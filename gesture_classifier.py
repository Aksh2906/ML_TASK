"""
Gesture classification from hand landmarks.
Recognizes: left_point, right_point, up_point, down_point, victory, rock, open_palm, closed_palm
"""
import numpy as np
from typing import List, Tuple


class GestureClassifier:
    """Classifies hand gestures from 21 MediaPipe landmarks."""
    
    # Landmark indices for quick reference
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18
    
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    
    THUMB_IP = 3
    THUMB_MCP = 2
    
    def __init__(self):
        self.gesture_history = []
        self.history_size = 5  # Use last 5 frames for smoothing
    
    def classify(self, landmarks: List[np.ndarray], handedness: str) -> Tuple[str, float]:
        """
        Classify gesture from landmarks.
        Args:
            landmarks: list of 21 normalized landmarks (each with x, y, z)
            handedness: 'Left' or 'Right'
        Returns:
            (gesture name string, confidence score 0-1)
        """
        if len(landmarks) != 21:
            return "unknown", 0.0
        
        lm = landmarks
        
        # Extract finger tip states (extended or not)
        thumb_extended = self._is_extended(lm[self.THUMB_TIP], lm[self.THUMB_IP])
        index_extended = self._is_extended(lm[self.INDEX_TIP], lm[self.INDEX_PIP])
        middle_extended = self._is_extended(lm[self.MIDDLE_TIP], lm[self.MIDDLE_PIP])
        ring_extended = self._is_extended(lm[self.RING_TIP], lm[self.RING_PIP])
        pinky_extended = self._is_extended(lm[self.PINKY_TIP], lm[self.PINKY_PIP])
        
        # Detect thumb gestures (thumbs up, down, left, right)
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            thumb_direction = self._get_thumb_direction(lm, handedness)
            return thumb_direction, 0.85
        
        # Detect closed palm (all fingers closed)
        if not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return "closed_palm", 0.90
        
        # Detect open palm (all fingers extended, spread)
        if all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            spread = self._finger_spread(lm)
            if spread > 0.3:  # Fingers spread apart
                confidence = min(0.95, 0.7 + spread * 0.25)
                return "open_palm", confidence
        
        # Detect victory (index and middle extended, ring and pinky closed)
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "victory", 0.88
        
        # Detect rock (index and pinky extended, middle and ring closed)
        if index_extended and pinky_extended and not middle_extended and not ring_extended:
            return "rock", 0.88
        
        # Detect pointing gestures (index extended, others closed)
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            direction = self._get_point_direction(lm, handedness)
            return direction, 0.80
        
        return "unknown", 0.0
    
    def smooth_gesture(self, gesture: str, confidence: float, history_weight: float = 0.7) -> Tuple[str, float]:
        """
        Smooth gestures using history to reduce jitter.
        Returns: (gesture, smoothed_confidence)
        """
        self.gesture_history.append((gesture, confidence))
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Calculate smoothed confidence based on history
        if len(self.gesture_history) > 0:
            recent_gestures = [g for g, _ in self.gesture_history[-3:]]
            recent_confidences = [c for _, c in self.gesture_history[-3:]]
            
            if recent_gestures.count(gesture) >= 2:
                smoothed_conf = sum(recent_confidences) / len(recent_confidences)
                return gesture, smoothed_conf
        
        if gesture != "unknown":
            return gesture, confidence
        elif len(self.gesture_history) > 1:
            last_gesture, last_conf = self.gesture_history[-2]
            return last_gesture, last_conf * 0.5
        
        return "unknown", 0.0
    
    def _is_extended(self, tip, pip, threshold: float = 0.05) -> bool:
        """Check if a finger is extended by comparing tip and PIP y-coordinates."""
        tip_y = tip.y if hasattr(tip, 'y') else tip[1]
        pip_y = pip.y if hasattr(pip, 'y') else pip[1]
        return tip_y < pip_y - threshold
    
    def _finger_spread(self, lm: List[np.ndarray]) -> float:
        """
        Measure how spread apart the fingers are.
        Returns a value 0-1, higher means more spread.
        """
        # Handle both landmark objects and numpy arrays
        tips = []
        for idx in [4, 8, 12, 16, 20]:
            x = lm[idx].x if hasattr(lm[idx], 'x') else lm[idx][0]
            tips.append(x)
        
        all_x = []
        for lm_pt in lm:
            x = lm_pt.x if hasattr(lm_pt, 'x') else lm_pt[0]
            all_x.append(x)
        
        spread = (max(tips) - min(tips)) / (max(all_x) - min(all_x) + 1e-6)
        return min(spread, 1.0)
    
    def _get_point_direction(self, lm: List[np.ndarray], handedness: str) -> str:
        """
        Determine pointing direction (left, right, up, down).
        Uses index finger tip relative to wrist.
        """
        wrist = lm[self.WRIST]
        index_tip = lm[self.INDEX_TIP]
        
        # Handle both landmark objects and numpy arrays
        wrist_x = wrist.x if hasattr(wrist, 'x') else wrist[0]
        wrist_y = wrist.y if hasattr(wrist, 'y') else wrist[1]
        tip_x = index_tip.x if hasattr(index_tip, 'x') else index_tip[0]
        tip_y = index_tip.y if hasattr(index_tip, 'y') else index_tip[1]
        
        dx = tip_x - wrist_x
        dy = tip_y - wrist_y
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Determine primary direction
        if abs_dx > abs_dy:
            # Left or right
            if dx < -0.1:  # Pointing left (negative x)
                return "left_point"
            elif dx > 0.1:  # Pointing right (positive x)
                return "right_point"
        else:
            # Up or down
            if dy < -0.1:  # Pointing up (negative y)
                return "up_point"
            elif dy > 0.1:  # Pointing down (positive y)
                return "down_point"
        
        return "unknown"
    
    def _get_thumb_direction(self, lm: List[np.ndarray], handedness: str) -> str:
        """
        Determine thumb direction (up, down, left, right).
        Uses thumb tip relative to wrist.
        """
        wrist = lm[self.WRIST]
        thumb_tip = lm[self.THUMB_TIP]
        
        # Handle both landmark objects and numpy arrays
        wrist_x = wrist.x if hasattr(wrist, 'x') else wrist[0]
        wrist_y = wrist.y if hasattr(wrist, 'y') else wrist[1]
        tip_x = thumb_tip.x if hasattr(thumb_tip, 'x') else thumb_tip[0]
        tip_y = thumb_tip.y if hasattr(thumb_tip, 'y') else thumb_tip[1]
        
        dx = tip_x - wrist_x
        dy = tip_y - wrist_y
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Determine primary direction
        if abs_dx > abs_dy:
            # Left or right
            if dx < -0.1:  # Thumb pointing left
                return "thumbs_left"
            elif dx > 0.1:  # Thumb pointing right
                return "thumbs_right"
        else:
            # Up or down
            if dy < -0.1:  # Thumb pointing up (negative y)
                return "thumbs_up"
            elif dy > 0.1:  # Thumb pointing down (positive y)
                return "thumbs_down"
        
        return "unknown"
