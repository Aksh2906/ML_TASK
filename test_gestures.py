#!/usr/bin/env python
"""
Example: Gesture Recognition with Command Output
Demonstrates all 8 gestures and 4 shapes with command feedback.
"""

import numpy as np
from gesture_classifier import GestureClassifier
from air_drawing import AirDrawingTracker
from command_mapper import get_gesture_command, get_shape_command, GESTURE_TO_COMMAND

def test_gesture_classifier():
    """Test all 8 gestures."""
    print("\n=== Testing Gesture Classifier ===\n")
    
    classifier = GestureClassifier()
    
    # Test data: Mock landmarks
    test_cases = {
        "closed_palm": lambda: create_mock_landmarks(
            [False, False, False, False, False]  # all fingers closed
        ),
        "open_palm": lambda: create_mock_landmarks(
            [True, True, True, True, True],  # all fingers extended
            spread=True
        ),
        "victory": lambda: create_mock_landmarks(
            [False, True, True, False, False]  # index + middle
        ),
        "rock": lambda: create_mock_landmarks(
            [False, True, False, False, True]  # index + pinky
        ),
        "left_point": lambda: create_mock_landmarks(
            [False, True, False, False, False],  # only index
            direction=(-0.4, 0)  # pointing left (strong dx)
        ),
        "right_point": lambda: create_mock_landmarks(
            [False, True, False, False, False],
            direction=(0.4, 0)  # pointing right
        ),
        "up_point": lambda: create_mock_landmarks(
            [False, True, False, False, False],
            direction=(0, -0.4)  # pointing up
        ),
        "down_point": lambda: create_mock_landmarks(
            [False, True, False, False, False],
            direction=(0, 0.4)  # pointing down
        ),
    }
    
    for gesture_name, create_func in test_cases.items():
        landmarks = create_func()
        detected = classifier.classify(landmarks, "Right")
        cmd_text, color = get_gesture_command(detected)
        
        status = "✅" if detected == gesture_name else "⚠️"
        print(f"{status} {gesture_name:15} → Detected: {detected:15} → Command: {cmd_text}")

def test_air_drawing():
    """Test shape recognition."""
    print("\n=== Testing Air Drawing & Shape Recognition ===\n")
    
    tracker = AirDrawingTracker()
    
    # Test circle (creates circular trail)
    print("🎨 Testing CIRCLE:")
    circle_trail = generate_circle_trail(100)
    shape = tracker.recognize_shape(circle_trail)
    print(f"   Recognized: {shape}")
    if shape:
        cmd_text, _ = get_shape_command(shape)
        print(f"   Command: {cmd_text}\n")
    
    # Test square
    print("🎨 Testing SQUARE:")
    tracker.reset()
    square_trail = generate_square_trail(100)
    shape = tracker.recognize_shape(square_trail)
    print(f"   Recognized: {shape}")
    if shape:
        cmd_text, _ = get_shape_command(shape)
        print(f"   Command: {cmd_text}\n")
    
    # Test line
    print("🎨 Testing LINE:")
    tracker.reset()
    line_trail = generate_line_trail(100)
    shape = tracker.recognize_shape(line_trail)
    print(f"   Recognized: {shape}")
    if shape:
        cmd_text, _ = get_shape_command(shape)
        print(f"   Command: {cmd_text}\n")

def test_command_mapping():
    """Show all gesture → command mappings."""
    print("\n=== Complete Gesture → Command Mapping ===\n")
    
    print("HAND GESTURES:")
    print("-" * 60)
    for gesture, (cmd, color) in GESTURE_TO_COMMAND.items():
        print(f"  {gesture:20} → {cmd}")
    
    print("\n" + "-" * 60)

def create_mock_landmarks(finger_states, spread=False, direction=None):
    """Create mock landmarks for testing."""
    # finger_states: [thumb, index, middle, ring, pinky]
    landmarks = np.zeros((21, 3))
    
    # Wrist at origin
    landmarks[0] = [0.5, 0.5, 0.0]
    
    # Thumb (landmarks 1-4: CMC, MCP, IP, TIP)
    landmarks[1] = [0.45, 0.5, 0.0]  # CMC
    landmarks[2] = [0.47, 0.48, 0.0]  # MCP
    landmarks[3] = [0.48, 0.45, 0.0]  # IP
    y = 0.3 if finger_states[0] else 0.55
    landmarks[4] = [0.5, y, 0.0]  # TIP
    
    # Index, Middle, Ring, Pinky
    # Structure: MCP (5,9,13,17), PIP (6,10,14,18), DIP (7,11,15,19), TIP (8,12,16,20)
    finger_ranges = [(5, 8), (9, 12), (13, 16), (17, 20)]
    
    for finger_idx, (mcp_idx, tip_idx) in enumerate(finger_ranges):
        extended = finger_states[finger_idx + 1]
        pip_idx = mcp_idx + 1
        dip_idx = mcp_idx + 2
        
        landmarks[mcp_idx] = [0.5, 0.48, 0.0]  # MCP
        
        if extended:  # finger is extended
            landmarks[pip_idx] = [0.5, 0.35, 0.0]  # PIP
            landmarks[dip_idx] = [0.5, 0.25, 0.0]  # DIP
            landmarks[tip_idx] = [0.5, 0.15, 0.0]  # TIP (way up)
        else:  # finger is closed
            landmarks[pip_idx] = [0.5, 0.50, 0.0]  # PIP
            landmarks[dip_idx] = [0.5, 0.52, 0.0]  # DIP
            landmarks[tip_idx] = [0.5, 0.55, 0.0]  # TIP (down)
    
    # Apply spread for open palm
    if spread:
        spread_x = [0.3, 0.4, 0.5, 0.6, 0.7]
        tip_indices = [4, 8, 12, 16, 20]
        for x, tip_idx in zip(spread_x, tip_indices):
            landmarks[tip_idx, 0] = x
    
    # Apply direction for pointing - modify index tip position (landmark 8)
    if direction:
        dx, dy = direction
        # Only modify X, keep Y to indicate it's extended
        landmarks[8, 0] = 0.5 + dx * 0.2
    
    return landmarks

def generate_circle_trail(num_points):
    """Generate a circular trail."""
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = 0.5 + 0.1 * np.cos(angles)
    y = 0.5 + 0.1 * np.sin(angles)
    return list(zip(x, y))

def generate_square_trail(num_points):
    """Generate a square trail."""
    points = []
    side_len = num_points // 4
    
    # Top-left to top-right
    for i in range(side_len):
        t = i / side_len
        points.append((0.3 + 0.2 * t, 0.3))
    # Top-right to bottom-right
    for i in range(side_len):
        t = i / side_len
        points.append((0.5, 0.3 + 0.2 * t))
    # Bottom-right to bottom-left
    for i in range(side_len):
        t = i / side_len
        points.append((0.5 - 0.2 * t, 0.5))
    # Bottom-left to top-left
    for i in range(side_len):
        t = i / side_len
        points.append((0.3, 0.5 - 0.2 * t))
    
    return points

def generate_line_trail(num_points):
    """Generate a line trail."""
    x = np.linspace(0.2, 0.8, num_points)
    y = np.ones(num_points) * 0.4
    return list(zip(x, y))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Gesture & Air Drawing Recognition - Test Suite")
    print("="*60)
    
    test_gesture_classifier()
    test_air_drawing()
    test_command_mapping()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60 + "\n")
