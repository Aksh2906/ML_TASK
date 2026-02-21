"""
Command mapper: Maps gestures and shapes to drone commands.
"""

GESTURE_TO_COMMAND = {
    "left_point": ("LEFT MOVE", (0, 255, 0)),
    "right_point": ("RIGHT MOVE", (0, 255, 0)),
    "up_point": ("UPWARD MOVE", (0, 255, 0)),
    "down_point": ("DOWNWARD MOVE", (0, 255, 0)),
    "thumbs_up": ("MOVE UP 👍", (255, 100, 100)),
    "thumbs_down": ("MOVE DOWN 👎", (255, 100, 100)),
    "thumbs_left": ("MOVE LEFT 👈", (100, 255, 100)),
    "thumbs_right": ("MOVE RIGHT 👉", (100, 255, 100)),
    "victory": ("VICTORY CELEBRATION! 🎉", (0, 200, 255)),
    "rock": ("BACKFLIP!", (255, 0, 255)),
    "open_palm": ("STOP IMMEDIATELY", (0, 0, 255)),
    "closed_palm": ("LAND", (255, 0, 0)),
}

SHAPE_TO_COMMAND = {
    "circle": ("CIRCLE MOVEMENT", (100, 100, 255)),
    "square": ("SQUARE PATTERN", (100, 255, 100)),
    "triangle": ("TRIANGLE PATTERN", (255, 100, 100)),
    "line": ("LINE MOVEMENT", (200, 200, 0)),
}

def get_gesture_command(gesture: str) -> tuple:
    """
    Get command text and color for a gesture.
    Returns: (command_text, color_bgr)
    """
    return GESTURE_TO_COMMAND.get(gesture, ("", (128, 128, 128)))

def get_shape_command(shape: str) -> tuple:
    """
    Get command text and color for a shape.
    Returns: (command_text, color_bgr)
    """
    return SHAPE_TO_COMMAND.get(shape, ("", (128, 128, 128)))

def is_critical_command(gesture: str) -> bool:
    """Check if gesture is a critical command (stop, land)."""
    return gesture in ["open_palm", "closed_palm"]
