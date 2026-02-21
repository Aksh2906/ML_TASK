# Hand Recognition & Gesture Recognition - Drone Control System

Advanced gesture and air-drawing recognition for drone/device control using MediaPipe.

## USP (Unique Selling Point)

This system recognizes **8 distinct hand gestures** and **4 air-drawn shapes**, mapping them to drone commands:

### 8 Hand Gestures
| Gesture | Command |
|---------|---------|
| **Left Pointing** (index extended left) | LEFT MOVE |
| **Right Pointing** (index extended right) | RIGHT MOVE |
| **Up Pointing** (index extended up) | UPWARD MOVE |
| **Down Pointing** (index extended down) | DOWNWARD MOVE |
| **Victory Sign** (index + middle extended) | VICTORY CELEBRATION 🎉 |
| **Rock Sign** (index + pinky extended) | BACKFLIP! |
| **Open Palm** (all fingers spread) | STOP IMMEDIATELY ⚠️ |
| **Closed Fist** (all fingers closed) | LAND |

### Air-Drawn Shapes
| Shape | Command |
|-------|---------|
| **Circle** | CIRCLE MOVEMENT |
| **Square** | SQUARE PATTERN |
| **Triangle** | TRIANGLE PATTERN |
| **Line** | LINE MOVEMENT |

## How It Works

### 1. Hand Landmark Detection
- Uses **MediaPipe Hand Landmarker** (official Google ML model)
- Detects **21 hand landmarks** per hand in real-time
- Tracks **2 hands simultaneously**

### 2. Gesture Classification
- **Landmark-based analysis**: Checks finger extension states (tips vs joints)
- **Direction detection**: Analyzes relative position of fingers to wrist
- **Smoothing**: Uses temporal filtering to reduce jitter and false positives

### 3. Air Drawing Recognition
- **Motion tracking**: Detects when hand is still (drawing mode)
- **Trail collection**: Records sequence of hand positions
- **Shape analysis**: 
  - Circularity measurement
  - Corner detection for polygons
  - Aspect ratio analysis
  - Direction change analysis

### 4. Command Mapping
- Real-time gesture → command translation
- Visual feedback on screen with color-coded commands
- Critical commands (STOP, LAND) trigger frame borders

## Setup

### 1. Create Virtual Environment
```bash
cd /Users/aksh-aggarwal/Desktop/Workspace/aims/myPro
./run.sh setup
```

### 2. Run the Demo
```bash
./run.sh run
```

Or manually:
```bash
source myENV/bin/activate
python hand_gesture.py
```

## Features

✅ **Real-time Hand Detection** - Processes video at ~30 FPS  
✅ **Multi-hand Support** - Recognizes up to 2 hands simultaneously  
✅ **Robust Gesture Recognition** - Uses temporal smoothing for stability  
✅ **Air Drawing** - Tracks and recognizes shapes drawn in air  
✅ **Visual Feedback** - Color-coded commands with live trail display  
✅ **Critical Commands** - Special highlighting for safety (STOP, LAND)  
# Hand Gesture Demo — Beginner Friendly

This project demonstrates simple real-time hand gesture recognition using MediaPipe. It's intended for learners who want a hands-on demo that maps a few hand gestures to commands (for example, to control a drone simulator or for UI experiments).

Quick overview:
- Detects hand landmarks (21 points per hand)
- Classifies a small set of gestures like pointing, open palm, victory, rock, and closed fist
- Shows live visual feedback in a window

Prerequisites
- Python 3.8+ and a webcam
- Create and activate the included virtual environment or install packages from `requirements.txt`.

Quick start
1. Activate the virtual environment (if present):
```bash
source myENV/bin/activate
```
2. Run the demo:
```bash
python hand_gesture.py
```

What you'll see
- A camera window with hand landmarks drawn
- Detected gesture shown on-screen and a simple command label
- Press `q` to quit

Files you need
- `hand_gesture.py` — main demo (run this)
- `gesture_classifier.py` — logic that interprets hand landmarks
- `command_mapper.py` — maps gestures to text/color for display
- `requirements.txt` — Python dependencies

Notes
- The project previously included an air-drawing helper (`air_drawing.py`). It is not required by the main demo and has been removed to keep things simple for beginners.
- The model file `hand_landmarker.task` will be downloaded automatically if missing.

Need help?
- Tell me what OS or Python version you're using and I can give exact install commands.

Enjoy experimenting with hand gestures!
 
Hi,

This is my small hand gestures project. I am learning and i wrote this. It is simple.

What it does
- Uses mediapipe to find hand points
- Tries to tell few gestures like point, open palm, victory, rock, closed fist
- Shows a window with landmarks and a text

What you need
- Python 3
- A webcam
- Install packages from requirements.txt or use the included env

How to run (simple)
```bash
source myENV/bin/activate
python hand_gesture.py
```

Notes
- The model file will download automatically when you run
- Press q to quit the window
- I removed some extra code (air drawing) because i was not using it in main script

Files (important)
- hand_gesture.py  (main script)
- gesture_classifier.py  (classify gestures)
- command_mapper.py  (maps gestures to text/colors)
- requirements.txt

If something breaks tell me and i try to fix it.

Thanks
