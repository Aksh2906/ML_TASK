#!/usr/bin/env bash
# Quick Start Guide for Hand Gesture & Air Drawing Drone Control

echo ""
echo "=========================================="
echo "🚀 Hand Gesture Drone Control System"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Check if myENV exists
if [ ! -d "myENV" ]; then
    echo "⚙️  Creating virtual environment..."
    python3 -m venv myENV
    source myENV/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt > /dev/null 2>&1
    echo "✅ Environment ready!"
else
    source myENV/bin/activate
    echo "✅ Environment activated"
fi

echo ""
echo "📋 Available Commands:"
echo "  gesture              - Show gesture recognition demo"
echo "  test                 - Run gesture recognition tests"
echo "  help                 - Show this help message"
echo ""

if [ -z "$1" ]; then
    echo "Running gesture recognition demo..."
    echo "  Press 'q' to quit"
    echo "  Perform hand gestures or draw shapes in air"
    echo ""
    python hand_gesture.py
elif [ "$1" = "test" ]; then
    echo "Running test suite..."
    python test_gestures.py
elif [ "$1" = "help" ] || [ "$1" = "-h" ]; then
    echo "Usage: ./quickstart.sh [command]"
    echo ""
    echo "Commands:"
    echo "  gesture   - Run main gesture recognition demo (default)"
    echo "  test      - Run unit tests for gesture & shape recognition"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./quickstart.sh gesture"
    echo "  ./quickstart.sh test"
else
    echo "Unknown command: $1"
    echo "Try './quickstart.sh help'"
fi

echo ""
