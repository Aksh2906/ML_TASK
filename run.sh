#!/usr/bin/env bash
set -e
ROOT="/Users/aksh-aggarwal/Desktop/Workspace/aims/myPro"
CMD="$1"
cd "$ROOT"

if [ "$CMD" = "setup" ]; then
  python3 -m venv myENV
  # shellcheck disable=SC1091
  source myENV/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  echo "Environment ready. Activate with: source myENV/bin/activate"
elif [ "$CMD" = "run" ]; then
  # shellcheck disable=SC1091
  source myENV/bin/activate
  python hand_gesture.py
else
  echo "Usage: ./run.sh [setup|run]"
fi
