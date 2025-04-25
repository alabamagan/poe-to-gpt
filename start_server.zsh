#!/bin/zsh -i

# Check if there's a session of this script running, if it exists, return 1
if pgrep -f "app.py" > /dev/null; then
  echo "Script is already running."
  exit 1
fi

# Activate conda environment: poe2gpt
conda activate poe2gpt

# Define the directory of this script
HERE=$(dirname "$0")

# Function to clean up and exit
cleanup() {
  echo "poe2gpt server stopped"
  exit 0
}

# Trap EXIT signal to call cleanup when both processes end
trap cleanup EXIT

# Activate FastAPI in the background
python $HERE/app.py > $HERE/poe2gpt_api.log &
PID1=$!

# Wait for both processes to finish
wait $PID1
