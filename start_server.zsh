#!/bin/zsh -i

# Check if there's a session of this script running, if it exists, return 1
if pgrep -f "api.py" > /dev/null; then
  echo "Script is already running."
  exit 1
fi

# Activate conda environment: poe2gpt
conda activate poe2gpt

# Define the directory of this script
HERE=$(dirname "$0")
echo "HERE=$HERE" >> ~/test.txt

# Function to clean up and exit
cleanup() {
  echo "Both processes have ended. Exiting."
  exit 0
}

# Trap EXIT signal to call cleanup when both processes end
trap cleanup EXIT

# Activate FastAPI in the background
python $HERE/external/api.py > $HERE/poe2gpt_api.log &
PID1=$!

# Sleep for 2 seconds to wait for api server setup
sleep 2

# Activate proxy
./poe-openai-proxy > $HERE/poe2gpt_proxy.log &
PID2=$!

# Wait for both processes to finish
wait $PID1
wait $PID2