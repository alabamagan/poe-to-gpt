#!/bin/zsh -i

# Check if there's a session of this script running, if it exists, return 1
if pgrep -f "app.py" > /dev/null; then
  echo "Script is already running."
  exit 1
fi

# Parse command line arguments
DEBUG_MODE=""
HOST_ARG=""
PORT_ARG=""
LOG_SUFFIX=""

# Parse arguments passed to this script
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG_MODE="--debug"
      LOG_SUFFIX="_debug"
      echo "Debug mode enabled"
      shift
      ;;
    --host)
      HOST_ARG="--host $2"
      shift 2
      ;;
    --port)
      PORT_ARG="--port $2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --debug         Enable debug mode with detailed request logging"
      echo "  --host HOST     Host to bind the server to (default: 0.0.0.0)"
      echo "  --port PORT     Port to run the server on (default: 3700)"
      echo "  --help, -h      Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                          # Start server in normal mode"
      echo "  $0 --debug                  # Start server in debug mode"
      echo "  $0 --debug --port 8080      # Start server in debug mode on port 8080"
      echo "  $0 --host 127.0.0.1 --port 8080  # Start server on specific host and port"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Activate conda environment: poe2gpt
echo "Activating conda environment: poe2gpt"
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

# Build the command with arguments
PYTHON_CMD="python $HERE/app.py"
if [[ -n "$DEBUG_MODE" ]]; then
  PYTHON_CMD="$PYTHON_CMD $DEBUG_MODE"
fi
if [[ -n "$HOST_ARG" ]]; then
  PYTHON_CMD="$PYTHON_CMD $HOST_ARG"
fi
if [[ -n "$PORT_ARG" ]]; then
  PYTHON_CMD="$PYTHON_CMD $PORT_ARG"
fi

# Log file name with debug suffix if in debug mode
LOG_FILE="$HERE/poe2gpt_api${LOG_SUFFIX}.log"

echo "Starting poe2gpt server..."
if [[ -n "$DEBUG_MODE" ]]; then
  echo "Debug mode: Detailed request logging enabled"
fi
echo "Command: $PYTHON_CMD"
echo "Log file: $LOG_FILE"

# Activate FastAPI in the background
eval "$PYTHON_CMD" > "$LOG_FILE" 2>&1 &
PID1=$!

echo "Server started with PID: $PID1"
echo "Log file: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' to view real-time logs"
if [[ -n "$DEBUG_MODE" ]]; then
  echo "Debug logs will show detailed request information"
fi

# Wait for both processes to finish
wait $PID1
