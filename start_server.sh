#!/bin/bash

# Ensure the script stops on errors
set -e

# Create a logs directory if it doesn't exist
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/server_$TIMESTAMP.log"

# Add the current directory to PYTHONPATH so python can find the 'app' module
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Starting ServeLLM server..."
echo "Logs will be written to $LOG_FILE"

# Start the uvicorn server in the background and redirect output to the log file
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &

PID=$!
echo "Server started with PID: $PID"
echo "To view the logs in real-time, run: tail -f $LOG_FILE"
echo "To stop the server, run: kill $PID"
