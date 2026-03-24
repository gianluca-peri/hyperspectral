#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <python_script_name>"
    exit 1
fi

PYTHON_SCRIPT=$1
LOG_DIR="Logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/${PYTHON_SCRIPT%.py}.log"

# Run in background with nohup, capture both stdout and stderr into one log, save PID
nohup python3 -u "$PYTHON_SCRIPT" >"$LOGFILE" 2>&1 &

PID=$!
echo "Started $PYTHON_SCRIPT with PID $PID"
echo "All output -> $LOGFILE"