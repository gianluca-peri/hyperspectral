#!/bin/bash

LOGFILE="cifar10.log"

# Run in background with nohup, capture both stdout and stderr into one log, save PID
nohup python3 -u cifar10.py >"$LOGFILE" 2>&1 &

PID=$!
echo "Started cifar10.py with PID $PID"
echo "All output -> $LOGFILE"
