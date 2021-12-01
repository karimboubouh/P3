#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "Please specify in which CPU(s) to run the program"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Please specify the delay before starting the execution"
  exit 1
fi
cpu=$1

echo "Configuring environment ..."
echo "Using CPU(s) #$cpu ..."

sudo cset shield --reset
sudo cset shield --cpu "$cpu" # --kthread=on
sudo cpupower --cpu "$cpu" frequency-set -g userspace -d 3.60Ghz -u 3.60Ghz
sudo cpupower --cpu 1-3 frequency-set -g userspace -d 1.2Ghz -u 1.2Ghz


delay=$2

# mark start time to >> execution.log
echo -e "Waiting for $delay seconds before starting the execution ...\n\n"
sudo powerstat 1 600000 -d $delay -R -c -z >execution.log &
pPid=$!

sleep $delay

# mark start time and end time

cset shield --exec python main.py

# mark end time to >> execution.log

sudo kill $pPid

echo -e "\n"
#sudo cset shield --reset
echo "Done."
