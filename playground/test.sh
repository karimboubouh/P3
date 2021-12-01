#!/bin/bash

source ./helpers.sh

# parse script arguments in format << --param value >>
process_arguments "$@"
#args_parser "$@"

# shield the selected CPU(s).
shield_cpu

# configure shielded CPU frequency
configure_cpu_performance

# run powerstat in idle state for 10min
idle_powerstat

# run a given program for $runs times
multi_run

# printf
echo "The program run on cpu $cpu, using $performance performance, after $delay s of delay for $runs times."
