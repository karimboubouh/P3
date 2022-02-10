#!/bin/bash

# Configurations
UUID=$(cat /dev/urandom | tr -dc 'A-Z0-9' | fold -w 4 | head -n 1)
VENV_PATH="/home/karim/Workspace/.pyenv"
TIMEFORMAT="%Rs"
MAX_FREQ="3.6Ghz"
AVG_FREQ="1.9Ghz"
MIN_FREQ="1.2Ghz"
IDLE_TIME=60
AVG_IDLE_POWER=12.65
STD_IDLE_POWER=0.25

cecho() {
  BOLD="\033[1m"
  RED="\033[1;31m"
  BLUE='\033[1;34m'
  GREEN="\033[1;32m"
  YELLOW="\033[1;33m"
  PURPLE='\033[1;35m'
  CYAN='\033[1;36m'
  NC="\033[0m" # No Color

  printf "${!1:-}${3:-"➜ "}${2} ${NC}\n"
}

average(){
  arr=("$@")
  sum=$(IFS="+";bc<<<"${arr[*]}")
  mean=$(bc <<< "scale=2; $sum/${#arr[@]}")
  echo "$mean"
}

usage() {
  cecho D "A script to evaluate the energy consumed running a given program." " "
  cecho GREEN "USAGE:" " "
  cecho D "$(basename $0) [-cpr] --exec <command line>" "\t"
  cecho GREEN "OPTIONS:" " "
  cecho GREEN "-e | --exec" "\n\t"
  cecho D "Required argument. Program to evaluate given as an executable command." "\t\t"
  cecho GREEN "-a | --args" "\n\t"
  cecho D "Provide additional arguments to the --exec command." "\t\t"
  cecho GREEN "-c | --cpu" "\n\t"
  cecho D "IDs of CPU to shield. For example: -c 0 or --cpu 1-3. (defaults to 0)" "\t\t"
  cecho GREEN "-p | --performance" "\n\t"
  cecho D "Performance profile of the shielded CPUs. Three values are accepted:" "\t\t"
  cecho D "max: for MAX_FREQ, set frequency of shielded CPUs to $MAX_FREQ." "\t\t\t"
  cecho D "avg: for AVG_FREQ, set frequency of shielded CPUs to $AVG_FREQ." "\t\t\t"
  cecho D "ondemand: for OnDemand, shielded CPUs are managed by OnDemand governor." "\t\t\t"
  cecho D "min: for MIN_FREQ, set frequency of shielded CPUs to $MIN_FREQ." "\t\t\t"
  cecho D "(defaults to max)." "\t\t"
  cecho GREEN "-r | --runs" "\n\t"
  cecho D "Number of times to run the evaluated program (defaults to 10) ." "\t\t"
  cecho GREEN "-d | --delay" "\n\t"
  cecho D "delay in seconds to wait for the system to be idle before running the program." "\t\t"
  cecho D "(defaults to 5)." "\t\t"
  cecho GREEN "-h | --help" "\n\t"
  cecho D "print this help." "\t\t"

  exit 0
}

process_arguments() {
  # Define default values for optional arguments
  export cpu=${cpu:-0}
  export delay=${delay:-10}
  export performance=${performance:-"max"} # max/min/avg/ondemand
  export args=${args:-""}                  # no arguments
  export runs=${runs:-10}
  export delay=${delay:-5}

  while [ -n "$1" ]; do
    case $1 in
    -e | --exec)
      export exec=$2
      shift
      ;;
    -a | --args)
      export args=$2
      shift
      ;;
    -c | --cpu)
      export cpu=$2
      shift
      ;;
    -p | --performance)
      export performance=$2
      shift
      ;;
    -r | --runs)
      export runs=$2
      shift
      ;;
    -d | --delay)
      export delay=$2
      shift
      ;;
    -h | --help)
      usage
      exit 1
      ;;
    *)
      cecho YELLOW "Unknown argument $1. run '$(basename $0) --help' for help."
      exit 1
      ;;
    esac
    shift
  done
  if [ -z "$exec" ]; then
    cecho RED "you must provide the program to evaluate. run '$(basename $0) --help' for help."
    exit 1
  fi
}

args_parser() {
  # Define default values
  export cpu=${cpu:-0}
  export delay=${delay:-10}
  export performance=${performance:-"max"} # max/min/avg
  export runs=${runs:-10}

  # Assign the values given by the user
  while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
      param="${1/--/}"
      export declare "$param"="$2"
    fi
    shift
  done
}

shield_cpu() {
  cecho GREEN "******************** SYSTEM CONFIGURATION ****************************\n" "\n"
  cecho BOLD "Execution ID: $UUID"
  mkdir -p "./out/${UUID}_${performance}"
  cecho D "Configuring the execution environment ..."
  # allow reading energy with normal user (security violation)
  sudo chmod +r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
  # activate virualenv
  source $VENV_PATH/bin/activate
  # shield CPU
  cecho D "Shielding CPU $cpu ..."
  sudo cset shield --reset
  sudo cset shield --cpu "$cpu" --kthread=on
}

configure_cpu_performance() {
  cecho D "Setting CPU performance to $performance."
  sudo cpupower --cpu all frequency-set -g userspace -d $MIN_FREQ -u $MIN_FREQ
  if [ "$performance" == "max" ]; then
    df=$MAX_FREQ
    uf=$MAX_FREQ
    governor="userspace"
  elif [ "$performance" == "min" ]; then
    df=$MIN_FREQ
    uf=$MIN_FREQ
    governor="userspace"
  elif [ "$performance" == "avg" ]; then
    df=$AVG_FREQ
    uf=$AVG_FREQ
    governor="userspace"
  elif [ "$performance" == "ondemand" ]; then
    df=$MIN_FREQ
    uf=$MAX_FREQ
    governor="ondemand"
  else
    cecho D "parameter < --performance > only accept values: [max; min; avg, ondemand]"
    exit 1
  fi
  sudo cpupower --cpu "$cpu" frequency-set -g $governor -d $df -u $uf
}

idle_powerstat() {
  cecho GREEN "******************** IDLE POWER USAGE ********************************\n" "\n"
  cecho D "Calculating power usage of the CPUs in idle state for $IDLE_TIME seconds ..."
  idle_file="./out/${UUID}_${performance}/idle.log"
  powerstat 1 $IDLE_TIME -Rf 2>&1 | tee $idle_file
  #  powerstat 1 $IDLE_TIME -Rf >$idle_file
  avgIdle=($(grep "Average" $idle_file | grep -Eo '[0-9]+([.][0-9]+)?' | tail -2))
  stdIdle=($(grep "StdDev" $idle_file | grep -Eo '[0-9]+([.][0-9]+)?' | tail -2))
  cecho GREEN "Average Idle Power      : ${avgIdle[0]} Watts; (StdDev of ${stdIdle[0]} Watts)."
  cecho GREEN "Average Frequency       : ${avgIdle[1]} Ghz; (StdDev of ${stdIdle[1]} Ghz)."
  avgJIdle=$(echo "${avgIdle[0]} * $IDLE_TIME" | bc)
  stdJIdle=$(echo "${stdIdle[0]} * $IDLE_TIME" | bc)
  cecho YELLOW "Average Energy consumed : $avgJIdle J; (StdDev of $stdJIdle J)."
}

multi_run() {
  # Default Idle power usage if idle_powerstat() was not executed.
  avgIP=${avgIdleWW[0]:-$AVG_IDLE_POWER}
  stdIP=${stdIdleWW[0]:-$STD_IDLE_POWER}
  # List of execution times
  T=()
  # List of average power usages
  P=()
  # List of average power usages
  PP=()
  # List of average CPU frequencies
  F=()
  # List of energy consumptions
  E=()

  for ((i = 1; i <= runs; i++)); do
    cecho GREEN "******************** RUN Nº $i ****************************************\n" "\n"
    run_program "$i" "$avgIP"
    T+=("$elapsed")
    P+=("${avg[0]}")
    PP+=("$program_power")
    F+=("${avg[1]}")
    E+=("$program_energy")
    cecho CYAN "Program running for round $i ended in $elapsed seconds."
  done
  printf "\n\n"
  cecho GREEN "  Experiment results summary: " "\t"
  cecho BLUE "-----------------------------------------------"
  avg_time=$(average "${T[@]}")
  stdJtime=$(echo "$stdIP * $avg_time" | bc)
  cecho YELLOW "Average Execution Time      : $avg_time seconds."
  cecho CYAN "Average Power Usage         : $(average "${P[@]}") Watts."
  cecho CYAN "Average Program Power Usage : $(average "${PP[@]}") Watts."
  cecho YELLOW "Average CPU Frequency       : $(average "${F[@]}") GHz."
  cecho CYAN "Average Energy Consumed     : $(average "${E[@]}") (+- $stdJtime) Joules."
  cecho BLUE "-----------------------------------------------"
}

run_program() {
  idleP=$2
  cecho D "Waiting for $delay seconds before starting the execution ..."
  filename="./out/${UUID}_${performance}/execution_$1.log"
  powerstat 1 600000 -Rf -d "$delay" >"$filename" &
  pPid=$!
  sleep "$delay"
  #  start_time=$(date +%s.%6N)
  start_time=$(date +%s)
  sudo -E env PATH=$PATH cset shield --exec $exec -- $args
  #  sudo cset shield --exec $exec -- $args
  #  end_time=$(date +%s.%6N)
  end_time=$(date +%s)
  elapsed=$(echo "scale=6; $end_time - $start_time" | bc)
  kill -SIGTERM $pPid
  kill -SIGQUIT $pPid
  sleep 1
  printf "\n"
  avg=($(grep "Average" $filename | grep -Eo '[0-9]+([.][0-9]+)?' | tail -2))
  std=($(grep "StdDev" $filename | grep -Eo '[0-9]+([.][0-9]+)?' | tail -2))
  cecho GREEN "Average Frequency                      : ${avg[1]} Ghz; (StdDev of ${std[1]} Ghz)."
  cecho GREEN "Average Power usage                    : ${avg[0]} Watts; (StdDev of ${std[0]} Watts)."
  avgJ=$(echo "${avg[0]} * $elapsed" | bc)
  stdJ=$(echo "${std[0]} * $elapsed" | bc)
  cecho GREEN "Average Energy consumed                : $avgJ J; (StdDev of $stdJ J)."
  printf "\n"
  program_power=$(echo "${avg[0]} - $idleP" | bc)
  program_energy=$(echo "$program_power * $elapsed" | bc)
  cecho YELLOW "Average Power usage of the program     : $program_power Watts; (StdDev of ${std[0]} Watts)."
  cecho YELLOW "Average Energy consumed by the program : $program_energy J; (StdDev of $stdJ J)."
}

reset_configurations() {
  cecho YELLOW "Restoring default configurations"
  #  sudo chmod -r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
  deactivate
  sudo cset shield --reset
  sudo cpupower --cpu all frequency-set -g ondemand -d $MIN_FREQ -u $MAX_FREQ
  cecho GREEN "Program terminated."
}
