# configurations
TIMEFORMAT="%Rs"
MAX_FREQ="3.6Ghz"
MIN_FREQ="1.2Ghz"
IDLE_TIME=60
IDLE_FILE="idle.log"

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

usage() {
  cecho D "A script to evaluate the energy consumed running a given program." " "
  cecho GREEN "USAGE:" " "
  cecho D "$(basename $0) [-cpr] --exec <command line>" "\t"
  cecho GREEN "OPTIONS:"
  cecho GREEN "-e | --exec" "\n\t"
  cecho D "Required argument. Program to evaluate given as an executable command." "\t\t"
  cecho GREEN "-c | --cpu" "\n\t"
  cecho D "IDs of CPU to shield. For example: -c 0 or --cpu 1-3. (defaults to 0)" "\t\t"
  cecho GREEN "-p | --performance" "\n\t"
  cecho D "Performance profile of the shielded CPUs. Three values are accepted:" "\t\t"
  cecho D "max: for MAX_FREQ, set frequency of shielded CPUs to $MAX_FREQ." "\t\t\t"
  cecho D "avg: for OnDemand, shielded CPUs are managed by OnDemand governor." "\t\t\t"
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
  export pre=$(head -200 /dev/urandom | cksum | cut -f1 -d " ")
  export cpu=${cpu:-0}
  export delay=${delay:-10}
  export performance=${performance:-"min"} # max/min/avg
  export runs=${runs:-10}
  export delay=${delay:-5}

  while [ -n "$1" ]; do
    case $1 in
    -e | --exec)
      export exec=$2
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
  cecho D "Configuring the execution environment ..."
  cecho D "Shielding CPU $cpu ..."
  sudo cset shield --reset
  sudo cset shield --cpu "$cpu" --kthread=on
}

configure_cpu_performance() {
  cecho D "Setting CPU performance to $performance."
  if [ "$performance" == "max" ]; then
    df=$MAX_FREQ
    uf=$MAX_FREQ
    governor="userspace"
  elif [ "$performance" == "min" ]; then
    df=$MIN_FREQ
    uf=$MIN_FREQ
    governor="userspace"
  elif [ "$performance" == "avg" ]; then
    df=$MIN_FREQ
    uf=$MAX_FREQ
    governor="ondemand"
  else
    cecho D "parameter < --performance > only accept values: [max; min; avg]"
    exit 1
  fi
  sudo cpupower --cpu "$cpu" frequency-set -g $governor -d $df -u $uf
  sudo cpupower --cpu 1-3 frequency-set -g userspace -d $MIN_FREQ -u $MIN_FREQ
}

idle_powerstat() {
  cecho GREEN "******************** IDLE POWER USAGE ********************************\n" "\n"
  cecho CYAN "Calculating power usage of the CPUs in idle state for $IDLE_TIME seconds ..."
  sudo powerstat 1 $IDLE_TIME -Rcf >"${pre}_${IDLE_FILE}"
  x=($(grep "Average" $IDLE_FILE | grep -Eo '[0-9]+([.][0-9]+)?' | tail -2))
  cecho GREEN "Average Idle Power: ${x[0]} Watts"
  cecho GREEN "Average Frequency : ${x[1]} Ghz"
}

multi_run() {
  execution_times=()
  for ((i = 1; i <= runs; i++)); do
    cecho GREEN "******************** RUN Nº $i ****************************************\n" "\n"
    run_program "$i"
    execution_times+=("$elapsed")
    cecho YELLOW "Program running for round $1 ended in $elapsed milliseconds."
  done
  cecho BLUE "${execution_times[*]}"
}

run_program() {
  cecho D "Waiting for $delay seconds before starting the execution ..."
  filename="./out/${pre}_execution_$1.log"
  sudo powerstat 1 600000 -d "$delay" -R -c -z >$filename &
  pPid=$!
  sleep "$delay"
  start_time=$(date +%s.%6N)
  cset shield --exec "$exec"
  end_time=$(date +%s.%6N)
  elapsed=$(echo "scale=6; $end_time - $start_time" | bc)
  sudo kill $pPid
}
