#!/bin/bash

WORKFLOW="/SNS/software/scd/garnet_reduction/src/garnet/workflow.py"

echo $WORKFLOW

REDUCTION="norm"
CONDA_ENV="scd-reduction-tools"

while getopts htnifd FLAG; do
    case $FLAG in
        h)
            echo "/SNS/software/scd/reduce.sh -i[n] reduction.yaml processes"
            echo "/SNS/software/scd/reduce.sh -t reduction.yaml instrument"
            echo "/SNS/software/scd/reduce.sh -d  use development version"
            exit 1
            ;;
        t)
            REDUCTION="temp"
            ;;
        n)
            REDUCTION="norm"
            ;;
        i)
            REDUCTION="int"
            ;;
        f)
            REDUCTION="fit"
            ;;
        d)
            CONDA_ENV="scd-reduction-tools-dev"
            ;;
    esac
done

shift $((OPTIND-1))

if [[ $# -ne 2 ]]; then
    echo "Requires input yaml file and number of processes"
    exit 1
fi

INPUT=$1
PROCESSES=$2

CONDA="/opt/anaconda/bin/activate"
if [ ! -f $CONDA ]; then
    CONDA="$HOME/miniconda3/bin/activate"
fi

rm -rf ~/.cache/fontconfig

echo $CONDA
echo $INPUT
source "${CONDA}" $CONDA_ENV
echo python $WORKFLOW $INPUT $REDUCTION $PROCESSES

START=$(date +%s)

/usr/bin/time -v python $WORKFLOW $INPUT $REDUCTION $PROCESSES

END=$(date +%s)
ELAPSED=$((END - START))
printf "Elapsed time: %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))