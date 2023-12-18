#!/bin/sh

export SAMPLE_NUM=4000

export EXP_IND=6
bash reformulation_experiment/de/run_experiment.sh

export EXP_IND=7
bash reformulation_experiment/de/run_experiment.sh

export EXP_IND=8
bash reformulation_experiment/de/run_experiment.sh
