#!/bin/sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
cfg=configs/$2.yaml
# mpirun -np $1 python -u main.py --config $cfg \
#    2>&1 | tee log/train_$1_mlu_$2.log-$now

/usr/local/neuware/bin/cnperf-cli record "python -u main.py --config $cfg" -o cnperf_trace
/usr/local/neuware/bin/cnperf-cli timechart -i cnperf_trace -o ./output
