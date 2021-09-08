#!/bin/sh

cfg=configs/$1.yaml
/usr/local/neuware/bin/cnperf-cli record " python -u main_half_camb.py --config $cfg --isPerf True" -o cnperf_trace
/usr/local/neuware/bin/cnperf-cli timechart -i cnperf_trace -o ./output

#Before run this script, you need mv python folder to python_bak to avoid cnperf bug
