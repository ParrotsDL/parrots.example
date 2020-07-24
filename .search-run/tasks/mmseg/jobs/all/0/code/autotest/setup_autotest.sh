#!/usr/bin/bash

ROOT=$(pwd)

if [ ! -f $ROOT/.search-run/summary.yaml ]; then
    rm -rf AutoParrots .search-run
    git clone git@gitlab.sz.sensetime.com:wangshiguang/AutoParrots.git
    cd AutoParrots
    python setup.py install --user
    cd ..
    search-init
fi

search-ctl show
