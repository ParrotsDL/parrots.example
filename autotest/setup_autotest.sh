#!/usr/bin/bash

ROOT=$(pwd)

if [ ! -f $ROOT/.search-run/summary.yaml ]; then
    rm -rf parrots.gml .search-run
    git clone git@gitlab.sz.sensetime.com:parrotsDL-sz/parrots.gml.git
    cd parrots.gml
    python setup.py install --user
    cd ..
    search-init
fi

search-ctl show
