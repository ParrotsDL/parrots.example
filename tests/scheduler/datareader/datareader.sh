#!/bin/bash

cd ../../..
git submodule update --init models/ssd
cd models/ssd
git pull origin pat0.6
cd ../..
search-run -t datareader_test tests/scheduler/datareader/search_config_datareadertest.yaml
search-ctl show -t datareader_test
