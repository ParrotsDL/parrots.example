#!/bin/bash

cd ../..
search-init
git submodule update --init models/mmaction
cd models/mmaction
git checkout jml/kill_case
cd ../../

echo 'test kill time limited job'
PARROTS_BENCHMARK=1 PARROTS_TEST_LIMITED=1 search-run -f -t test unitest/kill_job/search_config_mmaction.yaml

echo 'test kill log no change job'
PARROTS_TEST_LOG_NO_CHANGE=1 search-run -f -t test unitest/kill_job/search_config_mmaction.yaml

cd models/mmaction
git checkout parrots