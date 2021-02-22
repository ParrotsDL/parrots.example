#!/bin/sh
whlname=$1
cp /mnt/cache/share/platform/download/${whlname}.whl .
#cp /mnt/lustre/share/platform/download/parrots-pat20200730-py3.6-linux-x86_64.whl .

mv ${whlname}.whl ${whlname}.zip
unzip ${whlname}.zip -d parrots2
cd parrots2
pycfile=`find . -name "*.pyc" |wc -l`
echo "Total pycfile $pycfile"