#!/bin/sh
cp /mnt/cache/share/platform/download/parrots-pat20200730-py3.6-linux-x86_64.whl .
#cp /mnt/lustre/share/platform/download/parrots-pat20200730-py3.6-linux-x86_64.whl .

mv parrots-pat20200730-py3.6-linux-x86_64.whl parrots-pat20200730-py3.6-linux-x86_64.zip
unzip parrots-pat20200730-py3.6-linux-x86_64.zip -d parrots2
cd parrots2
pycfile=`find . -name "*.pyc" |wc -l`
echo "Total pycfile $pycfile"