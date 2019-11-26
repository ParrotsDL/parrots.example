source /mnt/lustre/share/platform/env/pat0.4.0b0
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH
name=$1
cfg=$ROOT/configs/${name}.yaml
python -u ../tools/main.py --config ${cfg} $2 $3
