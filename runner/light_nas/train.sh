mkdir -p log/light_nas
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/light_nas
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/light_nas/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
step='search'

case $name in
    "single_path_oneshot_search")
      step=search
      PYTHON_ARGS="python -m main \
         --config=configs/light_nas/single_path_oneshot/search.yaml \
         --step=$step"
      ;; 
    "single_path_oneshot_evolution")
      step=evolution
      PYTHON_ARGS="python -m main \
         --config=configs/light_nas/single_path_oneshot/evolution.yaml \
         --step=$step"
      ;; 
    *)
      echo "invalid $name"
      exit 1
      ;; 
esac

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=light_nas_${name} \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS}\
    $PYTHON_ARGS ${EXTRA_ARGS} \
    2>&1 | tee log/light_nas/train_${step}_${name}.log-$now
