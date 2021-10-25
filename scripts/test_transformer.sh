CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(readlink -f $0)))
IWSLT_CORPUS_PATH=/mnt/lustre/share_data/jiangyongjiu/datasets/

CORPORA_PATH=${EXAMPLES_DIR}/corpora

PYTHON_VERSION=`python -c 'import sys; version=sys.version_info[:3]; \
                               print("{0}.{1}.{2}".format(*version))'`
python_version=${PYTHON_VERSION:0:1}

# env
if [ -z ${IWSLT_CORPUS_PATH} ]; then
  echo "please set environment variable IWSLT_CORPUS_PATH."
  exit 1
fi

if [ ! -d ${CORPORA_PATH} ]; then
  ln -s ${IWSLT_CORPUS_PATH} ${CORPORA_PATH}
fi

# param
device='MLU'
iterations=10
num_epochs=10
resume="model_epoch_09.pth"
ckpt_dir=${EXAMPLES_DIR}/ckpt_model
log_dir=${EXAMPLES_DIR}/logs
DISTRIBUTED_TYPE=false
EVAL_TYPE=false


run_cmd="train.py  \
  --log-path ${log_dir} \
  --num_epochs ${num_epochs} \
  --iterations ${iterations} \
  --print-freq 1 \
  --dropout_rate 0.0"

args_cmd="--distributed"
distributed_cmd="-m torch.distributed.launch --nproc_per_node=4 --use_env"
check_cmd="scripts/compute_R2.py ${CKPT_MODEL_PATH}/logs/ ${log_dir} ${num_epochs}"
if [ $python_version -eq 2 ];then
  check_cmd="scripts/compute_R2.py ${CKPT_MODEL_PATH}/py2logs/ ${log_dir} ${num_epochs}"
fi

function usage
{
    echo "Usage:"
    echo "  $0 [-help|-h] [0|1] [-distributed|-d] [-eval|-e]"
    echo ""
    echo "  Parameter description:"
    echo "    -help or -h: usage instructions."
    echo "    parameter1: device type, 0)MLU, 1)GPU."
    echo "    -distributed or -d: distributed training."
    echo "    -eval or -e: select the method of accuracy verification."
		echo "         Default comparison degree of relative relationship of loss."
		echo "         Selecting the change mode will verify the reasoning accuracy."
}

BENCHMARK=0
while [[ $# -ge 1 ]]
do
    arg="$1"
    case $arg in
        -help | -h)
            usage
            exit 0
            ;;
        0)
            device='MLU'
            run_cmd="$run_cmd --device $device"
            ;;
        1)
            device='GPU'
            run_cmd="$run_cmd --device $device"
            ;;
        -distributed | -d)
            DISTRIBUTED_TYPE=true
            run_cmd="$run_cmd $args_cmd"
            check_cmd="scripts/compute_R2.py \
                ${CKPT_MODEL_PATH}/ddp_logs/ ${log_dir} ${num_epochs}"
            if [ $python_version -eq 2 ];then
                run_cmd="$distributed_cmd $run_cmd"
                check_cmd="scripts/compute_R2.py \
                  ${CKPT_MODEL_PATH}/py2ddp_logs/ ${log_dir} ${num_epochs}"
            fi
            ;;
        -eval | -e)
            EVAL_TYPE=true
            iterations=1000
            run_cmd="$run_cmd --iterations ${iterations}"
            check_cmd="eval.py \
              --pretrained ${EXAMPLES_DIR}/models/model_epoch_10.pth \
              --device $device"

            ;;

        -precheckin)
            EVAL_TYPE=true
            iterations=2
            run_cmd="$run_cmd --iterations ${iterations}"
            check_cmd="eval.py \
              --pretrained ${EXAMPLES_DIR}/models/model_epoch_10.pth --iterations ${iterations} \
              --device $device"
            ;;
        -daily)
            EVAL_TYPE=true
            iterations=1000
            run_cmd="$run_cmd --iterations ${iterations}"
            check_cmd="eval.py \
              --pretrained ${EXAMPLES_DIR}/models/model_epoch_10.pth \
              --device $device"
            ;;
        -weekly)
            EVAL_TYPE=true
            iterations=-1
            run_cmd="$run_cmd --iterations ${iterations}"
            check_cmd="eval.py \
              --pretrained ${EXAMPLES_DIR}/models/model_epoch_10.pth \
              --device $device"
            ;;
        -benchmark)
            BENCHMARK=1
            export MLU_ADAPTIVE_STRATEGY_COUNT=100
            iterations=`expr ${MLU_ADAPTIVE_STRATEGY_COUNT} + 200`
            num_epochs=1
            run_cmd="$run_cmd --iterations ${iterations} --num_epochs ${num_epochs}"
            ;;
        *)
            echo "[ERROR] Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
    shift
done

check_status() {
    if (($?!=0)); then
        echo "transformer network training failed!"
        exit -1
    fi
}

rm -rf ${log_dir}

if ((${BENCHMARK}==0)); then
  run_cmd="$run_cmd --resume ${resume}"
fi

pushd $EXAMPLES_DIR
echo $run_cmd
eval "python $run_cmd"
check_status
popd

# Trainging perf benchmark mode bypass $check_cmd
if ((${BENCHMARK}==1)); then
    exit
fi

# R2
pushd $EXAMPLES_DIR
echo $check_cmd
time $check_cmd
# eval "python $check_cmd"
# check_status
# popd
