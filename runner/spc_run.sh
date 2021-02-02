#!/bin/bash
set -x  
NAMESPACE=$1
FRAMEWORK_NAME=$2
MODEL_NAME=$3
GPUS=$4
 
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

# 导入mc的包 

# 首先需要将自己开发机home目录下的petreloss.conf 和 .pavi目录复制到nfs上自己的目录下, 不能是软连接

cp /home/${USER}/petreloss.conf /mnt/lustre/${USER}/petreloss.conf
cp -r /home/${USER}/.pavi /mnt/lustre/${USER}/.pavi

if [ -z ${container_job_name} ];then #“string”的长度为零则为真 取 ${framework_name}_${model_name}_${date} base64编码的前后16位
        # JOB_NAME不能超过64个字符,且不能有下划线
        JOB_NAME="${FRAMEWORK_NAME}_${MODEL_NAME}_`date +%s`"
        JOB_NAME=`echo -n ${FRAMEWORK_NAME}_${MODEL_NAME} | base64`
        jobname1=${JOB_NAME:0:16}
        jobname2=${JOB_NAME:0-18:16}
        JOB_NAME=${jobname1,,}${jobname2,,}
else
        JOB_NAME=${container_job_name}
fi
 
PARTITION=${NAMESPACE}
IMAGE="registry.sensetime.com/parrots/parrots:pat_latest"   #镜像名称可能也需要search_config.yaml中指定
## 资源信息
NODES=$((${GPUS}<=8?1:2))
GPU_PER_NODE=$((${GPUS}>8?8:${GPUS}))       ## 每个节点GPU的计算方法可能也是一个问题，因为有的机器运行着开发机，所以每个节点可能占不到8个GPU
CPU_PER_NODE="`expr 2 \* ${GPU_PER_NODE}`"
MEMORY_PER_NODE="`expr 32 \* ${GPU_PER_NODE}`Gi"
## 训练脚本

#wwl-修改1
#WORKING_DIR="/home/${USER}/parrots.test"    #工作空间可能也需要search_config.yaml中指定
#WORKING_DIR="/mnt/lustre/${USER}/parrots.test"    #工作空间可能也需要search_config.yaml中指定
WORKING_DIR=${PWD} 
TRAIN_SCRIPT="runner/${FRAMEWORK_NAME}/train_mpirun.sh"
TRAIN_SCRIPT_ARGS="${MODEL_NAME} ${NAMESPACE} ${EXTRA_ARGS}"
## 存储挂载
VOLUMES="nfs=/mnt/lustre"
VOLUME_MOUNTS="nfs=/mnt/lustre"
 
  
# 不变的参数
MPIRUN_CMD_BASE_ARGS='-x  CUDA_ROOT=/usr/local/cuda -x    ATEN_ROOT=/usr/local/PatATen  -x   PPLBASE_ROOT=/usr/local/pplbase -x    MPI_ROOT=/usr/local/openmpi-2.1.6-cuda9.0 -x    NCCL_ROOT=/usr/local/nccl-2.4.7-cuda9.0 -x    LD_PRELOAD=/usr/local/openmpi-2.1.6-cuda9.0/lib/libmpi.so --prefix  /usr/local/openmpi-2.1.6-cuda9.0 -x   LD_LIBRARY_PATH=/usr/local/nvidia/lib64/:/usr/lib64:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/nccl-2.4.7-cuda9.0/lib:/usr/local/pplbase/lib:/usr/local/PatATen/lib:/usr/local/cuda/lib64:/usr/local/libmemcached/lib/:/usr/local/memcached_client/lib/:/usr/local/boost/lib/:${LD_LIBRARY_PATH}  -x NCCL_DEBUG=INFO,--hostfile,/etc/mpirun.hosts'
NPERNODE=${GPU_PER_NODE}
NP=`expr ${NODES} \* ${GPU_PER_NODE}`
VOLUME_MOUNTS=${VOLUME_MOUNTS:-${VOLUMES}}
CONTAINER_NAME="parrots"
 
if [ ${PAVI_COMPARE_ID} ];then
        CUSTOM_ENV="PAVI_COMPARE_ID=${PAVI_COMPARE_ID}&AUTOML_SUBTASK_ID=${AUTOML_SUBTASK_ID}"
fi
 
if [ ${PARROTS_BENCHMARK} ];then
        CUSTOM_ENV="PARROTS_BENCHMARK=${PARROTS_BENCHMARK}&${CUSTOM_ENV}"
fi

PARROTS_BENCHMARK=${PARROTS_BENCHMARK:-""}
echo "**********"
echo $PARROTS_BENCHMARK
echo $CUSTOM_ENV

# download spc: wget -O spc http://file.intra.sensetime.com/d/3f3752597f/files/\?p\=/spc/v0.2.1-rc/spc-linux\&dl\=1
spc run mpi-job \
        -j ${JOB_NAME} \
        -N ${PARTITION} \
        -n ${NODES} \
        -i ${IMAGE} \
        -e "${CUSTOM_ENV}" \
        --cmd "/usr/bin/tini" \
        --cmd-args "-g,--,/mpirun_startup.sh" \
        --working-dir ${WORKING_DIR} \
        --mpirun-cmd "mpirun" \
        --mpirun-cmd-args "${MPIRUN_CMD_BASE_ARGS},--np,${NP},--npernode,${NPERNODE}" \
        --train-script ${TRAIN_SCRIPT} \
        --train-script-args "${TRAIN_SCRIPT_ARGS}" \
        --gpus-per-pod ${GPU_PER_NODE} \
        --cpus-per-pod ${CPU_PER_NODE} \
        --mems-per-pod ${MEMORY_PER_NODE} \
        -v ${VOLUMES} \
        -m ${VOLUME_MOUNTS} \
        --container ${CONTAINER_NAME}
 
sleep 5
spc log -N ${NAMESPACE} -p ${JOB_NAME}-1 --sync
