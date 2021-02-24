#!/bin/bash
set -x  
NAMESPACE=$1
FRAMEWORK_NAME=$2
MODEL_NAME=$3
GPUS=$4
 
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

# 首先需要将自己开发机home目录下的petreloss.conf 和 .pavi目录复制到nfs上自己的目录下, 不能是软连接
cp /home/${USER}/petreloss.conf /mnt/lustre/${USER}/petreloss.conf
cp -r /home/${USER}/.pavi /mnt/lustre/${USER}/.pavi

if [ -z ${container_job_name} ];then 
        current=`date "+%Y-%m-%d %H:%M:%S"`
        timeStamp=`date -d "$current" +%s`
        currentTimeStamp=$((timeStamp*1000+10#`date "+%N"`/1000000))
        JOB_NAME=${currentTimeStamp}
else
        JOB_NAME=${container_job_name}
fi
 

PARTITION=${NAMESPACE}
IMAGE="registry.sensetime.com/parrots/parrots:pat_latest"   #镜像名称可能也需要search_config.yaml中指定

## 资源信息
NODES=$((${GPUS}<=8?1:2))
GPU_PER_NODE=$((${GPUS}>8?8:${GPUS}))       ## 每个节点GPU的计算方法可能也是一个问题，因为有的机器运行着开发机，所以每个节点可能占不到8个GPU
CPU_PER_NODE="`expr 2 \* ${GPU_PER_NODE}`"
MEMORY_PER_NODE_VALUE=`expr 30 \* ${GPU_PER_NODE}`   ## 每个模型占用内存大小不同，如果不够再讨论解决办法
MEMORY_PER_NODE="$((${MEMORY_PER_NODE_VALUE}>=32?${MEMORY_PER_NODE_VALUE}:32))Gi"    # 最少分配32G内存(单卡模型)
## 训练脚本
WORKING_DIR=${PWD} 
TRAIN_SCRIPT="runner/${FRAMEWORK_NAME}/train_mpirun.sh"
TRAIN_SCRIPT_ARGS="${MODEL_NAME} ${EXTRA_ARGS}"
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
        MPIRUN_CMD_BASE_ARGS="${MPIRUN_CMD_BASE_ARGS} -x PAVI_COMPARE_ID=${PAVI_COMPARE_ID} -x AUTOML_SUBTASK_ID=${AUTOML_SUBTASK_ID}"
fi
 
if [ ${PARROTS_BENCHMARK} ];then
        MPIRUN_CMD_BASE_ARGS="${MPIRUN_CMD_BASE_ARGS} -x PARROTS_BENCHMARK=1"
fi


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
 
spc log -N ${NAMESPACE} -p ${JOB_NAME}-1 --sync
while (($? != 0))
do
        sleep 1
        spc log -N ${NAMESPACE} -p ${JOB_NAME}-1 --sync
done

