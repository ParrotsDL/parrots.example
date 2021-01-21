NAMESPACE="wenli"

JOB_NAME="${NAMESPACE}-test"

export container_job_name=${JOB_NAME}

spc cancel mpi-job -j "${JOB_NAME}" -N ${NAMESPACE}
spc cancel pod -p "${JOB_NAME}-1" -N ${NAMESPACE}
spc cancel pod -p "${JOB_NAME}-2" -N ${NAMESPACE}

sh runner/mmediting/spc_run.sh wenli  mmediting srcnn_x4k915_g1_1000k_div2k  1  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500

spc log -N ${NAMESPACE} -p ${JOB_NAME}-1 --sync