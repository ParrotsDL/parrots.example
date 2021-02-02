set -x
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

JOB_NAME="123456789"
jobname2=${JOB_NAME:0-9:4}
echo $jobname2