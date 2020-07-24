type=$1
mode=$2
partition=$3
num=$4

export PYTHONPATH=$(cd `dirname $0`; pwd):$PYTHONPATH

if [ $type = as ] || [ $type = after_shutdown ] || [ $type = sar ] || [ $type = shutdown_and_restart ]
then
    srun --mpi=pmi2 -p $partition --job-name=$type \
    --gres=gpu:8 -n1 --ntasks-per-node=8 \
    python dataloader_test.py --type $type --model $num --$mode

    if [ $mode = single ]
    then
        mode_info="single model."
    else
        mode_info="multi-models No.$num / total: 2."
    fi

    for pid in `cat $num.txt`
    do
        PID_EXIST=$(ps aux | awk '{print $2}'| grep -w $pid)
        if [ $PID_EXIST ]
        then
            if [ $type = as ] || [ $type = after_shutdown ]
            then
                type_info=after_shutdown
            else
                type_info=shutdown_and_restart
            fi

            echo Test Failed! Test: $type_info for $mode_info Some process has not been shutdown
            rm -f $num.txt
            exit
        fi
    done
    rm -f $num.txt

    if [ $type = sar ] || [ $type = shutdown_and_restart ]
    then
            srun --mpi=pmi2 -p $partition --job-name=$type \
        --gres=gpu:8 -n1 --ntasks-per-node=8 \
        python dataloader_test.py --type $type --model $num --$mode --restart
    else
        echo Test Passed! Test: after_shutdown for $mode_info
    fi

else
    srun --mpi=pmi2 -p $partition --job-name=$type \
    --gres=gpu:8 -n1 --ntasks-per-node=8 \
    python dataloader_test.py --type $type --model $num --$mode
fi
