models_list=(
"mmcls aaaa"
"mmdet bbbb"
"mmdet dddd"
"mmdet ffff"
"mmdet eeee"
)
models_num=${#models_list[@]}
max_parall=2

mkfifo ./fifo.$$ && exec 798<> ./fifo.$$ && rm -f ./fifo.$$
for ((i=0; i<$max_parall; i++)); do
    echo  "init add placed row $i" >&798
done

for ((i=0; i<$models_num; i++)); do
{
    read -u 798
    read frame model <<< ${models_list[i]}
    echo $frame xxxxx $model
    sleep $i
    echo  "after add place row $i"  1>&798
}&
done

wait

echo Done
