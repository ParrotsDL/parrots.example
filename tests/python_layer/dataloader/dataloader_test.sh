type=${1:-dfe}
mode=${2:-s}
partition=${3:-Platform}

if [ $mode = s ] || [ $mode = single ]
then
    sh run.sh $type single $partition 1
elif [ $mode = m ] || [ $mode = multi ]
then
    sh run.sh $type multi $partition 1 & sh run.sh $type multi $partition 2
else
    echo invalid argument for second argument, need to be one of ["s", "single", "m", "multi"], but got $mode
fi
