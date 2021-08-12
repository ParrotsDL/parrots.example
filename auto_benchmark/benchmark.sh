set -x
if  [ -f  "auto_benchmark/merged.sh"  ]; then
rm auto_benchmark/merged.sh
echo  "rm old version merged.sh"
else
echo "build new merged.sh"
fi
python $(cd `dirname $0`; pwd)/auto_gen_benchhmark.py
export PYTHONPATH=$PWD/auto_benchmark/:$PYTHONPATH
echo "!!! pavi's file is" && python -c 'import pavi;print(pavi.__file__)'
#python auto_benchmark/start.py $1
