#! /bin/bash
count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

echo "start benchmark $count gpus"
python3 benchmark_models.py -g $count 

echo 'benchmark end'
