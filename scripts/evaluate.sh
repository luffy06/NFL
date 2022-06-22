#!/bin/bash

set -e # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
exec=${build_dir}/evaluate
result_dir=${root_dir}/results
workload_dir=${root_dir}/workloads
config_dir=${root_dir}/configs

# Configurations
algorithms=('nfl' 'afli' 'lipp' 'alex' 'pgm-index' 'btree')
batch_size_list=(256)
req_dists=('zipf') # 'uniform')
repeat=3
declare -A key_type
key_type=([longitudes-200M]='float64'
          [longlat-200M]='float64'
          [ycsb-200M]='float64'
          [books-200M]='float64'
          [fb-200M]='float64'
          [wiki-ts-200M]='float64'
          [lognormal-200M]='float64')
# Configurations for workloads
read_frac_list=(100 80 20 0)
test_workloads=('longitudes-200M' 'longlat-200M' 'lognormal-200M' 'ycsb-200M'
                'books-200M' 'fb-200M' 'wiki-ts-200M')

if [ ! -d ${result_dir} ];
then
  mkdir ${result_dir}
fi

start_time=`echo $(($(date +%s%N)/1000000))`
for req_dist in ${req_dists[*]}
do
  # Initialize result files
  for workload in ${test_workloads[*]}
  do
    result_path=${result_dir}'/'${workload}'-'${req_dist}'.txt'
    echo '' > ${result_path}
  done
  # Evaluate workloads
  for batch_size in ${batch_size_list[*]}
  do
    for read_frac in ${read_frac_list[*]}
    do
      for workload in ${test_workloads[*]}
      do
        for algo in ${algorithms[*]}
        do
          workload_name=${workload}'-'${read_frac}'R-'${req_dist}
          workload_path=${workload_dir}'/'${workload_name}'.bin'
          config_path=${config_dir}'/'${algo}'_'${workload_name}'.in'
          result_path=${result_dir}'/'${workload}'-'${req_dist}'.txt'
          echo 'Batch size ['${batch_size}'] Algorithm ['${algo}'] Workload ['${workload_name}']'
          for i in $(seq 1 $repeat);
          do
            echo -e `${exec} ${algo} ${batch_size} ${workload_path} ${key_type[$workload]} ${config_path}` >> ${result_path}
          done
        done
      done
    done
  done
done
end_time=`echo $(($(date +%s%N)/1000000))`
time_cost=$((${end_time} - ${start_time}))
echo 'Time Cost '$((${time_cost}/1000))'(s)'