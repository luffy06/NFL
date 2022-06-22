#!/bin/bash

set -e  # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
trace_dir=${root_dir}/trace
pys_dir=${root_dir}/scripts
result_dir=${root_dir}/results

# Configurations
metrics=('bulkloading' 'throughput' 'avg-latency' 'p99-latency' 'p9999-latency' 'max-latency' 'index-size')
batch_size_list=(256)
req_dist='zipf'
flow_para='2D2H2L'
# Configurations for workloads
read_frac_list=(100 80 20 0)
workloads=('longitudes-200M' 'longlat-200M' 'lognormal-200M' 'ycsb-200M'
            'books-200M' 'fb-200M' 'wiki-ts-200M')

output_path=${root_dir}/results/${req_dist}_workload_${flow_para}_results.txt
echo '' > ${output_path}

# Load results
for batch_size in ${batch_size_list[*]}
do
  for metric in ${metrics[*]}
  do
    for read_frac in ${read_frac_list[*]}
    do
      echo ${metric}'-'${batch_size} >> ${output_path}
      for workload in ${workloads[*]}
      do
        echo ${workload}'-'${read_frac}'R' >> ${output_path}
      done
      for workload in ${workloads[*]}
      do
        result_path=${result_dir}'/'${workload}'-'${req_dist}'.txt'
        workload_name=${workload}'-'${read_frac}'R-'${req_dist}
        python3 ${pys_dir}/load_result.py ${result_path} ${workload_name} ${batch_size} ${metric} ${output_path}
      done
      echo '' >> ${output_path}
    done
  done
done
