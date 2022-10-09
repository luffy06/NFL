#!/bin/zsh

set -e  # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
flow_dir=${root_dir}/train/data
exec=${build_dir}/nf_convert

# Configurations
req_dist='zipf'
batch_size=256
read_frac_list=(100)
test_workloads=('longitudes-200M' 'longlat-200M' 'lognormal-200M' 'ycsb-200M'
                'books-200M' 'fb-200M' 'wiki-ts-200M')

declare -A key_type
key_type=([longitudes-200M]='float64'
          [longlat-200M]='float64'
          [ycsb-200M]='float64'
          [books-200M]='float64'
          [fb-200M]='float64'
          [wiki-ts-200M]='float64'
          [lognormal-200M]='float64')

# Process on workloads
for read_frac in ${read_frac_list[*]}
do
  for workload in ${test_workloads[*]}
  do
    workload_path=${root_dir}'/workloads/'${workload}'-'${read_frac}'R-'${req_dist}'.bin'
    echo 'Process '${workload_path}
    echo `${exec} ${workload_path} ${key_type[$workload]} 100 ${flow_dir}`
  done
done
