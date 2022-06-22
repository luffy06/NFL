#!/bin/bash

set -e # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
exec=${build_dir}/evaluate
config_dir=${root_dir}/configs
workload_dir=${root_dir}/workloads
weights_dir=${root_dir}/flow_weights

# Configurations
algorithms=('nfl' 'afli' 'lipp' 'alex' 'pgm-index' 'btree')
batch_size_list=(256)
req_dists=('zipf')
flow_para='2D2H2L'
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

if [ ! -d ${config_dir} ];
then
  mkdir ${config_dir}
fi

# Generate config for pgm
for req_dist in ${req_dists[*]}
do
  for workload in ${test_workloads[*]}
  do
    workload_name=${workload}'-100R-'${req_dist}
    config_path=${config_dir}'/pgm-index_'${workload_name}'.in'
    echo 'base=64' > ${config_path}

    workload_name=${workload}'-80R-'${req_dist}
    config_path=${config_dir}'/pgm-index_'${workload_name}'.in'
    echo -e 'base=32\nbuffer_level=1\nindex_level=5' > ${config_path}
    
    workload_name=${workload}'-20R-'${req_dist}
    config_path=${config_dir}'/pgm-index_'${workload_name}'.in'
    echo -e 'base=4\nbuffer_level=3\nindex_level=9' > ${config_path}
    
    workload_name=${workload}'-0R-'${req_dist}
    config_path=${config_dir}'/pgm-index_'${workload_name}'.in'
    echo -e 'base=2\nbuffer_level=4\nindex_level=26' > ${config_path}
    
  done
done

# Generate config for nfl
for req_dist in ${req_dists[*]}
do
  for read_frac in ${read_frac_list[*]}
  do
    for workload in ${test_workloads[*]}
    do
      workload_name=${workload}'-'${read_frac}'R-'${req_dist}
      config_path=${config_dir}'/nfl_'${workload_name}'.in'
      weights_path=${weights_dir}'/'${workload_name}'_'${flow_para}'_weights.txt'
      echo 'weights_path='${weights_path} > ${config_path}
    done
  done
done
