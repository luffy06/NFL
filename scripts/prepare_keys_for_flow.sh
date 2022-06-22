#!/bin/zsh

set -e  # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
exec=${build_dir}/nf_convert

# Configurations
req_dist='zipf'
batch_size=256
read_frac_list=(80 20) # 80 20 0)
init_frac_list=(100 80 40 20 10 1)
kks_frac_list=(50)
test_workloads=('longlat-200M' 'fb-200M')
                # 'longitudes-200M' 'longlat-200M' 'lognormal-200M' 'ycsb-200M'
                # 'books-200M' 'fb-200M' 'wiki-ts-200M')

declare -A key_type
key_type=([longitudes-200M]='float64'
          [longlat-200M]='float64'
          [ycsb-200M]='float64'
          [books-200M]='float64'
          [fb-200M]='float64'
          [wiki-ts-200M]='float64'
          [lognormal-200M]='float64')

# Process on workloads
# for init_frac in ${init_frac_list[*]}
# do
for kks_frac in ${kks_frac_list[*]}
do
  for read_frac in ${read_frac_list[*]}
  do
    for workload in ${test_workloads[*]}
    do
      # workload_name=${workload}'-'${init_frac}'I-'${read_frac}'R-'${kks_frac}'K-'${req_dist}
      workload_name=${workload}'-'${read_frac}'R-'${kks_frac}'K-'${req_dist}
      echo 'Process '${workload_name}
      echo `${exec} ${workload_name} ${key_type[$workload]} 100`
    done
  done
done
