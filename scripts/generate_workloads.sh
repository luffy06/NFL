#!/bin/bash

set -e  # fail and exit on any command erroring

# Path
root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
lib_dir=${root_dir}/lib
data_dir=${root_dir}/data
workload_dir=${root_dir}/workloads
scripts_dir=${root_dir}/scripts
pys_dir=${scripts_dir}/scripts
gen_exec=${build_dir}/gen
format_exec=${build_dir}/format

if [ ! -d ${data_dir} ];
then
  mkdir ${data_dir}
fi

if [ ! -f ${data_dir}/fb_200M_uint64 ];
then
  git clone https://github.com/learnedsystems/SOSD ${lib_dir}/SOSD
  cd ${lib_dir}/SOSD
  bash scripts/download.sh
  mv ${lib_dir}/SOSD/data/lognormal_200M_uint64 ${data_dir}/lognormal_200M_uint64
  mv ${lib_dir}/SOSD/data/books_200M_uint64 ${data_dir}/books_200M_uint64
  mv ${lib_dir}/SOSD/data/fb_200M_uint64 ${data_dir}/fb_200M_uint64
  mv ${lib_dir}/SOSD/data/wiki_ts_200M_uint64 ${data_dir}/wiki_ts_200M_uint64
  rm -rf ${lib_dir}/SOSD
  cd ${root_dir}
fi

if [ ! -f ${data_dir}/longitudes-200M.bin.data ];
then
  echo "Please manually download the longitudes datasets from " \
        "[https://drive.google.com/file/d/1zc90sD6Pze8UM_XYDmNjzPLqmKly8jKl/view]" \
        ", and save it inside "${data_dir}
  exit
fi

if [ ! -f ${data_dir}/longlat-200M.bin.data ];
then
  echo "Please manually download the longlat datasets from " \
        "[https://drive.google.com/file/d/1mH-y_PcLQ6p8kgAz9SB7ME4KeYAfRfmR/view]" \
        ", and save it inside "${data_dir}
  exit
fi

if [ ! -f ${data_dir}/ycsb-200M.bin.data ];
then
  echo "Please manually download the ycsb datasets from "\
        "[https://drive.google.com/file/d/1Q89-v4FJLEwIKL3YY3oCeOEs0VUuv5bD/view]" \
        ", and save it inside "${data_dir}
  exit
fi

if [ ! -d ${workload_dir} ];
then
  mkdir ${workload_dir}
fi

# Configurations
req_dist='zipf'
num_keys=190 # 190 million 
batch_size=256
synthetic=false
gen_source=true
declare -A key_type
key_type=([longitudes-200M]='float64'
          [longlat-200M]='float64'
          [ycsb-200M]='float64'
          [books-200M]='float64'
          [fb-200M]='float64'
          [wiki-ts-200M]='float64'
          [lognormal-200M]='float64')
# Configurations for synthetic dataset
mean=0
var_list=(1 2 4 8)
begin=0
end=2147483648 # 2^31
# Configurations for synthetic workloads
init_frac_list=(0.5)
read_frac_list=(100 80 20 0)
kks_frac_list=(1)
# Configurations for realistic workloads
float64_workloads=('longlat-200M' 'longitudes-200M')
int64_workloads=('ycsb-200M')
uint64_workloads=('fb-200M' 'lognormal-200M' 'books-200M' 'wiki-ts-200M')

for kks_frac in ${kks_frac_list[*]}
do
  for init_frac in ${init_frac_list[*]}
  do
    echo 'Request Distribution ['${req_dist}']'
    # Generate synthetic dataset and workloads
    # Generate synthetic dataset and workloads based on lognormal

    if [ ${synthetic} = true ];
    then
      for var in ${var_list[*]}
      do
        if [ ${gen_source} = true ];
        then
          echo 'Generate synthetic dataset, distribution [lognormal] number keys ['${num_keys}'] mean ['${mean}'] variance ['${var}']'
          echo `${gen_exec} dataset ${data_dir} lognormal ${num_keys} ${mean} ${var}`
        fi
        for read_frac in ${read_frac_list[*]}
        do
          workload='lognormal-'${num_keys}'M-var('${var}')'
          echo 'Generate workload ['${workload}'] with the read fraction ['${read_frac}']'
          echo `${gen_exec} workload ${data_dir} ${workload_dir} keyset ${workload} ${key_type[$workload]} ${req_dist} ${batch_size} ${init_frac} ${read_frac} ${kks_frac}`
        done
      done
    fi

    if [ ${synthetic} = true ];
    then
      # Generate synthetic dataset and workloads based on uniform
      if [ ${gen_source} = true ];
      then
        echo 'Generate synthetic data, distribution [uniform] number keys ['${num_keys}'] begin ['${begin}'] end ['${end}']'
        echo `${gen_exec} dataset ${data_dir} uniform ${num_keys} ${begin} ${end}`
      fi
      for read_frac in ${read_frac_list[*]}
      do
        workload='uniform-'${num_keys}'M'
        echo 'Generate workload ['${workload}'] with the initial fraction ['${init_frac}'] and the read fraction ['${read_frac}']'
        echo `${gen_exec} workload ${data_dir} ${workload_dir} keyset ${workload} ${key_type[$workload]} ${req_dist} ${batch_size} ${init_frac} ${read_frac} ${kks_frac}`
      done
    fi

    for workload in ${uint64_workloads[*]}
    do
      if [ ${gen_source} = true ];
      then
        echo 'Format '${workload}
        echo `${format_exec} ${root_dir} ${workload//'-'/'_'} uint64`
      fi
      for read_frac in ${read_frac_list[*]}
      do
        echo 'Generate workload ['${workload}'] with the initial fraction ['${init_frac}'] and the read fraction ['${read_frac}']'
        echo `${gen_exec} workload ${data_dir} ${workload_dir} keyset ${workload} ${key_type[$workload]} ${req_dist} ${batch_size} ${init_frac} ${read_frac} ${kks_frac}`
      done
    done

    # Generate workloads based on realistic workloads
    for workload in ${float64_workloads[*]}
    do
      if [ ${gen_source} = true ];
      then
        echo 'Format '${workload}
        echo `${format_exec} ${root_dir} ${workload} float64`
      fi
      for read_frac in ${read_frac_list[*]}
      do
        echo 'Generate workload ['${workload}'] with the initial fraction ['${init_frac}'] and the read fraction ['${read_frac}'] and the kks fraction ['${kks_frac}']'
        echo `${gen_exec} workload ${data_dir} ${workload_dir} keyset ${workload} ${key_type[$workload]} ${req_dist} ${batch_size} ${init_frac} ${read_frac} ${kks_frac}`
      done
    done

    # Generate workloads based on realistic workloads
    for workload in ${int64_workloads[*]}
    do
      if [ ${gen_source} = true ];
      then
        echo 'Format '${workload}
        echo `${format_exec} ${root_dir} ${workload} int64`
      fi
      for read_frac in ${read_frac_list[*]}
      do
        echo 'Generate workload ['${workload}'] with the initial fraction ['${init_frac}'] and the read fraction ['${read_frac}'] and the kks fraction ['${kks_frac}']'
        echo `${gen_exec} workload ${data_dir} ${workload_dir} keyset ${workload} ${key_type[$workload]} ${req_dist} ${batch_size} ${init_frac} ${read_frac} ${kks_frac}`
      done
    done
  done
done