#!/bin/bash

set -e  # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
lib_dir=${root_dir}/lib
config_dir=${root_dir}/configs
data_dir=${root_dir}/data
workload_dir=${root_dir}/workloads

if [ -d ${lib_dir}/PGM-index ];
then
  rm -rf ${lib_dir}/PGM-index
fi

if [ -d ${lib_dir}/ALEX ];
then
  rm -rf ${lib_dir}/ALEX
fi

if [ -d ${lib_dir}/lipp ];
then
  rm -rf ${lib_dir}/lipp
fi

if [ -d ${lib_dir}/BTree ];
then
  rm -rf ${lib_dir}/BTree
fi

if [ -d ${build_dir} ];
then
  rm -rf ${build_dir}
fi

if [ -d ${config_dir} ];
then
  rm -rf ${config_dir}
fi

if [ -d ${data_dir} ];
then
  rm -rf ${data_dir}/*.bin
  rm -rf ${data_dir}/*_uint64
fi

if [ -d ${workload_dir} ];
then
  rm -rf ${workload_dir}
fi


