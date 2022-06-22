#!/bin/bash

set -e  # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
lib_dir=${root_dir}/lib

# Create the directory of libraries
if [ ! -d ${lib_dir} ];
then
  mkdir ${lib_dir}  
fi

# Download source code of related works, e.g., PGM-Index, Alex, lipp, B-Tree
if [ ! -d ${lib_dir}/PGM-index ];
then
  git clone https://github.com/luffy06/PGM-index ${lib_dir}/PGM-index
fi

if [ ! -d ${lib_dir}/ALEX ];
then
  git clone https://github.com/luffy06/ALEX ${lib_dir}/ALEX
fi

if [ ! -d ${lib_dir}/lipp ];
then
  git clone https://github.com/luffy06/lipp ${lib_dir}/lipp
fi

if [ ! -d ${lib_dir}/BTree ];
then
  wget -P ${lib_dir} https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cpp-btree/cpp-btree-1.0.1.tar.gz
  tar -zxvf ${lib_dir}/cpp-btree-1.0.1.tar.gz -C ${lib_dir}
  mv ${lib_dir}/cpp-btree-1.0.1 ${lib_dir}/BTree
  rm ${lib_dir}/cpp-btree-1.0.1.tar.gz
fi

# Build source codes
if [ -d ${build_dir} ];
then
  rm -rf ${build_dir}
fi
mkdir -p ${build_dir} && cd ${build_dir}
cmake ${root_dir} && cmake --build ${build_dir}

