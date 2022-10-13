#!/bin/bash

root_dir=$(cd `dirname $0`/..; pwd)
weight_dir=${root_dir}/flow_weights

# Parameters for train the numerical flow
data_dir=data
req_dist='zipf'
test_workloads=('longitudes-200M' 'longlat-200M' 'ycsb-200M' 'lognormal-200M'
                'books-200M' 'fb-200M' 'wiki-ts-200M')
declare -A seeds
seeds=([longitudes-200M]=1000000013
        [longlat-200M]=1000000007
        [ycsb-200M]=1000000013
        [lognormal-200M]=1000000007
        [books-200M]=1000000007
        [fb-200M]=1000000003
        [wiki-ts-200M]=1000000013)
en_type='partition'
de_type='sum'
declare -A shifts
shifts=([longitudes-200M]=100000
        [longlat-200M]=1000000
        [ycsb-200M]=1000000
        [lognormal-200M]=1000000
        [books-200M]=1000000
        [fb-200M]=10000000
        [wiki-ts-200M]=1000000)
num_flows=1
num_layers=2
input_dim=2
hidden_dim=1
train_ratio=0.1
num_train=3
learning_rate=0.1
steps=15
batch_dim=4096
loss_func='normal'
output_dir=checkpoint

if [ -d ${output_dir} ];
then
  rm -rf ${output_dir}
fi

for workload in ${test_workloads[*]}
do
  data_name=${workload}'-100R-'${req_dist}
  python3 numerical_flow.py --data_name=${data_name} \
            --seed=${seeds[$workload]} --encoder_type=${en_type} \
            --decoder_type=${de_type} --shifts=${shifts[$workload]} \
            --num_flows=${num_flows} --num_layers=${num_layers} \
            --input_dim=${input_dim} --hidden_dim=${hidden_dim} \
            --train_ratio=${train_ratio} --num_train=${num_train} \
            --learning_rate=${learning_rate} --steps=${steps} \
            --batch_dim=${batch_dim} --loss_func=${loss_func}
  hidden_dim_actual=$(($hidden_dim * $input_dim))
  cp ${output_dir}/${data_name}-*/${data_name}-weights.txt ${weight_dir}/${workload}-100R-${req_dist}_${input_dim}D${hidden_dim_actual}H${num_layers}L_weights.txt
  cp ${output_dir}/${data_name}-*/${data_name}-weights.txt ${weight_dir}/${workload}-80R-${req_dist}_${input_dim}D${hidden_dim_actual}H${num_layers}L_weights.txt
  cp ${output_dir}/${data_name}-*/${data_name}-weights.txt ${weight_dir}/${workload}-20R-${req_dist}_${input_dim}D${hidden_dim_actual}H${num_layers}L_weights.txt
  cp ${output_dir}/${data_name}-*/${data_name}-weights.txt ${weight_dir}/${workload}-0R-${req_dist}_${input_dim}D${hidden_dim_actual}H${num_layers}L_weights.txt
done