# Introduction
NFL is a two-stage learned index framework, consisting of a normalizing flow that performs a **Distribution Transformation** on the key distribution and a learned index that achieves excellent performance on the near-uniform key distritbuion with small buffers.

Early access: https://arxiv.org/abs/2205.11807

# Requirements

* Intel MKL 11.3
* CMake 3.12
* GNU C++ 17
* OpenMP

# Getting Started

Acitivating intel mkl.
```bash
$ source ~/intel/oneapi/setvars.sh --force intel64
```

Downloading libraries and compiling codes.
```bash
$ bash scripts/bootstrap.sh
```

Generating workloads and configs.
```bash
$ bash scripts/generate_workloads.sh
$ bash scripts/generate_configs.sh
```

Reproducing results.
```bash
$ bash scripts/benchmark.sh
```

Clearing.
```bash
$ bash scripts/clear.sh
```

# Training

To train our numerical flowl, please follow the guideline in the `train` directory.

First, run the script to generate training data for the flow.
```bash
$ bash scripts/prepare_keys_for_flows.sh
```

Then, run the script in the `train` directory.
```bash
$ bash train/train_flow.sh
```

# Results

The results are shown in the following format.
```txt
(dataset name) (index name) (batch size) (bulk loading time) (transformation time in bulk loading) (model size) (index size) (overall throughput) (avg-T) (avg-I) (50-T) (50-I) (75-T) (75-I) (99-T) (99-I) (995-T) (995-I) (9999-T) (9999-I) (max-T) (max-I)
```
where 'T' represents the transformation time, 'I' represents the indexing latency.

# Contact

Please be free to contact us via shangyuwu2-c@my.cityu.edu.hk.