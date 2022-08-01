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
$ bash scripts/evaluate.sh
```
