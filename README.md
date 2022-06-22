# Introduction
NFL is a two-stage learned index framework, consisting of a normalizing flow that performs a **Distribution Transformation** on the key distribution and a learned index that achieves excellent performance on the near-uniform key distritbuion with small buffers.

Early access: https://arxiv.org/abs/2205.11807

# Requirements

* Intel MKL 11.3
* CMake 3.12
* GNU C++ 17
* OpenMP

# Bootstrap

```bash
$ source ~/intel/oneapi/setvars.sh --force intel64
$ bash bootstrap.sh
```

# Evaluation
