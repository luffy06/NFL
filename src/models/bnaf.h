#ifndef BNAF_H
#define BNAF_H

#include "util/common.h"

#include <mkl.h>
#include <mkl_cblas.h>

namespace nfl {

template<typename KT, typename VT>
class BNAF_Infer {
typedef std::pair<KT, VT> KVT;
typedef std::pair<KT, KVT> KKVT;
public:
  int num_layers_;
  MKL_INT batch_size_;
  MKL_INT in_dim_;
  MKL_INT hidden_dim_;
  double** weights_;
  // 1: in_dim_ * hidden_dim_
  // 2: hidden_dim_ * hidden_dim_
  // ....
  // n: hidden_dim_ * in_dim_
  double* inputs_;
  double* outputs_[2];
public:
  BNAF_Infer() : inputs_(nullptr), weights_(nullptr) {
    outputs_[0] = nullptr;
    outputs_[1] = nullptr;
  }

  ~BNAF_Infer() {
    for (int i = 0; i < num_layers_; ++ i) {
      if (weights_[i] != nullptr) {
        mkl_free(weights_[i]);
      }
    }
    if (inputs_ != nullptr) {
      mkl_free(inputs_);
    }
    if (outputs_[0] != nullptr) {
      mkl_free(outputs_[0]);
    }
    if (outputs_[1] != nullptr) {
      mkl_free(outputs_[1]);
    }
  }

  uint64_t model_size() {
    return 0;
  }

  uint64_t size() {
    return sizeof(BNAF_Infer<KT, VT>) + sizeof(double*) * num_layers_ 
          + sizeof(double) * (batch_size_ * in_dim_ + batch_size_ * hidden_dim_ * 2)
          + sizeof(double) * (in_dim_ * hidden_dim_ * 2 + (num_layers_ - 2) * hidden_dim_ * hidden_dim_);
  }

  void set_batch_size(uint32_t batch_size) {
    batch_size_ = batch_size;
    if (inputs_ != nullptr) {
      mkl_free(inputs_);
    }
    if (outputs_[0] != nullptr) {
      mkl_free(outputs_[0]);
    }
    if (outputs_[1] != nullptr) {
      mkl_free(outputs_[1]);
    }
    inputs_= (double*)mkl_calloc(batch_size_ * in_dim_, sizeof(double), 64);
    outputs_[0] = (double*)mkl_calloc(batch_size_ * hidden_dim_, sizeof(double), 64);
    outputs_[1] = (double*)mkl_calloc(batch_size_ * hidden_dim_, sizeof(double), 64);
  }

  void transform(KKVT* tran_kvs, uint32_t size) {
    prepare_inputs(tran_kvs, size);
    forward();
    prepare_outputs(tran_kvs, size);
  }
  void print_parameters() {
    std::cout << "Layers\t" << num_layers_ << std::endl;
    std::cout << "Input Dim\t" << in_dim_ << std::endl;
    std::cout << "Hidden Dim\t" << hidden_dim_ << std::endl;
    for (int i = 0; i < num_layers_; ++ i) {
      print_weight_matrix(i);
    }
  }

private:
  void prepare_inputs(const KKVT* tran_kvs, uint32_t size) {
    if (in_dim_ == 1) {
      for (uint32_t i = 0; i < size; ++ i) {
        inputs_[i] = tran_kvs[i].first;
      }
    } else if (in_dim_ == 2) {
      for (uint32_t i = 0; i < size; ++ i) {
        inputs_[2 * i] = tran_kvs[i].first;
        inputs_[2 * i + 1] = tran_kvs[i].first - std::floor(inputs_[2 * i]);
      }
    } else if (in_dim_ == 4) {
      for (uint32_t i = 0; i < size; ++ i) {
        inputs_[4 * i] = tran_kvs[i].first;
        inputs_[4 * i + 1] = std::floor(inputs_[2 * i]);
        double tmp = (tran_kvs[i].first - inputs_[4 * i + 1]) * 1000000;
        inputs_[4 * i + 2] = std::floor(tmp);
        inputs_[4 * i + 3] = tmp - inputs_[4 * i + 2];
      }
    } else {
      std::cout << "Unsupported dimensions\t" << in_dim_ << std::endl;
      exit(-1);
    }
  }

  void prepare_outputs(KKVT* tran_kvs, uint32_t size) {
    if (in_dim_ == 1) {
      for (uint32_t i = 0; i < size; ++ i) {
        tran_kvs[i] = {inputs_[i], tran_kvs[i].second};
      }
    } else if (in_dim_ == 2) {
      for (uint32_t i = 0; i < size; ++ i) {
        tran_kvs[i] = {inputs_[i * 2] + inputs_[i * 2 + 1], tran_kvs[i].second};
      }
    } else if (in_dim_ == 4) {
      for (uint32_t i = 0; i < size; ++ i) {
        tran_kvs[i] = {inputs_[i * 4] + inputs_[i * 4 + 1] + inputs_[i * 4 + 2] + inputs_[i * 4 + 3], tran_kvs[i].second};
      }
    } else {
      std::cout << "Unsupported dimensions\t" << in_dim_ << std::endl;
      exit(-1);
    }
  }

  void forward() {
    // print_outputs(-1, inputs_, batch_size_, in_dim_);
    // Compute the formula: 
    //            alpha * mat_a [m * k] * mat_b [k * n] + beta * mat_c [m * n]
    // cblas_dgemm(layout, trans_a, trans_b, m, n, k, alpha, mat_a, lda, 
    //              mat_b, ldb, beta, mat_c, ldc)
    // IN [batch_size_ * in_dim] * W_0 [in_dim * hidden_dim] = 
    // OUT [batch_size_ * hidden_dim]
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                batch_size_, hidden_dim_, in_dim_, 
                1, inputs_, in_dim_, 
                weights_[0], hidden_dim_, 
                0, outputs_[0], hidden_dim_);
    // print_weight_matrix(0);
    // print_outputs(0, outputs_[0], batch_size_, hidden_dim_);
    vdTanh(batch_size_ * hidden_dim_, outputs_[0], outputs_[1]);
    // print_outputs(0, outputs_[1], batch_size_, hidden_dim_);
    for (int i = 1; i < num_layers_ - 1; ++ i) {
      // IN [batch_size_ * hidden_dim] * W_i [hidden_dim * hidden_dim] = 
      // OUT [batch_size_ * hidden_dim]
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                  batch_size_, hidden_dim_, hidden_dim_, 
                  1, outputs_[1], hidden_dim_, 
                  weights_[i], hidden_dim_, 
                  0, outputs_[0], hidden_dim_);
      // print_weight_matrix(i);
      // print_outputs(i, outputs_[0], batch_size_, hidden_dim_);      
      vdTanh(batch_size_ * hidden_dim_, outputs_[0], outputs_[1]);
      // print_outputs(i, outputs_[1], batch_size_, hidden_dim_);      
    }
    // IN [batch_size_ * hidden_dim] * W_L [hidden_dim * in_dim] = 
    // OUT [batch_size_ * in_dim]
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                batch_size_, in_dim_, hidden_dim_, 
                1, outputs_[1], hidden_dim_, 
                weights_[num_layers_ - 1], in_dim_, 
                0, inputs_, in_dim_);
    // print_weight_matrix(num_layers_);
    // print_outputs(num_layers_, inputs_, batch_size_, in_dim_);
  }

  void print_weight_matrix(int l) {
    if (l == 0) {
      std::cout << std::fixed << "Weight (0)" << std::endl;
      for (int i = 0; i < in_dim_; ++ i) {
        for (int j = 0; j < hidden_dim_; ++ j) {
          std::cout << std::fixed << weights_[0][i * hidden_dim_ + j] << "\t";
        }
        std::cout << std::endl;
      }
    } else if (l == num_layers_) {
      std::cout << std::fixed << "Weight (" << num_layers_ << ")" << std::endl;
      for (int i = 0; i < hidden_dim_; ++ i) {
        for (int j = 0; j < in_dim_; ++ j) {
          std::cout << std::fixed << weights_[num_layers_][i * in_dim_ + j] << "\t";
        }
        std::cout << std::endl;
      }    
    } else {
      std::cout << std::fixed << "Weight (" << l << ")" << std::endl;
      for (int i = 0; i < hidden_dim_; ++ i) {
        for (int j = 0; j < hidden_dim_; ++ j) {
          std::cout << std::fixed << weights_[l][i * hidden_dim_ + j] << "\t";
        }
        std::cout << std::endl;
      }
    }
  }

  void print_outputs(int idx, double* outputs, int num_rows, int num_columns) {
    if (idx == -1) {
      std::cout << "Input" << std::endl;
    } else {
      std::cout << "Output (" << idx << ")" << std::endl;
    }
    // std::cout << "Shape\t" << num_rows << "*" << num_columns << std::endl;
    for (int i = 0; i < num_rows; ++ i) {
      for (int j = 0; j < num_columns; ++j) {
        std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) 
                  << outputs[i * num_columns + j] << "\t";
      }
      std::cout << std::endl;
    }
  }

};

}

#endif