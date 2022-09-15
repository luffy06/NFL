#ifndef NUMERICAL_FLOW_H
#define NUMERICAL_FLOW_H

#include "models/bnaf.h"
#include "util/common.h"

namespace nfl {

template<typename KT, typename VT>
class NumericalFlow {
typedef std::pair<KT, VT> KVT;
typedef std::pair<KT, KVT> KKVT;
public:
  double mean_;
  double var_;
  MKL_INT batch_size_;
  BNAF_Infer<KT, VT> model_;

public:
  explicit NumericalFlow(std::string weight_path, uint32_t batch_size) 
    : batch_size_(batch_size) {
    load(weight_path);
    model_.set_batch_size(batch_size);
  }

  uint64_t size() {
    return sizeof(NumericalFlow<KT, VT>) - sizeof(BNAF_Infer<KT, VT>) + model_.size();
  }

  void set_batch_size(uint32_t batch_size) {
    batch_size_ = batch_size;
    model_.set_batch_size(batch_size_);
  }

  void transform(const KVT* kvs, uint32_t size, KKVT* tran_kvs) {
    for (uint32_t i = 0; i < size; ++ i) {
      tran_kvs[i] = {(kvs[i].first - mean_) / var_, kvs[i]};
    }
    uint32_t num_batches = static_cast<uint32_t>(std::ceil(size * 1. / batch_size_));
    for (uint32_t i = 0; i < num_batches; ++ i) {
      uint32_t l = i * batch_size_;
      uint32_t r = std::min((i + 1) * batch_size_, size);
      model_.transform(tran_kvs + l, r - l);
    }
  }

  KKVT transform(const KVT kv) {
    KKVT t_kv = {(kv.first - mean_) / var_, kv};
    model_.transform(&t_kv, 1);
    return t_kv;
  }

private:
  void load(std::string path) {
    std::fstream in(path, std::ios::in);
    if (!in.is_open()) {
      std::cout << "File:" << path << " doesn't exist" << std::endl;
      exit(-1);
    }
    in >> model_.in_dim_ >> model_.hidden_dim_ >> model_.num_layers_;
    in >> mean_ >> var_;
    model_.weights_ = new double*[model_.num_layers_];
    for (uint32_t w = 0; w < model_.num_layers_; ++ w) {
      uint32_t n, m;
      in >> n >> m;
      model_.weights_[w] = (double*)mkl_calloc(n * m, sizeof(double), 64);
      for (uint32_t i = 0; i < n; ++ i) {
        for (uint32_t j = 0; j < m; ++ j) {
          in >> model_.weights_[w][i * m + j];
        }
      }
    }
    in.close();
  }

};

}

#endif