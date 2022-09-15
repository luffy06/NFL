#ifndef NFL_H
#define NFL_H

#include "afli/afli.h"
#include "afli/iterator.h"
#include "benchmark/workload.h"
#include "models/numerical_flow.h"
#include "util/common.h"

namespace nfl {

template<typename KT, typename VT>
class NFL {
typedef std::pair<KT, VT> KVT;
typedef std::pair<KT, KVT> KKVT;
private:
  AFLI<KT, VT>* index_;
  uint32_t batch_size_;
  KVT* batch_kvs_;

  bool enable_flow_;
  NumericalFlow<KT, VT>* flow_;
  AFLI<KT, KVT>* tran_index_;
  KKVT* tran_kvs_;

  const float kConflictsDecay = 0.1;
  const uint32_t kMaxBatchSize = 4196;
  const float kSizeAmplification = 1.5;
  const float kTailPercent = 0.99;
public:
  explicit NFL(std::string weights_path, uint32_t batch_size) : batch_size_(batch_size) { 
    enable_flow_ = true;
    flow_ = new NumericalFlow<KT, VT>(weights_path, batch_size);
    index_ = nullptr;
    tran_index_ = nullptr;
    tran_kvs_ = nullptr;
    batch_kvs_ = nullptr;
  }

  ~NFL() {
    if (index_ != nullptr) {
      delete index_;
    }
    if (flow_ != nullptr) {
      delete flow_;
    }
    if (tran_index_ != nullptr) {
      delete tran_index_;
    }
    if (tran_kvs_ != nullptr) {
      delete[] tran_kvs_;
    }
    if (batch_kvs_ != nullptr) {
      delete[] batch_kvs_;
    }
  }

  void set_batch_size(uint32_t batch_size) {
    if (batch_size > batch_size_) {
      if (enable_flow_) {
        delete[] tran_kvs_;
        tran_kvs_ = new KKVT[batch_size];
      } else {
        delete[] batch_kvs_;
        batch_kvs_ = new KVT[batch_size];
      }
    }
    batch_size_ = batch_size;
  }

  uint32_t auto_switch(const KVT* kvs, uint32_t size, uint32_t aggregate_size=0) {
    tran_kvs_ = new KKVT[size];
    uint32_t origin_tail_conflicts = compute_tail_conflicts<KT, VT>(kvs, size, kSizeAmplification, kTailPercent);
    flow_->set_batch_size(kMaxBatchSize);
    flow_->transform(kvs, size, tran_kvs_);
    std::sort(tran_kvs_, tran_kvs_ + size, [](const KKVT& a, const KKVT& b) {
      return a.first < b.first;
    });
    uint32_t tran_tail_conflicts = compute_tail_conflicts<KT, KVT>(tran_kvs_, size, kSizeAmplification, kTailPercent);
    if (origin_tail_conflicts <= tran_tail_conflicts
      || origin_tail_conflicts - tran_tail_conflicts 
        < static_cast<uint32_t>(origin_tail_conflicts * kConflictsDecay)) {
      enable_flow_ = false;
      delete[] tran_kvs_;
      tran_kvs_ = nullptr;
      return origin_tail_conflicts;
    } else {
      enable_flow_ = true;
      return tran_tail_conflicts;
    }
  }

  void bulk_load(const KVT* kvs, uint32_t size, uint32_t tail_conflicts, uint32_t aggregate_size=0) {
    if (enable_flow_) {
      tran_index_ = new AFLI<KT, KVT>();
      tran_index_->bulk_load(tran_kvs_, size, tail_conflicts, aggregate_size);
      flow_->set_batch_size(batch_size_);
      delete tran_kvs_;
      tran_kvs_ = new KKVT[batch_size_];
    } else {
      index_ = new AFLI<KT, VT>();
      index_->bulk_load(kvs, size, tail_conflicts, aggregate_size);
      batch_kvs_ = new KVT[batch_size_];      
    }
  }

  void transform(const KVT* kvs, uint32_t size) {
    if (enable_flow_) {
      flow_->transform(kvs, size, tran_kvs_);
    } else {
      std::memcpy(batch_kvs_, kvs, sizeof(KVT) * size);
    }
  }

  ResultIterator<KT, VT> find(uint32_t idx_in_batch) {
    if (enable_flow_) {
      auto it = tran_index_->find(tran_kvs_[idx_in_batch].first);
      if (!it.is_end()) {
        return {it.value_addr()};
      } else {
        return {};
      }
    } else {
      return index_->find(batch_kvs_[idx_in_batch].first);
    }
  }

  bool update(uint32_t idx_in_batch) {
    if (enable_flow_) {
      return tran_index_->update(tran_kvs_[idx_in_batch]);
    } else {
      return index_->update(batch_kvs_[idx_in_batch]);
    }
  }

  uint32_t remove(uint32_t idx_in_batch) {
    if (enable_flow_) {
      return tran_index_->remove(tran_kvs_[idx_in_batch].first);
    } else {
      return index_->remove(batch_kvs_[idx_in_batch].first);
    }
  }

  void insert(uint32_t idx_in_batch) {
    if (enable_flow_) {
      tran_index_->insert(tran_kvs_[idx_in_batch]);
    } else {
      index_->insert(batch_kvs_[idx_in_batch]);
    }
  }

  uint64_t model_size() {
    if (enable_flow_) {
      return tran_index_->model_size() + flow_->size();
    } else {
      return index_->model_size();
    }
  }

  uint64_t index_size() {
    if (enable_flow_) {
      return tran_index_->index_size() + flow_->size() 
            + sizeof(NFL<KT, VT>) + sizeof(KKVT) * batch_size_;
    } else {
      return index_->index_size() + sizeof(NFL<KT, VT>) + sizeof(KVT) * batch_size_;
    }
  }

  void print_stats() {
    if (enable_flow_) {
      tran_index_->print_stats();
    } else {
      index_->print_stats();
    }
  }
};

}

#endif