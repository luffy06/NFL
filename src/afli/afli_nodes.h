#ifndef AFLI_NODES_H
#define AFLI_NODES_H

#include "afli/buckets.h"
#include "afli/conflicts.h"
#include "models/linear_model.h"
#include "util/common.h"

#define BIT_TYPE uint8_t
#define BIT_SIZE (sizeof(BIT_TYPE) * 8)
#define BIT_LEN(x) (std::ceil((x) * 1. / BIT_SIZE))
#define BIT_IDX(x) ((x) / BIT_SIZE)
#define BIT_POS(x) ((x) % BIT_SIZE)
#define SET_BIT_ONE(x, n) ((x) |= (1 << (n)))
#define SET_BIT_ZERO(x, n) ((x) &= (~(1 << (n))))
#define REV_BIT(x, n) ((x) ^= (1 << (n)))
#define GET_BIT(x, n) (((x) >> (n)) & 1)

namespace nfl {

template<typename KT, typename VT>
class TNode;

struct HyperParameter {
  // Parameters
  uint32_t max_bucket_size_ = 6;
  uint32_t aggregate_size_ = 0;
  // Constant parameters
  const uint32_t kMaxBucketSize = 6;
  const uint32_t kMinBucketSize = 1;
  const double kSizeAmplification = 2;
  const double kTailPercent = 0.99;
};

enum EntryType {
  kNone = 0,
  kData = 1,
  kBucket = 2,
  kNode = 3
};

template<typename KT, typename VT>
union Entry {
  Bucket<KT, VT>*     bucket_;      // The bucket pointer.
  TNode<KT, VT>*      child_;       // The child node pointer.
  std::pair<KT, VT>   kv_;

  Entry() { }
};

template<typename KT, typename VT>
class TNode {
typedef std::pair<KT, VT> KVT;
public:
  LinearModel<KT>*    model_;
  uint32_t            size_;
  uint32_t            capacity_;
  uint32_t            size_sub_tree_;
  uint8_t*            bitmap0_;     // The i-th bit indicates whether the i-th 
                                    // position has a bucket or a child node.
  uint8_t*            bitmap1_;     // The i-th bit indicates whether the i-th 
                                    // position is a bucket.
  Entry<KT, VT>*      entries_;     // The pointer array that stores the pointer 
                                    // of buckets or child nodes.

public:
  // Constructor and deconstructor
  explicit TNode() : model_(nullptr), size_(0), capacity_(0), 
                      size_sub_tree_(0), bitmap0_(nullptr), 
                      bitmap1_(nullptr), entries_(nullptr) { }

  ~TNode() {
    destory_self();
  }

  // Get functions
  inline uint32_t size() const { return size_; }

  inline uint32_t capacity() const { return capacity_; }

  inline uint32_t size_sub_tree() const { return size_sub_tree_; }

  uint8_t entry_type(uint32_t idx) {
    uint32_t bit_idx = BIT_IDX(idx);
    uint32_t bit_pos = BIT_POS(idx);
    uint8_t bit0 = GET_BIT(bitmap0_[bit_idx], bit_pos);
    uint8_t bit1 = GET_BIT(bitmap1_[bit_idx], bit_pos);
    return ((bit1 << 1) | bit0);
  }

private:
  void set_entry_type(uint32_t idx, uint8_t type) {
    uint32_t bit_idx = BIT_IDX(idx);
    uint32_t bit_pos = BIT_POS(idx);
    if (GET_BIT(bitmap0_[bit_idx], bit_pos) ^ GET_BIT(type, 0)) {
      REV_BIT(bitmap0_[bit_idx], bit_pos);
    }
    if (GET_BIT(bitmap1_[bit_idx], bit_pos) ^ GET_BIT(type, 1)) {
      REV_BIT(bitmap1_[bit_idx], bit_pos);
    }
  }

public:
  // User API interfaces
  ResultIterator<KT, VT> find(KT key) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(key), 0L), 
                              static_cast<int64_t>(capacity_ - 1));
      uint8_t type = entry_type(idx);
      if (type == kData && compare(entries_[idx].kv_.first, key)) {
        return {&entries_[idx].kv_};
      } else if (type == kBucket) {
        return entries_[idx].bucket_->find(key);
      } else if (type == kNode) {
        return entries_[idx].child_->find(key);
      } else {
        return {};
      }
    } else {
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, key, 
                      [](const Entry<KT, VT>& kk, const KT k) {
                        return kk.kv_.first < k;
                      }) - entries_;
      if (idx < size_ && compare(entries_[idx].kv_.first, key)) {
        return {&entries_[idx].kv_};
      } else {
        return {};
      }
    }
  }

  bool update(KVT kv) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(kv.first), 0L), 
                              static_cast<int64_t>(size_ - 1));
      uint8_t type = entry_type(idx);
      if (type == kData && compare(entries_[idx].kv_.first, kv.first)) {
        entries_[idx].kv_ = kv;
        return true;
      } else if (type == kBucket) {
        return entries_[idx].bucket_->update(kv);
      } else if (type == kNode) {
        return entries_[idx].child_->update(kv);
      } else {
        return false;
      }
    } else {
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, kv.first, 
                      [](const Entry<KT, VT>& kk, const KT k) {
                        return kk.kv_.first < k;
                      }) - entries_;
      if (idx < size_ && compare(entries_[idx].kv_.first, kv.first)) {
        entries_[idx].kv_ = kv;
        return true;
      } else {
        return false;
      }
    }
  }

  uint32_t remove(KT key) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(key), 0L), 
                              static_cast<int64_t>(size_ - 1));
      uint8_t type = entry_type(idx);
      if (type == kData && compare(entries_[idx].kv_.first, key)) {
        set_entry_type(idx, kNone);
        size_ --;
        size_sub_tree_ --;
        return 1;
      } else if (type == kBucket) {
        uint32_t res = entries_[idx].bucket_->remove(key);
        size_sub_tree_ -= res;
        return res;
      } else if (type == kNode) {
        uint32_t res = entries_[idx].child_->remove(key);
        size_sub_tree_ -= res;
        return res;
      } else {
        return 0;
      }
    } else {
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, key, 
                      [](const Entry<KT, VT>& kk, const KT k) {
                        return kk.kv_.first < k;
                      }) - entries_;
      if (idx < size_ && compare(entries_[idx].kv_.first, key)) {
        if (size_ > 1) {
          for (uint32_t i = idx; i < size_ - 1; ++ i) {
            if (entries_[i].kv_.first != key) {
              break;
            }
            entries_[i].kv_ = entries_[i + 1].kv_;
          }
        }
        size_ --;
        size_sub_tree_ --;
        return 1;
      } else {
        return 0;
      }
    }
  }

  void insert(KVT kv, uint32_t depth, const HyperParameter& hyper_para) {
    size_sub_tree_ ++;
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(kv.first), 0L), 
                              static_cast<int64_t>(capacity_ - 1));
      uint8_t type = entry_type(idx);
      if (type == kNone) {
        set_entry_type(idx, kData);
        entries_[idx].kv_ = kv;
        size_ ++;
      } else if (type == kData || type == kBucket) {
        if (type == kData) {
          set_entry_type(idx, kBucket);
          KVT stored_kv = entries_[idx].kv_;
          entries_[idx].bucket_ = new Bucket<KT, VT>(&stored_kv, 1, 
                                                  hyper_para.max_bucket_size_);
          size_ --;
        }
        bool success = entries_[idx].bucket_->insert(kv, 
                                                  hyper_para.max_bucket_size_);
        if (!success) {
          // Copy data for rebuilding
          uint32_t bucket_size = entries_[idx].bucket_->size_;
          KVT* kvs = new KVT[bucket_size + 1];
          for (uint32_t i = 0; i < bucket_size; ++ i) {
            kvs[i] = entries_[idx].bucket_->data_[i];
          }
          kvs[bucket_size] = kv;
          std::sort(kvs, kvs + bucket_size + 1, 
            [](auto const& a, auto const& b) {
              return a.first < b.first;
            });
          // Clear entry
          delete entries_[idx].bucket_;
          // Create child node
          set_entry_type(idx, kNode);
          entries_[idx].child_ = new TNode<KT, VT>();
          entries_[idx].child_->build(kvs, bucket_size + 1, depth + 1, 
                                      hyper_para);
          delete[] kvs;
        }
      } else {
        entries_[idx].child_->insert(kv, depth + 1, hyper_para);
      }
    } else {
      if (size_ < capacity_) {
        uint32_t idx = std::lower_bound(entries_, entries_ + size_, kv.first, 
                    [](const Entry<KT, VT>& kk, KT k) {
                      return kk.kv_.first < k;
                    }) - entries_;
        for (int32_t i = size_; i > idx; -- i) {
          entries_[i].kv_ = entries_[i - 1].kv_;
        }
        entries_[idx].kv_ = kv;
        size_ ++;
      } else {
        // Copy data for rebuilding
        uint32_t node_size = size_;
        KVT* kvs = new KVT[node_size + 1];
        for (uint32_t i = 0; i < size_; ++ i) {
          kvs[i] = entries_[i].kv_;
        }
        kvs[node_size] = kv;
        std::sort(kvs, kvs + node_size + 1, 
          [](auto const& a, auto const& b) {
            return a.first < b.first;
          });
        // Clear entry
        destory_self();
        // Create child node
        build(kvs, node_size + 1, depth, hyper_para);
        delete[] kvs;
      }
    }
  }

  void destory_self() {    
    if (model_ != nullptr) {
      delete model_;
      model_ = nullptr;
      for (uint32_t i = 0; i < capacity_; ++ i) {
        uint8_t type_i = entry_type(i);
        if (type_i == kBucket) {
          delete entries_[i].bucket_;
        } else if (type_i == kNode) {
          uint32_t j = i;
          for (; j < capacity_; ++ j) {
            uint8_t type_j = entry_type(j);
            if (type_j != kNode || entries_[j].child_ != entries_[i].child_) {
              break;
            }
          }
          delete entries_[i].child_;
          i = j - 1;
        }
      }
      delete[] bitmap0_;
      bitmap0_ = nullptr;
      delete[] bitmap1_;
      bitmap1_ = nullptr;
    }
    if (entries_ != nullptr) {
      delete[] entries_;
      entries_ = nullptr;
    }
    size_ = 0;
    capacity_ = 0;
    size_sub_tree_ = 0;
  }

  void build_dense_node(const KVT* kvs, uint32_t size, uint32_t depth, 
                        uint32_t capacity) {
    model_ = nullptr;
    size_ = size;
    capacity_ = capacity;
    size_sub_tree_ = size;
    entries_ = new Entry<KT, VT>[capacity_];
    for (uint32_t i = 0; i < size; ++ i) {
      entries_[i].kv_ = kvs[i];
    }
  }

  void build(const KVT* kvs, uint32_t size, uint32_t depth, 
              const HyperParameter& hyper_para) {
    ConflictsInfo* ci = build_linear_model(kvs, size, model_, 
                                          hyper_para.kSizeAmplification);
    if (ci == nullptr) {
      build_dense_node(kvs, size, depth, size + hyper_para.max_bucket_size_);
    } else {
      // Allocate memory for the node
      uint32_t bit_len = BIT_LEN(ci->max_size_);
      capacity_ = ci->max_size_;
      size_ = 0;
      size_sub_tree_ = size;
      bitmap0_ = new BIT_TYPE[bit_len];
      bitmap1_ = new BIT_TYPE[bit_len];
      entries_ = new Entry<KT, VT>[ci->max_size_];
      memset(bitmap0_, 0, sizeof(BIT_TYPE) * bit_len);
      memset(bitmap1_, 0, sizeof(BIT_TYPE) * bit_len);
      // Recursively build the node
      for (uint32_t i = 0, j = 0; i < ci->num_conflicts_; ++ i) {
        uint32_t p = ci->positions_[i];
        uint32_t c = ci->conflicts_[i];
        if (c == 0) {
          continue;
        } else if (c == 1) {
          set_entry_type(p, kData);
          entries_[p].kv_ = kvs[j];
          size_ ++;
          j = j + c;
        } else if (c <= hyper_para.max_bucket_size_) {
          set_entry_type(p, kBucket);
          entries_[p].bucket_ = new Bucket<KT, VT>(kvs + j, c, 
                                                  hyper_para.max_bucket_size_);
          j = j + c;
        } else {
          uint32_t k = i + 1;
          uint32_t seg_size = c;
          uint32_t end = hyper_para.aggregate_size_ == 0 ? ci->num_conflicts_ 
                          : std::min(k + hyper_para.aggregate_size_, 
                                    ci->num_conflicts_);
          while (k < end && ci->positions_[k] - ci->positions_[k - 1] == 1 
                  && ci->conflicts_[k] > hyper_para.max_bucket_size_ + 1) {
            seg_size += ci->conflicts_[k];
            k ++;
          }
          if (seg_size == size) {
            // All conflicted positions are aggregated in one child node 
            // So we build a node for each conflicted position
            for (uint32_t u = i; u < k; ++ u) {
              uint32_t p_k = ci->positions_[u];
              uint32_t c_k = ci->conflicts_[u];
              set_entry_type(p_k, kNode);
              entries_[p_k].child_ = new TNode<KT, VT>();
              entries_[p_k].child_->build(kvs + j, c_k, depth + 1, hyper_para);
              j = j + c_k;
            }
          } else {
            set_entry_type(p, kNode);
            entries_[p].child_ = new TNode<KT, VT>();
            entries_[p].child_->build(kvs + j, seg_size, depth + 1, hyper_para);
            for (uint32_t u = i; u < k; ++ u) {
              uint32_t p_k = ci->positions_[u];
              set_entry_type(p_k, kNode);
              entries_[p_k].child_ = entries_[p].child_;
            }
            j = j + seg_size;
          }
          i = k - 1;
        }
      }
      delete ci;
    }
  }

};

}
#endif