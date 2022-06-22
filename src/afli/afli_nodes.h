#ifndef AFLI_NODES_H
#define AFLI_NODES_H

#include "afli/buckets.h"
#include "models/linear_model.h"
#include "util/common.h"

#define COLLECT_RUNNING_STATISTICS 0
#define BIT_TYPE uint8_t
#define BIT_SIZE (sizeof(BIT_TYPE) * 8)
#define BIT_LEN(x) (std::ceil((x) * 1. / BIT_SIZE))
#define BIT_IDX(x) ((x) / BIT_SIZE)
#define BIT_KEY(x) (1 << ((x) % BIT_SIZE))
// #define BS_DENSE
namespace kvevaluator {

template<typename KT, typename VT>
class TNode;

template<typename KT, typename VT>
struct RebuildStat {
  TNode<KT, VT>* node;
  std::pair<KT, VT>* kvs;
  uint32_t size;
  uint32_t depth;

  RebuildStat() : node(nullptr), kvs(nullptr), size(0), depth(0) { }

  ~RebuildStat() {
    if (kvs != nullptr) {
      delete[] kvs;
      kvs = nullptr;
    }
    size = 0;
    depth = 0;
  }
};

enum EntryType {
  kNone = 0,
  kData = 1,
  kBucket = 2,
  kNode = 3
};

template<typename KT, typename VT>
union Entry {
  Bucket<KT, VT>* bucket_;  // The bucket pointer.
  TNode<KT, VT>* child_;    // The child node pointer.
  std::pair<KT, VT> kv_;

  Entry() { }
};


template<typename KT, typename VT>
class TNode {
typedef std::pair<KT, VT> KVT;
public:
  LinearModel<KT>* model_;
  uint32_t size_;
  uint32_t num_ins_dense_;  // The number of data inserted into a dense node 
                            // after it is built.
  uint8_t* bitmap1_;        // The i-th bit indicates whether the i-th position 
                            // has a bucket or a child node.
  uint8_t* bitmap2_;        // The i-th bit indicates whether the i-th position 
                            // is a bucket.
  Entry<KT, VT>* entries_;  // The pointer array that stores the pointer of 
                            // buckets or child nodes.

public:
  explicit TNode() : model_(nullptr), size_(0), num_ins_dense_(0), 
                      bitmap1_(nullptr), bitmap2_(nullptr), 
                      entries_(nullptr) { }

  ~TNode() {
    DestroySelf();
  }

  ResultIterator<KT, VT> Find(KT key) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(
                              model_->Predict(key), 
                              0LL), static_cast<long long>(size_ - 1));
      uint8_t type = ((bitmap1_[BIT_IDX(idx)] & BIT_KEY(idx)) ? 1 : 0) 
                    | (((bitmap2_[BIT_IDX(idx)] & BIT_KEY(idx)) ? 1 : 0) << 1);
      if (type == kData && EQ(entries_[idx].kv_.first, key)) {
        return {&entries_[idx].kv_};
      } else if (type == kBucket) {
        return entries_[idx].bucket_->Find(key);
      } else if (type == kNode) {
        return entries_[idx].child_->Find(key);
      } else {
        return {};
      }
    } else {
      // Binary Search
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, key, 
                      [](const Entry<KT, VT>& kk, const KT k) {
                        return kk.kv_.first < k;
                      }) - entries_;
      if (idx < size_ && EQ(entries_[idx].kv_.first, key)) {
        return {&entries_[idx].kv_};
      } else {
        std::cout << "Dense Not Found\t" << key << std::endl;
        exit(-1);
        return {};
      }
    }
  }

  bool Update(KVT kv) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max( 
                              model_->Predict(kv.first), 
                              0LL), static_cast<long long>(size_ - 1));
      uint8_t type = ((bitmap1_[BIT_IDX(idx)] & BIT_KEY(idx)) ? 1 : 0) 
                    | (((bitmap2_[BIT_IDX(idx)] & BIT_KEY(idx)) ? 1 : 0) << 1);
      if (type == kData && EQ(entries_[idx].kv_.first, kv.first)) {
        entries_[idx].kv_ = kv;
        return true;
      } else if (type == kBucket) {
        return entries_[idx].bucket_->Update(kv);
      } else if (type == kNode) {
        return entries_[idx].child_->Update(kv);
      } else {
        return false;
      }
    } else {
      // Binary Search
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, kv.first, 
                      [](const Entry<KT, VT>& kk, const KT k) {
                        return kk.kv_.first < k;
                      }) - entries_;
      if (idx < size_ && EQ(entries_[idx].kv_.first, kv.first)) {
        entries_[idx].kv_ = kv;
        return true;
      } else {
        return false;
      }
    }
  }

  uint32_t Delete(KT key) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(
                              model_->Predict(key), 
                              0LL), static_cast<long long>(size_ - 1));
      uint32_t bit_idx = BIT_IDX(idx);
      uint32_t bit_key = BIT_KEY(idx);
      uint8_t type = ((bitmap1_[bit_idx] & bit_key) ? 1 : 0) 
                    | (((bitmap2_[bit_idx] & bit_key) ? 1 : 0) << 1);
      if (type == kData && EQ(entries_[idx].kv_.first, key)) {
        bitmap1_[bit_idx] &= (~bit_key);
        bitmap2_[bit_idx] &= (~bit_key);
        return 1;
      } else if (type == kBucket) {
        return entries_[idx].bucket_->Delete(key);
      } else if (type == kNode) {
        return entries_[idx].child_->Delete(key);
      } else {
        return 0;
      }
    } else {
      // Binary Search
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, key, 
                      [](const Entry<KT, VT>& kk, const KT k) {
                        return kk.kv_.first < k;
                      }) - entries_;
      if (idx < size_ && EQ(entries_[idx].kv_.first, key)) {
        if (size_ > 1) {
          for (uint32_t i = idx; i < size_ - 1; ++ i) {
            if (entries_[i].kv_.first != key) {
              break;
            }
            entries_[i].kv_ = entries_[i + 1].kv_;
          }
        }
        size_ --;
        return 1;
      } else {
        return 0;
      }
    }
  }

  RebuildStat<KT, VT>* Insert(KVT kv, uint32_t depth, 
                              const uint8_t kMaxBucketSize) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(
                              model_->Predict(kv.first), 
                              0LL), static_cast<long long>(size_ - 1));
      uint32_t bit_idx = BIT_IDX(idx);
      uint32_t bit_key = BIT_KEY(idx);
      uint8_t type = ((bitmap1_[bit_idx] & bit_key) ? 1 : 0) 
                    | (((bitmap2_[bit_idx] & bit_key) ? 1 : 0) << 1);
      if (type == kNone) {
        entries_[idx].kv_ = kv;
        bitmap1_[bit_idx] |= bit_key;
        return nullptr;
      } else if (type == kData || type == kBucket) {
        if (type == kData) {
          bitmap1_[bit_idx] &= (~bit_key);
          bitmap2_[bit_idx] |= bit_key;
          KVT stored_kv = entries_[idx].kv_;
          entries_[idx].bucket_ = new Bucket<KT, VT>(&stored_kv, 1, kMaxBucketSize);
        }
        bool success = entries_[idx].bucket_->Insert(kv, kMaxBucketSize);
        if (success) {
          // OPT: check whether there are too many buckets
          return nullptr;
        } else {
          RebuildStat<KT, VT>* rs = new RebuildStat<KT, VT>();
          rs->depth = depth + 1;
          rs->size = entries_[idx].bucket_->size_ + 1;
          rs->kvs = new KVT[rs->size];
          for (uint8_t i = 0; i < entries_[idx].bucket_->size_; ++ i) {
            rs->kvs[i] = entries_[idx].bucket_->data_[i];
          }
          rs->kvs[rs->size - 1] = kv;
          delete entries_[idx].bucket_;
          bitmap1_[bit_idx] |= bit_key;
          bitmap2_[bit_idx] |= bit_key;
          entries_[idx].child_ = new TNode<KT, VT>();
          rs->node = entries_[idx].child_;
          return rs;
        }
      } else {
        return entries_[idx].child_->Insert(kv, depth + 1, kMaxBucketSize);
      }
    } else {
      // Binary Search
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, kv.first, 
                  [](const Entry<KT, VT>& kk, KT k) {
                    return kk.kv_.first < k;
                  }) - entries_;
      if (num_ins_dense_ < kMaxBucketSize) {
        int left = -1, right = -1;
        for (int i = idx - 2; i >= 0; -- i) {
          if (EQ(entries_[i].kv_.first, entries_[i + 1].kv_.first)) {
            left = i;
            break;
          }
        }
        for (int i = idx + 1; i < size_; ++ i) {
          if (EQ(entries_[i].kv_.first, entries_[i - 1].kv_.first)) {
            right = i;
            break;
          }
        }
        if (left == -1 && right == -1) {
          ASSERT(false, "No enough gaps");
        } else if (left == -1 || (right - idx < idx - left)) {
          for (int i = right; i > idx; -- i) {
            entries_[i].kv_ = entries_[i - 1].kv_;
          }
          entries_[idx].kv_ = kv;
          num_ins_dense_ ++;
        } else if (right == -1 || (idx - left < right - idx)) {
          for (int i = left; i < idx - 1; ++ i) {
            entries_[i].kv_ = entries_[i + 1].kv_;
          }
          entries_[idx - 1].kv_ = kv;
          num_ins_dense_ ++;
        }
        return nullptr;
      } else {
        RebuildStat<KT, VT>* rs = new RebuildStat<KT, VT>();
        rs->depth = depth;
        rs->size = size_ + 1;
        rs->kvs = new KVT[rs->size];
        rs->kvs[0] = entries_[0].kv_;
        uint32_t j = 1;
        for (uint32_t i = 1; i < size_; ++ i) {
          if (!EQ(entries_[i].kv_.first, rs->kvs[j - 1].first)) {
            rs->kvs[j ++] = entries_[i].kv_;
          }
        }
        rs->kvs[j] = kv;
        DestroySelf();
        rs->node = this;
        return rs;
      }
    }
  }

  void DestroySelf() {    
    if (model_ != nullptr) {
      delete model_;
      model_ = nullptr;
      for (uint32_t i = 0; i < size_; ++ i) {
        uint8_t type_i = ((bitmap1_[BIT_IDX(i)] & BIT_KEY(i)) ? 1 : 0) 
                        | (((bitmap2_[BIT_IDX(i)] & BIT_KEY(i)) ? 1 : 0) << 1);
        if (type_i == kBucket) {
          delete entries_[i].bucket_;
        } else if (type_i == kNode) {
          uint32_t j = i;
          for (; j < size_; ++ j) {
            uint8_t type_j = ((bitmap1_[BIT_IDX(j)] & BIT_KEY(j)) ? 1 : 0) 
                            | (((bitmap2_[BIT_IDX(j)] & BIT_KEY(j)) ? 1 : 0) 
                            << 1);
            if (type_j != kNode || entries_[j].child_ != entries_[i].child_) {
              break;
            }
          }
          delete entries_[i].child_;
          i = j - 1;
        }
      }
      delete[] bitmap1_;
      bitmap1_ = nullptr;
      delete[] bitmap2_;
      bitmap2_ = nullptr;
    }
    if (entries_ != nullptr) {
      delete[] entries_;
      entries_ = nullptr;
    }
    size_ = 0;
    num_ins_dense_ = 0;
  }

};

}
#endif