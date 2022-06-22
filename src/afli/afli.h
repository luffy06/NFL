#ifndef AFLI_H
#define AFLI_H

#include "afli/afli_nodes.h"
#include "afli/conflicts.h"
#include "util/common.h"

namespace kvevaluator {

template <typename KT, typename VT>
class AFLI {
typedef std::pair<KT, VT> KVT;
private:
  TNode<KT, VT>* root_;
  uint32_t max_bucket_size_;
  uint32_t aggregate_size_;
  
  uint32_t max_aggregate_;

  const uint32_t kMaxBucketSize = 6;
  const uint32_t kMinBucketSize = 1;
  const double kSizeAmplification = 2;
  const double kTailPercent = 0.99;
public:
  AFLI() : root_(nullptr), max_bucket_size_(0), aggregate_size_(0), max_aggregate_(0) { }

  ~AFLI() {
    if (root_ != nullptr) {
      delete root_;
    }
  }

  void BulkLoad(const KVT* kvs, uint32_t size, 
                int bucket_size=-1, uint32_t aggregate_size=0) {
    ASSERT(root_ == nullptr, "The index must be empty before bulk loading");
    root_ = new TNode<KT, VT>();
    if (bucket_size == -1) {
      max_bucket_size_ = ComputeBucketSize(kvs, size);
    } else {
      max_bucket_size_ = std::min(std::max(static_cast<uint32_t>(bucket_size), kMinBucketSize), kMaxBucketSize);
    }
    aggregate_size_ = aggregate_size;
    Build(kvs, size, root_, 1);
  }

  ResultIterator<KT, VT> Find(KT key) {
    return root_->Find(key);
  }

  bool Update(KVT kv) {
    return root_->Update(kv);
  }

  uint32_t Delete(KT key) {
    return root_->Delete(key);
  }
    
  void Insert(KVT kv) {
    RebuildStat<KT, VT>* rs = root_->Insert(kv, 1, max_bucket_size_);
    if (rs != nullptr) {
      std::sort(rs->kvs, rs->kvs + rs->size, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });
      // Start to rebuild a child node based on the bucket
      Build(rs->kvs, rs->size, rs->node, rs->depth);
      delete rs;
    }
  }

  void PrintStat() {
    TreeStat ts;
    ts.bucket_size_ = max_bucket_size_;
    ts.max_aggregate_ = max_aggregate_;
    collect_tree_statistics(root_, 1, ts);
    ts.show();
  }

  uint64_t model_size() {
    TreeStat ts;
    ts.bucket_size_ = max_bucket_size_;
    ts.max_aggregate_ = max_aggregate_;
    collect_tree_statistics(root_, 1, ts);
    return ts.model_size_;
  }

  uint64_t index_size() {
    TreeStat ts;
    ts.bucket_size_ = max_bucket_size_;
    ts.max_aggregate_ = max_aggregate_;
    collect_tree_statistics(root_, 1, ts);
    return ts.index_size_;
  }

  uint32_t max_aggregate() { return max_aggregate_; }

private:
  uint8_t ComputeBucketSize(const KVT* kvs, uint32_t size) {
    uint32_t tail_conflicts = ComputeTailConflicts<KT, VT>(kvs, size, kSizeAmplification, kTailPercent);
    tail_conflicts = std::min(static_cast<uint32_t>(kMaxBucketSize), tail_conflicts);
    tail_conflicts = std::max(static_cast<uint32_t>(kMinBucketSize), tail_conflicts);
    return tail_conflicts;
  }

  void BuildDenseNode(const KVT* kvs, uint32_t size,
                      TNode<KT, VT>* node, uint32_t depth) {
    node->model_ = nullptr;
    node->size_ = size + max_bucket_size_;
    node->num_ins_dense_ = 0;
    node->entries_ = new Entry<KT, VT>[size + max_bucket_size_];
    uint32_t rep = max_bucket_size_ / size;
    uint32_t plus = max_bucket_size_ % size;
    uint32_t j = 0;
    for (uint32_t i = 0; i < size; ++ i) {
      uint32_t rep_i = 1 + rep + (i < plus ? 1 : 0);
      ASSERT(j + rep_i - 1 < node->size_, "Beyond index");
      for (uint32_t k = 0; k < rep_i; ++ k) {
        node->entries_[j + k].kv_ = kvs[i];
      }
      j += rep_i;
    }
    ASSERT(j == node->size_, "Fail to build");
  }

  void Build(const KVT* kvs, uint32_t size, TNode<KT, VT>* node, uint32_t depth) {
    ConflictsInfo* ci = BuildLinearModel(kvs, size, node->model_, kSizeAmplification);
    if (ci == nullptr) {
      BuildDenseNode(kvs, size, node, depth);
    } else {
      // Allocate memory for the node
      uint32_t bit_len = BIT_LEN(ci->max_size_);
      node->size_ = ci->max_size_;
      node->num_ins_dense_ = 0;
      node->bitmap2_ = new BIT_TYPE[bit_len];
      node->bitmap1_ = new BIT_TYPE[bit_len];
      node->entries_ = new Entry<KT, VT>[ci->max_size_];
      memset(node->bitmap1_, 0, sizeof(BIT_TYPE) * bit_len);
      memset(node->bitmap2_, 0, sizeof(BIT_TYPE) * bit_len);
      // Recursively build the node
      for (uint32_t i = 0, j = 0; i < ci->num_conflicts_; ++ i) {
        uint32_t p = ci->positions_[i];
        uint32_t c = ci->conflicts_[i];
        if (c == 0) {
          continue;
        } else if (c == 1) {
          node->bitmap1_[BIT_IDX(p)] |= BIT_KEY(p);
          node->entries_[p].kv_ = kvs[j];
          j = j + c;
        } else if (c <= max_bucket_size_) {
          node->bitmap2_[BIT_IDX(p)] |= BIT_KEY(p);
          node->entries_[p].bucket_ = new Bucket<KT, VT>(kvs + j, c, max_bucket_size_);
          j = j + c;
        } else {
          uint32_t k = i + 1;
          uint32_t seg_size = c;
          uint32_t end = aggregate_size_ == 0 ? ci->num_conflicts_ : std::min(k + aggregate_size_, ci->num_conflicts_);
          while (k < end && ci->positions_[k] - ci->positions_[k - 1] == 1 && ci->conflicts_[k] > max_bucket_size_ + 1) {
            seg_size += ci->conflicts_[k];
            k ++;
          }
          if (seg_size == size) {
            // All conflicted positions are aggregated in one child node 
            // So we build a node for each conflicted position
            for (uint32_t u = i; u < k; ++ u) {
              uint32_t p_k = ci->positions_[u];
              uint32_t c_k = ci->conflicts_[u];
              node->bitmap1_[BIT_IDX(p_k)] |= BIT_KEY(p_k);
              node->bitmap2_[BIT_IDX(p_k)] |= BIT_KEY(p_k);
              node->entries_[p_k].child_ = new TNode<KT, VT>();
              Build(kvs + j, c_k, node->entries_[p_k].child_, depth + 1);
              j = j + c_k;
            }
          } else {
            max_aggregate_ = std::max(max_aggregate_, k - i);
            node->bitmap1_[BIT_IDX(p)] |= BIT_KEY(p);
            node->bitmap2_[BIT_IDX(p)] |= BIT_KEY(p);
            node->entries_[p].child_ = new TNode<KT, VT>();
            Build(kvs + j, seg_size, node->entries_[p].child_, depth + 1);
            for (uint32_t u = i; u < k; ++ u) {
              uint32_t p_k = ci->positions_[u];
              node->bitmap1_[BIT_IDX(p_k)] |= BIT_KEY(p_k);
              node->bitmap2_[BIT_IDX(p_k)] |= BIT_KEY(p_k);
              node->entries_[p_k].child_ = node->entries_[p].child_;
            }
            j = j + seg_size;
          }
          i = k - 1;
        }
      }
      delete ci;
    }
  }

  uint32_t collect_tree_statistics(const TNode<KT, VT>* node, uint32_t depth, TreeStat& ts) {
    if (node->model_ != nullptr) {
      // Model node
      ts.num_model_nodes_ ++;
      ts.model_size_ += sizeof(TNode<KT, VT>) + sizeof(LinearModel<KT>);
      ts.index_size_ += sizeof(TNode<KT, VT>) + sizeof(LinearModel<KT>) 
                      + sizeof(BIT_TYPE) * 2 * BIT_LEN(node->size_) 
                      + sizeof(Entry<KT, VT>) * node->size_;
      bool is_leaf_node = true;
      uint32_t tot_kvs = 0;
      uint32_t tot_conflicts = 0;
      uint32_t num_conflicts = 0;
      for (uint32_t i = 0; i < node->size_; ++ i) {
        uint8_t type = ((node->bitmap1_[BIT_IDX(i)] & BIT_KEY(i)) ? 1 : 0)
                      | (((node->bitmap2_[BIT_IDX(i)] & BIT_KEY(i)) ? 1 : 0) 
                      << 1);
        if (type == kNone) {
          // Empty slot
          continue;
        } else if (type == kData) {
          // Data slot
          ts.num_data_model_ ++;
          ts.sum_depth_ += depth;
          tot_kvs ++;
          continue;
        } else if (type == kBucket) {
          // Bucket pointer
          ts.num_buckets_ ++;
          ts.num_data_bucket_ += node->entries_[i].bucket_->size_;
          ts.model_size_ += sizeof(Bucket<KT, VT>);
          ts.index_size_ += sizeof(Bucket<KT, VT>) 
                          + sizeof(KVT) * max_bucket_size_;
          ts.sum_depth_ += (depth + 1) * node->entries_[i].bucket_->size_;
          tot_kvs += node->entries_[i].bucket_->size_;
          tot_conflicts += node->entries_[i].bucket_->size_ - 1;
          num_conflicts ++;
        } else if (type == kNode) {
          // Child node pointer
          uint32_t num_kvs_child = collect_tree_statistics(node->entries_[i].child_, depth + 1, ts);
          tot_conflicts += num_kvs_child;
          num_conflicts ++;
          is_leaf_node = false;
          // Find the duplicated child node pointers
          uint32_t j = i + 1;
          for (; j < node->size_; ++ j, num_conflicts ++) {
            uint8_t type_j = ((node->bitmap1_[BIT_IDX(j)] & BIT_KEY(j)) 
                            ? 1 : 0) 
                            | (((node->bitmap2_[BIT_IDX(j)] & BIT_KEY(j)) 
                            ? 1 : 0) << 1);
            if (type_j != kNode 
                || node->entries_[j].child_ != node->entries_[i].child_) {
              break;
            }
          }
          i = j - 1;
        }
      }
      ts.node_conflicts_ += num_conflicts ? tot_conflicts * 1. / num_conflicts : 0;
      if (is_leaf_node) {
        ts.num_leaf_nodes_ ++;
        ts.max_depth_ = std::max(ts.max_depth_, depth);
      }
      return tot_conflicts;
    } else {
      // Dense node
      ts.num_dense_nodes_ ++;
      ts.num_data_dense_ ++;
      uint32_t tot_conflicts = 0;
      for (uint32_t i = 1; i < node->size_; ++ i) {
        if (!EQ(node->entries_[i].kv_.first, 
                node->entries_[i - 1].kv_.first)) {
          ts.num_data_dense_ ++;
          tot_conflicts ++;
        }
      }
      ts.sum_depth_ += depth * tot_conflicts;
      ts.node_conflicts_ += tot_conflicts;
      ts.model_size_ += sizeof(TNode<KT, VT>);
      ts.index_size_ += sizeof(TNode<KT, VT>) 
                      + sizeof(Entry<KT, VT>) * node->size_;
      ts.num_leaf_nodes_ ++;
      ts.sum_depth_ += depth;
      ts.max_depth_ = std::max(ts.max_depth_, depth);
      return tot_conflicts;
    }
  }
};

}
#endif