#ifndef AFLI_H
#define AFLI_H

#include "afli/afli_nodes.h"

namespace nfl {

template <typename KT, typename VT>
class AFLI {
typedef std::pair<KT, VT> KVT;
private:
  TNode<KT, VT>* root_;
  HyperParameter hyper_para_;
public:
  AFLI() : root_(nullptr) { }

  ~AFLI() {
    if (root_ != nullptr) {
      delete root_;
    }
  }

  void bulk_load(const KVT* kvs, uint32_t size, int32_t bucket_size=-1, 
                uint32_t aggregate_size=0) {
    assert_p(root_ == nullptr, "The index must be empty before bulk loading");
    root_ = new TNode<KT, VT>();
    if (bucket_size == -1) {
      hyper_para_.max_bucket_size_ = compute_bucket_size(kvs, size);
    } else {
      hyper_para_.max_bucket_size_ = std::min(std::max(
                                      static_cast<uint32_t>(bucket_size), 
                                      hyper_para_.kMinBucketSize), 
                                      hyper_para_.kMaxBucketSize);
    }
    hyper_para_.aggregate_size_ = aggregate_size;
    root_->build(kvs, size, 1, hyper_para_);
  }

  ResultIterator<KT, VT> find(KT key) {
    return root_->find(key);
  }

  bool update(KVT kv) {
    return root_->update(kv);
  }

  uint32_t remove(KT key) {
    return root_->remove(key);
  }
    
  void insert(KVT kv) {
    root_->insert(kv, 1, hyper_para_);
  }

  void print_stats() {
    TreeStat ts;
    ts.bucket_size_ = hyper_para_.max_bucket_size_;
    collect_tree_statistics(root_, 1, ts);
    ts.show();
  }

  uint64_t model_size() {
    TreeStat ts;
    ts.bucket_size_ = hyper_para_.max_bucket_size_;
    collect_tree_statistics(root_, 1, ts);
    return ts.model_size_;
  }

  uint64_t index_size() {
    TreeStat ts;
    ts.bucket_size_ = hyper_para_.max_bucket_size_;
    collect_tree_statistics(root_, 1, ts);
    return ts.index_size_;
  }

private:
  uint8_t compute_bucket_size(const KVT* kvs, uint32_t size) {
    uint32_t tail_conflicts = compute_tail_conflicts<KT, VT>(kvs, size, 
                                                hyper_para_.kSizeAmplification, 
                                                hyper_para_.kTailPercent);
    tail_conflicts = std::min(hyper_para_.kMaxBucketSize, tail_conflicts);
    tail_conflicts = std::max(hyper_para_.kMinBucketSize, tail_conflicts);
    return tail_conflicts;
  }

  uint32_t collect_tree_statistics(TNode<KT, VT>* node, 
                                    uint32_t depth, TreeStat& ts) {
    if (node->model_ != nullptr) {
      // Model node
      ts.num_model_nodes_ ++;
      ts.model_size_ += sizeof(TNode<KT, VT>) + sizeof(LinearModel<KT>);
      ts.index_size_ += sizeof(TNode<KT, VT>) + sizeof(LinearModel<KT>) 
                      + sizeof(BIT_TYPE) * 2 * BIT_LEN(node->capacity_) 
                      + sizeof(Entry<KT, VT>) * node->capacity_;
      bool is_leaf_node = true;
      uint32_t tot_kvs = 0;
      uint32_t tot_conflicts = 0;
      uint32_t num_conflicts = 0;
      for (uint32_t i = 0; i < node->capacity_; ++ i) {
        uint8_t type = node->entry_type(i);
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
                          + sizeof(KVT) * hyper_para_.max_bucket_size_;
          ts.sum_depth_ += (depth + 1) * node->entries_[i].bucket_->size_;
          tot_kvs += node->entries_[i].bucket_->size_;
          tot_conflicts += node->entries_[i].bucket_->size_ - 1;
          num_conflicts ++;
        } else if (type == kNode) {
          // Child node pointer
          uint32_t num_kvs_child = collect_tree_statistics(
                                    node->entries_[i].child_, depth + 1, ts);
          tot_conflicts += num_kvs_child;
          num_conflicts ++;
          is_leaf_node = false;
          // Find the duplicated child node pointers
          uint32_t j = i + 1;
          for (; j < node->size_; ++ j, num_conflicts ++) {
            uint8_t type_j = node->entry_type(j);
            if (type_j != kNode 
                || node->entries_[j].child_ != node->entries_[i].child_) {
              break;
            }
          }
          i = j - 1;
        }
      }
      ts.node_conflicts_ += num_conflicts ? 
                              tot_conflicts * 1. / num_conflicts : 0;
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
        if (!compare(node->entries_[i].kv_.first, 
                      node->entries_[i - 1].kv_.first)) {
          ts.num_data_dense_ ++;
          tot_conflicts ++;
        }
      }
      ts.sum_depth_ += depth * tot_conflicts;
      ts.node_conflicts_ += tot_conflicts;
      ts.model_size_ += sizeof(TNode<KT, VT>);
      ts.index_size_ += sizeof(TNode<KT, VT>) 
                      + sizeof(Entry<KT, VT>) * node->capacity_;
      ts.num_leaf_nodes_ ++;
      ts.sum_depth_ += depth;
      ts.max_depth_ = std::max(ts.max_depth_, depth);
      return tot_conflicts;
    }
  }
};

}
#endif