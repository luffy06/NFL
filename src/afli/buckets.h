#ifndef BUCKET_H
#define BUCKET_H

#include "afli/iterator.h"
#include "util/common.h"

namespace kvevaluator {

template<typename KT, typename VT>
class Bucket {
typedef std::pair<KT, VT> KVT;
public:
  KVT* data_;
  uint8_t size_;

public:
  Bucket() : data_(nullptr), size_(0) { }

  Bucket(const KVT* kvs, uint32_t size, const uint8_t kMaxBucketSize) : size_(size) {
    data_ = new KVT[kMaxBucketSize];
    for (uint32_t i = 0; i < size; ++ i) {
      data_[i] = kvs[i];
    }
  }

  ~Bucket() {
    if (data_ != nullptr) {
      delete[] data_;
      data_ = nullptr;
    }
  }

  inline uint8_t size() const { return size_; }

  ResultIterator<KT, VT> Find(KT key) {
    for (uint32_t i = 0; i < size_; ++ i) {
      if (EQ(data_[i].first, key)) {
        return {&data_[i]};
      }
    }
    return {};
  }

  bool Update(KVT kv) {
    for (uint8_t i = 0; i < size_; ++ i) {
      if (EQ(data_[i].first, kv.first)) {
        return true;
      }
    }
    return false;
  }

  uint32_t Delete(KT key) {
    bool copy = false;
    for (uint8_t i = 0; i < size_; ++ i) {
      if (EQ(data_[i].first, key)) {
        copy = true;
      }
      if (copy && i + 1 < size_) {
        data_[i] = data_[i + 1];
      }
    }
    if (copy) {
      size_ --;
      return 1;
    } else {
      return 0;
    }
  }

  bool Insert(KVT kv, const uint8_t kMaxBucketSize) {
    if (size_ < kMaxBucketSize) {
      data_[size_] = kv;
      size_ ++;
      return true;
    } else {
      return false;
    }
  }
};

}

#endif