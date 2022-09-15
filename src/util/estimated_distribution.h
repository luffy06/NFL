#ifndef ESTIMATED_DISTRIBUTION
#define ESTIMATED_DISTRIBUTION

#include "util/common.h"

namespace nfl {

template<typename KT>
class longitudes_distribution {
public:
  longitudes_distribution(std::string file_path, int n = 20000000) 
                          : dist_(0, n) {
    std::vector<KT> keys;
    load_keys(file_path, n, keys);
    estimate_prob(keys);
  }

  KT operator()(std::mt19937_64& rng) {
    double density = dist_(rng);
    int idx = std::lower_bound(count_.begin(), count_.end(), density, [](int ct, double density) {
                return ct <= density;
              }) - count_.begin();
    double prop = idx + (idx == count_.size() - 1 ? 0 : (density - count_[idx]) / (count_[idx + 1] - count_[idx]));
    return start_ + prop * step_;
  }

private:
  const int num_bin = 1e5;
  KT start_;
  KT step_;
  std::vector<int> count_;
  std::uniform_real_distribution<> dist_;

  void load_keys(std::string file_path, int n, std::vector<KT>& keys) {
    std::ifstream in(file_path, std::ios::binary | std::ios::in);
    if (!in.is_open()) {
      std::cout << "File: " << file_path << " does not exist" << std::endl;
      exit(-1);
    }
    keys.reserve(n);
    for (int i = 0; i < n; ++ i) {
      KT key;
      in.read((char*)&key, sizeof(KT));
      keys.push_back(key);
    }
    in.close();
    std::sort(keys.begin(), keys.end());
  }

  void estimate_prob(const std::vector<KT>& keys) {
    start_ = keys[0];
    step_ = (keys[keys.size() - 1] - keys[0]) / num_bin;
    count_.reserve(num_bin);
    int ct = 0;
    KT r = keys[0] + step_;
    for (int i = 0, j = 0; i < num_bin; ++ i, r += step_) {
      while (j < keys.size() && keys[j] < r) {
        ++ ct;
        ++ j;
      }
      count_.push_back(ct);
    }
  }
};

}
#endif