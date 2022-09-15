#ifndef WORKLOAD_H
#define WORKLOAD_H

#include "afli/conflicts.h"
#include "models/linear_model.h"
#include "util/common.h"

namespace nfl {

const double kSCALED = 1e9;

template<class DType, typename KT>
void generate_synthetic_keys(DType dist, int num_keys, std::vector<KT>& keys, 
                            std::string path = "") {
  std::mt19937_64 gen(kSEED);
  std::vector<KT> rep_keys;
  rep_keys.reserve(num_keys * 2);
  for (int i = 0; i < num_keys * 2; ++ i) {
    KT key = static_cast<KT>(dist(gen) * kSCALED + 0.5);
    rep_keys.push_back(key);
  }
  std::sort(rep_keys.begin(), rep_keys.end(), [](auto const& a, auto const& b) {
    return a < b;
  });
  keys.reserve(num_keys);
  for (int i = 0; i < rep_keys.size(); ++ i) {
    if (i == 0 || rep_keys[i] != rep_keys[i - 1]) {
      keys.push_back(rep_keys[i]);
    }
    if (keys.size() == num_keys) {
      break;
    }
  }
  std::cout << "[" << num_keys << "] unique keys are generated" << std::endl;
  if (path != "") {
    std::fstream out(path, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
      std::cout << "File [" << path << "] doesn't exist" << std::endl;
      exit(-1);
    }
    num_keys = keys.size();
    out.write((char*)&num_keys, sizeof(int));
    for (int i = 0; i < keys.size(); ++ i) {
      out.write((char*)&keys[i], sizeof(KT));
    }
    out.close();
  }
}

template<typename KT, typename VT>
void load_source_data(std::string path, std::vector<std::pair<KT, VT>>& kvs) {
  std::mt19937_64 gen(kSEED);
  std::ifstream in(path, std::ios::binary | std::ios::in);
  if (!in.is_open()) {
    std::cout << "File [" << path << "] does not exist" << std::endl;
    exit(-1);
  }
  int num_keys = 0;
  in.read((char*)&num_keys, sizeof(int));
  kvs.reserve(num_keys);
  for (int i = 0; i < num_keys; ++ i) {
    KT key;
    in.read((char*)&key, sizeof(KT));
    VT val = static_cast<VT>(gen());
    kvs.push_back({key, val});
  }
  in.close();
}

template<typename KT, typename VT>
void load_data(std::string path, std::vector<std::pair<KT, VT>>& init_data, 
                  std::vector<Request<KT, VT>>& requests) {
  std::ifstream in(path, std::ios::binary | std::ios::in);
  if (!in.is_open()) {
    std::cout << "File [" << path << "] does not exist" << std::endl;
    exit(-1);
  }
  int num_reqs = 0;
  in.read((char*)&num_reqs, sizeof(int));
  init_data.reserve(num_reqs);
  requests.reserve(num_reqs);
  for (int i = 0; i < num_reqs; ++ i) {
    Request<KT, VT> req;
    in.read((char*)&req, sizeof(Request<KT, VT>));
    if (req.op == kBulkLoad) {
      init_data.push_back(req.kv);
    } else {
      requests.push_back(req);
    }
  }
  in.close();
}

template<typename KT, typename VT>
void write_requests(std::string path, 
                    const std::vector<Request<KT, VT>>& tot_reqs) {
  std::ofstream out(path, std::ios::binary | std::ios::out);
  if (!out.is_open()) {
    std::cout << "File [" << path << "] does not exist" << std::endl;
    exit(-1);
  }
  int num_reqs = tot_reqs.size();
  out.write((char*)&num_reqs, sizeof(int));
  for (int i = 0; i < tot_reqs.size(); ++ i) {
    out.write((char*)&tot_reqs[i], sizeof(Request<KT, VT>));
  }
  out.close();
}

template<typename KT, typename VT>
void assess_data(const std::pair<KT, VT>* kvs, uint32_t size, bool pretty=false) {
  int num_unordered = 0;
  int num_duplicated = 0;
  for (int i = 1; i < size; ++ i) {
    if (kvs[i].first < kvs[i - 1].first) {
      num_unordered ++;
    }
    if (compare(kvs[i].first, kvs[i - 1].first)) {
      num_duplicated ++;
    }
  }
  if (pretty) {
    std::cout << "Unordered\t" << num_unordered << std::endl;
    std::cout << "Duplicated\t" << num_duplicated << std::endl;
  } else {
    std::cout << num_unordered << "\t" << num_duplicated << std::endl;
  }
  if (num_unordered == 0) {
    LinearModel<KT>* model = new LinearModel<KT>();
    ConflictsInfo* ci = build_linear_model<KT, VT>(kvs, size, model, 2);
    std::sort(ci->conflicts_, ci->conflicts_ + ci->num_conflicts_);
    uint32_t max_conflicts = ci->conflicts_[ci->num_conflicts_ - 1];
    uint32_t tail_conflicts = ci->conflicts_[std::max(0, int(ci->num_conflicts_ * 0.99) - 1)];
    double avg_conflicts = 0;
    for (uint32_t i = 0; i < ci->num_conflicts_; ++ i) {
      avg_conflicts += ci->conflicts_[i];
    }
    avg_conflicts /= ci->num_conflicts_;
    double space_amp = 1. * (model->predict(kvs[size - 1].first) - model->predict(kvs[0].first)) / size;
    if (pretty) {
      std::cout << "Max-Conflicts\t" << max_conflicts << std::endl;
      std::cout << "99\%-Conlicts\t" << tail_conflicts << std::endl;
      std::cout << "Avg-Conflicts\t" << avg_conflicts << std::endl;
      std::cout << "Space Amplification\t" << space_amp << std::endl;
    } else {
      std::cout << max_conflicts << "\t" << tail_conflicts << "\t" << avg_conflicts << "\t" << space_amp << std::endl;
    }
    delete ci;
    delete model;
  }
}


}

#endif