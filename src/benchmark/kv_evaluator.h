#ifndef KV_EVALUATOR_H
#define KV_EVALUATOR_H

#include "benchmark/workload.h"
#include "util/common.h"

#include "afli/afli.h"
#include "ALEX/src/core/alex.h"
#include "BTree/btree_map.h"
#include "lipp/src/core/lipp.h"
#include "nfl/nfl.h"
#include "PGM-index/include/pgm/pgm_index_dynamic.hpp"

namespace kvevaluator {

struct LIPPConfig {
  LIPPConfig(std::string path) { }
};

struct AlexConfig {
  AlexConfig(std::string path) { }
};

struct PGMConfig {
  int base;
  int buffer_level;
  int index_level;

  PGMConfig(std::string path) {
    base = 8;
    buffer_level = 0;
    index_level = 0;
    if (path != "") {
      std::ifstream in(path, std::ios::in);
      if (in.is_open()) {
        while (!in.eof()) {
          std::string kv;
          in >> kv;
          std::string::size_type n = kv.find("=");
          if (n != std::string::npos) {
            std::string key = kv.substr(0, n);
            std::string val = kv.substr(n + 1);
            if (key == "base") {
              base = STRTONUM<std::string, int>(val);
            } else if (key == "buffer_level") {
              buffer_level = STRTONUM<std::string, int>(val);
            } else if (key == "index_level") {
              index_level = STRTONUM<std::string, int>(val);
            }
          }
        }
        in.close();
      }
    }
  }
};

struct BTreeConfig {
  BTreeConfig(std::string path) { }
};

struct AFLIConfig {
  int bucket_size;
  int aggregate_size;

  AFLIConfig(std::string path) {
    bucket_size = -1;
    aggregate_size = 0;
    if (path != "") {
      std::ifstream in(path, std::ios::in);
      if (in.is_open()) {
        while (!in.eof()) {
          std::string kv;
          in >> kv;
          std::string::size_type n = kv.find("=");
          if (n != std::string::npos) {
            std::string key = kv.substr(0, n);
            std::string val = kv.substr(n + 1);
            if (key == "bucket_size") {
              bucket_size = STRTONUM<std::string, int>(val);
            } else if (key == "aggregate_size") {
              aggregate_size = STRTONUM<std::string, int>(val);
            }
          }
        }
        in.close();
      }
    }
  }
};

struct NFLConfig {
  int bucket_size;
  int aggregate_size;
  std::string weights_path;

  NFLConfig(std::string path) {
    bucket_size = -1;
    aggregate_size = 0;
    weights_path = "";
    if (path != "") {
      std::ifstream in(path, std::ios::in);
      if (in.is_open()) {
        while (!in.eof()) {
          std::string kv;
          in >> kv;
          std::string::size_type n = kv.find("=");
          if (n != std::string::npos) {
            std::string key = kv.substr(0, n);
            std::string val = kv.substr(n + 1);
            if (key == "bucket_size") {
              bucket_size = STRTONUM<std::string, int>(val);
            } else if (key == "aggregate_size") {
              aggregate_size = STRTONUM<std::string, int>(val);
            } else if (key == "weights_path") {
              weights_path = val;
            }
          }
        }
        in.close();
      }
    }
  }
};

template<typename KT, typename VT>
class KVEvaluator {
typedef std::pair<KT, VT> KVT;

public:
  
  std::vector<KVT> load_data;
  std::vector<Request<KT, VT>> reqs;
  const double conflicts_decay = 0.1;

  void TestWorkloads(std::string index_name, int batch_size, 
                    std::string workload_path, std::string config_path="",
                    bool show_incremental_throughputs=false) {
    load_data.clear();
    reqs.clear();
    std::string workload_name = GetWorkloadName(workload_path);
    LoadRequests(workload_path, load_data, reqs);
    // Check the order of load data.
    for (int i = 1; i < load_data.size(); ++ i) {
      if (load_data[i].first < load_data[i - 1].first || 
          EQ(load_data[i].first, load_data[i - 1].first)) {
        std::cout << std::fixed 
                  << std::setprecision(std::numeric_limits<KT>::digits10) 
                  << "Duplicated data" << std::endl << i - 1 << "th key [" 
                  << load_data[i - 1].first << "]" << std::endl << i 
                  << "th key [" << load_data[i].first << "]" << std::endl;
      }
      ASSERT(load_data[i].first > load_data[i - 1].first, 
              "Unordered case in load data");
    }
    bool categorical_keys = false;
    if (categorical_keys) {
      std::set<KT> key_set;
      for (int i = 0; i < load_data.size(); ++ i) {
        key_set.insert(load_data[i].first);
      }
      for (int i = 0; i < reqs.size(); ++ i) {
        key_set.insert(reqs[i].kv.first);
      }
      std::vector<KT> key_array;
      key_array.reserve(key_set.size());
      key_array.insert(key_array.end(), key_set.begin(), key_set.end());
      for (int i = 0; i < load_data.size(); ++ i) {
        KT opt_key = std::lower_bound(key_array.begin(), key_array.end(), 
                                      load_data[i].first) - key_array.begin();
        load_data[i] = {opt_key, load_data[i].second};
      }
      for (int i = 0; i < reqs.size(); ++ i) {
        KT opt_key = std::lower_bound(key_array.begin(), key_array.end(), 
                                      reqs[i].kv.first) - key_array.begin();
        reqs[i].kv = {opt_key, reqs[i].kv.second};
      }
    }
    // Start to evaluate
    bool show_stat = false;
    ExperimentalResults exp_res(batch_size);
    if (StartWith(index_name, "lipp")) {
      TestLIPP(batch_size, exp_res, config_path, show_stat);
    } else if (StartWith(index_name, "alex")) {
      TestAlex(batch_size, exp_res, config_path, show_stat);
    } else if (StartWith(index_name, "pgm-index")) {
      TestPGM(batch_size, exp_res, config_path, show_stat);
    } else if (StartWith(index_name, "btree")) {
      TestBTree(batch_size, exp_res, config_path, show_stat);
    } else if (StartWith(index_name, "afli")) {
      TestAFLI(batch_size, exp_res, config_path, show_stat);
    } else if (StartWith(index_name, "nfl")) {
      TestNFL(batch_size, exp_res, config_path, show_stat);
    } else {
      std::cout << "Unsupported model name [" << index_name << "]" << std::endl;
      exit(-1);
    }
    // Print results.
    std::cout << workload_name << "\t" << index_name << "\t" << batch_size 
              << std::endl;
    if (show_incremental_throughputs) {
      exp_res.show_incremental_throughputs();
    } else {
      exp_res.show();
    }
  }

  void TestLIPP(int batch_size, ExperimentalResults& exp_res, 
                std::string config_path, bool show_stat=false) {
    // Load config
    LIPPConfig config(config_path);
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    LIPP<KT, VT> lipp;
    lipp.bulk_load(load_data.data(), load_data.size());      
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end 
                                                    - bulk_load_start).count();

    std::vector<KVT> batch_data;
    batch_data.reserve(batch_size);
    // Perform operations in batch
    int num_batches = std::ceil(reqs.size() * 1. / batch_size);
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      int l = batch_idx * batch_size;
      int r = std::min((batch_idx + 1) * batch_size, 
                        static_cast<int>(reqs.size()));
      for (int i = l; i < r; ++ i) {
        batch_data.push_back(reqs[i].kv);
      }

      VT val_sum = 0;
      // Perform requests
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = l; i < r; ++ i) {
        int data_idx = i - l;
        if (reqs[i].op == kQuery) {
          VT res = lipp.at(batch_data[data_idx].first);
          val_sum += res;
        } else if (reqs[i].op == kUpdate) {
          VT res = lipp.at(batch_data[data_idx].first);
        } else if (reqs[i].op == kInsert) {
          lipp.insert(batch_data[data_idx]);
        } else if (reqs[i].op == kDelete) {
          std::cout << "Unsupport now" << std::endl;
          exit(-1);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end 
                                                              - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
    }
    exp_res.model_size = lipp.model_size();
    exp_res.index_size = lipp.index_size();
    if (show_stat) {
      lipp.print_depth();
    }
  }

  void TestAlex(int batch_size, ExperimentalResults& exp_res, 
                std::string config_path, bool show_stat=false) {
    AlexConfig config(config_path);
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    alex::Alex<KT, VT> alex;
    alex.bulk_load(load_data.data(), load_data.size());
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end 
                                                    - bulk_load_start).count();
    
    std::vector<KVT> batch_data;
    batch_data.reserve(batch_size);
    // Perform operations in batch
    int num_batches = std::ceil(reqs.size() * 1. / batch_size);
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      int l = batch_idx * batch_size;
      int r = std::min((batch_idx + 1) * batch_size, 
                        static_cast<int>(reqs.size()));
      for (int i = l; i < r; ++ i) {
        batch_data.push_back(reqs[i].kv);
      }

      VT val_sum = 0;
      // Perform requests
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = l; i < r; ++ i) {
        int data_idx = i - l;
        if (reqs[i].op == kQuery) {
          auto res = alex.find(batch_data[data_idx].first);
          if (res != alex.end()) {
            val_sum += res.payload();
          }
        } else if (reqs[i].op == kUpdate) {
          auto res = alex.find(batch_data[data_idx].first);
          if (res != alex.end()) {
            res.payload() = batch_data[data_idx].second;
          }
        } else if (reqs[i].op == kInsert) {
          alex.insert(batch_data[data_idx].first, batch_data[data_idx].second);
        } else if (reqs[i].op == kDelete) {
          int res = alex.erase(batch_data[data_idx].first);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end 
                                                              - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
    }
    exp_res.model_size = alex.model_size();
    exp_res.index_size = alex.model_size() + alex.data_size();
    if (show_stat) {
      alex.print_stats();
    }
  }

  void TestPGM(int batch_size, ExperimentalResults& exp_res, 
                std::string config_path, bool show_stat=false) {
    PGMConfig config(config_path);
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    pgm::DynamicPGMIndex<KT, VT, pgm::PGMIndex<KT, 16>> pgm_index(
      load_data.begin(), load_data.end(), config.base, config.buffer_level, 
      config.index_level);
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end 
                                                    - bulk_load_start).count();

    std::vector<KVT> batch_data;
    batch_data.reserve(batch_size);
    // Perform operations in batch
    int num_batches = std::ceil(reqs.size() * 1. / batch_size);
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      int l = batch_idx * batch_size;
      int r = std::min((batch_idx + 1) * batch_size, 
                        static_cast<int>(reqs.size()));
      for (int i = l; i < r; ++ i) {
        batch_data.push_back(reqs[i].kv);
      }

      VT val_sum = 0;
      // Perform requests
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = l; i < r; ++ i) {
        int data_idx = i - l;
        if (reqs[i].op == kQuery) {
          auto res = pgm_index.find(batch_data[data_idx].first);
          if (res != pgm_index.end()) {
            val_sum += res->second;
          }
        } else if (reqs[i].op == kUpdate) {
          pgm_index.insert_or_assign(batch_data[data_idx].first, 
                                      batch_data[data_idx].second);
        } else if (reqs[i].op == kInsert) {
          pgm_index.insert_or_assign(batch_data[data_idx].first, 
                                      batch_data[data_idx].second);
        } else if (reqs[i].op == kDelete) {
          pgm_index.erase(batch_data[data_idx].first);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end 
                                                              - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
    }
    exp_res.model_size = pgm_index.index_size_in_bytes();
    exp_res.index_size = pgm_index.size_in_bytes();
    if (show_stat) {
      pgm_index.print_stats();
    }
  }

  void TestBTree(int batch_size, ExperimentalResults& exp_res, 
                  std::string config_path, bool show_stat=false) {
    BTreeConfig config(config_path);
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    btree::btree_map<KT, VT> btree;
    for (int i = 0; i < load_data.size(); ++ i) {
      btree.insert(load_data[i]);
    }
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end 
                                                    - bulk_load_start).count();

    std::vector<KVT> batch_data;
    batch_data.reserve(batch_size);
    // Perform operations in batch
    int num_batches = std::ceil(reqs.size() * 1. / batch_size);
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      int l = batch_idx * batch_size;
      int r = std::min((batch_idx + 1) * batch_size, 
                        static_cast<int>(reqs.size()));
      for (int i = l; i < r; ++ i) {
        batch_data.push_back(reqs[i].kv);
      }

      VT val_sum = 0;
      // Perform requests
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = l; i < r; ++ i) {
        int data_idx = i - l;
        if (reqs[i].op == kQuery) {
          auto res = btree.find(batch_data[data_idx].first);
          val_sum += res->second;
        } else if (reqs[i].op == kUpdate) {
          auto res = btree.find(batch_data[data_idx].first);
        } else if (reqs[i].op == kInsert) {
          btree.insert(batch_data[data_idx]);
        } else if (reqs[i].op == kDelete) {
          int res = btree.erase(batch_data[data_idx].first);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end 
                                                              - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
    }
    exp_res.model_size = 0;
    exp_res.index_size = 0;
  }

  void TestAFLI(int batch_size, ExperimentalResults& exp_res, 
                std::string config_path, bool show_stat=false) {
    AFLIConfig config(config_path);
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    AFLI<KT, VT> afli;
    afli.BulkLoad(load_data.data(), load_data.size());
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end 
                                                    - bulk_load_start).count();
    if (show_stat) {
      afli.PrintStat();
    }

    std::vector<KVT> batch_data;
    batch_data.reserve(batch_size);
    // Perform operations in batch
    int num_batches = std::ceil(reqs.size() * 1. / batch_size);
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      int l = batch_idx * batch_size;
      int r = std::min((batch_idx + 1) * batch_size, 
                        static_cast<int>(reqs.size()));
      for (int i = l; i < r; ++ i) {
        batch_data.push_back(reqs[i].kv);
      }

      VT val_sum = 0;
      // Perform requests
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = l; i < r; ++ i) {
        int data_idx = i - l;
        if (reqs[i].op == kQuery) {
          auto it = afli.Find(batch_data[data_idx].first);
          if (!it.is_end()) {
            val_sum += it.value();
          }
        } else if (reqs[i].op == kUpdate) {
          bool res = afli.Update(batch_data[data_idx]);
        } else if (reqs[i].op == kInsert) {
          afli.Insert(batch_data[data_idx]);
        } else if (reqs[i].op == kDelete) {
          int res = afli.Delete(batch_data[data_idx].first);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
    }
    exp_res.model_size = afli.model_size();
    exp_res.index_size = afli.index_size();
    if (show_stat) {
      afli.PrintStat();
    }
  }

  void TestNFL(int batch_size, ExperimentalResults& exp_res, 
                std::string config_path, bool show_stat=false) {
    NFLConfig config(config_path);
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    NFL<KT, VT> nfl(config.weights_path, batch_size);
    uint32_t tail_conflicts = nfl.AutoSwitch(load_data.data(), 
                                              load_data.size());
    auto bulk_load_mid = std::chrono::high_resolution_clock::now();
    nfl.IndexBulkLoad(load_data.data(), load_data.size(), tail_conflicts);
    // nfl.BulkLoad(load_data.data(), load_data.size());
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_trans_time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_mid 
                                                    - bulk_load_start).count();
    exp_res.bulk_load_index_time = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end 
                                                      - bulk_load_mid).count();
    if (show_stat) {
      nfl.PrintStat();
    }

    std::vector<KVT> batch_data;
    batch_data.reserve(batch_size);
    // Perform operations in batch
    int num_batches = std::ceil(reqs.size() * 1. / batch_size);
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      int l = batch_idx * batch_size;
      int r = std::min((batch_idx + 1) * batch_size, 
                        static_cast<int>(reqs.size()));
      for (int i = l; i < r; ++ i) {
        batch_data.push_back(reqs[i].kv);
      }

      VT val_sum = 0;
      // Perform requests
      auto start = std::chrono::high_resolution_clock::now();
      nfl.Transform(batch_data.data(), batch_data.size());
      auto mid = std::chrono::high_resolution_clock::now();
      for (int i = l; i < r; ++ i) {
        int data_idx = i - l;
        if (reqs[i].op == kQuery) {
          auto it = nfl.Find(data_idx);
          if (!it.is_end()) {
              val_sum += it.value();
          }
        } else if (reqs[i].op == kUpdate) {
          bool res = nfl.Update(data_idx);
        } else if (reqs[i].op == kInsert) {
          nfl.Insert(data_idx);
        } else if (reqs[i].op == kDelete) {
          int res = nfl.Delete(data_idx);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(mid 
                                                              - start).count();
      double time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end 
                                                              - mid).count();
      exp_res.sum_transform_time += time1;
      exp_res.sum_indexing_time += time2;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({time1, time2});
      exp_res.step();
    }
    exp_res.model_size = nfl.model_size();
    exp_res.index_size = nfl.index_size();
    if (show_stat) {
      nfl.PrintStat();
    }
  }
};

}

#endif