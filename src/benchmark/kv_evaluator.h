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

enum ResultType {
  kSuccess = 0,
  kFail = 1,
  kNotFound = 2
};

template<typename KT, typename VT>
struct Status {
  ResultType state;
  std::pair<KT, VT> kv;

  Status() : state(kSuccess) { }

  Status(ResultType rt, KT key, VT val) : state(rt), kv({key, val}) { }

  Status(ResultType rt) : state(rt) { }
};

template<typename KT, typename VT>
class KVEvaluator {
typedef std::pair<KT, VT> KVT;

public:
  
  std::vector<KVT> load_data;
  std::vector<Request<KT, VT>> reqs;
  const double conflicts_decay = 0.1;

  void TestWorkloads(std::string index_name, int batch_size, 
                    std::string workload_path, std::string weights_path="",
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
      for (uint32_t i = 0; i < load_data.size(); ++ i) {
        key_set.insert(load_data[i].first);
      }
      for (uint32_t i = 0; i < reqs.size(); ++ i) {
        key_set.insert(reqs[i].kv.first);
      }
      std::vector<KT> key_array;
      key_array.reserve(key_set.size());
      key_array.insert(key_array.end(), key_set.begin(), key_set.end());
      for (uint32_t i = 0; i < load_data.size(); ++ i) {
        KT opt_key = std::lower_bound(key_array.begin(), key_array.end(), load_data[i].first) - key_array.begin();
        load_data[i] = {opt_key, load_data[i].second};
      }
      for (uint32_t i = 0; i < reqs.size(); ++ i) {
        KT opt_key = std::lower_bound(key_array.begin(), key_array.end(), reqs[i].kv.first) - key_array.begin();
        reqs[i].kv = {opt_key, reqs[i].kv.second};
      }
    }
    // Start to evaluate
    bool show_stat = false;
    bool check_correctness = false;
    ExperimentalResults exp_res(batch_size);
    if (StartWith(index_name, "lipp")) {
      TestLIPP(batch_size, exp_res, show_stat, check_correctness);
    } else if (StartWith(index_name, "alex")) {
      TestAlex(batch_size, exp_res, show_stat, check_correctness);
    } else if (StartWith(index_name, "pgm-index")) {
      TestPGMIndex(batch_size, exp_res, show_stat, check_correctness);
    } else if (StartWith(index_name, "btree")) {
      TestBTree(batch_size, exp_res, show_stat, check_correctness);
    } else if (StartWith(index_name, "afli")) {
      TestAFLI(batch_size, exp_res, show_stat, check_correctness);
    } else if (StartWith(index_name, "nfl")) {
      TestNFL(batch_size, exp_res, weights_path, show_stat, check_correctness);
    } else {
      std::cout << "Unsupported model name [" << index_name << "]" << std::endl;
      exit(-1);
    }
    // Print results.
    std::cout << workload_name << "\t" << index_name << "\t" << batch_size << std::endl;
    if (show_incremental_throughputs) {
      exp_res.show_incremental_throughputs();
    } else {
      exp_res.show();
    }
  }

  void TestLIPP(int batch_size, ExperimentalResults& exp_res, 
                bool show_stat=false, bool check_correctness=true) {
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    LIPP<KT, VT> lipp;
    lipp.bulk_load(load_data.data(), load_data.size());      
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end - bulk_load_start).count();

    std::vector<KVT> batch_data;
    std::vector<Status<KT, VT>> batch_results;
    batch_data.reserve(batch_size);
    batch_results.reserve(batch_size);
    // Perform operations in batch
    int num_batches = static_cast<int>(std::ceil(reqs.size() * 1. / batch_size));
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      batch_results.clear();
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
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
      if (check_correctness) {
        for (int i = l; i < r; ++ i) {
          int data_idx = i - l;
          if (reqs[i].op == kQuery) {
            if (batch_results[data_idx].state == kNotFound) {
              std::cout << "Queried key not found" << std::endl;
            }
            ASSERT(batch_results[data_idx].state == kSuccess, "Query fails");
            ASSERT(batch_results[data_idx].kv.first == batch_data[data_idx].first, "Key not match");
          } else if (reqs[i].op == kUpdate) {
            ASSERT(batch_results[data_idx].state == kSuccess, "Update fails");
          }
        }
      }
    }
    exp_res.model_size = lipp.model_size();
    exp_res.index_size = lipp.index_size();
    if (show_stat) {
      lipp.print_depth();
    }
  }

  void TestAlex(int batch_size, ExperimentalResults& exp_res, 
                bool show_stat=false, bool check_correctness=true) {
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    alex::Alex<KT, VT> alex;
    alex.bulk_load(load_data.data(), load_data.size());
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end - bulk_load_start).count();
    // alex.print_stats();
    
    std::vector<KVT> batch_data;
    std::vector<Status<KT, VT>> batch_results;
    batch_data.reserve(batch_size);
    batch_results.reserve(batch_size);
    // Perform operations in batch
    int num_batches = static_cast<int>(std::ceil(reqs.size() * 1. / batch_size));
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      batch_results.clear();
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
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
      if (check_correctness) {
        for (int i = l; i < r; ++ i) {
          int data_idx = i - l;
          if (reqs[i].op == kQuery) {
            if (batch_results[data_idx].state == kNotFound) {
              std::cout << "Queried key not found" << std::endl;
            }
            ASSERT(batch_results[data_idx].state == kSuccess, "Query fails");
            ASSERT(batch_results[data_idx].kv.first == batch_data[data_idx].first, "Key not match");
          } else if (reqs[i].op == kUpdate) {
            ASSERT(batch_results[data_idx].state == kSuccess, "Update fails");
          }
        }
      }
    }
    exp_res.model_size = alex.model_size();
    exp_res.index_size = alex.model_size() + alex.data_size();
    if (show_stat) {
      alex.print_stats();
    }
  }

  void TestPGMIndex(int batch_size, ExperimentalResults& exp_res, 
                    bool show_stat=false, bool check_correctness=true) {
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    pgm::DynamicPGMIndex<KT, VT> pgm_index(load_data.begin(), load_data.end());
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end - bulk_load_start).count();

    std::vector<KVT> batch_data;
    std::vector<Status<KT, VT>> batch_results;
    batch_data.reserve(batch_size);
    batch_results.reserve(batch_size);
    // Perform operations in batch
    int num_batches = static_cast<int>(std::ceil(reqs.size() * 1. / batch_size));
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      batch_results.clear();
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
          pgm_index.insert_or_assign(batch_data[data_idx].first, batch_data[data_idx].second);
        } else if (reqs[i].op == kInsert) {
          pgm_index.insert_or_assign(batch_data[data_idx].first, batch_data[data_idx].second);
        } else if (reqs[i].op == kDelete) {
          pgm_index.erase(batch_data[data_idx].first);
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
      if (check_correctness) {
        for (int i = l; i < r; ++ i) {
          int data_idx = i - l;
          if (reqs[i].op == kQuery) {
            if (batch_results[data_idx].state == kNotFound) {
              std::cout << "Queried key not found" << std::endl;
            }
            ASSERT(batch_results[data_idx].state == kSuccess, "Query fails");
            ASSERT(batch_results[data_idx].kv.first == batch_data[data_idx].first, "Key not match");
          } else if (reqs[i].op == kUpdate) {
            ASSERT(batch_results[data_idx].state == kSuccess, "Update fails");
          }
        }
      }
    }
    exp_res.model_size = pgm_index.index_size_in_bytes();
    exp_res.index_size = pgm_index.size_in_bytes();
    if (show_stat) {
      pgm_index.print_stats();
    }
  }

  void TestBTree(int batch_size, ExperimentalResults& exp_res, 
                  bool show_stat=false, bool check_correctness=true) {
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    btree::btree_map<KT, VT> btree;
    for (int i = 0; i < load_data.size(); ++ i) {
      btree.insert(load_data[i]);
    }
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end - bulk_load_start).count();

    std::vector<KVT> batch_data;
    std::vector<Status<KT, VT>> batch_results;
    batch_data.reserve(batch_size);
    batch_results.reserve(batch_size);
    // Perform operations in batch
    int num_batches = static_cast<int>(std::ceil(reqs.size() * 1. / batch_size));
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      batch_results.clear();
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
      double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      exp_res.sum_indexing_time += time;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({0, time});
      exp_res.step();
      if (check_correctness) {
        for (int i = l; i < r; ++ i) {
          int data_idx = i - l;
          if (reqs[i].op == kQuery) {
            if (batch_results[data_idx].state == kNotFound) {
              std::cout << "Queried key not found" << std::endl;
            }
            ASSERT(batch_results[data_idx].state == kSuccess, "Query fails");
            ASSERT(batch_results[data_idx].kv.first == batch_data[data_idx].first, "Key not match");
          } else if (reqs[i].op == kUpdate) {
            ASSERT(batch_results[data_idx].state == kSuccess, "Update fails");
          }
        }
      }
    }
    exp_res.model_size = 0;
    exp_res.index_size = 0;
  }

  void TestAFLI(int batch_size, ExperimentalResults& exp_res, 
                bool show_stat=false, bool check_correctness=true,
                int bucket_size=-1, int aggregate_size=0) {
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    AFLI<KT, VT> afli;
    afli.BulkLoad(load_data.data(), load_data.size());
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_index_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end - bulk_load_start).count();
    if (show_stat) {
      afli.PrintStat();
    }

    std::vector<KVT> batch_data;
    std::vector<Status<KT, VT>> batch_results;
    batch_data.reserve(batch_size);
    batch_results.reserve(batch_size);
    // Perform operations in batch
    int num_batches = static_cast<int>(std::ceil(reqs.size() * 1. / batch_size));
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      batch_results.clear();
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
      if (check_correctness) {
        for (int i = l; i < r; ++ i) {
          int data_idx = i - l;
          if (reqs[i].op == kQuery) {
            if (batch_results[data_idx].state == kNotFound) {
              std::cout << "Queried key not found" << std::endl;
            }
            ASSERT(batch_results[data_idx].state == kSuccess, "Query fails");
            ASSERT(batch_results[data_idx].kv.first == batch_data[data_idx].first, "Key not match");
          } else if (reqs[i].op == kUpdate) {
            ASSERT(batch_results[data_idx].state == kSuccess, "Update fails");
          }
        }
      }
    }
    exp_res.model_size = afli.model_size();
    exp_res.index_size = afli.index_size();
    if (show_stat) {
      afli.PrintStat();
    }
  }

  void TestNFL(int batch_size, ExperimentalResults& exp_res, 
                std::string weights_path, bool show_stat=false, bool check_correctness=true,
                int bucket_size=-1, int aggregate_size=0) {
    // Start to bulk load
    auto bulk_load_start = std::chrono::high_resolution_clock::now();
    NFL<KT, VT> nfl(weights_path, batch_size);
    uint32_t tail_conflicts = nfl.AutoSwitch(load_data.data(), load_data.size());
    auto bulk_load_mid = std::chrono::high_resolution_clock::now();
    nfl.IndexBulkLoad(load_data.data(), load_data.size(), tail_conflicts);
    // nfl.BulkLoad(load_data.data(), load_data.size());
    auto bulk_load_end = std::chrono::high_resolution_clock::now();
    exp_res.bulk_load_trans_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_mid - bulk_load_start).count();
    exp_res.bulk_load_index_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end - bulk_load_mid).count();
    if (show_stat) {
      nfl.PrintStat();
    }

    std::vector<KVT> batch_data;
    std::vector<Status<KT, VT>> batch_results;
    batch_data.reserve(batch_size);
    batch_results.reserve(batch_size);
    // Perform operations in batch
    int num_batches = static_cast<int>(std::ceil(reqs.size() * 1. / batch_size));
    exp_res.latencies.reserve(num_batches * 3);
    exp_res.need_compute.reserve(num_batches * 3);
    for (int batch_idx = 0; batch_idx < num_batches; ++ batch_idx) {
      batch_data.clear();
      batch_results.clear();
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
      double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(mid - start).count();
      double time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count();
      exp_res.sum_transform_time += time1;
      exp_res.sum_indexing_time += time2;
      exp_res.num_requests += batch_data.size();
      exp_res.latencies.push_back({time1, time2});
      exp_res.step();
      if (check_correctness) {
        for (int i = l; i < r; ++ i) {
          int data_idx = i - l;
          if (reqs[i].op == kQuery) {
            if (batch_results[data_idx].state == kNotFound) {
              std::cout << "Queried key not found" << std::endl;
            }
            ASSERT(batch_results[data_idx].state == kSuccess, "Query fails");
            ASSERT(batch_results[data_idx].kv.first == batch_data[data_idx].first, "Key not match");
          } else if (reqs[i].op == kUpdate) {
            ASSERT(batch_results[data_idx].state == kSuccess, "Update fails");
          }
        }
      }
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