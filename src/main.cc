#include "benchmark/kv_evaluator.h"

using namespace kvevaluator;

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "No enough parameters" << std::endl;
    std::cout << "Please input: evaluate (index name) (batch size) "
              << "(workload path) (key type) [config path] " 
              << "[show incremental updates]" << std::endl;
    exit(-1);
  }
  std::string index_name = std::string(argv[1]);
  int batch_size = STRTONUM<char*, int>(argv[2]);
  std::string workload_path = std::string(argv[3]);
  std::string key_type = std::string(argv[4]);
  std::string config_path = argc > 5 ? std::string(argv[5]) : "";
  std::string show_inc_thro = argc > 6 ? std::string(argv[6]) : "";
  srand(kSEED);
  if (key_type == "float64") {
    KVEvaluator<double, long long> kv_evaluator;
    kv_evaluator.TestWorkloads(index_name, batch_size, workload_path, 
                                config_path, show_inc_thro != "");
  } else {
    std::cout << "Unsupported key type [" << key_type << "]" << std::endl;
    exit(-1);
  }
  return 0;
}
