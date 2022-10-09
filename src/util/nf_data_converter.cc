#include "benchmark/workload.h"
#include "util/common.h"

using namespace nfl;

template<typename KT, typename VT>
void write_workload_keys(std::string workload_path, 
                          std::string flow_input_dir, 
                          double prop = 1) {
  std::string workload_name = get_workload_name(workload_path);
  std::cout << "Write raw keys [" << int(prop * 100) << "%] from [" 
            << workload_name << "]" << std::endl;
  std::vector<std::pair<KT, VT>> init_data;
  std::vector<Request<KT, VT>> run_reqs;
  load_data(workload_path, init_data, run_reqs);
  std::string output_path = path_join(flow_input_dir, workload_name + 
                            (std::fabs(prop - 1) < 1e-3 ? "" : "-small") + 
                            "-training.txt");
  std::ofstream out(output_path, std::ios::out);
  if (!out.is_open()) {
    std::cout << "File [" << output_path << "] does not exist" << std::endl;
    exit(-1);
  }
  assess_data(init_data.data(), init_data.size());
  int n = int(init_data.size() * prop);
  for (int i = 0; i < n; ++ i) {
    out << std::fixed << std::setprecision(std::numeric_limits<KT>::digits10) 
        << init_data[i].first << std::endl;
  }
  out.close();
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "No enough parameters" << std::endl;
    std::cout << "Please input: nf_convert (workload path) (key type) (proportion of keys)" 
              << "(flow input director)" << std::endl;
    exit(-1);
  }
  std::string workload_path = std::string(argv[1]);
  std::string key_type = std::string(argv[2]);
  double prop = ston<char*, int>(argv[3]) / 100.;
  std::string flow_input_dir = std::string(argv[4]);
  if (key_type == "float64") {
    write_workload_keys<double, long long>(workload_path, flow_input_dir, prop);
  } else {
    std::cout << "Unsupported key type [" << key_type << "]" << std::endl;
    exit(-1);
  }
  return 0;
}