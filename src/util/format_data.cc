#include "util/common.h"

using namespace nfl;

int get_num_keys(std::string name) {
  int tot_num = 0;
  for (int i = name.size() - 1; i >= 0; -- i) {
    if (name[i] == 'M') {
      int p = 1;
      for (int j = i - 1; j >= 0 && name[j] != '-'; j --, p *= 10) {
        tot_num = tot_num + (name[j] - '0') * p;
      }
      tot_num = tot_num * 1000000;
      break;
    }
  }
  return tot_num;
}

template<typename T>
T cut_key(T key, size_t len, bool verbose=false) {
  std::string key_str = str<T>(key);
  size_t start = key_str.size() < len ? 0 : key_str.size() - len;
  std::string key_cut_str = key_str.substr(start, len);
  if (verbose) {
    std::cout << std::fixed << std::setw(std::numeric_limits<T>::digits10) << key << std::endl;
    std::cout << std::fixed << std::setw(std::numeric_limits<T>::digits10) << key_str << std::endl;
    std::cout << std::fixed << std::setw(std::numeric_limits<T>::digits10) << key_cut_str << std::endl;
  }
  return ston<std::string, T>(key_cut_str);
}

template<typename T, typename P>
void format(std::string data_dir, std::string data_name, std::string suffix, int num_keys = 0) {
  std::string source_path = path_join(data_dir, data_name + suffix);
  for (int i = 0; i < data_name.size(); ++ i) {
    if (data_name[i] == '_') {
      data_name[i] = '-';
    }
  }
  std::string output_path = path_join(data_dir, data_name + ".bin");
  std::ifstream in(source_path, std::ios::binary | std::ios::in);
  if (!in.is_open()) {
    std::cout << "File [" << source_path << "] does not exist" << std::endl;
    exit(-1);
  }
  std::cout << "First type bytes [" << str<size_t>(sizeof(T)) << "], " 
            << "second type bytes [" << str<size_t>(sizeof(P)) << "]" 
            << std::endl;
  std::vector<T> origin_keys;
  std::vector<P> double_keys;
  if (num_keys == 0) {
    in.read((char*)&num_keys, sizeof(T));
  }
  std::cout << "[" << num_keys << "] keys found in " << data_name << std::endl;
  origin_keys.resize(num_keys);
  in.read((char*)origin_keys.data(), num_keys * sizeof(T));
  in.close();

  if (sizeof(T) > sizeof(P) || (sizeof(T) == sizeof(P) && !std::numeric_limits<T>::is_signed)) {
    std::cout << "Cut keys" << std::endl;
    assert_p(std::numeric_limits<T>::is_integer, "Cannot cut the non-integer type");
    size_t key_len = std::numeric_limits<P>::digits10;
    for (int i = 0; i < origin_keys.size(); ++ i) {
      origin_keys[i] = cut_key<T>(origin_keys[i], key_len);
    }
  }
  std::sort(origin_keys.begin(), origin_keys.end());

  double_keys.reserve(num_keys);
  for (int i = 0; i < num_keys; ++ i) {
    if (i == 0 || !compare<T>(origin_keys[i], origin_keys[i - 1])) {
      double_keys.push_back(static_cast<P>(origin_keys[i]));
    }
  }
  int num_unique = double_keys.size();
  std::cout << "[" << num_unique << "] unique keys in " << data_name << std::endl;
  std::ofstream out(output_path, std::ios::binary | std::ios::out);
  out.write((char*)&num_unique, sizeof(int));
  out.write((char*)double_keys.data(), num_keys * sizeof(P));
  out.close();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
      std::cout << "No enough parameters for formatting workloads\n"
                << "Please input: format (base path) (workload) (data type)" << std::endl;
      exit(-1);
  }
  std::string base_dir = std::string(argv[1]);
  std::string data_name = std::string(argv[2]);
  std::string data_type = std::string(argv[3]);
  if (data_type == "uint64") {
    format<uint64_t, double>(path_join(base_dir, "data"), data_name, std::string("_") + data_type);
  } else if (data_type == "float64") {
    int num_keys = get_num_keys(data_name);
    format<double, double>(path_join(base_dir, "data"), data_name, ".bin.data", num_keys);
  } else if (data_type == "int64") {
    int num_keys = get_num_keys(data_name);
    format<long long, double>(path_join(base_dir, "data"), data_name, ".bin.data", num_keys);
  } else {
    std::cout << "Unspported data type [" << data_type << "]" << std::endl;
    exit(-1);
  }
  return 0;
}