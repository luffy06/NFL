import os
import sys
import math

methods = ['nfl', 'afli', 'lipp', 'alex', 'pgm-index', 'btree']
metrics = ['bulkloading-transform', 'bulkloading-indexing', 
          'model-size', 'index-size', 'throughput', 
          'avg-latency-transform', 'avg-latency-indexing', 
          'p50-latency-transform', 'p50-latency-indexing', 
          'p75-latency-transform', 'p75-latency-indexing', 
          'p99-latency-transform', 'p99-latency-indexing', 
          'p995-latency-transform', 'p995-latency-indexing', 
          'p9999-latency-transform', 'p9999-latency-indexing', 
          'max-latency-transform', 'max-latency-indexing']

def check_methods(algo):
  for m in methods:
    if m in algo:
      return True
  return False

def order(k):
  if k == 'nfl':
    return 1
  elif k == 'afli':
    return 1
  elif k == 'lipp':
    return 2
  elif k == 'alex':
    return 3
  elif k == 'pgm-index':
    return 4
  elif k == 'btree':
    return 5
  else:
    return 6

def find_index(metric):
  idx_list = []
  for i, m in enumerate(metrics):
    if m.startswith(metric):
      idx_list.append(i)
  return idx_list

def load_result(result_path, workload, batch_size, metric, output_path):
  f = open(result_path, 'r')
  lines = list(map(lambda x: x.strip(), f.readlines()))
  f.close()
  idx_list = find_index(metric)
  n = len(idx_list)
  total_result = []
  for i, line in enumerate(lines):
    line = line.split()
    if len(line) < 3 or line[0] != workload or int(line[2]) != batch_size:
      continue
    if not check_methods(line[1]):
      continue
    if len(line) < 10:
      total_result.append((line[1], 0))
    else:
      s = 0
      for idx in idx_list:
        s += float(line[idx + 3])
      val = [s]
      for idx in idx_list:
        val.append(float(line[idx + 3]) / s if s != 0 else float(line[idx + 3]))
      total_result.append([line[1]] + val)
  total_result.sort(key=lambda x: order(x[0]))
  result = []
  j = 0
  ct = 0
  mean = [0 for i in range(n + 1)]
  for i, r in enumerate(total_result):
    if r[0] == total_result[j][0]:
      ct += 1
      for k in range(0, n + 1):
        mean[k] += r[k + 1]
    else:
      for k in range(n + 1):
        mean[k] /= ct
      var = [0 for i in range(n + 1)]
      for k in range(j, i):
        for u in range(n + 1):
          var[u] += (total_result[k][u + 1] - mean[u]) ** 2
      for u in range(n + 1):
        var[u] = math.sqrt(var[u] / ct)
      result.append([mean[0] * mean[i] for i in range(n, 0, -1)])
      j = i
      ct = 1
      mean = r[1:]
  if ct > 0:
    for k in range(n + 1):
      mean[k] /= ct
    var = [0 for i in range(n + 1)]
    for k in range(j, i):
      for u in range(n + 1):
        var[u] += (total_result[k][u + 1] - mean[u]) ** 2
    for u in range(n + 1):
      var[u] = math.sqrt(var[u] / ct)
    result.append([mean[0] * mean[i] for i in range(n, 0, -1)])
  # for i, res in enumerate(result):
  #   print(res[0], end='\t')
  # print()
  # Print mean results
  f = open(output_path, 'a+')
  for i, res in enumerate(result):
    if i > 0:
      f.write('\t')
    for j, r in enumerate(res):
      if j > 0:
         f.write('\t')
      f.write(str(r))
  f.write('\n')
  # Print variance results
  # for i, res in enumerate(result):
  #   print(res[2], end='\t')
  f.close()

if __name__ == '__main__':
  if len(sys.argv) < 6:
    sys.exit('usage: ./load_result.py (result path) (workload name) (batch size) (metric) (output path)')
  
  result_path = sys.argv[1]
  workload = sys.argv[2]
  batch_size = int(sys.argv[3])
  metric = sys.argv[4]
  output_path = sys.argv[5]
  load_result(result_path, workload, batch_size, metric, output_path)