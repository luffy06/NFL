import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_util import *

def compute_unordered_keys(keys: np.ndarray, verbose=False):
  return ((keys[1:] - keys[:-1]) < 0).sum().item()

def compute_number_duplicated_keys(keys: np.ndarray, verbose=False):
  sorted_keys = np.sort(keys, 0)
  num_dup = ((keys[1:] - keys[:-1]) == 0).sum().item()
  del sorted_keys
  return num_dup

def build_linear_model(keys: np.ndarray, y=None):
  n = keys.shape[0]
  if y == None:
    y = np.arange(n)
  elif y == 'prop':
    y = n * (keys - np.min(keys)) / (np.max(keys) - np.min(keys))
  x_sum = keys.sum()
  y_sum = y.sum()
  xx_sum = (keys ** 2).sum()
  xy_sum = (keys * y).sum()
  if n * xx_sum == x_sum * x_sum:
      slope = 0
  else:
      slope = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum)
  intercept = (y_sum - slope * x_sum) / n
  del y
  return slope, intercept

def compute_conflicts(keys: np.ndarray, slope, intercept, amp=-1, tail_percent=0.99, log_f=None, verbose=False):
  n = keys.shape[0]
  intercept = -keys[0] * slope + 0.5
  print('Slope {}, intercept {}'.format(slope, intercept))
  if log_f != None:
    print('Slope {}, intercept {}'.format(slope, intercept), file=log_f)
  pos = np.floor(keys * slope + intercept)
  if amp != -1:
      max_size = int(n * amp)
      pos[pos < 0] = 0
      pos[pos >= max_size] = max_size - 1
  else:
      max_size = pos[-1] + 1
  print('Space amplification {}'.format((pos[-1] - pos[0]) / n))
  if log_f != None:
    print('Space amplification {}'.format((pos[-1] - pos[0]) / n), file=log_f)
  conflicts = []
  max_conflicts = 0
  sum_conflicts = 0
  conflicts_per_pos = 0
  last_pos = pos[0]
  for i in range(1, n):
    if pos[i] == pos[i - 1]:
      conflicts_per_pos = conflicts_per_pos + 1
    else:
      last_pos = pos[i]
      max_conflicts = np.maximum(max_conflicts, conflicts_per_pos)
      sum_conflicts = sum_conflicts + conflicts_per_pos
      if conflicts_per_pos > 0:
        conflicts.append(conflicts_per_pos)
      conflicts_per_pos = 0
  if conflicts_per_pos > 0:
    max_conflicts = np.maximum(max_conflicts, conflicts_per_pos)
    sum_conflicts = sum_conflicts + conflicts_per_pos
    conflicts.append(conflicts_per_pos)
  conflicts.sort(key=lambda x: x)
  if len(conflicts) > 0:
    tail_conflicts = conflicts[int(tail_percent * len(conflicts)) - 1]
    avg_conflicts = sum_conflicts / len(conflicts)
  else:
    max_conflicts = 0
    tail_conflicts = 0
    avg_conflicts = 0
  del pos
  return sum_conflicts, max_conflicts, tail_conflicts, avg_conflicts

def evaluate_keys(keys: np.ndarray, log_f=None, verbose=False):
  print('#' * 100)
  if log_f != None:
    print('#' * 100, file=log_f)
  assert len(keys.shape) == 1
  print('Assess the {} of keys'.format(keys.shape[0]))
  if log_f != None:
    print('Assess the {} of keys'.format(keys.shape[0]), file=log_f)
  num_unordered = compute_unordered_keys(keys, verbose=verbose)
  print('Number of unordered keys {}'.format(num_unordered))
  if log_f != None:
    print('Number of unordered keys {}'.format(num_unordered), file=log_f)
  num_duplicated = compute_number_duplicated_keys(keys, verbose=verbose)
  print('Number of duplicated keys {}'.format(num_duplicated))
  if log_f != None:
    print('Number of duplicated keys {}'.format(num_duplicated), file=log_f)

  if num_duplicated != keys.shape[0] - 1:
    print('Range [{}, {}]'.format(np.min(keys), np.max(keys)))
    if log_f != None:
      print('Range [{}, {}]'.format(np.min(keys), np.max(keys)), file=log_f)

    slope, intercept = build_linear_model(keys)
    conf_stat = compute_conflicts(keys, slope, intercept, amp=-1, log_f=log_f, verbose=verbose)
    print('Absolute Conflicts\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))
    if log_f != None:
      print('Absolute Conflicts\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]), file=log_f)

    slope, intercept = build_linear_model(keys)
    conf_stat = compute_conflicts(keys, slope, intercept, amp=1.5, log_f=log_f, verbose=verbose)
    print('Limited Space Conflicts\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))
    if log_f != None:
      print('Limited Space Conflicts\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]), file=log_f)

    slope, intercept = build_linear_model(keys, 'prop')
    conf_stat = compute_conflicts(keys, slope, intercept, amp=1.5, log_f=log_f, verbose=verbose)
    print('Proportional Building\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))
    if log_f != None:
      print('Proportional Building\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]), file=log_f)
  else:
    print('All keys are the same')
    if log_f != None:
      print('All keys are the same', file=log_f)
  print('#' * 100)
  if log_f != None:
    print('#' * 100, file=log_f)

if __name__ == '__main__':
  workload_path = 'data/longlat-200M-100R-zipf-small'
  keys = load_data(workload_path, 'training')
  evaluate_keys(keys.detach().cpu().numpy().squeeze())
  keys = sample_data(keys, int(0.1 * keys.shape[0]))
  keys = torch.sort(keys, 0)[0]
  n = len(keys)
  fig = plt.figure()
  plt.plot(keys, [i for i in range(len(keys))])
  plt.savefig('origin.png')
  slope, intercept = build_linear_model(keys.detach().cpu().numpy().squeeze())
  k = 10
  slope = slope * k / n
  intercept = intercept * k / n
  pos = torch.floor(slope * keys + intercept)
  fig = plt.figure()
  plt.plot(pos, [i for i in range(len(pos))])
  plt.savefig('pos.png')

