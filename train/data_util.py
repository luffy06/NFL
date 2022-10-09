import scipy.stats as ss
import numpy as np
import torch
import os
import psutil

def load_data(location, data_type):
  inputs = np.loadtxt(location + '-' + data_type + '.txt', delimiter='\t', dtype=np.float64, ndmin=2)
  data = torch.from_numpy(inputs).double()
  return data
  
def sample_data(keys : torch.Tensor, num):
  assert num <= keys.shape[0]
  perm = np.random.permutation(keys.shape[0])
  index = perm[:num]
  return keys[index]

def print_tensor(x, name=None):
  assert len(x.shape) == 2
  if name != None:
    print(name, '(%d, %d)' % (x.shape[0], x.shape[1]))
  for i in range(x.shape[0]):
    # print('%d\t' % i, end='\t')
    for j in range(x.shape[1]):
      print('%.32f' % x[i][j], end='\t' if j != x.shape[1] - 1 else '\n')

