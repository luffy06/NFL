import os
import json
import argparse
import pprint
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import time
import sys
import random
from tqdm import trange
from data_util import *
from distribution_transformer import *
from assess_quality import *

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10, sci_mode=False)

def plot_dist(keys: np.ndarray, path, fig_type, normalize=False):
  if normalize:
    if np.max(keys) != np.min(keys):
      keys = (keys - np.min(keys)) / (np.max(keys) - np.min(keys))
    else:
      keys /= np.max(keys)
  colors = ['red', 'blue', 'black']
  fig = plt.figure()
  x = np.sort(keys, 0)
  # sns.distplot(x, hist=False, kde=False, fit=stats.norm, 
  #             fit_kws={'color': 'r', 'label':'u=0,s=1','linestyle':'-'})
  if fig_type == 'pdf':
    # plt.hist(x, 100, density=True, facecolor='blue', alpha=0.75)
    sns.kdeplot(np.array(x))
  elif fig_type == 'cdf':
    y = np.arange(0, x.shape[0])
    plt.plot(x.tolist(), y)
    plt.scatter(x.tolist(), y, s=7)
  plt.title(fig_type)
  plt.savefig(path + '_' + fig_type + '.png')
  plt.close()

def train(train_keys, model, optimizer, scheduler, best_loss, args, model_path):
  model.train()
  model = model.to(args.device)
  last_loss = None
  batch_size = args.batch_dim
  for epoch in range(args.steps):
    print('Epoch:{}'.format(epoch))
    start_time = time.time()
    epoch_loss = 0
    num_batches = int(np.ceil(train_keys.shape[0] / batch_size))
    iterator = trange(num_batches, smoothing=0, dynamic_ncols=True)
    for i in iterator:
      l = i * batch_size
      r = np.min([(i + 1) * batch_size, train_keys.shape[0]])
      x = train_keys[l:r].to(args.device)
      loss = model.loss(x)
      if np.isnan(loss.item()):
        print('Error: Loss is NaN')
        exit()
      if np.isinf(loss.item()):
        print('Error: Loss is INF')
        exit()

      epoch_loss += loss.item()
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
      optimizer.step()
      optimizer.zero_grad()
      scheduler.step(loss)

      iterator.set_postfix(
        loss="{:.2f}".format(loss.data.cpu().numpy()), refresh=False
      )

    epoch_time = time.time() - start_time
    print('Time Cost of Epoch {}'.format(epoch_time))

    print('Epoch mean loss:{}'.format(epoch_loss / train_keys.shape[0]))
    if last_loss != None and (epoch_loss > last_loss 
        or np.fabs((last_loss - epoch_loss) / last_loss) < 0.01):
      print('Process: Early stop at epoch-{}'.format(epoch))
      break
    if epoch_loss < best_loss:
      best_loss = epoch_loss
      save(model, optimizer, model_path)
    last_loss = epoch_loss

  return best_loss

def test(source_keys, model, args):
  model = model.to(args.device)
  model.eval()
  tran_keys = None
  with torch.no_grad():
    batch_size = 8192
    num_batches = int(np.ceil(source_keys.shape[0] / batch_size))
    iterator = trange(num_batches, smoothing=0, dynamic_ncols=True)
    for i in iterator:
      l = i * batch_size
      r = np.min([(i + 1) * batch_size, source_keys.shape[0]])
      x = source_keys[l:r].to(args.device)
      z = model(x).to('cpu')
      if torch.isnan(z).any():
        print('Error: Nan keys')
        exit()
      tran_keys = torch.cat((tran_keys, z), 0) if tran_keys != None else z
  return tran_keys

def load(model, optimizer, path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])

def save(model, optimizer, path):
  d = {}
  d["model"] = model.state_dict()
  d["optimizer"] = optimizer.state_dict()
  torch.save(d, path)

def save_weights(model, mean, var, args, weight_path):
  f = open(weight_path, 'w')
  model.eval()
  f.write('%d\t%d\t%d\n' % (args.input_dim, args.input_dim * args.hidden_dim, args.num_layers))
  f.write('%.16f\t%.16f\n' % (mean.item(), var.item()))
  model.save_weights(f)
  f.close()

if __name__ == '__main__':
  # Initializing parameters for flow models
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--data_dir', type=str, default='data')
  parser.add_argument('--data_name', type=str, default='lognormal-200M-100R-zipf')
  parser.add_argument('--seed', type=int, default=1000000007)
  parser.add_argument('--plot', type=bool, default=True)
  parser.add_argument('--log_file', type=bool, default=True)

  # Parameters for encoder/decoder
  parser.add_argument('--encoder_type', type=str, default='partition')
  parser.add_argument('--decoder_type', type=str, default='sum')
  parser.add_argument('--shifts', type=int, default=1000000)
  parser.add_argument('--keep_res', type=bool, default=True)
  parser.add_argument('--reduce_dim', type=int, default=-1)

  # Parameters for flow models
  parser.add_argument("--num_flows", type=int, default=1)
  parser.add_argument('--num_layers', type=int, default=2)
  parser.add_argument('--input_dim', type=int, default=1)
  parser.add_argument('--hidden_dim', type=int, default=1)

  # Parameters for training
  parser.add_argument('--train_ratio', type=float, default=0.1)
  parser.add_argument('--num_train', type=int, default=3)
  parser.add_argument('--learning_rate', type=float, default=1e-1)
  parser.add_argument('--clip_norm', type=float, default=.1)
  parser.add_argument('--steps', type=int, default=15)
  parser.add_argument('--patience', type=int, default=2000)
  parser.add_argument('--decay', type=float, default=0.5)
  parser.add_argument('--batch_dim', type=int, default=4096)
  parser.add_argument('--loss_func', type=str, default='normal')
  parser.add_argument('--load', type=str, default=None)
  args = parser.parse_args()
  if not os.path.exists('checkpoint'):
    os.mkdir('checkpoint')
  now = datetime.datetime.now()
  time_str = '-%4d-%02d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
  checkpoint_dir = os.path.join('checkpoint', args.data_name + time_str)
  if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
  model_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

  # Set random seed
  random_seed = args.seed
  seed = random_seed
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)

  if args.log_file:
    log_filename = os.path.join(checkpoint_dir, 'outputs.log')
    log_f = open(log_filename, 'w')
  else:
    log_f = None

  print('Process: Saving arguments...')
  with open(os.path.join(checkpoint_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4, sort_keys=True)

  print('Arguments')
  pprint.pprint(args.__dict__)

  # Loading data, analyzing data, segmenting data
  print('Process: Loading data...')
  start_time = time.time()
  load_keys = load_data(os.path.join(args.data_dir, args.data_name), 'training')
  load_durations = time.time() - start_time
  print('Time Cost of Loading data {}'.format(load_durations))

  print('Process: Evaluating original keys [{}]...'.format(load_keys.size()))
  evaluate_keys(load_keys.detach().cpu().numpy().squeeze(), log_f=log_f)

  print('Process: Using Min-Max Normalization')
  global_mean = torch.min(load_keys)
  global_var = (torch.max(load_keys) - torch.min(load_keys)) / args.shifts
  load_keys = (load_keys - global_mean) / global_var
  print('Min {} Max {}'.format(torch.min(load_keys), torch.max(load_keys)))
  if args.log_file:
    print('Min {} Max {}'.format(torch.min(load_keys), torch.max(load_keys)), file=log_f)

  encoder_config = {}
  decoder_config = {}
  if args.encoder_type == 'partition':
    encoder_config['shifts'] = args.shifts
    encoder_config['keep_res'] = args.keep_res
  
  if args.decoder_type == 'reduce':
    decoder_config['reduce_dim'] = args.reduce_dim
  elif args.decoder_type == 'shift_sum':
    decoder_config['shifts'] = args.shifts
    decoder_config['keep_res'] = args.keep_res
  print('Encoder configs\t{}'.format(encoder_config))
  print('Decoder configs\t{}'.format(decoder_config))
  if args.log_file:
    print('Encoder configs\t{}'.format(encoder_config), file=log_f)
    print('Decoder configs\t{}'.format(decoder_config), file=log_f)

  num_train_keys = np.max((np.clip(int(load_keys.shape[0] * args.train_ratio), 
                          1, load_keys.shape[0]), 10000))
  train_keys_list = []
  for i in range(args.num_train):
    print('Process: Spliting training data...')
    print('Number of training keys {}'.format(num_train_keys))
    if args.log_file:
      print('Number of training keys {}'.format(num_train_keys), file=log_f)
    train_keys = sample_data(load_keys, num_train_keys)
    train_keys_list.append(train_keys)

  model = DistTransformer(args.num_flows, args.num_layers, args.input_dim, 
                          args.hidden_dim, args.encoder_type, encoder_config, 
                          args.decoder_type, decoder_config)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                amsgrad=True)
  print('Parameters={}, n_dims={}'.format(sum((p != 0).sum() 
        if len(p.shape) > 1 else torch.tensor(p.shape).item() 
        for p in model.parameters()), args.input_dim))
  print('{}'.format(model))
  if args.log_file:
    print('Parameters={}, n_dims={}'.format(sum((p != 0).sum()
          if len(p.shape) > 1 else torch.tensor(p.shape).item() 
          for p in model.parameters()), args.input_dim), file=log_f)
    oldstdout = sys.stdout
    sys.stdout = log_f
    print('{}'.format(model))
    sys.stdout = oldstdout

  if args.load != None:
    load_path = os.path.join(args.load, 'checkpoint.pt')
    load(model, optimizer, load_path)
  else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                  factor=args.decay, patience=args.patience, min_lr=5e-4, 
                  verbose=True, threshold_mode='abs')

    best_loss = 1e20
    for i, train_keys in enumerate(train_keys_list):

      print('Process: {} Training...'.format(i + 1))
      start_time = time.time()
      best_loss = train(train_keys, model, optimizer, scheduler, best_loss, args, model_path)
      train_durations = time.time() - start_time
      print('Time Cost of Training {}'.format(train_durations))
      if args.log_file:
        print('Time Cost of Training {}'.format(train_durations), file=log_f)
      load(model, optimizer, model_path)
 
  print('Process: Saving weights for c++ inference...')
  weight_path = os.path.join(checkpoint_dir, args.data_name + '-weights.txt')
  save_weights(model, global_mean, global_var, args, weight_path)
  
  print('Process: Transforming keys...')
  tran_keys = test(load_keys, model, args)
  
  num_unordered = ((tran_keys[1:] - tran_keys[:-1]) < 0).sum().item()
  if num_unordered > 0:
    print('Process: Sorting keys...')
    tran_keys = torch.sort(tran_keys, 0)[0]

  print('Process: Evaluating the transformed keys...')
  evaluate_keys(tran_keys.detach().cpu().numpy().squeeze(), log_f=log_f)

  if args.plot:
    num_samples = 100000
    if load_keys.shape[0] > num_samples:
      print('Process: Sample data for plotting...')
      plot_keys = sample_data(load_keys, num_samples)
      plot_keys = torch.sort(plot_keys, 0)[0]
      tran_keys = test(plot_keys, model, args)
    print('Process: Plot the transofmred distribution...')
    plot_dist(plot_keys.detach().cpu().numpy().squeeze(), os.path.join(checkpoint_dir, args.data_name + '_load_dist'), 'pdf')
    plot_dist(tran_keys.detach().cpu().numpy().squeeze(), os.path.join(checkpoint_dir, args.data_name + '_tran_dist'), 'pdf')
    plot_dist(plot_keys.detach().cpu().numpy().squeeze(), os.path.join(checkpoint_dir, args.data_name + '_load_dist'), 'cdf')
    plot_dist(tran_keys.detach().cpu().numpy().squeeze(), os.path.join(checkpoint_dir, args.data_name + '_tran_dist'), 'cdf')