import ot
import torch
from models import *

class KernelEncoder(torch.nn.Module):
  def __init__(self, kernel_func_list):
    super(KernelEncoder, self).__init__()
    self.kernel_func_list = kernel_func_list

  def forward(self, x: torch.Tensor):
    z = None
    for kernel_func in self.kernel_func_list:
      y = kernel_func(x)
      z = torch.cat((z, y), 1) if z != None else y
    return z

  def save_weights(self, f):
    pass

  def __repr__(self):
    return 'KernelEncoder(num_kernels={})'.format(len(self.kernel_func_list))

class Partition(torch.nn.Module):
  def __init__(self, feature_dim, shifts, keep_res=True):
    super(Partition, self).__init__()
    assert feature_dim > 1
    self.feature_dim = feature_dim
    self.shifts = shifts
    self.keep_res = keep_res
  
  def forward(self, x: torch.Tensor):
    x_clone = x.clone()
    features = None
    for dim_i in range(self.feature_dim - 1):
      feature_i = torch.floor(x_clone)
      features = torch.cat((features, feature_i), 1) if features != None else feature_i
      x_clone = (x_clone - feature_i) * self.shifts
    if self.keep_res:
      x_clone /= self.shifts
      features = torch.cat((features, x_clone), 1)
    else:
      features = torch.cat((features, torch.floor(x_clone * self.shifts)), 1)
    return features
  
  def save_weights(self, f):
    pass

  def __repr__(self):
    return 'Partition(feature_dim={}, shifts={}, keep_res={})'.format(self.feature_dim, self.shifts, self.keep_res)

class ReduceDecoder(torch.nn.Module):
  def __init__(self, reduce_dim):
    super(ReduceDecoder, self).__init__()
    self.reduce_dim = reduce_dim
  
  def forward(self, x: torch.Tensor):
    assert self.reduce_dim >= 0 and self.reduce_dim < x.shape[1]
    return x[:, self.reduce_dim].unsqueeze(1)

  def save_weights(self, f):
    pass

  def __repr__(self):
    return 'ReduceDecoder(reduce_dim={})'.format(self.reduce_dim)

class SumDecoder(torch.nn.Module):
  def forward(self, x: torch.Tensor):
    return x.sum(-1).unsqueeze(1)

  def save_weights(self, f):
    pass

  def __repr__(self):
    return 'SumDecoder()'

class DistTransformer(torch.nn.Module):
  def __init__(self, num_flows=1, num_layers=2, input_dim=2, hidden_dim=1, 
                encoder_type='partition', encoder_config={'shifts':1000000}, 
                decoder_type='sum', decoder_config=None):
    super(DistTransformer, self).__init__()
    self.encoder = self.create_encoder(encoder_type, input_dim, encoder_config)
    self.flow = self.create_bnaf(num_flows, num_layers, input_dim, hidden_dim)
    self.decoder = self.create_decoder(decoder_type, decoder_config)
    
  def create_encoder(self, encoder_type, feature_dim, encoder_config):
    if encoder_type == 'partition':
      return Partition(feature_dim, encoder_config['shifts'], encoder_config['keep_res'])
    elif encoder_type == 'kernel_func':
      kernel_func_list = [torch.nn.Tanh(), torch.nn.Sigmoid()]
      return KernelEncoder(kernel_func_list)
    return None

  def create_decoder(self, decoder_type, decoder_config):
    if decoder_type == 'reduce':
      return ReduceDecoder(decoder_config['reduce_dim'])
    elif decoder_type == 'sum':
      return SumDecoder()
    elif decoder_type == 'shift_sum':
      return ShiftSumDecoder(decoder_config['shifts'], decoder_config['keep_res'])
    return None

  def create_bnaf(self, num_flows, num_layers, input_dim, hidden_dim):
    flows = []
    for f in range(num_flows):
        # First layer
        layers = [MaskedWeight(input_dim, input_dim * hidden_dim, 
                                dim=input_dim, bias=False), 
                  BNAFTanh()]
        # Inner layers
        for _ in range(num_layers - 2):
            layers.append(MaskedWeight(input_dim * hidden_dim, 
                                        input_dim * hidden_dim, 
                                        dim=input_dim, bias=False))
            layers.append(BNAFTanh())
        # Last layer
        layers.append(MaskedWeight(input_dim * hidden_dim, input_dim, 
                                    dim=input_dim, bias=False))
        flows.append(
            BNAF(*(layers), res=False)
        )
        if f < num_flows - 1:
            flows.append(Permutation(input_dim, 'flip'))
    return BNAFSequential(*flows)

  def loss(self, x: torch.Tensor):
    assert self.training
    z, log_diag = self.forward(x)
    # Compute log probability
    mean = torch.zeros_like(z)
    var = torch.ones_like(z)
    log_p_z = torch.distributions.Normal(mean, var).log_prob(z).sum(-1)
    loss = -(log_p_z + log_diag).mean()
    return loss

  def forward(self, x: torch.Tensor):
    if self.training:
      features = self.encoder(x) if self.encoder != None else x
      features, grad = self.flow(features)
      z = self.decoder(features) if self.decoder != None else features
      return z, grad
    else:
      features = self.encoder(x) if self.encoder != None else x
      features = self.flow(features)
      z = self.decoder(features) if self.decoder != None else features
      return z

  def save_weights(self, f):
    if self.encoder != None:
      self.encoder.save_weights(f)
    self.flow.save_weights(f)
    if self.decoder != None:
      self.decoder.save_weights(f)
  
  def __repr__(self):
    if self.encoder != None:
      print(self.encoder)
    print(self.flow)
    if self.decoder != None:
      print(self.decoder)
    return ''

