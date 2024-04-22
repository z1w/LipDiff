import torch
from torch import nn
from numpy import linalg as LA

def lanc_iter(mat, size, device):
  lan_subm = torch.zeros(size, size).to(device)
  dim = mat.shape[0]
  id = torch.eye(dim).to(device)
  rv = torch.rand(dim, 1).double().to(device)
  rand_vec = rv/torch.linalg.norm(rv)
  b = 1
  q_old = 0
  l_min = float('inf')
  for i in range(size):
    q = rand_vec/b
    a = torch.matmul(torch.t(q), torch.matmul(mat, q))
    rand_vec = torch.matmul((mat-a*id), q) - b*q_old
    b_old = b
    b = torch.linalg.norm(rand_vec)
    q_old=q
    lan_subm[i,i] = a
    l_min = min(l_min, a-abs(b_old)-abs(b))
    if b==0 or i== size-1:
      break
    lan_subm[i, i+1] = b
    lan_subm[i+1, i] = b
  return lan_subm


def max_eigv(A, device, lanc, lan_steps):
  if lanc:
    A = lanc_iter(A, lan_steps, device)
  L = torch.linalg.eigvalsh(A)
  return L[-1]


#def power_iteration_step(A, num_iterations, device):
#    A = A.double()
#    b_k = torch.rand((A.shape[1],1)).double().to(device)
    
#    for _ in range(num_iterations):
#        b_k1 = torch.matmul(A, b_k)
#        b_k1_norm = torch.linalg.norm(b_k1)
#        b_k = b_k1 / b_k1_norm

#    eig = torch.matmul(b_k.T,torch.matmul(A, b_k))/torch.matmul(b_k.T, b_k)
#    return eig

#def power_iteration(A, num_iteration, device):
#    e1 = power_iteration_step(A, num_iteration, device)
#    add = 2/3*torch.nn.functional.relu(-e1)*torch.eye(A.shape[1]).to(device)

#    e2 = power_iteration_step(A+add, num_iteration, device)

#    return e2-2/3*torch.nn.functional.relu(-e1)

def power_iteration_step(A, num_iterations, device, warm_start=None, tol=1e-8):
    A = A.double()
    if warm_start is not None:
      b_k = (warm_start+torch.rand((A.shape[1],1))*1e-1).to(device)
    else:
      b_k = torch.rand((A.shape[1],1)).double().to(device)
    
    last_lam = 0.
    for i in range(num_iterations):
        y_k = torch.matmul(A, b_k)
        lam_k = torch.matmul(b_k.T, y_k)
        b_k = y_k / torch.linalg.norm(y_k)
        if torch.abs((lam_k - last_lam)/lam_k) < tol:
          break
        last_lam = lam_k.clone().detach()
    return lam_k, b_k

def power_iteration(A, num_iteration, device, warm_start=None):
    e1, eig1 = power_iteration_step(A, num_iteration, device, warm_start=warm_start)
    print("e1 is", e1)
    if e1 > 0:
      return e1, eig1.clone().detach()
    A_tilde = A-(torch.eye(A.shape[1])*e1).to(device)
    e2, eig2 = power_iteration_step(A_tilde, num_iteration, device, warm_start=warm_start)
    print("e2 is", e2)
    return e2+e1, eig2.clone().detach()

def norm_prod(weights, pair):
    prod = 1.0
    num_weight = len(weights)
    for i in range(num_weight-1):
      prod *= LA.norm(weights[i], ord=2)
    #print("product:", prod)
    if len(pair) == 2:
      prod *= LA.norm(weights[-1][pair[0],:]-weights[-1][pair[1],:], ord=2)
    else:
      prod *= LA.norm(weights[-1][pair[0],:], ord=2)
    return prod

def flatten(x):
  return x.view(x.size()[0], -1)

def tensor_diff(x, y):
  assert x.shape == y.shape
  return torch.sum(torch.abs(x-y))

def extract_network(net, in_size = [28,28], channels=1):
  weights = []
  weight_types = []
  params = [[channels, 0, 0, 0, in_size]]
  for layer in net.modules():
    if type(layer) is nn.Conv2d:
        weight_types.append("conv2d")
        param = extract_cov(layer, in_size)
        W, in_size, _ = conv2mat(layer, in_size)
        params.append(param)
        weights.append(W)
    if type(layer) is nn.Linear:
        weight_types.append("linear")
        weight = layer.weight.cpu().detach().numpy()    
        weights.append(weight)
        params.append([0, 0, 0, 0, [0,0]])
  return weights, weight_types, params

def extract_cov(conv, in_shape):
  #For now I only need mat, stride, padding and input shape
  out_channel, kernel_mat, stride, padding, kernel_size = conv.out_channels, conv.weight.data, conv.stride, conv.padding, conv.kernel_size
  X_h, X_w = in_shape
  k_h, k_w = kernel_size
  #print(padding)
  Y_h, Y_w = int((X_h+padding[0]*2-k_h)/stride[0])+1, int((X_w+padding[1]*2-k_w)/stride[1])+1  
  return [out_channel, kernel_mat, stride, padding, [Y_h, Y_w]]

def test_model(model, data):
  acc = model.evaluate(data.test_loader)
  print(f'accuracy is {acc}%')
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Number of parameters: {total_params}")

def conv2mat(conv, in_shape):
  in_channel, out_channel, kernel_mat, stride, padding, kernel_size = conv.in_channels, conv.out_channels, conv.weight.data, conv.stride, conv.padding, conv.kernel_size
  X_h, X_w = in_shape
  k_h, k_w = kernel_size
  #print(padding)
  Y_h, Y_w = int((X_h+padding[0]*2-k_h)/stride[0])+1, int((X_w+padding[1]*2-k_w)/stride[1])+1
  W = torch.zeros(out_channel*Y_h*Y_w, in_channel*X_h*X_w)
  bias_data = conv.bias.data
  bias = torch.rand(out_channel*Y_h*Y_w)
  for out in range(out_channel):
    for i in range(Y_h):
      for j in range(Y_w):
        bias[out*(Y_h*Y_w)+i*Y_w+j] = bias_data[out]
  #print(W.shape)
  #out_shape = [Y_h, Y_w]
  print(Y_h, Y_w)
  for k in range(Y_h):
    pos_x_start = stride[0]*k-padding[0]
    for l in range(Y_w):
      pos_y_start = stride[1]*l-padding[1]
      for i in range(k_h):
        pos_x = pos_x_start+i
        if pos_x < 0 or pos_x >= X_h:
          continue
        for j in range(k_w):
          pos_y = pos_y_start+j
          if pos_y < 0 or pos_y >= X_w:
            continue
          #print(k*Y_w+l, pos_x*X_w+pos_y, i, j ,kernel_mat)
          for o in range(out_channel):
            for ic in range(in_channel):
              W[o*(Y_h*Y_w)+k*Y_w+l, ic*(X_h*X_w)+pos_x*X_w+pos_y] = kernel_mat[o, ic, i, j]
  return W, [Y_h, Y_w], bias