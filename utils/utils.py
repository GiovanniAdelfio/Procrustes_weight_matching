import copy
import torch

def slerp_state_dict(lam, t1, t2):
    '''
    creates a new state_dict with the interpolated weights and biases.
    '''
    new_sd = {}
    for k in t1.keys():
        new_sd[k] = slerp(lam, t1[k], t2[k])
    return new_sd

def slerp(lam: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor
    '''
    Applies SLERP interpolation between two given tensors
    '''
    
    # flatten
    low_flat = low.view(-1).to(torch.float64)
    high_flat = high.view(-1).to(torch.float64)

    # normalization
    low_norm = low_flat / low_flat.norm()
    high_norm = high_flat / high_flat.norm()

    # scalar product -> cos of the angle between the vectors -> applies arcos on the results -> applies sin on the results
    dot = torch.clamp(torch.dot(low_norm, high_norm), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    if so < 1e-8:
        # avoids singularities
        out = (1.0 - lam) * low_flat + lam * high_flat
    else:
        out = (torch.sin((1.0 - lam) * omega) / so) * low_norm + (torch.sin(lam * omega) / so) * high_norm

    # rimetti la scala come interpolazione delle norme originali
    out = out * ((1.0 - lam) * low_flat.norm() + lam * high_flat.norm())

    return out.to(low.dtype).view_as(low)

def lerp(lam, t1, t2):
  '''
  applies Linear Interpolation 
  '''
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3

def flatten_params(model):
  return model.state_dict()
