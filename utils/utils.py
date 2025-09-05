import copy
from xml.parsers.expat import model
import torch

def slerp_state_dict(lam, t1, t2):
    new_sd = {}
    for k in t1.keys():
        # appiattisci i tensori in 1D
        t1_flat = t1[k]
        t2_flat = t2[k]
        new = slerp(lam, t1_flat, t2_flat)
        new_sd[k] = new.view_as(t1[k])
    return new_sd

def slerp(lam: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    # flatten
    low_flat = low.view(-1).to(torch.float64)
    high_flat = high.view(-1).to(torch.float64)

    low_norm = low_flat / low_flat.norm()
    high_norm = high_flat / high_flat.norm()

    dot = torch.clamp(torch.dot(low_norm, high_norm), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    if so == 0:
        out = (1.0 - lam) * low_flat + lam * high_flat
    else:
        out = (torch.sin((1.0 - lam) * omega) / so) * low_norm + \
              (torch.sin(lam * omega) / so) * high_norm

    # rimetti la scala come interpolazione delle norme originali
    out = out * ((1.0 - lam) * low_flat.norm() + lam * high_flat.norm())

    return out.to(low.dtype).view_as(low)


def flatten_params(model):
  return model.state_dict()

def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3


