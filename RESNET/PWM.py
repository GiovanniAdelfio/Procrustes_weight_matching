from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
from jax import random
from scipy.optimize import linear_sum_assignment

from utils import rngmix
from weight_matching import PermutationSpec


# === Procrustes weight matching (rotazioni ortogonali) ========================

def _apply_Q_on_axis(w, Q, axis: int):
  """
  Applica la matrice ortogonale Q lungo l'asse `axis` di w.
  w: tensore con dimensione n lungo `axis`
  Q: (n, n), ortogonale
  """
  w = jnp.moveaxis(w, axis, 0)             # (n, ...)
  shape0 = w.shape
  w2 = (Q @ w.reshape(shape0[0], -1)).reshape(shape0)
  return jnp.moveaxis(w2, 0, axis)

def get_rotated_param(ps: PermutationSpec, Qs, k: str, params, except_axis=None):
  """
  Versione 'continua' di get_permuted_param:
  applica le rotazioni Qs[p] su tutti gli assi rilevanti di params[k],
  tranne `except_axis` (utile durante l'aggiornamento di quel blocco).
  """
  w = params[k]
  for axis, p in enumerate(ps.axes_to_perm[k]):
    if axis == except_axis:
      continue
    if p is not None:
      Q = Qs[p]
      w = _apply_Q_on_axis(w, Q, axis)
  return w

def apply_rotation(ps: PermutationSpec, Qs, params):
  """Applica tutte le rotazioni Qs ai parametri `params`."""
  return {k: get_rotated_param(ps, Qs, k, params) for k in params.keys()}

def procrustes_weight_matching(rng,
                               ps: PermutationSpec,
                               params_a,
                               params_b,
                               max_iter=10,
                               init_Q=None,
                               silent=False):
  """
  Trova, per ogni gruppo p del PermutationSpec, una matrice ortogonale Q[p]
  che allinei i neuroni di `params_b` a quelli di `params_a` risolvendo
  problemi di Orthogonal Procrustes in coordinate descent sui blocchi.
  """
  # dimensioni dei blocchi
  perm_sizes = {
      p: params_a[axes[0][0]].shape[axes[0][1]]
      for p, axes in ps.perm_to_axes.items()
  }

  # inizializzazione
  Qs = {p: jnp.eye(n) for p, n in perm_sizes.items()} if init_Q is None else init_Q
  perm_names = list(Qs.keys())

  for iteration in range(max_iter):
    progress = False

    # visita casuale dei blocchi come nel metodo a permutazioni
    for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]

      # Accumula M = sum_{(wk, axis) in p} B A^T  (A=w_a, B=w_b)
      M = jnp.zeros((n, n))
      # useremo anche una stima di loss per loggare
      cur_loss = 0.0
      count = 0

      for wk, axis in ps.perm_to_axes[p]:
        # Applica tutte le Q correnti a eccezione di quella sul blocco/asse che stiamo stimando
        w_a = params_a[wk]
        w_b = get_rotated_param(ps, Qs, wk, params_b, except_axis=axis)

        # porta l'asse target in testa e flattalo
        w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))

        # accumula matrice di Procrustes: M = B A^T
        M += w_b @ w_a.T

        # loss corrente (con Q attuale)
        Qa = Qs[p] @ w_a
        cur_loss += jnp.linalg.norm(Qa - w_b) ** 2
        count += 1

      # SVD e aggiornamento: argmin ||Q A - B||_F -> Q = U V^T con SVD(B A^T)
      U, S, Vt = jnp.linalg.svd(M, full_matrices=False)
      Q_new = U @ Vt

      # stima progresso (facoltativo)
      if not silent and count > 0:
        # ricalcola loss con Q nuovo su ultimo w_a/w_b elaborati? Meglio una stima media
        # Per coerenza facciamo un controllo su M (traccia aumenta).
        old_trace = jnp.trace(Qs[p] @ M)
        new_trace = jnp.trace(Q_new @ M)
        print(f"{iteration}/{p}: Δtrace={float(new_trace - old_trace):.6f}")

      progress = progress or (jnp.linalg.norm(Q_new - Qs[p]) > 1e-7)
      Qs[p] = Q_new

    if not progress:
      break

  return Qs

def test_procrustes_weight_matching():
  """Test minimo su MLP a 1 layer nascosto, confronto con permutazioni."""
  ps = mlp_permutation_spec(num_hidden_layers=1)
  rng = random.PRNGKey(123)
  num_hidden = 10
  shapes = {
      "Dense_0/kernel": (2, num_hidden),
      "Dense_0/bias": (num_hidden, ),
      "Dense_1/kernel": (num_hidden, 3),
      "Dense_1/bias": (3, )
  }
  params_a = {k: random.normal(rngmix(rng, f"a-{k}"), shape) for k, shape in shapes.items()}
  params_b = {k: random.normal(rngmix(rng, f"b-{k}"), shape) for k, shape in shapes.items()}

  # trova Q e applicalo a params_b
  Qs = procrustes_weight_matching(rng, ps, params_a, params_b, max_iter=10, silent=False)
  params_b_aligned = apply_rotation(ps, Qs, params_b)

  # misura somiglianza (Frobenius) post-allineamento
  frob = sum(jnp.linalg.norm(params_a[k] - params_b_aligned[k])**2 for k in params_a.keys())
  print("Frobenius distance after Procrustes alignment:", float(frob))


