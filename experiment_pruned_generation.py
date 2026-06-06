"""
Standalone experiment: pruned dataset generation with carry.

For each sequence in each dataset:
  - Try once with a random theta.
  - If invalid, try once more (fresh random theta).
  - If still invalid, discard the sequence entirely.

Carry: dataset k+1 starts each sequence from the final state of that same
sequence in dataset k (matching the training script). If a sequence was
discarded in dataset k, it restarts from m1x_0 in dataset k+1.

Prints how many sequences survive per dataset and overall.
"""

import torch
import Simulations.config as config
from Simulations.TDOA_2D.parameters import (
    m, n, m1x_0,
    Q_structure, R_structure,
    generate_dataset_random_theta_pruned,
    PX_MIN, PX_MAX, PY_MIN, PY_MAX, V_MAX,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Settings (same as training script) ────────────────────────────────────────
args = config.general_settings()
args.N_E    = 1000
args.N_CV   = 100
args.N_T    = 200
args.T      = 50
args.T_test = 50

T      = args.T
T_test = args.T_test
q2     = 0.001
r2     = 10
cycle  = 5
theta_max = 0.12

Q = (q2 * Q_structure).to(device)
R = (r2 * R_structure).to(device)

x0_default = m1x_0.reshape(-1).to(device)   # [m]

print("=" * 65)
print("Pruned generation experiment  (max_tries=2, WITH carry)")
print(f"  N_E={args.N_E}  N_CV={args.N_CV}  N_T={args.N_T}")
print(f"  T={T}  T_test={T_test}  q2={q2}  r2={r2}")
print(f"  cycle={cycle}  theta_max=±{theta_max}")
print(f"  Bounds: px=[{PX_MIN},{PX_MAX}]  py=[{PY_MIN},{PY_MAX}]  |v|<={V_MAX}")
print("=" * 65)


def _build_carry(targets, kept_mask, N, default_x0):
    """
    Build a [N, m] carry tensor for the next dataset.
    Kept sequences carry their final state; discarded ones restart from default_x0.
    """
    carry = default_x0.unsqueeze(0).expand(N, -1).clone()   # [N, m]
    j = 0
    for i, kept in enumerate(kept_mask):
        if kept:
            carry[i] = targets[j, :, -1]
            j += 1
    return carry


n_kept_train = []
n_kept_cv    = []
n_kept_test  = []

carry_train = None   # None on first dataset → generate_dataset_random_theta_pruned uses m1x_0
carry_cv    = None
carry_test  = None

for k in range(cycle):
    print(f"\nDataset {k}:")

    print(f"  train  ", end="", flush=True)
    _, tt, _, _, n_tr, mask_tr = generate_dataset_random_theta_pruned(
        args.N_E,  T,      2*theta_max, Q, R, x_init=carry_train, max_tries=2)

    print(f"  cv     ", end="", flush=True)
    _, ct, _, _, n_cv, mask_cv = generate_dataset_random_theta_pruned(
        args.N_CV, T,      2*theta_max, Q, R, x_init=carry_cv, max_tries=2)

    print(f"  test   ", end="", flush=True)
    _, xt, _, _, n_te, mask_te = generate_dataset_random_theta_pruned(
        args.N_T,  T_test, 2*theta_max, Q, R, x_init=carry_test, max_tries=2)

    n_kept_train.append(n_tr)
    n_kept_cv.append(n_cv)
    n_kept_test.append(n_te)

    # Build carry for next dataset — discard→restart, keep→final state
    carry_train = _build_carry(tt, mask_tr, args.N_E,  x0_default)
    carry_cv    = _build_carry(ct, mask_cv, args.N_CV, x0_default)
    carry_test  = _build_carry(xt, mask_te, args.N_T,  x0_default)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  {'ds':>4}  {'Train':>14}  {'CV':>12}  {'Test':>12}")
for k in range(cycle):
    print(f"  {k:>4}  "
          f"{n_kept_train[k]:>5}/{args.N_E:<5} ({100*n_kept_train[k]/args.N_E:5.1f}%)  "
          f"{n_kept_cv[k]:>3}/{args.N_CV:<3} ({100*n_kept_cv[k]/args.N_CV:5.1f}%)  "
          f"{n_kept_test[k]:>3}/{args.N_T:<3} ({100*n_kept_test[k]/args.N_T:5.1f}%)")

total_tr = sum(n_kept_train)
total_cv = sum(n_kept_cv)
total_te = sum(n_kept_test)
print(f"  {'ALL':>4}  "
      f"{total_tr:>5}/{cycle*args.N_E:<5} ({100*total_tr/(cycle*args.N_E):5.1f}%)  "
      f"{total_cv:>3}/{cycle*args.N_CV:<3} ({100*total_cv/(cycle*args.N_CV):5.1f}%)  "
      f"{total_te:>3}/{cycle*args.N_T:<3} ({100*total_te/(cycle*args.N_T):5.1f}%)")
print("=" * 65)
