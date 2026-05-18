"""
Multi-trajectory TDOA experiment — analytic baselines averaged over N_T sequences.

Steps:
  1. Generate N_T trajectories with random turning angles.
  2. Oracle ERTS  (true F per group  →  lower bound).
  3. Nominal ERTS (false F, model mismatch).
  4. Analytic EMKF (EM iterations estimating F from each sequence).
  5. Print averaged results + plot sequence 0.
"""

import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Simulations.Extended_sysmdl import SystemModel
from Simulations.TDOA_2D.parameters import (
    m, n, Q, R,
    m1x_0, m2x_0, M_mics,
    f, h,
    generate_dataset_random_theta,
    generate_false_F_list,
)
from Simulations.TDOA_2D.ekf_erts import (
    run_ekf_erts, compute_cross_covariances,
)
from emkf.main_emkf_func import E_EMKF_F_analitic_non_linear_h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss(reduction='mean')

# ─── Experiment settings ──────────────────────────────────────────────────────
T           = 20
N_T         = 100          # number of test sequences
max_em_iter = 10

theta_true_max  = 0.4      # true  F drawn Uniform(-theta_true_max/2,  +theta_true_max/2)
theta_false_max = 3.0      # false F = true_theta + Uniform(-theta_false_max/2, +theta_false_max/2)

# ── Noise covariances ─────────────────────────────────────────────────────────
Q_structure = torch.eye(m, dtype=torch.float32, device=device)
R_structure = torch.eye(n, dtype=torch.float32, device=device)

q2 = 1
r2 = 10
Q  = q2 * Q_structure
R  = r2 * R_structure

print("=" * 60)
print("2D TDOA tracking — analytic baselines averaged over sequences")
print(f"  T={T}  N_T={N_T}  max_em_iter={max_em_iter}")
print(f"  theta_true_max={theta_true_max} rad   theta_false_max={theta_false_max} rad")
print(f"  Microphones: {M_mics}   State dim: {m}   Obs dim: {n}")
print("=" * 60)

# ─── SystemModel (data container for EMKF) ───────────────────────────────────
sys_model = SystemModel(f=f, Q=Q, h=h, R=R,
                        T=T, T_test=T, m=m, n=n)
sys_model.InitSequence(m1x_0, m2x_0)

# ─── Step 1: Generate N_T trajectories ───────────────────────────────────────
print(f"\n[1/4] Generating {N_T} trajectories ...")
inputs, targets, theta_list, F_true_list = generate_dataset_random_theta(
    N_T, T, theta_true_max, Q_gen=Q, R_gen=R
)
F_false_list = generate_false_F_list(theta_list, theta_false_max)
# inputs:  [N_T, n, T]
# targets: [N_T, m, T]
# F_true_list / F_false_list: one per group of 10

n_groups = N_T // 10
print(f"  {n_groups} groups of 10 sequences each.")

# ─── Steps 2–4: Loop over all sequences ──────────────────────────────────────
print("\n[2-4/4] Running ERTS (true/false) and EMKF on each sequence ...")

mse_erts_true_list  = []
mse_erts_false_list = []
mse_emkf_false_list = []

for i in range(N_T):
    g      = i // 10
    states = targets[i]   # [m, T]
    obs    = inputs[i]    # [n, T]

    F_true  = F_true_list[g]
    F_false = F_false_list[g]

    get_F_true_fn  = lambda _t, _F=F_true:  _F
    get_F_false_fn = lambda _t, _F=F_false: _F

    # ERTS with true F
    x_erts_true, *_ = run_ekf_erts(obs, get_F_true_fn, Q_in=Q, R_in=R)
    mse_erts_true_list.append(loss_fn(x_erts_true, states).item())

    # ERTS with false F
    x_erts_false, *_ = run_ekf_erts(obs, get_F_false_fn, Q_in=Q, R_in=R)
    mse_erts_false_list.append(loss_fn(x_erts_false, states).item())

    # Analytic EMKF starting from false F
    Y_emkf = obs.unsqueeze(0)       # [1, n, T]
    X_emkf = states.unsqueeze(0)    # [1, m, T]

    F_matrices, *_ = E_EMKF_F_analitic_non_linear_h(
        sys_model=sys_model,
        F_0_matrices=[F_false.clone()],
        h=h,
        Q=Q,
        R=R,
        Y=Y_emkf,
        x_0=m1x_0,
        P_0=m2x_0,
        X=X_emkf,
        max_it=max_em_iter,
        generate_f=False,
        init_x_list=None,
        init_P_list=None,
    )

    F_final = F_matrices[0][-1]
    get_F_emkf_fn = lambda _t, _F=F_final: _F
    x_emkf, *_ = run_ekf_erts(obs, get_F_emkf_fn, Q_in=Q, R_in=R)
    mse_emkf_false_list.append(loss_fn(x_emkf, states).item())

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{N_T}] done")

# ─── Average results ─────────────────────────────────────────────────────────
mse_erts_true_avg  = sum(mse_erts_true_list)  / N_T
mse_erts_false_avg = sum(mse_erts_false_list) / N_T
mse_emkf_false_avg = sum(mse_emkf_false_list) / N_T

print("\n" + "=" * 60)
print("RESULTS SUMMARY  (averaged over all sequences)")
print("=" * 60)
print(f"  ERTS TRUE-F        : {10 * math.log10(mse_erts_true_avg):.2f} dB")
print(f"  ERTS FALSE-F       : {10 * math.log10(mse_erts_false_avg):.2f} dB")
print(f"  EMKF FALSE-init    : {10 * math.log10(mse_emkf_false_avg):.2f} dB")
print("=" * 60)

# ─── Plot sequence 0 ─────────────────────────────────────────────────────────
print("\nPlotting sequence 0 ...")

import numpy as np

states0 = targets[0]
obs0    = inputs[0]
g0      = 0
get_F_t0 = lambda _t, _F=F_true_list[g0]:  _F
get_F_f0 = lambda _t, _F=F_false_list[g0]: _F

x_plot_true,  *_ = run_ekf_erts(obs0, get_F_t0, Q_in=Q, R_in=R)
x_plot_false, *_ = run_ekf_erts(obs0, get_F_f0, Q_in=Q, R_in=R)

F_mat0, *_ = E_EMKF_F_analitic_non_linear_h(
    sys_model=sys_model,
    F_0_matrices=[F_false_list[g0].clone()],
    h=h, Q=Q, R=R,
    Y=obs0.unsqueeze(0),
    x_0=m1x_0, P_0=m2x_0,
    X=states0.unsqueeze(0),
    max_it=max_em_iter,
    generate_f=False,
    init_x_list=None,
    init_P_list=None,
)
get_F_emkf0 = lambda _t, _F=F_mat0[0][-1]: _F
x_plot_emkf, *_ = run_ekf_erts(obs0, get_F_emkf0, Q_in=Q, R_in=R)

t_axis = np.arange(T)

plt.figure(figsize=(12, 5))
plt.plot(t_axis, states0[1].cpu().numpy(),        'k-',  linewidth=2.5, label='true p_y')
plt.plot(t_axis, x_plot_true[1].cpu().numpy(),    '--',  linewidth=2,   label='ERTS true F')
plt.plot(t_axis, x_plot_false[1].cpu().numpy(),   ':',   linewidth=2,   label='ERTS false F')
plt.plot(t_axis, x_plot_emkf[1].cpu().numpy(),    '-',   linewidth=2,   label='EMKF false-init')
plt.xlabel('time')
plt.ylabel('y position')
plt.title('Y position vs time (sequence 0)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('y_position_tdoa_single.png', dpi=250)
print("  Saved: y_position_tdoa_single.png")
