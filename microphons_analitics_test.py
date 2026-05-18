"""
Single-trajectory TDOA experiment.

Steps (always in this order):
  1. Generate one trajectory with fixed block-wise turning angles.
  2. PLOT — inspect that target moves slowly and stays near the mic array.
  3. Oracle ERTS  (true piecewise F per block  →  lower bound).
  4. Nominal ERTS (straight-line F, model mismatch).
  5. Analytic EMKF (EM iterations estimating a single global F).
  6. PLOT — tracking results (estimates vs ground truth).
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
    mic_positions, make_F_block, f, F, h,
    generate_single_traj,
)
from Simulations.TDOA_2D.ekf_erts import (
    run_ekf_erts, compute_cross_covariances,
)
from emkf.main_emkf_func import E_EMKF_F_analitic_non_linear_h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss(reduction='mean')

# ─── Experiment settings ──────────────────────────────────────────────────────
T           = 20
max_em_iter = 10

# Turning angles — one entry per block; a single entry = constant F for whole trajectory
# thetas_deg = [0.0, 20.0, 0.0, -20.0]          # degrees, easy to read/edit
# thetas_rad = [t * math.pi / 180.0 for t in thetas_deg]
thetas_rad  =[0.2]
n_blocks   = len(thetas_rad)
block_size = T // n_blocks

# ── Noise covariances ─────────────────────────────────────────────────────────
Q_structure = torch.eye(m, dtype=torch.float32, device=device)   # [m, m] identity
R_structure = torch.eye(n, dtype=torch.float32, device=device)   # [n, n] identity

q2 = 1   # default — main scripts override with their own q2/r2
r2 = 10
Q  = q2 * Q_structure
R  = r2 * R_structure



print("=" * 60)
print("2D TDOA tracking — single trajectory")
print(f"  T={T}  blocks={n_blocks}  block_size={block_size}")
print(f"  Turning angles: {thetas_rad} deg")
print(f"  Microphones:    {M_mics} (on x-axis)")
print(f"  State dim: {m}   Obs dim: {n}")
print("=" * 60)


# ─── Step 1: Generate trajectory ─────────────────────────────────────────────
print("\n[1/6] Generating trajectory ...")
states, obs = generate_single_traj(T, thetas_rad, Q_gen=Q, R_gen=R)
# states: [4, T],   obs: [4, T]

print(f"  p_x  range: [{states[0].min():.3f},  {states[0].max():.3f}]")
print(f"  p_y  range: [{states[1].min():.3f},  {states[1].max():.3f}]")
print(f"  v_x  range: [{states[2].min():.4f}, {states[2].max():.4f}]")
print(f"  v_y  range: [{states[3].min():.4f}, {states[3].max():.4f}]")


# ─── SystemModel (needed only as a data container for EMKF_F_Mstep) ──────────
sys_model = SystemModel(f=f, Q=Q, h=h, R=R,
                        T=T, T_test=T, m=m, n=n)
sys_model.InitSequence(m1x_0, m2x_0)


# ─── Step 2: Build true-F and false-F models ─────────────────────────────────
print("\n[2/5] Building true-F and false-F settings ...")

theta_false_rad = 1   # choose the wrong theta you want
F_false = make_F_block(theta_false_rad)

def get_F_true(t: int) -> torch.Tensor:
    k = min(t // block_size, n_blocks - 1)
    return make_F_block(thetas_rad[k])

def get_F_false(_t: int) -> torch.Tensor:
    return F_false

# ─── Step 3: ERTS with true F ────────────────────────────────────────────────
print("\n[3/5] ERTS with TRUE F ...")

x_erts_true, P_s_true, P_f_true, sg_true, H_l_true, K_l_true = \
    run_ekf_erts(obs, get_F_true, Q_in=Q, R_in=R)

mse_erts_true = loss_fn(x_erts_true, states).item()
print(f"  ERTS TRUE-F MSE:  {10 * math.log10(mse_erts_true):.2f} dB")


# ─── Step 4: ERTS with false F ───────────────────────────────────────────────
print("\n[4/5] ERTS with FALSE F ...")

x_erts_false, P_s_false, P_f_false, sg_false, H_l_false, K_l_false = \
    run_ekf_erts(obs, get_F_false, Q_in=Q, R_in=R)

mse_erts_false = loss_fn(x_erts_false, states).item()
print(f"  ERTS FALSE-F MSE: {10 * math.log10(mse_erts_false):.2f} dB")

# ─── Step 5: Analytic EMKF ───────────────────────────────────────────────────
# ─── Step 5: EMKF with true-F init and false-F init ──────────────────────────
print("\n[5/5] EMKF ...")

# initial F guesses
F_init_false = [F_false.clone()]

# single-sequence tensors expected by your EMKF wrapper
Y_emkf = obs.unsqueeze(0)        # [1, n, T]
X_emkf = states.unsqueeze(0)     # [1, m, T]

F_matrices_false, mse_emkf_false_lin, last_x_false, last_P_false = \
    E_EMKF_F_analitic_non_linear_h(
        sys_model=sys_model,
        F_0_matrices=F_init_false,
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

F_final_false = F_matrices_false[0][-1]

def get_F_emkf_false(_t: int) -> torch.Tensor:
    return F_final_false

x_emkf_false, P_s_emkf_false, P_f_emkf_false, sg_emkf_false, H_l_emkf_false, K_l_emkf_false = \
    run_ekf_erts(obs, get_F_emkf_false, Q_in=Q, R_in=R)

mse_emkf_false = loss_fn(x_emkf_false, states).item()


print(f"  EMKF FALSE-init MSE: {10 * math.log10(mse_emkf_false):.2f} dB")
print(f"  Final F from FALSE init:\n{F_final_false}")


# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  ERTS TRUE-F        : {10 * math.log10(mse_erts_true):.2f} dB")
print(f"  ERTS FALSE-F       : {10 * math.log10(mse_erts_false):.2f} dB")
print(f"  EMKF FALSE-init    : {10 * math.log10(mse_emkf_false):.2f} dB")
print("=" * 60)

print("\nPlotting final results ...")

import numpy as np

px_true = states[0].cpu().numpy()
py_true = states[1].cpu().numpy()

px_erts_true = x_erts_true[0].cpu().numpy()
py_erts_true = x_erts_true[1].cpu().numpy()

px_erts_false = x_erts_false[0].cpu().numpy()
py_erts_false = x_erts_false[1].cpu().numpy()

px_emkf_false = x_emkf_false[0].cpu().numpy()
py_emkf_false = x_emkf_false[1].cpu().numpy()

mic_np = mic_positions.cpu().numpy()
t_axis = np.arange(T)


# Figure 3: y position vs time
plt.figure(figsize=(12, 5))

plt.plot(t_axis, py_true, 'k-', linewidth=2.5, label='true p_y')
plt.plot(t_axis, py_erts_true, '--', linewidth=2, label='ERTS true p_y')
plt.plot(t_axis, py_erts_false, ':', linewidth=2, label='ERTS false p_y')
plt.plot(t_axis, py_emkf_false, '-', linewidth=2, label='EMKF false p_y')

for b in range(1, n_blocks):
    plt.axvline(b * block_size, color='gray', linestyle=':', alpha=0.6)

plt.xlabel('time')
plt.ylabel('y position')
plt.title('Y position vs time')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('y_position_tdoa_single.png', dpi=250)