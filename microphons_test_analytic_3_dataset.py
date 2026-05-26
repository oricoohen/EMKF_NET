import os
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime

import Simulations.config as config

import matplotlib.patches as mpatches

from Simulations.TDOA_2D.parameters import (
    m, n, m1x_0, m2x_0, M_mics,
    Q_structure, R_structure,
    make_F_block, f, h, h_jacobian,
    generate_dataset_random_theta,
    generate_false_F_list,
    make_get_F_from_matrix,
    PX_MIN, PX_MAX, PY_MIN, PY_MAX,
    mic_positions,
)
from Simulations.TDOA_2D.ekf_erts import run_ekf_erts, compute_cross_covariances
from Simulations.Extended_sysmdl import SystemModel
from emkf.main_emkf_func import E_EMKF_F_analitic_non_linear_h, EMKF_F_solo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

loss_fn = nn.MSELoss(reduction="mean")

today = datetime.today()
now   = datetime.now()
strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
args.N_T   = 100
args.T     = 30
args.T_test = 30

T_test = args.T_test

q2 = 0.001
r2 = 1

cycle              = 5
theta_per_dataset  = [0.5, 2, 0.1, 0.8, 3]   # exact theta (rad) for each dataset
assert len(theta_per_dataset) == cycle

max_em_iter = 5

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

sys_model = SystemModel(f=f, Q=Q, h=h, R=R, T=T_test, T_test=T_test, m=m, n=n)
sys_model.InitSequence(m1x_0, m2x_0)

save_dir  = "RTSNet/tdoa_2d/10/"
cycle_dir = save_dir + f"{cycle}cycle/"
os.makedirs(cycle_dir, exist_ok=True)

print("=" * 70)
print(f"2D TDOA Analytic ERTS — {cycle}-cycle multi-dataset test")
print(f"  T_test={T_test}  q2={q2}  r2={r2}")
print(f"  cycle={cycle}  theta_per_dataset={theta_per_dataset}  false F fixed at theta=0")
print(f"  Microphones: {M_mics}   State dim: {m}   Obs dim: {n}")
print("=" * 70)

#########################################
###  Generate test data               ###
#########################################
print(f"\nGenerating {cycle} test datasets ...")

all_test_inputs  = []
all_test_targets = []
all_F_test_true  = []
all_F_test_false = []

carry_x_test = None

for k in range(cycle):
    theta_k = theta_per_dataset[k]
    print(f"  Dataset {k}: theta={theta_k:.4f} rad (fixed for all sequences)")

    xi, xt, _, F_te_t = generate_dataset_random_theta(
        args.N_T, T_test, 0, Q, R,
        x_init=carry_x_test, theta_base=[theta_k] * args.N_T,
    )
    F_te_f = [make_F_block(0.0)] * args.N_T   # false F always theta=0

    carry_x_test = xt[:, :, -1]   # [N_T, m] — per-sequence carry

    all_test_inputs.append(xi)
    all_test_targets.append(xt)
    all_F_test_true.append(F_te_t)
    all_F_test_false.append(F_te_f)

print(f"  Test per dataset: {all_test_targets[0].size()}")
print(f"  max_em_iter={max_em_iter}")

#########################################
###  Trajectory popup — sequences 0-2 ###
#########################################
_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

print("\nPlotting sequences 0-2 across all datasets ...")
for seq_idx in range(10):
    fig = plt.figure(figsize=(16, 12))
    ax_traj = fig.add_subplot(2, 2, 1)
    ax_pos  = fig.add_subplot(2, 2, 2)
    ax_vel  = fig.add_subplot(2, 2, 3)
    ax_tdoa = fig.add_subplot(2, 2, 4)

    t_offset = 0
    for k in range(cycle):
        states_k = all_test_targets[k][seq_idx].cpu()   # [m, T_test]
        obs_k    = all_test_inputs[k][seq_idx].cpu()    # [n, T_test]
        t_ax     = torch.arange(t_offset, t_offset + T_test).float()
        col      = _colors[k % len(_colors)]

        ax_traj.plot(states_k[0], states_k[1], color=col, label=f'ds{k} θ={theta_per_dataset[k]:.2f}')
        ax_traj.scatter(states_k[0, 0],  states_k[1, 0],  color=col, marker='o', s=40, zorder=5)
        ax_traj.scatter(states_k[0, -1], states_k[1, -1], color=col, marker='x', s=60, zorder=5)

        ax_pos.plot(t_ax, states_k[0], color=col, label=f'ds{k} px')
        ax_pos.plot(t_ax, states_k[1], color=col, linestyle='--')

        ax_vel.plot(t_ax, states_k[2], color=col, label=f'ds{k} vx')
        ax_vel.plot(t_ax, states_k[3], color=col, linestyle='--')

        for i in range(obs_k.size(0)):
            ax_tdoa.plot(t_ax, obs_k[i], color=col, alpha=0.6,
                         label=f'ds{k} ch{i}' if k == 0 else '_')

        if k > 0:
            for ax in [ax_pos, ax_vel, ax_tdoa]:
                ax.axvline(x=t_offset, color='k', linestyle='--', alpha=0.35)
        t_offset += T_test

    for idx, mic in enumerate(mic_positions):
        ax_traj.scatter(mic[0].item(), mic[1].item(), marker='^', color='black', s=80, zorder=6)
        ax_traj.annotate(f'm{idx}', (mic[0].item(), mic[1].item()),
                         textcoords='offset points', xytext=(4, 4), fontsize=7)

    _rect = mpatches.Rectangle(
        (PX_MIN, PY_MIN), PX_MAX - PX_MIN, PY_MAX - PY_MIN,
        linewidth=1.5, edgecolor='red', facecolor='lightyellow',
        linestyle='--', zorder=2, alpha=0.3, label='valid region',
    )
    ax_traj.add_patch(_rect)

    ax_traj.set_xlabel('p_x');    ax_traj.set_ylabel('p_y')
    ax_traj.set_title('2D trajectory (o=start, x=end per dataset)')
    ax_traj.legend(fontsize=7);   ax_traj.grid(True, alpha=0.4)

    ax_pos.set_ylabel('position');  ax_pos.set_title('p_x (solid) & p_y (dashed) vs time')
    ax_pos.legend(fontsize=7, ncol=cycle); ax_pos.grid(True, alpha=0.4)

    ax_vel.set_ylabel('velocity');  ax_vel.set_title('v_x (solid) & v_y (dashed) vs time')
    ax_vel.legend(fontsize=7, ncol=cycle); ax_vel.grid(True, alpha=0.4)
    ax_vel.set_xlabel('time step')

    ax_tdoa.set_ylabel('TDOA');   ax_tdoa.set_title('TDOA observations vs time')
    ax_tdoa.legend(fontsize=7);   ax_tdoa.grid(True, alpha=0.4)
    ax_tdoa.set_xlabel('time step')

    fig.suptitle(f'Sequence {seq_idx} — all datasets (test)  |  dashed = dataset boundary', fontsize=12)
    plt.tight_layout()
    _plot_path = os.path.abspath(cycle_dir + f'seq{seq_idx}_data_sanity.png')
    plt.savefig(_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {_plot_path}")
    if os.name == 'nt':
        os.startfile(_plot_path)

#########################################
###  Run ERTS across all datasets     ###
#########################################
# For each test sequence j, datasets are processed in order 0→cycle-1.
# x_0 and P_0 carry from the last EKF filtered state/covariance of dataset k
# into the initial condition of dataset k+1.

print("\nRunning ERTS (true F and false F) across all datasets ...")

N_T = args.N_T

# MSE arrays: [datasets, N_T]
mse_true_arr  = torch.zeros(cycle, N_T)
mse_false_arr = torch.zeros(cycle, N_T)
mse_emkf_arr  = torch.zeros(cycle, N_T)
mse_bigru_arr = torch.zeros(cycle, N_T)

# Store first-sequence outputs for plotting
out_true_seq0  = []   # smoother output per dataset for sequence 0
out_false_seq0 = []
out_emkf_seq0  = []
out_bigru_seq0 = []

for j in range(N_T):
    x0_true  = m1x_0.clone()
    P0_true  = m2x_0.clone()
    x0_false = m1x_0.clone()
    P0_false = m2x_0.clone()

    for data in range(cycle):
        y_seq  = all_test_inputs[data][j]
        x_true = all_test_targets[data][j]

        get_F_true  = make_get_F_from_matrix(all_F_test_true[data][j])
        get_F_false = make_get_F_from_matrix(all_F_test_false[data][j])

        # ERTS with true F
        x_s_true, P_s_true, P_f_true, *_ = run_ekf_erts(
            y_seq, get_F_true, Q_in=Q, R_in=R,
            x_init=x0_true, P_init=P0_true,
        )
        mse_true_arr[data, j] = loss_fn(x_s_true, x_true).item()
        x0_true = x_s_true[:, -1].detach()
        P0_true = P_f_true[:, :, -1].detach()

        # ERTS with false F
        x_s_false, P_s_false, P_f_false, *_ = run_ekf_erts(
            y_seq, get_F_false, Q_in=Q, R_in=R,
            x_init=x0_false, P_init=P0_false,
        )
        mse_false_arr[data, j] = loss_fn(x_s_false, x_true).item()
        x0_false = x_s_false[:, -1].detach()
        P0_false = P_f_false[:, :, -1].detach()

        if j == 0:
            out_true_seq0.append(x_s_true)
            out_false_seq0.append(x_s_false)

## ── Cross-check (confirmed equivalent, F diff ~1e-6) — commented out ──────────
## def run_emkf_ekf_erts(y_seq, F_init, x_0, P_0, max_it):
##     F_est = F_init.clone()
##     F_all = [F_est.clone()]
##     for _ in range(max_it):
##         x_s, P_s, P_f, sgains, H_last, K_last = run_ekf_erts(
##             y_seq, make_get_F_from_matrix(F_est), Q_in=Q, R_in=R,
##             x_init=x_0, P_init=P_0,
##         )
##         V     = compute_cross_covariances(F_est, H_last, K_last, P_f, sgains)
##         F_est = EMKF_F_solo(F_est, h, Q, R, y_seq, x_0, P_0, x_s, P_s, V, m, T_test)
##         F_all.append(F_est.clone())
##     x_s_final, *_ = run_ekf_erts(
##         y_seq, make_get_F_from_matrix(F_est), Q_in=Q, R_in=R,
##         x_init=x_0, P_init=P_0,
##     )
##     return F_all, x_s_final
##
## print("\n=== EMKF cross-check: S_Test_ext_old vs run_ekf_erts (seq 0, dataset 0) ===")
## _y0  = all_test_inputs[0][0]
## _xt0 = all_test_targets[0][0]
## _F0  = all_F_test_false[0][0].clone()
## _Fmats_A, _, _, _ = E_EMKF_F_analitic_non_linear_h(
##     sys_model=sys_model, F_0_matrices=[_F0.clone()],
##     h=h, Q=Q, R=R,
##     Y=_y0.unsqueeze(0), x_0=m1x_0, P_0=m2x_0,
##     X=_xt0.unsqueeze(0), max_it=max_em_iter, generate_f=False,
##     init_x_list=None, init_P_list=None,
## )
## _F_final_A = _Fmats_A[0][-1]
## _xs_A, *_ = run_ekf_erts(_y0, make_get_F_from_matrix(_F_final_A), Q_in=Q, R_in=R)
## _Fall_B, _xs_B = run_emkf_ekf_erts(_y0, _F0.clone(), m1x_0, m2x_0, max_em_iter)
## _F_final_B = _Fall_B[-1]
## print(f"  F_final diff  (A - B) norm : {(_F_final_A - _F_final_B).norm().item():.2e}")
## print(f"  x_smooth diff (A - B) norm : {(_xs_A - _xs_B).norm().item():.2e}")
## print(f"  MSE method A : {10*math.log10(loss_fn(_xs_A, _xt0).item()):.2f} dB")
## print(f"  MSE method B : {10*math.log10(loss_fn(_xs_B, _xt0).item()):.2f} dB")
## print("=" * 70)
## ─────────────────────────────────────────────────────────────────────────────


#########################################
###  Run analytic EMKF across datasets ###
#########################################
print("\nRunning analytic EMKF across all datasets ...")

emkf_x_carries = [m1x_0.clone() for _ in range(N_T)]
emkf_P_carries = [m2x_0.clone() for _ in range(N_T)]
emkf_F_carries = [make_F_block(0.0) for _ in range(N_T)]  # init with theta=0

for data in range(cycle):
    print(f"\n  [EMKF] Dataset {data} ...")

    F_mats_batch, _, last_x_emkf, last_P_emkf = E_EMKF_F_analitic_non_linear_h(
        sys_model=sys_model,
        F_0_matrices=emkf_F_carries,
        h=h, Q=Q, R=R,
        Y=all_test_inputs[data],
        x_0=m1x_0, P_0=m2x_0,
        X=all_test_targets[data],
        max_it=max_em_iter,
        generate_f=False,
        init_x_list=emkf_x_carries,
        init_P_list=emkf_P_carries,
        vel_only=True,
    )

    for j in range(N_T):
        F_final_j    = F_mats_batch[j][-1]
        get_F_emkf_j = make_get_F_from_matrix(F_final_j)
        y_seq  = all_test_inputs[data][j]
        x_true = all_test_targets[data][j]

        x_s_emkf, _, P_f_emkf, *_ = run_ekf_erts(
            y_seq, get_F_emkf_j, Q_in=Q, R_in=R,
            x_init=emkf_x_carries[j], P_init=emkf_P_carries[j],
        )
        mse_emkf_arr[data, j] = loss_fn(x_s_emkf, x_true).item()
        if j == 0:
            out_emkf_seq0.append(x_s_emkf)

    for j in range(N_T):
        emkf_x_carries[j] = last_x_emkf[j]
        emkf_P_carries[j] = last_P_emkf[j]
        emkf_F_carries[j] = F_mats_batch[j][-1]  # carry estimated F to next dataset

#########################################
###  Run BiGRU baseline               ###
#########################################
# Load the BiGRU trained by microphons_training_3_dataset.py — no retraining here.
bigru_train_dir = f"RTSNet/tdoa_2d/1/{cycle}cycle/"
load_path_bigru = bigru_train_dir + "BiGRU.pt"
print(f"\nLoading BiGRU from {load_path_bigru} ...")
bigru_model = torch.load(load_path_bigru, weights_only=False, map_location=device)
bigru_model.eval()

print("Running BiGRU inference across all datasets ...")

with torch.no_grad():
    for data in range(cycle):
        y_batch = all_test_inputs[data]    # [N_T, n, T]
        x_batch = all_test_targets[data]   # [N_T, m, T]
        x_hat   = bigru_model(y_batch)     # [N_T, m, T]
        for j in range(N_T):
            mse_bigru_arr[data, j] = loss_fn(x_hat[j], x_batch[j]).item()
        out_bigru_seq0.append(x_hat[0].cpu())

#########################################
###  Results summary                  ###
#########################################
print("\n" + "=" * 70)
print(f"RESULTS SUMMARY  (cycle={cycle}, theta_per_dataset={theta_per_dataset})")
print("=" * 70)

mse_true_db_per_dataset  = [10 * math.log10(mse_true_arr[k].mean().item())  for k in range(cycle)]
mse_false_db_per_dataset = [10 * math.log10(mse_false_arr[k].mean().item()) for k in range(cycle)]
mse_emkf_db_per_dataset  = [10 * math.log10(mse_emkf_arr[k].mean().item())  for k in range(cycle)]
mse_bigru_db_per_dataset = [10 * math.log10(mse_bigru_arr[k].mean().item()) for k in range(cycle)]

for k in range(cycle):
    print(f"  Dataset {k}  ERTS true-F : {mse_true_db_per_dataset[k]:.2f} dB"
          f"   ERTS false-F: {mse_false_db_per_dataset[k]:.2f} dB"
          f"   EMKF: {mse_emkf_db_per_dataset[k]:.2f} dB"
          f"   BiGRU: {mse_bigru_db_per_dataset[k]:.2f} dB")

mse_true_avg_db  = 10 * math.log10(mse_true_arr.mean().item())
mse_false_avg_db = 10 * math.log10(mse_false_arr.mean().item())
mse_emkf_avg_db  = 10 * math.log10(mse_emkf_arr.mean().item())
mse_bigru_avg_db = 10 * math.log10(mse_bigru_arr.mean().item())
print(f"\n  ERTS TRUE-F  (overall avg) : {mse_true_avg_db:.2f} dB")
print(f"  ERTS FALSE-F (overall avg) : {mse_false_avg_db:.2f} dB")
print(f"  EMKF         (overall avg) : {mse_emkf_avg_db:.2f} dB")
print(f"  BiGRU        (overall avg) : {mse_bigru_avg_db:.2f} dB")
print("=" * 70)

#########################################
###  Plot — sequence 0, all datasets  ###
#########################################
print("\nPlotting sequence 0 across all datasets ...")
t_axis = torch.arange(T_test)

fig, axes = plt.subplots(cycle, 1, figsize=(12, 3 * cycle), sharex=True)
for k in range(cycle):
    ax     = axes[k]
    states = all_test_targets[k][0]
    ax.plot(t_axis, states.cpu()[1],                  linewidth=2.5, label="true p_y")
    ax.plot(t_axis, out_true_seq0[k].cpu()[1],  "--",  linewidth=2,   label="ERTS true F")
    ax.plot(t_axis, out_false_seq0[k].cpu()[1], ":",   linewidth=2,   label="ERTS false F")
    ax.plot(t_axis, out_emkf_seq0[k].cpu()[1],  "-.",  linewidth=2,   label="EMKF")
    ax.plot(t_axis, out_bigru_seq0[k][1],        "-.",  linewidth=2,   label="BiGRU")
    ax.set_ylabel(f"Dataset {k}\ny position")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Dataset {k} — true: {mse_true_db_per_dataset[k]:.2f} dB  "
                 f"false: {mse_false_db_per_dataset[k]:.2f} dB  "
                 f"EMKF: {mse_emkf_db_per_dataset[k]:.2f} dB  "
                 f"BiGRU: {mse_bigru_db_per_dataset[k]:.2f} dB")

axes[-1].set_xlabel("time")
fig.suptitle(f"TDOA ERTS analytic — {cycle}-dataset sequential scenario", fontsize=13)
plt.tight_layout()

plot_path = cycle_dir + "analytic_erts_y_position.png"
plt.savefig(plot_path, dpi=250)
print(f"  Saved: {plot_path}")
