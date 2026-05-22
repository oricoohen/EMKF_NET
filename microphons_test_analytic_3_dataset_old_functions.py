import os
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime

import Simulations.config as config

from Simulations.TDOA_2D.parameters import (
    m, n, m1x_0, m2x_0, M_mics,
    Q_structure, R_structure,
    make_F_block, f, h, h_jacobian,
    generate_dataset_random_theta,
    generate_false_F_list,
)
from Simulations.Extended_sysmdl import SystemModel
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_old
from emkf.main_emkf_func import E_EMKF_F_analitic_non_linear_h

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
args.N_T   = 200
args.T     = 30
args.T_test = 30

T_test = args.T_test

q2 = 0.01
r2 = 1

cycle              = 5
theta_changed_list = [0.2, 0.2, 0.2, 0.2, 0.2]
assert len(theta_changed_list) == cycle

theta_false = 0.2
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
print(f"  cycle={cycle}  theta_changed_list={theta_changed_list}  theta_false={theta_false}")
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

carry_x_test    = None
carry_theta_test = None

for k in range(cycle):
    theta_changed = theta_changed_list[k]
    _te0 = carry_theta_test[0] if isinstance(carry_theta_test, list) else 0.0
    print(f"  Dataset {k}: theta_base[0]={_te0:.4f} + Uniform(-{theta_changed/2:.3f}, +{theta_changed/2:.3f})")

    xi, xt, th_te, F_te_t = generate_dataset_random_theta(
        args.N_T, T_test, theta_changed, Q, R,
        x_init=carry_x_test, theta_base=carry_theta_test,
    )
    F_te_f = generate_false_F_list(th_te, theta_false)

    carry_x_test    = xt[:, :, -1]   # [N_T, m] — per-sequence carry
    carry_theta_test = th_te

    all_test_inputs.append(xi)
    all_test_targets.append(xt)
    all_F_test_true.append(F_te_t)
    all_F_test_false.append(F_te_f)

print(f"  Test per dataset: {all_test_targets[0].size()}")
print(f"  max_em_iter={max_em_iter}")

#########################################
###  Run ERTS across all datasets     ###
#########################################
# Outer loop over datasets; S_Test_ext_old processes all N_T sequences in one call.
# Per-sequence x/P carries are maintained in lists and updated after each dataset.

print("\nRunning ERTS (true F and false F) across all datasets ...")

N_T = args.N_T

# MSE arrays: [datasets, N_T]
mse_true_arr  = torch.zeros(cycle, N_T)
mse_false_arr = torch.zeros(cycle, N_T)
mse_emkf_arr  = torch.zeros(cycle, N_T)

# Store first-sequence outputs for plotting
out_true_seq0  = []
out_false_seq0 = []
out_emkf_seq0  = []

# Per-sequence carries  (shape [m,1] initially, [m] after first update — both fine)
x0_true_carries  = [m1x_0.clone() for _ in range(N_T)]
P0_true_carries  = [m2x_0.clone() for _ in range(N_T)]
x0_false_carries = [m1x_0.clone() for _ in range(N_T)]
P0_false_carries = [m2x_0.clone() for _ in range(N_T)]

for data in range(cycle):
    # ── ERTS with true F — all N_T sequences at once ──────────────────────
    [mse_arr_true, _, _, X_smooth_true, P_smooth_true, _] = S_Test_ext_old(
        sys_model,
        test_input=all_test_inputs[data],       # [N_T, n, T]
        test_target=all_test_targets[data],     # [N_T, m, T]
        F_list=all_F_test_true[data],           # list of N_T [m,m] matrices
        generate_f=False,
        init_x_list=x0_true_carries,            # list of N_T initial states
        init_P_list=P0_true_carries,            # list of N_T initial covariances
    )
    mse_true_arr[data] = mse_arr_true.cpu()
    out_true_seq0.append(X_smooth_true[0].detach().clone())  # seq-0 output [m, T]

    # P_smooth[:,:,T-1] == P_filt[:,:,T-1] (smoother initialises at T-1 to filtered value)
    for j in range(N_T):
        x0_true_carries[j] = X_smooth_true[j, :, -1].detach()      # [m]
        P0_true_carries[j] = P_smooth_true[j, :, :, -1].detach()   # [m, m]

    # ── ERTS with false F — all N_T sequences at once ─────────────────────
    [mse_arr_false, _, _, X_smooth_false, P_smooth_false, _] = S_Test_ext_old(
        sys_model,
        test_input=all_test_inputs[data],
        test_target=all_test_targets[data],
        F_list=all_F_test_false[data],
        generate_f=False,
        init_x_list=x0_false_carries,
        init_P_list=P0_false_carries,
    )
    mse_false_arr[data] = mse_arr_false.cpu()
    out_false_seq0.append(X_smooth_false[0].detach().clone())

    for j in range(N_T):
        x0_false_carries[j] = X_smooth_false[j, :, -1].detach()
        P0_false_carries[j] = P_smooth_false[j, :, :, -1].detach()

#########################################
###  Run analytic EMKF across datasets ###
#########################################
print("\nRunning analytic EMKF across all datasets ...")

emkf_x_carries = [m1x_0.clone() for _ in range(N_T)]
emkf_P_carries = [m2x_0.clone() for _ in range(N_T)]
emkf_F_carries = [all_F_test_false[0][j].clone() for j in range(N_T)]  # init with false F

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
    )

    # ── Evaluate EMKF: run smoother once with final estimated F per sequence ──
    # Uses OLD carries (before update) to match the same initial condition as EM.
    F_list_emkf = [F_mats_batch[j][-1] for j in range(N_T)]   # list of N_T [m,m]
    [mse_arr_emkf, _, _, X_smooth_emkf, _, _] = S_Test_ext_old(
        sys_model,
        test_input=all_test_inputs[data],
        test_target=all_test_targets[data],
        F_list=F_list_emkf,
        generate_f=False,
        init_x_list=emkf_x_carries,   # OLD carries — not yet updated
        init_P_list=emkf_P_carries,
    )
    mse_emkf_arr[data] = mse_arr_emkf.cpu()
    out_emkf_seq0.append(X_smooth_emkf[0].detach().clone())

    # ── Update carries for next dataset ──────────────────────────────────────
    for j in range(N_T):
        emkf_x_carries[j] = last_x_emkf[j]
        emkf_P_carries[j] = last_P_emkf[j]
        emkf_F_carries[j] = F_mats_batch[j][-1]

#########################################
###  Results summary                  ###
#########################################
print("\n" + "=" * 70)
print(f"RESULTS SUMMARY  (cycle={cycle}, theta_changed_list={theta_changed_list})")
print("=" * 70)

mse_true_db_per_dataset  = [10 * math.log10(mse_true_arr[k].mean().item())  for k in range(cycle)]
mse_false_db_per_dataset = [10 * math.log10(mse_false_arr[k].mean().item()) for k in range(cycle)]
mse_emkf_db_per_dataset  = [10 * math.log10(mse_emkf_arr[k].mean().item())  for k in range(cycle)]

for k in range(cycle):
    print(f"  Dataset {k}  ERTS true-F : {mse_true_db_per_dataset[k]:.2f} dB"
          f"   ERTS false-F: {mse_false_db_per_dataset[k]:.2f} dB"
          f"   EMKF: {mse_emkf_db_per_dataset[k]:.2f} dB")

mse_true_avg_db  = 10 * math.log10(mse_true_arr.mean().item())
mse_false_avg_db = 10 * math.log10(mse_false_arr.mean().item())
mse_emkf_avg_db  = 10 * math.log10(mse_emkf_arr.mean().item())
print(f"\n  ERTS TRUE-F  (overall avg) : {mse_true_avg_db:.2f} dB")
print(f"  ERTS FALSE-F (overall avg) : {mse_false_avg_db:.2f} dB")
print(f"  EMKF         (overall avg) : {mse_emkf_avg_db:.2f} dB")
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
    ax.set_ylabel(f"Dataset {k}\ny position")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Dataset {k} — true: {mse_true_db_per_dataset[k]:.2f} dB  "
                 f"false: {mse_false_db_per_dataset[k]:.2f} dB  "
                 f"EMKF: {mse_emkf_db_per_dataset[k]:.2f} dB")

axes[-1].set_xlabel("time")
fig.suptitle(f"TDOA ERTS analytic — {cycle}-dataset sequential scenario", fontsize=13)
plt.tight_layout()

plot_path = cycle_dir + "analytic_erts_y_position.png"
plt.savefig(plot_path, dpi=250)
print(f"  Saved: {plot_path}")
