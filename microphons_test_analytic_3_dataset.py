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
    make_F_block, h, h_jacobian,
    generate_dataset_random_theta,
    generate_false_F_list,
    make_get_F_from_matrix,
)
from Simulations.TDOA_2D.ekf_erts import run_ekf_erts

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

q2 = 0.1
r2 = 10

cycle              = 5
theta_changed_list = [0.3, 0.3, 0.3, 0.3, 0.3]
assert len(theta_changed_list) == cycle

theta_false = 0.2

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

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
carry_theta_test = 0.0

for k in range(cycle):
    theta_changed = theta_changed_list[k]
    print(f"  Dataset {k}: theta_base={carry_theta_test:.4f} + Uniform(-{theta_changed/2:.3f}, +{theta_changed/2:.3f})")

    xi, xt, th_te, F_te_t = generate_dataset_random_theta(
        args.N_T, T_test, theta_changed, Q, R,
        x_init=carry_x_test, theta_base=carry_theta_test,
    )
    F_te_f = generate_false_F_list(th_te, theta_false)

    carry_x_test    = xt[-1, :, -1]
    carry_theta_test = th_te[-1]

    all_test_inputs.append(xi)
    all_test_targets.append(xt)
    all_F_test_true.append(F_te_t)
    all_F_test_false.append(F_te_f)

print(f"  Test per dataset: {all_test_targets[0].size()}")

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

# Store first-sequence outputs for plotting
out_true_seq0  = []   # smoother output per dataset for sequence 0
out_false_seq0 = []

for j in range(N_T):
    x0_true  = m1x_0.clone()
    P0_true  = m2x_0.clone()
    x0_false = m1x_0.clone()
    P0_false = m2x_0.clone()

    for data in range(cycle):
        y_seq  = all_test_inputs[data][j]
        x_true = all_test_targets[data][j]

        get_F_true  = make_get_F_from_matrix(all_F_test_true[data][j // 10])
        get_F_false = make_get_F_from_matrix(all_F_test_false[data][j // 10])

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

#########################################
###  Results summary                  ###
#########################################
print("\n" + "=" * 70)
print(f"RESULTS SUMMARY  (cycle={cycle}, theta_changed_list={theta_changed_list})")
print("=" * 70)

mse_true_db_per_dataset  = [10 * math.log10(mse_true_arr[k].mean().item())  for k in range(cycle)]
mse_false_db_per_dataset = [10 * math.log10(mse_false_arr[k].mean().item()) for k in range(cycle)]

for k in range(cycle):
    print(f"  Dataset {k}  ERTS true-F : {mse_true_db_per_dataset[k]:.2f} dB"
          f"   ERTS false-F: {mse_false_db_per_dataset[k]:.2f} dB")

mse_true_avg_db  = 10 * math.log10(mse_true_arr.mean().item())
mse_false_avg_db = 10 * math.log10(mse_false_arr.mean().item())
print(f"\n  ERTS TRUE-F  (overall avg) : {mse_true_avg_db:.2f} dB")
print(f"  ERTS FALSE-F (overall avg) : {mse_false_avg_db:.2f} dB")
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
    ax.plot(t_axis, states.cpu()[1],                 linewidth=2.5, label="true p_y")
    ax.plot(t_axis, out_true_seq0[k].cpu()[1],  "--", linewidth=2,   label="ERTS true F")
    ax.plot(t_axis, out_false_seq0[k].cpu()[1], ":",  linewidth=2,   label="ERTS false F")
    ax.set_ylabel(f"Dataset {k}\ny position")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Dataset {k} — true: {mse_true_db_per_dataset[k]:.2f} dB  "
                 f"false: {mse_false_db_per_dataset[k]:.2f} dB")

axes[-1].set_xlabel("time")
fig.suptitle(f"TDOA ERTS analytic — {cycle}-dataset sequential scenario", fontsize=13)
plt.tight_layout()

plot_path = cycle_dir + "analytic_erts_y_position.png"
plt.savefig(plot_path, dpi=250)
print(f"  Saved: {plot_path}")
