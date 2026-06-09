"""
5-cycle multi-dataset TDOA test — RTSNet and EMKFNet (MNet / Joint).

Same dataset generation and evaluation protocol as
microphons_test_analytic_3_dataset.py.

Methods compared
----------------
  ERTS true-F    : oracle upper bound (knows the true theta)
  ERTS false-F   : mismatched baseline (theta = 0 always)
  RTSNet true-F  : RTSNet trained with true F per sequence
  RTSNet false-F : RTSNet trained with false F (theta = 0)
  MNet           : RTSNet false-F + neural M-step for F estimation
  Joint          : RTSNet + MNet jointly trained
"""

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
    make_F_block, f, h, h_jacobian, make_f,
    generate_dataset_raw_batch,
    make_get_F_from_matrix,
    mic_positions,
)
from Simulations.TDOA_2D.ekf_erts import run_ekf_erts
from Simulations.Extended_sysmdl import SystemModel
from Pipelines.Pipeline_mic import Pipeline_mic as Pipeline
from RTSNet.RTSNet_nn import RTSNetNN
from Baselines.BiGRU_smoother import test_bigru_smoother

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
args.N_T    = 100
args.T      = 50
args.T_test = 50
args.n_steps = 1
args.n_batch = 1
args.lr      = 1e-3
args.wd      = 1e-3

T_test = args.T_test
N_T    = args.N_T       

q2 = 0.001
r2 = 1.

cycle             = 5
theta_per_dataset = [0.08, -0.08, 0.1, -0.1, 0.06]   # matches analytic test
# theta_per_dataset = [0.08,-0.08,0.1]   # matches analytic test
assert len(theta_per_dataset) == cycle

# Sparse-measurement mask — must match analytic test exactly so ERTS baselines are comparable.
MEASURE_EVERY_K = 1   # 1 = every step (no mask); >1 = sparse
obs_mask = torch.zeros(T_test, dtype=torch.bool)
obs_mask[::MEASURE_EVERY_K] = True
obs_mask = None if MEASURE_EVERY_K == 1 else obs_mask   # None → full observations (reversible)

num_em_iters = 2

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

save_dir  = "RTSNet/tdoa_2d/3mics/r1/cycle1/"
cycle_dir = save_dir + f"{cycle}cycle/"
# cycle_dir = save_dir
# cycle_dir = "RTSNet/tdoa_2d/3mics/r1/cycle1/"
os.makedirs(cycle_dir, exist_ok=True)

# Checkpoint paths — populated by microphons_training_3_dataset.py
path_rtsnet_true  = cycle_dir + "5dRTSNet_true0.001.pt"
path_rtsnet_true  =  "RTSNet/tdoa_2d/3mics/new_x0/r1/5cycle/"+ "5dRTSNet_true0.001.pt"
path_rtsnet_false = "RTSNet/tdoa_2d/3mics/new_x0/r1/5cycle/" + "5dRTSNet_false0.001.pt"
path_M_F          = "RTSNet/tdoa_2d/3mics/new_x0/r1/5cycle/" + "5dM_step_F_net0.001.pt"
path_rtsnet_joint = "RTSNet/tdoa_2d/3mics/new_x0/r1/5cycle/" + "5dRTSNet_falseF_joint0.001.pt"
path_M_F_joint    = "RTSNet/tdoa_2d/3mics/new_x0/r1/5cycle/" + "5dM_step_F_net_joint0.001.pt"
# path_rtsnet_true  = cycle_dir + "5dRTSNet_true0.001.pt"
# path_rtsnet_false = cycle_dir + "5dRTSNet_false0.001.pt"
# path_M_F          = cycle_dir + "5dM_step_F_net0.001.pt"
# path_rtsnet_joint = save_dir + "RTSNet_falseF_joint0.001.pt"
# path_M_F_joint    = save_dir + "M_step_F_net_joint0.001.pt"
path_bigru        = cycle_dir + "BiGRU.pt"

data_path = save_dir + "test_nn_3_dataset_data.pt"

###################
###    FLAGS     ###
###################
LOAD_DATA      = False  # True → skip generation, load data from data_path
OVERSAMPLE     = 1.15   # generate ceil(N_T × OVERSAMPLE) candidates, keep best N_T
VARY_NOISE     = False   # True → r2 changes per dataset; False = original fixed-noise behaviour
r2_per_dataset = [0.1, 0.5, 1.0, 1.5, 2.0]   # only used when VARY_NOISE=True
# Trajectory physics flags (edit in Simulations/TDOA_2D/parameters.py):
#   USE_BOUNDARIES — True: enforce px/py/v bounds   False: unbounded
#   USE_REFLECTION — True: bounce at walls           False: reject (good_seq=0)

print("=" * 70)
print(f"2D TDOA RTSNet / EMKFNet — {cycle}-cycle multi-dataset test")
print(f"  T_test={T_test}  q2={q2}  r2={r2}")
print(f"  cycle={cycle}  theta_per_dataset={theta_per_dataset}  false F fixed at theta=0")
print(f"  Microphones: {M_mics}   State dim: {m}   Obs dim: {n}")
print("=" * 70)

#########################################
###  Generate / load data             ###
#########################################
all_test_inputs  = []
all_test_targets = []
all_F_test_true  = []
all_F_test_false = []

if LOAD_DATA and os.path.exists(data_path):
    print(f"\nLoading saved data from {data_path} ...")
    _d = torch.load(data_path, weights_only=False, map_location=device)
    all_test_inputs  = _d["all_test_inputs"]
    all_test_targets = _d["all_test_targets"]
    all_F_test_true  = _d["all_F_test_true"]
    all_F_test_false = _d["all_F_test_false"]
    print("  Done.")
else:
    N_gen_T = math.ceil(N_T * OVERSAMPLE)
    good_seq_test = [1] * N_gen_T
    raw_test_inputs  = [];  raw_test_targets = [];  raw_F_test = []
    carry_test = None
    print(f"\nGenerating {cycle} test datasets (N_gen={N_gen_T}) ...")
    for k in range(cycle):
        theta_k = theta_per_dataset[k]
        R_k = (r2_per_dataset[k] * R_structure).to(device) if VARY_NOISE else R
        print(f"  Dataset {k}: theta={theta_k:.4f} rad  r2={r2_per_dataset[k] if VARY_NOISE else r2} ...", end="", flush=True)
        xi, xt, F_te, v_te = generate_dataset_raw_batch(
            N_gen_T, T_test, 0, Q, R_k,
            x_init=carry_test, theta_base=[theta_k] * N_gen_T,
        )
        for i in range(N_gen_T):
            if not v_te[i]: good_seq_test[i] = 0
        raw_test_inputs.append(xi);  raw_test_targets.append(xt);  raw_F_test.append(F_te)
        carry_test = xt[:, :, -1]
        print(f"  good so far: {sum(good_seq_test)}/{N_gen_T}")
    idx_te = [i for i in range(N_gen_T) if good_seq_test[i]][:N_T]
    if len(idx_te) < N_T:
        raise RuntimeError(
            f"Not enough valid test sequences: {len(idx_te)}/{N_T}. "
            f"Increase OVERSAMPLE ({OVERSAMPLE}).")
    idx_te_t = torch.tensor(idx_te, dtype=torch.long)
    for k in range(cycle):
        all_test_inputs.append(raw_test_inputs[k][idx_te_t])
        all_test_targets.append(raw_test_targets[k][idx_te_t])
        all_F_test_true.append([raw_F_test[k][i] for i in idx_te])
        all_F_test_false.append([make_F_block(0.0) for _ in idx_te])

    print(f"\n  Saving data to {data_path} ...")
    torch.save({
        "all_test_inputs":  all_test_inputs,
        "all_test_targets": all_test_targets,
        "all_F_test_true":  all_F_test_true,
        "all_F_test_false": all_F_test_false,
    }, data_path)
    print("  Done.")

print(f"  Test per dataset: {all_test_targets[0].size()}")

#########################################
###  System models                    ###
#########################################
H_prior = h_jacobian(m1x_0.reshape(-1))   # [n, m]
F_init  = make_F_block(0.0)
f_init  = make_f(F_init)

sys_model_true = SystemModel(f=f_init, Q=Q, h=h, R=R,
                             T=T_test, T_test=T_test, m=m, n=n, H=H_prior,
                             prior_S=torch.eye(n, device=device))
sys_model_true.F      = F_init
sys_model_true.F_test = all_F_test_true   # [cycle][N_T] — true F per sequence
sys_model_true.InitSequence(m1x_0, m2x_0)

sys_model_false = SystemModel(f=f_init, Q=Q, h=h, R=R,
                              T=T_test, T_test=T_test, m=m, n=n, H=H_prior,
                              prior_S=torch.eye(n, device=device))
sys_model_false.F      = F_init           # theta=0 seed for MNet carry
sys_model_false.F_test = all_F_test_false # [cycle][N_T] — false F per sequence
sys_model_false.InitSequence(m1x_0, m2x_0)

#########################################
###  Run ERTS baselines               ###
#########################################
print("\nRunning ERTS (true F and false F) ...")

mse_true_arr  = torch.zeros(cycle, N_T)
mse_false_arr = torch.zeros(cycle, N_T)

out_true_seq0  = []
out_false_seq0 = []

for j in range(N_T):
    x0_true  = m1x_0.clone()
    P0_true  = m2x_0.clone()
    x0_false = m1x_0.clone()
    P0_false = m2x_0.clone()

    for data in range(cycle):
        y_seq  = all_test_inputs[data][j]
        x_true = all_test_targets[data][j]
        R_erts = (r2_per_dataset[data] * R_structure).to(device) if VARY_NOISE else R

        get_F_true  = make_get_F_from_matrix(all_F_test_true[data][j])
        get_F_false = make_get_F_from_matrix(all_F_test_false[data][j])

        x_s_true, _, P_f_true, *_ = run_ekf_erts(
            y_seq, get_F_true, Q_in=Q, R_in=R_erts,
            x_init=x0_true, P_init=P0_true,
            obs_mask=obs_mask,
        )
        mse_true_arr[data, j] = loss_fn(x_s_true, x_true).item()
        x0_true = x_s_true[:, -1].detach()
        P0_true = P_f_true[:, :, -1].detach()

        x_s_false, _, P_f_false, *_ = run_ekf_erts(
            y_seq, get_F_false, Q_in=Q, R_in=R_erts,
            x_init=x0_false, P_init=P0_false,
            obs_mask=obs_mask,
        )
        mse_false_arr[data, j] = loss_fn(x_s_false, x_true).item()
        x0_false = x_s_false[:, -1].detach()
        P0_false = P_f_false[:, :, -1].detach()

        if j == 0:
            out_true_seq0.append(x_s_true)
            out_false_seq0.append(x_s_false)

#########################################
###  F-matrix proof                   ###
#########################################
print("\n" + "=" * 60)
print("F-MATRIX PROOF — verifying RTSNet receives different F")
print("=" * 60)
_F_t = all_F_test_true[0][0]    # true F, dataset 0, seq 0
_F_f = all_F_test_false[0][0]   # false F, dataset 0, seq 0
_x0  = m1x_0.reshape(-1)

print(f"  Dataset 0 theta_true  = {math.atan2(_F_t[3,2].item(), _F_t[2,2].item()):.4f} rad")
print(f"  Dataset 0 theta_false = {math.atan2(_F_f[3,2].item(), _F_f[2,2].item()):.4f} rad")
print(f"  True  F velocity block:  [[{_F_t[2,2]:.4f}, {_F_t[2,3]:.4f}], [{_F_t[3,2]:.4f}, {_F_t[3,3]:.4f}]]")
print(f"  False F velocity block:  [[{_F_f[2,2]:.4f}, {_F_f[2,3]:.4f}], [{_F_f[3,2]:.4f}, {_F_f[3,3]:.4f}]]")

_pred_true  = _F_t @ _x0
_pred_false = _F_f @ _x0
_diff1 = (_pred_true - _pred_false).abs()
print(f"\n  One-step prediction from x0={_x0.cpu().numpy().round(3)}:")
print(f"    F_true  @ x0 = {_pred_true.cpu().detach().numpy().round(4)}")
print(f"    F_false @ x0 = {_pred_false.cpu().detach().numpy().round(4)}")
print(f"    |diff| t=1    = pos {_diff1[:2].norm().item():.4f} m   vel {_diff1[2:].norm().item():.4f} m/s")

# Propagate T steps (no noise) to see cumulative divergence
_traj_true  = torch.zeros(m, T_test, device=device)
_traj_false = torch.zeros(m, T_test, device=device)
_xT_true  = _x0.clone()
_xT_false = _x0.clone()
for _t in range(T_test):
    _xT_true  = _F_t @ _xT_true
    _xT_false = _F_f @ _xT_false
    _traj_true[:, _t]  = _xT_true
    _traj_false[:, _t] = _xT_false
_diffT = (_xT_true - _xT_false).abs()
_mse_ol = loss_fn(_traj_false, _traj_true).item()
_mse_ol_db = 10 * math.log10(_mse_ol)
print(f"\n  After T={T_test} steps (no noise — pure model error):")
print(f"    F_true^T  @ x0 = {_xT_true.cpu().detach().numpy().round(3)}")
print(f"    F_false^T @ x0 = {_xT_false.cpu().detach().numpy().round(3)}")
print(f"    |diff| t=T     = pos {_diffT[:2].norm().item():.3f} m   vel {_diffT[2:].norm().item():.3f} m/s")
print(f"    Open-loop MSE (false vs true trajectory): {_mse_ol:.4f} = {_mse_ol_db:.2f} dB")
print(f"    → this is the MAXIMUM gap possible if no observations correct the error")

_tmp = RTSNetNN()
_tmp.NNBuild(sys_model_true, args)
_tmp.update_F(_F_t)
print(f"\n  model.F after update_F(F_true) [2,2:]  = [{_tmp.F[2,2]:.4f}, {_tmp.F[2,3]:.4f}, {_tmp.F[3,2]:.4f}, {_tmp.F[3,3]:.4f}]")
_tmp.update_F(_F_f)
print(f"  model.F after update_F(F_false) [2,2:] = [{_tmp.F[2,2]:.4f}, {_tmp.F[2,3]:.4f}, {_tmp.F[3,2]:.4f}, {_tmp.F[3,3]:.4f}]")
del _tmp, _F_t, _F_f, _x0, _pred_true, _pred_false, _diff1, _xT_true, _xT_false, _diffT, _traj_true, _traj_false, _t, _mse_ol, _mse_ol_db
print("=" * 60)

#########################################
###  RTSNet true-F                    ###
#########################################
print("\nRTSNet TRUE-F ...")
RTSNet_model_true = RTSNetNN()
RTSNet_model_true.NNBuild(sys_model_true, args)
RTSNet_Pipeline_true = Pipeline(strTime, "RTSNet", "RTSNet_TDOA_trueF")
RTSNet_Pipeline_true.setssModel(sys_model_true)
RTSNet_Pipeline_true.setModel(RTSNet_model_true, args)
RTSNet_Pipeline_true.setTrainingParams(args)

[MSE_arr_rt, MSE_avg_rt, MSE_dB_rt,
 rtsnet_out_true, _] = RTSNet_Pipeline_true.NNTest_3_datasets(
    sys_model_true, all_test_inputs, all_test_targets,
    path_rtsnet_true, generate_f=True, datasets=cycle,
    obs_mask=obs_mask,
)

mse_rt_db = [10 * math.log10(MSE_arr_rt[k * N_T:(k + 1) * N_T].mean().item())
             for k in range(cycle)]

#########################################
###  RTSNet false-F                   ###
#########################################
print("\nRTSNet FALSE-F ...")
RTSNet_model_false = RTSNetNN()
RTSNet_model_false.NNBuild(sys_model_false, args)
RTSNet_Pipeline_false = Pipeline(strTime, "RTSNet", "RTSNet_TDOA_falseF")
RTSNet_Pipeline_false.setssModel(sys_model_false)
RTSNet_Pipeline_false.setModel(RTSNet_model_false, args)
RTSNet_Pipeline_false.setTrainingParams(args)

[MSE_arr_rf, MSE_avg_rf, MSE_dB_rf,
 rtsnet_out_false, _] = RTSNet_Pipeline_false.NNTest_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    path_rtsnet_false, generate_f=True, datasets=cycle,
    obs_mask=obs_mask,
)

mse_rf_db = [10 * math.log10(MSE_arr_rf[k * N_T:(k + 1) * N_T].mean().item())
             for k in range(cycle)]

#########################################
###  MNet (EMKFNet)                   ###
#########################################
print("\nMNet (EMKFNet) ...")

[MSE_arr_mnet, MSE_avg_mnet, MSE_dB_mnet,
 rtsnet_out_mnet, _] = RTSNet_Pipeline_false.test_F_mstep_net_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    path_rtsnet_false, path_M_F,
    num_em_iters=num_em_iters, generate_f=True, datasets=cycle,
    propagate_F=False, obs_mask=obs_mask,
)

mse_mnet_db = [10 * math.log10(MSE_arr_mnet[k * N_T:(k + 1) * N_T].mean().item())
               for k in range(cycle)]

#########################################
###  Joint (RTSNet + MNet)            ###
#########################################
print("\nJoint (RTSNet + MNet) ...")

[MSE_arr_joint, MSE_avg_joint, MSE_dB_joint,
 rtsnet_out_joint, _] = RTSNet_Pipeline_false.test_F_mstep_net_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    path_rtsnet_joint, path_M_F_joint,
    num_em_iters=num_em_iters, generate_f=True, datasets=cycle,
    propagate_F=False, obs_mask=obs_mask,
)

mse_joint_db = [10 * math.log10(MSE_arr_joint[k * N_T:(k + 1) * N_T].mean().item())
                for k in range(cycle)]

#########################################
###  BiGRU baseline                   ###
#########################################
print("\nBiGRU ...")
mse_bigru_all, mse_bigru_db, bigru_x_hat = test_bigru_smoother(
    all_test_inputs, all_test_targets, path_bigru, device, obs_mask=obs_mask,
)
# x_hat shape: [N_T * cycle, m, T] — datasets concatenated in order 0..cycle-1
mse_bigru_per_ds = [
    10 * math.log10(
        ((bigru_x_hat[k * N_T:(k + 1) * N_T] -
          all_test_targets[k].to(device)) ** 2).mean().item()
    )
    for k in range(cycle)
]
# per-sequence outputs for plot: bigru_x_hat[k * N_T + j] → shape [m, T]
rtsnet_out_bigru = [bigru_x_hat[k * N_T] for k in range(cycle)]

#########################################
###  Results summary                  ###
#########################################
print("\n" + "=" * 70)
print(f"RESULTS SUMMARY  (cycle={cycle}, theta_per_dataset={[round(t,2) for t in theta_per_dataset]})")
print("=" * 70)

mse_true_db  = [10 * math.log10(mse_true_arr[k].mean().item())  for k in range(cycle)]
mse_false_db = [10 * math.log10(mse_false_arr[k].mean().item()) for k in range(cycle)]

for k in range(cycle):
    print(f"  Dataset {k} (theta={theta_per_dataset[k]:.2f})"
          f"  ERTS-T: {mse_true_db[k]:6.2f} dB"
          f"  ERTS-F: {mse_false_db[k]:6.2f} dB"
          f"  RTSNet-T: {mse_rt_db[k]:6.2f} dB"
          f"  RTSNet-F: {mse_rf_db[k]:6.2f} dB"
          f"  MNet: {mse_mnet_db[k]:6.2f} dB"
          f"  Joint: {mse_joint_db[k]:6.2f} dB"
          f"  BiGRU: {mse_bigru_per_ds[k]:6.2f} dB")

print()
print(f"  ERTS true-F    (overall) : {10*math.log10(mse_true_arr.mean().item()):.2f} dB")
print(f"  ERTS false-F   (overall) : {10*math.log10(mse_false_arr.mean().item()):.2f} dB")
print(f"  RTSNet true-F  (overall) : {MSE_dB_rt.item():.2f} dB")
print(f"  RTSNet false-F (overall) : {MSE_dB_rf.item():.2f} dB")
print(f"  MNet           (overall) : {MSE_dB_mnet.item():.2f} dB")
print(f"  Joint          (overall) : {MSE_dB_joint.item():.2f} dB")
print(f"  BiGRU          (overall) : {mse_bigru_db:.2f} dB")
print("=" * 70)

#########################################
###  Plot — sequence 0, all datasets  ###
#########################################
print("\nPlotting sequence 0 across all datasets ...")
t_axis = torch.arange(T_test)

# rtsnet_out_* is [N_T * cycle, m, T] ordered: x_out[j * cycle + data]
# so sequence 0, dataset k => index k

fig, axes = plt.subplots(cycle, 1, figsize=(14, 3 * cycle), sharex=True)
for k in range(cycle):
    ax = axes[k]
    states = all_test_targets[k][0]

    ax.plot(t_axis, states.cpu()[1],                              lw=2.5, label="true p_y")
    ax.plot(t_axis, out_true_seq0[k].cpu()[1],   "--",           lw=2,   label="ERTS true-F")
    ax.plot(t_axis, out_false_seq0[k].cpu()[1],  ":",            lw=2,   label="ERTS false-F")
    ax.plot(t_axis, rtsnet_out_true[k].cpu()[1],  "-.",          lw=2,   label="RTSNet true-F")
    ax.plot(t_axis, rtsnet_out_false[k].cpu()[1], "-",           lw=1.5, label="RTSNet false-F", alpha=0.8)
    ax.plot(t_axis, rtsnet_out_mnet[k].cpu()[1],  "-",           lw=1.5, label="MNet",           alpha=0.7)
    ax.plot(t_axis, rtsnet_out_joint[k].cpu()[1], "-",           lw=1.5, label="Joint",          alpha=0.7)
    ax.plot(t_axis, rtsnet_out_bigru[k].cpu()[1], "--",          lw=1.5, label="BiGRU",          alpha=0.7)

    ax.set_ylabel(f"ds{k} (θ={theta_per_dataset[k]:.2f})\ny position")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Dataset {k} — "
        f"ERTS-T: {mse_true_db[k]:.2f} dB  "
        f"ERTS-F: {mse_false_db[k]:.2f} dB  "
        f"RTSNet-T: {mse_rt_db[k]:.2f} dB  "
        f"RTSNet-F: {mse_rf_db[k]:.2f} dB  "
        f"MNet: {mse_mnet_db[k]:.2f} dB  "
        f"Joint: {mse_joint_db[k]:.2f} dB  "
        f"BiGRU: {mse_bigru_per_ds[k]:.2f} dB",
        fontsize=8,
    )

axes[-1].set_xlabel("time step")
fig.suptitle(
    f"TDOA RTSNet / EMKFNet — {cycle}-dataset sequential scenario",
    fontsize=13,
)
plt.tight_layout()

plot_path = cycle_dir + "nn_y_position.png"
plt.savefig(plot_path, dpi=250)
plt.close()
print(f"  Saved: {plot_path}")

#########################################
###  2-D trajectory popup — sequence 0 ###
#########################################
print("\nPlotting 2D trajectory for sequence 0 ...")

# Concatenate all datasets along time for sequence 0
def _cat(arr):
    return torch.cat([arr[k].cpu() for k in range(cycle)])

true_px   = _cat([all_test_targets[k][0][0] for k in range(cycle)])
true_py   = _cat([all_test_targets[k][0][1] for k in range(cycle)])
erts_t_px = _cat([out_true_seq0[k][0]       for k in range(cycle)])
erts_t_py = _cat([out_true_seq0[k][1]       for k in range(cycle)])
erts_f_px = _cat([out_false_seq0[k][0]      for k in range(cycle)])
erts_f_py = _cat([out_false_seq0[k][1]      for k in range(cycle)])
rt_t_px   = _cat([rtsnet_out_true[k][0]     for k in range(cycle)])
rt_t_py   = _cat([rtsnet_out_true[k][1]     for k in range(cycle)])
rt_f_px   = _cat([rtsnet_out_false[k][0]    for k in range(cycle)])
rt_f_py   = _cat([rtsnet_out_false[k][1]    for k in range(cycle)])
mnet_px   = _cat([rtsnet_out_mnet[k][0]     for k in range(cycle)])
mnet_py   = _cat([rtsnet_out_mnet[k][1]     for k in range(cycle)])
jnt_px    = _cat([rtsnet_out_joint[k][0]    for k in range(cycle)])
jnt_py    = _cat([rtsnet_out_joint[k][1]    for k in range(cycle)])
bgru_px   = _cat([rtsnet_out_bigru[k][0]    for k in range(cycle)])
bgru_py   = _cat([rtsnet_out_bigru[k][1]    for k in range(cycle)])

matplotlib.use("TkAgg")   # switch to interactive backend for popup

fig2d, ax2d = plt.subplots(figsize=(10, 8))

ax2d.plot(true_px,   true_py,   "k-",   lw=2.5,  label="True",           zorder=6)
ax2d.plot(erts_t_px, erts_t_py, "--",   lw=1.8,  label="ERTS true-F",    zorder=5)
ax2d.plot(erts_f_px, erts_f_py, ":",    lw=1.8,  label="ERTS false-F",   zorder=5)
ax2d.plot(rt_t_px,   rt_t_py,   "-.",   lw=1.6,  label="RTSNet true-F",  zorder=5)
ax2d.plot(rt_f_px,   rt_f_py,   "-",    lw=1.4,  label="RTSNet false-F", zorder=4, alpha=0.8)
ax2d.plot(mnet_px,   mnet_py,   "-",    lw=1.4,  label="MNet",           zorder=4, alpha=0.8)
ax2d.plot(jnt_px,    jnt_py,    "-",    lw=1.4,  label="Joint",          zorder=4, alpha=0.8)
ax2d.plot(bgru_px,   bgru_py,   "--",   lw=1.4,  label="BiGRU",          zorder=4, alpha=0.8)

# Dataset boundary markers (start of each dataset on the true trajectory)
for k in range(cycle):
    bx = all_test_targets[k][0][0, 0].cpu().item()
    by = all_test_targets[k][0][1, 0].cpu().item()
    ax2d.scatter(bx, by, color="black", s=60, zorder=7,
                 marker="o" if k == 0 else "D")
    ax2d.annotate(f"ds{k} θ={theta_per_dataset[k]:.2f}",
                  (bx, by), textcoords="offset points",
                  xytext=(6, 4), fontsize=8)

# Microphone positions
for idx, mic in enumerate(mic_positions):
    ax2d.scatter(mic[0].item(), mic[1].item(),
                 marker="^", color="red", s=100, zorder=8)
    ax2d.annotate(f"m{idx}", (mic[0].item(), mic[1].item()),
                  textcoords="offset points", xytext=(4, 4), fontsize=8)

ax2d.set_xlabel("p_x")
ax2d.set_ylabel("p_y")
ax2d.set_title(
    f"2D trajectory — seq 0 — all {cycle} datasets concatenated\n"
    + "  ".join(f"ds{k} θ={theta_per_dataset[k]:.2f}" for k in range(cycle)),
    fontsize=11,
)
ax2d.legend(fontsize=9, loc="upper right")
ax2d.grid(True, alpha=0.4)

plot_2d_path = os.path.abspath(cycle_dir + "traj_2d_seq0.png")
plt.savefig(plot_2d_path, dpi=200)
print(f"  Saved: {plot_2d_path}")

if os.name == "nt":
    os.startfile(plot_2d_path)
else:
    plt.show()
