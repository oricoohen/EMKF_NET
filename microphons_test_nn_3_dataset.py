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
args.N_T    = 100
args.T      = 30   # match training T (=30) so it loads the _T30 checkpoints
args.T_test = 30
args.n_steps = 1
args.n_batch = 1
args.lr      = 1e-3
args.wd      = 1e-3

T_test = args.T_test
N_T    = args.N_T       

q2 = 0.01   # matches training (microphons_training_3_dataset.py)
r2 = 1      # matches training

cycle    = 6
# Fixed per-dataset theta (matches analytic test). The TEST scenario is
# deterministic: every sequence in dataset k uses the same theta_per_dataset[k].
theta_per_dataset = [0.1, 0.08, -0.1, -0.08, 0.06, 0.1]
assert len(theta_per_dataset) == cycle

# Measurement mask — MUST match the training script's OBS_DROP so nets are tested
# under the regime they trained in. Dense now (0.0) to match training + the analytic test.
OBS_DROP = 0.0   # keep in sync with training; 0.0 = full observations (dense)
def _make_obs_mask(T, drop):
    mask = torch.ones(T, dtype=torch.bool)
    if drop and drop > 0:
        n_drop = int(round(T * drop))
        drop_idx = torch.linspace(1, T - 1, steps=n_drop).round().long().unique()  # never t=0
        mask[drop_idx] = False
    return mask
obs_mask = None if OBS_DROP == 0 else _make_obs_mask(T_test, OBS_DROP)
drop_tag = f"_drop{int(round(OBS_DROP * 100))}" if OBS_DROP > 0 else ""

num_em_iters = 2

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

save_dir  = "RTSNet/tdoa_2d/diff_3_mics/r1q001_6cycles/"   # matches training output dir
cycle_dir = save_dir + f"{cycle}cycle/"
os.makedirs(cycle_dir, exist_ok=True)

USE_BIG_MSTEP_NET = True
mstep_arch_tag = "big_mstep" if USE_BIG_MSTEP_NET else "base_mstep"
args.use_big_mstep_net = USE_BIG_MSTEP_NET
args.mstep_hidden_dim  = 512 if USE_BIG_MSTEP_NET else 256

# A1_RES: must match the training flag. Selects the "_a1res" network set so you
# can A/B the residual-feature MNet against the non-residual one.
A1_RES = False   # residual net failed to train from fresh init; use the working non-residual set
a1_tag = "_a1res" if A1_RES else ""
T_tag  = f"_T{args.T}"   # MUST match training T (loads the T=50 vs T=70 network set)

# Freshly trained by microphons_training_3_dataset.py in r10_drift/5cycle
path_rtsnet_false = cycle_dir + f"5dRTSNet_false0.001{T_tag}{drop_tag}.pt"                              # RTSNet-false (no A1_res dependence)
path_M_F          = cycle_dir + f"5dM_step_F_net0.001_big_mstep{a1_tag}{T_tag}{drop_tag}.pt"            # standalone MNet (big)
path_rtsnet_joint = cycle_dir + f"5dRTSNet_falseF_joint0.001_newbig_mstep{a1_tag}{T_tag}{drop_tag}.pt"
path_M_F_joint    = cycle_dir + f"5dM_step_F_net_joint0.001_newbig_mstep{a1_tag}{T_tag}{drop_tag}.pt"

path_rtsnet_true  = cycle_dir + f"5dRTSNet_true0.001{T_tag}{drop_tag}.pt"   # retrained under sparse regime
# BiGRU is NOT retrained under the mask (separate script). Dense-trained, tested sparse →
# treat its number as a handicapped lower bound until you retrain it with the same OBS_DROP.
path_bigru        = cycle_dir + f"BiGRU{T_tag}small.pt"

data_path = save_dir + "test_nn_3_dataset_data.pt"

###################
###    FLAGS     ###
###################
LOAD_DATA      = True  # True → skip generation, load data from data_path
OVERSAMPLE     = 1.3   # generate ceil(N_T × OVERSAMPLE) candidates, keep best N_T
VARY_NOISE     = False   # True → r2 changes per dataset; False = original fixed-noise behaviour
# A1_RES is defined above (with the checkpoint paths) so it can select the network set.
r2_per_dataset = [0.1, 0.5, 1.0, 1.5, 2.0, 1.0]   # only used when VARY_NOISE=True (len == cycle)
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

# Smoother outputs are kept for the first N_PLOT sequences (not just seq 0) so
# the trajectory figures can be produced for each of them: out_*_seqs[j][k].
N_PLOT = min(5, N_T)
out_true_seqs  = [[] for _ in range(N_PLOT)]
out_false_seqs = [[] for _ in range(N_PLOT)]

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

        if j < N_PLOT:
            out_true_seqs[j].append(x_s_true)
            out_false_seqs[j].append(x_s_false)

out_true_seq0  = out_true_seqs[0]    # aliases used by the seq-0 figures below
out_false_seq0 = out_false_seqs[0]

#########################################
###  EMKF regular (analytic EM)       ###
###  — full N_T evaluation, scored    ###
#########################################
# Classical analytic-EM F estimation, scored over ALL sequences like the NN
# methods. Per your spec: F RESETS to the theta=0 init each dataset
# (F_0 = make_F_block(0.0)), while the state (x_0/P_0) is CARRIED from the
# previous dataset's last EMKF estimate. Then the MSE is computed.
print("\nRunning EMKF regular (analytic EM, all sequences) ...")

EMKF_MAX_ITER = 3

mse_emkf_arr  = torch.zeros(cycle, N_T)
out_emkf_seqs = [[] for _ in range(N_PLOT)]   # out_emkf_seqs[j][k] (for the plots)

# Dedicated SystemModel for the EMKF computation — NOT sys_model_false.
# E_EMKF_F_analitic_non_linear_h calls sys_model.InitSequence(x_0, P_0)
# internally on every EM iteration, which overwrites sys_model.m1x_0/m2x_0
# as a side effect. Reusing sys_model_false here would silently corrupt
# the prior that RTSNet false-F / MNet / Joint read afterwards.
sys_model_emkf = SystemModel(f=f_init, Q=Q, h=h, R=R,
                              T=T_test, T_test=T_test, m=m, n=n, H=H_prior,
                              prior_S=torch.eye(n, device=device))
sys_model_emkf.InitSequence(m1x_0, m2x_0)

emkf_x_carries = [m1x_0.clone() for _ in range(N_T)]   # state carried across datasets
emkf_P_carries = [m2x_0.clone() for _ in range(N_T)]

for data in range(cycle):
    R_emkf = (r2_per_dataset[data] * R_structure).to(device) if VARY_NOISE else R
    F0_list = [make_F_block(0.0) for _ in range(N_T)]   # F reset to theta=0 each dataset

    F_mats_batch, _, last_x_emkf, last_P_emkf = E_EMKF_F_analitic_non_linear_h(
        sys_model=sys_model_emkf,
        F_0_matrices=F0_list,
        h=h, Q=Q, R=R_emkf,
        Y=all_test_inputs[data],
        x_0=m1x_0, P_0=m2x_0,
        X=all_test_targets[data],
        max_it=EMKF_MAX_ITER, generate_f=False,
        init_x_list=emkf_x_carries, init_P_list=emkf_P_carries,
        vel_only=True, obs_mask=obs_mask,
    )

    for j in range(N_T):
        F_final_j = F_mats_batch[j][-1]
        x_s_emkf, _, _, *_ = run_ekf_erts(
            all_test_inputs[data][j], make_get_F_from_matrix(F_final_j),
            Q_in=Q, R_in=R_emkf,
            x_init=emkf_x_carries[j], P_init=emkf_P_carries[j],
            obs_mask=obs_mask,
        )
        mse_emkf_arr[data, j] = loss_fn(x_s_emkf, all_test_targets[data][j]).item()
        if j < N_PLOT:
            out_emkf_seqs[j].append(x_s_emkf)

    # carry state (x, P) forward; F is reset (not propagated) into the next dataset
    for j in range(N_T):
        emkf_x_carries[j] = last_x_emkf[j]
        emkf_P_carries[j] = last_P_emkf[j]

out_emkf_seq0    = out_emkf_seqs[0]   # alias used by the seq-0 figures below
mse_emkf_db      = [10 * math.log10(mse_emkf_arr[k].mean().item()) for k in range(cycle)]
mse_emkf_seq0_db = [10 * math.log10(loss_fn(out_emkf_seq0[k], all_test_targets[k][0]).item())
                    for k in range(cycle)]

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
###  DIAGNOSTIC: RTSNet false-F fed TRUE F ###
###  If ≈ RTSNet-true  → bottleneck is MNet F-accuracy.
###  If ≈ RTSNet-false → RTSNet ignores F; only Joint can help.
#########################################
print("\nDIAGNOSTIC: RTSNet false-F + TRUE F (no MNet) ...")
_saved_Ftest = sys_model_false.F_test
sys_model_false.F_test = all_F_test_true          # feed the oracle F into the false-trained net
[MSE_arr_rf_trueF, _, MSE_dB_rf_trueF,
 _, _] = RTSNet_Pipeline_false.NNTest_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    path_rtsnet_false, generate_f=True, datasets=cycle,
    obs_mask=obs_mask,
)
sys_model_false.F_test = _saved_Ftest             # restore false F
mse_rf_trueF_db = [10 * math.log10(MSE_arr_rf_trueF[k * N_T:(k + 1) * N_T].mean().item())
                   for k in range(cycle)]

#########################################
###  MNet (EMKFNet)                   ###
#########################################
print("\nMNet (EMKFNet) ...")

sys_model_false.F_test_TRUE = all_F_test_true   # [cycle][N_T] — enables F_MSE logging in test

[MSE_arr_mnet, MSE_avg_mnet, MSE_dB_mnet,
 rtsnet_out_mnet, _] = RTSNet_Pipeline_false.test_F_mstep_net_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    path_rtsnet_false, path_M_F,
    num_em_iters=num_em_iters, generate_f=True, datasets=cycle,
    propagate_F=False, obs_mask=obs_mask, A1_res=A1_RES,   # F resets to theta=0 each dataset (matches training)
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
    propagate_F=False, obs_mask=obs_mask, A1_res=A1_RES,   # F resets to theta=0 each dataset (matches training)
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
          f"  EMKF: {mse_emkf_db[k]:6.2f} dB"
          f"  RTSNet-T: {mse_rt_db[k]:6.2f} dB"
          f"  RTSNet-F: {mse_rf_db[k]:6.2f} dB"
          f"  MNet: {mse_mnet_db[k]:6.2f} dB"
          f"  Joint: {mse_joint_db[k]:6.2f} dB"
          f"  BiGRU: {mse_bigru_per_ds[k]:6.2f} dB")

# ---- Overall dB ± STD -------------------------------------------------------
# The spread is computed over the per-sequence MSEs (every sequence in every
# dataset is one sample, so n = N_T * cycle) and converted to dB the same way
# the RTSNet pipeline does it:
#     std_dB = 10*log10(MSE_std + MSE_avg) - 10*log10(MSE_avg)
# i.e. the +1σ point expressed relative to the mean, in dB. It is deliberately
# NOT the std of the per-sequence dB values — MSE averaging happens in the
# linear domain, so the error bar has to be built there too. Note this makes
# the bar asymmetric in dB; the reported ± is the upper (larger-error) side.
def _db_pm_std(mse_samples):
    """mse_samples: 1-D tensor of per-sequence MSEs → (dB, std_dB)."""
    s       = mse_samples.reshape(-1).double()
    avg     = s.mean()
    std     = s.std(unbiased=True)
    db      = 10 * torch.log10(avg)
    std_db  = 10 * torch.log10(std + avg) - db
    return db.item(), std_db.item()

# BiGRU per-sequence MSE, same [dataset-major] ordering as bigru_x_hat
mse_bigru_arr = ((bigru_x_hat - torch.cat(all_test_targets, dim=0).to(device)) ** 2
                 ).mean(dim=(1, 2)).detach().cpu()

_overall = [
    ("ERTS true-F",             mse_true_arr,      ""),
    ("ERTS false-F",            mse_false_arr,     ""),
    ("EMKF",                    mse_emkf_arr,      ""),
    ("RTSNet true-F",           MSE_arr_rt,        ""),
    ("RTSNet false-F",          MSE_arr_rf,        ""),
    ("RTSNet false-F + TRUE F", MSE_arr_rf_trueF,  "   <-- diagnostic ceiling for MNet"),
    ("MNet",                    MSE_arr_mnet,      ""),
    ("Joint",                   MSE_arr_joint,     ""),
    ("BiGRU",                   mse_bigru_arr,     ""),
]

print()
print(f"  Overall (mean over all {N_T * cycle} sequence-instances, ± 1σ of the MSE):")
for _name, _arr, _note in _overall:
    _db, _std = _db_pm_std(_arr.detach().cpu() if torch.is_tensor(_arr) else torch.tensor(_arr))
    print(f"  {_name:<24}: {_db:7.2f} ± {_std:.2f} dB{_note}")
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
    ax.plot(t_axis, out_emkf_seq0[k].cpu()[1],   "-.",           lw=1.8, label="EMKF",           alpha=0.9)
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
        f"EMKF: {mse_emkf_seq0_db[k]:.2f} dB  "
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
emkf_px   = _cat([out_emkf_seq0[k][0]       for k in range(cycle)])
emkf_py   = _cat([out_emkf_seq0[k][1]       for k in range(cycle)])
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

fig2d, ax2d = plt.subplots(figsize=(10, 8))

ax2d.plot(true_px,   true_py,   "k-",   lw=2.5,  label="True",           zorder=6)
ax2d.plot(erts_t_px, erts_t_py, "--",   lw=1.8,  label="ERTS true-F",    zorder=5)
ax2d.plot(erts_f_px, erts_f_py, ":",    lw=1.8,  label="ERTS false-F",   zorder=5)
ax2d.plot(emkf_px,   emkf_py,   "-.",   lw=1.6,  label="EMKF",           zorder=4, alpha=0.9)
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
plt.close()
print(f"  Saved: {plot_2d_path}")

#########################################
###  Main comparison figure — seq 0    ###
###  ERTS-T / ERTS-F / EMKF /          ###
###  BiGRU / EMKalmanNet               ###
#########################################
# Single uncluttered trajectory figure with only the five headline methods
# (RTSNet true-F / false-F and standalone MNet are left out on purpose —
# they live in traj_2d_seq0.png above).
print("\nPlotting main comparison trajectory (5 methods, seq 0) ...")

_main_methods = [   # (label, px, py, style, colour, overall dB)
    ("ERTS true-F",  erts_t_px, erts_t_py, "--", "tab:blue",   10 * math.log10(mse_true_arr.mean().item())),
    ("ERTS false-F", erts_f_px, erts_f_py, ":",  "tab:red",    10 * math.log10(mse_false_arr.mean().item())),
    ("EMKF",         emkf_px,   emkf_py,   "-.", "tab:green",  10 * math.log10(mse_emkf_arr.mean().item())),
    ("BiGRU",        bgru_px,   bgru_py,   "--", "tab:orange", mse_bigru_db),
    ("EMKalmanNet",  jnt_px,    jnt_py,    "-",  "tab:purple", MSE_dB_joint.item()),
]

fig_main, ax_main = plt.subplots(figsize=(11, 9))
ax_main.plot(true_px, true_py, "k-", lw=2.6, label="Ground truth", zorder=10)

for _label, _px, _py, _style, _col, _db in _main_methods:
    ax_main.plot(_px, _py, _style, color=_col, lw=1.8, alpha=0.9,
                 label=f"{_label}  ({_db:.2f} dB)", zorder=5)

# dataset boundaries on the true trajectory
for k in range(cycle):
    bx = all_test_targets[k][0][0, 0].cpu().item()
    by = all_test_targets[k][0][1, 0].cpu().item()
    ax_main.scatter(bx, by, color="black", s=55, zorder=11,
                    marker="o" if k == 0 else "D")
    ax_main.annotate(f"ds{k} θ={theta_per_dataset[k]:.2f}", (bx, by),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
ax_main.scatter(true_px[-1], true_py[-1], color="black", s=70, marker="x", zorder=11)

for idx, mic in enumerate(mic_positions):
    ax_main.scatter(mic[0].item(), mic[1].item(), marker="^", color="dimgray", s=95, zorder=12)
    ax_main.annotate(f"m{idx}", (mic[0].item(), mic[1].item()),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)

ax_main.set_xlabel("p_x")
ax_main.set_ylabel("p_y")
ax_main.set_title(
    f"2D trajectory — seq 0, all {cycle} datasets concatenated\n"
    f"q2={q2}  r2={r2}  T={T_test}  (dB = overall MSE over all {N_T} sequences)",
    fontsize=11,
)
ax_main.legend(fontsize=9, loc="best")
ax_main.grid(True, alpha=0.4)
plt.tight_layout()

plot_main_path = os.path.abspath(cycle_dir + "traj_2d_seq0_main5.png")
plt.savefig(plot_main_path, dpi=200)
plt.close(fig_main)
print(f"  Saved: {plot_main_path}")

#########################################
###  Zoomed comparison figures         ###
###  first N_PLOT sequences, no mics   ###
#########################################
# One figure per sequence: the same five methods, microphones dropped and the
# axes cropped to the trajectories themselves, so the frame is filled by the
# path instead of the (far away) mic array. Limits span every plotted curve,
# not just the truth, so a diverging estimate is never silently clipped.
# The figures carry no numbers; each sequence's own full-state MSE (averaged
# over the `cycle` datasets) is printed to stdout instead.
print(f"\nPlotting zoomed trajectory comparisons (no microphones) "
      f"for the first {N_PLOT} sequences ...")

def _methods_for_seq(j):
    """(label, per-dataset [m,T] estimates, style, colour) for sequence j."""
    return [
        ("ERTS true-F",  out_true_seqs[j],                                 "--", "tab:blue"),
        ("ERTS false-F", out_false_seqs[j],                                ":",  "tab:red"),
        ("EMKF",         out_emkf_seqs[j],                                 "-.", "tab:green"),
        ("BiGRU",        [bigru_x_hat[k * N_T + j]     for k in range(cycle)], "--", "tab:orange"),
        ("EMKalmanNet",  [rtsnet_out_joint[j * cycle + k] for k in range(cycle)], "-",  "tab:purple"),
    ]

for j in range(N_PLOT):
    # Ground truth for this sequence, datasets concatenated along time.
    tgt_cat_j = torch.cat([all_test_targets[k][j].cpu() for k in range(cycle)], dim=1)   # [m, cycle*T]
    tpx_j, tpy_j = tgt_cat_j[0], tgt_cat_j[1]

    curves_j = []   # (label, px, py, style, colour, per-sequence dB)
    for _label, _outs, _style, _col in _methods_for_seq(j):
        est_cat = torch.cat([e.detach().cpu() for e in _outs], dim=1)                    # [m, cycle*T]
        _db = 10 * math.log10(((est_cat - tgt_cat_j) ** 2).mean().item())
        curves_j.append((_label, est_cat[0], est_cat[1], _style, _col, _db))

    print(f"  seq {j}: " + "   ".join(f"{c[0]}: {c[5]:.2f} dB" for c in curves_j))

    _xs = [tpx_j] + [c[1] for c in curves_j]
    _ys = [tpy_j] + [c[2] for c in curves_j]
    _xlo = min(float(a.min()) for a in _xs);  _xhi = max(float(a.max()) for a in _xs)
    _ylo = min(float(a.min()) for a in _ys);  _yhi = max(float(a.max()) for a in _ys)
    _pad = 0.05 * max(_xhi - _xlo, _yhi - _ylo, 1e-6)

    fig_zoom, ax_zoom = plt.subplots(figsize=(12, 9))
    ax_zoom.plot(tpx_j, tpy_j, "k-", lw=3.0, label="Ground truth", zorder=10)

    for _label, _px, _py, _style, _col, _db in curves_j:
        ax_zoom.plot(_px, _py, _style, color=_col, lw=2.0, alpha=0.9,
                     label=_label, zorder=5)

    # start / end of the true path + dataset boundaries (they explain the shape)
    ax_zoom.scatter(tpx_j[0],  tpy_j[0],  color="black", s=90,  marker="o", zorder=11)
    ax_zoom.scatter(tpx_j[-1], tpy_j[-1], color="black", s=110, marker="X", zorder=11)
    for k in range(1, cycle):
        bx = all_test_targets[k][j][0, 0].cpu().item()
        by = all_test_targets[k][j][1, 0].cpu().item()
        ax_zoom.scatter(bx, by, facecolors="none", edgecolors="black",
                        s=55, linewidths=1.2, zorder=11)
        ax_zoom.annotate(f"θ={theta_per_dataset[k]:.2f}", (bx, by),
                         textcoords="offset points", xytext=(6, 5), fontsize=12, alpha=0.75)

    ax_zoom.set_xlim(_xlo - _pad, _xhi + _pad)
    ax_zoom.set_ylim(_ylo - _pad, _yhi + _pad)
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.set_xlabel("p_x", fontsize=18)
    ax_zoom.set_ylabel("p_y", fontsize=18)
    ax_zoom.tick_params(axis="both", labelsize=15)
    ax_zoom.legend(fontsize=15, loc="best", framealpha=0.9)
    ax_zoom.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_zoom_path = os.path.abspath(cycle_dir + f"traj_2d_seq{j}_zoom.png")
    plt.savefig(plot_zoom_path, dpi=200)
    plt.close(fig_zoom)
    print(f"  Saved: {plot_zoom_path}")

#########################################
###  Per-method trajectory panels     ###
###  seq 0 — one subplot per method   ###
#########################################
# One panel per method, each showing that method's estimated x-trajectory
# (p_x vs p_y, all datasets concatenated) against the ground truth.
print("\nPlotting per-method trajectory panels for sequence 0 ...")

# (name, est_px, est_py, overall_dB, colour)
_panel_methods = [
    ("ERTS true-F",   erts_t_px, erts_t_py, 10 * math.log10(mse_true_arr.mean().item()), "tab:blue"),
    ("RTSNet false-F", rt_f_px,  rt_f_py,   MSE_dB_rf.item(),                             "tab:orange"),
    ("Joint (MNet)",  jnt_px,    jnt_py,    MSE_dB_joint.item(),                          "tab:green"),
    ("BiGRU",         bgru_px,   bgru_py,   mse_bigru_db,                                 "tab:red"),
]

# Shared axis limits (from the true trajectory) so panels are comparable.
_xmin = float(true_px.min()) - 2.0;  _xmax = float(true_px.max()) + 2.0
_ymin = float(true_py.min()) - 2.0;  _ymax = float(true_py.max()) + 2.0

figm, axesm = plt.subplots(2, 2, figsize=(14, 12))
axesm = axesm.flatten()
for ax, (name, epx, epy, db, col) in zip(axesm, _panel_methods):
    ax.plot(true_px, true_py, "k-", lw=2.5, label="Ground truth", zorder=5)
    ax.plot(epx, epy, "-", color=col, lw=1.8, label=name, zorder=4)

    # start (o) / end (x) of the true trajectory
    ax.scatter(true_px[0],  true_py[0],  color="black", s=55, marker="o", zorder=6)
    ax.scatter(true_px[-1], true_py[-1], color="black", s=65, marker="x", zorder=6)

    # microphone positions
    for idx, mic in enumerate(mic_positions):
        ax.scatter(mic[0].item(), mic[1].item(), marker="^", color="dimgray", s=70, zorder=7)
        ax.annotate(f"m{idx}", (mic[0].item(), mic[1].item()),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlim(_xmin, _xmax);  ax.set_ylim(_ymin, _ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("p_x");  ax.set_ylabel("p_y")
    ax.set_title(f"{name} — {db:.2f} dB")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.4)

figm.suptitle(
    f"Estimated x-trajectory per method — seq 0, all {cycle} datasets concatenated  "
    f"(q2={q2}  r2={r2}  T={T_test})",
    fontsize=13,
)
plt.tight_layout()
plot_panels_path = os.path.abspath(cycle_dir + "traj_2d_seq0_per_method.png")
plt.savefig(plot_panels_path, dpi=200)
plt.close()
print(f"  Saved: {plot_panels_path}")
