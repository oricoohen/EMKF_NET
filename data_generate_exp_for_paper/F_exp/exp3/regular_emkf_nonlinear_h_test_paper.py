"""
Analytic (non-learned) baselines for the non-linear-h experiment (exp 3).

Run from anywhere:  python regular_emkf_nonlinear_h_test_paper.py
"""
import os
import sys
from pathlib import Path

# Put the repo root on sys.path and anchor every path to it, so this script runs
# correctly from its own folder as well as from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS_ROOT = REPO_ROOT / 'RTSNet' / 'synthetic' / 'AI_M_step'
# NOTE: 'r_0001' selects the SNR bucket -- sweep with r2 below.
#       r2 = 10 -> 'r_10', 1 -> 'r_1', 0.1 -> 'r_01', 0.01 -> 'r_001', 0.001 -> 'r_0001'.
#       Must match EXP_DIR in exp3_train.py.
EXP_DIR = MODELS_ROOT / 'exp_3' / 'r_0001'

# exp3 keeps its OWN data folder; it used to share exp1_1/regular with exp1_2
# and the two scripts overwrote each other's cached test data.
DATA_DIR = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / 'exp_3_datasets'
os.makedirs(DATA_DIR, exist_ok=True)

import torch
from datetime import datetime

# S_Test_ext was refactored to (args, SysModel, ...) and no longer accepts a
# per-sequence F list or x/P carryover; the original function survives as
# S_Test_ext_old, which is what every call below is written against.
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_old as S_Test_ext


from Simulations.Extended_sysmdl import SystemModel, rotate_F, make_rotated_h_nonlinear
from Simulations.utils import DataGen

# Keep the non-linear h during data generation. GenerateBatch calls
# SystemModel.update_h(H) per group, which by default rebinds self.h to a LINEAR
# H@x (and a 3x3 H) and would corrupt the range-bearing observations. Patch it to
# only record H/H_T and leave self.h = h_nonlinear intact -- same fix exp3_train.py
# and exp3_test.py already apply.
def _update_h_keep_nonlinear(self, H):
    self.H = H
    self.H_T = H.T
SystemModel.update_h = _update_h_keep_nonlinear

# EKF/RTS call getJacobian from Lorenz_Atractor.parameters, which hardcodes
# view(-1, m=3) for the 3-D Lorenz state. This experiment is 2-D, so swap in a
# dimension-agnostic version (same patch exp3_train.py applies to Pipeline_ERTS).
import Smoothers.EKF as _ekf_mod
import Smoothers.Extended_RTS_Smoother as _erts_mod
def _getJacobian_nd(x, g):
    y = x.reshape(-1)
    J = torch.autograd.functional.jacobian(g, y)
    return J.reshape(-1, y.shape[0])
_ekf_mod.getJacobian = _getJacobian_nd
_erts_mod.getJacobian = _getJacobian_nd
import Simulations.config as config
# Use parameters_OLD, same as exp3_train.py / exp3_test.py: the newer parameters.py
# has no F / make_f (it is the 3-D spherical-h model), so this import used to fail.
from Simulations.Lorenz_Atractor.parameters_OLD import m1x_0, m2x_0, m, n, \
        F, make_f, h_nonlinear, Q_structure, R_structure

from emkf.main_emkf_func import E_EMKF_F_analitic
# Repro
torch.manual_seed(1)

# ---------------------
# Experiment settings
# ---------------------
args = config.general_settings()
args.N_T   = 40      # number of sequences for test
args.T     = 30      # generation length (train isn't used here)
args.T_test = 30     # test length
cycle      = 3       # number of datasets (rotations)

# Noise levels
q2 = 0.01
r2 = 0.001
Q  = q2 * Q_structure
R  = r2 * R_structure

# Base F guess (2x2). We'll build per-seq lists from this.
# Match the device of the imported params (m1x_0 / Q_structure are on cuda) --
# without this F0 lands on CPU and f_linear_1d dies on a cpu/cuda mismatch.
F0 = torch.tensor([[0.83, 0.20],
                   [0.20, 0.83]], dtype=Q.dtype, device=Q.device)

# ---------------------
# Build per-dataset F lists by rotating per sequence
# ---------------------
F_matrices_for_datasets_d = []
F_test_list = [F0.clone() for _ in range(args.N_T)]  # same F for all sequences initially

# produce (cycle) different F-list datasets by rotating
for i in range(cycle + 1):
    F_matrices_for_datasets_d.append([f.clone() for f in F_test_list])
    # rotate PER-SEQUENCE for the next dataset
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=0.2, many=True, randomit=False)

# drop the unrotated seed set; keep the last `cycle` sets
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]  # length == cycle

# ---------------------
# Generate data per dataset
# ---------------------
dataFolderName_base = str(DATA_DIR) + os.sep
all_inputs_by_F  = []
all_targets_by_F = []
all_F_matrices   = []

x0_list = None  # will carry last x_T as next dataset initials
h_nonlinear_rot = make_rotated_h_nonlinear(h_nonlinear, phi_deg=30.0)
for dataset_id in range(1, cycle + 1):
    print(f"\n=== Generating Dataset {dataset_id} ===")
    F_list_this = F_matrices_for_datasets[dataset_id - 1]  # list of length N_T

    # System model with nonlinear h and f(x) that wraps current F (will be replaced per seq in S_Test_ext)
    sys_model = SystemModel(make_f(F_list_this[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1x_0, m2x_0)

    dataFileName   = f'snr_0{args.T_test}_dataset_{dataset_id}_NL.pt'
    dataFileName_F = f'snr_0_F_dataset_{dataset_id}_NL.pt'

    # Generate
    DataGen(args, sys_model,
            dataFolderName_base + dataFileName,
            dataFolderName_base + dataFileName_F,
            delta=1,
            randomInit_train=False, randomInit_cv=False, randomInit_test=False,
            randomLength=False, Test=True,
            F_gen=F_list_this, x0_list=x0_list)

    # Load
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName_base + dataFileName, weights_only=True, map_location=Q.device)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(
        dataFolderName_base + dataFileName_F, weights_only=True, map_location=Q.device)

    # Prepare initials for *next* dataset from ground-truth last state (same as your linear exp)
    x_last = test_target[:, :, -1].clone()  # [N_T, m]
    x0_list = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]  # list of [m,1]

    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape:  {test_input.shape}")   # [N_T, n, T_test]
    print(f"Test target shape: {test_target.shape}")  # [N_T, m, T_test]
    print(f"Example stored F:  \n{F_test_mat_list[0]}")

    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)

# ---------------------
# TRUE F (upper bound)
# ---------------------
print('\n=== MSE with TRUE F matrices ===')
true_mse_lin_sum = 0.0
x0_last = None
p0_last = None

for dataset_id in range(cycle):
    test_input  = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]  # list length N_T

    # Build sys model (f will be replaced per sequence inside S_Test_ext)
    sys_model = SystemModel(make_f(true_F_for_this_dataset[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1x_0, m2x_0)

    if dataset_id == 0:
        # S_Test_ext returns: [MSE_arr, MSE_lin_avg, MSE_dB_avg, ERTS_out, P_smooth, V, P_filt, K_last]
        _mse_arr, _mse_avg, _mse_db, x_list, p_list, _ = S_Test_ext(sys_model, test_input, test_target,
                         true_F_for_this_dataset,  # F_list (positional)
                         False,                     # generate_f
                         False,                     # randomInit
                         None,                      # test_init
                         None, None)                # init_x_list, init_P_list
    else:
        _mse_arr, _mse_avg, _mse_db, x_list, p_list, _ = S_Test_ext(sys_model, test_input, test_target,
                         true_F_for_this_dataset,False, False, None,
                         x0_last, p0_last)


    # propagate initials for NEXT dataset
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone()            for k in range(args.N_T)]

    true_mse_lin_sum += _mse_avg.item()
    print(f"Dataset {dataset_id + 1} - TRUE F MSE: {_mse_db.item():.3f} dB")

average_true_F_mse_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycle))
print(f"Average MSE with TRUE F matrices: {average_true_F_mse_db:.3f} dB")

# ---------------------
# INITIAL GUESS (no learning)
# ---------------------
print('\n=== MSE with INITIAL GUESS F ===')
F_initial_guess = [F0.clone() for _ in range(args.N_T)]
mse_total_false = 0.0
x0_last = None
p0_last = None

for dataset_id in range(cycle):
    test_input  = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    sys_model = SystemModel(make_f(F_initial_guess[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1x_0, m2x_0)

    if dataset_id == 0:
        _mse_arr, _mse_avg, _mse_db, x_list, p_list, _ = S_Test_ext(sys_model, test_input, test_target,
                         F_initial_guess, False, False, None, None, None)
    else:
        _mse_arr, _mse_avg, _mse_db, x_list, p_list, _ = S_Test_ext(sys_model, test_input, test_target,
                         F_initial_guess, False, False, None, x0_last, p0_last)


    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone()            for k in range(args.N_T)]

    mse_total_false += _mse_avg.item()
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS MSE: {_mse_db.item():.3f} dB")

average_initial_guess_mse_db = 10 * torch.log10(torch.tensor(mse_total_false / cycle))
print(f"Average MSE with INITIAL GUESS F: {average_initial_guess_mse_db:.3f} dB")

# ---------------------
# EMKF (learn F per sequence with nonlinear h)
# ---------------------
print('\n=== MSE with EMKF matrices ===')
F_current_estimate = [F0.clone() for _ in range(args.N_T)]
mse_total_em = 0.0
x0_last = None
p0_last = None

for dataset_id in range(cycle):
    print(f"\n--- EMKF on Dataset {dataset_id + 1} ---")

    test_input  = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    sys_model = SystemModel(make_f(F_current_estimate[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1x_0, m2x_0)

    if dataset_id == 0:
        # Extended EMKF interface: returns (F_matrices, last_iter_mean_MSE_lin, last_x_list, last_P_list)
        F_mats, mse_lin_last, x_last_list, p_last_list = E_EMKF_F_analitic(
            sys_model, F_current_estimate,
            h_nonlinear, Q, R,
            test_input, m1x_0, m2x_0, test_target,
            max_it=3, generate_f=False
        )
    else:
        F_mats, mse_lin_last, x_last_list, p_last_list = E_EMKF_F_analitic(
            sys_model, F_current_estimate,
            h_nonlinear, Q, R,
            test_input, m1x_0, m2x_0, test_target,
            max_it=3, generate_f=False,
            init_x_list=x0_last, init_P_list=p0_last
        )

    # Propagate initials
    x0_last = [x_.clone() for x_ in x_last_list]  # each [m,1]
    p0_last = [P_.clone() for P_ in p_last_list]  # each [m,m]

    # Update current F estimates for next dataset: take the final iterate per sequence
    F_current_estimate = [Fs_per_seq[-1].clone() for Fs_per_seq in F_mats]
    # F_current_estimate = [F0.clone() for _ in range(args.N_T)]


    mse_total_em += float(mse_lin_last)
    print(f"Dataset {dataset_id + 1} - EMKF last-iter MSE: {10*torch.log10(torch.tensor(mse_lin_last)+1e-12):.3f} dB")
    print(f"Example EMKF F (seq 0):\n{F_mats[0][-1]}")
    print(f"Example TRUE F (seq 0):\n{true_F_for_this_dataset[0]}")

MSE_emkf_db = 10 * torch.log10(torch.tensor(mse_total_em / cycle))
print("\n=== EMKF iterations completed ===")
print(f"Final F estimate (seq 0):\n{F_current_estimate[0]}")
print(f"Average MSE across datasets (EMKF): {MSE_emkf_db:.3f} dB")

# ---------------------
# Summary
# ---------------------
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE F (perfect):        {average_true_F_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (learned):    {MSE_emkf_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - MSE_emkf_db):.3f} dB")
print(f"Gap to perfect (TRUE F): {(MSE_emkf_db - average_true_F_mse_db):.3f} dB")