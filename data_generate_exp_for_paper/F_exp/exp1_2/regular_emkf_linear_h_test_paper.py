"""
Analytic (non-learned) baselines for the linear-h / varying-F paper experiment.

Same 3-dataset setup as exp_1and2_testing.py -- the true F drifts by a FIXED
THETA_TEST rad per dataset, with x and P carried over between datasets -- but
evaluated with the closed-form smoothers instead of the learned ones:

  1) TRUE F          -- RTS smoother given the true F (upper bound)
  2) INITIAL GUESS F -- RTS smoother stuck with the nominal F (lower bound)
  3) analytic EMKF   -- EMKF_F_analitic, the classical EM estimate of F

Run from anywhere:  python regular_emkf_linear_h_test_paper.py
"""
import os
import sys
from pathlib import Path

# Put the repo root on sys.path and anchor every path to it, so this script runs
# correctly from its own folder as well as from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F

# ============================================================
# EXPERIMENT SWITCH -- which paper experiment this run produces
#   'gauss'       = exp 1 (Gaussian process/observation noise)
#   'exponential' = exp 2 (Exponential, heavy-tailed, non-zero-mean noise)
# Must be set BEFORE any DataGen call. Noise strength still comes from
# q2 / r2 below; this only changes the DISTRIBUTION.
# ============================================================
import Simulations.Linear_sysmdl as _lsm
_lsm.NOISE_DIST = 'gauss'
print(f"NOISE_DIST = {_lsm.NOISE_DIST}  (exp 1 = gauss, exp 2 = exponential)")

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from emkf.main_emkf_func import EMKF_F_analitic, EMKF_F_solo
from Simulations.utils import DataLoader, DataGen, estimate_QR

# Where the generated test data is cached. Same folder as exp_1and2_testing.py,
# so the two scripts evaluate the identical sequences.
DATA_DIR = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / 'exp1_1' / 'regular'
os.makedirs(DATA_DIR, exist_ok=True)

# # # For PyTorch
torch.manual_seed(1)

# Match the device of the imported params (m1_0, m2_0, Q_structure, R_structure are on cuda)
device = m1_0.device


args = config.general_settings()
args.N_T = 10  # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30       # Length of the time series (train/cv; unused here since Test=True).
args.T_test = 30  # Length of the time series for test sequences.
cycle = 3         # number of sequential datasets, each with a further-drifted true F

# True model noise variances.
q2 = 0.01
r2 = 0.01

# Alternative SNR-based parameterization (paper convention), if you prefer it:
#   v_db, snr_db = 0, 0          # v_db = 10*log10(r2/q2)
#   r2 = 10.0 ** (-snr_db / 10.0)
#   q2 = r2 / (10.0 ** (v_db / 10.0))

Q = q2 * Q_structure
R = r2 * R_structure

F = torch.tensor([[0.83, 0.2],
              [0.2, 0.83]], device=device)
H = torch.tensor([[1., 1.], [0.25, 1.]], device=device)


SystemModel.F_gen = False
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)

##############################################################################
# Build one true F per dataset: the base F rotated by a FIXED THETA_TEST rad,
# applied cumulatively so the drift accumulates across the datasets. Each
# dataset then generates its sequences from its own F, and we feed the nominal
# F plus the drifted sequences to the EM and sum the MSE.
#
# THETA_TEST matches exp_1and2_testing.py so the analytic and learned results
# are measured on the same drift.
##############################################################################
THETA_TEST = 0.2

F_matrices_for_datasets_d =[]
F_test_list = [F.clone() for _ in range(args.N_T)]  # 1 F per seq (same F)
for i in range(cycle +1):
    # deep copy the list of tensors
    F_matrices_for_datasets_d.append([f.clone() for f in F_test_list])
    # rotate PER-SEQUENCE for the next dataset (rotate_F returns a list of [n,n])
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=THETA_TEST, many=True, randomit=False)
# Drop the un-rotated base F: dataset k uses the F after k+1 rotations.
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]

# Store all inputs and targets organized by F matrix
all_inputs_by_F = []
all_targets_by_F = []
all_F_matrices = []

x0_list = None
# Fixed 2x2 observation matrix per sequence (one H per test sequence, all identical).
# Passed as H_gen so DataGen uses this instead of the random 3x3 Lorenz H generator.
H_test_list = [H.clone() for _ in range(args.N_T)]
# Generate one dataset per drifted F
for dataset_id in range(1, cycle+1):
    print(f"\n=== Generating Dataset {dataset_id} ===")

    # Select F matrix for this dataset. NOTE: deliberately not named `F` -- that
    # would shadow the base F, which the final summary prints as "Original F".
    F_current = F_matrices_for_datasets[dataset_id - 1]
    print(f"F matrix for dataset {dataset_id}:")
    print(F_current)
    # Create system model
    SystemModel.F_gen = False
    sys_model = SystemModel(F_matrices_for_datasets[dataset_id - 1][0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    # File names (DATA_DIR is absolute -- see the path block at the top)
    dataFilePath = str(DATA_DIR / f'snr_0{args.T_test}_dataset_{dataset_id}.pt')
    dataFilePath_F = str(DATA_DIR / f'snr_0_F_dataset_{dataset_id}.pt')

    # Generate data
    print(f"Generating data for dataset {dataset_id}...")

    DataGen(args, sys_model, dataFilePath, dataFilePath_F,
        delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
        randomLength=False, Test=True, F_gen=F_matrices_for_datasets[dataset_id - 1],
        H_gen=H_test_list, x0_list=x0_list)
    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFilePath, weights_only=True, map_location=device)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFilePath_F, map_location=device)

    # test_target: [N_T, m, T]
    x_last = test_target[:, :, -1].clone()
    x0_list = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]  # list of [m,1] tensors

    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")
    print(f"F matrix stored: {F_test_mat_list[0]}")

    # Store in our organized lists
    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)

#########################################################################################################
# RTS_out has shape [N_T, n, T] and is our "x_est"
# P_smooth has shape [N_T, n, n, T] and is the covariance we want to evaluate
# test_target has shape [N_T, n, T] and is our "x_true"

F_initial_gues_1 = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device)
F_current_estimate = [F_initial_gues_1 .clone() for _ in range(args.N_T)]
F_initial_estimate = [F_initial_gues_1 .clone() for _ in range(args.N_T)]

###############################################################################################
##estimate Q and R from data
# gauss = False
# if gauss:
#     combined_target = torch.cat(all_targets_by_F, dim=2)
#     combined_input = torch.cat(all_inputs_by_F, dim=2)
#     print('Combined shapes for QR estimation:', combined_input.shape, combined_target.shape)  # sanity: [N_T, n, 5*T_test], [N_T, m, 5*T_test]
#     Q_hat, R_hat = estimate_QR(combined_input, combined_target)
#     Q = Q_hat
#     R = R_hat
#     sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)

#################################################################################################



#############################################################################
# Calculate MSE for each dataset with TRUE F (what would happen without EMKF)
print('\n=== MSE with TRUE F matrices ===')
true_mse_lin_sum = 0.0
for dataset_id in range(cycle):
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    # Use the TRUE F matrix for this dataset
    sys_model = SystemModel(true_F_for_this_dataset[0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                                        F=true_F_for_this_dataset,generate_f=False,
                                                        H=H_test_list, generate_h=False)
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                        F=true_F_for_this_dataset,generate_f=False,
                                        H=H_test_list, generate_h=False, init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone() for k in range(args.N_T)]
    true_mse_lin_sum += _mse_avg.item()
    print(f"Dataset {dataset_id + 1} - TRUE F MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with true F
average_true_F_mse_db = 10*torch.log10(torch.tensor(true_mse_lin_sum / cycle))

print(f"Average MSE with TRUE F matrices: {average_true_F_mse_db:.3f} dB")

#############################################################################
# Calculate MSE for each dataset with INITIAL GUESS
print('\n=== MSE with INITIAL GUESS F  ===')
mse_total_false = 0
for dataset_id in range(cycle):
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    # Use the initial guess F for ALL datasets
    sys_model = SystemModel(F_initial_gues_1, Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                                        F=F_initial_estimate,generate_f=False,
                                                        H=H_test_list, generate_h=False)
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                        F=F_initial_estimate,generate_f=False,
                                        H=H_test_list, generate_h=False, init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone()             for k in range(args.N_T)]
    mse_total_false +=_mse_avg.item()
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with initial guess
average_initial_guess_mse_db = 10*torch.log10(torch.tensor(mse_total_false/cycle))
print(f"Average MSE with INITIAL GUESS F: {average_initial_guess_mse_db:.3f} dB")

###############################################################
# Calculate MSE for each dataset with emkf F
print('\n=== MSE with EMKF  matrices ===')
mse_total =0
for dataset_id in range(cycle):
    print(f"\n--- EMKF Iteration {dataset_id + 1} ---")
    print(f"Using dataset {dataset_id + 1}")

    # Get data for this dataset
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    # print(f"True F for this dataset: {true_F_for_this_dataset}")
    # print(f"Initial F guess: {F_current_estimate[0]}")

    # Create system model for EMKF
    sys_model = SystemModel(F_current_estimate[0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)
    # Run EMKF with current estimate as initial guess
    print(f"Running EMKF on dataset {dataset_id + 1}...")


    if dataset_id == 0:
        F_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last = EMKF_F_analitic(sys_model, F_current_estimate,
            H.unsqueeze(0), Q, R, test_input, m1_0.reshape(-1), m2_0,test_target, max_it=3,generate_f=False, tol_likelihood=0.01, tol_params=0.025)
    else:
        F_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last = EMKF_F_analitic(sys_model, F_current_estimate,
            H.unsqueeze(0), Q, R, test_input, m1_0.reshape(-1), m2_0,test_target, max_it=3,generate_f=False, tol_likelihood=0.01, tol_params=0.025,
                                                                                init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    # keep initials 1-D: compute_A1/A2 require [m]; the KF/S_Test squeeze internally so [m] is fine
    x0_last = [x_last[k].reshape(-1).clone() for k in range(args.N_T)]
    p0_last = [p_last[k].clone() for k in range(args.N_T)]
    #F_matrices has N_T(amount of seq) list inside where each list has max it + initial guess F matrices one for each T
    # Update F estimate for next iteration (use the result from EMKF)
    F_current_estimate = [Fs_per_seq[-1].clone() for Fs_per_seq in F_matrices]
    print(f"EMKF result for dataset {dataset_id + 1}: {F_matrices[0]}")
    print(f"True F was: {true_F_for_this_dataset[0]}")
    mse_total+= mse_avg_T.item()
MSE_total_db = 10 * torch.log10(torch.tensor(mse_total / cycle))

print("\n=== EMKF iterations completed ===")
print(f"Final F estimate: {F_current_estimate[0]}")
print(f"Original F: {F}")
print(f"\nAverage MSE across all datasets from final F estimate: {MSE_total_db:.3f} dB")

#############################################################################
# Summary comparison
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE F (perfect):        {average_true_F_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (learned):    {MSE_total_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - MSE_total_db):.3f} dB")
print(f"Gap to perfect (TRUE F): {(MSE_total_db - average_true_F_mse_db):.3f} dB")


