import torch
import time

# =====================================================================================
# Self-contained 2-D "linear-F + nonlinear-h" EMKF experiment.
#
# This script was originally written against an OLD Lorenz params module (make_f, F, m=n=2)
# that no longer exists. The current codebase is wired for 3-D Lorenz, so we rebuild the 2-D
# setup locally and shim the few 3-D-hardcoded pieces (THIS PROCESS ONLY; no file on disk is
# changed):
#   * getJacobian in EKF / Extended_rts_smoother hardcodes view(-1, m=3) -> replace with a
#     dimension-agnostic version.
#   * S_Test_ext was refactored to a new signature; this script and E_EMKF_F_analitic need the
#     OLD one (S_Test_ext_old) -> alias it + rebind the name inside main_emkf_func.
#   * Extended GenerateBatch calls update_h(), which would overwrite our nonlinear h with a
#     linear h=H@x -> make update_h a no-op so the nonlinear observation survives data-gen.
# =====================================================================================

# ---- dimension-agnostic Jacobian (Lorenz getJacobian hardcodes view(-1, m=3)) ----
def _getJacobian_nd(x, g):
    y = x.reshape(-1)
    Jac = torch.autograd.functional.jacobian(g, y)
    return Jac.reshape(-1, y.shape[0])

import Smoothers.EKF as _ekf_mod
import Smoothers.Extended_RTS_Smoother as _erts_mod
_ekf_mod.getJacobian = _getJacobian_nd
_erts_mod.getJacobian = _getJacobian_nd

# ---- old-signature smoother (matches this script and E_EMKF_F_analitic's internal call) ----
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_old as S_Test_ext
import emkf.main_emkf_func as _emkf_mod
_emkf_mod.S_Test_ext = S_Test_ext

from Simulations.Extended_sysmdl import SystemModel, rotate_F
from Simulations.utils import DataGen
import Simulations.config as config
from emkf.main_emkf_func import E_EMKF_F_analitic

# ---- keep the nonlinear h during data generation (GenerateBatch would otherwise linearize it) ----
SystemModel.update_h = lambda self, H: None

# Repro
torch.manual_seed(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === latency helpers ===
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
latency = {}  # algorithm name -> total wall-clock seconds

# -------------------------------------------------------------------------------------
# 2-D model: linear dynamics f(x) = F @ x (F is what EMKF estimates), nonlinear observation
# h(x) = (range, bearing) = (||x||, atan2(x1, x0)).
# -------------------------------------------------------------------------------------
m = 2
n = 2
m1x_0 = torch.tensor([[0.5], [0.5]], device=DEVICE)
m2x_0 = torch.eye(m, device=DEVICE)
Q_structure = torch.eye(m, device=DEVICE)
R_structure = torch.eye(n, device=DEVICE)
iter_num = 3
def make_f(F):
    F = F.to(DEVICE)
    def f(x):
        return (F @ x.reshape(m, 1)).reshape(m)   # accept [m] or [m,1]; return 1-D [m]
    return f

def h_nonlinear(x, alpha=0.3):
    # 2x2 linear (Cartesian) part + alpha * polar (range, bearing) part.
    # Matches the OLD Simulations/Lorenz_Atractor/parameters_OLD.py:h_nonlinear.
    x = x.reshape(2, 1)
    x1, x2 = x[0, 0], x[1, 0]
    eps = 1e-6
    r     = torch.sqrt(x1 * x1 + x2 * x2 + eps)              # polar range
    theta = torch.atan2(x2, x1 + eps)                        # polar bearing
    H = torch.tensor([[1.0, 1.0],
                      [0.25, 1.0]], device=x.device, dtype=x.dtype)
    lin = (H @ x).reshape(2)                                 # linear Cartesian part
    return lin + alpha * torch.stack([r, theta])            # return 1-D [n]

# ---------------------
# Experiment settings
# ---------------------
args = config.general_settings()
args.N_T   = 100      # number of sequences for test
args.T     = 30      # generation length (train isn't used here)
args.T_test = 30     # test length
cycle      = 3       # number of datasets (rotations)

# Noise levels
q2 = 0.01
r2 = 0.001
Q  = (q2 * Q_structure).to(DEVICE)
R  = (r2 * R_structure).to(DEVICE)

# Base F guess (2x2). We'll build per-seq lists from this.
F0 = torch.tensor([[0.83, 0.20],
                   [0.20, 0.83]], dtype=Q.dtype, device=DEVICE)

# ---------------------
# Build per-dataset F lists by rotating per sequence
# ---------------------
F_matrices_for_datasets_d = []
F_test_list = [F0.clone() for _ in range(args.N_T)]  # same F for all sequences initially
for i in range(cycle + 1):
    F_matrices_for_datasets_d.append([f.clone() for f in F_test_list])
    # rotate PER-SEQUENCE for the next dataset
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=0.2, many=True, randomit=False)
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]  # length == cycle

# ---------------------
# Generate data per dataset
# ---------------------
dataFolderName_base = 'Simulations/Linear_canonical/paper/exp1_1/regular/'
all_inputs_by_F  = []
all_targets_by_F = []
all_F_matrices   = []

# Fixed H list only to satisfy GenerateBatch's indexing; it's unused for the observation
# because update_h is a no-op, so h stays nonlinear.
H_fixed_list = [torch.eye(n, device=DEVICE) for _ in range(args.N_T)]

x0_list = None  # will carry last x_T as next dataset initials
for dataset_id in range(1, cycle + 1):
    print(f"\n=== Generating Dataset {dataset_id} ===")
    F_list_this = F_matrices_for_datasets[dataset_id - 1]  # list of length N_T

    # System model with nonlinear h and f(x)=F@x (f is replaced per seq in S_Test_ext_old)
    sys_model = SystemModel(make_f(F_list_this[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1x_0, m2x_0)

    dataFileName   = f'snr_0{args.T_test}_dataset_{dataset_id}_NL.pt'
    dataFileName_F = f'snr_0_F_dataset_{dataset_id}_NL.pt'

    DataGen(args, sys_model,
            dataFolderName_base + dataFileName,
            dataFolderName_base + dataFileName_F,
            delta=1,
            randomInit_train=False, randomInit_cv=False, randomInit_test=False,
            randomLength=False, Test=True,
            F_gen=F_list_this, H_gen=H_fixed_list, x0_list=x0_list)

    # Load
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName_base + dataFileName, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(
        dataFolderName_base + dataFileName_F, weights_only=True, map_location=DEVICE)

    # Prepare initials for *next* dataset from ground-truth last state
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
_sync(); _t_true_start = time.perf_counter()
for dataset_id in range(cycle):
    test_input  = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]  # list length N_T

    sys_model = SystemModel(make_f(true_F_for_this_dataset[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1x_0, m2x_0)

    if dataset_id == 0:
        # S_Test_ext_old(SysModel, test_input, test_target, F_list, generate_f, randomInit, test_init, init_x_list, init_P_list)
        _mse_arr, _mse_avg, _mse_db, x_list, p_list, _ = S_Test_ext(sys_model, test_input, test_target,
                         true_F_for_this_dataset, False, False, None, None, None)
    else:
        _mse_arr, _mse_avg, _mse_db, x_list, p_list, _ = S_Test_ext(sys_model, test_input, test_target,
                         true_F_for_this_dataset, False, False, None, x0_last, p0_last)

    # propagate initials for NEXT dataset
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone()            for k in range(args.N_T)]

    true_mse_lin_sum += _mse_avg.item()
    print(f"Dataset {dataset_id + 1} - TRUE F MSE: {_mse_db.item():.3f} dB")
_sync(); latency['TRUE F (S_Test_ext)'] = time.perf_counter() - _t_true_start

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
_sync(); _t_init_start = time.perf_counter()
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
_sync(); latency['INITIAL GUESS (S_Test_ext)'] = time.perf_counter() - _t_init_start

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
_sync(); _t_emkf_start = time.perf_counter()
for dataset_id in range(cycle):
    print(f"\n--- EMKF on Dataset {dataset_id + 1} ---")

    test_input  = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    sys_model = SystemModel(make_f(F_current_estimate[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1x_0, m2x_0)

    # x_0 must be 1-D [m]: compute_A1/A2 do x_0.unsqueeze(0); the EKF/S_Test reshape internally.
    if dataset_id == 0:
        F_mats, mse_lin_last, x_last_list, p_last_list = E_EMKF_F_analitic(
            sys_model, F_current_estimate,
            h_nonlinear, Q, R,
            test_input, m1x_0.reshape(-1), m2x_0, test_target,
            max_it=iter_num, generate_f=False
        )
    else:
        F_mats, mse_lin_last, x_last_list, p_last_list = E_EMKF_F_analitic(
            sys_model, F_current_estimate,
            h_nonlinear, Q, R,
            test_input, m1x_0.reshape(-1), m2x_0, test_target,
            max_it=iter_num, generate_f=False,
            init_x_list=x0_last, init_P_list=p0_last
        )

    # Propagate initials (keep 1-D for compute_A1/A2)
    x0_last = [x_.reshape(-1).clone() for x_ in x_last_list]  # each [m]
    p0_last = [P_.clone() for P_ in p_last_list]              # each [m,m]

    # Update current F estimates for next dataset: final iterate per sequence
    F_current_estimate = [Fs_per_seq[-1].clone() for Fs_per_seq in F_mats]

    mse_total_em += float(mse_lin_last)
    print(f"Dataset {dataset_id + 1} - EMKF last-iter MSE: {10*torch.log10(torch.tensor(mse_lin_last)+1e-12):.3f} dB")
    print(f"Example EMKF F (seq 0):\n{F_mats[0][-1]}")
    print(f"Example TRUE F (seq 0):\n{true_F_for_this_dataset[0]}")
_sync(); latency['EMKF (analytic M-step)'] = time.perf_counter() - _t_emkf_start

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

# ---------------------
# LATENCY (average time per sequence)
# ---------------------
total_seqs = cycle * args.N_T
print('\n=== LATENCY (avg per sequence) ===')
print(f"{'Algorithm':<28}{'total (s)':>12}{'per-seq (ms)':>16}")
for name, secs in latency.items():
    print(f"{name:<28}{secs:>12.3f}{1000.0 * secs / total_seqs:>16.3f}")
