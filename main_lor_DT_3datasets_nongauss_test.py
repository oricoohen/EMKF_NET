"""
NON-GAUSSIAN test experiment (contaminated-Gaussian measurement noise).

Add-on copy of main_lor_DT_3datasets_test.py. Differences:
  1. Data-generation SystemModel gets the same non-Gaussian measurement-noise law
     (set_noise_model) used at training time.
  2. Loads the nets trained by main_lor_DT_3datasets_nongauss_train.py.
  3. ADDS analytic baselines evaluated BEFORE the neural methods:
       ERTS TRUE H (analytic oracle) | ERTS FALSE H (analytic, no adapt) |
       classical analytic EMKF (EMKF_H_analitic_f_nonlinear, Gaussian-assuming).
  4. Prints a 7-method dB table:
       -- analytic:  ERTS TRUE H | ERTS FALSE H | classical EMKF
       -- neural:    EMKalmanNet (AI EMKF) | RTSNet TRUE H | RTSNet FALSE H | BiGRU

Keep NOISE_TYPE / NOISE_PARAMS identical to the training script.
"""
import torch
from datetime import datetime
import os
import time
import random

import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
    f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure, H_design
from Simulations.Linear_sysmdl import rotate_F, rotate_H
from Simulations.utils import DataGen

from RTSNet.RTSNet_nn import RTSNetNN
from Baselines.BiGRU_smoother import test_bigru_smoother
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from emkf.main_emkf_func import EMKF_H_analitic_f_nonlinear
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_H

print("Pipeline Start - NON-GAUSSIAN TEST")

DEVICE = torch.device("cuda")
DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)
H_Rotate = H_Rotate.to(device)
H_Rotate_inv = H_Rotate_inv.to(device)
Q_structure = Q_structure.to(device)
R_structure = R_structure.to(device)
H_design = H_design.to(device)

today = datetime.today(); now = datetime.now()
strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

# ============================================================================ #
#  NOISE MODEL -- MUST MATCH THE TRAINING SCRIPT                                 #
# ============================================================================ #
# Pick ONE noise law. Params from NOISE_PRESETS below. MUST match the training script.
#   'gaussian' | 'contaminated' | 'student_t' | 'laplace' | 'exponential'
NOISE_TYPE = 'contaminated'
NOISE_PRESETS = {
    'gaussian':     {},
    'contaminated': {'eps': 0.05, 'kappa': 3.0},   # ~1.4x var (easy); 0.05/5->2.2x; 0.1/10->11x (harsh)
    'student_t':    {'nu': 4.0},                    # nu=3 heavier, nu=8 milder
    'laplace':      {},
    'exponential':  {},
}
NOISE_PARAMS = NOISE_PRESETS[NOISE_TYPE if NOISE_TYPE else 'gaussian']

####################
### Settings ###
####################
args = config.general_settings()
args.N_T = 100
args.T = 30
args.T_test = 30

GENERATE_DATA = False
num_iters = 2       # EM iterations (EMKalmanNet and classical EMKF)
cycles = 5          # test over more H drift than trained (generalization)
max_iter = 3
r2 = torch.tensor([1], device=device)
vdB = -20
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
Q = q2[0] * Q_structure
R = r2[0] * R_structure
print('q2 is:', q2, ' r2 is:', r2)

# ---- load paths (match the nongauss training script) ----
_base = f'RTSNet/lorenz_nongauss_{NOISE_TYPE}/3datasets/30'
destination_path_rtsnet_full          = f'{_base}/RTSNet_full.pt'
destination_path_rtsnet_partial       = f'{_base}/RTSNet_partial.pt'
destination_path_M_joint              = f'{_base}/M_step_net_joint.pt'
destination_path_rtsnet_partial_joint = f'{_base}/RTSNet_partial_joint.pt'
bigru_path                            = f'{_base}/bigru_smoother.pt'

# ============================================================================ #
# Build drifting H matrices (same protocol as the Gaussian test)                #
# ============================================================================ #
initial_guess_H = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]
H_test_list = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]
H_matrices_for_datasets_d = []
for i in range(cycles + 1):
    H_matrices_for_datasets_d.append([hh.clone() for hh in H_test_list])
    H_test_list = rotate_H(H_matrices_for_datasets_d[i], theta=0.2, many=True, randomit=False)
H_matrices_for_datasets = H_matrices_for_datasets_d[1:]

all_inputs_by_H, all_targets_by_H, all_H_matrices = [], [], []
x0_last = None

print("\n" + "=" * 80)
print(f"GENERATING {cycles} TEST DATASETS WITH {NOISE_TYPE.upper()} MEASUREMENT NOISE")
print("=" * 80)

for dataset_id in range(1, cycles + 1):
    print(f"\n=== Generating Test Dataset {dataset_id} ===")
    H_current = H_matrices_for_datasets[dataset_id - 1]

    sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
    sys_model.InitSequence(m1x_0, m2x_0)
    # >>> non-Gaussian measurement noise (the only new line vs Gaussian test) <<<
    sys_model.set_noise_model(meas_type=NOISE_TYPE, meas_params=NOISE_PARAMS)

    r2_val = int(r2[0].item())
    dataFolderName = f'Simulations/Lorenz_Atractor/data/nongauss_{NOISE_TYPE}_r2_{r2_val}_T{args.T_test}_c{cycles}/'
    dataFileName = f'dataset_{dataset_id}.pt'
    dataFileName_H = f'dataset_{dataset_id}_H.pt'
    dataFileName_F = f'dataset_{dataset_id}_F.pt'
    os.makedirs(dataFolderName, exist_ok=True)

    if GENERATE_DATA:
        DataGen(args, sys_model,
                dataFolderName + dataFileName,
                dataFolderName + dataFileName_F,
                fileName_H=dataFolderName + dataFileName_H,
                delta=1,
                randomInit_train=False, randomInit_cv=False, randomInit_test=False,
                randomLength=False, Test=True, F_gen=False, H_gen=H_current,
                x0_list=x0_last, H_init=H_current)

    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(dataFolderName + dataFileName_H, map_location=DEVICE)

    x_last = test_target[:, :, -1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]

    all_inputs_by_H.append(test_input)
    all_targets_by_H.append(test_target)
    all_H_matrices.append(H_test_mat_list)
    print(f"Dataset {dataset_id}: test_input {test_input.shape}")

# ---- keep the H-lists sized to the ACTUAL loaded test set --------------------
# (guards the common case GENERATE_DATA=False + args.N_T changed: the on-disk
#  data may have a different number of sequences than args.N_T.)
N_actual = all_inputs_by_H[0].shape[0]
if N_actual != len(initial_guess_H):
    print(f"[fix] Loaded test set has {N_actual} sequences but H-lists were built for "
          f"{len(initial_guess_H)} (args.N_T={args.N_T}). Resizing H-lists to {N_actual}.")
    initial_guess_H = [H_Rotate.clone().to(DEVICE) for _ in range(N_actual)]
    H_test_list = [H_Rotate.clone().to(DEVICE) for _ in range(N_actual)]
    H_matrices_for_datasets_d = []
    for i in range(cycles + 1):
        H_matrices_for_datasets_d.append([hh.clone() for hh in H_test_list])
        H_test_list = rotate_H(H_matrices_for_datasets_d[i], theta=0.2, many=True, randomit=False)
    H_matrices_for_datasets = H_matrices_for_datasets_d[1:]

sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
sys_model.InitSequence(m1x_0, m2x_0)

# ============================================================================ #
# 1) BiGRU                                                                       #
# ============================================================================ #
print("\n=== BiGRU ===")
bigru_mse_lin_sum = 0.0
all_bigru_x = []
t0 = time.perf_counter()
for d in range(cycles):
    mse_bigru, mse_bigru_db, x_bigru = test_bigru_smoother(
        test_input=all_inputs_by_H[d], test_target=all_targets_by_H[d],
        load_path=bigru_path, device=device)
    all_bigru_x.append(x_bigru.detach().clone())
    bigru_mse_lin_sum += float(mse_bigru)
    print(f"Dataset {d + 1} - BiGRU MSE: {mse_bigru_db:.3f} dB")
t_bigru = time.perf_counter() - t0
bigru_mse_db = 10 * torch.log10(torch.tensor(bigru_mse_lin_sum / cycles, device=DEVICE))

# ############################################################################ #
# ANALYTIC BASELINES (evaluated BEFORE the neural methods):                     #
#   ERTS TRUE H | ERTS FALSE H | classical EMKF                                 #
# ############################################################################ #

# ============================================================================ #
# 2) ERTS TRUE H  (analytic Extended-RTS smoother, correct H -> analytic oracle)#
# ============================================================================ #
print("\n=== ERTS TRUE H (analytic smoother) ===")
erts_true_mse_lin_sum = 0.0
all_erts_true_x = []
xEt_last = None
t0 = time.perf_counter()
for d in range(cycles):
    sm = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
    sm.InitSequence(m1x_0, m2x_0)
    res = S_Test_ext_H(
        sm, all_inputs_by_H[d], all_targets_by_H[d],
        H_list=H_matrices_for_datasets[d], generate_h=False,
        init_x_list=xEt_last, init_P_list=None)
    all_erts_true_x.append(res[3].detach().clone())
    erts_true_mse_lin_sum += float(res[1])
    print(f"Dataset {d + 1} - ERTS TRUE H MSE: {res[2]:.3f} dB")
    x_last = res[3][:, :, -1].clone()
    xEt_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
t_erts_true = time.perf_counter() - t0
erts_true_mse_db = 10 * torch.log10(torch.tensor(erts_true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

# ============================================================================ #
# 3) ERTS FALSE H  (analytic Extended-RTS smoother, wrong/init H, no adaptation)#
# ============================================================================ #
print("\n=== ERTS FALSE (init-guess) H (analytic smoother) ===")
erts_false_mse_lin_sum = 0.0
all_erts_false_x = []
xEf_last = None
t0 = time.perf_counter()
for d in range(cycles):
    sm = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
    sm.InitSequence(m1x_0, m2x_0)
    res = S_Test_ext_H(
        sm, all_inputs_by_H[d], all_targets_by_H[d],
        H_list=initial_guess_H, generate_h=False,
        init_x_list=xEf_last, init_P_list=None)
    all_erts_false_x.append(res[3].detach().clone())
    erts_false_mse_lin_sum += float(res[1])
    print(f"Dataset {d + 1} - ERTS FALSE H MSE: {res[2]:.3f} dB")
    x_last = res[3][:, :, -1].clone()
    xEf_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
t_erts_false = time.perf_counter() - t0
erts_false_mse_db = 10 * torch.log10(torch.tensor(erts_false_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

# ============================================================================ #
# 4) CLASSICAL analytic EMKF (Gaussian-assuming baseline -> expected to fail)    #
# ============================================================================ #
print("\n=== Classical analytic EMKF (Gaussian-assuming) ===")
cls_mse_lin_sum = 0.0
all_emkf_cls_x = []
x0_cls_last = None
current_H_cls = None
t0 = time.perf_counter()
for d in range(cycles):
    H0 = ([Hm.clone() for Hm in initial_guess_H]) if d == 0 else current_H_cls

    sys_model_cls = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
    sys_model_cls.InitSequence(m1x_0, m2x_0)

    if d == 0:
        init_x, init_P = None, None
    else:
        P0 = sys_model_cls.m2x_0.clone()          # [m,m] prior covariance
        init_x = x0_cls_last
        init_P = [P0.clone() for _ in range(len(x0_cls_last))]

    (H_est_seq, likelihoods, iters_list, final_mean_mse,
     last_x_list, last_P_list, full_x_list) = EMKF_H_analitic_f_nonlinear(
        sys_model_cls, H0, all_inputs_by_H[d], m1x_0, m2x_0, all_targets_by_H[d],
        max_it=max_iter, generate_h=False, init_x_list=init_x, init_P_list=init_P)

    cls_mse_lin_sum += float(final_mean_mse)
    current_H_cls = [H_est_seq[j][-1].clone() for j in range(len(H_est_seq))]
    x0_cls_last = [last_x_list[j].clone() for j in range(len(last_x_list))]
    all_emkf_cls_x.append(torch.stack(full_x_list, dim=0).detach().clone())  # [N,m,T]
    print(f"Dataset {d + 1} - classical EMKF MSE: "
          f"{10 * torch.log10(torch.tensor(float(final_mean_mse)) + 1e-12):.3f} dB")
t_cls = time.perf_counter() - t0
cls_mse_db = 10 * torch.log10(torch.tensor(cls_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

# ############################################################################ #
# NEURAL METHODS:  EMKalmanNet | RTSNet TRUE H | RTSNet FALSE H                  #
# ############################################################################ #

# Pipeline for the RTSNet-based evaluations
RTSNet_model = RTSNetNN(); RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

# ============================================================================ #
# 5) EMKalmanNet (AI EMKF): joint RTSNet + M-net, sequential H learning          #
# ============================================================================ #
print("\n=== EMKalmanNet (AI EMKF) ===")
emkf_mse_lin_sum = 0.0
all_emkf_x = []
x0_em_last = None
current_H_estimate_prev = None
t0 = time.perf_counter()
for d in range(cycles):
    current_H_estimate = initial_guess_H if d == 0 else current_H_estimate_prev
    sys_model_ai = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
    sys_model_ai.InitSequence(m1x_0, m2x_0)
    sys_model_ai.H_test = current_H_estimate
    sys_model_ai.H_test_TRUE = H_matrices_for_datasets[d]

    test_losses, test_h_losses, final_H_list, last_x_list, list_x = RTSNet_Pipeline.test_H_mstep_net(
        sys_model_ai, all_inputs_by_H[d], all_targets_by_H[d],
        destination_path_RTS=destination_path_rtsnet_partial_joint,
        destination_path_M=destination_path_M_joint,
        num_em_iters=num_iters, generate_h=False,
        init_x_list=(None if d == 0 else x0_em_last), init_P_list=None)
    all_emkf_x.append(list_x.detach().clone())
    emkf_mse_lin_sum += float(test_losses[-1])
    current_H_estimate_prev = final_H_list
    x0_em_last = [last_x_list[j].clone() for j in range(len(last_x_list))]
t_emkf = time.perf_counter() - t0
emkf_mse_db = 10 * torch.log10(torch.tensor(emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

# ============================================================================ #
# 6) RTSNet with TRUE H (upper bound)                                            #
# ============================================================================ #
print("\n=== RTSNet TRUE H ===")
true_mse_lin_sum = 0.0
all_rts_trueH_x = []
xT0_last = None
t0 = time.perf_counter()
for d in range(cycles):
    sys_model_true = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
    sys_model_true.InitSequence(m1x_0, m2x_0)
    sys_model_true.H_test = H_matrices_for_datasets[d]
    results = RTSNet_Pipeline.NNTest(
        sys_model_true, all_inputs_by_H[d], all_targets_by_H[d],
        destination_path_rtsnet_full, generate_h=False, generate_f=None,
        init_x_list=xT0_last, init_P_list=None)
    all_rts_trueH_x.append(results[3].detach().clone())
    true_mse_lin_sum += float(results[1])
    print(f"Dataset {d + 1} - TRUE H MSE: {results[2]:.3f} dB")
    x_last = results[3][:, :, -1].clone()
    xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
t_true = time.perf_counter() - t0
true_mse_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

# ============================================================================ #
# 7) RTSNet with FALSE / initial-guess H (no adaptation)                         #
# ============================================================================ #
print("\n=== RTSNet FALSE (init-guess) H ===")
init_mse_lin_sum = 0.0
all_initH_x = []
xH0_last = None
t0 = time.perf_counter()
for d in range(cycles):
    sys_model_init = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
    sys_model_init.InitSequence(m1x_0, m2x_0)
    sys_model_init.H_test = initial_guess_H
    results = RTSNet_Pipeline.NNTest(
        sys_model_init, all_inputs_by_H[d], all_targets_by_H[d],
        destination_path_rtsnet_partial, generate_h=False, generate_f=None,
        init_x_list=xH0_last, init_P_list=None)
    all_initH_x.append(results[3].detach().clone())
    init_mse_lin_sum += float(results[1])
    print(f"Dataset {d + 1} - FALSE H MSE: {results[2]:.3f} dB")
    x_last = results[3][:, :, -1].clone()
    xH0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
t_init = time.perf_counter() - t0
init_mse_db = 10 * torch.log10(torch.tensor(init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

# ============================================================================ #
# SUMMARY                                                                        #
# ============================================================================ #
print("\n" + "=" * 60)
print(f"SUMMARY  (noise = {NOISE_TYPE} {NOISE_PARAMS}, averaged over {cycles} datasets)")
print("=" * 60)
print("-- analytic baselines --")
print(f"{'ERTS TRUE H (analytic oracle)':<32} {erts_true_mse_db:8.3f} dB")
print(f"{'ERTS FALSE H (analytic, no adapt)':<32} {erts_false_mse_db:8.3f} dB")
print(f"{'Classical EMKF (Gaussian)':<32} {cls_mse_db:8.3f} dB")
print("-- neural methods --")
print(f"{'EMKalmanNet (AI EMKF, ours)':<32} {emkf_mse_db:8.3f} dB")
print(f"{'RTSNet TRUE H (upper bound)':<32} {true_mse_db:8.3f} dB")
print(f"{'RTSNet FALSE H (no adapt)':<32} {init_mse_db:8.3f} dB")
print(f"{'BiGRU (model-free)':<32} {bigru_mse_db:8.3f} dB")
print("-" * 60)
print(f"EMKalmanNet gap to RTSNet TRUE H: {(emkf_mse_db - true_mse_db):.3f} dB")
print(f"EMKalmanNet gain vs BiGRU:        {(bigru_mse_db - emkf_mse_db):.3f} dB")
print(f"EMKalmanNet gain vs classical EMKF:{(cls_mse_db - emkf_mse_db):.3f} dB")
print(f"EMKalmanNet gain vs ERTS FALSE H: {(erts_false_mse_db - emkf_mse_db):.3f} dB")

_N = cycles * args.N_T
print(f"\n{'Algorithm':<28}{'Total (s)':>10}{'ms/seq':>9}")
for name, tt in [('BiGRU', t_bigru), ('ERTS TRUE H', t_erts_true),
                 ('ERTS FALSE H', t_erts_false), ('Classical EMKF', t_cls),
                 ('EMKalmanNet (AI EMKF)', t_emkf), ('RTSNet TRUE H', t_true),
                 ('RTSNet FALSE H', t_init)]:
    print(f"{name:<28}{tt:>10.2f}{tt / _N * 1000:>9.1f}")

# ---- save x-estimates for the shared plot script ----
sample_idx = 0
T_len = all_targets_by_H[0].shape[2]
x_true_glued  = torch.cat([all_targets_by_H[d][sample_idx] for d in range(cycles)], dim=1).detach().cpu()
x_rtsnet_false = torch.cat([all_initH_x[d][sample_idx]     for d in range(cycles)], dim=1).detach().cpu()
x_emkalmanet   = torch.cat([all_emkf_x[d][sample_idx]      for d in range(cycles)], dim=1).detach().cpu()
x_bigru_glued  = torch.cat([all_bigru_x[d][sample_idx]     for d in range(cycles)], dim=1).detach().cpu()
x_emkf_cls     = torch.cat([all_emkf_cls_x[d][sample_idx]  for d in range(cycles)], dim=1).detach().cpu()
x_erts_true    = torch.cat([all_erts_true_x[d][sample_idx] for d in range(cycles)], dim=1).detach().cpu()
x_erts_false   = torch.cat([all_erts_false_x[d][sample_idx] for d in range(cycles)], dim=1).detach().cpu()

save_dir = "x_estimates"
os.makedirs(save_dir, exist_ok=True)
torch.save({
    "sample_idx": sample_idx, "cycles": cycles, "T_len": T_len,
    "noise_type": NOISE_TYPE, "noise_params": NOISE_PARAMS,
    "algos": {
        "true x":         x_true_glued,
        "erts true":      x_erts_true,
        "erts false":     x_erts_false,
        "emkf classical": x_emkf_cls,
        "emkalmanet":     x_emkalmanet,
        "rtsnet false":   x_rtsnet_false,
        "bgru":           x_bigru_glued,
    },
}, os.path.join(save_dir, "neural_x_nongauss.pt"))
print(f"\nSaved x-estimates -> {os.path.join(save_dir, 'neural_x_nongauss.pt')}")
