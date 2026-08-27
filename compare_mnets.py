"""
Rank every M-step network in one SNR bucket against the same test protocol used by
data_generate_exp_for_paper/F_exp/exp1_2/exp_1and2_testing.py.

Each candidate mnet is run through 3 sequential test datasets whose true F drifts by
a fixed 0.2 rad per dataset (F and x carried over between datasets), exactly as in the
paper test, and scored by the final-EM-iteration state MSE averaged over the 3 datasets.
TRUE F and INITIAL GUESS F are reported as the upper/lower bounds.

Architecture: KalmanNet_nn_with_F (forward, FC8 F-embedding)
              RTSNet_nn_with_F    (backward smoother, FC_F_bw F-embedding)

Run from anywhere:  python compare_mnets.py
Results are also streamed to <bucket>/mnet_comparison.csv as each model finishes, so a
crash partway through the sweep does not lose the models already scored.
"""
import os
import sys
import csv
import glob
import time
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from Simulations.Linear_sysmdl import SystemModel, rotate_F

# exp 1 = gauss, exp 2 = exponential. Must be set BEFORE any DataGen call.
import Simulations.Linear_sysmdl as _lsm
_lsm.NOISE_DIST = 'gauss'

from emkf.main_emkf_func_AI import EMKF_F
from Simulations.utils import DataLoader, DataGen, estimate_QR
import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure, m1_0, m2_0
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
from RTSNet.RTSNet_nn_with_F import RTSNetNN
from emkf.AI_M_step_old_F import DeltaF_MStepNet as MStepNet
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

# The saved .pt files were pickled as RTSNet.RTSNet_nn.RTSNetNN.
# Redirect that class lookup to RTSNet_nn_with_F.RTSNetNN so torch.load
# restores them with the correct F-embedding methods (FC_F_bw, FC8).
import RTSNet.RTSNet_nn as _rts_nn_module
_rts_nn_module.RTSNetNN = RTSNetNN

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda")
DTYPE  = torch.float32

# SNR bucket. The folder tag and r2 must always change as a PAIR, which is why r2 is
# looked up from the tag instead of being a second free knob.
R_BUCKET = 'r_0001'
R2_BY_BUCKET = {'r_10': 10, 'r_1': 1, 'r_01': 0.1, 'r_001': 0.01, 'r_0001': 0.001}

# The cached datasets under DATA_DIR are r2-specific, so each bucket gets its own
# sub-folder (below) and switching R_BUCKET cannot silently reuse another bucket's cache.
GENERATE_DATA = True

# Which checkpoints to sweep. INCLUDE_OLD also pulls in the archived EMKF/False/old/
# folder; MNET_GLOB narrows the sweep (e.g. 'm_step_e_q*.pt') when you only want a subset.
INCLUDE_OLD = False
MNET_GLOB   = '*.pt'

NUM_EM_ITERS = 3

EXP_DIR       = REPO_ROOT / 'RTSNet' / 'synthetic' / 'AI_M_step' / 'exp_1' / R_BUCKET
MNET_FOLDER   = EXP_DIR / 'EMKF' / 'False'
RTS_TRUE_PATH = str(EXP_DIR / 'True_F'  / 'RTSNET_true.pt')
RTS_FALSE_PATH= str(EXP_DIR / 'False_F' / 'RTSNET_false.pt')
# One sub-folder per SNR bucket. This used to write into exp1_1/regular/, a cache shared
# with the exp1_2 scripts, so a run at one r2 left behind files the next script would load
# as if they matched its own r2.
DATA_DIR      = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / 'exp1_1' / R_BUCKET
CSV_PATH      = EXP_DIR / 'mnet_comparison.csv'
os.makedirs(DATA_DIR, exist_ok=True)

for _p in (RTS_TRUE_PATH, RTS_FALSE_PATH):
    if not os.path.isfile(_p):
        raise FileNotFoundError(f"RTSNet checkpoint missing for bucket {R_BUCKET}: {_p}")

torch.cuda.empty_cache()

today  = datetime.today()
now    = datetime.now()
strTime= today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

# ─────────────────────────────────────────────
# SYSTEM PARAMETERS  (identical to paper test)
# ─────────────────────────────────────────────
args = config.general_settings()
args.N_T      = 100
args.T        = 30
args.T_test   = 30
torch.manual_seed(1)

cycles  = 3
q2 = 0.01
r2 = R2_BY_BUCKET[R_BUCKET]
print(f"bucket = {R_BUCKET}   q2 = {q2}   r2 = {r2}")
print(f"data dir = {DATA_DIR}")

Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)

F = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
H = torch.tensor([[1., 1.], [0.25, 1.]],     device=DEVICE, dtype=DTYPE)

m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
SystemModel.F_gen = False
sys_model.InitSequence(m1_0, m2_0)

# ─────────────────────────────────────────────
# BUILD F MATRICES FOR 3 DATASETS
# ─────────────────────────────────────────────
THETA_TEST = 0.2
F_test_list = [F.clone().to(DEVICE) for _ in range(args.N_T)]
H_test_list = [H.clone().to(DEVICE) for _ in range(args.N_T)]
F_matrices_for_datasets_d = []
for i in range(cycles + 1):
    F_matrices_for_datasets_d.append([f.clone() for f in F_test_list])
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=THETA_TEST, many=True, randomit=False)
# Drop the un-rotated base F: dataset k uses the F after k+1 rotations.
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]

# ─────────────────────────────────────────────
# GENERATE / LOAD DATA
# ─────────────────────────────────────────────
all_inputs_by_F  = []
all_targets_by_F = []
all_F_matrices   = []
all_H_matrices   = []
x0_last = None

for dataset_id in range(1, cycles + 1):
    print(f"\n=== Dataset {dataset_id} ===")
    F_current = F_matrices_for_datasets[dataset_id - 1]
    SystemModel.F_gen = False
    sys_model = SystemModel(F_current[0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    dataFilePath   = str(DATA_DIR / f'snr_0{args.T_test}_dataset_{dataset_id}.pt')
    dataFilePath_F = str(DATA_DIR / f'snr_0_F_dataset_{dataset_id}.pt')

    if GENERATE_DATA:
        print(f"Generating data for dataset {dataset_id} at r2={r2}...")
        DataGen(args, sys_model, dataFilePath, dataFilePath_F,
                delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
                randomLength=False, Test=True, F_gen=F_current, H_gen=H_test_list, x0_list=x0_last)
    else:
        print(f"Loading existing data for dataset {dataset_id} "
              f"(WARNING: cached data is r2-specific -- make sure it was generated at r2={r2})")

    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFilePath, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFilePath_F, map_location=DEVICE)

    x_last = test_target[:, :, -1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]

    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)
    all_H_matrices.append(H_test_list)

# ─────────────────────────────────────────────
# RTSNet PIPELINE  (shared across all tests)
# ─────────────────────────────────────────────
sys_model_base = SystemModel(F_matrices_for_datasets[0][0], Q, H, R, args.T, args.T_test)
sys_model_base.InitSequence(m1_0, m2_0)

RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_base, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model_base)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

F_initial_guess = [torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
                   for _ in range(args.N_T)]

# ─────────────────────────────────────────────
# BASELINE: TRUE F
# ─────────────────────────────────────────────
print('\n=== Baseline: TRUE F ===')
true_mse_lin_sum = 0.0
xT0_last = pT0_last = None
t_start_true_f = time.perf_counter()
for dataset_id in range(cycles):
    test_input  = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F      = F_matrices_for_datasets[dataset_id][0]

    sys_model_true = SystemModel(true_F, Q, H, R, args.T, args.T_test)
    sys_model_true.InitSequence(m1_0, m2_0)
    sys_model_true.F_test  = F_matrices_for_datasets[dataset_id]
    sys_model_true.H_test  = all_H_matrices

    kw = dict(load_model_path=RTS_TRUE_PATH, generate_f=False)
    if dataset_id > 0:
        kw.update(init_x_list=xT0_last, init_P_list=pT0_last)
    results = RTSNet_Pipeline.NNTest_no_p(sys_model_true, test_input, test_target, **kw)

    mse_db  = results[2]
    true_mse_lin_sum += float(results[1])
    print(f"  Dataset {dataset_id+1} TRUE F MSE: {mse_db:.3f} dB")

    x_last  = results[3][:, :, -1].clone()
    xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
    pT0_last = sys_model_true.m2x_0.clone().detach()
t_end_true_f = time.perf_counter()

avg_true_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"  Average TRUE F MSE: {avg_true_db:.3f} dB")

# ─────────────────────────────────────────────
# BASELINE: INITIAL GUESS F
# ─────────────────────────────────────────────
print('\n=== Baseline: INITIAL GUESS F ===')
init_mse_lin_sum = 0.0
xF0_last = pF0_last = None
t_start_init_f = time.perf_counter()
for dataset_id in range(cycles):
    test_input  = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    sys_model_init = SystemModel(F_initial_guess[0], Q, H, R, args.T, args.T_test)
    sys_model_init.InitSequence(m1_0, m2_0)
    sys_model_init.F_test = F_initial_guess
    sys_model_init.H_test = all_H_matrices

    kw = dict(load_model_path=RTS_FALSE_PATH, generate_f=False)
    if dataset_id > 0:
        kw.update(init_x_list=xF0_last, init_P_list=pF0_last)
    results = RTSNet_Pipeline.NNTest_no_p(sys_model_init, test_input, test_target, **kw)

    mse_db = results[2]
    init_mse_lin_sum += float(results[1])
    print(f"  Dataset {dataset_id+1} INIT GUESS MSE: {mse_db:.3f} dB")

    x_last  = results[3][:, :, -1].clone()
    xF0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
    pF0_last = sys_model_init.m2x_0.clone().detach()
t_end_init_f = time.perf_counter()

avg_init_db = 10 * torch.log10(torch.tensor(init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"  Average INIT GUESS MSE: {avg_init_db:.3f} dB")

# ─────────────────────────────────────────────
# DISCOVER ALL MNET MODELS
# ─────────────────────────────────────────────
mnet_files = sorted(glob.glob(str(MNET_FOLDER / MNET_GLOB)))
if INCLUDE_OLD:
    mnet_files += sorted(glob.glob(str(MNET_FOLDER / 'old' / MNET_GLOB)))

if not mnet_files:
    print(f"\nNo .pt files found in {MNET_FOLDER}")
else:
    print(f"\nFound {len(mnet_files)} MNet model(s) to compare in {MNET_FOLDER}:")
    for p in mnet_files:
        print(f"  {os.path.relpath(p, MNET_FOLDER)}")

# ─────────────────────────────────────────────
# COMPARE EACH MNET
# ─────────────────────────────────────────────
results_table = []  # list of (name, avg_mse_db, total_s, ms_per_seq)

csv_file = open(CSV_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['model', 'avg_mse_db', 'ds1_db', 'ds2_db', 'ds3_db',
                     'total_s', 'ms_per_seq', 'error'])
csv_file.flush()

for idx, mnet_path in enumerate(mnet_files, start=1):
    model_name = os.path.relpath(mnet_path, MNET_FOLDER)
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(mnet_files)}] Testing MNet: {model_name}")
    print('='*60, flush=True)

    emkf_mse_lin_sum     = 0.0
    per_dataset_db       = []
    current_F_estimate   = None
    x0_em_last = p0_em_last = None

    try:
        t_start_mnet = time.perf_counter()
        for dataset_id in range(cycles):
            test_input  = all_inputs_by_F[dataset_id]
            test_target = all_targets_by_F[dataset_id]
            true_F_list = F_matrices_for_datasets[dataset_id]

            if dataset_id == 0:
                current_F_estimate = F_initial_guess
            # else: carry over the F learned on the previous dataset (set below)

            sys_model_ai = SystemModel(current_F_estimate[0], Q, H, R, args.T, args.T_test)
            sys_model_ai.InitSequence(m1_0, m2_0)
            sys_model_ai.F_test      = current_F_estimate
            sys_model_ai.F_test_TRUE = true_F_list
            sys_model_ai.H_test      = all_H_matrices

            kw = dict(
                destination_path_RTS=RTS_FALSE_PATH,
                destination_path_M  =mnet_path,
                num_em_iters        =NUM_EM_ITERS,
                generate_f          =False,
            )
            if dataset_id > 0:
                kw.update(init_x_list=x0_em_last, init_P_list=p0_em_last)

            test_losses, test_f_losses, final_F_list, last_x_list = \
                RTSNet_Pipeline.test_mstep_net(sys_model_ai, test_input, test_target, **kw)

            final_loss = float(test_losses[-1])
            emkf_mse_lin_sum  += final_loss
            current_F_estimate = final_F_list

            # Chain the EMKF's OWN smoothed last state into the next dataset -- same as
            # exp_1and2_testing.py. Using test_target here would be an oracle warm-start
            # and would rank the mnets under conditions the paper test never runs.
            p0_em_last = sys_model_ai.m2x_0.clone().detach()
            x0_em_last = last_x_list

            loss_db = 10 * torch.log10(torch.tensor(final_loss, device=DEVICE, dtype=DTYPE))
            per_dataset_db.append(float(loss_db))
            print(f"  Dataset {dataset_id+1}: final loss = {loss_db:.3f} dB", flush=True)

        t_end_mnet = time.perf_counter()
        avg_mse_db = 10 * torch.log10(torch.tensor(emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
        _lat_total = t_end_mnet - t_start_mnet
        _lat_per_seq = _lat_total / (cycles * args.N_T) * 1000
        results_table.append((model_name, float(avg_mse_db), _lat_total, _lat_per_seq))
        print(f"  >>> Average over {cycles} datasets: {avg_mse_db:.3f} dB  |  latency: {_lat_total:.2f}s total, {_lat_per_seq:.1f} ms/seq")
        csv_writer.writerow([model_name, f"{float(avg_mse_db):.4f}"]
                            + [f"{d:.4f}" for d in per_dataset_db]
                            + [''] * (cycles - len(per_dataset_db))
                            + [f"{_lat_total:.2f}", f"{_lat_per_seq:.1f}", ''])

    except Exception as e:
        # Several checkpoints in this folder were trained with variant M-net
        # architectures (different z_in layout / no A2 / new entries) that
        # test_mstep_net cannot drive. Record and move on instead of aborting the sweep.
        print(f"  ERROR with {model_name}: {type(e).__name__}: {e}")
        results_table.append((model_name, float('inf'), float('inf'), float('inf')))
        csv_writer.writerow([model_name, '', '', '', '', '', '', f"{type(e).__name__}: {e}"])

    csv_file.flush()

csv_file.close()

# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
results_table.sort(key=lambda x: x[1])

_N_seqs = cycles * args.N_T
_lat_true_f_total = t_end_true_f - t_start_true_f
_lat_init_f_total = t_end_init_f - t_start_init_f

print('\n' + '='*80)
print(f'SUMMARY COMPARISON -- exp_1 / {R_BUCKET} (r2={r2})   (lower dB = better)')
print('='*80)
print(f"{'Model':<50} {'Avg MSE (dB)':>12} {'Total (s)':>10} {'ms/seq':>8}")
print('-'*82)
print(f"{'TRUE F (oracle upper bound)':<50} {float(avg_true_db):>12.3f} {_lat_true_f_total:>10.2f} {_lat_true_f_total / _N_seqs * 1000:>8.1f}")
print('-'*82)
for name, db, lat_total, lat_per_seq in results_table:
    marker = ' <-- BEST' if name == results_table[0][0] else ''
    if db == float('inf'):
        print(f"{name:<50} {'err':>12} {'err':>10} {'err':>8}")
    else:
        print(f"{name:<50} {db:>12.3f} {lat_total:>10.2f} {lat_per_seq:>8.1f}{marker}")
print('-'*82)
print(f"{'INITIAL GUESS (no EMKF)':<50} {float(avg_init_db):>12.3f} {_lat_init_f_total:>10.2f} {_lat_init_f_total / _N_seqs * 1000:>8.1f}")
print('='*80)
print(f"Full results written to {CSV_PATH}")

if results_table and results_table[0][1] != float('inf'):
    best_name, best_db, best_lat, best_per_seq = results_table[0]
    print(f"\nBest MNet:  {best_name}")
    print(f"  full path:    {MNET_FOLDER / best_name}")
    print(f"  vs TRUE F:    {best_db - float(avg_true_db):+.3f} dB gap")
    print(f"  vs INIT GUESS:{float(avg_init_db) - best_db:+.3f} dB improvement")
