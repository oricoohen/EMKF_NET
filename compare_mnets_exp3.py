"""
Compare all MNet models across every r-folder in RTSNet/AI_M_step/exp_3/
based on M_network_AI_emkf_testing_non_linear_h_paper.py

Non-linear observation: h_nonlinear (Extended system model, EKF/ERTS).
For each r-folder: TRUE-F baseline, INIT-GUESS baseline, then all MNets ranked.

Architecture: RTSNet_nn_with_F (F-embedding, same as models were saved with)
"""
import os
import glob
import time
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Extended_sysmdl import SystemModel, rotate_F
from Simulations.Lorenz_Atractor.parameters_OLD import (
    m1x_0 as m1_0, m2x_0 as m2_0,
    m, n, h_nonlinear, Q_structure, R_structure, make_f
)
from Simulations.utils import DataLoader, DataGen
import Simulations.config as config

from RTSNet.RTSNet_nn_with_F import RTSNetNN
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline


# Redirect RTSNet_nn class lookup so old pickled models load with F-embedding methods
import RTSNet.RTSNet_nn as _rts_nn_module
_rts_nn_module.RTSNetNN = RTSNetNN

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda")
DTYPE  = torch.float32
GENERATE_DATA = True  # use existing datasets generated during training

EXP_ROOT     = 'RTSNet/AI_M_step/exp_3/'
NUM_EM_ITERS = 3

torch.cuda.empty_cache()
torch.manual_seed(1)

today   = datetime.today()
now     = datetime.now()
strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

# ─────────────────────────────────────────────
# SYSTEM PARAMETERS  (match paper test)
# ─────────────────────────────────────────────
args = config.general_settings()
args.N_T    = 100
args.T      = 30
args.T_test = 30

cycles = 3
q2     = 0.01   # fixed; r2 varies per r-folder

F_base = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)

m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

F_initial_guess = [F_base.clone() for _ in range(args.N_T)]

# ─────────────────────────────────────────────
# BUILD F ROTATION MATRICES (same for all r)
# ─────────────────────────────────────────────
F_test_list = [F_base.clone() for _ in range(args.N_T)]
F_matrices_for_datasets_d = []
a = 1
for i in range(cycles + 1):
    F_matrices_for_datasets_d.append([(f * a).clone() for f in F_test_list])
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=0.2,
                           many=True, randomit=False)
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]

# ─────────────────────────────────────────────
# FOLDER → r2 MAPPING
# ─────────────────────────────────────────────
def r2_from_folder(folder_name):
    tag = folder_name.replace('r_', '')
    if tag.startswith('0'):
        decimals = len(tag)
        return 10.0 ** (-decimals + 1)
    else:
        return float(tag)

# ─────────────────────────────────────────────
# DISCOVER R-FOLDERS
# ─────────────────────────────────────────────
r_folders = ['r_1', 'r_10']

print(f"\nFound {len(r_folders)} r-folders in {EXP_ROOT}:")
for rf in r_folders:
    r2 = r2_from_folder(rf)
    n_mnets = len(glob.glob(os.path.join(EXP_ROOT, rf, 'EMKF/False/*.pt')))
    print(f"  {rf:10s}  r2={r2:<8g}  MNets={n_mnets}")

# ─────────────────────────────────────────────
# MAIN LOOP OVER R-FOLDERS
# ─────────────────────────────────────────────
all_summaries = []

for r_folder in r_folders:
    r2   = r2_from_folder(r_folder)
    base = os.path.join(EXP_ROOT, r_folder)
    RTS_TRUE_PATH  = os.path.join(base, 'True_F/best-rts_true.pt')
    RTS_FALSE_PATH = os.path.join(base, 'False_F/best-rts_false.pt')
    MNET_FOLDER    = os.path.join(base, 'EMKF/False/')

    mnet_files = sorted(glob.glob(os.path.join(MNET_FOLDER, '*.pt')))

    print(f"\n{'='*70}")
    print(f"  r-folder: {r_folder}   r2={r2}   ({len(mnet_files)} MNets)")
    print(f"{'='*70}")

    if not os.path.exists(RTS_TRUE_PATH):
        print(f"  SKIP — missing {RTS_TRUE_PATH}")
        continue
    if not os.path.exists(RTS_FALSE_PATH):
        print(f"  SKIP — missing {RTS_FALSE_PATH}")
        continue
    if not mnet_files:
        print(f"  SKIP — no .pt files in {MNET_FOLDER}")
        continue

    # ── Covariance matrices for this r2 ──
    Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
    R = (r2  * R_structure).to(DEVICE, dtype=DTYPE)

    # ── Generate / load datasets ──
    all_inputs_by_F  = []
    all_targets_by_F = []
    x0_last = None

    for dataset_id in range(1, cycles + 1):
        F_current = F_matrices_for_datasets[dataset_id - 1]
        SystemModel.F_gen = False
        sys_model = SystemModel(make_f(F_current[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
        sys_model.InitSequence(m1_0, m2_0)

        dataFolderName = 'Simulations/Linear_canonical/paper/exp1_1/regular/'
        dataFileName   = f'snr_0{args.T_test}_dataset_{dataset_id}.pt'
        dataFileName_F = f'snr_0_F_dataset_{dataset_id}.pt'

        if GENERATE_DATA:
            DataGen(args, sys_model,
                    dataFolderName + dataFileName, dataFolderName + dataFileName_F,
                    delta=1, randomInit_train=False, randomInit_cv=False,
                    randomInit_test=False, randomLength=False, Test=True,
                    F_gen=F_current, x0_list=x0_last)

        [_, _, _, _, test_input, test_target] = torch.load(
            dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
        [_, _, F_test_mat_list] = torch.load(
            dataFolderName + dataFileName_F, map_location=DEVICE)

        x_last  = test_target[:, :, -1].clone()
        x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]

        all_inputs_by_F.append(test_input)
        all_targets_by_F.append(test_target)

    # ── Build pipeline ──
    sys_model_base = SystemModel(F_matrices_for_datasets[0][0], Q, h_nonlinear, R,
                                 args.T, args.T_test, m, n)
    sys_model_base.InitSequence(m1_0, m2_0)

    RTSNet_model = RTSNetNN()
    RTSNet_model.NNBuild(sys_model_base, args)
    RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
    RTSNet_Pipeline.setssModel(sys_model_base)
    RTSNet_Pipeline.setModel(RTSNet_model, args)
    RTSNet_Pipeline.setTrainingParams(args)

    # ── Baseline: TRUE F ──
    print('\n  --- Baseline: TRUE F ---')
    true_mse_lin_sum = 0.0
    xT0_last = pT0_last = None

    for dataset_id in range(cycles):
        test_input  = all_inputs_by_F[dataset_id]
        test_target = all_targets_by_F[dataset_id]
        true_F      = F_matrices_for_datasets[dataset_id][0]

        sys_model_true = SystemModel(true_F, Q, h_nonlinear, R, args.T, args.T_test, m, n)
        sys_model_true.InitSequence(m1_0, m2_0)
        sys_model_true.F_test = F_matrices_for_datasets[dataset_id]

        kw = dict(load_model_path=RTS_TRUE_PATH, generate_f=False, non_linear_h=True)
        if dataset_id > 0:
            kw.update(init_x_list=xT0_last, init_P_list=pT0_last)

        try:
            results = RTSNet_Pipeline.NNTest_no_p(sys_model_true, test_input, test_target, **kw)
            mse_db  = results[2]
            true_mse_lin_sum += float(results[1])
            print(f"    Dataset {dataset_id+1} TRUE F MSE: {mse_db:.3f} dB")
            x_last   = results[3][:, :, -1].clone()
            xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
            pT0_last = sys_model_true.m2x_0.clone().detach()
        except Exception as e:
            print(f"    ERROR true-F baseline dataset {dataset_id+1}: {e}")
            true_mse_lin_sum = float('nan')
            break

    avg_true_db = 10 * torch.log10(torch.tensor(
        true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
    print(f"  Average TRUE F MSE: {avg_true_db:.3f} dB")

    # ── Baseline: INITIAL GUESS F ──
    print('\n  --- Baseline: INITIAL GUESS F ---')
    init_mse_lin_sum = 0.0
    xF0_last = pF0_last = None

    for dataset_id in range(cycles):
        test_input  = all_inputs_by_F[dataset_id]
        test_target = all_targets_by_F[dataset_id]

        sys_model_init = SystemModel(F_initial_guess[0], Q, h_nonlinear, R, args.T, args.T_test, m, n)
        sys_model_init.InitSequence(m1_0, m2_0)
        sys_model_init.F_test = F_initial_guess

        kw = dict(load_model_path=RTS_FALSE_PATH, generate_f=False, non_linear_h=True)
        if dataset_id > 0:
            kw.update(init_x_list=xF0_last, init_P_list=pF0_last)

        try:
            results = RTSNet_Pipeline.NNTest_no_p(sys_model_init, test_input, test_target, **kw)
            mse_db  = results[2]
            init_mse_lin_sum += float(results[1])
            print(f"    Dataset {dataset_id+1} INIT GUESS MSE: {mse_db:.3f} dB")
            x_last   = results[3][:, :, -1].clone()
            xF0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
            pF0_last = sys_model_init.m2x_0.clone().detach()
        except Exception as e:
            print(f"    ERROR init-guess baseline dataset {dataset_id+1}: {e}")
            init_mse_lin_sum = float('nan')
            break

    avg_init_db = 10 * torch.log10(torch.tensor(
        init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
    print(f"  Average INIT GUESS MSE: {avg_init_db:.3f} dB")

    # ── Compare each MNet ──
    results_table = []

    for mnet_path in mnet_files:
        model_name = os.path.basename(mnet_path)
        print(f"\n  Testing MNet: {model_name}")

        emkf_mse_lin_sum   = 0.0
        current_F_estimate = None
        x0_em_last = p0_em_last = None

        try:
            for dataset_id in range(cycles):
                test_input  = all_inputs_by_F[dataset_id]
                test_target = all_targets_by_F[dataset_id]
                true_F_list = F_matrices_for_datasets[dataset_id]

                if dataset_id == 0:
                    current_F_estimate = F_initial_guess

                sys_model_ai = SystemModel(current_F_estimate[0], Q, h_nonlinear, R,
                                           args.T, args.T_test, m, n)
                sys_model_ai.InitSequence(m1_0, m2_0)
                sys_model_ai.F_test      = current_F_estimate
                sys_model_ai.F_test_TRUE = true_F_list

                kw = dict(
                    destination_path_RTS=RTS_FALSE_PATH,
                    destination_path_M  =mnet_path,
                    num_em_iters        =NUM_EM_ITERS,
                    generate_f          =False,
                    non_linear_h        =True,
                )
                if dataset_id > 0:
                    kw.update(init_x_list=x0_em_last, init_P_list=p0_em_last)

                test_losses, _, final_F_list, last_x_list = \
                    RTSNet_Pipeline.test_mstep_net(sys_model_ai, test_input, test_target, **kw)

                final_loss = float(test_losses[-1])
                emkf_mse_lin_sum  += final_loss
                current_F_estimate = final_F_list

                # use predicted last state (non-linear: don't substitute true target)
                p0_em_last = sys_model_ai.m2x_0.clone().detach()
                x0_em_last = last_x_list

                assert x0_em_last[0].ndim == 2 and x0_em_last[0].shape[1] == 1, \
                    f"x0 shape off: {x0_em_last[0].shape}"

                loss_db = 10 * torch.log10(torch.tensor(final_loss, device=DEVICE, dtype=DTYPE))
                print(f"    Dataset {dataset_id+1}: {loss_db:.3f} dB")

            avg_mse_db = 10 * torch.log10(torch.tensor(
                emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
            results_table.append((model_name, float(avg_mse_db)))
            print(f"  >>> Avg: {avg_mse_db:.3f} dB")

        except Exception as e:
            print(f"  ERROR: {e}")
            results_table.append((model_name, float('inf')))

    # ── Summary for this r-folder ──
    results_table.sort(key=lambda x: x[1])
    all_summaries.append((r_folder, r2, results_table,
                          float(avg_true_db), float(avg_init_db)))

    print(f"\n  {'─'*60}")
    print(f"  SUMMARY  {r_folder}  (r2={r2})")
    print(f"  {'─'*60}")
    print(f"  {'Model':<48} {'Avg MSE (dB)':>12}")
    print(f"  {'-'*60}")
    print(f"  {'TRUE F (oracle)':<48} {float(avg_true_db):>12.3f}")
    print(f"  {'-'*60}")
    for name, db in results_table:
        marker = ' <-- BEST' if name == results_table[0][0] and db != float('inf') else ''
        print(f"  {name:<48} {db:>12.3f}{marker}")
    print(f"  {'-'*60}")
    print(f"  {'INIT GUESS (no EMKF)':<48} {float(avg_init_db):>12.3f}")

# ─────────────────────────────────────────────
# GRAND SUMMARY
# ─────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("GRAND SUMMARY — exp_3, non-linear h (all r-folders, best MNet per folder)")
print('='*70)
print(f"{'r-folder':<10} {'r2':<8} {'TRUE F':>10} {'INIT':>10} {'BEST MNet':>10}  {'Model name'}")
print('-'*70)
for r_folder, r2, table, true_db, init_db in all_summaries:
    if table and table[0][1] != float('inf'):
        best_name, best_db = table[0]
    else:
        best_name, best_db = 'N/A', float('nan')
    print(f"{r_folder:<10} {r2:<8g} {true_db:>10.3f} {init_db:>10.3f} {best_db:>10.3f}  {best_name}")
print('='*70)
