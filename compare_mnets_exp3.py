"""
Rank every M-step network in one exp_3 SNR bucket against the same test protocol used by
data_generate_exp_for_paper/F_exp/exp3/exp3_test.py.

exp 3 = NON-LINEAR observation (h_nonlinear, range-bearing; Extended system model, EKF/ERTS).

Two kinds of candidate are scored side by side:
  * regular -- an M-net trained against the FROZEN stage-1 RTSNet. Run with
               False_F/best-rts_false.pt as the smoother.
  * joint   -- an M-net that was trained JOINTLY with its own RTSNet. It must be run with
               THAT RTSNet, not the frozen one; pairing it with best-rts_false.pt scores a
               pair that was never trained together and understates it badly.
The pairing is by filename: <...>joint<...>mnet<...>.pt  <->  same name with mnet->rtsnet.
Files whose name contains 'rtsnet' or 'bigru' are smoother/baseline checkpoints, not
M-nets, and are excluded as candidates.

Each candidate is run through 3 sequential test datasets whose true F drifts by a fixed
0.2 rad per dataset (F and x carried over between datasets), and is scored by the
final-EM-iteration state MSE averaged over the 3 datasets.

Run from anywhere:  python compare_mnets_exp3.py
Results are also streamed to <bucket>/mnet_comparison.csv as each model finishes.
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

from Simulations.Extended_sysmdl import SystemModel, rotate_F
from Simulations.Lorenz_Atractor.parameters_OLD import (
    m1x_0 as m1_0, m2x_0 as m2_0,
    m, n, h_nonlinear, Q_structure, R_structure, make_f
)
from Simulations.utils import DataLoader, DataGen
import Simulations.config as config

# The exp3 RTS checkpoints use the F-aware architecture (FC8, FC_F_bw, no FC9); the base
# RTSNet_nn.py is H-aware (FC9). Import the F-aware class and remap the name the pickles
# reference so torch.load reconstructs them with matching forward/backward code.
from RTSNet.RTSNet_nn_with_F import RTSNetNN
import RTSNet.RTSNet_nn as _rtsnet_nn_mod
_rtsnet_nn_mod.RTSNetNN = RTSNetNN

# The pre-trained checkpoints pickled self.h BY REFERENCE to
# Simulations.Lorenz_Atractor.parameters.h_nonlinear (now the 3-D spherical h). Rebind that
# name to the 2-D h_nonlinear BEFORE any torch.load so they reconstruct with the matching
# 2-D observation (otherwise InitSequence -> self.h(x) crashes on 3-D input).
import Simulations.Lorenz_Atractor.parameters as _lor_params
_lor_params.h_nonlinear = h_nonlinear

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

# The nonlinear-h path calls getJacobian (imported into Pipeline_ERTS), which hardcodes
# view(-1, m=3). Replace with a dimension-agnostic version for the 2-D model.
import Pipelines.Pipeline_ERTS as _pipe_mod
def _getJacobian_nd(x, g):
    y = x.reshape(-1)
    Jac = torch.autograd.functional.jacobian(g, y)
    return Jac.reshape(-1, y.shape[0])
_pipe_mod.getJacobian = _getJacobian_nd

# Keep the non-linear h during data generation. GenerateBatch calls
# SystemModel.update_h(H) per group, which by default rebinds self.h to a LINEAR H@x and
# would corrupt the range-bearing observations. Patch it to only record H/H_T and leave
# self.h = h_nonlinear intact (mirrors exp3_train.py / exp3_test.py).
def _update_h_keep_nonlinear(self, H):
    self.H = H
    self.H_T = H.T
SystemModel.update_h = _update_h_keep_nonlinear

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda")
DTYPE  = torch.float32

# SNR bucket. The folder tag and r2 must always change as a PAIR, which is why r2 is
# looked up from the tag instead of being a second free knob.
R_BUCKET = 'r_10'
R2_BY_BUCKET = {'r_10': 10, 'r_1': 1, 'r_01': 0.1, 'r_001': 0.01, 'r_0001': 0.001}

# exp3 keeps its OWN data folder, per bucket -- it used to share exp1_1/regular with
# exp1_2 and the two scripts overwrote each other's cached test data.
DATA_DIR_BY_BUCKET = {
    'r_10':   'exp_3_datasets_r10',
    'r_1':    'exp_3_datasets_r1',
    'r_01':   'exp_3_datasets_r01',
    'r_001':  'exp_3_datasets_r001',
    'r_0001': 'exp_3_datasets',
}

GENERATE_DATA = True

# MNET_GLOB narrows the sweep; INCLUDE_OLD also pulls in the archived EMKF/False/old/.
INCLUDE_OLD = False
MNET_GLOB   = '*.pt'

NUM_EM_ITERS = 3

EXP_DIR       = REPO_ROOT / 'RTSNet' / 'synthetic' / 'AI_M_step' / 'exp_3' / R_BUCKET
MNET_FOLDER   = EXP_DIR / 'EMKF' / 'False'
RTS_TRUE_PATH = str(EXP_DIR / 'True_F'  / 'best-rts_true.pt')
RTS_FALSE_PATH= str(EXP_DIR / 'False_F' / 'best-rts_false.pt')
DATA_DIR      = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / DATA_DIR_BY_BUCKET[R_BUCKET]
CSV_PATH      = EXP_DIR / 'mnet_comparison.csv'
os.makedirs(DATA_DIR, exist_ok=True)

for _p in (RTS_TRUE_PATH, RTS_FALSE_PATH):
    if not os.path.isfile(_p):
        raise FileNotFoundError(f"RTSNet checkpoint missing for exp_3 bucket {R_BUCKET}: {_p}")

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

today  = datetime.today()
now    = datetime.now()
strTime= today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

# ─────────────────────────────────────────────
# SYSTEM PARAMETERS  (match exp3_test.py)
# ─────────────────────────────────────────────
args = config.general_settings()
args.N_T      = 100
args.T        = 30
args.T_test   = 30
torch.manual_seed(1)

cycles  = 3
q2 = 0.01
r2 = R2_BY_BUCKET[R_BUCKET]
print(f"exp_3 (non-linear h)  bucket = {R_BUCKET}   q2 = {q2}   r2 = {r2}")
print(f"data dir = {DATA_DIR}")

Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)

F = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)

m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

# H=eye(n) placeholder: the non-linear-h model has no linear H, but NNTest_no_p sets
# self.model.H = SysModel.H when the (F-arch) checkpoint has no H attribute. The F forward
# never reads it, but it must not be None.
H_EYE = torch.eye(n, device=DEVICE, dtype=DTYPE)

sys_model = SystemModel(F, Q, h_nonlinear, R, args.T, args.T_test, m, n)
SystemModel.F_gen = False
sys_model.InitSequence(m1_0, m2_0)

# ─────────────────────────────────────────────
# BUILD F MATRICES FOR 3 DATASETS
# ─────────────────────────────────────────────
THETA_TEST = 0.2
F_test_list = [F.clone().to(DEVICE) for _ in range(args.N_T)]
F_matrices_for_datasets_d = []
for i in range(cycles + 1):
    F_matrices_for_datasets_d.append([f.clone() for f in F_test_list])
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=THETA_TEST, many=True, randomit=False)
# Drop the un-rotated base F: dataset k uses the F after k+1 rotations.
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]

# ─────────────────────────────────────────────
# GENERATE / LOAD DATA  (non-linear h -- see update_h patch above)
# ─────────────────────────────────────────────
all_inputs_by_F  = []
all_targets_by_F = []
all_F_matrices   = []
x0_last = None

H_gen_list = [H_EYE.clone() for _ in range(args.N_T)]

for dataset_id in range(1, cycles + 1):
    print(f"\n=== Dataset {dataset_id} ===")
    F_current = F_matrices_for_datasets[dataset_id - 1]
    SystemModel.F_gen = False
    sys_model = SystemModel(F_current[0], Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1_0, m2_0)

    dataFilePath   = str(DATA_DIR / f'snr_0{args.T_test}_dataset_{dataset_id}.pt')
    dataFilePath_F = str(DATA_DIR / f'snr_0_F_dataset_{dataset_id}.pt')

    if GENERATE_DATA:
        print(f"Generating non-linear-h data for dataset {dataset_id} at r2={r2}...")
        print(f"  -> {dataFilePath}")
        DataGen(args, sys_model, dataFilePath, dataFilePath_F,
                delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
                randomLength=False, Test=True, F_gen=F_current, H_gen=H_gen_list, x0_list=x0_last)
    else:
        if not os.path.isfile(dataFilePath):
            raise FileNotFoundError(
                f"No cached exp_3 data at {dataFilePath}. Set GENERATE_DATA = True.")
        print(f"Loading cached data for dataset {dataset_id}...")

    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFilePath, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFilePath_F, map_location=DEVICE)

    x_last = test_target[:, :, -1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]

    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)

# ─────────────────────────────────────────────
# RTSNet PIPELINE  (shared across all tests)
# ─────────────────────────────────────────────
sys_model_base = SystemModel(F_matrices_for_datasets[0][0], Q, h_nonlinear, R,
                             args.T, args.T_test, m, n, H=H_EYE.clone())
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

    sys_model_true = SystemModel(true_F, Q, h_nonlinear, R, args.T, args.T_test, m, n,
                                 H=H_EYE.clone())
    sys_model_true.InitSequence(m1_0, m2_0)
    sys_model_true.F_test = F_matrices_for_datasets[dataset_id]

    kw = dict(load_model_path=RTS_TRUE_PATH, generate_f=False, non_linear_h=True)
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

    sys_model_init = SystemModel(F_initial_guess[0], Q, h_nonlinear, R, args.T, args.T_test, m, n,
                                 H=H_EYE.clone())
    sys_model_init.InitSequence(m1_0, m2_0)
    sys_model_init.F_test = F_initial_guess

    kw = dict(load_model_path=RTS_FALSE_PATH, generate_f=False, non_linear_h=True)
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
# DISCOVER CANDIDATES  (regular M-nets + jointly-trained M-net/RTSNet pairs)
# ─────────────────────────────────────────────
def discover_candidates(paths):
    """-> list of (name, mnet_path, rts_path_or_None, kind).

    'joint' in the filename means the M-net was trained together with its own RTSNet and
    must be evaluated with it; everything else is a regular M-net that runs against the
    frozen stage-1 RTSNet (RTS_FALSE_PATH).
    """
    out = []
    for p in paths:
        name = os.path.relpath(p, MNET_FOLDER)
        base = os.path.basename(p)
        low  = base.lower()
        if 'rtsnet' in low or 'bigru' in low:
            # smoother / black-box baseline checkpoint, not an M-net
            continue
        if 'joint' in low:
            partner = os.path.join(os.path.dirname(p), base.replace('mnet', 'rtsnet'))
            if os.path.isfile(partner):
                out.append((name, p, partner, 'joint'))
            else:
                # Do NOT silently fall back to the frozen RTSNet: that pair was never
                # trained together and the resulting number would be meaningless.
                out.append((name, p, None, 'joint-no-partner'))
        else:
            out.append((name, p, RTS_FALSE_PATH, 'regular'))
    return out

all_pt = sorted(glob.glob(str(MNET_FOLDER / MNET_GLOB)))
if INCLUDE_OLD:
    all_pt += sorted(glob.glob(str(MNET_FOLDER / 'old' / MNET_GLOB)))

candidates = discover_candidates(all_pt)
skipped    = [os.path.relpath(p, MNET_FOLDER) for p in all_pt
              if os.path.relpath(p, MNET_FOLDER) not in {c[0] for c in candidates}]

if not candidates:
    print(f"\nNo M-net candidates found in {MNET_FOLDER}")
else:
    print(f"\nFound {len(candidates)} M-net candidate(s) in {MNET_FOLDER}:")
    for name, _, rts, kind in candidates:
        rts_show = os.path.basename(rts) if rts else '(MISSING)'
        print(f"  [{kind:<16}] {name}   smoother: {rts_show}")
if skipped:
    print(f"Skipped {len(skipped)} non-M-net checkpoint(s) (rtsnet / bigru):")
    for s in skipped:
        print(f"  {s}")

# ─────────────────────────────────────────────
# COMPARE EACH CANDIDATE
# ─────────────────────────────────────────────
results_table = []  # list of (name, kind, avg_mse_db, total_s, ms_per_seq)

csv_file = open(CSV_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['model', 'kind', 'smoother', 'avg_mse_db', 'ds1_db', 'ds2_db', 'ds3_db',
                     'total_s', 'ms_per_seq', 'error'])
csv_file.flush()

for idx, (model_name, mnet_path, rts_path, kind) in enumerate(candidates, start=1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(candidates)}] Testing MNet: {model_name}  ({kind})")
    print('='*60, flush=True)

    if rts_path is None:
        msg = (f"joint M-net but partner RTSNet "
               f"{os.path.basename(mnet_path).replace('mnet', 'rtsnet')} not found")
        print(f"  SKIP: {msg}")
        results_table.append((model_name, kind, float('inf'), float('inf'), float('inf')))
        csv_writer.writerow([model_name, kind, '', '', '', '', '', '', '', msg])
        csv_file.flush()
        continue

    print(f"  smoother: {rts_path}")

    emkf_mse_lin_sum   = 0.0
    per_dataset_db     = []
    current_F_estimate = None
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

            sys_model_ai = SystemModel(current_F_estimate[0], Q, h_nonlinear, R,
                                       args.T, args.T_test, m, n, H=H_EYE.clone())
            sys_model_ai.InitSequence(m1_0, m2_0)
            sys_model_ai.F_test      = current_F_estimate
            sys_model_ai.F_test_TRUE = true_F_list

            kw = dict(
                destination_path_RTS=rts_path,      # joint pair -> its own RTSNet
                destination_path_M  =mnet_path,
                num_em_iters        =NUM_EM_ITERS,
                generate_f          =False,
                non_linear_h        =True,
            )
            if dataset_id > 0:
                kw.update(init_x_list=x0_em_last, init_P_list=p0_em_last)

            test_losses, test_f_losses, final_F_list, last_x_list = \
                RTSNet_Pipeline.test_mstep_net(sys_model_ai, test_input, test_target, **kw)

            final_loss = float(test_losses[-1])
            emkf_mse_lin_sum  += final_loss
            current_F_estimate = final_F_list

            # Chain the EMKF's OWN smoothed last state into the next dataset (never the
            # true target -- that would be an oracle warm-start).
            p0_em_last = sys_model_ai.m2x_0.clone().detach()
            x0_em_last = last_x_list
            assert x0_em_last[0].ndim == 2 and x0_em_last[0].shape[1] == 1, \
                f"x0 shape off: {x0_em_last[0].shape}"

            loss_db = 10 * torch.log10(torch.tensor(final_loss, device=DEVICE, dtype=DTYPE))
            per_dataset_db.append(float(loss_db))
            print(f"  Dataset {dataset_id+1}: final loss = {loss_db:.3f} dB", flush=True)

        t_end_mnet = time.perf_counter()
        avg_mse_db = 10 * torch.log10(torch.tensor(emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
        _lat_total = t_end_mnet - t_start_mnet
        _lat_per_seq = _lat_total / (cycles * args.N_T) * 1000
        results_table.append((model_name, kind, float(avg_mse_db), _lat_total, _lat_per_seq))
        print(f"  >>> Average over {cycles} datasets: {avg_mse_db:.3f} dB  |  latency: {_lat_total:.2f}s total, {_lat_per_seq:.1f} ms/seq")
        csv_writer.writerow([model_name, kind, os.path.basename(rts_path), f"{float(avg_mse_db):.4f}"]
                            + [f"{d:.4f}" for d in per_dataset_db]
                            + [''] * (cycles - len(per_dataset_db))
                            + [f"{_lat_total:.2f}", f"{_lat_per_seq:.1f}", ''])

    except Exception as e:
        # Checkpoints trained with variant M-net architectures cannot be driven by
        # test_mstep_net. Record and move on instead of aborting the sweep.
        print(f"  ERROR with {model_name}: {type(e).__name__}: {e}")
        results_table.append((model_name, kind, float('inf'), float('inf'), float('inf')))
        csv_writer.writerow([model_name, kind, os.path.basename(rts_path), '', '', '', '',
                             '', '', f"{type(e).__name__}: {e}"])

    csv_file.flush()

csv_file.close()

# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
results_table.sort(key=lambda x: x[2])

_N_seqs = cycles * args.N_T
_lat_true_f_total = t_end_true_f - t_start_true_f
_lat_init_f_total = t_end_init_f - t_start_init_f

print('\n' + '='*94)
print(f'SUMMARY COMPARISON -- exp_3 (non-linear h) / {R_BUCKET} (r2={r2})   (lower dB = better)')
print('='*94)
print(f"{'Model':<46} {'Kind':<10} {'Avg MSE (dB)':>12} {'Total (s)':>10} {'ms/seq':>8}")
print('-'*94)
print(f"{'TRUE F (oracle upper bound)':<46} {'baseline':<10} {float(avg_true_db):>12.3f} {_lat_true_f_total:>10.2f} {_lat_true_f_total / _N_seqs * 1000:>8.1f}")
print('-'*94)
for name, kind, db, lat_total, lat_per_seq in results_table:
    marker = ' <-- BEST' if name == results_table[0][0] else ''
    if db == float('inf'):
        print(f"{name:<46} {kind:<10} {'err':>12} {'err':>10} {'err':>8}")
    else:
        print(f"{name:<46} {kind:<10} {db:>12.3f} {lat_total:>10.2f} {lat_per_seq:>8.1f}{marker}")
print('-'*94)
print(f"{'INITIAL GUESS (no EMKF)':<46} {'baseline':<10} {float(avg_init_db):>12.3f} {_lat_init_f_total:>10.2f} {_lat_init_f_total / _N_seqs * 1000:>8.1f}")
print('='*94)
print(f"Full results written to {CSV_PATH}")

# Best of each kind, so a joint model is not hidden by a better regular one and vice versa.
for kind_label in ('regular', 'joint'):
    of_kind = [r for r in results_table if r[1] == kind_label and r[2] != float('inf')]
    if of_kind:
        name, _, db, _, _ = of_kind[0]
        print(f"\nBest {kind_label:<8} M-net:  {name}   {db:.3f} dB")
        print(f"  full path:    {MNET_FOLDER / name}")
        print(f"  vs TRUE F:    {db - float(avg_true_db):+.3f} dB gap")
        print(f"  vs INIT GUESS:{float(avg_init_db) - db:+.3f} dB improvement")
    else:
        print(f"\nBest {kind_label:<8} M-net:  none scored successfully")
