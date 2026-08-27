"""
Rank every H-step network in one SNR bucket against the same test protocol used by
data_generate_exp_for_paper/H_exp/for_h_M_network_AI_emkf_testing_paper1.py.

H EXPERIMENT (the mirror image of compare_mnets.py):
  * F is FIXED and KNOWN for every dataset -- the dynamics are not the unknown here.
  * H is DIVERSE and UNKNOWN. The true H drifts by a fixed 0.2 rad per dataset, and the
    M-net's job is to recover it from the observation statistics.
Because the unknown is H, candidates are driven by Pipeline_ERTS.test_H_mstep_net (which
predicts deltaH from [A_yx, A_xx, S_nu, C_nu_x, H_current]), NOT test_mstep_net.

Set NOISE = 'gauss' (exp 1) or 'exponential' (exp 2) at the top; the checkpoint folder and
the data cache both follow from it. q2 == r2 in this experiment, both taken from R_BUCKET.

Architecture: RTSNet_nn.RTSNetNN with the FC8(F) + FC9(H) KGain_step restored -- see the
RTSNetNN_FH subclass below. Not RTSNet_nn_with_F (no FC9/update_H) and not the current
base class (FC8 commented out, so in_Sigma is 14 wide where the weights need 34).

Each candidate is run through 3 sequential test datasets (H and x carried over between
datasets) and scored by the final-EM-iteration state MSE averaged over the 3 datasets.
TRUE H and INITIAL GUESS H are reported as the upper/lower bounds.

Run from anywhere:  python compare_mnets_H.py
Results are also streamed to <bucket>/mnet_H_comparison.csv as each model finishes, so a
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
import Simulations.Linear_sysmdl as _lsm

# ============================================================
# NOISE SWITCH -- the only knob. 'gauss' (exp 1) or 'exponential' (exp 2).
# The noise family, the checkpoint folder and the data cache all derive from it, so they
# cannot drift apart: scoring gauss-generated data against exponential-trained
# checkpoints silently produces a meaningless number.
# An Exponential(lam) draw has variance matching the diagonal of Q_gen/R_gen but
# mean 1/lam = sqrt(var), i.e. the noise is deliberately NOT zero-mean.
# MUST be set before any DataGen call.
# ============================================================
NOISE = 'exponential'          # <-- 'gauss' or 'exponential'

EXP_TAG_BY_NOISE = {'gauss': 'exp_1', 'exponential': 'exp_2'}
if NOISE not in EXP_TAG_BY_NOISE:
    raise ValueError(f"NOISE must be 'gauss' or 'exponential', got {NOISE!r}")
_lsm.NOISE_DIST = NOISE

from Simulations.utils import DataLoader, DataGen, estimate_QR
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0

# H-aware smoother for the H experiment. RTSNet_nn_for_H.RTSNetNN is the base RTSNetNN
# with the F embedding restored in both Sigma-GRUs -- see that file's header for the full
# story. It is REQUIRED here: the H checkpoints store GRU_Sigma=34 / GRU_Sigma_bw=44, while
# the current base class builds 14 / 34 and dies with
#     RuntimeError: input.size(-1) must be equal to input_size. Expected 34, got 14
from RTSNet.RTSNet_nn_for_H import RTSNetNN
import RTSNet.RTSNet_nn as _rtsnet_nn_mod
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

# The .pt files were pickled as RTSNet.RTSNet_nn.RTSNetNN, so point that name at the
# for_H class before any torch.load. This is in-memory only -- no file on disk changes,
# and no other script is affected.
_rtsnet_nn_mod.RTSNetNN = RTSNetNN

# Same fix one layer down, for the M-nets themselves: these checkpoints were saved with a
# single LayerNorm (self.ln), while emkf/AI_M_step_for_h.py now uses per-block LayerNorms,
# so running one raises
#     AttributeError: 'DeltaH_MStepNet' object has no attribute 'block_lns'
# See emkf/AI_M_step_for_h_single_ln.py for the full story.
from emkf.AI_M_step_for_h_single_ln import DeltaH_MStepNet
import emkf.AI_M_step_for_h as _mstep_h_mod
_mstep_h_mod.DeltaH_MStepNet = DeltaH_MStepNet

# Pipeline_ERTS.NNTest swaps the observation matrix per sequence via SysModel.update_h(H),
# but only Extended_sysmdl ever defined that method -- the LINEAR SystemModel used here
# does not have it, so NNTest dies with AttributeError on the first test sequence.
# Supply the linear equivalent: h(x) = self.H @ x reads self.H directly, so recording H
# (and its transpose, which some call sites expect) is all that is needed.
if not hasattr(SystemModel, 'update_h'):
    def _update_h_linear(self, H):
        self.H   = H
        self.H_T = H.T
    SystemModel.update_h = _update_h_linear

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda")
DTYPE  = torch.float32

# The H checkpoints live under changed_H_v_0/<EXP_TAG>/<R_BUCKET>/, and EXP_TAG follows
# from the noise family above -- never set it by hand.
EXP_TAG = EXP_TAG_BY_NOISE[NOISE]

# SNR bucket. The folder tag and r2 must always change as a PAIR, which is why r2 is
# looked up from the tag instead of being a second free knob.
R_BUCKET = 'r_1'
R2_BY_BUCKET = {'r_10': 10, 'r_1': 1, 'r_01': 0.1, 'r_001': 0.01, 'r_0001': 0.001}

# The cached datasets under DATA_DIR are r2-specific, so each bucket gets its own
# sub-folder (below) and switching R_BUCKET cannot silently reuse another bucket's cache.
GENERATE_DATA = True

# Which checkpoints to sweep. INCLUDE_OLD also pulls in the archived EMKF/False/old/
# folder; MNET_GLOB narrows the sweep when you only want a subset.
INCLUDE_OLD = False
MNET_GLOB   = '*.pt'

NUM_EM_ITERS = 3

EXP_DIR     = REPO_ROOT / 'RTSNet' / 'synthetic' / 'changed_H_v_0' / EXP_TAG / R_BUCKET
MNET_FOLDER = EXP_DIR / 'EMKF' / 'False'
# One sub-folder per (noise family, SNR bucket), so a run at one r2 -- or one noise
# family -- cannot leave behind files that another run would load as its own.
DATA_DIR    = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / 'exp1_H' / EXP_TAG / R_BUCKET
CSV_PATH    = EXP_DIR / 'mnet_H_comparison.csv'
os.makedirs(DATA_DIR, exist_ok=True)


def _resolve_rts(sub_dir, names):
    """Return the first existing checkpoint among `names` inside EXP_DIR/sub_dir.

    The H buckets are not consistently named -- most hold best-rts_{true,false}.pt but
    e.g. exp_1/r_1/False_H holds RTSNET_false.pt instead. Probe rather than hard-code, and
    fail loudly (listing what IS there) instead of dying on a bare missing-file error.
    """
    folder = EXP_DIR / sub_dir
    for nm in names:
        p = folder / nm
        if p.is_file():
            return str(p)
    present = sorted(q.name for q in folder.glob('*.pt')) if folder.is_dir() else []
    raise FileNotFoundError(
        f"No RTSNet checkpoint for {EXP_TAG}/{R_BUCKET} in {folder}.\n"
        f"  tried:  {', '.join(names)}\n"
        f"  found:  {', '.join(present) if present else '(folder missing or empty)'}")


RTS_TRUE_PATH  = _resolve_rts('True_H',  ['best-rts_true.pt',  'RTSNET_true.pt'])
RTS_FALSE_PATH = _resolve_rts('False_H', ['best-rts_false.pt', 'RTSNET_false.pt'])

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

today  = datetime.today()
now    = datetime.now()
strTime= today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

# ─────────────────────────────────────────────
# SYSTEM PARAMETERS  (match for_h_M_network_AI_emkf_testing_paper1.py)
# ─────────────────────────────────────────────
args = config.general_settings()
args.N_T      = 50
args.T        = 30
args.T_test   = 30
torch.manual_seed(1)

cycles  = 3
# q2 == r2 always in the H experiment (unlike the F scripts, which pin q2 = 0.01).
r2 = R2_BY_BUCKET[R_BUCKET]
q2 = r2
print(f"H exp ({EXP_TAG})  noise = {NOISE}   bucket = {R_BUCKET}   q2 = {q2}   r2 = {r2}")
print(f"data dir = {DATA_DIR}")
print(f"true  smoother = {RTS_TRUE_PATH}")
print(f"false smoother = {RTS_FALSE_PATH}")

Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)

# F is FIXED and known for the whole H experiment -- only H drifts.
F_fixed = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
H_true  = torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)

m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

F_fixed_list = [F_fixed.clone() for _ in range(args.N_T)]

# ─────────────────────────────────────────────
# BUILD H MATRICES FOR 3 DATASETS
# ─────────────────────────────────────────────
THETA_TEST = 0.2
H_test_list = [H_true.clone().to(DEVICE) for _ in range(args.N_T)]
H_matrices_for_datasets_d = []
for i in range(cycles + 1):
    H_matrices_for_datasets_d.append([h.clone() for h in H_test_list])
    H_test_list = rotate_F(H_matrices_for_datasets_d[i], i=0, j=1, theta=THETA_TEST,
                           many=True, randomit=False)
# Drop the un-rotated base H: dataset k uses the H after k+1 rotations.
H_matrices_for_datasets = H_matrices_for_datasets_d[1:]

# ─────────────────────────────────────────────
# GENERATE / LOAD DATA  (FIXED F, DIVERSE H)
# ─────────────────────────────────────────────
all_inputs_by_H  = []
all_targets_by_H = []
all_H_matrices   = []
x0_last = None

for dataset_id in range(1, cycles + 1):
    print(f"\n=== Dataset {dataset_id} ===")
    H_current = H_matrices_for_datasets[dataset_id - 1]
    SystemModel.F_gen = False
    sys_model = SystemModel(F_fixed, Q, H_current[0], R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    dataFilePath   = str(DATA_DIR / f'snr_0{args.T_test}_dataset_{dataset_id}.pt')
    dataFilePath_H = str(DATA_DIR / f'snr_0_H_dataset_{dataset_id}.pt')

    if GENERATE_DATA:
        print(f"Generating fixed-F / diverse-H data for dataset {dataset_id} at r2={r2}...")
        print(f"  -> {dataFilePath}")
        DataGen(args, sys_model, dataFilePath, dataFilePath_H,
                fileName_H=dataFilePath_H,
                delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
                randomLength=False, Test=True,
                F_gen=F_fixed_list, H_gen=H_current, x0_list=x0_last)
    else:
        if not os.path.isfile(dataFilePath):
            raise FileNotFoundError(
                f"No cached H-exp data at {dataFilePath}. Set GENERATE_DATA = True.")
        print(f"Loading cached data for dataset {dataset_id}...")

    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFilePath, weights_only=True, map_location=DEVICE)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(dataFilePath_H, map_location=DEVICE)

    x_last = test_target[:, :, -1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]

    all_inputs_by_H.append(test_input)
    all_targets_by_H.append(test_target)
    all_H_matrices.append(H_test_mat_list)

# ─────────────────────────────────────────────
# RTSNet PIPELINE  (shared across all tests)
# ─────────────────────────────────────────────
sys_model_base = SystemModel(F_fixed, Q, H_matrices_for_datasets[0][0], R, args.T, args.T_test)
sys_model_base.InitSequence(m1_0, m2_0)

RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_base, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model_base)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

# The M-net starts from the un-rotated H every run.
H_initial_guess = [H_true.clone() for _ in range(args.N_T)]

# NNTest (not NNTest_no_p) is the baseline runner here: it applies H_test[j] per sequence
# via update_H, which NNTest_no_p never does -- it only ever updates F.
# ─────────────────────────────────────────────
# BASELINE: TRUE H
# ─────────────────────────────────────────────
print('\n=== Baseline: TRUE H ===')
true_mse_lin_sum = 0.0
xT0_last = pT0_last = None
t_start_true_h = time.perf_counter()
for dataset_id in range(cycles):
    test_input  = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]
    true_H_list = H_matrices_for_datasets[dataset_id]

    sys_model_true = SystemModel(F_fixed, Q, true_H_list[0], R, args.T, args.T_test)
    sys_model_true.InitSequence(m1_0, m2_0)
    sys_model_true.F_test = F_fixed_list
    sys_model_true.H_test = true_H_list

    kw = dict(load_model_path=RTS_TRUE_PATH, generate_f=False, generate_h=False)
    if dataset_id > 0:
        kw.update(init_x_list=xT0_last, init_P_list=pT0_last)
    results = RTSNet_Pipeline.NNTest(sys_model_true, test_input, test_target, **kw)

    mse_db  = results[2]
    true_mse_lin_sum += float(results[1])
    print(f"  Dataset {dataset_id+1} TRUE H MSE: {mse_db:.3f} dB")

    x_last  = results[3][:, :, -1].clone()
    xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
    pT0_last = sys_model_true.m2x_0.clone().detach()
t_end_true_h = time.perf_counter()

avg_true_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"  Average TRUE H MSE: {avg_true_db:.3f} dB")

# ─────────────────────────────────────────────
# BASELINE: INITIAL GUESS H
# ─────────────────────────────────────────────
print('\n=== Baseline: INITIAL GUESS H ===')
init_mse_lin_sum = 0.0
xH0_last = pH0_last = None
t_start_init_h = time.perf_counter()
for dataset_id in range(cycles):
    test_input  = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]

    sys_model_init = SystemModel(F_fixed, Q, H_initial_guess[0], R, args.T, args.T_test)
    sys_model_init.InitSequence(m1_0, m2_0)
    sys_model_init.F_test = F_fixed_list
    sys_model_init.H_test = H_initial_guess

    kw = dict(load_model_path=RTS_FALSE_PATH, generate_f=False, generate_h=False)
    if dataset_id > 0:
        kw.update(init_x_list=xH0_last, init_P_list=pH0_last)
    results = RTSNet_Pipeline.NNTest(sys_model_init, test_input, test_target, **kw)

    mse_db = results[2]
    init_mse_lin_sum += float(results[1])
    print(f"  Dataset {dataset_id+1} INIT GUESS MSE: {mse_db:.3f} dB")

    x_last  = results[3][:, :, -1].clone()
    xH0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
    pH0_last = sys_model_init.m2x_0.clone().detach()
t_end_init_h = time.perf_counter()

avg_init_db = 10 * torch.log10(torch.tensor(init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"  Average INIT GUESS MSE: {avg_init_db:.3f} dB")

# ─────────────────────────────────────────────
# DISCOVER ALL MNET MODELS
# ─────────────────────────────────────────────
# Smoother / black-box baselines sit in the same folder and would only crash
# test_H_mstep_net. Match on the PREFIX, not a substring: real M-nets are named after what
# they were trained against, so e.g. M_rand_false_trained_12_20_f_rtsnet_new_net.pt
# contains 'rtsnet' but is a genuine candidate, while RTSNET_joint_false.pt is a smoother.
_NON_MNET_PREFIXES = ('rtsnet', 'best-rts', 'best-psmooth', 'bigru', 'new_bigru', 'psmooth')


def is_mnet(path):
    low = os.path.basename(path).lower()
    return not low.startswith(_NON_MNET_PREFIXES)


all_pt = sorted(glob.glob(str(MNET_FOLDER / MNET_GLOB)))
if INCLUDE_OLD:
    all_pt += sorted(glob.glob(str(MNET_FOLDER / 'old' / MNET_GLOB)))

mnet_files = [p for p in all_pt if is_mnet(p)]
skipped    = [p for p in all_pt if not is_mnet(p)]

if not mnet_files:
    print(f"\nNo H M-net candidates found in {MNET_FOLDER}")
else:
    print(f"\nFound {len(mnet_files)} H M-net candidate(s) in {MNET_FOLDER}:")
    for p in mnet_files:
        print(f"  {os.path.relpath(p, MNET_FOLDER)}")
if skipped:
    print(f"Skipped {len(skipped)} non-M-net checkpoint(s) (bigru / rtsnet / psmooth):")
    for p in skipped:
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
    print(f"[{idx}/{len(mnet_files)}] Testing H M-net: {model_name}")
    print('='*60, flush=True)

    emkf_mse_lin_sum   = 0.0
    per_dataset_db     = []
    current_H_estimate = None
    x0_em_last = p0_em_last = None

    try:
        t_start_mnet = time.perf_counter()
        for dataset_id in range(cycles):
            test_input  = all_inputs_by_H[dataset_id]
            test_target = all_targets_by_H[dataset_id]
            true_H_list = H_matrices_for_datasets[dataset_id]

            if dataset_id == 0:
                current_H_estimate = H_initial_guess
            # else: carry over the H learned on the previous dataset (set below)

            sys_model_ai = SystemModel(F_fixed, Q, current_H_estimate[0], R, args.T, args.T_test)
            sys_model_ai.InitSequence(m1_0, m2_0)
            sys_model_ai.F_test      = F_fixed_list
            sys_model_ai.H_test      = current_H_estimate
            sys_model_ai.H_test_TRUE = true_H_list

            kw = dict(
                destination_path_RTS=RTS_FALSE_PATH,
                destination_path_M  =mnet_path,
                num_em_iters        =NUM_EM_ITERS,
                generate_h          =False,
            )
            if dataset_id > 0:
                kw.update(init_x_list=x0_em_last, init_P_list=p0_em_last)

            # test_H_mstep_net returns 5 values (the paper1 reference unpacks 4 and is
            # stale against the current pipeline): per-EM-iter x MSE, per-EM-iter H MSE,
            # final H list, final x list, stacked x.
            test_losses, test_h_losses, final_H_list, last_x_list, _x_all = \
                RTSNet_Pipeline.test_H_mstep_net(sys_model_ai, test_input, test_target, **kw)

            final_loss = float(test_losses[-1])
            emkf_mse_lin_sum  += final_loss
            current_H_estimate = final_H_list

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
        results_table.append((model_name, float(avg_mse_db), _lat_total, _lat_per_seq))
        print(f"  >>> Average over {cycles} datasets: {avg_mse_db:.3f} dB  |  latency: {_lat_total:.2f}s total, {_lat_per_seq:.1f} ms/seq")
        csv_writer.writerow([model_name, f"{float(avg_mse_db):.4f}"]
                            + [f"{d:.4f}" for d in per_dataset_db]
                            + [''] * (cycles - len(per_dataset_db))
                            + [f"{_lat_total:.2f}", f"{_lat_per_seq:.1f}", ''])

    except Exception as e:
        # Checkpoints trained with variant M-net architectures cannot be driven by
        # test_H_mstep_net. Record and move on instead of aborting the sweep.
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
_lat_true_h_total = t_end_true_h - t_start_true_h
_lat_init_h_total = t_end_init_h - t_start_init_h

print('\n' + '='*94)
print(f'SUMMARY COMPARISON -- H exp ({EXP_TAG}) / {R_BUCKET} (r2={r2})   (lower dB = better)')
print('='*94)
print(f"{'Model':<46} {'Avg MSE (dB)':>12} {'Total (s)':>10} {'ms/seq':>8}")
print('-'*94)
print(f"{'TRUE H (oracle upper bound)':<46} {float(avg_true_db):>12.3f} {_lat_true_h_total:>10.2f} {_lat_true_h_total / _N_seqs * 1000:>8.1f}")
print('-'*94)
for name, db, lat_total, lat_per_seq in results_table:
    marker = ' <-- BEST' if name == results_table[0][0] else ''
    if db == float('inf'):
        print(f"{name:<46} {'err':>12} {'err':>10} {'err':>8}")
    else:
        print(f"{name:<46} {db:>12.3f} {lat_total:>10.2f} {lat_per_seq:>8.1f}{marker}")
print('-'*94)
print(f"{'INITIAL GUESS (no EMKF)':<46} {float(avg_init_db):>12.3f} {_lat_init_h_total:>10.2f} {_lat_init_h_total / _N_seqs * 1000:>8.1f}")
print('='*94)
print(f"Full results written to {CSV_PATH}")

_ok = [r for r in results_table if r[1] != float('inf')]
if _ok:
    name, db, _, _ = _ok[0]
    print(f"\nBest H M-net:  {name}   {db:.3f} dB")
    print(f"  full path:    {MNET_FOLDER / name}")
    print(f"  vs TRUE H:    {db - float(avg_true_db):+.3f} dB gap")
    print(f"  vs INIT GUESS:{float(avg_init_db) - db:+.3f} dB improvement")
else:
    print("\nBest H M-net:  none scored successfully")
