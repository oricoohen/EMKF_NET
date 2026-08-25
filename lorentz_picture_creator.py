"""
lorentz_picture_creator.py

Generate ONE shared Lorenz test set (5 cycles / datasets, r2 = 1) and run every
algorithm on that SAME data, then save one picture (estimated x vs time) per
algorithm plus a combined 3D attractor picture.

Algorithms produced:
    1. true x        - ground-truth state
    2. rts true      - analytic RTS smoother, TRUE H
    3. rts false     - analytic RTS smoother, initial/false H
    4. rtsnet false  - RTSNet, initial/false H
    5. emkf          - analytic EMKF (learned H)
    6. emkalmanet    - AI EMKF / neural EM (learned H)
    7. bgru          - BiGRU smoother

Output: PNGs in ./lorenz_pictures/
    x_true_x.png, x_rts_true.png, x_rts_false.png, x_rtsnet_false.png,
    x_emkf.png, x_emkalmanet.png, x_bgru.png, attractor_3d.png

Run:
    python lorentz_picture_creator.py
"""
import os
import time
import math
import random
from datetime import datetime

import torch
import matplotlib
matplotlib.use("Agg")  # write files, no interactive window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import Simulations.config as config
from Simulations.utils import DataGen
from Simulations.Extended_sysmdl import SystemModel
from Simulations.Lorenz_Atractor.parameters import (
    m1x_0, m2x_0, m, n, f, hRotate, H_Rotate, H_Rotate_inv,
    Q_structure, R_structure, H_design,
)
from Simulations.Linear_sysmdl import rotate_H

from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_H
from emkf.main_emkf_func import EMKF_H_analitic_f_nonlinear
from RTSNet.RTSNet_nn import RTSNetNN
from Baselines.BiGRU_smoother import test_bigru_smoother
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

# ============================================================
# Reproducibility / device
# ============================================================
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
print("Using device:", DEVICE)

m1x_0 = m1x_0.to(DEVICE)
m2x_0 = m2x_0.to(DEVICE)
H_Rotate = H_Rotate.to(DEVICE)
H_Rotate_inv = H_Rotate_inv.to(DEVICE)
Q_structure = Q_structure.to(DEVICE)
R_structure = R_structure.to(DEVICE)
H_design = H_design.to(DEVICE)

today = datetime.today()
strTime = today.strftime("%m.%d.%y") + "_" + datetime.now().strftime("%H:%M:%S")
print("Current Time =", strTime)

# ============================================================
# Settings  (5 datasets, r2 = 1, T = 30 to match trained neural models)
# ============================================================
args = config.general_settings()
args.N_T = 100      # number of test sequences
args.T = 30         # sequence length (train/cv)
args.T_test = 30    # sequence length (test)

cycles = 10          # 5 datasets / cycles
sample_idx = 0      # which test sequence to draw

GENERATE_DATA = True
num_em_iters = 2    # neural EM iterations (emkalmanet)
max_iter =  3     # analytic EMKF iterations (emkf)

# --- noise: r2 = 1 for EVERYTHING (data generation AND every algorithm) ---
r2 = torch.tensor([10], device=DEVICE)
vdB = -20
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
Q_true = q2[0] * Q_structure
R_true = r2[0] * R_structure

print(f"q2 = {q2.item()}, r2 = {r2.item()} (used by every algorithm)")

# ============================================================
# Model paths (T = 30 trained networks)
# ============================================================
path_rtsnet_partial        = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_10/10datasets/RTSNet_partial.pt'
path_rtsnet_partial_joint  = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_10/10datasets/RTSNet_partial_joint.pt'
path_M_joint               = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_10/10datasets/M_step_net_joint.pt'
bigru_path                 = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_10/10datasets/old/bigru_smoother.pt'

# ============================================================
# Build the 5 rotated H matrices (F fixed)
# ============================================================
initial_guess_H = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]

H_matrices_for_datasets_d = []
H_test_list = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]
for i in range(cycles + 1):
    H_matrices_for_datasets_d.append([hh.clone() for hh in H_test_list])
    H_test_list = rotate_H(H_matrices_for_datasets_d[i], theta=0.1, many=True, randomit=False)
H_matrices_for_datasets = H_matrices_for_datasets_d[1:]

# ============================================================
# Generate the shared datasets (r2 = 1)
# ============================================================
print("\n" + "=" * 80)
print(f"GENERATING {cycles} DATASETS (r2 = {int(r2.item())}, T = {args.T_test})")
print("=" * 80)

all_inputs_by_H = []
all_targets_by_H = []
all_H_matrices = []

r2_val = int(r2[0].item())
dataFolderName = f'Simulations/Lorenz_Atractor/data/picture_r2_{r2_val}_T{args.T_test}_c{cycles}/'
os.makedirs(dataFolderName, exist_ok=True)

x0_last = None
for dataset_id in range(1, cycles + 1):
    print(f"\n=== Generating Dataset {dataset_id} ===")
    H_current = H_matrices_for_datasets[dataset_id - 1]

    sys_model = SystemModel(f, Q_true, hRotate, R_true, args.T, args.T_test, m, n, H_Rotate)
    sys_model.InitSequence(m1x_0, m2x_0)

    dataFileName   = f'dataset_{dataset_id}.pt'
    dataFileName_H = f'dataset_{dataset_id}_H.pt'
    dataFileName_F = f'dataset_{dataset_id}_F.pt'

    if GENERATE_DATA:
        DataGen(args, sys_model,
                dataFolderName + dataFileName,
                dataFolderName + dataFileName_F,
                fileName_H=dataFolderName + dataFileName_H,
                delta=1,
                randomInit_train=False, randomInit_cv=False, randomInit_test=False,
                randomLength=False, Test=True,
                F_gen=False, H_gen=H_current,
                x0_list=x0_last, H_init=H_current)

    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(
        dataFolderName + dataFileName_H, map_location=DEVICE)

    # continuity: next dataset starts where this one ended
    x_last = test_target[:, :, -1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]

    all_inputs_by_H.append(test_input)
    all_targets_by_H.append(test_target)
    all_H_matrices.append(H_test_mat_list)
    print(f"Dataset {dataset_id}: input {test_input.shape}, target {test_target.shape}")

# ============================================================
# Storage (per-dataset [N_T, m, T]) for each algorithm
# ============================================================
all_true_x       = [all_targets_by_H[d].clone() for d in range(cycles)]
all_rts_true     = []
all_rts_false    = []
all_rtsnet_false = []
all_emkf         = []
all_emkalmanet   = []
all_bgru         = []

# ------------------------------------------------------------
# per-algorithm wall-clock timing (total seconds over all datasets)
# ------------------------------------------------------------
timings = {}
def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

# ------------------------------------------------------------
# 2. rts true  — analytic RTS smoother with TRUE H (r2 = 1)
# ------------------------------------------------------------
print("\n=== [rts true] analytic RTS smoother, TRUE H ===")
x0_last, p0_last = None, None
_sync(); _t_start = time.perf_counter()
for d in range(cycles):
    sm = SystemModel(f, Q_true, hRotate, R_true, args.T, args.T_test, m, n, H_Rotate)
    sm.InitSequence(m1x_0, m2x_0)
    _a, _avg, _db, x_list, p_list, _ = S_Test_ext_H(
        sm, all_inputs_by_H[d], all_targets_by_H[d],
        H_list=H_matrices_for_datasets[d], generate_h=False,
        init_x_list=x0_last, init_P_list=p0_last)
    # (r2 = 1 system model above)
    all_rts_true.append(x_list.detach().clone())
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone() for k in range(args.N_T)]
    print(f"  dataset {d + 1}: {_db.item():.3f} dB")
_sync(); timings["rts true"] = time.perf_counter() - _t_start

# ------------------------------------------------------------
# 3. rts false — analytic RTS smoother with INITIAL/FALSE H
# ------------------------------------------------------------
print("\n=== [rts false] analytic RTS smoother, initial/false H ===")
x0_last, p0_last = None, None
_sync(); _t_start = time.perf_counter()
for d in range(cycles):
    sm = SystemModel(f, Q_true, hRotate, R_true, args.T, args.T_test, m, n, H_Rotate)
    sm.InitSequence(m1x_0, m2x_0)
    _a, _avg, _db, x_list, p_list, _ = S_Test_ext_H(
        sm, all_inputs_by_H[d], all_targets_by_H[d],
        H_list=initial_guess_H, generate_h=False,
        init_x_list=x0_last, init_P_list=p0_last)
    all_rts_false.append(x_list.detach().clone())
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [m2x_0.clone() for _ in range(args.N_T)]
    print(f"  dataset {d + 1}: {_db.item():.3f} dB")
_sync(); timings["rts false"] = time.perf_counter() - _t_start

# ------------------------------------------------------------
# 5. emkf — analytic EMKF (learned H), sequential
# ------------------------------------------------------------
print("\n=== [emkf] analytic EMKF ===")
x0_last, p0_last = None, None
H_current_estimate = [initial_guess_H[k].clone() for k in range(args.N_T)]
_sync(); _t_start = time.perf_counter()
for d in range(cycles):
    sm = SystemModel(f, Q_true, hRotate, R_true, args.T, args.T_test, m, n, H_Rotate)
    sm.InitSequence(m1x_0, m2x_0)
    H_matrices, _lk, _it, _mse, x_last, p_last, x_list_emkf = EMKF_H_analitic_f_nonlinear(
        sm, H_current_estimate, all_inputs_by_H[d], m1x_0, m2x_0, all_targets_by_H[d],
        max_it=max_iter, generate_h=False, init_x_list=x0_last, init_P_list=p0_last)
    all_emkf.append(x_list_emkf.detach().clone() if torch.is_tensor(x_list_emkf) else x_list_emkf)
    x0_last = [x_last[k].clone() for k in range(args.N_T)]
    p0_last = [p_last[k].clone() for k in range(args.N_T)]
    H_current_estimate = [Hs[-1].clone() for Hs in H_matrices]
    print(f"  dataset {d + 1}: {10 * torch.log10(_mse).item():.3f} dB")
_sync(); timings["emkf"] = time.perf_counter() - _t_start

# ============================================================
# Neural pipeline setup
# ============================================================
sys_model = SystemModel(f, Q_true, hRotate, R_true, args.T, args.T_test, m, n, H_Rotate)
sys_model.InitSequence(m1x_0, m2x_0)

RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

# ------------------------------------------------------------
# 4. rtsnet false — RTSNet with INITIAL/FALSE H (true Q,R)
# ------------------------------------------------------------
print("\n=== [rtsnet false] RTSNet, initial/false H ===")
xH0_last = None
_sync(); _t_start = time.perf_counter()
for d in range(cycles):
    sm = SystemModel(f, Q_true, hRotate, R_true, args.T, args.T_test, m, n, H_Rotate)
    sm.InitSequence(m1x_0, m2x_0)
    sm.H_test = initial_guess_H
    results = RTSNet_Pipeline.NNTest(
        sm, all_inputs_by_H[d], all_targets_by_H[d], path_rtsnet_partial,
        generate_h=False, generate_f=None,
        init_x_list=(None if d == 0 else xH0_last), init_P_list=None)
    all_rtsnet_false.append(results[3].detach().clone())
    x_last = results[3][:, :, -1].clone()
    xH0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
    print(f"  dataset {d + 1}: {results[2]:.3f} dB")
_sync(); timings["rtsnet false"] = time.perf_counter() - _t_start

# ------------------------------------------------------------
# 6. emkalmanet — AI EMKF / neural EM (learned H), sequential
# ------------------------------------------------------------
print("\n=== [emkalmanet] AI EMKF (neural EM) ===")
x0_em_last = None
current_H_estimate = None
_sync(); _t_start = time.perf_counter()
for d in range(cycles):
    sm = SystemModel(f, Q_true, hRotate, R_true, args.T, args.T_test, m, n, H_Rotate)
    sm.InitSequence(m1x_0, m2x_0)
    sm.H_test = initial_guess_H if d == 0 else current_H_estimate
    sm.H_test_TRUE = H_matrices_for_datasets[d]

    test_losses, test_h_losses, final_H_list, last_x_list, list_x = RTSNet_Pipeline.test_H_mstep_net(
        sm, all_inputs_by_H[d], all_targets_by_H[d],
        destination_path_RTS=path_rtsnet_partial_joint,
        destination_path_M=path_M_joint,
        num_em_iters=num_em_iters, generate_h=False,
        init_x_list=(None if d == 0 else x0_em_last), init_P_list=None)

    all_emkalmanet.append(list_x.detach().clone())
    current_H_estimate = final_H_list
    x0_em_last = [last_x_list[j].clone() for j in range(len(last_x_list))]
    print(f"  dataset {d + 1}: final loss {float(test_losses[-1]):.4e}")
_sync(); timings["emkalmanet"] = time.perf_counter() - _t_start

# ------------------------------------------------------------
# 7. bgru — BiGRU smoother
# ------------------------------------------------------------
print("\n=== [bgru] BiGRU smoother ===")
# seq_by_seq=True: one sequence per forward, so the runtime is comparable with
# the other algorithms (all of which loop sequence by sequence). return_time=True
# measures the forward passes only -- the per-call torch.load of the checkpoint
# is NOT counted as inference time.
timings["bgru"] = 0.0
for d in range(cycles):
    _mse, _mse_db, x_bigru, _t_d = test_bigru_smoother(
        test_input=all_inputs_by_H[d], test_target=all_targets_by_H[d],
        load_path=bigru_path, device=DEVICE, return_time=True)
    all_bgru.append(x_bigru.detach().clone())
    timings["bgru"] += _t_d
    print(f"  dataset {d + 1}: {_mse_db:.3f} dB")

# ============================================================
# MSE summary for every algorithm (vs. the true state), printed at the end
# ============================================================
def _to_batch(item):
    # normalize one dataset's output to a tensor [N_T, m, T]
    if torch.is_tensor(item):
        return item
    return torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t) for t in item])


def algo_mse_linear(per_dataset):
    """mean squared error (linear) across all datasets / samples / time."""
    se_sum, count = 0.0, 0
    for d in range(cycles):
        est = _to_batch(per_dataset[d]).to(DEVICE).float()   # [N_T, m, T]
        tgt = all_targets_by_H[d].to(DEVICE).float()         # [N_T, m, T]
        se_sum += torch.sum((est - tgt) ** 2).item()
        count  += est.numel()
    return se_sum / count


mse_algos = {
    "rts true":     all_rts_true,
    "rts false":    all_rts_false,
    "rtsnet false": all_rtsnet_false,
    "emkf":         all_emkf,
    "emkalmanet":   all_emkalmanet,
    "bgru":         all_bgru,
}

# Each algorithm processes cycles*N_T segments of args.T_test steps each, so
# dividing the total by n_seqs gives the cost of ONE args.T_test-step segment.
n_seqs = cycles * args.N_T   # total segments processed by each algorithm

print("\n" + "=" * 76)
print(f"MSE / TIMING SUMMARY  (r2={r2_val}, {cycles} datasets, N_T={args.N_T}, T={args.T_test})")
print("=" * 76)
print(f"{'algorithm':<14}{'MSE (linear)':>16}{'MSE (dB)':>12}"
      f"{'total (s)':>12}{f'ms/{args.T_test}steps':>14}")
print("-" * 76)
mse_lin_total = 0.0
for name, store in mse_algos.items():
    lin = algo_mse_linear(store)
    mse_lin_total += lin
    tot = timings.get(name, float('nan'))
    ms_per_seq = tot / n_seqs * 1000.0
    print(f"{name:<14}{lin:>16.6f}{10.0 * math.log10(lin + 1e-12):>12.3f}"
          f"{tot:>12.2f}{ms_per_seq:>14.3f}")
print("-" * 76)
print(f"{'SUM':<14}{mse_lin_total:>16.6f}{10.0 * math.log10(mse_lin_total + 1e-12):>12.3f}")
print("=" * 76)
print(f"note: ms/{args.T_test}steps = time to smooth ONE {args.T_test}-step sequence "
      f"(total / {cycles}*{args.N_T}).")
print("      every algorithm, bgru included, is timed sequence-by-sequence.")

# ============================================================
# Glue ONE sample's trajectory across datasets -> [dim, cycles*T]
# (single clean 3D trajectory, exactly like the combined plot the
#  user liked — just one algorithm per figure).
# ============================================================
def glue(per_dataset):
    # per_dataset[d] is either a tensor [N_T, m, T] or a list of [m, T]
    # tensors (EMKF). Indexing [sample_idx] gives [m, T] in both cases.
    parts = [per_dataset[d][sample_idx] for d in range(cycles)]
    return torch.cat(parts, dim=1).detach().cpu()


# (title, glued trajectory [dim, cycles*T], color) — colors match the
# combined-figure legend order the user showed.
algorithms = [
    ("true x",       glue(all_true_x),       "#1f77b4"),  # blue
    ("rts true",     glue(all_rts_true),     "#ff7f0e"),  # orange
    ("rts false",    glue(all_rts_false),    "#2ca02c"),  # green
    ("rtsnet false", glue(all_rtsnet_false), "#d62728"),  # red
    ("emkf",         glue(all_emkf),         "#9467bd"),  # purple
    ("emkalmanet",   glue(all_emkalmanet),   "#8c564b"),  # brown
    ("bgru",         glue(all_bgru),         "#e377c2"),  # pink
]

OUT_DIR = "lorenz_pictures"
os.makedirs(OUT_DIR, exist_ok=True)


def safe(name):
    return name.replace(" ", "_").replace("/", "")


print("\n" + "=" * 80)
print("SAVING 3D LORENZ ATTRACTOR PICTURES (one per algorithm) ->", OUT_DIR)
print("=" * 80)

# ---- one 3D trajectory picture PER algorithm (axes shown, like the sample) ----
saved_paths = {}
for name, traj, color in algorithms:
    xn = traj.numpy()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xn[0], xn[1], xn[2], color=color, linewidth=1.5, label=name)
    ax.set_title(f"Lorenz attractor — {name} (sample {sample_idx}, {cycles} datasets)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"attractor_{safe(name)}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths[name] = out_path
    print(f"  {name:<14} -> {out_path}")

print("\nAll pictures saved in:", os.path.abspath(OUT_DIR))
for name, p in saved_paths.items():
    print(f"  {name:<14}: {p}")
