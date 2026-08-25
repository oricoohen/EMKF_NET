import os
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — save plots to disk
import matplotlib.pyplot as plt

from datetime import datetime

from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config

from Pipelines.Pipeline_mic import Pipeline_mic as Pipeline
from RTSNet.RTSNet_nn import RTSNetNN
from Baselines.BiGRU_smoother import train_bigru_smoother

from Simulations.TDOA_2D.parameters import (
    m, n, m1x_0, m2x_0, M_mics,
    Q_structure, R_structure,
    make_F_block, h, h_jacobian,
    generate_dataset_raw_batch,
    make_f,
    PX_MIN, PX_MAX, PY_MIN, PY_MAX,
)
from Simulations.TDOA_2D.ekf_erts import run_ekf_erts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

loss_fn = nn.MSELoss(reduction="mean")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 400
args.N_CV = 150
args.N_T = 100
args.T = 30
args.T_test = 30
### training parameters
args.n_steps = 500
args.n_batch = 15
args.lr = 1e-4
args.wd = 1e-3
args.use_amp = False

T      = 30
T_test = 30

### noise levels
q2 = 0.01
r2 = 1

### cycle: number of datasets
cycle = 6
# Every sequence-segment (each sequence, each dataset) draws theta independently
# from Uniform(-theta_max, +theta_max). No dataset-level base theta, no drift —
# fully independent per segment. Matches the test generation exactly.
theta_max = 0.12       # drawn range = Uniform(-0.12, +0.12)

### EM iterations
num_em_iters = 2

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

### paths
# Fresh output location for the random-theta + F-reset-per-dataset run
# (each dataset restarts F from theta=0, no propagate_F).
# Warm-start LOAD paths below still read the old r10/cycle1 checkpoints.
save_dir  = "RTSNet/tdoa_2d/diff_3_mics/r1q001_6cycles/"
cycle_dir = save_dir + f"{cycle}cycle/"
os.makedirs(save_dir,  exist_ok=True)
os.makedirs(cycle_dir, exist_ok=True)

# Warm-start load paths — r=10 5-cycle checkpoints are the closest available.
# MNet is trained fresh in this script (see standalone MNet section below) —
# no MNet warm-start path needed; the r=1 joint MNet checkpoint was tried
# previously and found to be collapsed/broken, so it is deliberately not used.
load_path_rtsnet_true    = "RTSNet/tdoa_2d/3mics/r10/cycle1/5cycle/5dRTSNet_true0.001.pt"
# load_path_rtsnet_false   = "RTSNet/tdoa_2d/3mics/r10/cycle1/5cycle/5dRTSNet_false0.001.pt"
load_path_rtsnet_F_joint = "RTSNet/tdoa_2d/3mics/r10/cycle1/5cycle/5dRTSNet_false0.001.pt"
bigru_load_path          = "RTSNet/tdoa_2d/3mics/r10/cycle1/5cycle/BiGRU.pt"

# Sparse-measurement mask: OBS_DROP fraction of timesteps have NO measurement
# (pure F-prediction across the gap), forcing the smoother to lean on F so a good
# F estimate actually pays off. Same construction is used in the test script so the
# train/test measurement schedule is byte-identical (T_train == T_test).
OBS_DROP = 0.0   # 0.0 = full obs (dense, matches the analytic test); >0 drops that fraction of steps
def _make_obs_mask(T, drop):
    mask = torch.ones(T, dtype=torch.bool)
    if drop and drop > 0:
        n_drop = int(round(T * drop))
        drop_idx = torch.linspace(1, T - 1, steps=n_drop).round().long().unique()  # never t=0
        mask[drop_idx] = False
    return mask
obs_mask = None if OBS_DROP == 0 else _make_obs_mask(args.T, OBS_DROP)
drop_tag = f"_drop{int(round(OBS_DROP * 100))}" if OBS_DROP > 0 else ""

# Cycle-dataset experiment outputs
T_tag = f"_T{args.T}"   # encode sequence length so T=50 / T=70 networks don't collide
destination_path_rtsnet_true   = cycle_dir + f"5dRTSNet_true0.001{T_tag}{drop_tag}.pt"
load_path_rtsnet_false = destination_path_rtsnet_true
destination_path_rtsnet_false  = cycle_dir + f"5dRTSNet_false0.001{T_tag}{drop_tag}.pt"
destination_path_bigru         = cycle_dir + f"BiGRU{T_tag}small.pt"   # trained below; NN test loads this

data_path = save_dir + "training_3_dataset_data.pt"

###################
###    FLAGS     ###
USE_BIG_MSTEP_NET = True   # False -> same M-step net as before, True -> wider residual M-step net
MSTEP_HIDDEN_DIM  = 512    # used only when USE_BIG_MSTEP_NET=True
args.use_big_mstep_net = USE_BIG_MSTEP_NET
args.mstep_hidden_dim = MSTEP_HIDDEN_DIM if USE_BIG_MSTEP_NET else 256

mstep_arch_tag = "big_mstep" if USE_BIG_MSTEP_NET else "base_mstep"

# A1_RES: feed the M-step residual (A1 - F@A2) to the MNet instead of raw A1.
# Failed to train from a fresh init (expansive-F blow-up), so parked at False.
# MUST match the A1_RES flag in the test script.
A1_RES = False
a1_tag = "_a1res" if A1_RES else ""

destination_path_M_F           = cycle_dir + f"5dM_step_F_net0.001_{mstep_arch_tag}{a1_tag}{T_tag}{drop_tag}.pt"
destination_path_rtsnet_jointF = cycle_dir + f"5dRTSNet_falseF_joint0.001_new{mstep_arch_tag}{a1_tag}{T_tag}{drop_tag}.pt"
destination_path_M_F_joint     = cycle_dir + f"5dM_step_F_net_joint0.001_new{mstep_arch_tag}{a1_tag}{T_tag}{drop_tag}.pt"
###################
def _noisy_false_theta(F_true, noise_half=0.1, clip=0.14):
    theta = math.atan2(F_true[3, 2].item(), F_true[2, 2].item())
    lo = max(-noise_half, -clip - theta)
    hi = min( noise_half,  clip - theta)
    return theta + lo + (hi - lo) * torch.rand(1).item()

LOAD_DATA  = True  # True → skip generation, load data from data_path
OVERSAMPLE = 1.7   # generate this × more candidates than N_E/N_CV/N_T

# Trajectory physics flags (edit in Simulations/TDOA_2D/parameters.py):
#   USE_BOUNDARIES — True: enforce px/py/v bounds   False: unbounded
#   USE_REFLECTION — True: bounce at walls           False: reject (good_seq=0)

print("=" * 70)
print(f"2D TDOA RTSNet — {cycle}-cycle multi-dataset experiment (rotation theta model)")
print(f"  T={T}  T_test={T_test}  q2={q2}  r2={r2}")
print(f"  cycle={cycle}  theta_max=±{theta_max}  (each seq independent)  false F = make_F_block(0.0)")
print(f"  M-step net: {'BIG' if USE_BIG_MSTEP_NET else 'BASE'}  hidden={args.mstep_hidden_dim}")
print(f"  Microphones: {M_mics}   State dim: {m}   Obs dim: {n}")
print("=" * 70)

#########################################
###  Generate / load data             ###
#########################################

all_train_inputs  = []
all_train_targets = []
all_cv_inputs     = []
all_cv_targets    = []
all_test_inputs   = []
all_test_targets  = []
all_F_train_true  = []
all_F_cv_true     = []
all_F_test_true   = []
all_F_train_false = []
all_F_cv_false    = []
all_F_test_false  = []

if LOAD_DATA and os.path.exists(data_path):
    print(f"\nLoading saved data from {data_path} ...")
    _d = torch.load(data_path, weights_only=False, map_location=device)
    all_train_inputs  = _d["all_train_inputs"]
    all_train_targets = _d["all_train_targets"]
    all_cv_inputs     = _d["all_cv_inputs"]
    all_cv_targets    = _d["all_cv_targets"]
    all_test_inputs   = _d["all_test_inputs"]
    all_test_targets  = _d["all_test_targets"]
    all_F_train_true  = _d["all_F_train_true"]
    all_F_cv_true     = _d["all_F_cv_true"]
    all_F_test_true   = _d["all_F_test_true"]
    all_F_train_false = _d["all_F_train_false"]
    all_F_cv_false    = _d["all_F_cv_false"]
    all_F_test_false  = _d["all_F_test_false"]
    print("  Done.")
else:
    N_gen_E  = math.ceil(args.N_E  * OVERSAMPLE)
    N_gen_CV = math.ceil(args.N_CV * OVERSAMPLE)
    N_gen_T  = math.ceil(args.N_T  * OVERSAMPLE)

    # good_seq[i] == 1 means candidate i was valid in every dataset so far
    good_seq_train = [1] * N_gen_E
    good_seq_cv    = [1] * N_gen_CV
    good_seq_test  = [1] * N_gen_T

    # Raw (unfiltered) storage — shape [N_gen, n/m, T] per dataset
    raw_train_inputs  = [];  raw_train_targets = [];  raw_F_train = []
    raw_cv_inputs     = [];  raw_cv_targets    = [];  raw_F_cv    = []
    raw_test_inputs   = [];  raw_test_targets  = [];  raw_F_test  = []

    carry_train = None
    carry_cv    = None
    carry_test  = None

    # ── Independent random theta per segment ──
    # Every sequence-segment (each sequence, each dataset) draws its own theta
    # ~ Uniform(-theta_max, +theta_max), fresh per dataset. Covers sign flips
    # between datasets and the full ±theta_max range. Matches the test's
    # per-segment random draw exactly.
    import random as _rand
    def _rand_theta(N):
        return [(_rand.random() - 0.5) * 2 * theta_max for _ in range(N)]

    print(f"\nGenerating {cycle} datasets  "
          f"(N_gen train={N_gen_E}  cv={N_gen_CV}  test={N_gen_T}) ...")

    for k in range(cycle):
        print(f"  Dataset {k} ...", end="", flush=True)

        # fresh independent theta per sequence for this segment/dataset
        theta_tr = _rand_theta(N_gen_E)
        theta_cv = _rand_theta(N_gen_CV)
        theta_te = _rand_theta(N_gen_T)

        # theta_true_max=0.0 → theta = theta_base exactly (per-segment random)
        ti, tt, F_tr, v_tr = generate_dataset_raw_batch(
            N_gen_E,  T,      0.0, Q, R, x_init=carry_train, theta_base=theta_tr)
        ci, ct, F_cv, v_cv = generate_dataset_raw_batch(
            N_gen_CV, T,      0.0, Q, R, x_init=carry_cv,    theta_base=theta_cv)
        xi, xt, F_te, v_te = generate_dataset_raw_batch(
            N_gen_T,  T_test, 0.0, Q, R, x_init=carry_test,  theta_base=theta_te)

        for i in range(N_gen_E):
            if not v_tr[i]: good_seq_train[i] = 0
        for i in range(N_gen_CV):
            if not v_cv[i]: good_seq_cv[i] = 0
        for i in range(N_gen_T):
            if not v_te[i]: good_seq_test[i] = 0

        raw_train_inputs.append(ti);  raw_train_targets.append(tt);  raw_F_train.append(F_tr)
        raw_cv_inputs.append(ci);     raw_cv_targets.append(ct);     raw_F_cv.append(F_cv)
        raw_test_inputs.append(xi);   raw_test_targets.append(xt);   raw_F_test.append(F_te)

        carry_train = tt[:, :, -1]   # [N_gen_E,  m]
        carry_cv    = ct[:, :, -1]   # [N_gen_CV, m]
        carry_test  = xt[:, :, -1]   # [N_gen_T,  m]

        n_ok_tr = sum(good_seq_train)
        n_ok_cv = sum(good_seq_cv)
        n_ok_te = sum(good_seq_test)
        print(f"  good so far → train={n_ok_tr}/{N_gen_E}  "
              f"cv={n_ok_cv}/{N_gen_CV}  test={n_ok_te}/{N_gen_T}")

    # Select first N valid candidates
    idx_tr = [i for i in range(N_gen_E)  if good_seq_train[i]][:args.N_E]
    idx_cv = [i for i in range(N_gen_CV) if good_seq_cv[i]][:args.N_CV]
    idx_te = [i for i in range(N_gen_T)  if good_seq_test[i]][:args.N_T]

    if len(idx_tr) < args.N_E:
        raise RuntimeError(
            f"Not enough valid train sequences: got {len(idx_tr)}, need {args.N_E}. "
            f"Increase OVERSAMPLE (currently {OVERSAMPLE}).")
    if len(idx_cv) < args.N_CV:
        raise RuntimeError(
            f"Not enough valid CV sequences: got {len(idx_cv)}, need {args.N_CV}. "
            f"Increase OVERSAMPLE (currently {OVERSAMPLE}).")
    if len(idx_te) < args.N_T:
        raise RuntimeError(
            f"Not enough valid test sequences: got {len(idx_te)}, need {args.N_T}. "
            f"Increase OVERSAMPLE (currently {OVERSAMPLE}).")

    idx_tr_t = torch.tensor(idx_tr, dtype=torch.long)
    idx_cv_t = torch.tensor(idx_cv, dtype=torch.long)
    idx_te_t = torch.tensor(idx_te, dtype=torch.long)

    for k in range(cycle):
        all_train_inputs.append(raw_train_inputs[k][idx_tr_t])
        all_train_targets.append(raw_train_targets[k][idx_tr_t])
        all_cv_inputs.append(raw_cv_inputs[k][idx_cv_t])
        all_cv_targets.append(raw_cv_targets[k][idx_cv_t])
        all_test_inputs.append(raw_test_inputs[k][idx_te_t])
        all_test_targets.append(raw_test_targets[k][idx_te_t])

        all_F_train_true.append([raw_F_train[k][i] for i in idx_tr])
        all_F_cv_true.append(   [raw_F_cv[k][i]    for i in idx_cv])
        all_F_test_true.append( [raw_F_test[k][i]  for i in idx_te])

        all_F_train_false.append([make_F_block(0.0) for _ in idx_tr])
        all_F_cv_false.append(   [make_F_block(0.0) for _ in idx_cv])
        all_F_test_false.append( [make_F_block(0.0) for _ in idx_te])

    print(f"\n  Saving data to {data_path} ...")
    torch.save({
        "all_train_inputs":  all_train_inputs,
        "all_train_targets": all_train_targets,
        "all_cv_inputs":     all_cv_inputs,
        "all_cv_targets":    all_cv_targets,
        "all_test_inputs":   all_test_inputs,
        "all_test_targets":  all_test_targets,
        "all_F_train_true":  all_F_train_true,
        "all_F_cv_true":     all_F_cv_true,
        "all_F_test_true":   all_F_test_true,
        "all_F_train_false": all_F_train_false,
        "all_F_cv_false":    all_F_cv_false,
        "all_F_test_false":  all_F_test_false,
    }, data_path)
    print("  Done.")

print(f"\n  Train per dataset: {all_train_targets[0].size()}")
print(f"  CV per dataset:    {all_cv_targets[0].size()}")
print(f"  Test per dataset:  {all_test_targets[0].size()}")

# Plot sequences 0-3 across all datasets — 4 panels each
import matplotlib.patches as mpatches
from Simulations.TDOA_2D.parameters import mic_positions
_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

print("\nPlotting sequences 0-3 across all datasets ...")
for seq_idx in range(4):
    fig = plt.figure(figsize=(16, 12))
    ax_traj = fig.add_subplot(2, 2, 1)
    ax_pos  = fig.add_subplot(2, 2, 2)
    ax_vel  = fig.add_subplot(2, 2, 3)
    ax_tdoa = fig.add_subplot(2, 2, 4)

    t_offset = 0
    for k in range(cycle):
        states_k = all_train_targets[k][seq_idx].cpu()   # [m, T]
        obs_k    = all_train_inputs[k][seq_idx].cpu()    # [n, T]
        t_ax     = torch.arange(t_offset, t_offset + T).float()
        col      = _colors[k % len(_colors)]

        ax_traj.plot(states_k[0], states_k[1], color=col, label=f'ds{k}')
        ax_traj.scatter(states_k[0, 0],  states_k[1, 0],  color=col, marker='o', s=40, zorder=5)
        ax_traj.scatter(states_k[0, -1], states_k[1, -1], color=col, marker='x', s=60, zorder=5)

        ax_pos.plot(t_ax, states_k[0], color=col, label=f'ds{k} px')
        ax_pos.plot(t_ax, states_k[1], color=col, linestyle='--')

        ax_vel.plot(t_ax, states_k[2], color=col, label=f'ds{k} vx')
        ax_vel.plot(t_ax, states_k[3], color=col, linestyle='--')

        for i in range(obs_k.size(0)):
            ax_tdoa.plot(t_ax, obs_k[i], color=col, alpha=0.6,
                         label=f'ds{k} ch{i}' if k == 0 else '_')

        if k > 0:
            for ax in [ax_pos, ax_vel, ax_tdoa]:
                ax.axvline(x=t_offset, color='k', linestyle='--', alpha=0.35)
        t_offset += T

    for idx, mic in enumerate(mic_positions):
        ax_traj.scatter(mic[0].item(), mic[1].item(), marker='^', color='black', s=80, zorder=6)
        ax_traj.annotate(f'm{idx}', (mic[0].item(), mic[1].item()), textcoords='offset points', xytext=(4, 4), fontsize=7)

    _rect = mpatches.Rectangle(
        (PX_MIN, PY_MIN), PX_MAX - PX_MIN, PY_MAX - PY_MIN,
        linewidth=1.5, edgecolor='red', facecolor='lightyellow',
        linestyle='--', zorder=2, alpha=0.3, label='valid region',
    )
    ax_traj.add_patch(_rect)

    ax_traj.set_xlabel('p_x');    ax_traj.set_ylabel('p_y')
    ax_traj.set_xlim(-20, 20);    ax_traj.set_ylim(-10, 20)
    ax_traj.set_title('2D trajectory (o=start, x=end per dataset)')
    ax_traj.legend(fontsize=7);   ax_traj.grid(True, alpha=0.4)

    ax_pos.set_ylabel('position');  ax_pos.set_title('p_x (solid) & p_y (dashed) vs time')
    ax_pos.legend(fontsize=7, ncol=cycle); ax_pos.grid(True, alpha=0.4)

    ax_vel.set_ylabel('velocity');  ax_vel.set_title('v_x (solid) & v_y (dashed) vs time')
    ax_vel.legend(fontsize=7, ncol=cycle); ax_vel.grid(True, alpha=0.4)
    ax_vel.set_xlabel('time step')

    ax_tdoa.set_ylabel('TDOA');   ax_tdoa.set_title('TDOA observations vs time')
    ax_tdoa.legend(fontsize=7);   ax_tdoa.grid(True, alpha=0.4)
    ax_tdoa.set_xlabel('time step')

    fig.suptitle(f'Sequence {seq_idx} — all datasets (train)  |  dashed = dataset boundary', fontsize=12)
    plt.tight_layout()
    plt.savefig(cycle_dir + f"seq{seq_idx}_datasets.png", dpi=150, bbox_inches="tight")
    plt.close()

#########################################
###  System models                     ###
#########################################
# H_prior: linearized h at x0 — used as fixed GRU feature in RTSNet (FC9)
# h:       nonlinear TDOA function — called inside RTSNet for innovations + MNet statistics
H_prior = h_jacobian(m1x_0.reshape(-1))   # [n, m]
F_init  = make_F_block(0.0)   # theta=0 starting point (false F)
f_init  = make_f(F_init)

sys_model_true = SystemModel(f=f_init, Q=Q, h=h, R=R,
                             T=T, T_test=T_test, m=m, n=n, H=H_prior,
                             prior_S=torch.eye(n, device=device))
sys_model_true.F       = F_init
sys_model_true.F_train = all_F_train_true    # [cycle][group_idx]
sys_model_true.F_valid = all_F_cv_true
sys_model_true.F_test  = all_F_test_true
sys_model_true.InitSequence(m1x_0, m2x_0)

sys_model_false = SystemModel(f=f_init, Q=Q, h=h, R=R,
                              T=T, T_test=T_test, m=m, n=n, H=H_prior,
                              prior_S=torch.eye(n, device=device))
sys_model_false.F       = F_init
sys_model_false.F_train = all_F_train_false
sys_model_false.F_valid = all_F_cv_false
sys_model_false.F_test  = all_F_test_false
sys_model_false.InitSequence(m1x_0, m2x_0)
sys_model_false.F_train_TRUE = all_F_train_true
sys_model_false.F_valid_TRUE = all_F_cv_true
sys_model_false.F_test_TRUE  = all_F_test_true


########################################
### BiGRU baseline                   ###
########################################
print("\nBiGRU — training on all cycle datasets ...")
train_bigru_smoother(
    train_input=all_train_inputs,
    train_target=all_train_targets,
    cv_input=all_cv_inputs,
    cv_target=all_cv_targets,
    n=n, m=m,
    save_path=destination_path_bigru,
    device=device,
    epochs=300,
    batch_size=10,
    lr=args.lr,

)
sthrhtrtsthrrthtrh
#######################
### RTSNet true-F   ###
#######################
print("\nRTSNet TRUE-F — cycle fine-tuning")
RTSNet_model_true = RTSNetNN()
RTSNet_model_true.NNBuild(sys_model_true, args)
RTSNet_Pipeline_true = Pipeline(strTime, "RTSNet", "RTSNet_TDOA_trueF")
RTSNet_Pipeline_true.setssModel(sys_model_true)
RTSNet_Pipeline_true.setModel(RTSNet_model_true, args)
RTSNet_Pipeline_true.setTrainingParams(args)

RTSNet_Pipeline_true.train_RTS_net_3_datasets(
    sys_model_true,
    all_cv_inputs,    all_cv_targets,
    all_train_inputs, all_train_targets,
    destination_path_RTS=destination_path_rtsnet_true,
    load_path_RTS=load_path_rtsnet_true,
    generate_f=True,
    datasets=cycle,
    obs_mask=obs_mask,   # dense now (obs_mask=None)
)

sys_model_true.F_test = all_F_test_true   # [cycle][group_idx]
[MSE_test_arr_true, MSE_test_avg_true, MSE_test_dB_avg_true,
 rtsnet_out_true, RunTime_true] = RTSNet_Pipeline_true.NNTest_3_datasets(
    sys_model_true,
    all_test_inputs,
    all_test_targets,
    destination_path_rtsnet_true,
    generate_f=True,
    datasets=cycle,
    obs_mask=obs_mask,
)

#######################
### RTSNet false-F  ###
#######################
print("\nRTSNet FALSE-F — cycle fine-tuning")
RTSNet_model_false = RTSNetNN()
RTSNet_model_false.NNBuild(sys_model_false, args)
RTSNet_Pipeline_false = Pipeline(strTime, "RTSNet", "RTSNet_TDOA_falseF")
RTSNet_Pipeline_false.setssModel(sys_model_false)
RTSNet_Pipeline_false.setModel(RTSNet_model_false, args)
RTSNet_Pipeline_false.setTrainingParams(args)

RTSNet_Pipeline_false.train_RTS_net_3_datasets(
    sys_model_false,
    all_cv_inputs,    all_cv_targets,
    all_train_inputs, all_train_targets,
    destination_path_RTS=destination_path_rtsnet_false,
    load_path_RTS=load_path_rtsnet_false,   # warm-start from old r10 RTSNet-false
    generate_f=True,
    datasets=cycle,
    obs_mask=obs_mask,   # dense now (obs_mask=None); MNet/joint load this checkpoint
)

sys_model_false.F_test = all_F_test_false   # [cycle][group_idx]
[MSE_test_arr_false, MSE_test_avg_false, MSE_test_dB_avg_false,
 rtsnet_out_false, RunTime_false] = RTSNet_Pipeline_false.NNTest_3_datasets(
    sys_model_false,
    all_test_inputs,
    all_test_targets,
    destination_path_rtsnet_false,
    generate_f=True,
    datasets=cycle,
    obs_mask=obs_mask,
)

#############################
### MNet cycle training    ###
#############################
# Standalone MNet pretraining — RTSNet is FROZEN here, so the M-step net
# learns to recover F purely from smoothing statistics before joint
# fine-tuning starts moving RTSNet too. Fresh init (load_mnet=None):
# warm-starting from the r=1 joint MNet checkpoint was found to produce
# near-zero / wrong-signed ΔF (collapsed by the old lambda_F=0.1 run),
# which then poisoned joint training. Train clean instead.
print(f"\nMNet {cycle}-cycle training (standalone, frozen RTSNet) ...")

RTSNet_Pipeline_false.train_F_mstep_net_3_datasets(
    sys_model_false,
    all_cv_inputs,    all_cv_targets,
    all_train_inputs, all_train_targets,
    destination_path_M=destination_path_M_F,
    load_path_RTS=destination_path_rtsnet_false,   # freshly-trained RTSNet-false above
    # Warm-start from a working MNet (NOT fresh) — a random net emits uncalibrated
    # ΔF that pushes F expansive and NaNs every epoch. This one is calibrated.
    load_mnet="RTSNet/tdoa_2d/3mics/r10/5cycle/5dM_step_F_net_joint0.001_newbig_mstep.pt",
    num_em_iters=num_em_iters,
    alpha=(0.5, 1.0, 0.85),   # EM1 decent, EM2 priority (was (0.3,1.0) — EM1 under-trained)
    lambda_F=1e-3,
    generate_f=True,
    datasets=cycle,
    propagate_F=False,          # each dataset restarts F estimation from the base
    F_init=make_F_block(0.0),   # ...F (theta=0), NOT the previous dataset's estimate
    A1_res=A1_RES,
    use_big_mstep_net=USE_BIG_MSTEP_NET,
    mstep_hidden_dim=args.mstep_hidden_dim,
    obs_mask=obs_mask,
)

###############################
### Joint cycle training     ###
###############################
print(f"\nJoint {cycle}-cycle training ...")

RTSNet_Pipeline_false.train_F_mstep_net_3_datasets_joint(
    sys_model_false,
    all_cv_inputs,    all_cv_targets,
    all_train_inputs, all_train_targets,
    destination_path_M=destination_path_M_F_joint,
    destination_path_RTS=destination_path_rtsnet_jointF,
    load_path_RTS=destination_path_rtsnet_false,   # freshly-trained RTSNet-false above
    load_mnet=destination_path_M_F,   # initialised by MNet training above
    num_em_iters=num_em_iters,
    alpha=(0.5, 1.0, 0.85),   # EM1 decent, EM2 priority (was (0.1,1.0) — EM1 under-trained)
    lambda_F=1e-3,
    generate_f=True,
    datasets=cycle,
    propagate_F=False,          # each dataset restarts F estimation from the base
    F_init=make_F_block(0.0),   # ...F (theta=0), NOT the previous dataset's estimate
    A1_res=A1_RES,
    use_big_mstep_net=USE_BIG_MSTEP_NET,
    mstep_hidden_dim=args.mstep_hidden_dim,
    obs_mask=obs_mask,
)
###############################
### Test MNet + Joint       ###
###############################
sys_model_false.F_test      = all_F_test_false   # [cycle][group_idx]
sys_model_false.F_test_TRUE = all_F_test_true     # [cycle][group_idx]

# Standalone MNet not trained — skip its test
# print("\nTesting MNet ...")
# [MSE_test_arr_mnet, ...] = RTSNet_Pipeline_false.test_F_mstep_net_3_datasets(...)

print("\nTesting joint ...")
[MSE_test_arr_joint, MSE_test_avg_joint, MSE_test_dB_avg_joint,
 rtsnet_out_joint, RunTime_joint] = RTSNet_Pipeline_false.test_F_mstep_net_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    destination_path_rtsnet_jointF, destination_path_M_F_joint,
    num_em_iters=num_em_iters, generate_f=True, datasets=cycle, propagate_F=False, A1_res=A1_RES,   # F resets to theta=0 each dataset (matches training)
    obs_mask=obs_mask,
)

# BiGRU training is commented out above, so its checkpoint may not exist.
# Only test it when the file is actually present — otherwise skip cleanly.
HAS_BIGRU = os.path.exists(destination_path_bigru)
bigru_outputs   = []   # list[dataset] of [N_T, m, T]
mse_bigru_avg_db = None
if HAS_BIGRU:
    print("\nBiGRU — testing ...")
    bigru_model = torch.load(destination_path_bigru, weights_only=False, map_location=device)
    bigru_model.eval()

    mse_bigru_per_dataset = torch.zeros(cycle)
    with torch.no_grad():
        for k in range(cycle):
            y    = all_test_inputs[k].to(device)
            tgt  = all_test_targets[k].to(device)
            xhat = bigru_model(y)          # [N_T, m, T]
            mse_bigru_per_dataset[k] = loss_fn(xhat, tgt)
            bigru_outputs.append(xhat.cpu())

    mse_bigru_avg_db = 10 * math.log10(mse_bigru_per_dataset.mean().item())
else:
    print(f"\nBiGRU — skipped (no checkpoint at {destination_path_bigru}).")

########################################
### Plot                              ###
########################################
print("\nPlotting test sequence 0 ...")
t_axis = torch.arange(T_test)
states = all_test_targets[0][0]
plt.figure(figsize=(12, 5))
plt.plot(t_axis, states.cpu()[1],                              linewidth=2.5, label="true p_y")
plt.plot(t_axis, rtsnet_out_joint[0].cpu()[1], "-.",          linewidth=2,   label=f"Joint {cycle}-cycle")
if HAS_BIGRU:
    plt.plot(t_axis, bigru_outputs[0][0][1],    "--",          linewidth=2,   label="BiGRU")
plt.xlabel("time")
plt.ylabel("y position")
plt.title(f"TDOA tracking: y position — {cycle}-cycle  r2={r2}")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(cycle_dir + "training_result.png", dpi=150, bbox_inches="tight")
plt.close()

########################################
### Results summary                   ###
########################################
print("\n" + "=" * 70)
print(f"RESULTS SUMMARY  (cycle={cycle}  q2={q2}  r2={r2})")
print("=" * 70)
print(f"  Joint {cycle}-cycle (avg)     : {MSE_test_dB_avg_joint.item():.2f} dB")
if HAS_BIGRU:
    print(f"  BiGRU          (avg)          : {mse_bigru_avg_db:.2f} dB")
print("=" * 70)
