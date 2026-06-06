import os
import math
import torch
import torch.nn as nn
import matplotlib
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
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 50
args.T_test = 50
### training parameters
args.n_steps = 400
args.n_batch = 20
args.lr = 1e-3
args.wd = 1e-3

T      = args.T
T_test = args.T_test

### noise levelsbut
q2 = 0.001
r2 = 1

### cycle: number of datasets
cycle = 5
# Each sequence draws theta independently from Uniform(-theta_max, +theta_max).
# No dataset-level base theta — every sequence is fully independent.
theta_max = 0.12   # drawn range = Uniform(-0.12, +0.12), covers test ±0.10

### EM iterations
num_em_iters = 2

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

### paths
save_dir  = "RTSNet/tdoa_2d/3mics/r1/cycle1/"
cycle_dir = save_dir + f"{cycle}cycle/"
os.makedirs(save_dir,  exist_ok=True)
os.makedirs(cycle_dir, exist_ok=True)

# Load paths: pre-trained networks from the 1-dataset experiment

load_path_rtsnet_true  = save_dir + "RTSNet_true0.001.pt"
load_path_rtsnet_false = save_dir + "RTSNet_false0.001.pt"
load_path_M_F = save_dir + "M_step_F_net0.001.pt"
load_path_rtsnet_F_joint = save_dir + "RTSNet_falseF_joint0.001.pt"
load_path_M_F_joint = save_dir + "M_step_F_net_joint0.001.pt"

# Cycle-dataset experiment outputs
destination_path_rtsnet_true   = cycle_dir + "5dRTSNet_true0.001.pt"
destination_path_rtsnet_false  = cycle_dir + "5dRTSNet_false0.001.pt"
destination_path_M_F           = cycle_dir + "5dM_step_F_net0.001.pt"
destination_path_rtsnet_jointF = cycle_dir + "5dRTSNet_falseF_joint0.001.pt"
destination_path_M_F_joint     = cycle_dir + "5dM_step_F_net_joint0.001.pt"
destination_path_bigru         = cycle_dir + "BiGRU.pt"
# destination_path_rtsnet_true   = save_dir + "RTSNet_true.pt"
# destination_path_rtsnet_false  = save_dir + "RTSNet_false.pt"
# destination_path_M_F           = save_dir + "M_step_F_net.pt"
# destination_path_rtsnet_jointF = save_dir + "RTSNet_falseF_joint.pt"
# destination_path_M_F_joint     = save_dir + "M_step_F_net_joint.pt"

data_path = save_dir + "training_3_dataset_data.pt"

###################
###    FLAGS     ###
###################
LOAD_DATA  = False  # True → skip generation, load data from data_path
OVERSAMPLE = 1.5   # generate this × more candidates than N_E/N_CV/N_T

# Trajectory physics flags (edit in Simulations/TDOA_2D/parameters.py):
#   USE_BOUNDARIES — True: enforce px/py/v bounds   False: unbounded
#   USE_REFLECTION — True: bounce at walls           False: reject (good_seq=0)

print("=" * 70)
print(f"2D TDOA RTSNet — {cycle}-cycle multi-dataset experiment (rotation theta model)")
print(f"  T={T}  T_test={T_test}  q2={q2}  r2={r2}")
print(f"  cycle={cycle}  theta_max=±{theta_max}  (each seq independent)  false F = make_F_block(0.0)")
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

    print(f"\nGenerating {cycle} datasets  "
          f"(N_gen train={N_gen_E}  cv={N_gen_CV}  test={N_gen_T}) ...")

    for k in range(cycle):
        print(f"  Dataset {k} ...", end="", flush=True)

        ti, tt, F_tr, v_tr = generate_dataset_raw_batch(
            N_gen_E,  T,      2*theta_max, Q, R, x_init=carry_train)
        ci, ct, F_cv, v_cv = generate_dataset_raw_batch(
            N_gen_CV, T,      2*theta_max, Q, R, x_init=carry_cv)
        xi, xt, F_te, v_te = generate_dataset_raw_batch(
            N_gen_T,  T_test, 2*theta_max, Q, R, x_init=carry_test)

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
    plt.show()

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

# RTSNet_Pipeline_true.train_RTS_net_3_datasets(
#     sys_model_true,
#     all_cv_inputs,    all_cv_targets,
#     all_train_inputs, all_train_targets,
#     destination_path_RTS=destination_path_rtsnet_true,
#     load_path_RTS=load_path_rtsnet_true,
#     generate_f=True,
#     datasets=cycle,
# )

# sys_model_true.F_test = all_F_test_true   # [cycle][group_idx]
# [MSE_test_arr_true, MSE_test_avg_true, MSE_test_dB_avg_true,
#  rtsnet_out_true, RunTime_true] = RTSNet_Pipeline_true.NNTest_3_datasets(
#     sys_model_true,
#     all_test_inputs,
#     all_test_targets,
#     destination_path_rtsnet_true,
#     generate_f=True,
#     datasets=cycle,
# )

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

# RTSNet_Pipeline_false.train_RTS_net_3_datasets(
#     sys_model_false,
#     all_cv_inputs,    all_cv_targets,
#     all_train_inputs, all_train_targets,
#     destination_path_RTS=destination_path_rtsnet_false,
#     load_path_RTS=load_path_rtsnet_false,
#     generate_f=True,
#     datasets=cycle,
# )

# sys_model_false.F_test = all_F_test_false   # [cycle][group_idx]
# [MSE_test_arr_false, MSE_test_avg_false, MSE_test_dB_avg_false,
#  rtsnet_out_false, RunTime_false] = RTSNet_Pipeline_false.NNTest_3_datasets(
#     sys_model_false,
#     all_test_inputs,
#     all_test_targets,
#     destination_path_rtsnet_false,
#     generate_f=True,
#     datasets=cycle,
# )

#############################
### MNet cycle training    ###
#############################
print(f"\nMNet {cycle}-cycle training ...")

# RTSNet_Pipeline_false.train_F_mstep_net_3_datasets(
#     sys_model_false,
#     all_cv_inputs,    all_cv_targets,
#     all_train_inputs, all_train_targets,
#     destination_path_M=destination_path_M_F,
#     load_path_RTS=destination_path_rtsnet_false,
#     load_mnet=load_path_M_F,       # initialise from training-1 MNet
#     num_em_iters=num_em_iters,
#     alpha=(0.3, 1.0, 0.85),
#     lambda_F=1e-3,
#     generate_f=True,
#     datasets=cycle,
#     propagate_F=False,
# )

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
    load_path_RTS=destination_path_rtsnet_false,
    load_mnet=destination_path_M_F,   # initialised by MNet training above
    num_em_iters=num_em_iters,
    alpha=(0.3, 1.0, 0.85),
    lambda_F=1e-3,
    generate_f=True,
    datasets=cycle,
    propagate_F=False,
)

# RTSNet_Pipeline_false.train_F_mstep_net_3_datasets_joint(
#     sys_model_false,
#     all_cv_inputs,    all_cv_targets,
#     all_train_inputs, all_train_targets,
#     destination_path_M=destination_path_M_F_joint,
#     destination_path_RTS=destination_path_rtsnet_jointF,
#     load_path_RTS=destination_path_rtsnet_jointF,
#     load_mnet=destination_path_M_F_joint,   # initialised by MNet training above
#     num_em_iters=num_em_iters,
#     alpha=(0.3, 1.0, 0.85),
#     lambda_F=1e-3,
#     generate_f=True,
#     datasets=cycle,
#     propagate_F=False,
# )
###############################
### Test MNet + Joint       ###
###############################
sys_model_false.F_test      = all_F_test_false   # [cycle][group_idx]
sys_model_false.F_test_TRUE = all_F_test_true     # [cycle][group_idx]

print("\nTesting MNet ...")
[MSE_test_arr_mnet, MSE_test_avg_mnet, MSE_test_dB_avg_mnet,
 rtsnet_out_mnet, RunTime_mnet] = RTSNet_Pipeline_false.test_F_mstep_net_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    destination_path_rtsnet_false, destination_path_M_F,
    num_em_iters=num_em_iters, generate_f=True, datasets=cycle,
)

print("\nTesting joint ...")
[MSE_test_arr_joint, MSE_test_avg_joint, MSE_test_dB_avg_joint,
 rtsnet_out_joint, RunTime_joint] = RTSNet_Pipeline_false.test_F_mstep_net_3_datasets(
    sys_model_false, all_test_inputs, all_test_targets,
    destination_path_rtsnet_jointF, destination_path_M_F_joint,
    num_em_iters=num_em_iters, generate_f=True, datasets=cycle,
)

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
    epochs=args.n_steps,
    batch_size=args.n_batch,
    lr=args.lr,
)

print("\nBiGRU — testing ...")
bigru_model = torch.load(destination_path_bigru, weights_only=False, map_location=device)
bigru_model.eval()

bigru_outputs         = []   # list[dataset] of [N_T, m, T]
mse_bigru_per_dataset = torch.zeros(cycle)
with torch.no_grad():
    for k in range(cycle):
        y    = all_test_inputs[k].to(device)
        tgt  = all_test_targets[k].to(device)
        xhat = bigru_model(y)          # [N_T, m, T]
        mse_bigru_per_dataset[k] = loss_fn(xhat, tgt)
        bigru_outputs.append(xhat.cpu())

mse_bigru_avg_db = 10 * math.log10(mse_bigru_per_dataset.mean().item())

########################################
### Plot                              ###
########################################
print("\nPlotting test sequence 0 ...")
t_axis = torch.arange(T_test)
states = all_test_targets[0][0]
plt.figure(figsize=(12, 5))
plt.plot(t_axis, states.cpu()[1],                              linewidth=2.5, label="true p_y")
plt.plot(t_axis, rtsnet_out_true[0].cpu()[1],                 linewidth=2,   label="RTSNet true F")
plt.plot(t_axis, rtsnet_out_false[0].cpu()[1],                linewidth=2,   label="RTSNet false F")
plt.plot(t_axis, rtsnet_out_mnet[0].cpu()[1],                 linewidth=2,   label=f"MNet {cycle}-cycle")
plt.plot(t_axis, rtsnet_out_joint[0].cpu()[1], "-.",          linewidth=2,   label=f"Joint {cycle}-cycle")  # type: ignore[index]
plt.plot(t_axis, bigru_outputs[0][0][1],        "--",          linewidth=2,   label="BiGRU")
plt.xlabel("time")
plt.ylabel("y position")
plt.title(f"TDOA tracking: y position — {cycle}-cycle")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

########################################
### Results summary                   ###
########################################
print("\n" + "=" * 70)
print(f"RESULTS SUMMARY  (cycle={cycle}, a_range={a_range}, b_range={b_range})")
print("=" * 70)
print(f"  RTSNet TRUE-F  (avg)          : {MSE_test_dB_avg_true.item():.2f} dB")
print(f"  RTSNet FALSE-F (avg)          : {MSE_test_dB_avg_false.item():.2f} dB")
print(f"  MNet {cycle}-cycle (avg)      : {MSE_test_dB_avg_mnet.item():.2f} dB")  # type: ignore[union-attr]
print(f"  Joint {cycle}-cycle (avg)     : {MSE_test_dB_avg_joint.item():.2f} dB")
print(f"  BiGRU          (avg)          : {mse_bigru_avg_db:.2f} dB")
print("=" * 70)
