import os
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
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
    generate_dataset_random_theta,
    generate_false_F_list,
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
args.T = 30
args.T_test = 30
### training parameters
args.n_steps = 400
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

T      = args.T
T_test = args.T_test

### noise levels
q2 = 0.001
r2 = 1

### cycle: number of datasets
cycle = 5
# Max theta per dataset [rad] — each group of 10 sequences draws theta ~ Uniform(-max/2, +max/2)
theta_changed_list = [3, 3, 3, 3, 3]
assert len(theta_changed_list) == cycle

### false F mismatch — always assume theta=0 (straight-line motion)
theta_false = 0.0   # [rad]

### EM iterations
num_em_iters = 2

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

### paths
save_dir  = "RTSNet/tdoa_2d/1/"
cycle_dir = save_dir + f"{cycle}cycle/"
os.makedirs(save_dir,  exist_ok=True)
os.makedirs(cycle_dir, exist_ok=True)

# Load paths: pre-trained networks from the 1-dataset experiment

load_path_rtsnet_true  = save_dir + "RTSNet_true.pt"
load_path_rtsnet_false = save_dir + "RTSNet_false.pt"
load_path_M_F = save_dir + "M_step_F_net.pt"
load_path_rtsnet_F_joint = save_dir + "RTSNet_falseF_joint_arge_f_loss.pt"
load_path_M_F_joint = save_dir + "M_step_F_net_joint_large_f_loss.pt"

# Cycle-dataset experiment outputs
destination_path_rtsnet_true   = cycle_dir + "RTSNet_true.pt"
destination_path_rtsnet_false  = cycle_dir + "RTSNet_false.pt"
destination_path_M_F           = cycle_dir + "M_step_F_net.pt"
destination_path_rtsnet_jointF = cycle_dir + "RTSNet_falseF_joint.pt"
destination_path_M_F_joint     = cycle_dir + "M_step_F_net_joint.pt"
destination_path_bigru         = cycle_dir + "BiGRU.pt"
# destination_path_rtsnet_true   = save_dir + "RTSNet_true.pt"
# destination_path_rtsnet_false  = save_dir + "RTSNet_false.pt"
# destination_path_M_F           = save_dir + "M_step_F_net.pt"
# destination_path_rtsnet_jointF = save_dir + "RTSNet_falseF_joint.pt"
# destination_path_M_F_joint     = save_dir + "M_step_F_net_joint.pt"
print("=" * 70)
print(f"2D TDOA RTSNet — {cycle}-cycle multi-dataset experiment")
print(f"  T={T}  T_test={T_test}  q2={q2}  r2={r2}")
print(f"  cycle={cycle}  theta_changed_list={theta_changed_list}  theta_false={theta_false}")
print(f"  Microphones: {M_mics}   State dim: {m}   Obs dim: {n}")
print("=" * 70)

#########################################
###  Generate data — nested lists only ###
#########################################
# Matches ori_main_lor_DT_3_datasets_train.py:
#   all_train_inputs[data]        -> [N_E, n, T]
#   all_F_train_true[data][group] -> F matrix for that group of 10 sequences

print(f"\nGenerating {cycle} datasets ...")

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

# Carry state between datasets: last state of last trajectory and last group theta
carry_x_train  = None;  carry_theta_train  = None
carry_x_cv     = None;  carry_theta_cv     = None
carry_x_test   = None;  carry_theta_test   = None

for k in range(cycle):
    theta_changed = theta_changed_list[k]
    print(f"  Dataset {k}: theta ~ theta_prev + Uniform(-{theta_changed/2:.3f}, +{theta_changed/2:.3f}) rad per group")
    _tr0 = carry_theta_train[0] if isinstance(carry_theta_train, list) else 0.0
    _cv0 = carry_theta_cv[0]    if isinstance(carry_theta_cv,    list) else 0.0
    _te0 = carry_theta_test[0]  if isinstance(carry_theta_test,  list) else 0.0
    print(f"            theta_base[0]: train={_tr0:.4f}  cv={_cv0:.4f}  test={_te0:.4f}")

    ti, tt, th_tr, F_tr_t = generate_dataset_random_theta(args.N_E,  T,      theta_changed, Q, R, x_init=carry_x_train, theta_base=carry_theta_train)
    ci, ct, th_cv, F_cv_t = generate_dataset_random_theta(args.N_CV, T,      theta_changed, Q, R, x_init=carry_x_cv,    theta_base=carry_theta_cv)
    xi, xt, th_te, F_te_t = generate_dataset_random_theta(args.N_T,  T_test, theta_changed, Q, R, x_init=carry_x_test,  theta_base=carry_theta_test)

    # Update carry: last state of EVERY sequence, last group's theta
    carry_x_train  = tt[:, :, -1];  carry_theta_train  = th_tr   # [N_E,  m]
    carry_x_cv     = ct[:, :, -1];  carry_theta_cv     = th_cv   # [N_CV, m]
    carry_x_test   = xt[:, :, -1];  carry_theta_test   = th_te   # [N_T,  m]

    # False F is always theta=0 (straight-line assumption) for every dataset
    F_tr_f = [make_F_block(0.0) for _ in range(len(th_tr))]
    F_cv_f = [make_F_block(0.0) for _ in range(len(th_cv))]

    # test always starts from theta=0 (matches MNet training starting point)
    F_te_f = [make_F_block(0.0).to(device) for _ in range(args.N_T)]

    all_train_inputs.append(ti);   all_train_targets.append(tt)
    all_cv_inputs.append(ci);      all_cv_targets.append(ct)
    all_test_inputs.append(xi);    all_test_targets.append(xt)

    all_F_train_true.append(F_tr_t);   all_F_train_false.append(F_tr_f)
    all_F_cv_true.append(F_cv_t);      all_F_cv_false.append(F_cv_f)
    all_F_test_true.append(F_te_t);    all_F_test_false.append(F_te_f)

print(f"  Train per dataset: {all_train_targets[0].size()}")
print(f"  CV per dataset:    {all_cv_targets[0].size()}")
print(f"  Test per dataset:  {all_test_targets[0].size()}")

#########################################
###  Data sanity check                 ###
#########################################
# Verify x_0 carry: last state of seq-0 in dataset k-1 must equal
# the x_0 used to start seq-0 in dataset k.
print("\nData sanity check — x_0 carry for sequence 0 (train):")
print(f"  ds0 x_0 (fixed start): {m1x_0.reshape(-1).cpu().numpy().round(3)}")
for k in range(cycle):
    x_end = all_train_targets[k][0, :, -1].cpu()
    print(f"  ds{k} last state : [{x_end[0]:.3f}, {x_end[1]:.3f}, {x_end[2]:.3f}, {x_end[3]:.3f}]", end="")
    if k < cycle - 1:
        x_next_0 = carry_x_train[0].cpu() if k == cycle - 1 else all_train_targets[k][0, :, -1].cpu()
        print(f"  ← x_0 for ds{k+1}", end="")
    print()

print("\nData sanity check — theta per sequence (train, first 5 sequences, all datasets):")
for k in range(cycle):
    thetas_k = [f"{all_F_train_true[k][s][2,2].item():.3f}" for s in range(min(5, args.N_E))]
    print(f"  ds{k} F[2,2] (cos θ) seq0-4: {thetas_k}")

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
    _plot_path = os.path.abspath(save_dir + f'seq{seq_idx}_data_sanity.png')
    plt.savefig(_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {_plot_path}")
    # os.startfile(_plot_path)
    if os.name == "nt":
        os.startfile(_plot_path)

#########################################
###  System models                     ###
#########################################
# H_prior: linearized h at x0 — used as fixed GRU feature in RTSNet (FC9)
# h:       nonlinear TDOA function — called inside RTSNet for innovations + MNet statistics
H_prior = h_jacobian(m1x_0.reshape(-1))   # [n, m]
F_init  = make_F_block(0.0)
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
    # sys_model_true,
    # all_cv_inputs,    all_cv_targets,
    # all_train_inputs, all_train_targets,
    # destination_path_RTS=destination_path_rtsnet_true,
    # load_path_RTS=load_path_rtsnet_true,
    # generate_f=True,
    # datasets=cycle,
# )

sys_model_true.F_test = all_F_test_true   # [cycle][group_idx]
[MSE_test_arr_true, MSE_test_avg_true, MSE_test_dB_avg_true,
 rtsnet_out_true, RunTime_true] = RTSNet_Pipeline_true.NNTest_3_datasets(
    sys_model_true,
    all_test_inputs,
    all_test_targets,
    destination_path_rtsnet_true,
    generate_f=True,
    datasets=cycle,
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

# RTSNet_Pipeline_false.train_RTS_net_3_datasets(
#     sys_model_false,
#     all_cv_inputs,    all_cv_targets,
#     all_train_inputs, all_train_targets,
#     destination_path_RTS=destination_path_rtsnet_false,
#     load_path_RTS=load_path_rtsnet_false,
#     generate_f=True,
#     datasets=cycle,
# )

sys_model_false.F_test = all_F_test_false   # [cycle][group_idx]
[MSE_test_arr_false, MSE_test_avg_false, MSE_test_dB_avg_false,
 rtsnet_out_false, RunTime_false] = RTSNet_Pipeline_false.NNTest_3_datasets(
    sys_model_false,
    all_test_inputs,
    all_test_targets,
    destination_path_rtsnet_false,
    generate_f=True,
    datasets=cycle,
)

#############################
### MNet cycle training    ###
#############################
print(f"\nMNet {cycle}-cycle training ...")

RTSNet_Pipeline_false.train_F_mstep_net_3_datasets(
    sys_model_false,
    all_cv_inputs,    all_cv_targets,
    all_train_inputs, all_train_targets,
    destination_path_M=destination_path_M_F,
    load_path_RTS=destination_path_rtsnet_false,
    load_mnet=load_path_M_F,
    num_em_iters=num_em_iters,
    alpha=(0.3, 1.0, 0.85),
    lambda_F=1e-3,
    generate_f=True,
    datasets=cycle,
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
    load_path_RTS=destination_path_rtsnet_false,
    load_mnet=load_path_M_F_joint,
    num_em_iters=num_em_iters,
    alpha=(0.3, 1.0, 0.85),
    lambda_F=1e-3,
    generate_f=True,
    datasets=cycle,
)

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
plt.savefig(cycle_dir + "tdoa_rtsnet_y_position.png", dpi=250)
print(f"  Saved: {cycle_dir}tdoa_rtsnet_y_position.png")

########################################
### Results summary                   ###
########################################
print("\n" + "=" * 70)
print(f"RESULTS SUMMARY  (cycle={cycle}, theta_changed_list={theta_changed_list})")
print("=" * 70)
print(f"  RTSNet TRUE-F  (avg)          : {MSE_test_dB_avg_true.item():.2f} dB")
print(f"  RTSNet FALSE-F (avg)          : {MSE_test_dB_avg_false.item():.2f} dB")
print(f"  MNet {cycle}-cycle (avg)      : {MSE_test_dB_avg_mnet.item():.2f} dB")  # type: ignore[union-attr]
print(f"  Joint {cycle}-cycle (avg)     : {MSE_test_dB_avg_joint.item():.2f} dB")
print(f"  BiGRU          (avg)          : {mse_bigru_avg_db:.2f} dB")
print("=" * 70)
