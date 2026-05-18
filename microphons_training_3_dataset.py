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

from Simulations.TDOA_2D.parameters import (
    m, n, m1x_0, m2x_0, M_mics,
    Q_structure, R_structure,
    make_F_block, h, h_jacobian,
    generate_dataset_random_theta,
    generate_false_F_list,
    make_f,
    make_get_F_from_matrix,
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
args.T = 20
args.T_test = 20
### training parameters
args.n_steps = 400
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

T      = args.T
T_test = args.T_test

### noise levels
q2 = 1e-4
r2 = 1e-3

### cycle: number of datasets
cycle = 5
# Max theta per dataset [rad] — each group of 10 sequences draws theta ~ Uniform(-max/2, +max/2)
theta_changed_list = [0.2, 0.2, 0.2,0.2,0.2]
assert len(theta_changed_list) == cycle

### false F mismatch
theta_false = 0.2   # [rad]

### EM iterations
num_em_iters = 2

Q     = (q2 * Q_structure).to(device)
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

### paths
save_dir  = "RTSNet/tdoa_2d/"
cycle_dir = save_dir + f"{cycle}cycle/"
os.makedirs(save_dir,  exist_ok=True)
os.makedirs(cycle_dir, exist_ok=True)

# Load paths: pre-trained networks from the 1-dataset experiment
load_path_rtsnet_true  = save_dir + "RTSNet_trueF.pt"
load_path_rtsnet_false = save_dir + "RTSNet_falseF.pt"
load_path_M_F          = save_dir + "M_step_F_net.pt"
load_path_M_F_joint    = save_dir + "M_step_F_net_jointF.pt"

# Cycle-dataset experiment outputs
destination_path_rtsnet_true   = cycle_dir + "RTSNet_trueF.pt"
destination_path_rtsnet_false  = cycle_dir + "RTSNet_falseF.pt"
destination_path_M_F           = cycle_dir + "M_step_F_net.pt"
destination_path_rtsnet_jointF = cycle_dir + "RTSNet_falseF_jointF.pt"
destination_path_M_F_joint     = cycle_dir + "M_step_F_net_jointF.pt"

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

for k in range(cycle):
    theta_changed = theta_changed_list[k]
    print(f"  Dataset {k}: theta ~ Uniform(-{theta_changed/2:.3f}, +{theta_changed/2:.3f}) rad per group")

    ti, tt, th_tr, F_tr_t = generate_dataset_random_theta(args.N_E,  T,      theta_changed, Q, R)
    ci, ct, th_cv, F_cv_t = generate_dataset_random_theta(args.N_CV, T,      theta_changed, Q, R)
    xi, xt, th_te, F_te_t = generate_dataset_random_theta(args.N_T,  T_test, theta_changed, Q, R)

    F_tr_f = generate_false_F_list(th_tr, theta_false)
    F_cv_f = generate_false_F_list(th_cv, theta_false)
    F_te_f = generate_false_F_list(th_te, theta_false)

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

########################################
### Analytic baselines (one sequence) ###
########################################
print("\nAnalytic baselines on test sequence 0 (dataset 0) ...")
states = all_test_targets[0][0]
obs    = all_test_inputs[0][0]
get_F_true_fn  = make_get_F_from_matrix(all_F_test_true[0][0])
get_F_false_fn = make_get_F_from_matrix(all_F_test_false[0][0])
x_erts_true,  *_ = run_ekf_erts(obs, get_F_true_fn)
x_erts_false, *_ = run_ekf_erts(obs, get_F_false_fn)
mse_erts_true  = loss_fn(x_erts_true,  states).item()
mse_erts_false = loss_fn(x_erts_false, states).item()
print(f"  ERTS TRUE-F  MSE: {10*math.log10(mse_erts_true):.2f} dB")
print(f"  ERTS FALSE-F MSE: {10*math.log10(mse_erts_false):.2f} dB")

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
)

# NNTest expects flat [N*cycle, n, T] — concatenate inline
sys_model_true.F_test = [F for Fs in all_F_test_true for F in Fs]
[MSE_test_arr_true, MSE_test_avg_true, MSE_test_dB_avg_true,
 rtsnet_out_true, RunTime_true] = RTSNet_Pipeline_true.NNTest(
    sys_model_true,
    torch.cat(all_test_inputs,  dim=0),
    torch.cat(all_test_targets, dim=0),
    destination_path_rtsnet_true,
    generate_f=True,
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
    load_path_RTS=load_path_rtsnet_false,
    generate_f=True,
    datasets=cycle,
)

sys_model_false.F_test = [F for Fs in all_F_test_false for F in Fs]
[MSE_test_arr_false, MSE_test_avg_false, MSE_test_dB_avg_false,
 rtsnet_out_false, RunTime_false] = RTSNet_Pipeline_false.NNTest(
    sys_model_false,
    torch.cat(all_test_inputs,  dim=0),
    torch.cat(all_test_targets, dim=0),
    destination_path_rtsnet_false,
    generate_f=True,
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
# test_F_mstep_net / test_joint_F_mstep_net expect flat format — flatten inline
sys_model_false.F_test      = [F for Fs in all_F_test_false for F in Fs]
sys_model_false.F_test_TRUE = [F for Fs in all_F_test_true  for F in Fs]
test_input_flat  = torch.cat(all_test_inputs,  dim=0)
test_target_flat = torch.cat(all_test_targets, dim=0)

print("\nTesting MNet ...")
[MSE_test_arr_mnet, MSE_test_avg_mnet, MSE_test_dB_avg_mnet,
 rtsnet_out_mnet, RunTime_mnet] = RTSNet_Pipeline_false.test_F_mstep_net(
    sys_model_false, test_input_flat, test_target_flat,
    destination_path_rtsnet_false, destination_path_M_F,
    num_em_iters=num_em_iters, lambda_F=1e-3, generate_f=True,
)

print("\nTesting joint ...")
[MSE_test_arr_joint, MSE_test_avg_joint, MSE_test_dB_avg_joint,
 rtsnet_out_joint, RunTime_joint] = RTSNet_Pipeline_false.test_F_mstep_net(
    sys_model_false, test_input_flat, test_target_flat,
    destination_path_rtsnet_jointF, destination_path_M_F_joint,
    num_em_iters=num_em_iters, lambda_F=1e-3, generate_f=True,
)

########################################
### Plot                              ###
########################################
print("\nPlotting test sequence 0 ...")
t_axis = torch.arange(T_test)

plt.figure(figsize=(12, 5))
plt.plot(t_axis, states.cpu()[1],                              linewidth=2.5, label="true p_y")
plt.plot(t_axis, x_erts_true.cpu()[1],   "--",                linewidth=2,   label="ERTS true F")
plt.plot(t_axis, x_erts_false.cpu()[1],  ":",                 linewidth=2,   label="ERTS false F")
plt.plot(t_axis, rtsnet_out_true[0].cpu()[1],                 linewidth=2,   label="RTSNet true F")
plt.plot(t_axis, rtsnet_out_false[0].cpu()[1],                linewidth=2,   label="RTSNet false F")
plt.plot(t_axis, rtsnet_out_mnet[0].cpu()[1],                 linewidth=2,   label=f"MNet {cycle}-cycle")
plt.plot(t_axis, rtsnet_out_joint[0].cpu()[1], "-.",          linewidth=2,   label=f"Joint {cycle}-cycle")  # type: ignore[index]
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
print(f"  ERTS TRUE-F  (seq 0)          : {10*math.log10(mse_erts_true):.2f} dB")
print(f"  ERTS FALSE-F (seq 0)          : {10*math.log10(mse_erts_false):.2f} dB")
print(f"  RTSNet TRUE-F  (avg)          : {MSE_test_dB_avg_true.item():.2f} dB")
print(f"  RTSNet FALSE-F (avg)          : {MSE_test_dB_avg_false.item():.2f} dB")
print(f"  MNet {cycle}-cycle (avg)      : {MSE_test_dB_avg_mnet.item():.2f} dB")  # type: ignore[union-attr]
print(f"  Joint {cycle}-cycle (avg)     : {MSE_test_dB_avg_joint.item():.2f} dB")
print("=" * 70)
