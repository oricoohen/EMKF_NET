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
args.T = 50
args.T_test = 50
### training parameters
args.n_steps = 300
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

T      = args.T
T_test = args.T_test

### noise levels
q2 = 0.01   # process noise variance   (Q = q2 * I_m)
r2 = 1       # measurement noise variance (R = r2 * I_n)

### theta range for rotation model
# each sequence draws theta ~ Uniform(-theta_true_max/2, +theta_true_max/2)
# false F is always make_F_block(0.0) (straight-line / theta=0 assumption)
theta_true_max = 0.2   # actual drawn range = Uniform(-0.1, +0.1), covers test ±0.10 with margin

Q     = (q2 * Q_structure).to(device) 
R     = (r2 * R_structure).to(device)
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)

### paths
save_dir = "RTSNet/tdoa_2d/3mics/r1/cycle1/"
os.makedirs(save_dir, exist_ok=True)    
destination_path_rtsnet_true  = save_dir + "RTSNet_true0.01.pt"
destination_path_rtsnet_false = save_dir + "RTSNet_false0.01.pt"
destination_path_M_F          = save_dir + "M_step_F_net0.01.pt"
destination_path_rtsnet_jointF = save_dir + "RTSNet_falseF_joint0.01.pt"
destination_path_M_F_joint    = save_dir + "M_step_F_net_joint0.01.pt"

print("=" * 70)
print("2D TDOA RTSNet experiment — rotation theta model")
print(f"  T={T}  T_test={T_test}")
print(f"  q2={q2}  r2={r2}")
print(f"  theta_true_max={theta_true_max}  false F = make_F_block(0.0) [theta=0, straight line]")
print(f"  Microphones: {M_mics}   State dim: {m}   Obs dim: {n}")
print("=" * 70)

#########################################
###  Generate data                     ###
#########################################
# Data generation — see Simulations/TDOA_2D/parameters.py:
#   generate_dataset_random_theta  →  groups of 10 sequences share one random theta
#   generate_single_traj           →  called with [theta] = constant F for all T steps
#
# F is NOT random per time-step — it is constant within a trajectory and
# drawn Uniform(-theta_true/2, +theta_true/2) once per group of 10.

print(f"\nGenerating datasets ...  theta_true_max={theta_true_max}")

train_input, train_target, _, F_train_true = generate_dataset_random_theta(args.N_E,  T,      theta_true_max, Q, R)
cv_input,    cv_target,    _, F_cv_true    = generate_dataset_random_theta(args.N_CV, T,      theta_true_max, Q, R)
test_input,  test_target,  _, F_test_true  = generate_dataset_random_theta(args.N_T,  T_test, theta_true_max, Q, R)

# False F: theta=0 (straight-line constant-velocity assumption)
F_train_false = [make_F_block(0.0) for _ in range(args.N_E)]
F_cv_false    = [make_F_block(0.0) for _ in range(args.N_CV)]
F_test_false  = [make_F_block(0.0) for _ in range(args.N_T)]

print("trainset size:", train_target.size())
print("cvset size:   ", cv_target.size())
print("testset size: ", test_target.size())

# ── True-F vs False-F mismatch summary ──────────────────────────────────────
print("\n" + "─" * 60)
print("TRUE-F  vs  FALSE-F  mismatch  (all groups, test set):")
print("─" * 60)
all_mse = []
for i, (Ft, Ff) in enumerate(zip(F_test_true, F_test_false)):
    mse_g = ((Ft - Ff) ** 2).mean().item()
    all_mse.append(mse_g)
    dB_g  = 10 * math.log10(max(mse_g, 1e-10))
    theta_i = math.atan2(Ft[3, 2].item(), Ft[2, 2].item())
    print(f"  seq {i:3d}  theta={theta_i:.4f} rad  "
          f"||F_true-F_false||^2={mse_g:.6f}  ({dB_g:.2f} dB)")
mse_mean = sum(all_mse) / len(all_mse)
print(f"  MEAN  ||F_true-F_false||^2 = {mse_mean:.6f}  "
      f"({10*math.log10(max(mse_mean,1e-10)):.2f} dB)")
print("  (seq 0 — F_true):")
print(" ", F_test_true[0].cpu().numpy())
print("  (seq 0 — F_false):")
print(" ", F_test_false[0].cpu().numpy())
print("  (seq 0 — diff  F_false - F_true):")
print(" ", (F_test_false[0] - F_test_true[0]).cpu().numpy())
print("─" * 60)
# ─────────────────────────────────────────────────────────────────────────────

#########################################
###  System models                     ###
#########################################
# Linearized H at x0 — prior feature for RTSNet's FC9 (S-GRU). Nonlinear h stays intact.
H_prior = h_jacobian(m1x_0.reshape(-1))   # [n, m]

# Neutral initial F for NNBuild — constant velocity (false F starting point)
F_init  = make_F_block(0.0)
f_init  = make_f(F_init)

### True-F model: RTSNet sees the actual F used to generate each sequence group
sys_model_true = SystemModel(f=f_init, Q=Q, h=h, R=R,
                             T=T, T_test=T_test, m=m, n=n, H=H_prior,
                             prior_S=torch.eye(n, device=device))
sys_model_true.F      = F_init
sys_model_true.F_train = F_train_true
sys_model_true.F_valid = F_cv_true
sys_model_true.F_test  = F_test_true
sys_model_true.InitSequence(m1x_0, m2x_0)

### False-F model: RTSNet sees a perturbed F (true + noise) for each sequence group
sys_model_false = SystemModel(f=f_init, Q=Q, h=h, R=R,
                              T=T, T_test=T_test, m=m, n=n, H=H_prior,
                              prior_S=torch.eye(n, device=device))
sys_model_false.F      = F_init
sys_model_false.F_train = F_train_false
sys_model_false.F_valid = F_cv_false
sys_model_false.F_test  = F_test_false
sys_model_false.InitSequence(m1x_0, m2x_0)
# For MNet training: false F is the input/init, true F is the target
sys_model_false.F_train_TRUE = F_train_true
sys_model_false.F_valid_TRUE = F_cv_true
sys_model_false.F_test_TRUE  = F_test_true

########################################
### Analytic baselines (all test seqs) ###
########################################
print("\nRunning analytic baselines averaged over all test sequences ...")
mse_erts_true_sum  = 0.0
mse_erts_false_sum = 0.0
for i in range(args.N_T):
    states_i = test_target[i]
    obs_i    = test_input[i]
    x_et, *_ = run_ekf_erts(obs_i, make_get_F_from_matrix(F_test_true[i]),  Q_in=Q, R_in=R)
    x_ef, *_ = run_ekf_erts(obs_i, make_get_F_from_matrix(F_test_false[i]), Q_in=Q, R_in=R)
    mse_erts_true_sum  += loss_fn(x_et, states_i).item()
    mse_erts_false_sum += loss_fn(x_ef, states_i).item()

mse_erts_true  = mse_erts_true_sum  / args.N_T
mse_erts_false = mse_erts_false_sum / args.N_T

print(f"  ERTS TRUE-F  MSE (avg): {10 * math.log10(mse_erts_true):.2f} dB")
print(f"  ERTS FALSE-F MSE (avg): {10 * math.log10(mse_erts_false):.2f} dB")

# keep seq-0 outputs for the plot
states = test_target[0]
obs    = test_input[0]
x_erts_true,  *_ = run_ekf_erts(obs, make_get_F_from_matrix(F_test_true[0]),  Q_in=Q, R_in=R)
x_erts_false, *_ = run_ekf_erts(obs, make_get_F_from_matrix(F_test_false[0]), Q_in=Q, R_in=R)

#######################
### RTSNet true-F   ###
#######################
print("\nRTSNet with TRUE-F model (generate_f=True — correct F per group)")
RTSNet_model_true = RTSNetNN()
RTSNet_model_true.NNBuild(sys_model_true, args)

RTSNet_Pipeline_true = Pipeline(strTime, "RTSNet", "RTSNet_TDOA_trueF")
RTSNet_Pipeline_true.setssModel(sys_model_true)
RTSNet_Pipeline_true.setModel(RTSNet_model_true, args)
RTSNet_Pipeline_true.setTrainingParams(args)

print("Number of trainable parameters for RTSNet:",
      sum(p.numel() for p in RTSNet_model_true.parameters() if p.requires_grad))

# RTSNet_Pipeline_true.NNTrain(
#     sys_model_true, cv_input, cv_target, train_input, train_target,
#     destination_path_rtsnet_true,
#     generate_f=True,
# )

[MSE_test_arr_true, MSE_test_avg_true, MSE_test_dB_avg_true,
 rtsnet_out_true, RunTime_true] = RTSNet_Pipeline_true.NNTest(
    sys_model_true, test_input, test_target, destination_path_rtsnet_true,
    generate_f=True,
)

#######################
### RTSNet false-F  ###
#######################
print("\nRTSNet with FALSE-F model (generate_f=True — wrong F per group)")
RTSNet_model_false = RTSNetNN()
RTSNet_model_false.NNBuild(sys_model_false, args)

RTSNet_Pipeline_false = Pipeline(strTime, "RTSNet", "RTSNet_TDOA_falseF")
RTSNet_Pipeline_false.setssModel(sys_model_false)
RTSNet_Pipeline_false.setModel(RTSNet_model_false, args)
RTSNet_Pipeline_false.setTrainingParams(args)

print("Number of trainable parameters for RTSNet:",
      sum(p.numel() for p in RTSNet_model_false.parameters() if p.requires_grad))

# RTSNet_Pipeline_false.NNTrain(
#     sys_model_false, cv_input, cv_target, train_input, train_target,
#     destination_path_rtsnet_false, load_model_path=destination_path_rtsnet_true,
#     generate_f=True,
# )



[MSE_test_arr_false, MSE_test_avg_false, MSE_test_dB_avg_false,
 rtsnet_out_false, RunTime_false] = RTSNet_Pipeline_false.NNTest(
    sys_model_false, test_input, test_target, destination_path_rtsnet_false,
    generate_f=True,
)

#############################
### EMKalmanNet / MNet F   ###
#############################
print("\nEMKalmanNet: training MNet to update F")

# DELETE
diff = sys_model_false.F_train[0] - sys_model_false.F_train_TRUE[0]
print("F_false[0]:\n", sys_model_false.F_train[0])
print("F_true[0]:\n",  sys_model_false.F_train_TRUE[0])
print("||F_false - F_true||^2 =", diff.pow(2).mean().item(),
      f"({10*math.log10(max(diff.pow(2).mean().item(), 1e-10)):.2f} dB)")
# DELETE

num_em_iters = 2

RTSNet_Pipeline_false.train_F_mstep_net(
    sys_model_false,
    cv_input,
    cv_target,
    train_input,
    train_target,
    destination_path_M=destination_path_M_F,
    destination_path_RTS=destination_path_rtsnet_false,
    load_destination_path_M=None,
    num_em_iters=num_em_iters,
    alpha=(0.3, 1.0, 0.85),
    lambda_F=1e-3,
    generate_f=True,
)

[MSE_test_arr_mnet, MSE_test_avg_mnet, MSE_test_dB_avg_mnet,
 rtsnet_out_mnet, RunTime_mnet] = RTSNet_Pipeline_false.test_F_mstep_net(
    sys_model_false,
    test_input,
    test_target,
    destination_path_rtsnet_false,
    destination_path_M_F,
    num_em_iters=num_em_iters,
    lambda_F=1e-3,
    generate_f=True,
)

#############################
### Joint RTSNet + MNet F  ###
#############################
print("\nJoint training: RTSNet + MNet F together")

RTSNet_Pipeline_false.train_joint_F_mstep_net(
    sys_model_false,
    cv_input,
    cv_target,
    train_input,
    train_target,
    destination_path_M=destination_path_M_F_joint,
    destination_path_RTS=destination_path_rtsnet_jointF,
    load_destination_path_RTS=destination_path_rtsnet_false,
    load_destination_path_M=destination_path_M_F,
    num_em_iters=num_em_iters,
    alpha=(0.3, 1.0),
    lambda_F=1e-3,
    generate_f=True,
)

[MSE_test_arr_joint, MSE_test_avg_joint, MSE_test_dB_avg_joint,
 rtsnet_out_joint, RunTime_joint] = RTSNet_Pipeline_false.test_F_mstep_net(
    sys_model_false,
    test_input,
    test_target,
    destination_path_rtsnet_jointF,
    destination_path_M_F_joint,
    num_em_iters=num_em_iters,
    lambda_F=1e-3,
    generate_f=True,
)

########################################
### Plot one example                  ###
########################################
print("\nPlotting test sequence 0 ...")

t_axis = torch.arange(T_test)

plt.figure(figsize=(12, 5))
plt.plot(t_axis, states.cpu()[1],                              linewidth=2.5, label="true p_y")
plt.plot(t_axis, x_erts_true.cpu()[1],   "--",                linewidth=2,   label="ERTS true F")
plt.plot(t_axis, x_erts_false.cpu()[1],  ":",                 linewidth=2,   label="ERTS false F")
plt.plot(t_axis, rtsnet_out_true[0].cpu()[1],                 linewidth=2,   label="RTSNet true F")
plt.plot(t_axis, rtsnet_out_false[0].cpu()[1],                linewidth=2,   label="RTSNet false F")
plt.plot(t_axis, rtsnet_out_mnet[0].cpu()[1],                 linewidth=2,   label="EMKalmanNet / MNet F")
plt.plot(t_axis, rtsnet_out_joint[0].cpu()[1], "-.",          linewidth=2,   label="Joint RTSNet+MNet F")

plt.xlabel("time")
plt.ylabel("y position")
plt.title("TDOA tracking: y position (test seq 0)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir + "tdoa_rtsnet_y_position.png", dpi=250)
print("  Saved:", save_dir + "tdoa_rtsnet_y_position.png")

########################################
### Results summary                   ###
########################################
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"  ERTS TRUE-F  (avg)        : {10 * math.log10(mse_erts_true):.2f} dB")
print(f"  ERTS FALSE-F (avg)        : {10 * math.log10(mse_erts_false):.2f} dB")
print(f"  RTSNet TRUE-F  (avg)      : {MSE_test_dB_avg_true.item():.2f} dB")
print(f"  RTSNet FALSE-F (avg)      : {MSE_test_dB_avg_false.item():.2f} dB")
print(f"  EMKalmanNet F-MNet        : {MSE_test_dB_avg_mnet.item():.2f} dB")
print(f"  Joint RTSNet+MNet F (avg) : {MSE_test_dB_avg_joint.item():.2f} dB")
print("=" * 70)
