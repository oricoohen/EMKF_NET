import torch
from emkf.main_emkf_func import EMKF_H_analitic_f_nonlinear  # Changed to H estimation
import random
device = torch.device("cuda")
DTYPE = torch.float32
import torch.backends.cudnn as cudnn
import torch
from datetime import datetime
from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
    f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure,H_design
from Simulations.Linear_sysmdl import rotate_F, rotate_H,estimate_Q_R_from_true_data, estimate_H_ls
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_H

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


cudnn.benchmark = True
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



import shutil
print("Pipeline Start")
import random
# === ADD: global device/dtype ===
DEVICE = torch.device("cuda")
DTYPE  = torch.float32
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # optional

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
SEED = 0
# ============================================
# STORAGE FOR PLOTTING PREDICTED X
# ============================================
all_true_x = []
all_rts_trueH_x = []
all_emkf_x = []
all_initH_x = []
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False
DEVICE = torch.device("cuda")
DTYPE = torch.float32
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)
H_Rotate = H_Rotate.to(device)
H_Rotate_inv = H_Rotate_inv.to(device)
Q_structure = Q_structure.to(device)
R_structure = R_structure.to(device)
H_design = H_design.to(device)
args = config.general_settings()

############################################################################
#################################################################################
args.N_T = 100   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

torch.manual_seed(1)

max_iter = 10

cycles = 3

r2 = torch.tensor([1], device=device)  # [100, 10, 1, 0.1, 0.01]
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
Q = q2[0] * Q_structure
R = r2[0] * R_structure
print('q2 is:', q2)
print('r2 is:', r2)

H_matrices_for_datasets_d = []
H_deji = [torch.eye(n, m, device=DEVICE) for _ in range(args.N_T)]

H_initial_estimate = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]
H_test_list = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]
for i in range(cycles+1):
    H_matrices_for_datasets_d.append([(h).clone() for h in H_test_list])
    # Rotate H for next dataset
    H_test_list = rotate_H(H_matrices_for_datasets_d[i], theta=0.2, many=True, randomit=False)

H_matrices_for_datasets = H_matrices_for_datasets_d[1:]

# Store all data organized by H matrix
all_inputs_by_H = []
all_targets_by_H = []
all_H_matrices = []

x0_last = None
# Generate datasets with diverse H (F is FIXED)
for dataset_id in range(1, cycles+1):
    print(f"\n=== Generating Dataset {dataset_id} ===")

    H_current = H_matrices_for_datasets[dataset_id - 1]
    print(f"H matrix for dataset {dataset_id}:")
    print(H_current[0])
    print(H_current[1])

    # Create system model with FIXED F and current H
    sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp1_H/regular/test_'
    dataFileName = f'snr_0{args.T_test}_dataset00_{dataset_id}.pt'
    dataFileName_H = f'snr_0_H_dataset00_{dataset_id}.pt'
    dataFileName_F = f'snr_0_F_dataset00_{dataset_id}.pt'
    # Generate data with FIXED F and DIVERSE H
    print(f"Generating data for dataset {dataset_id}...")

    DataGen(args, sys_model,
            dataFolderName + dataFileName,
            dataFolderName + dataFileName_F,
            fileName_H=dataFolderName + dataFileName_H,
            delta=1,
            randomInit_train=False,
            randomInit_cv=False,
            randomInit_test=False,
            randomLength=False,
            Test=True,
            F_gen=False,  # F is FIXED across datasets
            H_gen=H_current,
            x0_list=x0_last,
            H_init=H_current)  # Use x0_last for continuity in test set
    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(dataFolderName + dataFileName_H, map_location=DEVICE)
    # print(f"H matrices loaded for dataset {H_test_mat_list}:")
    x_last = test_target[:,:,-1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))] #list of [m,1]
    print('x_last from target:', x0_last[0])
    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")
    print(f"H matrix stored: {H_test_mat_list[0]}")

    # Store in our organized lists
    all_inputs_by_H.append(test_input)
    all_targets_by_H.append(test_target)
    all_H_matrices.append(H_test_mat_list)
for d in range(cycles):
    all_true_x.append(all_targets_by_H[d].clone())

# ============================================================
# Estimate Q,R from the TRUE generated data
# ============================================================
qr_est = estimate_Q_R_from_true_data(
    all_inputs_by_H=all_inputs_by_H,
    all_targets_by_H=all_targets_by_H,
    all_H_matrices=all_H_matrices,
    f=f,
    Q_structure=None,
    R_structure=None,
    device=DEVICE,
    dtype=DTYPE
)

Q_hat = qr_est["Q_hat_full"]
R_hat = qr_est["R_hat_full"]

print("\n=== Estimated noise statistics from TRUE data ===")
print(f"count_q = {qr_est['count_q']}, count_r = {qr_est['count_r']}")
print("Q_hat (structured) =\n", Q_hat)
print("R_hat (structured) =\n", R_hat)
Q = Q_hat
R = R_hat
#############################################################################
# Calculate MSE for each dataset with TRUE H (what would happen without EMKF)
x0_last = None
p0_last = None
print('\n=== MSE with TRUE H matrices ===')
true_mse_lin_sum = 0.0
for dataset_id in range(cycles):
    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]
    true_H_for_this_dataset = H_matrices_for_datasets[dataset_id]

    # Use the TRUE H matrix for this dataset
    sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0


    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test_ext_H(sys_model, test_input, test_target,
            H_list=true_H_for_this_dataset,
            generate_h=False,
            init_x_list=None,
            init_P_list=None)
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test_ext_H(sys_model, test_input, test_target,
            H_list=true_H_for_this_dataset,
            generate_h=False,
            init_x_list=x0_last,
            init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    all_rts_trueH_x.append(x_list.detach().clone())
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone() for k in range(args.N_T)]
    true_mse_lin_sum += _mse_avg.item()
    print(f"Dataset {dataset_id + 1} - TRUE H MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with true H
average_true_H_mse_db = 10*torch.log10(torch.tensor(true_mse_lin_sum / cycles))

print(f"Average MSE with TRUE H matrices: {average_true_H_mse_db:.3f} dB")

#############################################################################
# Calculate MSE for each dataset with INITIAL GUESS H
x0_last = None
p0_last = None
print('\n=== MSE with INITIAL GUESS H  ===')
mse_total_false = 0
for dataset_id in range(cycles):
    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]

    # Use the TRUE H matrix for this dataset
    sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0


    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test_ext_H(
            sys_model, test_input, test_target,
            H_list=H_initial_estimate,
            generate_h=False,
            init_x_list=None,
            init_P_list=None
        )
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test_ext_H(
            sys_model, test_input, test_target,
            H_list=H_initial_estimate,
            generate_h=False,
            init_x_list=x0_last,
            init_P_list=p0_last
        )
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    all_initH_x.append(x_list.detach().clone())
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone()             for k in range(args.N_T)]
    mse_total_false +=_mse_avg.item()
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS H MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with initial guess
average_initial_guess_mse_db = 10 * torch.log10(torch.tensor(mse_total_false / cycles))
print(f"Average MSE with INITIAL GUESS H: {average_initial_guess_mse_db:.3f} dB")

###############################################################
# Calculate MSE for each dataset with EMKF H
print('\n=== MSE with EMKF H matrices ===')
mse_total = 0
H_current_estimate = [H_initial_estimate[k].clone() for k in range(args.N_T)]  # Start with initial guess
for dataset_id in range(cycles):
    print(f"\n--- EMKF Iteration {dataset_id + 1} ---")
    print(f"Using dataset {dataset_id + 1}")

    # Get data for this dataset
    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]
    true_H_for_this_dataset = all_H_matrices[dataset_id]

    print(f"True H for this dataset:\n{true_H_for_this_dataset[0]}")


    # Create system model for EMKF with FIXED F
    sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0


    # Run EMKF_H with current estimate as initial guess
    print(f"Running EMKF_H on dataset {dataset_id + 1}...")

    if dataset_id == 0:
        H_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last,x_list_emkf = EMKF_H_analitic_f_nonlinear(
            sys_model,  H_current_estimate, test_input,m1x_0,m2x_0,  test_target,
            max_it=3, generate_h=False,init_x_list=None, init_P_list=None)
    else:
        H_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last,x_list_emkf = EMKF_H_analitic_f_nonlinear(
            sys_model, H_current_estimate,test_input,m1x_0,m2x_0, test_target,
            max_it=3, generate_h=False,init_x_list=x0_last, init_P_list=p0_last)
    all_emkf_x.append(x_list_emkf)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_last[k].clone() for k in range(args.N_T)]
    p0_last = [p_last[k].clone() for k in range(args.N_T)]

    # H_matrices has N_T (amount of seq) list inside where each list has max_it + initial guess H matrices
    # Update H estimate for next iteration (use the result from EMKF)
    H_current_estimate = [Hs_per_seq[-1].clone() for Hs_per_seq in H_matrices]

    print(f"EMKF H evolution for first sequence:")
    for iter_idx, H_est in enumerate(H_matrices[0]):
        h_error = torch.norm(H_est - true_H_for_this_dataset[0]).item()
        print(f"  Iteration {iter_idx}: H error = {h_error:.4f}")

    print(f"Final H estimate (seq 0):\n{H_current_estimate[0]}")
    print(f"True H was:\n{true_H_for_this_dataset[0]}")

    mse_total += mse_avg_T.item()

MSE_total_db = 10 * torch.log10(torch.tensor(mse_total / cycles))

print("\n=== EMKF iterations completed ===")
print(f"Final H estimate:\n{H_current_estimate[0]}")
print(f"\nAverage MSE across all datasets from final H estimate: {MSE_total_db:.3f} dB")

#############################################################################
# Summary comparison
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE H (perfect):        {average_true_H_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (learned):    {MSE_total_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - MSE_total_db):.3f} dB")
print(f"Gap to perfect (TRUE H): {(MSE_total_db - average_true_H_mse_db):.3f} dB")


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("\n" + "="*80)
print("PLOTTING GLUED TRUE X AND PREDICTED X")
print("="*80)

sample_idx = 0  # choose which test sequence to display

# Glue across datasets: result shape [m, cycles*T]
x_true_glued = torch.cat([all_true_x[d][sample_idx] for d in range(cycles)], dim=1).detach().cpu()
x_rts_trueH_glued = torch.cat([all_rts_trueH_x[d][sample_idx] for d in range(cycles)], dim=1).detach().cpu()
x_emkf_glued = torch.cat([all_emkf_x[d][sample_idx] for d in range(cycles)], dim=1).detach().cpu()
x_initH_glued = torch.cat([all_initH_x[d][sample_idx] for d in range(cycles)], dim=1).detach().cpu()

T_len = all_true_x[0].shape[2]

for dim in range(x_true_glued.shape[0]):
    plt.figure(figsize=(12, 5))

    plt.plot(x_true_glued[dim].numpy(), label="True x", linewidth=2)
    plt.plot(x_rts_trueH_glued[dim].numpy(), label="RTS (TRUE H)", linewidth=2)
    plt.plot(x_emkf_glued[dim].numpy(), label="EMKF / learned H", linewidth=2)
    plt.plot(x_initH_glued[dim].numpy(), label="RTS (initial H)", linewidth=2)

    for d in range(1, cycles):
        plt.axvline(d * T_len, linestyle='--')

    plt.title(f"Glued trajectories for x[{dim}] - sample {sample_idx}")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_true_glued[0].numpy(), x_true_glued[1].numpy(), x_true_glued[2].numpy(), label="True x", linewidth=2)
ax.plot(x_rts_trueH_glued[0].numpy(), x_rts_trueH_glued[1].numpy(), x_rts_trueH_glued[2].numpy(), label="RTS TRUE H", linewidth=2)
ax.plot(x_emkf_glued[0].numpy(), x_emkf_glued[1].numpy(), x_emkf_glued[2].numpy(), label="EMKF learned H", linewidth=2)
ax.plot(x_initH_glued[0].numpy(), x_initH_glued[1].numpy(), x_initH_glued[2].numpy(), label="RTS initial H", linewidth=2)

ax.set_title(f"3D glued trajectories - sample {sample_idx}")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.legend()
plt.tight_layout()
plt.show()