####the old one without the f
import torch
from datetime import datetime
from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
    f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure,H_design
from Simulations.Linear_sysmdl import rotate_F, rotate_H
from Simulations.Linear_sysmdl import estimate_Q_R_from_true_data
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN
from Baselines.BiGRU_smoother import train_bigru_smoother, test_bigru_smoother
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

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
SEED = 1
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
args.N_T = 100   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

torch.manual_seed(1)

num_iters = 2

cycles = 5

r2 = torch.tensor([10], device=device)  # [100, 10, 1, 0.1, 0.01]
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
Q = q2[0] * Q_structure
R = r2[0] * R_structure
print('q2 is:', q2)
print('r2 is:', r2)

print("\n" + "="*80)
print("GENERATING 3 DATASETS WITH DIFFERENT H MATRICES (F IS FIXED)")
print("="*80)
destination_path_rtsnet_full = 'RTSNet/lorenz_rotated_10/3datasets/RTSNet_full.pt'
destination_path_rtsnet_partial = 'RTSNet/lorenz_rotated_10/3datasets/RTSNet_partial.pt'
# destination_path_rtsnet_partial_joint = 'RTSNet/lorenz_rotated_1/3datasets/RTSNet_partial_joint.pt'
# destination_path_M_joint = 'RTSNet/lorenz_rotated_1/3datasets/M_step_net_joint.pt'
# destination_path_M_joint = 'RTSNet/lorenz_rotated_1/3datasets/M_step_net_joint0.6.pt'
# destination_path_M_joint = 'RTSNet/lorenz_rotated_1/3datasets/M_step_net_joint0.3.pt'
# destination_path_rtsnet_partial_joint = 'RTSNet/lorenz_rotated_1/3datasets/RTSNet_partial_joint0.3.pt'
destination_path_M_joint1 = 'RTSNet/lorenz_rotated_1/3datasets/M_step_net_joint0.4_5 datasets_finallll.pt'
destination_path_rtsnet_partial_joint1 = 'RTSNet/lorenz_rotated_1/3datasets/RTSNet_partial_joint0.4_5datasets_finalllll.pt'
destination_path_M_joint = 'RTSNet/lorenz_rotated_10/3datasets/2iter_mnet.pt'
destination_path_rtsnet_partial_joint = 'RTSNet/lorenz_rotated_10/3datasets/RTSNet_partial.pt'
# Generate diverse H matrices for datasets (F is FIXED)

H_matrices_for_datasets_d = []

initial_guess_H = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]
H_test_list = [H_Rotate.clone().to(DEVICE) for _ in range(args.N_T)]
for i in range(cycles+1):
    H_matrices_for_datasets_d.append([(h).clone() for h in H_test_list])
    # Rotate H for next dataset
    H_test_list = rotate_H(H_matrices_for_datasets_d[i], theta=0.2, many=True, randomit=False)

H_matrices_for_datasets = H_matrices_for_datasets_d[1:]
H_deji = [torch.eye(n, m, device=DEVICE) for _ in range(args.N_T)]

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
    dataFolderName = f'Simulations/Lorenz_Atractor/data/test_r=1/'
    dataFileName = f'snr_0{args.T_test}_dataset0_{dataset_id}.pt'
    dataFileName_H = f'snr_0_H_dataset0_{dataset_id}.pt'
    dataFileName_F = f'snr_0_F_dataset0_{dataset_id}.pt'
    # Generate data with FIXED F and DIVERSE H
    print(f"Generating data for dataset {dataset_id}...")

    if dataset_id == 1:
        if InitIsRandom_test:
            DataGen(args, sys_model,
            dataFolderName + dataFileName,
            dataFolderName + dataFileName_F,
            fileName_H=dataFolderName + dataFileName_H,
            delta=1,
            randomInit_train=False,
            randomInit_cv=False,
            randomInit_test=True,
            randomLength=False,
            Test=True,
            F_gen=False,  # F is FIXED across datasets
            H_gen=H_current,
            x0_list=x0_last,
            H_init=H_current)  # Use x0_last for continuity in test set
            [training_input, training_target, training_init, cv_input, cv_target, cv_init, test_input, test_target,test_init] = torch.load(
            dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
        else:
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
            [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(
                    dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    else:
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
        #Load the generated data
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


# ============================================================
# Estimate Q,R from the TRUE generated data
# ============================================================
# qr_est = estimate_Q_R_from_true_data(
#     all_inputs_by_H=all_inputs_by_H,
#     all_targets_by_H=all_targets_by_H,
#     all_H_matrices=all_H_matrices,
#     f=f,
#     Q_structure=None,
#     R_structure=None,
#     device=DEVICE,
#     dtype=DTYPE
# )
#
# Q_hat = qr_est["Q_hat_full"]
# R_hat = qr_est["R_hat_full"]
#
# print("\n=== Estimated noise statistics from TRUE data ===")
# print(f"count_q = {qr_est['count_q']}, count_r = {qr_est['count_r']}")
# print("Q_hat (structured) =\n", Q_hat)
# print("R_hat (structured) =\n", R_hat)
# Q = Q_hat
# R = R_hat

sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)  # x
for d in range(cycles):
    all_true_x.append(all_targets_by_H[d].clone())

bigru_path = 'RTSNet/lorenz_rotated_10/3datasets/benchmarks/bigru_smoother.pt'

bigru_mse_lin_sum = 0.0
bigru_results = []

for dataset_id in range(cycles):

    print(f"\n--- BiGRU Dataset {dataset_id + 1} ---")

    mse_bigru, mse_bigru_db, x_bigru = test_bigru_smoother(
        test_input=all_inputs_by_H[dataset_id],
        test_target=all_targets_by_H[dataset_id],
        load_path=bigru_path,
        device=device
    )

    bigru_results.append(mse_bigru_db)  # just for printing
    bigru_mse_lin_sum += mse_bigru      # accumulate LINEAR MSE

    print(f"Dataset {dataset_id+1} - BiGRU MSE: {mse_bigru_db:.3f} dB")
avg_bigru_mse = bigru_mse_lin_sum / cycles
avg_bigru_mse_db = 10 * torch.log10(torch.tensor(avg_bigru_mse, device=device))

print("\n=== BiGRU SUMMARY ===")
print(f"Average BiGRU MSE: {avg_bigru_mse_db.item():.3f} dB")


# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model,args)
RTSNet_Pipeline.setTrainingParams(args)

#########################################################################################################
# AI EMKF EXPERIMENT
#########################################################################################################


print('\n=== Starting AI EMKF Experiment with Pre-trained RTSNet ===')

#############################################################################
# Baseline: Test with TRUE H matrices using NNTest
print('\n=== Baseline: MSE with TRUE H matrices ===')
true_H_results = []
true_mse_lin_sum = 0.0
for dataset_id in range(cycles):
    print(f"\n--- Testing Dataset {dataset_id + 1} with TRUE H ---")

    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]
    true_H_for_this_dataset = H_matrices_for_datasets[dataset_id]

    # Set up system model with true H
    sys_model_true = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model_true.InitSequence(m1x_0, m2x_0)  # x0 and P0

    # Set H_test for the model (needed by NNTest)
    sys_model_true.H_test = true_H_for_this_dataset
    # sys_model_true.H_test = H_deji

    if dataset_id == 0:
        # Use NNTest to get results with TRUE H
        #[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, rtsnet_out, RunTime]
        # results = RTSNet_Pipeline.NNTest(
        #     sys_model_true, test_input, test_target, destination_path_rtsnet_full,generate_h=False,generate_f=None,init_x_list=None, init_P_list=None)
        if InitIsRandom_test:
            xT0_last = test_init.clone()
        else:
            xT0_last = None
        results = RTSNet_Pipeline.NNTest(
            sys_model_true, test_input, test_target, destination_path_rtsnet_full,generate_h=False,generate_f=None,init_x_list=xT0_last, init_P_list=None)
    else:
        results = RTSNet_Pipeline.NNTest(
            sys_model_true, test_input, test_target, destination_path_rtsnet_full,generate_h=False,generate_f=None,init_x_list=xT0_last, init_P_list=None)

    all_rts_trueH_x.append(results[3].detach().clone())  # [N_T, m, T]
    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    true_H_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - TRUE H MSE: {mse_db:.3f} dB")
    mse_lin = float(results[1])  # results[1] = linear MSE avg
    true_mse_lin_sum += mse_lin

    # >>> propagate last smoothed x_T and P_T to next dataset <<<
    x_last = results[3][:, :, -1].clone()            # [N_T, m]
    xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]  # list of [m,1]
    pT0_last = sys_model_true.m2x_0.clone().detach()

average_true_H_mse_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

#############################################################################
# AI EMKF Sequential Testing
print('\n=== AI EMKF Sequential Learning and Testing ===')

emkf_mse_lin_sum = 0.0
#################################################################################################################################

for dataset_id in range(cycles):
    print(f"\n--- AI EMKF Processing Dataset {dataset_id + 1} ---")

    # Get current dataset
    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]
    true_H_for_this_dataset = H_matrices_for_datasets[dataset_id]

    print(f"Dataset {dataset_id + 1} input shape: {test_input.shape}")

    # Set up system model for this dataset
    if dataset_id == 0:
        # For first dataset, use initial guess
        current_H_estimate = initial_guess_H
        print("Using initial H guess for first dataset")
    else:
        # For subsequent datasets, use previous dataset's estimate
        current_H_estimate = current_H_estimate_prev
        print(f"Using previous dataset's H as estimate: {current_H_estimate[0]}")

    # Create system model with current H estimate
    sys_model_ai = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model_ai.InitSequence(m1x_0, m2x_0)  # x0 and P0

    # Set up H_test and H_test_TRUE for EMKF
    sys_model_ai.H_test = current_H_estimate
    # sys_model_ai.H_test = H_deji
    sys_model_ai.H_test_TRUE = true_H_for_this_dataset


    # Run test_H_mstep_net (this will iteratively improve H estimates)
    print(f"Running test_H_mstep_net on dataset {dataset_id + 1}...")

    if dataset_id == 0:
        # x0_em_last = test_init.clone()
        test_losses, test_h_losses, final_H_list, last_x_list,list_x = RTSNet_Pipeline.test_H_mstep_net(
            sys_model_ai, test_input, test_target,
            destination_path_RTS=destination_path_rtsnet_partial_joint,
            destination_path_M=destination_path_M_joint,
            num_em_iters=num_iters,
            generate_h=False)
        # test_losses, test_h_losses, final_H_list, last_x_list,list_x = RTSNet_Pipeline.test_H_mstep_net(
        #     sys_model_ai, test_input, test_target,
        #     destination_path_RTS=destination_path_rtsnet_partial_joint,
        #     destination_path_M=destination_path_M_joint,
        #     num_em_iters=num_iters,
        #     generate_h=False,
        #     init_x_list=x0_em_last,
        #     init_P_list=None)
    else:
        test_losses, test_h_losses, final_H_list, last_x_list,list_x = RTSNet_Pipeline.test_H_mstep_net(
            sys_model_ai, test_input, test_target,
            destination_path_RTS=destination_path_rtsnet_partial_joint,
            destination_path_M=destination_path_M_joint,
            num_em_iters=num_iters,
            generate_h=False,
            init_x_list=x0_em_last,
            init_P_list=None)
    all_emkf_x.append(list_x.detach().clone())
    emkf_mse_lin_sum += float(test_losses[-1])
    current_H_estimate_prev = final_H_list

    # Prepare initials for NEXT dataset
    p0_em_last = sys_model_ai.m2x_0.clone().detach()

    # Use TRUE last state for continuity
    # last_x_list = test_target[:,:,-1]
    # print('last_x_list TRUE:',last_x_list[0])
    # x0_em_last = [last_x_list[j].unsqueeze(-1).clone() for j in range(len(last_x_list))]
    x0_em_last = [last_x_list[j].clone() for j in range(len(last_x_list))]

    assert x0_em_last[0].ndim == 2 and x0_em_last[0].shape[1] == 1, f"x0 shape off: {x0_em_last[0].shape}"
emkf_final_mse_db = 10 * torch.log10(torch.tensor(emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

#############################################################################
# Baseline: Test with INITIAL GUESS H using NNTest
print('\n=== Baseline: MSE with INITIAL GUESS H ===')
initial_guess_results = []
init_mse_lin_sum = 0.0

for dataset_id in range(cycles):
    print(f"\n--- Testing Dataset {dataset_id + 1} with INITIAL GUESS H ---")

    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]

    # Set up system model with initial guess H
    sys_model_init = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model_init.InitSequence(m1x_0, m2x_0)  # x0 and P0


    sys_model_init.H_test = initial_guess_H
    # sys_model_init.H_test = H_deji
    # destination_path_rtsnet_partial_exp_3

    # Use NNTest to get results with initial guess H
    if dataset_id == 0:
        # xH0_last = test_init.clone()
        results = RTSNet_Pipeline.NNTest(
            sys_model_init, test_input, test_target, destination_path_rtsnet_partial_joint,generate_h=False,generate_f=None,init_x_list=None, init_P_list=None)
        # results = RTSNet_Pipeline.NNTest(
        #     sys_model_init, test_input, test_target, destination_path_rtsnet_partial_joint, generate_h=False,
        #     generate_f=None, init_x_list=xH0_last, init_P_list=None)
    else:
        results = RTSNet_Pipeline.NNTest(
            sys_model_init, test_input, test_target, destination_path_rtsnet_partial_joint,generate_h=False,generate_f=None,init_x_list=xH0_last, init_P_list=None)

    all_initH_x.append(results[3].detach().clone())  # [N_T, m, T]
    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    init_mse_lin_sum += float(results[1])  # results[1] = linear MSE avg

    # >>> propagate last smoothed x_T and P_T to next dataset <<<
    x_last = results[3][:, :, -1].clone()  # [N_T, m]
    xH0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]  # list of [m,1]
    pH0_last = sys_model_init.m2x_0.clone().detach()

    initial_guess_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS H MSE: {mse_db:.3f} dB")

average_initial_guess_mse_db = 10 * torch.log10(torch.tensor(init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"Average MSE with INITIAL GUESS H: {average_initial_guess_mse_db:.3f} dB")

#############################################################################
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE H (perfect):        {average_true_H_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (learned):    {emkf_final_mse_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - emkf_final_mse_db):.3f} dB")
print(f"Gap to perfect (TRUE H): {(emkf_final_mse_db - average_true_H_mse_db):.3f} dB")


import matplotlib.pyplot as plt

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
total_T = x_true_glued.shape[1]

for dim in range(x_true_glued.shape[0]):
    plt.figure(figsize=(12, 5))

    plt.plot(x_true_glued[dim].numpy(), label="True x", linewidth=2)
    plt.plot(x_rts_trueH_glued[dim].numpy(), label="RTSNet (TRUE H)", linewidth=2)
    plt.plot(x_emkf_glued[dim].numpy(), label="EMKF / learned H", linewidth=2)
    plt.plot(x_initH_glued[dim].numpy(), label="RTSNet (initial H)", linewidth=2)

    for d in range(1, cycles):
        plt.axvline(d * T_len, linestyle='--')

    plt.title(f"Glued trajectories for x[{dim}] - sample {sample_idx}")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_true_glued[0].numpy(), x_true_glued[1].numpy(), x_true_glued[2].numpy(), label="True x", linewidth=2)
ax.plot(x_rts_trueH_glued[0].numpy(), x_rts_trueH_glued[1].numpy(), x_rts_trueH_glued[2].numpy(), label="RTSNet TRUE H", linewidth=2)
ax.plot(x_emkf_glued[0].numpy(), x_emkf_glued[1].numpy(), x_emkf_glued[2].numpy(), label="EMKF learned H", linewidth=2)
ax.plot(x_initH_glued[0].numpy(), x_initH_glued[1].numpy(), x_initH_glued[2].numpy(), label="RTSNet initial H", linewidth=2)

ax.set_title(f"3D glued trajectories - sample {sample_idx}")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.legend()
plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# print("\n" + "="*80)
# print("PLOTTING GLUED TRUE X FROM ALL DATASETS")
# print("="*80)
#
# sample_idx = 0  # choose which sequence to inspect
#
# # concatenate the same sample across datasets along time
# x_glued = torch.cat(
#     [all_targets_by_H[d][sample_idx] for d in range(len(all_targets_by_H))],
#     dim=1
# ).detach().cpu()   # shape [m, cycles*T]
#
# # 3D trajectory
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot(
#     x_glued[0, :].numpy(),
#     x_glued[1, :].numpy(),
#     x_glued[2, :].numpy(),
#     linewidth=2
# )
#
# # mark dataset boundaries
# T_len = all_targets_by_H[0].shape[2]
# for d in range(len(all_targets_by_H)):
#     start_t = d * T_len
#     ax.scatter(
#         x_glued[0, start_t].item(),
#         x_glued[1, start_t].item(),
#         x_glued[2, start_t].item(),
#         marker='o',
#         s=60
#     )
#
# ax.scatter(
#     x_glued[0, -1].item(),
#     x_glued[1, -1].item(),
#     x_glued[2, -1].item(),
#     marker='x',
#     s=80,
#     label='final point'
# )
#
# ax.set_title(f"Glued True X Trajectory - sample {sample_idx}")
# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("x3")
# ax.legend()
# plt.tight_layout()
# plt.show()
# print("\n" + "="*80)
# print("PLOTTING GLUED TRUE X COORDINATES VS TIME")
# print("="*80)
#
# plt.figure(figsize=(10, 5))
# for dim in range(x_glued.shape[0]):
#     plt.plot(x_glued[dim, :].numpy(), label=f"x[{dim}]")
#
# # vertical lines between datasets
# for d in range(1, len(all_targets_by_H)):
#     plt.axvline(d * T_len, linestyle='--')
#
# plt.title(f"Glued True X vs Time - sample {sample_idx}")
# plt.xlabel("t")
# plt.ylabel("value")
# plt.legend()
# plt.tight_layout()
# plt.show()