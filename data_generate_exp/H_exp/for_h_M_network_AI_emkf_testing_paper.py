####the old one without the f
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F,det
from emkf.main_emkf_func_AI import EMKF_F

from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN


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
SEED = 0

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
path_results_True = '../../RTSNet/synthetic/changed_H_v_0/exp_2/r_01/True_H/'
gauss = False
path_results_False = '../../RTSNet/synthetic/changed_H_v_0/exp_2/r_01/False_H/'


####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_T = 50   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

torch.manual_seed(1)

max_iter = 4

cycles = 3

# True model
q2 = 0.1
r2 = 0.1

# v_db = 0
# snr_db =20.0################################################################################################################################################################################################
# r2 = 10.0**(-snr_db/10.0)
# q2 = r2/(10.0**v_db/10.0)

print('q2 is:',q2)
print('r2 is:',r2)



Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)

# F is now FIXED (known dynamics) - NO diversity
F_fixed = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=DEVICE, dtype=DTYPE)

# H is TRUE but will be DIVERSE and unknown
H_true = torch.tensor([[1., 1.],
                  [0.25, 1.]], device=DEVICE, dtype=DTYPE)

sys_model = SystemModel(F_fixed, Q, H_true, R, args.T, args.T_test)
SystemModel.F_gen = False
m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix (FIXED):", F_fixed)
print("Observation Matrix (will be diverse):", H_true)


# Generate diverse H matrices for datasets (F is FIXED)

H_matrices_for_datasets_d = []

H_test_list = [H_true.clone().to(DEVICE) for _ in range(args.N_T)]
a = 1
for i in range(cycles+1):
    H_matrices_for_datasets_d.append([(h*a).clone() for h in H_test_list])
    # Rotate H for next dataset
    H_test_list = rotate_F(H_matrices_for_datasets_d[i], i=0, j=1, theta=0.2, many=True, randomit=False)

H_matrices_for_datasets = H_matrices_for_datasets_d[1:]

# F is FIXED for all datasets (create lists for consistency)
F_fixed_list = [F_fixed.clone() for _ in range(args.N_T)]

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
    print(f"F matrix (FIXED for all datasets):")
    print(F_fixed)

    # Create system model with FIXED F and current H
    SystemModel.F_gen = False
    sys_model = SystemModel(F_fixed, Q, H_matrices_for_datasets[dataset_id - 1][0], R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp1_H/regular/'
    dataFileName = f'snr_0{args.T_test}_dataset0_{dataset_id}.pt'
    dataFileName_H = f'snr_0_H_dataset0_{dataset_id}.pt'

    # Generate data with FIXED F and DIVERSE H
    print(f"Generating data for dataset {dataset_id}...")
    DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_H,
            fileName_H=dataFolderName + dataFileName_H,
            delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
            randomLength=False, Test=True,
            F_gen=F_fixed_list,
            H_gen=H_matrices_for_datasets[dataset_id - 1],
            x0_list=x0_last)

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(dataFolderName + dataFileName_H, map_location=DEVICE)

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


path_results_True_rts = path_results_True+'best-rts_true.pt'
path_results_wrong_rts = path_results_False+'best-rts_false.pt'
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
    sys_model_true = SystemModel(F_fixed, Q, true_H_for_this_dataset[0], R, args.T, args.T_test)
    sys_model_true.InitSequence(m1_0, m2_0)

    # Set H_test for the model (needed by NNTest)
    sys_model_true.H_test = true_H_for_this_dataset
    sys_model_true.F_test = F_fixed_list
    if dataset_id == 0:
        # Use NNTest to get results with TRUE H
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_true, test_input, test_target,
                                              load_model_path=path_results_True_rts,
                                              generate_f=False, generate_h=False,
                                              init_x_list=None, init_P_list=None)
    else:
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_true, test_input, test_target,
                                              load_model_path=path_results_True_rts,
                                              generate_f=False, generate_h=False,
                                              init_x_list=xT0_last, init_P_list=pT0_last)

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



############################################################################# create the datadestination for the models
# The folder where the new copies will be saved.
destination_folder = 'RTSNet/changed_H_v_0/exp_2/r_01/EMKF/False/'###############################################################################################################################################
destination_path_M = destination_folder + 'M_net_H_trained_3_datasets2.pt'
# destination_path_M_2 =  destination_folder + 'try_one_iter_just_x_mix_f.pt'
# destination_path_M = destination_folder +f"M_rand_false_trained_12_20_f_rtsnet_new_net.pt"
path_results_wrong_psmooth = path_results_False+'best-psmooth_false.pt'
#############################################################################
# AI EMKF Sequential Testing
print('\n=== AI EMKF Sequential Learning and Testing ===')

# Initial H guess for all datasets
H_initial_guess_1 = torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)
H_initial_guess = [H_initial_guess_1.clone() for _ in range(args.N_T)]
# Process each dataset sequentially
emkf_mse_lin_sum = 0.0
#################################################################################################################################

# # model_pathes = []
# m_step_pathes = []
# for i in range(max_iter):
#     # Create the new filename, e.g., "expert_0.pt", "expert_1.pt", etc.
#
#     file_rtsnet = f"model_e_q{i}_rand_false_trained.pt"
#     # file_m_step = f"m_step_e_q{i}M_rand_false_trained_only_f_15_net_no_pass_between_paper.pt"
#     if i ==2:
#         file_m_step = f"M_rand_false_trained_only_f_0.1_net_paper.pt"
#     if i==1:
#         file_m_step = f"M_rand_false_trained_only_f_0.1_net_paper.pt"
#     else:
#         file_m_step = f"M_rand_false_trained_only_f_one_net_paper.pt"
#     # Build the full destination path
#     # destination_path_RTS = destination_folder + file_rtsnet
#     destination_path_m_step = destination_folder + file_m_step
#     # model_pathes.append(destination_path_RTS)
#     m_step_pathes.append(destination_path_m_step)


#
# model_pathes = []
# m_step_pathes = []
# for i in range(max_iter):
#     # Create the new filename, e.g., "expert_0.pt", "expert_1.pt", etc.
#
#     file_rtsnet = f"model_e_q{i}_rand_false_trained_no_rts_10f_5fmid.pt"
#     file_m_step = f"m_step_e_q{i}M_rand_false_trained_no_rts_10f_5fmid.pt"
#     # Build the full destination path
#     destination_path_RTS = destination_folder + file_rtsnet
#     destination_path_m_step = destination_folder + file_m_step
#     model_pathes.append(destination_path_RTS)
#     m_step_pathes.append(destination_path_m_step)


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
        current_H_estimate = H_initial_guess
        print("Using initial H guess for first dataset")
    else:
        # For subsequent datasets, use previous dataset's estimate
        current_H_estimate = current_H_estimate_prev
        print(f"Using previous dataset's H as estimate: {current_H_estimate[0]}")

    # Create system model with current H estimate
    sys_model_ai = SystemModel(F_fixed, Q, current_H_estimate[0], R, args.T, args.T_test)
    sys_model_ai.InitSequence(m1_0, m2_0)

    # Set up H_test and H_test_TRUE for EMKF
    sys_model_ai.H_test = current_H_estimate
    sys_model_ai.H_test_TRUE = true_H_for_this_dataset
    sys_model_ai.F_test = F_fixed_list

    # Run test_H_mstep_net (this will iteratively improve H estimates)
    print(f"Running test_H_mstep_net on dataset {dataset_id + 1}...")

    if dataset_id == 0:
        test_losses, test_h_losses, final_H_list, last_x_list = RTSNet_Pipeline.test_H_mstep_net(
            sys_model_ai, test_input, test_target,
            destination_path_RTS=path_results_wrong_rts,
            destination_path_M=destination_path_M,
            num_em_iters=3,
            generate_h=False)
    else:
        test_losses, test_h_losses, final_H_list, last_x_list = RTSNet_Pipeline.test_H_mstep_net(
            sys_model_ai, test_input, test_target,
            destination_path_RTS=path_results_wrong_rts,
            destination_path_M=destination_path_M,
            num_em_iters=3,
            generate_h=False,
            init_x_list=x0_em_last,
            init_P_list=p0_em_last)

    emkf_mse_lin_sum += float(test_losses[-1])
    current_H_estimate_prev = final_H_list

    # Prepare initials for NEXT dataset
    p0_em_last = sys_model_ai.m2x_0.clone().detach()

    # Use TRUE last state for continuity
    last_x_list = test_target[:,:,-1]
    print('last_x_list TRUE:',last_x_list[0])
    x0_em_last = [last_x_list[j].unsqueeze(-1).clone() for j in range(len(last_x_list))]

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
    sys_model_init = SystemModel(F_fixed, Q, H_initial_guess[0], R, args.T, args.T_test)
    sys_model_init.InitSequence(m1_0, m2_0)

    # Set H_test for the model
    H_test_list = H_initial_guess
    sys_model_init.H_test = H_test_list
    sys_model_init.F_test = F_fixed_list

    # Use NNTest to get results with initial guess H
    if dataset_id == 0:
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_init, test_input, test_target,
                                              load_model_path=path_results_wrong_rts,
                                              generate_f=False, generate_h=False)
    else:
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_init, test_input, test_target,
                                              load_model_path=path_results_wrong_rts,
                                              generate_f=False, generate_h=False,
                                              init_x_list=xH0_last, init_P_list=pH0_last)

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
