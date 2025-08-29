####the old one without the f
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F,det
from emkf.main_emkf_func_AI import EMKF_F

from Simulations.utils import DataLoader, DataGen

import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN


from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

import shutil
print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results_True = 'RTSNet/paper/exp_1/True_F/'
path_results_False = 'RTSNet/paper/exp_1/False_F/'

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_T = 20   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

torch.manual_seed(1)

max_iter = 3


# True model
q2 = 0.01
r2 =0.1
Q = q2 * Q_structure
R = r2 * R_structure
F = torch.tensor([[0.83, 0.2],[0.2, 0.83]]) # State transition matrix
H = torch.tensor([[1., 1.],
                  [0.25, 1.]])
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
SystemModel.F_gen = False
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)


# Generate 5 different F matrices for datasets (same as original)
F_matrices_for_datasets_d = []
F_i = F

for i in range(6):
    F_test_list = [F_i for _ in range(args.N_T)]
    F_matrices_for_datasets_d.append(F_test_list)
    F_i = rotate_F(F_matrices_for_datasets_d[i][0], i=0, j=1, theta=0.78, many=False, randomit=False)
    print('rotate_f =', F_i)
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]  # Use the last 5 matrices


# Store all data organized by F matrix
all_inputs_by_F = []
all_targets_by_F = []
all_F_matrices = []

# Generate 5 datasets (same as original)
for dataset_id in range(1, 6):
    print(f"\n=== Generating Dataset {dataset_id} ===")

    F_current = F_matrices_for_datasets[dataset_id - 1]
    print(f"F matrix for dataset {dataset_id}:")
    print(F_current)

    # Create system model
    SystemModel.F_gen = False
    sys_model = SystemModel(F_matrices_for_datasets[dataset_id - 1][0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/data/dataset/'
    dataFileName = f'2x2_rq3030_T{args.T_test}_dataset_{dataset_id}.pt'
    dataFileName_F = f'2x2_F_dataset_{dataset_id}.pt'

    # Generate data
    print(f"Generating data for dataset {dataset_id}...")
    DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_F,
            delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
            randomLength=False, Test=True, F_gen=F_matrices_for_datasets[dataset_id - 1])

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location="cpu")
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F)

    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")

    # Store in our organized lists
    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)



path_results_True_rts = path_results_True+'best-model_gauss.pt'
# path_results_True_rts2 = path_results_True+'best-modelwith_p2_q.pt'
path_results_True_psmooth = path_results_True+'best-psmooth_true_gauss.pt'
path_results_wrong_rts = path_results_False+'best-rts_false_gauss.pt'
# path_results_2_rts2 = path_results_False+'best-rts_with_pJOINT_q.pt'
path_results_wrong_psmooth = path_results_False+'best-psmooth_false_gauss.pt'
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
# Baseline: Test with TRUE F matrices using NNTest
print('\n=== Baseline: MSE with TRUE F matrices ===')
true_F_results = []

for dataset_id in range(5):
    print(f"\n--- Testing Dataset {dataset_id + 1} with TRUE F ---")

    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id][0]

    # Set up system model with true F
    sys_model_true = SystemModel(true_F_for_this_dataset, Q, H, R, args.T, args.T_test)
    sys_model_true.InitSequence(m1_0, m2_0)

    # Set F_test for the model (needed by NNTest)
    F_test_list = F_matrices_for_datasets[dataset_id]
    sys_model_true.F_test = F_test_list

    # Use NNTest to get results with TRUE F
    results = RTSNet_Pipeline.NNTest(sys_model_true, test_input, test_target, load_model_path=path_results_True_rts, load_p_smoothe_model_path=path_results_True_psmooth,
        generate_f=False)

    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    true_F_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - TRUE F MSE: {mse_db:.3f} dB")

average_true_F_mse_db = sum(true_F_results) / len(true_F_results)
print(f"Average MSE with TRUE F matrices: {average_true_F_mse_db:.3f} dB")

############################################################################# create the datadestination for the models
model_pathes = []
psmooth_pathes = []
# The folder where the new copies will be saved.
destination_folder = 'RTSNet/paper/exp_1/EMKF/False/'
for i in range(max_iter):
    file_rtsnet = f"model_e_q{i}_rand_false_gauss.pt"
    file_psmooth = f"psmooth_e_q{i}_rand_false_gauss.pt"
    # Build the full destination path
    destination_path_RTS = destination_folder + file_rtsnet
    destination_path_PSMOOTH = destination_folder + file_psmooth
    model_pathes.append(destination_path_RTS)
    psmooth_pathes.append(destination_path_PSMOOTH)
#############################################################################
# AI EMKF Sequential Testing
print('\n=== AI EMKF Sequential Learning and Testing ===')

# Initial F guess for all datasets
F_initial_guess_1 = torch.tensor([[0.83, 0.2], [0.2, 0.83]])
F_initial_guess = [F_initial_guess_1.clone() for _ in range(args.N_T)]
# Process each dataset sequentially
for dataset_id in range(5):
    print(f"\n--- AI EMKF Processing Dataset {dataset_id + 1} ---")

    # Get current dataset
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    print(f"True F for this dataset: {true_F_for_this_dataset}")
    print(f"Dataset {dataset_id + 1} input shape: {test_input.shape}")

    # Set up system model for this dataset
    if dataset_id == 0:
        # For first dataset, use initial guess
        current_F_estimate = F_initial_guess
        print("Using initial F guess for first dataset")
    else:
        # For subsequent datasets, we would normally use AI prediction
        # For now, let's use the previous dataset's true F as a simple heuristic
        # (you can replace this with actual AI prediction later)
        prev_true_F = F_matrices_for_datasets[dataset_id - 1]
        current_F_estimate = prev_true_F
        print(f"Using previous dataset's F as estimate: {current_F_estimate}")

    # Create system model with current F estimate
    sys_model_ai = SystemModel(current_F_estimate[0], Q, H, R, args.T, args.T_test)
    sys_model_ai.InitSequence(m1_0, m2_0)

    # Set up F_test and F_test_TRUE for EMKF
    F_test_list = current_F_estimate
    F_test_TRUE_list = true_F_for_this_dataset

    sys_model_ai.F_test = F_test_list
    sys_model_ai.F_test_TRUE = F_test_TRUE_list

    # Run Test_Only_EMKF (this will iteratively improve F estimates)
    print(f"Running Test_Only_EMKF on dataset {dataset_id + 1}...")


    test_losses, test_f_losses, final_F_list  = RTSNet_Pipeline.Test_Only_EMKF(sys_model_ai, test_input, test_target,
        load_base_rtsnet=model_pathes, load_base_psmooth=psmooth_pathes,emkf_iterations=3)
    current_F_estimate = final_F_list
#############################################################################
# Baseline: Test with INITIAL GUESS F using NNTest
print('\n=== Baseline: MSE with INITIAL GUESS F ===')
initial_guess_results = []

for dataset_id in range(5):
    print(f"\n--- Testing Dataset {dataset_id + 1} with INITIAL GUESS F ---")

    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    # Set up system model with initial guess F
    sys_model_init = SystemModel(F_initial_guess[0], Q, H, R, args.T, args.T_test)
    sys_model_init.InitSequence(m1_0, m2_0)

    # Set F_test for the model - one F per sequence
    # Since we have 20 sequences (args.N_T), we need 20 F matrices
    F_test_list = F_initial_guess
    sys_model_init.F_test = F_test_list#THIS IS A F IN THE LONG OF THE SEQ

    # Use NNTest to get results with initial guess F
    results = RTSNet_Pipeline.NNTest(sys_model_init, test_input, test_target, load_model_path=path_results_wrong_rts,
        load_p_smoothe_model_path=path_results_wrong_psmooth, generate_f=False)

    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    initial_guess_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS F MSE: {mse_db:.3f} dB")

average_initial_guess_mse_db = sum(initial_guess_results) / len(initial_guess_results)
print(f"Average MSE with INITIAL GUESS F: {average_initial_guess_mse_db:.3f} dB")

#############################################################################
