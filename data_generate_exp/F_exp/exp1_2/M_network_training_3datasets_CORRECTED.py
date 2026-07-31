"""
Corrected version of M_network_training_for_nirr.py
This properly sets up data for train_mstep_net_3_datasets function
"""
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F, det
from emkf.main_emkf_func_AI import EMKF_F

from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure, m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

import shutil
print("Pipeline Start")

# === ADD: global device/dtype ===
DEVICE = torch.device("cuda")
DTYPE = torch.float32
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

path_results_True = '../../../RTSNet/AI_M_step/exp_1/r_0001/True_F/'
path_results_False = '../../../RTSNet/AI_M_step/exp_1/r_0001/False_F/'

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_E = 400   # Number of training examples
args.N_CV = 100  # Number of CV examples
args.N_T = 50    # Number of test examples

args.T = 30      # Length of the time series
args.T_test = 30

### training parameters
args.n_steps = 175
args.n_batch = 10
args.lr = 1e-4
args.wd = 1e-3

torch.manual_seed(1)

max_iter = 4
cycles = 3  # Number of datasets (each represents 30 timesteps with different F)

# True model parameters
q2 = 0.01
r2 = 0.001

print('q2 is:', q2)
print('r2 is:', r2)

Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)
H = torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)
m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

print("\n" + "="*80)
print("GENERATING 3 DATASETS WITH DIFFERENT F MATRICES")
print("="*80)


# Storage for all datasets - CORRECTED: Now storing train, cv, AND test data
all_train_inputs = []
all_train_targets = []
all_cv_inputs = []
all_cv_targets = []
all_test_inputs = []
all_test_targets = []
all_F_matrices_train = []
all_F_matrices_cv = []
all_F_matrices_test = []

x0_last = None

# Generate datasets
F_init = None  # No specific F_init for first dataset
for dataset_id in range(cycles):
    print(f"\n{'='*80}")
    print(f"Generating Dataset {dataset_id}")
    print(f"{'='*80}")

    # Create system model with this F
    F_current = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
    sys_model = SystemModel(F_current, Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    print(f"F matrix for dataset {dataset_id}:")
    print(F_current)

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp_3_datasets/'
    dataFileName = f'dataset_{dataset_id}_data.pt'
    dataFileName_F = f'dataset_{dataset_id}_F.pt'

    # Generate data for this F matrix
    print(f"\nGenerating data for dataset {dataset_id}...")

    # IMPORTANT: We need to generate TRAINING and CV data, not just test
    # So we call DataGen with Test=True to generate all splits
    DataGen(args, sys_model,
            dataFolderName + dataFileName,
            dataFolderName + dataFileName_F,
            delta=1,
            randomInit_train=InitIsRandom_train,
            randomInit_cv=InitIsRandom_cv,
            randomInit_test=InitIsRandom_test,
            randomLength=LengthIsRandom,
            Test=False,
            x0_list=x0_last, F_init=F_init)  # Use x0_last for continuity in test set

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F, map_location=DEVICE)

    F_init = [F_train_mat, F_val_mat, F_test_mat_list]  # For next dataset

    # Prepare x0_last for next dataset (for continuity in test sequences)
    X_0_train = train_target[:, :, -1].clone()
    X_0_cv = cv_target[:, :, -1].clone()
    X_0_test = test_target[:, :, -1].clone()
    x0_last = [None,None,None]
    x0_last[0] = [X_0_train[j].unsqueeze(-1).clone() for j in range(X_0_train.size(0))]
    x0_last[1] = [X_0_cv[j].unsqueeze(-1).clone() for j in range(X_0_cv.size(0))]
    x0_last[2] = [X_0_test[j].unsqueeze(-1).clone() for j in range(X_0_test.size(0))]

    print(f"\nDataset {dataset_id} shapes:")
    print(f"  Train input: {train_input.shape}")    # [N_E, n, 30]
    print(f"  Train target: {train_target.shape}")  # [N_E, m, 30]
    print(f"  CV input: {cv_input.shape}")          # [N_CV, n, 30]
    print(f"  CV target: {cv_target.shape}")        # [N_CV, m, 30]
    print(f"  Test input: {test_input.shape}")      # [N_T, n, 30]
    print(f"  Test target: {test_target.shape}")    # [N_T, m, 30]

    # Store ALL data (train, cv, test) - CORRECTED
    all_train_inputs.append(train_input)
    all_train_targets.append(train_target)
    all_cv_inputs.append(cv_input)
    all_cv_targets.append(cv_target)
    all_test_inputs.append(test_input)
    all_test_targets.append(test_target)
    all_F_matrices_train.append(F_train_mat)
    all_F_matrices_cv.append(F_val_mat)
    all_F_matrices_test.append(F_test_mat_list)

print("\n" + "="*80)
print("DATA GENERATION COMPLETE")
print("="*80)

print(f"\nData structure verification:")
print(f"  Number of datasets: {len(all_train_inputs)}")
print(f"  Training sequences per dataset: {all_train_inputs[0].shape[0]}")
print(f"  CV sequences per dataset: {all_cv_inputs[0].shape[0]}")
print(f"  Test sequences per dataset: {all_test_inputs[0].shape[0]}")
print(f"  Timesteps per sequence: {all_train_inputs[0].shape[2]}")
print(f"  Total effective sequence length: {cycles * args.T} timesteps")

# Verify structure
assert len(all_train_inputs) == 3, "Should have 3 datasets"
assert all_train_inputs[0].shape[0] == args.N_E, f"Train should have {args.N_E} sequences"
assert all_cv_inputs[0].shape[0] == args.N_CV, f"CV should have {args.N_CV} sequences"
assert all_test_inputs[0].shape[0] == args.N_T, f"Test should have {args.N_T} sequences"
print("✓ Data structure verified!")

print("\n" + "="*80)
print("SETTING UP SYSTEM MODEL AND PIPELINE")
print("="*80)

# Create system model (F will be updated during training)
F_initial = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
sys_model = SystemModel(F_initial, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)

# CRITICAL: Set F lists as 3-dataset structure
sys_model.F_train = all_F_matrices_train     # List of 3 F lists
sys_model.F_valid = all_F_matrices_cv        # List of 3 F lists
sys_model.F_test = all_F_matrices_test       # List of 3 F lists

# These _TRUE versions are what the loss is computed against
sys_model.F_train_TRUE = all_F_matrices_train
sys_model.F_valid_TRUE = all_F_matrices_cv
sys_model.F_test_TRUE = all_F_matrices_test

print("✓ System model configured with 3-dataset F structure")

# Paths for models
path_results_True_rts = path_results_True + 'best-rts_true.pt'
path_results_wrong_rts = path_results_False + 'best-rts_false.pt'
destination_folder = 'RTSNet/AI_M_step/exp_1/r_0001/EMKF/False/'
destination_path_M = destination_folder + 'M_net_trained_3_datasets_no_mult.pt'
destination_path_M_laod = destination_folder + 'try_15_on_last_f_3_iter_mixed_f.pt'
# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

print("✓ Pipeline configured")

#########################################################################################################
# TRAIN M-STEP NETWORK ON 3 DATASETS
#########################################################################################################

print("\n" + "="*80)
print("TRAINING M-STEP NETWORK ON 3 DATASETS")
print("="*80)

print("\nTraining configuration:")
print(f"  Datasets: {cycles}")
print(f"  Training sequences per dataset: {args.N_E}")
print(f"  CV sequences per dataset: {args.N_CV}")
print(f"  Timesteps per dataset: {args.T}")
print(f"  Total effective sequence length: {cycles * args.T} timesteps")
print(f"  EM iterations per dataset: 3")
print(f"  Training epochs: {args.n_steps}")
print(f"  Batch size: {args.n_batch}")
print(f"  Learning rate: {args.lr}")
print(f"  Weight decay: {args.wd}")
print(f"  EM iteration weights (alpha): (0.05, 0.1, 0.85)")
print(f"  F regularization (lambda_F): 1e-3")

print("\nStarting training...")

# Call the training function - CORRECTED: Pass train and cv data, not test data
RTSNet_Pipeline.train_mstep_net_3_datasets(
    SysModel=sys_model,
    cv_input=all_cv_inputs,           # List of 3 CV datasets [N_CV, n, 30]
    cv_target=all_cv_targets,         # List of 3 CV targets [N_CV, m, 30]
    train_input=all_train_inputs,     # List of 3 train datasets [N_E, n, 30]
    train_target=all_train_targets,   # List of 3 train targets [N_E, m, 30]
    destination_path_M=destination_path_M,
    destination_path_RTS=path_results_wrong_rts,
    num_em_iters=3,
    alpha=(0.05, 0.1, 0.85),          # Weights for EM iterations
    lambda_F=1e-3,                    # Regularization on ΔF
    generate_f=True,                  # Use grouped F (f_index = n_e // 10)
    non_linear_h=False,               # Linear observation model
    load=destination_path_M_laod,datasets=3                        # Number of datasets
)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best M-network model saved to: {destination_path_M}")


