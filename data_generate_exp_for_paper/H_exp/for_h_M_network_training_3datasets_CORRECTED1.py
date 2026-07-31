"""
Corrected version for H estimation (EMKF_H)
This properly sets up data for train_H_mstep_net_3_datasets function
F is FIXED (known), H is DIVERSE (unknown, to be estimated)
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
print("Pipeline Start - EMKF H Estimation")

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

path_results_True = '../../RTSNet/changed_H_v_0/exp_1/r_1/True_H/'
path_results_False = '../../RTSNet/changed_H_v_0/exp_1/r_1/False_H/'

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
q2 = 1
r2 = 1

print('q2 is:', q2)
print('r2 is:', r2)

Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)

# F is now FIXED (known dynamics) - NO diversity
F_fixed = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)

# H is TRUE but will be DIVERSE and unknown
H_true = torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)

m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

print("\n" + "="*80)
print("GENERATING 3 DATASETS WITH DIFFERENT H MATRICES (F IS FIXED)")
print("="*80)


# Storage for all datasets - CORRECTED: Now storing train, cv, AND test data
all_train_inputs = []
all_train_targets = []
all_cv_inputs = []
all_cv_targets = []
all_test_inputs = []
all_test_targets = []
all_H_matrices_train = []
all_H_matrices_cv = []
all_H_matrices_test = []

x0_last = None

# ─────────── build diverse H per dataset UP FRONT, then feed into DataGen ───────────
# rotate_H is 3-D-only (builds a 3x3 rotation) -> crashes on 2x2 H. rotate_F is
# dimension-agnostic (R = eye(n) sized to the matrix, R@H@R.T), so we use it to make
# diverse 2x2 H. Each group gets its own random rotation, chained across datasets.
USE_EYE_H = False   # True -> base H = eye(2) ;  False -> base H = [[1,1],[0.25,1]]
theta_H = 0.2       # max random H rotation per dataset (randomit=True)
base_H = torch.eye(2, device=DEVICE, dtype=DTYPE) if USE_EYE_H \
         else torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)

_prev_H = [base_H.clone() for _ in range(args.N_E)]
H_by_dataset = []
for _k in range(cycles):
    _rot = rotate_F(_prev_H, i=0, j=1, theta=theta_H, many=True, randomit=True)  # 2x2 Givens R@H@R.T
    _prev_H = [_rot[i] for i in range(_rot.shape[0])] if torch.is_tensor(_rot) else list(_rot)
    H_by_dataset.append([h.clone() for h in _prev_H])

# Generate datasets with FIXED F and DIVERSE H
H_init = None  # (unused now: H is precomputed above and passed via H_gen)
for dataset_id in range(cycles):
    print(f"\n{'='*80}")
    print(f"Generating Dataset {dataset_id}")
    print(f"{'='*80}")

    # Create system model with FIXED F and this H
    H_current = torch.tensor([[1, 1], [0.25, 1]], device=DEVICE, dtype=DTYPE)
    sys_model = SystemModel(F_fixed, Q, H_current, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    print(f"F matrix (FIXED for all datasets):")
    print(F_fixed)
    print(f"H matrix for dataset {dataset_id}:")
    print(H_current)

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp_3_datasets_H/'
    dataFileName = f'dataset_{dataset_id}_data.pt'
    dataFileName_H = f'dataset_{dataset_id}_H.pt'

    # Generate data for this H matrix (F is FIXED)
    print(f"\nGenerating data for dataset {dataset_id}...")

    # Create F_fixed_list (same F for all sequences)
    F_fixed_list = [F_fixed.clone() for _ in range(args.N_E)]

    # Create H_list for this dataset (same H for all sequences in this dataset)
    H_current_list = [H_current.clone() for _ in range(args.N_E)]

    # IMPORTANT: We need to generate TRAINING and CV data, not just test
    # So we call DataGen with Test=False to generate all splits
    DataGen(args, sys_model,
            dataFolderName + dataFileName,
            dataFolderName + dataFileName_H,
            fileName_H=dataFolderName + dataFileName_H,
            delta=1,
            randomInit_train=InitIsRandom_train,
            randomInit_cv=InitIsRandom_cv,
            randomInit_test=InitIsRandom_test,
            randomLength=LengthIsRandom,
            Test=False,
            F_gen=F_fixed_list,
            H_gen=H_by_dataset[dataset_id],   # precomputed diverse 2x2 H (per group)
            x0_list=x0_last)

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(dataFolderName + dataFileName_H, map_location=DEVICE)

    H_init = [H_train_mat, H_val_mat, H_test_mat_list]  # For next dataset

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
    all_H_matrices_train.append(H_train_mat)
    all_H_matrices_cv.append(H_val_mat)
    all_H_matrices_test.append(H_test_mat_list)

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

# Create system model with FIXED F and initial H
H_initial = torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)
sys_model = SystemModel(F_fixed, Q, H_initial, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)

# CRITICAL: Set H lists as 3-dataset structure
sys_model.H_train = all_H_matrices_train     # List of 3 H lists
sys_model.H_valid = all_H_matrices_cv        # List of 3 H lists
sys_model.H_test = all_H_matrices_test       # List of 3 H lists

# These _TRUE versions are what the loss is computed against
sys_model.H_train_TRUE = all_H_matrices_train
sys_model.H_valid_TRUE = all_H_matrices_cv
sys_model.H_test_TRUE = all_H_matrices_test

print("✓ System model configured with 3-dataset H structure")

# Paths for models
path_results_True_rts = path_results_True + 'best-rts_true.pt'
path_results_wrong_rts = path_results_False + 'best-rts_false.pt'
destination_folder = 'RTSNet/changed_H_v_0/exp_2/r_1/EMKF/False/'
destination_path_M = destination_folder + 'M_net_H_trained_3_datasets2.pt'
destination_path_M_load = destination_folder + 'M_rand_false_trained_12_20_f_rtsnet_new_net.pt'


#########################################################################################################
# BiGRU SMOOTHER BASELINE (black-box y -> x; no model knowledge). Same setup as the F experiment.
# BiGRU doesn't use F or H, so this is identical whether F or H is the changing quantity.
#########################################################################################################
from Baselines.BiGRU_smoother import train_bigru_smoother
bigru_save_path = destination_folder + 'bigru_H_3ds.pt'
print("\n===== BiGRU baseline training (3 datasets) =====")
train_bigru_smoother(
    train_input=all_train_inputs, train_target=all_train_targets,
    cv_input=all_cv_inputs, cv_target=all_cv_targets,
    n=all_train_inputs[0].shape[1], m=all_train_targets[0].shape[1],
    save_path=bigru_save_path, device=DEVICE,
    epochs=300, batch_size=32, lr=1e-3, hidden_size=128, num_layers=2,
)
print("BiGRU baseline saved to:", bigru_save_path)
asdgadgadgssdg





# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

print("✓ Pipeline configured")

#########################################################################################################
# TRAIN M-STEP NETWORK ON 3 DATASETS FOR H ESTIMATION
#########################################################################################################

print("\n" + "="*80)
print("TRAINING M-STEP NETWORK ON 3 DATASETS FOR H ESTIMATION")
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
print(f"  H regularization (lambda_H): 1e-3")
print(f"  F is FIXED (not estimated)")

print("\nStarting training...")

# Call the H training function - CORRECTED: Pass train and cv data, not test data
RTSNet_Pipeline.train_H_mstep_net_3_datasets(
    SysModel=sys_model,
    cv_input=all_cv_inputs,           # List of 3 CV datasets [N_CV, n, 30]
    cv_target=all_cv_targets,         # List of 3 CV targets [N_CV, m, 30]
    train_input=all_train_inputs,     # List of 3 train datasets [N_E, n, 30]
    train_target=all_train_targets,   # List of 3 train targets [N_E, m, 30]
    destination_path_M=destination_path_M,
    destination_path_RTS=path_results_wrong_rts,
    num_em_iters=3,
    alpha=(0.05, 0.1, 0.85),          # Weights for EM iterations
    lambda_H=1e-3,                    # Regularization on ΔH
    generate_h=True,                  # Use grouped H (h_index = n_e // 10)
    non_linear_f=False,               # Linear observation model
    load=destination_path_M_load,
    datasets=3                        # Number of datasets
)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best M-network model saved to: {destination_path_M}")



#########################################################################################################
# OPTIONAL: TEST THE TRAINED MODEL
#########################################################################################################

print("\n" + "="*80)
print("TESTING ON 3 DATASETS (OPTIONAL)")
print("="*80)

