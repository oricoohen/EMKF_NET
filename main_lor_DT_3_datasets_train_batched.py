"""
BATCHED version of main_lor_DT_3_datasets_train.py  (delete-to-revert add-on).

Identical data generation and system-model setup as the original; the ONLY changes are:
  1. Pipeline import  -> Pipelines.Pipeline_ERTS_batched.Pipeline_ERTS_batched
  2. training call    -> *_batched twin (all N_B samples run together; ~13-15x faster)

The batched trainers are numerically equivalent to the sequential ones (B=1 is
bit-for-bit identical; B>1 differs only at the GPU float-kernel level).

To revert: just run the original main_lor_DT_3_datasets_train.py instead, and
delete RTSNet/RTSNet_nn_batched.py + Pipelines/Pipeline_ERTS_batched.py.
"""
import torch
from datetime import datetime
from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
    f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure,H_design
from Simulations.Linear_sysmdl import rotate_F, rotate_H

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN
from Baselines.BiGRU_smoother import train_bigru_smoother

# === CHANGED: use the batched pipeline ===
from Pipelines.Pipeline_ERTS_batched import Pipeline_ERTS_batched as Pipeline

import shutil
import os
print("Pipeline Start - EMKF H Estimation (BATCHED)")

# === ADD: global device/dtype ===
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
args.n_steps = 500
args.n_batch = 15
args.lr = 1e-3
args.wd = 1e-3

torch.manual_seed(1)

cycles = 3  # Number of datasets (each represents 30 timesteps with different F)
num_em_iters = 2
# noise q and r
r2 = torch.tensor([1], device=device)  # [100, 10, 1, 0.1, 0.01]
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
Q = q2[0] * Q_structure
R = r2[0] * R_structure
print('q2 is:', q2)
print('r2 is:', r2)


sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0

print("\n" + "="*80)
print("GENERATING 3 DATASETS WITH DIFFERENT H MATRICES (F IS FIXED)")
print("="*80)
load_path_rtsnet_partial = 'RTSNet/lorenz_rotated_1/3datasets/RTSNet_partial.pt'
load_path_rtsnet_partial_joint = 'RTSNet/lorenz_rotated_1/3datasets/RTSNet_partial_joint.pt'
load_path_M_joint = 'RTSNet/lorenz_rotated_1/3datasets/M_step_net_joint.pt'
destination_path_rtsnet_full = 'RTSNet/lorenz_rotated_1/3datasets/50/RTSNet_full.pt'
destination_path_rtsnet_partial = 'RTSNet/lorenz_rotated_1/3datasets/50/RTSNet_partial.pt'
destination_path_M_reg = 'RTSNet/lorenz_rotated_1/3datasets/50/M_step_net.pt'
destination_path_M_joint = 'RTSNet/lorenz_rotated_1/3datasets/50/M_step_net_joint.pt'
destination_path_rtsnet_partial_joint = 'RTSNet/lorenz_rotated_1/3datasets/50/RTSNet_partial_joint.pt'
destination_path_bigru = 'RTSNet/lorenz_rotated_1/3datasets/50/bigru_smoother.pt'

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
all_H_matrices_train_rotate = []
all_H_matrices_cv_rotate = []
all_H_matrices_test_rotate = []
x0_last = None

# Generate datasets with FIXED F and DIVERSE H
H_init = None  # No specific H_init for first dataset
for dataset_id in range(cycles):
    print(f"\n{'='*80}")
    print(f"Generating Dataset {dataset_id}")
    print(f"{'='*80}")

    # Create system model with FIXED F and this H
    sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
    sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp_3_datasets_H/'
    dataFileName = f'dataset_{dataset_id}_data.pt'
    dataFileName_H = f'dataset_{dataset_id}_H.pt'

    # Generate data for this H matrix (F is FIXED)
    print(f"\nGenerating data for dataset {dataset_id}...")


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
            F_gen=False,  # F is FIXED across datasets
            H_gen=True,
            x0_list=x0_last,
            H_init=H_init)  # Use x0_last for continuity in test set

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(dataFolderName + dataFileName_H, map_location=DEVICE)

    # train_rotate =rotate_H(H_train_mat, theta=0.4, many=True, randomit=True)
    # val_rotate = rotate_H(H_val_mat, theta=0.4, many=True, randomit=True)
    # test_rotate = rotate_H(H_test_mat_list, theta=0.4, many=True, randomit=True)

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
    # all_H_matrices_train_rotate.append(train_rotate)
    # all_H_matrices_cv_rotate.append(val_rotate)
    # all_H_matrices_test_rotate.append(test_rotate)

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
sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0

# CRITICAL: Set H lists as 3-dataset structure
sys_model.H_train = all_H_matrices_train_rotate
sys_model.H_valid = all_H_matrices_cv_rotate
sys_model.H_test =  all_H_matrices_test_rotate

# These _TRUE versions are what the loss is computed against
sys_model.H_train_TRUE = all_H_matrices_train
sys_model.H_valid_TRUE = all_H_matrices_cv
sys_model.H_test_TRUE = all_H_matrices_test

# TRUE system model: H_train holds the TRUE H matrices (oracle smoother reference).
# Used to train the "true" RTSNet, which sees each sequence's correct H.
sys_model_true = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
sys_model_true.InitSequence(m1x_0, m2x_0)
sys_model_true.H_train = all_H_matrices_train
sys_model_true.H_valid = all_H_matrices_cv
sys_model_true.H_test = all_H_matrices_test
sys_model_true.H_train_TRUE = all_H_matrices_train
sys_model_true.H_valid_TRUE = all_H_matrices_cv
sys_model_true.H_test_TRUE = all_H_matrices_test

print("✓ System model configured with 3-dataset H structure")

# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

print("✓ Pipeline configured (BATCHED)")

#########################################################################################################
# TRAIN M-STEP NETWORK ON 3 DATASETS FOR H ESTIMATION  (BATCHED)
#########################################################################################################

print("\n" + "="*80)
print("TRAINING M-STEP NETWORK ON 3 DATASETS FOR H ESTIMATION (BATCHED)")
print("="*80)

print("\nTraining configuration:")
print(f"  Datasets: {cycles}")
print(f"  Training sequences per dataset: {args.N_E}")
print(f"  CV sequences per dataset: {args.N_CV}")
print(f"  Timesteps per dataset: {args.T}")
print(f"  Total effective sequence length: {cycles * args.T} timesteps")
print(f"  EM iterations per dataset: {num_em_iters}")
print(f"  Training epochs: {args.n_steps}")
print(f"  Batch size (parallelized): {args.n_batch}")
print(f"  Learning rate: {args.lr}")
print(f"  Weight decay: {args.wd}")
print(f"  EM iteration weights (alpha): (0.5, 1., 0.85)")
print(f"  H regularization (lambda_H): 1e-3")
print(f"  F is FIXED (not estimated)")

print("\nStarting training (batched)...")

# Make sure the output directories exist
for _p in [destination_path_rtsnet_full, destination_path_rtsnet_partial,
           destination_path_rtsnet_partial_joint, destination_path_M_joint,
           destination_path_bigru]:
    os.makedirs(os.path.dirname(_p), exist_ok=True)

# The wrong observation matrix every sequence starts from (single [n,m], shared).
H_init_common = H_Rotate


def _fresh_rtsnet(ssm):
    """A freshly-built RTSNet so each training starts from scratch (not warm-started)."""
    net = RTSNetNN()
    net.NNBuild(ssm, args)
    return net.to(device)


# ============================================================================ #
# BASELINE - BiGRU smoother: data-driven baseline, no system model needed.      #
# Concatenates all datasets; already fully batched (standard minibatch GRU).    #
# ============================================================================ #
print("\n[BASELINE] Training BiGRU smoother...")
# train_bigru_smoother(
#     train_input=all_train_inputs,   # list of 3 tensors [N_E, n, T] — auto-concatenated
#     train_target=all_train_targets,
#     cv_input=all_cv_inputs,
#     cv_target=all_cv_targets,
#     n=n,
#     m=m,
#     save_path=destination_path_bigru,
#     device=device,
#     epochs=300,
#     batch_size=32,
#     lr=1e-3,
#     hidden_size=128,
#     num_layers=2,
# )


# ============================================================================ #
# STEP 1a - TRUE RTSNet: trained with the TRUE per-sequence H (oracle smoother). #
# Uses sys_model_true (H_train = true H) and H_init=None.                        #
# ============================================================================ #
print("\n[STEP 1a] Training TRUE RTSNet (batched, true H per sequence)...")
RTSNet_Pipeline.model = _fresh_rtsnet(sys_model_true)
# RTSNet_Pipeline.train_RTS_net_3_datasets_batched(
#     sys_model_true,
#     all_cv_inputs, all_cv_targets,
#     all_train_inputs, all_train_targets,
#     destination_path_RTS=destination_path_rtsnet_full,
#     load_path_RTS=None,               # train from scratch
#     H_init=None,                      # per-sequence TRUE H (oracle)
#     datasets=cycles)

# ============================================================================ #
# STEP 1b - FALSE RTSNet: trained with the WRONG H_init (H_Rotate), the same     #
# matrix for every sequence.                                                     #
# ============================================================================ #
print("\n[STEP 1b] Training FALSE RTSNet (batched, wrong H_init=H_Rotate)...")
RTSNet_Pipeline.model = _fresh_rtsnet(sys_model)
# RTSNet_Pipeline.train_RTS_net_3_datasets_batched(
#     sys_model,
#     all_cv_inputs, all_cv_targets,
#     all_train_inputs, all_train_targets,
#     destination_path_RTS=destination_path_rtsnet_partial,
#     load_path_RTS=load_path_rtsnet_partial_joint,               # train from scratch
#     H_init=H_init_common,             # SAME wrong H for every sequence
#     datasets=cycles)

# Save an initial M-net checkpoint so STEP 2 has one to load.
torch.save(RTSNet_Pipeline.M_model_H, destination_path_M_joint)

# ============================================================================ #
# STEP 2 - joint RTSNet + M-net training (batched).                              #
# Loads the FALSE RTSNet from STEP 1b; every sequence starts from the shared      #
# wrong H_init and the M-net refines H via EM.                                    #
# ============================================================================ #
print("\n[STEP 2] Joint RTSNet + M-net training (batched)...")
RTSNet_Pipeline.train_H_mstep_net_3_datasets_joint_batched(
    SysModel=sys_model,
    cv_input=all_cv_inputs,           # List of 3 CV datasets [N_CV, n, T]
    cv_target=all_cv_targets,         # List of 3 CV targets [N_CV, m, T]
    train_input=all_train_inputs,     # List of 3 train datasets [N_E, n, T]
    train_target=all_train_targets,   # List of 3 train targets [N_E, m, T]
    destination_path_M=destination_path_M_joint,
    destination_path_RTS=destination_path_rtsnet_partial_joint,
    load_path_RTS=destination_path_rtsnet_partial_joint,   # FALSE RTSNet from STEP 1b
    load_mnet=destination_path_M_joint,
    num_em_iters=num_em_iters,
    alpha=(0.5, 1., 0.85),            # Weights for EM iterations
    lambda_H=1e-3,                    # Regularization on ΔH
    generate_h=True,                  # H_true indexed per group (h_index = n_e // 10)
    H_init=H_init_common,             # SAME wrong H for every sequence
    datasets=cycles)                  # Number of datasets

print("\n" + "="*80)
print("TRAINING COMPLETE (BATCHED)")
print("="*80)
print(f"BiGRU smoother saved to: {destination_path_bigru}")
print(f"TRUE  RTSNet saved to:   {destination_path_rtsnet_full}")
print(f"FALSE RTSNet saved to:   {destination_path_rtsnet_partial}")
print(f"Joint RTSNet saved to:   {destination_path_rtsnet_partial_joint}")
print(f"M-network    saved to:   {destination_path_M_joint}")
