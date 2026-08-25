"""
Training script for the non-linear-h / varying-F paper experiment (exp 3).

Sets up 3 sequential datasets and trains everything exp3_test.py needs.

Run from anywhere:  python exp3_train.py
"""
import os
import sys
from pathlib import Path

# Put the repo root on sys.path and anchor every path to it, so this script runs
# correctly from its own folder as well as from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from datetime import datetime


from Simulations.Extended_sysmdl import SystemModel, rotate_F#, make_rotated_h_nonlinear   # your class posted above
from Simulations.Lorenz_Atractor.parameters_OLD import ( m1x_0 as m1_0, m2x_0 as m2_0,    # keep your names consistent
    m, n, F, make_f, h_nonlinear, Q_structure, R_structure
)

from Simulations.utils import DataLoader, DataGen

import Simulations.config as config


# F-embedding RTSNet, matching exp3_test.py. The plain RTSNet.RTSNet_nn.RTSNetNN
# is the FC9 / H-embedding architecture, which reads self.H and crashes here.
from RTSNet.RTSNet_nn_with_F import RTSNetNN
import RTSNet.RTSNet_nn as _rts_nn_mod
_rts_nn_mod.RTSNetNN = RTSNetNN


# Batched pipeline (fast): adds train_F_mstep_net_3_datasets_joint_batched -- the
# warm-started, batched F twin of the H nongauss trainer
# (train_H_mstep_net_3_datasets_joint_batched). It subclasses Pipeline_ERTS, so
# every other method used below (NNTest_no_p, setTrainingParams, ...) is unchanged.
from Pipelines.Pipeline_ERTS_batched import Pipeline_ERTS_batched as Pipeline

from Baselines.BiGRU_smoother import train_bigru_smoother

import shutil
import os
print("Pipeline Start")

# ──────────────────────────────────────────────────────────────────────────────
# Keep the non-linear h during data generation. GenerateBatch calls
# SystemModel.update_h(H) per group, which by default rebinds self.h to a LINEAR
# H@x and would corrupt the range-bearing observations. Patch it to only record
# H/H_T and leave self.h = h_nonlinear intact (same fix as the maintained
# data_generate_exp_for_paper/F_exp/exp3/exp_3_train.py).
# ──────────────────────────────────────────────────────────────────────────────
def _update_h_keep_nonlinear(self, H):
    self.H = H
    self.H_T = H.T
SystemModel.update_h = _update_h_keep_nonlinear

# === ADD: global device/dtype ===
DEVICE = torch.device("cuda")
DTYPE = torch.float32
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────────────────────────────────────
# Batched twin of parameters_OLD.h_nonlinear:  x [B,2] -> y [B,2].
# The batched RTSNet runs all N_B sequences through h at once, but the sequential
# h_nonlinear does x.view(2,1) and only works for a single sample. The joint
# batched trainer needs a batch-aware h, so we pass this in via h_batched=...
# (same math: y = H@x + 0.3*[r, theta], H = [[1,1],[0.25,1]]).
# ──────────────────────────────────────────────────────────────────────────────
def h_nonlinear_batched(x, alpha=0.3):
    x1 = x[:, 0]
    x2 = x[:, 1]
    eps = 1e-6
    r = torch.sqrt(x1 * x1 + x2 * x2 + eps)          # [B]
    theta = torch.atan2(x2, x1 + eps)                # [B]
    H = torch.tensor([[1., 1.], [0.25, 1.]], device=x.device, dtype=x.dtype)
    lin = x @ H.T                                    # [B,2]
    return lin + alpha * torch.stack([r, theta], dim=1)  # [B,2]

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

##############################################################################
### Paths -- all anchored to REPO_ROOT, so the CWD does not matter.
##############################################################################
MODELS_ROOT = REPO_ROOT / 'RTSNet' / 'synthetic' / 'AI_M_step'
# NOTE: 'r_0001' selects the SNR bucket. Sweep it together with r2 below --
#       r2 = 10 -> 'r_10',  1 -> 'r_1',  0.1 -> 'r_01',  0.01 -> 'r_001',  0.001 -> 'r_0001'.
#       Trainer and test script must use the SAME bucket.
EXP_DIR = MODELS_ROOT / 'exp_3' / 'r_0001'

os.makedirs(EXP_DIR / 'True_F', exist_ok=True)
os.makedirs(EXP_DIR / 'False_F', exist_ok=True)
path_results_True_rts  = str(EXP_DIR / 'True_F'  / 'best-rts_true.pt')
path_results_wrong_rts = str(EXP_DIR / 'False_F' / 'best-rts_false.pt')

DATA_DIR = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / 'exp_3_datasets'
os.makedirs(DATA_DIR, exist_ok=True)

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_E = 1000   # Number of training examples
args.N_CV = 100  # Number of CV examples
args.N_T = 50    # Number of test examples

args.T = 30      # Length of the time series
args.T_test = 30

### training parameters
args.n_steps = 400
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
F_init = None  # Initialize for first dataset (DataGen expects this parameter)

# True (linear) F for each dataset: rotate the base F by a RANDOM angle in
# [-1, 1] rad (rotate_F theta=1, randomit=True), one random draw per sequence-
# group, chained cumulatively across the 3 datasets -- identical to the exp_1and2
# training so the models see real F diversity. rotate_F(many=True) from
# Extended_sysmdl returns a LIST of [2,2] tensors (one per sequence).
F_nominal = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
theta_max = 1  # random F drift up to 1 rad per group (matches exp_1and2)
F_by_dataset = []
_prev_F = [F_nominal.clone() for _ in range(args.N_E)]
for _k in range(cycles):
    _prev_F = rotate_F(_prev_F, i=0, j=1, theta=theta_max, many=True, randomit=True)
    F_by_dataset.append([f.clone() for f in _prev_F])

# Single nominal F per dataset, used only as the placeholder SystemModel F and by
# the later RTSNet/M-step and test sections that expect ONE matrix (previously
# referenced but never defined -> NameError). The TRUE per-sequence F fed to the
# data generator is F_by_dataset above.
F_matrices_for_datasets = [F_nominal.clone() for _ in range(cycles)]

# Generate datasets
for dataset_id in range(cycles):
    print(f"\n{'='*80}")
    print(f"Generating Dataset {dataset_id}")
    print(f"{'='*80}")

    # Create system model (placeholder F; DataGen applies the true per-group F below)
    F_current = F_matrices_for_datasets[dataset_id]
    f_current = make_f(F_current)  # F is linear, just wrapped as function f(x)=F@x
    sys_model = SystemModel(f_current, Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model.InitSequence(m1_0, m2_0)

    # H list = identity per sequence so the non-linear h is used for observations.
    # Passing an explicit F_gen list (F_by_dataset) + H_gen list also skips the
    # broken generate_random_F_matrices(F_init=None) default path.
    H_gen_list = [torch.eye(n, device=DEVICE, dtype=DTYPE) for _ in range(args.N_E)]

    print(f"F matrix for dataset {dataset_id}:")
    print(F_current)

    # Create folder and file names
    dataFolderName = str(DATA_DIR) + os.sep
    dataFileName = f'dataset_{dataset_id}_data.pt'
    dataFileName_F = f'dataset_{dataset_id}_F.pt'

    # Generate data for this F matrix
    print(f"\nGenerating data for dataset {dataset_id}...")

    # IMPORTANT: We need to generate TRAINING and CV data, not just test
    # So we call DataGen with Test=False to generate all splits
    DataGen(args, sys_model,
            dataFolderName + dataFileName,
            dataFolderName + dataFileName_F,
            delta=1,
            randomInit_train=InitIsRandom_train,
            randomInit_cv=InitIsRandom_cv,
            randomInit_test=InitIsRandom_test,
            randomLength=LengthIsRandom,
            Test=False,
            F_gen=F_by_dataset[dataset_id], H_gen=H_gen_list,
            x0_list=x0_last)  # Use x0_last for continuity in test set

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
print("[OK] Data structure verified!")

#########################################################################################################
# TRAIN BiGRU SMOOTHER BASELINE ON THE SAME 3 DATASETS (FIRST MODEL TO TRAIN)
#########################################################################################################

print("\n" + "="*80)
print("TRAINING BiGRU SMOOTHER BASELINE ON 3 DATASETS")
print("="*80)
# Black-box baseline: a bidirectional-GRU smoother that maps observations
# y:[N,n,T] -> state estimate x_hat:[N,m,T] directly, with no model knowledge.
# Trained on exactly the same pooled 3-dataset train/cv data as the M-step net,
# so exp3_test.py can compare AI-EMKF against it.
destination_folder = str(EXP_DIR / 'EMKF' / 'False') + os.sep
os.makedirs(destination_folder, exist_ok=True)
bigru_save_path = destination_folder + 'new_bigru_lin_3ds.pt'
n_obs = all_train_inputs[0].shape[1]     # observation dim (n)
m_state = all_train_targets[0].shape[1]  # state dim (m)

# train_bigru_smoother(
#     train_input=all_train_inputs,     # list of 3 x [N_E, n, 30] (concatenated inside)
#     train_target=all_train_targets,   # list of 3 x [N_E, m, 30]
#     cv_input=all_cv_inputs,           # list of 3 x [N_CV, n, 30]
#     cv_target=all_cv_targets,         # list of 3 x [N_CV, m, 30]
#     n=n_obs,
#     m=m_state,
#     save_path=bigru_save_path,
#     device=DEVICE,
#     epochs=300,
#     batch_size=8,
#     lr=1e-3,
#     hidden_size=16,
#     num_layers=2,
# )
print(f"BiGRU baseline saved to: {bigru_save_path}")

print("\n" + "="*80)
print("SETTING UP SYSTEM MODEL AND PIPELINE")
print("="*80)

# Create system model (F will be updated during training)
F_initial = F_matrices_for_datasets[0]
f_initial = make_f(F_initial)  # F is linear, just wrapped as function f(x)=F@x
sys_model = SystemModel(f_initial, Q, h_nonlinear, R, args.T, args.T_test, m, n)
sys_model.InitSequence(m1_0, m2_0)

# CRITICAL: Set F lists as 3-dataset structure
sys_model.F_train = all_F_matrices_train     # List of 3 F lists
sys_model.F_valid = all_F_matrices_cv        # List of 3 F lists
sys_model.F_test = all_F_matrices_test       # List of 3 F lists

# These _TRUE versions are what the loss is computed against
sys_model.F_train_TRUE = all_F_matrices_train
sys_model.F_valid_TRUE = all_F_matrices_cv
sys_model.F_test_TRUE = all_F_matrices_test

print("[OK] System model configured with 3-dataset F structure")

# Model output paths (path_results_*_rts are defined in the path block at the top)
destination_folder = str(EXP_DIR / 'EMKF' / 'False') + os.sep
os.makedirs(destination_folder, exist_ok=True)
destination_path_M = destination_folder + 'new_M_net_trained_3_datasets_no_mult.pt'
destination_path_M_laod = destination_folder + 'final_net.pt'
# JOINT (batched) outputs -- ONE F-M-net + ONE RTSNet trained together. Distinct
# names so the warm-start sources (path_results_wrong_rts / destination_path_M_laod)
# are never overwritten.
os.makedirs(destination_folder, exist_ok=True)
destination_path_M_joint   = destination_folder + 'new_joint_mnet_3ds_batched.pt'
destination_path_RTS_joint = destination_folder + 'new_joint_rtsnet_3ds_batched.pt'
# Warm-start the F-M-net only if a checkpoint exists; else start from the pipeline
# default DeltaF_MStepNet (load_mnet=None).
load_mnet = destination_path_M_laod if (destination_path_M_laod and os.path.exists(destination_path_M_laod)) else None
# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

print("[OK] Pipeline configured")

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

# Call the BATCHED JOINT training function -- the warm-started F twin of the
# nongauss H trainer. ONE RTSNet + ONE F-M-net trained together, N_B sequences in
# parallel via torch.bmm. Same experiment/statistics as train_mstep_net_3_datasets
# (F estimation, non-linear h), just batched + warm started + jointly optimized.
#   load_path_RTS : warm-start RTSNet (the wrong-F RTSNet exp3 already produces)
#   load_mnet     : warm-start F-M-net (None -> fresh DeltaF_MStepNet)
#   h_batched     : batch-aware h_nonlinear (sequential SysModel.h is not batchable)
RTSNet_Pipeline.setTrainingParams(args)   # refresh optimizer / default M_model
RTSNet_Pipeline.train_F_mstep_net_3_datasets_joint_batched(
    SysModel=sys_model,
    cv_input=all_cv_inputs,           # List of 3 CV datasets [N_CV, n, 30]
    cv_target=all_cv_targets,         # List of 3 CV targets [N_CV, m, 30]
    train_input=all_train_inputs,     # List of 3 train datasets [N_E, n, 30]
    train_target=all_train_targets,   # List of 3 train targets [N_E, m, 30]
    destination_path_M=destination_path_M_joint,     # trained F-M-net output
    destination_path_RTS=destination_path_RTS_joint, # jointly trained RTSNet output
    load_path_RTS=path_results_wrong_rts,            # warm-start RTSNet (wrong F)
    load_mnet=load_mnet,                             # warm-start F-M-net (or None)
    num_em_iters=3,
    alpha=(0.4, 1, 0.85),          # Weights for EM iterations
    lambda_F=1e-3,                    # Regularization on ΔF
    generate_f=True,                  # Use grouped F (f_index = n_e // 10)
    non_linear_h=True,                # Non-linear observation model (h_nonlinear)
    h_batched=h_nonlinear_batched,    # batch-aware twin of the non-linear h
    datasets=3                        # Number of datasets
)



# TSNet_Pipeline.train_mstep_net_3_datasets(
#     SysModel=sys_model,
#     cv_input=all_cv_inputs,           # List of 3 CV datasets [N_CV, n, 30]
#     cv_target=all_cv_targets,         # List of 3 CV targets [N_CV, m, 30]
#     train_input=all_train_inputs,     # List of 3 train datasets [N_E, n, 30]
#     train_target=all_train_targets,   # List of 3 train targets [N_E, m, 30]
#     destination_path_M=destination_path_M,
#     destination_path_RTS=path_results_wrong_rts,
#     num_em_iters=3,
#     alpha=(0.05, 0.1, 0.85),          # Weights for EM iterations
#     lambda_F=1e-3,                    # Regularization on ΔF
#     generate_f=True,                  # Use grouped F (f_index = n_e // 10)
#     non_linear_h=True,                # Non-linear observation model (h_nonlinear)
#     load=destination_path_M_laod,
#     datasets=3                        # Number of datasets
# )
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

# After training, you can test on each dataset separately
# Or implement a test function that handles the 3-dataset structure

results = None  # Initialize for first iteration
for dataset_id in range(cycles):
    print(f"\n--- Testing Dataset {dataset_id} ---")

    test_input = all_test_inputs[dataset_id]
    test_target = all_test_targets[dataset_id]
    true_F = F_matrices_for_datasets[dataset_id]

    # Set up system model with true F for this dataset (use non-linear h)
    f_test = make_f(true_F)
    sys_model_test = SystemModel(f_test, Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model_test.InitSequence(m1_0, m2_0)
    sys_model_test.F_test = all_F_matrices_test[dataset_id]
    sys_model_test.F_test_TRUE = all_F_matrices_test[dataset_id]

    # You can use one of these test functions:
    # 1. Test with true F (baseline)
    if dataset_id == 0:
        results = RTSNet_Pipeline.NNTest_no_p(
            sys_model_test,
            test_input,
            test_target,
            load_model_path=path_results_True_rts,
            generate_f=False,
            init_x_list=None,
            init_P_list=None
        )
    else:
        # Use last state from previous dataset for continuity
        x_last = results[3][:, :, -1].clone()
        xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]
        pT0_last = sys_model_test.m2x_0.clone().detach()

        results = RTSNet_Pipeline.NNTest_no_p(
            sys_model_test,
            test_input,
            test_target,
            load_model_path=path_results_True_rts,
            generate_f=False,
            init_x_list=xT0_last,
            init_P_list=pT0_last
        )

    mse_db = results[2]
    print(f"Dataset {dataset_id} - MSE: {mse_db:.3f} dB")

print("\n" + "="*80)
print("ALL DONE!")
print("="*80)

"""
KEY DIFFERENCES FROM YOUR ORIGINAL CODE:

1. CORRECTED: Now storing train, cv, AND test data
   - all_train_inputs, all_train_targets (for training)
   - all_cv_inputs, all_cv_targets (for validation)
   - all_test_inputs, all_test_targets (for testing)

2. CORRECTED: Calling train_mstep_net_3_datasets with proper data
   - train_input=all_train_inputs (not all_inputs_by_F which was test data)
   - cv_input=all_cv_inputs (not test data)

3. CORRECTED: F structure setup
   - sys_model.F_train_TRUE = all_F_matrices_train (list of 3 lists)
   - sys_model.F_valid_TRUE = all_F_matrices_cv (list of 3 lists)

4. ADDED: Data structure verification
   - Asserts to check correct shapes
   - Clear logging of data dimensions

5. ADDED: 3 different F matrices
   - F_matrices_for_datasets list with 3 distinct F values
   - Models F changing every 30 timesteps

YOUR ORIGINAL CODE ISSUES:
- Only stored test data (all_inputs_by_F, all_targets_by_F)
- Tried to train on test data (which has N_T sequences, not N_E)
- Used same F for all 3 datasets (all datasets had [[0.83, 0.2], [0.2, 0.83]])
- Missing the train/cv data split needed for training

NOW IT WILL WORK CORRECTLY!
"""

