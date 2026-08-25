"""
Training script for the linear-h / varying-F paper experiment (exp 1 & 2).

Trains, on 3 sequential datasets:
  1) the regular M-step net  -> train_mstep_net_3_datasets
  2) the joint (1 mnet + 1 RTSNet) net -> joint_train_mnet_rtsnet_3_datasets
  3) a BiGRU smoother baseline (Baselines/BiGRU_smoother.py), on exactly the same
     pooled train/cv data, so exp_1and2_testing.py can compare AI-EMKF against it.

The true F of each dataset is the base F rotated by a RANDOM angle in
[-THETA_TRAIN, THETA_TRAIN] rad (rotate_F randomit=True), chained cumulatively
across the 3 datasets; x0 is chained across datasets too. The test script uses a
FIXED 0.2 rad drift -- training on wider random drifts is deliberate, so the mnet
generalizes instead of overfitting one fixed step.

Sensor h is linear: h(x) = H @ x with H = [[1,1],[0.25,1]].
Noise: q2 = 0.01, r2 = 10.

Run from anywhere:  python exp_1and2_training.py
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

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F, det
from emkf.main_emkf_func_AI import EMKF_F

from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure, m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Baselines.BiGRU_smoother import train_bigru_smoother

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

##############################################################################
### Paths -- all anchored to REPO_ROOT, so the CWD does not matter.
###
### EXP_DIR is the ONE knob that says which experiment/SNR bucket this run
### reads and writes. exp_1and2_testing.py has the identical constant and the
### two MUST point at the same folder, otherwise the test loads checkpoints
### this script never wrote.
##############################################################################
EXP_DIR = REPO_ROOT / 'RTSNet' / 'AI_M_step' / 'exp_1' / 'r_10'

# Pre-trained RTSNet references. NOTE: these live under r_1 while the data below
# is generated at r2 = 10 -- kept as-is from the original script.
RTS_REF_DIR = REPO_ROOT / 'RTSNet' / 'AI_M_step' / 'exp_1' / 'r_1'
path_results_True_rts  = str(RTS_REF_DIR / 'True_F'  / 'best-rts_true.pt')
path_results_wrong_rts = str(RTS_REF_DIR / 'False_F' / 'best-rts_false.pt')

# Outputs of this script.
os.makedirs(EXP_DIR, exist_ok=True)
destination_path_M         = str(EXP_DIR / 'mnet_lin_3ds.pt')           # regular mnet
destination_path_M_joint   = str(EXP_DIR / 'joint_mnet_1m1r_lin.pt')    # joint mnet
destination_path_RTS_joint = str(EXP_DIR / 'joint_rtsnet_1m1r_lin.pt')  # joint RTSNet
bigru_save_path            = str(EXP_DIR / 'new_bigru_lin_3ds.pt')      # BiGRU baseline

# Where the generated 3-dataset data is cached.
DATA_DIR = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / 'exp_3_datasets_r1'
os.makedirs(DATA_DIR, exist_ok=True)

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_E = 400  # Number of training examples (match exp_3)
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
r2 = 10  # observation-noise variance for this run (see EXP_DIR above)

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


# Storage for all datasets - storing train, cv, AND test data
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

# Fixed observation matrix for the linear-h experiment. We MUST pass H_gen explicitly:
# if omitted, DataGen defaults to H_gen=True -> generate_random_H_matrices, which builds
# a 3x3 H (torch.eye(3)) and crashes against the 2-D state. length >= N_E covers all splits.
H_gen_list = [H.clone() for _ in range(args.N_E)]

# Build the true F for each dataset by rotating the base F by a RANDOM angle in
# [-THETA_TRAIN, THETA_TRAIN] rad (rotate_F randomit=True), one draw per sequence,
# chained cumulatively across the 3 datasets. This REPLACES DataGen's default
# random-F path (generate_random_F_matrices), so the drift magnitude is explicit
# and reproducible here rather than buried in the generator.
#
# The test script (exp_1and2_testing.py) evaluates on a FIXED 0.2 rad drift. The
# wider random drift here is deliberate: it makes the mnet generalize over a range
# of drifts instead of memorizing one fixed step.
THETA_TRAIN = 1.0   # matches exp_3's theta_rot (F drift up to 1 rad per dataset)
theta_max = THETA_TRAIN
base_F = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
F_by_dataset = []
_prev_F = [base_F.clone() for _ in range(args.N_E)]
for _k in range(cycles):
    # each element gets its own random angle in [-theta_max, theta_max]; rotate_F
    # returns a stacked [N_E,2,2] tensor -> convert to a list of [2,2] (GenerateBatch
    # does `if F_gen == True` and needs a list, not a tensor).
    _rot = rotate_F(_prev_F, i=0, j=1, theta=theta_max, many=True, randomit=True)
    _prev_F = [_rot[i] for i in range(_rot.shape[0])]
    F_by_dataset.append(_prev_F)
for dataset_id in range(cycles):
    print(f"\n{'='*80}")
    print(f"Generating Dataset {dataset_id}")
    print(f"{'='*80}")

    # Create system model (F here is only a placeholder; DataGen generates the true F)
    F_current = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
    sys_model = SystemModel(F_current, Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    print(f"F matrix for dataset {dataset_id}:")
    print(F_current)

    # Create file names (DATA_DIR is absolute -- see the path block at the top)
    dataFilePath = str(DATA_DIR / f'dataset_{dataset_id}_data.pt')
    dataFilePath_F = str(DATA_DIR / f'dataset_{dataset_id}_F.pt')

    # Generate data for this dataset (Test=False -> train/cv/test all produced)
    print(f"\nGenerating data for dataset {dataset_id}...")
    DataGen(args, sys_model,
            dataFilePath,
            dataFilePath_F,
            delta=1,
            randomInit_train=InitIsRandom_train,
            randomInit_cv=InitIsRandom_cv,
            randomInit_test=InitIsRandom_test,
            randomLength=LengthIsRandom,
            Test=False,
            F_gen=F_by_dataset[dataset_id],  # explicit true F: base rotated by random theta<=0.3
            H_gen=H_gen_list,   # fixed 2x2 H (linear-h); avoids the random 3x3 default
            x0_list=x0_last)    # x0 chained across datasets for continuity

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFilePath, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFilePath_F, map_location=DEVICE)

    # Prepare x0_last for next dataset (for continuity in the sequences)
    X_0_train = train_target[:, :, -1].clone()
    X_0_cv = cv_target[:, :, -1].clone()
    X_0_test = test_target[:, :, -1].clone()
    x0_last = [None, None, None]
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

    # Store ALL data (train, cv, test)
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

#########################################################################################################
# TRAIN BiGRU SMOOTHER BASELINE ON THE SAME 3 DATASETS
#########################################################################################################

print("\n" + "="*80)
print("TRAINING BiGRU SMOOTHER BASELINE ON 3 DATASETS")
print("="*80)
# Black-box baseline: a bidirectional-GRU smoother that maps observations
# y:[N,n,T] -> state estimate x_hat:[N,m,T] directly, with no model knowledge.
# Trained on exactly the same pooled 3-dataset train/cv data as the M-step net,
# so exp_1and2_testing.py can compare AI-EMKF against it. Saved to bigru_save_path
# (defined in the path block at the top).
n_obs = all_train_inputs[0].shape[1]     # observation dim (n)
m_state = all_train_targets[0].shape[1]  # state dim (m)

train_bigru_smoother(
    train_input=all_train_inputs,     # list of 3 x [N_E, n, 30] (concatenated inside)
    train_target=all_train_targets,   # list of 3 x [N_E, m, 30]
    cv_input=all_cv_inputs,           # list of 3 x [N_CV, n, 30]
    cv_target=all_cv_targets,         # list of 3 x [N_CV, m, 30]
    n=n_obs,
    m=m_state,
    save_path=bigru_save_path,
    device=DEVICE,
    epochs=300,
    batch_size=16,
    lr=1e-3,
    hidden_size=16,
    num_layers=2,
)
print(f"BiGRU baseline saved to: {bigru_save_path}")

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

# ── 1) REGULAR M-step net on 3 datasets (RTSNet frozen), LINEAR h -> non_linear_h=False ──
print("\n===== REGULAR M-step training (3 datasets, linear H, RTSNet frozen) =====")
RTSNet_Pipeline.train_mstep_net_3_datasets(
    SysModel=sys_model,
    cv_input=all_cv_inputs, cv_target=all_cv_targets,
    train_input=all_train_inputs, train_target=all_train_targets,
    destination_path_M=destination_path_M,
    destination_path_RTS=path_results_wrong_rts,
    num_em_iters=3, alpha=(0.05, 0.1, 0.85), lambda_F=1e-3,
    generate_f=True, non_linear_h=False, load=None, datasets=3,
)
print("Regular mnet saved to:", destination_path_M)

# ── 2) JOINT: ONE mnet + ONE RTSNet trained together (RTSNet->mnet->RTSNet->…), linear h ──
print("\n===== JOINT M-step training (3 datasets, 1 mnet + 1 RTSNet, linear H) =====")
RTSNet_Pipeline.setTrainingParams(args)
RTSNet_Pipeline.joint_train_mnet_rtsnet_3_datasets(
    SysModel=sys_model,
    cv_input=all_cv_inputs, cv_target=all_cv_targets,
    train_input=all_train_inputs, train_target=all_train_targets,
    destination_path_M=destination_path_M_joint,
    destination_path_RTS=destination_path_RTS_joint,
    load_rts=path_results_wrong_rts,   # base linear wrong-F RTSNet
    load_m=destination_path_M,         # warm-start joint mnet from the regular mnet
    num_em_iters=3, alpha=(0.05, 0.1, 0.85), lambda_F=1e-3,
    generate_f=True, non_linear_h=False, datasets=3,
)
print("Joint mnet saved to:", destination_path_M_joint)
print("Joint RTS saved to:", destination_path_RTS_joint)

print("\n" + "=" * 80)
print("TRAINING COMPLETE -- artifacts written to:", EXP_DIR)
print(f"  regular mnet : {destination_path_M}")
print(f"  joint mnet   : {destination_path_M_joint}")
print(f"  joint RTSNet : {destination_path_RTS_joint}")
print(f"  BiGRU        : {bigru_save_path}")
print("=" * 80)
