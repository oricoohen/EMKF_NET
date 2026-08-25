"""
Test script for the linear-h / varying-F paper experiment (exp 1 & 2).

Evaluates, on 3 sequential test datasets whose true F drifts by a FIXED 0.2 rad
per dataset (F and x carried over between datasets), five methods:

  1) TRUE F          -- RTSNet given the true F (upper bound)
  2) INITIAL GUESS F -- RTSNet stuck with the nominal F (lower bound, no EMKF)
  3) BiGRU           -- black-box smoother baseline, no model knowledge
  4) AI-EMKF regular -- frozen RTSNet + learned M-step net
  5) AI-EMKF joint   -- jointly trained RTSNet + M-step net

All learned models come from exp_1and2_training.py; EXP_DIR below must match the
EXP_DIR in that script.

Run from anywhere:  python exp_1and2_testing.py
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

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F,det

# ============================================================
# EXPERIMENT SWITCH -- which paper experiment this run produces
#   'gauss'       = exp 1 (Gaussian process/observation noise)
#   'exponential' = exp 2 (Exponential, heavy-tailed, non-zero-mean noise)
# Must be set BEFORE any DataGen call. Noise strength still comes from
# q2 / r2 below; this only changes the DISTRIBUTION.
# ============================================================
import Simulations.Linear_sysmdl as _lsm
_lsm.NOISE_DIST = 'gauss'
print(f"NOISE_DIST = {_lsm.NOISE_DIST}  (exp 1 = gauss, exp 2 = exponential)")

from emkf.main_emkf_func_AI import EMKF_F

from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from Baselines.BiGRU_smoother import test_bigru_smoother

# ============================================================
# VERSION FLAGS — change these to switch architectures
# RTSNET_VERSION : 'F' = F-embedding, 'H' = H-embedding
# MNET_VERSION   : 'old_F' = original LN (saved models),
#                  'new_F' = per-block LN + tanh,
#                  'H'     = H-embedding M-net
# ============================================================
RTSNET_VERSION = 'F'
MNET_VERSION   = 'old_F'
GENERATE_DATA  = True  # True = regenerate datasets, False = load existing saved data

if RTSNET_VERSION == 'F':
    from RTSNet.RTSNet_nn_with_F import RTSNetNN
else:
    from RTSNet.RTSNet_nn import RTSNetNN

# The saved RTSNet checkpoints (best-rts_true/false.pt) are F-embedding models
# (FC8 + FC_F_bw, no FC9) but were pickled under the name RTSNet.RTSNet_nn.RTSNetNN.
# That module's code has since become the FC9 / H-embedding architecture, so a plain
# torch.load rebuilds them with FC9 (which the saved weights lack) -> AttributeError.
# Redirect that pickled class name to the selected (F-embedding) RTSNetNN so the
# checkpoints reconstruct against the matching FC8 architecture. Same patch the
# non-linear-h scripts already use.
import RTSNet.RTSNet_nn as _rts_nn_mod
_rts_nn_mod.RTSNetNN = RTSNetNN

if MNET_VERSION == 'old_F':
    from emkf.AI_M_step_old_F import DeltaF_MStepNet as MStepNet
elif MNET_VERSION == 'new_F':
    from emkf.AI_M_step_for_f import DeltaF_MStepNet as MStepNet
elif MNET_VERSION == 'H':
    from emkf.AI_M_step_for_h import DeltaH_MStepNet as MStepNet


from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

print("Pipeline Start")

# === ADD: global device/dtype ===
DEVICE = torch.device("cuda")
DTYPE  = torch.float32
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # optional

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
### loads its learned models from. It MUST match EXP_DIR in
### exp_1and2_training.py, otherwise this script looks for checkpoints the
### trainer never wrote.
##############################################################################
MODELS_ROOT = REPO_ROOT / 'RTSNet' / 'synthetic' / 'AI_M_step'
# NOTE: 'r_10' selects the SNR bucket. Sweep it together with r2 below --
#       r2 = 10 -> 'r_10',  1 -> 'r_1',  0.1 -> 'r_01',  0.01 -> 'r_001',  0.001 -> 'r_0001'.
#       The folder tag and r2 must always be changed as a pair, and the trainer
#       and the test script must both use the SAME bucket.
EXP_DIR = MODELS_ROOT / 'exp_1' / 'r_10'

# Learned models produced by exp_1and2_training.py.
destination_path_M         = str(EXP_DIR / 'mnet_lin_3ds.pt')           # regular mnet
destination_path_M_joint   = str(EXP_DIR / 'joint_mnet_1m1r_lin.pt')    # joint mnet
destination_path_RTS_joint = str(EXP_DIR / 'joint_rtsnet_1m1r_lin.pt')  # joint RTSNet
bigru_path                 = str(EXP_DIR / 'new_bigru_lin_3ds.pt')      # BiGRU baseline

# RTSNets trained by stage 1 of exp_1and2_training.py, in the SAME EXP_DIR bucket.
path_results_True_rts  = str(EXP_DIR / 'True_F'  / 'best-rts_true.pt')
path_results_wrong_rts = str(EXP_DIR / 'False_F' / 'best-rts_false.pt')

# Where the generated test data is cached.
DATA_DIR = REPO_ROOT / 'Simulations' / 'Linear_canonical' / 'paper' / 'exp1_1' / 'regular'
os.makedirs(DATA_DIR, exist_ok=True)

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_T = 100   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

torch.manual_seed(1)

cycles = 3   # number of sequential datasets, each with a further-drifted true F

# True model noise variances. keep the q2 fixed and change only r2 between 10 to 0.001
q2 = 0.01
r2 = 0.001

print('q2 is:',q2)
print('r2 is:',r2)



Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)
# s=0.95
F = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=DEVICE, dtype=DTYPE) # State transition matrix
# F = torch.tensor([[0.999, 0.1],[0., 0.999]], device=DEVICE, dtype=DTYPE) # State transition matrix
H = torch.tensor([[1., 1.],
                  [0.25, 1.]], device=DEVICE, dtype=DTYPE)
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
SystemModel.F_gen = False
m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)


# Build the true F of each test dataset: the base F rotated by a FIXED THETA_TEST
# rad, applied cumulatively so the drift accumulates across the 3 datasets.
# (The trainer uses a wider RANDOM drift on purpose -- see its docstring.)
THETA_TEST = 0.2

F_matrices_for_datasets_d = []
F_test_list = [F.clone().to(DEVICE) for _ in range(args.N_T)]
H_test_list = [H.clone().to(DEVICE) for _ in range(args.N_T)]
for i in range(cycles + 1):
    F_matrices_for_datasets_d.append([f.clone() for f in F_test_list])
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=THETA_TEST, many=True, randomit=False)
# Drop the un-rotated base F: dataset k uses the F after k+1 rotations.
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]

# Store all data organized by F matrix
all_inputs_by_F = []
all_targets_by_F = []
all_F_matrices = []
all_H_matrices=[]
x0_last = None
# Generate 5 datasets (same as original)
for dataset_id in range(1, cycles+1):
    print(f"\n=== Generating Dataset {dataset_id} ===")

    F_current = F_matrices_for_datasets[dataset_id - 1]
    print(f"F matrix for dataset {dataset_id}:")
    print(F_current)

    # Create system model
    SystemModel.F_gen = False
    sys_model = SystemModel(F_matrices_for_datasets[dataset_id - 1][0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    # File names (DATA_DIR is absolute -- see the path block at the top)
    dataFilePath = str(DATA_DIR / f'snr_0{args.T_test}_dataset_{dataset_id}.pt')
    dataFilePath_F = str(DATA_DIR / f'snr_0_F_dataset_{dataset_id}.pt')

    print("DATA PATH:", dataFilePath)
    print("F DATA PATH:", dataFilePath_F)
    # Generate or load data
    if GENERATE_DATA:
        print(f"Generating data for dataset {dataset_id}...")
        DataGen(args, sys_model, dataFilePath, dataFilePath_F,
                delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
                randomLength=False, Test=True, F_gen=F_matrices_for_datasets[dataset_id - 1],H_gen = H_test_list, x0_list= x0_last)
    else:
        print(f"Loading existing data for dataset {dataset_id}...")

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFilePath, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFilePath_F, map_location=DEVICE)

    # Chain x0: the last true state of this dataset seeds the next one, so the 3
    # datasets form one continuous trajectory.
    x_last = test_target[:,:,-1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))] #list of [m,1]

    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")

    # Store in our organized lists
    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)
    all_H_matrices.append(H_test_list)


#############################################################################
# Baseline: BiGRU smoother (black-box, no model knowledge)
# Trained by exp_1and2_training.py on the same pooled 3-dataset train/cv data,
# then evaluated here on the same 3 test sets as AI-EMKF. bigru_path is defined
# in the path block at the top.
print('\n=== Baseline: BiGRU smoother (black-box) ===')
bigru_mse_lin_sum = 0.0
for dataset_id in range(cycles):
    print(f"\n--- Testing Dataset {dataset_id + 1} with BiGRU ---")

    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    mse_all, mse_db, _ = test_bigru_smoother(test_input, test_target, bigru_path, DEVICE)
    bigru_mse_lin_sum += float(mse_all)  # linear MSE, averaged across datasets below
    print(f"Dataset {dataset_id + 1} - BiGRU MSE: {mse_db:.3f} dB")

average_bigru_mse_db = 10 * torch.log10(torch.tensor(bigru_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"Average MSE with BiGRU: {average_bigru_mse_db:.3f} dB")

#################################################################################################

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
true_mse_lin_sum = 0.0
for dataset_id in range(cycles):
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
    sys_model_true.H_test = all_H_matrices


    # Use NNTest to get results with TRUE F
    kw = dict(load_model_path=path_results_True_rts, generate_f=False)
    if dataset_id > 0:
        kw.update(init_x_list=xT0_last, init_P_list=pT0_last)
    results = RTSNet_Pipeline.NNTest_no_p(sys_model_true, test_input, test_target, **kw)

    # results = [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, x_out,
    #            t, P_smooth, V_list, K_T_list, MSE_test_psmooth_dB_avg, MSE_test_psmooth_std]
    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    true_F_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - TRUE F MSE: {mse_db:.3f} dB")
    mse_lin = float(results[1])  # results[1] = linear MSE avg
    true_mse_lin_sum += mse_lin
   # >>> propagate last smoothed x_T and P_T to next dataset <<<
    x_last = results[3][:, :, -1].clone()            # [N_T, m]
    xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]  # list of [m,1]
    pT0_last = sys_model_true.m2x_0.clone().detach()

average_true_F_mse_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))



#############################################################################
# AI EMKF Sequential Testing
print('\n=== AI EMKF Sequential Learning and Testing ===')

# Initial F guess for all datasets
F_initial_guess_1 = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
F_initial_guess = [F_initial_guess_1.clone() for _ in range(args.N_T)]
# Process each dataset sequentially
emkf_mse_lin_sum = 0.0

for dataset_id in range(cycles):
    print(f"\n--- AI EMKF Processing Dataset {dataset_id + 1} ---")

    # Get current dataset
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    #print(f"True F for this dataset: {true_F_for_this_dataset}")
    print(f"Dataset {dataset_id + 1} input shape: {test_input.shape}")

    # Set up system model for this dataset
    if dataset_id == 0:
        # For the first dataset there is nothing to carry over: start from the nominal F.
        current_F_estimate = F_initial_guess
        print("Using initial F guess for first dataset")
    else:
        # Carry the F the EMKF learned on the previous dataset into this one.
        current_F_estimate = current_F_estimate_prev
        print(f"Using previous dataset's F as estimate: {current_F_estimate[0]}")

    # Create system model with current F estimate
    sys_model_ai = SystemModel(current_F_estimate[0], Q, H, R, args.T, args.T_test)
    sys_model_ai.InitSequence(m1_0, m2_0)

    # Set up F_test and F_test_TRUE for EMKF
    sys_model_ai.F_test = current_F_estimate
    sys_model_ai.F_test_TRUE = true_F_for_this_dataset
    sys_model_ai.H_test = all_H_matrices
    # Run the EM loop: each iteration re-estimates F with the learned M-step net.
    print(f"Running AI-EMKF on dataset {dataset_id + 1}...")

    kw = dict(destination_path_RTS=path_results_wrong_rts,
              destination_path_M=destination_path_M,
              num_em_iters=3, generate_f=False)
    if dataset_id > 0:
        kw.update(init_x_list=x0_em_last, init_P_list=p0_em_last)
    test_losses, test_f_losses, final_F_list, last_x_list = \
        RTSNet_Pipeline.test_mstep_net(sys_model_ai, test_input, test_target, **kw)

    emkf_mse_lin_sum += float(test_losses[-1])
    current_F_estimate_prev = final_F_list

    # Prepare initials for NEXT dataset
    p0_em_last = sys_model_ai.m2x_0.clone().detach()
    # Chain the EMKF's OWN estimated last smoothed state into the next dataset.
    # (Previously this block overwrote it with test_target[:,:,-1] -- the TRUE state,
    # an oracle warm-start that flattered the EMKF and was inconsistent with the
    # other baselines, which chain their estimate.) last_x_list is final_x_list from
    # test_mstep_net: a list of [m,1] smoothed last-state estimates, already in the
    # x0_em_last format the assert below expects.
    x0_em_last = last_x_list

    assert x0_em_last[0].ndim == 2 and x0_em_last[0].shape[1] == 1, f"x0 shape off: {x0_em_last[0].shape}"
emkf_final_mse_db = 10 * torch.log10(torch.tensor(emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

#############################################################################
# JOINT AI EMKF Sequential Testing (1 mnet + 1 RTSNet, jointly trained; linear H)
# Same test_mstep_net, but with the joint RTSNet + joint mnet, F/x carryover.
#############################################################################
print('\n=== JOINT AI EMKF Sequential Testing ===')
joint_mse_lin_sum = 0.0
current_F_estimate_prev_j = None
x0j_last = p0j_last = None
for dataset_id in range(cycles):
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    current_F_estimate_j = F_initial_guess if dataset_id == 0 else current_F_estimate_prev_j

    sys_model_aj = SystemModel(current_F_estimate_j[0], Q, H, R, args.T, args.T_test)
    sys_model_aj.InitSequence(m1_0, m2_0)
    sys_model_aj.F_test = current_F_estimate_j
    sys_model_aj.F_test_TRUE = true_F_for_this_dataset
    sys_model_aj.H_test = all_H_matrices

    kw = dict(destination_path_RTS=destination_path_RTS_joint,
              destination_path_M=destination_path_M_joint,
              num_em_iters=3, generate_f=False)
    if dataset_id > 0:
        kw.update(init_x_list=x0j_last, init_P_list=p0j_last)
    jt_losses, jt_f_losses, jt_final_F_list, jt_last_x_list = \
        RTSNet_Pipeline.test_mstep_net(sys_model_aj, test_input, test_target, **kw)

    joint_mse_lin_sum += float(jt_losses[-1])
    current_F_estimate_prev_j = jt_final_F_list
    x0j_last = jt_last_x_list
    p0j_last = sys_model_aj.m2x_0.clone().detach()
    ds_db = 10 * torch.log10(torch.tensor(float(jt_losses[-1]), device=DEVICE, dtype=DTYPE))
    print(f"  Joint dataset {dataset_id + 1}: {ds_db:.3f} dB")
joint_final_mse_db = 10 * torch.log10(torch.tensor(joint_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

#############################################################################
# Baseline: Test with INITIAL GUESS F using NNTest
print('\n=== Baseline: MSE with INITIAL GUESS F ===')
initial_guess_results = []
init_mse_lin_sum = 0.0

for dataset_id in range(cycles):
    print(f"\n--- Testing Dataset {dataset_id + 1} with INITIAL GUESS F ---")

    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    # Set up system model with initial guess F
    sys_model_init = SystemModel(F_initial_guess[0], Q, H, R, args.T, args.T_test)
    sys_model_init.InitSequence(m1_0, m2_0)
    sys_model_init.H_test = all_H_matrices
    # One F per sequence (args.N_T of them), all the nominal guess: this baseline
    # never updates F, so it stays wrong as the true F drifts across datasets.
    sys_model_init.F_test = F_initial_guess

    # Use NNTest to get results with initial guess F
    kw = dict(load_model_path=path_results_wrong_rts, generate_f=False)
    if dataset_id > 0:
        kw.update(init_x_list=xF0_last, init_P_list=pF0_last)
    results = RTSNet_Pipeline.NNTest_no_p(sys_model_init, test_input, test_target, **kw)

    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    init_mse_lin_sum += float(results[1])  # results[1] = linear MSE avg

    # >>> propagate last smoothed x_T and P_T to next dataset <<<
    x_last = results[3][:, :, -1].clone()  # [N_T, m]
    xF0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]  # list of [m,1]
    pF0_last = sys_model_init.m2x_0.clone().detach()


    initial_guess_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS F MSE: {mse_db:.3f} dB")

average_initial_guess_mse_db = 10 * torch.log10(torch.tensor(init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"Average MSE with INITIAL GUESS F: {average_initial_guess_mse_db:.3f} dB")



#############################################################################
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE F (perfect):        {average_true_F_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"BiGRU (black-box):       {average_bigru_mse_db:.3f} dB")
print(f"EMKF FINAL (regular):    {emkf_final_mse_db:.3f} dB")
print(f"EMKF FINAL (joint):      {joint_final_mse_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - emkf_final_mse_db):.3f} dB")
print(f"EMKF improvement over BiGRU:    {(average_bigru_mse_db - emkf_final_mse_db):.3f} dB")
print(f"Gap to perfect (TRUE F): {(emkf_final_mse_db - average_true_F_mse_db):.3f} dB")
