####the old one without the f
import torch
import torch.nn as nn
from datetime import datetime


from Simulations.Extended_sysmdl import SystemModel, rotate_F#, make_rotated_h_nonlinear   # your class posted above
# The old Lorenz params module (m1x_0, m2x_0, m, n, F, make_f, h_nonlinear) this script was written
# against no longer exists (current params are 3-D Lorenz, no make_f/F). Rebuild the 2-D
# "linear-F + nonlinear-h" setup locally (matches ori_regular_emkf_nonlinear_h_test_paper.py).
m = 2
n = 2
m1_0 = torch.tensor([[0.5], [0.5]])
m2_0 = torch.eye(m)
Q_structure = torch.eye(m)
R_structure = torch.eye(n)

def make_f(F):
    def f(x):
        return (F.to(x.device) @ x.reshape(m, 1)).reshape(m)   # accept [m]/[m,1]; return 1-D [m]
    return f

def h_nonlinear(x, alpha=0.3):
    # MATCHES TRAINING (parameters_OLD.h_nonlinear):  h = H@x + 0.3*(cartesian->polar)
    x = x.reshape(2, 1)
    x1, x2 = x[0, 0], x[1, 0]
    eps = 1e-6
    r     = torch.sqrt(x1 * x1 + x2 * x2 + eps)
    theta = torch.atan2(x2, x1 + eps)
    H = torch.tensor([[1., 1.],
                      [0.25, 1.]], device=x.device, dtype=x.dtype)
    lin = (H @ x).view(2)
    return lin + alpha * torch.stack([r, theta])                # return 1-D [n]

# The pre-trained (2-D) checkpoints pickled self.h BY REFERENCE to
# Simulations.Lorenz_Atractor.parameters.h_nonlinear, which is now the 3-D spherical h. Pickle
# re-resolves that name at load time, so rebind it to our 2-D h BEFORE any torch.load below, so
# the models reconstruct with the matching 2-D observation function.
import Simulations.Lorenz_Atractor.parameters as _lor_params
_lor_params.h_nonlinear = h_nonlinear

from Simulations.utils import DataLoader, DataGen

import Simulations.config as config


# The pre-trained RTS checkpoints use the F-aware architecture (FC8, FC_F_bw, no FC9); the current
# base RTSNet_nn.py is H-aware (FC9). Use the F-aware class and remap the name the pickles reference
# so torch.load reconstructs them with matching forward/backward code (this process only).
from RTSNet.RTSNet_nn_with_F import RTSNetNN
import RTSNet.RTSNet_nn as _rtsnet_nn_mod
_rtsnet_nn_mod.RTSNetNN = RTSNetNN


from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Baselines.BiGRU_smoother import test_bigru_smoother   # BiGRU baseline (obs -> states)
import shutil

# The nonlinear-h path calls getJacobian (imported into Pipeline_ERTS), which hardcodes view(-1, m=3).
# Replace with a dimension-agnostic version for the 2-D model.
import Pipelines.Pipeline_ERTS as _pipe_mod
def _getJacobian_nd(x, g):
    y = x.reshape(-1)
    Jac = torch.autograd.functional.jacobian(g, y)
    return Jac.reshape(-1, y.shape[0])
_pipe_mod.getJacobian = _getJacobian_nd

# Extended GenerateBatch calls update_h(), which would overwrite our nonlinear h with a linear h=H@x.
# Make it a no-op so the nonlinear observation survives data generation.
SystemModel.update_h = lambda self, H: None

print("Pipeline Start")

# === ADD: global device/dtype ===
DEVICE = torch.device("cuda")
DTYPE  = torch.float32
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # optional

# === latency measurement helpers ===
import time
def _sync():
    # sync CUDA so timings are accurate (kernels are async); no-op if no GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
latency = {}  # algorithm name -> total wall-clock seconds

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results_True = 'RTSNet/AI_M_step/exp_3/r_1/True_F/'  ###############################################################################################################################################
gauss = False
path_results_False = 'RTSNet/AI_M_step/exp_3/r_1/False_F/'  ###############################################################################################################################################


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

max_iter = 3

cycles = 3

# True model
q2 = 0.01
r2 = 10

# v_db = 0
# snr_db =20.0################################################################################################################################################################################################
# r2 = 10.0**(-snr_db/10.0)
# q2 = r2/(10.0**v_db/10.0)

print('q2 is:',q2)
print('r2 is:',r2)



Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)
F = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=DEVICE, dtype=DTYPE) # State transition matrix
# F = torch.tensor([[0.999, 0.1],[0., 0.999]], device=DEVICE, dtype=DTYPE) # State transition matrix
sys_model = SystemModel(make_f(F), Q, h_nonlinear, R, args.T, args.T_test,m,n)
SystemModel.F_gen = False
m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)



# Generate 5 different F matrices for datasets (same as original)

F_matrices_for_datasets_d = []

F_test_list = [F.clone().to(DEVICE) for _ in range(args.N_T)]
a= 1
for i in range(cycles+1):
    F_matrices_for_datasets_d.append([(f*a).clone() for f in F_test_list])
    # a=a*0.95
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=0.2, many=True, randomit=False)
    # if i ==0:
    #     F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=0.2, many=True, randomit=False)
    # F_7 = torch.tensor([[0.63, 0.0021], [0.0021, 1.0299]], device=DEVICE)#DELET
    # F_test_list= [F_7.clone().to(DEVICE) for _ in range(args.N_T)]#DELET
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]

# Store all data organized by F matrix
all_inputs_by_F = []
all_targets_by_F = []
all_F_matrices = []

x0_last = None
# Generate 5 datasets (same as original)
for dataset_id in range(1, cycles+1):
    print(f"\n=== Generating Dataset {dataset_id} ===")

    F_current = F_matrices_for_datasets[dataset_id - 1]
    print(f"F matrix for dataset {dataset_id}:")
    print(F_current)

    # Create system model
    SystemModel.F_gen = False
    sys_model = SystemModel(make_f(F_matrices_for_datasets[dataset_id - 1][0]), Q, h_nonlinear, R, args.T, args.T_test,m,n)
    sys_model.InitSequence(m1_0, m2_0)

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp1_1/regular/'
    dataFileName = f'snr_0{args.T_test}_dataset_{dataset_id}.pt'
    dataFileName_F = f'snr_0_F_dataset_{dataset_id}.pt'

    # Generate data
    print(f"Generating data for dataset {dataset_id}...")
    DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_F,
            delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
            randomLength=False, Test=True, F_gen=F_matrices_for_datasets[dataset_id - 1],
            H_gen=[torch.eye(n, device=DEVICE, dtype=DTYPE) for _ in range(args.N_T)], x0_list= x0_last)

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F, map_location=DEVICE)

    x_last = test_target[:,:,-1].clone()
    x0_last = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))] #list of [m,1]
    print('x_000000000000000000',x0_last)

    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")

    # Store in our organized lists
    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)

##############################################################################################
##estimate Q and R from data
# if gauss:
#     combined_target = torch.cat(all_targets_by_F, dim=2)
#     combined_input = torch.cat(all_inputs_by_F, dim=2)
#     print('Combined shapes for QR estimation:', combined_input.shape, combined_target.shape)  # sanity: [N_T, n, 5*T_test], [N_T, m, 5*T_test]
#     Q_hat, R_hat = estimate_QR(combined_input, combined_target)
#     Q = Q_hat
#     R = R_hat
#     sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)


#############################################################################
# BiGRU baseline (supervised smoother, obs -> states; no EMKF, no F).
# Loads the model trained by for_paper_M_network_training_3datasets_no_linear_h.py.
#############################################################################
destination_folder = 'RTSNet//AI_M_step/exp_3/r_10/EMKF/False/'
bgru_path = destination_folder + 'bigru_smoother_3ds.pt'
print('\n=== BiGRU baseline test ===')
bgru_mse_lin_sum = 0.0
_sync(); _t_bgru_start = time.perf_counter()
for dataset_id in range(cycles):
    b_mse, b_db, _ = test_bigru_smoother(all_inputs_by_F[dataset_id], all_targets_by_F[dataset_id],
                                         load_path=bgru_path, device=DEVICE)
    bgru_mse_lin_sum += float(b_mse)
    print(f"  BiGRU dataset {dataset_id + 1}: {b_db:.3f} dB")
_sync(); latency['BiGRU (test)'] = time.perf_counter() - _t_bgru_start
bgru_final_mse_db = 10 * torch.log10(torch.tensor(bgru_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"Average BiGRU MSE: {bgru_final_mse_db:.3f} dB")

#################################################################################################

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
# Baseline: Test with TRUE F matrices using NNTest
print('\n=== Baseline: MSE with TRUE F matrices ===')
true_F_results = []
true_mse_lin_sum = 0.0
_sync(); _t_true_start = time.perf_counter()
for dataset_id in range(cycles):
    print(f"\n--- Testing Dataset {dataset_id + 1} with TRUE F ---")

    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id][0]

    # Set up system model with true F
    # Nominal H (identity): the checkpoints lack an H attr and NNTest_no_p restores it from SysModel.H;
    # the true observation Jacobian is recomputed per-step via getJacobian for the nonlinear h.
    sys_model_true = SystemModel(make_f(true_F_for_this_dataset), Q,h_nonlinear, R, args.T, args.T_test,m,n,
                                 H=torch.eye(n, device=DEVICE, dtype=DTYPE))
    sys_model_true.InitSequence(m1_0, m2_0)

    # Set F_test for the model (needed by NNTest)
    F_test_list = F_matrices_for_datasets[dataset_id]
    sys_model_true.F_test = F_test_list


    if dataset_id == 0:# Use NNTest to get results with TRUE F
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_true, test_input, test_target, load_model_path=path_results_True_rts, generate_f=False,init_x_list=None, init_P_list=None,non_linear_h=True)
    else:
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_true, test_input, test_target, load_model_path=path_results_True_rts,generate_f=False,init_x_list=xT0_last, init_P_list=pT0_last,non_linear_h=True)


    #[self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, torch.stack(x_out_list), t, torch.stack(P_smooth_list), V_list, self.model.K_T_list,
                # self.MSE_test_psmooth_dB_avg, self.MSE_test_psmooth_std]
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

_sync(); latency['TRUE F (NNTest_no_p)'] = time.perf_counter() - _t_true_start
average_true_F_mse_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))



############################################################################# create the datadestination for the models
# The folder where the new copies will be saved.
###############################################################################################################################################
destination_path_M = destination_folder + 'mnet_r001_3ds.pt'
# destination_path_M = destination_folder + 'M_net_trained_3_datasets_no_mult.pt'
# joint (1 mnet + 1 RTSNet) checkpoints from joint_train_mnet_rtsnet_3_datasets
# destination_path_M_joint   = destination_folder + 'joint_mnet_1m1r_r001.pt'
# destination_path_RTS_joint = destination_folder + 'joint_rtsnet_1m1r_r001.pt'
destination_path_M_joint   = destination_folder + 'joint_mnet_1m1r_r10_new.pt'
destination_path_RTS_joint = destination_folder + 'joint_rtsnet_1m1r_r10_new.pt'
#############################################################################
# AI EMKF Sequential Testing
print('\n=== AI EMKF Sequential Learning and Testing ===')

# Initial F guess for all datasets
F_initial_guess_1 = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)
F_initial_guess = [F_initial_guess_1.clone() for _ in range(args.N_T)]
# Process each dataset sequentially
emkf_mse_lin_sum = 0.0
_sync(); _t_emkf_start = time.perf_counter()
for dataset_id in range(cycles):
    print(f"\n--- AI EMKF Processing Dataset {dataset_id + 1} ---")

    # Get current dataset
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    #print(f"True F for this dataset: {true_F_for_this_dataset}")
    print(f"Dataset {dataset_id + 1} input shape: {test_input.shape}")

    # Set up system model for this dataset++++++++++++++++++
    if dataset_id == 0:
        # For first dataset, use initial guess
        current_F_estimate = F_initial_guess
        print("Using initial F guess for first dataset")
    else:
        # For subsequent datasets, we would normally use AI prediction
        current_F_estimate = current_F_estimate_prev
        print(f"Using previous dataset's F as estimate: {current_F_estimate[0]}")

    # Create system model with current F estimate
    sys_model_ai = SystemModel(make_f(current_F_estimate[0]), Q, h_nonlinear, R, args.T, args.T_test,m,n,
                               H=torch.eye(n, device=DEVICE, dtype=DTYPE))
    sys_model_ai.InitSequence(m1_0, m2_0)

    # Set up F_test and F_test_TRUE for EMKF
    sys_model_ai.F_test = current_F_estimate
    sys_model_ai.F_test_TRUE = true_F_for_this_dataset

    # Run Test_Only_EMKF (this will iteratively improve F estimates)
    print(f"Running Test_Only_EMKF on dataset {dataset_id + 1}...")

    if dataset_id == 0:
        test_losses, test_f_losses, final_F_list,  last_x_list,   = RTSNet_Pipeline.test_mstep_net(sys_model_ai, test_input, test_target,
            destination_path_RTS=path_results_wrong_rts, destination_path_M=destination_path_M, num_em_iters=3,generate_f= False,non_linear_h=True)
    else:
        test_losses, test_f_losses, final_F_list,  last_x_list,   = RTSNet_Pipeline.test_mstep_net(sys_model_ai, test_input, test_target,
            destination_path_RTS=path_results_wrong_rts, destination_path_M=destination_path_M ,num_em_iters=3, generate_f= False, init_x_list=x0_em_last, init_P_list=p0_em_last,non_linear_h=True)

    emkf_mse_lin_sum += float(test_losses[-1])
    current_F_estimate_prev = final_F_list
    # current_F_estimate_prev = F_initial_guess
    # Prepare initials for NEXT dataset

    x0_em_last = last_x_list
    p0_em_last = sys_model_ai.m2x_0.clone().detach()
###############################delet
    # last_x_list = test_target[:,:,-1]
    # last_P_list = torch.eye(2, device="cuda")
    # x0_em_last = [last_x_list[j].unsqueeze(-1).clone() for j in range(len(last_x_list))]
    # p0_em_last = [last_P_list.clone() for j in range(len(last_x_list))]
    ##########################

    assert x0_em_last[0].ndim == 2 and x0_em_last[0].shape[1] == 1, f"x0 shape off: {x0_em_last[0].shape}"
_sync(); latency['AI EMKF (test_mstep_net)'] = time.perf_counter() - _t_emkf_start
emkf_final_mse_db = 10 * torch.log10(torch.tensor(emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

#############################################################################
# JOINT AI EMKF Sequential Testing (1 mnet + 1 RTSNet, jointly trained).
# Same test_mstep_net, but with the joint RTSNet + joint mnet, and the same
# F/x carryover across the 3 datasets.
#############################################################################
print('\n=== JOINT AI EMKF Sequential Testing ===')
joint_mse_lin_sum = 0.0
current_F_estimate_prev_j = None
x0j_last = p0j_last = None
_sync(); _t_joint_start = time.perf_counter()
for dataset_id in range(cycles):
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    current_F_estimate_j = F_initial_guess if dataset_id == 0 else current_F_estimate_prev_j

    sys_model_aj = SystemModel(make_f(current_F_estimate_j[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n,
                               H=torch.eye(n, device=DEVICE, dtype=DTYPE))
    sys_model_aj.InitSequence(m1_0, m2_0)
    sys_model_aj.F_test = current_F_estimate_j
    sys_model_aj.F_test_TRUE = true_F_for_this_dataset

    kw = dict(destination_path_RTS=destination_path_RTS_joint,
              destination_path_M=destination_path_M_joint,
              num_em_iters=3, generate_f=False, non_linear_h=True)
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
_sync(); latency['JOINT AI EMKF (test_mstep_net)'] = time.perf_counter() - _t_joint_start
joint_final_mse_db = 10 * torch.log10(torch.tensor(joint_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

#############################################################################
# Baseline: Test with INITIAL GUESS F using NNTest
print('\n=== Baseline: MSE with INITIAL GUESS F ===')
initial_guess_results = []
init_mse_lin_sum = 0.0

_sync(); _t_init_start = time.perf_counter()
for dataset_id in range(cycles):
    print(f"\n--- Testing Dataset {dataset_id + 1} with INITIAL GUESS F ---")

    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    # Set up system model with initial guess F
    sys_model_init = SystemModel(make_f(F_initial_guess[0]), Q,h_nonlinear , R, args.T, args.T_test,m,n,
                                 H=torch.eye(n, device=DEVICE, dtype=DTYPE))
    sys_model_init.InitSequence(m1_0, m2_0)

    # Set F_test for the model - one F per sequence
    # Since we have 20 sequences (args.N_T), we need 20 F matrices
    F_test_list = F_initial_guess
    sys_model_init.F_test = F_test_list#THIS IS A F IN THE LONG OF THE SEQ

    # Use NNTest to get results with initial guess F

    if dataset_id ==0:
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_init, test_input, test_target, load_model_path=path_results_wrong_rts,
                                          generate_f=False,non_linear_h=True)
    else:
        results = RTSNet_Pipeline.NNTest_no_p(sys_model_init, test_input, test_target, load_model_path=path_results_wrong_rts,
            generate_f=False,init_x_list =xF0_last,init_P_list = pF0_last,non_linear_h=True)

    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    init_mse_lin_sum += float(results[1])  # results[1] = linear MSE avg                model_e_q0_rand_true

    # >>> propagate last smoothed x_T and P_T to next dataset <<<
    x_last = results[3][:, :, -1].clone()  # [N_T, m]
    xF0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]  # list of [m,1]
    pF0_last = sys_model_init.m2x_0.clone().detach()





    initial_guess_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS F MSE: {mse_db:.3f} dB")

_sync(); latency['INITIAL GUESS (NNTest_no_p)'] = time.perf_counter() - _t_init_start
average_initial_guess_mse_db = 10 * torch.log10(torch.tensor(init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"Average MSE with INITIAL GUESS F: {average_initial_guess_mse_db:.3f} dB")



#############################################################################
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE F (perfect):          {average_true_F_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF):   {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (regular mnet): {emkf_final_mse_db:.3f} dB")
print(f"EMKF FINAL (joint mnet):   {joint_final_mse_db:.3f} dB")
print(f"BiGRU (baseline):          {bgru_final_mse_db:.3f} dB")
print(f"Regular vs init:  {(average_initial_guess_mse_db - emkf_final_mse_db):.3f} dB   Gap to TRUE F: {(emkf_final_mse_db - average_true_F_mse_db):.3f} dB")

#############################################################################
# LATENCY REPORT (average time per sequence)
#############################################################################
# Each algorithm processes the same workload: `cycles` datasets x args.N_T sequences.
total_seqs = cycles * args.N_T
print('\n=== LATENCY (avg per sequence) ===')
print(f"{'Algorithm':<30}{'total (s)':>12}{'per-seq (ms)':>16}")
for name, secs in latency.items():
    print(f"{name:<30}{secs:>12.3f}{1000.0 * secs / total_seqs:>16.3f}")
