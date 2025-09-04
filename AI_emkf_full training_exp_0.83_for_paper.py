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
path_results_True = 'RTSNet/paper/exp_1/r_001/True_F/'###############################################################################################################################################
gauss = False
path_results_False = 'RTSNet/paper/exp_1/r_001/False_F/'###############################################################################################################################################


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

max_iter = 3

cycles = 3

# True model
q2 = 0.01
r2 =0.01

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
H = torch.tensor([[1., 1.],
                  [0.25, 1.]], device=DEVICE, dtype=DTYPE)
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
SystemModel.F_gen = False
m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)


# Generate 5 different F matrices for datasets (same as original)

F_matrices_for_datasets_d = []

F_test_list = [F.clone().to(DEVICE) for _ in range(args.N_T)]
a=1
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
    sys_model = SystemModel(F_matrices_for_datasets[dataset_id - 1][0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp1_1/regular/'
    dataFileName = f'snr_0{args.T_test}_dataset_{dataset_id}.pt'
    dataFileName_F = f'snr_0_F_dataset_{dataset_id}.pt'

    # Generate data
    print(f"Generating data for dataset {dataset_id}...")
    DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_F,
            delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
            randomLength=False, Test=True, F_gen=F_matrices_for_datasets[dataset_id - 1], x0_list= x0_last)

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
if gauss:
    combined_target = torch.cat(all_targets_by_F, dim=2)
    combined_input = torch.cat(all_inputs_by_F, dim=2)
    print('Combined shapes for QR estimation:', combined_input.shape, combined_target.shape)  # sanity: [N_T, n, 5*T_test], [N_T, m, 5*T_test]
    Q_hat, R_hat = estimate_QR(combined_input, combined_target)
    Q = Q_hat
    R = R_hat
    sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)

#################################################################################################

path_results_True_rts = path_results_True+'best-rts_true.pt'
# path_results_True_rts2 = path_results_True+'best-modelwith_p2_q.pt'
path_results_True_psmooth = path_results_True+'best-psmooth_true.pt'
path_results_wrong_rts = path_results_False+'best-rts_false.pt'
# path_results_2_rts2 = path_results_False+'best-rts_with_pJOINT_q.pt'
path_results_wrong_psmooth = path_results_False+'best-psmooth_false.pt'
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


    if dataset_id == 0:# Use NNTest to get results with TRUE F
        results = RTSNet_Pipeline.NNTest(sys_model_true, test_input, test_target, load_model_path=path_results_True_rts, load_p_smoothe_model_path=path_results_True_psmooth,
            generate_f=False,init_x_list=None, init_P_list=None)
    else:
        results = RTSNet_Pipeline.NNTest(sys_model_true, test_input, test_target, load_model_path=path_results_True_rts, load_p_smoothe_model_path=path_results_True_psmooth,
                                         generate_f=False,init_x_list=xT0_last, init_P_list=pT0_last)


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
    p_last = results[5][:, :, :, -1].clone()         # [N_T, m, m]
    xT0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]  # list of [m,1]
    pT0_last = [p_last[j] for j in range(p_last.size(0))]

average_true_F_mse_db = 10 * torch.log10(torch.tensor(true_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))



############################################################################# create the datadestination for the models
model_pathes = []
psmooth_pathes = []
# The folder where the new copies will be saved.
destination_folder = 'RTSNet/paper/exp_1/r_001/EMKF/False/'###############################################################################################################################################
for i in range(max_iter):
    file_rtsnet = f"model_e_q{i}_rand_false_trained.pt"
    file_psmooth = f"psmooth_e_q{i}_rand_false_trained.pt"
    # Build the full destination path
    destination_path_RTS = destination_folder + file_rtsnet
    destination_path_PSMOOTH = destination_folder + file_psmooth
    model_pathes.append(destination_path_RTS)
    psmooth_pathes.append(destination_path_PSMOOTH)
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

    print(f"True F for this dataset: {true_F_for_this_dataset}")
    print(f"Dataset {dataset_id + 1} input shape: {test_input.shape}")

    # Set up system model for this dataset++++++++++++++++++
    if dataset_id == 0:
        # For first dataset, use initial guess
        current_F_estimate = F_initial_guess
        print("Using initial F guess for first dataset")
    else:
        # For subsequent datasets, we would normally use AI prediction
        current_F_estimate = current_F_estimate_prev
        print(f"Using previous dataset's F as estimate: {current_F_estimate}")

    # Create system model with current F estimate
    sys_model_ai = SystemModel(current_F_estimate[0], Q, H, R, args.T, args.T_test)
    sys_model_ai.InitSequence(m1_0, m2_0)

    # Set up F_test and F_test_TRUE for EMKF
    sys_model_ai.F_test = current_F_estimate
    sys_model_ai.F_test_TRUE = true_F_for_this_dataset

    # Run Test_Only_EMKF (this will iteratively improve F estimates)
    print(f"Running Test_Only_EMKF on dataset {dataset_id + 1}...")

    if dataset_id == 0:
        test_losses, test_f_losses, final_F_list,  last_x_list, last_P_list,final_F_list2  = RTSNet_Pipeline.Test_Only_EMKF(sys_model_ai, test_input, test_target,
            load_base_rtsnet=model_pathes, load_base_psmooth=psmooth_pathes,emkf_iterations=3,generate_f= False)
    else:
        test_losses, test_f_losses, final_F_list,  last_x_list, last_P_list,final_F_list2  = RTSNet_Pipeline.Test_Only_EMKF(sys_model_ai, test_input, test_target,
            load_base_rtsnet=model_pathes, load_base_psmooth=psmooth_pathes,emkf_iterations=3, generate_f= False, init_x_list=x0_em_last, init_P_list=p0_em_last)

    emkf_mse_lin_sum += float(test_losses[-1])
    # current_F_estimate_prev = final_F_list
    current_F_estimate_prev = F_initial_guess
    # Prepare initials for NEXT dataset

    x0_em_last = [last_x_list[j].clone() for j in range(len(last_x_list))]
    p0_em_last = [last_P_list[j].clone() for j in range(len(last_P_list))]
###############################delet
    # last_x_list = test_target[:,:,-1]
    # last_P_list = torch.eye(2, device="cuda")
    # x0_em_last = [last_x_list[j].unsqueeze(-1).clone() for j in range(len(last_x_list))]
    # p0_em_last = [last_P_list.clone() for j in range(len(last_x_list))]
    ##########################

    assert x0_em_last[0].ndim == 2 and x0_em_last[0].shape[1] == 1, f"x0 shape off: {x0_em_last[0].shape}"
emkf_final_mse_db = 10 * torch.log10(torch.tensor(emkf_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))

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

    # Set F_test for the model - one F per sequence
    # Since we have 20 sequences (args.N_T), we need 20 F matrices
    F_test_list = F_initial_guess
    sys_model_init.F_test = F_test_list#THIS IS A F IN THE LONG OF THE SEQ

    # Use NNTest to get results with initial guess F

    if dataset_id ==0:
        results = RTSNet_Pipeline.NNTest(sys_model_init, test_input, test_target, load_model_path=path_results_wrong_rts,
                                         load_p_smoothe_model_path=path_results_wrong_psmooth, generate_f=False)
    else:
        results = RTSNet_Pipeline.NNTest(sys_model_init, test_input, test_target, load_model_path=path_results_wrong_rts,
            load_p_smoothe_model_path=path_results_wrong_psmooth, generate_f=False,init_x_list =xF0_last,init_P_list = pF0_last)

    # Extract MSE in dB
    mse_db = results[2]  # MSE_test_dB_avg
    init_mse_lin_sum += float(results[1])  # results[1] = linear MSE avg                model_e_q0_rand_true

    # >>> propagate last smoothed x_T and P_T to next dataset <<<
    x_last = results[3][:, :, -1].clone()  # [N_T, m]
    p_last = results[5][:, :, :, -1].clone()  # [N_T, m, m]
    xF0_last = [x_last[j].unsqueeze(-1) for j in range(x_last.size(0))]  # list of [m,1]
    pF0_last = [p_last[j] for j in range(p_last.size(0))]





    initial_guess_results.append(mse_db)
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS F MSE: {mse_db:.3f} dB")

average_initial_guess_mse_db = 10 * torch.log10(torch.tensor(init_mse_lin_sum / cycles, device=DEVICE, dtype=DTYPE))
print(f"Average MSE with INITIAL GUESS F: {average_initial_guess_mse_db:.3f} dB")

#############################################################################
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE F (perfect):        {average_true_F_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (learned):    {emkf_final_mse_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - emkf_final_mse_db):.3f} dB")
print(f"Gap to perfect (TRUE F): {(emkf_final_mse_db - average_true_F_mse_db):.3f} dB")
