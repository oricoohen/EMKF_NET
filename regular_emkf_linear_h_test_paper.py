import torch
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from emkf.main_emkf_func import EMKF_F_analitic, EMKF_F_solo
from Simulations.utils import DataLoader, DataGen, estimate_QR
from RTSNet.PsmoothNN import PsmoothNN
import numpy as np
from torch.distributions import Exponential




# # # For PyTorch
torch.manual_seed(1)


args = config.general_settings()
args.N_T = 50  # Number of test examples (size of the test dataset used to evaluate performance).100

args.T_test = 30 # Length of the time series for test sequences.
state = 1
cycle = 3
# torch.manual_seed(state)          # add this one line
# True model
q2 = 0.01
r2 = 0.01
# v_db   = 0   # in dB, = 10*log10(r2/q2)
# snr_db = 0   # in dB, paper convention

# compute variances:
# r2 = 10.0 ** (-snr_db / 10.0)
# q2 = r2 / (10.0 ** (v_db / 10.0))
Q = q2 * Q_structure
R = r2 * R_structure

# Q = torch.tensor([[0.0202, 0.0102],
#               [0.0102, 0.02]])
# R = torch.tensor([[2.054, 1.023],
#               [1.023, 1.9329]])

F = torch.tensor([[0.83, 0.2],
              [0.2, 0.83]])
F_rot = torch.tensor([[0.63, 0.0021], [0.0021, 1.0299]])  # State transition matrix
F_initial_guest_1 = F
H = torch.tensor([[1., 1.], [0.25, 1.]])


SystemModel.F_gen = False
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)

# Define 5 different F matrices for the 5 datasets
#############################################
'''i create 5 different F matrices which from every one of them i compute to seq
we will enter the f initial and 2 seq from rotated f into the em ans sum up the mse
'''
#############################################

F_matrices_for_datasets_d =[]
F_test_list = [F.clone() for _ in range(args.N_T)]  # 1 F per seq (same F)
# F_rot_list = [F_rot.clone() for _ in range(args.N_T)]  # delet
for i in range(cycle +1):
    # deep copy the list of tensors
    F_matrices_for_datasets_d.append([f.clone() for f in F_test_list])
    # rotate PER-SEQUENCE for the next dataset (rotate_F should return a list of [n,n])
    F_test_list = rotate_F(F_matrices_for_datasets_d[i], i=0, j=1, theta=0.2, many=True, randomit=False)
    # F_test_list = F_rot_list#delet
F_matrices_for_datasets = F_matrices_for_datasets_d[1:]  # Use the last 5 matrices for datasets 1 to 5

# Store all inputs and targets organized by F matrix
all_inputs_by_F = []
all_targets_by_F = []
all_F_matrices = []

x0_list = None
# Generate 5 datasets
for dataset_id in range(1, cycle+1):
    print(f"\n=== Generating Dataset {dataset_id} ===")

    # Select F matrix for this dataset
    F = F_matrices_for_datasets[dataset_id - 1]
    print(f"F matrix for dataset {dataset_id}:")
    print(F)
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
        randomLength=False, Test=True, F_gen=F_matrices_for_datasets[dataset_id - 1],x0_list=x0_list)
    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName + dataFileName, weights_only=True, map_location="cpu")
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F)

    # test_target: [N_T, m, T]
    x_last = test_target[:, :, -1].clone()
    x0_list = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]  # list of [m,1] tensors

    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")
    print(f"F matrix stored: {F_test_mat_list[0]}")

    # Store in our organized lists
    all_inputs_by_F.append(test_input)
    all_targets_by_F.append(test_target)
    all_F_matrices.append(F_test_mat_list)

#########################################################################################################
# RTS_out has shape [N_T, n, T] and is our "x_est"
# P_smooth has shape [N_T, n, n, T] and is the covariance we want to evaluate
# test_target has shape [N_T, n, T] and is our "x_true"

F_initial_gues_1 = torch.tensor([[0.83, 0.2],[0.2, 0.83]])
# F_initial_gues_1 = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])#delete
F_current_estimate = [F_initial_gues_1 .clone() for _ in range(args.N_T)]
F_initial_estimate = [F_initial_gues_1 .clone() for _ in range(args.N_T)]

###############################################################################################
##estimate Q and R from data
# gauss = False
# if gauss:
#     combined_target = torch.cat(all_targets_by_F, dim=2)
#     combined_input = torch.cat(all_inputs_by_F, dim=2)
#     print('Combined shapes for QR estimation:', combined_input.shape, combined_target.shape)  # sanity: [N_T, n, 5*T_test], [N_T, m, 5*T_test]
#     Q_hat, R_hat = estimate_QR(combined_input, combined_target)
#     Q = Q_hat
#     R = R_hat
#     sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)

#################################################################################################



#############################################################################
# Calculate MSE for each dataset with TRUE F (what would happen without EMKF)
print('\n=== MSE with TRUE F matrices ===')
true_mse_lin_sum = 0.0
for dataset_id in range(cycle):
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    # Use the TRUE F matrix for this dataset
    sys_model = SystemModel(true_F_for_this_dataset[0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                                        F=true_F_for_this_dataset,generate_f=False)
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                        F=true_F_for_this_dataset,generate_f=False, init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone() for k in range(args.N_T)]
    true_mse_lin_sum += _mse_avg.item()
    print(f"Dataset {dataset_id + 1} - TRUE F MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with true F
average_true_F_mse_db = 10*torch.log10(torch.tensor(true_mse_lin_sum / cycle))

print(f"Average MSE with TRUE F matrices: {average_true_F_mse_db:.3f} dB")

#############################################################################
# Calculate MSE for each dataset with INITIAL GUESS
print('\n=== MSE with INITIAL GUESS F  ===')
mse_total_false = 0
for dataset_id in range(cycle):
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]

    # Use the initial guess F for ALL datasets
    sys_model = SystemModel(F_initial_gues_1, Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                                        F=F_initial_estimate,generate_f=False)
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                        F=F_initial_estimate,generate_f=False, init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone()             for k in range(args.N_T)]
    mse_total_false +=_mse_avg.item()
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with initial guess
average_initial_guess_mse_db = 10*torch.log10(torch.tensor(mse_total_false/cycle))
print(f"Average MSE with INITIAL GUESS F: {average_initial_guess_mse_db:.3f} dB")

###############################################################
# Calculate MSE for each dataset with emkf F
print('\n=== MSE with EMKF  matrices ===')
mse_total =0
for dataset_id in range(cycle):
    print(f"\n--- EMKF Iteration {dataset_id + 1} ---")
    print(f"Using dataset {dataset_id + 1}")

    # Get data for this dataset
    test_input = all_inputs_by_F[dataset_id]
    test_target = all_targets_by_F[dataset_id]
    true_F_for_this_dataset = F_matrices_for_datasets[dataset_id]

    # print(f"True F for this dataset: {true_F_for_this_dataset}")
    # print(f"Initial F guess: {F_current_estimate[0]}")

    # Create system model for EMKF
    sys_model = SystemModel(F_current_estimate[0], Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)
    # Run EMKF with current estimate as initial guess
    print(f"Running EMKF on dataset {dataset_id + 1}...")


    if dataset_id == 0:
        F_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last = EMKF_F_analitic(sys_model, F_current_estimate,
            H, Q, R, test_input, m1_0, m2_0,test_target, max_it=3,generate_f=False, tol_likelihood=0.01, tol_params=0.025)
    else:
        F_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last = EMKF_F_analitic(sys_model, F_current_estimate,
            H, Q, R, test_input, m1_0, m2_0,test_target, max_it=3,generate_f=False, tol_likelihood=0.01, tol_params=0.025,
                                                                                init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_last[k].clone() for k in range(args.N_T)]
    p0_last = [p_last[k].clone() for k in range(args.N_T)]
    #F_matrices has N_T(amount of seq) list inside where each list has max it + initial guess F matrices one for each T
    # Update F estimate for next iteration (use the result from EMKF)
    F_current_estimate = [Fs_per_seq[-1].clone() for Fs_per_seq in F_matrices]
    print(f"EMKF result for dataset {dataset_id + 1}: {F_matrices[0]}")
    print(f"True F was: {true_F_for_this_dataset[0]}")
    mse_total+= mse_avg_T.item()
MSE_total_db = 10 * torch.log10(torch.tensor(mse_total / cycle))

print("\n=== EMKF iterations completed ===")
print(f"Final F estimate: {F_current_estimate[0]}")
print(f"Original F: {F}")
print(f"\nAverage MSE across all datasets from final F estimate: {MSE_total_db:.3f} dB")

#############################################################################
# Summary comparison
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE F (perfect):        {average_true_F_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (learned):    {MSE_total_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - MSE_total_db):.3f} dB")
print(f"Gap to perfect (TRUE F): {(MSE_total_db - average_true_F_mse_db):.3f} dB")



# # ---- Combine the 5 blocks into one long sequence per sequence index ----
# # all_inputs_by_F[k] has shape [N_T, n, T_test]
# # all_targets_by_F[k] has shape [N_T, m, T_test]
# combined_input  = torch.cat(all_inputs_by_F,  dim=2)   # [N_T, n, 5*T_test]
# combined_target = torch.cat(all_targets_by_F, dim=2)   # [N_T, m, 5*T_test]
# T_long = combined_input.size(-1)                       # = 5 * T_test
# print('ori what is T', T_long)
# print("Combined shapes:", combined_input.shape, combined_target.shape)  # sanity: [N_T, n, T_long], [N_T, m, T_long]
#
# # ---- Evaluate INITIAL GUESS on the combined data (no learning) ----
# sys_model_long = SystemModel(F_initial_gues_1, Q, H, R, args.T, T_long)
# sys_model_long.InitSequence(m1_0, m2_0)
# args.T_test = T_long
#
# # F_initial_estimate must be a list of length N_T (one F per sequence) – you already have it
# [_arr0, init_mse_lin_avg, init_mse_db, _, _, _] = S_Test( sys_model_long, combined_input, combined_target,
#     F=F_initial_estimate, generate_f=False)
# print(f"[Combined] INITIAL-GUESS MSE (linear avg): {init_mse_lin_avg.item():.6f}")
# print(f"[Combined] INITIAL-GUESS MSE (dB):         {init_mse_db.item():.3f} dB")
#
# # ---- Run EMKF on the combined data (single big block) ----
# F_current_estimate_long = [F_initial_gues_1.clone() for _ in range(args.N_T)]
#
# F_mats_long, liks_long, iters_long,  last_iter_mse_lin, x_last_long, p_last_long = EMKF_F_analitic(
#     sys_model_long, F_current_estimate_long,
#     H, Q, R,combined_input, m1_0, m2_0, combined_target, max_it=3, generate_f=False, tol_likelihood=0.01, tol_params=0.025)
#
# print(f"[Combined] EMKF last-iter MSE (linear avg): {last_iter_mse_lin:.6f}")
# print(f"[Combined] EMKF last-iter MSE (dB):         {10*torch.log10(torch.tensor(last_iter_mse_lin)).item():.3f} dB")
#
# # If you want the final learned F per sequence:
# F_final_long = [Fs_per_seq[-1].clone() for Fs_per_seq in F_mats_long]
# print("[Combined] Example final F for seq 0:\n", F_final_long[0])









#
#
#
# #############################################################################
# print('True F matrices 1', F)
# #print('end of EMKF first:',F_matrices[0],'second', F_matrices[1])
# print(test_target.size())
# i=0
# mean_norms = []
# for seq in test_target:
#     # 1) ℓ2-norm at each time step → shape (T,)
#     norm_per_t = torch.norm(seq, p=2, dim=0)
#     if i ==0:
#         i =1
#         print('norm per t', norm_per_t)
#     # 2) average over the T time-steps → scalar
#     mean_norms.append(norm_per_t.mean())
# mean_norms_tensor = torch.stack(mean_norms)    # S = number of sequences
#
# # 2) compute the ℓ2‐norm across sequences
# norm_x = torch.norm(mean_norms_tensor, p=2, dim=0)  # scalar
#
# # 3) convert to dB
# eps = 1e-12
# norm_db = 20 * torch.log10(norm_x + eps)
#
# print('norm x',norm_db)
#
# #######FALSE_1#######
#
#
#
# #######FALSE_2#######