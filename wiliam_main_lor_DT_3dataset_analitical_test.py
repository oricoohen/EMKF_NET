import torch
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from emkf.main_emkf_func import EMKF_H_analitic  # Changed to H estimation
from Simulations.utils import DataLoader, DataGen, estimate_QR

import random
device = torch.device("cuda")
DTYPE = torch.float32
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



args = config.general_settings()
args.N_T = 100  # Number of test examples (size of the test dataset used to evaluate performance).100

args.T_test = 30 # Length of the time series for test sequences.
state = 1
cycle = 3
# torch.manual_seed(state)          # add this one line
# True model
q2 = 1.
r2 = 1.
# v_db   = 0   # in dB, = 10*log10(r2/q2)
# snr_db = 0   # in dB, paper convention

# compute variances:
# r2 = 10.0 ** (-snr_db / 10.0)
# q2 = r2 / (10.0 ** (v_db / 10.0))
Q = (q2 * Q_structure).to(device=device, dtype=DTYPE)
R = (r2 * R_structure).to(device=device, dtype=DTYPE)

# F is now FIXED (known true model) - NO diversity
F_fixed = torch.tensor([[0.83, 0.2],
              [0.2, 0.83]], device=device, dtype=DTYPE)

# H is TRUE but will be diverse and unknown
H_true = torch.tensor([[1., 1.], [0.25, 1.]], device=device, dtype=DTYPE)

# Move initial conditions to GPU
m1_0_gpu = m1_0.to(device=device, dtype=DTYPE)
m2_0_gpu = m2_0.to(device=device, dtype=DTYPE)

SystemModel.F_gen = False
sys_model = SystemModel(F_fixed, Q, H_true, R, args.T, args.T_test)
sys_model.InitSequence(m1_0_gpu, m2_0_gpu)

# Define diverse H matrices for the datasets (cycle+1 different H matrices)
#############################################
'''Create diverse H matrices which change across datasets
F is FIXED (same for all), H varies
'''
#############################################

H_matrices_for_datasets_d = []
H_test_list = [H_true.clone() for _ in range(args.N_T)]  # Start with true H for all sequences

for i in range(cycle + 1):
    # Deep copy the list of H tensors
    H_matrices_for_datasets_d.append([h.clone() for h in H_test_list])
    # Rotate H for the next dataset
    H_test_list = rotate_F(H_test_list, i=0, j=1, theta=0.2, many=True, randomit=False)

H_matrices_for_datasets = H_matrices_for_datasets_d[1:]  # Use the rotated H matrices for datasets

# F is FIXED for all datasets (create lists for consistency)
F_fixed_list = [F_fixed.clone() for _ in range(args.N_T)]  # Same F for all sequences

# Store all inputs and targets organized by H matrix
all_inputs_by_H = []
all_targets_by_H = []
all_H_matrices = []

x0_list = None

# Generate datasets with diverse H (F is fixed)
for dataset_id in range(0, cycle):
    print(f"\n=== Generating Dataset {dataset_id} ===")

    # Select H matrix for this dataset (F is FIXED)
    H_current = H_matrices_for_datasets[dataset_id ]
    print(f"H matrix for dataset {dataset_id}:")
    print(H_current[0])
    print(f"F matrix (FIXED for all datasets):")
    print(F_fixed)

    # Create system model with FIXED F and current H
    sys_model = SystemModel(F_fixed, Q, H_current[0], R, args.T, args.T_test)
    sys_model.InitSequence(m1_0_gpu, m2_0_gpu)

    # Create folder and file names
    dataFolderName = f'Simulations/Linear_canonical/paper/exp1_H/regular/'
    dataFileName = f'snr_0{args.T_test}_dataset_{dataset_id}.pt'
    dataFileName_H = f'snr_0_H_dataset_{dataset_id}.pt'

    # Generate data with FIXED F and DIVERSE H
    print(f"Generating data for dataset {dataset_id}...")

    DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_H,
        fileName_H=dataFolderName + dataFileName_H,
        delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
        randomLength=False, Test=True,
        F_gen=F_fixed_list,  # Fixed F for all
        H_gen=H_current,  # Diverse H for this dataset
        x0_list=x0_list)

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=device)
    [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(
        dataFolderName + dataFileName_H, map_location=device)

    # test_target: [N_T, m, T]
    x_last = test_target[:, :, -1].clone()
    x0_list = [x_last[j].unsqueeze(-1).clone() for j in range(x_last.size(0))]

    print(f"Dataset {dataset_id} created successfully!")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")
    print(f"H matrix stored: {H_test_mat_list[0]}")

    # Store in our organized lists
    all_inputs_by_H.append(test_input)
    all_targets_by_H.append(test_target)
    all_H_matrices.append(H_test_mat_list)

#########################################################################################################
# RTS_out has shape [N_T, m, T] and is our "x_est"
# P_smooth has shape [N_T, m, m, T] and is the covariance we want to evaluate
# test_target has shape [N_T, m, T] and is our "x_true"

# Initial H guess (wrong H to start EMKF) - ON GPU
H_initial_guess = torch.tensor([[1., 1.], [0.25, 1.]], device=device, dtype=DTYPE)
H_current_estimate = [H_initial_guess.clone() for _ in range(args.N_T)]
H_initial_estimate = [H_initial_guess.clone() for _ in range(args.N_T)]


#############################################################################
# Calculate MSE for each dataset with TRUE H (what would happen without EMKF)
print('\n=== MSE with TRUE H matrices ===')
true_mse_lin_sum = 0.0
for dataset_id in range(cycle):
    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]
    true_H_for_this_dataset = H_matrices_for_datasets[dataset_id]

    # Use the TRUE H matrix for this dataset
    sys_model = SystemModel(F_fixed, Q, true_H_for_this_dataset[0], R, args.T, args.T_test)
    sys_model.InitSequence(m1_0_gpu, m2_0_gpu)

    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                                        F=F_fixed_list, H=true_H_for_this_dataset, generate_f=False, generate_h=False)
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                        F=F_fixed_list, H=true_H_for_this_dataset, generate_f=False, generate_h=False, init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone() for k in range(args.N_T)]
    true_mse_lin_sum += _mse_avg.item()
    print(f"Dataset {dataset_id + 1} - TRUE H MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with true H
average_true_H_mse_db = 10*torch.log10(torch.tensor(true_mse_lin_sum / cycle))

print(f"Average MSE with TRUE H matrices: {average_true_H_mse_db:.3f} dB")

#############################################################################
# Calculate MSE for each dataset with INITIAL GUESS H
print('\n=== MSE with INITIAL GUESS H  ===')
mse_total_false = 0
for dataset_id in range(cycle):
    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]

    # Use the initial guess H for ALL datasets
    sys_model = SystemModel(F_fixed, Q, H_initial_guess, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0_gpu, m2_0_gpu)

    if dataset_id == 0:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                                        F=F_fixed_list, H=H_initial_estimate, generate_f=False, generate_h=False)
    else:
        [_mse_arr, _mse_avg, _mse_db, x_list, p_list, _] = S_Test(sys_model, test_input, test_target,
                                        F=F_fixed_list, H=H_initial_estimate, generate_f=False, generate_h=False, init_x_list=x0_last, init_P_list=p0_last)
    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_list[k, :, -1].unsqueeze(-1).clone() for k in range(args.N_T)]
    p0_last = [p_list[k, :, :, -1].clone()             for k in range(args.N_T)]
    mse_total_false +=_mse_avg.item()
    print(f"Dataset {dataset_id + 1} - INITIAL GUESS H MSE: {_mse_db.item():.3f} dB")

# Calculate and print average with initial guess
average_initial_guess_mse_db = 10*torch.log10(torch.tensor(mse_total_false/cycle))
print(f"Average MSE with INITIAL GUESS H: {average_initial_guess_mse_db:.3f} dB")

###############################################################
# Calculate MSE for each dataset with EMKF H
print('\n=== MSE with EMKF H matrices ===')
mse_total = 0
for dataset_id in range(cycle):
    print(f"\n--- EMKF Iteration {dataset_id + 1} ---")
    print(f"Using dataset {dataset_id + 1}")

    # Get data for this dataset
    test_input = all_inputs_by_H[dataset_id]
    test_target = all_targets_by_H[dataset_id]
    true_H_for_this_dataset = H_matrices_for_datasets[dataset_id]

    print(f"True H for this dataset:\n{true_H_for_this_dataset[0]}")
    print(f"Initial H guess:\n{H_current_estimate[0]}")

    # Create system model for EMKF with FIXED F
    sys_model = SystemModel(F_fixed, Q, H_current_estimate[0], R, args.T, args.T_test)
    sys_model.InitSequence(m1_0_gpu, m2_0_gpu)

    # Store true H for comparison in EMKF
    sys_model.H_test_TRUE = true_H_for_this_dataset

    # Run EMKF_H with current estimate as initial guess
    print(f"Running EMKF_H on dataset {dataset_id + 1}...")

    if dataset_id == 0:
        H_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last = EMKF_H_analitic(
            sys_model, F_fixed_list, H_current_estimate,Q, R, test_input, m1_0_gpu, m2_0_gpu, test_target,
            max_it=3, generate_h=False,init_x_list=None, init_P_list=None)
    else:
        H_matrices, likelihoods, iterations_list, mse_avg_T, x_last, p_last = EMKF_H_analitic(
            sys_model, F_fixed_list, H_current_estimate,
            Q, R, test_input, m1_0_gpu, m2_0_gpu, test_target,
            max_it=3, generate_h=False,init_x_list=x0_last, init_P_list=p0_last)

    # >>> propagate: last smoothed x_T and P_T become next dataset's initials <<<
    x0_last = [x_last[k].clone() for k in range(args.N_T)]
    p0_last = [p_last[k].clone() for k in range(args.N_T)]

    # H_matrices has N_T (amount of seq) list inside where each list has max_it + initial guess H matrices
    # Update H estimate for next iteration (use the result from EMKF)
    H_current_estimate = [Hs_per_seq[-1].clone() for Hs_per_seq in H_matrices]

    print(f"EMKF H evolution for first sequence:")
    for iter_idx, H_est in enumerate(H_matrices[0]):
        h_error = torch.norm(H_est - true_H_for_this_dataset[0]).item()
        print(f"  Iteration {iter_idx}: H error = {h_error:.4f}")

    print(f"Final H estimate (seq 0):\n{H_current_estimate[0]}")
    print(f"True H was:\n{true_H_for_this_dataset[0]}")

    mse_total += mse_avg_T.item()

MSE_total_db = 10 * torch.log10(torch.tensor(mse_total / cycle))

print("\n=== EMKF iterations completed ===")
print(f"Final H estimate:\n{H_current_estimate[0]}")
print(f"Original H:\n{H_true}")
print(f"\nAverage MSE across all datasets from final H estimate: {MSE_total_db:.3f} dB")

#############################################################################
# Summary comparison
print('\n=== SUMMARY COMPARISON ===')
print(f"TRUE H (perfect):        {average_true_H_mse_db:.3f} dB")
print(f"INITIAL GUESS (no EMKF): {average_initial_guess_mse_db:.3f} dB")
print(f"EMKF FINAL (learned):    {MSE_total_db:.3f} dB")
print(f"EMKF improvement over initial: {(average_initial_guess_mse_db - MSE_total_db):.3f} dB")
print(f"Gap to perfect (TRUE H): {(MSE_total_db - average_true_H_mse_db):.3f} dB")


