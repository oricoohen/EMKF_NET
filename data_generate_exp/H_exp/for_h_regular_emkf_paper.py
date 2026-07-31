import torch
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F, generate_random_H_matrices
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from emkf.main_emkf_func import EMKF_H_analitic  # Changed from EMKF_F_analitic
from Simulations.utils import DataLoader, DataGen, estimate_QR
from RTSNet.PsmoothNN import PsmoothNN
import numpy as np
from torch.distributions import Exponential


# # For NumPy
# np.random.seed(1)
#
# For PyTorch
torch.manual_seed(1)
DEVICE = torch.device("cuda")
DTYPE  = torch.float32
args = config.general_settings()
args.N_T = 175  # Number of test examples (size of the test dataset used to evaluate performance).100

args.T_test = 30 # Length of the time series for test sequences.

# True model
# choose your targets:
v_db   = 0   # in dB, = 10*log10(r2/q2)
snr_db = 0   # in dB, paper convention

# compute variances:
# r2 = 10.0 ** (-snr_db / 10.0)
# q2 = r2 / (10.0 ** (v_db / 10.0))
# print('r2=',r2)
# print('q2=',q2)
q2 = 0.01
r2 = 0.01
Q = q2 * Q_structure
R = r2 * R_structure

# F is KNOWN (fixed true model) - no diversity needed
F = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=DEVICE, dtype=DTYPE)

F_in = [F for _ in range(args.N_E)]
# H is TRUE but UNKNOWN - we'll generate diverse H matrices
H = torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)
H_in = [H for _ in range(args.N_E)]
# System model with true F and H
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)


###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName = '2x2_rq3030_T100.pt'
dataFileName_H = '2x2_H_diverse'  # Changed from 2x2_F

print("Start Data Gen")
# Generate data with FIXED F but DIVERSE H
DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_H,
        fileName_H=dataFolderName + dataFileName_H,
        delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
        randomLength=False, Test=True, F_gen=F_in)  # F_gen=F_in (fixed), H_gen=True (diverse)

print("Data Load")
[train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName + dataFileName)
[H_train_mat, H_val_mat, H_test_mat] = torch.load(dataFolderName + dataFileName_H)

print("testset size:", test_target.size())
print("Number of H matrices:", len(H_test_mat))
H_test_mat_0 = H_test_mat[0]
print('First true H matrix:\n', H_test_mat_0)


# F is FIXED (true model) - same for all sequences
F_fixed = F

# Create list of fixed F for testing (all the same)
F_test_mat = []
for i in range(args.N_T):
    F_test_mat.append(F_fixed)



############kalman_TRUE (with true diverse H)############################
print("\n--- Running Kalman Filter with TRUE diverse H ---")
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target, F=F_test_mat, H=H_test_mat)

############rts_TRUE (with true diverse H)##############################
print("\n--- Running Classical RTS Smoother with TRUE diverse H ---")
[MSE_RTS_linear_arr, MSE_RTS_linear_avg_1, MSE_RTS_dB_avg, RTS_out, P_smooth, V_test] = S_Test(sys_model, test_input, test_target,
                                                                                               F=F_test_mat, H=H_test_mat)
#########################################################################################################
# RTS_out has shape [N_T, m, T] and is our "x_est"
# P_smooth has shape [N_T, m, m, T] and is the covariance we want to evaluate
# test_target has shape [N_T, m, T] and is our "x_true"



# Initialize WRONG H matrices for EMKF to refine
# Create DIVERSE wrong H by rotating the true diverse H matrices
# This maintains the diversity structure but makes them "wrong"
H_test_mat_wrong = [H.clone() for H in H_test_mat]
H_test_mat_wrong = rotate_F(H_test_mat_wrong, i=0, j=1, theta=1., mult=1, many=True, randomit=False)


#################Test with WRONG H###################
print('\n--- Testing with WRONG H ---')
print('Wrong H initial (first group):\n', H_test_mat_wrong[0])
print('True H (first group):\n', H_test_mat[0])
print('H difference norm:', torch.norm(H_test_mat_wrong[0] - H_test_mat[0]).item())
S_Test(sys_model, test_input, test_target, F=F_test_mat, H=H_test_mat_wrong)

# Create second system model for EMKF testing
sys_model_2 = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model_2.InitSequence(m1_0, m2_0)
sys_model_2.F_test = F_test_mat
sys_model_2.H_test = H_test_mat_wrong  # Wrong H for estimation
sys_model_2.H_test_TRUE = H_test_mat   # True H for reference

########EMKF for H estimation##########
print('\n--- Start EMKF for H Estimation ---')
print('Fixed F (known):\n', F_fixed)
print('Initial wrong H (first group):\n', H_test_mat_wrong[0])
print('True diverse H (first group):\n', H_test_mat[0])

H_matrices, likelihoods, iterations_list, _, _, _ = EMKF_H_analitic(sys_model_2,sys_model_2.F_test,H_test_mat_wrong,Q, R,test_input,m1_0, m2_0,test_target,max_it=3,generate_h=True, init_x_list=None,init_P_list=None)

print('\n--- EMKF Results ---')
print('Number of sequences processed:', len(H_matrices))

# # Extract final H estimates (last iteration for each sequence)
# H_final_estimates = [H_seq[-1] for H_seq in H_matrices]
#
# print('H evolution for sequence 0:')
# for iter_idx, H_est in enumerate(H_matrices[0]):
#     print(f'  Iteration {iter_idx}: H =\n{H_est}')
#     h_error = torch.norm(H_est - H_test_mat[0]).item()
#     print(f'    Error from true H: {h_error:.6f}')
#
# print('\nFinal estimated H (seq 0, last iteration):\n', H_final_estimates[0])
# print('True H (seq 0):\n', H_test_mat[0])
# print('Initial wrong H (seq 0):\n', H_test_mat_wrong[0])
# print('Final H estimation error:', torch.norm(H_final_estimates[0] - H_test_mat[0]).item())
# print('Initial H error:', torch.norm(H_test_mat_wrong[0] - H_test_mat[0]).item())
#
# # Optional: Evaluate with estimated H
# print('\n--- Testing with Estimated H (final iteration) ---')
# S_Test(sys_model, test_input, test_target, F=F_test_mat, H=H_final_estimates)
