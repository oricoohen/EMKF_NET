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
from emkf.second_main_emkf_paper_func import EMKF_FHB_decrypt_style_batch


device = torch.device("cuda")
# # For NumPy
# np.random.seed(1)
#
# For PyTorch
torch.manual_seed(1)

args = config.general_settings()
args.N_T = 100  # Number of test examples (size of the test dataset used to evaluate performance).100

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
r2 = 0.1
Q = q2 * Q_structure
R = r2 * R_structure
# F = torch.tensor([[0.999, 0.1],
#                             [0.0,   0.999]]) # State transition matrix
F = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]], device=device)
# F = torch.tensor([[0.83, 0.2],
#               [0.2, 0.83]])
H = torch.tensor([[1., 1.], [0.25, 1.]], device=device)
F_in =[F for _ in range(args.N_E)]
H_in = [H for _ in range(args.N_E)]
SystemModel.F_gen = False
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)


###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName = '2x2_rq3030_T100.pt'
dataFileName_F = '2x2_F'
dataFileName_H = '2x2_H'
print("Start Data Gen")
# Generate data with FIXED F and H (same for all samples)
# F_gen=False and H_gen=False means use the F and H from sys_model
DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_F,
        fileName_H=dataFolderName + dataFileName_H, delta=1, randomInit_train=False,
        randomInit_cv=False, randomInit_test=False, randomLength=False, Test=True,
        F_gen=F_in, H_gen=H_in)
print("Data Load")
#
#[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
[train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName + dataFileName, weights_only=False)
[F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F, weights_only=False)
[H_train_mat, H_val_mat, H_test_mat_list] = torch.load(dataFolderName + dataFileName_H, weights_only=False)

# Get the device from loaded data and move all tensors to the same device
data_device = test_target.device
m1_0_device = m1_0.to(data_device)
m2_0_device = m2_0.to(data_device)
Q = Q.to(data_device)
R = R.to(data_device)

print("testset size:",test_target.size())
print("Data generated with F:", F_test_mat_list[0][0])
print("Data generated with H:", H_test_mat_list[0][0] if H_test_mat_list[0].dim() == 3 else H_test_mat_list[0])
F_test_mat_0 = F_test_mat_list[0]
H_true = H_test_mat_list[0] if H_test_mat_list[0].dim() == 2 else H_test_mat_list[0][0]
print('True F used for data generation:', F_test_mat_0[0] if F_test_mat_0.dim() == 3 else F_test_mat_0)
print('True H used for data generation:', H_true)



# Use the same H that was used for data generation
H_in = [H_true for _ in range(args.N_T)]

# F_test_mat_0 = torch.tensor([[0.83, 0.2],
#               [0.2, 0.83]])
# F_test_mat_1 = torch.tensor([[0.83, 0.2], [0.2, 0.83]])
# Use a DIFFERENT F for testing to evaluate robustness
F_test_mat_0_wrong = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]], device=data_device) # Different F for testing
sys_model.InitSequence(m1_0_device, m2_0_device)
# F_test_mat_0= torch.tensor([[0.83, 0.2],
#               [0.2, 0.83]])
# Create list with the TRUE F for testing ground truth
F_test_mat_true = []


############kalman_TRUE############################
print("\n--- Running Kalman Filter with TRUE F (used for data generation) ---")
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target, F=F_test_mat_list)
# ############rts_TRUE##############################
# 1. Run S_Test to get the outputs from the classical RTS smoother
print("\n--- Running Classical RTS Smoother with TRUE F and TRUE H (used for data generation) ---")
[MSE_RTS_linear_arr, MSE_RTS_linear_avg_1, MSE_RTS_dB_avg, RTS_out, P_smooth, V_test] = S_Test(sys_model, test_input, test_target,
                                                F=F_test_mat_list, H=H_in)
#########################################################################################################
# RTS_out has shape [N_T, n, T] and is our "x_est"
# P_smooth has shape [N_T, n, n, T] and is the covariance we want to evaluate
# test_target has shape [N_T, n, T] and is our "x_true"



# F_initial_1 = torch.tensor([[0.85, 0.2],
#                             [0.2,   0.85]])
# F_initial_2 = torch.tensor([[0.85, 0.2],
#                             [0.2,   0.85]])
# F_initial_1 = rotate_F([F_test_mat_0])
# F_initial_2 = rotate_F([F_test_mat_1])


# F_initial_1 = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])
# F_initial_2  = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])
# Use a DIFFERENT F for testing (F_initial_1 != True F used for data generation)
F_initial_1 = torch.tensor([[0.83, 0.2],
                            [0.2,   0.83]], device=data_device)
F_initial_2  = torch.tensor([[0.83, 0.2],
                            [0.2,   0.83]], device=data_device)
# F_initial_1 = torch.tensor([[1., 1.],[1., 0.]])
# F_initial_2  = torch.tensor([[1., 1.],[1., 0.]])



# F_initial_1 = rotate_F(F_test_mat_0)
# F_initial_2 = rotate_F(F_test_mat_1)
# F_initial_1 = rotate_F(F, i=0, j=1, theta=0.087, many=True, randomit=False)
# F_initial_2 = F_test_mat_1
F_test_mat =[]
for i in range(args.N_T):
    F_test_mat.append(F_initial_1)
# F_test_mat.append(F_initial_1)
# F_test_mat.append(F_initial_2)
sys_model.F_test = F_test_mat

#################false test###################
print('\n--- Running RTS Smoother with WRONG F (different from true F used for data generation) ---')
print('Wrong F:', F_test_mat[0])
print('True F:', F_test_mat_0[0] if F_test_mat_0.dim() == 3 else F_test_mat_0)
# S_Test(sys_model, test_input[0].unsqueeze(0), test_target[0].unsqueeze(0), F=F_test_mat)
S_Test(sys_model, test_input, test_target, F=F_test_mat, H=H_in)
########EMKF##########
#####TRUE######
print('\n--- Running EMKF with WRONG initial F (should converge to true F) ---')
F_matrices, likelihoods, iterations_list,_,_,_ = EMKF_F_analitic(sys_model,F_test_mat, H_in, Q, R, test_input, m1_0_device, m2_0_device,
                                                                 test_target, max_it=4, tol_likelihood=0.01, tol_params=0.025)


I_m = torch.eye(2, device=data_device)
factors_init = {
    "T10": I_m.clone(),
    "T11": I_m.clone(),
    "T12": F_initial_1.clone(),

    # H factors (we will NOT update H, but still need valid shapes)
    # Make D0@D1@D2 = H
    "D0": torch.eye(H_true.shape[0], device=data_device, dtype=H_true.dtype),   # [n,n] = [2,2]
    "D1": torch.eye(2, device=data_device, dtype=H_true.dtype),            # [m,m] = [2,2]
    "D2": H_true.clone(),                                             # [n,m] = [2,2]
}
print('\n--- Running second EMKF with decrypt style (factored parameterization) ---')

hist_paper, x_last_paper, p_last_paper = EMKF_FHB_decrypt_style_batch(sys_model=sys_model,Y=test_input,X_true=test_target,x_0=m1_0_device,P_0=m2_0_device,
factors_init=factors_init,U_in=None,max_it=3,n_sweeps_factor=1,init_x_list=None,init_P_list=None,update_F=True,update_H=False,update_B=False,H_fixed=H_true,F_fixed=None,B_fixed=None)


