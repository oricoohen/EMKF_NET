import torch
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from emkf.main_emkf_func import EMKF_F_analitic, EMKF_F_solo
from Simulations.utils import DataLoader, DataGen
from RTSNet.PsmoothNN import PsmoothNN
import numpy as np




torch.manual_seed(1)
args = config.general_settings()
args.N_T = 2   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T_test = 100 # Length of the time series for test sequences.
state = 1
torch.manual_seed(state)          # add this one line


# True model

q2 = 0.01
r2 = 0.1

Q = q2 * Q_structure
R = r2 * R_structure
# F = torch.tensor([[0.999, 0.1],
#                             [0.0,   0.999]]) # State transition matrix
F = torch.tensor([[0.83, 0.2],
              [0.2, 0.83]])
H = torch.tensor([[1., 1.], [0.25, 1.]])

# Q= torch.tensor([
#     [ 8.025e-4, -5.330e-4, -4.853e-4],
#     [-5.330e-4,  1.912e-3,  1.443e-3],
#     [-4.853e-4,  1.443e-3,  1.929e-3]])
#
# H = torch.tensor([
#     [-0.945,  0.507,  0.076],
#     [-0.341,  0.577, -0.394],])
# R = torch.tensor([
#     [1.913e-3, 6.096e-4],
#     [6.096e-4, 3.264e-4],])
# F = torch.tensor([
#     [0.901, 0.045, -0.036],
#     [0.045, 0.881, -0.008],
#     [0.033, -0.009, 0.905], ])
# H = torch.tensor([[1., 0.], [0., 1.]])

SystemModel.F_gen = False
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName = '2x2_rq3030_T100.pt'
dataFileName_F = '2x2_F'
print("Start Data Gen")
# DataGen(args, sys_model, dataFolderName + dataFileName,dataFolderName + dataFileName_F,delta=1, randomInit_train=False,randomInit_cv=False,randomInit_test=False,randomLength=False)
# print("Data Load")
#
# [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
# [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F)
# print("testset size:",test_target.size())
# F_test_mat_0 = F_test_mat_list[0]
# print('for ori', F_test_mat_0)
###############################################################################################


# F_test_mat_0 =F + torch.empty_like(F).uniform_(-0.1, 0.1)
# F_test_mat_1 =F + torch.empty_like(F).uniform_(-0.1, 0.1)
F_test_mat_0 = torch.tensor([[0.83, 0.2],
                            [0.2,   0.83]]) # State transition matrix
F_test_mat_1 = torch.tensor([[0.83, 0.2],
                            [0.2,   0.83]]) # State transition matrix
F_test_mat =[]
F_test_mat.append(F_test_mat_0)
F_test_mat.append(F_test_mat_1)

#####create data######

T = 100
n = 2
p = 2

state = 1
gen = np.random.default_rng(seed=state)


F_sim2 = np.array([[0.83, 0.2],
                   [0.2, 0.83]])          #   [[1,1],[0.1,1]]


Q_sim2 = 0.01 * np.eye(n)               #   process-noise covariance
H_sim2 = np.array([[1., 1.], [0.25, 1.]])                  #   full observation
R_sim2 = 0.10 * np.eye(p)               #   measurement-noise covariance

xi_sim2 = np.array([0.5, 0.5])          #   x₀ mean
L_sim2  = np.eye(n)

x_sim2 = np.array([gen.multivariate_normal(xi_sim2, L_sim2)])
z_sim2 = np.empty((1, p))

for t in range(T):
    x_sim2 = np.append(x_sim2, [F_sim2 @ x_sim2[t] + gen.multivariate_normal(np.zeros(n), Q_sim2)], axis=0)
    z_sim2 = np.append(z_sim2, [H_sim2 @ x_sim2[t + 1] + gen.multivariate_normal(np.zeros(p), R_sim2)], axis=0)

z_sim2 = np.delete(z_sim2, 0, axis=0)

# make PyTorch default to 32-bit floats
torch.set_default_dtype(torch.float32)

x_true_t = torch.from_numpy(x_sim2[1:].T).contiguous().float()
z_meas_t = torch.from_numpy(z_sim2.T).contiguous().float()

# Pack to match S_Test/KFTest interface: (N_T, dim, T)
test_target = x_true_t.unsqueeze(0)    # (1, n, T)
test_input  = z_meas_t.unsqueeze(0)    # (1, p, T)






############kalman_TRUE############################
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target,F =F_test_mat)
# ############rts_TRUE##############################
# 1. Run S_Test to get the outputs from the classical RTS smoother
print("\n--- Running Classical RTS Smoother ---")
[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out, P_smooth, V_test] = S_Test(sys_model, test_input, test_target, F=F_test_mat)
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
F_initial_1 = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])
F_initial_2  = torch.tensor([[0.63, 0.0021],[0.0021, 1.0029]])
# F_initial_1 = rotate_F(F_test_mat_0)
# F_initial_2 = rotate_F(F_test_mat_1)
F_test_mat =[]
F_test_mat.append(F_initial_1)
F_test_mat.append(F_initial_2)


#################false test###################
print('regular error F', F_test_mat)

[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out, P_smooth, V_test] = S_Test(sys_model, test_input, test_target, F=F_test_mat)

########EMKF##########
#####TRUE######
print('start EMKF')
F_matrices, likelihoods, iterations_list = EMKF_F_analitic(sys_model,F_test_mat, H, Q, R, test_input, m1_0, m2_0, test_target, max_it=300, tol_likelihood=0.01, tol_params=0.025)
print('True F matrices 1', F)
print('end of EMKF first:',F_matrices[0],'second', F_matrices[1])
print(test_target.size())
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

#######FALSE_1#######



#######FALSE_2#######