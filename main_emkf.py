import torch
import matplotlib.pyplot as plt
from emkf.func import  compute_A1, compute_A2, compute_A3, Ell
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from emkf.main_emkf_func import EMKF_F, EMKF_F_solo
from Simulations.utils import DataLoader, DataGen



args = config.general_settings()
args.N_T = 2   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T_test = 30 # Length of the time series for test sequences.


r2 = torch.tensor([1e-3])
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))
#################################################START DATA#######################################
# True model
Q = q2 * Q_structure
R = r2 * R_structure
F = torch.tensor([[1, 0.1],[1, 1]]) # State transition matrix
H = torch.tensor([[1., 1.],
                  [0.25, 1.]])

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
DataGen(args, sys_model, dataFolderName + dataFileName,dataFolderName + dataFileName_F,delta=1, randomInit_train=False,randomInit_cv=False,randomInit_test=False,randomLength=False)
print("Data Load")

[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
[F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName + dataFileName_F)
print("testset size:",test_target.size())

###############################################################################################


F_initial_1 = torch.tensor([[1., 1.], [0.1, 1.]])
F_initial_2  = torch.tensor([[1., 1.], [0.1, 1.]])

############kalman_TRUE############################
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, K_list] = KFTest(args, sys_model, test_input, test_target,F =F_test_mat)
############rts_TRUE###############################
[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out,P_smooth,V] = S_Test(sys_model, test_input, test_target,F= F_test_mat,K_T_list = K_list)




########EMKF##########
#####TRUE######

F_matrices, likelihoods, iterations_list = EMKF_F(F_test_mat, H, Q, R, test_input, m1_0, m2_0, test_target, P_smooth, V, K_list, max_it=100, tol_likelihood=0.01, tol_params=0.005)
print('True F matrices 1', F_test_mat[0])
print('end of EMKF',F_matrices )


######mse with emkf F #######

[MSE_KF_linear_arr1, MSE_KF_linear_avg1, MSE_KF_dB_avg1, K_list1] = KFTest(args, sys_model, test_input, test_target,F= F_matrices)
[MSE_RTS_linear_arr2, MSE_RTS_linear_avg2, MSE_RTS_dB_avg2, RTS_out2,P_smooth_2,V_2] = S_Test(sys_model, test_input, test_target,F= F_matrices,K_T_list= K_list1)



print('mse of smoother',MSE_RTS_dB_avg)
print('mse of emkf',MSE_RTS_dB_avg2)


#######FALSE_1#######



#######FALSE_2#######