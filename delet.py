import torch
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0

from Simulations.utils import DataLoader, DataGen
from RTSNet.PsmoothNN import PsmoothNN
import numpy as np
from torch.distributions import Exponential


# # For NumPy
# np.random.seed(1)
#
# For PyTorch
torch.manual_seed(1)

args = config.general_settings()
args.N_T = 10   # Number of test examples (size of the test dataset used to evaluate performance).100
args.N_E = 0
args.T_test = 30 # Length of the time series for test sequences.

# True model
q2 = 0.01
r2 = 0.1
Q = q2 * Q_structure
R = r2 * R_structure
# F = torch.tensor([[0.999, 0.1],
#                             [0.0,   0.999]]) # State transition matrix
F = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])
F = torch.tensor([[0.83, 0.2],
              [0.2, 0.83]])
H = torch.tensor([[1., 1.], [0.25, 1.]])


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
DataGen(args, sys_model, dataFolderName + dataFileName,dataFolderName + dataFileName_F,delta=1, randomInit_train=False,
        randomInit_cv=False,randomInit_test=False,randomLength=False,Test = False)
print("Data Load")
#
[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
[F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F)
print("testset size:",test_target.size())
F_test_mat_0 = F_test_mat_list[0]
print('for ori', F_test_mat_0)
###############################################################################################



# F_test_mat_0 = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]]) # State transition matrix
# F_test_mat_1 = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]]) # State transition matrix
F_test_mat_0 = torch.tensor([[0.83, 0.2],
              [0.2, 0.83]])
F_test_mat_1 = torch.tensor([[0.83, 0.2],
              [0.2, 0.83]])
F_test_mat =[]
F_test_mat.append(F_test_mat_0)
F_test_mat.append(F_test_mat_1)


############kalman_TRUE############################
KFTest(args, sys_model, test_input, test_target,F =F_test_mat)
# ############rts_TRUE##############################
# 1. Run S_Test to get the outputs from the classical RTS smoother
print("\n--- Running Classical RTS Smoother TRUE---")
S_Test(sys_model, test_input, test_target, F=F_test_mat)
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
# F_initial_1 = torch.tensor([[0.83, 0.2],
#                             [0.2,   0.83]])
# F_initial_2  = torch.tensor([[0.83, 0.2],
#                             [0.2,   0.83]])

F_initial_1 = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])
F_initial_2  = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]])
# F_initial_1 = torch.tensor([[1., 1.],[1., 0.]])
# F_initial_2  = torch.tensor([[1., 1.],[1., 0.]])



# F_initial_1 = rotate_F(F_test_mat_0)
# F_initial_2 = rotate_F(F_test_mat_1)
F_test_mat =[]
F_test_mat.append(F_initial_1)
F_test_mat.append(F_initial_2)


#################false test###################
print('regular error F', F_test_mat)
print('wrong FFFFFFFFFFFF')
S_Test(sys_model, test_input[0].unsqueeze(0),test_target[0].unsqueeze(0), F=F_test_mat)
S_Test(sys_model, test_input[1].unsqueeze(0),test_target[1].unsqueeze(0), F=F_test_mat)

########EMKF##########
#####TRUE######
print('start EMKF')



