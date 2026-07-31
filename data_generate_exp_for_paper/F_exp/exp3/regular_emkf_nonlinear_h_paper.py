import torch
from datetime import datetime

from Smoothers.EKF_test import EKFTest
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext


from Simulations.Extended_sysmdl import SystemModel,make_rotated_h_nonlinear
from Simulations.utils import DataGen
import Simulations.config as config
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
        F, make_f, h_nonlinear, Q_structure, R_structure

from emkf.main_emkf_func import E_EMKF_F_analitic

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 100
args.N_CV = 100
args.N_T = 100
args.T = 30
args.T_test = 30



offset = 0
chop = False
sequential_training = False
path_results = '../../../RTSNet/'
RTSNetPass1_path = '../../../RTSNet/checkpoints/LorenzAttracotor/DT/HNL/rq3030_T20.pt'
# r2 = torch.tensor([1e-3]) # [10, 1, 0.1, 0.01, 1e-3]
# vdB = 0 # ratio v=q2/r2
# v = 10**(vdB/10)
# q2 = torch.mul(v,r2)

q2 = 0.01
r2 = 0.1

Q = q2 * Q_structure
R = r2 * R_structure



dataFolderName = f'Simulations/Linear_canonical/paper/exp1_1/regular/'
dataFileName = f'snr_0{args.T_test}_dataset_non_linear.pt'
dataFileName_F = f'snr_0_F_dataset_non_linear.pt'

#########################################
###  Generate and load data DT case   ###
#########################################

# h_nonlinear_rot = make_rotated_h_nonlinear(h_nonlinear, phi_deg=30.0)

sys_model = SystemModel(make_f(F), Q, h_nonlinear, R, args.T, args.T_test, m, n)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0
## Model wrong F

print('old f',F)
# ---- pick a small mismatch so results are still good ----
# rot_deg = 45.0
# theta = torch.deg2rad(torch.tensor(rot_deg, dtype=F.dtype))
# c, s = torch.cos(theta), torch.sin(theta)
# R_dyn = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # [2,2]
# F_wrong = F @ R_dyn   # slight rotation of the true dynamics
F_wrong = torch.tensor([[0.83, 0.2],
                            [0.2,   0.83]])
print('new f',F_wrong)
sys_model_2 = SystemModel(make_f(F_wrong), Q, h_nonlinear, R, args.T,args.T_test, m, n)
sys_model_2.InitSequence(m1x_0, m2x_0)

# print("Start Data Gen")
DataGen(args, sys_model, dataFolderName + dataFileName, dataFolderName + dataFileName_F,
        delta=1, randomInit_train=False, randomInit_cv=False, randomInit_test=False,
        randomLength=False, Test=True)
print("Data Load")

[train_input,train_target, cv_input, cv_target, test_input, test_target] =  torch.load(dataFolderName + dataFileName)
[F_train_mat, F_val_mat, F_test_mat_list] = torch.load(dataFolderName + dataFileName_F)

# print("trainset size:",train_target.size())
# print("cvset size:",cv_target.size())
print("testset size:",test_target.size())

######################################
### Evaluate Filters and Smoothers ###
######################################
### Evaluate EKF full

F_wrong_list = [F_wrong.clone() for _ in range(len(test_input))]

F_matrices,  mse_avg_T, x_last, p_last = E_EMKF_F_analitic(sys_model, F_wrong_list,h_nonlinear, Q, R, test_input, m1x_0, m2x_0,
                                                                                      test_target, max_it=3,
                                                                                      generate_f=False)

print("Evaluate EKF full")
EKFTest(sys_model, test_input, test_target)

### Evaluate RTS full
print("Evaluate RTS full")
S_Test_ext(sys_model, test_input, test_target,F_test_mat_list)

F_wrong_list = [F_wrong.clone() for _ in range(args.N_T//10)]
print("Evaluate EKF wrong_F")
EKFTest(sys_model_2, test_input, test_target)

### Evaluate RTS full
print("Evaluate RTS wrong_F")
S_Test_ext(sys_model_2, test_input, test_target,F_wrong_list)