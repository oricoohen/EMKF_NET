####the old one without the f
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F,det
from emkf.main_emkf_func_AI_no_p import EMKF_F_N

from Simulations.utils import DataLoader, DataGen

import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN


from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

import shutil
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
path_results_full = 'RTSNet/full_info/exp_cov'
path_results_2 = 'RTSNet/wrong_F/exp_cov'

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_E = 500  # Number of training examples (size of the training dataset).50
args.N_CV = 100  # Number of cross-validation examples (size of the CV dataset used to tune hyperparameters).30
args.N_T = 100   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

### training parameters
args.n_steps = 175  # Number of training steps or iterations for optimization.
args.n_batch = 30    # Batch size: the number of sequences processed at each training step.10
args.lr = 1e-4       # Learning rate: controls how quickly the model updates during training.
args.wd = 1e-3       # Weight decay (L2 regularization): penalizes large weights to reduce overfitting.


###########################################################
# True model
q2 = 0.01
r2 =0.1
Q = q2 * Q_structure
R = r2 * R_structure
F = torch.tensor([[0.7, 0.3],[0.3, 0.7]]) # State transition matrix
H = torch.tensor([[1., 1.],
                  [0.25, 1.]])
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
SystemModel.F_gen = True
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)


###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName = '2x2_rq3030_T100.pt'
dataFileName_F = '2x2_F'
print("Start Data Gen")
DataGen(args, sys_model, dataFolderName + dataFileName,dataFolderName + dataFileName_F,delta=1, randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv,randomInit_test=InitIsRandom_test,randomLength=LengthIsRandom)
print("Data Load")



[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
[F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName + dataFileName_F)
print("trainset size:",train_target.size())#(seq,m,T)
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())



sys_model.F_train = F_train_mat
sys_model.F_valid = F_val_mat
sys_model.F_test = F_test_mat
sys_model.F_train_TRUE = F_train_mat
sys_model.F_valid_TRUE = F_val_mat
sys_model.F_test_TRUE = F_test_mat
# print(F_train_mat,'111111111111111')
# print(F_val_mat,'22222222222222222222')
# print(F_test_mat,'333333333333333')
########################################
### Evaluate Observation Noise Floor ###
########################################




loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T)# MSE [Linear]
for j in range(0, args.N_T):
   MSE_obs_linear_arr[j] = loss_obs(test_input[j], test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor - STD:", obs_std_dB, "[dB]")



# ##############################
# ### Evaluate Kalman Filter ###
# ##############################
print("Evaluate Kalman Filter True")
KFTest(args, sys_model, test_input, test_target,F = F_test_mat)
# #############################
### Evaluate RTS Smoother ###
############################

print("Evaluate RTS Smoother True")
_,_,_,RTS_TRUE,P_TRUE= S_Test(sys_model, test_input, test_target,F = F_test_mat)

######BAD F############################

##################second training
sys_model_2 = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model_2.InitSequence(m1_0, m2_0)
######create new data


dataFolderName_2 = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName_2 = '2x2_rq3030_T100_2.pt'
dataFileName_F_2 = '2x2_F_2'
print("Start Data Gen")
print("Data Load")



sys_model_2.F_train = F_train_mat
sys_model_2.F_valid = F_val_mat
sys_model_2.F_test = F_test_mat
sys_model_2.F_train_TRUE = F_train_mat
sys_model_2.F_valid_TRUE = F_val_mat
sys_model_2.F_test_TRUE = F_test_mat

#########change to wrong f option A
# sys_model_2.F_train = det(F_train_mat)
# sys_model_2.F_valid = det(F_val_mat)
# sys_model_2.F_test= det(F_test_mat)
# sys_model_2.args = args
#########change to wrong f option B

sys_model_2.F_train = rotate_F(F_train_mat)
sys_model_2.F_valid = rotate_F(F_val_mat)
sys_model_2.F_test= rotate_F(F_test_mat)

print('just to make suree', sys_model_2.F_test)


###################check the regu;ar rts
print('regular kalman and rts with wrong F')
# KFTest(args, sys_model_2, test_input, test_target,F = sys_model_2.F_test)
#
# _,_,_,RTS_out,P,V = S_Test(sys_model_2, test_input, test_target,F = sys_model_2.F_test)

#######################
### RTSNet Pipeline ###
#######################
########emkf#################################

# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model,args)
RTSNet_Pipeline.setTrainingParams(args)

path_results_full_rts = path_results_full+'best-model_no.pt'
path_results_2_rts = path_results_2+'best-rts_no.pt'
path_results_full_PSMOOTH = path_results_full+'PSMOOTH-best-model.pt'
path_results_full_rts2 = path_results_full+'best-model_no2.pt'
path_results_full_PSMOOTH2 = path_results_full+'PSMOOTH-best-model2.pt'

eps = 0.00001
#####TRAIN GOOD F#####
print('rtssnet with trueeeeeeee F')
RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results_full_rts)
X_1,P_1,V_1 = RTSNet_Pipeline.NNTest_HybridP(sys_model, test_input, test_target, load_model_path=path_results_full_rts)
######TRAIN GOOD F########
# RTSNet_Pipeline.P_smooth_Train(sys_model,cv_input, cv_target,
#                      train_input, train_target, path_results =path_results_full_PSMOOTH, path_rtsnet = path_results_full_rts,load_psmooth_path = None, generate_f=True)
# RTSNet_Pipeline.Train_Joint(sys_model, cv_input, cv_target, train_input, train_target, path_results_rtsnet=path_results_full_rts2 ,path_results_psmooth=path_results_full_PSMOOTH2,
#                             load_rtsnet = path_results_full_rts,load_psmooth = path_results_full_PSMOOTH, generate_f=True)
# _,_,_,_,_, P_1,_,_,_,_= RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, load_model_path=path_results_full_rts,load_p_smoothe_model_path= path_results_full_PSMOOTH)#



rel_all = 0
for Pt, Pe in zip(P_TRUE, P_1):          # Pt, Pe: [m, m, T]
    for i in range(len(Pt)):
        diff = Pe[i] - Pt[i]                                    # [m, m, T]
        num = torch.linalg.norm(diff, ord='fro',dim=(0, 1))         # Frobenius ‖ΔP_t‖, shape [T]
        den = torch.linalg.norm(Pt[i],ord='fro' ,dim=(0, 1)) + eps    # Frobenius ‖P_t‖,   shape [T]
        err =(num / den)
        rel_all += err
mean_rel = rel_all/len(P_TRUE)
mean_rel = mean_rel/args.T_test
rel_db = 20 * torch.log10(mean_rel)
rel_db = rel_db.item()

print(f"Mean relative error = {rel_db:.2f} dB")











### Test Neural Network
# X_1,P_1,V_1 = RTSNet_Pipeline.NNTest_HybridP(sys_model, test_input, test_target, load_model_path=path_results_full_rts)


d




