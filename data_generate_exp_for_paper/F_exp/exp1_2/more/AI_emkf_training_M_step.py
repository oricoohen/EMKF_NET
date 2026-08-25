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
print(torch.cuda.is_available())  # should be True
print(torch.cuda.get_device_name(0))
device = torch.device("cuda")
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results_True = '../../../../RTSNet/AI_M_step/exp_1/r_1/True_F/'  ######################################################################################################################################################################
gauss = False
path_results_False = '../../../../RTSNet/AI_M_step/exp_1/r_1/False_F/'  ######################################################################################################################################################################

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_E = 400  # Number of training examples (size of the training dataset).50
args.N_CV = 100  # Number of cross-validation examples (size of the CV dataset used to tune hyperparameters).30
args.N_T = 100   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

### training parameters
args.n_steps = 175  # Number of training steps or iterations for optimization.
args.n_batch = 10    # Batch size: the number of sequences processed at each training step.10
args.lr = 1e-4       # Learning rate: controls how quickly the model updates during training.
args.wd = 1e-3       # Weight decay (L2 regularization): penalizes large weights to reduce overfitting.

max_iter = 3


# True model
q2 = 0.01
r2 =1.
v_db = 0
# snr_db =10.0######################################################################################################################################################################
# r2 = 10.0**(-snr_db/10.0)
# q2 = r2/(10.0**v_db/10.0)

Q = q2 * Q_structure.to(device)
R = r2 * R_structure.to(device)
F = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device) # State transition matrix
H = torch.tensor([[1., 1.], [0.25, 1.]], device=device)
H_in = [H.clone() for _ in range(args.N_T)]
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
SystemModel.F_gen = True
m1_0 = m1_0.to(device)
m2_0 = m2_0.to(device)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)


###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/paper/exp1_1/full' + '/'
dataFileName = '2x2_1.pt'
dataFileName_F = '2x2_F_reg+5_deg'
print("Start Data Gen")
DataGen(args, sys_model, dataFolderName + dataFileName,dataFolderName + dataFileName_F,delta=1, randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv
        ,randomInit_test=InitIsRandom_test,randomLength=LengthIsRandom,Test=False,F_gen=True,H_gen=H_in)
print("Data Load")



[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
[F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName + dataFileName_F, map_location=device)
print("trainset size:",train_target.size())#(seq,m,T)
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())


###############################################################################################
# ##estimate Q and R from data
# if gauss:
#     Q_hat, R_hat = estimate_QR(train_input, train_target)
#     Q = Q_hat
#     R = R_hat
#     sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)

#################################################################################################

############################
# --- GPU moves for datasets (dtype aligned with F) ---
ddtype = F.dtype
train_input = train_input.to(device=device, dtype=ddtype)
train_target = train_target.to(device=device, dtype=ddtype)
cv_input = cv_input.to(device=device, dtype=ddtype)
cv_target = cv_target.to(device=device, dtype=ddtype)
test_input = test_input.to(device=device, dtype=ddtype)
test_target = test_target.to(device=device, dtype=ddtype)
############################
sys_model.F_train = F_train_mat
sys_model.F_valid = F_val_mat
sys_model.F_test = F_test_mat
sys_model.F_train_TRUE = F_train_mat
sys_model.F_valid_TRUE = F_val_mat
sys_model.F_test_TRUE = F_test_mat

sys_model.H_test = H_in
# print(F_train_mat,'111111111111111')
# print(F_val_mat,'22222222222222222222')
print(F_test_mat,'333333333333333')
########################################
### Evaluate Observation Noise Floor ###
########################################




loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T, device=device)# MSE [Linear]
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
KFTest(args, sys_model, test_input, test_target,F = F_test_mat,H=H_in)
# #############################
### Evaluate RTS Smoother ###
############################

print("Evaluate RTS Smoother True")
S_Test(sys_model, test_input, test_target,F = F_test_mat,H=H_in)

######BAD F############################

##################second training
sys_model_2 = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model_2.InitSequence(m1_0, m2_0)
######create new data



sys_model_2.F_train_TRUE = F_train_mat
sys_model_2.F_valid_TRUE = F_val_mat
sys_model_2.F_test_TRUE = F_test_mat



#########change to wrong f option B

# sys_model_2.F_train = rotate_F(F_train_mat, i=0, j=1, theta=0.5,mult=1, many=True, randomit=True)
# sys_model_2.F_valid = rotate_F(F_val_mat, i=0, j=1, theta=0.5,mult=1, many=True, randomit=True)
# sys_model_2.F_test= rotate_F(F_test_mat, i=0, j=1, theta=0.5,mult=1, many=True, randomit=True)
sys_model_2.F_train = F_train_mat.copy()
sys_model_2.F_valid = F_val_mat.copy()
sys_model_2.F_test = F_test_mat.copy()
# for i in range(len(F_train_mat)):
#     sys_model_2.F_train[i] =torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device, dtype=ddtype)
# for i in range(len(F_val_mat)):
#     sys_model_2.F_valid[i] =torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device, dtype=ddtype)
# for i in range(len(F_test_mat)):
    # sys_model_2.F_test[i] = torch.tensor([[1.2237, -0.0927],[1.8518, 0.0819]], device=device, dtype=ddtype)
    # sys_model_2.F_test[i] = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device, dtype=ddtype)

# random WRONG F per sequence using your helper
sys_model_2.F_train = rotate_F(F_train_mat, i=0, j=1, theta=1, mult=1, many=True, randomit=True)
sys_model_2.F_valid = rotate_F(F_val_mat,  i=0, j=1, theta=1, mult=1, many=True, randomit=True)
sys_model_2.F_test  = rotate_F(F_test_mat, i=0, j=1, theta=1, mult=1, many=True, randomit=True)
sys_model_2.H_test = H_in

sys_model_2.args = args
print("F WRONGGGGGG:",sys_model_2.F_test)



###################check the regu;ar rts
print('regular kalman and rts with wrong F')
KFTest(args, sys_model_2, test_input, test_target,F = sys_model_2.F_test,H=H_in)

S_Test(sys_model_2, test_input, test_target,F = sys_model_2.F_test,H=H_in)

#######################
### RTSNet Pipeline ###
#######################
########emkf#################################

rtsnet_models= []


# Q2 = q2 *5* Q_structure.to(device)
# R2= r2 *100* R_structure.to(device)
# sys_model.Q = Q2
# sys_model.R = R2
# sys_model_2.Q = Q2
# sys_model_2.R = R2
# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model,args)
RTSNet_Pipeline.setTrainingParams(args)
RTSNet_model.to(device)

path_results_True_rts = path_results_True+'best-rts_true.pt'
path_results_wrong_rts = path_results_False+'best-rts_false.pt'
#####TRAIN GOOD F#####
print('rtssnet and psmooth with trueeeeeeee F')
# RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results_True_rts)

### Test Neural Network
RTSNet_Pipeline.NNTest_no_p(sys_model, test_input, test_target,load_model_path=path_results_True_rts, generate_f=True,init_x_list=None, init_P_list=None,non_linear_h=False)


#RTSNet_Pipeline.setTrainingParams(args_big)
print('rtssnet and psmooth with WRONGGGGGGG F')
#######TRAIN BAD F########
# RTSNet_Pipeline.NNTrain(sys_model_2, cv_input, cv_target, train_input, train_target, path_results = path_results_wrong_rts,load_model_path= path_results_True_rts,generate_f=True)

## Test Neural Network
RTSNet_Pipeline.NNTest_no_p(sys_model_2, test_input, test_target,load_model_path=path_results_wrong_rts, generate_f=True,init_x_list=None, init_P_list=None,non_linear_h=False)
5
# The folder where the new copies will be saved.
destination_folder = 'RTSNet/AI_M_step/exp_1/r_1/EMKF/False/'######################################################################################################################################################################

# --- Step 2: Loop 5 times and copy the file ---
# Create the new filename, e.g., "expert_0.pt", "expert_1.pt", etc.
# Build the full destination path
# destination_path_RTS = destination_folder + file_rtsnet

destination_path_M= destination_folder + f"M_rand_false_trained_12_20_f_rtsnet_new.pt"

# destination_path_M = [destination_folder + "M_iter0.pt",destination_folder + "M_iter1.pt",destination_folder + "M_iter2.pt"]
# load_m= destination_folder + f"M_rand_false_trained.pt"
######START THE EMKF TRAINING##########


sys_model_2.args = args
RTSNet_Pipeline.setTrainingParams(args)


# RTSNet_Pipeline.end_To_end_m_net(sys_model_2,cv_input,cv_target,train_input,train_target,destination_path_M,destination_path_RTS=path_results_wrong_rts,
#                     load_base_m_mmodel=load_m,num_em_iters=3,alpha=(0.05, 0.1, 0.85),lambda_F=0.01,generate_f=True,non_linear_h=False)

# RTSNet_Pipeline.end_to_end_test_mstep_net(sys_model_2, test_input, test_target,path_results_wrong_rts,destination_path_M,num_em_iters=3,
#                    alpha=(0.05, 0.1, 0.85), lambda_F=1e-3, generate_f=True, non_linear_h=False)

# RTSNet_Pipeline.train_mstep_net(sys_model_2,cv_input, cv_target, train_input, train_target,
#                         destination_path_M, path_results_wrong_rts, num_em_iters=3,alpha=(0.05, 0.1, 0.85), lambda_F=1e-4, generate_f=True)



# print('check FFFFFFFFFFFF', sys_model_2.F_test)
RTSNet_Pipeline.test_mstep_net(sys_model_2, test_input, test_target,path_results_wrong_rts,destination_path_M,num_em_iters=3,
                   alpha=(0.05, 0.1, 0.85), lambda_F=1e-4, generate_f=True, non_linear_h=False)


# RTSNet_Pipeline.one_train_m_step_net( sys_model_2, cv_input, cv_target, train_input, train_target,
#                         destination_path_M, destination_path_RTS = path_results_wrong_rts,  lambda_F=1e-3, generate_f=True, non_linear_h=False)

# RTSNet_Pipeline.one_test_mstep_net(sys_model_2, test_input, test_target,
#                        destination_path_RTS =path_results_wrong_rts ,destination_path_M=destination_path_M, lambda_F=1e-3, generate_f=True, init_x_list=None, init_P_list=None, non_linear_h=False)
#
# sys_model_2.F_test = rotate_F(F_test_mat)
# print('ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
# sys_model_2.F_test = F_test_mat
# for i in range(len(F_test_mat)):
#     # sys_model_2.F_train =torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device, dtype=ddtype)
#     # sys_model_2.F_valid =torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device, dtype=ddtype)
#     sys_model_2.F_test[i] = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device, dtype=ddtype)
# EMKF_F(sys_model_2,RTSNet_Pipeline,train_input, train_target, cv_input, cv_target,test_input, test_target,model_pathes,psmooth_pathes,3)

d

