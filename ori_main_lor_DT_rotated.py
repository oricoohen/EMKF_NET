import torch
from matplotlib import pyplot as plt

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
from Smoothers.EKF_test import EKFTest
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext
# from Smoothers.ParticleSmoother_test import PSTest
# from Smoothers.PF_test import PFTest

from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen, Short_Traj_Split
import Simulations.config as config

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_EKF import Pipeline_EKF
from Pipelines.Pipeline_concat_models import Pipeline_twoRTSNets

from datetime import datetime

from RTSNet.RTSNet_nn import RTSNetNN
from Plot import Plot_extended as Plot
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
    f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure,H_design
from Simulations.Linear_sysmdl import rotate_F, rotate_H
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Pipeline Start")

m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)
H_Rotate = H_Rotate.to(device)
H_Rotate_inv = H_Rotate_inv.to(device)
Q_structure = Q_structure.to(device)
R_structure = R_structure.to(device)
H_design = H_design.to(device)



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
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 30
args.T_test = 30
### training parameters
args.n_steps = 300
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

offset = 0  # offset for the data
chop = False  # whether to chop data sequences into shorter sequences
# path_results = 'RTSNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/T100_Hrot1' + '/'
switch = 'partial'  # 'full' or 'partial' or 'estH' or rotated_true or rotated_partial
destination_path_rtsnet_full = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_01/1dataset/RTSNet_full.pt'
destination_path_rtsnet_partial = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_01/1dataset/RTSNet_partial.pt'
destination_path_M_reg = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_01/1dataset/M_step_net.pt'
destination_path_rtsnet_partial_joint = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_01/1dataset/RTSNet_partial_joint.pt'
destination_path_M_joint = 'RTSNet/lorenz/lorenz_gauss/lorenz_rotated_01/1dataset/M_step_net_joint.pt'
import os

paths = [
    destination_path_rtsnet_full,
    destination_path_rtsnet_partial,
    destination_path_M_reg,
    destination_path_rtsnet_partial_joint,
    destination_path_M_joint
]

for p in paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
emkalmanet =True
emkalmanet_joint =True
num_em_iters = 2
# noise q and r
r2 = torch.tensor([0.1], device=device)  # [100, 10, 1, 0.1, 0.01]
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[0]))

traj_resultName = ['traj_lorDT_rq-1010_T10000.pt']
dataFileName = ['data_lor_v20_rq-1010_T10000.pt']
dataFName = 'data_F0.pt'
#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)  # parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0


fileName_H = "Simulations/Lorenz_Atractor/data/T100_Hrot1/2ndPass/partial/H_per_seq00.pt"
print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0],DatafolderName + dataFName[0],fileName_H =fileName_H ,H_gen = True,randomInit_train=InitIsRandom_train,
    randomInit_cv=InitIsRandom_cv,
    randomInit_test=InitIsRandom_test)
print("Data Load")
print(dataFileName[0])
[train_input_long, train_target_long, cv_input, cv_target, test_input, test_target] = torch.load(
    DatafolderName + dataFileName[0])

# Load H matrices per sequence
print("Loading H matrices per sequence...")

[H_matrices_train, H_matrices_val, H_matrices_test] = torch.load(fileName_H, map_location=device)
print(f"H matrices loaded successfully")
print(f"H_train shape: {H_matrices_train[0].shape if isinstance(H_matrices_train, list) else 'list'}, count: {len(H_matrices_train)}")
print(f"H_val shape: {H_matrices_val[0].shape if isinstance(H_matrices_val, list) else 'list'}, count: {len(H_matrices_val)}")
print(f"H_test shape: {H_matrices_test[0].shape if isinstance(H_matrices_test, list) else 'list'}, count: {len(H_matrices_test)}")

# H is DIVERSE - different H matrices
sys_model.H_train = H_matrices_train
sys_model.H_valid = H_matrices_val
sys_model.H_test = H_matrices_test
sys_model.H_train_TRUE = H_matrices_train
sys_model.H_valid_TRUE = H_matrices_val
sys_model.H_test_TRUE = H_matrices_test
print('checks for loaded H matrices:')
print(f"First H_trains matrix:\n{sys_model.H_train[0]}\n{sys_model.H_train[1]}\n{sys_model.H_train[2]}")
print(f"First H_valid matrix:\n{sys_model.H_valid[0]}\n{sys_model.H_valid[1]}\n{sys_model.H_valid[2]}")
print(f"First H_test matrix:\n{sys_model.H_test[0]}\n{sys_model.H_valid[1]}\n{sys_model.H_valid[2]}")


# Move data to device
train_input_long = train_input_long.to(device)
train_target_long = train_target_long.to(device)
cv_input = cv_input.to(device)
cv_target = cv_target.to(device)
test_input = test_input.to(device)
test_target = test_target.to(device)

if chop:
    print("chop training data")
    [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T)
    # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, args.T)
else:
    print("no chopping")
    train_target = train_target_long[:, :, 0:args.T]
    train_input = train_input_long[:, :, 0:args.T]
    # cv_target = cv_target[:,:,0:args.T]
    # cv_input = cv_input[:,:,0:args.T]

print("trainset size:", train_target.size())
print("cvset size:", cv_target.size())
print("testset size:", test_target.size())


# Model with partial info
sys_model_partial = SystemModel(f, Q, h, R, args.T, args.T_test, m, n,H_design)
sys_model_partial.InitSequence(m1x_0, m2x_0)
######SWITCH H
print("True Observation matrix H:", H_Rotate)
####################################################### computation of avarage H matrix across sequences (for partial model)
# # X_all: [N*T, m]
# X_all = train_target.permute(0, 2, 1).reshape(-1, train_target.size(1))
#
# # Y_all: [N*T, n]
# Y_all = train_input.permute(0, 2, 1).reshape(-1, train_input.size(1))
#
# # Solve X_all @ H^T ≈ Y_all
# # => H^T = argmin ||X_all H^T - Y_all||^2
# H_hat_T = torch.linalg.lstsq(X_all, Y_all).solution   # [m, n]
# H_hat = H_hat_T.T                                     # [n, m]
#
# print("Estimated Observation matrix H:", H_hat)

#######################################################
H_hat_ls = torch.tensor([
    [9.6155e-01, 3.3093e-02, 1.2045e-02],
    [-7.0018e-04, 9.6525e-01, 1.1512e-02],
    [1.5618e-02, 8.7818e-03, 9.6918e-01]
], device=device)

# Copy H matrices (will be rotated to create WRONG H)
sys_model_partial.H_train = [H.clone() for H in H_matrices_train]
sys_model_partial.H_valid = [H.clone() for H in H_matrices_val]
sys_model_partial.H_test = [H.clone() for H in H_matrices_test]

# Create WRONG H matrices by rotating them (F stays correct)
sys_model_partial.H_train = rotate_H(sys_model_partial.H_train,  theta=0.4, many=True, randomit=True)
sys_model_partial.H_valid = rotate_H(sys_model_partial.H_valid, theta=0.4,  many=True, randomit=True)
sys_model_partial.H_test  = rotate_H(sys_model_partial.H_test,   theta=0.4, many=True, randomit=True)
# H is DIVERSE - different H matrices

sys_model_partial.H_train_TRUE = H_matrices_train
sys_model_partial.H_valid_TRUE = H_matrices_val
sys_model_partial.H_test_TRUE = H_matrices_test
print('checkori',sys_model_partial.H_train[0],sys_model_partial.H_train[1],sys_model_partial.H_train[2])

# Model for 2nd pass
sys_model_pass2 = SystemModel(f, Q, h, R, args.T, args.T_test, m, n,H_design)  # parameters for GT
sys_model_pass2.InitSequence(m1x_0, m2x_0)  # x0 and P0

########################################
### Evaluate Observation Noise Floor ###
########################################
N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(N_T)  # MSE [Linear]

for j in range(0, N_T):
    reversed_target = torch.matmul(H_Rotate_inv, test_input[j])
    MSE_obs_linear_arr[j] = loss_obs(reversed_target, test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")

# ######################################
# ### Evaluate Filters and Smoothers ###
# ######################################
# ### Evaluate EKF true
# print("Evaluate EKF true")
# [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input,
#                                                                                           test_target)
# ### Evaluate EKF partial
# print("Evaluate EKF partial")
# [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial,
#  EKF_out_partial] = EKFTest(sys_model_partial, test_input, test_target)
#
# ## Evaluate RTS true
# print("Evaluate RTS true")
# [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test_ext(args, sys_model, test_input, test_target)
# ### Evaluate RTS partial
# print("Evaluate RTS partial")
# [MSE_ERTS_linear_arr_partial, MSE_ERTS_linear_avg_partial, MSE_ERTS_dB_avg_partial, ERTS_out_partial] = S_Test_ext(args,
#                                                                                                                sys_model_partial,
#                                                                                                                test_input,
#                                                                                                                test_target)
#
#
target_sample = torch.reshape(test_target[0, :, :], [1, m, args.T_test])
input_sample = torch.reshape(test_input[0, :, :], [1, n, args.T_test])

#######################
### Evaluate RTSNet ###
#######################
# if switch == 'full':

## RTSNet - 1 full ###
######################
## Build Neural Network
print("RTSNet with full model info")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
# ## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model,args)
print("Number of trainable parameters for RTSNet:",
      sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
RTSNet_Pipeline.setTrainingParams(args)
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model,
                    cv_input, cv_target, train_input, train_target, destination_path_rtsnet_full,load_model_path=None ,generate_h=True)
# Test Neural Network
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, rtsnet_out, RunTime] = RTSNet_Pipeline.NNTest(
#                     sys_model, test_input, test_target, destination_path_rtsnet_full,generate_h=True,generate_f=None)

## Save trained model
torch.save(RTSNet_Pipeline.model, destination_path_rtsnet_full)

    ####################################################################################

####################################################################################
# elif switch == 'partial':
## RTSNet with model mismatch ####################################################################################
#########################
## RTSNet - 1 partial ###
#########################
## Build Neural Network
print("RTSNet with observation model mismatch")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_partial, args)
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model_partial)
RTSNet_Pipeline.setModel(RTSNet_model,args)
RTSNet_Pipeline.setTrainingParams(args)

[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch,MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partial, cv_input, cv_target, train_input,
                                    train_target, destination_path_rtsnet_partial,destination_path_rtsnet_full,generate_h=True)
## Test Neural Network
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, rtsnet_out, RunTime] = RTSNet_Pipeline.NNTest(
#     sys_model_partial, test_input, test_target, destination_path_rtsnet_partial,generate_h=True,generate_f=None)
# Save trained model

torch.save(RTSNet_Pipeline.model, destination_path_rtsnet_partial)

if emkalmanet:
    RTSNet_Pipeline.train_H_mstep_net(sys_model_partial, cv_input, cv_target, train_input, train_target,
                      destination_path_M_reg, destination_path_rtsnet_partial, load_destination_path_M = None,num_em_iters=num_em_iters,
                      alpha=(0.3, 1, 0.85), lambda_H=1e-3, generate_h=True)
    # if emkalmanet_joint:
    RTSNet_Pipeline.train_jointH_mstep_net(sys_model_partial, cv_input, cv_target, train_input, train_target,
                          destination_path_M =destination_path_M_joint, destination_path_RTS =destination_path_rtsnet_partial_joint,
                                           load_destination_path_M= destination_path_M_reg,load_path_RTS =destination_path_rtsnet_partial,num_em_iters=num_em_iters,
                          alpha=(0.3, 1, 0.85), lambda_H=1e-3, generate_h=True)
    # RTSNet_Pipeline.test_H_mstep_net(sys_model_partial, test_input, test_target,
    #                  destination_path_rtsnet_partial_joint, destination_path_M_joint, num_em_iters=num_em_iters,
    #                 lambda_H=1e-3, generate_h=True)

###################################################################################

    print("True Observation matrix H:", H_Rotate)
    ### Least square estimation of H
    X = torch.squeeze(train_target[:, :, 0])
    Y = torch.squeeze(train_input[:, :, 0])
    for t in range(1, args.T):
        X_t = torch.squeeze(train_target[:, :, t])
        Y_t = torch.squeeze(train_input[:, :, t])
        X = torch.cat((X, X_t), 0)
        Y = torch.cat((Y, Y_t), 0)
    Y_1 = torch.unsqueeze(Y[:, 0], 1)
    Y_2 = torch.unsqueeze(Y[:, 1], 1)
    Y_3 = torch.unsqueeze(Y[:, 2], 1)
    H_row1 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_1)
    H_row2 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_2)
    H_row3 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_3)
    H_hat = torch.cat((H_row1.T, H_row2.T, H_row3.T), 0)
    print("Estimated Observation matrix H:", H_hat)
#
#
#     def h_hat(x):
#         return torch.matmul(H_hat, x)
#

#     # Estimated model
#     sys_model_esth = SystemModel(f, Q, h_hat, R, args.T, args.T_test, m, n, H_hat)
#     sys_model_esth.InitSequence(m1x_0, m2x_0)
#
#     if load_trained_pass1:
#         print("Load RTSNet pass 1")
#     else:
#         ######################
#         ## RTSNet - 1 estH ###
#         ######################
#         print("RTSNet with estimated H")
#         RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNetEstH_" + dataFileName[0])
#         RTSNet_Pipeline.setssModel(sys_model_esth)
#         RTSNet_model = RTSNetNN()
#         RTSNet_model.NNBuild(sys_model_esth, args)
#         RTSNet_Pipeline.setModel(RTSNet_model)
#         RTSNet_Pipeline.setTrainingParams(args)
#         # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(
#         #     sys_model_esth, cv_input, cv_target, train_input, train_target, path_results)
#         ## Test Neural Network
#         [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, rtsnet_out, RunTime] = RTSNet_Pipeline.NNTest(
#             sys_model_esth, test_input, test_target, path_results)
#         # Save trained model
#         # torch.save(RTSNet_Pipeline.model, RTSNetPass1_path)
#     ###################################################################################
#     if two_pass:
#         ######################
#         ## RTSNet - 2 estH ###
#         ######################
#         if load_dataset_for_pass2:
#             print("Load dataset for pass 2")
#             [train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input,
#              test_target] = torch.load(DatasetPass1_path)
#
#             print("Data Shape for RTSNet pass 2:")
#             print("testset state x size:", test_target.size())
#             print("testset observation y size:", test_input.size())
#             print("trainset state x size:", train_target_pass2.size())
#             print("trainset observation y size:", len(train_input_pass2), train_input_pass2[0].size())
#             print("cvset state x size:", cv_target_pass2.size())
#             print("cvset observation y size:", len(cv_input_pass2), cv_input_pass2[0].size())
#         else:
#             ### save result of RTSNet1 as dataset for RTSNet2
#             RTSNet_model_pass1 = RTSNetNN()
#             RTSNet_model_pass1.NNBuild(sys_model_esth, args)
#             RTSNet_Pipeline_pass1 = Pipeline(strTime, "RTSNet", "RTSNet")
#             RTSNet_Pipeline_pass1.setssModel(sys_model_esth)
#             RTSNet_Pipeline_pass1.setModel(RTSNet_model_pass1)
#             ### Optional to test it on test-set, just for checking
#             print("Test RTSNet pass 1 on test set")
#             [_, _, _, rtsnet_out_test, _] = RTSNet_Pipeline_pass1.NNTest(sys_model_esth, test_input, test_target,
#                                                                          path_results, load_model=True,
#                                                                          load_model_path=RTSNetPass1_path)
#
#             print("Test RTSNet pass 1 on training set")
#             [_, _, _, rtsnet_out_train, _] = RTSNet_Pipeline_pass1.NNTest(sys_model_esth, train_input, train_target,
#                                                                           path_results, load_model=True,
#                                                                           load_model_path=RTSNetPass1_path)
#             print("Test RTSNet pass 1 on cv set")
#             [_, _, _, rtsnet_out_cv, _] = RTSNet_Pipeline_pass1.NNTest(sys_model_esth, cv_input, cv_target,
#                                                                        path_results, load_model=True,
#                                                                        load_model_path=RTSNetPass1_path)
#
#             train_input_pass2 = rtsnet_out_train
#             train_target_pass2 = train_target
#             cv_input_pass2 = rtsnet_out_cv
#             cv_target_pass2 = cv_target
#
#             torch.save(
#                 [train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target],
#                 DatasetPass1_path)
#         #######################################
#         ## RTSNet_2passes with estimated H ###
#         # Build Neural Network
#         print("RTSNet estH pass 2 pipeline start!")
#         RTSNet_model = RTSNetNN()
#         RTSNet_model.NNBuild(sys_model_pass2, args)
#         print("Number of trainable parameters for RTSNet pass 2:",
#               sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
#         ## Train Neural Network
#         RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet_pass2")
#         RTSNet_Pipeline.setssModel(sys_model_pass2)
#         RTSNet_Pipeline.setModel(RTSNet_model)
#         RTSNet_Pipeline.setTrainingParams(args)
#         #######################################
#         # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(
#         #     sys_model_pass2, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results)
#         # RTSNet_Pipeline.save()
#         print("RTSNet pass 2 pipeline end!")
#         #######################################
#         # load trained Neural Network
#         print("Concat two RTSNets")
#         RTSNet_model1 = torch.load(RTSNetPass1_path)
#         RTSNet_model2 = torch.load('RTSNet/best-model.pt')
#         ## Set up Neural Network
#         RTSNet_Pipeline_2passes = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
#         RTSNet_Pipeline_2passes.setModel(RTSNet_model1, RTSNet_model2)
#         NumofParameter = RTSNet_Pipeline_2passes.count_parameters()
#         print("Number of parameters for RTSNet with 2 passes: ", NumofParameter)
#         ## Test Neural Network
#         [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, rtsnet_out,
#          RunTime] = RTSNet_Pipeline_2passes.NNTest(sys_model_esth, test_input, test_target, path_results)
#
# ###################################################################################
#
# else:
#     print("Error in switch! Please try 'full' or 'partial' or 'estH'.")
#
#
#
#
#
#
#
