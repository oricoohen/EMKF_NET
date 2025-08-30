####the old one without the f
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F,det
from emkf.main_emkf_func_AI import EMKF_F

from Simulations.utils import DataLoader, DataGen

import Simulations.config as config
from Simulations.Linear_canonical.parameters import  H, Q_structure, R_structure,m1_0, m2_0

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
path_results_full = 'RTSNet/full_info/'
path_results_2 = 'RTSNet/wrong_F/'

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_E = 200  # Number of training examples (size of the training dataset).50
args.N_CV = 100  # Number of cross-validation examples (size of the CV dataset used to tune hyperparameters).30
args.N_T = 100   # Number of test examples (size of the test dataset used to evaluate performance).100

args.T = 30    # Length of the time series for training and cross-validation sequences.
args.T_test = 30 # Length of the time series for test sequences.

### training parameters
args.n_steps = 100  # Number of training steps or iterations for optimization.
args.n_batch = 10    # Batch size: the number of sequences processed at each training step.10
args.lr = 1e-4       # Learning rate: controls how quickly the model updates during training.
args.wd = 1e-3       # Weight decay (L2 regularization): penalizes large weights to reduce overfitting.

###################ORI CHANGE
args_big   = config.general_settings()

args_big.N_E    =  200       # larger training set
args_big.N_CV   =   100
args_big.N_T    =   100       # test size can stay the same
args_big.T      =    30
args_big.T_test =    30

args_big.n_steps = 175        # extra epochs for fine-tune
args_big.n_batch =  30        # maybe different batch size
args_big.lr      = 1e-4       # usually ↓ a bit for fine-tune
args_big.wd      = 1e-3

max_iter = 3
###########################################################


# True model
q2 = 0.01
r2 =0.1
Q = q2 * Q_structure
R = r2 * R_structure
F = torch.tensor([[0.999, 0.1],[0, 0.999]]) # State transition matrix
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
S_Test(sys_model, test_input, test_target,F = F_test_mat)

######BAD F############################

##################second training
sys_model_2 = SystemModel(F, Q, H, R, args_big.T, args_big.T_test)
sys_model_2.InitSequence(m1_0, m2_0)
######create new data


dataFolderName_2 = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName_2 = '2x2_rq3030_T100_2.pt'
dataFileName_F_2 = '2x2_F_2'
print("Start Data Gen")
DataGen(args_big, sys_model_2, dataFolderName_2 + dataFileName_2,dataFolderName_2 + dataFileName_F_2,delta = 0.5, randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv,randomInit_test=InitIsRandom_test,randomLength=LengthIsRandom)
print("Data Load")


[train_input_2, train_target_2, cv_input_2, cv_target_2, test_input_2, test_target_2] = DataLoader(dataFolderName_2 + dataFileName_2)
[F_train_mat_2, F_val_mat_2, F_test_mat_2] = torch.load(dataFolderName_2 + dataFileName_F_2)
sys_model_2.F_train_TRUE = F_train_mat_2
sys_model_2.F_valid_TRUE = F_val_mat_2
sys_model_2.F_test_TRUE = F_test_mat_2

####create data######
#
# T = 30
# n = 2
# p = 2
#
# state = 1
# gen = np.random.default_rng(seed=state)
#
#
# F_sim2 = np.array([[0.83, 0.2],
#                    [0.2, 0.83]])          #   [[1,1],[0.1,1]]
#
#
# Q_sim2 = 0.01 * np.eye(n)               #   process-noise covariance
# H_sim2 = np.array([[1., 1.], [0.25, 1.]])                  #   full observation
# R_sim2 = 0.10 * np.eye(p)               #   measurement-noise covariance
#
# xi_sim2 = np.array([0.5, 0.5])          #   x₀ mean
# L_sim2  = np.eye(n)
#
# # x_sim2 = np.array([gen.multivariate_normal(xi_sim2, L_sim2)])
# x_sim2 = xi_sim2.reshape(1, n)  #   initial state
# z_sim2 = np.empty((1, p))
#
# for t in range(T):
#     x_sim2 = np.append(x_sim2, [F_sim2 @ x_sim2[t] + gen.multivariate_normal(np.zeros(n), Q_sim2)], axis=0)
#     z_sim2 = np.append(z_sim2, [H_sim2 @ x_sim2[t + 1] + gen.multivariate_normal(np.zeros(p), R_sim2)], axis=0)
# z_sim2 = np.delete(z_sim2, 0, axis=0)
# x_0 = x_sim2[0].reshape(n, 1)  #   initial state
# x_sim2 = np.delete(x_sim2, 0, axis=0)
# # make PyTorch default to 32-bit floats
# torch.set_default_dtype(torch.float32)
#
# x_true_t = torch.from_numpy(x_sim2.T).contiguous().float()
# z_meas_t = torch.from_numpy(z_sim2.T).contiguous().float()
#
# # Pack to match S_Test/KFTest interface: (N_T, dim, T)
# test_target = x_true_t.unsqueeze(0)    # (1, n, T)
# test_input  = z_meas_t.unsqueeze(0)    # (1, p, T)

#################################################################################################
# Set seeds
torch.manual_seed(1)


# # Parameters - EXACTLY matching your NumPy code
# T = 30
# n = 2
# p = 2
#
# # Use NumPy random generator for consistency with your original
# gen = np.random.default_rng(seed=1)
#
# # System matrices - EXACTLY matching
# F_sim2 = torch.tensor([[0.83, 0.2],
#                        [0.2, 0.83]], dtype=torch.float32)
#
# Q_sim2 = 0.01 * torch.eye(n, dtype=torch.float32)
# H_sim2 = torch.tensor([[1., 1.],
#                        [0.25, 1.]], dtype=torch.float32)
# R_sim2 = 0.10 * torch.eye(p, dtype=torch.float32)
#
# xi_sim2 = torch.tensor([0.5, 0.5], dtype=torch.float32)
#
# # Initialize - EXACTLY like NumPy
# x_sim2 = xi_sim2.unsqueeze(0)  # Shape: (1, 2) like numpy reshape(1, n)
# z_sim2 = torch.empty((1, p), dtype=torch.float32)  # Empty first row
#
# print("Initial state:", x_sim2)
# print("F matrix:", F_sim2)
# print("Q matrix:", Q_sim2)
# print("H matrix:", H_sim2)
# print("R matrix:", R_sim2)
#
# # Generate sequence - EXACTLY like NumPy loop
# for t in range(T):
#     # State evolution: x_{t+1} = F @ x_t + process_noise
#     current_state = x_sim2[t]  # Current state
#
#     # Generate process noise using NumPy (to match exactly)
#     process_noise_np = gen.multivariate_normal(np.zeros(n), Q_sim2.numpy())
#     process_noise = torch.from_numpy(process_noise_np).float()
#
#     # Evolve state
#     next_state = F_sim2 @ current_state + process_noise
#     x_sim2 = torch.cat([x_sim2, next_state.unsqueeze(0)], dim=0)
#
#     # Generate observation: z_t = H @ x_{t+1} + measurement_noise
#     measurement_noise_np = gen.multivariate_normal(np.zeros(p), R_sim2.numpy())
#     measurement_noise = torch.from_numpy(measurement_noise_np).float()
#
#     observation = H_sim2 @ next_state + measurement_noise
#     z_sim2 = torch.cat([z_sim2, observation.unsqueeze(0)], dim=0)
#
# # Remove first empty observation
# z_sim2 = z_sim2[1:]  # Remove first empty row
#
# # Remove initial state (like NumPy x_sim2[1:])
# x_0 = x_sim2[0].unsqueeze(1)  # Initial state as column vector
# x_sim2 = x_sim2[1:]  # Remove initial state
#
# print("\nGenerated data shapes:")
# print("x_sim2 shape:", x_sim2.shape)  # Should be (T, n)
# print("z_sim2 shape:", z_sim2.shape)  # Should be (T, p)
#
# # Convert to final format like your NumPy code
# x_true_t = x_sim2.T  # Transpose to (n, T)
# z_meas_t = z_sim2.T  # Transpose to (p, T)
#
# # Pack to match interface
# test_target = x_true_t.unsqueeze(0)  # (1, n, T)
# test_input = z_meas_t.unsqueeze(0)  # (1, p, T)
#
# print("\nFinal shapes:")
# print("test_target shape:", test_target.shape)
# print("test_input shape:", test_input.shape)
#
# print("\nFirst few states:")
# print("x_true_t[:, :5]:")
# print(x_true_t[:, :5])
#
# print("\nFirst few observations:")
# print("z_meas_t[:, :5]:")
# print(z_meas_t[:, :5])
#
# # Verify the relationship between observations and states
# print("\nVerification - First observation should be related to first state:")
# print("First state x_true_t[:, 0]:", x_true_t[:, 0])
# print("H @ first state:", H_sim2 @ x_true_t[:, 0])
# print("First observation z_meas_t[:, 0]:", z_meas_t[:, 0])
# print("Difference (should be noise):", z_meas_t[:, 0] - H_sim2 @ x_true_t[:, 0])
# # #########################################################################################################
# import torch
# from torch.distributions.multivariate_normal import MultivariateNormal
# #
# # #Set seed
# # torch.manual_seed(1)
#
# # Parameters - EXACTLY matching your NumPy code
# T = 30
# n = 2
# p = 2
#
# # System matrices - EXACTLY matching
# F_sim2 = torch.tensor([[0.83, 0.2],
#                        [0.2, 0.83]], dtype=torch.float32)
#
# #Q_sim2 = 0.01 * torch.eye(n, dtype=torch.float32)
# Q_sim2 = torch.eye(n, dtype=torch.float32)
#
# H_sim2 = torch.tensor([[1., 1.],
#                        [0.25, 1.]], dtype=torch.float32)
# # R_sim2 = 0.10 * torch.eye(p, dtype=torch.float32)
# R_sim2 = torch.eye(p, dtype=torch.float32)
# xi_sim2 = torch.tensor([0.5, 0.5], dtype=torch.float32)
#
# # Initialize - EXACTLY like NumPy
# x_sim2 = xi_sim2.unsqueeze(0)  # Shape: (1, 2) like numpy reshape(1, n)
# z_sim2 = torch.empty((1, p), dtype=torch.float32)  # Empty first row
#
# print("Initial state:", x_sim2)
# print("F matrix:", F_sim2)
# print("Q matrix:", Q_sim2)
# print("H matrix:", H_sim2)
# print("R matrix:", R_sim2)
#
# # Create distributions for noise generation
# process_noise_dist = MultivariateNormal(torch.zeros(n), Q_sim2)
#
# measurement_noise_dist = MultivariateNormal(torch.zeros(p), R_sim2)
#
# lam_r = 1.
#
# # Generate sequence - EXACTLY like NumPy loop
# for t in range(T):
#     # State evolution: x_{t+1} = F @ x_t + process_noise
#     current_state = x_sim2[t]  # Current state
#
#     # Generate process noise using PyTorch
#     # process_noise = process_noise_dist.sample()
#     # Sample from it
#
#     lam_vec_r = torch.full((2,), lam_r)
#     er = Exponential(lam_vec_r).sample()    # shape (n,)
#     process_noise = torch.reshape(er[:], z_sim2.size())
#     # Evolve state
#     next_state = F_sim2 @ current_state + process_noise
#     x_sim2 = torch.cat([x_sim2, next_state.unsqueeze(0)], dim=0)
#
#     # Generate observation: z_t = H @ x_{t+1} + measurement_noise
#     # measurement_noise = measurement_noise_dist.sample()
#
#     # measurement_noise = exp_dist.sample((1,))
#     lam_vec_q = torch.full((2,), lam_r)
#     eq = Exponential(lam_vec_r).sample()  # shape (n,)
#     measurement_noise = torch.reshape(eq[:], z_sim2.size())
#     observation = H_sim2 @ next_state + measurement_noise
#     z_sim2 = torch.cat([z_sim2, observation.unsqueeze(0)], dim=0)
#
# # Remove first empty observation
# z_sim2 = z_sim2[1:]  # Remove first empty row
#
# # Remove initial state (like NumPy x_sim2[1:])
# x_0 = x_sim2[0].unsqueeze(1)  # Initial state as column vector
# x_sim2 = x_sim2[1:]  # Remove initial state
#
# print("\nGenerated data shapes:")
# print("x_sim2 shape:", x_sim2.shape)  # Should be (T, n)
# print("z_sim2 shape:", z_sim2.shape)  # Should be (T, p)
#
# # Convert to final format like your NumPy code
# x_true_t = x_sim2.T  # Transpose to (n, T)
# z_meas_t = z_sim2.T  # Transpose to (p, T)
#
# # Pack to match interface
# test_target = x_true_t.unsqueeze(0)  # (1, n, T)
# test_input = z_meas_t.unsqueeze(0)  # (1, p, T)
#
# print("\nFinal shapes:")
# print("test_target shape:", test_target.shape)
# print("test_input shape:", test_input.shape)
#
# print("\nFirst few states:")
# print("x_true_t[:, :5]:")
# print(x_true_t[:, :5])
#
# print("\nFirst few observations:")
# print("z_meas_t[:, :5]:")
# print(z_meas_t[:, :5])
#
# # Verify the relationship between observations and states
# print("\nVerification - First observation should be related to first state:")
# print("First state x_true_t[:, 0]:", x_true_t[:, 0])
# print("H @ first state:", H_sim2 @ x_true_t[:, 0])
# print("First observation z_meas_t[:, 0]:", z_meas_t[:, 0])
# print("Difference (should be noise):", z_meas_t[:, 0] - H_sim2 @ x_true_t[:, 0])


#########change to wrong f option A
# sys_model_2.F_train = change_F(F_train_mat_2)
# sys_model_2.F_valid = change_F(F_val_mat_2)
# sys_model_2.F_test= change_F(F_test_mat_2)
# sys_model_2.args = args_big
#########change to wrong f option B

# sys_model_2.F_train = rotate_F(F_train_mat_2, i=0,j=1,theta = 0.087,mult = 1,many=True, randomit=True)
# sys_model_2.F_valid = rotate_F(F_val_mat_2, i=0,j=1,theta = 0.087,mult = 1,many=True, randomit=True)
# sys_model_2.F_test= rotate_F(F_test_mat_2, i=0,j=1,theta = 0.087,mult = 1,many=True, randomit=True)
# sys_model_2.args = args_big
#########change to wrong f option C

# sys_model_2.F_train = det(F_train_mat_2)
# sys_model_2.F_valid = det(F_val_mat_2)
# sys_model_2.F_test= det(F_test_mat_2)
# sys_model_2.F_train = F_train_mat_2
# sys_model_2.F_valid = F_val_mat_2
# sys_model_2.F_test = F_test_mat_2
sys_model_2.F_train = rotate_F(F_train_mat_2,0,1,0.87, many=True)
sys_model_2.F_valid = rotate_F(F_val_mat_2,0,1,0.87, many=True)
sys_model_2.F_test= rotate_F(F_test_mat_2,0,1,0.87, many=True)
print("det F_train:",sys_model_2.F_train)
sys_model_2.args = args_big




###################check the regu;ar rts
print('regular kalman and rts with wrong F')
KFTest(args, sys_model_2, test_input_2, test_target_2,F = sys_model_2.F_test)

S_Test(sys_model_2, test_input_2, test_target_2,F = sys_model_2.F_test)

#######################
### RTSNet Pipeline ###
#######################
########emkf#################################

rtsnet_models= []


# Create RTSNet
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(args)

path_results_full_rts = path_results_full+'best-model.pt'
path_results_full_rts2 = path_results_full+'best-model21.pt'
path_results_full_psmooth = path_results_full+'best-psmooth.pt'
path_results_2_rts = path_results_2+'best-model.pt'
path_results_2_psmooth = path_results_2+'best-psmooth.pt'
#####TRAIN GOOD F#####
print('rtssnet and psmooth with trueeeeeeee F')
#RTSNet_Pipeline.NNTrain( sys_model, cv_input, cv_target, train_input, train_target, path_results_full)
######TRAIN GOOD F########
#[MSE_train_p_smooth_dB_epoch,MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model,cv_input, cv_target,
#                    train_input, train_target, path_results =path_results_full_psmooth, path_rtsnet = path_results_full_rts,load_psmooth_path = None, generate_f=True)
#RTSNet_Pipeline.Train_Joint(sys_model, cv_input, cv_target, train_input, train_target, path_results_rtsnet=path_results_full_rts2 ,path_results_psmooth=path_results_full_psmooth,
#                            load_rtsnet = path_results_full_rts,load_psmooth = None, generate_f=True,beta=0.)

### Test Neural Network
RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results_full, True)


h

RTSNet_Pipeline.setTrainingParams(args_big)
print('rtssnet and psmooth with WRONGGGGGGG F')
#######TRAIN BAD F########
RTSNet_Pipeline.NNTrain(sys_model_2, cv_input_2, cv_target_2, train_input_2, train_target_2, path_results = path_results_2,generate_f=True)
#########TRAIN BAD F############
#[MSE_train_p_smooth_dB_epoch_2,MSE_cv_p_smooth_dB_epoch_2] = RTSNet_Pipeline.P_smooth_Train(sys_model_2, cv_input_2, cv_target_2, train_input_2,
                #train_target_2, path_results = path_results_2_psmooth,path_rtsnet = path_results_2_rts, load_psmooth_path=path_results_full_psmooth, generate_f=True)

# ## Test Neural Network
RTSNet_Pipeline.NNTest(sys_model_2, test_input_2, test_target_2, path_results_2, True)
# The folder where the new copies will be saved.
destination_folder = 'RTSNet/EMKF/'
a
# --- Step 2: Loop 5 times and copy the file ---
model_pathes = []
psmooth_pathes = []




sys_model_2.args = args
RTSNet_Pipeline.setTrainingParams(args)

EMKF_F(sys_model_2,RTSNet_Pipeline,train_input_2, train_target_2, cv_input_2, cv_target_2,test_input_2,
                                                              test_target_2,model_pathes,psmooth_pathes)
































































### RTSNet with full info ##############################################################################################
# Build Neural Network
print("RTSNet with full model info")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(args)
#

[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results_full,True)
# [MSE_train_p_smooth_dB_epoch,MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model, cv_input, cv_target, train_input, train_target, path_results_full, generate_f=None,randomInit=False, cv_init=None, train_init=None)
# ## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime,P_smooth_list, V_list,K,MSE_test_p_smooth_dB_avg,MSE_test_p_smooth_std] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results_full,True)



sys_model_2 = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model_2.InitSequence(m1_0, m2_0)
######create new data


dataFolderName_2 = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName_2 = '2x2_rq3030_T100_2.pt'
dataFileName_F_2 = '2x2_F_2'
print("Start Data Gen")
DataGen(args, sys_model_2, dataFolderName_2 + dataFileName_2,dataFolderName_2 + dataFileName_F_2,delta = 1, randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv,randomInit_test=InitIsRandom_test,randomLength=LengthIsRandom)
print("Data Load")


[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName_2 + dataFileName_2)
[F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName_2 + dataFileName_F_2)
sys_model_2.F_train = F_train_mat
sys_model_2.F_valid = F_val_mat
sys_model_2.F_test = F_test_mat


print("RTSNet true test2222222222222222")

[MSE_test_linear_arr2, MSE_test_linear_avg2, MSE_test_dB_avg2,rtsnet_out2,RunTime2,P_smooth_list2, V_list2,K2,MSE_test_p_smooth_dB_avg2,MSE_test_p_smooth_std2] = RTSNet_Pipeline.NNTest(sys_model_2, test_input, test_target, path_results_full,True)


