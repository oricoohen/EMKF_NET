####the old one without the f
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
# from emkf.main_emkf_func import EMKF_F

from Simulations.utils import DataLoader, DataGen

import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

from Plot import Plot_RTS as Plot

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
path_results_wrongF = 'RTSNet/wrong_F/'

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
args.N_E = 2000  # Number of training examples (size of the training dataset).50
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

args_big.N_E    =  2000       # larger training set
args_big.N_CV   =   100
args_big.N_T    =   100       # test size can stay the same
args_big.T      =    30
args_big.T_test =    30

args_big.n_steps = 400        # extra epochs for fine-tune
args_big.n_batch =  30        # maybe different batch size
args_big.lr      = 1e-4       # usually ↓ a bit for fine-tune
args_big.wd      = 1e-3


###########################################################
r2 = torch.tensor([1e-3])
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

# True model
Q = q2 * Q_structure
R = r2 * R_structure
F = torch.tensor([[1, 0.1],[1, 1]]) # State transition matrix
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
DataGen(args, sys_model, dataFolderName + dataFileName,dataFolderName + dataFileName_F,delta =1, randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv,randomInit_test=InitIsRandom_test,randomLength=LengthIsRandom)
print("Data Load")


if(InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   [train_input, train_target, train_init, cv_input, cv_target, cv_init, test_input, test_target, test_init] = torch.load(dataFolderName + dataFileName)
   [F_train_mat,F_val_mat,F_test_mat] = torch.load(dataFolderName + dataFileName_F)
   # print("trainset size:",train_target.size())
   # print("cvset size:",cv_target.size())
   # print("testset size:",test_target.size())
elif(LengthIsRandom):
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName + dataFileName)
   [F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName + dataFileName_F)
   ### Check sequence lengths
   # for sequences in train_target:
   #    print("trainset size:",sequences.size())
   # for sequences in test_target:
   #    print("testset size:",sequences.size())
else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
   [F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName + dataFileName_F)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())



sys_model.F_train = F_train_mat
sys_model.F_valid = F_val_mat
sys_model.F_test = F_test_mat
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
if InitIsRandom_test:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target,F = F_test_mat, randomInit = True, test_init=test_init)
else:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target,F = F_test_mat)

#
# #############################
### Evaluate RTS Smoother ###
#############################



print("Evaluate RTS Smoother True")
if InitIsRandom_test:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target,F = F_test_mat, randomInit = True,test_init=test_init)
else:
    [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out,_,_] = S_Test(sys_model, test_input, test_target, F=F_test_mat)

PlotfolderName = 'Smoothers' + '/'
ComparedmodelName = 'Dataset'
Plot = Plot(PlotfolderName, ComparedmodelName)
print("Plot")
Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, MSE_obs_linear_arr)



#######################
### RTSNet Pipeline ###
#######################

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
if (InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results_full,True, randomInit = True, cv_init=cv_init,train_init=train_init)
   [MSE_train_p_smooth_dB_epoch,MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model, cv_input, cv_target, train_input, train_target, path_results_full, generate_f=True,
                 randomInit=True, cv_init=None, train_init=None)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime,P_smooth_list, V_list] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results_full,True,randomInit=True,test_init=test_init)

else:#e
    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results_full,True)
    [MSE_train_p_smooth_dB_epoch,MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model, cv_input, cv_target, train_input, train_target, path_results_full, generate_f=True,randomInit=False, cv_init=None, train_init=None)
    # ## Test Neural Network
    print("trsnet test11111111111111")
    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime,P_smooth_list, V_list,K,MSE_test_p_smooth_dB_avg,MSE_test_p_smooth_std] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results_full,True)

RTSNet_Pipeline.save()
# MSE_KF_dB_avg_true = MSE_KF_dB_avg          # KF  (true F)
# MSE_RTS_dB_avg_true = MSE_RTS_dB_avg         # E/RTS (true F)
# MSE_test_RTSNet_dB_avg_true = MSE_test_dB_avg
# MSE_cv_RTSNet_dB_avg_true = MSE_cv_dB_epoch
# MSE_train_RTSNet_dB_avg_true = MSE_train_dB_epoch
# MSE_test_p_smooth_dB_avg_true = MSE_test_p_smooth_dB_avg
# MSE_std_test_p_smooth_dB_true= MSE_test_p_smooth_std
# MSE_cv_p_smooth_dB_avg_true = MSE_cv_p_smooth_dB_epoch
# MSE_train_p_smooth_dB_avg_true = MSE_train_p_smooth_dB_epoch



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


[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName_2 + dataFileName_2)
[F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName_2 + dataFileName_F_2)
sys_model_2.F_train = F_train_mat
sys_model_2.F_valid = F_val_mat
sys_model_2.F_test = F_test_mat


#########change to wrong f
sys_model_2.F_train = change_F(F_train_mat,many=True)
sys_model_2.F_valid = change_F(F_val_mat,many=True)
sys_model_2.F_test= change_F(F_test_mat,many=True)
###################check the regu;ar rts
print('regular kalman and rts with wrong F')
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model_2, test_input, test_target,F =sys_model_2.F_test)

[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg2, RTS_out] = S_Test(sys_model_2, test_input, test_target,F = sys_model_2.F_test)
#############



###############################################################################
# 5) RE-USE **THE SAME** MODEL ─ tell the pipeline about new data + hyper-params
###############################################################################
RTSNet_Pipeline.setssModel(sys_model_2)           # new system matrices & F list
RTSNet_Pipeline.setTrainingParams(args_big)         # replaces optimiser with new LR, etc.

###############################################################################
# 6) ─── SECOND (big) TRAIN PASS (FINE-TUNE) ──────────────────────────────────
###############################################################################
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_2, cv_input, cv_target, train_input, train_target, path_results_full,True)
[MSE_train_p_smooth_dB_epoch,MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model_2, cv_input, cv_target, train_input, train_target, path_results_full, generate_f=True,randomInit=False, cv_init=None, train_init=None)
# ## Test Neural Network
print("trsnet test222222222")
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, rtsnet_out, RunTime, P_smooth_list, V_list, K,MSE_test_p_smooth_dB_avg, MSE_test_p_smooth_std] = (
    RTSNet_Pipeline.NNTest(sys_model_2, test_input, test_target, path_results_full, True))

###############################################################################
RTSNet_Pipeline.save()


D
### RTSNet with wrong F info ##############################################################################################
# Build Neural Network
# Create a new instance for the wrong model

sys_model_wrongF = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model_wrongF.InitSequence(m1_0, m2_0)

# # Wrong F
# sys_model_wrongF.F_train = rotate_F(F_train_mat,theta=0.03*torch.pi,mult=1.,many=True)
# sys_model_wrongF.F_valid = rotate_F(F_val_mat,theta=0.03*torch.pi,mult=1.,many=True)
# sys_model_wrongF.F_test= rotate_F(F_test_mat,theta=0.03*torch.pi,mult=1.,many=True)

sys_model_wrongF.F_train = change_F(F_train_mat,many=True)
sys_model_wrongF.F_valid = change_F(F_val_mat,many=True)
sys_model_wrongF.F_test= change_F(F_test_mat,many=True)


#sys_model_wrongF.F_test = [torch.tensor([[10., 0.],[20., 0.]]) for _ in range(len(F_test_mat))]

# Assume F_train_mat is a tensor of shape [N, m, m]
# Replace all F matrices with zeros of same shape
#
# zero_like_F_train = [torch.tensor([[1.4012,  0.8262],
#                   [-0.0738, 0.5988]]) for _ in range(len(F_train_mat))]
# zero_like_F_val = [torch.tensor([[1.4012,  0.8262],
#                   [-0.0738, 0.5988]]) for _ in range(len(F_val_mat))]
# zero_like_F_test = [torch.tensor([[1.4012,  0.8262],
#                   [-0.0738, 0.5988]]) for _ in range(len(F_test_mat))]

# Assign to wrongF system model
# sys_model_wrongF.F_train = zero_like_F_train
# sys_model_wrongF.F_valid = zero_like_F_val
# sys_model_wrongF.F_test = zero_like_F_test


[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model_wrongF, test_input, test_target,F =sys_model_wrongF.F_test)

[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg2, RTS_out] = S_Test(sys_model_wrongF, test_input, test_target,F = sys_model_wrongF.F_test)
print('oriiiiiiiii wrong',MSE_RTS_dB_avg2)
print("Zero F example:\n", sys_model_wrongF.F_test[0])



print('ori to checkkkkkkkkkkkkkkkkkkkkkkkkkk first',F_train_mat[3],'versoso',sys_model_wrongF.F_train[3] )

print("RTSNet with with wrong F")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_wrongF, args)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model_wrongF)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(args)


if (InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_wrongF, cv_input, cv_target, train_input, train_target, path_results_wrongF,generate_f=True, randomInit = True, cv_init=cv_init,train_init=train_init)
   [MSE_train_p_smooth_dB_epoch,MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model_wrongF, cv_input, cv_target, train_input, train_target, path_results_wrongF, generate_f=True,
                  randomInit=True, cv_init=None, train_init=None)
    ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime,P_smooth_test_list, V_test_list] = RTSNet_Pipeline.NNTest(sys_model_wrongF, test_input, test_target, path_results_wrongF,generate_f=True, randomInit=True,test_init=test_init)
else:
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_wrongF, cv_input, cv_target, train_input, train_target, path_results_wrongF,generate_f=True)
   # [MSE_train_p_smooth_dB_epoch,MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model_wrongF, cv_input, cv_target, train_input, train_target, path_results_wrongF,
   #                                 generate_f=True, randomInit=False, cv_init=None, train_init=None)
   # # Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime,P_smooth_test_list, V_test_list, K_T,MSE_test_p_smooth_dB_avg,MSE_test_p_smooth_std] = RTSNet_Pipeline.NNTest(sys_model_wrongF, test_input, test_target, path_results_wrongF,generate_f=True)

RTSNet_Pipeline.save()



MSE_KF_dB_avg_wrong = MSE_KF_dB_avg          # KF  (true F)
MSE_RTS_dB_avg_wrong = MSE_RTS_dB_avg         # E/RTS (true F)
MSE_test_RTSNet_dB_avg_wrong = MSE_test_dB_avg
MSE_cv_RTSNet_dB_avg_wrong = MSE_cv_dB_epoch
MSE_train_RTSNet_dB_avg_wrong = MSE_train_dB_epoch
MSE_test_p_smooth_dB_avg_wrong = MSE_test_p_smooth_dB_avg
MSE_std_test_p_smooth_dB_wrong= MSE_test_p_smooth_std
MSE_cv_p_smooth_dB_avg_wrong = MSE_cv_p_smooth_dB_epoch
MSE_train_p_smooth_dB_avg_wrong = MSE_train_p_smooth_dB_epoch

print('emkf in the pipeline')
#####do emkf###############
EMKF_F_model,_,_ = EMKF_F(sys_model_wrongF.F_test, H, Q, R, test_input, m1_0, m2_0,rtsnet_out, P_smooth_test_list,V_test_list, K_T)

# F_true, F_wrong, F_emkf are lists of torch.Tensor [m x m]

dist_wrong = []
dist_emkf = []
F_true = []
for F in sys_model.F_test:
    F_true.extend([F] * 10)
F_wrong = []
for F in sys_model_wrongF.F_test:
    F_wrong.extend([F] * 10)

for i in range(len(EMKF_F_model)):
    d_wrong = torch.norm(F_true[i] - F_wrong[i], p='fro')  # Frobenius norm
    d_emkf = torch.norm(F_true[i] - EMKF_F_model[i], p='fro')

    dist_wrong.append(d_wrong.item())
    dist_emkf.append(d_emkf.item())

    print(f"F[{i}] → Wrong Dist: {d_wrong.item():.6f}, EMKF Dist: {d_emkf.item():.6f} → Closer: {'EMKF' if d_emkf < d_wrong else 'Wrong'}")


    #summeration
    # 2.  Pretty‑print ------------------------------------------------------
    def _fmt(x):
        """Return a nicely formatted string for scalars or 'N/A' otherwise."""
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "   N/A   "

    print("\n" + "=" * 106)
    print("{:^106}".format("PERFORMANCE SUMMARY (ALL IN dB UNLESS NOTED)"))
    print("=" * 106)
    print("{:<36}|{:^16}|{:^16}|".format("Metric", "True‑F", "Wrong‑F"))
    print("-" * 106)
    print("{:<36}|{:>16}|{:>16}|".format("KF – Test MSE",
                                         _fmt(MSE_KF_dB_avg_true), _fmt(MSE_KF_dB_avg_wrong)))
    print("{:<36}|{:>16}|{:>16}|".format("RTS / ERTS – Test MSE",
                                         _fmt(MSE_RTS_dB_avg_true), _fmt(MSE_RTS_dB_avg_wrong)))
    print("{:<36}|{:>16}|{:>16}|".format("RTSNet – Test MSE",
                                         _fmt(MSE_test_RTSNet_dB_avg_true), _fmt(MSE_test_RTSNet_dB_avg_wrong)))
    print("{:<36}|{:>16}|{:>16}|".format("RTSNet – Best CV MSE",
                                         _fmt(min(MSE_cv_RTSNet_dB_avg_true)), _fmt(min(MSE_cv_RTSNet_dB_avg_wrong))))
    print("{:<36}|{:>16}|{:>16}|".format("RTSNet – Final Train MSE",_fmt(MSE_train_RTSNet_dB_avg_true[-1]),
                                         _fmt(MSE_train_RTSNet_dB_avg_wrong[-1])))
    print("{:<36}|{:>16}|{:>16}|".format("P‑smooth – Test mean MSE",
                                         _fmt(MSE_test_p_smooth_dB_avg_true), _fmt(MSE_test_p_smooth_dB_avg_wrong)))
    print("{:<36}|{:>16}|{:>16}|".format("P‑smooth – Test std (±)",
                                         _fmt(MSE_std_test_p_smooth_dB_true), _fmt(MSE_std_test_p_smooth_dB_wrong)))
    print("{:<36}|{:>16}|{:>16}|".format("P‑smooth – CV mean MSE",
                                         _fmt(MSE_cv_p_smooth_dB_avg_true), _fmt(MSE_cv_p_smooth_dB_avg_wrong)))
    print("{:<36}|{:>16}|{:>16}|".format("P‑smooth – Train mean MSE",
                                         _fmt(MSE_train_p_smooth_dB_avg_true), _fmt(MSE_train_p_smooth_dB_avg_wrong)))
    print("=" * 106 + "\n")