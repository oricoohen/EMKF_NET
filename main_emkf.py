import torch
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from emkf.main_emkf_func import EMKF_F_analitic, EMKF_F_solo
from Simulations.utils import DataLoader, DataGen
from RTSNet.PsmoothNN import PsmoothNN


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
Q = q2 * Q_structure*30
R = r2 * R_structure
F = torch.tensor([[1, 1], [0.1, 1]]) # State transition matrix
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
DataGen(args, sys_model, dataFolderName + dataFileName,dataFolderName + dataFileName_F,delta=1, randomInit_train=False,randomInit_cv=False,randomInit_test=False,randomLength=False)
print("Data Load")

[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
[F_train_mat, F_val_mat, F_test_mat] = torch.load(dataFolderName + dataFileName_F)
print("testset size:",test_target.size())

###############################################################################################

F_test_mat =[]
F_test_mat.append(F)
F_test_mat.append(F)

############kalman_TRUE############################
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target,F =F_test_mat)
############rts_TRUE###############################
[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out,P_smooth,V] = S_Test(sys_model, test_input, test_target,F= F_test_mat)

# 1. Run S_Test to get the outputs from the classical RTS smoother
print("\n--- Running Classical RTS Smoother ---")
[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out, P_smooth, V_test] = S_Test(sys_model, test_input, test_target, F=F_test_mat)
#########################################################################################################
# RTS_out has shape [N_T, n, T] and is our "x_est"
# P_smooth has shape [N_T, n, n, T] and is the covariance we want to evaluate
# test_target has shape [N_T, n, T] and is our "x_true"


# 2. Create a temporary PsmoothNN object. We only need it to call its compute_loss method.
#    This model does not need to be trained. It's just a tool for calculation.
loss_calculator = PsmoothNN(sys_model.m, args)


print("\n--- Comparing Estimated Covariance to Actual Error ---")

# 1. Initialize a variable to store the total error
total_frobenius_distance = 0.0
num_sequences = test_target.shape[0]

# This loop will run for each sequence in your test set
for i in range(num_sequences):

    # Get the data for the i-th sequence
    p_smooth_i = P_smooth[i]  # Shape: [m, m, T]
    rts_out_i = RTS_out[i]  # Shape: [m, T]
    test_target_i = test_target[i]  # Shape: [m, T]

    # 2. Calculate the actual error covariance for each time step
    p_error_list = []
    # Loop over every time step T for this specific sequence
    for t in range(test_target_i.shape[1]):
        # This is a 1D vector of shape [m]
        error_vec = test_target_i[:, t] - rts_out_i[:, t]
        # This correctly calculates the [m, m] outer product matrix
        error_covariance = error_vec.unsqueeze(1) @ error_vec.unsqueeze(0)
        p_error_list.append(error_covariance)

    # 3. Convert the list of [m, m] matrices into a single [m, m, T] tensor
    p_error_tensor = torch.stack(p_error_list, dim=2)

    # 4. As requested, print the matrices for the first sequence to compare
    if i == 0:
        print(f"\n--- Detailed Comparison for First Sequence (Sequence {i}) ---")
        # To make them comparable, we print the mean of each matrix over all time steps
        print("the first_perrorlist",p_error_list)
        print("the second real_p",p_smooth_i)

    # 5. Calculate the difference for this one sequence
    difference = p_smooth_i - p_error_tensor

    # 6. Compute the Frobenius norm of the difference
    frobenius_distance_for_sequence = torch.norm(difference, p='fro')

    # 7. Add this sequence's error to the total
    total_frobenius_distance += frobenius_distance_for_sequence.item()

# 8. Calculate the mean error (average distance) AFTER the loop is finished
mean_frobenius_error = total_frobenius_distance / num_sequences

print(f"\n--- Final Result ---")
print(f"The Mean Frobenius Distance between Estimated P and Actual Error P is: {mean_frobenius_error:.4f}")
print('msedb',10* torch.log10(torch.tensor(mean_frobenius_error)))
#######################################################################################













F_initial_1 = torch.tensor([[0.9, 1],[0.2, 1.1]])
F_initial_2  = torch.tensor([[0.9, 1],[0.2, 1.1]])
F_test_mat =[]
F_test_mat.append(F_initial_1)
F_test_mat.append(F_initial_2)



########EMKF##########
#####TRUE######
print('start EMKF')
F_matrices, likelihoods, iterations_list = EMKF_F_analitic(sys_model,F_test_mat, H, Q, R, test_input, m1_0, m2_0, test_target, max_it=100, tol_likelihood=0.01, tol_params=0.005)
print('True F matrices 1', F)
print('end of EMKF first:',F_matrices[0],'second', F_matrices[1])


#######FALSE_1#######



#######FALSE_2#######