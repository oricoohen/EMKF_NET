import torch
import matplotlib.pyplot as plt
from emkf.func_AI import  compute_A1, compute_A2, compute_A3, Ell
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
#from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

def EMKF_F_Mstep(sys_model,X_s, P_smooth_s, V_s,m):
    """
    Perform EM for a single sequence to estimate the state transition matrix F.

    Args:
        F_0 (torch.Tensor): Initial estimate of F, shape (n, n)
        H (torch.Tensor): Observation matrix, shape (m, n)
        Q (torch.Tensor): Process noise covariance, shape (n, n)
        R (torch.Tensor): Measurement noise covariance, shape (m, m)
        y (torch.Tensor): Observations, shape (m, T)
        x_0 (torch.Tensor): Initial state estimate, shape (n,)
        P_0 (torch.Tensor): Initial covariance, shape (n, n)
        X (torch.Tensor): Smoothed states, shape (seq,n, T)
        P_smooth (torch.Tensor): Smoothed covariances, shape (seq,n, n, T)
        V (torch.Tensor): Lag-1 autocovariances, shape (seq,n, n, T)


    Returns:
        F_all (torch.Tensor): All estimated F matrices during iterations, shape (num_iter+1, n, n)
        likelihood (torch.Tensor): Log-likelihood per iteration, shape (num_iter+1,)
        iterations (int): Number of EM iterations
    """
    SEQ = X_s.shape[0]
    A_1 = compute_A1(sys_model.m1x_0, X_s, V_s)  # (seq,n, n)
    A_2 = compute_A2(sys_model.m1x_0, sys_model.m2x_0, X_s, P_smooth_s)  # (seq,n, n)
    # Update equation for F: F^(i+1) = A_1^(i) @ inv(A_2^(i))
    eps = 1e-4 * torch.eye(m, device=A_2.device)
    A_2 = A_2 + eps
    F_estimates_tensor = A_1 @ torch.linalg.pinv(A_2)
    # F_estimates_tensor = torch.zeros_like(A_1)
    # for s in range(SEQ):
    #     # solve A2_reg[s] @ X = A1[s]  =>  X = A2_reg[s]^{-1} @ A1[s]
    #     F_estimates_tensor[s] = torch.linalg.solve(A_2[s].T, A_1[s].T).T

    averaged_blocks_list = []
    # 2. Loop through the F_estimates_tensor in steps of 10.
    # if SEQ != 1:
    #     for i in range(0, SEQ, 10):
    #         # 3. Get the current chunk of up to 10 matrices.
    #         current_block = F_estimates_tensor[i: i + 10]
    #         # 4. Calculate the average of this chunk.
    #         average_of_block = current_block.mean(dim=0)
    #         # 5. Add the resulting average matrix to our list.
    #         averaged_blocks_list.append(average_of_block)
    #     # 6. After the loop, stack the list of averaged matrices into a single tensor.
    #     block_averages = torch.stack(averaged_blocks_list)
    #     return block_averages
    return F_estimates_tensor





def EMKF_F(sys_model,RTSNet_Pipeline,train_input, train_target, cv_input, cv_target,test_input,
                                                              test_target,model_pathes, psmooth_pathes,max_it=3):
    """
     EMKF_F:  Run EMKF_F_solo across multiple sequences in tensor form.
     Notation:
       • N_seq = number of sequences
       • T     = length of each time series
       • m     = measurement dimension
       • n     = state dimension
     Inputs:
       Y            : all measurements,          Tensor (N_seq, m, T)
       x_0          : prior mean of x₀,          Tensor (n, 1)
       P_0          : prior covariance of x₀,    Tensor (n, n)
       X            : smoothed state means,      Tensor (N_seq, n, T)
       P_smooth     : smoothed covariances,      Tensor (N_seq, n, n, T)
       V            : cross-covariances,         Tensor (N_seq, n, n, T)
       K_all        : Kalman gains per time,     Tensor (N_seq, n, m, T)
     Returns:
       F_out        : list of estimated F, length N_seq, each (n×n)
       ll_out       : list of final log-likelihoods, length N_seq
       it_out       : list of iteration counts, length N_seq
     """
    F_matrices = []
    F_matrices.append(torch.stack(sys_model.F_test))
    delta_F =[]


    for q in range(max_it):
        #############E STEP rts###############################
        ######TRAIN RTSNET###############
        #RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results=model_pathes[q], load_model_path=model_pathes[q], generate_f=True)
        # #####TRAIN PSMOOTH########
        # [MSE_train_p_smooth_dB_epoch, MSE_cv_p_smooth_dB_epoch] = RTSNet_Pipeline.P_smooth_Train(sys_model,cv_input, cv_target,
        #  train_input,train_target,path_results = psmooth_pathes[q],path_rtsnet = model_pathes[q],load_psmooth_path = psmooth_pathes[q],generate_f=True)
        # if q != 0:
        #     RTSNet_Pipeline.Train_Joint(sys_model, cv_input, cv_target, train_input, train_target,path_results_rtsnet=model_pathes[q],path_results_psmooth=psmooth_pathes[q],
        #                                load_rtsnet=model_pathes[q], load_psmooth=psmooth_pathes[q],generate_f=True)
            # sys_model.F_train = F_train
            # sys_model.F_test = F_valid
        #####TEST AND PSMOOTH###########
        if q == 0:
            [_,_,_,x_out_tensor,_,P_smooth_tensor,V_list,K_T_list,_,_] = RTSNet_Pipeline.NNTest(sys_model, test_input,test_target,load_model_path = model_pathes[q],
                load_p_smoothe_model_path = psmooth_pathes[q], generate_f=True,)
        else:
            [_,_,_,x_out_tensor,_,P_smooth_tensor,V_list,K_T_list,_,_] = RTSNet_Pipeline.NNTest(sys_model, test_input,test_target,load_model_path = model_pathes[q],
                load_p_smoothe_model_path = psmooth_pathes[q], generate_f=False,)
             #############M STEP rts###############################

        F_est = EMKF_F_Mstep(sys_model,x_out_tensor,P_smooth_tensor,V_list,sys_model.m)
        #alpha = 0.5/(q+1)  # 0 < α ≤ 1  (smaller = safer)
        # alpha = 1
        # F_est = (1-alpha) * F_matrices[q] + alpha * F_est
        F_matrices.append(F_est)
        print('q_iter:', q, 'F_est:', F_est)
        # Check convergence
        # if q > 0:
        #     delta_F.append(torch.abs(F_matrices[q] - F_matrices[q-1]).max())
        ####return the f to a list
        new_F_list = []
        for f_matrix in F_est:
            new_F_list.append(f_matrix)
        # new_F_list is now a Python list of 100 tensors, each with shape [2, 2]
        sys_model.F_test = new_F_list
    RTSNet_Pipeline.NNTest(sys_model, test_input,test_target,load_model_path = model_pathes[q],load_p_smoothe_model_path = psmooth_pathes[q], generate_f=True,)
    return F_matrices, delta_F