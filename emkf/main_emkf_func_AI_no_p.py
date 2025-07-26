import torch
from emkf.func_AI import  compute_A1, compute_A2, compute_A3, Ell
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

def EMKF_F_Mstep(sys_model,X_s, P_smooth_s, V_s,n):
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
        X (torch.Tensor): Smoothed states, shape (n, T)
        P_smooth (torch.Tensor): Smoothed covariances, shape (n, n, T)
        V (torch.Tensor): Lag-1 autocovariances, shape (n, n, T)
        K_T (torch.Tensor): Kalman gain at final time step, shape (n, m)

    Returns:
        F_all (torch.Tensor): All estimated F matrices during iterations, shape (num_iter+1, n, n)
        likelihood (torch.Tensor): Log-likelihood per iteration, shape (num_iter+1,)
        iterations (int): Number of EM iterations
    """
    SEQ = X_s.shape[0]
    A_1 = compute_A1(sys_model.m1x_0, X_s, V_s)  # (seq,n, n)
    A_2 = compute_A2(sys_model.m1x_0, sys_model.m2x_0, X_s, P_smooth_s)  # (seq,n, n)
    # Update equation for F: F^(i+1) = A_1^(i) @ inv(A_2^(i))
    eps = 1e-4 * torch.eye(n, device=A_2.device)
    A_2 = A_2 + eps
    #F_estimates_tensor = A_1 @ torch.linalg.pinv(A_2)

    F_estimates_tensor = torch.zeros_like(A_1)
    for s in range(SEQ):
        # solve A2_reg[s] @ X = A1[s]  =>  X = A2_reg[s]^{-1} @ A1[s]
        F_estimates_tensor[s] = torch.linalg.solve(A_2[s].T, A_1[s].T).T
    averaged_blocks_list = []
    # 2. Loop through the F_estimates_tensor in steps of 10.
    for i in range(0, SEQ, 10):
        # 3. Get the current chunk of up to 10 matrices.
        current_block = F_estimates_tensor[i: i + 10]
       # 4. Calculate the average of this chunk.
        average_of_block = current_block.mean(dim=0)
        # 5. Add the resulting average matrix to our list.
        averaged_blocks_list.append(average_of_block)
    # 6. After the loop, stack the list of averaged matrices into a single tensor.
    block_averages = torch.stack(averaged_blocks_list)

    return block_averages



# def EMKF_F_Mstep(sys_model, X_s, P_smooth_s, V_s, n, block_size=10, eps=1e-6):
#     """
#     X_s        : [S, n, T]          smoothed states
#     P_smooth_s : [S, n, n, T]        smoothed covariances
#     V_s        : [S, n, n, T]        lag-1 cross covariances
#     n          : state dimension
#     block_size : how many sequences share the same F (e.g. 10)
#     """
#     S, n_chk, T = X_s.shape
#     assert n_chk == n
#
#     F_each_seq = torch.zeros(S, n, n, device=X_s.device)
#     V_s = torch.stack(V_s)
#     for s in range(S):
#         A1 = torch.zeros(n, n, device=X_s.device)
#         A2 = torch.zeros(n, n, device=X_s.device)
#         # sum over t = 1..T-1
#         for t in range(1, T):
#             x_t   = X_s[s, :, t]      # [n]
#             x_tm1 = X_s[s, :, t-1]    # [n]
#
#             # add V_t
#             A1 += V_s[s, :, :, t]
#             # add x_t x_{t-1}^T
#             A1 += torch.outer(x_t, x_tm1)
#
#             # add P_{t-1}
#             A2 += P_smooth_s[s, :, :, t-1]
#             # add x_{t-1} x_{t-1}^T
#             A2 += torch.outer(x_tm1, x_tm1)
#
#                 # regularize and solve
#         A2 = A2 + eps * torch.eye(n, device=X_s.device)
#         F_each_seq[s] = torch.linalg.solve(A2.T, A1.T).T
#         # average every block_size sequences
#
#     blocks = []
#     for i in range(0, S, block_size):
#         # take the next up to block_size estimates
#         chunk = F_each_seq[i: i + block_size]  # shape [<=block_size, n, n]
#         blocks.append(chunk.mean(dim=0))  # average over that chunk
#         # blocks.append(F_each_seq[i])
#
#
#     F_blocks = torch.stack(blocks)  # shape [num_blocks, n, n]
#     return F_blocks

























def EMKF_F_N(sys_model,RTSNet_Pipeline,train_input, train_target, cv_input, cv_target,test_input,
                                                              test_target,model_pathes,max_it=3):
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
        # if q!=0:
        # # RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results=model_pathes[q], load_model_path=model_pathes[q], generate_f=True)
        #     F_train,F_valid = RTSNet_Pipeline.NNTrain_with_F(sys_model, cv_input, cv_target, train_input, train_target,
        #                                path_results=model_pathes[q], load_model_path=model_pathes[q], generate_f=True, beta=0.99)
        #     sys_model.F_train = F_train
        #     sys_model.F_test = F_valid
        #####TEST AND PSMOOTH###########

        [x_out_tensor,P_smooth_tensor,V_list] = RTSNet_Pipeline.NNTest_HybridP(sys_model, test_input, test_target, load_model_path=model_pathes[q])
             #############M STEP rts###############################
        F_est = EMKF_F_Mstep(sys_model,x_out_tensor,P_smooth_tensor,V_list,sys_model.m)
        # alpha = 0.2/(q+1)  # 0 < α ≤ 1  (smaller = safer)
        alpha = 1
        F_est = (1-alpha) * F_matrices[q] + alpha * F_est
        F_matrices.append(F_est)
        print('q_iter:', q, 'F_est:', F_est)
        # Check convergence
        if q > 0:
            delta_F.append(torch.abs(F_matrices[q] - F_matrices[q-1]).max())
        ####return the f to a list
        new_F_list = []
        for f_matrix in F_est:
            new_F_list.append(f_matrix)
        # new_F_list is now a Python list of 100 tensors, each with shape [2, 2]
        sys_model.F_test = new_F_list

    print('final_test')
    RTSNet_Pipeline.NNTest_HybridP(sys_model, test_input, test_target, load_model_path=model_pathes[-1])
    return F_matrices, delta_F