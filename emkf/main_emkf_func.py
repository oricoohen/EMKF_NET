import torch
import matplotlib.pyplot as plt
from emkf.func import  compute_A1, compute_A2, compute_A3, Ell
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0

from Simulations.utils import DataLoader, DataGen


def EMKF_F_solo(F_0, H, Q, R, y, x_0, P_0, X_s, P_smooth_s, V_s,n,T, max_it=100, tol_likelihood=0.01, tol_params=0.005):
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
    A_1 = compute_A1(x_0, X_s, V_s,n,T)  # (n, n)
    A_2 = compute_A2(x_0, P_0, X_s, P_smooth_s,n,T)  # (n, n)
    # Update equation for F: F^(i+1) = A_1^(i) @ inv(A_2^(i))
    eps = 1e-4 * torch.eye(n, device=A_2.device)
    A_2inv = torch.linalg.pinv(A_2+eps)
    F_fin = A_1 @ A_2inv
    # print('f_i shape',F_i.shape)
    return F_fin




# def EMKF_F(F_0_matrices, H, Q, R, Y_list, x_0, P_0, X_list, P_smooth_list, V_list, K_list, max_it=100, tol_likelihood=0.01, tol_params=0.005):
#     """
#     Run EMKF_F_solo across a list of sequences.
#
#     Args:
#         F_0_matrices (List[Tensor]): List of F_0 matrices, each (n, n)
#         H, Q, R (Tensor): System model parameters
#         Y_list (List[Tensor]): Observation sequences, each (m, T)
#         X_list (List[Tensor]): Smoothed states, each (n, T)
#         P_smooth_list (List[Tensor]): Smoothed covariances, each (n, n, T)
#         V_list (List[Tensor]): Lag-1 autocovariances, each (n, n, T)
#         K_list (List[Tensor]): Kalman gains
#
#     Returns:
#         Tuple of (List of estimated F, List of likelihoods, List of iteration counts)
#     """
#     F_matrices = []
#     likelihoods = []
#     iterations_list = []
#
#     for j in range(len(X_list)):
#         index = j // 10
#         F_0 = F_0_matrices[index]
#         Y = Y_list[j]
#         X = X_list[j]
#         P_smooth = P_smooth_list[j]
#         V = V_list[j]
#         K_T = K_list[j]
#
#         # print(f"Running EMKF on sequence {j + 1}/{len(X_list)} using F[{index}]")
#
#         F_est, likelihood, iterations = EMKF_F_solo(F_0, H, Q, R, Y, x_0, P_0, X, P_smooth, V, K_T, max_it,
#                                                     tol_likelihood, tol_params)
#         F_matrices.append(F_est)
#         likelihoods.append(likelihood)
#         iterations_list.append(iterations)
#
#     return F_matrices, likelihoods, iterations_list

# def EMKF_F(F_0_matrices, H, Q, R, Y_list, x_0, P_0, X_list, P_smooth_list, V_list, K_list, max_it=100, tol_likelihood=0.01, tol_params=0.005):
#     """
#     Run EMKF_F_solo across a list of sequences.
#
#     Args:
#         F_0_matrices (List[Tensor]): List of F_0 matrices, each (n, n)
#         H, Q, R (Tensor): System model parameters
#         Y_list (List[Tensor]): Observation sequences, each (m, T)
#         X_list (List[Tensor]): Smoothed states, each (n, T)
#         P_smooth_list (List[Tensor]): Smoothed covariances, each (n, n, T)
#         V_list (List[Tensor]): Lag-1 autocovariances, each (n, n, T)
#         K_list (List[Tensor]): Kalman gains
#
#     Returns:
#         Tuple of (List of estimated F, List of likelihoods, List of iteration counts)
#     """
#
#     # print(f"Running EMKF on sequence {j + 1}/{len(X_list)} using F[{index}]")
#
#     F_est, likelihood, iterations = EMKF_F_solo(F_0_matrices[0], H, Q, R, Y_list[0], x_0, P_0, X_list[0], P_smooth_list[0], V_list[0],K_list[0], max_it,
#                                                 tol_likelihood, tol_params)
#
#
#     return F_est, likelihood, iterations




def EMKF_F_analitic(sys_model,F_0_matrices, H, Q, R, Y, x_0, P_0, X, max_it=100, tol_likelihood=0.01, tol_params=0.005):
    """
     EMKF_F:  Run EMKF_F_solo across multiple sequences in tensor form.
     Notation:
       • N_seq = number of sequences
       • T     = length of each time series
       • m     = measurement dimension
       • n     = state dimension
     Inputs:
       F_0_matrices : list of initial F guesses, each Tensor (n×n)
       H            : observation matrix,        Tensor (m×n)
       Q            : process noise covariance,  Tensor (n×n)
       R            : measurement noise covariance, Tensor (m×m)
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
    likelihoods = []
    iterations_list = []
    n = sys_model.n
    T = sys_model.T_test
    for j in range(len(X)):
        index = j // 10
        F_est = F_0_matrices[index]
        Y_t = Y[j]
        X_t = X[j]
        F_all_j = []
        F_all_j.append(F_est)
        likelihood_j =[]
        for q in range(max_it):
            #############E STEP rts###############################
            [_,_,_, X_smooth, P_smooth_t, V_t] = S_Test(sys_model, Y_t.unsqueeze(0), X_t.unsqueeze(0), F=F_est.unsqueeze(0))
            likelihood = 0
            #############M STEP rts###############################
            F_est = EMKF_F_solo(F_est, H, Q, R, Y_t, x_0, P_0, X_smooth.squeeze(0), P_smooth_t.squeeze(0), V_t.squeeze(0),n,T,max_it, tol_likelihood, tol_params)
            alpha = 0.2/(q+1)  # 0 < α ≤ 1  (smaller = safer)
            F_est = alpha * F_all_j[q-1] + (1 - alpha) * F_est
            F_all_j.append(F_est)
            likelihood_j.append(likelihood)
            print('q_iter:', q, 'F_est:', F_est)
            # Check convergence
            if q > 0:
                delta_F = torch.abs(F_all_j[q] - F_all_j[q-1]).max()
                if delta_F < tol_params:
                    print(f"Converged on F at iteration {q}")
                    break
        F_matrices.append(F_all_j)
        iterations_list.append(q)
        likelihoods.append(likelihood_j)


    return F_matrices, likelihoods, iterations_list