import torch
import matplotlib.pyplot as plt
from emkf.func import  compute_A1, compute_A2, compute_A3, Ell


def EMKF_F_solo(F_0, H, Q, R, y, x_0, P_0, X, P_smooth, V, K_T, max_it=100, tol_likelihood=0.01, tol_params=0.005):
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

    T = y.shape[1]
    n = x_0.shape[0]

    F_all = [F_0.clone()]
    likelihood = []
    print('f_0 shape', F_0.shape)
    for i in range(max_it):
        A_1 = compute_A1(x_0, X, V)  # (n, n)
        A_2 = compute_A2(x_0, P_0, X, P_smooth)  # (n, n)
        # Update equation for F: F^(i+1) = A_1^(i) @ inv(A_2^(i))
        F_i = A_1 @ torch.linalg.pinv(A_2)
        print('f_i shape',F_i.shape)
        F_all.append(F_i)

        # Check convergence
        if i > 0:
            delta_F = torch.abs(F_all[i + 1] - F_all[i]).max()
            if delta_F < tol_params:
                print(f"Converged on F at iteration {i}")
                break

    return torch.stack(F_all), torch.tensor(likelihood), len(F_all) - 1




def EMKF_F(F_0_matrices, H, Q, R, Y_list, x_0, P_0, X_list, P_smooth_list, V_list, K_list, max_it=100, tol_likelihood=0.01, tol_params=0.005):
    """
    Run EMKF_F_solo across a list of sequences.

    Args:
        F_0_matrices (List[Tensor]): List of F_0 matrices, each (n, n)
        H, Q, R (Tensor): System model parameters
        Y_list (List[Tensor]): Observation sequences, each (m, T)
        X_list (List[Tensor]): Smoothed states, each (n, T)
        P_smooth_list (List[Tensor]): Smoothed covariances, each (n, n, T)
        V_list (List[Tensor]): Lag-1 autocovariances, each (n, n, T)
        K_list (List[Tensor]): Kalman gains

    Returns:
        Tuple of (List of estimated F, List of likelihoods, List of iteration counts)
    """
    F_matrices = []
    likelihoods = []
    iterations_list = []

    for j in range(len(X_list)):
        index = j // 10
        F_0 = F_0_matrices[index]
        Y = Y_list[j]
        X = X_list[j]
        P_smooth = P_smooth_list[j]
        V = V_list[j]
        K_T = K_list[j]

        print(f"Running EMKF on sequence {j + 1}/{len(X_list)} using F[{index}]")

        F_est, likelihood, iterations = EMKF_F_solo(F_0, H, Q, R, Y, x_0, P_0, X, P_smooth, V, K_T, max_it,
                                                    tol_likelihood, tol_params)
        F_matrices.append(F_est)
        likelihoods.append(likelihood)
        iterations_list.append(iterations)

    return F_matrices, likelihoods, iterations_list
