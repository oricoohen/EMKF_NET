import torch
import matplotlib.pyplot as plt
from emkf.func import  compute_A1, compute_A2, compute_A3, Ell
from Simulations.Linear_sysmdl import SystemModel, rotate_F, change_F
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_H
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_old as S_Test_ext
import Simulations.config as config
from Simulations.Linear_canonical.parameters import Q_structure, R_structure, m1_0, m2_0

from Simulations.utils import DataLoader, DataGen


def EMKF_F_solo(F_0, H, Q, R, y, x_0, P_0, X_s, P_smooth_s, V_s,n,T, tol_likelihood=0.01, tol_params=0.005, vel_only=False):
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
    eps = 1e-7 * torch.eye(n, device=A_2.device)
    # A_2inv = torch.linalg.pinv(A_2+eps) ####istead of solving linlag we will solve the equation
    # F_fin = A_1 @ A_2inv
    if vel_only:
        # Estimate only the 2x2 velocity block; position rows are fixed kinematics
        A_1_vv = A_1[2:4, 2:4]
        A_2_vv = A_2[2:4, 2:4] + 1e-7 * torch.eye(2, device=A_2.device)
        F_vv = torch.linalg.solve(A_2_vv.T, A_1_vv.T).T
        F_fin = F_0.clone()
        F_fin[2:4, 2:4] = F_vv
    else:
        A_2 = A_2 + eps
        F_fin = torch.linalg.solve(A_2.T, A_1.T).T

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




def EMKF_F_analitic(sys_model,F_0_matrices, H, Q, R, Y, x_0, P_0, X, max_it=3, generate_f=True, tol_likelihood=0.01,
                    tol_params=0.005,init_x_list=None, init_P_list=None):
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
    m = sys_model.m
    T = Y.size(-1)
    # Accumulators for MSE across sequences, per iteration
    sum_lin_per_iter = torch.zeros(max_it, dtype=torch.float64, device=Y.device)
    last_x_list = []   #ori
    last_P_list = []   #

    for j in range(len(X)):
        if generate_f ==True:
            index = j // 10
        else:
            index= j
        F_est = F_0_matrices[index]

        Y_t = Y[j]
        X_t = X[j]
        F_all_j = []
        F_all_j.append(F_est)
        likelihood_j =[]
        for q in range(max_it):
            #############E STEP rts###############################
            # pick per-seq initials if provided
            if init_x_list is not None:
                x_0 = init_x_list[j]
                P_0 = init_P_list[j]
                sys_model.InitSequence(x_0, P_0)
                x0_j = [init_x_list[j]]
                P0_j = [init_P_list[j]]
            else:
                x0_j = None
                P0_j = None
                sys_model.InitSequence(x_0, P_0)
            # print('q_iter:', q, 'F_est:', F_est)
            [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth_t, V_t] = S_Test(sys_model, Y_t.unsqueeze(0), X_t.unsqueeze(0),
                F=F_est.unsqueeze(0),H=H,generate_f=False, init_x_list=x0_j,
                init_P_list=P0_j)
            # Compute *our* MSE for this sequence & iteration (linear, not dB)
            sum_lin_per_iter[q] += float(_mse_avg)
            #############M STEP rts###############################
            F_est = EMKF_F_solo(F_est, H, Q, R, Y_t, x_0, P_0, X_smooth.squeeze(0), P_smooth_t.squeeze(0), V_t.squeeze(0),
                                m,T, tol_likelihood, tol_params)
            #alpha = 0.6/(q/5+1)  # 0 < α ≤ 1  (smaller = safer)
            alpha = 0
            F_est = alpha * F_all_j[q-1] + (1 - alpha) * F_est
            F_all_j.append(F_est)


        F_matrices.append(F_all_j)
        iterations_list.append(q)
        likelihoods.append(likelihood_j)
        # after final iteration, keep last smoothed x_T, P_T (for next dataset)
        x_T = X_smooth[0, :, -1].unsqueeze(-1).clone()  # [m,1]
        P_T = P_smooth_t[0, :, :, -1].clone()  # [m,m]
        last_x_list.append(x_T)
        last_P_list.append(P_T)
    # -------- After all sequences: report mean MSE per iteration --------
    mean_mse_per_seq_lin = (sum_lin_per_iter/len(X)).clone()
    mean_mse_db = 10.0 * torch.log10(mean_mse_per_seq_lin + 1e-12)

    print("\n=== Mean MSE across sequences per iteration ===")
    for q in range(max_it):
        print(f"Iter {q:02d}: mean MSE = {mean_mse_db[q].item():.3f} dB")

    return F_matrices, likelihoods, iterations_list, mean_mse_per_seq_lin[-1], last_x_list, last_P_list



def E_EMKF_F_analitic_non_linear_h(sys_model,F_0_matrices, h, Q, R, Y, x_0, P_0, X, max_it=3, generate_f=True,init_x_list=None, init_P_list=None, vel_only=False, obs_mask=None):
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
    iterations_list = []
    m = sys_model.m
    T = Y.size(-1)
    # Accumulators for MSE across sequences, per iteration
    sum_lin_per_iter = torch.zeros(max_it, dtype=torch.float64, device=Y.device)
    last_x_list = []   #ori
    last_P_list = []   #

    for j in range(len(X)):
        if generate_f ==True:
            index = j // 10
        else:
            index= j
        F_est = F_0_matrices[index]

        Y_t = Y[j]
        X_t = X[j]
        F_all_j = []
        F_all_j.append(F_est)

        if j == 0:
            import math as _math
            F_vv0 = F_est[2:4, 2:4]
            theta0 = _math.atan2(F_vv0[1, 0].item(), F_vv0[0, 0].item())
            sr0 = torch.linalg.eigvals(F_vv0).abs().max().item()
            print(f"\n  [Seq0 INIT] theta_F0={theta0:.4f}rad  spec_rad(Fvv)={sr0:.4f}")
            print(f"  {'Iter':>4}  {'MSE(dB)':>9}  {'theta_est(rad)':>15}  {'spec_rad':>9}")

        for q in range(max_it):
            #############E STEP rts###############################
            # pick per-seq initials if provided
            if init_x_list is not None:
                x_0 = init_x_list[j]
                P_0 = init_P_list[j]
                sys_model.InitSequence(x_0, P_0)
                x0_j = [init_x_list[j]]
                P0_j = [init_P_list[j]]
            else:
                x0_j = None
                P0_j = None
                sys_model.InitSequence(x_0, P_0)

            if obs_mask is None:
                # Original path — full measurements every step (no mask)
                [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth_t, V_t] = S_Test_ext(
                    sys_model, Y_t.unsqueeze(0), X_t.unsqueeze(0),
                    F_list=[F_est.clone()], generate_f=False,
                    init_x_list=x0_j, init_P_list=P0_j)
                x_s_j    = X_smooth.squeeze(0)    # [m, T]
                P_s_j    = P_smooth_t.squeeze(0)  # [m, m, T]
                V_j      = V_t.squeeze(0)         # [m, m, T]
            else:
                # Fair path — same sparse measurements as ERTS baselines
                from Simulations.TDOA_2D.ekf_erts import run_ekf_erts as _run_erts
                from Simulations.TDOA_2D.ekf_erts import compute_cross_covariances as _cc
                import torch.nn as _nn
                def _gF(_F):
                    def _f(_t): return _F
                    return _f
                x_s_j, P_s_j, P_f_j, sgains_j, H_last_j, K_last_j = _run_erts(
                    Y_t, _gF(F_est), Q_in=Q, R_in=R,
                    x_init=x_0, P_init=P_0, obs_mask=obs_mask,
                )
                V_j      = _cc(F_est, H_last_j, K_last_j, P_f_j, sgains_j)
                _mse_avg = _nn.MSELoss(reduction='mean')(x_s_j, X_t).item()

            # Compute *our* MSE for this sequence & iteration (linear, not dB)
            sum_lin_per_iter[q] += float(_mse_avg)
            #############M STEP rts###############################
            F_est = EMKF_F_solo(F_est, h, Q, R, Y_t, x_0, P_0, x_s_j, P_s_j, V_j,
                                m, T, vel_only=vel_only)
            #alpha = 0.6/(q/5+1)  # 0 < α ≤ 1  (smaller = safer)
            alpha = 0
            F_est = alpha * F_all_j[q-1] + (1 - alpha) * F_est
            F_all_j.append(F_est)

            if j == 0:
                F_vv = F_est[2:4, 2:4]
                theta_est = _math.atan2(F_vv[1, 0].item(), F_vv[0, 0].item())
                spec_rad = torch.linalg.eigvals(F_vv).abs().max().item()
                mse_db_q = 10.0 * _math.log10(float(_mse_avg) + 1e-12)
                print(f"  {q:>4d}  {mse_db_q:>9.3f}  {theta_est:>15.4f}  {spec_rad:>9.4f}")


        F_matrices.append(F_all_j)
        # after final iteration, keep last smoothed x_T, P_T (for next dataset)
        x_T = x_s_j[:, -1].unsqueeze(-1).clone()  # [m,1]
        P_T = P_s_j[:, :, -1].clone()             # [m,m]
        last_x_list.append(x_T)
        last_P_list.append(P_T)
    # -------- After all sequences: report mean MSE per iteration --------
    mean_mse_per_seq_lin = (sum_lin_per_iter/len(X)).clone()
    mean_mse_db = 10.0 * torch.log10(mean_mse_per_seq_lin + 1e-12)

    print("\n=== Mean MSE across sequences per iteration ===")
    for q in range(max_it):
        print(f"Iter {q:02d}: mean MSE = {mean_mse_db[q].item():.3f} dB")

    return F_matrices, mean_mse_per_seq_lin[-1], last_x_list, last_P_list



def EMKF_H_analitic(sys_model, F, H_0_matrices, Q, R, Y, x_0, P_0, X, max_it=3, generate_h=True,init_x_list=None, init_P_list=None):
    """
    EM algorithm for estimating observation matrix H.

    This function performs EM iterations to estimate H while keeping F FIXED.
    Structure: For each sequence, run all EM iterations before moving to next sequence.

    Notation:
      • N_seq = number of sequences
      • T     = length of each time series
      • m     = state dimension
      • n     = observation dimension

    Inputs:
      sys_model    : SystemModel object containing system parameters
      F            : FIXED state transition matrix, Tensor (m×m) - SAME for all sequences
      H_0_matrices : list of initial H guesses (TO BE ESTIMATED), each Tensor (n×m)
      Q            : process noise covariance, Tensor (m×m)
      R            : measurement noise covariance, Tensor (n×n)
      Y            : all measurements, Tensor (N_seq, n, T)
      x_0          : prior mean of x₀, Tensor (m, 1)
      P_0          : prior covariance of x₀, Tensor (m, m)
      X            : true states (for MSE computation), Tensor (N_seq, m, T)
      max_it       : maximum number of EM iterations per sequence
      generate_h   : if True, use grouped H (index = j // 10); else one H per sequence
      init_x_list  : optional list of initial state means per sequence
      init_P_list  : optional list of initial covariances per sequence

    Returns:
      H_matrices   : list of H estimates per sequence, length N_seq, each list has (max_it+1) H estimates
      mean_mse_final : final mean MSE across sequences (scalar)
      last_x_list  : list of final smoothed states x_T for each sequence
      last_P_list  : list of final smoothed covariances P_T for each sequence
    """

    m = sys_model.m  # state dimension
    n = sys_model.n  # observation dimension
    T = Y.size(-1)   # sequence length
    N_seq = Y.size(0)  # number of sequences

    # Accumulators for MSE across sequences, per iteration
    sum_mse_per_iter = torch.zeros(max_it, dtype=torch.float64, device=Y.device)
    last_x_list = []
    last_P_list = []
    H_matrices = []  # Store H evolution for each sequence
    F_i = F[0].clone()

    print(f"\n{'='*60}")
    print(f"Starting EM for H estimation")
    print(f"  Sequences: {N_seq}, State dim: {m}, Obs dim: {n}, Length: {T}")
    print(f"  Iterations per sequence: {max_it}")
    print(f"  F is FIXED (same for all), estimating H")
    print(f"{'='*60}\n")

    # ========== OUTER LOOP: For each sequence ==========
    for j in range(N_seq):

        # Select appropriate H initial guess for this sequence/group
        if generate_h:
            h_index = j // 10
        else:
            h_index = j
        H_est = H_0_matrices[h_index].clone()

        Y_j = Y[j]      # [n, T]
        X_true_j = X[j] # [m, T]

        # Initialize H evolution list for this sequence
        H_all_j = [H_est.clone()]

        # Set initial conditions for this sequence
        if init_x_list is not None:
            x0_j = init_x_list[j]
            P0_j = init_P_list[j]
        else:
            x0_j = x_0
            P0_j = P_0


        # ========== INNER LOOP: EM iterations for this sequence ==========
        for q in range(max_it):
            # ========== E-STEP: Run RTS smoother with current H and FIXED F ==========
            sys_model.InitSequence(x0_j, P0_j)

            [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, V_smooth] = S_Test(sys_model,Y_j.unsqueeze(0),X_true_j.unsqueeze(0), F=F_i.unsqueeze(0),H=[H_est],generate_f=False, generate_h = False,init_x_list=[x0_j],init_P_list=[P0_j])



            # Accumulate MSE for this iteration
            sum_mse_per_iter[q] += float(_mse_avg)

            # Extract results (remove batch dimension)
            X_s = X_smooth.squeeze(0)  # [m, T]
            P_s = P_smooth.squeeze(0)  # [m, m, T]

            # ========== M-STEP: Update H for this sequence ==========
            # Accumulate sufficient statistics over time
            C1 = torch.zeros((n, m), dtype=Y.dtype, device=Y.device)  # Σ_t y_t x_t^T
            C2 = torch.zeros((m, m), dtype=Y.dtype, device=Y.device)  # Σ_t (x_t x_t^T + P_t)

            for t in range(T):
                # C1: Σ_t y_t x_t^T
                C1 += Y_j[:, t].unsqueeze(1) @ X_s[:, t].unsqueeze(0)  # [n,1] @ [1,m] = [n,m]

                # C2: Σ_t (x_t x_t^T + P_t)
                C2 += X_s[:, t].unsqueeze(1) @ X_s[:, t].unsqueeze(0) + P_s[:, :, t]  # [m,m]

            # Add regularization for numerical stability
            eps = 1e-6 * torch.eye(m, device=C2.device, dtype=C2.dtype)
            C2 = C2 + eps

            # H_new = C1 @ inv(C2)
            # Using solve for numerical stability: H_new^T = C2^T \ C1^T
            H_est = torch.linalg.solve(C2.T, C1.T).T  # [n, m]
            H_all_j.append(H_est.clone())

        # Store H evolution for this sequence
        H_matrices.append(H_all_j)

        # Store final state for continuity across sequences
        x_T = X_s[:, -1].unsqueeze(-1).clone()  # [m, 1]
        P_T = P_s[:, :, -1].clone()             # [m, m]
        last_x_list.append(x_T)
        last_P_list.append(P_T)

    # -------- After all sequences: report mean MSE per iteration --------
    mean_mse_per_seq_lin = (sum_mse_per_iter / N_seq).clone()
    mean_mse_db = 10.0 * torch.log10(mean_mse_per_seq_lin + 1e-12)

    print(f"\n{'='*60}")
    print("EM iterations completed for all sequences")
    print(f"{'='*60}\n")

    print("=== Mean MSE across sequences per iteration ===")
    for q in range(max_it):
        print(f"Iter {q:02d}: mean MSE = {mean_mse_db[q].item():.3f} dB")

    final_mean_mse = mean_mse_per_seq_lin[-1]

    # Create empty lists for compatibility with expected return format
    likelihoods = [[] for _ in range(N_seq)]
    iterations_list = [max_it-1 for _ in range(N_seq)]

    return H_matrices, likelihoods, iterations_list, final_mean_mse, last_x_list, last_P_list


def EMKF_H_analitic_f_nonlinear(sys_model, H_0_matrices, Y, x_0, P_0, X,max_it=3, generate_h=True, init_x_list=None, init_P_list=None):
    """
    EM algorithm for estimating observation matrix H.

    This function performs EM iterations to estimate H while keeping F FIXED.
    Structure: For each sequence, run all EM iterations before moving to next sequence.

    Notation:
      • N_seq = number of sequences
      • T     = length of each time series
      • m     = state dimension
      • n     = observation dimension

    Inputs:
      sys_model    : SystemModel object containing system parameters
      F            : FIXED state transition matrix, Tensor (m×m) - SAME for all sequences
      H_0_matrices : list of initial H guesses (TO BE ESTIMATED), each Tensor (n×m)
      Q            : process noise covariance, Tensor (m×m)
      R            : measurement noise covariance, Tensor (n×n)
      Y            : all measurements, Tensor (N_seq, n, T)
      x_0          : prior mean of x₀, Tensor (m, 1)
      P_0          : prior covariance of x₀, Tensor (m, m)
      X            : true states (for MSE computation), Tensor (N_seq, m, T)
      max_it       : maximum number of EM iterations per sequence
      generate_h   : if True, use grouped H (index = j // 10); else one H per sequence
      init_x_list  : optional list of initial state means per sequence
      init_P_list  : optional list of initial covariances per sequence

    Returns:
      H_matrices   : list of H estimates per sequence, length N_seq, each list has (max_it+1) H estimates
      mean_mse_final : final mean MSE across sequences (scalar)
      last_x_list  : list of final smoothed states x_T for each sequence
      last_P_list  : list of final smoothed covariances P_T for each sequence
    """

    m = sys_model.m  # state dimension
    n = sys_model.n  # observation dimension
    T = Y.size(-1)   # sequence length
    N_seq = Y.size(0)  # number of sequences

    # Accumulators for MSE across sequences, per iteration
    sum_mse_per_iter = torch.zeros(max_it, dtype=torch.float64, device=Y.device)
    last_x_list = []
    full_x_list = []  # Store full smoothed trajectories for each sequence
    last_P_list = []
    H_matrices = []  # Store H evolution for each sequence

    print(f"\n{'='*60}")
    print(f"Starting EM for H estimation")
    print(f"  Sequences: {N_seq}, State dim: {m}, Obs dim: {n}, Length: {T}")
    print(f"  Iterations per sequence: {max_it}")
    print(f"  F is FIXED (same for all), estimating H")
    print(f"{'='*60}\n")

    # ========== OUTER LOOP: For each sequence ==========
    for j in range(N_seq):

        # Select appropriate H initial guess for this sequence/group
        if generate_h:
            h_index = j // 10
        else:
            h_index = j
        H_est = H_0_matrices[h_index].clone()

        Y_j = Y[j]      # [n, T]
        X_true_j = X[j] # [m, T]

        # Initialize H evolution list for this sequence
        H_all_j = [H_est.clone()]

        # Set initial conditions for this sequence
        if init_x_list is not None:
            x0_j = init_x_list[j]
            P0_j = init_P_list[j]
        else:
            x0_j = x_0
            P0_j = P_0


        # ========== INNER LOOP: EM iterations for this sequence ==========
        for q in range(max_it):
            # ========== E-STEP: Run RTS smoother with current H and FIXED F ==========
            sys_model.InitSequence(x0_j, P0_j)

            [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, V_smooth] = S_Test_ext_H(
                sys_model,
                Y_j.unsqueeze(0),
                X_true_j.unsqueeze(0),
                H_list=[H_est],
                generate_h=False,
                init_x_list=[x0_j],
                init_P_list=[P0_j]
            )

            # Accumulate MSE for this iteration
            sum_mse_per_iter[q] += float(_mse_avg)

            # Extract results (remove batch dimension)
            X_s = X_smooth.squeeze(0)  # [m, T]
            P_s = P_smooth.squeeze(0)  # [m, m, T]

            # ========== M-STEP: Update H for this sequence ==========
            # Accumulate sufficient statistics over time
            C1 = torch.zeros((n, m), dtype=Y.dtype, device=Y.device)  # Σ_t y_t x_t^T
            C2 = torch.zeros((m, m), dtype=Y.dtype, device=Y.device)  # Σ_t (x_t x_t^T + P_t)

            for t in range(T):
                # C1: Σ_t y_t x_t^T
                C1 += Y_j[:, t].unsqueeze(1) @ X_s[:, t].unsqueeze(0)  # [n,1] @ [1,m] = [n,m]

                # C2: Σ_t (x_t x_t^T + P_t)
                C2 += X_s[:, t].unsqueeze(1) @ X_s[:, t].unsqueeze(0) + P_s[:, :, t]  # [m,m]

            # Add regularization for numerical stability
            eps = 1e-6 * torch.eye(m, device=C2.device, dtype=C2.dtype)
            C2 = C2 + eps

            # H_new = C1 @ inv(C2)
            # Using solve for numerical stability: H_new^T = C2^T \ C1^T
            H_est = torch.linalg.solve(C2.T, C1.T).T  # [n, m]
            H_all_j.append(H_est.clone())

        # Store H evolution for this sequence
        H_matrices.append(H_all_j)

        # Store final state for continuity across sequences
        x_T = X_s[:, -1].unsqueeze(-1).clone()  # [m, 1]
        P_T = P_s[:, :, -1].clone()             # [m, m]
        last_x_list.append(x_T)
        last_P_list.append(P_T)
        full_x_list.append(X_s)  # Store full smoothed trajectory for this sequence

    # -------- After all sequences: report mean MSE per iteration --------
    mean_mse_per_seq_lin = (sum_mse_per_iter / N_seq).clone()
    mean_mse_db = 10.0 * torch.log10(mean_mse_per_seq_lin + 1e-12)

    print(f"\n{'='*60}")
    print("EM iterations completed for all sequences")
    print(f"{'='*60}\n")

    print("=== Mean MSE across sequences per iteration ===")
    for q in range(max_it):
        print(f"Iter {q:02d}: mean MSE = {mean_mse_db[q].item():.3f} dB")

    final_mean_mse = mean_mse_per_seq_lin[-1]

    # Create empty lists for compatibility with expected return format
    likelihoods = [[] for _ in range(N_seq)]
    iterations_list = [max_it-1 for _ in range(N_seq)]

    return H_matrices, likelihoods, iterations_list, final_mean_mse, last_x_list, last_P_list,full_x_list


def EMKF_FH_analytic(sys_model, F_init_list, H_init_list, Q, R, Y, x_0, P_0, X_true,max_it=5, generate_f=True, generate_h=True,init_x_list=None, init_P_list=None,  update_F=True, update_H=True):

    N = Y.size(0)
    T = Y.size(-1)
    m = sys_model.m
    n = sys_model.n

    F_hist, H_hist = [], []
    last_x_list, last_P_list = [], []
    smooth_x_list = []  # Store full smoothed trajectories

    # NEW: Track MSE per iteration across all sequences
    mse_per_iter = torch.zeros(max_it, device=Y.device)
    iter_count = torch.zeros(max_it, device=Y.device)

    for j in range(N):
        f_idx = j // 10 if generate_f else j
        h_idx = j // 10 if generate_h else j

        F_est = F_init_list[f_idx].clone()
        H_est = H_init_list[h_idx].clone()

        Y_j = Y[j]
        X_true_j = X_true[j]

        F_all_j = [F_est.clone()]
        H_all_j = [H_est.clone()]

        if init_x_list is not None:
            x0_j = init_x_list[j]
            P0_j = init_P_list[j]
        else:
            x0_j = x_0
            P0_j = P_0

        for q in range(max_it):
            # ---------- E-step ----------
            sys_model.InitSequence(x0_j, P0_j)
            [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, V_smooth] = S_Test(sys_model,Y_j.unsqueeze(0),X_true_j.unsqueeze(0),F=F_est.unsqueeze(0),H=[H_est],generate_f=False,generate_h=False,init_x_list=[x0_j],
                init_P_list=[P0_j],)

            X_s = X_smooth.squeeze(0)    # [m,T]
            P_s = P_smooth.squeeze(0)    # [m,m,T]
            V_s = V_smooth.squeeze(0)    # [m,m,T]

            # NEW: Accumulate MSE for this iteration
            mse_per_iter[q] += _mse_avg
            iter_count[q] += 1

            # ---------- M-step (F) ----------
            if update_F:
                A_1 = compute_A1(x0_j, X_s, V_s, m, T)  # (m,m )
                A_2 = compute_A2(x0_j, P0_j, X_s, P_s, m, T)  # (m,m)
                # Update equation for F: F^(i+1) = A_1^(i) @ inv(A_2^(i))
                eps = 1e-7 * torch.eye(m, device=A_2.device)
                A_2 = A_2 + eps
                F_new = torch.linalg.solve(A_2.T, A_1.T).T
            else:
                F_new = F_est
            # F_new = clamp(F_new)

            # ---------- M-step (H) ----------
            if update_H:
                C1 = torch.zeros((n, m), dtype=Y.dtype, device=Y.device)
                C2 = torch.zeros((m, m), dtype=Y.dtype, device=Y.device)
                for t in range(T):
                    C1 += Y_j[:, t].unsqueeze(1) @ X_s[:, t].unsqueeze(0)
                    C2 += X_s[:, t].unsqueeze(1) @ X_s[:, t].unsqueeze(0) + P_s[:, :, t]

                eps = 1e-6 * torch.eye(m, device=Y.device, dtype=Y.dtype)
                H_new = torch.linalg.solve((C2 + eps).T, C1.T).T
            else:
                H_new = H_est
            # ---------- commit ----------
            F_est = F_new
            H_est = H_new
            F_all_j.append(F_est.clone())
            H_all_j.append(H_est.clone())
            # print(f"Seq {j+1}/{N}, Iter {q+1}/{max_it}, MSE: {_mse_avg:.6f} ({_mse_db:.2f} dB),F_est={F_new}")
        F_hist.append(F_all_j)
        H_hist.append(H_all_j)

        x_T = X_s[:, -1].unsqueeze(-1).clone()
        P_T = P_s[:, :, -1].clone()
        last_x_list.append(x_T)
        last_P_list.append(P_T)
        smooth_x_list.append(X_s.clone())  # Append full smoothed trajectory [m, T]

    # NEW: Compute mean MSE per iteration
    mean_mse_per_iter = mse_per_iter / iter_count

    return F_hist, H_hist, last_x_list, last_P_list, smooth_x_list, mean_mse_per_iter
