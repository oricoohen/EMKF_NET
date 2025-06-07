import torch
import torch.nn as nn
import time
from Smoothers.Linear_KF import KalmanFilter
from Smoothers.RTS_Smoother import rts_smoother



def enforce_covariance_properties( P, eps=1e-6):
    """
    Ensure that the covariance matrix P is positive    definite (PSD).

Args:
    P: A square matrix [m, m] representing the covariance matrix.
    eps: Small constant to ensure positive semi-definiteness if necessary.

Returns:
    P: Adjusted covariance matrix that is PSD.
"""
    # Check if P is positive semi-definite
    # We will check the eigenvalues to determine if all are >= 0
    P = (P + P.T) / 2  # Ensure P is symmetric
    # Compute eigenvalues and eigenvectors. Use torch.linalg.eigh for symmetric tensors.
    eigenvalues, eigenvectors = torch.linalg.eigh(P)
    if torch.any(eigenvalues.real < 0):  # If there are negative eigenvalues
        # Clamp eigenvalues to ensure they are at least eps.
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        # Reconstruct the matrix
        P = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

    return P


def compute_cross_covariances( F, H, Ks, Ps, SGains):
    """
    Compute the cross-covariance matrices for RTS smoothing.

    Args:
        F:      Tensor [m, m], state transition matrix
        H:      Tensor [n, m], measurement matrix
        Ks:     Tensor [m, n], Kalman gain at last step
        Ps:     Tensor [m, m, T], filtered covariances over T time steps
        SGains: List of length T-1 of [m, m] tensors, smoother gains

    Returns:
        V: Tensor [m, m, T], where V[:,:,t] = Cov(x_t, x_{t-1}|Y)
    """
    m, _, T = Ps.shape
    # Preallocate output tensor
    V = torch.zeros((m, m, T), dtype=Ps.dtype, device=Ps.device)

    # Identity matrix for dimension m
    I = torch.eye(m, dtype=Ps.dtype, device=Ps.device)
    # Cross-covariance at final time step (T-1)
    V[:, :, T - 1] = (I - Ks @ H) @ F @ Ps[:, :, T - 2]
    V[:, :, T - 1] = enforce_covariance_properties(V[:, :, T - 1])

    # Backward recursion from t = T-2 down to 0
    for t in range(T - 2, -1, -1):
        Pt = Ps[:, :, t]
        St = SGains[T - 2 - t]
        Stm1_T = SGains[T - 1 - t]
        V[:, :, t] = (Pt @ Stm1_T.T + St @ (V[:, :, t + 1] - F @ Pt) @ Stm1_T.T)
        V[:, :, t] = enforce_covariance_properties(V[:, :, t])

    return V



def S_Test(SysModel, test_input, test_target,F=None, allStates=True, randomInit = False,test_init=None,K_T_list=None):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    T = test_input[0].size(-1)
    m = SysModel.m
    N_T = len(test_input)
    MSE_RTS_linear_arr = torch.empty(N_T)
    start = time.time()
    KF = KalmanFilter(SysModel)
    RTS = rts_smoother(SysModel)
    RTS_out = torch.zeros(N_T, m, T)
    P_smooth = torch.zeros(N_T, m, m, T)
    V_test = None
    if K_T_list is not None:
        V_test = torch.zeros(N_T, m, m, T)



    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2:
            loc = torch.tensor([True,False]) # for position only


    # mask = torch.tensor([True,True,True,False,False,False])# for kitti

    for j,(sequence_target,sequence_input) in enumerate(zip(test_target,test_input)):
        if F is not None:
            F_index = j//10
            SysModel.F = F[F_index]
            SysModel.F_T = SysModel.F.T
            KF.F = F[F_index]
            KF.F_T = F[F_index].T
            RTS.F = F[F_index]
            RTS.F_T = F[F_index].T


        if(randomInit):
            KF.InitSequence(torch.unsqueeze(test_init[j,:],1), SysModel.m2x_0)  
        else:
            KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)   
            
        KF.GenerateSequence(sequence_input, sequence_input.size()[-1])
        RTS.GenerateSequence(KF.x, KF.sigma, sequence_input.size()[-1])

        
        if(allStates):
            MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x, sequence_target).item()
        else:           
            MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x[loc,:], sequence_target[loc,:]).item()
        RTS_out[j] =RTS.s_x

        P_smooth[j] = RTS.s_sigma.clone()
        SGains = RTS.SGains
        if K_T_list is not None: #####JUST IF THEY GAVE US K WE CAN COMPUTE THE V
            V_now = compute_cross_covariances(SysModel.F, SysModel.H, K_T_list[j], P_smooth[j], SGains)
            V_test[j] = V_now




    end = time.time()
    t = end - start

    # Average
    MSE_RTS_linear_avg = torch.mean(MSE_RTS_linear_arr)
    MSE_RTS_dB_avg = 10 * torch.log10(MSE_RTS_linear_avg)

    # Standard deviation
    MSE_RTS_linear_std = torch.std(MSE_RTS_linear_arr, unbiased=True)

    # Confidence interval
    RTS_std_dB = 10 * torch.log10(MSE_RTS_linear_std + MSE_RTS_linear_avg) - MSE_RTS_dB_avg


    print("RTS Smoother - MSE LOSS:", MSE_RTS_dB_avg, "[dB]")
    print("RTS Smoother - STD:", RTS_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg ,RTS_out,P_smooth,V_test]



