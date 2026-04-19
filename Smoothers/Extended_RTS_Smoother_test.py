import torch
import torch.nn as nn
import time
from Smoothers.EKF import ExtendedKalmanFilter
from Smoothers.Extended_RTS_Smoother import Extended_rts_smoother

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def compute_cross_covariances(F, H_last, K_last, P_filt, Sgains):
    """
    Cross-covariances V[:,:,t] = Cov(x_t, x_{t-1} | Y).
    Uses only the LAST measurement Jacobian/matrix (H_last).

    Args:
        F:       [m,m]
        H_last:  [n,m]  (Jacobian at t = T-1; for linear case it's just H)
        K_last:  [m,n]  (KF gain at t = T-1)
        P_filt:  [m,m,T] filtered P_t|t (pre-smoothing)
        Sgains:  list of length T-1, each [m,m], smoother gains

    Returns:
        V: [m,m,T]
    """
    m, _, T = P_filt.shape
    V = torch.zeros((m, m, T), dtype=P_filt.dtype, device=P_filt.device)
    I = torch.eye(m, dtype=P_filt.dtype, device=P_filt.device)

    # init at t = T-1
    V[:, :, T - 1] = (I - K_last @ H_last) @ F @ P_filt[:, :, T - 2]

    # same indexing you used in the linear version
    for t in range(T - 2, -1, -1):
        Pt = P_filt[:, :, t]
        St = Sgains[T - 2 - t]
        Stm1_T = Sgains[T - 1 - t]
        V[:, :, t] = Pt @ Stm1_T.T + St @ (V[:, :, t + 1] - F @ Pt) @ Stm1_T.T

    return V


def S_Test_ext(args, SysModel, test_input, test_target, randomInit = False,test_init=None):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_ERTS_linear_arr = torch.empty(args.N_T,device=device)
    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel)
    ERTS = Extended_rts_smoother(SysModel)
    # Allocate empty list for output
    ERTS_out = []
    j=0

    for sequence_target,sequence_input in zip(test_target,test_input):
        if(randomInit):
            EKF.InitSequence(torch.unsqueeze(test_init[j,:],1), SysModel.m2x_0)
        else:
            EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

        EKF.GenerateSequence(sequence_input, sequence_input.size()[-1])
        ERTS.GenerateSequence(EKF.x, EKF.sigma, sequence_input.size()[-1])
        MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x, sequence_target).item()
        ERTS_out.append(ERTS.s_x)
        j=j+1
    end = time.time()
    t = end - start
    MSE_ERTS_linear_avg = torch.mean(MSE_ERTS_linear_arr)
    MSE_ERTS_dB_avg = 10 * torch.log10(MSE_ERTS_linear_avg)

    # Standard deviation
    MSE_ERTS_linear_std = torch.std(MSE_ERTS_linear_arr, unbiased=True)

    # Confidence interval
    ERTS_std_dB = 10 * torch.log10(MSE_ERTS_linear_std + MSE_ERTS_linear_avg) - MSE_ERTS_dB_avg

    print("Extended RTS Smoother - MSE LOSS:", MSE_ERTS_dB_avg, "[dB]")
    print("Extended RTS Smoother - STD:", ERTS_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out]

def S_Test_ext_old(SysModel, test_input, test_target, F_list, generate_f=True, randomInit=False, test_init=None,
               init_x_list=None, init_P_list=None):
    N_T = len(test_input)
    m = SysModel.m
    n = SysModel.n
    T = test_input[0].size(-1)  # or: SysModel.T or SysModel.T_test
    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_ERTS_linear_arr = torch.empty(N_T, device=test_target[0].device)
    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel)
    ERTS = Extended_rts_smoother(SysModel)

    ERTS_out = torch.zeros(N_T, m, T, dtype=test_target[0].dtype, device=test_target[0].device)
    P_smooth = torch.zeros(N_T, m, m, T, dtype=test_target[0].dtype, device=test_target[0].device)
    P_filt = torch.zeros(N_T, m, m, T, dtype=test_target[0].dtype, device=test_target[0].device)  # pre-smoothing
    V_test = torch.zeros(N_T, m, m, T, dtype=test_target[0].dtype, device=test_target[0].device)
    last_gain = torch.empty(N_T, m, n, dtype=test_target[0].dtype, device=test_target[0].device)

    for j, (sequence_target, sequence_input) in enumerate(zip(test_target, test_input)):

        if generate_f == True:
            F_index = j // 10
            SysModel.F = F_list[F_index]
            SysModel.F_T = SysModel.F.T
            SysModel.update_f(F_list[F_index])
            EKF.f = SysModel.f
            ERTS.f = SysModel.f

        else:
            SysModel.F = F_list[j]
            SysModel.F_T = F_list[j].T
            SysModel.update_f(F_list[j])
            EKF.f = SysModel.f
            ERTS.f = SysModel.f

        if (randomInit):
            EKF.InitSequence(torch.unsqueeze(test_init[j, :], 1), SysModel.m2x_0)
        else:
            if init_x_list is not None:
                SysModel.m1x_0 = init_x_list[j]
                SysModel.m2x_0 = init_P_list[j]
            EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

        EKF.GenerateSequence(sequence_input, sequence_input.size()[-1])

        # store filtered
        P_filt[j] = EKF.sigma.clone()  # P_t|t
        last_gain[j] = EKF.KG_array[-1].clone()  # K at t = T-1

        ERTS.GenerateSequence(EKF.x, EKF.sigma, sequence_input.size()[-1])

        # outputs
        ERTS_out[j] = ERTS.s_x.clone()
        P_smooth[j] = ERTS.s_sigma.clone()

        # V_test with last H and last K
        H_last = EKF.H_last
        V_now = compute_cross_covariances(SysModel.F, H_last, last_gain[j], P_filt[j], ERTS.SGains)
        V_test[j] = V_now

        MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x, sequence_target).item()

    end = time.time()
    t = end - start
    MSE_ERTS_linear_avg = torch.mean(MSE_ERTS_linear_arr)
    MSE_ERTS_dB_avg = 10 * torch.log10(MSE_ERTS_linear_avg)

    # Standard deviation
    MSE_ERTS_linear_std = torch.std(MSE_ERTS_linear_arr, unbiased=True)

    # Confidence interval
    ERTS_std_dB = 10 * torch.log10(MSE_ERTS_linear_std + MSE_ERTS_linear_avg) - MSE_ERTS_dB_avg

    print("Extended RTS Smoother - MSE LOSS:", MSE_ERTS_dB_avg, "[dB]")
    print("Extended RTS Smoother - STD:", ERTS_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out, P_smooth, V_test]


#######for linear h non linear f
def S_Test_ext_H(SysModel, test_input, test_target, H_list, generate_h=False,
                 randomInit=False, test_init=None, init_x_list=None, init_P_list=None):
    N_T = len(test_input)
    m = SysModel.m
    n = SysModel.n
    T = test_input[0].size(-1)  # or: SysModel.T or SysModel.T_test
    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_ERTS_linear_arr = torch.empty(N_T, device=test_target[0].device)
    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel)
    ERTS = Extended_rts_smoother(SysModel)

    ERTS_out = torch.zeros(N_T, m, T, dtype=test_target[0].dtype, device=test_target[0].device)
    P_smooth = torch.zeros(N_T, m, m, T, dtype=test_target[0].dtype, device=test_target[0].device)
    P_filt = torch.zeros(N_T, m, m, T, dtype=test_target[0].dtype, device=test_target[0].device)  # pre-smoothing
    V_test = torch.zeros(N_T, m, m, T, dtype=test_target[0].dtype, device=test_target[0].device)
    last_gain = torch.empty(N_T, m, n, dtype=test_target[0].dtype, device=test_target[0].device)

    for j, (sequence_target, sequence_input) in enumerate(zip(test_target, test_input)):

        if generate_h == True:
            H_index = j // 10
            H_curr = H_list[H_index]
        else:
            H_curr = H_list[j]

        SysModel.H = H_curr
        SysModel.update_h(H_curr)
        EKF.h = SysModel.h
        ERTS.h = SysModel.h

        if randomInit:
            EKF.InitSequence(torch.unsqueeze(test_init[j, :], 1), SysModel.m2x_0)
        else:
            if init_x_list is not None:
                SysModel.m1x_0 = init_x_list[j]
            if init_P_list is not None:
                SysModel.m2x_0 = init_P_list[j]
            EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

        EKF.GenerateSequence(sequence_input, sequence_input.size()[-1])

        # store filtered
        P_filt[j] = EKF.sigma.clone()  # P_t|t
        last_gain[j] = EKF.KG_array[-1].clone()  # K at t = T-1

        ERTS.GenerateSequence(EKF.x, EKF.sigma, sequence_input.size()[-1])

        # outputs
        ERTS_out[j] = ERTS.s_x.clone()
        P_smooth[j] = ERTS.s_sigma.clone()

        # V_test with last H and last K
        H_last = EKF.H_last
        # V_now = compute_cross_covariances(SysModel.F, H_last, last_gain[j], P_filt[j], ERTS.SGains) #return if F is linear
        # V_test[j] = V_now

        MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x, sequence_target).item()

    end = time.time()
    t = end - start
    MSE_ERTS_linear_avg = torch.mean(MSE_ERTS_linear_arr)
    MSE_ERTS_dB_avg = 10 * torch.log10(MSE_ERTS_linear_avg)

    # Standard deviation
    MSE_ERTS_linear_std = torch.std(MSE_ERTS_linear_arr, unbiased=True)

    # Confidence interval
    ERTS_std_dB = 10 * torch.log10(MSE_ERTS_linear_std + MSE_ERTS_linear_avg) - MSE_ERTS_dB_avg

    # print("Extended RTS Smoother - MSE LOSS:", MSE_ERTS_dB_avg, "[dB]")
    # print("Extended RTS Smoother - STD:", ERTS_std_dB, "[dB]")
    # # Print Run Time
    # print("Inference Time:", t)

    return [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out, P_smooth, V_test]