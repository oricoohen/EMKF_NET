import torch
import torch.nn as nn
import time
from Smoothers.Linear_KF import KalmanFilter

def KFTest(args, SysModel, test_input, test_target,F =None, allStates=True, randomInit = False, test_init=None):
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(args.N_T)
    start = time.time()
    KF = KalmanFilter(SysModel)

    last_gains = torch.empty(args.N_T, SysModel.m, SysModel.n)

    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only


    for j,(sequence_target,sequence_input) in enumerate(zip(test_target,test_input)):
        if F is not None:
            F_index = j//10
            SysModel.F = F[F_index]
            KF.F = F[F_index]
            KF.F_T = F[F_index].T

        if(randomInit):
            KF.InitSequence(torch.unsqueeze(test_init[j,:],1), SysModel.m2x_0)        
        else:
            KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
            
        KF.GenerateSequence(sequence_input, sequence_input.size()[-1])



        MSE_KF_linear_arr[j] = loss_fn(KF.x, sequence_target).item()



        #    KF.K should have shape (m, n)
        last_gains[j] = KF.KG.clone()





    end = time.time()
    t = end - start
    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]



