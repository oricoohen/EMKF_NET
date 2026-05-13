"""
This file contains the class Pipeline_ERTS,
which is used to train and test RTSNet in both linear and non-linear cases.
"""
import copy

import torch
import torch.nn as nn
import time
import random
from Plot import Plot_extended as Plot
from RTSNet.PsmoothNN import PsmoothNN  # Ensure the PsmoothNN class is correctly imported
from RTSNet.PsmoothNN_combined import PsmoothFromPnot, PNotSmoothNN
import torch.nn as nn
from emkf.main_emkf_func_AI import EMKF_F_Mstep
from Simulations.Lorenz_Atractor.parameters import getJacobian
from emkf.AI_M_step_for_f import DeltaF_MStepNet
from emkf.AI_M_step_for_h import DeltaH_MStepNet
import math
from Smoothers.Extended_RTS_Smoother_test import S_Test_ext_H
import os
device =torch.device("cuda")

# thresholds you can tweak
_JAC_MAXABS_THRESH = 1e3      # entries bigger than this are "large"
_JAC_COND_THRESH   = 1e6      # rough condition-number alarm
_R_FLOOR_WARN      = 1e-3     # "small radius" alarm


def enforce_covariance_properties(P, eps=1e-6):
    # Ensure P is symmetric positive semidefinite
    P = (P + P.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(P)
    if torch.any(eigenvalues.real < 0):
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        P = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return P


def _jacobian_watchdog(H: torch.Tensor, x: torch.Tensor, h_fn, tag: str = ""):
    """
    H: [n,m], x: [m,1], h_fn(x)->[n,1]
    Prints ONLY when something looks bad.
    """
    with torch.no_grad():
        nfin = not torch.isfinite(H).all()
        maxabs = H.abs().max().item()
        fro = H.norm().item()
        try:
            svals = torch.linalg.svdvals(H)
            smin = svals.min().item()
            smax = svals.max().item()
            cond = (smax / max(smin, 1e-12))
        except Exception:
            smin = smax = 0.0
            cond = float("inf")

        # also look at measurement radius r from your h: y=[r, theta]
        try:
            y = h_fn(x).view(-1)
            r = float(y[0].abs().item())
        except Exception:
            r = float("nan")

        flags = []
        if nfin: flags.append("non-finite")
        if maxabs > _JAC_MAXABS_THRESH: flags.append("large|H|")
        if cond   > _JAC_COND_THRESH:   flags.append("ill-cond(H)")
        if r != r or r < _R_FLOOR_WARN: flags.append(f"small-r({r:.2e})")  # r!=r catches NaN

        if flags:
            print(f"[JAC] {tag} flags={','.join(flags)} "
                  f"max|H|={maxabs:.2e} ||H||F={fro:.2e} cond≈{cond:.2e} r={r:.2e}")
            # optional: show the worst offending row/col once
            if maxabs > _JAC_MAXABS_THRESH:
                i,j = torch.where(H.abs() == H.abs().max())
                print(f"[JAC] worst H[{int(i[0])},{int(j[0])}]={H[i[0],j[0]].item():.3e}")

def normalize_mstep_statistics(A1, A2, S_delta_x, S_nu, C_delta_x_xminus, debug=False, seq_id=None):
    """
    Normalize M-step statistics to be scale-invariant.

    Args:
        A1: [m, m] cross-moment matrix
        A2: [m, m] auto-moment matrix
        S_delta_x: [m, m] innovation covariance matrix
        S_nu: [n, n] observation noise covariance matrix
        C_delta_x_xminus: [m, m] cross-covariance
        debug: whether to print debug info
        seq_id: sequence ID for debug printing

    Returns:
        A1_normalized, A2_normalized, S_delta_x_normalized, S_nu_normalized,
        C_delta_normalized, A2_scale, S_nu_scale
    """
    # Compute state magnitude scale (proportional to ||x||²)
    A2_scale = torch.diagonal(A2).mean()  # Scalar
    A2_scale_safe = torch.clamp(A2_scale, min=1e-6)  # Avoid division by zero

    # Compute observation noise scale
    S_nu_scale = torch.diagonal(S_nu).mean()  # Scalar
    S_nu_scale_safe = torch.clamp(S_nu_scale, min=1e-6)

    # Normalize by respective scales
    A1_normalized = A1 / A2_scale_safe
    A2_normalized = A2 / A2_scale_safe
    S_delta_x_normalized = S_delta_x / A2_scale_safe  # Also scales with ||x||²!
    S_nu_normalized = S_nu / S_nu_scale_safe  # Normalize by its own scale
    C_delta_normalized = C_delta_x_xminus / A2_scale_safe

    # Debug output
    if debug and (seq_id is None or seq_id % 10 == 0):
        print(f"\n[NORM DEBUG] Seq {seq_id if seq_id is not None else '?'}:")
        print(f"  A2_scale (state) = {A2_scale.item():.6f}")
        print(f"  S_nu_scale (obs) = {S_nu_scale.item():.6f}")
        print(f"  A1 range: [{A1.min().item():.4f}, {A1.max().item():.4f}]")
        print(f"  A1_norm range: [{A1_normalized.min().item():.4f}, {A1_normalized.max().item():.4f}]")
        print(f"  S_delta_x range: [{S_delta_x.min().item():.4f}, {S_delta_x.max().item():.4f}]")
        print(f"  S_delta_x_norm range: [{S_delta_x_normalized.min().item():.4f}, {S_delta_x_normalized.max().item():.4f}]")
        print(f"  A2_norm diag mean: {torch.diagonal(A2_normalized).mean().item():.6f}")  # Should be ~1.0
        print(f"  S_delta_x_norm diag mean: {torch.diagonal(S_delta_x_normalized).mean().item():.6f}")
        print(f"  S_nu_norm diag mean: {torch.diagonal(S_nu_normalized).mean().item():.6f}")  # Should be ~1.0

    return A1_normalized, A2_normalized, S_delta_x_normalized, S_nu_normalized, C_delta_normalized, A2_scale, S_nu_scale

def _rho(F):
    try:
        return torch.linalg.eigvals(F).abs().max().real.item()
    except Exception:
        return float("inf")

def _cond_sym(A):
    try:
        s = torch.linalg.svdvals(A)
        return (s.max() / torch.clamp(s.min(), 1e-12)).item()
    except Exception:
        return float("inf")





class Pipeline_ERTS:


    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.SysModel = ssModel #the dinamic system model contains the F, Q, H, R, T, T_test

    def setModel(self, model,args):
        self.args = args
        self.model = model # the RTSNet model contains the parameters of the RTSNet
        # Initialize PsmoothNN
        self.PsmoothNN = PsmoothNN(self.SysModel.m, self.args)
        self.model.to(self.device)
        self.PsmoothNN.to(self.device)

        self.PNotSmoothNN = PNotSmoothNN(self.SysModel.m, self.SysModel.n, self.SysModel.m2x_0.clone().detach())
        self.PsmoothFromPnot = PsmoothFromPnot(self.SysModel.m)

        # M-step network that outputs ΔF from statistics z_in
        self.M_model = DeltaF_MStepNet(self.SysModel.m, self.SysModel.n).to(self.device)
        self.M_model_H = DeltaH_MStepNet(self.SysModel.m, self.SysModel.n).to(self.device)

    def setTrainingParams(self, args, alpha=0.5, b=0.5):
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch  # Number of Samples in Batch
        self.learningRate = args.lr  # Learning Rate
        self.weightDecay = args.wd  # L2 Weight Regularization - Weight Decay
        self.alpha = alpha  # Composition loss factor
        self.b = b  # Weight factor between main loss and P-smooth loss
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')



        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.9, patience=20)

        # ori add p smoothed
        # Optimizer for PsmoothNN
        self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
                                                    weight_decay=self.weightDecay)
        # P-smooth loss weight
        self.p_smooth_weight = 0.1  # Default weight for p-smooth loss

        self.M_optimizer = torch.optim.Adam(self.M_model.parameters(),lr=self.learningRate,weight_decay=self.weightDecay)
        self.M_optimizer_H = torch.optim.Adam(self.M_model_H.parameters(),lr=self.learningRate, weight_decay=self.weightDecay)

    def P_smooth_Train(self,SysModel, cv_input, cv_target, train_input, train_target, path_results,path_rtsnet=None,load_psmooth_path = None, generate_f=True,generate_h=False):

        '''train P-smooth network with RTSNet fixed. dont change the RTSNet'''

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.model = torch.load(path_rtsnet, map_location=self.device, weights_only=False)  # Load the best RTSNet model
        self.model.to(self.device).eval() # Freeze RTSNet if needed, so it doesn't change
        if load_psmooth_path != None:
            self.PsmoothNN = torch.load(load_psmooth_path,  map_location=self.device,weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
                                                        weight_decay=self.weightDecay)
        self.PsmoothNN.to(self.device).train()  # Set P-smooth network to train mode
        # Preallocate arrays for logging training performance

        self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_idx_opt = 0
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)
        ##############
        ### Epochs ###
        ##############


        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Zero gradients for both optimizers
            self.PsmoothNN_optimizer.zero_grad()
            self.PsmoothNN.train()
            Batch_Psmooth_LOSS_sum = 0

            for j in range(0, self.N_B):

                n_e = random.randint(0, self.N_E - 1)
                if generate_f is True:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                    # self.PsmoothNN.update_F(SysModel.F)
                if generate_h is True:  ####if we train with different h
                    index = n_e // 10
                    SysModel.H = SysModel.H_train[index]
                    self.model.update_H(SysModel.H)
                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                self.model.init_hidden()
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)

                x_out_training_forward = torch.empty(SysModel.m, SysModel.T,
                                     device=y_training.device, dtype=y_training.dtype)
                x_out_training = torch.empty(SysModel.m, SysModel.T,
                                     device=y_training.device, dtype=y_training.dtype)
                ########add changes to compute P and S
                #####compute P ori
                self.model.sigma_list = []  # is added in every step_KGain_est(self, y) [1, 1, m²]
                self.model.smoother_gain_list = []  # is added in every step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2)
                for t in range(0, SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                    P_forward = self.model.h_Sigma.clone().detach()  # [1, 1, m²]
                    self.model.sigma_list.append(P_forward)  # [1, 1, m²]
                # -------- RTSNet Backward Pass: Compute smoother gains and smoothed states --------
                # Start backward smoothing: initialize using last forward estimate
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:,SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T - 1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
                                                               x_out_training_forward[:, SysModel.T - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # ori save the T-1 sgain
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],
                                                      x_out_training_forward[:, t + 1], x_out_training[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # Save detached copy ori shape[m, m]
                # ---- Handle initial smoothed P at time T ----
                P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T,
                             device=self.device)
                dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m,
                             device=self.device)  # shape: [1, 1, m²] input to PsmoothNN
                sigma_T = self.model.sigma_list[-1] # shape: [1, 1, m²] input to PsmoothNN
                self.PsmoothNN.start = 0
                ####compute the P(T)
                P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)# shape: [1, 1, m²] to [m²]
                P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(SysModel.m,SysModel.m))# shape: [m, m]
                P_smoothed_seq[:, :, SysModel.T - 1] = P_matrix  # shape: [m, m]
                # ---- Loop over t = 1 to T for learning P_smooth ----
                for t in range(SysModel.T - 2, -1, -1):  # Loop from T-2 down to 0
                    sigma_t = self.model.sigma_list[t].view(1, 1, -1)  # sigma_t: shape [1, 1, m²]
                    # Compute the proper index for smoother_gain_list
                    index = (SysModel.T - 2) - t
                    sgain_t = self.model.smoother_gain_list[index].reshape(1, 1, -1)  # Now sgain_t: [1, 1, m²]
                    # Forward pass through PsmoothNN
                    P_flat = self.PsmoothNN(sigma_t, sgain_t)  # [1, 1, m²]
                    # Enforce PSD properties
                    P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(-1).view(SysModel.m,SysModel.m))  # [m, m]
                    # Save result in the sequence tensor
                    P_smoothed_seq[:, :, t] = P_matrix  # [ m, m]


                # Compute P-smooth loss using PsmoothNN's compute_loss method
                # Detach x_out_training to prevent gradient flow to RTSNet
                #oprion 1
                psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training.detach())
                #option 2
                #psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq, train_target[n_e], x_out_training.detach())

                # Accumulate losses
                Batch_Psmooth_LOSS_sum += psmooth_loss

            # Average losses for this batch
            Batch_Psmooth_LOSS_mean = Batch_Psmooth_LOSS_sum / self.N_B
            MSE_train_psmooth_batch = Batch_Psmooth_LOSS_mean


            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_psmooth_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])



            # Then train PsmoothNN
            Batch_Psmooth_LOSS_mean.backward()
            # right after Batch_Psmooth_LOSS_mean.backward()
            total_grad = 0.0
            # for p in self.PsmoothNN.parameters():
            #     if p.grad is not None:
            #         total_grad += p.grad.norm().item()
            # print(f"Epoch {ti:03d} – gradient L2-norm on PsmoothNN = {total_grad:.4e}")
            self.PsmoothNN_optimizer.step()


            ##################
            ### Optimizing ###
            ##################

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.PsmoothNN.eval()  # Set PsmoothNN to eval mode
            with torch.no_grad():
                MSE_cv_psmooth_batch = torch.empty([self.N_CV], device=self.device)

                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]
                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test,
                               device=y_cv.device, dtype=y_cv.dtype)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test,
                               device=y_cv.device, dtype=y_cv.dtype)

                    if generate_f is True:  ####if we valid with different f
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)
                        # self.PsmoothNN.update_F(SysModel.F)
                    if generate_h is True:  ####if we valid with different h
                        index = j // 10
                        SysModel.H = SysModel.H_valid[index]
                        self.model.update_H(SysModel.H)

                    self.model.init_hidden()
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T)

                    # Forward pass and compute P-smooth
                    # Initialize lists to store intermediate values
                    self.model.sigma_list = []  # List of [1, 1, m²] tensors for each time step
                    self.model.smoother_gain_list = []  # List of [m, m] tensors for each time step

                    # Forward pass through RTSNet
                    for t in range(0, SysModel.T_test):
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                        P_cv_forward = self.model.h_Sigma.clone().detach()
                        self.model.sigma_list.append(P_cv_forward)  # [1, 1, m²]
                    # Initialize backward pass
                    x_out_cv[:, SysModel.T_test-1] = x_out_cv_forward[:, SysModel.T_test-1]  # [m]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test-1])
                    # First backward step
                    x_out_cv[:, SysModel.T_test-2] = self.model(None, x_out_cv_forward[:, SysModel.T_test-2], x_out_cv_forward[:, SysModel.T_test-1],None)  # [m]
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # [m, m]
                    # Remaining backward steps
                    for t in range(SysModel.T_test-3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t+1],x_out_cv[:, t+2])  # [m]
                        self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # [m, m]

                    # Initialize P-smooth sequence tensor
                    P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T_test,
                             device=sigma_T.device, dtype=sigma_T.dtype)  # [m, m, T_test]
                    dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m,
                             device=sigma_T.device, dtype=sigma_T.dtype)  # shape: [1, 1, m²] input to PsmoothNN
                    sigma_T = self.model.sigma_list[-1]  # shape: [1, 1, m²] input to PsmoothNN
                    self.PsmoothNN.start = 0
                    # Handle initial P-smooth at time T_test
                    P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
                    P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(SysModel.m, SysModel.m))  # shape: [m, m]
                    P_smoothed_seq[:, :, SysModel.T_test - 1] = P_matrix  # shape: [m, m]
                    # Compute P-smooth for remaining time steps
                    for t in range(SysModel.T_test - 2, -1, -1):
                        sigma_t = self.model.sigma_list[t].view(1, 1, -1)  # [1, 1, m²]
                        index = (SysModel.T_test - 2) - t
                        sgain_t = self.model.smoother_gain_list[index].reshape(1, 1, -1)  # [1, 1, m²]
                        P_flat = self.PsmoothNN(sigma_t, sgain_t)  # [1, 1, m²]
                        P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(-1).view(SysModel.m,SysModel.m))  # [m, m]
                        P_smoothed_seq[:, :, t] = P_matrix  # [m, m]

                    # Compute P-smooth validation loss
                    #option 1
                    MSE_cv_psmooth_batch[j] = self.PsmoothNN.compute_loss(P_smoothed_seq, cv_target[j], x_out_cv)  # Scalar
                    #option 2
                    #MSE_cv_psmooth_batch[j] = self.compute_gaussian_loss1(P_smoothed_seq, cv_target[j], x_out_cv)
                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_psmooth_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.PsmoothNN, path_results)


            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE P_smoothe Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE P_smoothe Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return  [self.MSE_train_dB_epoch[ti],self.MSE_cv_dB_epoch[ti]]

    def P_joint_Train(self, SysModel, cv_input, cv_target, train_input, train_target, path_results_pnot, path_results_psfp, path_rtsnet=None, load_pnot_path=None, load_psfp_path=None, generate_f=True, generate_h=False):
        """
        Joint training (equal weights) for PNotSmoothNN and PsmoothNN while RTSNet is frozen.
        - Uses RTSNet only to produce: filtered states x_fwd, filtered covariances (h_Sigma),
          Kalman gains (KGain), and smoother gains (SGain) via its backward pass.
        - Trains both covariance nets together with a single Adam optimizer:
            Loss_total = 0.5 * L_not + 0.5 * L_smooth
          where L_not compares P_not_t to (x_true - x_fwd)(x_true - x_fwd)^T,
                L_smooth compares P_smooth_t to (x_true - x_smooth)(x_true - x_smooth)^T.
        - Style is kept identical to your P_smooth_Train: epoch/batch loops, prints, CV save.
        """

        # ---------- Setup ----------
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        # Load RTSNet and freeze it: eval() for deterministic layers; requires_grad_(False) to avoid autograd graph
        self.model = torch.load(path_rtsnet, map_location=self.device, weights_only=False).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Optionally load existing checkpoints for the two covariance networks
        if load_pnot_path is not None:
            print("loading model_and keep training them")
            self.PNotSmoothNN = torch.load(load_pnot_path, map_location=self.device, weights_only=False)
        if load_psfp_path  is not None:
            self.PsmoothFromPnot = torch.load(load_psfp_path, map_location=self.device, weights_only=False)

        # Put both nets in train mode (we are learning their parameters)
        self.PNotSmoothNN.to(self.device).train()
        self.PsmoothFromPnot.to(self.device).train()

        # Single optimizer over both nets (keeps latency/overhead low)
        self.Joint_optimizer = torch.optim.Adam(list(self.PNotSmoothNN.parameters()) + list(self.PsmoothFromPnot.parameters()),
            lr=self.learningRate, weight_decay=self.weightDecay)

        # Allocate logging buffers (same style as your function)
        self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_idx_opt = 0
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_pnot_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_psmooth_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_pnot_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_psmooth_dB_epoch = torch.empty([self.N_steps], device=self.device)

        self.CV_pnot_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.CV_psmooth_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.CV_pnot_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.CV_psmooth_dB_epoch = torch.empty([self.N_steps], device=self.device)

        # ---------- Training Loop ----------
        for ti in range(self.N_steps):

            self.Joint_optimizer.zero_grad()
            Batch_LOSS_sum = 0.0
            # Put both nets in train mode (we are learning their parameters)
            self.PNotSmoothNN.train()
            self.PsmoothFromPnot.train()
            sum_Lnot = 0.0
            sum_Lsmooth = 0.0
            for _ in range(self.N_B):

                # Sample a training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_training = train_input[n_e]  # [n, T]
                x_true = train_target[n_e]  # [m, T]
                SysModel.T = y_training.size(-1)
                m, n = SysModel.m, SysModel.n

                # (Optional) vary F across sequences (as in your code)
                if generate_f is True:
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                    self.PNotSmoothNN.F = SysModel.F
                if generate_h is True:
                    index = n_e // 10
                    SysModel.H = SysModel.H_train[index]
                    self.model.update_H(SysModel.H)

                # --------- RTSNet Forward (collect x_fwd, sigma_list, K_gain) ---------

                self.model.init_hidden()
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                x_out_training_forward = torch.empty(m, SysModel.T, device=self.device,dtype=y_training.dtype )
                x_out_training = torch.empty(m, SysModel.T,device=self.device, dtype=y_training.dtype)
                kgain_list = []
                self.model.smoother_gain_list = []

                for t in range(SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                    kgain_list.append(self.model.KGain.clone().detach())
                # --------- RTSNet Backward (x_smooth, S_gain) ---------
                # Your RTSNet stores SGain during the backward calls; it is appended in reverse time
                # (T-2, T-3, ..., 0). We reverse that list to align sgain_list[t] with time t.
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T - 1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
                                                               x_out_training_forward[:, SysModel.T - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # ori save the T-1 sgain
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],
                                                      x_out_training_forward[:, t + 1], x_out_training[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())  # Save detached copy ori shape[m, m]

                # --------- Compute P_not_smooth sequence ---------
                # PNotSmoothNN input at each time is (F_t, K_t, P_prev).
                # We roll it sequentially, seeding with p_0 and feeding its previous prediction.
                self.PNotSmoothNN.reset_state()
                P_prev = self.PNotSmoothNN.p_0
                P_not_seq = torch.empty(m, m, SysModel.T, device=self.device)
                for t in range(SysModel.T):
                    K_t = kgain_list[t]
                    P_t = self.PNotSmoothNN(K_t, P_prev)
                    P_not_seq[:, :, t] = P_t
                    P_prev = P_t.detach()  # stop grads across time for stability

                # --------- Compute P_smooth sequence ---------
                # PsmoothNN takes (P_not_t, SGain_t). At the last time step there is no SGain,
                # so we use a zero matrix as a simple convention.
                self.PsmoothFromPnot.reset_state()
                P_smooth_seq = torch.empty_like(P_not_seq)
                P_smooth_seq[:, :, SysModel.T - 1] = P_not_seq[:, :, SysModel.T - 1]
                for t in range(SysModel.T - 2, -1, -1):  # Loop from T-2 down to 0
                    index = (SysModel.T - 2) - t
                    sgain_t = self.model.smoother_gain_list[index]  # Now sgain_t: [1, 1, m²]
                    P_smooth_seq[:, :, t] = self.PsmoothFromPnot(P_not_seq[:, :, t], sgain_t)

                # --------- Compute Losses ---------
                # L_not: match P_not_t to empirical error covariance of (x_true - x_fwd)
                # L_smooth: match P_smooth_t to empirical error covariance of (x_true - x_smooth)
                L_not = 0.0
                L_smooth = 0.0
                for t in range(SysModel.T):
                    err_not = (x_true[:, t] - x_out_training_forward[:, t]).unsqueeze(1)  # [m,1]
                    P_true_not = err_not @ err_not.T  # [m,m]
                    P_pred_not = P_not_seq[:, :, t]  # [m,m]
                    L_not = L_not + torch.norm(P_pred_not - P_true_not, p='fro') ** 2

                    err_sm = (x_true[:, t] - x_out_training[:, t]).unsqueeze(1)  # [m,1]
                    P_true_sm = err_sm @ err_sm.T  # [m,m]
                    P_pred_sm = P_smooth_seq[:, :, t]  # [m,m]
                    L_smooth = L_smooth + torch.norm(P_pred_sm - P_true_sm, p='fro') ** 2

                L_not = L_not / SysModel.T
                L_smooth = L_smooth / SysModel.T
                Loss_total = 0.5 * L_not + 0.5 * L_smooth  # equal weights, as requested
                sum_Lnot += L_not.detach()
                sum_Lsmooth += L_smooth.detach()
                Batch_LOSS_sum += Loss_total



            # --------- Backpropagation for this batch ---------
            Batch_LOSS_mean = Batch_LOSS_sum / self.N_B
            mean_Lnot = sum_Lnot / self.N_B
            mean_Lsmooth = sum_Lsmooth / self.N_B
            Batch_LOSS_mean.backward()
            self.Joint_optimizer.step()

            # Log training metrics in your style (linear and dB)
            self.MSE_train_linear_epoch[ti] = Batch_LOSS_mean.detach()
            self.MSE_pnot_linear_epoch[ti] = mean_Lnot
            self.MSE_psmooth_linear_epoch[ti] = mean_Lsmooth
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
            self.MSE_pnot_dB_epoch[ti] = 10 * torch.log10(self.MSE_pnot_linear_epoch[ti])
            self.MSE_psmooth_dB_epoch[ti] = 10 * torch.log10(self.MSE_psmooth_linear_epoch[ti])


            # --------- Validation (no gradient) ---------
            self.PNotSmoothNN.eval()
            self.PsmoothFromPnot.eval()
            with torch.no_grad():
                MSE_cv_batch = torch.empty([self.N_CV], device=self.device)
                cv_sum_Lnot = 0.0
                cv_sum_Lsmooth = 0.0
                for j in range(self.N_CV):
                    y_cv = cv_input[j]
                    x_true_cv = cv_target[j]
                    SysModel.T_test = y_cv.size(-1)
                    m, n = SysModel.m, SysModel.n

                    # Optional F rotation for CV – keep RTSNet and PNot aligned
                    if generate_f is True and hasattr(SysModel, "F_valid"):
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)
                        self.PNotSmoothNN.F = SysModel.F
                    if generate_h is True and hasattr(SysModel, "H_valid"):
                        index = j // 10
                        SysModel.H = SysModel.H_valid[index]
                        self.model.update_H(SysModel.H)

                    # --- RTSNet forward (CV) ---
                    self.model.init_hidden()
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)

                    x_out_cv_forward = torch.empty(m, SysModel.T_test, device=self.device, dtype=y_cv.dtype)
                    x_out_cv = torch.empty(m, SysModel.T_test, device=self.device, dtype=y_cv.dtype)
                    kgain_list = []
                    self.model.smoother_gain_list = []

                    for t in range(SysModel.T_test):
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                        kg = self.model.KGain
                        kgain_list.append(kg.clone().detach())

                    # --- RTSNet backward (CV) ---
                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])

                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2], x_out_cv_forward[:,SysModel.T_test - 1],None)
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None,x_out_cv_forward[:, t],x_out_cv_forward[:, t + 1],x_out_cv[:, t + 2])
                        self.model.smoother_gain_list.append(self.model.SGain.clone().detach())


                    # --- Roll P_not on CV ---
                    self.PNotSmoothNN.reset_state()
                    P_prev = self.PNotSmoothNN.p_0
                    P_not_seq = torch.empty(m, m, SysModel.T_test, device=self.device, dtype=y_cv.dtype)
                    for t in range(SysModel.T_test):
                        K_t = kgain_list[t]
                        P_t = self.PNotSmoothNN(K_t, P_prev)
                        P_not_seq[:, :, t] = P_t
                        P_prev = P_t  # no need to detach under no_grad

                    # --- Roll P_smooth on CV ---
                    self.PsmoothFromPnot.reset_state()
                    P_smooth_seq = torch.empty_like(P_not_seq)
                    P_smooth_seq[:, :, SysModel.T_test - 1] = P_not_seq[:, :, SysModel.T_test - 1]
                    for t in range(SysModel.T_test - 2, -1, -1):
                        index = (SysModel.T_test - 2) - t
                        sgain_t = self.model.smoother_gain_list[index]
                        P_smooth_seq[:, :, t] = self.PsmoothFromPnot(P_not_seq[:, :, t], sgain_t)

                    # --- CV Loss (equal weights) ---
                    L_not = 0.0
                    L_smooth = 0.0
                    for t in range(SysModel.T_test):
                        err_f = (x_true_cv[:, t] - x_out_cv_forward[:, t]).unsqueeze(1)  # [m,1]
                        P_true_not = err_f @ err_f.T  # [m,m]
                        L_not = L_not + torch.norm(P_not_seq[:, :, t] - P_true_not, p='fro') ** 2

                        err_s = (x_true_cv[:, t] - x_out_cv[:, t]).unsqueeze(1)
                        P_true_sm = err_s @ err_s.T
                        L_smooth = L_smooth + torch.norm(P_smooth_seq[:, :, t] - P_true_sm, p='fro') ** 2

                    L_not = L_not / SysModel.T_test
                    L_smooth = L_smooth / SysModel.T_test
                    MSE_cv_batch[j] = 0.5 * L_not + 0.5 * L_smooth
                    cv_sum_Lnot += (L_not).detach()
                    cv_sum_Lsmooth += (L_smooth).detach()

                # Aggregate CV results and log
                self.CV_pnot_linear_epoch[ti] = cv_sum_Lnot / self.N_CV
                self.CV_psmooth_linear_epoch[ti] = cv_sum_Lsmooth / self.N_CV
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_batch)
                self.CV_pnot_dB_epoch[ti] = 10 * torch.log10(self.CV_pnot_linear_epoch[ti])
                self.CV_psmooth_dB_epoch[ti] = 10 * torch.log10(self.CV_psmooth_linear_epoch[ti])

                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.PNotSmoothNN, path_results_pnot)
                    torch.save(self.PsmoothFromPnot, path_results_psfp)
                    print("New best CV at epoch", ti, "→",
                          "Total:", self.MSE_cv_dB_epoch[ti].item(), "[dB],",
                          "P_not:", self.CV_pnot_dB_epoch[ti].item(), "[dB],",
                          "P_smooth:", self.CV_psmooth_dB_epoch[ti].item(), "[dB]")

            #########################
            ### Training Summary ###
            ########################
            print(ti,
                  "MSE Train P_not :", self.MSE_pnot_dB_epoch[ti], "[dB]",
                  "MSE Train P_smooth :", self.MSE_psmooth_dB_epoch[ti], "[dB]",
                  "MSE Train Total :", self.MSE_train_dB_epoch[ti], "[dB]",
                  "MSE Val P_not :", self.CV_pnot_dB_epoch[ti], "[dB]",
                  "MSE Val P_smooth :", self.CV_psmooth_dB_epoch[ti], "[dB]",
                  "MSE Val Total :", self.MSE_cv_dB_epoch[ti], "[dB]")

            if (ti > 1):
                d_train_total = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_val_total = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]

                print("diff Train Total :", d_train_total, "[dB]",
                      "diff Val Total :", d_val_total, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")



    def NNTrain_old(self, SysModel, cv_input, cv_target, train_input, train_target,path_results, load_model_path=None,generate_f=True,generate_h=False,
                CompositionLoss=False):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)

        MSE_train_linear_batch = torch.empty([self.N_B], device=self.device)
        self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)


        if load_model_path is not None:
            print("loading model_and keep training them")
            self.model = torch.load(load_model_path, map_location=self.device, weights_only=False).to(self.device).eval()
            # Re-link the optimizer to the parameters of the newly loaded model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                              weight_decay=self.weightDecay)

        # Training Mode
        self.model.train()


        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        nan_streak = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Zero gradients for both optimizers
            self.model.train()
            self.optimizer.zero_grad()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):

                self.model.init_hidden()
                n_e = random.randint(0, self.N_E - 1)
                if generate_f is True:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                    # Debug check
                    # print(f"[DEBUG] Sample {j}:")
                    # print("F matrix:\n", SysModel.F)
                if generate_h is True:  ####if we train with different h
                    index = n_e // 10
                    SysModel.H = SysModel.H_train[index]
                    self.model.update_H(SysModel.H)

                    # print("f(x) output for [1.0, 1.0]:", SysModel.f(torch.tensor([1.0, 1.0])))
                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                x_out_training_forward = torch.empty(SysModel.m, SysModel.T,
                                     device=y_training.device, dtype=y_training.dtype)
                x_out_training = torch.empty(SysModel.m, SysModel.T,
                                     device=y_training.device, dtype=y_training.dtype)

                # Init Hidden State
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                self.model.init_hidden()



                for t in range(0, SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:,SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T-1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],x_out_training_forward[:, SysModel.T - 1], None)
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],x_out_training_forward[:, t + 1], x_out_training[:, t + 2])

                # Compute losses separately
                if (CompositionLoss):
                    y_hat = torch.empty([SysModel.n, SysModel.T],
                                     device=y_training.device, dtype=y_training.dtype)
                    for t in range(SysModel.T):
                        y_hat[:, t] = SysModel.h(x_out_training[:, t])
                    rtsnet_loss = self.alpha * self.loss_fn(x_out_training, train_target[n_e]) + (1 - self.alpha) * self.loss_fn(y_hat, train_input[n_e])
                else:
                    rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])


                # Accumulate losses
                Batch_Optimizing_LOSS_sum += rtsnet_loss

                MSE_train_linear_batch[j] = rtsnet_loss.item()

            # Average losses for this batch
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            # Train RTSNet first
            Batch_Optimizing_LOSS_mean.backward()
            # 1) check every gradient tensor ori 2 blocks
            bad_grad = False
            for p in self.model.parameters():
                if p.grad is None:  # this param wasn’t used this pass
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad_grad = True
                    break

            if bad_grad:  # → skip this batch
                print("NaN/Inf gradients → batch skipped")
                nan_streak += 1
                if nan_streak >= 3:  # three bad batches in a row
                    print("Stopping training (3 consecutive bad batches).")
                continue  # start next epoch iteration


                # Calling the step function on an Optimizer makes an update to its
                # parameters
                nan_streak = 0


            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)#ori
            self.optimizer.step()


            # Average for logging
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)

                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test,
                               device=y_cv.device, dtype=y_cv.dtype)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test,
                               device=y_cv.device, dtype=y_cv.dtype)

                    if generate_f is True:  ####if we valid with different f
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)
                    if generate_h is True:  ####if we valid with different h
                        index = j // 10
                        SysModel.H = SysModel.H_valid[index]
                        self.model.update_H(SysModel.H)

                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                    self.model.init_hidden()


                    # Forward pass through RTSN et
                    for t in range(0, SysModel.T_test):
                        # x_out_cv_forward: [m] - Forward state estimates
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                    # Initialize backward pass
                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]  # [m]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    # First backward step
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],x_out_cv_forward[:, SysModel.T_test - 1], None)  # [m]
                    # Remaining backward steps
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],x_out_cv[:, t + 2])  # [m]


                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j]).item()  # Scalar

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.model, path_results)

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")



        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]
    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target,path_results, load_model_path=None,generate_f=False,generate_h=False,
                CompositionLoss=False, train_init=None, cv_init=None):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)

        MSE_train_linear_batch = torch.empty([self.N_B], device=self.device)
        self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)


        if load_model_path is not None:
            print("loading model_and keep training them")
            self.model = torch.load(load_model_path, map_location=self.device, weights_only=False).to(self.device).eval()
            # Re-link the optimizer to the parameters of the newly loaded model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                              weight_decay=self.weightDecay)

        # Training Mode
        self.model.train()


        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        nan_streak = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Zero gradients for both optimizers
            self.model.train()
            self.optimizer.zero_grad()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):

                self.model.init_hidden()
                n_e = random.randint(0, self.N_E - 1)
                if generate_f is True:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                    # Debug check
                    # print(f"[DEBUG] Sample {j}:")
                    # print("F matrix:\n", SysModel.F)
                if generate_h is True:  ####if we train with different h
                    index = n_e // 10
                    SysModel.H = SysModel.H_train[index]
                    self.model.update_H(SysModel.H)

                    # print("f(x) output for [1.0, 1.0]:", SysModel.f(torch.tensor([1.0, 1.0])))
                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                x_out_training_forward = torch.empty(SysModel.m, SysModel.T,
                                     device=y_training.device, dtype=y_training.dtype)
                x_out_training = torch.empty(SysModel.m, SysModel.T,
                                     device=y_training.device, dtype=y_training.dtype)

                # Init Hidden State
                if train_init is not None:
                    SysModel.m1x_0 = train_init[n_e]
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                self.model.init_hidden()



                for t in range(0, SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:,SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T-1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],x_out_training_forward[:, SysModel.T - 1], None)
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],x_out_training_forward[:, t + 1], x_out_training[:, t + 2])

                # Compute losses separately
                if (CompositionLoss):
                    y_hat = torch.empty([SysModel.n, SysModel.T],
                                     device=y_training.device, dtype=y_training.dtype)
                    for t in range(SysModel.T):
                        y_hat[:, t] = SysModel.h(x_out_training[:, t])
                    rtsnet_loss = self.alpha * self.loss_fn(x_out_training, train_target[n_e]) + (1 - self.alpha) * self.loss_fn(y_hat, train_input[n_e])
                else:
                    rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])


                # Accumulate losses
                Batch_Optimizing_LOSS_sum += rtsnet_loss

                MSE_train_linear_batch[j] = rtsnet_loss.item()

            # Average losses for this batch
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            # Train RTSNet first
            Batch_Optimizing_LOSS_mean.backward()
            # 1) check every gradient tensor ori 2 blocks
            bad_grad = False
            for p in self.model.parameters():
                if p.grad is None:  # this param wasn’t used this pass
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad_grad = True
                    break

            if bad_grad:  # → skip this batch
                print("NaN/Inf gradients → batch skipped")
                nan_streak += 1
                if nan_streak >= 3:  # three bad batches in a row
                    print("Stopping training (3 consecutive bad batches).")
                continue  # start next epoch iteration


                # Calling the step function on an Optimizer makes an update to its
                # parameters
                nan_streak = 0


            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)#ori
            self.optimizer.step()


            # Average for logging
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)

                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test,
                               device=y_cv.device, dtype=y_cv.dtype)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test,
                               device=y_cv.device, dtype=y_cv.dtype)

                    if generate_f is True:  ####if we valid with different f
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)
                    if generate_h is True:  ####if we valid with different h
                        index = j // 10
                        SysModel.H = SysModel.H_valid[index]
                        self.model.update_H(SysModel.H)

                    if cv_init is not None:
                        SysModel.m1x_0 = cv_init[j]
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                    self.model.init_hidden()


                    # Forward pass through RTSN et
                    for t in range(0, SysModel.T_test):
                        # x_out_cv_forward: [m] - Forward state estimates
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                    # Initialize backward pass
                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]  # [m]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    # First backward step
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],x_out_cv_forward[:, SysModel.T_test - 1], None)  # [m]
                    # Remaining backward steps
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],x_out_cv[:, t + 2])  # [m]


                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j]).item()  # Scalar

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.model, path_results)

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")



        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]











    def NNTrain_with_F(self, SysModel, cv_input, cv_target, train_input, train_target,path_results, load_model_path=None,generate_f=True,generate_h=False,beta = 0.5):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])

        m = SysModel.m

        F_train_copy = [F.clone() for F in SysModel.F_train]
        F_valid_copy = [F.clone() for F in SysModel.F_valid]

        if load_model_path is not None:
            print("loading model_and keep training them")
            self.model = torch.load(load_model_path, weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                              weight_decay=self.weightDecay)

        # Training Mode
        self.model.train()


        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        nan_streak = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Zero gradients for both optimizers
            self.model.train()
            self.optimizer.zero_grad()
            Batch_Optimizing_LOSS_sum = 0
            F_loss_batch = 0

            for j in range(0, self.N_B):


                n_e = random.randint(0, self.N_E - 1)
                if generate_f is True:  ####if we train with different f
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                if generate_h is True:  ####if we train with different h
                    index = n_e // 10
                    SysModel.H = SysModel.H_train[index]
                    self.model.update_H(SysModel.H)

                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                V_list = []
                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)

                # Init Hidden State
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                self.model.init_hidden()

                # Lists to store the results from our analytical filter
                P_filtered_seq = torch.empty(m, m, SysModel.T)
                P_predicted_seq = torch.empty(m, m, SysModel.T)
                # Initialize P for the filter using the prior
                P_filt_prev = SysModel.m2x_0

                for t in range(0, SysModel.T):
                    # 1. ANALYTICAL PREDICTION STEP for covariance
                    P_pred = SysModel.F @ P_filt_prev @ SysModel.F.T + SysModel.Q
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                    K_t = self.model.KGain.clone()
                    # 3. ANALYTICAL UPDATE STEP using the Kalman Gain from the network
                    I = torch.eye(m)
                    # Using the numerically stable Joseph form for the covariance update
                    P_filt = (I - K_t @ SysModel.H) @ P_pred @ (I - K_t @ SysModel.H).T + K_t @ SysModel.R @ K_t.T
                    # 4. Save results and update for next step
                    P_predicted_seq[:, :, t] = P_pred
                    P_filtered_seq[:, :, t] = P_filt
                    P_filt_prev = P_filt

                self.model.smoother_gain_list = []  # Clear the list before populating
                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:,SysModel.T - 1]  # backward smoothing starts from x_T|T
                self.model.InitBackward(x_out_training[:, SysModel.T-1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],x_out_training_forward[:, SysModel.T - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone())
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],x_out_training_forward[:, t + 1], x_out_training[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone())
                #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.linalg.inv(P_1_0_pred)  ######COMPUTE S_0
                self.model.smoother_gain_list.append(s_0.clone())  # [m, m]

                P_smoothed_seq = torch.empty(m, m, SysModel.T)
                P_smoothed_seq[:, :, -1] = P_filtered_seq[:, :, -1]  # P_T|T is the last filtered P
                for t in range(SysModel.T - 2, -1, -1):
                    # Get necessary matrices for this time step
                    P_filt_t = P_filtered_seq[:, :, t]
                    P_pred_t_plus_1 = P_predicted_seq[:, :, t + 1]
                    P_smooth_t_plus_1 = P_smoothed_seq[:, :, t + 1]

                    # Get the Smoother Gain from the network for this time step
                    reverse_time = SysModel.T - 1 - t
                    S_t = self.model.smoother_gain_list[reverse_time]  ####S[T-1] = S(0), S[0] = S(T-1)

                    # ANALYTICAL RTS UPDATE using the smoother gain from the network
                    P_smooth_t = P_filt_t + S_t @ (P_smooth_t_plus_1 - P_pred_t_plus_1) @ S_t.T
                    P_smoothed_seq[:, :, t] = P_smooth_t

                V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq,
                                                   self.model.smoother_gain_list)
                V_list.append(V)  # [seq](tensor(m,m,T))

                ###################### 2) run M‑step on **this sequence** (batch of size 1)
                X_s = x_out_training.unsqueeze(0)  # → [1, m, T]
                P_smooth_s = P_smoothed_seq.unsqueeze(0)  # → [1, m, m, T]
                V_s = V.unsqueeze(0)  # → [1, m, m, T]
                m_state = SysModel.F.shape[0]

                F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, m_state)
                F_est = F_est[0]
                index_1 = n_e // 10
                ###########CHANGE F AS TRAINING##########
                F_train_copy[index_1] = F_est
                #####################################
                F_TRUE = SysModel.F_train_TRUE[index_1]
                eps_f = F_est - F_TRUE
                eps = torch.linalg.norm(eps_f, ord='fro')

                F_loss_batch += eps
                rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])
                # Accumulate losses
                Batch_Optimizing_LOSS_sum += rtsnet_loss
                MSE_train_linear_batch[j] = rtsnet_loss.item()

            # Average losses for this batch
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            F_loss_mean = F_loss_batch/ self.N_B
            loss_total = beta*Batch_Optimizing_LOSS_mean + (1-beta)*F_loss_mean
            loss_total_training_db = 10 * torch.log10(loss_total)

            # Train RTSNet first
            loss_total.backward()
            # 1) check every gradient tensor ori 2 blocks
            # bad_grad = False
            # for p in self.model.parameters():
            #     if p.grad is None:  # this param wasn’t used this pass
            #         continue
            #     if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            #         bad_grad = True
            #         break
            #
            # if bad_grad:  # → skip this batch
            #     print("NaN/Inf gradients → batch skipped")
            #     nan_streak += 1
            #     if nan_streak >= 3:  # three bad batches in a row
            #         print("Stopping training (3 consecutive bad batches).")
            #         return  # leave NNTrain early
            #     self.model.zero_grad(set_to_none=True)  # throw away bad grads
            #     continue  # start next epoch iteration
            # ── DEBUG F‑LOSS GRADIENT CHECK ──
            # print("=== DEBUG: gradient norms after F_loss.backward() ===")
            # no_grad = True
            # for name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         print(f"{name:30s} grad is None")
            #     else:
            #         gnorm = param.grad.norm().item()
            #         print(f"{name:30s} grad norm = {gnorm:.6e}")
            #         if gnorm > 0:
            #             no_grad = False
            # print(">>> Any nonzero grads? ", not no_grad)
            # print(V.requires_grad)  # should be True
            # print(V.grad_fn)  # should NOT be None
            # print(type(V.grad_fn))  # e.g. <class 'StackBackward0'>




            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)#ori
            self.optimizer.step()


            # Average for logging
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])


            ##################
            ### Optimizing ###
            ##################

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                F_loss_batch_cv = 0.0
                MSE_cv_linear_batch = torch.empty([self.N_CV])

                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)

                    P_filtered_seq = torch.empty(m, m, SysModel.T_test)
                    P_predicted_seq = torch.empty(m, m, SysModel.T_test)


                    if generate_f is True:  ####if we valid with different f
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)
                    if generate_h is True:  ####if we valid with different h
                        index = j // 10
                        SysModel.H = SysModel.H_valid[index]
                        self.model.update_H(SysModel.H)

                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                    self.model.init_hidden()

                    P_filt_prev = SysModel.m2x_0
                    # Forward pass through RTSN et
                    for t in range(0, SysModel.T_test):
                        P_pred = SysModel.F @ P_filt_prev @ SysModel.F.T + SysModel.Q
                        # x_out_cv_forward: [m] - Forward state estimates
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                        K_t = self.model.KGain.clone()
                        I = torch.eye(m, device=P_pred.device)
                        P_filt = (I - K_t @ SysModel.H) @ P_pred @ (I - K_t @ SysModel.H).T + K_t @ SysModel.R @ K_t.T

                        P_predicted_seq[:, :, t] = P_pred
                        P_filtered_seq[:, :, t] = P_filt
                        P_filt_prev = P_filt

                    # ---------- BACKWARD (RTS for x, analytic P_smooth using S from net) ----------
                    self.model.smoother_gain_list = []
                    # Initialize backward pass
                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]  # [m]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    # First backward step
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],x_out_cv_forward[:, SysModel.T_test - 1], None)  # [m]
                    self.model.smoother_gain_list.append(self.model.SGain.clone())
                    # Remaining backward steps
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],x_out_cv[:, t + 2])  # [m]
                        self.model.smoother_gain_list.append(self.model.SGain.clone())

                    # s_0
                    P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                    s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.linalg.inv(P_1_0_pred)
                    self.model.smoother_gain_list.append(s_0.clone())

                    ###################em steps
                    # analytic RTS for P
                    P_smoothed_seq = torch.empty(m, m, SysModel.T_test)
                    P_smoothed_seq[:, :, -1] = P_filtered_seq[:, :, -1]
                    for t in range(SysModel.T_test - 2, -1, -1):
                        P_filt_t = P_filtered_seq[:, :, t]
                        P_pred_t_plus_1 = P_predicted_seq[:, :, t + 1]
                        P_smooth_t_plus_1 = P_smoothed_seq[:, :, t + 1]
                        reverse_time = SysModel.T_test - 1 - t
                        S_t = self.model.smoother_gain_list[reverse_time]
                        P_smooth_t = P_filt_t + S_t @ (P_smooth_t_plus_1 - P_pred_t_plus_1) @ S_t.T
                        P_smoothed_seq[:, :, t] = P_smooth_t

                    # V_t tensor
                    V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq,
                                                       self.model.smoother_gain_list)
                    # ---------- M-step ----------
                    X_s = x_out_cv.unsqueeze(0)  # [1, m, T]
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)  # [1, m, m, T]
                    V_s = V.unsqueeze(0)  # [1, m, m, T]

                    F_est_cv = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s,m)[0]
                    # write into copy
                    index = j // 10
                    F_valid_copy[index] = F_est_cv.detach()

                    # F-loss for logging
                    F_TRUE_cv = SysModel.F_valid_TRUE[index]
                    F_loss_batch_cv += torch.linalg.norm(F_est_cv - F_TRUE_cv, ord='fro')

                    # state MSE
                    MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j]).item()
                    #######################################################

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                F_loss_batch_cv_av = F_loss_batch_cv/self.N_CV
                mse_cv_total = self.MSE_cv_linear_epoch[ti].item()*0.8 + 0.2*F_loss_batch_cv_av
                mse_cv_total_loss_db = 10 * torch.log10(mse_cv_total)

                # if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):this is whtiout F
                #     self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                #     self.MSE_cv_idx_opt = ti
                #
                #     torch.save(self.model, path_results)

                if (mse_cv_total_loss_db < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = mse_cv_total_loss_db
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.model, path_results)



            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE rts Training :", self.MSE_train_dB_epoch[ti], "[dB]","MSE F LOSS Training :", F_loss_mean, "MSE F LOSS TOTAL :",loss_total_training_db,"[db]",
                  "MSE rts Validation :",self.MSE_cv_dB_epoch[ti],"[dB]","MSE total Validation :",mse_cv_total_loss_db,"[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")



        return F_train_copy, F_valid_copy







    def compute_cross_covariances(self, F, H, Ks, Ps, SGains):
        """
        Computes lag-one cross-covariances and returns them as a single tensor.

        Returns:
            V_tensor (torch.Tensor): A single tensor of shape [m, m, T]
            F [m,m]
            ks [m,n]
            Ps [m,m,T]

        """
        # Get dimensions from input tensors
        m = Ps.shape[0]
        T = Ps.shape[2]

        # 1. Create a single empty tensor with the target shape [m, m, T]
        V_tensor = torch.empty(m, m, T, device=F.device)

        # Note: Ks is the Kalman Gain for the last time step, T-1.
        # Ps is the sequence of filtered covariances P_t|t
        # SGains is the sequence of Smoother Gains S_t for t = T-2 down to 0.

        # --- Calculation for V_{T-1, T-2 | T} ---
        # The last element of V_tensor will actually be V_{T-1}
        I = torch.eye(m, device=F.device)
        # P_{T-2|T-2} is at index T-2
        # V_{T-1} uses the filtered covariance from T-2, not T-1
        V_T_minus_1 = (I - Ks @ H) @ F @ Ps[:, :, T - 2]

        # 2. Assign the result to the last time-slice of the tensor
        V_tensor[:, :, T - 1] = V_T_minus_1

        # --- Backward recursion for t = T-2 down to 0 ---
        for t in range(T - 2, -1, -1):
            # Get values for time t
            Pt = Ps[:, :, t]
            # Smoother gain S_t has been stored in reverse order
            # For t=T-2, we need the first element of SGains (index 0)
            # For t=T-3, we need the second element (index 1), and so on.
            # print('size ,p,v,k',Ps.size, len(SGains),SGains[0],Ks.size)
            index = (T - 2) - t
            St = SGains[index]
            St_minus1 = SGains[index +1]
            # Get V_{t+1, t | T} from the tensor we are filling
            V_t_plus_1 = V_tensor[:, :, t + 1]

            # The cross-covariance update equation
            V_t = Pt @ St_minus1.T + St @ (V_t_plus_1 - F @ Pt) @ St_minus1.T

            # 3. Assign the result to the correct slice [:, :, t]
            V_tensor[:, :, t] = V_t

        # 4. Return the single tensor
        return V_tensor

    def NNTest(self, SysModel, test_input, test_target,load_model_path, generate_f=False,generate_h=False,init_x_list=None, init_P_list=None,non_linear_h=False):

        tp = torch.float32
        print("Testing RTSNet...")
        self.N_T = len(test_input)


        self.MSE_test_linear_arr = torch.empty([self.N_T],device=self.device, dtype=tp)


             # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Load models
        self.model = torch.load(load_model_path,weights_only=False).to(self.device).eval()

        torch.no_grad()

        x_out_list = []

        start = time.time()

        with torch.no_grad():
            for j in range(0, self.N_T):
                y_mdl_tst = test_input[j]
                SysModel.T_test = y_mdl_tst.size()[-1]
                x_out_test_forward_1 = torch.empty(SysModel.m, SysModel.T_test, device=self.device, dtype=tp)
                x_out_test = torch.empty(SysModel.m, SysModel.T_test,  device=self.device, dtype=tp)

                # choose initials for this sequence j (same logic as NNTest)
                if (init_x_list is not None):
                    P0 = SysModel.m2x_0
                    x0 = init_x_list[j]
                    SysModel.m1x_0=x0
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0
                # initialize prior Sigma and hidden for this sequence
                self.model.prior_Sigma = P0
                self.model.InitSequence(x0, SysModel.T_test)
                self.model.init_hidden()

                if generate_f ==False:  ####if we valid with different f
                    SysModel.F = SysModel.F_test[j]
                    self.model.update_F(SysModel.F)
                elif generate_f ==True:  ####if we valid with different f
                    index = j // 10
                    SysModel.F = SysModel.F_test[index]
                    self.model.update_F(SysModel.F)
                if generate_h ==False:  ####if we valid with different h
                    SysModel.H = SysModel.H_test[j]
                    SysModel.update_h(SysModel.H)
                    self.model.update_H(SysModel.H)
                elif generate_h ==True:  ####if we valid with different h
                    index = j // 10
                    SysModel.H = SysModel.H_test[index]
                    SysModel.update_h(SysModel.H)
                    self.model.update_H(SysModel.H)
                # Forward pass and compute P-smooth
                self.model.sigma_list = []
                self.model.smoother_gain_list = []
                for t in range(0, SysModel.T_test):
                    x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)
                    P_test_forward = self.model.h_Sigma.clone().detach()
                    self.model.sigma_list.append(P_test_forward)  # [1, 1, m²]
                x_out_test[:, SysModel.T_test - 1] = x_out_test_forward_1[:, SysModel.T_test - 1]
                self.model.InitBackward(x_out_test[:, SysModel.T_test - 1])
                x_out_test[:, SysModel.T_test - 2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test - 2],
                                                                x_out_test_forward_1[:, SysModel.T_test - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())

                for t in range(SysModel.T_test - 3, -1, -1):#T-3 to 0
                    x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                                  x_out_test[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())##there are T-1 s gain


                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()

                x_out_list.append(x_out_test)
                # print('x_latst',x_out_test[:, -1])

        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)


        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg


        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, torch.stack(x_out_list), t]

    def NNTest_with_p(self, SysModel, test_input, test_target, load_model_path,load_p_smoothe_model_path=None, generate_f=True,generate_h=False,init_x_list=None, init_P_list=None,non_linear_h =False):

        tp = torch.float32
        print("Testing RTSNet...")
        self.N_T = len(test_input)


        self.MSE_test_linear_arr = torch.empty([self.N_T], device=self.device, dtype=tp)
        self.MSE_test_psmooth_arr = torch.empty([self.N_T], device=self.device, dtype=tp)

             # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Load models
        if load_p_smoothe_model_path is not None:
            self.PsmoothNN = torch.load(load_p_smoothe_model_path, map_location=self.device, weights_only=False).to(self.device).eval()
        else:
            self.PsmoothNN = torch.load('RTSNet/full_info/best-model.pt', map_location=self.device, weights_only=False).to(self.device).eval()
        self.model = torch.load(load_model_path, map_location=self.device, weights_only=False).to(self.device).eval()

        torch.no_grad()

        x_out_list = []
        P_smooth_list = []
        V_list = []
        start = time.time()
        self.model.K_T_list = []
        print('ORICHECK DEHIL FFFFFFFFFFFFFFFF',SysModel.F_test[0] )
        for j in range(0, self.N_T):
            y_mdl_tst = test_input[j]
            SysModel.T_test = y_mdl_tst.size()[-1]
            x_out_test_forward_1 = torch.empty(SysModel.m, SysModel.T_test, device=self.device, dtype=tp)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test, device=self.device, dtype=tp)

            # choose initials for this sequence j
            if (init_x_list is not None) and (init_P_list is not None):
                P0 = init_P_list[j].to(self.device)
                x0 = init_x_list[j].to(self.device)
            else:
                P0 = SysModel.m2x_0.to(self.device)
                x0 = SysModel.m1x_0.to(self.device)

            # --- initialize prior Sigma and hidden for this sequence ---
            self.model.prior_Sigma = P0  # 1) set prior_Sigma to P0 for this seq
            self.model.InitSequence(x0, SysModel.T_test)  # 3) set mean x0, T
            self.model.init_hidden()  # 2) reset hidden -> seeds h_Sigma from prior_Sigma


            if generate_f == False:  ####if we valid with different f
                SysModel.F = SysModel.F_test[j]
                self.model.update_F(SysModel.F)
            else:
                index = j // 10
                SysModel.F = SysModel.F_test[index]
                self.model.update_F(SysModel.F)
            if generate_h == False:  ####if we valid with different h
                SysModel.H = SysModel.H_test[j]
                self.model.update_H(SysModel.H)
            else:
                index = j // 10
                SysModel.H = SysModel.H_test[index]
                self.model.update_H(SysModel.H)
            # Forward pass and compute P-smooth
            self.model.sigma_list = []
            self.model.smoother_gain_list = []
            for t in range(0, SysModel.T_test):
                x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)
                P_test_forward = self.model.h_Sigma.clone().detach()
                self.model.sigma_list.append(P_test_forward)  # [1, 1, m²]
                if t == SysModel.T_test - 1:
                    K = self.model.KGain.clone().detach()
                    self.model.K_T_list.append(K)  # [m, n]
                    if non_linear_h == True:
                        # Compute H_last = ∂h/∂x at time T-1 (use filtered state at T-1 as approximation)
                        x_last = x_out_test_forward_1[:, SysModel.T_test - 1].view(SysModel.m, 1)
                        H_last = getJacobian(x_last, SysModel.h)
                        _jacobian_watchdog(H_last, x_last, SysModel.h,)#ori del

            x_out_test[:, SysModel.T_test - 1] = x_out_test_forward_1[:, SysModel.T_test - 1]
            self.model.InitBackward(x_out_test[:, SysModel.T_test - 1])
            x_out_test[:, SysModel.T_test - 2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test - 2],
                                                            x_out_test_forward_1[:, SysModel.T_test - 1], None)
            self.model.smoother_gain_list.append(self.model.SGain.clone().detach())

            for t in range(SysModel.T_test - 3, -1, -1):#T-3 to 0
                x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                              x_out_test[:, t + 2])
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())##there are T-1 s gain

            # Compute P-smooth predictions
            P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T_test,  device=device)
            dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m, device=device)  # shape: [1, 1, m²] input to PsmoothNN
            sigma_T = self.model.sigma_list[-1]  # shape: [1, 1, m²] input to PsmoothNN
            self.PsmoothNN.start = 0 #initial the model
            # Handle initial P-smooth at time T_test
            P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
            P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(SysModel.m, SysModel.m))  # shape: [m, m]
            P_smoothed_seq[:, :, SysModel.T_test - 1] = P_matrix  # shape: [m, m]

            for t in range(SysModel.T_test - 2, -1, -1):
                sigma_t = self.model.sigma_list[t].view(1, 1, -1)
                index = (SysModel.T_test - 2) - t
                sgain_t = self.model.smoother_gain_list[index].reshape(1, 1, -1)
                P_flat = self.PsmoothNN(sigma_t, sgain_t)
                P_matrix = self.PsmoothNN.enforce_covariance_properties(P_flat.view(-1).view(SysModel.m,SysModel.m))
                P_smoothed_seq[:, :, t] = P_matrix

            #compute s(0) for later use, by the # S_t = P_t * F.T * (P_t+1)^-1
            P_1_0_pred = SysModel.F @ P0 @ SysModel.F.T + SysModel.Q
            s_0 = P0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
            self.model.smoother_gain_list.append(s_0.clone().detach())  # [m, m]


            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()

            # Compute P-smooth loss
            #option 1
            self.MSE_test_psmooth_arr[j] = self.PsmoothNN.compute_loss(P_smoothed_seq, test_target[j], x_out_test).item()
            #option 2
            #self.MSE_test_psmooth_arr[j] = self.compute_gaussian_loss1(P_smoothed_seq, test_target[j], x_out_test).item()


            x_out_list.append(x_out_test)
            P_smooth_list.append(P_smoothed_seq)

            if non_linear_h == True:
                #######compute V############
                V =  self.compute_cross_covariances(SysModel.F, H_last, K, P_smoothed_seq, self.model.smoother_gain_list)
            else:
                #######compute V############
                V =  self.compute_cross_covariances(SysModel.F, SysModel.H, K, P_smoothed_seq, self.model.smoother_gain_list)
            V_list.append(V)#[seq](tensor(m,m,T))


        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        self.MSE_test_psmooth_avg = torch.mean(self.MSE_test_psmooth_arr)
        self.MSE_test_psmooth_dB_avg = 10 * torch.log10(self.MSE_test_psmooth_avg)
        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_psmooth_std = torch.std(self.MSE_test_psmooth_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg
        self.test_std_dB = 10 * torch.log10(self.MSE_test_psmooth_std + self.MSE_test_psmooth_avg) - self.MSE_test_psmooth_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        str = self.modelName + "-" + "P-smooth MSE Test:"
        print(str, self.MSE_test_psmooth_dB_avg, "[dB]")
        str = self.modelName + "-" + "P-smooth STD Test:"
        print(str, self.MSE_test_psmooth_std, "[dB]")
        # Print Run Time
        print("Inference Time:", t)




        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, torch.stack(x_out_list), t, torch.stack(P_smooth_list), V_list, self.model.K_T_list,
                self.MSE_test_psmooth_dB_avg, self.MSE_test_psmooth_std]


    def NNTest_p(self, SysModel, test_input, test_target, load_model_path,load_pnot_path=None, load_psfp_path=None, generate_f=True,generate_h=False,init_x_list=None, init_P_list=None,non_linear_h =False):

        tp = torch.float32
        print("Testing RTSNet...")
        self.N_T = len(test_input)


        self.MSE_test_linear_arr = torch.empty([self.N_T], device=self.device, dtype=tp)
        self.MSE_test_psmooth_arr = torch.empty([self.N_T], device=self.device, dtype=tp)
        self.MSE_test_pnot_arr = torch.empty([self.N_T], device=self.device, dtype=tp)

             # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Load RTSNet (frozen for test)
        self.model = torch.load(load_model_path, map_location=self.device, weights_only=False).to(self.device).eval()

        if load_pnot_path is not None:
            self.PNotSmoothNN = torch.load(load_pnot_path, map_location=self.device, weights_only=False).to(self.device).eval()
            self.PsmoothFromPnot = torch.load(load_psfp_path, map_location=self.device, weights_only=False).to(self.device).eval()

        with torch.no_grad():

            x_out_list = []
            P_smooth_list = []
            V_list = []
            start = time.time()
            self.model.K_T_list = []
            print('ORICHECK DEHIL FFFFFFFFFFFFFFFF',SysModel.F_test[0] )
            for j in range(0, self.N_T):
                y_mdl_tst = test_input[j]
                SysModel.T_test = y_mdl_tst.size()[-1]
                x_out_test_forward_1 = torch.empty(SysModel.m, SysModel.T_test, device=self.device, dtype=tp)
                x_out_test = torch.empty(SysModel.m, SysModel.T_test, device=self.device, dtype=tp)

                # choose initials for this sequence j
                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list[j]
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                # --- initialize prior Sigma and hidden for this sequence ---
                self.model.prior_Sigma = P0  # 1) set prior_Sigma to P0 for this seq
                self.model.InitSequence(x0, SysModel.T_test)  # 3) set mean x0, T
                self.model.init_hidden()  # 2) reset hidden -> seeds h_Sigma from prior_Sigma


                if generate_f == False:  ####if we valid with different f
                    SysModel.F = SysModel.F_test[j]
                    self.model.update_F(SysModel.F)
                else:
                    index = j // 10
                    SysModel.F = SysModel.F_test[index]
                    self.model.update_F(SysModel.F)
                if generate_h == False:  ####if we valid with different h
                    SysModel.H = SysModel.H_test[j]
                    self.model.update_H(SysModel.H)
                else:
                    index = j // 10
                    SysModel.H = SysModel.H_test[index]
                    self.model.update_H(SysModel.H)
                self.PNotSmoothNN.F = SysModel.F  # keep in sync with RTSNet's F
                # Forward pass and compute P-smooth
                kgain_list = []
                self.model.smoother_gain_list = []
                for t in range(0, SysModel.T_test):
                    x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)
                    K = self.model.KGain.clone().detach()
                    kgain_list.append(K)# for the p estimation
                    if t == SysModel.T_test - 1:
                        self.model.K_T_list.append(K)  # [m, n] for the v estimation
                        if non_linear_h == True:
                            # Compute H_last = ∂h/∂x at time T-1 (use filtered state at T-1 as approximation)
                            x_last = x_out_test_forward_1[:, SysModel.T_test - 1].view(SysModel.m, 1)
                            H_last = getJacobian(x_last, SysModel.h)
                            _jacobian_watchdog(H_last, x_last, SysModel.h,)#ori del

                x_out_test[:, SysModel.T_test - 1] = x_out_test_forward_1[:, SysModel.T_test - 1]
                self.model.InitBackward(x_out_test[:, SysModel.T_test - 1])
                x_out_test[:, SysModel.T_test - 2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test - 2],
                                                                x_out_test_forward_1[:, SysModel.T_test - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone().detach())

                for t in range(SysModel.T_test - 3, -1, -1):#T-3 to 0
                    x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                                  x_out_test[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone().detach())##there are T-1 s gain

                # ---------- NEW: roll P_not first, then P_smooth ----------
                # 1) Roll P_not with (K_t, P_prev) and F (set per sequence)
                self.PNotSmoothNN.reset_state()
                P_not_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T_test, device=self.device, dtype=test_input[j].dtype)
                P_prev = SysModel.m2x_0

                for t in range(SysModel.T_test):
                    P_t = self.PNotSmoothNN(kgain_list[t], P_prev)
                    P_not_seq[:, :, t] = P_t
                    P_prev = P_t  # no detach needed in no_grad()
                # 2) Roll P_smooth with (P_not_t, SGain_t)
                self.PsmoothFromPnot.reset_state()
                P_smoothed_seq = torch.empty_like(P_not_seq)
                # At T-1 we don't have SGain; use the convention P_smooth(T-1) = P_not(T-1)
                P_smoothed_seq[:, :, SysModel.T_test - 1] = P_not_seq[:, :, SysModel.T_test - 1]
                for t in range(SysModel.T_test - 2, -1, -1):
                    index = (SysModel.T_test - 2) - t  # aligns SGain_list[0] with time t = T-2
                    sgain_t = self.model.smoother_gain_list[index]  # [m, m]
                    P_smoothed_seq[:, :, t] = self.PsmoothFromPnot(P_not_seq[:, :, t], sgain_t)

                #compute s(0) for later use, by the # S_t = P_t * F.T * (P_t+1)^-1
                P_1_0_pred = SysModel.F @ P0 @ SysModel.F.T + SysModel.Q
                s_0 = P0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
                self.model.smoother_gain_list.append(s_0.clone().detach())  # [m, m]

                # --- State MSE ---
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()

                # --- P_not MSE (use x_fwd) ---
                self.MSE_test_pnot_arr[j] = self.PNotSmoothNN.compute_loss( P_not_seq, test_target[j], x_out_test_forward_1).item()

                # --- P_smooth MSE (use x_smooth) ---
                self.MSE_test_psmooth_arr[j] = self.PNotSmoothNN.compute_loss( P_smoothed_seq, test_target[j], x_out_test).item()


                x_out_list.append(x_out_test)
                P_smooth_list.append(P_smoothed_seq)

                if non_linear_h == True:
                    #######compute V############
                    V =  self.compute_cross_covariances(SysModel.F, H_last, K, P_not_seq, self.model.smoother_gain_list)
                else:
                    #######compute V############
                    V =  self.compute_cross_covariances(SysModel.F, SysModel.H, K, P_not_seq, self.model.smoother_gain_list)
                V_list.append(V)#[seq](tensor(m,m,T))


            end = time.time()
            t = end - start

            # --- Aggregates for state MSE ---
            self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
            self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
            self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
            # Optional CI-like dB spread (same style you used elsewhere)
            self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

            # --- Aggregates for P_not ---
            self.MSE_test_pnot_avg = torch.mean(self.MSE_test_pnot_arr)
            self.MSE_test_pnot_dB_avg = 10 * torch.log10(self.MSE_test_pnot_avg)
            self.MSE_test_pnot_std = torch.std(self.MSE_test_pnot_arr, unbiased=True)

            # --- Aggregates for P_smooth (you already had psmooth arr) ---
            self.MSE_test_psmooth_avg = torch.mean(self.MSE_test_psmooth_arr)
            self.MSE_test_psmooth_dB_avg = 10 * torch.log10(self.MSE_test_psmooth_avg)
            self.MSE_test_psmooth_std = torch.std(self.MSE_test_psmooth_arr, unbiased=True)

            # Prints
            print(self.modelName + "-" + "MSE Test:", self.MSE_test_dB_avg, "[dB]")
            print(self.modelName + "-" + "STD Test:", self.test_std_dB, "[dB]")

            print(self.modelName + "-" + "P-not MSE Test:", self.MSE_test_pnot_dB_avg, "[dB]")
            print(self.modelName + "-" + "P-not STD Test:", self.MSE_test_pnot_std, "[linear]")

            print(self.modelName + "-" + "P-smooth MSE Test:", self.MSE_test_psmooth_dB_avg, "[dB]")
            print(self.modelName + "-" + "P-smooth STD Test:", self.MSE_test_psmooth_std, "[linear]")




        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, torch.stack(x_out_list), t, torch.stack(P_smooth_list), V_list, self.model.K_T_list,
                self.MSE_test_psmooth_dB_avg, self.MSE_test_psmooth_std,self.MSE_test_pnot_dB_avg, self.MSE_test_pnot_std]



    def NNTest_HybridP(self, SysModel, test_input, test_target, load_model_path):
        """
        This function tests the RTSNet, but computes the smoothed covariance P
        analytically using the Kalman Gain (K) and Smoother Gain (S) produced by the RTSNet.
        This replaces the PsmoothNN.
        """
        print("Testing Hybrid Smoother (RTSNet states/gains, Analytical P)...")
        self.N_T = len(test_input)

        # Load the trained RTSNet model
        self.model = torch.load(load_model_path, weights_only=False)
        self.model.eval()

        # Initialize a tensor to store the MSE for each sequence >>>
        self.MSE_test_linear_arr = torch.empty([self.N_T])
        loss_fn = nn.MSELoss(reduction='mean')

        # To store the final results for all sequences
        x_out_list = []
        P_smooth_list_analytical = []
        V_list = []
        with torch.no_grad():
            for j in range(0, self.N_T):
                SysModel.T_test = test_input[j].size()[-1]
                m = SysModel.m

                # Get the correct F for this sequence
                # This uses the j // 10 logic. If you change to the fundamental
                # per-sequence F list, this line becomes: SysModel.F = SysModel.F_test[j]
                index = j // 10
                SysModel.F = SysModel.F_test[index]
                self.model.update_F(SysModel.F)
                # Get the correct H for this sequence (if using H diversity)
                if hasattr(SysModel, 'H_test') and SysModel.H_test is not None:
                    SysModel.H = SysModel.H_test[index]
                    self.model.update_H(SysModel.H)

                # --- INITIALIZATION ---
                self.model.init_hidden()
                self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)

                # Lists to store the results from our analytical filter
                P_filtered_seq = torch.empty(m, m, SysModel.T_test)
                P_predicted_seq = torch.empty(m, m, SysModel.T_test)
                # Initialize P for the filter using the prior
                P_filt_prev = SysModel.m2x_0
                # <<<Initialize the tensor to store the forward sequence here >>>
                x_out_test_forward_1 = torch.empty(m, SysModel.T_test)
                K_t = None
                y_mdl_tst = test_input[j]
                # --- FORWARD PASS: HYBRID KALMAN FILTER ---
                for t in range(0, SysModel.T_test):
                    # 1. ANALYTICAL PREDICTION STEP for covariance
                    P_pred = SysModel.F @ P_filt_prev @ SysModel.F.T + SysModel.Q

                    # <<<Capture the output of the forward pass into our tensor >>>
                    x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)

                    K_t = self.model.KGain.clone()

                    # 3. ANALYTICAL UPDATE STEP using the Kalman Gain from the network
                    I = torch.eye(m)
                    # Using the numerically stable Joseph form for the covariance update
                    P_filt = (I - K_t @ SysModel.H) @ P_pred @ (I - K_t @ SysModel.H).T + K_t @ SysModel.R @ K_t.T

                    # 4. Save results and update for next step
                    P_predicted_seq[:, :, t] = P_pred
                    P_filtered_seq[:, :, t] = P_filt
                    P_filt_prev = P_filt

                # --- BACKWARD PASS: HYBRID RTS SMOOTHER ---
                # Run the RTSNet backward pass once to get all state estimates and smoother gains
                x_out_test = torch.empty(m, SysModel.T_test)
                self.model.smoother_gain_list = []  # Clear the list before populating

                x_out_test[:, -1] = x_out_test_forward_1[:, -1]
                self.model.InitBackward(x_out_test[:, -1])
                # Special first backward step for t = T-2
                x_out_test[:, SysModel.T_test - 2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test - 2],
                                                                x_out_test_forward_1[:, SysModel.T_test - 1], None)
                self.model.smoother_gain_list.append(self.model.SGain.clone())
                for t in range(SysModel.T_test - 3, -1, -1):  #### T-3 all the way to 0 includes [T-3,0]
                    x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t + 1],
                                                  x_out_test[:, t + 2])
                    self.model.smoother_gain_list.append(self.model.SGain.clone())
                #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
                    P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
                self.model.smoother_gain_list.append(s_0.clone().detach())  # [m, m]

                # Now, run the analytical RTS backward pass for the covariance
                P_smoothed_seq = torch.empty(m, m, SysModel.T_test)
                P_smoothed_seq[:, :, -1] = P_filtered_seq[:, :, -1]  # P_T|T is the last filtered P

                for t in range(SysModel.T_test - 2, -1, -1):
                    # Get necessary matrices for this time step
                    P_filt_t = P_filtered_seq[:, :, t]
                    P_pred_t_plus_1 = P_predicted_seq[:, :, t + 1]
                    P_smooth_t_plus_1 = P_smoothed_seq[:, :, t + 1]

                    # Get the Smoother Gain from the network for this time step
                    reverse_time = SysModel.T_test - 1 - t
                    S_t = self.model.smoother_gain_list[reverse_time]  ####S[T-1] = S(0), S[0] = S(T-1)

                    # ANALYTICAL RTS UPDATE using the smoother gain from the network
                    P_smooth_t = P_filt_t + S_t @ (P_smooth_t_plus_1 - P_pred_t_plus_1) @ S_t.T
                    P_smoothed_seq[:, :, t] = P_smooth_t

                # Save the final results for this sequence
                x_out_list.append(x_out_test)
                P_smooth_list_analytical.append(P_smoothed_seq)

                # <<<Calculate and store the MSE for the j-th sequence >>>
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()
                #######compute V############
                V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq,
                                                   self.model.smoother_gain_list)
                V_list.append(V)  # [seq](tensor(m,m,T))

        # <<< Average the MSEs over all sequences and print the result >>>
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        print(f"Hybrid RTSNet - MSE Test: {self.MSE_test_dB_avg:.4f} [dB]")


        #
        # Return the full tensors of results
        return torch.stack(x_out_list), torch.stack(P_smooth_list_analytical), V_list

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, self.N_B, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

    def PlotTrain_RTS(self, MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_E, self.N_steps, self.N_B, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, self.MSE_test_linear_arr)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)




    def Train_Joint(self, SysModel, cv_input, cv_target, train_input, train_target, path_results_rtsnet,path_results_psmooth,load_rtsnet = None,load_psmooth = None,
                    generate_f=True,beta=0.0):
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        # Logging arrays
        self.MSE_train_rts_dB_epoch = torch.empty([self.N_steps])
        self.MSE_train_psmooth_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_rts_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_psmooth_dB_epoch = torch.empty([self.N_steps])
        self.MSE_train_total_dB_epoch = torch.empty([self.N_steps])
        self.MSE_cv_total_dB_epoch = torch.empty([self.N_steps])

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        if load_rtsnet is not None:
            self.model = torch.load(load_rtsnet, map_location=self.device, weights_only=False)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,weight_decay=self.weightDecay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,weight_decay=self.weightDecay)
        if load_psmooth != None:
            self.PsmoothNN = torch.load(load_psmooth, map_location=self.device, weights_only=False)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,weight_decay=self.weightDecay)
        else:
            self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
                                                        weight_decay=self.weightDecay)


        eps_f = SysModel.F_train[2]- SysModel.F_train_TRUE[2]
        eps = torch.linalg.norm(eps_f, ord='fro')
        print('initial diviation is' , eps,'the first',SysModel.F_train_TRUE[2], 'the second', SysModel.F_train[2] )

        F_train_copy = [F.clone() for F in SysModel.F_train]
        F_valid_copy = [F.clone() for F in SysModel.F_valid]

        for ti in range(0, self.N_steps):

            # Set both models to train mode
            self.model.train()
            self.PsmoothNN.train()

            # Zero gradients for both optimizers
            self.optimizer.zero_grad()
            self.PsmoothNN_optimizer.zero_grad()

            Batch_RTS_LOSS_sum = 0
            Batch_Psmooth_LOSS_sum = 0
            Batch_Total_LOSS_sum = 0
            F_loss_batch = 0


            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)
                if generate_f:
                    index = n_e // 10
                    SysModel.F = SysModel.F_train[index]
                    self.model.update_F(SysModel.F)
                if generate_h:
                    index = n_e // 10
                    SysModel.H = SysModel.H_train[index]
                    self.model.update_H(SysModel.H)

                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                # Run RTSNet forward and backward pass to get smoothed states and intermediate values
                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)

                self.model.init_hidden()
                self.model.InitSequence(SysModel.m1x_0, SysModel.T)

                sigma_list = []
                smoother_gain_list = []

                for t in range(SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                    sigma_list.append(self.model.h_Sigma.clone())  # We need to keep the graph attached
                    K_t = self.model.KGain.clone()### save the last one

                x_out_training[:, SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
                self.model.InitBackward(x_out_training[:, SysModel.T - 1])
                x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
                                                               x_out_training_forward[:, SysModel.T - 1], None)
                smoother_gain_list.append(self.model.SGain.clone())
                for t in range(SysModel.T - 3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t], x_out_training_forward[:, t + 1],
                                                      x_out_training[:, t + 2])
                    smoother_gain_list.append(self.model.SGain.clone())
                #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
                smoother_gain_list.append(s_0.clone())  # [m, m]

                # Run PsmoothNN using the stateless method
                P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T)
                dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)

                sigma_T = sigma_list[-1]
                # sigma_T_processed = self.PsmoothNN.FC8(sigma_T.view(1, -1)).view(1, 1, -1)
                # in_Psmooth_T = torch.cat((sigma_T_processed, dummy_sgain), dim=2)
                # h_current = in_Psmooth_T[:, :, :self.PsmoothNN.d_hidden_Psmooth].clone()
                self.PsmoothNN.start = 0
                P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
                P_smoothed_seq[:, :, SysModel.T - 1] = self.PsmoothNN.enforce_covariance_properties(
                    P_flat.view(SysModel.m, SysModel.m))

                for t in range(SysModel.T - 2, -1, -1):
                    sigma_t = sigma_list[t]
                    index = (SysModel.T - 2) - t
                    sgain_t = smoother_gain_list[index]
                    P_flat = self.PsmoothNN(sigma_t, sgain_t) # [1, 1, m²] and [1, 1, d_hidden_Psmooth]
                    P_smoothed_seq[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
                        P_flat.view(-1).view(SysModel.m, SysModel.m))
                ###################compute the M step for F#########################
                p_tilde_tensor = torch.empty(SysModel.m,SysModel.m, SysModel.T)
                for i,p_1 in enumerate(sigma_list):
                    p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
                    p_tilde_tensor[:,:,i] =   self.PsmoothNN.enforce_covariance_properties(p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
                V = self.compute_cross_covariances(SysModel.F,SysModel.H, K_t,P_smoothed_seq, smoother_gain_list)
                # 2) run M‑step on **this sequence** (batch of size 1)
                X_s = x_out_training.unsqueeze(0)  # → [1, m, T]
                P_smooth_s = P_smoothed_seq.unsqueeze(0)  # → [1, m, m, T]
                V_s = V.unsqueeze(0)  # → [1, m, m, T]
                n_state = SysModel.F.shape[0]

                F_est = EMKF_F_Mstep(SysModel,X_s, P_smooth_s, V_s, n_state)
                F_est = F_est[0]
                index = n_e // 10
                F_TRUE = SysModel.F_train_TRUE[index]
                if j == 0:
                    print('true',F_TRUE, 'F_false', F_est)

                eps_f = F_est -F_TRUE

                eps = torch.linalg.norm(eps_f, ord='fro')
                F_loss_batch += eps
                # Calculate the two separate losses
                rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])
                #option_1
                psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training)
                #option 2
                #psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq, train_target[n_e], x_out_training)
                # Combine them into a total loss
                # beta_change = beta/(ti/5+1)
                beta_change =0.8
                total_loss = beta_change*rtsnet_loss + (1-beta_change)* psmooth_loss
                # Accumulate for logging
                Batch_RTS_LOSS_sum += rtsnet_loss
                Batch_Psmooth_LOSS_sum += psmooth_loss
                Batch_Total_LOSS_sum += total_loss

            # Average losses for the batch
            old_weights = [p.clone() for p in self.PsmoothNN.parameters()]
            Total_LOSS_mean = Batch_Total_LOSS_sum / self.N_B
            RTSNET_LOSS_mean = Batch_RTS_LOSS_sum / self.N_B
            Psmooth_LOSS_mean = Batch_Psmooth_LOSS_sum / self.N_B
            F_loss_mean = F_loss_batch/self.N_B
            Total_LOSS_mean = Total_LOSS_mean
            print('F_loss is:', F_loss_mean)
            # Backward pass on the combined loss
            Total_LOSS_mean.backward(retain_graph=True)
            # Clip gradients and step both optimizers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # bach_to_p = Psmooth_LOSS_mean
            # bach_to_p.backward()
            torch.nn.utils.clip_grad_norm_(self.PsmoothNN.parameters(), max_norm=1.0)
            self.PsmoothNN_optimizer.step()

            # Log training losses
            self.MSE_train_rts_dB_epoch[ti] = 10 * torch.log10(Batch_RTS_LOSS_sum / self.N_B)
            self.MSE_train_psmooth_dB_epoch[ti] = 10 * torch.log10(Batch_Psmooth_LOSS_sum / self.N_B)
            self.MSE_train_total_dB_epoch[ti] = 10 * torch.log10(Batch_Total_LOSS_sum / self.N_B)
            # Validation#####################################################
            self.model.eval()
            self.PsmoothNN.eval()
            with ((torch.no_grad())):
                CV_RTS_LOSS_sum = 0
                CV_Psmooth_LOSS_sum = 0
                CV_Total_LOSS_sum = 0
                F_loss_batch_cv = 0
                for j in range(self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    if generate_f:
                        index = j // 10
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)
                    if generate_h:
                        index = j // 10
                        SysModel.H = SysModel.H_valid[index]
                        self.model.update_H(SysModel.H)

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)

                    sigma_list_cv, smoother_gain_list_cv = [], []
                    for t in range(SysModel.T_test):
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                        sigma_list_cv.append(self.model.h_Sigma)
                        K_t = self.model.KGain.clone()  ### save the last one

                    x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                    x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],
                                                                  x_out_cv_forward[:, SysModel.T_test - 1], None)
                    smoother_gain_list_cv.append(self.model.SGain.clone())
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],
                                                    x_out_cv[:, t + 2])
                        smoother_gain_list_cv.append(self.model.SGain.clone())
                    #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
                    P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
                    s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
                        P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
                    smoother_gain_list_cv.append(s_0.clone())  # [m, m]

                    P_smoothed_seq_cv = torch.empty(SysModel.m, SysModel.m, SysModel.T_test)
                    dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
                    sigma_T_cv = sigma_list_cv[-1]
                    self.PsmoothNN.start = 0

                    P_flat_cv = self.PsmoothNN(sigma_T_cv, dummy_sgain)  # shape: [1, 1, m²] to [m²]
                    P_smoothed_seq_cv[:, :, SysModel.T_test - 1] = self.PsmoothNN.enforce_covariance_properties(P_flat_cv.view(-1).view(SysModel.m, SysModel.m))

                    for t in range(SysModel.T_test - 2, -1, -1):
                        sigma_t_cv = sigma_list_cv[t]
                        index = (SysModel.T_test - 2) - t
                        sgain_t_cv = smoother_gain_list_cv[index]
                        P_flat_cv = self.PsmoothNN(sigma_t_cv, sgain_t_cv)
                        P_smoothed_seq_cv[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
                            P_flat_cv.view(-1).view(SysModel.m, SysModel.m))

                    ###################compute the M step for F#########################
                    p_tilde_tensor = torch.empty(SysModel.m, SysModel.m, SysModel.T)
                    for i, p_1 in enumerate(sigma_list_cv):
                        p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
                        p_tilde_tensor[:, :, i] = self.PsmoothNN.enforce_covariance_properties(
                            p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
                    V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_smoothed_seq_cv,
                                                       smoother_gain_list_cv)
                    # 2) run M‑step on **this sequence** (batch of size 1)
                    X_s = x_out_cv.unsqueeze(0)  # → [1, m, T]
                    P_smooth_s = P_smoothed_seq_cv.unsqueeze(0)  # → [1, m, m, T]
                    V_s = V.unsqueeze(0)  # → [1, m, m, T]
                    n_state = SysModel.F.shape[0]

                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, n_state)
                    F_est = F_est[0]
                    index = j // 10
                    F_TRUE = SysModel.F_valid_TRUE[index]
                    eps_f = F_est - F_TRUE

                    eps = torch.linalg.norm(eps_f, ord='fro')
                    F_loss_batch_cv += eps

                    # option_1
                    psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()
                    # option 2
                    #psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()


                    CV_RTS_LOSS_sum += self.loss_fn(x_out_cv, cv_target[j]).item()
                    CV_Psmooth_LOSS_sum += psmooth_loss

                    CV_Total_LOSS_sum += beta_change*self.loss_fn(x_out_cv, cv_target[j]).item() + (1 - beta_change)* psmooth_loss

                f_loss = F_loss_batch_cv/self.N_CV
                CV_Total_LOSS_sum =  0.6*CV_Total_LOSS_sum / self.N_CV + 0.4*f_loss
                self.MSE_cv_rts_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_RTS_LOSS_sum / self.N_CV))
                self.MSE_cv_psmooth_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Psmooth_LOSS_sum / self.N_CV))
                self.MSE_cv_total_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Total_LOSS_sum))
                print('cv f loss is:', f_loss)


                # Save best models based on the main RTSNet validation loss
                if self.MSE_cv_total_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_total_dB_epoch[ti]
                    self.MSE_cv_rts_dB_opt = self.MSE_cv_rts_dB_epoch[ti]
                    self.MSE_cv_psmooth_dB_opt = self.MSE_cv_psmooth_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, path_results_rtsnet)
                    torch.save(self.PsmoothNN, path_results_psmooth)
                    print(f"**** Best Models Saved at Epoch {ti} with CV Loss {self.MSE_cv_dB_opt:.4f} dB ****")

            # --- Comprehensive Logging ---
            # Get current epoch's values for printing
            train_rts_loss = self.MSE_train_rts_dB_epoch[ti]
            train_psmooth_loss = self.MSE_train_psmooth_dB_epoch[ti]
            train_total_loss = self.MSE_train_total_dB_epoch[ti]

            cv_rts_loss = self.MSE_cv_rts_dB_epoch[ti]
            cv_psmooth_loss = self.MSE_cv_psmooth_dB_epoch[ti]
            cv_total_loss = self.MSE_cv_total_dB_epoch[ti]

            # Use an f-string for clean, aligned printing
            print(f"Epoch {ti:03d}/{self.N_steps - 1} | "
                  f"TRAIN: [RTS: {train_rts_loss:8.3f}, PSmooth: {train_psmooth_loss:8.3f}, Total: {train_total_loss:8.3f}] dB | "
                  f"CV: [RTS: {cv_rts_loss:8.3f}, PSmooth: {cv_psmooth_loss:8.3f}, Total: {cv_total_loss:8.3f}] dB | "
                  f"BEST Total: {self.MSE_cv_dB_opt:8.3f} dB (at epoch {self.MSE_cv_idx_opt})")

        # After all epochs are done, return the logged histories for plotting
        return



    def compute_gaussian_loss1(self,Sigma,x_target,x_est):
        err = x_target - x_est  # [m, T]
        m, T = err.shape
        eps = 1e-6
        eye_m = torch.eye(m, device=err.device, dtype=err.dtype) * eps

        total = 0.0
        const = 0.5 * m * torch.log(torch.tensor(2 * torch.pi, device=err.device))

        for t in range(T):
            Σ = Sigma[:, :, t] + eye_m  # jitter for PD
            δ = err[:, t].unsqueeze(1)  # [m,1]

            Σ_inv = torch.inverse(Σ)  # explicit inverse
            maha = (δ.transpose(0, 1) @ Σ_inv @ δ).squeeze()

            sign, logdet = torch.slogdet(Σ)  # numerically stable log-det
            if sign <= 0:
                # in case numerical issues lead to non-PD
                logdet = torch.log(torch.det(Σ) + eps)

            total += maha + logdet + 2 * const  # note we’ll divide by 2T below

        return total / (2 * T)



    def compute_gaussian_loss(self,P_seq,  # [m, m, T]  – predicted covariances
                              x_target_seq,  # [m, T]      – ground–truth states
                              x_est_seq):  # [m, T]      – RTSNet state output
        """
        Negative log-likelihood  (up to the additive constant ½·m·log(2π))
        ℓ_t = (x_t − μ_t)ᵀ P_t^{-1} (x_t − μ_t) + log |P_t|
        averaged over the T time steps.
        """
        m, T = x_target_seq.shape
        eps = 1e-5  # keeps P positive-definite numerically
        total = 0.0

        for t in range(T):
            P_t = P_seq[:, :, t] + torch.eye(m, device=P_seq.device) * eps
            δ = (x_target_seq[:, t] - x_est_seq[:, t]).unsqueeze(1)  # [m,1]

            # ---------- Mahalanobis term without explicit inverse ----------
            # L is lower-triangular s.t. P_t = L Lᵀ  (Cholesky factorisation)
            L = torch.linalg.cholesky(P_t)
            # Solve L Lᵀ α = δ   → α = P_t^{-1} δ
            α = torch.cholesky_solve(δ, L)  # same size as δ
            mahal = δ.T @ α  # scalar

            # ---------- log-det term (log |P_t|) -----------
            # log |P_t| = 2·Σ log diag(L)
            log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))

            total += mahal + log_det

        return total / T/2  # average over the sequence

    # def Train_Joint(self, SysModel, cv_input, cv_target, train_input, train_target, path_results_rtsnet,
    #                 path_results_psmooth, load_rtsnet=None, load_psmooth=None,
    #                 generate_f=True, beta=0.0):
    #     self.N_E = len(train_input)
    #     self.N_CV = len(cv_input)
    #
    #     # Logging arrays
    #     self.MSE_train_rts_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_train_psmooth_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_cv_rts_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_cv_psmooth_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_train_total_dB_epoch = torch.empty([self.N_steps])
    #     self.MSE_cv_total_dB_epoch = torch.empty([self.N_steps])
    #
    #     self.MSE_cv_dB_opt = 1000
    #     self.MSE_cv_idx_opt = 0
    #
    #     if load_rtsnet is not None:
    #         self.model = torch.load(load_rtsnet, weights_only=False)
    #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
    #                                           weight_decay=self.weightDecay)
    #     else:
    #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
    #                                           weight_decay=self.weightDecay)
    #     if load_psmooth != None:
    #         self.PsmoothNN = torch.load(load_psmooth, weights_only=False)
    #         # Re-link the optimizer to the parameters of the newly loaded model
    #         self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
    #                                                     weight_decay=self.weightDecay)
    #     else:
    #         self.PsmoothNN_optimizer = torch.optim.Adam(self.PsmoothNN.parameters(), lr=self.learningRate,
    #                                                     weight_decay=self.weightDecay)
    #
    #     eps_f = SysModel.F_train[2] - SysModel.F_train_TRUE[2]
    #     eps = torch.linalg.norm(eps_f, ord='fro')
    #     print('initial diviation is', eps, 'the first', SysModel.F_train_TRUE[2], 'the second', SysModel.F_train[2])
    #
    #     F_train_copy = [F.clone() for F in SysModel.F_train]
    #     F_valid_copy = [F.clone() for F in SysModel.F_valid]
    #
    #     for ti in range(0, self.N_steps):
    #
    #         # Set both models to train mode
    #         self.model.train()
    #         self.PsmoothNN.train()
    #
    #         # Zero gradients for both optimizers
    #         self.optimizer.zero_grad()
    #         self.PsmoothNN_optimizer.zero_grad()
    #
    #         Batch_RTS_LOSS_sum = 0
    #         Batch_Psmooth_LOSS_sum = 0
    #         Batch_Total_LOSS_sum = 0
    #         F_loss_batch = []
    #
    #         for j in range(0, self.N_B):
    #             n_e = random.randint(0, self.N_E - 1)
    #             if generate_f:
    #                 index = n_e // 10
    #                 SysModel.F = SysModel.F_train[index]
    #                 self.model.update_F(SysModel.F)
    #
    #             y_training = train_input[n_e]
    #             SysModel.T = y_training.size()[-1]
    #
    #             # Run RTSNet forward and backward pass to get smoothed states and intermediate values
    #             x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
    #             x_out_training = torch.empty(SysModel.m, SysModel.T)
    #
    #             self.model.init_hidden()
    #             self.model.InitSequence(SysModel.m1x_0, SysModel.T)
    #
    #             sigma_list = []
    #             smoother_gain_list = []
    #
    #             for t in range(SysModel.T):
    #                 x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
    #                 sigma_list.append(self.model.h_Sigma.clone())  # We need to keep the graph attached
    #                 K_t = self.model.KGain.clone()  ### save the last one
    #
    #             x_out_training[:, SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
    #             self.model.InitBackward(x_out_training[:, SysModel.T - 1])
    #             x_out_training[:, SysModel.T - 2] = self.model(None, x_out_training_forward[:, SysModel.T - 2],
    #                                                            x_out_training_forward[:, SysModel.T - 1], None)
    #             smoother_gain_list.append(self.model.SGain.clone())
    #             for t in range(SysModel.T - 3, -1, -1):
    #                 x_out_training[:, t] = self.model(None, x_out_training_forward[:, t],
    #                                                   x_out_training_forward[:, t + 1],
    #                                                   x_out_training[:, t + 2])
    #                 smoother_gain_list.append(self.model.SGain.clone())
    #             #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
    #             P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
    #             s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
    #                 P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
    #             smoother_gain_list.append(s_0.clone())  # [m, m]
    #
    #             # Run PsmoothNN using the stateless method
    #             P_smoothed_seq = torch.empty(SysModel.m, SysModel.m, SysModel.T)
    #             dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)
    #
    #             sigma_T = sigma_list[-1]
    #             # sigma_T_processed = self.PsmoothNN.FC8(sigma_T.view(1, -1)).view(1, 1, -1)
    #             # in_Psmooth_T = torch.cat((sigma_T_processed, dummy_sgain), dim=2)
    #             # h_current = in_Psmooth_T[:, :, :self.PsmoothNN.d_hidden_Psmooth].clone()
    #             self.PsmoothNN.start = 0
    #             P_flat = self.PsmoothNN(sigma_T, dummy_sgain).view(-1)  # shape: [1, 1, m²] to [m²]
    #             P_smoothed_seq[:, :, SysModel.T - 1] = self.PsmoothNN.enforce_covariance_properties(
    #                 P_flat.view(SysModel.m, SysModel.m))
    #
    #             for t in range(SysModel.T - 2, -1, -1):
    #                 sigma_t = sigma_list[t]
    #                 index = (SysModel.T - 2) - t
    #                 sgain_t = smoother_gain_list[index]
    #                 P_flat = self.PsmoothNN(sigma_t, sgain_t)  # [1, 1, m²] and [1, 1, d_hidden_Psmooth]
    #                 P_smoothed_seq[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
    #                     P_flat.view(-1).view(SysModel.m, SysModel.m))
    #             ###################compute the M step for F#########################
    #             p_tilde_tensor = torch.empty(SysModel.m, SysModel.m, SysModel.T)
    #             for i, p_1 in enumerate(sigma_list):
    #                 p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
    #                 p_tilde_tensor[:, :, i] = self.PsmoothNN.enforce_covariance_properties(
    #                     p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
    #             V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, p_tilde_tensor, smoother_gain_list)
    #             # 2) run M‑step on **this sequence** (batch of size 1)
    #             X_s = x_out_training.unsqueeze(0)  # → [1, m, T]
    #             P_smooth_s = P_smoothed_seq.unsqueeze(0)  # → [1, m, m, T]
    #             V_s = V.unsqueeze(0)  # → [1, m, m, T]
    #             n_state = SysModel.F.shape[0]
    #
    #             F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, n_state)
    #             F_est = F_est[0]
    #             index = n_e // 10
    #             F_TRUE = SysModel.F_train_TRUE[index]
    #             eps_f = F_est - F_TRUE
    #
    #             eps = torch.linalg.norm(eps_f, ord='fro')
    #             F_loss_batch.append(eps)
    #             # Calculate the two separate losses
    #             rtsnet_loss = self.loss_fn(x_out_training, train_target[n_e])
    #             # option_1
    #             psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq, train_target[n_e], x_out_training)
    #             # option 2
    #             # psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq, train_target[n_e], x_out_training)
    #             # Combine them into a total loss
    #             # beta_change = beta/(ti/5+1)
    #             beta_change = 0.6
    #             total_loss = beta_change * rtsnet_loss + (1 - beta_change) * psmooth_loss
    #             # Accumulate for logging
    #             Batch_RTS_LOSS_sum += rtsnet_loss
    #             Batch_Psmooth_LOSS_sum += psmooth_loss
    #             Batch_Total_LOSS_sum += total_loss
    #
    #         # Average losses for the batch
    #         old_weights = [p.clone() for p in self.PsmoothNN.parameters()]
    #         Total_LOSS_mean = Batch_Total_LOSS_sum / self.N_B
    #         RTSNET_LOSS_mean = Batch_RTS_LOSS_sum / self.N_B
    #         Psmooth_LOSS_mean = Batch_Psmooth_LOSS_sum / self.N_B
    #         F_loss_mean = torch.stack(F_loss_batch).mean()
    #         Total_LOSS_mean = Total_LOSS_mean * 1 + F_loss_mean * 0
    #         print('F_loss is:', F_loss_mean)
    #         # Backward pass on the combined loss
    #         Total_LOSS_mean.backward(retain_graph=True)
    #         # Clip gradients and step both optimizers
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #         self.optimizer.step()
    #         # bach_to_p = Psmooth_LOSS_mean
    #         # bach_to_p.backward()
    #         torch.nn.utils.clip_grad_norm_(self.PsmoothNN.parameters(), max_norm=1.0)
    #         self.PsmoothNN_optimizer.step()
    #
    #         # Log training losses
    #         self.MSE_train_rts_dB_epoch[ti] = 10 * torch.log10(Batch_RTS_LOSS_sum / self.N_B)
    #         self.MSE_train_psmooth_dB_epoch[ti] = 10 * torch.log10(Batch_Psmooth_LOSS_sum / self.N_B)
    #         self.MSE_train_total_dB_epoch[ti] = 10 * torch.log10(Batch_Total_LOSS_sum / self.N_B)
    #         # Validation#####################################################
    #         self.model.eval()
    #         self.PsmoothNN.eval()
    #         with ((torch.no_grad())):
    #             CV_RTS_LOSS_sum = 0
    #             CV_Psmooth_LOSS_sum = 0
    #             CV_Total_LOSS_sum = 0
    #             F_loss_batch_cv = []
    #             for j in range(self.N_CV):
    #                 y_cv = cv_input[j]
    #                 SysModel.T_test = y_cv.size()[-1]
    #
    #                 if generate_f:
    #                     index = j // 10
    #                     SysModel.F = SysModel.F_valid[index]
    #                     self.model.update_F(SysModel.F)
    #
    #                 x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
    #                 x_out_cv = torch.empty(SysModel.m, SysModel.T_test)
    #                 self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
    #
    #                 sigma_list_cv, smoother_gain_list_cv = [], []
    #                 for t in range(SysModel.T_test):
    #                     x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
    #                     sigma_list_cv.append(self.model.h_Sigma)
    #                     K_t = self.model.KGain.clone()  ### save the last one
    #
    #                 x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
    #                 self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
    #                 x_out_cv[:, SysModel.T_test - 2] = self.model(None, x_out_cv_forward[:, SysModel.T_test - 2],
    #                                                               x_out_cv_forward[:, SysModel.T_test - 1], None)
    #                 smoother_gain_list_cv.append(self.model.SGain.clone())
    #                 for t in range(SysModel.T_test - 3, -1, -1):
    #                     x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t + 1],
    #                                                 x_out_cv[:, t + 2])
    #                     smoother_gain_list_cv.append(self.model.SGain.clone())
    #                 #  P_1_0_predicted = F @ P_0_0 @ F.T + Q
    #                 P_1_0_pred = SysModel.F @ SysModel.m2x_0 @ SysModel.F.T + SysModel.Q
    #                 s_0 = SysModel.m2x_0 @ SysModel.F.T @ torch.inverse(
    #                     P_1_0_pred.view(SysModel.m, SysModel.m))  ######COMPUTE S_0
    #                 smoother_gain_list_cv.append(s_0.clone())  # [m, m]
    #
    #                 P_smoothed_seq_cv = torch.empty(SysModel.m, SysModel.m, SysModel.T_test)
    #                 dummy_sgain = torch.zeros(1, 1, SysModel.m * SysModel.m)  # shape: [1, 1, m²] input to PsmoothNN
    #                 sigma_T_cv = sigma_list_cv[-1]
    #                 self.PsmoothNN.start = 0
    #
    #                 P_flat_cv = self.PsmoothNN(sigma_T_cv, dummy_sgain)  # shape: [1, 1, m²] to [m²]
    #                 P_smoothed_seq_cv[:, :, SysModel.T_test - 1] = self.PsmoothNN.enforce_covariance_properties(
    #                     P_flat_cv.view(-1).view(SysModel.m, SysModel.m))
    #
    #                 for t in range(SysModel.T_test - 2, -1, -1):
    #                     sigma_t_cv = sigma_list_cv[t]
    #                     index = (SysModel.T_test - 2) - t
    #                     sgain_t_cv = smoother_gain_list_cv[index]
    #                     P_flat_cv = self.PsmoothNN(sigma_t_cv, sgain_t_cv)
    #                     P_smoothed_seq_cv[:, :, t] = self.PsmoothNN.enforce_covariance_properties(
    #                         P_flat_cv.view(-1).view(SysModel.m, SysModel.m))
    #
    #                 ###################compute the M step for F#########################
    #                 p_tilde_tensor = torch.empty(SysModel.m, SysModel.m, SysModel.T)
    #                 for i, p_1 in enumerate(sigma_list_cv):
    #                     p_1 = p_1.view(4, 4).mean(dim=1)  # shape: (4,)
    #                     p_tilde_tensor[:, :, i] = self.PsmoothNN.enforce_covariance_properties(
    #                         p_1.view(SysModel.m, SysModel.m), eps=1e-6)  # tensor  (n×n×T)
    #                 V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, p_tilde_tensor,
    #                                                    smoother_gain_list_cv)
    #                                                    smoother_gain_list_cv)
    #                 # 2) run M‑step on **this sequence** (batch of size 1)
    #                 X_s = x_out_cv.unsqueeze(0)  # → [1, m, T]
    #                 P_smooth_s = P_smoothed_seq_cv.unsqueeze(0)  # → [1, m, m, T]
    #                 V_s = V.unsqueeze(0)  # → [1, m, m, T]
    #                 n_state = SysModel.F.shape[0]
    #
    #                 F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, V_s, n_state)
    #                 F_est = F_est[0]
    #                 index = j // 10
    #                 F_TRUE = SysModel.F_valid_TRUE[index]
    #                 eps_f = F_est - F_TRUE
    #
    #                 eps = torch.linalg.norm(eps_f, ord='fro')
    #                 F_loss_batch_cv.append(eps)
    #
    #                 # option_1
    #                 psmooth_loss = self.PsmoothNN.compute_loss(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()
    #                 # option 2
    #                 # psmooth_loss = self.compute_gaussian_loss1(P_smoothed_seq_cv, cv_target[j], x_out_cv).item()
    #
    #                 CV_RTS_LOSS_sum += self.loss_fn(x_out_cv, cv_target[j]).item()
    #                 CV_Psmooth_LOSS_sum += psmooth_loss
    #
    #                 CV_Total_LOSS_sum += beta_change * self.loss_fn(x_out_cv, cv_target[j]).item() + (
    #                             1 - beta_change) * psmooth_loss
    #
    #             f_loss = torch.stack(F_loss_batch_cv).mean()
    #             CV_Total_LOSS_sum = 0.5 * CV_Total_LOSS_sum / self.N_CV + 0 * f_loss
    #             self.MSE_cv_rts_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_RTS_LOSS_sum / self.N_CV))
    #             self.MSE_cv_psmooth_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Psmooth_LOSS_sum / self.N_CV))
    #             self.MSE_cv_total_dB_epoch[ti] = 10 * torch.log10(torch.tensor(CV_Total_LOSS_sum))
    #             print('cv f loss is:', f_loss)
    #
    #             # Save best models based on the main RTSNet validation loss
    #             if self.MSE_cv_total_dB_epoch[ti] < self.MSE_cv_dB_opt:
    #                 self.MSE_cv_dB_opt = self.MSE_cv_total_dB_epoch[ti]
    #                 self.MSE_cv_rts_dB_opt = self.MSE_cv_rts_dB_epoch[ti]
    #                 self.MSE_cv_psmooth_dB_opt = self.MSE_cv_psmooth_dB_epoch[ti]
    #                 self.MSE_cv_idx_opt = ti
    #                 torch.save(self.model, path_results_rtsnet)
    #                 torch.save(self.PsmoothNN, path_results_psmooth)
    #                 print(f"**** Best Models Saved at Epoch {ti} with CV Loss {self.MSE_cv_dB_opt:.4f} dB ****")
    #
    #         # --- Comprehensive Logging ---
    #         # Get current epoch's values for printing
    #         train_rts_loss = self.MSE_train_rts_dB_epoch[ti]
    #         train_psmooth_loss = self.MSE_train_psmooth_dB_epoch[ti]
    #         train_total_loss = self.MSE_train_total_dB_epoch[ti]
    #
    #         cv_rts_loss = self.MSE_cv_rts_dB_epoch[ti]
    #         cv_psmooth_loss = self.MSE_cv_psmooth_dB_epoch[ti]
    #         cv_total_loss = self.MSE_cv_total_dB_epoch[ti]
    #
    #         # Use an f-string for clean, aligned printing
    #         print(f"Epoch {ti:03d}/{self.N_steps - 1} | "
    #               f"TRAIN: [RTS: {train_rts_loss:8.3f}, PSmooth: {train_psmooth_loss:8.3f}, Total: {train_total_loss:8.3f}] dB | "
    #               f"CV: [RTS: {cv_rts_loss:8.3f}, PSmooth: {cv_psmooth_loss:8.3f}, Total: {cv_total_loss:8.3f}] dB | "
    #               f"BEST Total: {self.MSE_cv_dB_opt:8.3f} dB (at epoch {self.MSE_cv_idx_opt})")
    #
    #     # After all epochs are done, return the logged histories for plotting
    #     return

    def _run_rtsnet_sequence(self, SysModel, y_seq, model_index, x0=None, p0=None,non_linear_h = False):
        """
        Run RTSNet forward and backward pass for a single sequence.
        """
        T = y_seq.size()[-1]
        m = SysModel.m
        dev = y_seq.device
        dt = y_seq.dtype

        # Initialize
        x_out_forward = torch.empty(m, T,device=dev, dtype=dt)
        x_out_smoothed = torch.empty(m, T,device=dev, dtype=dt)

        if x0 is not None:
            # 1) set prior Sigma (P0) for THIS sequence
            x0 = x0.to(dev, dtype=dt)
            p0 = self.psmooth_models[0].enforce_covariance_properties(p0)
            p0_use = p0.to(dev, dtype=dt)
            self.rtsnet_models[model_index].prior_Sigma = p0_use
            # 3) set mean/x0 and T (does NOT touch GRUs)
            self.rtsnet_models[model_index].InitSequence(x0, T)
            # 2) reset GRU hiddens so h_Sigma seeds from prior_Sigma == P0
            self.rtsnet_models[model_index].init_hidden()

        else:
            p0_use = SysModel.m2x_0
            self.rtsnet_models[model_index].prior_Sigma = SysModel.m2x_0
            self.rtsnet_models[model_index].InitSequence(SysModel.m1x_0, T)
            self.rtsnet_models[model_index].init_hidden()
        sigma_list = []
        smoother_gain_list = []

        # Forward pass
        for t in range(T):
            x_out_forward[:, t] = self.rtsnet_models[model_index](y_seq[:, t], None, None, None)
            sigma_list.append(self.rtsnet_models[model_index].h_Sigma.clone())
            if t == T-1:
                K_t = self.rtsnet_models[model_index].KGain.clone()
                H_last = None
                if non_linear_h == True:
                    # ADD (immediately after the loop ends, before the backward pass):
                    x_last = x_out_forward[:, T - 1].view(m, 1)
                    # make sure getJacobian(SysModel.h) is in scope; import if needed
                    with torch.enable_grad():
                        H_last = getJacobian(x_last, SysModel.h)
                        _jacobian_watchdog(H_last, x_last, SysModel.h)

        # Backward pass
        x_out_smoothed[:, T - 1] = x_out_forward[:, T - 1]


        self.rtsnet_models[model_index].InitBackward(x_out_smoothed[:, T - 1])
        x_out_smoothed[:, T - 2] = self.rtsnet_models[model_index](None, x_out_forward[:, T - 2],
                                                                   x_out_forward[:, T - 1], None)
        smoother_gain_list.append(self.rtsnet_models[model_index].SGain.clone())

        for t in range(T - 3, -1, -1):
            x_out_smoothed[:, t] = self.rtsnet_models[model_index](None, x_out_forward[:, t],
                                                                   x_out_forward[:, t + 1],
                                                                   x_out_smoothed[:, t + 2])
            smoother_gain_list.append(self.rtsnet_models[model_index].SGain.clone())

        # Run PsmoothNet
        P_smoothed_seq = torch.empty(m, m, T, device=dev, dtype=dt)
        dummy_sgain = torch.zeros(1, 1, m * m, device=dev, dtype=dt)

        # Final time step
        sigma_T = sigma_list[-1]
        self.psmooth_models[model_index].start = 0
        P_flat = self.psmooth_models[model_index](sigma_T, dummy_sgain).view(-1)
        P_smoothed_seq[:, :, T - 1] = self.psmooth_models[model_index].enforce_covariance_properties(
            P_flat.view(m, m))

        # Backward in time
        for t in range(T - 2, -1, -1):
            sigma_t = sigma_list[t]
            sgain_index = (T - 2) - t
            sgain_t = smoother_gain_list[sgain_index].reshape(1, 1, -1)
            P_flat = self.psmooth_models[model_index](sigma_t, sgain_t)
            P_smoothed_seq[:, :, t] = self.psmooth_models[model_index].enforce_covariance_properties(
                P_flat.view(-1).view(m, m))

        # Compute S_0
        P_1_0_pred = SysModel.F @ p0_use@ SysModel.F.T + SysModel.Q
        s_0 = p0_use@ SysModel.F.T @ torch.inverse(P_1_0_pred.view(m, m))# ori return this and delet the tow row down
        smoother_gain_list.append(s_0.clone())

        # Extract filtered covariances
        P_filtered_seq = torch.empty(m, m, T, device=dev, dtype=dt)
        # for i, sigma in enumerate(sigma_list):
        #     sigma_processed = sigma.view(4, 4).mean(dim=1)
        #     P_filtered_seq[:, :, i] = self.psmooth_models[model_index].enforce_covariance_properties(
        #         sigma_processed.view(SysModel.m, SysModel.m), eps=1e-6)

        return x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t, H_last

    def Test_Only_EMKF(self, SysModel, test_input, test_target,
                       load_base_rtsnet=None, load_base_psmooth=None, emkf_iterations=3, generate_f=True,
                       init_x_list=None, init_P_list=None,non_linear_h =False):
        """
        Test-only version - No training, no optimization, just run EMKF on test data
        """

        # Initialize multiple models
        self.rtsnet_models = []
        self.psmooth_models = []

        for i in range(emkf_iterations):
            rtsnet_model = torch.load(load_base_rtsnet[i], map_location=self.device, weights_only=False).to(self.device)
            self.rtsnet_models.append(rtsnet_model)

            psmooth_model = torch.load(load_base_psmooth[i], map_location=self.device, weights_only=False).to(
                self.device)
            self.psmooth_models.append(psmooth_model)

        print(f"Starting Test-Only EMKF with {emkf_iterations} EM iterations")

        # Run test only
        test_losses, test_f_losses, final_F_list, x_last, p_last,final_F_list2 = self._run_test_simple(SysModel, test_input,
                                  test_target, emkf_iterations, generate_f=generate_f, init_x_list=init_x_list, init_P_list=init_P_list,non_linear_h=non_linear_h)

        # # Compute weighted total test loss
        # iteration_weights = [0.1, 0.2, 0.7]
        # total_test_loss = sum(w * loss for w, loss in zip(iteration_weights, test_losses))
        # total_test_f_loss = sum(w * loss for w, loss in zip(iteration_weights, test_f_losses))

        # # LOGGING
        # test_loss_db = 10 * torch.log10(total_test_loss.detach())
        #
        # print(f"TEST RESULTS:")
        # print(f"  Total Loss: {test_loss_db:.3f} dB")
        # print(f"  F-loss: {total_test_f_loss:.6f}")

        # Log individual iteration losses
        for i, test_loss in enumerate(test_losses):
            print(f"    Iter {i}: {10 * torch.log10(test_loss.detach()):.2f}dB")

        return test_losses, test_f_losses, final_F_list, x_last, p_last, final_F_list2

    def _run_test_simple(self, SysModel, input_data, target_data, emkf_iterations, generate_f=True, init_x_list=None,
                         init_P_list=None, non_linear_h = None):
        """
        Simple test - just loop through each sequence one by one
        """
        N_data = len(input_data)

        final_F_list = [None] * N_data
        final_F_list2 = [None] * N_data
        # Initialize F matrices
        F_current = [f.clone().detach() for f in SysModel.F_test]
        F_true = SysModel.F_test_TRUE

        # Accumulate losses
        all_iter_losses = [[] for _ in range(emkf_iterations)]
        all_iter_f_losses = [[] for _ in range(emkf_iterations)]

        x_last_all = []
        p_last_all = []
        # SIMPLE LOOP - one sequence at a time
        for seq_idx in range(N_data):
            y_seq = input_data[seq_idx]
            target_seq = target_data[seq_idx]
            f_index = seq_idx

            if (init_x_list is not None) and (init_P_list is not None):
                P0 = init_P_list[seq_idx]
                x0 = init_x_list[seq_idx]
            else:
                P0 = SysModel.m2x_0
                x0 = SysModel.m1x_0

            if generate_f == True:
                f_index = seq_idx // 10
            # Store losses for each EM iteration for this sequence
            seq_iter_losses = []
            seq_iter_f_losses = []

            # Start with F
            F_seq = F_current[f_index].clone().detach()

            # EM ITERATIONS for this sequence
            for em_iter in range(emkf_iterations):
                # Set model modes - EVAL ONLY
                self.rtsnet_models[em_iter].eval()
                self.psmooth_models[em_iter].eval()

                # Use current F
                SysModel.F = F_seq
                self.rtsnet_models[em_iter].update_F(SysModel.F)

                # E-STEP: Run networks
                with torch.no_grad():

                    x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t, H_last = \
                        self._run_rtsnet_sequence(SysModel, y_seq, em_iter, x0, P0,non_linear_h=non_linear_h)

                    # Compute losses
                    rts_loss = self.loss_fn(x_out_smoothed, target_seq)
                    psmooth_loss = self.psmooth_models[em_iter].compute_loss(P_smoothed_seq, target_seq, x_out_smoothed)
                    total_seq_loss = 1 * rts_loss + 0 * psmooth_loss
                    seq_iter_losses.append(total_seq_loss)

                    # Debug print for first sequence
                    if seq_idx== 0:
                        print(f'EM iter: {em_iter}, loss: {total_seq_loss:.4f}')
                        print(f'F_seq: {F_seq}')


                    # M-STEP
                    if non_linear_h == True:
                        V = self.compute_cross_covariances(SysModel.F,H_last, K_t, P_smoothed_seq,
                                                           smoother_gain_list)
                    else:
                        V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_smoothed_seq, smoother_gain_list)
                    X_s = x_out_smoothed.unsqueeze(0)
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)
                    list_V_s = []
                    list_V_s.append(V)
                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, list_V_s, SysModel.m)[0]

                    deltaF = (F_est - F_seq).norm()
                    if seq_idx == 0:
                        print(f"[EM {em_iter}] ||ΔF|| = {deltaF.item():.3e}")
                    F_seq = F_est

                    # F loss
                    F_true_seq = F_true[f_index]
                    f_loss = torch.linalg.norm(F_seq.detach() - F_true_seq, ord='fro')
                    seq_iter_f_losses.append(f_loss)

                    if em_iter == (emkf_iterations-2):
                        final_F_list2[seq_idx] = F_seq.detach().clone()

                    if seq_idx % 5 == 0:
                        print(f'EM iter: {em_iter}, avg loss: {rts_loss:.4f}, avg f loss: {f_loss:.6f}')



            ##########################add a last computation for after the F computation
            # --- after the for em_iter in range(emkf_iterations) loop ---

            # # Use the final F_seq (after the last M-step)
            # SysModel.F = F_seq
            #
            # # Pick a model index to run the forward; reusing the last one is fine
            # last_idx = emkf_iterations - 1
            # self.rtsnet_models[last_idx].eval()
            # self.psmooth_models[last_idx].eval()
            #
            # with torch.no_grad():
            #     x_out_forward2, x_out_smoothed2, P_smoothed_seq2, P_filtered_seq2, smoother_gain_list2, K_t2 = \
            #         self._run_rtsnet_sequence(SysModel, y_seq, last_idx)
            #     final_seq_loss = self.loss_fn(x_out_smoothed2, target_seq)
            #
            # # Append the “after K updates” loss as the (K)-th index (i.e., emkf_iterations)
            # seq_iter_losses.append(final_seq_loss)
            #
            # # Optional: F-distance for the final F
            # F_true_seq = F_true[f_index]
            # final_f_loss = torch.linalg.norm(F_seq.detach() - F_true_seq, ord='fro')
            # seq_iter_f_losses.append(final_f_loss)

            ################################################################

            # terminals from the last EM iteration run above
            x_last_all.append(x_out_smoothed[:, -1].unsqueeze(-1).detach().clone())  # [m,1]
            p_last_all.append(P_smoothed_seq[:, :, -1].detach().clone())  # [m,m]

            final_F_list[seq_idx] = F_seq.detach().clone()

            # Add this sequence's losses to accumulators
            for em_iter in range(emkf_iterations):
                all_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                all_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])

        # Compute final averages
        final_iter_losses = []
        final_iter_f_losses = []

        for em_iter in range(emkf_iterations):
            if all_iter_losses[em_iter]:
                iter_avg_loss = torch.stack(all_iter_losses[em_iter]).mean()
                iter_avg_f_loss = torch.stack(all_iter_f_losses[em_iter]).mean()

            else:
                print('wrongggggggggggggggggggggggggggggggggggggggg')

            final_iter_losses.append(iter_avg_loss)
            final_iter_f_losses.append(iter_avg_f_loss)

        return final_iter_losses, final_iter_f_losses, final_F_list, x_last_all, p_last_all, final_F_list2

    def _run_sequential_emkf_epoch(self, SysModel, input_data, target_data, emkf_iterations, is_training,non_linear_h = False):
        """
        FIXED VERSION - Run one epoch with BATCH → SEQUENCE → EM order and F propagation.
        """
        N_data = len(input_data)

        # FIX 1: Correct batch calculation
        if is_training:
            N_batches = self.N_B  # Training: 400//10 = 40 batches
        else:
            N_batches = self.N_B  # Validation: ceil(100/10) = 10 batches

        # Initialize F matrices
        if is_training:
            F_current = [f.clone().detach() for f in SysModel.F_train]  # FIX: Detach initial F
            F_true = SysModel.F_train_TRUE
        else:
            F_current = [f.clone().detach() for f in SysModel.F_valid]  # FIX: Detach initial F
            F_true = SysModel.F_valid_TRUE

        # Accumulate all iteration losses across all batches
        all_iter_losses = [[] for _ in range(emkf_iterations)]
        all_iter_f_losses = [[] for _ in range(emkf_iterations)]

        # BATCH LOOP (outermost)
        for batch_idx in range(N_batches):
            # Get batch indices
            if is_training:
                batch_indices = [random.randint(0, N_data - 1) for _ in range(self.N_B)]
                for i in range(emkf_iterations):
                    self.rtsnet_optimizers[i].zero_grad()
                    self.psmooth_optimizers[i].zero_grad()
            else:
                start_idx = batch_idx * self.N_B
                end_idx = min(start_idx + self.N_B, N_data)
                batch_indices = list(range(start_idx, end_idx))


            # Make copy of F for this batch (so F updates happen within batch)
            F_batch = [f.clone() for f in F_current]

            # Track batch losses for printing
            batch_iter_losses = [[] for _ in range(emkf_iterations)]
            batch_iter_f_losses = [[] for _ in range(emkf_iterations)]
            last_iter = emkf_iterations - 1

            # SEQUENCE LOOP (middle) - Process each sequence in the batch
            for seq_idx in batch_indices:
                y_seq = input_data[seq_idx]
                target_seq = target_data[seq_idx]
                f_index = seq_idx // 10

                # Store losses for each EM iteration for this sequence
                seq_iter_losses = []
                seq_iter_f_losses = []

                # FIX 2: Start with stable F initialization
                F_seq = F_batch[f_index].clone().detach().requires_grad_(True)  # Ensure no gradients from previous sequences

                # EM ITERATIONS LOOP (innermost) - Sequential F updates for this sequence
                for em_iter in range(emkf_iterations):
                    # Set model modes
                    if is_training:
                        self.rtsnet_models[em_iter].train()
                        self.psmooth_models[em_iter].train()

                    # # FIX 3: Ensure F has gradients for this iteration only
                    # if is_training:
                    #     F_seq = F_seq.detach().requires_grad_(True)
                    # else:
                    #     F_seq = F_seq.detach()

                    # Use current F (updated from previous EM iteration)
                    SysModel.F = F_seq
                    self.rtsnet_models[em_iter].update_F(SysModel.F)
                    # E-STEP: Run networks
                    x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t, H_last = \
                        self._run_rtsnet_sequence(SysModel, y_seq, em_iter, non_linear_h=non_linear_h)
##########################################################################################################################################################
                    #ori just return this
                    #
                    # # Compute losses
                    # rts_loss = self.loss_fn(x_out_smoothed, target_seq)
                    # psmooth_loss = self.psmooth_models[em_iter].compute_loss(P_smoothed_seq, target_seq,
                    #                                                          x_out_smoothed)
                    # total_seq_loss = 0.9 * rts_loss + 0.1 * psmooth_loss
                    # seq_iter_losses.append(total_seq_loss)

                    ###########################################################################################
                    # Compute losses
                    rts_loss = self.loss_fn(x_out_smoothed, target_seq)

                    # --- CHANGED: do NOT backprop psmooth_loss for the last iteration ---
                    if em_iter == last_iter:
                        # we only want psmooth_3 to be trained by F-loss, so its "state" loss is for logging only
                        with torch.no_grad():
                            psmooth_loss = self.psmooth_models[em_iter].compute_loss(
                                P_smoothed_seq, target_seq, x_out_smoothed
                            )
                    else:
                        psmooth_loss = self.psmooth_models[em_iter].compute_loss(
                            P_smoothed_seq, target_seq, x_out_smoothed
                        )

                    total_seq_loss = 0.9 * rts_loss + 0.1 * psmooth_loss
                    seq_iter_losses.append(total_seq_loss)



##########################################################################################################################################################



                    ####################################################################################################################################################ori del
                    # # ---- after you compute total_seq_loss = 0.9*rts_loss + 0.1*psmooth_loss ----
                    #
                    # # print F if this sequence's loss in this EM iteration is worse than -7 dB
                    # HIGH_LOSS_DB = -7.0
                    # loss_db = 10.0 * torch.log10(
                    #     total_seq_loss.detach().clamp_min(torch.tensor(1e-12, device=total_seq_loss.device)))
                    #
                    # if loss_db.item() > HIGH_LOSS_DB:
                    #     # spectral radius (useful context)
                    #     def _spectral_radius(F):
                    #         vals = torch.linalg.eigvals(F)
                    #         return vals.abs().max().real.item()
                    #
                    #     rho = _spectral_radius(F_seq)
                    #
                    #     # identify context, then print the F that produced this loss
                    #     print(f"[HIGH-LOSS] batch= seq={seq_idx} em_iter={em_iter} "
                    #           f"loss={loss_db.item():.2f} dB  rho(F)={rho:.4f}  ||F||F={torch.linalg.norm(F_seq).item():.4f}")
                    #     print(F_seq.detach())  # F that was used for this E-step

                    ####################################################################################################################################################





                    # Debug print for first sequence in first batch
                    if batch_idx == 0 and seq_idx == batch_indices[0]:
                        print(f'EM iter: {em_iter}, loss: {total_seq_loss:.4f}')
                        print(f'F_seq: {F_seq}')

                    # FIX 4: Stable M-STEP with regularization
                    # print(SysModel.F)

                    if non_linear_h ==True:
                        V = self.compute_cross_covariances(SysModel.F, H_last, K_t, P_smoothed_seq,
                                                           smoother_gain_list)
                    else:
                        V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_smoothed_seq,smoother_gain_list)
                    X_s = x_out_smoothed.unsqueeze(0)
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)
                    list_V_s = []
                    list_V_s.append(V)

                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, list_V_s, SysModel.m)[0]
                    F_seq = F_est
                    # F loss
                    F_true_seq = F_true[f_index]
                    f_loss = torch.linalg.norm(F_seq - F_true_seq, ord='fro')
                    seq_iter_f_losses.append(f_loss)

                # Add this sequence's losses to both global and batch accumulators
                for em_iter in range(emkf_iterations):
                    all_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                    all_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])
                    batch_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                    batch_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])

                # ADDED: Batch-level optimization
            if is_training:
                # Compute batch losses
                batch_losses = []
                F_LOSS_WEIGHT = 0.
                for em_iter in range(emkf_iterations):
                    if batch_iter_losses[em_iter]:
                        batch_avg_state_loss = torch.stack(batch_iter_losses[em_iter]).mean()
#################################################################################################################
                        if em_iter == last_iter:
                            # 3rd iteration: state loss only, NO F-loss in Part 1
                            batch_avg_loss = batch_avg_state_loss
                        else:
                            # Iter 0 and 1: add small F-loss
                            batch_avg_f_loss = torch.stack(batch_iter_f_losses[em_iter]).mean()
                            batch_avg_loss = batch_avg_state_loss + F_LOSS_WEIGHT * batch_avg_f_loss

#################################################################################################################
                        # batch_avg_f_loss = torch.stack(batch_iter_f_losses[em_iter]).mean()
                        # batch_avg_loss = batch_avg_state_loss + F_LOSS_WEIGHT * batch_avg_f_loss
#################################################################################################################
                        batch_losses.append(batch_avg_loss)
                    else:
                        print('problemmmmmmmmmmmmmmmmmmmmmmmm')

                    # Compute weighted total batch loss
                iteration_weights = [0.01, 0.1, 0.89]
                total_batch_loss = sum(w * loss for w, loss in zip(iteration_weights, batch_losses))


                if torch.isfinite(total_batch_loss) and total_batch_loss < 1000.0:

                    # batch-mean F-loss for EM-iter = last_iter (3rd iteration)
                    ps3_f_loss = torch.stack(batch_iter_f_losses[last_iter]).mean()
                    # zero ONLY Psmooth_3 grads
                    self.psmooth_optimizers[last_iter].zero_grad()
                    # backprop this F-loss: only Psmooth_3 has useful grads here
                    ps3_f_loss.backward(retain_graph=True)
                    # Clip & STEP Psmooth[2] ONLY (Psmooth 3)
                    torch.nn.utils.clip_grad_norm_(self.psmooth_models[last_iter].parameters(), max_norm=0.5)
                    self.psmooth_optimizers[last_iter].step()

                    for i in range(emkf_iterations):
                        self.rtsnet_optimizers[i].zero_grad()
                        self.psmooth_optimizers[i].zero_grad()
                    total_batch_loss.backward()
                    # Do we get gradient in iter-0 params when using only iter-2 loss?
                    nz0 = sum(
                        (p.grad is not None) and (p.grad.abs().sum() > 0) for p in self.rtsnet_models[0].parameters())
                    print("nonzero grads in iter-0 params:", bool(nz0))

                    ###################################


                    for i in range(emkf_iterations):
                        torch.nn.utils.clip_grad_norm_(self.rtsnet_models[i].parameters(), max_norm=0.5)
                        torch.nn.utils.clip_grad_norm_(self.psmooth_models[i].parameters(), max_norm=0.5)
                        self.rtsnet_optimizers[i].step()
                        if i != last_iter:
                            self.psmooth_optimizers[i].step()
                else:
                    print(
                        f"WARNING: Skipping backward pass for batch {batch_idx} due to invalid loss: {total_batch_loss}")

            # BATCH LOSS PRINTING (only every 5 batches to reduce spam)
            if batch_idx % 2 == 0 or batch_idx == N_batches - 1:
                mode_str = "TRAIN" if is_training else "VALID"
                print(f"  {mode_str} Batch {batch_idx + 1}/{N_batches}:")
                for em_iter in range(emkf_iterations):
                    if batch_iter_losses[em_iter]:  # Check if batch has losses
                        batch_avg_loss = torch.stack(batch_iter_losses[em_iter]).mean()
                        batch_avg_f_loss = torch.stack(batch_iter_f_losses[em_iter]).mean()
                        batch_loss_db = 10 * torch.log10(batch_avg_loss.detach())
                        print(f"    EM-Iter {em_iter}: Loss={batch_loss_db:.2f}dB, F-loss={batch_avg_f_loss:.4f}")
                    else:
                        print(f"    EM-Iter {em_iter}: No data processed")

            # Update global F with final F from this batch
            for f_idx, f_val in enumerate(F_batch):
                F_current[f_idx] = f_val.detach().clone()

        # Compute final averages across all sequences for each iteration
        final_iter_losses = []
        final_iter_f_losses = []

        for em_iter in range(emkf_iterations):
            if all_iter_losses[em_iter]:  # This check prevents torch.stack([]) error
                iter_avg_loss = torch.stack(all_iter_losses[em_iter]).mean()
                iter_avg_f_loss = torch.stack(all_iter_f_losses[em_iter]).mean()
            else:
                # Fallback values when no losses were collected
                iter_avg_loss = torch.tensor(1000.0, device=self.device)  # High loss indicates problem
                iter_avg_f_loss = torch.tensor(0.0, device=self.device)  # Zero F-loss as neutral value

            final_iter_losses.append(iter_avg_loss)
            final_iter_f_losses.append(iter_avg_f_loss)

        return final_iter_losses, final_iter_f_losses

    def Train_EndToEnd_EMKF(self, SysModel, cv_input, cv_target, train_input, train_target,
                                  rtsnet_model_paths, psmooth_model_paths, emkf_iterations=3,
                                  load_base_rtsnet=None, load_base_psmooth=None,non_linear_h = False):
        """
        FIXED VERSION - Main training function for end-to-end EMKF training.
        """

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        # Initialize multiple models
        self.rtsnet_models = []
        self.psmooth_models = []

        for i in range(emkf_iterations):
            rtsnet_model = torch.load(load_base_rtsnet, map_location=self.device, weights_only=False).to(self.device)
            self.rtsnet_models.append(rtsnet_model)

            psmooth_model = torch.load(load_base_psmooth, map_location=self.device, weights_only=False).to(self.device)
            self.psmooth_models.append(psmooth_model)

        # Create separate optimizers with LOWER learning rate for stability
        self.rtsnet_optimizers = []
        self.psmooth_optimizers = []

        # FIX 7: Reduce learning rate for stability
        stable_lr = self.learningRate   # 10x smaller learning rate

        for i in range(emkf_iterations):
            self.rtsnet_optimizers.append(torch.optim.Adam(self.rtsnet_models[i].parameters(), lr=stable_lr,
                                                           weight_decay=self.weightDecay))
            self.psmooth_optimizers.append(torch.optim.Adam(self.psmooth_models[i].parameters(), lr=stable_lr,
                                                            weight_decay=self.weightDecay))

        # Logging arrays
        self.MSE_train_total_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_total_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.F_loss_train_epoch = torch.empty([self.N_steps], device=self.device)
        self.F_loss_cv_epoch = torch.empty([self.N_steps], device=self.device)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        print(f"Starting FIXED End-to-End EMKF Training with {emkf_iterations} EM iterations")
        print(f"Using reduced learning rate: {stable_lr}")

        # MAIN EPOCH LOOP
        for epoch in range(self.N_steps):

             # TRAINING
            train_losses, train_f_losses = self._run_sequential_emkf_epoch(SysModel, train_input, train_target,
                                                                           emkf_iterations, is_training=True, non_linear_h=non_linear_h )

            # FIX 8: Check for NaN/Inf losses
            valid_train_losses = []
            for loss in train_losses:
                if torch.isfinite(loss):
                    valid_train_losses.append(loss)
                else:
                    print(f"WARNING: Invalid loss detected: {loss}, skipping")
                    valid_train_losses.append(torch.tensor(1000.0))  # High penalty
            train_losses = valid_train_losses

            # Compute weighted total training loss
            iteration_weights = [0.01, 0.1, 0.89]
            total_train_loss = sum(w * loss for w, loss in zip(iteration_weights, train_losses))

            # # FIX 9: Gradient clipping BEFORE backprop
            # if torch.isfinite(total_train_loss) and total_train_loss < 1000.0:
            #     # BACKPROPAGATION
            #     total_train_loss.backward()
            #
            #     # OPTIMIZER STEP with more aggressive clipping
            #     for i in range(emkf_iterations):
            #         torch.nn.utils.clip_grad_norm_(self.rtsnet_models[i].parameters(), max_norm=0.5)  # Reduced from 1.0
            #         torch.nn.utils.clip_grad_norm_(self.psmooth_models[i].parameters(),
            #                                        max_norm=0.5)  # Reduced from 1.0
            #         self.rtsnet_optimizers[i].step()
            #         self.psmooth_optimizers[i].step()
            # else:
            #     print(f"WARNING: Skipping backward pass due to invalid loss: {total_train_loss}")

            # VALIDATION
            with torch.no_grad():
                for i in range(emkf_iterations):
                    self.rtsnet_models[i].eval()
                    self.psmooth_models[i].eval()

                cv_losses, cv_f_losses = self._run_sequential_emkf_epoch(SysModel, cv_input, cv_target, emkf_iterations,
                                                                         is_training=False,non_linear_h=non_linear_h)

            # Compute weighted averages
            total_cv_loss = sum(w * loss for w, loss in zip(iteration_weights, cv_losses))
            total_train_f_loss = sum(w * loss for w, loss in zip(iteration_weights, train_f_losses))
            total_cv_f_loss = sum(w * loss for w, loss in zip(iteration_weights, cv_f_losses))

            # LOGGING
            self.MSE_train_total_dB_epoch[epoch] = 10 * torch.log10(total_train_loss.detach())
            self.MSE_cv_total_dB_epoch[epoch] = 10 * torch.log10(total_cv_loss.detach())
            self.F_loss_train_epoch[epoch] = total_train_f_loss.detach()
            self.F_loss_cv_epoch[epoch] = total_cv_f_loss.detach()

            # SAVE BEST MODELS
            if self.MSE_cv_total_dB_epoch[epoch] < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = self.MSE_cv_total_dB_epoch[epoch]
                self.MSE_cv_idx_opt = epoch

                for i in range(emkf_iterations):
                    torch.save(self.rtsnet_models[i], rtsnet_model_paths[i])
                    torch.save(self.psmooth_models[i], psmooth_model_paths[i])

                print(f"**** Best Models Saved at Epoch {epoch} with CV Loss {self.MSE_cv_dB_opt:.4f} dB ****")

            # PROGRESS PRINTING (every epoch)
            print(f"Epoch {epoch:03d}/{self.N_steps - 1} | ")
            print(
                f"  TRAIN: [Total: {self.MSE_train_total_dB_epoch[epoch]:.3f}, F-loss: {self.F_loss_train_epoch[epoch]:.6f}]")
            print(
                f"  CV:    [Total: {self.MSE_cv_total_dB_epoch[epoch]:.3f}, F-loss: {self.F_loss_cv_epoch[epoch]:.6f}]")
            print(f"  BEST:  {self.MSE_cv_dB_opt:.3f} dB (epoch {self.MSE_cv_idx_opt})")

            # Log individual iteration losses
            for i, (train_loss, cv_loss) in enumerate(zip(train_losses, cv_losses)):
                print(
                    f"    Iter {i}: Train={10 * torch.log10(train_loss.detach()):.2f}dB, CV={10 * torch.log10(cv_loss.detach()):.2f}dB")

        return [self.MSE_train_total_dB_epoch, self.MSE_cv_total_dB_epoch, self.F_loss_train_epoch,
                self.F_loss_cv_epoch]

######################################################################

    def _run_rtsnet_sequence_P(self, SysModel, y_seq, model_index, x0=None, p0=None, non_linear_h=False):
        """
        Run RTSNet forward + backward on a single sequence,
        then produce:
          - P_not_smooth_t via PNotSmoothNN(F_t, K_t, P_prev)
          - P_smooth_t via PsmoothFromPnot(P_not_t, S_t)

        Returns:
          x_out_forward        : [m, T]
          x_out_smoothed       : [m, T]
          P_smoothed_seq       : [m, m, T]
          P_not_seq            : [m, m, T]     # (returned in place of filtered covs)
          smoother_gain_list   : list of length (T-1)+1 with s_0 appended at the end
          K_t_list             : list of T tensors [m, n]
          H_last               : tensor or None
        """
        T = y_seq.size()[-1]
        m = SysModel.m
        dev = y_seq.device
        dt = y_seq.dtype

        # Initialize
        x_out_forward = torch.empty(m, T, device=dev, dtype=dt)
        x_out_smoothed = torch.empty(m, T, device=dev, dtype=dt)

        if x0 is not None:
            # 1) set prior Sigma (P0) for THIS sequence
            x0 = x0.to(dev, dtype=dt)
            p0 = enforce_covariance_properties(p0)
            p0_use = p0.to(dev, dtype=dt)
            self.rtsnet_models[model_index].prior_Sigma = p0_use
            self.rtsnet_models[model_index].InitSequence(x0, T)
            self.rtsnet_models[model_index].init_hidden()

        else:
            p0_use = SysModel.m2x_0
            self.rtsnet_models[model_index].prior_Sigma = SysModel.m2x_0
            self.rtsnet_models[model_index].InitSequence(SysModel.m1x_0, T)
            self.rtsnet_models[model_index].init_hidden()

        # --- INIT PNot for this sequence (even in else branch) ---
        pnot = self.pnot_models[model_index]
        pnot.reset_state()
        pnot.p_0 = p0_use.clone()

        K_t_list = []
        smoother_gain_list = []

        # Forward pass
        for t in range(T):
            x_out_forward[:, t] = self.rtsnet_models[model_index](y_seq[:, t], None, None, None)
            K_t_list.append(self.rtsnet_models[model_index].KGain.clone())
            if t == T - 1:
                H_last = None
                if non_linear_h == True:
                    # ADD (immediately after the loop ends, before the backward pass):
                    x_last = x_out_forward[:, T - 1].view(m, 1)
                    # make sure getJacobian(SysModel.h) is in scope; import if needed
                    with torch.enable_grad():
                        H_last = getJacobian(x_last, SysModel.h)
                        _jacobian_watchdog(H_last, x_last, SysModel.h)

        # ----------------- Backward pass (collect S in time order) -----------------
        x_out_smoothed[:, T - 1] = x_out_forward[:, T - 1]

        self.rtsnet_models[model_index].InitBackward(x_out_smoothed[:, T - 1])
        x_out_smoothed[:, T - 2] = self.rtsnet_models[model_index](None, x_out_forward[:, T - 2],
                                                                   x_out_forward[:, T - 1], None)
        smoother_gain_list.append(self.rtsnet_models[model_index].SGain.clone())

        for t in range(T - 3, -1, -1):
            x_out_smoothed[:, t] = self.rtsnet_models[model_index](None, x_out_forward[:, t],
                                                                   x_out_forward[:, t + 1],
                                                                   x_out_smoothed[:, t + 2])
            smoother_gain_list.append(self.rtsnet_models[model_index].SGain.clone())

        # --- P_not via PNotSmoothNN (no sigma_list) ---
        P_not_seq = torch.empty(m, m, T, device=dev, dtype=dt)
        P_prev = p0_use.clone()
        for t in range(T):
            K_t = K_t_list[t].detach()
            P_not_t = pnot(K_t, P_prev)
            P_not_seq[:, :, t] = enforce_covariance_properties(P_not_t)
            P_prev = P_not_t.detach()

        # --- P_smooth via PsmoothFromPnot (use your existing smoother_gain_list order) ---
        psfp = self.psfp_models[model_index]
        psfp.reset_state()
        P_smoothed_seq = torch.empty(m, m, T, device=dev, dtype=dt)
        # 1) Final step: P_smooth[T-1] = P_not[T-1]
        P_smoothed_seq[:, :, T - 1] = P_not_seq[:, :, T - 1]
        self.psfp_models[model_index].start = 0
        # Backward in time
        for t in range(T - 2, -1, -1):
            sgain_index = (T - 2) - t
            sgain_t = smoother_gain_list[sgain_index].reshape(1, 1, -1)
            P_smoothed_seq[:, :, t] = psfp(P_not_seq[:, :, t].view(-1), sgain_t)
        # Compute S_0
        P_1_0_pred = SysModel.F @ p0_use @ SysModel.F.T + SysModel.Q
        s_0 = p0_use @ SysModel.F.T @ torch.inverse(P_1_0_pred.view(m, m))  # ori return this and delet the tow row down
        smoother_gain_list.append(s_0.clone())


        return x_out_forward, x_out_smoothed, P_smoothed_seq, P_not_seq, smoother_gain_list, K_t_list[-1], H_last



    def _run_sequential_emkf_epoch_rts_pnot_psfp(self, SysModel, input_data, target_data, emkf_iterations, is_training,
                                   non_linear_h=False):
        """
        FIXED VERSION - Run one epoch with BATCH → SEQUENCE → EM order and F propagation.
        """
        N_data = len(input_data)


        N_batches = self.N_B  # Validation: ceil(100/10) = 10 batches

        # Initialize F matrices
        if is_training:
            F_current = [f.clone().detach() for f in SysModel.F_train]  # FIX: Detach initial F
            F_true = SysModel.F_train_TRUE
        else:
            F_current = [f.clone().detach() for f in SysModel.F_valid]  # FIX: Detach initial F
            F_true = SysModel.F_valid_TRUE

        # Accumulate all iteration losses across all batches
        all_iter_losses = [[] for _ in range(emkf_iterations)]
        all_iter_f_losses = [[] for _ in range(emkf_iterations)]

        # BATCH LOOP (outermost)
        for batch_idx in range(N_batches):
            # Get batch indices
            if is_training:
                batch_indices = [random.randint(0, N_data - 1) for _ in range(self.N_B)]
                for i in range(emkf_iterations):
                    self.rtsnet_optimizers[i].zero_grad()
                    self.pnot_optimizers[i].zero_grad()
                    self.psfp_optimizers[i].zero_grad()
            else:
                start_idx = batch_idx * self.N_B
                end_idx = min(start_idx + self.N_B, N_data)
                batch_indices = list(range(start_idx, end_idx))

            # Make copy of F for this batch (so F updates happen within batch)
            F_batch = [f.clone() for f in F_current]

            # Track batch losses for printing
            batch_iter_losses = [[] for _ in range(emkf_iterations)]
            batch_iter_f_losses = [[] for _ in range(emkf_iterations)]

            # SEQUENCE LOOP (middle) - Process each sequence in the batch
            for seq_idx in batch_indices:
                y_seq = input_data[seq_idx]
                target_seq = target_data[seq_idx]
                f_index = seq_idx // 10

                # Store losses for each EM iteration for this sequence
                seq_iter_losses = []
                seq_iter_f_losses = []
                seq_iter_losses_with_F = []

                # FIX 2: Start with stable F initialization
                F_seq = F_batch[f_index].clone().detach().requires_grad_(True)  # Ensure no gradients from previous sequences
                # EM ITERATIONS LOOP (innermost) - Sequential F updates for this sequence
                for em_iter in range(emkf_iterations):
                    # Set model modes
                    if is_training:
                        self.rtsnet_models[em_iter].train()
                        self.pnot_models[em_iter].train()
                        self.psfp_models[em_iter].train()

                    # Use current F (updated from previous EM iteration)
                    SysModel.F = F_seq
                    self.rtsnet_models[em_iter].update_F(SysModel.F)
                    self.pnot_models[em_iter].F = SysModel.F
                    # E-STEP: Run networks
                    x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t, H_last = \
                        self._run_rtsnet_sequence_P(SysModel, y_seq, em_iter, non_linear_h=non_linear_h)

                    # 3) Losses (IMPORTANT: build P_true internally from x)
                    rts_loss = self.loss_fn(x_out_smoothed, target_seq)
                    pnot_loss = self.pnot_models[em_iter].compute_loss(P_filtered_seq, target_seq,x_out_forward)
                    psfp_loss = self.psfp_models[em_iter].compute_loss(P_smoothed_seq, target_seq,x_out_smoothed)

                    # combine (tune weights as you like)
                    total_seq_loss = 0.8 * rts_loss + 0.1 * pnot_loss + 0.1 * psfp_loss
                    seq_iter_losses.append(total_seq_loss)




                    ####################################################################################################################################################ori del
                    # # ---- after you compute total_seq_loss = 0.9*rts_loss + 0.1*psmooth_loss ----
                    #
                    # # print F if this sequence's loss in this EM iteration is worse than -7 dB
                    # HIGH_LOSS_DB = -7.0
                    # loss_db = 10.0 * torch.log10(
                    #     total_seq_loss.detach().clamp_min(torch.tensor(1e-12, device=total_seq_loss.device)))
                    #
                    # if loss_db.item() > HIGH_LOSS_DB:
                    #     # spectral radius (useful context)
                    #     def _spectral_radius(F):
                    #         vals = torch.linalg.eigvals(F)
                    #         return vals.abs().max().real.item()
                    #
                    #     rho = _spectral_radius(F_seq)
                    #
                    #     # identify context, then print the F that produced this loss
                    #     print(f"[HIGH-LOSS] batch= seq={seq_idx} em_iter={em_iter} "
                    #           f"loss={loss_db.item():.2f} dB  rho(F)={rho:.4f}  ||F||F={torch.linalg.norm(F_seq).item():.4f}")
                    #     print(F_seq.detach())  # F that was used for this E-step

                    ####################################################################################################################################################

                    # Debug print for first sequence in first batch
                    if batch_idx == 0 and seq_idx == batch_indices[0]:
                        print(f'EM iter: {em_iter}, loss: {total_seq_loss:.4f}')
                        print(f'F_seq: {F_seq}')

                    # FIX 4: Stable M-STEP with regularization
                    # print(SysModel.F)

                    if non_linear_h == True:
                        V = self.compute_cross_covariances(SysModel.F, H_last, K_t, P_filtered_seq,smoother_gain_list)
                    else:
                        V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq,smoother_gain_list)
                    X_s = x_out_smoothed.unsqueeze(0)
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)
                    list_V_s = []
                    list_V_s.append(V)

                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, list_V_s, SysModel.m)[0]
                    F_seq = F_est
                    # F loss
                    F_true_seq = F_true[f_index]
                    f_loss = torch.linalg.norm(F_seq.detach() - F_true_seq, ord='fro')
                    seq_iter_f_losses.append(f_loss)

                    ####################################################################################
                    vals = torch.linalg.eigvals(F_est)
                    rho_tensor = vals.abs().max()
                    rho_target = 1.07  # desired max eigenvalue magnitude
                    alpha_rho = 1  # small weight, can tune
                    rho_value = rho_tensor.item()

                    if rho_value > rho_target:
                        rho_penalty = alpha_rho * (rho_tensor - rho_target)
                        print(f"[rho penalty] rho={rho_value:.3f} > {rho_target}, penalty={rho_penalty.item():.3e}")
                    else:
                        rho_penalty = 0.0 * rho_tensor  # tensor zero (keeps gradient path valid)

                    # NOW form the final loss for this EM iter & sequence:
                    total_seq_loss_with_F = total_seq_loss + rho_penalty
                    seq_iter_losses_with_F.append(total_seq_loss_with_F)

                    ####################################################################################

                # Add this sequence's losses to both global and batch accumulators
                for em_iter in range(emkf_iterations):
                    all_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                    all_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])
                    # batch_iter_losses[em_iter].append(seq_iter_losses[em_iter]) return if you dont want the f panelty
                    batch_iter_losses[em_iter].append(seq_iter_losses_with_F[em_iter])
                    batch_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])

                # ADDED: Batch-level optimization
            if is_training:
                # Compute batch losses
                batch_losses = []
                for em_iter in range(emkf_iterations):
                    if batch_iter_losses[em_iter]:
                        batch_avg_loss = torch.stack(batch_iter_losses[em_iter]).mean()
                        batch_losses.append(batch_avg_loss)
                    else:
                        print('problemmmmmmmmmmmmmmmmmmmmmmmm')

                    # Compute weighted total batch loss
                iteration_weights = [0.01, 0.1, 0.89]
                total_batch_loss = sum(w * loss for w, loss in zip(iteration_weights, batch_losses))   # ADD F_rho penalty to total loss

                if torch.isfinite(total_batch_loss) and total_batch_loss < 1000.0:

                    #################################### ori delete
                    # Right before total_batch_loss.backward()
                    print("F_seq.requires_grad:",
                          F_seq.requires_grad)  # should be True (for the last F_seq in the loop)
                    print("F_seq.grad_fn:", F_seq.grad_fn is not None)

                    total_batch_loss.backward(retain_graph=True)

                    # Do we get gradient in iter-0 params when using only iter-2 loss?
                    nz0 = sum(
                        (p.grad is not None) and (p.grad.abs().sum() > 0) for p in self.rtsnet_models[0].parameters())
                    print("nonzero grads in iter-0 params:", bool(nz0))

                    ###################################

                    # total_batch_loss.backward()

                    for i in range(emkf_iterations):
                        torch.nn.utils.clip_grad_norm_(self.rtsnet_models[i].parameters(), max_norm=0.5)
                        torch.nn.utils.clip_grad_norm_(self.pnot_models[i].parameters(), max_norm=0.5)
                        torch.nn.utils.clip_grad_norm_(self.psfp_models[i].parameters(), max_norm=0.5)
                        self.rtsnet_optimizers[i].step()
                        self.pnot_optimizers[i].step()
                        self.psfp_optimizers[i].step()
                else:
                    print(
                        f"WARNING: Skipping backward pass for batch {batch_idx} due to invalid loss: {total_batch_loss}")

            # BATCH LOSS PRINTING (only every 5 batches to reduce spam)
            if batch_idx % 2 == 0 or batch_idx == N_batches - 1:
                mode_str = "TRAIN" if is_training else "VALID"
                print(f"  {mode_str} Batch {batch_idx + 1}/{N_batches}:")
                for em_iter in range(emkf_iterations):
                    if batch_iter_losses[em_iter]:  # Check if batch has losses
                        batch_avg_loss = torch.stack(batch_iter_losses[em_iter]).mean()
                        batch_avg_f_loss = torch.stack(batch_iter_f_losses[em_iter]).mean()
                        batch_loss_db = 10 * torch.log10(batch_avg_loss.detach())
                        print(f"    EM-Iter {em_iter}: Loss={batch_loss_db:.2f}dB, F-loss={batch_avg_f_loss:.4f}")
                    else:
                        print(f"    EM-Iter {em_iter}: No data processed")

            # Update global F with final F from this batch
            for f_idx, f_val in enumerate(F_batch):
                F_current[f_idx] = f_val.detach().clone()

        # Compute final averages across all sequences for each iteration
        final_iter_losses = []
        final_iter_f_losses = []

        for em_iter in range(emkf_iterations):
            if all_iter_losses[em_iter]:  # This check prevents torch.stack([]) error
                iter_avg_loss = torch.stack(all_iter_losses[em_iter]).mean()
                iter_avg_f_loss = torch.stack(all_iter_f_losses[em_iter]).mean()
            else:
                # Fallback values when no losses were collected
                iter_avg_loss = torch.tensor(1000.0, device=self.device)  # High loss indicates problem
                iter_avg_f_loss = torch.tensor(0.0, device=self.device)  # Zero F-loss as neutral value

            final_iter_losses.append(iter_avg_loss)
            final_iter_f_losses.append(iter_avg_f_loss)

        return final_iter_losses, final_iter_f_losses

    def Train_EndToEnd_P_EMKF(self, SysModel, cv_input, cv_target, train_input, train_target,
                        rtsnet_model_paths, pnot_model_paths, psfp_model_paths, emkf_iterations=3,
                        load_base_rtsnet=None, load_base_pnot=None, load_base_psfp=None, non_linear_h=False):
        """
        FIXED VERSION - Main training function for end-to-end EMKF training.
        """

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        # Initialize multiple models
        self.rtsnet_models = []
        self.pnot_models = []
        self.psfp_models = []

        for i in range(emkf_iterations):
            rtsnet_model = torch.load(load_base_rtsnet, map_location=self.device, weights_only=False).to(self.device)
            self.rtsnet_models.append(rtsnet_model)

            pnot_model = torch.load(load_base_pnot, map_location=self.device, weights_only=False).to(self.device)
            self.pnot_models.append(pnot_model)

            psfp_model = torch.load(load_base_psfp, map_location=self.device, weights_only=False).to(self.device)
            self.psfp_models.append(psfp_model)

        # Create separate optimizers with LOWER learning rate for stability
        self.rtsnet_optimizers = []
        self.pnot_optimizers = []
        self.psfp_optimizers = []

        # FIX 7: Reduce learning rate for stability
        stable_lr = self.learningRate*0.1   # 10x smaller learning rate

        for i in range(emkf_iterations):
            self.rtsnet_optimizers.append(torch.optim.Adam(self.rtsnet_models[i].parameters(), lr=stable_lr,
                                                           weight_decay=self.weightDecay))
            self.pnot_optimizers.append(torch.optim.Adam(self.pnot_models[i].parameters(), lr=stable_lr,
                                                         weight_decay=self.weightDecay))
            self.psfp_optimizers.append(torch.optim.Adam(self.psfp_models[i].parameters(), lr=stable_lr,
                                                         weight_decay=self.weightDecay))

        # Logging arrays
        self.MSE_train_total_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_total_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.F_loss_train_epoch = torch.empty([self.N_steps], device=self.device)
        self.F_loss_cv_epoch = torch.empty([self.N_steps], device=self.device)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        print(f"Starting FIXED End-to-End EMKF Training with {emkf_iterations} EM iterations")
        print(f"Using reduced learning rate: {stable_lr}")

        # MAIN EPOCH LOOP
        for epoch in range(self.N_steps):

         # TRAINING
         train_losses, train_f_losses = self._run_sequential_emkf_epoch_rts_pnot_psfp(SysModel, train_input, train_target, emkf_iterations, is_training=True, non_linear_h=non_linear_h)


        # FIX 8: Check for NaN/Inf losses
        valid_train_losses = []
        for loss in train_losses:
            if torch.isfinite(loss):
                valid_train_losses.append(loss)
            else:
                print(f"WARNING: Invalid loss detected: {loss}, skipping")
                valid_train_losses.append(torch.tensor(1000.0))  # High penalty
        train_losses = valid_train_losses

        # Compute weighted total training loss
        iteration_weights = [0.01, 0.1, 0.89]
        total_train_loss = sum(w * loss for w, loss in zip(iteration_weights, train_losses))

        # FIX 9: Gradient clipping BEFORE backprop
        if torch.isfinite(total_train_loss) and total_train_loss < 1000.0:
            # BACKPROPAGATION
            total_train_loss.backward()

            # OPTIMIZER STEP with more aggressive clipping
            for i in range(emkf_iterations):
                torch.nn.utils.clip_grad_norm_(self.rtsnet_models[i].parameters(), max_norm=0.5)  # Reduced from 1.0
                torch.nn.utils.clip_grad_norm_(self.psmooth_models[i].parameters(),
                                               max_norm=0.5)  # Reduced from 1.0
                self.rtsnet_optimizers[i].step()
                self.psmooth_optimizers[i].step()
        else:
            print(f"WARNING: Skipping backward pass due to invalid loss: {total_train_loss}")

        # VALIDATION
        with torch.no_grad():
            for i in range(emkf_iterations):
                self.rtsnet_models[i].eval()
                self.pnot_models[i].eval()
                self.psfp_models[i].eval()

            cv_losses, cv_f_losses = self._run_sequential_emkf_epoch_rts_pnot_psfp(SysModel, cv_input, cv_target, emkf_iterations, is_training=False, non_linear_h=non_linear_h)

    # Compute weighted averages
        total_cv_loss = sum(w * loss for w, loss in zip(iteration_weights, cv_losses))
        total_train_f_loss = sum(w * loss for w, loss in zip(iteration_weights, train_f_losses))
        total_cv_f_loss = sum(w * loss for w, loss in zip(iteration_weights, cv_f_losses))

        # LOGGING
        self.MSE_train_total_dB_epoch[epoch] = 10 * torch.log10(total_train_loss.detach())
        self.MSE_cv_total_dB_epoch[epoch] = 10 * torch.log10(total_cv_loss.detach())
        self.F_loss_train_epoch[epoch] = total_train_f_loss.detach()
        self.F_loss_cv_epoch[epoch] = total_cv_f_loss.detach()

        # SAVE BEST MODELS
        if self.MSE_cv_total_dB_epoch[epoch] < self.MSE_cv_dB_opt:
            self.MSE_cv_dB_opt = self.MSE_cv_total_dB_epoch[epoch]
            self.MSE_cv_idx_opt = epoch

            for i in range(emkf_iterations):
                torch.save(self.rtsnet_models[i], rtsnet_model_paths[i])
                torch.save(self.pnot_models[i], pnot_model_paths[i])
                torch.save(self.psfp_models[i], psfp_model_paths[i])

            print(f"**** Best Models Saved at Epoch {epoch} with CV Loss {self.MSE_cv_dB_opt:.4f} dB ****")

        # PROGRESS PRINTING (every epoch)
        print(f"Epoch {epoch:03d}/{self.N_steps - 1} | ")
        print(
            f"  TRAIN: [Total: {self.MSE_train_total_dB_epoch[epoch]:.3f}, F-loss: {self.F_loss_train_epoch[epoch]:.6f}]")
        print(
            f"  CV:    [Total: {self.MSE_cv_total_dB_epoch[epoch]:.3f}, F-loss: {self.F_loss_cv_epoch[epoch]:.6f}]")
        print(f"  BEST:  {self.MSE_cv_dB_opt:.3f} dB (epoch {self.MSE_cv_idx_opt})")

        # Log individual iteration losses
        for i, (train_loss, cv_loss) in enumerate(zip(train_losses, cv_losses)):
            print(
                f"    Iter {i}: Train={10 * torch.log10(train_loss.detach()):.2f}dB, CV={10 * torch.log10(cv_loss.detach()):.2f}dB")

        return [self.MSE_train_total_dB_epoch, self.MSE_cv_total_dB_epoch, self.F_loss_train_epoch,
            self.F_loss_cv_epoch]


    def Test_Only_EMKF_p(self, SysModel, test_input, test_target,load_base_rtsnet=None, load_base_pnot=None, load_base_psfp=None,
                   emkf_iterations=3, generate_f=True,init_x_list=None, init_P_list=None, non_linear_h=False):
        """
        Test-only version - No training, no optimization, just run EMKF on test data
        """

        # Initialize multiple models
        self.rtsnet_models = []
        self.pnot_models = []
        self.psfp_models = []

        for i in range(emkf_iterations):
            rtsnet_model = torch.load(load_base_rtsnet[i], map_location=self.device, weights_only=False).to(self.device)
            self.rtsnet_models.append(rtsnet_model)

            # PNotSmoothNN
            pnot_model = torch.load(load_base_pnot[i], map_location=self.device, weights_only=False).to(self.device)
            self.pnot_models.append(pnot_model)

            # PsmoothFromPnot
            psfp_model = torch.load(load_base_psfp[i], map_location=self.device, weights_only=False).to(self.device)
            self.psfp_models.append(psfp_model)

        print(f"Starting Test-Only EMKF with {emkf_iterations} EM iterations")

        # Run test only
        test_losses, test_f_losses, final_F_list, x_last, p_last,final_F_list2 = self._run_test_simple_p(SysModel, test_input,
                                  test_target, emkf_iterations, generate_f=generate_f, init_x_list=init_x_list, init_P_list=init_P_list,non_linear_h=non_linear_h)

        # # Compute weighted total test loss
        # iteration_weights = [0.1, 0.2, 0.7]
        # total_test_loss = sum(w * loss for w, loss in zip(iteration_weights, test_losses))
        # total_test_f_loss = sum(w * loss for w, loss in zip(iteration_weights, test_f_losses))

        # # LOGGING
        # test_loss_db = 10 * torch.log10(total_test_loss.detach())
        #
        # print(f"TEST RESULTS:")
        # print(f"  Total Loss: {test_loss_db:.3f} dB")
        # print(f"  F-loss: {total_test_f_loss:.6f}")

        # Log individual iteration losses
        for i, test_loss in enumerate(test_losses):
            print(f"    Iter {i}: {10 * torch.log10(test_loss.detach()):.2f}dB")

        return test_losses, test_f_losses, final_F_list, x_last, p_last, final_F_list2

    def _run_test_simple_p(self, SysModel, input_data, target_data, emkf_iterations, generate_f=True, init_x_list=None,
                         init_P_list=None, non_linear_h = None):
        """
        Simple test - just loop through each sequence one by one
        """
        N_data = len(input_data)

        final_F_list = [None] * N_data
        final_F_list2 = [None] * N_data
        # Initialize F matrices
        F_current = [f.clone().detach() for f in SysModel.F_test]
        F_true = SysModel.F_test_TRUE

        # Accumulate losses
        all_iter_losses = [[] for _ in range(emkf_iterations)]
        all_iter_f_losses = [[] for _ in range(emkf_iterations)]

        x_last_all = []
        p_last_all = []
        # SIMPLE LOOP - one sequence at a time
        for seq_idx in range(N_data):
            y_seq = input_data[seq_idx]
            target_seq = target_data[seq_idx]
            f_index = seq_idx

            if (init_x_list is not None) and (init_P_list is not None):
                P0 = init_P_list[seq_idx]
                x0 = init_x_list[seq_idx]
            else:
                P0 = SysModel.m2x_0
                x0 = SysModel.m1x_0

            if generate_f == True:
                f_index = seq_idx // 10
            # Store losses for each EM iteration for this sequence
            seq_iter_losses = []
            seq_iter_f_losses = []

            # Start with F
            F_seq = F_current[f_index].clone().detach()

            # EM ITERATIONS for this sequence
            for em_iter in range(emkf_iterations):
                # Set model modes - EVAL ONLY
                self.rtsnet_models[em_iter].eval()
                self.pnot_models[em_iter].eval()
                self.psfp_models[em_iter].eval()

                # Use current F
                SysModel.F = F_seq
                self.rtsnet_models[em_iter].update_F(SysModel.F)
                self.pnot_models[em_iter].F = SysModel.F

                # E-STEP: Run networks
                with torch.no_grad():

                    x_out_forward, x_out_smoothed, P_smoothed_seq, P_filtered_seq, smoother_gain_list, K_t, H_last = \
                        self._run_rtsnet_sequence_P(SysModel, y_seq, em_iter, x0=x0, p0=P0,non_linear_h=non_linear_h)

                    # Compute losses
                    rts_loss = self.loss_fn(x_out_smoothed, target_seq)
                    total_seq_loss = 1 * rts_loss
                    seq_iter_losses.append(total_seq_loss)

                    # Debug print for first sequence
                    if seq_idx == 0:
                        print(f'EM iter: {em_iter}, loss: {total_seq_loss:.4f}')
                        print(f'F_seq: {F_seq}')


                    # M-STEP
                    if non_linear_h == True:
                        V = self.compute_cross_covariances(SysModel.F,H_last, K_t, P_filtered_seq,
                                                           smoother_gain_list)
                    else:
                        V = self.compute_cross_covariances(SysModel.F, SysModel.H, K_t, P_filtered_seq, smoother_gain_list)
                    X_s = x_out_smoothed.unsqueeze(0)
                    P_smooth_s = P_smoothed_seq.unsqueeze(0)
                    list_V_s = []
                    list_V_s.append(V)
                    F_est = EMKF_F_Mstep(SysModel, X_s, P_smooth_s, list_V_s, SysModel.m)[0]

                    deltaF = (F_est - F_seq).norm()
                    if seq_idx == 0:
                        print(f"[EM {em_iter}] ||ΔF|| = {deltaF.item():.3e}")
                    F_seq = F_est

                    # F loss
                    F_true_seq = F_true[f_index]
                    f_loss = torch.linalg.norm(F_seq.detach() - F_true_seq, ord='fro')
                    seq_iter_f_losses.append(f_loss)

                    if em_iter == (emkf_iterations-2):
                        final_F_list2[seq_idx] = F_seq.detach().clone()



            ##########################add a last computation for after the F computation
            # --- after the for em_iter in range(emkf_iterations) loop ---

            # # Use the final F_seq (after the last M-step)
            # SysModel.F = F_seq
            #
            # # Pick a model index to run the forward; reusing the last one is fine
            # last_idx = emkf_iterations - 1
            # self.rtsnet_models[last_idx].eval()
            # self.psmooth_models[last_idx].eval()
            #
            # with torch.no_grad():
            #     x_out_forward2, x_out_smoothed2, P_smoothed_seq2, P_filtered_seq2, smoother_gain_list2, K_t2 = \
            #         self._run_rtsnet_sequence(SysModel, y_seq, last_idx)
            #     final_seq_loss = self.loss_fn(x_out_smoothed2, target_seq)
            #
            # # Append the “after K updates” loss as the (K)-th index (i.e., emkf_iterations)
            # seq_iter_losses.append(final_seq_loss)
            #
            # # Optional: F-distance for the final F
            # F_true_seq = F_true[f_index]
            # final_f_loss = torch.linalg.norm(F_seq.detach() - F_true_seq, ord='fro')
            # seq_iter_f_losses.append(final_f_loss)

            ################################################################

            # terminals from the last EM iteration run above
            x_last_all.append(x_out_smoothed[:, -1].unsqueeze(-1).detach().clone())  # [m,1]
            p_last_all.append(P_smoothed_seq[:, :, -1].detach().clone())  # [m,m]

            final_F_list[seq_idx] = F_seq.detach().clone()

            # Add this sequence's losses to accumulators
            for em_iter in range(emkf_iterations):
                all_iter_losses[em_iter].append(seq_iter_losses[em_iter])
                all_iter_f_losses[em_iter].append(seq_iter_f_losses[em_iter])

        # Compute final averages
        final_iter_losses = []
        final_iter_f_losses = []

        for em_iter in range(emkf_iterations):
            if all_iter_losses[em_iter]:
                iter_avg_loss = torch.stack(all_iter_losses[em_iter]).mean()
                iter_avg_f_loss = torch.stack(all_iter_f_losses[em_iter]).mean()
            else:
                print('wrongggggggggggggggggggggggggggggggggggggggg')

            final_iter_losses.append(iter_avg_loss)
            final_iter_f_losses.append(iter_avg_f_loss)

        return final_iter_losses, final_iter_f_losses, final_F_list, x_last_all, p_last_all, final_F_list2


#################################################################################################################################################


# python
    def train_mstep_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS, num_em_iters=3,
                        alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, non_linear_h=False):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
            # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model.train()


        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0


        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            train_loss_sum = 0.0


            for _ in range(self.N_B):
                self.M_optimizer.zero_grad()

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]   # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select F_i and F_true by group
                if generate_f is True:
                    f_index = n_e // 10
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                # --------- EM unrolling over F ---------
                F_current = F_base  # this will be updated each EM iteration
                total_loss = 0.0

                for em_iter in range(num_em_iters):

                    self.model.update_F(F_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    x_prev = torch.empty_like(x_curr)  # [m, T]
                    x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                    # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                    A1 = (x_curr @ x_prev.T)/T
                    # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                    A2 = (x_prev @ x_prev.T)/T
                    # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                    x_minus = F_current @ x_prev  # [m, T_eff]
                    # Δx_t = x_t - F*x_{t-1|T}
                    delta_x = x_curr - x_minus  # [m, T]
                    delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                    delta_centered = delta_x - delta_mean
                    # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                    S_delta_x = (delta_centered @ delta_centered.T) / T
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr
                    # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                    nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T
                    # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    z_in = torch.cat([A1.reshape(-1),A2.reshape(-1),S_delta_x.reshape(-1),S_nu.reshape(-1),C_delta_x_xminus.reshape(-1),
                        F_current.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]

                    # Predict ΔF and update F
                    deltaF = model_mstep(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat

                    # Loss: Frobenius(F_next - F_true)^2 + regularization
                    f_loss = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################
                    if em_iter == num_em_iters - 1:
                        loss_em = 15*f_loss + reg + x_loss
                    else:
                        loss_em = f_loss + reg+ x_loss
#############################################################################################################
                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss += weight*loss_em
                    F_current = F_next  # use updated F in next EM iteration


                # after `for em_iter in range(num_em_iters):`
                loss = total_loss / float(num_em_iters)   # average over EM iterations
                loss_mult = loss
                loss_mult.backward()
                torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                self.M_optimizer.step()

                train_loss_sum += loss.detach().item()

                # ---------------- Validation ----------------
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    # choose base F and F_true for this CV sequence
                    if generate_f is True:
                        f_index_cv = j // 10
                        F_base_cv = SysModel.F_valid[f_index_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()
                    total_loss_cv = 0.0

                    for em_iter in range(num_em_iters):

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_F(F_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        x_prev = torch.empty_like(x_curr)
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)
                        x_prev[:, 1:] = x_curr[:, :-1]

                        A1_cv = (x_curr @ x_prev.T) / T_cv
                        A2_cv = (x_prev @ x_prev.T) / T_cv

                        x_minus_cv = F_current_cv @ x_prev
                        delta_x_cv = x_curr - x_minus_cv

                        delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                        delta_centered_cv = delta_x_cv - delta_mean_cv
                        S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv


                        if non_linear_h:
                            y_hat_cv_list = []
                            for t in range(T_cv):
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)  # non-linear h
                                y_hat_cv_list.append(y_t_hat.view(-1))
                            Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                        else:
                            H = SysModel.H.to(device)
                            Hx_curr_cv = H @ x_curr
                        nu_cv = y_cv - Hx_curr_cv

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_centered_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                        C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                        z_cv = torch.cat([
                            A1_cv.reshape(-1),
                            A2_cv.reshape(-1),
                            S_delta_x_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_delta_x_xminus_cv.reshape(-1),
                            F_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dF_cv = model_mstep(z_cv)
                        dF_cv_mat = dF_cv.view(m, m)
                        F_next_cv = F_current_cv + dF_cv_mat

                        # same loss as train (but no backward)
                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                        reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################
                        if em_iter == num_em_iters - 1:
                            loss_em_cv = 15*f_loss_cv + reg_cv + x_loss_cv
                        else:
                            loss_em_cv =  f_loss_cv + reg_cv + x_loss_cv
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight =  alpha[2]  # if you really want the same scaling as in train

                        total_loss_cv += weight * loss_em_cv
                        F_current_cv = F_next_cv
                    cv_loss_seq = total_loss_cv / float(num_em_iters)
                    cv_loss_sum += cv_loss_seq.item()

            train_epoch = train_loss_sum / max(1, self.N_B)
            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)

            print(f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")

    def train_mstep_net_batch(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS, num_em_iters=3,
                        alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, non_linear_h=False):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model.train()

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            self.M_optimizer.zero_grad()
            epoch_loss_sum = 0.0

            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select F_i and F_true by group
                if generate_f is True:
                    f_index = n_e // 10
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                # NEW: Select H for this training sequence
                if generate_h is True:
                    h_index = n_e // 10
                    H_current = SysModel.H_train[h_index]
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                # --------- EM unrolling over F ---------
                F_current = F_base  # this will be updated each EM iteration
                total_loss = 0.0

                for em_iter in range(num_em_iters):

                    self.model.update_F(F_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1],
                                                    x_smooth[:, t + 2])

                    # ---------------- Stats for M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    x_prev = torch.empty_like(x_curr)  # [m, T]
                    x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                    # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                    A1 = (x_curr @ x_prev.T) / T
                    # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                    A2 = (x_prev @ x_prev.T) / T
                    # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                    x_minus = F_current @ x_prev  # [m, T_eff]
                    # Δx_t = x_t - F*x_{t-1|T}
                    delta_x = x_curr - x_minus  # [m, T]
                    delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                    delta_centered = delta_x - delta_mean
                    # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                    S_delta_x = (delta_centered @ delta_centered.T) / T
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr
                    # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                    nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T
                    # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    z_in = torch.cat([A1.reshape(-1), A2.reshape(-1), S_delta_x.reshape(-1), S_nu.reshape(-1),
                                      C_delta_x_xminus.reshape(-1),
                                      F_current.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]

                    # Predict ΔF and update F
                    deltaF = model_mstep(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat

                    # Loss: Frobenius(F_next - F_true)^2 + regularization
                    f_loss = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################
                    if em_iter == num_em_iters - 1:
                        loss_em = 3*f_loss + reg + x_loss
                    elif em_iter == num_em_iters-2:
                        loss_em =  f_loss + reg + x_loss
                    else:
                        loss_em = f_loss + reg + x_loss
                    #############################################################################################################
                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss += weight * loss_em
                    F_current = F_next  # use updated F in next EM iteration

                # after `for em_iter in range(num_em_iters):`
                loss = total_loss / float(num_em_iters)  # average over EM iterations
                # accumulate loss over sequences in the epoch
                epoch_loss_sum += loss

            # average over the N_B sequences in this epoch
            epoch_loss_mean = epoch_loss_sum / float(self.N_B)

            # backprop once for the whole epoch ("batch")
            epoch_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.M_optimizer.step()

            # for logging
            train_epoch = epoch_loss_mean.item()

                # ---------------- Validation ----------------
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    # choose base F and F_true for this CV sequence
                    if generate_f is True:
                        f_index_cv = j // 10
                        F_base_cv = SysModel.F_valid[f_index_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()
                    total_loss_cv = 0.0

                    for em_iter in range(num_em_iters):

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_F(F_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        x_prev = torch.empty_like(x_curr)
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)
                        x_prev[:, 1:] = x_curr[:, :-1]

                        A1_cv = (x_curr @ x_prev.T) / T_cv
                        A2_cv = (x_prev @ x_prev.T) / T_cv

                        x_minus_cv = F_current_cv @ x_prev
                        delta_x_cv = x_curr - x_minus_cv

                        delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                        delta_centered_cv = delta_x_cv - delta_mean_cv
                        S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv

                        if non_linear_h:
                            y_hat_cv_list = []
                            for t in range(T_cv):
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)  # non-linear h
                                y_hat_cv_list.append(y_t_hat.view(-1))
                            Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                        else:
                            H = SysModel.H.to(device)
                            Hx_curr_cv = H @ x_curr
                        nu_cv = y_cv - Hx_curr_cv

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_centered_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                        C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                        z_cv = torch.cat([
                            A1_cv.reshape(-1),
                            A2_cv.reshape(-1),
                            S_delta_x_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_delta_x_xminus_cv.reshape(-1),
                            F_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dF_cv = model_mstep(z_cv)
                        dF_cv_mat = dF_cv.view(m, m)
                        F_next_cv = F_current_cv + dF_cv_mat

                        # same loss as train (but no backward)
                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                        reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################
                        if em_iter == num_em_iters - 1:
                            loss_em_cv = 3*f_loss_cv + reg_cv + x_loss_cv
                        elif em_iter == num_em_iters - 2:
                            loss_em_cv = f_loss_cv + reg_cv + x_loss_cv
                        else:
                            loss_em_cv = f_loss_cv + reg_cv + x_loss_cv
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]  # if you really want the same scaling as in train

                        total_loss_cv += weight * loss_em_cv
                        F_current_cv = F_next_cv
                    cv_loss_seq = total_loss_cv / float(num_em_iters)
                    cv_loss_sum += cv_loss_seq.item()

            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)

            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")

    def end_To_end_m_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS,load_base_m_mmodel = None,load_rts=None, num_em_iters=3,
                        alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, non_linear_h=False):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load base RTSNet and create one RTSNet per EM iteration (all trainable)
        base_rts = torch.load(load_rts, weights_only=False).to(self.device).train()

        self.RTS_models = []
        for i in range(num_em_iters):
            RTS_i = copy.deepcopy(base_rts).to(self.device).train()
            self.RTS_models.append(RTS_i)


        self.M_models = []
        if load_base_m_mmodel !=None:
            for i in range(num_em_iters):
                M_model = torch.load(load_base_m_mmodel,map_location=self.device,weights_only=False).to(self.device)
                self.M_models.append(M_model.train())
        else:
            for i in range(num_em_iters):
                M_model = copy.deepcopy(self.M_model)
                self.M_models.append(M_model.train())

        self.M_optimizers = []

        stable_lr = self.learningRate * 0.1
        for i in range(num_em_iters):
            self.M_optimizers.append(torch.optim.Adam(self.M_models[i].parameters(),lr=stable_lr,weight_decay=self.weightDecay))

        self.RTS_optimizers = []
        for i in range(num_em_iters):
            self.RTS_optimizers.append(
                torch.optim.Adam(self.RTS_models[i].parameters(), lr=stable_lr, weight_decay=self.weightDecay))





        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            for M_model in self.M_models:
                M_model.train()
            for RTS_model in self.RTS_models:
                RTS_model.train()
            train_loss_sum = 0.0

            # zero grad for all M nets
            for opt in self.M_optimizers:
                opt.zero_grad()

            # zero grad for all RTS nets
            for opt in self.RTS_optimizers:
                opt.zero_grad()
            epoch_loss_sum = 0.0  # NEW: accumulate loss over N_B sequences

            for _ in range(self.N_B):



                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select F_i and F_true by group
                if generate_f is True:
                    f_index = n_e // 10
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                # NEW: Select H for this training sequence
                if generate_h is True:
                    h_index = n_e // 10
                    H_current = SysModel.H_train[h_index]
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                # --------- EM unrolling over F ---------
                F_current = F_base  # this will be updated each EM iteration
                total_loss = 0.0

                for em_iter in range(num_em_iters):

                    M_k = self.M_models[em_iter]
                    self.model = self.RTS_models[em_iter]

                    self.model.update_F(F_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    x_prev = torch.empty_like(x_curr)  # [m, T]
                    x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                    # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                    A1 = (x_curr @ x_prev.T) / T
                    # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                    A2 = (x_prev @ x_prev.T) / T
                    # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                    x_minus = F_current @ x_prev  # [m, T_eff]
                    # Δx_t = x_t - F*x_{t-1|T}
                    delta_x = x_curr - x_minus  # [m, T]
                    delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                    delta_centered = delta_x - delta_mean
                    # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                    S_delta_x = (delta_centered @ delta_centered.T) / T
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr
                    # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                    nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T
                    # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    z_in = torch.cat([A1.reshape(-1), A2.reshape(-1), S_delta_x.reshape(-1), S_nu.reshape(-1),
                                      C_delta_x_xminus.reshape(-1),
                                      F_current.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]

                    # Predict ΔF and update F
                    deltaF = M_k(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat

                    # Loss: Frobenius(F_next - F_true)^2 + regularization
                    f_loss = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################
                    if em_iter == num_em_iters - 1:
                        loss_em = 10*f_loss + reg + x_loss
                    else:
                        loss_em = 5*f_loss + reg+ x_loss
                    #############################################################################################################
                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss += weight * loss_em
                    F_current = F_next  # use updated F in next EM iteration

                # after `for em_iter in range(num_em_iters):`
                loss = total_loss / float(num_em_iters)  # average over EM iterations

                # NEW: accumulate this sequence loss into the epoch batch loss
                epoch_loss_sum += total_loss

                # still keep scalar for logging if you want
                train_loss_sum += loss.detach().item()

            epoch_loss_mean = epoch_loss_sum / float(self.N_B)

            # one backward pass for the whole batch
            epoch_loss_mean.backward()

            for M_model, opt in zip(self.M_models, self.M_optimizers):
                torch.nn.utils.clip_grad_norm_(M_model.parameters(), max_norm=1.0)
                opt.step()

            for RTS_model, opt_rts in zip(self.RTS_models, self.RTS_optimizers):
                torch.nn.utils.clip_grad_norm_(RTS_model.parameters(), max_norm=1.0)
                opt_rts.step()

            train_epoch = (epoch_loss_mean/3).item()

                # ---------------- Validation ----------------
            for M_model in self.M_models:
                M_model.eval()
            for RTS_model in self.RTS_models:
                RTS_model.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    # choose base F and F_true for this CV sequence
                    if generate_f is True:
                        f_index_cv = j // 10
                        F_base_cv = SysModel.F_valid[f_index_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()
                    total_loss_cv = 0.0

                    for em_iter in range(num_em_iters):

                        M_k = self.M_models[em_iter]
                        self.model = self.RTS_models[em_iter]

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_F(F_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        ################################################
                        if epoch % 10 == 0 and j == 2:
                                # מרחק של F_current לפני העדכון מה-F_true
                            f_err_before = torch.mean((F_current_cv - F_true_cv) ** 2).item()
                            x_err_before = torch.mean((x_curr - x_true_cv_seq) ** 2).item()
                            print(
                                f"[DEBUG][epoch={epoch} em={em_iter}] F_err_before={f_err_before:.3e}, x_err_before={x_err_before:.3e}")
                                #######################################
                        x_prev = torch.empty_like(x_curr)
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)
                        x_prev[:, 1:] = x_curr[:, :-1]

                        A1_cv = (x_curr @ x_prev.T) / T_cv
                        A2_cv = (x_prev @ x_prev.T) / T_cv

                        x_minus_cv = F_current_cv @ x_prev
                        delta_x_cv = x_curr - x_minus_cv

                        delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                        delta_centered_cv = delta_x_cv - delta_mean_cv
                        S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv

                        if non_linear_h:
                            y_hat_cv_list = []
                            for t in range(T_cv):
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)  # non-linear h
                                y_hat_cv_list.append(y_t_hat.view(-1))
                            Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                        else:
                            H = SysModel.H.to(device)
                            Hx_curr_cv = H @ x_curr
                        nu_cv = y_cv - Hx_curr_cv

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_centered_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                        C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                        z_cv = torch.cat([
                            A1_cv.reshape(-1),
                            A2_cv.reshape(-1),
                            S_delta_x_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_delta_x_xminus_cv.reshape(-1),
                            F_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dF_cv = M_k(z_cv)
                        dF_cv_mat = dF_cv.view(m, m)
                        F_next_cv = F_current_cv + dF_cv_mat
################################################################################################################
                        # if epoch % 10 == 0 and j == 2:
                        #
                        #     # RTS עם F_next (רק לצורך בדיקה, בלי גרדיאנט)
                        #     self.model.update_F(F_next_cv)
                        #     self.model.InitSequence(SysModel.m1x_0, T)
                        #     self.model.init_hidden()
                        #     self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                        #
                        #     x_f2 = torch.empty(m, T, device=device)
                        #     x_s2 = torch.empty(m, T, device=device)
                        #     for t in range(T):
                        #         x_f2[:, t] = self.model(y_cv[:, t], None, None, None)
                        #     x_s2[:, T - 1] = x_f2[:, T - 1]
                        #     self.model.InitBackward(x_s2[:, T - 1])
                        #     x_s2[:, T - 2] = self.model(None, x_f2[:, T - 2], x_f2[:, T - 1], None)
                        #     for t in range(T - 3, -1, -1):
                        #         x_s2[:, t] = self.model(None, x_f2[:, t], x_f2[:, t + 1], x_s2[:, t + 2])
                        #
                        #     x_err_after = torch.mean((x_s2 - x_true_cv_seq) ** 2).item()
                        #     f_err_after = torch.mean((F_next_cv - F_true_cv) ** 2).item()
                        #
                        #     print(
                        #         f"[DEBUG][epoch={epoch} em={em_iter}] F_err_after={f_err_after:.3e}, x_err_after={x_err_after:.3e}")
                                #############################################################################################################




                        # same loss as train (but no backward)
                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                        reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################
                        if em_iter == num_em_iters - 1:
                            loss_em_cv = 10*f_loss_cv + reg_cv + x_loss_cv
                        else:
                            loss_em_cv =  5*f_loss_cv + reg_cv + x_loss_cv
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]  # if you really want the same scaling as in train

                        total_loss_cv += weight * loss_em_cv
                        F_current_cv = F_next_cv
                    cv_loss_seq = total_loss_cv / float(num_em_iters)
                    cv_loss_sum += cv_loss_seq.item()


            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                for k in range(num_em_iters):
                    torch.save(self.M_models[k], destination_path_M[k])
                    torch.save(self.RTS_models[k], destination_path_RTS[k])
            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")

    def test_mstep_net(self, SysModel, test_input, test_target,
                       destination_path_RTS,destination_path_M, num_em_iters=3,
                       alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, generate_h=False, init_x_list=None, init_P_list=None, non_linear_h=False):
        """
        Testing-only version for the M-step network.
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Use CURRENT M-step network (self.M_model) to predict ΔF.
        - Run num_em_iters EM iterations per test sequence.
        - No training, no optimizer step.
        - Returns per-sequence loss and mean loss.
        """


        N_T = len(test_input)
        m = SysModel.m

        all_test_losses = []
        all_f_losses = []


        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load M-step network from checkpoint (NO training)
        model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        final_F_list = []
        final_x_list = []
        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]    # [n, T]
                x_true_seq = test_target[j]  # [m, T]

                T = y_seq.size(-1)

                # Select F_base and F_true for this test sequence
                if generate_f is True:
                    f_index = j // 10
                    F_base = SysModel.F_test[f_index].to(device)
                    F_true = SysModel.F_test_TRUE[f_index].to(device)
                else:
                    # fallback: sequence-wise
                    F_base = SysModel.F_test[j].to(device)
                    F_true = SysModel.F_test_TRUE[j].to(device)

                # NEW: Select H for this test sequence
                if generate_h is True:
                    h_index = j // 10
                    H_current = SysModel.H_test[h_index].to(device)
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                F_current = F_base.clone()
                total_loss = 0.0
                F_estimates =[]
                F_losses_mse = []
                F_losses_total = []
                x_losses_mse = []
                y_loss_tot = []

                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                for em_iter in range(num_em_iters):

                    # ----- RTS smoother with current F_current -----
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = P0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ----- stats for M-network -----
                    x_curr = x_smooth                    # [m, T]
                    x_prev = torch.empty_like(x_curr)    # [m, T]
                    x_prev[:, 0] = x0.view(-1)   # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]                      # x_{t-1|T}

                    A1 = (x_curr @ x_prev.T) / T
                    A2 = (x_prev @ x_prev.T) / T

                    x_minus = F_current @ x_prev        # [m, T]
                    delta_x = x_curr - x_minus          # [m, T]

                    delta_mean = delta_x.mean(dim=1, keepdim=True)
                    delta_centered = delta_x - delta_mean
                    S_delta_x = (delta_centered @ delta_centered.T) / T


                    # ---------- linear vs non-linear h ----------
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            x_t = x_curr[:, t].view(SysModel.m, 1)  # [m,1]
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))  # flatten to [n]
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H.to(device)
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T

                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    z_in = torch.cat([
                        A1.reshape(-1),
                        A2.reshape(-1),
                        S_delta_x.reshape(-1),
                        S_nu.reshape(-1),
                        C_delta_x_xminus.reshape(-1),
                        F_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only -----
                    deltaF = model_mstep(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat
                    #############################################

                    # rho = eigvals.abs().max()
                    #
                    # # rho is a scalar complex tensor -> take real part for safety
                    # rho_real = rho.real
                    # max_rho = 1.07
                    # if rho_real > max_rho:
                    #     scale = (max_rho / rho_real)
                    #     F_next = F_next * scale
                    #############################################

                    mse_F = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    # x-loss: same as in training
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat  = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat  - y_seq) ** 2)
                    # y_loss_tot.append(y_loss.item())
                    ##########################################################
                    loss_em = 3*mse_F + reg + x_loss
                    x_loss_sum_per_iter[em_iter] += x_loss.item()

                    # same alpha weighting as in training
                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]

                    total_loss += weight * loss_em
                    F_current = F_next

                    all_test_losses.append(loss_em.item())
                    all_f_losses.append(mse_F.item())

                    # store F estimates for the chosen sequence
                    if j %5 ==0:
                        F_estimates.append(F_next.detach())
                        F_losses_mse.append(mse_F.item())
                        F_losses_total.append(loss_em.item())
                        x_losses_mse.append(x_loss.item())

                # store final F and final x_smooth for this sequence (after last EM iter)
                final_F_list.append(F_current.detach().clone())   # [m, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())      # [m, T]
                        # final loss for this sequence (already weighted)
                loss_list[j] = total_loss / float(num_em_iters)
                # Mean x-loss for this sequence
                # if this is the chosen sequence, print F_true and all F_est
                if j %5 ==0:
                    print(f"\n[M-step TEST] sequence {j} summary")
                    print("F_true:\n", F_true.detach())
                    print("F_init (F_base):\n", F_base.detach())

                    mse_F_init = torch.mean((F_base - F_true) ** 2).item()
                    print(f"Initial F MSE loss = {mse_F_init:.6e}")
                    for k, (F_est, f_mse, x_mse, total_val) in enumerate(zip(F_estimates, F_losses_mse, x_losses_mse, F_losses_total)):
                        f_db = 10.0 * math.log10(f_mse)
                        x_db = 10.0 * math.log10(x_mse)
                        tot_db = 10.0 * math.log10(total_val)
                        print(f"\n  EM iter {k + 1}:")
                        print("  F_est:\n", F_est)
                        print(f"  F-loss (MSE_F)                 = {f_db:.2f} dB")
                        print(f"  x-loss (MSE_x)                 = {x_db:.2f} dB")
                        print(f"  total loss (F + reg + x)       = {tot_db:.2f} dB")
                        # print(f"y_loss = {y_loss_tot[k]:2f}")


        mean_loss = loss_list.mean().item()
        print(f"[M-step TEST] mean_loss={mean_loss:.6f}")
        # average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)

        print("[M-step TEST] Mean x-MSE per EM iteration:")
        for k in range(num_em_iters):
            mse_k = mean_x_mse_per_iter[k].item()
            db_k = 10.0 * math.log10(mse_k )
            print(f"  EM iter {k + 1}: mean x-MSE = {mse_k:.6e}  ({db_k: .2f} dB)")

        # convert per-iteration mean x-MSE to numpy (linear and dB)
        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()


        return mean_x_mse_per_iter_np,mean_x_mse_per_iter_db_np, final_F_list,final_x_list

    def end_to_end_test_mstep_net(self, SysModel, test_input, test_target,destination_path_RTS,destination_path_M, num_em_iters=3,
                       alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, init_x_list=None, init_P_list=None, non_linear_h=False):
        """
        Testing-only version for the M-step network.
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Use CURRENT M-step network (self.M_model) to predict ΔF.
        - Run num_em_iters EM iterations per test sequence.
        - No training, no optimizer step.
        - Returns per-sequence loss and mean loss.
        """


        N_T = len(test_input)
        m = SysModel.m

        all_test_losses = []
        all_f_losses = []

        RTS_models = []
        for k in range(num_em_iters):
            rts_k = torch.load(destination_path_RTS[k],weights_only=False,map_location=device).to(device).eval()
            for p in rts_k.parameters():
                p.requires_grad_(False)
            RTS_models.append(rts_k)


        M_models = []
        for k in range(num_em_iters):
            M_k = torch.load(destination_path_M[k], weights_only=False, map_location=device)
            M_k = M_k.to(device).eval()
            M_models.append(M_k)

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        final_F_list = []
        final_x_list = []
        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]    # [n, T]
                x_true_seq = test_target[j]  # [m, T]

                T = y_seq.size(-1)

                # Select F_base and F_true for this test sequence
                if generate_f is True:
                    f_index = j // 10
                    F_base = SysModel.F_test[f_index].to(device)
                    F_true = SysModel.F_test_TRUE[f_index].to(device)
                else:
                    # fallback: sequence-wise
                    F_base = SysModel.F_test[j].to(device)
                    F_true = SysModel.F_test_TRUE[j].to(device)

                # NEW: Select H for this test sequence
                if generate_h is True:
                    h_index = j // 10
                    H_current = SysModel.H_test[h_index].to(device)
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                F_current = F_base.clone()
                total_loss = 0.0
                F_estimates =[]
                F_losses_mse = []
                F_losses_total = []
                x_losses_mse = []

                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                for em_iter in range(num_em_iters):

                    self.model = RTS_models[em_iter]
                    M_k = M_models[em_iter]

                    # ----- RTS smoother with current F_current -----
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = P0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ----- stats for M-network -----
                    x_curr = x_smooth                    # [m, T]
                    x_prev = torch.empty_like(x_curr)    # [m, T]
                    x_prev[:, 0] = x0.view(-1)   # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]                      # x_{t-1|T}

                    A1 = (x_curr @ x_prev.T) / T
                    A2 = (x_prev @ x_prev.T) / T

                    x_minus = F_current @ x_prev        # [m, T]
                    delta_x = x_curr - x_minus          # [m, T]

                    delta_mean = delta_x.mean(dim=1, keepdim=True)
                    delta_centered = delta_x - delta_mean
                    S_delta_x = (delta_centered @ delta_centered.T) / T


                    # ---------- linear vs non-linear h ----------
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            x_t = x_curr[:, t].view(SysModel.m, 1)  # [m,1]
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))  # flatten to [n]
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H.to(device)
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T

                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    z_in = torch.cat([
                        A1.reshape(-1),
                        A2.reshape(-1),
                        S_delta_x.reshape(-1),
                        S_nu.reshape(-1),
                        C_delta_x_xminus.reshape(-1),
                        F_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only -----
                    deltaF = M_k(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat

                    mse_F = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    # x-loss: same as in training
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    loss_em = 3*mse_F + reg + x_loss
                    x_loss_sum_per_iter[em_iter] += x_loss.item()

                    # same alpha weighting as in training
                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]

                    total_loss += weight * loss_em
                    F_current = F_next

                    all_test_losses.append(loss_em.item())
                    all_f_losses.append(mse_F.item())

                    # store F estimates for the chosen sequence
                    if j %5 ==0:
                        F_estimates.append(F_next.detach())
                        F_losses_mse.append(mse_F.item())
                        F_losses_total.append(loss_em.item())
                        x_losses_mse.append(x_loss.item())

                # store final F and final x_smooth for this sequence (after last EM iter)
                final_F_list.append(F_current.detach().clone())   # [m, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())      # [m, T]
                        # final loss for this sequence (already weighted)
                loss_list[j] = total_loss / float(num_em_iters)
                # Mean x-loss for this sequence
                # if this is the chosen sequence, print F_true and all F_est
                if j %5 ==0:
                    print(f"\n[M-step TEST] sequence {j} summary")
                    print("F_true:\n", F_true.detach())
                    print("F_init (F_base):\n", F_base.detach())

                    mse_F_init = torch.mean((F_base - F_true) ** 2).item()
                    print(f"Initial F MSE loss = {mse_F_init:.6e}")
                    for k, (F_est, f_mse, x_mse, total_val) in enumerate(zip(F_estimates, F_losses_mse, x_losses_mse, F_losses_total)):
                        f_db = 10.0 * math.log10(f_mse)
                        x_db = 10.0 * math.log10(x_mse)
                        tot_db = 10.0 * math.log10(total_val)
                        print(f"\n  EM iter {k + 1}:")
                        print("  F_est:\n", F_est)
                        print(f"  F-loss (MSE_F)                 = {f_db:.2f} dB")
                        print(f"  x-loss (MSE_x)                 = {x_db:.2f} dB")
                        print(f"  total loss (F + reg + x)       = {tot_db:.2f} dB")


        mean_loss = loss_list.mean().item()
        print(f"[M-step TEST] mean_loss={mean_loss:.6f}")
        # average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)

        print("[M-step TEST] Mean x-MSE per EM iteration:")
        for k in range(num_em_iters):
            mse_k = mean_x_mse_per_iter[k].item()
            db_k = 10.0 * math.log10(mse_k )
            print(f"  EM iter {k + 1}: mean x-MSE = {mse_k:.6e}  ({db_k: .2f} dB)")

        # convert per-iteration mean x-MSE to numpy (linear and dB)
        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()


        return mean_x_mse_per_iter_np,mean_x_mse_per_iter_db_np, final_F_list,final_x_list

    ################################################################################################################################################
    def one_train_m_step_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS,  lambda_F=1e-3, generate_f=True, non_linear_h=False):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model.train()

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        ######################################################
        # for name, param in model_mstep.named_parameters():
        #     if "weight" in name or "bias" in name:
        #         param.data.zero_()
        ##################################################
        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            train_loss_sum = 0.0
            batch_loss_sum = 0.0
            batch_x_before_sum = 0.0
            batch_x_after_sum = 0.0
            batch_f_loss_sum = 0.0
            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select F_i and F_true by group
                if generate_f is True:
                    f_index = n_e // 10
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                # NEW: Select H for this training sequence
                if generate_h is True:
                    h_index = n_e // 10
                    H_current = SysModel.H_train[h_index]
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                # --------- EM unrolling over F ---------
                F_current = F_base  # this will be updated each EM iteration


                self.model.update_F(F_current)

                # E-step via frozen RTSNet → x_smooth
                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.init_hidden()
                self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                x_smooth[:, T - 1] = x_forward[:, T - 1]

                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                x_loss_before = torch.mean((x_smooth - x_true_seq) ** 2)


                # ---------------- Stats for M-network ----------------
                x_curr = x_smooth  # [m, T]
                x_prev = torch.empty_like(x_curr)  # [m, T]
                x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                A1 = (x_curr @ x_prev.T) / T
                # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                A2 = (x_prev @ x_prev.T) / T
                # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                x_minus = F_current @ x_prev  # [m, T_eff]
                # Δx_t = x_t - F*x_{t-1|T}
                delta_x = x_curr - x_minus  # [m, T]
                delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                delta_centered = delta_x - delta_mean
                # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                S_delta_x = (delta_centered @ delta_centered.T) / T
                if non_linear_h:
                    # y_hat_t = h(x_t) for each t
                    y_hat_list = []
                    for t in range(T):
                        # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                        x_t = x_curr[:, t].view(SysModel.m, 1)
                        y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                        y_hat_list.append(y_t_hat.view(-1))
                    Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                else:
                    H = SysModel.H
                    Hx_curr = H @ x_curr  # [n, T]
                nu = y_seq - Hx_curr
                # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                nu_centered = nu - nu_mean
                S_nu = (nu_centered @ nu_centered.T) / T
                # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                C_delta_x_xminus = (delta_x @ x_minus.T) / T

                z_in = torch.cat([A1.reshape(-1), A2.reshape(-1), S_delta_x.reshape(-1), S_nu.reshape(-1),
                                  C_delta_x_xminus.reshape(-1),
                                  F_current.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]

                # Predict ΔF and update F
                deltaF = model_mstep(z_in)
                deltaF_mat = deltaF.view(m, m)
                # print('for nir delta f =',deltaF_mat)
                F_next = F_current + deltaF_mat

                F_current =F_next

                self.model.update_F(F_current)

                # E-step via frozen RTSNet → x_smooth
                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.init_hidden()
                self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                x_smooth[:, T - 1] = x_forward[:, T - 1]

                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])



                # Loss: Frobenius(F_next - F_true)^2 + regularization
                f_loss = torch.mean((F_current - F_true) ** 2)
                reg = lambda_F * torch.mean(deltaF_mat ** 2)
                # x-loss AFTER M-step (with updated F)
                x_loss_after = torch.mean((x_smooth - x_true_seq) ** 2)

                # ---- compute loss ----
                loss = 15* f_loss + reg + x_loss_after

                # ---- backward + step (one SGD step per sequence) ----
                self.M_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                self.M_optimizer.step()

                # ---- only for logging / printing ----
                batch_loss_sum     += loss.detach()
                batch_x_before_sum += x_loss_before.detach()
                batch_x_after_sum  += x_loss_after.detach()
                batch_f_loss_sum   += f_loss.detach()

            batch_loss = batch_loss_sum / self.N_B
            train_loss_sum += batch_loss.item()
            # averages over sequences in this batch
            mean_x_before = batch_x_before_sum / self.N_B
            mean_x_after  = batch_x_after_sum / self.N_B
            mean_f_loss   = batch_f_loss_sum / self.N_B

            eps = 1e-12
            batch_loss_db   = 10.0 * math.log10(batch_loss.item()   + eps)
            x_before_db     = 10.0 * math.log10(mean_x_before.item() + eps)
            x_after_db      = 10.0 * math.log10(mean_x_after.item()  + eps)
            f_loss_db       = 10.0 * math.log10(mean_f_loss.item()   + eps)
            print(
                f"[M-step][train] epoch={epoch:03d} "
                f"batch_loss={batch_loss.item():.6e} ({batch_loss_db:.2f} dB) "
                f"x_before={mean_x_before.item():.6e} ({x_before_db:.2f} dB) "
                f"x_after={mean_x_after.item():.6e} ({x_after_db:.2f} dB) "
                f"F_loss={mean_f_loss.item():.6e} ({f_loss_db:.2f} dB)"
            )

                # ---------------- Validation ----------------
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    # choose base F and F_true for this CV sequence
                    if generate_f is True:
                        f_index_cv = j // 10
                        F_base_cv = SysModel.F_valid[f_index_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()

                    # --- RTS smoother with current F_current_cv ---
                    self.model.update_F(F_current_cv)
                    self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                    x_f_cv = torch.empty(m, T_cv, device=device)
                    x_s_cv = torch.empty(m, T_cv, device=device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    # -------- stats, same as training --------
                    x_curr = x_s_cv  # [m, T_cv]
                    x_prev = torch.empty_like(x_curr)
                    x_prev[:, 0] = SysModel.m1x_0.view(-1)
                    x_prev[:, 1:] = x_curr[:, :-1]

                    A1_cv = (x_curr @ x_prev.T) / T_cv
                    A2_cv = (x_prev @ x_prev.T) / T_cv

                    x_minus_cv = F_current_cv @ x_prev
                    delta_x_cv = x_curr - x_minus_cv

                    delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                    delta_centered_cv = delta_x_cv - delta_mean_cv
                    S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv

                    if non_linear_h:
                        y_hat_cv_list = []
                        for t in range(T_cv):
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)  # non-linear h
                            y_hat_cv_list.append(y_t_hat.view(-1))
                        Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                    else:
                        H = SysModel.H.to(device)
                        Hx_curr_cv = H @ x_curr
                    nu_cv = y_cv - Hx_curr_cv

                    nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                    nu_centered_cv = nu_cv - nu_mean_cv
                    S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                    C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                    z_cv = torch.cat([
                        A1_cv.reshape(-1),
                        A2_cv.reshape(-1),
                        S_delta_x_cv.reshape(-1),
                        S_nu_cv.reshape(-1),
                        C_delta_x_xminus_cv.reshape(-1),
                        F_current_cv.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # --- M-step forward only (no grad) ---
                    dF_cv = model_mstep(z_cv)
                    dF_cv_mat = dF_cv.view(m, m)
                    F_next_cv = F_current_cv + dF_cv_mat

                    F_current_cv = F_next_cv

                    # --- RTS smoother with current F_current_cv ---
                    self.model.update_F(F_current_cv)
                    self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                    x_f_cv = torch.empty(m, T_cv, device=device)
                    x_s_cv = torch.empty(m, T_cv, device=device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    # same loss as train (but no backward)
                    f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                    reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                    x_loss_cv = torch.mean((x_s_cv - x_true_cv_seq) ** 2)

                    cv_loss_seq =15*f_loss_cv + reg_cv + x_loss_cv

                    cv_loss_sum += cv_loss_seq.item()


            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)


            train_epoch_db = 10.0 * math.log10(train_loss_sum )
            cv_epoch_db = 10.0 * math.log10(cv_epoch)
            best_cv_db = 10.0 * math.log10(self.MSE_cv_dB_opt)

            print(
                f"[M-step] epoch={epoch:03d} "
                f"train={train_loss_sum:.6e} ({train_epoch_db:.2f} dB) "
                f"cv={cv_epoch:.6e} ({cv_epoch_db:.2f} dB) "
                f"best_cv={self.MSE_cv_dB_opt:.6e} ({best_cv_db:.2f} dB)"
            )




    def one_test_mstep_net(self, SysModel, test_input, test_target,
                       destination_path_RTS,destination_path_M, lambda_F=1e-3, generate_f=True, generate_h=False, init_x_list=None, init_P_list=None, non_linear_h=False):
        """
        Testing-only version for the M-step network.
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Use CURRENT M-step network (self.M_model) to predict ΔF.
        - Run num_em_iters EM iterations per test sequence.
        - No training, no optimizer step.
        - Returns per-sequence loss and mean loss.
        """


        N_T = len(test_input)
        m = SysModel.m

        all_test_losses = []
        all_f_losses = []


        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load M-step network from checkpoint (NO training)
        model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum = 0.0
        x_loss_sum_before = 0.0
        f_loss_sum_before = 0.0
        f_loss_sum_after = 0.0
        final_F_list = []
        final_x_list = []
        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]    # [n, T]
                x_true_seq = test_target[j]  # [m, T]

                T = y_seq.size(-1)

                # Select F_base and F_true for this test sequence
                if generate_f is True:
                    f_index = j // 10
                    F_base = SysModel.F_test[f_index].to(device)
                    F_true = SysModel.F_test_TRUE[f_index].to(device)
                else:
                    # fallback: sequence-wise
                    F_base = SysModel.F_test[j].to(device)
                    F_true = SysModel.F_test_TRUE[j].to(device)

                # NEW: Select H for this test sequence
                if generate_h is True:
                    h_index = j // 10
                    H_current = SysModel.H_test[h_index].to(device)
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                F_current = F_base.clone()
                F_estimates =[]
                F_losses_mse = []
                F_losses_total = []
                x_losses_mse = []


                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                # ----- RTS smoother with current F_current -----
                self.model.update_F(F_current)
                self.model.InitSequence(x0.clone().detach(), T)
                self.model.init_hidden()
                self.model.prior_Sigma = P0.clone().detach()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                x_smooth[:, T - 1] = x_forward[:, T - 1]
                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                x_loss_before = torch.mean((x_smooth - x_true_seq) ** 2)
                x_loss_sum_before += x_loss_before.item()
                x_before_db = 10.0 * math.log10(x_loss_before.item())
                # ----- stats for M-network -----
                x_curr = x_smooth                    # [m, T]
                x_prev = torch.empty_like(x_curr)    # [m, T]
                x_prev[:, 0] = x0.view(-1)   # x_0
                x_prev[:, 1:] = x_curr[:, :-1]                      # x_{t-1|T}

                A1 = (x_curr @ x_prev.T) / T
                A2 = (x_prev @ x_prev.T) / T

                x_minus = F_current @ x_prev        # [m, T]
                delta_x = x_curr - x_minus          # [m, T]

                delta_mean = delta_x.mean(dim=1, keepdim=True)
                delta_centered = delta_x - delta_mean
                S_delta_x = (delta_centered @ delta_centered.T) / T


                # ---------- linear vs non-linear h ----------
                if non_linear_h:
                    # y_hat_t = h(x_t) for each t
                    y_hat_list = []
                    for t in range(T):
                        x_t = x_curr[:, t].view(SysModel.m, 1)  # [m,1]
                        y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                        y_hat_list.append(y_t_hat.view(-1))  # flatten to [n]
                    Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                else:
                    H = SysModel.H.to(device)
                    Hx_curr = H @ x_curr  # [n, T]
                nu = y_seq - Hx_curr

                nu_mean = nu.mean(dim=1, keepdim=True)
                nu_centered = nu - nu_mean
                S_nu = (nu_centered @ nu_centered.T) / T

                C_delta_x_xminus = (delta_x @ x_minus.T) / T

                # # ===== NORMALIZATION: Remove dependence on ||x|| magnitude ===== #
                # A1_normalized, A2_normalized, S_delta_x_normalized, S_nu_normalized, C_delta_normalized, A2_scale, S_nu_scale = \
                #     normalize_mstep_statistics(A1, A2, S_delta_x, S_nu, C_delta_x_xminus, debug=True, seq_id=j)
                #
                # # Optional: Print initial state norm for context
                # if j % 10 == 0:
                #     print(f"  ||x_0|| = {torch.norm(x0).item():.4f}")

                # Create input with NORMALIZED statistics
                z_in = torch.cat([
                    A1.reshape(-1),          # Normalized by A2_scale!
                    A2.reshape(-1),          # Normalized by A2_scale!
                    S_delta_x.reshape(-1),   # Normalized by A2_scale!
                    S_nu.reshape(-1),        # Normalized by S_nu_scale!
                    C_delta_x_xminus.reshape(-1),     # Normalized by A2_scale!
                    F_current.reshape(-1)               # Keep as-is
                ], dim=0).reshape(1, -1)

                # ----- M-step forward only -----
                deltaF = model_mstep(z_in)
                deltaF_mat = deltaF.view(m, m)
##########################################################################################################################################
                # # DEBUG: Check what the M-network is predicting
                # if j % 5 == 0 or x_loss_before.item() < 3.0:  # Always show if x_loss is decent
                #     # Compute analytical solution for comparison
                #     A1_mat = A1
                #     A2_mat = A2
                #     I = torch.eye(m, device=A2_mat.device, dtype=A2_mat.dtype)
                #     A2_reg = A2_mat + 1e-3 * I
                #     F_analytical = torch.linalg.solve(A2_reg.T, A1_mat.T).T
                #     analytical_deltaF = F_analytical - F_current
                #
                #     # Compute fit error manually to verify
                #     mismatch = A1_mat - F_current @ A2_mat
                #     fit_err_manual = ((mismatch ** 2).mean()).item()
                #
                #     print(f"\n[DEBUG] Sequence {j} - Statistics Quality:")
                #     print(f"  x_loss BEFORE:      {x_loss_before.item():.6e} ({10*math.log10(x_loss_before.item()):.2f} dB)")
                #     print(f"  S_nu diag mean:     {torch.diagonal(S_nu).mean().item():.6e}")
                #     print(f"  S_delta_x diag mean:{torch.diagonal(S_delta_x).mean().item():.6e}")
                #     print(f"  Fit error (manual): {fit_err_manual:.6e}")
                #     print(f"\n[DEBUG] Gate Behavior:")
                #     has_debug = hasattr(model_mstep, '_debug_gate_value')
                #     if not has_debug and j == 0:
                #         print(f"  ⚠️  Model doesn't have debug attributes - using old checkpoint!")
                #     print(f"  Gate value (g):     {getattr(model_mstep, '_debug_gate_value', 'N/A')}")
                #     print(f"  Fit error (log):    {getattr(model_mstep, '_debug_fit_err', 'N/A')}")
                #     print(f"  Noise level (log):  {getattr(model_mstep, '_debug_noise_level', 'N/A')}")
                #     print(f"\n[DEBUG] Delta F Comparison:")
                #     print(f"  F_analytical:\n{F_analytical}")
                #     print(f"  ΔF_analytical:\n{analytical_deltaF}")
                #     print(f"  ΔF_analytical norm: {analytical_deltaF.norm().item():.6f}")
                #     print(f"  ΔF_network:\n{deltaF_mat}")
                #     print(f"  ΔF_network norm:    {deltaF_mat.norm().item():.6f}")
                #     print(f"  Ratio (network/analytical): {deltaF_mat.norm().item() / (analytical_deltaF.norm().item() + 1e-8):.3f}")
                #
                #     # Check if analytical would be better
                #     F_analytical_result = F_current + analytical_deltaF
                #     analytical_f_mse = torch.mean((F_analytical_result - F_true) ** 2).item()
                #     analytical_f_db = 10 * math.log10(analytical_f_mse)
                #     print(f"\n[DEBUG] If we used analytical solution:")
                #     print(f"  F_analytical MSE:   {analytical_f_mse:.6e} ({analytical_f_db:.2f} dB)")
                F_next = F_current + deltaF_mat
                #
                # # Check eigenvalue stability
                # eigvals = torch.linalg.eigvals(F_next)
                # max_eigval = torch.abs(eigvals).max().item()
                # if max_eigval > 1.07:
                #     print(f"\n⚠️  WARNING: Sequence {j} - Unstable F predicted!")
                #     print(f"   Max eigenvalue: {max_eigval:.4f} (threshold: 1.07)")
                #     print(f"   All eigenvalues: {torch.abs(eigvals)}")
                #     print(f"   F_next:\n{F_next}")
                ######################################################################################################################
                F_current = F_next

                # ----- RTS smoother with current F_current -----
                self.model.update_F(F_current)
                self.model.InitSequence(x0.clone().detach(), T)
                self.model.init_hidden()
                self.model.prior_Sigma = P0.clone().detach()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                x_smooth[:, T - 1] = x_forward[:, T - 1]
                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])


                mse_F = torch.mean((F_next - F_true) ** 2)
                reg = lambda_F * torch.mean(deltaF_mat ** 2)
                # x-loss: same as in training
                x_loss_after = torch.mean((x_smooth - x_true_seq) ** 2)
                x_after_db = 10.0 * math.log10(x_loss_after.item())
                x_loss =x_loss_after
                loss_em = 3*mse_F + reg + x_loss

                x_loss_sum += x_loss_after.item()
                loss_list[j] = loss_em

                all_test_losses.append(loss_em.item())
                all_f_losses.append(mse_F.item())

                # store F estimates for the chosen sequence
                if j %5 ==0:
                    F_estimates.append(F_next.detach())
                    F_losses_mse.append(mse_F.item())
                    F_losses_total.append(loss_em.item())
                    x_losses_mse.append(x_loss.item())

                # store final F and final x_smooth for this sequence (after last EM iter)
                mse_F_init = torch.mean((F_base - F_true) ** 2).item()
                f_loss_sum_before += mse_F_init
                f_loss_sum_after += mse_F.item()
                final_F_list.append(F_current.detach().clone())   # [m, m]
                final_x_list.append(x_smooth[:, -1].unsqueeze(-1).detach().clone())      # [m, T]
                        # final loss for this sequence (already weighted)
                    # Mean x-loss for this sequence
                # if this is the chosen sequence, print F_true and all F_est
                if j %5 ==0:
                    print(f"\n[M-step TEST] sequence {j} summary")
                    print("F_true:\n", F_true.detach())
                    print("F_init (F_base):\n", F_base.detach())

                    mse_F_init = torch.mean((F_base - F_true) ** 2).item()
                    mse_F_init_db = 10.0 * math.log10(mse_F_init+0.00001)
                    print(f"Initial F MSE loss = {mse_F_init_db:.2f}db")

                    print(f"x-loss BEFORE M-step = {x_before_db:.2f} dB")
                    print(f"x-loss AFTER  M-step = {x_after_db:.2f} dB")

                    f_db = 10.0 * math.log10(mse_F.item())
                    x_db = 10.0 * math.log10(x_loss_after.item())
                    tot_db = 10.0 * math.log10(loss_em.item())

                    print("F_est:\n", F_next.detach())
                    print(f"F-loss (MSE_F)        = {f_db:.2f} dB")
                    print(f"x-loss (MSE_x)        = {x_db:.2f} dB")
                    print(f"total loss (F+reg+x)   = {tot_db:.2f} dB")


        # average x-MSE after M-step over all sequences
        # ----- average x-MSE BEFORE and AFTER M-step over all sequences -----
        mean_x_mse_before = x_loss_sum_before / float(N_T)
        mean_x_mse_before_tensor = torch.tensor([mean_x_mse_before], device=device)
        mean_x_mse_before_db = (10.0 * torch.log10(mean_x_mse_before_tensor)).detach()

        mean_x_mse_after = x_loss_sum / float(N_T)
        mean_x_mse_after_tensor = torch.tensor([mean_x_mse_after], device=device)
        mean_x_mse_after_db = (10.0 * torch.log10(mean_x_mse_after_tensor)).detach()

        print(f"[M-step TEST] mean x-MSE BEFORE M-step = {mean_x_mse_before:.6e} "
              f"({mean_x_mse_before_db[0].item():.2f} dB)")
        print(f"[M-step TEST] mean x-MSE AFTER  M-step = {mean_x_mse_after:.6e} "
              f"({mean_x_mse_after_db[0].item():.2f} dB)")

        # keep return based on AFTER-M-step (like before)
        mean_x_mse_per_iter_np = mean_x_mse_after_tensor.detach()
        mean_x_mse_per_iter_db_np = mean_x_mse_after_db

        mean_f_mse_before = f_loss_sum_before / float(N_T)
        mean_f_mse_before_tensor = torch.tensor([mean_f_mse_before], device=device)
        mean_f_mse_before_db = (10.0 * torch.log10(mean_f_mse_before_tensor)).detach()

        mean_f_mse_after = f_loss_sum_after / float(N_T)
        mean_f_mse_after_tensor = torch.tensor([mean_f_mse_after], device=device)
        mean_f_mse_after_db = (10.0 * torch.log10(mean_f_mse_after_tensor)).detach()

        print(f"[M-step TEST] mean F-MSE BEFORE M-step = {mean_f_mse_before:.6e} "
              f"({mean_f_mse_before_db[0].item():.2f} dB)")
        print(f"[M-step TEST] mean F-MSE AFTER  M-step = {mean_f_mse_after:.6e} "
              f"({mean_f_mse_after_db[0].item():.2f} dB)")
        return mean_x_mse_per_iter_np,mean_x_mse_per_iter_db_np, final_F_list,final_x_list





    def train_mstep_net_with_p(self, SysModel, cv_input, cv_target, train_input, train_target,
                            destination_path_M, destination_path_RTS,destination_path_PSMOOTH, num_em_iters=3,
                            alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, non_linear_h=False):
            """
            Single-function M-step training (no helpers, no .to(...)).
            - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
            - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
            - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
            """
                # Basic sizes
            self.N_E = len(train_input)
            self.N_CV = len(cv_input)
            m = SysModel.m
            dev = self.device

            # Load and freeze RTSNet (smoother only)
            self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
            for p in self.model.parameters():
                p.requires_grad_(False)

            # 2) Load and FREEZE PsmoothNet (for P_smooth)
            self.psmooth_model = torch.load(destination_path_PSMOOTH,
                                            weights_only=False,
                                            map_location=self.device).to(self.device).train()
            for p in self.psmooth_model.parameters():
                p.requires_grad_(False)


            # M-step model and optimizer
            model_mstep = self.M_model.train()


            self.MSE_cv_dB_opt = 1000
            self.MSE_cv_idx_opt = 0


            for epoch in range(self.N_steps):
                # ---------------- Training ----------------
                model_mstep.train()
                train_loss_sum = 0.0


                for _ in range(self.N_B):
                    self.M_optimizer.zero_grad()

                    # Pick one training sequence
                    n_e = random.randint(0, self.N_E - 1)
                    y_seq = train_input[n_e]   # [n, T]
                    x_true_seq = train_target[n_e]  # [m, T]
                    T = y_seq.size(-1)

                    # Select F_i and F_true by group
                    if generate_f is True:
                        f_index = n_e // 10
                        F_base = SysModel.F_train[f_index]
                        F_true = SysModel.F_train_TRUE[f_index]
                    else:
                        F_base = SysModel.F_train[n_e]
                        F_true = SysModel.F_train_TRUE[n_e]
                    # --------- EM unrolling over F ---------
                    F_current = F_base  # this will be updated each EM iteration
                    total_loss = 0.0

                    for em_iter in range(num_em_iters):

                        self.model.update_F(F_current)

                        # ---------- E-STEP: RTSNET FOR x_smooth + sigma_list + smoother_gain_list ----------
                        self.model.InitSequence(SysModel.m1x_0, T)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                        x_forward = torch.empty(m, T, device=dev)
                        x_smooth = torch.empty(m, T, device=dev)

                        sigma_list = []
                        smoother_gain_list = []

                        # Forward pass
                        for t in range(T):
                            x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                            # store encoded covariance representation
                            sigma_list.append(self.model.h_Sigma.clone())

                        # Backward pass for x_smooth
                        x_smooth[:, T - 1] = x_forward[:, T - 1]
                        self.model.InitBackward(x_smooth[:, T - 1])

                        x_smooth[:, T - 2] = self.model(None,
                                                        x_forward[:, T - 2],
                                                        x_forward[:, T - 1],
                                                        None)
                        smoother_gain_list.append(self.model.SGain.clone())

                        for t in range(T - 3, -1, -1):
                            x_smooth[:, t] = self.model(None,
                                                        x_forward[:, t],
                                                        x_forward[:, t + 1],
                                                        x_smooth[:, t + 2])
                            smoother_gain_list.append(self.model.SGain.clone())

                        # ---------- PsmoothNet: P_smoothed_seq and its average ----------
                        dt = y_seq.dtype
                        P_smoothed_seq = torch.empty(m, m, T, device=dev, dtype=dt)
                        dummy_sgain = torch.zeros(1, 1, m * m, device=dev, dtype=dt)

                        # Final time step
                        sigma_T = sigma_list[-1]
                        self.psmooth_model.start = 0
                        P_flat = self.psmooth_model(sigma_T, dummy_sgain).view(-1)
                        P_smoothed_seq[:, :, T - 1] = self.psmooth_model.enforce_covariance_properties(
                            P_flat.view(m, m))

                        # Backward in time for P_t
                        for t in range(T - 2, -1, -1):
                            sigma_t = sigma_list[t]
                            sgain_index = (T - 2) - t
                            sgain_t = smoother_gain_list[sgain_index].reshape(1, 1, -1)
                            P_flat = self.psmooth_model(sigma_t, sgain_t)
                            P_smoothed_seq[:, :, t] = self.psmooth_model.enforce_covariance_properties(
                                P_flat.view(-1).view(m, m))

                        # Average P_smooth over time: P̄ = (1/T) Σ P_t
                        P_avg = P_smoothed_seq.mean(dim=2)  # [m, m]

                        # ---------- Stats for M-network using x_smooth & P_avg ----------

                        x_curr = x_smooth  # [m, T]
                        x_prev = torch.empty_like(x_curr)
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                        x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}

                        # A1, A2
                        A1 = (x_curr @ x_prev.T) / T
                        A2 = (x_prev @ x_prev.T) / T
                        # Predicted previous state and delta_x
                        x_minus = F_current @ x_prev  # [m, T]
                        delta_x = x_curr - x_minus  # [m, T]

                        # Cross term C_{Δx, x⁻} = 1/T Σ Δx_t x⁻_t^T
                        C_delta_x_xminus = (delta_x @ x_minus.T) / T

                        # y_hat and innovation covariance S_nu
                        if non_linear_h:
                            y_hat_list = []
                            for t in range(T):
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)
                                y_hat_list.append(y_t_hat.view(-1))
                            Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                        else:
                            H = SysModel.H
                            Hx_curr = H @ x_curr  # [n, T]

                        nu = y_seq - Hx_curr
                        nu_mean = nu.mean(dim=1, keepdim=True)
                        nu_centered = nu - nu_mean
                        S_nu = (nu_centered @ nu_centered.T) / T

                        delta_mean = delta_x.mean(dim=1, keepdim=True)
                        delta_centered = delta_x - delta_mean
                        S_delta_x = (delta_centered @ delta_centered.T) / T

                        # ---------- BUILD INPUT z_in = [A1, A2, S_delta_x, S_nu, C_delta_x_xminus, P_avg] ----------
                        z_in = torch.cat([
                            A1.reshape(-1),
                            A2.reshape(-1),
                            S_delta_x.reshape(-1),
                            S_nu.reshape(-1),
                            C_delta_x_xminus.reshape(-1),
                            P_avg.reshape(-1)  # <-- new
                        ], dim=0).reshape(1, -1)

                        # ---------- M-step network: predict ΔF and update ----------
                        deltaF = model_mstep(z_in)
                        deltaF_mat = deltaF.view(m, m)
                        F_next = F_current + deltaF_mat

                        # Loss components
                        f_loss = torch.mean((F_next - F_true) ** 2)
                        reg = lambda_F * torch.mean(deltaF_mat ** 2)
                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)

                        # EM-iteration weighting
                        if em_iter == num_em_iters - 1:
                            loss_em = 15 * f_loss + reg + x_loss
                        else:
                            loss_em = f_loss + reg + x_loss

                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]

                        total_loss += weight * loss_em
                        F_current = F_next


                    # after `for em_iter in range(num_em_iters):`
                    loss = total_loss / float(num_em_iters)   # average over EM iterations
                    loss_mult = loss
                    loss_mult.backward()
                    torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                    self.M_optimizer.step()

                    train_loss_sum += loss.detach().item()

                # ---------------- Validation ----------------
                model_mstep.eval()
                cv_loss_sum = 0.0

                with torch.no_grad():
                    for j in range(self.N_CV):
                        y_cv = cv_input[j]  # [n, T_cv]
                        x_true_cv_seq = cv_target[j]  # [m, T_cv]
                        T_cv = y_cv.size(-1)
                        m = SysModel.m

                        # choose base F and F_true for this CV sequence
                        if generate_f is True:
                            f_index_cv = j // 10
                            F_base_cv = SysModel.F_valid[f_index_cv]
                            F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                        else:
                            F_base_cv = SysModel.F_valid[j]
                            F_true_cv = SysModel.F_valid_TRUE[j]

                        F_current_cv = F_base_cv.clone()
                        total_loss_cv = 0.0

                        for em_iter in range(num_em_iters):

                            # --- RTS smoother with current F_current_cv ---
                            self.model.update_F(F_current_cv)
                            self.model.InitSequence(SysModel.m1x_0.to(dev), T_cv)
                            self.model.init_hidden()
                            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(dev)

                            dev = y_cv.device
                            dt = y_cv.dtype

                            x_f_cv = torch.empty(m, T_cv, device=dev, dtype=dt)
                            x_s_cv = torch.empty(m, T_cv, device=dev, dtype=dt)

                            sigma_list_cv = []
                            smoother_gain_list_cv = []

                            # forward pass
                            for t in range(T_cv):
                                x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)
                                sigma_list_cv.append(self.model.h_Sigma.clone())

                            # backward pass for x_s_cv
                            x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                            self.model.InitBackward(x_s_cv[:, T_cv - 1])

                            x_s_cv[:, T_cv - 2] = self.model(None,
                                                             x_f_cv[:, T_cv - 2],
                                                             x_f_cv[:, T_cv - 1],
                                                             None)
                            smoother_gain_list_cv.append(self.model.SGain.clone())

                            for t in range(T_cv - 3, -1, -1):
                                x_s_cv[:, t] = self.model(None,
                                                          x_f_cv[:, t],
                                                          x_f_cv[:, t + 1],
                                                          x_s_cv[:, t + 2])
                                smoother_gain_list_cv.append(self.model.SGain.clone())

                            # --- PsmoothNet: compute P_smoothed_seq and P_avg (no grad) ---
                            P_smoothed_seq_cv = torch.empty(m, m, T_cv, device=dev, dtype=dt)
                            dummy_sgain = torch.zeros(1, 1, m * m, device=dev, dtype=dt)

                            # final time step
                            sigma_T = sigma_list_cv[-1]
                            self.psmooth_model.start = 0
                            P_flat = self.psmooth_model(sigma_T, dummy_sgain).view(-1)
                            P_smoothed_seq_cv[:, :, T_cv - 1] = self.psmooth_model.enforce_covariance_properties(
                                P_flat.view(m, m)
                            )

                            # backward in time for P_t
                            for t in range(T_cv - 2, -1, -1):
                                sigma_t = sigma_list_cv[t]
                                sgain_index = (T_cv - 2) - t
                                sgain_t = smoother_gain_list_cv[sgain_index].reshape(1, 1, -1)
                                P_flat = self.psmooth_model(sigma_t, sgain_t)
                                P_smoothed_seq_cv[:, :, t] = self.psmooth_model.enforce_covariance_properties(
                                    P_flat.view(-1).view(m, m)
                                )

                            # average P_smooth over time
                            P_avg_cv = P_smoothed_seq_cv.mean(dim=2)  # [m, m]

                            # -------- stats, same as training but with P_avg_cv and NO S_delta_x / C_delta_x --------
                            x_curr = x_s_cv  # [m, T_cv]
                            x_prev = torch.empty_like(x_curr)
                            x_prev[:, 0] = SysModel.m1x_0.view(-1)
                            x_prev[:, 1:] = x_curr[:, :-1]

                            A1_cv = (x_curr @ x_prev.T) / T_cv
                            A2_cv = (x_prev @ x_prev.T) / T_cv

                            # Predicted previous state and delta_x for CV
                            x_minus_cv = F_current_cv @ x_prev  # [m, T_cv]
                            delta_x_cv = x_curr - x_minus_cv  # [m, T_cv]

                            # Cross term C_{Δx, x⁻} for CV
                            C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                            if non_linear_h:
                                y_hat_cv_list = []
                                for t in range(T_cv):
                                    x_t = x_curr[:, t].view(SysModel.m, 1)
                                    y_t_hat = SysModel.h(x_t)  # non-linear h
                                    y_hat_cv_list.append(y_t_hat.view(-1))
                                Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                            else:
                                H = SysModel.H.to(device)
                                Hx_curr_cv = H @ x_curr

                            nu_cv = y_cv - Hx_curr_cv
                            nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                            nu_centered_cv = nu_cv - nu_mean_cv
                            S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                            delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                            delta_centered_cv = delta_x_cv - delta_mean_cv
                            S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv

                            z_cv = torch.cat([
                                A1_cv.reshape(-1),
                                A2_cv.reshape(-1),
                                S_delta_x_cv.reshape(-1),
                                S_nu_cv.reshape(-1),
                                C_delta_x_xminus_cv.reshape(-1),
                                P_avg_cv.reshape(-1)  # <-- new
                            ], dim=0).reshape(1, -1)

                            # --- M-step forward only (no grad) ---
                            dF_cv = model_mstep(z_cv)
                            dF_cv_mat = dF_cv.view(m, m)
                            F_next_cv = F_current_cv + dF_cv_mat

                            # same loss as train (but no backward)
                            f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)
                            reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                            x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)

                            if em_iter == num_em_iters - 1:
                                loss_em_cv = 15 * f_loss_cv + reg_cv + x_loss_cv
                            else:
                                loss_em_cv =  f_loss_cv + reg_cv + x_loss_cv

                            if em_iter == 0:
                                weight = alpha[0]
                            elif em_iter == 1:
                                weight = alpha[1]
                            else:
                                weight = alpha[2]  # keep same scaling as train

                            total_loss_cv += weight * loss_em_cv
                            F_current_cv = F_next_cv

                        cv_loss_seq = total_loss_cv / float(num_em_iters)
                        cv_loss_sum += cv_loss_seq.item()

                    train_epoch = train_loss_sum / max(1, self.N_B)
                    cv_epoch = cv_loss_sum / max(1, self.N_CV)

                    if cv_epoch < self.MSE_cv_dB_opt:
                        self.MSE_cv_dB_opt = cv_epoch
                        torch.save(model_mstep, destination_path_M)

                    print(
                        f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")

    def test_mstep_net_with_p(self, SysModel, test_input, test_target,
                              destination_path_RTS, destination_path_M, destination_path_PSMOOTH,
                              num_em_iters=3,
                              alpha=(0.0, 0.0, 1.0), lambda_F=1e-3,
                              generate_f=True, init_x_list=None, init_P_list=None, non_linear_h=False):
        """
        Testing-only version for the M-step network WITH Psmooth.
        - Freeze RTSNet (destination_path_RTS) to compute x_smooth and sigma_list.
        - Freeze PsmoothNet (destination_path_PSMOOTH) to compute P_smooth and P_avg.
        - Use M-step network (destination_path_M) to predict ΔF.
        - z_in = [A1, A2, S_nu, C_delta_x_xminus, F_current, P_avg]  (NO S_delta_x).
        """

        N_T = len(test_input)
        m = SysModel.m

        all_test_losses = []
        all_f_losses = []

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load and freeze PsmoothNet
        self.psmooth_model = torch.load(destination_path_PSMOOTH,
                                        weights_only=False,
                                        map_location=device).to(device).eval()
        for p in self.psmooth_model.parameters():
            p.requires_grad_(False)

        # Load M-step network from checkpoint (NO training)
        model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        final_F_list = []
        final_x_list = []

        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]  # [n, T]
                x_true_seq = test_target[j]  # [m, T]
                T = y_seq.size(-1)

                # Select F_base and F_true for this test sequence
                if generate_f is True:
                    f_index = j // 10
                    F_base = SysModel.F_test[f_index].to(device)
                    F_true = SysModel.F_test_TRUE[f_index].to(device)
                else:
                    F_base = SysModel.F_test[j].to(device)
                    F_true = SysModel.F_test_TRUE[j].to(device)

                F_current = F_base.clone()
                total_loss = 0.0
                F_estimates = []
                F_losses_mse = []
                F_losses_total = []
                x_losses_mse = []
                y_loss_tot = []

                # initial x0, P0 (for EM initialization)
                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                for em_iter in range(num_em_iters):

                    # ----- RTS smoother with current F_current -----
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = P0.clone().detach()

                    dev = y_seq.device
                    dt = y_seq.dtype

                    x_forward = torch.empty(m, T, device=dev, dtype=dt)
                    x_smooth = torch.empty(m, T, device=dev, dtype=dt)

                    sigma_list = []
                    smoother_gain_list = []

                    # forward pass
                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                        sigma_list.append(self.model.h_Sigma.clone())

                    # backward pass for x_smooth
                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])

                    x_smooth[:, T - 2] = self.model(
                        None,
                        x_forward[:, T - 2],
                        x_forward[:, T - 1],
                        None
                    )
                    smoother_gain_list.append(self.model.SGain.clone())

                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(
                            None,
                            x_forward[:, t],
                            x_forward[:, t + 1],
                            x_smooth[:, t + 2]
                        )
                        smoother_gain_list.append(self.model.SGain.clone())

                    # ----- PsmoothNet: compute P_smoothed_seq and P_avg (no grad) -----
                    P_smoothed_seq = torch.empty(m, m, T, device=dev, dtype=dt)
                    dummy_sgain = torch.zeros(1, 1, m * m, device=dev, dtype=dt)

                    # final time step
                    sigma_T = sigma_list[-1]
                    self.psmooth_model.start = 0
                    P_flat = self.psmooth_model(sigma_T, dummy_sgain).view(-1)
                    P_smoothed_seq[:, :, T - 1] = self.psmooth_model.enforce_covariance_properties(
                        P_flat.view(m, m)
                    )

                    # backward in time for P_t
                    for t in range(T - 2, -1, -1):
                        sigma_t = sigma_list[t]
                        sgain_index = (T - 2) - t
                        sgain_t = smoother_gain_list[sgain_index].reshape(1, 1, -1)
                        P_flat = self.psmooth_model(sigma_t, sgain_t)
                        P_smoothed_seq[:, :, t] = self.psmooth_model.enforce_covariance_properties(
                            P_flat.view(-1).view(m, m)
                        )

                    # average P_smooth over time
                    P_avg = P_smoothed_seq.mean(dim=2)  # [m, m]

                    # ----- stats for M-network (same as training) -----
                    x_curr = x_smooth  # [m, T]
                    x_prev = torch.empty_like(x_curr)  # [m, T]
                    x_prev[:, 0] = x0.view(-1)  # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}

                    A1 = (x_curr @ x_prev.T) / T
                    A2 = (x_prev @ x_prev.T) / T

                    x_minus = F_current @ x_prev  # [m, T]
                    delta_x = x_curr - x_minus  # [m, T]

                    # C_{Δx, x⁻} = 1/T Σ Δx_t x⁻_t^T
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T
                    delta_mean = delta_x.mean(dim=1, keepdim=True)
                    delta_centered = delta_x - delta_mean
                    S_delta_x = (delta_centered @ delta_centered.T) / T

                    # ---------- linear vs non-linear h ----------
                    if non_linear_h:
                        y_hat_list = []
                        for t in range(T):
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)
                            y_hat_list.append(y_t_hat.view(-1))
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H.to(device)
                        Hx_curr = H @ x_curr  # [n, T]

                    nu = y_seq - Hx_curr
                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T

                    # ---------- BUILD INPUT z_in (NO S_delta_x) ----------
                    z_in = torch.cat([
                        A1.reshape(-1),
                        A2.reshape(-1),
                        S_delta_x.reshape(-1),
                        S_nu.reshape(-1),
                        C_delta_x_xminus.reshape(-1),
                        P_avg.reshape(-1)  # same as training
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only -----
                    deltaF = model_mstep(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat

                    mse_F = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)

                    loss_em = 3 * mse_F + reg + x_loss
                    x_loss_sum_per_iter[em_iter] += x_loss.item()

                    # same alpha weighting as in training
                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]

                    total_loss += weight * loss_em
                    F_current = F_next

                    all_test_losses.append(loss_em.item())
                    all_f_losses.append(mse_F.item())

                    if j % 5 == 0:
                        F_estimates.append(F_next.detach())
                        F_losses_mse.append(mse_F.item())
                        F_losses_total.append(loss_em.item())
                        x_losses_mse.append(x_loss.item())

                # store final F and final x_smooth for this sequence (after last EM iter)
                final_F_list.append(F_current.detach().clone())  # [m, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())

                # final loss for this sequence (already weighted)
                loss_list[j] = total_loss / float(num_em_iters)

                if j % 5 == 0:
                    print(f"\n[M-step TEST with P] sequence {j} summary")
                    print("F_true:\n", F_true.detach())
                    print("F_init (F_base):\n", F_base.detach())
                    mse_F_init = torch.mean((F_base - F_true) ** 2).item()
                    print(f"Initial F MSE loss = {mse_F_init:.6e}")
                    for k, (F_est, f_mse, x_mse, total_val) in enumerate(
                            zip(F_estimates, F_losses_mse, x_losses_mse, F_losses_total)):
                        f_db = 10.0 * math.log10(f_mse)
                        x_db = 10.0 * math.log10(x_mse)
                        tot_db = 10.0 * math.log10(total_val)
                        print(f"\n  EM iter {k + 1}:")
                        print("  F_est:\n", F_est)
                        print(f"  F-loss (MSE_F)           = {f_db:.2f} dB")
                        print(f"  x-loss (MSE_x)           = {x_db:.2f} dB")
                        print(f"  total loss (F+reg+x)     = {tot_db:.2f} dB")

        mean_loss = loss_list.mean().item()
        print(f"[M-step TEST with P] mean_loss={mean_loss:.6f}")

        # average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)
        print("[M-step TEST with P] Mean x-MSE per EM iteration:")
        for k in range(num_em_iters):
            mse_k = mean_x_mse_per_iter[k].item()
            db_k = 10.0 * math.log10(mse_k)
            print(f"  EM iter {k + 1}: mean x-MSE = {mse_k:.6e}  ({db_k: .2f} dB)")

        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()

        return mean_x_mse_per_iter_np, mean_x_mse_per_iter_db_np, final_F_list, final_x_list

    def no_rts_end_To_end_m_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                         destination_path_M, load_base_m_mmodel=None, load_rts=None,
                         num_em_iters=3,
                         alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, non_linear_h=False):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load a single RTSNet and freeze it (same for all EM iterations)
        self.RTS_model = torch.load(load_rts, weights_only=False).to(self.device)
        self.RTS_model.train()
        for p in self.RTS_model.parameters():
            p.requires_grad = False


        self.M_models = []
        if load_base_m_mmodel != None:
            for i in range(num_em_iters):
                M_model = torch.load(load_base_m_mmodel, map_location=self.device, weights_only=False).to(
                    self.device)
                self.M_models.append(M_model.train())
        else:
            for i in range(num_em_iters):
                M_model = copy.deepcopy(self.M_model)
                self.M_models.append(M_model.train())

        self.M_optimizers = []

        stable_lr = self.learningRate * 0.1
        for i in range(num_em_iters):
            self.M_optimizers.append(
                torch.optim.Adam(self.M_models[i].parameters(), lr=stable_lr, weight_decay=self.weightDecay))


        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            for M_model in self.M_models:
                M_model.train()
            self.RTS_model.train()
            train_loss_sum = 0.0

            # zero grad for all M nets
            for opt in self.M_optimizers:
                opt.zero_grad()
            epoch_loss_sum = 0.0

            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select F_i and F_true by group
                if generate_f is True:
                    f_index = n_e // 10
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                # NEW: Select H for this training sequence
                if generate_h is True:
                    h_index = n_e // 10
                    H_current = SysModel.H_train[h_index]
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                # --------- EM unrolling over F ---------
                F_current = F_base  # this will be updated each EM iteration
                total_loss_12 = 0.0
                total_loss_3 = 0.0

                for em_iter in range(num_em_iters):


                    M_k = self.M_models[em_iter]
                    self.model = self.RTS_model

                    self.model.update_F(F_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    x_prev = torch.empty_like(x_curr)  # [m, T]
                    x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                    # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                    A1 = (x_curr @ x_prev.T) / T
                    # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                    A2 = (x_prev @ x_prev.T) / T
                    # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                    x_minus = F_current @ x_prev  # [m, T_eff]
                    # Δx_t = x_t - F*x_{t-1|T}
                    delta_x = x_curr - x_minus  # [m, T]
                    delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                    delta_centered = delta_x - delta_mean
                    # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                    S_delta_x = (delta_centered @ delta_centered.T) / T
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr
                    # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                    nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T
                    # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T


                    if em_iter == num_em_iters - 1:
                        # last iter: detach so f3 won't affect M1/M2
                        F_for_update = F_current.detach()
                        z_in = torch.cat([A1.reshape(-1).detach(), A2.reshape(-1).detach(), S_delta_x.reshape(-1).detach(), S_nu.reshape(-1).detach(),
                                          C_delta_x_xminus.detach().reshape(-1),
                                          F_for_update.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]
                    else:
                        F_for_update = F_current
                        z_in = torch.cat([A1.reshape(-1), A2.reshape(-1), S_delta_x.reshape(-1), S_nu.reshape(-1),
                                          C_delta_x_xminus.reshape(-1),
                                          F_for_update.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]

                    # Predict ΔF and update F
                    deltaF = M_k(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_for_update  + deltaF_mat

                    # Loss: Frobenius(F_next - F_true)^2 + regularization
                    f_loss = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################

                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)

                    if em_iter < num_em_iters - 1:
                        # Iter 1 & 2:
                        #  - f1,x1,f2,x2 go into total_loss_12 (train M1,M2)
                        loss_iter_12 = 0.5*f_loss + reg + x_loss
                        total_loss_12 += weight * loss_iter_12
                    else:
                        # Iter 3:
                        #  - x3 goes to total_loss_12 (so M1,M2 see it)
                        #  - f3 goes to total_loss_3 (trains M3 only)
                        total_loss_12 += weight * x_loss
                        total_loss_3 += weight * (f_loss + reg)

                    F_current = F_next  # use updated F in next EM iteration

                # after `for em_iter in range(num_em_iters):`
                total_loss = total_loss_12 + total_loss_3  # full loss used for backward

                loss = total_loss / float(num_em_iters)  # for logging

                epoch_loss_sum += total_loss
                train_loss_sum += loss.detach().item()

            epoch_loss_mean = epoch_loss_sum / float(self.N_B)
            # one backward pass for the whole batch
            epoch_loss_mean.backward()

            for idx, M_model in enumerate(self.M_models):
                grad_norm = 0.0
                for p in M_model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                print(f"[DEBUG] epoch={epoch} M{idx + 1} grad_norm={grad_norm:.3e}")

            for M_model, opt in zip(self.M_models, self.M_optimizers):
                torch.nn.utils.clip_grad_norm_(M_model.parameters(), max_norm=1.0)
                opt.step()

            train_epoch = epoch_loss_mean.item()

            # ---------------- Validation ----------------
            for M_model in self.M_models:
                M_model.eval()
            self.RTS_model.eval()
            cv_loss_sum = 0.0


            with torch.no_grad():
                for j in range(self.N_CV):
                    total_loss_12_cv = 0.0
                    total_loss_3_cv = 0.0
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    # choose base F and F_true for this CV sequence
                    if generate_f is True:
                        f_index_cv = j // 10
                        F_base_cv = SysModel.F_valid[f_index_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()
                    total_loss_cv = 0.0

                    for em_iter in range(num_em_iters):

                        M_k = self.M_models[em_iter]
                        self.model = self.RTS_model

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_F(F_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        ################################################
                        if epoch % 10 == 0 and j == 2:
                            # מרחק של F_current לפני העדכון מה-F_true
                            f_err_before = torch.mean((F_current_cv - F_true_cv) ** 2).item()
                            x_err_before = torch.mean((x_curr - x_true_cv_seq) ** 2).item()
                            print(
                                f"[DEBUG][epoch={epoch} em={em_iter}] F_err_before={f_err_before:.3e}, x_err_before={x_err_before:.3e}")
                            #######################################
                        x_prev = torch.empty_like(x_curr)
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)
                        x_prev[:, 1:] = x_curr[:, :-1]

                        A1_cv = (x_curr @ x_prev.T) / T_cv
                        A2_cv = (x_prev @ x_prev.T) / T_cv

                        x_minus_cv = F_current_cv @ x_prev
                        delta_x_cv = x_curr - x_minus_cv

                        delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                        delta_centered_cv = delta_x_cv - delta_mean_cv
                        S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv

                        if non_linear_h:
                            y_hat_cv_list = []
                            for t in range(T_cv):
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)  # non-linear h
                                y_hat_cv_list.append(y_t_hat.view(-1))
                            Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                        else:
                            H = SysModel.H.to(device)
                            Hx_curr_cv = H @ x_curr
                        nu_cv = y_cv - Hx_curr_cv

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_centered_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                        C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                        z_cv = torch.cat([
                            A1_cv.reshape(-1),
                            A2_cv.reshape(-1),
                            S_delta_x_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_delta_x_xminus_cv.reshape(-1),
                            F_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dF_cv = M_k(z_cv)
                        dF_cv_mat = dF_cv.view(m, m)
                        F_next_cv = F_current_cv + dF_cv_mat

                        # same loss as train (but no backward)
                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                        reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################

                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]  # if you really want the same scaling as in train

                        if em_iter < num_em_iters - 1:
                            # Iter 1 & 2:
                            #  - f1,x1,f2,x2 go into total_loss_12 (train M1,M2)
                            loss_iter_12_cv = 0.5 * f_loss_cv + reg_cv + x_loss_cv
                            total_loss_12_cv += weight * loss_iter_12_cv
                        else:
                            # Iter 3:
                            #  - x3 goes to total_loss_12 (so M1,M2 see it)
                            #  - f3 goes to total_loss_3 (trains M3 only)
                            total_loss_12_cv += weight * x_loss_cv
                            total_loss_3_cv += weight * (f_loss_cv + reg_cv)
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        F_current_cv = F_next_cv

                    total_loss_cv = total_loss_12_cv + total_loss_3_cv
                    cv_loss_seq = total_loss_cv / float(num_em_iters)
                    cv_loss_sum += cv_loss_seq.item()

            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                for k in range(num_em_iters):
                    torch.save(self.M_models[k], destination_path_M[k])
            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")


    def no_rts_end_To_end_m_net_2(self, SysModel, cv_input, cv_target, train_input, train_target,
                         destination_path_M, load_base_m_mmodel=None, load_rts=None,
                         num_em_iters=3,
                         lambda_F=1e-3, generate_f=True, non_linear_h=False):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load a single RTSNet and freeze it (same for all EM iterations)
        self.RTS_model = torch.load(load_rts, weights_only=False).to(self.device)
        self.RTS_model.train()
        for p in self.RTS_model.parameters():
            p.requires_grad = False


        self.M_models = []
        if load_base_m_mmodel != None:
            for i in range(num_em_iters):
                M_model = torch.load(load_base_m_mmodel, map_location=self.device, weights_only=False).to(
                    self.device)
                self.M_models.append(M_model.train())
        else:
            for i in range(num_em_iters):
                M_model = copy.deepcopy(self.M_model)
                self.M_models.append(M_model.train())

        self.M_optimizers = []

        stable_lr = self.learningRate * 0.1
        for i in range(num_em_iters):
            self.M_optimizers.append(
                torch.optim.Adam(self.M_models[i].parameters(), lr=stable_lr, weight_decay=self.weightDecay))


        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            for M_model in self.M_models:
                M_model.train()
            self.RTS_model.train()
            train_loss_sum = 0.0

            # zero grad for all M nets
            for opt in self.M_optimizers:
                opt.zero_grad()
            epoch_loss_sum = 0.0

            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select F_i and F_true by group
                if generate_f is True:
                    f_index = n_e // 10
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                # NEW: Select H for this training sequence
                if generate_h is True:
                    h_index = n_e // 10
                    H_current = SysModel.H_train[h_index]
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                # --------- EM unrolling over F ---------
                F_current = F_base  # this will be updated each EM iteration
                total_loss_1 = 0.0
                total_loss_2 = 0.0
                total_loss_3 = 0.0

                for em_iter in range(num_em_iters):


                    M_k = self.M_models[em_iter]
                    self.model = self.RTS_model

                    self.model.update_F(F_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    x_prev = torch.empty_like(x_curr)  # [m, T]
                    x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                    # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                    A1 = (x_curr @ x_prev.T) / T
                    # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                    A2 = (x_prev @ x_prev.T) / T
                    # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                    x_minus = F_current @ x_prev  # [m, T_eff]
                    # Δx_t = x_t - F*x_{t-1|T}
                    delta_x = x_curr - x_minus  # [m, T]
                    delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                    delta_centered = delta_x - delta_mean
                    # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                    S_delta_x = (delta_centered @ delta_centered.T) / T
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr
                    # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                    nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T
                    # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    if em_iter == 0:
                        F_for_update = F_current  # no detach, so F1 depends on ΔF0
                    else:
                        F_for_update = F_current.detach()  # cut gradients to previous ΔF's

                    # Stats do NOT need gradient, detach them before feeding to M_k:
                    z_in = torch.cat([
                        A1.reshape(-1).detach(),
                        A2.reshape(-1).detach(),
                        S_delta_x.reshape(-1).detach(),
                        S_nu.reshape(-1).detach(),
                        C_delta_x_xminus.reshape(-1).detach(),
                        F_for_update.reshape(-1)  # only this may have grad (for current Mk)
                    ], dim=0).reshape(1, -1)  # [1, feature_dim]

                    # Predict ΔF and update F
                    deltaF = M_k(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_for_update  + deltaF_mat

                    # Loss: Frobenius(F_next - F_true)^2 + regularization
                    f_loss = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################

                    if em_iter ==0:
                        # Iter 1 & 2:
                        #  - f1,x1,f2,x2 go into total_loss_12 (train M1,M2)
                        loss_iter_1 = 15*f_loss + reg
                        total_loss_1 +=  loss_iter_1
                    elif em_iter ==1:
                        loss_iter_2 = 15*f_loss + reg
                        total_loss_2 +=  loss_iter_2
                        loss_iter_1 = x_loss
                        total_loss_1 +=  loss_iter_1
                    else:
                        # Iter 3:
                        #  - x3 goes to total_loss_12 (so M1,M2 see it)
                        total_loss_3 += f_loss + reg
                        loss_iter_2 = x_loss
                        total_loss_2 +=  loss_iter_2

                    F_current = F_next  # use updated F in next EM iteration

                # after `for em_iter in range(num_em_iters):`
                total_loss = total_loss_1 +total_loss_2 + total_loss_3  # full loss used for backward

                loss = total_loss / float(num_em_iters)  # for logging

                epoch_loss_sum += total_loss
                train_loss_sum += loss.detach().item()

            epoch_loss_mean = epoch_loss_sum / float(self.N_B)
            # one backward pass for the whole batch
            epoch_loss_mean.backward()

            for idx, M_model in enumerate(self.M_models):
                grad_norm = 0.0
                for p in M_model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                print(f"[DEBUG] epoch={epoch} M{idx + 1} grad_norm={grad_norm:.3e}")

            for M_model, opt in zip(self.M_models, self.M_optimizers):
                torch.nn.utils.clip_grad_norm_(M_model.parameters(), max_norm=1.0)
                opt.step()

            train_epoch = epoch_loss_mean.item()

            # ---------------- Validation ----------------
            for M_model in self.M_models:
                M_model.eval()
            self.RTS_model.eval()
            cv_loss_sum = 0.0


            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    # choose base F and F_true for this CV sequence
                    if generate_f is True:
                        f_index_cv = j // 10
                        F_base_cv = SysModel.F_valid[f_index_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()
                    total_loss_cv = 0.0
                    total_loss_1_cv = 0.0
                    total_loss_2_cv = 0.0
                    total_loss_3_cv = 0.0

                    for em_iter in range(num_em_iters):

                        M_k = self.M_models[em_iter]
                        self.model = self.RTS_model

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_F(F_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        ################################################
                        if epoch % 10 == 0 and j == 2:
                            # מרחק של F_current לפני העדכון מה-F_true
                            f_err_before = torch.mean((F_current_cv - F_true_cv) ** 2).item()
                            x_err_before = torch.mean((x_curr - x_true_cv_seq) ** 2).item()
                            print(
                                f"[DEBUG][epoch={epoch} em={em_iter}] F_err_before={f_err_before:.3e}, x_err_before={x_err_before:.3e}")
                            #######################################
                        x_prev = torch.empty_like(x_curr)
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)
                        x_prev[:, 1:] = x_curr[:, :-1]

                        A1_cv = (x_curr @ x_prev.T) / T_cv
                        A2_cv = (x_prev @ x_prev.T) / T_cv

                        x_minus_cv = F_current_cv @ x_prev
                        delta_x_cv = x_curr - x_minus_cv

                        delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                        delta_centered_cv = delta_x_cv - delta_mean_cv
                        S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv

                        if non_linear_h:
                            y_hat_cv_list = []
                            for t in range(T_cv):
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)  # non-linear h
                                y_hat_cv_list.append(y_t_hat.view(-1))
                            Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                        else:
                            H = SysModel.H.to(device)
                            Hx_curr_cv = H @ x_curr
                        nu_cv = y_cv - Hx_curr_cv

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_centered_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                        C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                        z_cv = torch.cat([
                            A1_cv.reshape(-1),
                            A2_cv.reshape(-1),
                            S_delta_x_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_delta_x_xminus_cv.reshape(-1),
                            F_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dF_cv = M_k(z_cv)
                        dF_cv_mat = dF_cv.view(m, m)
                        F_next_cv = F_current_cv + dF_cv_mat

                        # same loss as train (but no backward)
                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                        reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################

                        if em_iter == 0:
                            # Iter 1 & 2:
                            #  - f1,x1,f2,x2 go into total_loss_12 (train M1,M2)
                            loss_iter_1_cv = 0.05*f_loss_cv + reg_cv
                        elif em_iter == 1:
                            loss_iter_2_cv = 0.1*f_loss_cv + reg_cv+ 0.1*x_loss_cv
                            total_loss_2_cv += loss_iter_2_cv
                        else:
                            # Iter 3:
                            #  - x3 goes to total_loss_12 (so M1,M2 see it)
                            total_loss_3_cv += 5*f_loss_cv + reg_cv +x_loss_cv

                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        F_current_cv = F_next_cv

                    total_loss_cv = total_loss_1_cv + total_loss_2_cv + total_loss_3_cv
                    cv_loss_seq = total_loss_cv / float(num_em_iters)
                    cv_loss_sum += cv_loss_seq.item()

            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                for k in range(num_em_iters):
                    torch.save(self.M_models[k], destination_path_M[k])
            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")



    def NO_RTS_end_to_end_test_mstep_net(self, SysModel, test_input, test_target, destination_path_RTS, destination_path_M,
                                  num_em_iters=3,alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, init_x_list=None,
                                  init_P_list=None, non_linear_h=False):
        """
        Testing-only version for the M-step network.
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Use CURRENT M-step network (self.M_model) to predict ΔF.
        - Run num_em_iters EM iterations per test sequence.
        - No training, no optimizer step.
        - Returns per-sequence loss and mean loss.
        """

        N_T = len(test_input)
        m = SysModel.m

        all_test_losses = []
        all_f_losses = []

        RTS_models = []

        self.model = torch.load(destination_path_RTS, weights_only=False, map_location=device).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)


        M_models = []
        for k in range(num_em_iters):
            M_k = torch.load(destination_path_M[k], weights_only=False, map_location=device)
            M_k = M_k.to(device).eval()
            M_models.append(M_k)

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        final_F_list = []
        final_x_list = []
        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]  # [n, T]
                x_true_seq = test_target[j]  # [m, T]

                T = y_seq.size(-1)

                # Select F_base and F_true for this test sequence
                if generate_f is True:
                    f_index = j // 10
                    F_base = SysModel.F_test[f_index].to(device)
                    F_true = SysModel.F_test_TRUE[f_index].to(device)
                else:
                    # fallback: sequence-wise
                    F_base = SysModel.F_test[j].to(device)
                    F_true = SysModel.F_test_TRUE[j].to(device)

                # NEW: Select H for this test sequence
                if generate_h is True:
                    h_index = j // 10
                    H_current = SysModel.H_test[h_index].to(device)
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                F_current = F_base.clone()
                total_loss = 0.0
                F_estimates = []
                F_losses_mse = []
                F_losses_total = []
                x_losses_mse = []

                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                for em_iter in range(num_em_iters):

                    M_k = M_models[em_iter]

                    # ----- RTS smoother with current F_current -----
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = P0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ----- stats for M-network -----
                    x_curr = x_smooth  # [m, T]
                    x_prev = torch.empty_like(x_curr)  # [m, T]
                    x_prev[:, 0] = x0.view(-1)  # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}

                    A1 = (x_curr @ x_prev.T) / T
                    A2 = (x_prev @ x_prev.T) / T

                    x_minus = F_current @ x_prev  # [m, T]
                    delta_x = x_curr - x_minus  # [m, T]

                    delta_mean = delta_x.mean(dim=1, keepdim=True)
                    delta_centered = delta_x - delta_mean
                    S_delta_x = (delta_centered @ delta_centered.T) / T

                    # ---------- linear vs non-linear h ----------
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            x_t = x_curr[:, t].view(SysModel.m, 1)  # [m,1]
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))  # flatten to [n]
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H.to(device)
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T

                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    z_in = torch.cat([
                        A1.reshape(-1),
                        A2.reshape(-1),
                        S_delta_x.reshape(-1),
                        S_nu.reshape(-1),
                        C_delta_x_xminus.reshape(-1),
                        F_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only -----
                    deltaF = M_k(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat

                    mse_F = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    # x-loss: same as in training
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    loss_em = 3 * mse_F + reg + x_loss
                    x_loss_sum_per_iter[em_iter] += x_loss.item()

                    # same alpha weighting as in training
                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]

                    total_loss += weight * loss_em
                    F_current = F_next

                    all_test_losses.append(loss_em.item())
                    all_f_losses.append(mse_F.item())

                    # store F estimates for the chosen sequence
                    if j % 5 == 0:
                        F_estimates.append(F_next.detach())
                        F_losses_mse.append(mse_F.item())
                        F_losses_total.append(loss_em.item())
                        x_losses_mse.append(x_loss.item())

                # store final F and final x_smooth for this sequence (after last EM iter)
                final_F_list.append(F_current.detach().clone())  # [m, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())  # [m, T]
                # final loss for this sequence (already weighted)
                loss_list[j] = total_loss / float(num_em_iters)
                # Mean x-loss for this sequence
                # if this is the chosen sequence, print F_true and all F_est
                if j % 5 == 0:
                    print(f"\n[M-step TEST] sequence {j} summary")
                    print("F_true:\n", F_true.detach())
                    print("F_init (F_base):\n", F_base.detach())

                    mse_F_init = torch.mean((F_base - F_true) ** 2).item()
                    print(f"Initial F MSE loss = {mse_F_init:.6e}")
                    for k, (F_est, f_mse, x_mse, total_val) in enumerate(
                            zip(F_estimates, F_losses_mse, x_losses_mse, F_losses_total)):
                        f_db = 10.0 * math.log10(f_mse)
                        x_db = 10.0 * math.log10(x_mse)
                        tot_db = 10.0 * math.log10(total_val)
                        print(f"\n  EM iter {k + 1}:")
                        print("  F_est:\n", F_est)
                        print(f"  F-loss (MSE_F)                 = {f_db:.2f} dB")
                        print(f"  x-loss (MSE_x)                 = {x_db:.2f} dB")
                        print(f"  total loss (F + reg + x)       = {tot_db:.2f} dB")

        mean_loss = loss_list.mean().item()
        print(f"[M-step TEST] mean_loss={mean_loss:.6f}")
        # average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)

        print("[M-step TEST] Mean x-MSE per EM iteration:")
        for k in range(num_em_iters):
            mse_k = mean_x_mse_per_iter[k].item()
            db_k = 10.0 * math.log10(mse_k)
            print(f"  EM iter {k + 1}: mean x-MSE = {mse_k:.6e}  ({db_k: .2f} dB)")

        # convert per-iteration mean x-MSE to numpy (linear and dB)
        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()

        return mean_x_mse_per_iter_np, mean_x_mse_per_iter_db_np, final_F_list, final_x_list


    def batch_one_train_m_step_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS,  lambda_F=1e-3, generate_f=True, non_linear_h=False):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model.train()

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        ######################################################
        # for name, param in model_mstep.named_parameters():
        #     if "weight" in name or "bias" in name:
        #         param.data.zero_()
        ##################################################

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            train_loss_sum = 0.0
            batch_x_before_sum = 0.0
            batch_x_after_sum = 0.0
            batch_f_loss_sum = 0.0
            self.M_optimizer.zero_grad()
            total_loss_batch = 0.0
            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select F_i and F_true by group
                if generate_f is True:
                    f_index = n_e // 10
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                # NEW: Select H for this training sequence
                if generate_h is True:
                    h_index = n_e // 10
                    H_current = SysModel.H_train[h_index]
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                # --------- EM unrolling over F ---------
                F_current = F_base  # this will be updated each EM iteration


                self.model.update_F(F_current)

                # E-step via frozen RTSNet → x_smooth
                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.init_hidden()
                self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                x_smooth[:, T - 1] = x_forward[:, T - 1]

                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                x_loss_before = torch.mean((x_smooth - x_true_seq) ** 2)


                # ---------------- Stats for M-network ----------------
                x_curr = x_smooth  # [m, T]
                x_prev = torch.empty_like(x_curr)  # [m, T]
                x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                A1 = (x_curr @ x_prev.T) / T
                # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                A2 = (x_prev @ x_prev.T) / T
                # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                x_minus = F_current @ x_prev  # [m, T_eff]
                # Δx_t = x_t - F*x_{t-1|T}
                delta_x = x_curr - x_minus  # [m, T]
                delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                delta_centered = delta_x - delta_mean
                # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                S_delta_x = (delta_centered @ delta_centered.T) / T
                if non_linear_h:
                    # y_hat_t = h(x_t) for each t
                    y_hat_list = []
                    for t in range(T):
                        # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                        x_t = x_curr[:, t].view(SysModel.m, 1)
                        y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                        y_hat_list.append(y_t_hat.view(-1))
                    Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                else:
                    H = SysModel.H
                    Hx_curr = H @ x_curr  # [n, T]
                nu = y_seq - Hx_curr
                # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                nu_centered = nu - nu_mean
                S_nu = (nu_centered @ nu_centered.T) / T
                # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                C_delta_x_xminus = (delta_x @ x_minus.T) / T

                z_in = torch.cat([A1.reshape(-1), A2.reshape(-1), S_delta_x.reshape(-1), S_nu.reshape(-1),
                                  C_delta_x_xminus.reshape(-1),
                                  F_current.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]

                # Predict ΔF and update F
                deltaF = model_mstep(z_in)
                deltaF_mat = deltaF.view(m, m)
                # print('for nir delta f =',deltaF_mat)
                F_next = F_current + deltaF_mat

                F_current =F_next

                self.model.update_F(F_current)

                # E-step via frozen RTSNet → x_smooth
                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.init_hidden()
                self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                x_smooth[:, T - 1] = x_forward[:, T - 1]

                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])



                # Loss: Frobenius(F_next - F_true)^2 + regularization
                f_loss = torch.mean((F_current - F_true) ** 2)
                reg = lambda_F * torch.mean(deltaF_mat ** 2)
                # x-loss AFTER M-step (with updated F)
                x_loss_after = torch.mean((x_smooth - x_true_seq) ** 2)

                # ---- compute loss ----
                loss = 5*f_loss + reg + x_loss_after

                # for optimization (keep graph)
                total_loss_batch += loss
                # ---- only for logging / printing ----

                batch_x_before_sum += x_loss_before.detach()
                batch_x_after_sum  += x_loss_after.detach()
                batch_f_loss_sum   += f_loss.detach()

            # ---- backward + step (one SGD step per sequence) ----

            mean_total_loss_batch = total_loss_batch / self.N_B
            mean_total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.M_optimizer.step()
            batch_loss = mean_total_loss_batch.detach()
            train_loss_sum += batch_loss.item()
            # averages over sequences in this batch
            mean_x_before = batch_x_before_sum / self.N_B
            mean_x_after  = batch_x_after_sum / self.N_B
            mean_f_loss   = batch_f_loss_sum / self.N_B

            eps = 1e-12
            batch_loss_db   = 10.0 * math.log10(batch_loss.item()   + eps)
            x_before_db     = 10.0 * math.log10(mean_x_before.item() + eps)
            x_after_db      = 10.0 * math.log10(mean_x_after.item()  + eps)
            f_loss_db       = 10.0 * math.log10(mean_f_loss.item()   + eps)
            print(
                f"[M-step][train] epoch={epoch:03d} "
                f"batch_loss={batch_loss.item():.6e} ({batch_loss_db:.2f} dB) "
                f"x_before={mean_x_before.item():.6e} ({x_before_db:.2f} dB) "
                f"x_after={mean_x_after.item():.6e} ({x_after_db:.2f} dB) "
                f"F_loss={mean_f_loss.item():.6e} ({f_loss_db:.2f} dB)"
            )

                # ---------------- Validation ----------------
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    # choose base F and F_true for this CV sequence
                    if generate_f is True:
                        f_index_cv = j // 10
                        F_base_cv = SysModel.F_valid[f_index_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()

                    # --- RTS smoother with current F_current_cv ---
                    self.model.update_F(F_current_cv)
                    self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                    x_f_cv = torch.empty(m, T_cv, device=device)
                    x_s_cv = torch.empty(m, T_cv, device=device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    # -------- stats, same as training --------
                    x_curr = x_s_cv  # [m, T_cv]
                    x_prev = torch.empty_like(x_curr)
                    x_prev[:, 0] = SysModel.m1x_0.view(-1)
                    x_prev[:, 1:] = x_curr[:, :-1]

                    A1_cv = (x_curr @ x_prev.T) / T_cv
                    A2_cv = (x_prev @ x_prev.T) / T_cv

                    x_minus_cv = F_current_cv @ x_prev
                    delta_x_cv = x_curr - x_minus_cv

                    delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                    delta_centered_cv = delta_x_cv - delta_mean_cv
                    S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv

                    if non_linear_h:
                        y_hat_cv_list = []
                        for t in range(T_cv):
                            x_t = x_curr[:, t].view(SysModel.m, 1)
                            y_t_hat = SysModel.h(x_t)  # non-linear h
                            y_hat_cv_list.append(y_t_hat.view(-1))
                        Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                    else:
                        H = SysModel.H.to(device)
                        Hx_curr_cv = H @ x_curr
                    nu_cv = y_cv - Hx_curr_cv

                    nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                    nu_centered_cv = nu_cv - nu_mean_cv
                    S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                    C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                    z_cv = torch.cat([
                        A1_cv.reshape(-1),
                        A2_cv.reshape(-1),
                        S_delta_x_cv.reshape(-1),
                        S_nu_cv.reshape(-1),
                        C_delta_x_xminus_cv.reshape(-1),
                        F_current_cv.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # --- M-step forward only (no grad) ---
                    dF_cv = model_mstep(z_cv)
                    dF_cv_mat = dF_cv.view(m, m)
                    F_next_cv = F_current_cv + dF_cv_mat

                    F_current_cv = F_next_cv

                    # --- RTS smoother with current F_current_cv ---
                    self.model.update_F(F_current_cv)
                    self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                    x_f_cv = torch.empty(m, T_cv, device=device)
                    x_s_cv = torch.empty(m, T_cv, device=device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    # same loss as train (but no backward)
                    f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                    reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                    x_loss_cv = torch.mean((x_s_cv - x_true_cv_seq) ** 2)

                    cv_loss_seq =5*f_loss_cv + reg_cv + x_loss_cv

                    cv_loss_sum += cv_loss_seq.item()


            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)


            train_epoch_db = 10.0 * math.log10(train_loss_sum )
            cv_epoch_db = 10.0 * math.log10(cv_epoch)
            best_cv_db = 10.0 * math.log10(self.MSE_cv_dB_opt)

            print(f"[M-step] epoch={epoch:03d} "
                f"train={train_loss_sum:.6e} ({train_epoch_db:.2f} dB) "
                f"cv={cv_epoch:.6e} ({cv_epoch_db:.2f} dB) "
                f"best_cv={self.MSE_cv_dB_opt:.6e} ({best_cv_db:.2f} dB)")

    def train_mstep_net_with_new_enters(self, SysModel, cv_input, cv_target, train_input, train_target,
                            destination_path_M, destination_path_RTS, num_em_iters=3,
                            alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, non_linear_h=False):
            """
            Single-function M-step training (no helpers, no .to(...)).
            - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
            - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
            - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
            """
                # Basic sizes
            self.N_E = len(train_input)
            self.N_CV = len(cv_input)
            m = SysModel.m

            # Load and freeze RTSNet (smoother only)
            self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
            for p in self.model.parameters():
                p.requires_grad_(False)

            # M-step model and optimizer
            model_mstep = self.M_model.train()


            self.MSE_cv_dB_opt = 1000
            self.MSE_cv_idx_opt = 0


            for epoch in range(self.N_steps):
                # ---------------- Training ----------------
                model_mstep.train()
                train_loss_sum = 0.0


                for _ in range(self.N_B):
                    self.M_optimizer.zero_grad()

                    # Pick one training sequence
                    n_e = random.randint(0, self.N_E - 1)
                    y_seq = train_input[n_e]   # [n, T]
                    x_true_seq = train_target[n_e]  # [m, T]
                    T = y_seq.size(-1)

                    # Select F_i and F_true by group
                    if generate_f is True:
                        f_index = n_e // 10
                        F_base = SysModel.F_train[f_index]
                        F_true = SysModel.F_train_TRUE[f_index]
                    else:
                        F_base = SysModel.F_train[n_e]
                        F_true = SysModel.F_train_TRUE[n_e]
                    # --------- EM unrolling over F ---------
                    F_current = F_base  # this will be updated each EM iteration
                    total_loss = 0.0

                    for em_iter in range(num_em_iters):

                        self.model.update_F(F_current)

                        # E-step via frozen RTSNet → x_smooth
                        self.model.InitSequence(SysModel.m1x_0, T)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                        x_forward = torch.empty(m, T, device=device)
                        x_smooth = torch.empty(m, T, device=device)

                        for t in range(T):
                            x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                        x_smooth[:, T - 1] = x_forward[:, T - 1]

                        self.model.InitBackward(x_smooth[:, T - 1])
                        x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                        # ---------------- Stats for M-network ----------------
                        x_curr = x_smooth  # [m, T]
                        x_prev = torch.empty_like(x_curr)  # [m, T]
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                        x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                        # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                        A1 = (x_curr @ x_prev.T)/T
                        # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                        A2 = (x_prev @ x_prev.T)/T
                        # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                        x_minus = F_current @ x_prev  # [m, T_eff]
                        # Δx_t = x_t - F*x_{t-1|T}
                        delta_x = x_curr - x_minus  # [m, T]
                        delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                        delta_centered = delta_x - delta_mean
                        # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                        S_delta_x = (delta_centered @ delta_centered.T) / T
                        if non_linear_h:
                            # y_hat_t = h(x_t) for each t
                            y_hat_list = []
                            for t in range(T):
                                # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                                y_hat_list.append(y_t_hat.view(-1))
                            Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                        else:
                            H = SysModel.H
                            Hx_curr = H @ x_curr  # [n, T]
                        nu = y_seq - Hx_curr
                        # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                        nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                        nu_centered = nu - nu_mean
                        S_nu = (nu_centered @ nu_centered.T) / T
                        # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                        C_delta_x_xminus = (delta_x @ x_minus.T) / T

                        epsI = 1e-4 * torch.eye(m, device=A2.device, dtype=A2.dtype)
                        A2_reg = A2 + epsI
                        F_em = A1 @ torch.linalg.inv(A2_reg)  # [m, m]
                        deltaF_em = F_em - F_current  # [m, m]

                        # ----- 2) sequence SNR: snr_y = E_x / (E_nu + eps) -----
                        # E_x = mean_t ||x_t||^2
                        E_x = (x_curr ** 2).sum(dim=0).mean()  # scalar

                        # E_nu = mean_t ||nu_t||^2
                        E_nu = (nu ** 2).sum(dim=0).mean()  # scalar

                        snr_y = E_x / (E_nu + 1e-6)  # scalar

                        # make it a 1D tensor so we can cat
                        snr_y_vec = snr_y.view(1)
                        z_in = torch.cat([A1.reshape(-1),A2.reshape(-1),S_delta_x.reshape(-1),S_nu.reshape(-1),C_delta_x_xminus.reshape(-1),
                            F_current.reshape(-1),snr_y_vec], dim=0).reshape(1, -1)  # [1, feature_dim]

                        # Predict ΔF and update F
                        deltaF = model_mstep(z_in)
                        deltaF_mat = deltaF.view(m, m)
                        F_next = F_current + deltaF_mat

                        # Loss: Frobenius(F_next - F_true)^2 + regularization
                        f_loss = torch.mean((F_next - F_true) ** 2)
                        reg = lambda_F * torch.mean(deltaF_mat ** 2)
                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat = H @ x_curr  # [n, T]
                        # y_loss = torch.mean((y_hat - y_seq) ** 2)
                        # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                        ##########################################################
                        if em_iter == num_em_iters - 1:
                            loss_em = 15*f_loss + reg + x_loss
                        else:
                            loss_em = f_loss + reg+ x_loss
    #############################################################################################################
                        # loss_em = 3 * f_loss + reg + x_loss
                        # Apply your specific weighting: 0.05, 0.1, 0.85
                        if em_iter == 0:
                            weight = alpha[0]  # First EM iteration
                        elif em_iter == 1:
                            weight = alpha[1]  # Second EM iteration
                        else:
                            weight = alpha[2]  # Third EM iteration (rest)
                        total_loss += weight*loss_em
                        F_current = F_next  # use updated F in next EM iteration


                    # after `for em_iter in range(num_em_iters):`
                    loss = total_loss / float(num_em_iters)   # average over EM iterations
                    loss_mult = loss
                    loss_mult.backward()
                    torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                    self.M_optimizer.step()

                    train_loss_sum += loss.detach().item()

                    # ---------------- Validation ----------------
                model_mstep.eval()
                cv_loss_sum = 0.0

                with torch.no_grad():
                    for j in range(self.N_CV):
                        y_cv = cv_input[j]  # [n, T_cv]
                        x_true_cv_seq = cv_target[j]  # [m, T_cv]
                        T_cv = y_cv.size(-1)

                        # choose base F and F_true for this CV sequence
                        if generate_f is True:
                            f_index_cv = j // 10
                            F_base_cv = SysModel.F_valid[f_index_cv]
                            F_true_cv = SysModel.F_valid_TRUE[f_index_cv]
                        else:
                            F_base_cv = SysModel.F_valid[j]
                            F_true_cv = SysModel.F_valid_TRUE[j]

                        F_current_cv = F_base_cv.clone()
                        total_loss_cv = 0.0

                        for em_iter in range(num_em_iters):

                            # --- RTS smoother with current F_current_cv ---
                            self.model.update_F(F_current_cv)
                            self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                            self.model.init_hidden()
                            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                            x_f_cv = torch.empty(m, T_cv, device=device)
                            x_s_cv = torch.empty(m, T_cv, device=device)

                            for t in range(T_cv):
                                x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                            x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                            self.model.InitBackward(x_s_cv[:, T_cv - 1])
                            x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                            for t in range(T_cv - 3, -1, -1):
                                x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                            # -------- stats, same as training --------
                            x_curr = x_s_cv  # [m, T_cv]
                            x_prev = torch.empty_like(x_curr)
                            x_prev[:, 0] = SysModel.m1x_0.view(-1)
                            x_prev[:, 1:] = x_curr[:, :-1]

                            A1_cv = (x_curr @ x_prev.T) / T_cv
                            A2_cv = (x_prev @ x_prev.T) / T_cv

                            x_minus_cv = F_current_cv @ x_prev
                            delta_x_cv = x_curr - x_minus_cv

                            delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                            delta_centered_cv = delta_x_cv - delta_mean_cv
                            S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv


                            if non_linear_h:
                                y_hat_cv_list = []
                                for t in range(T_cv):
                                    x_t = x_curr[:, t].view(SysModel.m, 1)
                                    y_t_hat = SysModel.h(x_t)  # non-linear h
                                    y_hat_cv_list.append(y_t_hat.view(-1))
                                Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                            else:
                                H = SysModel.H.to(device)
                                Hx_curr_cv = H @ x_curr
                            nu_cv = y_cv - Hx_curr_cv

                            nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                            nu_centered_cv = nu_cv - nu_mean_cv
                            S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                            C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv
                            epsI = 1e-4 * torch.eye(m, device=A2_cv.device, dtype=A2.dtype)
                            A2_reg_cv = A2_cv + epsI
                            F_em_cv = A1_cv @ torch.linalg.inv(A2_reg_cv)  # [m, m]
                            deltaF_em_cv = F_em_cv - F_current_cv  # [m, m]

                            # ----- 2) sequence SNR: snr_y = E_x / (E_nu + eps) -----
                            # E_x = mean_t ||x_t||^2
                            E_x_cv = (x_curr ** 2).sum(dim=0).mean()  # scalar

                            # E_nu = mean_t ||nu_t||^2
                            E_nu_cv = (nu_cv ** 2).sum(dim=0).mean()  # scalar

                            snr_y_cv = E_x_cv / (E_nu_cv + 1e-6)  # scalar
                            # make it a 1D tensor so we can cat
                            snr_y_vec_cv = snr_y_cv.view(1)

                            z_cv = torch.cat([
                                A1_cv.reshape(-1),
                                A2_cv.reshape(-1),
                                S_delta_x_cv.reshape(-1),
                                S_nu_cv.reshape(-1),
                                C_delta_x_xminus_cv.reshape(-1),
                                F_current_cv.reshape(-1),snr_y_vec_cv
                                ], dim=0).reshape(1, -1)

                            # --- M-step forward only (no grad) ---
                            dF_cv = model_mstep(z_cv)
                            dF_cv_mat = dF_cv.view(m, m)
                            F_next_cv = F_current_cv + dF_cv_mat

                            # same loss as train (but no backward)
                            f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                            reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                            x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                            ##########################################################
                            # # y-loss (measurement-space loss)
                            # H = SysModel.H.to(device)  # [n, m]
                            # y_hat_cv  = H @ x_curr  # [n, T]
                            # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                            # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                            ##########################################################
                            if em_iter == num_em_iters - 1:
                                loss_em_cv = 15*f_loss_cv + reg_cv + x_loss_cv
                            else:
                                loss_em_cv =  f_loss_cv + reg_cv + x_loss_cv
                            #########################################################################
                            # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                            if em_iter == 0:
                                weight = alpha[0]
                            elif em_iter == 1:
                                weight = alpha[1]
                            else:
                                weight =  alpha[2]  # if you really want the same scaling as in train

                            total_loss_cv += weight * loss_em_cv
                            F_current_cv = F_next_cv
                        cv_loss_seq = total_loss_cv / float(num_em_iters)
                        cv_loss_sum += cv_loss_seq.item()

                train_epoch = train_loss_sum / max(1, self.N_B)
                cv_epoch = cv_loss_sum / max(1, self.N_CV)

                if cv_epoch < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = cv_epoch
                    torch.save(model_mstep, destination_path_M)

                print(f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")




    def test_mstep_net_with_new_enteries(self, SysModel, test_input, test_target,
                       destination_path_RTS,destination_path_M, num_em_iters=3,
                       alpha=(0.0, 0.0, 1.0), lambda_F=1e-3, generate_f=True, init_x_list=None, init_P_list=None, non_linear_h=False):
        """
        Testing-only version for the M-step network.
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Use CURRENT M-step network (self.M_model) to predict ΔF.
        - Run num_em_iters EM iterations per test sequence.
        - No training, no optimizer step.
        - Returns per-sequence loss and mean loss.
        """


        N_T = len(test_input)
        m = SysModel.m

        all_test_losses = []
        all_f_losses = []


        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load M-step network from checkpoint (NO training)
        model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        final_F_list = []
        final_x_list = []
        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]    # [n, T]
                x_true_seq = test_target[j]  # [m, T]

                T = y_seq.size(-1)

                # Select F_base and F_true for this test sequence
                if generate_f is True:
                    f_index = j // 10
                    F_base = SysModel.F_test[f_index].to(device)
                    F_true = SysModel.F_test_TRUE[f_index].to(device)
                else:
                    # fallback: sequence-wise
                    F_base = SysModel.F_test[j].to(device)
                    F_true = SysModel.F_test_TRUE[j].to(device)

                # NEW: Select H for this test sequence
                if generate_f is True:
                    h_index = j // 10
                    H_current = SysModel.H_test[h_index].to(device)
                    SysModel.H = H_current
                    self.model.update_H(H_current)
                F_current = F_base.clone()
                total_loss = 0.0
                F_estimates =[]
                F_losses_mse = []
                F_losses_total = []
                x_losses_mse = []
                y_loss_tot = []

                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                for em_iter in range(num_em_iters):

                    # ----- RTS smoother with current F_current -----
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = P0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ----- stats for M-network -----
                    x_curr = x_smooth                    # [m, T]
                    x_prev = torch.empty_like(x_curr)    # [m, T]
                    x_prev[:, 0] = x0.view(-1)   # x_0
                    x_prev[:, 1:] = x_curr[:, :-1]                      # x_{t-1|T}

                    A1 = (x_curr @ x_prev.T) / T
                    A2 = (x_prev @ x_prev.T) / T

                    x_minus = F_current @ x_prev        # [m, T]
                    delta_x = x_curr - x_minus          # [m, T]

                    delta_mean = delta_x.mean(dim=1, keepdim=True)
                    delta_centered = delta_x - delta_mean
                    S_delta_x = (delta_centered @ delta_centered.T) / T


                    # ---------- linear vs non-linear h ----------
                    if non_linear_h:
                        # y_hat_t = h(x_t) for each t
                        y_hat_list = []
                        for t in range(T):
                            x_t = x_curr[:, t].view(SysModel.m, 1)  # [m,1]
                            y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                            y_hat_list.append(y_t_hat.view(-1))  # flatten to [n]
                        Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                    else:
                        H = SysModel.H.to(device)
                        Hx_curr = H @ x_curr  # [n, T]
                    nu = y_seq - Hx_curr

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_centered = nu - nu_mean
                    S_nu = (nu_centered @ nu_centered.T) / T

                    C_delta_x_xminus = (delta_x @ x_minus.T) / T

                    epsI = 1e-4 * torch.eye(m, device=A2.device, dtype=A2.dtype)
                    A2_reg = A2 + epsI
                    F_em = A1 @ torch.linalg.inv(A2_reg)  # [m, m]
                    deltaF_em = F_em - F_current  # [m, m]

                    # ----- 2) sequence SNR: snr_y = E_x / (E_nu + eps) -----
                    # E_x = mean_t ||x_t||^2
                    E_x = (x_curr ** 2).sum(dim=0).mean()  # scalar

                    # E_nu = mean_t ||nu_t||^2
                    E_nu = (nu ** 2).sum(dim=0).mean()  # scalar

                    snr_y = E_x / (E_nu + 1e-6)  # scalar
                    # make it a 1D tensor so we can cat
                    snr_y_vec = snr_y.view(1)

                    z_in = torch.cat([
                        A1.reshape(-1),
                        A2.reshape(-1),
                        S_delta_x.reshape(-1),
                        S_nu.reshape(-1),
                        C_delta_x_xminus.reshape(-1),
                        F_current.reshape(-1),snr_y_vec
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only -----
                    deltaF = model_mstep(z_in)
                    deltaF_mat = deltaF.view(m, m)
                    F_next = F_current + deltaF_mat
                   ####################################################################
                    # eigvals = torch.linalg.eigvals(F_next)
                    # rho = eigvals.abs().max()
                    #
                    # # rho is a scalar complex tensor -> take real part for safety
                    # rho_real = rho.real
                    # max_rho = 1.07
                    # if rho_real > max_rho:
                    #     scale = (max_rho / rho_real)
                    #     F_next = F_next * scale
                    ##########################################################
                    mse_F = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF_mat ** 2)
                    # x-loss: same as in training
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat  = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat  - y_seq) ** 2)
                    # y_loss_tot.append(y_loss.item())
                    ##########################################################
                    loss_em = 3*mse_F + reg + x_loss
                    x_loss_sum_per_iter[em_iter] += x_loss.item()

                    # same alpha weighting as in training
                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]

                    total_loss += weight * loss_em
                    F_current = F_next

                    all_test_losses.append(loss_em.item())
                    all_f_losses.append(mse_F.item())

                    # store F estimates for the chosen sequence
                    if j %5 ==0:
                        F_estimates.append(F_next.detach())
                        F_losses_mse.append(mse_F.item())
                        F_losses_total.append(loss_em.item())
                        x_losses_mse.append(x_loss.item())

                # store final F and final x_smooth for this sequence (after last EM iter)
                final_F_list.append(F_current.detach().clone())   # [m, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())      # [m, T]
                        # final loss for this sequence (already weighted)
                loss_list[j] = total_loss / float(num_em_iters)
                # Mean x-loss for this sequence
                # if this is the chosen sequence, print F_true and all F_est
                if j %5 ==0:
                    print(f"\n[M-step TEST] sequence {j} summary")
                    print("F_true:\n", F_true.detach())
                    print("F_init (F_base):\n", F_base.detach())

                    mse_F_init = torch.mean((F_base - F_true) ** 2).item()
                    print(f"Initial F MSE loss = {mse_F_init:.6e}")
                    for k, (F_est, f_mse, x_mse, total_val) in enumerate(zip(F_estimates, F_losses_mse, x_losses_mse, F_losses_total)):
                        f_db = 10.0 * math.log10(f_mse)
                        x_db = 10.0 * math.log10(x_mse)
                        tot_db = 10.0 * math.log10(total_val)
                        print(f"\n  EM iter {k + 1}:")
                        print("  F_est:\n", F_est)
                        print(f"  F-loss (MSE_F)                 = {f_db:.2f} dB")
                        print(f"  x-loss (MSE_x)                 = {x_db:.2f} dB")
                        print(f"  total loss (F + reg + x)       = {tot_db:.2f} dB")
                        # print(f"y_loss = {y_loss_tot[k]:2f}")


        mean_loss = loss_list.mean().item()
        print(f"[M-step TEST] mean_loss={mean_loss:.6f}")
        # average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)

        print("[M-step TEST] Mean x-MSE per EM iteration:")
        for k in range(num_em_iters):
            mse_k = mean_x_mse_per_iter[k].item()
            db_k = 10.0 * math.log10(mse_k )
            print(f"  EM iter {k + 1}: mean x-MSE = {mse_k:.6e}  ({db_k: .2f} dB)")

        # convert per-iteration mean x-MSE to numpy (linear and dB)
        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()


        return mean_x_mse_per_iter_np,mean_x_mse_per_iter_db_np, final_F_list,final_x_list



    def train_mstep_net_3_datasets(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS, num_em_iters=3,
                        alpha=(0.05, 0.1, 0.85), lambda_F=1e-3, generate_f=True, non_linear_h=False,load = None,datasets = 3):
        """
        Single-function M-step training (no helpers, no .to(...)).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build A1/A2 from x_smooth, zeros for the rest, feed M-net to predict ΔF.
        - Minimize Frobenius loss to ground-truth F (train/CV), save best M-model.
        """
            # Basic sizes - FIXED: train_input is list of datasets, need length of first dataset
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        if load is not None:
            self.M_model = torch.load(load, weights_only=False).to(self.device)
        model_mstep = self.M_model.train()
        self.M_optimizer = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)  # use your LR var

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0


        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            train_loss_sum = 0.0
            model_mstep.train()
            for _ in range(self.N_B):
                self.M_optimizer.zero_grad()
                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                F_base =  torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=device)
                x_0 = SysModel.m1x_0
                total_loss = 0.0
                for data in range(datasets):

                    y_seq = train_input[data][n_e]   # [n, T]
                    x_true_seq = train_target[data][n_e]  # [m, T]
                    T = y_seq.size(-1)

                    # Select F_i and F_true by group
                    if generate_f is True:
                        f_index = n_e // 10
                        F_true = SysModel.F_train_TRUE[data][f_index]
                    else:
                        F_true = SysModel.F_train_TRUE[data][n_e]
                    # --------- EM unrolling over F ---------
                    F_current = F_base  # this will be updated each EM iteration


                    for em_iter in range(num_em_iters):

                        self.model.update_F(F_current)

                        # E-step via frozen RTSNet → x_smooth
                        self.model.InitSequence(x_0, T)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                        x_forward = torch.empty(m, T, device=device)
                        x_smooth = torch.empty(m, T, device=device)

                        for t in range(T):
                            x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                        x_smooth[:, T - 1] = x_forward[:, T - 1]

                        self.model.InitBackward(x_smooth[:, T - 1])
                        x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                        # ---------------- Stats for M-network ----------------
                        x_curr = x_smooth  # [m, T]
                        x_prev = torch.empty_like(x_curr)  # [m, T]
                        x_prev[:, 0] = SysModel.m1x_0.view(-1)  # x_0
                        x_prev[:, 1:] = x_curr[:, :-1]  # x_{t-1|T}, t>=1
                        # Ā1 = 1/T Σ x_t x_{t-1|T}^T  (cross moment)
                        A1 = (x_curr @ x_prev.T)/T
                        # Ā2 = 1/T Σ x_{t-1|T} x_{t-1|T}^T  (auto moment)
                        A2 = (x_prev @ x_prev.T)/T
                        # Predicted previous state: x⁻_t = F_current x_{t-1|T}
                        x_minus = F_current @ x_prev  # [m, T_eff]
                        # Δx_t = x_t - F*x_{t-1|T}
                        delta_x = x_curr - x_minus  # [m, T]
                        delta_mean = delta_x.mean(dim=1, keepdim=True)  # \bar{Δx}
                        delta_centered = delta_x - delta_mean
                        # S_Δx = 1/T Σ (Δx_t - \bar{Δx})(Δx_t - \bar{Δx})^T
                        S_delta_x = (delta_centered @ delta_centered.T) / T
                        if non_linear_h:
                            # y_hat_t = h(x_t) for each t
                            y_hat_list = []
                            for t in range(T):
                                # shape [m] -> whatever h expects; .view(m,1) is safe with your linear h
                                x_t = x_curr[:, t].view(SysModel.m, 1)
                                y_t_hat = SysModel.h(x_t)  # [n,1] or [n]
                                y_hat_list.append(y_t_hat.view(-1))
                            Hx_curr = torch.stack(y_hat_list, dim=1)  # [n, T]
                        else:
                            H = SysModel.H
                            Hx_curr = H @ x_curr  # [n, T]
                        nu = y_seq - Hx_curr
                        # S_ν = 1/T Σ (ν_t − \bar{ν})(ν_t − \bar{ν})^T
                        nu_mean = nu.mean(dim=1, keepdim=True)  # [n,1]
                        nu_centered = nu - nu_mean
                        S_nu = (nu_centered @ nu_centered.T) / T
                        # C_{Δx,x⁻} = 1/T Σ Δx_t x⁻_t^T   (no centering in your formula)
                        C_delta_x_xminus = (delta_x @ x_minus.T) / T

                        z_in = torch.cat([A1.reshape(-1),A2.reshape(-1),S_delta_x.reshape(-1),S_nu.reshape(-1),C_delta_x_xminus.reshape(-1),
                            F_current.reshape(-1)], dim=0).reshape(1, -1)  # [1, feature_dim]

                        # Predict ΔF and update F
                        deltaF = model_mstep(z_in)
                        deltaF_mat = deltaF.view(m, m)
                        F_next = F_current + deltaF_mat

                        # Loss: Frobenius(F_next - F_true)^2 + regularization
                        f_loss = torch.mean((F_next - F_true) ** 2)
                        reg = lambda_F * torch.mean(deltaF_mat ** 2)
                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat = H @ x_curr  # [n, T]
                        # y_loss = torch.mean((y_hat - y_seq) ** 2)
                        # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                        ##########################################################
                        if em_iter == num_em_iters - 1:
                            loss_em = f_loss + reg + x_loss
                        else:
                            loss_em = f_loss + reg+ x_loss
    #############################################################################################################
                        # loss_em = 3 * f_loss + reg + x_loss
                        # Apply your specific weighting: 0.05, 0.1, 0.85
                        if em_iter == 0:
                            weight = alpha[0]  # First EM iteration
                        elif em_iter == 1:
                            weight = alpha[1]  # Second EM iteration
                        else:
                            weight = alpha[2]  # Third EM iteration (rest)
                        total_loss += weight*loss_em
                        F_current = F_next  # use updated F in next EM iteration

                    F_base = F_current.detach()  # detach to avoid backprop through datasets
                    x_0 = x_curr[:,-1].detach()  # use last smoothed x as x_0 for next dataset

                # right before backward






                # after `for em_iter in range(num_em_iters):`
                # FIXED: Divide by datasets first to keep as tensor, then normalize properly
                #w_before = next(model_mstep.parameters()).detach().clone()
                loss = total_loss / float(datasets * num_em_iters)   # average over datasets and EM iterations
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                self.M_optimizer.step()
                #w_after = next(model_mstep.parameters()).detach()
                #print("Δw:", (w_after - w_before).abs().max().item())


                train_loss_sum += loss.detach().item()

                # ---------------- Validation ----------------
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():

                for j in range(self.N_CV):
                    F_base_cv = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=device)
                    x_0_cv = SysModel.m1x_0
                    total_loss_cv = 0.0

                    for data in range(datasets):

                        y_cv = cv_input[data][j]  # [n, T_cv]
                        x_true_cv_seq = cv_target[data][j]  # [m, T_cv]
                        T_cv = y_cv.size(-1)

                        # choose base F and F_true for this CV sequence
                        if generate_f is True:
                            f_index_cv = j // 10
                            F_true_cv = SysModel.F_valid_TRUE[data][f_index_cv]
                        else:
                            F_true_cv = SysModel.F_valid_TRUE[data][j]

                        F_current_cv = F_base_cv.clone()

                        for em_iter in range(num_em_iters):

                            # --- RTS smoother with current F_current_cv ---
                            self.model.update_F(F_current_cv)
                            self.model.InitSequence(x_0_cv, T_cv)
                            self.model.init_hidden()
                            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                            x_f_cv = torch.empty(m, T_cv, device=device)
                            x_s_cv = torch.empty(m, T_cv, device=device)

                            for t in range(T_cv):
                                x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                            x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                            self.model.InitBackward(x_s_cv[:, T_cv - 1])
                            x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                            for t in range(T_cv - 3, -1, -1):
                                x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                            # -------- stats, same as training --------
                            x_curr = x_s_cv  # [m, T_cv]
                            x_prev = torch.empty_like(x_curr)
                            x_prev[:, 0] = SysModel.m1x_0.view(-1)
                            x_prev[:, 1:] = x_curr[:, :-1]

                            A1_cv = (x_curr @ x_prev.T) / T_cv
                            A2_cv = (x_prev @ x_prev.T) / T_cv

                            x_minus_cv = F_current_cv @ x_prev
                            delta_x_cv = x_curr - x_minus_cv

                            delta_mean_cv = delta_x_cv.mean(dim=1, keepdim=True)
                            delta_centered_cv = delta_x_cv - delta_mean_cv
                            S_delta_x_cv = (delta_centered_cv @ delta_centered_cv.T) / T_cv


                            if non_linear_h:
                                y_hat_cv_list = []
                                for t in range(T_cv):
                                    x_t = x_curr[:, t].view(SysModel.m, 1)
                                    y_t_hat = SysModel.h(x_t)  # non-linear h
                                    y_hat_cv_list.append(y_t_hat.view(-1))
                                Hx_curr_cv = torch.stack(y_hat_cv_list, dim=1)  # [n, T_cv]
                            else:
                                H = SysModel.H.to(device)
                                Hx_curr_cv = H @ x_curr
                            nu_cv = y_cv - Hx_curr_cv

                            nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                            nu_centered_cv = nu_cv - nu_mean_cv
                            S_nu_cv = (nu_centered_cv @ nu_centered_cv.T) / T_cv

                            C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                            z_cv = torch.cat([
                                A1_cv.reshape(-1),
                                A2_cv.reshape(-1),
                                S_delta_x_cv.reshape(-1),
                                S_nu_cv.reshape(-1),
                                C_delta_x_xminus_cv.reshape(-1),
                                F_current_cv.reshape(-1)
                            ], dim=0).reshape(1, -1)

                            # --- M-step forward only (no grad) ---
                            dF_cv = model_mstep(z_cv)
                            dF_cv_mat = dF_cv.view(m, m)
                            F_next_cv = F_current_cv + dF_cv_mat

                            # same loss as train (but no backward)
                            f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)  # MSE on F
                            reg_cv = lambda_F * torch.mean(dF_cv_mat ** 2)
                            x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                            ##########################################################
                            # # y-loss (measurement-space loss)
                            # H = SysModel.H.to(device)  # [n, m]
                            # y_hat_cv  = H @ x_curr  # [n, T]
                            # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                            # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                            ##########################################################
                            if em_iter == num_em_iters - 1:
                                loss_em_cv = f_loss_cv + reg_cv + x_loss_cv
                            else:
                                loss_em_cv =  f_loss_cv + reg_cv + x_loss_cv
                            #########################################################################
                            # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                            if em_iter == 0:
                                weight = alpha[0]
                            elif em_iter == 1:
                                weight = alpha[1]
                            else:
                                weight =  alpha[2]  # if you really want the same scaling as in train

                            total_loss_cv += weight * loss_em_cv
                            F_current_cv = F_next_cv
                        x_0_cv = x_curr[:, -1].detach()  # use last smoothed x as x_0 for next CV sequence
                        F_base_cv = F_current_cv.detach()
                    # FIXED: Properly average CV loss over EM iterations
                    cv_loss_seq = total_loss_cv / float(num_em_iters*datasets)
                    cv_loss_sum += cv_loss_seq.item()

            # FIXED: Average over batches and CV sequences (already accounted for datasets in cv_loss_seq)
            train_epoch = train_loss_sum / max(1, self.N_B)
            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)

            print(f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")

    def train_RTS_net_3_datasets(self, SysModel, cv_input, cv_target, train_input, train_target,destination_path_RTS
                        , load_path_RTS,H_init=None
                        , datasets=3):
        """
        Train RTSNet on 3 datasets.
        Each dataset contains sequences of length 30, with different H.
        RTSNet parameters are trained, while H is supplied per dataset/sequence.
        Loss is the state reconstruction MSE between smoothed states and true states.
        """
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])

        MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)

        MSE_train_linear_batch = torch.empty([self.N_B], device=self.device)
        self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)

        if load_path_RTS is not None:
            print("loading model_and keep training them")
            self.model = torch.load(load_path_RTS, map_location=self.device, weights_only=False).to(
                self.device)
            # Re-link the optimizer to the parameters of the newly loaded model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                              weight_decay=self.weightDecay)

        # Training Mode
        self.model.train()

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        nan_streak = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Zero gradients for both optimizers
            self.model.train()
            self.optimizer.zero_grad()

            Batch_Optimizing_LOSS_sum = 0
            batch_dataset_losses = [0.0] * datasets
            H_enter = H_init
            self.model.update_H(H_enter)
            for j in range(0, self.N_B):

                n_e = random.randint(0, self.N_E - 1)
                x_0 = SysModel.m1x_0.clone().detach().to(self.device)

                loss_3sets = 0.0
                for data in range(datasets):


                    y_training = train_input[data][n_e].to(self.device)
                    x_target = train_target[data][n_e].to(self.device)
                    SysModel.T = y_training.size(-1)

                    x_out_training_forward = torch.empty(
                        SysModel.m, SysModel.T, device=self.device, dtype=y_training.dtype
                    )
                    x_out_training = torch.empty(
                        SysModel.m, SysModel.T, device=self.device, dtype=y_training.dtype
                    )

                    self.model.InitSequence(x_0, SysModel.T)
                    self.model.init_hidden()

                    for t in range(SysModel.T):
                        x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)

                    x_out_training[:, SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
                    self.model.InitBackward(x_out_training[:, SysModel.T - 1])
                    x_out_training[:, SysModel.T - 2] = self.model(
                        None,
                        x_out_training_forward[:, SysModel.T - 2],
                        x_out_training_forward[:, SysModel.T - 1],
                        None
                    )

                    for t in range(SysModel.T - 3, -1, -1):
                        x_out_training[:, t] = self.model(
                            None,
                            x_out_training_forward[:, t],
                            x_out_training_forward[:, t + 1],
                            x_out_training[:, t + 2]
                        )

                    rtsnet_loss = self.loss_fn(x_out_training, x_target)
                    batch_dataset_losses[data] += rtsnet_loss.detach().item()
                    loss_3sets = loss_3sets + rtsnet_loss

                    # preserve x_0 for next dataset
                    x_0 = x_out_training[:, -1]

                loss_3sets = loss_3sets / datasets
                Batch_Optimizing_LOSS_sum += loss_3sets
                MSE_train_linear_batch[j] = loss_3sets.item()

            avg_dataset_losses = [x / self.N_B for x in batch_dataset_losses]
            # Average losses for this batch
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            print(f"[epoch {ti:03d}] "
                  f"loss_d0={avg_dataset_losses[0]:.6f} "
                  f"loss_d1={avg_dataset_losses[1]:.6f} "
                  f"loss_d2={avg_dataset_losses[2]:.6f} "
                  f"loss_all={Batch_Optimizing_LOSS_mean.item():.6f}")
            # Train RTSNet first
            Batch_Optimizing_LOSS_mean.backward()
            # 1) check every gradient tensor ori 2 blocks
            bad_grad = False
            for p in self.model.parameters():
                if p.grad is None:  # this param wasn’t used this pass
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad_grad = True
                    break

            if bad_grad:  # → skip this batch
                print("NaN/Inf gradients → batch skipped")
                nan_streak += 1
                if nan_streak >= 3:  # three bad batches in a row
                    print("Stopping training (3 consecutive bad batches).")
                continue  # start next epoch iteration

                # Calling the step function on an Optimizer makes an update to its
                # parameters
            nan_streak = 0

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # ori
            self.optimizer.step()

            # Average for logging
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)

                batch_cv_dataset_losses = [0.0] * datasets
                for j in range(0, self.N_CV):

                    x_0_cv = SysModel.m1x_0.clone().detach().to(self.device)
                    cv_loss_3sets = 0.0
                    for data in range(datasets):

                        y_cv = cv_input[data][j].to(self.device)
                        x_cv_target = cv_target[data][j].to(self.device)
                        SysModel.T_test = y_cv.size(-1)

                        x_out_cv_forward = torch.empty(
                            SysModel.m, SysModel.T_test, device=self.device, dtype=y_cv.dtype
                        )
                        x_out_cv = torch.empty(
                            SysModel.m, SysModel.T_test, device=self.device, dtype=y_cv.dtype
                        )

                        self.model.InitSequence(x_0_cv, SysModel.T_test)
                        self.model.init_hidden()

                        for t in range(SysModel.T_test):
                            x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_out_cv[:, SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
                        self.model.InitBackward(x_out_cv[:, SysModel.T_test - 1])
                        x_out_cv[:, SysModel.T_test - 2] = self.model(
                            None,
                            x_out_cv_forward[:, SysModel.T_test - 2],
                            x_out_cv_forward[:, SysModel.T_test - 1],
                            None
                        )

                        for t in range(SysModel.T_test - 3, -1, -1):
                            x_out_cv[:, t] = self.model(
                                None,
                                x_out_cv_forward[:, t],
                                x_out_cv_forward[:, t + 1],
                                x_out_cv[:, t + 2]
                            )

                        cv_loss_curr = self.loss_fn(x_out_cv, x_cv_target)
                        batch_cv_dataset_losses[data] += cv_loss_curr.item()
                        cv_loss_3sets += cv_loss_curr

                        # preserve x_0 for next dataset
                        x_0_cv = x_out_cv[:, -1].detach()

                    MSE_cv_linear_batch[j] = (cv_loss_3sets / datasets).item()
                avg_cv_dataset_losses = [x / self.N_CV for x in batch_cv_dataset_losses]
                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                print(f"[epoch {ti:03d}] "
                      f"cv_d0={avg_cv_dataset_losses[0]:.6f} "
                      f"cv_d1={avg_cv_dataset_losses[1]:.6f} "
                      f"cv_d2={avg_cv_dataset_losses[2]:.6f} "
                      f"cv_all={self.MSE_cv_linear_epoch[ti].item():.6f}")

                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.model, destination_path_RTS)

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti], "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch,
                self.MSE_train_dB_epoch]

    def train_H_mstep_net_3_datasets_with_rts(self, SysModel, cv_input, cv_target, train_input, train_target,
                                     destination_path_M, load_mnet, num_em_iters=3, H_init=None,
                                     alpha=(0.05, 0.1, 0.85), lambda_H=1e-3, generate_h=True, datasets=3):
        """
        M-step training for H (observation matrix) across 3 datasets.
        - F is FIXED (known dynamics) - same for all sequences and datasets
        - H is DIVERSE (unknown, changes across datasets)
        - Trains M-network to predict ΔH given smoothed states and statistics
        """
        # Basic sizes
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m
        n = SysModel.n

        # M-step model for H
        self.M_model_H = torch.load(load_mnet, weights_only=False).to(self.device)
        model_mstep = self.M_model_H.train()
        self.M_optimizer_H = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            model_mstep.train()
            self.M_optimizer_H.zero_grad()

            batch_total_loss = 0.0
            batch_x_loss_start = 0.0
            batch_x_loss_em = [0.0] * num_em_iters

            for _ in range(self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                # H_base = H_init.clone().to(self.device) if H_init is not None else SysModel.H.clone().detach().to(
                #     self.device)
                x_0 = SysModel.m1x_0.clone().detach().to(self.device)

                sample_total_loss = 0.0


                if H_init is None:
                    if generate_h:
                        h_index = n_e // 10
                        H_base = SysModel.H_train[0][h_index].clone().detach().to(self.device)
                    else:
                        H_base = SysModel.H_train[0][n_e].clone().detach().to(self.device)
                else:
                    H_base = H_init.clone().to(self.device)

                for data in range(datasets):
                    y_seq = train_input[data][n_e]
                    x_true_seq = train_target[data][n_e]
                    T = y_seq.size(-1)

                    # Get true H for this dataset
                    if generate_h is True:
                        h_index = n_e // 10
                        H_true = SysModel.H_train_TRUE[data][h_index]
                    else:
                        H_true = SysModel.H_train_TRUE[data][n_e]

                    H_current = H_base

                    [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, _] = S_Test_ext_H(
                        SysModel,
                        y_seq.unsqueeze(0),
                        x_true_seq.unsqueeze(0),
                        H_list=[H_current],
                        generate_h=False,
                        init_x_list=[x_0],
                        init_P_list=[SysModel.m2x_0]
                    )

                    x_curr = X_smooth.squeeze(0)  # [m, T]
                    P_curr = P_smooth.squeeze(0)  # [m, m, T]
                    y_curr = y_seq

                    x_loss_start = torch.mean((x_curr - x_true_seq) ** 2)
                    batch_x_loss_start += x_loss_start.detach().item()
                    for em_iter in range(num_em_iters):

                        # residuals
                        Hx = H_current @ x_curr
                        nu = y_curr - Hx
                        nu_mean = nu.mean(dim=1, keepdim=True)
                        nu_c = nu - nu_mean
                        x_mean = x_curr.mean(dim=1, keepdim=True)
                        x_c = x_curr - x_mean

                        S_nu = (nu_c @ nu_c.T) / T
                        C_nu_x = (nu_c @ x_c.T) / T

                        # EM-style sufficient statistics
                        C1 = (y_curr @ x_curr.T) / T
                        C2 = torch.zeros((m, m), device=x_curr.device, dtype=x_curr.dtype)

                        for t in range(T):
                            xt = x_curr[:, t].unsqueeze(1)
                            C2 += xt @ xt.T + P_curr[:, :, t]

                        C2 = C2 / T
                        eps = 1e-5 * torch.eye(m, device=x_curr.device, dtype=x_curr.dtype)
                        C2 = 0.5 * (C2 + C2.T) + eps

                        H_em = torch.linalg.solve(C2.T, C1.T).T
                        z_in = torch.cat([
                            H_current.detach().reshape(-1),
                            H_em.detach().reshape(-1),
                            S_nu.detach().reshape(-1),
                            C_nu_x.detach().reshape(-1)
                        ], dim=0).reshape(1, -1)

                        deltaH = model_mstep(z_in)
                        deltaH_mat = deltaH.view(n, m)

                        beta = 0.1
                        H_next = H_em + beta * deltaH_mat
                        H_current = H_next  # update for next EM iteration
                        [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, _] = S_Test_ext_H(
                            SysModel,
                            y_seq.unsqueeze(0),
                            x_true_seq.unsqueeze(0),
                            H_list=[H_current],
                            generate_h=False,
                            init_x_list=[x_0],
                            init_P_list=[SysModel.m2x_0]
                        )

                        x_curr = X_smooth.squeeze(0)
                        P_curr = P_smooth.squeeze(0)
                        y_curr = y_seq

                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                        batch_x_loss_em[em_iter] += x_loss.detach().item()

                        h_loss = torch.mean((H_next - H_true) ** 2)
                        reg = lambda_H * torch.mean(deltaH_mat ** 2)

                        loss_em = 0.5 * (h_loss + reg) + x_loss

                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]
                        sample_total_loss += weight * loss_em

                    H_base = H_current.detach()
                    x_0 = x_curr[:, -1].detach()
                sample_total_loss = sample_total_loss / float(datasets)
                batch_total_loss += sample_total_loss
            loss = batch_total_loss / self.N_B
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.M_optimizer_H.step()
            denom = self.N_B * datasets
            avg_x_loss_start = batch_x_loss_start / denom
            avg_x_loss_em = [x / denom for x in batch_x_loss_em]

            em_msg = " ".join([f"x_loss_em{k}={avg_x_loss_em[k]:.6f}" for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] x_loss_start={avg_x_loss_start:.6f} {em_msg} loss_all={loss.item():.6f}")
            # Validation
            model_mstep.eval()
            cv_loss_sum = 0.0
            batch_cv_x_loss_start = 0.0
            batch_cv_x_loss_em = [0.0] * num_em_iters

            with torch.no_grad():
                for j in range(self.N_CV):
                    x_0_cv = SysModel.m1x_0
                    sample_total_loss_cv = 0.0
                    if H_init is None:
                        if generate_h:
                            h_index = j// 10
                            H_base_cv = SysModel.H_valid[0][h_index].clone().detach().to(self.device)
                        else:
                            H_base_cv = SysModel.H_valid[0][j].clone().detach().to(self.device)
                    else:
                        H_base_cv = H_init
                    for data in range(datasets):
                        y_cv = cv_input[data][j]
                        x_true_cv_seq = cv_target[data][j]
                        T_cv = y_cv.size(-1)
                        if generate_h is True:
                            h_index_cv = j // 10
                            H_true_cv = SysModel.H_valid_TRUE[data][h_index_cv]
                        else:
                            H_true_cv = SysModel.H_valid_TRUE[data][j]

                        H_current_cv = H_base_cv.clone()

                        [_mse_arr_cv, _mse_avg_cv, _mse_db_cv, X_smooth_cv, P_smooth_cv, _] = S_Test_ext_H(
                            SysModel,
                            y_cv.unsqueeze(0),
                            x_true_cv_seq.unsqueeze(0),
                            H_list=[H_current_cv],
                            generate_h=False,
                            init_x_list=[x_0_cv],
                            init_P_list=[SysModel.m2x_0]
                        )

                        x_curr = X_smooth_cv.squeeze(0)
                        P_curr_cv = P_smooth_cv.squeeze(0)
                        y_curr = y_cv
                        x_loss_start_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        batch_cv_x_loss_start += x_loss_start_cv.item()
                        for em_iter in range(num_em_iters):
                            nu_cv = y_curr - (H_current_cv @ x_curr)
                            nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                            nu_c_cv = nu_cv - nu_mean_cv

                            x_mean_cv = x_curr.mean(dim=1, keepdim=True)
                            x_c_cv = x_curr - x_mean_cv

                            S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv
                            C_nu_x_cv = (nu_c_cv @ x_c_cv.T) / T_cv

                            C1_cv = (y_curr @ x_curr.T) / T_cv
                            C2_cv = torch.zeros((m, m), device=x_curr.device, dtype=x_curr.dtype)

                            for t in range(T_cv):
                                xt = x_curr[:, t].unsqueeze(1)
                                C2_cv += xt @ xt.T + P_curr_cv[:, :, t]

                            C2_cv = C2_cv / T_cv
                            eps_cv = 1e-5 * torch.eye(m, device=x_curr.device, dtype=x_curr.dtype)
                            C2_cv = 0.5 * (C2_cv + C2_cv.T) + eps_cv

                            H_em_cv = torch.linalg.solve(C2_cv.T, C1_cv.T).T

                            z_cv = torch.cat([
                                H_current_cv.reshape(-1),
                                H_em_cv.reshape(-1),
                                S_nu_cv.reshape(-1),
                                C_nu_x_cv.reshape(-1)
                            ], dim=0).reshape(1, -1)

                            dH_cv = model_mstep(z_cv)
                            dH_cv_mat = dH_cv.view(n, m)

                            beta = 0.1
                            H_next_cv = H_em_cv + beta * dH_cv_mat
                            H_current_cv = H_next_cv  # update for next EM iteration
                            h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                            reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                            [_mse_arr_cv, _mse_avg_cv, _mse_db_cv, X_smooth_cv, P_smooth_cv, _] = S_Test_ext_H(
                                SysModel,
                                y_cv.unsqueeze(0),
                                x_true_cv_seq.unsqueeze(0),
                                H_list=[H_current_cv],
                                generate_h=False,
                                init_x_list=[x_0_cv],
                                init_P_list=[SysModel.m2x_0]
                            )

                            x_curr = X_smooth_cv.squeeze(0)
                            P_curr_cv = P_smooth_cv.squeeze(0)
                            y_curr = y_cv
                            x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                            batch_cv_x_loss_em[em_iter] += x_loss_cv.item()

                            loss_em_cv = 0.5 * (h_loss_cv + reg_cv) + x_loss_cv

                            if em_iter == 0:
                                weight = alpha[0]
                            elif em_iter == 1:
                                weight = alpha[1]
                            else:
                                weight = alpha[2]

                            sample_total_loss_cv += weight * loss_em_cv

                        x_0_cv = x_curr[:, -1].detach()
                        H_base_cv = H_current_cv.detach()

                    sample_total_loss_cv = sample_total_loss_cv / float(datasets)
                    cv_loss_sum += sample_total_loss_cv.item()

            cv_epoch = cv_loss_sum / max(1, self.N_CV)
            cv_denom = self.N_CV * datasets
            avg_cv_x_loss_start = batch_cv_x_loss_start / cv_denom
            avg_cv_x_loss_em = [x / cv_denom for x in batch_cv_x_loss_em]

            cv_em_msg = " ".join([f"cv_x_loss_em{k}={avg_cv_x_loss_em[k]:.6f}" for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] cv_x_loss_start={avg_cv_x_loss_start:.6f} {cv_em_msg} cv_all={cv_epoch:.6f}")
            print(f"BEST: epoch={self.MSE_cv_idx_opt}  best_cv_loss={self.MSE_cv_dB_opt:.6f}")
            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)


    def train_H_mstep_net_3_datasets(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, load_path_RTS,load_mnet, num_em_iters=3,H_init = None,
                        alpha=(0.05, 0.1, 0.85), lambda_H=1e-3, generate_h=True, datasets=3):
        """
        M-step training for H (observation matrix) across 3 datasets.
        - F is FIXED (known dynamics) - same for all sequences and datasets
        - H is DIVERSE (unknown, changes across datasets)
        - Trains M-network to predict ΔH given smoothed states and statistics
        """
        # Basic sizes
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m
        n = SysModel.n

        # Load and freeze RTSNet
        self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model for H
        self.M_model_H = torch.load(load_mnet, weights_only=False).to(self.device)
        model_mstep = self.M_model_H.train()
        self.M_optimizer_H = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            train_loss_sum = 0.0
            model_mstep.train()
            self.M_optimizer_H.zero_grad()

            batch_total_loss = 0.0
            batch_x_loss_start = 0.0
            batch_x_loss_em = [0.0] * num_em_iters

            for _ in range(self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                H_base = H_init.clone().to(self.device) if H_init is not None else SysModel.H.clone().detach().to(
                    self.device)
                x_0 = SysModel.m1x_0.clone().detach().to(self.device)
                sample_total_loss = 0.0

                for data in range(datasets):
                    y_seq = train_input[data][n_e]
                    x_true_seq = train_target[data][n_e]
                    T = y_seq.size(-1)

                    # Get true H for this dataset
                    if generate_h is True:
                        h_index = n_e // 10
                        H_true = SysModel.H_train_TRUE[data][h_index]
                    else:
                        H_true = SysModel.H_train_TRUE[data][n_e]

                    H_current = H_base

                    self.model.update_H(H_current)
                    self.model.InitSequence(x_0, T)
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                    self.model.init_hidden()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # Stats for H M-network
                    x_curr = x_smooth
                    y_curr = y_seq
                    x_loss_start = torch.mean((x_curr - x_true_seq) ** 2)
                    batch_x_loss_start += x_loss_start.detach().item()
                    for em_iter in range(num_em_iters):

                        A_yx = (y_curr @ x_curr.T) / T
                        A_xx = (x_curr @ x_curr.T) / T

                        Hx = H_current @ x_curr
                        nu = y_curr - Hx

                        nu_mean = nu.mean(dim=1, keepdim=True)
                        nu_c = nu - nu_mean
                        S_nu = (nu_c @ nu_c.T) / T

                        C_nu_x = (nu @ x_curr.T) / T

                        z_in = torch.cat([
                            A_yx.reshape(-1).detach(),
                            A_xx.reshape(-1).detach(),
                            S_nu.reshape(-1).detach(),
                            C_nu_x.reshape(-1).detach(),
                            H_current.reshape(-1).detach()
                        ], dim=0).reshape(1, -1).detach()

                        deltaH = model_mstep(z_in)
                        deltaH_mat = deltaH.view(n, m)
                        H_next = H_current + deltaH_mat
                        H_current = H_next  # update for next EM iteration
                        h_loss = torch.mean((H_next - H_true) ** 2)
                        reg = lambda_H * torch.mean(deltaH_mat ** 2)
                        self.model.update_H(H_current)
                        self.model.InitSequence(x_0, T)
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                        self.model.init_hidden()


                        x_forward = torch.empty(m, T, device=device)
                        x_smooth = torch.empty(m, T, device=device)

                        for t in range(T):
                            x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                        x_smooth[:, T - 1] = x_forward[:, T - 1]

                        self.model.InitBackward(x_smooth[:, T - 1])
                        x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                        # Stats for H M-network
                        x_curr = x_smooth
                        y_curr = y_seq

                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                        batch_x_loss_em[em_iter] += x_loss.detach().item()
                        loss_em =5* (h_loss + reg) + x_loss

                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]
                        sample_total_loss += weight * loss_em


                    H_base = H_current.detach()
                    x_0 = x_curr[:, -1].detach()
                sample_total_loss = sample_total_loss / float(datasets)
                batch_total_loss += sample_total_loss
            loss = batch_total_loss / self.N_B
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.M_optimizer_H.step()
            denom = self.N_B * datasets
            avg_x_loss_start = batch_x_loss_start / denom
            avg_x_loss_em = [x / denom for x in batch_x_loss_em]

            em_msg = " ".join([f"x_loss_em{k}={avg_x_loss_em[k]:.6f}" for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] x_loss_start={avg_x_loss_start:.6f} {em_msg} loss_all={loss.item():.6f}")
            # Validation
            model_mstep.eval()
            cv_loss_sum = 0.0
            batch_cv_x_loss_start = 0.0
            batch_cv_x_loss_em = [0.0] * num_em_iters

            with torch.no_grad():
                for j in range(self.N_CV):
                    H_base_cv = H_init
                    x_0_cv = SysModel.m1x_0
                    sample_total_loss_cv = 0.0

                    for data in range(datasets):
                        y_cv = cv_input[data][j]
                        x_true_cv_seq = cv_target[data][j]
                        T_cv = y_cv.size(-1)
                        if generate_h is True:
                            h_index_cv = j // 10
                            H_true_cv = SysModel.H_valid_TRUE[data][h_index_cv]
                        else:
                            H_true_cv = SysModel.H_valid_TRUE[data][j]

                        H_current_cv = H_base_cv.clone()

                        self.model.update_H(H_current_cv)
                        self.model.InitSequence(x_0_cv, T_cv)
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                        self.model.init_hidden()

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        x_curr = x_s_cv
                        y_curr = y_cv
                        x_loss_start_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        batch_cv_x_loss_start += x_loss_start_cv.item()
                        for em_iter in range(num_em_iters):

                            A_yx_cv = (y_curr @ x_curr.T) / T_cv
                            A_xx_cv = (x_curr @ x_curr.T) / T_cv

                            nu_cv = y_curr - (H_current_cv @ x_curr)

                            nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                            nu_c_cv = nu_cv - nu_mean_cv
                            S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv

                            C_nu_x_cv = (nu_cv @ x_curr.T) / T_cv

                            z_cv = torch.cat([
                                A_yx_cv.reshape(-1),
                                A_xx_cv.reshape(-1),
                                S_nu_cv.reshape(-1),
                                C_nu_x_cv.reshape(-1),
                                H_current_cv.reshape(-1)
                            ], dim=0).reshape(1, -1)

                            dH_cv = model_mstep(z_cv)
                            dH_cv_mat = dH_cv.view(n, m)
                            H_next_cv = H_current_cv + dH_cv_mat
                            H_current_cv = H_next_cv  # update for next EM iteration
                            h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                            reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                            self.model.update_H(H_current_cv)
                            self.model.InitSequence(x_0_cv, T_cv)
                            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                            self.model.init_hidden()

                            x_f_cv = torch.empty(m, T_cv, device=device)
                            x_s_cv = torch.empty(m, T_cv, device=device)

                            for t in range(T_cv):
                                x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                            x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                            self.model.InitBackward(x_s_cv[:, T_cv - 1])
                            x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                            for t in range(T_cv - 3, -1, -1):
                                x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                            x_curr = x_s_cv
                            y_curr = y_cv
                            x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                            batch_cv_x_loss_em[em_iter] += x_loss_cv.item()

                            loss_em_cv =  5*(h_loss_cv + reg_cv) + x_loss_cv

                            if em_iter == 0:
                                weight = alpha[0]
                            elif em_iter == 1:
                                weight = alpha[1]
                            else:
                                weight = alpha[2]

                            sample_total_loss_cv += weight * loss_em_cv

                        x_0_cv = x_curr[:, -1].detach()
                        H_base_cv = H_current_cv.detach()


                    sample_total_loss_cv = sample_total_loss_cv / float(datasets )
                    cv_loss_sum += sample_total_loss_cv.item()


            cv_epoch = cv_loss_sum / max(1, self.N_CV)
            cv_denom = self.N_CV * datasets
            avg_cv_x_loss_start = batch_cv_x_loss_start / cv_denom
            avg_cv_x_loss_em = [x / cv_denom for x in batch_cv_x_loss_em]

            cv_em_msg = " ".join([f"cv_x_loss_em{k}={avg_cv_x_loss_em[k]:.6f}" for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] cv_x_loss_start={avg_cv_x_loss_start:.6f} {cv_em_msg} cv_all={cv_epoch:.6f}")
            print(f"BEST: epoch={self.MSE_cv_idx_opt}  best_cv_loss={self.MSE_cv_dB_opt:.6f}")
            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)

    def train_H_mstep_net_3_datasets_joint(self, SysModel, cv_input, cv_target, train_input, train_target,
                                     destination_path_M, destination_path_RTS,load_path_RTS, load_mnet, num_em_iters=3, H_init=None,
                                     alpha=(0.05, 0.1, 0.85), lambda_H=1e-3, generate_h=True, datasets=3,x_0_train_list = None,x_0_cv_list =None):
        """
        M-step training for H (observation matrix) across 3 datasets.
        - F is FIXED (known dynamics) - same for all sequences and datasets
        - H is DIVERSE (unknown, changes across datasets)
        - Trains M-network to predict ΔH given smoothed states and statistics
        """
        # Basic sizes
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m
        n = SysModel.n

        # Load and freeze RTSNet
        self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        # M-step model for H
        self.M_model_H = torch.load(load_mnet, weights_only=False).to(self.device)
        model_mstep = self.M_model_H.train()
        self.optimizer_joint = torch.optim.Adam(
            list(self.model.parameters()) + list(model_mstep.parameters()),
            lr=self.learningRate
        )

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            model_mstep.train()
            self.model.train()
            self.optimizer_joint.zero_grad()

            batch_total_loss = 0.0
            batch_x_loss_start = 0.0
            batch_x_loss_em = [0.0] * num_em_iters
            batch_h_loss_em = [0.0] * num_em_iters
            batch_reg_em = [0.0] * num_em_iters

            for _ in range(self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                H_base = H_init.clone().to(self.device) if H_init is not None else SysModel.H.clone().detach().to(
                    self.device)
                if x_0_train_list != None:
                    SysModel.m1x_0 = x_0_train_list[n_e]

                x_0 = SysModel.m1x_0.clone().detach().to(self.device)
                sample_total_loss = 0.0

                if H_init is None:
                    if generate_h:
                        h_index = n_e // 10
                        H_current = SysModel.H_train[0][h_index].clone().detach().to(self.device)
                    else:
                        H_current = SysModel.H_train[0][n_e].clone().detach().to(self.device)
                else:
                    H_current = H_base

                for data in range(datasets):
                    y_seq = train_input[data][n_e]
                    x_true_seq = train_target[data][n_e]
                    T = y_seq.size(-1)

                    # Get true H for this dataset
                    if generate_h is True:
                        h_index = n_e // 10
                        H_true = SysModel.H_train_TRUE[data][h_index]
                    else:
                        H_true = SysModel.H_train_TRUE[data][n_e]



                    self.model.update_H(H_current)
                    self.model.InitSequence(x_0, T)
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                    self.model.init_hidden()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # Stats for H M-network
                    x_curr = x_smooth
                    y_curr = y_seq
                    x_loss_start = torch.mean((x_curr - x_true_seq) ** 2)
                    batch_x_loss_start += x_loss_start.detach().item()
                    for em_iter in range(num_em_iters):

                        A_yx = (y_curr @ x_curr.T) / T
                        A_xx = (x_curr @ x_curr.T) / T

                        Hx = H_current @ x_curr
                        nu = y_curr - Hx

                        nu_mean = nu.mean(dim=1, keepdim=True)
                        nu_c = nu - nu_mean
                        S_nu = (nu_c @ nu_c.T) / T

                        C_nu_x = (nu @ x_curr.T) / T

                        z_in = torch.cat([
                            A_yx.reshape(-1).detach(),
                            A_xx.reshape(-1).detach(),
                            S_nu.reshape(-1).detach(),
                            C_nu_x.reshape(-1).detach(),
                            H_current.reshape(-1).detach()
                        ], dim=0).reshape(1, -1)

                        deltaH = model_mstep(z_in)
                        deltaH_mat = deltaH.view(n, m)
                        H_next = H_current + deltaH_mat
                        H_current = H_next  # update for next EM iteration
                        h_loss = torch.mean((H_next - H_true) ** 2)
                        reg = lambda_H * torch.mean(deltaH_mat ** 2)
                        batch_h_loss_em[em_iter] += h_loss.detach().item()
                        batch_reg_em[em_iter] += reg.detach().item()
                        self.model.update_H(H_current)
                        self.model.InitSequence(x_0, T)
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                        self.model.init_hidden()

                        x_forward = torch.empty(m, T, device=device)
                        x_smooth = torch.empty(m, T, device=device)

                        for t in range(T):
                            x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                        x_smooth[:, T - 1] = x_forward[:, T - 1]

                        self.model.InitBackward(x_smooth[:, T - 1])
                        x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                        # Stats for H M-network
                        x_curr = x_smooth
                        y_curr = y_seq

                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                        batch_x_loss_em[em_iter] += x_loss.detach().item()
                        loss_em = 5*h_loss + reg*0.5 + x_loss

                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]
                        sample_total_loss += weight * loss_em

                    H_base = H_current.detach()
                    x_0 = x_curr[:, -1].detach()
                sample_total_loss = sample_total_loss / float(datasets)
                batch_total_loss += sample_total_loss
            loss = batch_total_loss / self.N_B
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(model_mstep.parameters()),
                max_norm=1.5
            )
            self.optimizer_joint.step()
            denom = self.N_B * datasets
            avg_x_loss_start = batch_x_loss_start / denom
            avg_x_loss_em = [x / denom for x in batch_x_loss_em]
            avg_h_loss_em = [x / denom for x in batch_h_loss_em]
            avg_reg_em = [x / denom for x in batch_reg_em]
            em_msg = " ".join([
                f"x{k}={avg_x_loss_em[k]:.4f} "
                f"h{k}={avg_h_loss_em[k]:.4f} "
                f"reg{k}={avg_reg_em[k]:.4f}"
                for k in range(num_em_iters)
            ])
            print(f"[epoch {epoch:03d}] x_loss_start={avg_x_loss_start:.6f} {em_msg} loss_all={loss.item():.6f}")
            # Validation
            model_mstep.eval()
            self.model.eval()
            cv_loss_sum = 0.0
            batch_cv_x_loss_start = 0.0
            batch_cv_x_loss_em = [0.0] * num_em_iters
            batch_cv_h_loss_em = [0.0] * num_em_iters
            batch_cv_reg_em = [0.0] * num_em_iters
            with torch.no_grad():
                for j in range(self.N_CV):
                    H_base_cv = H_init
                    sample_total_loss_cv = 0.0

                    if H_init is None:
                        if generate_h:
                            h_index = j // 10
                            H_base_cv = SysModel.H_valid[0][h_index].clone().detach().to(self.device)
                        else:
                            H_base_cv= SysModel.H_valid[0][j].clone().detach().to(self.device)
                    if x_0_cv_list != None:
                        SysModel.m1x_0 = x_0_cv_list[j]
                    x_0_cv = SysModel.m1x_0.clone()
                    for data in range(datasets):
                        y_cv = cv_input[data][j]
                        x_true_cv_seq = cv_target[data][j]
                        T_cv = y_cv.size(-1)
                        if generate_h is True:
                            h_index_cv = j // 10
                            H_true_cv = SysModel.H_valid_TRUE[data][h_index_cv]
                        else:
                            H_true_cv = SysModel.H_valid_TRUE[data][j]

                        H_current_cv = H_base_cv.clone()

                        self.model.update_H(H_current_cv)
                        self.model.InitSequence(x_0_cv, T_cv)
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                        self.model.init_hidden()

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        x_curr = x_s_cv
                        y_curr = y_cv
                        x_loss_start_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        batch_cv_x_loss_start += x_loss_start_cv.item()
                        for em_iter in range(num_em_iters):

                            A_yx_cv = (y_curr @ x_curr.T) / T_cv
                            A_xx_cv = (x_curr @ x_curr.T) / T_cv

                            nu_cv = y_curr - (H_current_cv @ x_curr)

                            nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                            nu_c_cv = nu_cv - nu_mean_cv
                            S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv

                            C_nu_x_cv = (nu_cv @ x_curr.T) / T_cv

                            z_cv = torch.cat([
                                A_yx_cv.reshape(-1),
                                A_xx_cv.reshape(-1),
                                S_nu_cv.reshape(-1),
                                C_nu_x_cv.reshape(-1),
                                H_current_cv.reshape(-1)
                            ], dim=0).reshape(1, -1)

                            dH_cv = model_mstep(z_cv)
                            dH_cv_mat = dH_cv.view(n, m)
                            H_next_cv = H_current_cv + dH_cv_mat
                            H_current_cv = H_next_cv  # update for next EM iteration
                            h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                            reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                            batch_cv_h_loss_em[em_iter] += h_loss_cv.item()
                            batch_cv_reg_em[em_iter] += reg_cv.item()
                            self.model.update_H(H_current_cv)
                            self.model.InitSequence(x_0_cv, T_cv)
                            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                            self.model.init_hidden()

                            x_f_cv = torch.empty(m, T_cv, device=device)
                            x_s_cv = torch.empty(m, T_cv, device=device)

                            for t in range(T_cv):
                                x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                            x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                            self.model.InitBackward(x_s_cv[:, T_cv - 1])
                            x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                            for t in range(T_cv - 3, -1, -1):
                                x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                            x_curr = x_s_cv
                            y_curr = y_cv
                            x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                            batch_cv_x_loss_em[em_iter] += x_loss_cv.item()

                            loss_em_cv = 5*h_loss_cv + reg_cv*0.5 + x_loss_cv

                            if em_iter == 0:
                                weight = alpha[0]
                            elif em_iter == 1:
                                weight = alpha[1]
                            else:
                                weight = alpha[2]

                            sample_total_loss_cv += weight * loss_em_cv

                        x_0_cv = x_curr[:, -1].detach()
                        H_base_cv = H_current_cv.detach()

                    sample_total_loss_cv = sample_total_loss_cv / float(datasets)
                    cv_loss_sum += sample_total_loss_cv.item()

            cv_epoch = cv_loss_sum / max(1, self.N_CV)
            cv_denom = self.N_CV * datasets
            avg_cv_x_loss_start = batch_cv_x_loss_start / cv_denom
            avg_cv_x_loss_em = [x / cv_denom for x in batch_cv_x_loss_em]
            avg_cv_h_loss_em = [x / cv_denom for x in batch_cv_h_loss_em]
            avg_cv_reg_em = [x / cv_denom for x in batch_cv_reg_em]

            cv_em_msg = " ".join([
                f"cv_x{k}={avg_cv_x_loss_em[k]:.6f} "
                f"cv_h{k}={avg_cv_h_loss_em[k]:.6f} "
                f"cv_reg{k}={avg_cv_reg_em[k]:.6f}"
                for k in range(num_em_iters)
            ])
            print(f"[epoch {epoch:03d}] cv_x_loss_start={avg_cv_x_loss_start:.6f} {cv_em_msg} cv_all={cv_epoch:.6f}")
            print(f"BEST: epoch={self.MSE_cv_idx_opt}  best_cv_loss={self.MSE_cv_dB_opt:.6f}")
            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)
                torch.save(self.model, destination_path_RTS)

    def train_H_mstep_net_3_datasets_old(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS, num_em_iters=3,
                        alpha=(0.05, 0.1, 0.85), lambda_H=1e-3, generate_h=True, non_linear_f=False, load=None, datasets=3):
        """
        M-step training for H (observation matrix) across 3 datasets.
        - F is FIXED (known dynamics) - same for all sequences and datasets
        - H is DIVERSE (unknown, changes across datasets)
        - Trains M-network to predict ΔH given smoothed states and statistics
        """
        # Basic sizes
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m
        n = SysModel.n

        # Load and freeze RTSNet
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model for H
        if load is not None:
            self.M_model_H = torch.load(load, weights_only=False).to(self.device)
        model_mstep = self.M_model_H.train()
        self.M_optimizer_H = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            train_loss_sum = 0.0
            model_mstep.train()

            for _ in range(self.N_B):
                self.M_optimizer_H.zero_grad()
                n_e = random.randint(0, self.N_E - 1)

                # Initial "wrong" H (hardcoded baseline)
                H_base = torch.tensor([[1.0, 1.0], [0.25, 1.0]], device=device, dtype=torch.float32)
                x_0 = SysModel.m1x_0
                total_loss = 0.0

                for data in range(datasets):
                    y_seq = train_input[data][n_e]
                    x_true_seq = train_target[data][n_e]
                    T = y_seq.size(-1)

                    # Get true H for this dataset
                    if generate_h is True:
                        h_index = n_e // 10
                        H_true = SysModel.H_train_TRUE[data][h_index]
                    else:
                        H_true = SysModel.H_train_TRUE[data][n_e]

                    H_current = H_base

                    for em_iter in range(num_em_iters):
                        self.model.update_H(H_current)
                        self.model.InitSequence(x_0, T)
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                        self.model.init_hidden()


                        x_forward = torch.empty(m, T, device=device)
                        x_smooth = torch.empty(m, T, device=device)

                        for t in range(T):
                            x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                        x_smooth[:, T - 1] = x_forward[:, T - 1]

                        self.model.InitBackward(x_smooth[:, T - 1])
                        x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                        # Stats for H M-network
                        x_curr = x_smooth
                        y_curr = y_seq

                        A_yx = (y_curr @ x_curr.T) / T
                        A_xx = (x_curr @ x_curr.T) / T

                        Hx = H_current @ x_curr
                        nu = y_curr - Hx

                        nu_mean = nu.mean(dim=1, keepdim=True)
                        nu_c = nu - nu_mean
                        S_nu = (nu_c @ nu_c.T) / T

                        C_nu_x = (nu @ x_curr.T) / T

                        z_in = torch.cat([
                            A_yx.reshape(-1),
                            A_xx.reshape(-1),
                            S_nu.reshape(-1),
                            C_nu_x.reshape(-1),
                            H_current.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        deltaH = model_mstep(z_in)
                        deltaH_mat = deltaH.view(n, m)
                        H_next = H_current + deltaH_mat

                        h_loss = torch.mean((H_next - H_true) ** 2)
                        reg = lambda_H * torch.mean(deltaH_mat ** 2)
                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)

                        if em_iter == num_em_iters - 1:
                            loss_em = h_loss + reg + x_loss
                        else:
                            loss_em = h_loss + reg + x_loss

                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]

                        total_loss += weight * loss_em
                        H_current = H_next

                    H_base = H_current.detach()
                    x_0 = x_curr[:, -1].detach()

                loss = total_loss / float(datasets * num_em_iters)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                self.M_optimizer_H.step()

                train_loss_sum += loss.detach().item()

            # Validation
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    H_base_cv = torch.tensor([[1.0, 1.0], [0.25, 1.0]], device=device, dtype=torch.float32)
                    x_0_cv = SysModel.m1x_0
                    total_loss_cv = 0.0

                    for data in range(datasets):
                        y_cv = cv_input[data][j]
                        x_true_cv_seq = cv_target[data][j]
                        T_cv = y_cv.size(-1)

                        if generate_h is True:
                            h_index_cv = j // 10
                            H_true_cv = SysModel.H_valid_TRUE[data][h_index_cv]
                        else:
                            H_true_cv = SysModel.H_valid_TRUE[data][j]

                        H_current_cv = H_base_cv.clone()

                        for em_iter in range(num_em_iters):
                            self.model.update_H(H_current_cv)
                            self.model.InitSequence(x_0_cv, T_cv)
                            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                            self.model.init_hidden()


                            x_f_cv = torch.empty(m, T_cv, device=device)
                            x_s_cv = torch.empty(m, T_cv, device=device)

                            for t in range(T_cv):
                                x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                            x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                            self.model.InitBackward(x_s_cv[:, T_cv - 1])
                            x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                            for t in range(T_cv - 3, -1, -1):
                                x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                            x_curr = x_s_cv
                            y_curr = y_cv

                            A_yx_cv = (y_curr @ x_curr.T) / T_cv
                            A_xx_cv = (x_curr @ x_curr.T) / T_cv

                            nu_cv = y_curr - (H_current_cv @ x_curr)

                            nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                            nu_c_cv = nu_cv - nu_mean_cv
                            S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv

                            C_nu_x_cv = (nu_cv @ x_curr.T) / T_cv

                            z_cv = torch.cat([
                                A_yx_cv.reshape(-1),
                                A_xx_cv.reshape(-1),
                                S_nu_cv.reshape(-1),
                                C_nu_x_cv.reshape(-1),
                                H_current_cv.reshape(-1)
                            ], dim=0).reshape(1, -1)

                            dH_cv = model_mstep(z_cv)
                            dH_cv_mat = dH_cv.view(n, m)
                            H_next_cv = H_current_cv + dH_cv_mat

                            h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                            reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                            x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)

                            if em_iter == num_em_iters - 1:
                                loss_em_cv =  h_loss_cv + reg_cv + x_loss_cv
                            else:
                                loss_em_cv = h_loss_cv + reg_cv + x_loss_cv

                            if em_iter == 0:
                                weight = alpha[0]
                            elif em_iter == 1:
                                weight = alpha[1]
                            else:
                                weight = alpha[2]

                            total_loss_cv += weight * loss_em_cv
                            H_current_cv = H_next_cv

                        x_0_cv = x_curr[:, -1].detach()
                        H_base_cv = H_current_cv.detach()

                    cv_loss_seq = total_loss_cv / float(num_em_iters * datasets)
                    cv_loss_sum += cv_loss_seq.item()

            train_epoch = train_loss_sum / max(1, self.N_B)
            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)

            print(f"[M-step H 3datasets] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")

    def normelize_train_H_mstep_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, destination_path_RTS, num_em_iters=3,
                        alpha=(0.0, 0.0, 1.0), lambda_H=1e-3, generate_h=True, non_linear_f=False):
        """
        Single-function M-step training for H (observation matrix).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build observation-state statistics from x_smooth, feed M-net to predict ΔH.
        - Minimize Frobenius loss to ground-truth H (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: Need observation dimension!

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model_H.train()


        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            train_loss_sum = 0.0

            for _ in range(self.N_B):
                self.M_optimizer_H.zero_grad()

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select H_i and H_true by group
                if generate_h is True:
                    h_index = n_e // 10
                    H_base = SysModel.H_train[h_index]
                    H_true = SysModel.H_train_TRUE[h_index]
                else:
                    H_base = SysModel.H_train[n_e]
                    H_true = SysModel.H_train_TRUE[n_e]

                # --------- EM unrolling over H ---------
                H_current = H_base  # this will be updated each EM iteration
                total_loss = 0.0

                for em_iter in range(num_em_iters):

                    self.model.update_H(H_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for H M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    y_curr = y_seq  # [n, T]

                    A_yx = (y_curr @ x_curr.T) / T  # [n, m]
                    A_xx = (x_curr @ x_curr.T) / T  # [m, m]

                    # residuals with current H
                    Hx = H_current @ x_curr  # [n, T]
                    nu = y_curr - Hx  # [n, T]

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    S_nu = (nu_c @ nu_c.T) / T  # [n, n]

                    # correlation-like normalization for direction only (scale-invariant)
                    diag = torch.diagonal(S_nu, 0)  # [n]
                    inv_sqrt = torch.rsqrt(diag.clamp_min(1e-6))  # [n]
                    S_nu_corr = (inv_sqrt[:, None] * S_nu) * inv_sqrt[None, :]  # [n, n]

                    C_nu_x = (nu @ x_curr.T) / T  # [n, m]

                    z_in = torch.cat([
                        A_yx.reshape(-1),
                        A_xx.reshape(-1),
                        S_nu_corr.reshape(-1),
                        C_nu_x.reshape(-1),
                        H_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)
                    H_next = H_current + deltaH_mat


                    h_loss = torch.mean((H_next - H_true) ** 2)
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################
                    if em_iter == num_em_iters - 1:
                        loss_em = 15 * h_loss + reg + x_loss
                    else:
                        loss_em = h_loss + reg + x_loss
                    #############################################################################################################
                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss += weight * loss_em
                    H_current = H_next

                # after `for em_iter in range(num_em_iters):`
                loss = total_loss / float(num_em_iters)  # average over EM iterations
                loss_mult = loss
                loss_mult.backward()
                torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                self.M_optimizer_H.step()

                train_loss_sum += loss.detach().item()

                # ---------------- Validation ----------------
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    if generate_h is True:
                        h_index_cv = j // 10
                        H_base_cv = SysModel.H_valid[h_index_cv]
                        H_true_cv = SysModel.H_valid_TRUE[h_index_cv]
                    else:
                        H_base_cv = SysModel.H_valid[j]
                        H_true_cv = SysModel.H_valid_TRUE[j]

                    H_current_cv = H_base_cv.clone()
                    total_loss_cv = 0.0

                    for em_iter in range(num_em_iters):

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_H(H_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        y_curr = y_cv  # [n, T_cv]

                        A_yx_cv = (y_curr @ x_curr.T) / T_cv  # [n, m]
                        A_xx_cv = (x_curr @ x_curr.T) / T_cv  # [m, m]

                        nu_cv = y_curr - (H_current_cv @ x_curr)  # [n, T_cv]

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_c_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv  # [n, n]

                        diag_cv = torch.diagonal(S_nu_cv, 0)  # [n]
                        inv_sqrt_cv = torch.rsqrt(diag_cv.clamp_min(1e-6))  # [n]
                        S_nu_corr_cv = (inv_sqrt_cv[:, None] * S_nu_cv) * inv_sqrt_cv[None, :]  # [n, n]

                        C_nu_x_cv = (nu_cv @ x_curr.T) / T_cv  # [n, m]

                        z_cv = torch.cat([
                            A_yx_cv.reshape(-1),
                            A_xx_cv.reshape(-1),
                            S_nu_corr_cv.reshape(-1),
                            C_nu_x_cv.reshape(-1),
                            H_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dH_cv = model_mstep(z_cv)
                        dH_cv_mat = dH_cv.view(n, m)
                        H_next_cv = H_current_cv + dH_cv_mat

                        # same loss as train (but no backward)
                        h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                        reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################
                        if em_iter == num_em_iters - 1:
                            loss_em_cv = 15 * h_loss_cv + reg_cv + x_loss_cv
                        else:
                            loss_em_cv = h_loss_cv + reg_cv + x_loss_cv
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]  # if you really want the same scaling as in train

                        total_loss_cv += weight * loss_em_cv
                        H_current_cv = H_next_cv
                    cv_loss_seq = total_loss_cv / float(num_em_iters)
                    cv_loss_sum += cv_loss_seq.item()

            train_epoch = train_loss_sum / max(1, self.N_B)
            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)

            print(f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")





    def normelize_test_H_mstep_net(self, SysModel, test_input, test_target,
                       destination_path_RTS,destination_path_M, num_em_iters=3,
                       alpha=(0.0, 0.0, 1.0), lambda_H=1e-3, generate_h=True, init_x_list=None, init_P_list=None, non_linear_f=False):


        N_T = len(test_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: observation dimension

        all_test_losses = []
        all_h_losses = []

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load H M-step network from checkpoint (NO training)
        model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        final_H_list = []
        final_x_list = []
        
        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]    # [n, T]
                x_true_seq = test_target[j]  # [m, T]
                T = y_seq.size(-1)

                # Select H_base and H_true for this test sequence
                if generate_h is True:
                    h_index = j // 10
                    H_base = SysModel.H_test[h_index].to(device)
                    H_true = SysModel.H_test_TRUE[h_index].to(device)
                else:
                    # fallback: sequence-wise
                    H_base = SysModel.H_test[j].to(device)
                    H_true = SysModel.H_test_TRUE[j].to(device)

                # --------- EM unrolling over H ---------
                H_current = H_base.clone()
                total_loss = 0.0
                H_estimates = []
                H_losses_mse = []
                H_losses_total = []
                x_losses_mse = []

                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                for em_iter in range(num_em_iters):
                    # ----- RTS smoother with current H_current -----
                    self.model.update_H(H_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = P0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for H M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    y_curr = y_seq  # [n, T]

                    A_yx = (y_curr @ x_curr.T) / T  # [n, m]
                    A_xx = (x_curr @ x_curr.T) / T  # [m, m]

                    # residuals with current H
                    Hx = H_current @ x_curr  # [n, T]
                    nu = y_curr - Hx  # [n, T]

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    S_nu = (nu_c @ nu_c.T) / T  # [n, n]

                    # correlation normalization
                    diag = torch.diagonal(S_nu, 0)  # [n]
                    inv_sqrt = torch.rsqrt(diag.clamp_min(1e-6))  # [n]
                    S_nu_corr = (inv_sqrt[:, None] * S_nu) * inv_sqrt[None, :]  # [n, n]

                    C_nu_x = (nu @ x_curr.T) / T  # [n, m]

                    z_in = torch.cat([
                        A_yx.reshape(-1),
                        A_xx.reshape(-1),
                        S_nu_corr.reshape(-1),
                        C_nu_x.reshape(-1),
                        H_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only (no grad) -----
                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)
                    H_next = H_current + deltaH_mat

                    # Loss components
                    h_loss = torch.mean((H_next - H_true) ** 2)
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    
                    if em_iter == num_em_iters - 1:
                        loss_em = 15 * h_loss + reg + x_loss
                    else:
                        loss_em = h_loss + reg + x_loss

                    x_loss_sum_per_iter[em_iter] += x_loss.item()

                    # Alpha weighting
                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]

                    total_loss += weight * loss_em
                    H_current = H_next

                    all_test_losses.append(loss_em.item())
                    all_h_losses.append(h_loss.item())

                    # Store H estimates for selected sequences
                    if j % 5 == 0:
                        H_estimates.append(H_next.detach())
                        H_losses_mse.append(h_loss.item())
                        H_losses_total.append(loss_em.item())
                        x_losses_mse.append(x_loss.item())

                # Store final H and final x_smooth for this sequence
                final_H_list.append(H_current.detach().clone())  # [n, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())  # [m, 1]
                
                # Final loss for this sequence
                loss_list[j] = total_loss / float(num_em_iters)

                # Print summary for selected sequences
                if j % 5 == 0:
                    print(f"\n[H M-step TEST] sequence {j} summary")
                    print("H_true:\n", H_true.detach())
                    print("H_init (H_base):\n", H_base.detach())

                    mse_H_init = torch.mean((H_base - H_true) ** 2).item()
                    print(f"Initial H MSE loss = {mse_H_init:.6e}")
                    
                    for k, (H_est, h_mse, x_mse, total_val) in enumerate(
                            zip(H_estimates, H_losses_mse, x_losses_mse, H_losses_total)):
                        h_db = 10.0 * math.log10(h_mse)
                        x_db = 10.0 * math.log10(x_mse)
                        tot_db = 10.0 * math.log10(total_val)
                        print(f"\n  EM iter {k + 1}:")
                        print("  H_est:\n", H_est)
                        print(f"  H-loss (MSE_H)                 = {h_db:.2f} dB")
                        print(f"  x-loss (MSE_x)                 = {x_db:.2f} dB")
                        print(f"  total loss (H + reg + x)       = {tot_db:.2f} dB")

        mean_loss = loss_list.mean().item()
        print(f"[H M-step TEST] mean_loss={mean_loss:.6f}")
        
        # Average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)

        print("[H M-step TEST] Mean x-MSE per EM iteration:")
        for k in range(num_em_iters):
            mse_k = mean_x_mse_per_iter[k].item()
            db_k = 10.0 * math.log10(mse_k)
            print(f"  EM iter {k + 1}: mean x-MSE = {mse_k:.6e}  ({db_k: .2f} dB)")

        # Convert per-iteration mean x-MSE to numpy (linear and dB)
        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()

        return loss_list, final_H_list, final_x_list, mean_loss, mean_x_mse_per_iter_np, mean_x_mse_per_iter_db_np

    def train_H_mstep_net_old(self, SysModel, cv_input, cv_target, train_input, train_target,
                          destination_path_M, destination_path_RTS, num_em_iters=3,
                          alpha=(0.0, 0.0, 1.0), lambda_H=1e-3, generate_h=True, non_linear_f=False):
        """
        Single-function M-step training for H (observation matrix).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build observation-state statistics from x_smooth, feed M-net to predict ΔH.
        - Minimize Frobenius loss to ground-truth H (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: Need observation dimension!

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model_H.train()

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            train_loss_sum = 0.0

            for _ in range(self.N_B):
                self.M_optimizer_H.zero_grad()

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select H_i and H_true by group
                if generate_h is True:
                    h_index = n_e // 10
                    H_base = SysModel.H_train[h_index]
                    H_true = SysModel.H_train_TRUE[h_index]
                else:
                    H_base = SysModel.H_train[n_e]
                    H_true = SysModel.H_train_TRUE[n_e]

                # --------- EM unrolling over H ---------
                H_current = H_base  # this will be updated each EM iteration
                total_loss = 0.0

                for em_iter in range(num_em_iters):

                    self.model.update_H(H_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for H M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    y_curr = y_seq  # [n, T]

                    A_yx = (y_curr @ x_curr.T) / T  # [n, m]
                    A_xx = (x_curr @ x_curr.T) / T  # [m, m]

                    # residuals with current H
                    Hx = H_current @ x_curr  # [n, T]
                    nu = y_curr - Hx  # [n, T]

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    S_nu = (nu_c @ nu_c.T) / T  # [n, n]

                    C_nu_x = (nu @ x_curr.T) / T  # [n, m]

                    z_in = torch.cat([
                        A_yx.reshape(-1),
                        A_xx.reshape(-1),
                        S_nu.reshape(-1),
                        C_nu_x.reshape(-1),
                        H_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)
                    H_next = H_current + deltaH_mat

                    h_loss = torch.mean((H_next - H_true) ** 2)
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################
                    if em_iter == num_em_iters - 1:
                        loss_em = 15 * h_loss + reg + x_loss
                    else:
                        loss_em = h_loss + reg + x_loss
                    #############################################################################################################
                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss += weight * loss_em
                    H_current = H_next

                # after `for em_iter in range(num_em_iters):`
                loss = total_loss / float(num_em_iters)  # average over EM iterations
                loss_mult = loss
                loss_mult.backward()
                torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
                self.M_optimizer_H.step()

                train_loss_sum += loss.detach().item()

                # ---------------- Validation ----------------
            model_mstep.eval()
            cv_loss_sum = 0.0

            with torch.no_grad():
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    if generate_h is True:
                        h_index_cv = j // 10
                        H_base_cv = SysModel.H_valid[h_index_cv]
                        H_true_cv = SysModel.H_valid_TRUE[h_index_cv]
                    else:
                        H_base_cv = SysModel.H_valid[j]
                        H_true_cv = SysModel.H_valid_TRUE[j]

                    H_current_cv = H_base_cv.clone()
                    total_loss_cv = 0.0

                    for em_iter in range(num_em_iters):

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_H(H_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.init_hidden()
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        y_curr = y_cv  # [n, T_cv]

                        A_yx_cv = (y_curr @ x_curr.T) / T_cv  # [n, m]
                        A_xx_cv = (x_curr @ x_curr.T) / T_cv  # [m, m]

                        nu_cv = y_curr - (H_current_cv @ x_curr)  # [n, T_cv]

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_c_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv  # [n, n]

                        C_nu_x_cv = (nu_cv @ x_curr.T) / T_cv  # [n, m]

                        z_cv = torch.cat([
                            A_yx_cv.reshape(-1),
                            A_xx_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_nu_x_cv.reshape(-1),
                            H_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dH_cv = model_mstep(z_cv)
                        dH_cv_mat = dH_cv.view(n, m)
                        H_next_cv = H_current_cv + dH_cv_mat

                        # same loss as train (but no backward)
                        h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                        reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################
                        if em_iter == num_em_iters - 1:
                            loss_em_cv = 15 * h_loss_cv + reg_cv + x_loss_cv
                        else:
                            loss_em_cv = h_loss_cv + reg_cv + x_loss_cv
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]  # if you really want the same scaling as in train

                        total_loss_cv += weight * loss_em_cv
                        H_current_cv = H_next_cv
                    cv_loss_seq = total_loss_cv / float(num_em_iters)
                    cv_loss_sum += cv_loss_seq.item()

            train_epoch = train_loss_sum / max(1, self.N_B)
            cv_epoch = cv_loss_sum / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                torch.save(model_mstep, destination_path_M)

            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")
    def train_H_mstep_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                          destination_path_M, destination_path_RTS,load_destination_path_M=None, num_em_iters=3,
                          alpha=(0.0, 0.0, 1.0), lambda_H=1e-3, generate_h=True, train_init=None, cv_init=None):
        """
        Single-function M-step training for H (observation matrix).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build observation-state statistics from x_smooth, feed M-net to predict ΔH.
        - Minimize Frobenius loss to ground-truth H (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: Need observation dimension!

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model_H.train()
        if load_destination_path_M is not None:
            print(f"Loading H M-step model from: {load_destination_path_M}")
            model_mstep = torch.load(load_destination_path_M, weights_only=False).to(self.device)
            self.M_model_H = model_mstep
            model_mstep.train()
        self.M_optimizer_H = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            self.M_optimizer_H.zero_grad()
            total_loss_batch = 0.0
            loss_x_per_iter = torch.zeros(num_em_iters+1, device=device)
            loss_total_per_iter = torch.zeros(num_em_iters + 1, device=device)
            # ===== EPOCH ACCUMULATORS =====
            x_loss_first_epoch = 0.0
            h_loss_first_epoch = 0.0

            x_loss_em_epoch = [0.0 for _ in range(num_em_iters)]
            h_loss_em_epoch = [0.0 for _ in range(num_em_iters)]

            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select H_i and H_true by group
                if generate_h is True:
                    h_index = n_e // 10
                    H_base = SysModel.H_train[h_index]
                    H_true = SysModel.H_train_TRUE[h_index]
                else:
                    H_base = SysModel.H_train[n_e]
                    H_true = SysModel.H_train_TRUE[n_e]

                # --------- EM unrolling over H ---------
                H_current = H_base.clone()  # this will be updated each EM iteration

                total_loss_x_seq = 0.0
                total_loss_h_seq = 0.0
                total_loss_seq_backward = 0.0
###############first rts####################
                self.model.update_H(H_current)

                # E-step via frozen RTSNet → x_smooth
                if train_init is not None:
                    SysModel.m1x_0 = train_init[n_e]
                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                self.model.init_hidden()


                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                x_smooth[:, T - 1] = x_forward[:, T - 1]

                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                # ---------------- Stats for H M-network ----------------
                x_curr = x_smooth  # [m, T]
                y_curr = y_seq  # [n, T]
                # loss_x_per_iter[0] += torch.mean((x_curr - x_true_seq) ** 2).item()
                # h_loss1 = torch.mean((H_current - H_true) ** 2)
                x_loss_first = torch.mean((x_curr - x_true_seq) ** 2).item()
                h_loss1 = torch.mean((H_current - H_true) ** 2)
                h_loss_first = h_loss1.item()
                x_loss_first_epoch += x_loss_first
                h_loss_first_epoch += h_loss_first
                # reg1 = lambda_H
                # loss_total_per_iter[0] += 0.1*h_loss1 + 0.1*reg1 + loss_x_per_iter[0]
                for em_iter in range(num_em_iters):
                  #####firstm_step######
                    A_yx = (y_curr @ x_curr.T) / T  # [n, m]
                    A_xx = (x_curr @ x_curr.T) / T  # [m, m]

                    # residuals with current H
                    Hx = H_current @ x_curr  # [n, T]
                    nu = y_curr - Hx  # [n, T]

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    S_nu = (nu_c @ nu_c.T) / T  # [n, n]

                    C_nu_x = (nu @ x_curr.T) / T  # [n, m]

                    z_in = torch.cat([
                        A_yx.reshape(-1),
                        A_xx.reshape(-1),
                        S_nu.reshape(-1),
                        C_nu_x.reshape(-1),
                        H_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)
                    H_next = H_current + deltaH_mat
                    # print("deltaH mean abs:", deltaH.abs().mean().item())
                    # print("deltaH max abs:", deltaH.abs().max().item())
                    # print("H error:", torch.mean((H_current - H_true) ** 2).item())
                    # print("H next error:", torch.mean((H_next - H_true) ** 2).item())
                    # print("x loss:", torch.mean((x_curr - x_true_seq) ** 2).item())
                    h_loss = torch.mean((H_next - H_true) ** 2)
                    h_loss_em_epoch[em_iter] += h_loss.item()
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    total_loss_h_seq += h_loss +reg
                    h_loss_em = h_loss + reg
                    H_current = H_next
                    self.model.update_H(H_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                    self.model.init_hidden()


                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_smooth[:, T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for H M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    y_curr = y_seq  # [n, T]

                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    x_loss_em_epoch[em_iter] += x_loss.item()
                    total_loss_x_seq += x_loss
                    x_loss_em = x_loss
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################
                    # if em_iter == num_em_iters - 1:
                    #     loss_em = 15 * h_loss + reg + x_loss
                    # else:
                    #     loss_em = h_loss + reg + x_loss
                    #############################################################################################################
                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    # if em_iter == 0:
                    #     weight = alpha[0]  # First EM iteration
                    # elif em_iter == 1:
                    #     weight = alpha[1]  # Second EM iteration
                    # else:
                    #     weight = alpha[2]  # Third EM iteration (rest)
                    # total_loss += weight * loss_em
                    # H_current = H_next

                    # after `for em_iter in range(num_em_iters):`
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss_seq_backward += weight*(5*h_loss_em + x_loss_em)
                loss_seq_avarage = total_loss_seq_backward
                total_loss_batch += loss_seq_avarage
            total_loss_batch = total_loss_batch / float(self.N_B)
            total_loss_batch.backward()
            # for name, p in model_mstep.named_parameters():
            #     print(name, None if p.grad is None else p.grad.norm().item())
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.M_optimizer_H.step()
            print(f"[epoch {epoch}] FIRST RTS: "
                  f"x_loss={x_loss_first_epoch / self.N_B:.6f}, "
                  f"H_loss={h_loss_first_epoch / self.N_B:.6f}")

            for em_iter in range(num_em_iters):
                    print(f"[epoch {epoch}] MSTEP {em_iter}: "
                          f"x_loss={x_loss_em_epoch[em_iter] / self.N_B:.6f}, "
                          f"H_loss={h_loss_em_epoch[em_iter] / self.N_B:.6f}")
            # print('backwardloss', total_loss_batch.item())
            # print('first loss_x', loss_x_per_iter[0].item()/float(self.N_B), 'second loss_x', loss_x_per_iter[1].item()/float(self.N_B))
            # print('first loss', loss_total_per_iter[0].item() / float(self.N_B), 'second loss',
            #       loss_total_per_iter[1].item() / float(self.N_B))
                # ---------------- Validation ----------------
            model_mstep.eval()
            with torch.no_grad():
                total_loss_cv =0.0
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    if generate_h is True:
                        h_index_cv = j // 10
                        H_base_cv = SysModel.H_valid[h_index_cv]
                        H_true_cv = SysModel.H_valid_TRUE[h_index_cv]
                    else:
                        H_base_cv = SysModel.H_valid[j]
                        H_true_cv = SysModel.H_valid_TRUE[j]

                    H_current_cv = H_base_cv.clone()
                    total_loss_seq_cv = 0.0

                    # --- RTS smoother with current F_current_cv ---
                    self.model.update_H(H_current_cv)
                    if cv_init is not None:
                        SysModel.m1x_0 = cv_init[j]
                    self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                    x_f_cv = torch.empty(m, T_cv, device=device)
                    x_s_cv = torch.empty(m, T_cv, device=device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    # -------- stats, same as training --------
                    x_curr = x_s_cv  # [m, T_cv]
                    y_curr = y_cv  # [n, T_cv]

                    for em_iter in range(num_em_iters):


                        A_yx_cv = (y_curr @ x_curr.T) / T_cv  # [n, m]
                        A_xx_cv = (x_curr @ x_curr.T) / T_cv  # [m, m]

                        nu_cv = y_curr - (H_current_cv @ x_curr)  # [n, T_cv]

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_c_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv  # [n, n]

                        C_nu_x_cv = (nu_cv @ x_curr.T) / T_cv  # [n, m]

                        z_cv = torch.cat([
                            A_yx_cv.reshape(-1),
                            A_xx_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_nu_x_cv.reshape(-1),
                            H_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dH_cv = model_mstep(z_cv)
                        dH_cv_mat = dH_cv.view(n, m)
                        H_next_cv = H_current_cv + dH_cv_mat

                        # same loss as train (but no backward)
                        h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                        reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                        H_current_cv = H_next_cv.clone()
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################
                        # if em_iter == num_em_iters - 1:
                        #     loss_em_cv = 15 * h_loss_cv + reg_cv + x_loss_cv
                        # else:
                        #     loss_em_cv = h_loss_cv + reg_cv + x_loss_cv
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        # if em_iter == 0:
                        #     weight = alpha[0]
                        # elif em_iter == 1:
                        #     weight = alpha[1]
                        # else:
                        #     weight = alpha[2]  # if you really want the same scaling as in train

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_H(H_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                        self.model.init_hidden()


                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        y_curr = y_cv  # [n, T_cv]
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        if em_iter == 0:
                                weight = alpha[0]
                        elif em_iter == 1:
                                weight = alpha[1]
                        total_loss_seq_cv += weight*(x_loss_cv + 5*h_loss_cv.item() + 0.1*reg_cv.item())
                    cv_loss_seq_mean = total_loss_seq_cv
                    total_loss_cv += cv_loss_seq_mean.item()

            train_epoch = total_loss_batch
            cv_epoch = total_loss_cv / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)

            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} "
                f"cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f} "
                f"best_epoch={self.MSE_cv_idx_opt}"
            )

    def train_jointH_mstep_net(self, SysModel, cv_input, cv_target, train_input, train_target,
                          destination_path_M, destination_path_RTS, load_destination_path_M=None, load_path_RTS=None,num_em_iters=3,
                          alpha=(0.0, 0.0, 1.0), lambda_H=1e-3, generate_h=True,train_init = None,cv_init =None):
        """
        Single-function M-step training for H (observation matrix).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build observation-state statistics from x_smooth, feed M-net to predict ΔH.
        - Minimize Frobenius loss to ground-truth H (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: Need observation dimension!

        # Load and train RTSNet (smoother only)
        self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        # M-step model and optimizer
        model_mstep = self.M_model_H.train()
        if load_destination_path_M is not None:
            print(f"Loading H M-step model from: {load_destination_path_M}")
            model_mstep = torch.load(load_destination_path_M, weights_only=False).to(self.device)
            self.M_model_H = model_mstep
            model_mstep.train()
        self.optimizer_joint = torch.optim.Adam(list(self.model.parameters()) + list(model_mstep.parameters()),
            lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            self.model.train()
            self.optimizer_joint.zero_grad()
            total_loss_batch = 0.0
            loss_x_per_iter = torch.zeros(num_em_iters + 1, device=device)
            loss_total_per_iter = torch.zeros(num_em_iters + 1, device=device)
            # ===== EPOCH ACCUMULATORS =====
            x_loss_first_epoch = 0.0
            h_loss_first_epoch = 0.0

            x_loss_em_epoch = [0.0 for _ in range(num_em_iters)]
            h_loss_em_epoch = [0.0 for _ in range(num_em_iters)]

            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select H_i and H_true by group
                if generate_h is True:
                    h_index = n_e // 10
                    H_base = SysModel.H_train[h_index]
                    H_true = SysModel.H_train_TRUE[h_index]
                else:
                    H_base = SysModel.H_train[n_e]
                    H_true = SysModel.H_train_TRUE[n_e]

                # --------- EM unrolling over H ---------
                H_current = H_base.clone()  # this will be updated each EM iteration

                total_loss_x_seq = 0.0
                total_loss_h_seq = 0.0
                total_loss_seq_backward = 0.0
                ###############first rts####################
                self.model.update_H(H_current)

                # E-step via frozen RTSNet → x_smooth
                if train_init:
                    x_0 = train_init[n_e]
                    SysModel.m1x_0 = x_0
                else:
                    x_0 =SysModel.m1x_0

                self.model.InitSequence(x_0, T)
                self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                self.model.init_hidden()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                x_forward_list = []
                for t in range(T):
                    x_forward_list.append(self.model(y_seq[:, t], None, None, None))
                x_forward = torch.stack(x_forward_list, dim=1)  # [m, T]

                x_smooth_list = [None] * T
                x_smooth_list[T - 1] = x_forward[:, T - 1]

                self.model.InitBackward(x_smooth_list[T - 1])
                x_smooth_list[T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth_list[t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth_list[t + 2])

                x_smooth = torch.stack(x_smooth_list, dim=1)  # [m, T]

                # ---------------- Stats for H M-network ----------------
                x_curr = x_smooth  # [m, T]
                y_curr = y_seq  # [n, T]
                # loss_x_per_iter[0] += torch.mean((x_curr - x_true_seq) ** 2).item()
                # h_loss1 = torch.mean((H_current - H_true) ** 2)
                x_loss_first = torch.mean((x_curr - x_true_seq) ** 2).item()
                h_loss1 = torch.mean((H_current - H_true) ** 2)
                h_loss_first = h_loss1.item()
                x_loss_first_epoch += x_loss_first
                h_loss_first_epoch += h_loss_first
                # reg1 = lambda_H
                # loss_total_per_iter[0] += 0.1*h_loss1 + 0.1*reg1 + loss_x_per_iter[0]
                for em_iter in range(num_em_iters):
                    #####firstm_step######
                    A_yx = (y_curr @ x_curr.T) / T  # [n, m]
                    A_xx = (x_curr @ x_curr.T) / T  # [m, m]

                    # residuals with current H
                    Hx = H_current @ x_curr  # [n, T]
                    nu = y_curr - Hx  # [n, T]

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    S_nu = (nu_c @ nu_c.T) / T  # [n, n]

                    C_nu_x = (nu @ x_curr.T) / T  # [n, m]

                    z_in = torch.cat([
                        A_yx.reshape(-1),
                        A_xx.reshape(-1),
                        S_nu.reshape(-1),
                        C_nu_x.reshape(-1),
                        H_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)
                    H_next = H_current + deltaH_mat
                    # print("deltaH mean abs:", deltaH.abs().mean().item())
                    # print("deltaH max abs:", deltaH.abs().max().item())
                    # print("H error:", torch.mean((H_current - H_true) ** 2).item())
                    # print("H next error:", torch.mean((H_next - H_true) ** 2).item())
                    # print("x loss:", torch.mean((x_curr - x_true_seq) ** 2).item())
                    h_loss = torch.mean((H_next - H_true) ** 2)
                    h_loss_em_epoch[em_iter] += h_loss.item()
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    total_loss_h_seq += h_loss + reg
                    h_loss_em = h_loss + 0.5*reg
                    H_current = H_next
                    self.model.update_H(H_current)

                    # E-step via frozen RTSNet → x_smooth
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach()
                    self.model.init_hidden()

                    x_forward_list = []
                    for t in range(T):
                        x_forward_list.append(self.model(y_seq[:, t], None, None, None))
                    x_forward = torch.stack(x_forward_list, dim=1)

                    x_smooth_list = [None] * T
                    x_smooth_list[T - 1] = x_forward[:, T - 1]

                    self.model.InitBackward(x_smooth_list[T - 1])
                    x_smooth_list[T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth_list[t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth_list[t + 2])

                    x_smooth = torch.stack(x_smooth_list, dim=1)

                    # ---------------- Stats for H M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    y_curr = y_seq  # [n, T]

                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    x_loss_em_epoch[em_iter] += x_loss.item()
                    total_loss_x_seq += x_loss
                    x_loss_em = x_loss
                    ##########################################################
                    # # y-loss (measurement-space loss)
                    # H = SysModel.H.to(device)  # [n, m]
                    # y_hat = H @ x_curr  # [n, T]
                    # y_loss = torch.mean((y_hat - y_seq) ** 2)
                    # loss_em = 3 * f_loss + reg + x_loss + 1e-2 * y_loss
                    ##########################################################
                    # if em_iter == num_em_iters - 1:
                    #     loss_em = 15 * h_loss + reg + x_loss
                    # else:
                    #     loss_em = h_loss + reg + x_loss
                    #############################################################################################################
                    # loss_em = 3 * f_loss + reg + x_loss
                    # Apply your specific weighting: 0.05, 0.1, 0.85
                    # if em_iter == 0:
                    #     weight = alpha[0]  # First EM iteration
                    # elif em_iter == 1:
                    #     weight = alpha[1]  # Second EM iteration
                    # else:
                    #     weight = alpha[2]  # Third EM iteration (rest)
                    # total_loss += weight * loss_em
                    # H_current = H_next

                    # after `for em_iter in range(num_em_iters):`
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss_seq_backward += weight * (2 * h_loss_em + x_loss_em)
                loss_seq_avarage = total_loss_seq_backward
                total_loss_batch += loss_seq_avarage
            total_loss_batch = total_loss_batch / float(self.N_B)
            total_loss_batch.backward()
            # for name, p in model_mstep.named_parameters():
            #     print(name, None if p.grad is None else p.grad.norm().item())
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.optimizer_joint.step()
            print(f"[epoch {epoch}] FIRST RTS: "
                  f"x_loss={x_loss_first_epoch / self.N_B:.6f}, "
                  f"H_loss={h_loss_first_epoch / self.N_B:.6f}")

            for em_iter in range(num_em_iters):
                print(f"[epoch {epoch}] MSTEP {em_iter}: "
                      f"x_loss={x_loss_em_epoch[em_iter] / self.N_B:.6f}, "
                      f"H_loss={h_loss_em_epoch[em_iter] / self.N_B:.6f}")
            # print('backwardloss', total_loss_batch.item())
            # print('first loss_x', loss_x_per_iter[0].item()/float(self.N_B), 'second loss_x', loss_x_per_iter[1].item()/float(self.N_B))
            # print('first loss', loss_total_per_iter[0].item() / float(self.N_B), 'second loss',
            #       loss_total_per_iter[1].item() / float(self.N_B))
            # ---------------- Validation ----------------
            model_mstep.eval()
            with torch.no_grad():
                total_loss_cv = 0.0
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    if generate_h is True:
                        h_index_cv = j // 10
                        H_base_cv = SysModel.H_valid[h_index_cv]
                        H_true_cv = SysModel.H_valid_TRUE[h_index_cv]
                    else:
                        H_base_cv = SysModel.H_valid[j]
                        H_true_cv = SysModel.H_valid_TRUE[j]

                    H_current_cv = H_base_cv.clone()
                    total_loss_seq_cv = 0.0

                    # --- RTS smoother with current F_current_cv ---
                    self.model.update_H(H_current_cv)
                    if cv_init:
                        x_0_cv = cv_init[j]
                        SysModel.m1x_0 = x_0_cv
                    else:
                        x_0_cv = SysModel.m1x_0
                    self.model.InitSequence(x_0_cv, T_cv)
                    self.model.init_hidden()
                    self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)

                    x_f_cv = torch.empty(m, T_cv, device=device)
                    x_s_cv = torch.empty(m, T_cv, device=device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    # -------- stats, same as training --------
                    x_curr = x_s_cv  # [m, T_cv]
                    y_curr = y_cv  # [n, T_cv]

                    for em_iter in range(num_em_iters):

                        A_yx_cv = (y_curr @ x_curr.T) / T_cv  # [n, m]
                        A_xx_cv = (x_curr @ x_curr.T) / T_cv  # [m, m]

                        nu_cv = y_curr - (H_current_cv @ x_curr)  # [n, T_cv]

                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_c_cv = nu_cv - nu_mean_cv
                        S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv  # [n, n]

                        C_nu_x_cv = (nu_cv @ x_curr.T) / T_cv  # [n, m]

                        z_cv = torch.cat([
                            A_yx_cv.reshape(-1),
                            A_xx_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_nu_x_cv.reshape(-1),
                            H_current_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        # --- M-step forward only (no grad) ---
                        dH_cv = model_mstep(z_cv)
                        dH_cv_mat = dH_cv.view(n, m)
                        H_next_cv = H_current_cv + dH_cv_mat

                        # same loss as train (but no backward)
                        h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                        reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                        H_current_cv = H_next_cv.clone()
                        ##########################################################
                        # # y-loss (measurement-space loss)
                        # H = SysModel.H.to(device)  # [n, m]
                        # y_hat_cv  = H @ x_curr  # [n, T]
                        # y_loss_cv = torch.mean((y_hat_cv  - y_cv) ** 2)
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv + 1e-2 * y_loss_cv
                        ##########################################################
                        # if em_iter == num_em_iters - 1:
                        #     loss_em_cv = 15 * h_loss_cv + reg_cv + x_loss_cv
                        # else:
                        #     loss_em_cv = h_loss_cv + reg_cv + x_loss_cv
                        #########################################################################
                        # loss_em_cv = 3 * f_loss_cv + reg_cv + x_loss_cv
                        # if em_iter == 0:
                        #     weight = alpha[0]
                        # elif em_iter == 1:
                        #     weight = alpha[1]
                        # else:
                        #     weight = alpha[2]  # if you really want the same scaling as in train

                        # --- RTS smoother with current F_current_cv ---
                        self.model.update_H(H_current_cv)
                        self.model.InitSequence(SysModel.m1x_0.to(device), T_cv)
                        self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                        self.model.init_hidden()

                        x_f_cv = torch.empty(m, T_cv, device=device)
                        x_s_cv = torch.empty(m, T_cv, device=device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)

                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        # -------- stats, same as training --------
                        x_curr = x_s_cv  # [m, T_cv]
                        y_curr = y_cv  # [n, T_cv]
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]
                        total_loss_seq_cv += weight * (x_loss_cv + 2*h_loss_cv.item() + 0.5 * reg_cv.item())
                    cv_loss_seq_mean = total_loss_seq_cv
                    total_loss_cv += cv_loss_seq_mean.item()

            train_epoch = total_loss_batch
            cv_epoch = total_loss_cv / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)
                torch.save(self.model, destination_path_RTS)

            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} "
                f"cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f} "
                f"best_epoch={self.MSE_cv_idx_opt}"
            )

    def train_H_mstep_net_RTS(self, SysModel, cv_input, cv_target, train_input, train_target,
                          destination_path_M,  load_destination_path_M=None, num_em_iters=3,
                          alpha=(0.0, 0.0, 1.0), lambda_H=1e-3, generate_h=True, non_linear_f=False):
        """
        Single-function M-step training for H (observation matrix).
        - Freeze RTSNet loaded from destination_path_RTS and use it only to compute x_smooth.
        - Build observation-state statistics from x_smooth, feed M-net to predict ΔH.
        - Minimize Frobenius loss to ground-truth H (train/CV), save best M-model.
        """
        # Basic sizes
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: Need observation dimension!

        # M-step model and optimizer
        model_mstep = self.M_model_H.train()
        if load_destination_path_M is not None:
            print(f"Loading H M-step model from: {load_destination_path_M}")
            model_mstep = torch.load(load_destination_path_M, weights_only=False).to(self.device)
            self.M_model_H = model_mstep
            model_mstep.train()
        self.M_optimizer_H = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ---------------- Training ----------------
            model_mstep.train()
            self.M_optimizer_H.zero_grad()
            total_loss_batch = 0.0
            # ===== EPOCH ACCUMULATORS =====
            x_loss_first_epoch = 0.0
            h_loss_first_epoch = 0.0

            x_loss_em_epoch = [0.0 for _ in range(num_em_iters)]
            h_loss_em_epoch = [0.0 for _ in range(num_em_iters)]

            for _ in range(self.N_B):

                # Pick one training sequence
                n_e = random.randint(0, self.N_E - 1)
                y_seq = train_input[n_e]  # [n, T]
                x_true_seq = train_target[n_e]  # [m, T]
                T = y_seq.size(-1)

                # Select H_i and H_true by group
                if generate_h is True:
                    h_index = n_e // 10
                    H_base = SysModel.H_train[h_index]
                    H_true = SysModel.H_train_TRUE[h_index]
                else:
                    H_base = SysModel.H_train[n_e]
                    H_true = SysModel.H_train_TRUE[n_e]

                # --------- EM unrolling over H ---------
                H_current = H_base.clone()  # this will be updated each EM iteration

                total_loss_x_seq = 0.0
                total_loss_h_seq = 0.0
                total_loss_seq_backward = 0.0
                ###############first rts####################
                [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, _] = S_Test_ext_H(
                    SysModel,
                    y_seq.unsqueeze(0),
                    x_true_seq.unsqueeze(0),
                    H_list=[H_current],
                    generate_h=False,
                    init_x_list=[SysModel.m1x_0],
                    init_P_list=[SysModel.m2x_0]
                )

                x_curr = X_smooth.squeeze(0)  # [m, T]
                P_curr = P_smooth.squeeze(0)  # [m, m, T]
                y_curr = y_seq  # [n, T]

                x_loss_first = torch.mean((x_curr - x_true_seq) ** 2).item()
                h_loss1 = torch.mean((H_current - H_true) ** 2)
                h_loss_first = h_loss1.item()
                x_loss_first_epoch += x_loss_first
                h_loss_first_epoch += h_loss_first
                # reg1 = lambda_H
                # loss_total_per_iter[0] += 0.1*h_loss1 + 0.1*reg1 + loss_x_per_iter[0]
                for em_iter in range(num_em_iters):
                    #####firstm_step######

                    # residuals with current H
                    Hx = H_current @ x_curr
                    nu = y_curr - Hx
                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    x_mean = x_curr.mean(dim=1, keepdim=True)
                    x_c = x_curr - x_mean

                    S_nu = (nu_c @ nu_c.T) / T
                    C_nu_x = (nu_c @ x_c.T) / T

                    # EM-style H update statistics
                    C1 = (y_curr @ x_curr.T) / T  # same role as A_yx
                    C2 = torch.zeros((m, m), device=x_curr.device, dtype=x_curr.dtype)

                    for t in range(T):
                        xt = x_curr[:, t].unsqueeze(1)
                        C2 += xt @ xt.T + P_curr[:, :, t]

                    C2 = C2 / T
                    eps = 1e-5 * torch.eye(m, device=x_curr.device, dtype=x_curr.dtype)
                    C2 = 0.5 * (C2 + C2.T) + eps

                    H_em = torch.linalg.solve(C2.T, C1.T).T

                    z_in = torch.cat([
                        H_current.detach().reshape(-1),
                        H_em.detach().reshape(-1),
                        S_nu.detach().reshape(-1),
                        C_nu_x.detach().reshape(-1)
                    ], dim=0).reshape(1, -1)

                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)

                    beta = 0.1
                    H_next = H_em + beta * deltaH_mat
                    h_loss = torch.mean((H_next - H_true) ** 2)
                    h_loss_em_epoch[em_iter] += h_loss.item()
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    total_loss_h_seq += h_loss + reg
                    h_loss_em = h_loss + reg
                    H_current = H_next


                    ###############first rts####################
                    [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, _] = S_Test_ext_H(
                        SysModel,
                        y_seq.unsqueeze(0),
                        x_true_seq.unsqueeze(0),
                        H_list=[H_current],
                        generate_h=False,
                        init_x_list=[SysModel.m1x_0],
                        init_P_list=[SysModel.m2x_0]
                    )

                    x_curr = X_smooth.squeeze(0)  # [m, T]
                    P_curr = P_smooth.squeeze(0)  # [m, m, T]
                    y_curr = y_seq  # [n, T]


                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                    x_loss_em_epoch[em_iter] += x_loss.item()
                    total_loss_x_seq += x_loss
                    x_loss_em = x_loss

                    # after `for em_iter in range(num_em_iters):`
                    if em_iter == 0:
                        weight = alpha[0]  # First EM iteration
                    elif em_iter == 1:
                        weight = alpha[1]  # Second EM iteration
                    else:
                        weight = alpha[2]  # Third EM iteration (rest)
                    total_loss_seq_backward += weight * (0.1 * h_loss_em + x_loss_em)
                loss_seq_avarage = total_loss_seq_backward
                total_loss_batch += loss_seq_avarage
            total_loss_batch = total_loss_batch / float(self.N_B)
            total_loss_batch.backward()
            # for name, p in model_mstep.named_parameters():
            #     print(name, None if p.grad is None else p.grad.norm().item())
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.M_optimizer_H.step()
            print(f"[epoch {epoch}] FIRST RTS: "
                  f"x_loss={x_loss_first_epoch / self.N_B:.6f}, "
                  f"H_loss={h_loss_first_epoch / self.N_B:.6f}")

            for em_iter in range(num_em_iters):
                print(f"[epoch {epoch}] MSTEP {em_iter}: "
                      f"x_loss={x_loss_em_epoch[em_iter] / self.N_B:.6f}, "
                      f"H_loss={h_loss_em_epoch[em_iter] / self.N_B:.6f}")

            # ---------------- Validation ----------------
            model_mstep.eval()
            with torch.no_grad():
                total_loss_cv = 0.0
                for j in range(self.N_CV):
                    y_cv = cv_input[j]  # [n, T_cv]
                    x_true_cv_seq = cv_target[j]  # [m, T_cv]
                    T_cv = y_cv.size(-1)

                    if generate_h is True:
                        h_index_cv = j // 10
                        H_base_cv = SysModel.H_valid[h_index_cv]
                        H_true_cv = SysModel.H_valid_TRUE[h_index_cv]
                    else:
                        H_base_cv = SysModel.H_valid[j]
                        H_true_cv = SysModel.H_valid_TRUE[j]

                    H_current_cv = H_base_cv.clone()
                    total_loss_seq_cv = 0.0

                    [_mse_arr_cv, _mse_avg_cv, _mse_db_cv, X_smooth_cv, P_smooth_cv, _] = S_Test_ext_H(
                        SysModel,
                        y_cv.unsqueeze(0),
                        x_true_cv_seq.unsqueeze(0),
                        H_list=[H_current_cv],
                        generate_h=False,
                        init_x_list=[SysModel.m1x_0],
                        init_P_list=[SysModel.m2x_0]
                    )

                    x_curr = X_smooth_cv.squeeze(0)  # [m, T_cv]
                    P_curr_cv = P_smooth_cv.squeeze(0)  # [m, m, T_cv]
                    y_curr = y_cv  # [n, T_cv]

                    for em_iter in range(num_em_iters):

                        nu_cv = y_curr - (H_current_cv @ x_curr)
                        nu_mean_cv = nu_cv.mean(dim=1, keepdim=True)
                        nu_c_cv = nu_cv - nu_mean_cv
                        x_mean_cv = x_curr.mean(dim=1, keepdim=True)
                        x_c_cv = x_curr - x_mean_cv

                        S_nu_cv = (nu_c_cv @ nu_c_cv.T) / T_cv
                        C_nu_x_cv = (nu_c_cv @ x_c_cv.T) / T_cv

                        C1_cv = (y_curr @ x_curr.T) / T_cv
                        C2_cv = torch.zeros((m, m), device=x_curr.device, dtype=x_curr.dtype)

                        for t in range(T_cv):
                            xt = x_curr[:, t].unsqueeze(1)
                            C2_cv += xt @ xt.T + P_curr_cv[:, :, t]

                        C2_cv = C2_cv / T_cv
                        eps_cv = 1e-5 * torch.eye(m, device=x_curr.device, dtype=x_curr.dtype)
                        C2_cv = 0.5 * (C2_cv + C2_cv.T) + eps_cv

                        H_em_cv = torch.linalg.solve(C2_cv.T, C1_cv.T).T

                        z_cv = torch.cat([
                            H_current_cv.reshape(-1),
                            H_em_cv.reshape(-1),
                            S_nu_cv.reshape(-1),
                            C_nu_x_cv.reshape(-1)
                        ], dim=0).reshape(1, -1)

                        dH_cv = model_mstep(z_cv)
                        dH_cv_mat = dH_cv.view(n, m)

                        beta = 0.1
                        H_next_cv = H_em_cv + beta * dH_cv_mat

                        # same loss as train (but no backward)
                        h_loss_cv = torch.mean((H_next_cv - H_true_cv) ** 2)
                        reg_cv = lambda_H * torch.mean(dH_cv_mat ** 2)
                        H_current_cv = H_next_cv.clone()
                        [_mse_arr_cv, _mse_avg_cv, _mse_db_cv, X_smooth_cv, P_smooth_cv, _] = S_Test_ext_H(
                            SysModel,
                            y_cv.unsqueeze(0),
                            x_true_cv_seq.unsqueeze(0),
                            H_list=[H_current_cv],
                            generate_h=False,
                            init_x_list=[SysModel.m1x_0],
                            init_P_list=[SysModel.m2x_0]
                        )

                        x_curr = X_smooth_cv.squeeze(0)  # [m, T_cv]
                        P_curr_cv = P_smooth_cv.squeeze(0)  # [m, m, T_cv]
                        y_curr = y_cv  # [n, T_cv]

                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        if em_iter == 0:
                            weight = alpha[0]
                        elif em_iter == 1:
                            weight = alpha[1]
                        else:
                            weight = alpha[2]
                        total_loss_seq_cv += weight * (x_loss_cv + 0.1 * h_loss_cv.item() + 0.1 * reg_cv.item())
                    cv_loss_seq_mean = total_loss_seq_cv
                    total_loss_cv += cv_loss_seq_mean.item()

            train_epoch = total_loss_batch
            cv_epoch = total_loss_cv / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)

            print(
                f"[M-step] epoch={epoch:03d} train={train_epoch:.6f} "
                f"cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f} "
                f"best_epoch={self.MSE_cv_idx_opt}"
            )




    def test_H_mstep_net(self, SysModel, test_input, test_target,
                         destination_path_RTS, destination_path_M, num_em_iters=1,
                         lambda_H=1e-3, generate_h=True, init_x_list=None,init_H=None, init_P_list=None,):

        N_T = len(test_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: observation dimension

        all_test_losses = []

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load H M-step network from checkpoint (NO training)
        model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        h_loss_per_iter = torch.zeros(num_em_iters, device=device)
        reg_per_iter = torch.zeros(num_em_iters, device=device)
        final_H_list = []
        final_x_list = []
        list_x = []
        loss_x_list = torch.zeros(num_em_iters+1, device=device)
        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]  # [n, T]
                x_true_seq = test_target[j]  # [m, T]
                T = y_seq.size(-1)

                # Select H_base and H_true for this test sequence
                if generate_h is True:
                    h_index = j // 10
                    H_base = SysModel.H_test[h_index].to(device)
                    H_true = SysModel.H_test_TRUE[h_index].to(device)
                else:
                    # fallback: sequence-wise
                    # sequence-wise
                    if init_H is not None:
                        H_base = init_H[j]
                    else:
                        H_base = SysModel.H_test[j]
                    H_true = SysModel.H_test_TRUE[j].to(device)

                # --------- EM unrolling over H ---------
                H_current = H_base.clone()
                total_loss = 0.0

                if init_x_list is not None:
                    P0 = SysModel.m2x_0
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                # ----- RTS smoother with current H_current -----
                self.model.update_H(H_current)
                self.model.InitSequence(x0.clone().detach(), T)
                self.model.prior_Sigma = P0.clone().detach()
                self.model.init_hidden()

                x_forward = torch.empty(m, T, device=device)
                x_smooth = torch.empty(m, T, device=device)

                for t in range(T):
                    x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                x_smooth[:, T - 1] = x_forward[:, T - 1]
                self.model.InitBackward(x_smooth[:, T - 1])
                x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                # ---------------- Stats for H M-network ----------------
                x_curr = x_smooth  # [m, T]
                y_curr = y_seq  # [n, T]
                x_loss_init = torch.mean((x_curr - x_true_seq) ** 2).item()
                loss_x_list[0] += x_loss_init
                for em_iter in range(num_em_iters):


                    A_yx = (y_curr @ x_curr.T) / T  # [n, m]
                    A_xx = (x_curr @ x_curr.T) / T  # [m, m]

                    # residuals with current H
                    Hx = H_current @ x_curr  # [n, T]
                    nu = y_curr - Hx  # [n, T]

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    S_nu = (nu_c @ nu_c.T) / T  # [n, n]


                    C_nu_x = (nu @ x_curr.T) / T  # [n, m]

                    z_in = torch.cat([
                        A_yx.reshape(-1),
                        A_xx.reshape(-1),
                        S_nu.reshape(-1),
                        C_nu_x.reshape(-1),
                        H_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only (no grad) -----
                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)
                    H_next = H_current + deltaH_mat

                    # Loss components
                    h_loss = torch.mean((H_next - H_true) ** 2)
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    H_current = H_next.clone()
                    # ----- RTS smoother with current H_current -----
                    self.model.update_H(H_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.prior_Sigma = P0.clone().detach()
                    self.model.init_hidden()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for H M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    y_curr = y_seq  # [n, T]
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)

                    loss_x_list[em_iter + 1] += x_loss.item()

                    # if em_iter == num_em_iters - 1:
                    #     loss_em = 15 * h_loss + reg + x_loss
                    # else:
                    #     loss_em = h_loss + reg + x_loss

                    x_loss_sum_per_iter[em_iter] += x_loss.item()
                    h_loss_per_iter[em_iter] += h_loss.item()
                    reg_per_iter[em_iter] += reg.item()
                    loss_em = h_loss + reg + x_loss
                    total_loss += loss_em.item()
                    # total_loss += weight * loss_em
                    # H_current = H_next
                    #
                    # all_test_losses.append(loss_em.item())
                    # all_h_losses.append(h_loss.item())
                    all_test_losses.append(loss_em.item())
                    # Store H estimates for selected sequences
                list_x.append(x_curr.detach().clone())
                # Store final H and final x_smooth for this sequence
                final_H_list.append(H_current.detach().clone())  # [n, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())  # [m, 1]

                # Final loss for this sequence
                loss_list[j] = total_loss / float(num_em_iters)


        mean_loss = loss_list.mean().item()
        print(f"[H M-step TEST] mean_loss={mean_loss:.6f}")
        # Average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)
        mean_H_mse_per_iter = h_loss_per_iter / float(N_T)
        mean_reg_per_iter = reg_per_iter / float(N_T)
        mean_x_loss_list = loss_x_list / float(N_T)

        print("[H M-step TEST] Mean losses per EM iteration:")

        x0 = mean_x_loss_list[0].item()
        print(f"  init: x_loss = {x0:.6e} ({10.0 * math.log10(x0):.2f} dB)")

        for k in range(num_em_iters):
            x_k = mean_x_loss_list[k + 1].item()
            h_k = mean_H_mse_per_iter[k].item()
            reg_k = mean_reg_per_iter[k].item()

            print(
                f"  EM iter {k + 1}: "
                f"x_loss = {x_k:.6e} ({10.0 * math.log10(x_k):.2f} dB), "
                f"H_loss = {h_k:.6e} ({10.0 * math.log10(h_k):.2f} dB), "
                f"reg = {reg_k:.6e} ({10.0 * math.log10(reg_k):.2f} dB)"
            )


        # Convert per-iteration mean x-MSE to numpy (linear and dB)
        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()

        # return loss_list, final_H_list, final_x_list, mean_loss, mean_x_mse_per_iter_np, mean_x_mse_per_iter_db_np
        tensor_x=torch.stack(list_x)
        return mean_x_mse_per_iter_np,mean_H_mse_per_iter, final_H_list, final_x_list,tensor_x


    def test_H_mstep_net_old(self, SysModel, test_input, test_target,
                         destination_path_RTS, destination_path_M, num_em_iters=3,
                         alpha=(0.0, 0.0, 1.0), lambda_H=1e-3, generate_h=True, init_x_list=None, init_P_list=None,
                         non_linear_f=False):

        N_T = len(test_input)
        m = SysModel.m
        n = SysModel.n  # CRITICAL: observation dimension

        all_test_losses = []
        all_h_losses = []

        # Load and freeze RTSNet (smoother only)
        self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load H M-step network from checkpoint (NO training)
        model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

        loss_list = torch.zeros(N_T, device=device)
        x_loss_sum_per_iter = torch.zeros(num_em_iters, device=device)
        h_loss_per_iter = torch.zeros(num_em_iters, device=device)
        final_H_list = []
        final_x_list = []

        with torch.no_grad():
            for j in range(N_T):
                # ----- one test sequence -----
                y_seq = test_input[j]  # [n, T]
                x_true_seq = test_target[j]  # [m, T]
                T = y_seq.size(-1)

                # Select H_base and H_true for this test sequence
                if generate_h is True:
                    h_index = j // 10
                    H_base = SysModel.H_test[h_index].to(device)
                    H_true = SysModel.H_test_TRUE[h_index].to(device)
                else:
                    # fallback: sequence-wise
                    H_base = SysModel.H_test[j].to(device)
                    H_true = SysModel.H_test_TRUE[j].to(device)

                # --------- EM unrolling over H ---------
                H_current = H_base.clone()
                total_loss = 0.0
                H_estimates = []
                H_losses_mse = []
                H_losses_total = []
                x_losses_mse = []

                if (init_x_list is not None) and (init_P_list is not None):
                    P0 = init_P_list
                    x0 = init_x_list[j]
                else:
                    P0 = SysModel.m2x_0
                    x0 = SysModel.m1x_0

                for em_iter in range(num_em_iters):
                    # ----- RTS smoother with current H_current -----
                    self.model.update_H(H_current)
                    self.model.InitSequence(x0.clone().detach(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = P0.clone().detach()

                    x_forward = torch.empty(m, T, device=device)
                    x_smooth = torch.empty(m, T, device=device)

                    for t in range(T):
                        x_forward[:, t] = self.model(y_seq[:, t], None, None, None)

                    x_smooth[:, T - 1] = x_forward[:, T - 1]
                    self.model.InitBackward(x_smooth[:, T - 1])
                    x_smooth[:, T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_smooth[:, t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth[:, t + 2])

                    # ---------------- Stats for H M-network ----------------
                    x_curr = x_smooth  # [m, T]
                    y_curr = y_seq  # [n, T]

                    A_yx = (y_curr @ x_curr.T) / T  # [n, m]
                    A_xx = (x_curr @ x_curr.T) / T  # [m, m]

                    # residuals with current H
                    Hx = H_current @ x_curr  # [n, T]
                    nu = y_curr - Hx  # [n, T]

                    nu_mean = nu.mean(dim=1, keepdim=True)
                    nu_c = nu - nu_mean
                    S_nu = (nu_c @ nu_c.T) / T  # [n, n]


                    C_nu_x = (nu @ x_curr.T) / T  # [n, m]

                    z_in = torch.cat([
                        A_yx.reshape(-1),
                        A_xx.reshape(-1),
                        S_nu.reshape(-1),
                        C_nu_x.reshape(-1),
                        H_current.reshape(-1)
                    ], dim=0).reshape(1, -1)

                    # ----- M-step forward only (no grad) -----
                    deltaH = model_mstep(z_in)
                    deltaH_mat = deltaH.view(n, m)
                    H_next = H_current + deltaH_mat

                    # Loss components
                    h_loss = torch.mean((H_next - H_true) ** 2)
                    reg = lambda_H * torch.mean(deltaH_mat ** 2)
                    x_loss = torch.mean((x_curr - x_true_seq) ** 2)


                    if em_iter == num_em_iters - 1:
                        loss_em = 15 * h_loss + reg + x_loss
                    else:
                        loss_em = h_loss + reg + x_loss

                    x_loss_sum_per_iter[em_iter] += x_loss.item()
                    h_loss_per_iter[em_iter] += h_loss.item()

                    # Alpha weighting
                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]

                    total_loss += weight * loss_em
                    H_current = H_next

                    all_test_losses.append(loss_em.item())
                    all_h_losses.append(h_loss.item())

                    # Store H estimates for selected sequences
                    if j % 5 == 0:
                        H_estimates.append(H_next.detach())
                        H_losses_mse.append(h_loss.item())
                        H_losses_total.append(loss_em.item())
                        x_losses_mse.append(x_loss.item())

                # Store final H and final x_smooth for this sequence
                final_H_list.append(H_current.detach().clone())  # [n, m]
                final_x_list.append(x_curr[:, -1].unsqueeze(-1).detach().clone())  # [m, 1]

                # Final loss for this sequence
                loss_list[j] = total_loss / float(num_em_iters)

                # Print summary for selected sequences
                if j % 5 == 0:
                    print(f"\n[H M-step TEST] sequence {j} summary")
                    print("H_true:\n", H_true.detach())
                    print("H_init (H_base):\n", H_base.detach())

                    mse_H_init = torch.mean((H_base - H_true) ** 2).item()
                    print(f"Initial H MSE loss = {mse_H_init:.6e}")

                    for k, (H_est, h_mse, x_mse, total_val) in enumerate(
                            zip(H_estimates, H_losses_mse, x_losses_mse, H_losses_total)):
                        h_db = 10.0 * math.log10(h_mse)
                        x_db = 10.0 * math.log10(x_mse)
                        tot_db = 10.0 * math.log10(total_val)
                        print(f"\n  EM iter {k + 1}:")
                        print("  H_est:\n", H_est)
                        print(f"  H-loss (MSE_H)                 = {h_db:.2f} dB")
                        print(f"  x-loss (MSE_x)                 = {x_db:.2f} dB")
                        print(f"  total loss (H + reg + x)       = {tot_db:.2f} dB")

        mean_loss = loss_list.mean().item()
        print(f"[H M-step TEST] mean_loss={mean_loss:.6f}")

        # Average x-MSE for each EM iteration over all sequences
        mean_x_mse_per_iter = x_loss_sum_per_iter / float(N_T)
        mean_H_mse_per_iter = h_loss_per_iter / float(N_T)

        print("[H M-step TEST] Mean x-MSE per EM iteration:")
        for k in range(num_em_iters):
            mse_k = mean_x_mse_per_iter[k].item()
            db_k = 10.0 * math.log10(mse_k)
            print(f"  EM iter {k + 1}: mean x-MSE = {mse_k:.6e}  ({db_k: .2f} dB)")

        # Convert per-iteration mean x-MSE to numpy (linear and dB)
        mean_x_mse_per_iter_np = mean_x_mse_per_iter.detach()
        mean_x_mse_per_iter_db_np = (10.0 * torch.log10(mean_x_mse_per_iter)).detach()

        # return loss_list, final_H_list, final_x_list, mean_loss, mean_x_mse_per_iter_np, mean_x_mse_per_iter_db_np
        return mean_x_mse_per_iter_np,mean_H_mse_per_iter, final_H_list, final_x_list

#     def NNTrain_stocks(self, SysModel,cv_input, cv_target,train_input, train_target,path_results,load_model_path=None,generate_f=True, generate_h=False,
#                 train_x0=None, cv_x0=None):
#         """
#         Inputs:
#             SysModel:
#                 Contains the fixed observation model H, initial F (learned/updated in training),
#                 and dimensions m (state) and n (measurement).
#
#             train_input / cv_input:
#                 List of sequences (rolling windows) of measurements.
#                 Each element is a Tensor of shape [n, T] (usually n=1 for a single stock price).
#                 Example: y_window = [y(t0), ..., y(t0+T-1)].
#
#             train_target / cv_target:
#                 List of "next-day" target sequences aligned to each window.
#                 Each element is a Tensor containing the true next measurements:
#                     y_next = [y(t0+1), ..., y(t0+T)]
#                 Shape should match the loss indexing:
#                     - If you use y_next[:, t] for t=0..T-1, then y_next is [n, T]
#                     - If you store [y(t0), ..., y(t0+T)] (length T+1), then use y_next[:, t+1]
#
#             train_x0 / cv_x0:
#                 List of scalars (one per window) giving the measurement BEFORE the window:
#                     y(t0-1)
#                 Used to build the initial state:
#                     x0 = [ y(t0-1) , 0.5 ]   (fixed momentum)
#
#         Training objective:
#             For each t in 0..T-1, predict the next-day measurement:
#                 y_pred(t+1|t) = H F x_forward(t)
#             and minimize a weighted MSE:
#                 loss = sum_t w_t * MSE(y_pred(t+1|t), y_true(t+1))
#             with increasing weights w_t so the last prediction gets the most weight.
#         """
#         self.N_E = len(train_input)
#         self.N_CV = len(cv_input)
#
#
#         self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
#         self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)
#
#         MSE_train_linear_batch = torch.empty([self.N_B], device=self.device)
#         self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
#         self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)
#
#         if load_model_path is not None:
#             print("loading model_and keep training them")
#             self.model = torch.load(load_model_path, map_location=self.device, weights_only=False).to(self.device).eval()
#             self.optimizer = torch.optim.Adam(self.model.parameters(),
#                                               lr=self.learningRate,
#                                               weight_decay=self.weightDecay)
#
#         # Training Mode
#         self.model.train()
#
#         ##############
#         ### Epochs ###
#         ##############
#         self.MSE_cv_dB_opt = 1000
#         self.MSE_cv_idx_opt = 0
#         nan_streak = 0
#
#         for ti in range(0, self.N_steps):
#
#             ###############################
#             ### Training Sequence Batch ###
#             ###############################
#             self.model.train()
#             self.optimizer.zero_grad()
#
#             Batch_Optimizing_LOSS_sum = 0
#
#             for j in range(0, self.N_B):
#
#                 self.model.init_hidden()
#                 n_e = random.randint(0, self.N_E - 1)
#                 y_next_day = train_target[n_e]        # [n, T]
#                 y_training = train_input[n_e]         # [n, T]
#
#                 # =========================================================
#                 # PER-WINDOW NORMALIZATION (can be reproduced at test time)
#                 # Normalize each window independently by its own statistics
#                 # =========================================================
#                 y_mean = y_training.mean()
#                 y_std = y_training.std()
#                 if y_std < 1e-6:  # avoid division by zero for constant windows
#                     y_std = torch.tensor(1.0, device=y_training.device, dtype=y_training.dtype)
#
#                 y_training_norm = (y_training - y_mean) / y_std
#                 y_next_day_norm = (y_next_day - y_mean) / y_std
#
#                 if generate_f is True:  ####if we train with different f
#                     index = n_e // 10
#                     SysModel.F = SysModel.F_train[index]
#                     self.model.update_F(SysModel.F)
#                 else:
#                     # Use the first (and only) F matrix when not varying F
#                     if isinstance(SysModel.F_train, list):
#                         SysModel.F = SysModel.F_train[0]
#                     else:
#                         SysModel.F = SysModel.F_train
#                     self.model.update_F(SysModel.F)
#
#
#                 SysModel.T = y_training_norm.size()[-1]
#
#                 # =========================================================
#                 # PER-SEQUENCE x0:
#                 # x0[0] = price_before_window  → normalize by window stats
#                 # x0[1] = trend0               → keep as-is (already a price-difference scale)
#                 # =========================================================
#                 x0_raw = train_x0[n_e]  # [2] tensor: [price, trend]
#                 # Normalize both components by y_std so x0 lives in same space as RTSNet states
#                 # price: subtract mean AND divide by std; trend: divide by std only (it's a difference)
#                 x0_norm = torch.stack([
#                     (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                     x0_raw[1] / y_std               # trend: scale only (no mean shift, same units as price)
#                 ])
#                 SysModel.m1x_0 = x0_norm.view(SysModel.m, 1)  # [m, 1] = [2, 1]
#
#                 # Init Hidden State
#                 self.model.InitSequence(SysModel.m1x_0, SysModel.T)
#                 self.model.init_hidden()
#
#                 # FIXED: Forward pass - use list comprehension to preserve computation graph
#                 x_out_training_forward_list = [self.model(y_training_norm[:, t], None, None, None)
#                                                for t in range(SysModel.T)]
#                 x_out_training_forward = torch.stack(x_out_training_forward_list, dim=1)  # [m, T]
#
#                 # FIXED: Backward smoothing - use list to preserve computation graph
#                 x_out_training_list = [None] * SysModel.T
#                 x_out_training_list[SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
#                 self.model.InitBackward(x_out_training_list[SysModel.T - 1])
#
#                 if SysModel.T >= 2:
#                     x_out_training_list[SysModel.T - 2] = self.model(None,
#                                                                       x_out_training_forward[:, SysModel.T - 2],
#                                                                       x_out_training_forward[:, SysModel.T - 1],
#                                                                       None)
#                 for t in range(SysModel.T - 3, -1, -1):
#                     x_out_training_list[t] = self.model(None,
#                                                         x_out_training_forward[:, t],
#                                                         x_out_training_forward[:, t + 1],
#                                                         x_out_training_list[t + 2])
#
#                 x_out_training = torch.stack(x_out_training_list, dim=1)  # [m, T]
#
#                 # =========================================================
#                 # LOSS: weighted next-day prediction (per t)
#                 # Using SMOOTHED states for better estimates
#                 # ASSUMPTION: T >= 2 always
#                 # y_next_day_norm is [y_2, y_3, ..., y_{T+1}] (length T)
#                 # We predict y_{t+1} from x_t using H*F*x_t
#                 # PLUS extra prediction y_{T+1} from x_T with weight=2
#                 # =========================================================
#                 HF = SysModel.H @ SysModel.F  # [n, m]
#
#                 # Weights for standard predictions: increasing over time
#                 weights = torch.arange(1, SysModel.T + 1, device=y_training_norm.device, dtype=y_training_norm.dtype)
#                 weights = weights / torch.sum(weights)  # normalize
#
#                 rtsnet_loss = 0
#                 for t in range(0, SysModel.T):
#                     y_pred_next_t = HF @ x_out_training[:, t]   # [n] - using SMOOTHED state
#                     y_true_next_t = y_next_day_norm[:, t]       # [n] - this is y_{t+2} for t=0, y_{t+3} for t=1, etc.
#                     rtsnet_loss = rtsnet_loss + weights[t] * self.loss_fn(y_pred_next_t, y_true_next_t)
#
#                 # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                 x_last = x_out_training[:, -1]  # x_T
#                 y_pred_Tp1 = HF @ x_last  # predict y_{T+1}
#                 y_true_Tp1 = y_next_day_norm[:, -1]  # y_{T+1}
#                 loss_last = self.loss_fn(y_pred_Tp1, y_true_Tp1)
#                 rtsnet_loss = rtsnet_loss + 2.0 * loss_last  # Double weight for last prediction
#
#                 # =========================================================
#                 # MINI-BATCH: Accumulate loss across all sequences in batch
#                 # DON'T call backward inside loop!
#                 # =========================================================
#                 Batch_Optimizing_LOSS_sum += rtsnet_loss  # keep gradient graph alive
#                 MSE_train_linear_batch[j] = rtsnet_loss.detach().item()  # log without gradient
#
#             # =========================================================
#             # MINI-BATCH: Single backward on accumulated batch loss
#             # =========================================================
#             Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
#             Batch_Optimizing_LOSS_mean.backward()  # single backward for entire batch
#
#             # Gradient check
#             bad_grad = False
#             for p in self.model.parameters():
#                 if p.grad is None:
#                     continue
#                 if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
#                     bad_grad = True
#                     break
#
#             if bad_grad:
#                 print("NaN/Inf gradients → batch skipped")
#                 nan_streak += 1
#                 if nan_streak >= 3:
#                     print("Stopping training (3 consecutive bad batches).")
#                     # Save the best model found so far before early exit
#                     if self.MSE_cv_idx_opt < ti and hasattr(self, 'best_model_state'):
#                         os.makedirs(os.path.dirname(path_results) if os.path.dirname(path_results) else '.', exist_ok=True)
#                         torch.save(self.best_model_state, path_results)
#                         print(f"Saved best model from epoch {self.MSE_cv_idx_opt} to {path_results}")
#                     return
#                 self.model.zero_grad(set_to_none=True)
#                 continue
#
#             nan_streak = 0
#
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             self.optimizer.step()  # Single update per mini-batch
#
#             # Logging
#             self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
#             self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
#
#             #################################
#             ### Validation Sequence Batch ###
#             #################################
#             self.model.eval()
#             with torch.no_grad():
#                 MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)
#
#                 for j in range(0, self.N_CV):
#                     y_cv = cv_input[j]                    # [n, T_test]
#                     y_next_day_cv = cv_target[j]          # [n, T_test]
#
#                     # =========================================================
#                     # PER-WINDOW NORMALIZATION (same as training)
#                     # =========================================================
#                     y_mean = y_cv.mean()
#                     y_std = y_cv.std()
#                     if y_std < 1e-6:
#                         y_std = torch.tensor(1.0, device=y_cv.device, dtype=y_cv.dtype)
#
#                     y_cv_norm = (y_cv - y_mean) / y_std
#                     y_next_day_cv_norm = (y_next_day_cv - y_mean) / y_std
#
#                     SysModel.T_test = y_cv_norm.size()[-1]
#
#                     if generate_f is True:  ####if we valid with different f
#                         index = j // 10
#                         SysModel.F = SysModel.F_valid[index]
#                         self.model.update_F(SysModel.F)
#                     else:
#                         # Use the first (and only) F matrix when not varying F
#                         if isinstance(SysModel.F_valid, list):
#                             SysModel.F = SysModel.F_valid[0]
#                         else:
#                             SysModel.F = SysModel.F_valid
#                         self.model.update_F(SysModel.F)
#
#                     if generate_h is True:  ####if we valid with different h
#                         index = j // 10
#                         SysModel.H = SysModel.H_valid[index]
#                         # Note: update_H not available in base RTSNet
#                     else:
#                         # Use the first (and only) H matrix when not varying H
#                         if isinstance(SysModel.H_valid, list):
#                             SysModel.H = SysModel.H_valid[0]
#                         else:
#                             SysModel.H = SysModel.H_valid
#
#                     # x0 (CV): normalize both components by y_std
#                     x0_raw = cv_x0[j]  # [2] tensor
#                     x0_norm_cv = torch.stack([
#                         (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                         x0_raw[1] / y_std               # trend: scale only (same units as price)
#                     ])
#                     SysModel.m1x_0 = x0_norm_cv.view(SysModel.m, 1)  # [m, 1] = [2, 1]
#                     self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
#                     self.model.init_hidden()
#
#                     # FIXED: Forward pass - use list comprehension (consistency in no_grad context)
#                     x_out_cv_forward_list = [self.model(y_cv_norm[:, t], None, None, None)
#                                              for t in range(SysModel.T_test)]
#                     x_out_cv_forward = torch.stack(x_out_cv_forward_list, dim=1)  # [m, T_test]
#
#                     # FIXED: Backward pass - use list comprehension (SMOOTHING IN CV!)
#                     x_out_cv_list = [None] * SysModel.T_test
#                     x_out_cv_list[SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
#                     self.model.InitBackward(x_out_cv_list[SysModel.T_test - 1])
#
#                     if SysModel.T_test >= 2:
#                         x_out_cv_list[SysModel.T_test - 2] = self.model(None,
#                                                                          x_out_cv_forward[:, SysModel.T_test - 2],
#                                                                          x_out_cv_forward[:, SysModel.T_test - 1],
#                                                                          None)
#                     for t in range(SysModel.T_test - 3, -1, -1):
#                         x_out_cv_list[t] = self.model(None,
#                                                       x_out_cv_forward[:, t],
#                                                       x_out_cv_forward[:, t + 1],
#                                                       x_out_cv_list[t + 2])
#
#                     x_out_cv = torch.stack(x_out_cv_list, dim=1)  # [m, T_test]
#
#                     # =========================================================
#                     # CV LOSS: weighted next-day prediction (per t)
#                     # FIXED: Use SMOOTHED states (x_out_cv) just like training!
#                     # PLUS extra prediction y_{T+1} from x_T with weight=2
#                     # =========================================================
#                     HF = SysModel.H @ SysModel.F
#
#                     weights = torch.arange(1, SysModel.T_test + 1, device=y_cv_norm.device, dtype=y_cv_norm.dtype)
#                     weights = weights / torch.sum(weights)
#
#                     cv_loss = 0
#                     for t in range(0, SysModel.T_test):
#                         y_pred_next_t = HF @ x_out_cv[:, t]  # FIXED: Use smoothed states
#                         y_true_next_t = y_next_day_cv_norm[:, t]
#                         cv_loss = cv_loss + weights[t] * self.loss_fn(y_pred_next_t, y_true_next_t)
#
#                     # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                     x_last = x_out_cv[:, -1]
#                     y_pred_Tp1 = HF @ x_last
#                     y_true_Tp1 = y_next_day_cv_norm[:, -1]
#                     loss_last = self.loss_fn(y_pred_Tp1, y_true_Tp1)
#                     cv_loss = cv_loss + 2.0 * loss_last
#
#                     MSE_cv_linear_batch[j] = cv_loss.item()
#
#                 # Average CV
#                 self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
#                 self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
#
#                 if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
#                     self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
#                     self.MSE_cv_idx_opt = ti
#                     torch.save(self.model, path_results)
#
#             ########################
#             ### Training Summary ###
#             ########################
#             print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
#                   self.MSE_cv_dB_epoch[ti], "[dB]")
#
#             if (ti > 1):
#                 d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
#                 d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
#                 print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")
#
#             print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
#
#         return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch,
#                 self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]
#
#     def NNTest_stocks_last(self, SysModel, test_input, test_target, load_model_path,
#                            generate_f=False, generate_h=False, test_x0=None):
#
#         tp = torch.float32
#         print("Testing RTSNet (stocks – last step only, forward+backward)")
#
#         self.N_T = len(test_input)
#
#         # Load trained RTSNet
#         self.model = torch.load(load_model_path, weights_only=False).eval()
#
#         pred_prices = torch.empty(self.N_T, device=self.device, dtype=tp)
#         real_prices = torch.empty(self.N_T, device=self.device, dtype=tp)
#         sq_err_arr = torch.empty(self.N_T, device=self.device, dtype=tp)
#         rel_err_arr = torch.empty(self.N_T, device=self.device, dtype=tp)
#         rel_err_arr_abs = torch.empty(self.N_T, device=self.device, dtype=tp)
#
#         with torch.no_grad():
#             for j in range(0, self.N_T):
#
#                 # --------------------------------------------------
#                 # Window + target (target is y(t0+TAU))
#                 # --------------------------------------------------
#                 y_win = test_input[j]  # [n, TAU]
#                 y_true = test_target[j]  # scalar tensor (or [1])
#
#                 T = y_win.size(-1)
#                 SysModel.T_test = T
#
#                 # --------------------------------------------------
#                 # Per-window normalization (per feature row if n>1)
#                 # --------------------------------------------------
#                 y_mean = y_win.mean()
#                 y_std = y_win.std()
#                 if y_std == 0:
#                     y_std = torch.tensor(1.0, device=self.device)
#
#                 y_win_norm = (y_win - y_mean) / y_std
#
#                 # --------------------------------------------------
#                 # F / H selection (same logic as your codebase)
#                 # --------------------------------------------------
#                 if generate_f is True:
#                     index = j // 10
#                     SysModel.F = SysModel.F_test[index]
#                     self.model.update_F(SysModel.F)
#                 else:
#                     # Use the first (and only) F matrix when not varying F
#                     if isinstance(SysModel.F_test, list):
#                         SysModel.F = SysModel.F_test[0]
#                     else:
#                         SysModel.F = SysModel.F_test
#                     self.model.update_F(SysModel.F)
#
#                 if generate_h is True:
#                     index = j // 10
#                     SysModel.H = SysModel.H_test[index]
#                     self.model.update_H(SysModel.H)
#
#                 # --------------------------------------------------
#                 # x0: x0[0]=price normalized, x0[1]=trend scale-only (no mean shift)
#                 # --------------------------------------------------
#                 x0_raw = test_x0[j]  # [2] tensor
#                 x0_norm = torch.stack([
#                     (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                     x0_raw[1] / y_std               # trend: scale only (same units as price)
#                 ])
#                 SysModel.m1x_0 = x0_norm.view(SysModel.m, 1)  # [m, 1] = [2, 1]
#
#                 # --------------------------------------------------
#                 # Init sequence
#                 # --------------------------------------------------
#                 self.model.InitSequence(SysModel.m1x_0, T)
#                 self.model.init_hidden()
#
#                 # --------------------------------------------------
#                 # Forward pass
#                 # ASSUMPTION: T >= 2 always
#                 # --------------------------------------------------
#                 x_fwd_list = [self.model(y_win_norm[:, t], None, None, None) for t in range(T)]
#                 x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                 # --------------------------------------------------
#                 # Backward smoothing - ALWAYS smooth
#                 # --------------------------------------------------
#                 x_smooth_list = [None] * T
#                 x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                 self.model.InitBackward(x_smooth_list[T - 1])
#                 x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                 for t in range(T - 3, -1, -1):
#                     x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#
#                 x_smooth = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                 # --------------------------------------------------
#                 # LAST prediction ONLY: y_{T+1} from x_T (smoothed)
#                 # This is the ONLY prediction that matters for testing!
#                 # y_pred_{T+1} = H * F * x_smooth[:, T-1]
#                 # --------------------------------------------------
#                 x_last = x_smooth[:, T - 1].view(SysModel.m, 1)  # x_T
#
#                 # H[0:1,:] extracts the price row; denorm with price-row stats
#                 y_pred_norm = (SysModel.H @ (SysModel.F @ x_last))[0, 0]
#                 y_pred = y_pred_norm * y_std + y_mean
#
#                 # --------------------------------------------------
#                 # Metrics
#                 # --------------------------------------------------
#                 # make y_true scalar: take LAST element = z[t0+TAU] = next-day price after the window
#                 if y_true.numel() > 1:
#                     y_true_s = y_true.view(-1)[-1]
#                 else:
#                     y_true_s = y_true.view(())
#
#                 pred_prices[j] = y_pred
#                 real_prices[j] = y_true_s
#
#                 sq_err_arr[j] = (y_pred - y_true_s) ** 2
#                 rel_err_arr[j] = (y_pred - y_true_s) / y_true_s
#                 rel_err_arr_abs[j] = abs((y_pred - y_true_s) / y_true_s)
#
#         mse_price = torch.mean(sq_err_arr)
#         rel_err_mean = torch.mean(rel_err_arr_abs)
#
#         print("MSE(price):", mse_price.item())
#         print("Mean relative error:", rel_err_mean.item())
#
#         return (pred_prices, real_prices, mse_price, rel_err_mean, sq_err_arr, rel_err_arr)
#
#     # -------------------------
#
#     def train_emkalmannet_F_from_price(self,SysModel,cv_input, cv_target, cv_x0,train_input, train_target, train_x0,destination_path_M,destination_path_RTS,
#                                            num_em_iters=3,alpha=(0.05, 0.10, 0.85),lambda_F=1,generate_f=False,generate_h=False,use_smoothed=True,clip_grad=1.0,):
#         """
#         Train an M-step network to estimate/update F using a frozen RTSNet smoother and a price-domain loss.
#
#         Assumptions (consistent with your NNTrain_stocks):
#         - Each sample is a window y_win:      train_input[i]  shape [n, T]
#         - Each target is next-day aligned:    train_target[i] shape [n, T]
#             i.e., train_target[i][:, t] = y(t0 + t + 1)
#         - train_x0[i] is y(t0-1) (scalar), and x0 = [normalized_y(t0-1), 0.5]
#         - Per-window normalization: (y - mean)/std for both input and target.
#         - RTSNet is used ONLY to compute x_forward / x_smooth given current F.
#         - M-net predicts ΔF; we update F_current -> F_next and compute y_pred = H * F_next * x_state.
#         - Loss = weighted MSE over t plus regularization on ΔF, unrolled for num_em_iters with alpha weights.
#         """
#
#         device = self.device
#         dtype = train_input[0].dtype
#         m = SysModel.m
#         n = SysModel.n
#
#         self.N_E = len(train_input)
#         self.N_CV = len(cv_input)
#
#         # -------------------------
#         # Load & freeze RTSNet
#         # IMPORTANT: Keep in .train() mode for CuDNN RNN backward compatibility
#         self.model = torch.load(destination_path_RTS, map_location=device, weights_only=False).to(device).train()
#         for p in self.model.parameters():
#             p.requires_grad_(False)
#
#         batch_size = 10
#         # M-step model
#         model_mstep = self.M_model.train()
#
#         self.MSE_cv_dB_opt = 1e18
#
#         for epoch in range(self.N_steps):
#
#             # =========================
#             # TRAIN
#             # =========================
#             model_mstep.train()
#             train_loss_sum = 0.0
#             for j in range(self.N_B):
#                 self.M_optimizer.zero_grad()
#                 batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
#                 for _ in range(batch_size):
#                     # sample one window
#                     idx = random.randint(0, self.N_E - 1)
#                     y_win = train_input[idx].to(device)       # [n, T]
#                     y_next = train_target[idx].to(device)     # [n, T]
#                     T = int(y_win.size(-1))  # FIXED: ensure T is an int
#
#                     # per-window normalization (same as NNTrain_stocks)
#                     y_mean = y_win.mean()
#                     y_std = y_win.std()
#                     # FIXED: Use .item() for safe scalar comparison and ensure y_std is tensor on device
#                     if float(y_std.item()) < 1e-6:
#                         y_std = torch.tensor(1.0, device=device, dtype=dtype)
#
#                     y_win_n = (y_win - y_mean) / y_std
#                     y_next_n = (y_next - y_mean) / y_std
#
#                     # choose base F/H (usually fixed for stocks)
#                     if generate_f:
#                         f_index = idx // 10
#                         F_base = SysModel.F_train[f_index].to(device)
#                     else:
#                         F_base = SysModel.F_train[0].to(device) if isinstance(SysModel.F_train, list) else SysModel.F_train.to(device)
#
#                     if generate_h:
#                         h_index = idx // 10
#                         H = SysModel.H_train[h_index].to(device)
#                     else:
#                         H = SysModel.H_train[0].to(device) if isinstance(SysModel.H_train, list) else SysModel.H.to(device)
#
#                     # x0: x0[0]=price normalized, x0[1]=trend scale-only
#                     x0_raw = train_x0[idx]  # [2] tensor: [price, trend]
#                     x0_norm = torch.stack([
#                         (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                         x0_raw[1] / y_std               # trend: scale only (same units as price)
#                     ])
#                     SysModel.m1x_0 = x0_norm.view(m, 1).to(device)  # [2, 1]
#
#                     # init covariance prior for RTSNet (as in your code)
#                     if hasattr(SysModel, "m2x_0"):
#                         prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
#                     else:
#                         prior_Sigma = torch.eye(m, device=device, dtype=dtype)
#
#                     # M-step K times (num_em_iters iterations)
#                     # ASSUMPTION: T >= 2 always
#                     F_current = F_base.clone().detach()
#                     total_loss = torch.tensor(0.0, device=device, dtype=dtype)
#
#                     # Increasing weights over t: t=0 gets weight 1, t=T-1 gets weight T
#                     w = torch.arange(1, T + 1, device=device, dtype=dtype)
#                     w = w / (w.sum() + 1e-12)  # normalize
#
#                     for em_iter in range(num_em_iters):
#
#                         # --- E-step: smooth x using frozen RTSNet under F_current ---
#                         self.model.update_F(F_current)
#                         self.model.InitSequence(SysModel.m1x_0, T)
#                         self.model.init_hidden()
#                         self.model.prior_Sigma = prior_Sigma
#
#                         # Forward pass
#                         x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                         x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                         # Backward smoothing - ALWAYS smooth
#                         x_sm_list = [None] * T
#                         x_sm_list[T - 1] = x_fwd[:, T - 1]
#                         self.model.InitBackward(x_sm_list[T - 1])
#                         x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                         for t in range(T - 3, -1, -1):
#                             x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
#                         x_state = torch.stack(x_sm_list, dim=1)  # [m, T]
#
#                         # DO NOT DETACH - need gradients to flow through F to M-network!
#                         # x_state stays with gradients so M-network can learn
#
#                         nu = y_win_n - (H @ x_state)  # [n, T]
#
#                         # Compute M-step statistics
#                         A1 = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(1, T):
#                             A1 += x_state[:, t].view(m, 1) @ x_state[:, t-1].view(1, m)
#
#                         A2 = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(T-1):
#                             A2 += x_state[:, t].view(m, 1) @ x_state[:, t].view(1, m)
#
#                         S_delta_x = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(1, T):
#                             delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
#                             S_delta_x += delta_x.view(m, 1) @ delta_x.view(1, m)
#                         S_delta_x = S_delta_x / max(T-1, 1)
#
#                         S_nu = torch.zeros(n, n, device=device, dtype=dtype)
#                         for t in range(T):
#                             S_nu += nu[:, t].view(n, 1) @ nu[:, t].view(1, n)
#                         S_nu = S_nu / T
#
#                         C_delta_x_xminus = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(1, T):
#                             delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
#                             C_delta_x_xminus += delta_x.view(m, 1) @ x_state[:, t-1].view(1, m)
#                         C_delta_x_xminus = C_delta_x_xminus / max(T-1, 1)
#
#                         # Build feature vector
#                         feat = torch.cat([
#                             A1.reshape(-1), A2.reshape(-1),
#                             S_delta_x.reshape(-1), S_nu.reshape(-1),
#                             C_delta_x_xminus.reshape(-1), F_current.reshape(-1)
#                         ], dim=0).view(1, -1)  # [1, 5*m^2 + n^2]
#
#                         # predict ΔF, update
#                         dF = model_mstep(feat).view(m, m)
#                         F_next = F_current + dF
#
#                         # ===== Compute y_pred loss using F_NEXT (the updated F) =====
#                         # CRITICAL: Must use F_next so gradient flows through dF to M-network!
#                         HF_iter = H @ F_current  # [n, m]  ← Use UPDATED F, not old F_current!
#
#                         # Predict all y_{t+1} with increasing weights
#                         y_pred_iter_list = [(HF_iter @ x_state[:, t].view(m, 1)).view(-1) for t in range(T)]
#                         y_pred_iter = torch.stack(y_pred_iter_list, dim=1)  # [n, T]
#
#                         # Weighted MSE: sum(w[t] * mse_y_t)
#                         mse_t_iter = (y_pred_iter - y_next_n) ** 2  # [n, T]
#                         mse_time = mse_t_iter.mean(dim=0, keepdim=True)  # [1, T] average over n
#                         # loss_y_iter = (w.view(1, T) * mse_time).sum()  # scalar return_it
#
#                         # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                         x_last_iter = x_state[:, -1].view(m, 1)
#                         y_pred_Tp1_iter = (HF_iter @ x_last_iter).view(-1)
#                         y_true_Tp1 = y_next_n[:, -1]
#                         loss_y_Tp1_iter = torch.mean((y_pred_Tp1_iter - y_true_Tp1) ** 2)
#                         # loss_y_iter = loss_y_iter + 2.0 * loss_y_Tp1_iter  return_it
#                         loss_y_iter =   2.0 * loss_y_Tp1_iter
#                         # Regularize ΔF
#                         reg = lambda_F * torch.mean(dF ** 2)
#
#                         # alpha weighting: BOTH y_pred loss and reg for this EM iteration
#                         weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
#                         total_loss = total_loss + weight * (loss_y_iter + reg)
#
#                         # advance F for next EM iteration
#                         F_current = F_next
#
#                     # --- ONE FINAL RTS with final F for y_pred loss (HIGHEST ALPHA WEIGHT) ---
#                     self.model.update_F(F_current)
#                     self.model.InitSequence(SysModel.m1x_0, T)
#                     self.model.init_hidden()
#
#
#                     x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                     x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                     # Smooth - ALWAYS
#                     x_sm_list = [None] * T
#                     x_sm_list[T - 1] = x_fwd[:, T - 1]
#                     self.model.InitBackward(x_sm_list[T - 1])
#                     x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                     for t in range(T - 3, -1, -1):
#                         x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
#                     x_state_final = torch.stack(x_sm_list, dim=1)  # [m, T]
#
#                     # --- price prediction loss: y_hat(t+1|t) = H * F_current * x_state_final(:,t) ---
#                     HF_final = H @ F_current  # [n, m]
#                     # Use list comprehension to preserve gradients
#                     y_pred_list = [(HF_final @ x_state_final[:, t].view(m, 1)).view(-1) for t in range(T)]
#                     y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]
#
#                     # weighted MSE over t (increasing weights) - per-element then weighted
#                     mse_t = (y_pred - y_next_n) ** 2  # [n, T]
#                     mse_time = mse_t.mean(dim=0, keepdim=True)
#                     loss_y = (w.view(1, T) * mse_time).sum()
#                     # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                     x_last = x_state_final[:, -1].view(m, 1)
#                     y_pred_Tp1 = (HF_final @ x_last).view(-1)
#                     y_true_Tp1 = y_next_n[:, -1]
#                     loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
#                     # loss_y = loss_y + 2.0 * loss_y_Tp1 return_it
#                     loss_y =  2.0 * loss_y_Tp1
#                     # Add y_pred loss with HIGHEST alpha weight (alpha[num_em_iters])
#                     final_weight = alpha[-1]
#                     total_loss = total_loss + final_weight * loss_y
#
#                     # Accumulate batch loss
#                     batch_loss = batch_loss + total_loss
#                 # FIXED: Single backward call per optimizer step with defensive try/except
#                 loss = batch_loss / float(batch_size)
#                 try:
#                     loss.backward()
#                 except Exception as e:
#                     print(f"Warning: backward failed at epoch {epoch} with error: {e}; skipping this batch")
#                     self.M_optimizer.zero_grad()
#                     continue
#                 if clip_grad is not None and clip_grad > 0:
#                     torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=float(clip_grad))
#                 self.M_optimizer.step()
#
#                 train_loss_sum += loss.detach().item()
#
#             # =========================
#             # VALIDATION
#             # =========================
#             model_mstep.eval()
#             cv_loss_sum = 0.0
#
#             with torch.no_grad():
#                 for j in range(self.N_CV):
#                     y_win = cv_input[j].to(device)
#                     y_next = cv_target[j].to(device)
#                     T = int(y_win.size(-1))  # FIXED: ensure T is an int
#
#                     y_mean = y_win.mean()
#                     y_std = y_win.std()
#                     # FIXED: Safe scalar comparison
#                     if float(y_std.item()) < 1e-6:
#                         y_std = torch.tensor(1.0, device=device, dtype=dtype)
#
#                     y_win_n = (y_win - y_mean) / y_std
#                     y_next_n = (y_next - y_mean) / y_std
#
#                     if generate_f:
#                         f_index = j // 10
#                         F_base = SysModel.F_valid[f_index].to(device)
#                     else:
#                         F_base = SysModel.F_valid[0].to(device) if isinstance(SysModel.F_valid, list) else SysModel.F_valid.to(device)
#
#                     if generate_h:
#                         h_index = j // 10
#                         H = SysModel.H_valid[h_index].to(device)
#                     else:
#                         H = SysModel.H_valid[0].to(device) if isinstance(SysModel.H_valid, list) else SysModel.H.to(device)
#
#                     x0_raw = cv_x0[j]  # [2] tensor: [price, trend]
#                     x0_norm_cv = torch.stack([
#                         (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                         x0_raw[1] / y_std               # trend: scale only (same units as price)
#                     ])
#                     SysModel.m1x_0 = x0_norm_cv.view(m, 1).to(device)  # [2, 1]
#
#                     if hasattr(SysModel, "m2x_0"):
#                         prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
#                     else:
#                         prior_Sigma = torch.eye(m, device=device, dtype=dtype)
#
#                     w = torch.arange(1, T + 1, device=device, dtype=dtype)
#                     w = w / (w.sum() + 1e-12)
#
#                     F_current = F_base.clone().detach()
#                     # FIXED: Use tensor accumulator on device
#                     total_loss = torch.tensor(0.0, device=device, dtype=dtype)
#
#                     # M-step runs num_em_iters times
#                     for em_iter in range(num_em_iters):
#                         self.model.update_F(F_current)
#                         self.model.InitSequence(SysModel.m1x_0, T)
#                         self.model.init_hidden()
#                         self.model.prior_Sigma = prior_Sigma
#
#                         # Forward pass
#                         x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                         x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                         # Backward smoothing - ALWAYS smooth
#                         x_sm_list = [None] * T
#                         x_sm_list[T - 1] = x_fwd[:, T - 1]
#                         self.model.InitBackward(x_sm_list[T - 1])
#                         x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                         for t in range(T - 3, -1, -1):
#                             x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
#                         x_state = torch.stack(x_sm_list, dim=1)  # [m, T]
#
#                         nu = y_win_n - (H @ x_state)
#
#                         # Compute M-step statistics
#                         A1 = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(1, T):
#                             A1 += x_state[:, t].view(m, 1) @ x_state[:, t-1].view(1, m)
#
#                         A2 = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(T-1):
#                             A2 += x_state[:, t].view(m, 1) @ x_state[:, t].view(1, m)
#
#                         S_delta_x = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(1, T):
#                             delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
#                             S_delta_x += delta_x.view(m, 1) @ delta_x.view(1, m)
#                         S_delta_x = S_delta_x / max(T-1, 1)
#
#                         S_nu = torch.zeros(n, n, device=device, dtype=dtype)
#                         for t in range(T):
#                             S_nu += nu[:, t].view(n, 1) @ nu[:, t].view(1, n)
#                         S_nu = S_nu / T
#
#                         C_delta_x_xminus = torch.zeros(m, m, device=device, dtype=dtype)
#                         for t in range(1, T):
#                             delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
#                             C_delta_x_xminus += delta_x.view(m, 1) @ x_state[:, t-1].view(1, m)
#                         C_delta_x_xminus = C_delta_x_xminus / max(T-1, 1)
#
#                         feat = torch.cat([
#                             A1.reshape(-1), A2.reshape(-1),
#                             S_delta_x.reshape(-1), S_nu.reshape(-1),
#                             C_delta_x_xminus.reshape(-1), F_current.reshape(-1),
#                         ], dim=0).view(1, -1)
#
#                         dF = model_mstep(feat).view(m, m)
#                         F_next = F_current + dF
#
#                         # ===== Compute y_pred loss using F_NEXT (same fix as training) =====
#                         HF_iter = H @ F_current  # Use UPDATED F!
#                         y_pred_iter_list = [(HF_iter @ x_state[:, t].view(m, 1)).view(-1) for t in range(T)]
#                         y_pred_iter = torch.stack(y_pred_iter_list, dim=1)  # [n, T]
#
#                         # Weighted MSE: sum(w[t] * mse_y_t)
#                         mse_t_iter = (y_pred_iter - y_next_n) ** 2
#                         mse_time = mse_t_iter.mean(dim=0, keepdim=True)
#                         loss_y_iter = (w.view(1, T) * mse_time).sum()
#
#                         # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                         x_last_iter = x_state[:, -1].view(m, 1)
#                         y_pred_Tp1_iter = (HF_iter @ x_last_iter).view(-1)
#                         y_true_Tp1 = y_next_n[:, -1]
#                         loss_y_Tp1_iter = torch.mean((y_pred_Tp1_iter - y_true_Tp1) ** 2)
#                         # loss_y_iter = loss_y_iter + 2.0 * loss_y_Tp1_iter return_it
#                         loss_y_iter =2.0 * loss_y_Tp1_iter
#                         # Regularize ΔF
#                         reg = lambda_F * torch.mean(dF ** 2)
#
#                         # alpha weighting (same as training)
#                         weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
#                         total_loss = total_loss + weight * (loss_y_iter + reg)
#
#                         F_current = F_next
#
#                     # ONE FINAL RTS with final F for prediction
#                     self.model.update_F(F_current)
#                     self.model.InitSequence(SysModel.m1x_0, T)
#                     self.model.init_hidden()
#                     self.model.prior_Sigma = prior_Sigma
#
#                     x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                     x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                     # Smooth - ALWAYS
#                     x_sm_list = [None] * T
#                     x_sm_list[T - 1] = x_fwd[:, T - 1]
#                     self.model.InitBackward(x_sm_list[T - 1])
#                     x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                     for t in range(T - 3, -1, -1):
#                         x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
#                     x_state_final = torch.stack(x_sm_list, dim=1)  # [m, T]
#
#                     # Prediction with final F
#                     HF_final = H @ F_current
#                     # FIXED: Use list comprehension
#                     y_pred_list = [(HF_final @ x_state_final[:, t].view(m, 1)).view(-1) for t in range(T)]
#                     y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]
#
#                     # weighted MSE over t (increasing weights) - per-element then weighted
#                     mse_t = (y_pred - y_next_n) ** 2  # [n, T]
#                     mse_time = mse_t.mean(dim=0, keepdim=True)
#                     loss_y_final = (w.view(1, T) * mse_time).sum()
#                     x_last = x_state_final[:, -1].view(m, 1)
#                     y_pred_Tp1 = (HF_final @ x_last).view(-1)
#                     y_true_Tp1 = y_next_n[:, -1]
#                     loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
#                     # loss_y_final = loss_y_final + 2.0 * loss_y_Tp1 return_it
#                     loss_y_final = 2.0 * loss_y_Tp1
#
#                     # Add with HIGHEST alpha weight (same as training)
#                     final_weight = alpha[-1]
#                     total_loss = total_loss + final_weight * loss_y_final
#
#                     cv_loss_sum += total_loss.item()
#
#             train_epoch = train_loss_sum / max(1, self.N_B)
#             cv_epoch = cv_loss_sum / max(1, self.N_CV)
#
#             # FIXED: Safe comparison with float
#             if float(cv_epoch) < float(self.MSE_cv_dB_opt):
#                 self.MSE_cv_dB_opt = float(cv_epoch)
#                 torch.save(model_mstep, destination_path_M)
#
#             print(f"[F-MNet via RTSNet] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")
#
#
#     def test_mstep_net_price(self,
#             SysModel,
#             test_input,  # list: each [n, T]
#             test_target,  # list: each [n, T]  (next-day aligned, like your NNTrain_stocks)
#             test_x0,  # list: scalar y(t0-1) per window (like your stocks pipeline)
#             destination_path_RTS,
#             destination_path_M,
#             num_em_iters=3,
#             generate_f=False,
#             generate_h=False
#     ):
#         """
#         Test M-step network for STOCK PRICE prediction.
#         - Load frozen RTSNet from destination_path_RTS.
#         - Load trained M-step net from destination_path_M.
#         - For each test window:
#             * normalize window once (mean/std of input window)
#             * build x0 = [normalized y(t0-1), 0.5]
#             * unroll num_em_iters:
#                 - smooth x with current F (frozen RTSNet)
#                 - build z_in features
#                 - predict ΔF, update F
#             * after final F, compute y_pred(t+1|t) = H * F_final * x_state(:,t)
#             * compute MSE vs test_target (normalized)
#         Returns:
#           mean_price_mse_per_iter (tensor [num_em_iters])   # how price error evolves across EM iters
#           mean_price_mse_db_per_iter (tensor [num_em_iters])
#           final_F_list (list of [m,m])
#           (optional) predictions (list of dicts)
#         """
#
#         device = self.device
#         m = SysModel.m
#         n = SysModel.n
#         N_T = len(test_input)
#
#         # --- Load and freeze RTSNet ---
#         self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
#         for p in self.model.parameters():
#             p.requires_grad_(False)
#
#         # --- Load M-step net ---
#         model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()
#
#         # Track mean price MSE per EM iteration
#         price_mse_sum_per_iter = torch.zeros(num_em_iters, device=device)
#
#         final_F_list = []
#         preds_out = []
#
#         with torch.no_grad():
#             for j in range(N_T):
#
#                 y_win = test_input[j]  # [n, T]  (assumed already on device)
#                 y_next = test_target[j]  # [n, T]  (next-day aligned)
#                 T = y_win.size(-1)
#
#                 # Choose base F and H
#                 if generate_f:
#                     f_index = j // 10
#                     F_current = SysModel.F_test[f_index].clone()
#                 else:
#                     # common in stocks: one global base F
#                     F_current = SysModel.F_test[0].clone() if isinstance(SysModel.F_test,
#                                                                          list) else SysModel.F_test.clone()
#
#                 if generate_h:
#                     h_index = j // 10
#                     H = SysModel.H_test[h_index].clone()
#                     SysModel.H = H
#                     self.model.update_H(H)
#                 else:
#                     H = SysModel.H.clone()
#
#                 # ---- Normalize ONCE per window ----
#                 y_mean = y_win.mean()
#                 y_std = y_win.std()
#                 if y_std < 1e-6:
#                     y_std = torch.tensor(1.0, device=device, dtype=y_win.dtype)
#
#                 y_win_n = (y_win - y_mean) / y_std
#                 y_next_n = (y_next - y_mean) / y_std
#
#                 # # OLD: split normalization (price normalized, trend kept as 0.5)
#                 # # x0_raw = test_x0[j]  # [2] tensor: [price_before_window, 0.5]
#                 # # x0_norm = torch.zeros_like(x0_raw)
#                 # # x0_norm[0] = (x0_raw[0] - y_mean) / y_std
#                 # # x0_norm[1] = x0_raw[1]  # Keep 0.5 as-is
#                 # # x0 = x0_norm.view(m, 1)
#                 # # OLD: normalize both components
#                 # # x0_norm = (x0_raw - y_mean) / y_std
#                 # # x0 = x0_norm.view(m, 1)
#
#                 # NEW: x0[0]=price normalized, x0[1]=trend scale-only (no mean shift)
#                 x0_raw = test_x0[j]  # [2] tensor: [price_before_window, trend0]
#                 x0_norm = torch.stack([
#                     (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                     x0_raw[1] / y_std               # trend: scale only (same units as price)
#                 ])
#                 x0 = x0_norm.view(m, 1)  # [m, 1] = [2, 1]
#
#
#                 # prior covariance (same style as your code)
#                 P0 = SysModel.m2x_0.clone().detach()
#
#                 # ========= M-step K times (num_em_iters iterations) =========
#                 # ASSUMPTION: T >= 2 always
#                 for em_iter in range(num_em_iters):
#
#                     # --- E-step: RTSNet smoothing under current F ---
#                     self.model.update_F(F_current)
#                     self.model.InitSequence(x0.clone().detach(), T)
#                     self.model.init_hidden()
#                     self.model.prior_Sigma = P0.clone().detach()
#
#                     # Forward pass
#                     x_forward_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                     x_forward = torch.stack(x_forward_list, dim=1)  # [m, T]
#
#                     # Backward smoothing - ALWAYS smooth
#                     x_smooth_list = [None] * T
#                     x_smooth_list[T - 1] = x_forward[:, T - 1]
#                     self.model.InitBackward(x_smooth_list[T - 1])
#                     x_smooth_list[T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
#                     for t in range(T - 3, -1, -1):
#                         x_smooth_list[t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth_list[t + 2])
#                     x_state = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                     # --- Build z_in features for M-step ---
#                     x_curr = x_state  # [m, T]
#                     x_prev = torch.cat([x0, x_curr[:, :-1]], dim=1)  # [m, T]
#
#                     A1 = (x_curr @ x_prev.T) / T
#                     A2 = (x_prev @ x_prev.T) / T
#
#                     x_minus = F_current @ x_prev
#                     delta_x = x_curr - x_minus
#
#                     delta_mean = delta_x.mean(dim=1, keepdim=True)
#                     delta_centered = delta_x - delta_mean
#                     S_delta_x = (delta_centered @ delta_centered.T) / T
#
#                     Hx_curr = H @ x_curr
#                     nu = y_win_n - Hx_curr
#
#                     nu_mean = nu.mean(dim=1, keepdim=True)
#                     nu_centered = nu - nu_mean
#                     S_nu = (nu_centered @ nu_centered.T) / T
#
#                     C_delta_x_xminus = (delta_x @ x_minus.T) / T
#
#                     z_in = torch.cat([
#                         A1.reshape(-1),
#                         A2.reshape(-1),
#                         S_delta_x.reshape(-1),
#                         S_nu.reshape(-1),
#                         C_delta_x_xminus.reshape(-1),
#                         F_current.reshape(-1),
#                     ], dim=0).view(1, -1)
#
#                     # --- M-step: predict ΔF and update ---
#                     dF = model_mstep(z_in).view(m, m)
#                     F_next = F_current + dF
#
#                     # Update F for next EM iteration
#                     F_current = F_next
#
#                 # ========= ONE FINAL RTS pass with final F for prediction =========
#                 self.model.update_F(F_current)
#                 self.model.InitSequence(x0.clone().detach(), T)
#                 self.model.init_hidden()
#                 self.model.prior_Sigma = P0.clone().detach()
#
#                 # Forward
#                 x_forward_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                 x_forward = torch.stack(x_forward_list, dim=1)  # [m, T]
#
#                 # Smooth - ALWAYS
#                 x_smooth_list = [None] * T
#                 x_smooth_list[T - 1] = x_forward[:, T - 1]
#                 self.model.InitBackward(x_smooth_list[T - 1])
#                 x_smooth_list[T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
#                 for t in range(T - 3, -1, -1):
#                     x_smooth_list[t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth_list[t + 2])
#                 x_state_final = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                 # --- ONLY predict y_{T+1} from x_T (the last smoothed state) ---
#                 HF_final = H @ F_current  # [n, m]
#                 x_last = x_state_final[:, -1].view(m, 1)  # x_T
#                 y_pred_Tp1 = (HF_final @ x_last).view(-1)  # predict y_{T+1}
#                 y_true_Tp1 = y_next_n[:, -1]  # y_{T+1} (last element of target)
#
#                 # MSE on ONLY this prediction (the only one that matters!)
#                 mse_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
#
#                 # Store this as the final MSE for this sequence
#
#                 seq_price_mse_per_iter = mse_Tp1  # Only last slot matters
#
#                 # accumulate mean over sequences
#                 price_mse_sum_per_iter += seq_price_mse_per_iter
#                 final_F_list.append(F_current.detach().clone())
#
#
#                 # optionally return denormalized predictions from FINAL iteration
#                 # y_pred_Tp1 is normalized, denormalize for output
#                 preds_out.append({
#                     "seq_index": j,
#                     "y_mean": y_mean.detach().cpu(),
#                     "y_std": y_std.detach().cpu(),
#                     # Predictions for y_{T+1} (ONLY)
#                     "y_pred_Tp1_norm": y_pred_Tp1.detach().cpu(),
#                     "y_true_Tp1_norm": y_true_Tp1.detach().cpu(),
#                     "y_pred_Tp1": (y_pred_Tp1 * y_std + y_mean).detach().cpu(),
#                     "y_true_Tp1": (y_true_Tp1 * y_std + y_mean).detach().cpu(),
#                 })
#
#         mean_price_mse_per_iter = price_mse_sum_per_iter / float(N_T)
#         mean_price_mse_db_per_iter = 10.0 * torch.log10(mean_price_mse_per_iter + 1e-12)
#
#         print("[M-step PRICE TEST] Mean price MSE per EM iteration:")
#         for k in range(num_em_iters):
#             print(f"  EM iter {k + 1}: mse={mean_price_mse_per_iter[k].item():.6e}  "
#                   f"({mean_price_mse_db_per_iter[k].item():.2f} dB)")
#
#
#         return mean_price_mse_per_iter, mean_price_mse_db_per_iter, final_F_list, preds_out
#
#
# def train_joint_rtsnet_and_mnet_em2_batch5(
#     self,
#     SysModel,
#     train_input, train_target, train_x0,
#     cv_input, cv_target, cv_x0,
#     path_rts_in,
#     path_m_in,
#     path_rts_out,
#     path_m_out,
#     batch_size=5,
#     num_em_iters=2,
#     lambda_F=1e-1,
#     clip_grad=1.0,
#     alpha=[0.05,0.1,0.85],
#     lr_rts=1e-4,
#     lr_m=1e-4,
#     wd_rts=1e-5,
#     wd_m=1e-5,
# ):
#     """
#     Joint end-to-end fine-tuning of ONE RTSNet + ONE M-step net with:
#       - EM unroll length = 2
#       - Batch size = 5 (one backward per 5 sequences)
#       - price prediction loss: y_hat = H * F * x_t  (sequence target) OR y_hat_last (last target)
#
#     Data format supported:
#       - train_input[i]  : [n, T]
#       - train_target[i] : [n, T]  (next-day aligned)  OR  [n] / [n,1] (last only)
#       - train_x0[i]     : scalar y(t0-1)
#
#     Saves best models (based on CV subset MSE):
#       - RTSNet -> path_rts_out
#       - MNet   -> path_m_out
#     """
#
#     device = self.device
#     m, n = SysModel.m, SysModel.n
#     N_E = len(train_input)
#     N_CV = len(cv_input)
#     dtype = train_input[0].dtype
#
#     # Initialize Pipeline attributes (required by validation loop)
#     self.N_E = N_E
#     self.N_CV = N_CV
#
#     # ---------- Load models ONCE ----------
#     self.model = torch.load(path_rts_in, weights_only=False).to(device).train()
#     model_mstep = torch.load(path_m_in, weights_only=False).to(device).train()
#
#     # ---------- Optimizers ----------
#     rts_opt = torch.optim.Adam(self.model.parameters(), lr=lr_rts, weight_decay=wd_rts)
#     m_opt   = torch.optim.Adam(model_mstep.parameters(), lr=lr_m,   weight_decay=wd_m)
#
#     best_cv = float("inf")
#
#     # EPOCH LOOP (like train_emkalmannet_F_from_price)
#     for epoch in range(self.N_steps):
#
#         # ==========================
#         # TRAIN: self.N_B mini-batch updates per epoch
#         # ==========================
#         self.model.train()
#         model_mstep.train()
#         train_loss_sum = 0.0
#
#         for j in range(self.N_B):  # N_B updates per epoch
#             rts_opt.zero_grad()
#             m_opt.zero_grad()
#
#             batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
#
#             for b in range(batch_size):  # batch_size sequences per update
#
#                 idx = random.randint(0, N_E - 1)
#
#                 y_win = train_input[idx]      # [n, T]
#                 y_tgt = train_target[idx]     # [n, T] or [n]
#                 T = y_win.size(-1)
#
#                 # ---- normalize ONCE per window ----
#                 y_mean = y_win.mean()
#                 y_std  = y_win.std()
#                 if y_std < 1e-6:
#                     y_std = torch.tensor(1.0, device=device, dtype=dtype)
#
#                 y_win_n = (y_win - y_mean) / y_std
#
#                 if y_tgt.dim() == 2:
#                     y_tgt_n = (y_tgt - y_mean) / y_std
#                     target_is_sequence = True
#                 else:
#                     y_tgt_n = (y_tgt.view(-1) - y_mean.view(-1)) / y_std
#                     target_is_sequence = False
#
#                 # ---- base F, H (stocks: usually fixed) ----
#                 # assume SysModel.F and SysModel.H exist on the correct device
#                 F_base = SysModel.F.clone().detach()
#                 H = SysModel.H
#
#                 # x0: normalize both components so x0 lives in same space as RTSNet states
#                 x0_raw  = train_x0[idx]  # [2] tensor
#                 x0_norm = torch.stack([
#                     (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                     x0_raw[1] / y_std               # trend: scale only (same units as price)
#                 ])
#                 x0 = x0_norm.view(m, 1)  # [m, 1] = [2, 1]
#
#                 P0 = SysModel.m2x_0.clone().detach()
#
#                 # ==================================
#                 # M-step K times (num_em_iters iterations)
#                 # ASSUMPTION: T >= 2 always
#                 # ==================================
#                 F_current = F_base.detach()  # Detached at initialization (from SysModel.F.clone().detach())
#                 total_loss = torch.tensor(0.0, device=device, dtype=dtype)
#
#                 # Increasing weights over t
#                 w = torch.arange(1, T + 1, device=device, dtype=dtype)
#                 w = w / (w.sum() + 1e-12)
#
#                 for em_iter in range(num_em_iters):
#
#                     # ---- E-step: RTSNet smoother under current F ----
#                     self.model.update_F(F_current)
#                     self.model.InitSequence(x0, T)
#                     self.model.init_hidden()
#                     self.model.prior_Sigma = P0
#
#                     # Forward pass
#                     x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                     x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                     # Backward smoothing - ALWAYS smooth
#                     x_smooth_list = [None] * T
#                     x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                     self.model.InitBackward(x_smooth_list[T - 1])
#                     x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                     for t in range(T - 3, -1, -1):
#                         x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                     x_state = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                     # ---- build stats for M-net input ----
#                     x_curr = x_state
#                     # x0 fully normalized (trend/y_std) so same space as x_curr — correct x_{-1}
#                     x_prev = torch.cat([x0, x_curr[:, :-1]], dim=1)  # [m, T]
#
#                     A1 = (x_curr @ x_prev.T) / T
#                     A2 = (x_prev @ x_prev.T) / T
#
#                     x_minus = F_current @ x_prev
#                     delta_x = x_curr - x_minus
#                     delta_centered = delta_x - delta_x.mean(dim=1, keepdim=True)
#                     S_delta_x = (delta_centered @ delta_centered.T) / T
#
#                     nu = y_win_n - (H @ x_curr)
#                     nu_centered = nu - nu.mean(dim=1, keepdim=True)
#                     S_nu = (nu_centered @ nu_centered.T) / T
#
#                     C_delta_x_xminus = (delta_x @ x_minus.T) / T
#
#                     z_in = torch.cat([
#                         A1.reshape(-1),
#                         A2.reshape(-1),
#                         S_delta_x.reshape(-1),
#                         S_nu.reshape(-1),
#                         C_delta_x_xminus.reshape(-1),
#                         F_current.reshape(-1),
#                     ], dim=0).view(1, -1)
#
#                     # ---- M-step: ΔF and update F (gradients flow to MNet) ----
#                     dF = model_mstep(z_in).view(m, m)
#                     F_next = F_current + dF
#
#                     # Use F_current for prediction (x_state was smoothed under F_current)
#                     HF_iter = H @ F_current  # [n, m]
#
#                     # Predict all y_{t+1} with increasing weights
#                     y_pred_iter_list = [(HF_iter @ x_state[:, t].view(m, 1)).view(-1) for t in range(T)]
#                     y_pred_iter = torch.stack(y_pred_iter_list, dim=1)  # [n, T]
#
#                     # Weighted MSE over full sequence
#                     mse_t_iter = (y_pred_iter - y_tgt_n) ** 2  # [n, T]
#                     mse_time = mse_t_iter.mean(dim=0, keepdim=True)  # [1, T]
#                     loss_y_iter = (w.view(1, T) * mse_time).sum()  # scalar
#
#                     # y_{T+1} prediction from x_T with DOUBLE weight
#                     x_last_iter = x_state[:, -1].view(m, 1)
#                     y_pred_Tp1_iter = (HF_iter @ x_last_iter).view(-1)
#                     y_true_Tp1 = y_tgt_n[:, -1]
#                     loss_y_Tp1_iter = torch.mean((y_pred_Tp1_iter - y_true_Tp1) ** 2)
#                     loss_y_iter = loss_y_iter + 2.0 * loss_y_Tp1_iter  # full sequence + double last
#
#                     # Regularize ΔF
#                     reg_iter = lambda_F * torch.mean(dF ** 2)
#
#                     # alpha weighting
#                     weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
#                     total_loss = total_loss + weight * (loss_y_iter + reg_iter)
#
#                     # advance F (NO detach - gradient flows to both RTSNet and M-net!)
#                     F_current = F_next
#
#                 # ==================================
#                 # ONE FINAL RTS with final F for y_pred loss (HIGHEST ALPHA WEIGHT)
#                 # ==================================
#                 self.model.update_F(F_current)
#                 self.model.InitSequence(x0, T)
#                 self.model.init_hidden()
#                 self.model.prior_Sigma = P0
#
#                 x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                 x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                 # Smooth - ALWAYS
#                 x_smooth_list = [None] * T
#                 x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                 self.model.InitBackward(x_smooth_list[T - 1])
#                 x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                 for t in range(T - 3, -1, -1):
#                     x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                 x_state_last = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                 # NO detach - need gradients to flow through F to both nets!
#                 # ==========================
#                 HF = H @ F_current  # [n, m]
#
#                 # Predict y_{t+1} from x_t for all t
#                 y_pred_list = [(HF @ x_state_last[:, t].view(m, 1)).view(-1) for t in range(T)]
#                 y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]
#
#                 # Weighted MSE: sum(w[t] * mse_y_t)
#                 mse_t = (y_pred - y_tgt_n) ** 2  # [n, T]
#                 mse_time = mse_t.mean(dim=0, keepdim=True)  # [1,T]
#                 loss_y = (w.view(1, T) * mse_time).sum()
#
#                 # EXTRA: y_{T+1} from x_T with DOUBLE weight
#                 x_last = x_state_last[:, -1].view(m, 1)  # x_T
#                 y_pred_Tp1 = (HF @ x_last).view(-1)
#                 y_true_Tp1 = y_tgt_n[:, -1]
#                 loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
#                 loss_y = loss_y + 2.0 * loss_y_Tp1  # full sequence + double last
#                 final_weight = alpha[-1]
#                 total_loss = total_loss + final_weight * loss_y
#
#                 batch_loss = batch_loss + total_loss / float(batch_size)
#
#             # ---- One backward for BOTH nets (end of mini-batch) ----
#             batch_loss.backward()
#
#             if clip_grad and clip_grad > 0:
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(clip_grad))
#                 torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=float(clip_grad))
#
#             rts_opt.step()
#             m_opt.step()
#
#             train_loss_sum += batch_loss.detach().item()
#
#         # ==========================
#         # VALIDATION (end of epoch)
#         # ==========================
#         self.model.eval()
#         model_mstep.eval()
#
#         cv_loss = 0.0
#         with torch.no_grad():
#             for j in range(self.N_CV):
#                 y_win = cv_input[j]
#                 y_tgt = cv_target[j]
#                 T = y_win.size(-1)
#
#                 y_mean = y_win.mean()
#                 y_std  = y_win.std()
#                 if y_std < 1e-6:
#                     y_std = torch.tensor(1.0, device=device, dtype=dtype)
#                 y_win_n = (y_win - y_mean) / y_std
#
#                 if y_tgt.dim() == 2:
#                     y_tgt_n = (y_tgt - y_mean) / y_std
#                     target_is_sequence = True
#                 else:
#                     y_tgt_n = (y_tgt.view(-1) - y_mean.view(-1)) / y_std
#                     target_is_sequence = False
#
#                 F_current = SysModel.F.clone().detach()
#                 H = SysModel.H
#
#                 x0_raw  = cv_x0[j]  # [2] tensor
#                 x0_norm_cv = torch.stack([
#                     (x0_raw[0] - y_mean) / y_std,   # price: full normalization
#                     x0_raw[1] / y_std               # trend: scale only (same units as price)
#                 ])
#                 x0 = x0_norm_cv.view(m, 1)  # [m, 1] = [2, 1]
#                 P0 = SysModel.m2x_0.clone().detach()
#
#                 # Compute time weights
#                 w_cv = torch.arange(1, T + 1, device=device, dtype=dtype)
#                 w_cv = w_cv / (w_cv.sum() + 1e-12)
#
#                 total_loss_cv = torch.tensor(0.0, device=device, dtype=dtype)
#
#                 # M-step K times (num_em_iters iterations)
#                 # ASSUMPTION: T >= 2 always
#                 for em_iter in range(num_em_iters):
#                     self.model.update_F(F_current)
#                     self.model.InitSequence(x0, T)
#                     self.model.init_hidden()
#                     self.model.prior_Sigma = P0
#
#                     # Forward pass
#                     x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                     x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                     # Backward smoothing - ALWAYS smooth
#                     x_smooth_list = [None] * T
#                     x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                     self.model.InitBackward(x_smooth_list[T - 1])
#                     x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                     for t in range(T - 3, -1, -1):
#                         x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                     x_state = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                     x_curr = x_state
#                     # x0 fully normalized (trend/y_std) so same space as x_curr — correct x_{-1}
#                     x_prev = torch.cat([x0, x_curr[:, :-1]], dim=1)  # [m, T]
#
#                     A1 = (x_curr @ x_prev.T) / T
#                     A2 = (x_prev @ x_prev.T) / T
#
#                     x_minus = F_current @ x_prev
#                     delta_x = x_curr - x_minus
#                     delta_centered = delta_x - delta_x.mean(dim=1, keepdim=True)
#                     S_delta_x = (delta_centered @ delta_centered.T) / T
#
#                     nu = y_win_n - (H @ x_curr)
#                     nu_centered = nu - nu.mean(dim=1, keepdim=True)
#                     S_nu = (nu_centered @ nu_centered.T) / T
#
#                     C_delta_x_xminus = (delta_x @ x_minus.T) / T
#
#                     z_in = torch.cat([
#                         A1.reshape(-1), A2.reshape(-1),
#                         S_delta_x.reshape(-1), S_nu.reshape(-1),
#                         C_delta_x_xminus.reshape(-1),
#                         F_current.reshape(-1),
#                     ], dim=0).view(1, -1)
#
#                     dF = model_mstep(z_in).view(m, m)
#                     F_next = F_current + dF
#
#                     # Use F_current for prediction (x_state was smoothed under F_current)
#                     HF_iter = H @ F_current
#                     y_pred_iter_list = [(HF_iter @ x_state[:, t].view(m, 1)).view(-1) for t in range(T)]
#                     y_pred_iter = torch.stack(y_pred_iter_list, dim=1)  # [n, T]
#
#                     # Weighted MSE over full sequence
#                     mse_t_iter = (y_pred_iter - y_tgt_n) ** 2
#                     mse_time = mse_t_iter.mean(dim=0, keepdim=True)
#                     loss_y_iter = (w_cv.view(1, T) * mse_time).sum()
#
#                     x_last_iter = x_state[:, -1].view(m, 1)
#                     y_pred_Tp1_iter = (HF_iter @ x_last_iter).view(-1)
#                     y_true_Tp1 = y_tgt_n[:, -1]
#                     loss_y_Tp1_iter = torch.mean((y_pred_Tp1_iter - y_true_Tp1) ** 2)
#                     loss_y_iter = loss_y_iter + 2.0 * loss_y_Tp1_iter  # full sequence + double last
#
#                     # Regularize ΔF
#                     reg_iter = lambda_F * torch.mean(dF ** 2)
#
#                     # alpha weighting (same as training)
#                     weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
#                     total_loss_cv = total_loss_cv + weight * (loss_y_iter + reg_iter)
#
#                     F_current = F_next
#
#                 # ONE FINAL RTS with final F for prediction (HIGHEST ALPHA WEIGHT)
#                 self.model.update_F(F_current)
#                 self.model.InitSequence(x0, T)
#                 self.model.init_hidden()
#                 self.model.prior_Sigma = P0
#
#                 x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                 x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                 # Smooth - ALWAYS
#                 x_smooth_list = [None] * T
#                 x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                 self.model.InitBackward(x_smooth_list[T - 1])
#                 x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                 for t in range(T - 3, -1, -1):
#                     x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                 x_state_last = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                 HF = H @ F_current
#
#                 # Predict all y_{t+1}
#                 y_pred_list = [(HF @ x_state_last[:, t].view(m, 1)).view(-1) for t in range(T)]
#                 y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]
#
#                 # Weighted MSE: sum(w[t] * mse_y_t)
#                 mse_t = (y_pred - y_tgt_n) ** 2
#                 mse_time = mse_t.mean(dim=0, keepdim=True)
#                 loss_y_final = (w_cv.view(1, T) * mse_time).sum()
#
#                 # EXTRA: y_{T+1} from x_T with DOUBLE weight
#                 x_last = x_state_last[:, -1].view(m, 1)
#                 y_pred_Tp1 = (HF @ x_last).view(-1)
#                 y_true_Tp1 = y_tgt_n[:, -1]
#                 loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
#                 loss_y_final = loss_y_final + 2.0 * loss_y_Tp1  # full sequence + double last
#
#                 # Add with HIGHEST alpha weight (same as training)
#                 final_weight = alpha[-1]
#                 total_loss_cv = total_loss_cv + final_weight * loss_y_final
#
#                 cv_loss += total_loss_cv.item()
#
#         # Normalize losses by their respective dataset sizes
#         cv_loss = cv_loss / float(self.N_CV)
#         train_epoch = train_loss_sum / float(self.N_B)
#
#         if cv_loss < best_cv:
#             best_cv = cv_loss
#             torch.save(self.model, path_rts_out)
#             torch.save(model_mstep, path_m_out)
#
#         print(f"[JOINT EM={num_em_iters} B={batch_size}] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_loss:.6f} best_cv={best_cv:.6f}")
#
#
#         ##########################only_f##############################################
#
#
# def y_train_joint_rtsnet_and_mnet_em2_batch5(
#         self,
#         SysModel,
#         train_input, train_target, train_x0,
#         cv_input, cv_target, cv_x0,
#         path_rts_in,
#         path_m_in,
#         path_rts_out,
#         path_m_out,
#         batch_size=5,
#         num_em_iters=2,
#         lambda_F=1e-1,
#         clip_grad=1.0,
#         alpha=[0.05, 0.1, 0.85],
#         lr_rts=1e-4,
#         lr_m=1e-4,
#         wd_rts=1e-5,
#         wd_m=1e-5,
# ):
#     """
#     Joint end-to-end fine-tuning of ONE RTSNet + ONE M-step net with:
#       - EM unroll length = 2
#       - Batch size = 5 (one backward per 5 sequences)
#       - price prediction loss: y_hat = H * F * x_t  (sequence target) OR y_hat_last (last target)
#
#     Data format supported:
#       - train_input[i]  : [n, T]
#       - train_target[i] : [n, T]  (next-day aligned)  OR  [n] / [n,1] (last only)
#       - train_x0[i]     : scalar y(t0-1)
#
#     Saves best models (based on CV subset MSE):
#       - RTSNet -> path_rts_out
#       - MNet   -> path_m_out
#     """
#
#     device = self.device
#     m, n = SysModel.m, SysModel.n
#     N_E = len(train_input)
#     N_CV = len(cv_input)
#     dtype = train_input[0].dtype
#
#     # Initialize Pipeline attributes (required by validation loop)
#     self.N_E = N_E
#     self.N_CV = N_CV
#
#     # ---------- Load models ONCE ----------
#     self.model = torch.load(path_rts_in, weights_only=False).to(device).train()
#     model_mstep = torch.load(path_m_in, weights_only=False).to(device).train()
#
#     # ---------- Optimizers ----------
#     rts_opt = torch.optim.Adam(self.model.parameters(), lr=lr_rts, weight_decay=wd_rts)
#     m_opt = torch.optim.Adam(model_mstep.parameters(), lr=lr_m, weight_decay=wd_m)
#
#     best_cv = float("inf")
#
#     # EPOCH LOOP (like train_emkalmannet_F_from_price)
#     for epoch in range(self.N_steps):
#
#         # ==========================
#         # TRAIN: self.N_B mini-batch updates per epoch
#         # ==========================
#         self.model.train()
#         model_mstep.train()
#         train_loss_sum = 0.0
#
#         for j in range(self.N_B):  # N_B updates per epoch
#             rts_opt.zero_grad()
#             m_opt.zero_grad()
#
#             batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
#
#             for b in range(batch_size):  # batch_size sequences per update
#
#                 idx = random.randint(0, N_E - 1)
#
#                 y_win = train_input[idx]  # [n, T]
#                 y_tgt = train_target[idx]  # [n, T] or [n]
#                 T = y_win.size(-1)
#
#                 # ---- normalize ONCE per window ----
#                 y_mean = y_win.mean()
#                 y_std = y_win.std()
#                 if y_std < 1e-6:
#                     y_std = torch.tensor(1.0, device=device, dtype=dtype)
#
#                 y_win_n = (y_win - y_mean) / y_std
#
#                 if y_tgt.dim() == 2:
#                     y_tgt_n = (y_tgt - y_mean) / y_std
#                     target_is_sequence = True
#                 else:
#                     y_tgt_n = (y_tgt.view(-1) - y_mean.view(-1)) / y_std
#                     target_is_sequence = False
#
#                 # ---- base F, H (stocks: usually fixed) ----
#                 # assume SysModel.F and SysModel.H exist on the correct device
#                 F_base = SysModel.F.clone().detach()
#                 H = SysModel.H
#
#                 # ---- x0 = [normalized y(t0-1), 0.5] ----
#                 x0_raw = float(train_x0[idx])
#                 x0_norm = (x0_raw - y_mean.item()) / y_std.item()
#
#                 x0 = torch.empty(m, device=device, dtype=dtype)
#                 x0[0] = torch.tensor(x0_norm, device=device, dtype=dtype)
#                 x0[1] = torch.tensor(0.5, device=device, dtype=dtype)
#                 x0 = x0.view(m, 1)
#
#                 P0 = SysModel.m2x_0.clone().detach()
#
#                 # ==================================
#                 # M-step K times (num_em_iters iterations)
#                 # ASSUMPTION: T >= 2 always
#                 # ==================================
#                 F_current = F_base.detach()  # Detached at initialization (from SysModel.F.clone().detach())
#                 total_loss = torch.tensor(0.0, device=device, dtype=dtype)
#
#                 # Increasing weights over t
#                 w = torch.arange(1, T + 1, device=device, dtype=dtype)
#                 w = w / (w.sum() + 1e-12)
#
#                 for em_iter in range(num_em_iters):
#
#                     # ---- E-step: RTSNet smoother under current F ----
#                     self.model.update_F(F_current)
#                     self.model.InitSequence(x0, T)
#                     self.model.init_hidden()
#                     self.model.prior_Sigma = P0
#
#                     # Forward pass
#                     x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                     x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                     # Backward smoothing - ALWAYS smooth
#                     x_smooth_list = [None] * T
#                     x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                     self.model.InitBackward(x_smooth_list[T - 1])
#                     x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                     for t in range(T - 3, -1, -1):
#                         x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                     x_state = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                     # ---- build stats for M-net input ----
#                     x_curr = x_state
#                     # x0 fully normalized (trend/y_std) — correct x_{-1} for EM statistics
#                     x_prev = torch.cat([x0, x_curr[:, :-1]], dim=1)  # [m, T]
#
#                     A1 = (x_curr @ x_prev.T) / T
#                     A2 = (x_prev @ x_prev.T) / T
#
#                     x_minus = F_current @ x_prev
#                     delta_x = x_curr - x_minus
#                     delta_centered = delta_x - delta_x.mean(dim=1, keepdim=True)
#                     S_delta_x = (delta_centered @ delta_centered.T) / T
#
#                     nu = y_win_n - (H @ x_curr)
#                     nu_centered = nu - nu.mean(dim=1, keepdim=True)
#                     S_nu = (nu_centered @ nu_centered.T) / T
#
#                     C_delta_x_xminus = (delta_x @ x_minus.T) / T
#
#                     z_in = torch.cat([
#                         A1.reshape(-1),
#                         A2.reshape(-1),
#                         S_delta_x.reshape(-1),
#                         S_nu.reshape(-1),
#                         C_delta_x_xminus.reshape(-1),
#                         F_current.reshape(-1),
#                     ], dim=0).view(1, -1)
#
#                     # ---- M-step: ΔF and update F (gradients flow to MNet) ----
#                     dF = model_mstep(z_in).view(m, m)
#                     F_next = F_current + dF
#
#                     # ===== Compute y_pred loss IN THIS EM iteration (NO detach on x_state!) =====
#                     HF_iter = H @ F_current  # [n, m]
#
#                     # Predict all y_{t+1} with increasing weights
#                     y_pred_iter_list = [(HF_iter @ x_state[:, t].view(m, 1)).view(-1) for t in range(T)]
#                     y_pred_iter = torch.stack(y_pred_iter_list, dim=1)  # [n, T]
#
#                     # Weighted MSE: sum(w[t] * mse_y_t)
#                     mse_t_iter = (y_pred_iter - y_tgt_n) ** 2  # [n, T]
#                     mse_time = mse_t_iter.mean(dim=0, keepdim=True)  # [1, T] average over n
#                     loss_y_iter = (w.view(1, T) * mse_time).sum()  # scalar
#
#                     # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                     x_last_iter = x_state[:, -1].view(m, 1)
#                     y_pred_Tp1_iter = (HF_iter @ x_last_iter).view(-1)
#                     y_true_Tp1 = y_tgt_n[:, -1]
#                     loss_y_Tp1_iter = torch.mean((y_pred_Tp1_iter - y_true_Tp1) ** 2)
#                     loss_y_iter = loss_y_iter + 2.0 * loss_y_Tp1_iter
#
#                     # Regularize ΔF
#                     reg_iter = lambda_F * torch.mean(dF ** 2)
#
#                     # alpha weighting: BOTH y_pred loss and reg for this EM iteration
#                     weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
#                     total_loss = total_loss + weight * (loss_y_iter + 4 * reg_iter)
#
#                     # advance F (NO detach - gradient flows to both RTSNet and M-net!)
#                     F_current = F_next
#
#                 # ==================================
#                 # ONE FINAL RTS with final F for y_pred loss (HIGHEST ALPHA WEIGHT)
#                 # ==================================
#                 self.model.update_F(F_current)
#                 self.model.InitSequence(x0, T)
#                 self.model.init_hidden()
#                 self.model.prior_Sigma = P0
#
#                 x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                 x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                 # Smooth - ALWAYS
#                 x_smooth_list = [None] * T
#                 x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                 self.model.InitBackward(x_smooth_list[T - 1])
#                 x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                 for t in range(T - 3, -1, -1):
#                     x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                 x_state_last = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                 # NO detach - need gradients to flow through F to both nets!
#                 # ==========================
#                 HF = H @ F_current  # [n, m]
#
#                 # Predict y_{t+1} from x_t for all t
#                 y_pred_list = [(HF @ x_state_last[:, t].view(m, 1)).view(-1) for t in range(T)]
#                 y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]
#
#                 # Weighted MSE: sum(w[t] * mse_y_t)
#                 mse_t = (y_pred - y_tgt_n) ** 2  # [n, T]
#                 mse_time = mse_t.mean(dim=0, keepdim=True)  # [1,T]
#                 loss_y = (w.view(1, T) * mse_time).sum()
#
#                 # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                 x_last = x_state_last[:, -1].view(m, 1)  # x_T
#                 y_pred_Tp1 = (HF @ x_last).view(-1)  # predict y_{T+1}
#                 y_true_Tp1 = y_tgt_n[:, -1]  # y_{T+1}
#                 loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
#                 loss_y = loss_y + 2.0 * loss_y_Tp1
#                 final_weight = alpha[-1]
#                 total_loss = total_loss + final_weight * loss_y
#
#                 batch_loss = batch_loss + total_loss / float(batch_size)
#
#             # ---- One backward for BOTH nets (end of mini-batch) ----
#             batch_loss.backward()
#
#             if clip_grad and clip_grad > 0:
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(clip_grad))
#                 torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=float(clip_grad))
#
#             rts_opt.step()
#             m_opt.step()
#
#             train_loss_sum += batch_loss.detach().item()
#
#         # ==========================
#         # VALIDATION (end of epoch)
#         # ==========================
#         self.model.eval()
#         model_mstep.eval()
#
#         cv_loss = 0.0
#         with torch.no_grad():
#             for j in range(self.N_CV):
#                 y_win = cv_input[j]
#                 y_tgt = cv_target[j]
#                 T = y_win.size(-1)
#
#                 y_mean = y_win.mean()
#                 y_std = y_win.std()
#                 if y_std < 1e-6:
#                     y_std = torch.tensor(1.0, device=device, dtype=dtype)
#                 y_win_n = (y_win - y_mean) / y_std
#
#                 if y_tgt.dim() == 2:
#                     y_tgt_n = (y_tgt - y_mean) / y_std
#                     target_is_sequence = True
#                 else:
#                     y_tgt_n = (y_tgt.view(-1) - y_mean.view(-1)) / y_std
#                     target_is_sequence = False
#
#                 F_current = SysModel.F.clone().detach()
#                 H = SysModel.H
#
#                 x0_raw = float(cv_x0[j])
#                 x0_norm = (x0_raw - y_mean.item()) / y_std.item()
#                 x0 = torch.tensor([[x0_norm], [0.5]], device=device, dtype=dtype)
#
#                 P0 = SysModel.m2x_0.clone().detach()
#
#                 # Compute time weights
#                 w_cv = torch.arange(1, T + 1, device=device, dtype=dtype)
#                 w_cv = w_cv / (w_cv.sum() + 1e-12)
#
#                 total_loss_cv = torch.tensor(0.0, device=device, dtype=dtype)
#
#                 # M-step K times (num_em_iters iterations)
#                 # ASSUMPTION: T >= 2 always
#                 for em_iter in range(num_em_iters):
#                     self.model.update_F(F_current)
#                     self.model.InitSequence(x0, T)
#                     self.model.init_hidden()
#                     self.model.prior_Sigma = P0
#
#                     # Forward pass
#                     x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                     x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                     # Backward smoothing - ALWAYS smooth
#                     x_smooth_list = [None] * T
#                     x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                     self.model.InitBackward(x_smooth_list[T - 1])
#                     x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                     for t in range(T - 3, -1, -1):
#                         x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                     x_state = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                     x_curr = x_state
#                     # x0 fully normalized (trend/y_std) — correct x_{-1} for EM statistics
#                     x_prev = torch.cat([x0, x_curr[:, :-1]], dim=1)  # [m, T]
#
#                     A1 = (x_curr @ x_prev.T) / T
#                     A2 = (x_prev @ x_prev.T) / T
#
#                     x_minus = F_current @ x_prev
#                     delta_x = x_curr - x_minus
#                     delta_centered = delta_x - delta_x.mean(dim=1, keepdim=True)
#                     S_delta_x = (delta_centered @ delta_centered.T) / T
#
#                     nu = y_win_n - (H @ x_curr)
#                     nu_centered = nu - nu.mean(dim=1, keepdim=True)
#                     S_nu = (nu_centered @ nu_centered.T) / T
#
#                     C_delta_x_xminus = (delta_x @ x_minus.T) / T
#
#                     z_in = torch.cat([
#                         A1.reshape(-1), A2.reshape(-1),
#                         S_delta_x.reshape(-1), S_nu.reshape(-1),
#                         C_delta_x_xminus.reshape(-1),
#                         F_current.reshape(-1),
#                     ], dim=0).view(1, -1)
#
#                     dF = model_mstep(z_in).view(m, m)
#                     F_next = F_current + dF
#
#                     # ===== Compute y_pred loss IN THIS EM iteration (same as training) =====
#                     HF_iter = H @ F_current
#                     y_pred_iter_list = [(HF_iter @ x_state[:, t].view(m, 1)).view(-1) for t in range(T)]
#                     y_pred_iter = torch.stack(y_pred_iter_list, dim=1)  # [n, T]
#
#                     # Weighted MSE: sum(w[t] * mse_y_t)
#                     mse_t_iter = (y_pred_iter - y_tgt_n) ** 2
#                     mse_time = mse_t_iter.mean(dim=0, keepdim=True)
#                     loss_y_iter = (w_cv.view(1, T) * mse_time).sum()
#
#                     x_last_iter = x_state[:, -1].view(m, 1)
#                     y_pred_Tp1_iter = (HF_iter @ x_last_iter).view(-1)
#                     y_true_Tp1 = y_tgt_n[:, -1]
#                     loss_y_Tp1_iter = torch.mean((y_pred_Tp1_iter - y_true_Tp1) ** 2)
#                     loss_y_iter = loss_y_iter + 2.0 * loss_y_Tp1_iter
#
#                     # Regularize ΔF
#                     reg_iter = lambda_F * torch.mean(dF ** 2)
#
#                     # alpha weighting (same as training)
#                     weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
#                     total_loss_cv = total_loss_cv + weight * (loss_y_iter + 4 * reg_iter)
#
#                     F_current = F_next
#
#                 # ONE FINAL RTS with final F for prediction (HIGHEST ALPHA WEIGHT)
#                 self.model.update_F(F_current)
#                 self.model.InitSequence(x0, T)
#                 self.model.init_hidden()
#                 self.model.prior_Sigma = P0
#
#                 x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
#                 x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]
#
#                 # Smooth - ALWAYS
#                 x_smooth_list = [None] * T
#                 x_smooth_list[T - 1] = x_fwd[:, T - 1]
#                 self.model.InitBackward(x_smooth_list[T - 1])
#                 x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
#                 for t in range(T - 3, -1, -1):
#                     x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])
#                 x_state_last = torch.stack(x_smooth_list, dim=1)  # [m, T]
#
#                 HF = H @ F_current
#
#                 # Predict all y_{t+1}
#                 y_pred_list = [(HF @ x_state_last[:, t].view(m, 1)).view(-1) for t in range(T)]
#                 y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]
#
#                 # Weighted MSE: sum(w[t] * mse_y_t)
#                 mse_t = (y_pred - y_tgt_n) ** 2
#                 mse_time = mse_t.mean(dim=0, keepdim=True)
#                 loss_y_final = (w_cv.view(1, T) * mse_time).sum()
#
#                 # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
#                 x_last = x_state_last[:, -1].view(m, 1)
#                 y_pred_Tp1 = (HF @ x_last).view(-1)
#                 y_true_Tp1 = y_tgt_n[:, -1]
#                 loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
#                 loss_y_final = loss_y_final + 2.0 * loss_y_Tp1
#
#                 # Add with HIGHEST alpha weight (same as training)
#                 final_weight = alpha[-1]
#                 total_loss_cv = total_loss_cv + final_weight * loss_y_final
#
#                 cv_loss += total_loss_cv.item()
#
#         if cv_loss < best_cv:
#             best_cv = cv_loss
#             torch.save(self.model, path_rts_out)
#             torch.save(model_mstep, path_m_out)
#         train_epoch = train_loss_sum / float(self.N_B)
#         print(
#             f"[JOINT EM={num_em_iters} B={batch_size}] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_loss:.6f} best_cv={best_cv:.6f}")
#
#
#
