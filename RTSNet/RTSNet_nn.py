"""# **Class: RTSNet (batch-first)**

Vectorized backward smoother — processes B sequences in parallel.
Same conventions as KalmanNet_nn (batch-first): state [B, m, 1], per-sequence
F [B, m, m], GRU hidden [1, B, d], feature vectors [B, feat], batched matmuls,
and PER-ROW standardize().
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

import sys

from RTSNet.KalmanNet_nn import KalmanNetNN

class RTSNetNN(KalmanNetNN):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.dev = torch.device("cuda")
        self.dt  = torch.float32

    #############
    ### Build ###
    #############
    def NNBuild(self, ssModel, args):


        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.m, ssModel.n)

        self.InitKGainNet(ssModel.prior_Q, ssModel.prior_Sigma, ssModel.prior_S, args)

        self.InitRTSGainNet(ssModel.prior_Q, ssModel.prior_Sigma, args)


    def standardize(self, x, eps: float = 1e-5):
        # x: [B, feat]  ->  per-row (over feat) standardization with a
        # near-constant-row guard (zeros that row), matching the sequential guard.
        if x.shape[-1] <= 1:
            return torch.zeros_like(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        out = (x - mean) / (std + eps)
        return torch.where(std < eps, torch.zeros_like(out), out)

    #################################################
    ### Initialize Backward Smoother Gain Network ###
    #################################################
    def InitRTSGainNet(self, prior_Q, prior_Sigma, args):

        mult_bw = 4  # ← use the SAME factor as FC8/Σ-GRU in KalmanNet

        self.seq_len_input = 1
        self.batch_size = 1   # updated by update_F / InitSequence at run time

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma

        # ─── NEW: two-layer embedder for F.flatten() ────────────────────────ori
        self.d_input_FF_bw = self.m * self.m  # raw F.flatten() size
        self.d_hidden_FF1_bw = self.d_input_FF_bw  # can choose any intermediate
        self.d_hidden_FF2_bw = self.m * args.in_mult_RTSNet  # final embed dim
        self.FC_F_bw = nn.Sequential(nn.Linear(self.d_input_FF_bw, self.d_hidden_FF1_bw), nn.ReLU(),
        nn.LayerNorm(self.d_hidden_FF1_bw),
        nn.Linear(self.d_hidden_FF1_bw, self.d_hidden_FF2_bw), nn.ReLU(),
        nn.LayerNorm(self.d_hidden_FF2_bw))

        # BW GRU to track Q
        self.d_input_Q_bw = self.m * args.in_mult_RTSNet
        self.d_hidden_Q_bw = self.m ** 2
        self.GRU_Q_bw = nn.GRU(self.d_input_Q_bw, self.d_hidden_Q_bw)
        self.h_Q_bw = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q_bw,
                              device=self.dev, dtype=self.dt)

        # BW GRU to track Sigma
        #self.d_input_Sigma_bw = self.d_hidden_Q_bw + 2 * self.m * args.in_mult_RTSNet oriiiiiii
        self.d_input_Sigma_bw = (self.d_hidden_Q_bw + 2 * self.m * args.in_mult_RTSNet+ self.d_hidden_FF2_bw)
        self.d_hidden_Sigma_bw = mult_bw*self.m ** 2
        self.GRU_Sigma_bw = nn.GRU(self.d_input_Sigma_bw, self.d_hidden_Sigma_bw)
        self.h_Sigma_bw = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma_bw,
                              device=self.dev, dtype=self.dt)


        # BW Fully connected 1
        self.d_input_FC1_bw = self.d_hidden_Sigma_bw # + self.d_hidden_Q
        self.d_output_FC1_bw = self.m * self.m
        self.d_hidden_FC1_bw = self.d_input_FC1_bw * args.out_mult_RTSNet
        self.FC1_bw = nn.Sequential(
                nn.Linear(self.d_input_FC1_bw, self.d_hidden_FC1_bw),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC1_bw, self.d_output_FC1_bw))

        # BW Fully connected 2
        self.d_input_FC2_bw = self.d_hidden_Sigma_bw + self.d_output_FC1_bw
        self.d_output_FC2_bw = self.d_hidden_Sigma_bw
        self.FC2_bw = nn.Sequential(
                nn.Linear(self.d_input_FC2_bw, self.d_output_FC2_bw),
                nn.ReLU())

        # BW Fully connected 3
        self.d_input_FC3_bw = self.m
        self.d_output_FC3_bw = self.m * args.in_mult_RTSNet
        self.FC3_bw = nn.Sequential(
                nn.Linear(self.d_input_FC3_bw, self.d_output_FC3_bw),
                nn.ReLU())

        # BW Fully connected 4
        self.d_input_FC4_bw = 2 * self.m
        self.d_output_FC4_bw = 2 * self.m * args.in_mult_RTSNet
        self.FC4_bw = nn.Sequential(
                nn.Linear(self.d_input_FC4_bw, self.d_output_FC4_bw),
                nn.ReLU())


    ####################################
    ### Initialize Backward Sequence ###
    ####################################
    def InitBackward(self, filter_x):
        # filter_x: [B, m] or [B, m, 1]  ->  [B, m, 1]
        self.s_m1x_nexttime = filter_x.reshape(self.batch_size, self.m, 1)

    ##############################
    ### Innovation Computation ###
    ##############################
    def S_Innovation(self, filter_x):
        # filter_x: [B, m, 1]
        self.filter_x_prior = self.f(filter_x)                          # [B, m, 1]
        self.dx = self.s_m1x_nexttime - self.filter_x_prior            # [B, m, 1]

    ################################
    ### Smoother Gain Estimation ###
    ################################
    def step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2):
        # filter_x_nexttime: [B, m, 1];  smoother_x_tplus2: [B, m, 1] or None

        # Feature: Delta tilde x_t+1 = x_t+1|T - x_t+1|t+1
        dm1x_tilde = (self.s_m1x_nexttime - filter_x_nexttime).squeeze(-1)   # [B, m]
        bw_innov_diff = self.standardize(dm1x_tilde)

        if smoother_x_tplus2 is None:
            # Delta x_t+1 = x_t+1|t+1 - x_t+1|t  (for t = T-1)
            dm1x_input2 = (filter_x_nexttime - self.filter_x_prior).squeeze(-1)  # [B, m]
        else:
            # Delta x_t+1|T = x_t+2|T - x_t+1|T  (for t = 1:T-2)
            dm1x_input2 = (smoother_x_tplus2 - self.s_m1x_nexttime).squeeze(-1)  # [B, m]
        bw_evol_diff = self.standardize(dm1x_input2)

        # Feature 7:  x_t+1|T - x_t+1|t
        dm1x_f7 = (self.s_m1x_nexttime - filter_x_nexttime).squeeze(-1)      # [B, m]
        bw_update_diff = self.standardize(dm1x_f7)

        # Smoother Gain Network Step
        SG = self.RTSGain_step(bw_innov_diff, bw_evol_diff, bw_update_diff)  # [1, B, m*m]

        # Reshape Smoother Gain to a batched Matrix
        self.SGain = SG.squeeze(0).reshape(self.batch_size, self.m, self.m)  # [B, m, m]


    ####################
    ### RTS Net Step ###
    ####################
    def RTSNet_step(self, filter_x, filter_x_nexttime, smoother_x_tplus2):
        B, m = self.batch_size, self.m
        filter_x = filter_x.reshape(B, m, 1)
        filter_x_nexttime = filter_x_nexttime.reshape(B, m, 1)
        if smoother_x_tplus2 is not None:
            smoother_x_tplus2 = smoother_x_tplus2.reshape(B, m, 1)

        # Compute Innovation
        self.S_Innovation(filter_x)

        # Compute Smoother Gain
        self.step_RTSGain_est(filter_x_nexttime, smoother_x_tplus2)

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.SGain, self.dx)                 # [B, m, 1]
        self.s_m1x_nexttime = filter_x + INOV                 # [B, m, 1]

        # return
        return self.s_m1x_nexttime.squeeze(-1)                # [B, m]

    ##########################
    ### Smoother Gain Step ###
    ##########################
    def RTSGain_step(self, bw_innov_diff, bw_evol_diff, bw_update_diff):

        def expand_dim(x):
            # x: [B, feat]  ->  [1, B, feat]
            return x.unsqueeze(0)

        bw_innov_diff = expand_dim(bw_innov_diff)
        bw_evol_diff = expand_dim(bw_evol_diff)
        bw_update_diff = expand_dim(bw_update_diff)

        ####################
        ### Forward Flow ###
        ####################

        # FC 3
        in_FC3 = bw_update_diff
        out_FC3 = self.FC3_bw(in_FC3)

        # Q-GRU
        in_Q = out_FC3
        out_Q, self.h_Q_bw = self.GRU_Q_bw(in_Q, self.h_Q_bw)

        # FC 4
        in_FC4 = torch.cat((bw_innov_diff, bw_evol_diff), 2)
        out_FC4 = self.FC4_bw(in_FC4)

        # embed the current F
        F_vec = self.F.reshape(self.batch_size, -1).unsqueeze(0)  # [1, B, m²]
        F_emb = self.FC_F_bw(F_vec)  # [1, B, d_hidden_FF2_bw]
        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC4, F_emb), 2)#ori

        out_Sigma, self.h_Sigma_bw = self.GRU_Sigma_bw(in_Sigma, self.h_Sigma_bw)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1_bw(in_FC1)

        #####################
        ### Backward Flow ###
        #####################

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_FC1), 2)
        out_FC2 = self.FC2_bw(in_FC2)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma_bw = out_FC2

        return out_FC1

    ###############
    ### Forward ###
    ###############
    def forward(self, yt, filter_x, filter_x_nexttime, smoother_x_tplus2):
        if yt is None:
            # BW pass
            return self.RTSNet_step(filter_x, filter_x_nexttime, smoother_x_tplus2)
        else:
            # FW pass — accept [B, n] or [B, n, 1]
            yt = yt.reshape(self.batch_size, self.n, 1)
            return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        B = self.batch_size

        ### FW GRUs (zero-then-fill; prior broadcasts identically across B rows)
        hidden = weight.new(1, B, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, :, :self.n ** 2] = self.prior_S.flatten()

        hidden = weight.new(1, B, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, :, :self.m ** 2] = self.prior_Sigma.flatten()

        hidden = weight.new(1, B, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, :, :] = self.prior_Q.flatten()

        ### BW GRUs
        hidden_bw = weight.new(1, B, self.d_hidden_Q_bw).zero_()
        self.h_Q_bw = hidden_bw.data
        self.h_Q_bw[0, :, :] = self.prior_Q.flatten()

        hidden_bw = weight.new(1, B, self.d_hidden_Sigma_bw).zero_()
        self.h_Sigma_bw = hidden_bw.data
        self.h_Sigma_bw[0, :, :self.m ** 2] = self.prior_Sigma.flatten()
