"""# **Class: KalmanNet (batch-first)**

Vectorized rewrite: the forward filter processes B sequences in parallel.

Conventions
-----------
  state / obs are batched column vectors:  x -> [B, m, 1],  y -> [B, n, 1]
  per-sequence F:                          self.F -> [B, m, m]
  GRU hidden states:                       [1, B, d]
  feature vectors for FC/GRU:              [B, feat]  (expand_dim -> [1, B, feat])
  matmuls are batched (torch.bmm).
  standardize() is PER-ROW (over the feature dim), so sequences never couple.

B = 1 is the degenerate case, so single-sequence inference works through the
same path. `self.batch_size` is (re)set by update_F / InitSequence.
"""
#the newone
import torch
import torch.nn as nn
import torch.nn.functional as func

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.dev = torch.device("cuda")
        self.dt  = torch.float32

    def NNBuild(self, SysModel, args):

        self.F = SysModel.F
        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):

        self.seq_len_input = 1
        self.batch_size = 1   # updated by update_F / InitSequence at run time

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S

        mult = 4

        # GRU to track Q
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q, device=self.dev, dtype=self.dt)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet + (self.m ** 2) * args.in_mult_KNet # (self.m ** 2) * args.in_mult_KNet is the F output
        self.d_hidden_Sigma = (self.m ** 2) * mult
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma, device=self.dev, dtype=self.dt)

        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = (self.n ** 2)* mult
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_S, device=self.dev, dtype=self.dt)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma # This is \hat{\Sigma}_{t|t
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU())

        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU())

        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU())

        # Fully connected F

        self.d_input_FC8 = self.m ** 2
        self.d_output_FC8 = self.d_input_FC8 * args.in_mult_KNet  # latent size h
        # new:
        self.d_hidden_FC8_1 = self.d_input_FC8  * 4  # e.g. the 4 is with no reason
        self.d_hidden_FC8_2 = self.d_input_FC8  * 2  # e.g. the 2 is with no reason
        # new deeper FC8
        self.FC8 = nn.Sequential(nn.Linear(self.d_input_FC8, self.d_hidden_FC8_1),
            nn.LayerNorm(self.d_hidden_FC8_1),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC8_1, self.d_hidden_FC8_2),
            nn.LayerNorm(self.d_hidden_FC8_2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC8_2, self.d_output_FC8),
            nn.LayerNorm(self.d_output_FC8),
            nn.ReLU())


    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def f_new(self, x):
        # self.F: [B, m, m],  x: [B, m, 1]  ->  [B, m, 1]
        return torch.bmm(self.F, x)

    def update_F(self, F):
        # F: [B, m, m] (per-sequence transition matrices)
        self.F = F
        self.batch_size = F.shape[0]
        self.f = self.f_new

    def InitSystemDynamics(self, f, h, m, n):

        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        # M1_0: [B, m, 1]
        self.T = T
        self.batch_size = M1_0.shape[0]

        self.m1x_posterior = M1_0                              # [B, m, 1]
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)          # [B, n, 1]

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)           # [B, m, 1]
        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)                     # [B, n, 1]

    ##############################
    ### Kalman Gain Estimation ###
    ##############################

    def standardize(self, x, eps=1e-5):
        # x: [B, feat]  ->  per-row (over feat) standardization
        if x.shape[-1] <= 1:
            return torch.zeros_like(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + eps)

    def step_KGain_est(self, y):
        # y: [B, n, 1];  feature diffs are [B, feat]
        obs_diff       = (y - self.y_previous).squeeze(-1)                          # [B, n]
        obs_innov_diff = (y - self.m1y).squeeze(-1)                                 # [B, n]
        fw_evol_diff   = (self.m1x_posterior - self.m1x_posterior_previous).squeeze(-1)  # [B, m]
        fw_update_diff = (self.m1x_posterior - self.m1x_prior_previous).squeeze(-1)      # [B, m]

        obs_diff = self.standardize(obs_diff)
        obs_innov_diff = self.standardize(obs_innov_diff)
        fw_evol_diff = self.standardize(fw_evol_diff)
        fw_update_diff = self.standardize(fw_update_diff)
        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)  # [1, B, n*m]
        # Reshape Kalman Gain to a batched Matrix
        self.KGain = KG.squeeze(0).reshape(self.batch_size, self.m, self.n)          # [B, m, n]



    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # y: [B, n, 1]

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y                                     # [B, n, 1]

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)                      # [B, m, 1]
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV            # [B, m, 1]

        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        # return
        return self.m1x_posterior.squeeze(-1)                 # [B, m]

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            # x: [B, feat]  ->  [1, B, feat]
            return x.unsqueeze(0)

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################

        in_FC8 = self.F.reshape(self.batch_size, -1).unsqueeze(0)  # [1, B, m²]
        out_FC8 = self.FC8(in_FC8)  # [1, B, h]

        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)


        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6, out_FC8), 2)#ori changed
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)


        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU - THIS IS THE NEW P
        self.h_Sigma = out_FC4

        return out_FC2

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        # accept [B, n] or [B, n, 1]  ->  [B, n, 1]
        y = y.reshape(self.batch_size, self.n, 1)
        return self.KNet_step(y)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        B = self.batch_size
        # zero-then-fill; the prior broadcasts identically across all B rows.
        hidden = weight.new(1, B, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, :, :self.n ** 2] = self.prior_S.flatten()
        hidden = weight.new(1, B, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, :, :self.m ** 2] = self.prior_Sigma.flatten()
        hidden = weight.new(1, B, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, :, :] = self.prior_Q.flatten()
