"""
Batched RTSNet  (delete-to-revert add-on).

`RTSNetNN_batched` subclasses the existing `RTSNetNN`, so it reuses *all* layer
definitions (NNBuild / InitKGainNet / InitRTSGainNet) untouched and therefore has
identical parameter names. Only the forward-path math is overridden to run B
sequences in parallel via `torch.bmm`, with per-sample F/H carried as
[B, m, m] / [B, n, m] tensors.

Conventions
-----------
- batch size B = self.batch_size (set per call via set_batch_size)
- state vectors stored as [B, m]; observations as [B, n]
- F: [B, m, m], H: [B, n, m]
- GRU hidden states: [1, B, d]   (nn.GRU processes each batch element independently)

For B == 1 this reproduces the sequential model's math exactly (mod float order),
which is what the equivalence test relies on.
"""

import torch
from RTSNet.RTSNet_nn import RTSNetNN


class RTSNetNN_batched(RTSNetNN):

    # ----------------------------------------------------------------- helpers
    def set_batch_size(self, B):
        self.batch_size = B

    def standardize(self, x, eps: float = 1e-5):
        # Per-sample standardization over the LAST dim (matches the sequential
        # model, which standardizes each single-sample vector globally).
        # unbiased=False to match RTSNetNN.standardize.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / (std + eps)

    @staticmethod
    def _bmv(M, v):
        # batched matrix-vector:  M [B,a,b] @ v [B,b] -> [B,a]
        return torch.bmm(M, v.unsqueeze(-1)).squeeze(-1)

    # --------------------------------------------------- system dynamics (bmm)
    def f_new(self, x):
        return self._bmv(self.F, x)

    def update_F(self, F):
        # F may be [m,m] (shared) or [B,m,m]
        self.F = F
        self.f = self.f_new
        self._f_is_batched = True

    def _f_apply(self, x):
        """Apply the state-evolution function to a batch x [B,m] -> [B,m].

        If update_F installed the batched f_new, call it directly. Otherwise
        (the usual case: f is the fixed nonlinear Lorenz map, NOT batch-aware),
        evaluate it per-sample and stack. f is cheap relative to the gain GRUs,
        so this loop is not the bottleneck and guarantees exact equivalence with
        the sequential path."""
        if getattr(self, "_f_is_batched", False):
            return self.f(x)
        outs = [self.f(x[b]).reshape(-1) for b in range(x.shape[0])]
        return torch.stack(outs, dim=0)

    def h_new(self, x):
        return self._bmv(self.H, x)

    def update_H(self, H):
        self.H = H
        self.h = self.h_new

    def _F_b(self):
        B = self.batch_size
        if self.F.dim() == 2:
            return self.F.unsqueeze(0).expand(B, -1, -1)
        return self.F

    def _H_b(self):
        B = self.batch_size
        if self.H.dim() == 2:
            return self.H.unsqueeze(0).expand(B, -1, -1)
        return self.H

    # --------------------------------------------------------- init sequence
    def InitSequence(self, M1_0, T):
        self.T = T
        B = self.batch_size

        m1 = M1_0
        if m1.dim() == 1:                      # [m]
            m1 = m1.unsqueeze(0).expand(B, -1)
        elif m1.dim() == 2 and m1.shape[-1] == 1:   # [m,1]
            m1 = m1.squeeze(-1).unsqueeze(0).expand(B, -1)
        # else already [B, m]
        self.m1x_posterior = m1.contiguous()
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)   # [B,n]

    # --------------------------------------------------------- forward filter
    def step_prior(self):
        self.m1x_prior = self._f_apply(self.m1x_posterior)    # [B,m]
        self.m1y = self.h(self.m1x_prior)                     # [B,n]

    def step_KGain_est(self, y):
        obs_diff = y - self.y_previous                                  # [B,n]
        obs_innov_diff = y - self.m1y                                   # [B,n]
        fw_evol_diff = self.m1x_posterior - self.m1x_posterior_previous # [B,m]
        fw_update_diff = self.m1x_posterior - self.m1x_prior_previous   # [B,m]

        obs_diff = self.standardize(obs_diff)
        obs_innov_diff = self.standardize(obs_innov_diff)
        fw_evol_diff = self.standardize(fw_evol_diff)
        fw_update_diff = self.standardize(fw_update_diff)

        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = KG.reshape(self.batch_size, self.m, self.n)

    def KNet_step(self, y):
        self.step_prior()
        self.step_KGain_est(y)
        dy = y - self.m1y                                  # [B,n]
        INOV = self._bmv(self.KGain, dy)                   # [B,m]
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        return self.m1x_posterior                          # [B,m]

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):                 # [B,feat] -> [1,B,feat]
            return x.unsqueeze(0)

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        # FC9 for H matrix  -> [1,B,n*m]   (no standardize, matching original)
        in_FC9 = self._H_b().reshape(self.batch_size, -1).unsqueeze(0)
        out_FC9 = self.FC9(in_FC9)

        # FC5 -> Q-GRU
        out_FC5 = self.FC5(fw_evol_diff)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        # FC6 -> Sigma-GRU
        out_FC6 = self.FC6(fw_update_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC1
        out_FC1 = self.FC1(out_Sigma)

        # FC7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7, out_FC9), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        # Backward flow
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)
        self.h_Sigma = out_FC4

        return out_FC2                                     # [1,B,m*n]

    # --------------------------------------------------------- backward smoother
    def InitBackward(self, filter_x):
        self.s_m1x_nexttime = filter_x                     # [B,m]

    def S_Innovation(self, filter_x):
        self.filter_x_prior = self._f_apply(filter_x)      # [B,m]
        self.dx = self.s_m1x_nexttime - self.filter_x_prior

    def step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2):
        dm1x_tilde = self.s_m1x_nexttime - filter_x_nexttime
        bw_innov_diff = self.standardize(dm1x_tilde)

        if smoother_x_tplus2 is None:
            dm1x_input2 = filter_x_nexttime - self.filter_x_prior
        else:
            dm1x_input2 = smoother_x_tplus2 - self.s_m1x_nexttime
        bw_evol_diff = self.standardize(dm1x_input2)

        dm1x_f7 = self.s_m1x_nexttime - filter_x_nexttime
        bw_update_diff = self.standardize(dm1x_f7)

        SG = self.RTSGain_step(bw_innov_diff, bw_evol_diff, bw_update_diff)
        self.SGain = SG.reshape(self.batch_size, self.m, self.m)

    def RTSGain_step(self, bw_innov_diff, bw_evol_diff, bw_update_diff):
        def expand_dim(x):                 # [B,feat] -> [1,B,feat]
            return x.unsqueeze(0)

        bw_innov_diff = expand_dim(bw_innov_diff)
        bw_evol_diff = expand_dim(bw_evol_diff)
        bw_update_diff = expand_dim(bw_update_diff)

        # FC3 -> Q-GRU
        out_FC3 = self.FC3_bw(bw_update_diff)
        out_Q, self.h_Q_bw = self.GRU_Q_bw(out_FC3, self.h_Q_bw)

        # FC4
        in_FC4 = torch.cat((bw_innov_diff, bw_evol_diff), 2)
        out_FC4 = self.FC4_bw(in_FC4)

        # H embedding (standardized, matching original)
        H_vec = self._H_b().reshape(self.batch_size, -1).unsqueeze(0)   # [1,B,n*m]
        H_vec = self.standardize(H_vec)
        H_emb = self.FC_H_bw(H_vec)

        # Sigma-GRU
        in_Sigma = torch.cat((out_Q, out_FC4, H_emb), 2)
        out_Sigma, self.h_Sigma_bw = self.GRU_Sigma_bw(in_Sigma, self.h_Sigma_bw)

        # FC1
        out_FC1 = self.FC1_bw(out_Sigma)

        # FC2 (backward) -> update Sigma-GRU hidden
        in_FC2 = torch.cat((out_Sigma, out_FC1), 2)
        out_FC2 = self.FC2_bw(in_FC2)
        self.h_Sigma_bw = out_FC2

        return out_FC1                                     # [1,B,m*m]

    def RTSNet_step(self, filter_x, filter_x_nexttime, smoother_x_tplus2):
        self.S_Innovation(filter_x)
        self.step_RTSGain_est(filter_x_nexttime, smoother_x_tplus2)
        INOV = self._bmv(self.SGain, self.dx)              # [B,m]
        self.s_m1x_nexttime = filter_x + INOV
        return self.s_m1x_nexttime                         # [B,m]

    # --------------------------------------------------------- forward dispatch
    def forward(self, yt, filter_x, filter_x_nexttime, smoother_x_tplus2):
        if yt is None:
            return self.RTSNet_step(filter_x, filter_x_nexttime, smoother_x_tplus2)
        return self.KNet_step(yt)

    # --------------------------------------------------------- hidden states
    def init_hidden(self):
        weight = next(self.parameters()).data
        B = self.batch_size

        ### FW GRUs
        self.h_S = weight.new(1, B, self.d_hidden_S).zero_()
        self.h_S[0, :, :self.n ** 2] = self.prior_S.flatten()

        self.h_Sigma = weight.new(1, B, self.d_hidden_Sigma).zero_()
        self.h_Sigma[0, :, :self.m ** 2] = self.prior_Sigma.flatten()

        self.h_Q = weight.new(1, B, self.d_hidden_Q).zero_()
        self.h_Q[0, :, :] = self.prior_Q.flatten()

        ### BW GRUs
        self.h_Q_bw = weight.new(1, B, self.d_hidden_Q_bw).zero_()
        self.h_Q_bw[0, :, :] = self.prior_Q.flatten()

        self.h_Sigma_bw = weight.new(1, B, self.d_hidden_Sigma_bw).zero_()
        self.h_Sigma_bw[0, :, :self.m ** 2] = self.prior_Sigma.flatten()


def make_batched_from(model_seq, batch_size):
    """
    Turn an already-loaded sequential RTSNetNN into its batched twin.

    RTSNetNN_batched adds NO new layers or constructor state (only batch_size and
    overridden methods), so we simply re-class the live object. This keeps the
    exact same parameter tensors -> any optimizer built on model.parameters()
    stays valid, and the on-disk checkpoint is untouched (caller loads a fresh
    copy). The sequential training path loads its own separate copy, so A/B
    comparison is preserved.
    """
    model_seq.__class__ = RTSNetNN_batched
    model_seq.set_batch_size(batch_size)
    return model_seq
