"""RTSNet variant for the H EXPERIMENT (diverse / unknown H, fixed & known F).

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
The H-experiment checkpoints (RTSNet/synthetic/changed_H_v_0/**/True_H/*.pt and
False_H/*.pt) were trained with a version of KGain_step / RTSGain_step that fed BOTH
matrices into the GRUs: the F matrix via FC8 / FC_F_bw, and the H matrix via FC9 / FC_H_bw.

The .pt files store the LAYERS with their input widths frozen at save time:

        layer            width stored in the .pt
        GRU_Sigma                34   = out_Q(4) + out_FC6(10) + out_FC8(20)
        GRU_S                    44   = out_FC1(4) + out_FC7(20) + out_FC9(20)
        GRU_Sigma_bw             44   = out_Q(4) + out_FC4(20) + F_emb(10) + H_emb(10)

They do NOT store the code that decides what to concatenate -- that comes from whatever
class torch.load reconstructs them with. Commit 1b7e810 ("m_net_ new exp") commented the
F-embedding out of RTSNet/KalmanNet_nn.py and RTSNet/RTSNet_nn.py, so those files now build

        GRU_Sigma                14   = out_Q(4) + out_FC6(10)              <- FC8 dropped
        GRU_S                    44   (unchanged)
        GRU_Sigma_bw             34   = out_Q(4) + out_FC4(20) + H_emb(10)  <- F_emb dropped

Loading an H checkpoint with the current base class therefore dies with

        RuntimeError: input.size(-1) must be equal to input_size. Expected 34, got 14

and, once that one is fixed, again with 'Expected 44, got 34' on the backward pass.
(This is why data_generate_exp_for_paper/H_exp/old/exp_H_test.py and the other for_h_*
scripts no longer run either -- they predate that commit.)

None of the existing classes is usable for these checkpoints; all seven were tried by
actually loading a checkpoint and running a forward+backward pass:

        RTSNet_nn.py            Sigma=14  S=44  Sigma_bw=34   -> too narrow
        RTSNet_nn_with_H.py     Sigma=14  S=44  Sigma_bw=34   -> byte-identical to the base
        RTSNet_nn_with_F.py     Sigma=34  S=24  Sigma_bw=34   -> S too narrow, no update_H
        (batched / multipass / old- also fail)

=============================================================================
WHAT THIS FILE DOES
=============================================================================
It restores the F embedding for the H experiment ONLY, without editing any shared file,
because other models depend on the current 'no_F' behaviour of RTSNet_nn.py and must keep
loading exactly as they do today.

It subclasses the existing RTSNetNN and overrides just the two methods that assemble the
mismatched vectors -- 4 lines of real change in KGain_step, 2 in RTSGain_step. Everything
else (all layers, update_F, update_H, InitSequence, KNet_step, the forward flow) is
inherited unchanged, so future fixes to RTSNet_nn.py are picked up automatically here.

The restored code is taken verbatim from commit be11017 ("include h emkf"), i.e. the
version these checkpoints were actually trained with.

=============================================================================
HOW TO USE IT
=============================================================================
The .pt files record the class name 'RTSNet.RTSNet_nn.RTSNetNN' inside them, so that name
must be pointed at this class BEFORE torch.load, exactly the way compare_mnets.py does it
for RTSNet_nn_with_F:

        from RTSNet.RTSNet_nn_for_H import RTSNetNN
        import RTSNet.RTSNet_nn as _m
        _m.RTSNetNN = RTSNetNN          # in-memory only; writes nothing to disk

See compare_mnets_H.py for a working example.

NOTE (training): this class is for LOADING pre-trained H checkpoints -- torch.load brings
the layers in from the .pt file. To TRAIN a fresh network with this architecture, NNBuild
would also need overriding: the base NNBuild sizes GRU_Sigma without the F term and leaves
FC_F_bw commented out (RTSNet/RTSNet_nn.py), so a newly built model would have no FC_F_bw
to embed F with. Add that here if joint H training is ever needed.
"""

import torch

from RTSNet.RTSNet_nn import RTSNetNN as _BaseRTSNetNN


class RTSNetNN(_BaseRTSNetNN):
    """Base RTSNetNN with the F embedding put back into both Sigma-GRUs."""

    ###############################
    ### FORWARD (KalmanNet) step ##
    ###############################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        """Identical to KalmanNetNN.KGain_step except that FC8(F) is fed to the Sigma-GRU.

        The base version has the FC8 block and the 3-way cat commented out with '#no_F',
        which makes in_Sigma 14 wide; the checkpoints' GRU_Sigma needs 34.
        """

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1],
                                   device=x.device, dtype=x.dtype)
            expanded[0, 0, :] = x
            return expanded

        obs_diff       = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff   = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################

        # FC8 for the F matrix. THIS IS THE RESTORED BLOCK -- standardized to match FC5/FC6,
        # exactly as at commit be11017.
        in_FC8  = self.F.flatten().unsqueeze(0).unsqueeze(0)   # [1,1,m²]
        in_FC8  = self.standardize(in_FC8)
        out_FC8 = self.FC8(in_FC8)                             # [1,1,20]

        # FC9 for the H matrix. Still live in the base class, and deliberately NOT
        # standardized there -- keep it that way or the weights no longer match.
        in_FC9  = self.H.flatten().unsqueeze(0).unsqueeze(0)   # [1,1,n*m]
        out_FC9 = self.FC9(in_FC9)                             # [1,1,20]

        # FC 5
        out_FC5 = self.FC5(fw_evol_diff)

        # Q-GRU
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)        # [1,1,4]

        # FC 6
        out_FC6 = self.FC6(fw_update_diff)                     # [1,1,10]

        # Sigma-GRU: 4 + 10 + 20 = 34  (base builds only 4 + 10 = 14)
        in_Sigma = torch.cat((out_Q, out_FC6, out_FC8), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        out_FC1 = self.FC1(out_Sigma)                          # [1,1,4]

        # FC 7
        out_FC7 = self.FC7(torch.cat((obs_diff, obs_innov_diff), 2))   # [1,1,20]

        # S-GRU: 4 + 20 + 20 = 44  (unchanged from the base class)
        in_S = torch.cat((out_FC1, out_FC7, out_FC9), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        out_FC2 = self.FC2(torch.cat((out_Sigma, out_S), 2))

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        out_FC3 = self.FC3(torch.cat((out_S, out_FC2), 2))

        # FC 4
        out_FC4 = self.FC4(torch.cat((out_Sigma, out_FC3), 2))

        # updating hidden state of the Sigma-GRU - THIS IS THE NEW P
        self.h_Sigma = out_FC4

        return out_FC2

    ################################
    ### BACKWARD (smoother) step ###
    ################################
    def RTSGain_step(self, bw_innov_diff, bw_evol_diff, bw_update_diff):
        """Identical to RTSNetNN.RTSGain_step except that F_emb is fed to GRU_Sigma_bw.

        RTSNet_nn.py already computes H_emb and even carries the correct line
            # in_Sigma = torch.cat((out_Q, out_FC4, F_emb, H_emb), 2)  # With F and H new_exp
        commented out directly above the active 3-way one. That 3-way version is 34 wide;
        the checkpoints' GRU_Sigma_bw needs 44.
        """

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1],
                                   device=x.device, dtype=x.dtype)
            expanded[0, 0, :] = x
            return expanded

        bw_innov_diff  = expand_dim(bw_innov_diff)
        bw_evol_diff   = expand_dim(bw_evol_diff)
        bw_update_diff = expand_dim(bw_update_diff)

        # FC 3
        out_FC3 = self.FC3_bw(bw_update_diff)

        # Q-GRU
        out_Q, self.h_Q_bw = self.GRU_Q_bw(out_FC3, self.h_Q_bw)   # [1,1,4]

        # FC 4
        out_FC4 = self.FC4_bw(torch.cat((bw_innov_diff, bw_evol_diff), 2))   # [1,1,20]

        # Embed the current F. THIS IS THE RESTORED BLOCK (FC_F_bw exists in the trained
        # checkpoints; it is only the *code* that stopped using it).
        F_vec = self.F.flatten().view(1, 1, -1)      # [1,1,m²]
        F_vec = self.standardize(F_vec)
        F_emb = self.FC_F_bw(F_vec)                  # [1,1,10]

        # Embed the current H (H diversity) -- standardized, as in the base class.
        H_vec = self.H.flatten().view(1, 1, -1)      # [1,1,n*m]
        H_vec = self.standardize(H_vec)
        H_emb = self.FC_H_bw(H_vec)                  # [1,1,10]

        # Sigma-GRU: 4 + 20 + 10 + 10 = 44  (base builds only 4 + 20 + 10 = 34)
        in_Sigma = torch.cat((out_Q, out_FC4, F_emb, H_emb), 2)
        out_Sigma, self.h_Sigma_bw = self.GRU_Sigma_bw(in_Sigma, self.h_Sigma_bw)

        # FC 1
        out_FC1 = self.FC1_bw(out_Sigma)

        # FC 2 -- updating hidden state of the Sigma-GRU
        out_FC2 = self.FC2_bw(torch.cat((out_Sigma, out_FC1), 2))
        self.h_Sigma_bw = out_FC2

        return out_FC1
