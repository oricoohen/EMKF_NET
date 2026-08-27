"""H M-step network variant for the H EXPERIMENT -- single-LayerNorm version.

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
Exactly the same situation as RTSNet/RTSNet_nn_for_H.py, one layer down.

The trained H M-nets (RTSNet/synthetic/changed_H_v_0/**/EMKF/False/M_*.pt) were saved
when DeltaH_MStepNet normalized the input with ONE LayerNorm over the whole vector:

        self.ln = nn.LayerNorm(self.d_z)        # d_z = 20 for m = n = 2
        ...
        z = self.ln(z_in)
        deltaH = self.mlp(z).view(B, n, m)

Commit 0107cb2 ("m_net_ new exp") replaced that single `ln` with a ModuleList of
per-block LayerNorms:

        # self.ln = nn.LayerNorm(self.d_z) change ori     <- commented out
        self.block_lns = nn.ModuleList([nn.LayerNorm(dim) for dim in self.block_dims])

A pickled .pt stores the OBJECT (its layers and attributes), not the forward code. The
saved M-nets therefore contain `ln` and `mlp` and have no `block_lns` at all, while
emkf/AI_M_step_for_h.py's current forward iterates over `self.block_lns`. Loading one and
running it fails with

        AttributeError: 'DeltaH_MStepNet' object has no attribute 'block_lns'

which is what filled the `error` column of mnet_H_comparison.csv for every candidate.

=============================================================================
WHAT THIS FILE DOES
=============================================================================
It subclasses the current DeltaH_MStepNet and overrides ONLY forward(), restoring the
single-LayerNorm path verbatim from commit be11017 ("include h emkf") -- the version these
checkpoints were trained with.

Nothing in emkf/AI_M_step_for_h.py is modified, so M-nets trained after 0107cb2 (which do
have block_lns) keep working exactly as they do today.

__init__ is deliberately NOT overridden: torch.load reconstructs a pickled module without
calling __init__, so the object arrives with its own `ln`, `mlp`, `m`, `n` and `block_dims`
already populated from the file. Only the forward code has to match.

=============================================================================
HOW TO USE IT
=============================================================================
The .pt files record the class name 'emkf.AI_M_step_for_h.DeltaH_MStepNet', so that name
must be pointed at this class BEFORE torch.load:

        from emkf.AI_M_step_for_h_single_ln import DeltaH_MStepNet
        import emkf.AI_M_step_for_h as _m
        _m.DeltaH_MStepNet = DeltaH_MStepNet     # in-memory only; writes nothing to disk

See compare_mnets_H.py for a working example.
"""

from emkf.AI_M_step_for_h import DeltaH_MStepNet as _BaseDeltaH_MStepNet


class DeltaH_MStepNet(_BaseDeltaH_MStepNet):
    """DeltaH_MStepNet whose forward uses the single `ln` LayerNorm, not `block_lns`."""

    def forward(self, z_in):
        """z_in: [B, d_z]  ->  deltaH: [B, n, m]

        Restored from commit be11017. The current base forward normalizes each block of
        z_in with its own LayerNorm from self.block_lns; these checkpoints have a single
        self.ln covering the whole d_z-wide vector instead.

        z_in block layout (flattened, concatenated), d_z = 20 for m = n = 2:
            [ A_yx (n*m) | A_xx (m*m) | S_nu (n*n) | C_nu_x (n*m) | H_current (n*m) ]
        """
        B, dz = z_in.shape

        z = self.ln(z_in)

        deltaH_vec = self.mlp(z)                       # [B, n*m]
        deltaH = deltaH_vec.view(B, self.n, self.m)
        return deltaH
