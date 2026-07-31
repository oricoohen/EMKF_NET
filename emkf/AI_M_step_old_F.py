import torch
import torch.nn as nn

device = torch.device("cuda")


class DeltaF_MStepNet(nn.Module):
    """
    Original M-step network architecture (global LayerNorm, no tanh bound).
    Used for loading checkpoints trained before the per-block LN architecture.
    """
    def __init__(self, m, n, d_hidden=256, dF_scale=0.1):
        super().__init__()
        self.m = m
        self.n = n
        self.d_z = 5 * (m * m) + (n * n)

        self.ln = nn.LayerNorm(self.d_z)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_z, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, m * m))

    def forward(self, z_in):
        B, _ = z_in.shape
        z = self.ln(z_in)
        deltaF_vec = self.mlp(z)
        return deltaF_vec.view(B, self.m, self.m)
