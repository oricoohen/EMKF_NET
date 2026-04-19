import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- fixed device -----
device = torch.device("cuda")

#####################################################################################regular rts
# ============================================================
# 1. M-step network: ΔH = MLP(z_in)
# ============================================================
class DeltaH_MStepNet(nn.Module):
    """
    Single M-step network for H:
      ΔH = MLP(z_in),  H^{(i+1)} = H^{(i)} + ΔH

   z_in block layout (all flattened, then concatenated):
  [ H_current (n*m) | H_em (n*m) | S_nu_corr (n*n) | C_nu_x (n*m) ]

Input:
  - z_in: [B, d_z] where d_z = 3*(n*m) + (n*n)
      """
    def __init__(self, m, n, d_hidden=256,dH_scale=1):
        super().__init__()
        self.m = m
        self.n = n
        self.dH_scale = dH_scale #change ori
        self.block_dims = [
            n * m,  # H_current
            n * m,  # H_em
            n * n,  # S_nu_corr
            n * m  # C_nu_x
        ]
        self.d_z = sum(self.block_dims)

        # self.ln = nn.LayerNorm(self.d_z) change ori
        self.block_lns = nn.ModuleList([
            nn.LayerNorm(dim) for dim in self.block_dims
        ])
        self.mlp = nn.Sequential(
            nn.Linear(self.d_z, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n * m)
        )

    def _norm_block(self, block, eps=1e-6):
        # block: [B, D_block]
        mean = block.mean(dim=1, keepdim=True)
        std = block.std(dim=1, keepdim=True)
        return (block - mean) / (std + eps)

    def forward(self, z_in):
        """
        z_in: [B, d_z] on CUDA
        returns ΔH: [B, m, m] on CUDA
        """
        m = self.m
        n=self.n
        B, dz = z_in.shape
        # blocks = []
        # start = 0
        # for dim in self.block_dims:
        #     end = start + dim
        #     b = z_in[:, start:end]  # [B, dim]
        #     b = self._norm_block(b)  # normalize this block
        #     blocks.append(b)
        #     start = end
        #
        # z_in = torch.cat(blocks, dim=1)
        # z_in[:, 0: (m * m)] = 0          # no A1
        # z_in[:, (m * m): (2 * m * m)] = 0#no A_2
        # z_in[:, 2 * (self.m * self.m): 2 * (self.m * self.m) + (self.n * self.n)] = 0 #no S_delta_x
        # # z_in[:, 3 * (m * m): 3 * (m * m) + (n * n)] = 0 #no S_nu
        # z_in[:, 3 * (m * m) + (n * n): 4 * (m * m) + (n * n)] = 0# no C_delta_x_xminus
        # z_in[:, 4 * (m * m) + (n * n): 5 * (m * m) + (n * n)] = 0 #NO F_current

        # z = self.ln(z_in) change ori
        #
        # deltaH_vec = self.mlp(z)  # [B, n*m]
        # deltaH = deltaH_vec.view(B, self.n, self.m)
        # return deltaH
        blocks = []
        start = 0
        for ln, dim in zip(self.block_lns, self.block_dims):
            end = start + dim
            block = z_in[:, start:end]
            block = ln(block)
            blocks.append(block)
            start = end

        z = torch.cat(blocks, dim=1)

        raw = self.mlp(z)  # [B, n*m]
        deltaH_vec = self.dH_scale * torch.tanh(raw)
        deltaH = deltaH_vec.view(B, self.n, self.m)
        return deltaH

