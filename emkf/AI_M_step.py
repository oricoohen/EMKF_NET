import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- fixed device -----
device = torch.device("cuda")


# ============================================================
# 1. M-step network: ΔF = MLP(z_in)
# ============================================================
class DeltaF_MStepNet(nn.Module):
    """
    Single M-step network:
      ΔF = MLP(z_in),  F^{(i+1)} = F^{(i)} + ΔF

    Inputs:
      - z_in: [B, d_z] where d_z = 5*m^2 + n^2
    Output:
      - ΔF:  [B, m, m]
    """
    def __init__(self, m, n, d_hidden=256):
        super().__init__()
        self.m = m
        self.n = n
        self.d_z = 5 * (m * m) + (n * n)
        self.block_dims = [
            m * m,  # A1
            m * m,  # A2
            m * m,  # S_delta_x
            n * n,  # S_nu
            m * m,  # C_delta_x_xminus
            m * m   # F_current
        ]

        self.ln = nn.LayerNorm(self.d_z)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_z, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, m * m))

    def _norm_block(self, block, eps=1e-6):
        # block: [B, D_block]
        mean = block.mean(dim=1, keepdim=True)
        std = block.std(dim=1, keepdim=True)
        return (block - mean) / (std + eps)

    def forward(self, z_in):
        """
        z_in: [B, d_z] on CUDA
        returns ΔF: [B, m, m] on CUDA
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

        z = self.ln(z_in)

        deltaF_vec = self.mlp(z)           # [B, m^2]
        deltaF = deltaF_vec.view(B, self.m, self.m)
        return deltaF


# ============================================================
# 2. Helpers for statistics and loss
# ============================================================
# def compute_mstep_stats(x_smooth, x_prev_smooth, delta_x, nu, x_prev_hat, x_0):
#     """
#     All inputs are CUDA tensors.
#     Shapes (YOUR convention):
#       x_smooth      : [B, m, T]
#       x_prev_smooth : [B, m, T]
#       delta_x       : [B, m, T]
#       nu            : [B, n, T]
#       x_prev_hat    : [B, m, T]
#
#     Returns:
#       A1, A2, Sdx, Sv, Cdx
#       A1, A2, Sdx, Cdx : [B, m, m]
#       Sv               : [B, n, n]
#     """
#     B, m, T = x_smooth.shape
#     _, n, _ = nu.shape
#
#     # ---------- A1 = (1/T) Σ_t x_t x_{t-1}^T ----------
#     A1 = torch.zeros(B, m, m, device=x_smooth.device)
#
#     for t in range(T):
#         # x_t, x_prev: [B, m]
#         x_t = x_smooth[:, :, t]        # [B, m]
#         x_prev = x_prev_smooth[:, :, t]   # [B, m]
#
#         # [B, m, 1] * [B, 1, m] -> [B, m, m]
#         A1 += x_t.unsqueeze(2) * x_prev.unsqueeze(1)
#
#     A1 = A1 / T
#
#     # ---------- A2 = (1/T) Σ_t x_{t-1} x_{t-1}^T ----------
#     A2 = torch.zeros(B, m, m, device=x_smooth.device)
#
#     for t in range(T):
#         x_prev = x_prev_smooth[:, :, t]   # [B, m]
#         A2 += x_prev.unsqueeze(2) * x_prev.unsqueeze(1)
#
#     A2 = A2 / T
#
#     # ---------- S_Δx ----------
#     # delta_x: [B, m, T]
#     dx_mean   = delta_x.mean(dim=2, keepdim=True)   # [B, m, 1]  (mean over time)
#     dx_center = delta_x - dx_mean                   # [B, m, T]
#
#     Sdx = torch.zeros(B, m, m, device=delta_x.device)
#     for t in range(T):
#         dx_t = dx_center[:, :, t]                   # [B, m]
#         Sdx += dx_t.unsqueeze(2) * dx_t.unsqueeze(1)
#     Sdx = Sdx / T
#
#     # ---------- S_ν ----------
#     # nu: [B, n, T]
#     nu_mean   = nu.mean(dim=2, keepdim=True)        # [B, n, 1]
#     nu_center = nu - nu_mean                        # [B, n, T]
#
#     Sv = torch.zeros(B, n, n, device=nu.device)
#     for t in range(T):
#         nu_t = nu_center[:, :, t]                   # [B, n]
#         Sv += nu_t.unsqueeze(2) * nu_t.unsqueeze(1)
#     Sv = Sv / T
#
#
#     # ---------- C_{Δx, x-} = (1/T) Σ_t Δx_t x_{t-1}^T ----------
#     # delta_x: [B, m, T], x_prev_hat: [B, m, T]
#     Cdx = torch.zeros(B, m, m, device=delta_x.device)
#     for t in range(T):
#         dx_t    = delta_x[:, :, t]                  # [B, m]
#         xprev_t = x_prev_hat[:, :, t]               # [B, m]
#         Cdx += dx_t.unsqueeze(2) * xprev_t.unsqueeze(1)
#     Cdx = Cdx / T
#
#
#     return A1, A2, Sdx, Sv, Cdx



def mstep_loss(x_true, x_hat_list, F_last, F_true,
               alpha=(1.0, 1.0, 1.0), lambda_F=1e-3):
    """
    x_true:     [B, m, T]
    x_hat_list: list of 3 tensors [B, m, T]
    F_last:     [B, m, m]   F^(3)
    F_true:     [m, m] or [B, m, m]
    """

    # ----- state loss: Σ_k α_k * MSE(x_hat_k, x_true) -----
    state_loss = 0.0
    for k in range(3):
        err = x_hat_list[k] - x_true       # [B, m, T]
        loss_k = (err ** 2).mean()         # scalar
        state_loss = state_loss + alpha[k] * loss_k

    # ----- F loss: Frobenius norm between F_last and F_true -----
    if F_true.dim() == 2:
        # broadcast same true F to all batch elements
        F_true_b = F_true.unsqueeze(0).expand_as(F_last)   # [B, m, m]
    else:
        F_true_b = F_true                                  # assume [B, m, m]

    F_err = F_last - F_true_b                              # [B, m, m]
    # squared Frobenius per batch, then mean over batch
    F_loss = lambda_F * (F_err ** 2).sum(dim=[1, 2]).mean()

    total_loss = state_loss + F_loss
    return total_loss, state_loss, F_loss


# ============================================================
# 3. Training skeleton (shared M-step net, 3 EM iterations)
# ============================================================
