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
      - z_in: [B, d_z] where d_z = 6*m^2 + n^2
    Output:
      - ΔF:  [B, m, m]
    """
    def __init__(self, m, n, d_hidden=256, use_analytical_bias=False):
        super().__init__()
        self.m = m
        self.n = n
        self.use_analytical_bias = use_analytical_bias

        # ===== STATISTIC ENCODERS (separate processing) =====
        self.encode_A1 = nn.Sequential(
            nn.Linear(m*m, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.encode_A2 = nn.Sequential(
            nn.Linear(m*m, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.encode_S_delta = nn.Sequential(
            nn.Linear(m*m, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.encode_S_nu = nn.Sequential(
            nn.Linear(n*n, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.encode_C_delta = nn.Sequential(
            nn.Linear(m*m, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.encode_F_current = nn.Sequential(
            nn.Linear(m*m, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ===== NOISE ESTIMATOR =====
        # Learns: "How reliable are the statistics?"
        # Input: S_nu (observation noise covariance)
        # Output: reliability score [0, 1]
        self.g_min = 0.05  # floor so gate never kills learning

        # Gate uses two scalars: [fit_err, noise_level]
        self.step_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # ===== FUSION NETWORK =====
        # Combines all encoded statistics
        self.fusion = nn.Sequential(
            nn.Linear(6 * d_hidden, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ===== DELTA-F PREDICTOR =====
        # Two branches: base correction + residual refinement
        self.base_correction = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, m * m)
        )

        self.residual_refinement = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, m * m)
        )



    def forward(self, z_in):
        """
        z_in: [B, 5*m² + n²] = [A1, A2, S_delta, S_nu, C_delta, F_current]

        Returns: deltaF [B, m, m]
        """
        B = z_in.shape[0]
        m = self.m
        n = self.n


        # ===== PARSE INPUT =====
        idx = 0
        A1_flat = z_in[:, idx:idx + m*m]
        idx += m*m

        A2_flat = z_in[:, idx:idx + m*m]
        idx += m*m

        S_delta_flat = z_in[:, idx:idx + m*m]
        idx += m*m

        S_nu_flat = z_in[:, idx:idx + n*n]
        idx += n*n

        C_delta_flat = z_in[:, idx:idx + m*m]
        idx += m*m

        F_current_flat = z_in[:, idx:idx + m*m]

        # Compute analytical solution: F_analytical = A1 @ inv(A2)
        A1_mat = A1_flat.view(B,m, m)
        A2_mat = A2_flat.view(B,m, m)

        # Add regularization for stability
        I = torch.eye(m, device=A2_mat.device, dtype=A2_mat.dtype).unsqueeze(0)  # [1,m,m]

        A2_reg = A2_mat + 1e-3 * I
        F_analytical = torch.linalg.solve(A2_reg.transpose(-1, -2),A1_mat.transpose(-1, -2)).transpose(-1, -2)

        F_current_mat = F_current_flat.view(B, m, m)
        # Analytical correction
        analytical_deltaF = F_analytical - F_current_mat
        analytical_deltaF_flat = analytical_deltaF.view(B, -1)

        ################################################################################################################
        eps = 1e-6
        F_cur = F_current_flat.view(B, m, m)
        Snu_mat = S_nu_flat.view(B, n, n)

        # 1) noise magnitude (big when noise is big)
        noise_level = torch.log(torch.diagonal(Snu_mat, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True) + eps)  # [B,1]

        # 2) fit error (big when F doesn't fit the data)
        mismatch = A1_mat - torch.bmm(F_cur, A2_mat)  # [B,m,m]
        fit_err = torch.log((mismatch ** 2).mean(dim=(1, 2), keepdim=True) + eps).squeeze(-1)   # [B,1]

        # gate in [g_min, 1]
        gate_in = torch.cat([fit_err.detach(), noise_level.detach()], dim=1)  # [B,2]
        g = self.step_gate(gate_in)  # [B,1] in [0,1]
        g = self.g_min + (1.0 - self.g_min) * g
        ################################################################################################################

        # ===== ENCODE EACH STATISTIC =====
        enc_A1 = self.encode_A1(A1_flat)          # [B, d_hidden]
        enc_A2 = self.encode_A2(A2_flat)
        enc_S_delta = self.encode_S_delta(S_delta_flat)
        enc_S_nu = self.encode_S_nu(S_nu_flat)
        enc_C_delta = self.encode_C_delta(C_delta_flat)
        # enc_F_current = self.encode_F_current(F_current_flat)
        enc_F_current = self.encode_F_current(analytical_deltaF_flat)

        # ===== FUSE ALL STATISTICS =====
        fused = torch.cat([
            enc_A1, enc_A2, enc_S_delta,
            enc_S_nu, enc_C_delta, enc_F_current
        ], dim=1)  # [B, 6*d_hidden]

        fused = self.fusion(fused)  # [B, d_hidden]

        # ===== PREDICT DELTA-F (two branches) =====
        base = self.base_correction(fused)          # [B, m²]
        residual = self.residual_refinement(fused)  # [B, m²]

        # Combine with residual connection
        deltaF_flat = (base + 0.1 * residual)* g


        deltaF_flat =  deltaF_flat


        # Reshape to matrix
        deltaF = deltaF_flat.view(B, m, m)

        return deltaF