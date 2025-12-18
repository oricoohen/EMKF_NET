import torch
import torch.nn as nn

class DeltaF_MStepNet_Minimal(nn.Module):
    """
    MINIMAL M-step network:
      1. Compute analytical: F_analytical = A1 @ inv(A2)
      2. Predict confidence: α = MLP(quality_features) in [0, 1]
      3. Output: ΔF = α * analytical

    The network ONLY decides how much to scale analytical, cannot ignore it!
    """
    def __init__(self, m, n, d_hidden=64):
        super().__init__()
        self.m = m
        self.n = n

        # Input: 5 quality scalars (including C_delta cross-covariance)
        quality_dim = 5

        self.confidence_net = nn.Sequential(
            nn.Linear(quality_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, z_in):
        """
        z_in: [B, 5*m² + n²] = [A1, A2, S_delta, S_nu, C_delta, F_current]
        Returns: deltaF [B, m, m]
        """
        B = z_in.shape[0]
        m = self.m
        n = self.n

        # Parse input
        idx = 0
        A1_flat = z_in[:, idx:idx + m*m]; idx += m*m
        A2_flat = z_in[:, idx:idx + m*m]; idx += m*m
        S_delta_flat = z_in[:, idx:idx + m*m]; idx += m*m
        S_nu_flat = z_in[:, idx:idx + n*n]; idx += n*n
        C_delta_flat = z_in[:, idx:idx + m*m]; idx += m*m
        F_current_flat = z_in[:, idx:idx + m*m]

        A1_mat = A1_flat.view(B, m, m)
        A2_mat = A2_flat.view(B, m, m)
        F_current_mat = F_current_flat.view(B, m, m)

        # Compute analytical solution
        I = torch.eye(m, device=A2_mat.device, dtype=A2_mat.dtype)
        A2_reg = A2_mat + 1e-3 * I  # Broadcasting works automatically [B,m,m] + [m,m]
        F_analytical = torch.linalg.solve(A2_reg.transpose(-1, -2), A1_mat.transpose(-1, -2)).transpose(-1, -2)
        analytical_deltaF = F_analytical - F_current_mat

        # Compute quality indicators
        eps = 1e-6

        # 1. Fit error
        mismatch = A1_mat - torch.bmm(F_current_mat, A2_mat)
        fit_err = torch.log((mismatch ** 2).mean(dim=(1, 2)) + eps).unsqueeze(-1)  # [B, 1]

        # 2. Noise level
        Snu_mat = S_nu_flat.view(B, n, n)
        noise_level = torch.log(torch.diagonal(Snu_mat, dim1=-2, dim2=-1).mean(dim=-1) + eps).unsqueeze(-1)  # [B, 1]

        # 3. State uncertainty
        S_delta_mat = S_delta_flat.view(B, m, m)
        state_uncertainty = torch.log(torch.diagonal(S_delta_mat, dim1=-2, dim2=-1).mean(dim=-1) + eps).unsqueeze(-1)  # [B, 1]

        # 4. Analytical magnitude (how big is the correction?)
        analytical_norm = torch.log((analytical_deltaF ** 2).sum(dim=(1, 2)).sqrt() + eps).unsqueeze(-1)  # [B, 1]

        # 5. Cross-covariance magnitude: How do innovations correlate with predictions?
        C_delta_mat = C_delta_flat.view(B, m, m)
        cross_cov_mag = torch.log((C_delta_mat ** 2).sum(dim=(1, 2)) + eps).unsqueeze(-1)  # [B, 1]

        # Concatenate quality features
        quality_features = torch.cat([fit_err, noise_level, state_uncertainty, analytical_norm, cross_cov_mag], dim=1)  # [B, 5]

        # Predict confidence (how much to trust analytical)
        confidence = self.confidence_net(quality_features)  # [B, 1]

        # Final output: scale analytical by confidence
        deltaF = confidence.unsqueeze(-1) * analytical_deltaF  # [B, 1, 1] * [B, m, m]

        return deltaF

