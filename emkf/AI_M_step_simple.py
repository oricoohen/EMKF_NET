import torch
import torch.nn as nn

class DeltaF_MStepNet_Simple(nn.Module):
    """
    SIMPLIFIED M-step network:
      1. Compute analytical: F_analytical = A1 @ inv(A2)
      2. Predict small correction: residual = MLP(quality_features)
      3. Output: ΔF = analytical + residual

    This FORCES the network to use analytical as base!
    """
    def __init__(self, m, n, d_hidden=128, correction_scale=0.3):
        super().__init__()
        self.m = m
        self.n = n
        self.correction_scale = correction_scale  # How much correction can deviate from analytical

        # Only encode QUALITY indicators, not raw statistics
        # Network decides how much to trust/correct analytical
        quality_dim = 5  # fit_err, noise_level, state_uncertainty, obs_uncertainty, cross_cov_mag

        self.correction_net = nn.Sequential(
            nn.Linear(quality_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, m * m)
        )

        # Small initialization so network starts by trusting analytical
        with torch.no_grad():
            self.correction_net[-1].weight.data *= 0.01
            self.correction_net[-1].bias.data.zero_()

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

        # Compute quality indicators (scalars that indicate reliability)
        eps = 1e-6

        # 1. Fit error: How well does F_current fit the data?
        mismatch = A1_mat - torch.bmm(F_current_mat, A2_mat)
        fit_err = torch.log((mismatch ** 2).mean(dim=(1, 2)) + eps).unsqueeze(-1)  # [B, 1]

        # 2. Noise level: How noisy are observations?
        Snu_mat = S_nu_flat.view(B, n, n)
        noise_level = torch.log(torch.diagonal(Snu_mat, dim1=-2, dim2=-1).mean(dim=-1) + eps).unsqueeze(-1)  # [B, 1]

        # 3. State uncertainty: How uncertain is smoothing?
        S_delta_mat = S_delta_flat.view(B, m, m)
        state_uncertainty = torch.log(torch.diagonal(S_delta_mat, dim1=-2, dim2=-1).mean(dim=-1) + eps).unsqueeze(-1)  # [B, 1]

        # 4. Observation uncertainty trace
        obs_uncertainty = torch.log(torch.diagonal(Snu_mat, dim1=-2, dim2=-1).sum(dim=-1) + eps).unsqueeze(-1)  # [B, 1]

        # 5. Cross-covariance magnitude: How do innovations correlate with predictions?
        C_delta_mat = C_delta_flat.view(B, m, m)
        cross_cov_mag = torch.log((C_delta_mat ** 2).sum(dim=(1, 2)) + eps).unsqueeze(-1)  # [B, 1]

        # Concatenate quality features
        quality_features = torch.cat([fit_err, noise_level, state_uncertainty, obs_uncertainty, cross_cov_mag], dim=1)  # [B, 5]

        # Predict small correction
        correction_flat = self.correction_net(quality_features)  # [B, m²]
        correction_mat = correction_flat.view(B, m, m)

        # Final output: analytical + learned correction
        # correction_scale controls how much we can deviate:
        #   0.1 = stay very close to analytical (when analytical is usually good)
        #   0.5 = allow moderate corrections (when analytical has systematic errors)
        #   1.0 = full correction freedom (when analytical is often wrong)
        deltaF = analytical_deltaF + self.correction_scale * correction_mat

        # Store debug info
        self._debug_analytical_norm = analytical_deltaF.norm().item() if analytical_deltaF.numel() > 0 else 0.0
        self._debug_correction_norm = correction_mat.norm().item() if correction_mat.numel() > 0 else 0.0

        return deltaF

