import torch
import torch.nn as nn

# ----- fixed device (simple) -----
device = torch.device("cuda")

# ----- your PSD projector (kept as-is style) -----
def enforce_covariance_properties(P, eps=1e-6):
    # Ensure P is symmetric positive semidefinite
    P = (P + P.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(P)
    if torch.any(eigenvalues.real < 0):
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        P = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return P
def standardize(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)


class PNotSmoothNN(nn.Module):
    """
    Estimates P_not_smooth_t from:
      - F_t    : [m, m]
      - K_t    : [m, n]
      - P_prev : [m, m]  (previous P_not_smooth)

    Input to GRU is: [vec(F_t), vec(K_t), vec(P_prev)]  (no innovation)
    GRU hidden size = 4*m^2, head maps to m^2, then reshape to [m, m] and PSD-enforce.
    """
    def __init__(self, m, n, p_0):
        super().__init__()
        self.m = m
        self.n = n
        self.start = 0
        self.p_0 = p_0  # tensor of shape [m, m]

        # dims
        self.d_in  = (m*m) + (m*n) + (m*m)      # F + K + P_prev
        self.d_hid = 4 * (m*m)                  # your choice; keep as you set

        # layers
        self.ln   = nn.LayerNorm(self.d_in)
        self.gru  = nn.GRU(input_size=self.d_in, hidden_size=self.d_hid, batch_first=True)
        self.head = nn.Linear(self.d_hid, m*m)

        # GRU hidden state
        self.h = None
        self.F = None

    def reset_state(self):
        self.h = None
        self.start = 0

    def forward(self, K_t, P_prev):
        """
        F_t : [m,m]
        K_t : [m,n]
        P_prev : [m,m]   (tip: feed PSD(P_prev).detach() when rolling during training)
        returns: P_not_t [m,m]
        """
        # build input [1,1,d_in]
        F_t = self.F

        # scale = P_prev.norm(p='fro').clamp_min(1e-6)
        # P_prev = P_prev / scale




        # x = torch.cat([F_t.reshape(-1),K_t.reshape(-1),P_prev.reshape(-1)], dim=0).unsqueeze(0).unsqueeze(0)   # [B=1, T=1, d_in]
        P_prev = P_prev.detach()
        F_t = F_t.detach()

        x = torch.cat([standardize(F_t.reshape(-1)), standardize(K_t.reshape(-1)), standardize(P_prev.reshape(-1))], dim=0).unsqueeze(0).unsqueeze(0)

        x = self.ln(x)

        # first-step hidden init: vec(P0) padded with zeros to d_hid
        if self.start == 0:
            P0_vec = self.p_0.reshape(1, 1, -1)                         # [1,1,m^2]
            pad = torch.zeros(1, 1, self.d_hid - P0_vec.shape[-1],
                              device=P0_vec.device, dtype=P0_vec.dtype)  # [1,1,d_hid - m^2]
            self.h = torch.cat([P0_vec, pad], dim=-1)                    # [1,1,d_hid]
            self.start = 1

        out, self.h = self.gru(x, self.h)          # out: [1,1,d_hid]
        P_vec = self.head(out).squeeze(0).squeeze(0)  # [m^2]
        P = P_vec.view(self.m, self.m)             # [m,m]
        P = enforce_covariance_properties(P)
        return P

    def compute_loss(self, P_pred_seq, x_target, x_not_smooth):
        """
        Same as your Psmooth loss but with x_not_smooth.

        P_pred_seq   : [m, m, T]
        x_target     : [m, T]
        x_not_smooth : [m, T]
        """
        m, T = x_target.shape
        loss = 0.0
        for t in range(T):
            err = (x_target[:, t] - x_not_smooth[:, t]).unsqueeze(1)  # [m,1]
            P_true = err @ err.T                                      # [m,m]
            P_pred = P_pred_seq[:, :, t]                              # [m,m]
            loss = loss + torch.norm(P_pred - P_true, p='fro')**2
        return loss / T

class PsmoothFromPnot(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.start = 0
        self.m = m

        # Input = vec(P_not) || vec(SGain)  -> size = 2*m^2
        self.d_input_Psmooth  = 2 * (m * m)
        self.d_hidden_Psmooth = m * m*4

        # GRU expects input as [seq_len, batch, feat] (batch_first=False by default)
        self.GRU_Psmooth = nn.GRU(self.d_input_Psmooth, self.d_hidden_Psmooth)

        # Use only LayerNorm on the concatenated feature vector
        self.layernorm_Psmooth = nn.LayerNorm(self.d_input_Psmooth)

        # Map GRU hidden -> vec(P_smooth)
        self.FC_Psmooth = nn.Linear(self.d_hidden_Psmooth, m * m)

        # GRU hidden state
        self.h_Psmooth = None

    def reset_state(self):
        self.h_Psmooth = None
        self.start = 0

    def forward(self, P_not_t, SGain_t):
        """
        P_not_t : [m, m]
        SGain_t : [m, m]
        returns P_smooth : [m, m]
        """
        p_flat = P_not_t.view(1, 1, -1)   # [seq=1, batch=1, m^2]
        s_flat = SGain_t.view(1, 1, -1)   # [1,1,m^2]
        input_l = torch.cat((p_flat, s_flat), dim=2)   # [1,1,2*m^2]

        input_i = self.layernorm_Psmooth(input_l)

        # seed hidden once from the first m^2 features (your style)
        if self.start == 0:
            self.h_Psmooth = input_i[:, :, :self.d_hidden_Psmooth].clone()  # [1,1,m^2]
            self.start = 1

        out, self.h_Psmooth = self.GRU_Psmooth(input_i, self.h_Psmooth)     # out: [1,1,m^2]
        P_vec = self.FC_Psmooth(out).squeeze(0).squeeze(0)            # [m^2]
        P = P_vec.view(self.m, self.m)
        P = enforce_covariance_properties(P)
        return P
    def compute_loss(self, P_pred_seq, x_target, x_not_smooth):
        """
        Same as your Psmooth loss but with x_not_smooth.

        P_pred_seq   : [m, m, T]
        x_target     : [m, T]
        x_not_smooth : [m, T]
        """
        m, T = x_target.shape
        loss = 0.0
        for t in range(T):
            err = (x_target[:, t] - x_not_smooth[:, t]).unsqueeze(1)  # [m,1]
            P_true = err @ err.T                                      # [m,m]
            P_pred = P_pred_seq[:, :, t]                              # [m,m]
            loss = loss + torch.norm(P_pred - P_true, p='fro')**2
        return loss / T