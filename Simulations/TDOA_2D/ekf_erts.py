"""
EKF forward pass + ERTS backward pass for the 2-D TDOA experiment.

Kept in a dedicated module so that the main scripts stay clean and so
we never drag in the Lorenz-specific EKF / Extended_RTS_Smoother classes
(which hard-code m=3 and import getJacobian from the Lorenz parameters).
"""

import torch
import torch.nn as nn

from Simulations.TDOA_2D.parameters import (
    m, n, Q, R,
    m1x_0, m2x_0,
    h, h_jacobian,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_loss = nn.MSELoss(reduction='mean')


# ── Cross-covariance (kept here to avoid the Lorenz import chain) ─────────────

def compute_cross_covariances(
    F: torch.Tensor,
    H_last: torch.Tensor,
    K_last: torch.Tensor,
    P_filt: torch.Tensor,
    Sgains: list,
) -> torch.Tensor:
    """
    Lag-1 cross-covariances V[:,:,t] = Cov(x_t, x_{t-1} | Y).

    Parameters
    ----------
    F       : [m, m]   state-transition matrix (constant, e.g. current EM estimate)
    H_last  : [n, m]   measurement Jacobian at t = T-1
    K_last  : [m, n]   Kalman gain at t = T-1
    P_filt  : [m, m, T]  filtered covariances P_{t|t}
    Sgains  : list[T]  smoother gains; Sgains[k] = J at t = T-2-k,
                        Sgains[T-1] = extra gain for t = -1

    Returns
    -------
    V : [m, m, T]
    """
    m, _, T = P_filt.shape
    V = torch.zeros(m, m, T, dtype=P_filt.dtype, device=P_filt.device)
    I = torch.eye(m, dtype=P_filt.dtype, device=P_filt.device)

    # Initialise at t = T-1
    V[:, :, T - 1] = (I - K_last @ H_last) @ F @ P_filt[:, :, T - 2]

    for t in range(T - 2, -1, -1):
        Pt     = P_filt[:, :, t]
        St     = Sgains[T - 2 - t]      # J_t
        Stm1_T = Sgains[T - 1 - t]     # J_{t-1}
        V[:, :, t] = (Pt @ Stm1_T.T
                      + St @ (V[:, :, t + 1] - F @ Pt) @ Stm1_T.T)
    return V


# ── Single-sequence EKF + ERTS ────────────────────────────────────────────────

def run_ekf_erts(y_seq: torch.Tensor, get_F,
                 Q_in: torch.Tensor = None,
                 R_in: torch.Tensor = None,
                 x_init: torch.Tensor = None,
                 P_init: torch.Tensor = None) -> tuple:
    """
    One EKF forward pass followed by one ERTS backward pass.

    Parameters
    ----------
    y_seq  : [n_obs, T_len]   noisy observations
    get_F  : callable(t: int) -> [m_state, m_state]
    Q_in   : [m, m]  process noise covariance (defaults to module-level Q)
    R_in   : [n, n]  measurement noise covariance (defaults to module-level R)
    x_init : [m] or [m,1]  initial state mean   (defaults to m1x_0)
    P_init : [m, m]        initial state covariance (defaults to m2x_0)

    Returns
    -------
    x_smooth : [m, T]
    P_smooth : [m, m, T]
    P_filt   : [m, m, T]
    sgains   : list[T]  smoother gains (Sgains[0] = J at t=T-2, …,
                         Sgains[T-1] = extra J_{-1})
    H_last   : [n, m]
    K_last   : [m, n]
    """
    Q_use  = Q if Q_in is None else Q_in
    R_use  = R if R_in is None else R_in
    P0_use = m2x_0.to(device) if P_init is None else P_init.to(device)

    T_len = y_seq.shape[1]

    x_f   = torch.zeros(m, T_len, device=device)
    P_f   = torch.zeros(m, m, T_len, device=device)
    F_seq = torch.zeros(m, m, T_len, device=device)
    I_m   = torch.eye(m, device=device)

    x_p = m1x_0.reshape(-1).clone().to(device) if x_init is None else x_init.reshape(-1).clone().to(device)
    P_p = P0_use.clone()
    H_last = None
    K_last = None

    # ── EKF forward ──────────────────────────────────────────────────────────
    for t in range(T_len):
        F_t = get_F(t).to(device)
        F_seq[:, :, t] = F_t

        xpr = F_t @ x_p
        Ppr = F_t @ P_p @ F_t.T + Q_use

        H_t = h_jacobian(xpr)
        S_t = H_t @ Ppr @ H_t.T + R_use
        K_t = Ppr @ H_t.T @ torch.linalg.inv(S_t)

        innov = y_seq[:, t] - h(xpr).reshape(-1)
        x_p   = xpr + K_t @ innov
        P_p   = (I_m - K_t @ H_t) @ Ppr
        P_p   = (P_p + P_p.T) / 2

        x_f[:, t]    = x_p
        P_f[:, :, t] = P_p
        H_last = H_t
        K_last = K_t

    # ── ERTS backward ─────────────────────────────────────────────────────────
    x_s    = x_f.clone()
    P_s    = P_f.clone()
    sgains = []

    for t in range(T_len - 2, -1, -1):
        F_tp1   = F_seq[:, :, t + 1]
        xpr_tp1 = F_tp1 @ x_f[:, t]
        Ppr_tp1 = F_tp1 @ P_f[:, :, t] @ F_tp1.T + Q_use

        J_t = P_f[:, :, t] @ F_tp1.T @ torch.linalg.inv(Ppr_tp1)
        sgains.append(J_t.clone())

        x_s[:, t]    = x_f[:, t] + J_t @ (x_s[:, t + 1] - xpr_tp1)
        dP           = J_t @ (P_s[:, :, t + 1] - Ppr_tp1) @ J_t.T
        P_s[:, :, t] = P_f[:, :, t] + dP
        P_s[:, :, t] = (P_s[:, :, t] + P_s[:, :, t].T) / 2

    # Extra smoother gain for cross-covariance recursion at t = -1
    F_0   = F_seq[:, :, 0]
    Ppr_0 = F_0 @ P0_use @ F_0.T + Q_use
    J_m1  = P0_use @ F_0.T @ torch.linalg.inv(Ppr_0)
    sgains.append(J_m1.clone())

    return x_s, P_s, P_f, sgains, H_last, K_last


# ── Batch EKF + ERTS ──────────────────────────────────────────────────────────

def run_batch_erts(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    get_F_fn,
    Q_in: torch.Tensor = None,
    R_in: torch.Tensor = None,
) -> tuple:
    """
    Run EKF+ERTS for every sequence in `inputs`.

    Parameters
    ----------
    inputs  : [N, n_obs, T]
    targets : [N, m_state, T]
    get_F_fn: callable(t: int) -> [m_state, m_state]

    Returns
    -------
    mean_mse : float
    X_s_all  : [N, m, T]
    P_s_all  : [N, m, m, T]
    V_all    : [N, m, m, T]
    F_ref    : [m, m]   F used for cross-covariance (= get_F_fn(0))
    """
    N     = inputs.shape[0]
    T_len = inputs.shape[2]

    X_s_all = torch.zeros(N, m, T_len, device=device)
    P_s_all = torch.zeros(N, m, m, T_len, device=device)
    V_all   = torch.zeros(N, m, m, T_len, device=device)
    mse_arr = torch.zeros(N)

    F_ref = get_F_fn(0)

    for j in range(N):
        y_j = inputs[j].to(device)
        x_j = targets[j].to(device)

        x_s, P_s, P_f, sg, H_l, K_l = run_ekf_erts(y_j, get_F_fn, Q_in=Q_in, R_in=R_in)
        V_j = compute_cross_covariances(F_ref, H_l, K_l, P_f, sg)

        X_s_all[j] = x_s
        P_s_all[j] = P_s
        V_all[j]   = V_j
        mse_arr[j] = _loss(x_s, x_j).item()

    return mse_arr.mean().item(), X_s_all, P_s_all, V_all, F_ref
