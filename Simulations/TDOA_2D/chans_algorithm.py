"""
Chan's algorithm for 2D TDOA source localization + linear KF/RTS smoother.

Reference: Chan & Ho, "A Simple and Efficient Estimator for Hyperbolic Location",
           IEEE Trans. Signal Processing, 1994.

For collinear microphone arrays (all mics share the same y-coordinate), the
standard linear system loses the p_y component. This implementation recovers
p_y from the estimated range R_0 to the reference microphone.
"""

import torch


def chans_position_and_cov(tdoa, mics, r2, c=1.0):
    """
    Chan's Stage-1 TDOA localization.

    tdoa : [n_tdoa]  TDOA measurements  = (dist_i - dist_0) / c
    mics : [M, 2]    mic positions; mics[0] is the reference
    r2   : float     TDOA noise variance per channel
    c    : float     speed of sound

    Returns
    -------
    pos     : [2]    estimated [p_x, p_y]
    cov_pos : [2, 2] position covariance (from WLS propagation)
    """
    device, dtype = tdoa.device, tdoa.dtype
    r0 = mics[0]
    d  = tdoa * c           # range differences  [n]
    n  = len(d)

    # K_i = ||r_i||^2 - ||r_0||^2  (known scalar per mic)
    K = torch.stack([(mics[i + 1] ** 2).sum() - (r0 ** 2).sum() for i in range(n)])

    # Linear system columns:  -2*(r_i - r_0)  and  -2*d_i  (coefficient of R_0)
    dx = torch.stack([-2.0 * (mics[i + 1][0] - r0[0]) for i in range(n)])
    dy = torch.stack([-2.0 * (mics[i + 1][1] - r0[1]) for i in range(n)])
    b  = d ** 2 - K

    # WLS weights: Var(b_i) ≈ 4*r2*d_i^2  (first-order noise propagation)
    w = 1.0 / (4.0 * r2 * d ** 2 + 1e-6)
    W = torch.diag(w)

    collinear = (dy.abs() < 1e-6).all()

    if collinear:
        # All mics at same y → dy column is zero → solve for [p_x, R_0]
        # then recover p_y = sqrt(R_0^2 - (p_x - r_0x)^2)
        A   = torch.stack([dx, -2.0 * d], dim=1)               # [n, 2]
        reg = 1e-7 * torch.eye(2, dtype=dtype, device=device)
        AtWA = A.T @ W @ A + reg
        z    = torch.linalg.solve(AtWA, A.T @ (W @ b))         # [p_x, R_0]
        Cov_z = torch.linalg.inv(AtWA)

        px  = z[0]
        R0  = z[1]
        py2 = R0 ** 2 - (px - r0[0]) ** 2
        py  = torch.sqrt(torch.clamp(py2, min=0.01))           # enforce py > 0
        pos = torch.stack([px, py])

        # Propagate covariance [p_x, R_0] → [p_x, p_y] via Jacobian
        J = torch.zeros(2, 2, dtype=dtype, device=device)
        J[0, 0] = 1.0
        J[1, 0] = -(px - r0[0]) / py
        J[1, 1] = R0 / py
        cov_pos = J @ Cov_z @ J.T
    else:
        A   = torch.stack([dx, dy, -2.0 * d], dim=1)           # [n, 3]
        reg = 1e-7 * torch.eye(3, dtype=dtype, device=device)
        AtWA = A.T @ W @ A + reg
        z    = torch.linalg.solve(AtWA, A.T @ (W @ b))
        Cov_z = torch.linalg.inv(AtWA)
        pos     = z[:2]
        cov_pos = Cov_z[:2, :2]

    return pos, cov_pos


def run_chans_smoother(Y, mics, F_kf, Q_kf, m1x_0, m2x_0, r2, c=1.0):
    """
    Two-stage TDOA tracker: Chan's per time step then linear KF + RTS smoother.

    Y      : [n_tdoa, T]  TDOA observations
    mics   : [M, 2]       mic positions
    F_kf   : [m, m]       state transition matrix for KF
    Q_kf   : [m, m]       process noise covariance
    m1x_0  : [m] or [m,1] initial state mean
    m2x_0  : [m, m]       initial covariance
    r2     : float        TDOA noise variance
    c      : float        speed of sound

    Returns
    -------
    x_smooth : [m, T]   smoothed state estimates
    x_last   : [m]      last filtered state  (for carry-over across datasets)
    P_last   : [m, m]   last filtered covariance
    """
    T      = Y.shape[-1]
    device = Y.device
    dtype  = Y.dtype
    m_dim  = F_kf.shape[0]
    I      = torch.eye(m_dim, dtype=dtype, device=device)

    # Observation matrix: directly observe [p_x, p_y]
    H = torch.zeros(2, m_dim, dtype=dtype, device=device)
    H[0, 0] = 1.0
    H[1, 1] = 1.0

    x = m1x_0.reshape(-1).to(dtype=dtype, device=device).clone()
    P = m2x_0.reshape(m_dim, m_dim).to(dtype=dtype, device=device).clone()

    x_filt = torch.zeros(m_dim, T, dtype=dtype, device=device)
    P_filt = torch.zeros(m_dim, m_dim, T, dtype=dtype, device=device)

    # ── Forward KF ──────────────────────────────────────────────────────────────
    for t in range(T):
        pos_est, R_pos = chans_position_and_cov(Y[:, t], mics, r2, c)

        x_pred = F_kf @ x
        P_pred = F_kf @ P @ F_kf.T + Q_kf

        S = H @ P_pred @ H.T + R_pos
        K = P_pred @ H.T @ torch.linalg.inv(S)
        x = x_pred + K @ (pos_est - H @ x_pred)
        P = (I - K @ H) @ P_pred
        P = (P + P.T) / 2          # keep symmetric

        x_filt[:, t] = x
        P_filt[:, :, t] = P

    x_last = x_filt[:, -1].clone()
    P_last = P_filt[:, :, -1].clone()

    # ── RTS Smoother ─────────────────────────────────────────────────────────────
    x_smooth = x_filt.clone()
    P_smooth = P_filt.clone()

    for t in range(T - 2, -1, -1):
        P_pred = F_kf @ P_filt[:, :, t] @ F_kf.T + Q_kf
        G = P_filt[:, :, t] @ F_kf.T @ torch.linalg.inv(P_pred)
        x_smooth[:, t] = x_filt[:, t] + G @ (x_smooth[:, t + 1] - F_kf @ x_filt[:, t])
        P_smooth[:, :, t] = P_filt[:, :, t] + G @ (P_smooth[:, :, t + 1] - P_pred) @ G.T

    return x_smooth, x_last, P_last
