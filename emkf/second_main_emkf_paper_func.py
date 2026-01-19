import torch
from Smoothers.RTS_Smoother_test import S_Test
# ============================================================
# Helpers (paper-style positivity projection = ReLU)
# ============================================================
device = torch.device("cuda")
@torch.no_grad()
def relu_proj_(A):
    return A.clamp_min_(0.0)

@torch.no_grad()
def factorize_3_als_relu(M_hat, A0, A1, A2, n_sweeps=1, eps=1e-7):
    """
    Factorize M_hat ≈ A0 A1 A2 via ALS sweeps with ReLU projection.
    Shapes:
      M_hat: [r, c]
      A0:    [r, k1]
      A1:    [k1, k2]
      A2:    [k2, c]
    """
    for _ in range(n_sweeps):
        # A0 <- M_hat pinv(A1 A2)
        R = A1 @ A2
        A0.copy_(M_hat @ torch.linalg.pinv(R))
        relu_proj_(A0)

        # A1 <- pinv(A0) M_hat pinv(A2)
        A1.copy_(torch.linalg.pinv(A0) @ M_hat @ torch.linalg.pinv(A2))
        relu_proj_(A1)

        # A2 <- pinv(A0 A1) M_hat
        L = A0 @ A1
        A2.copy_(torch.linalg.pinv(L) @ M_hat)
        relu_proj_(A2)

    return A0, A1, A2

# ============================================================
# Sufficient statistics (paper Eq. 11) using YOUR RTS outputs
# ============================================================
@torch.no_grad()
def compute_stats_decrypt(X_s, P_s, V_s, Y, U_in=None):
    """
    Args:
      X_s:  [m, T]     smoothed state means
      P_s:  [m, m, T]  smoothed covariances
      V_s:  [m, m, T]  lag-1 cross-covariances (your RTS output, aligned like your compute_A1)
      Y:    [n, T]     measurements
      U_in: [p, T]     control input (optional)

    Returns dict with:
      Rz: [m,m] = (1/T) Σ (x_t x_t^T + P_t)
      Uz: [m,m] = (1/T) Σ (x_{t-1} x_{t-1}^T + P_{t-1})
      Cxz: [m,m] = (1/T) Σ (x_t x_{t-1}^T + V_t)     = E[x_t x_{t-1}^T]
      YX: [n,m] = (1/T) Σ y_t x_t^T
      UU: [p,p] = (1/T) Σ u_t u_t^T                 (if U_in)
      XU: [m,p] = (1/T) Σ x_t u_t^T                 (if U_in)
      XprevU: [m,p] = (1/T) Σ x_{t-1} u_t^T         (if U_in)
    """
    device = X_s.device
    dtype = X_s.dtype
    m, T = X_s.shape
    n = Y.shape[0]

    Rz = torch.zeros((m, m), device=device, dtype=dtype)
    Uz = torch.zeros((m, m), device=device, dtype=dtype)
    Cxz = torch.zeros((m, m), device=device, dtype=dtype)
    YX = torch.zeros((n, m), device=device, dtype=dtype)

    for t in range(T):
        xt = X_s[:, t].unsqueeze(1)  # [m,1]
        Rz += xt @ xt.T + P_s[:, :, t]
        YX += Y[:, t].unsqueeze(1) @ xt.T

    for t in range(1, T):
        xprev = X_s[:, t-1].unsqueeze(1)
        Uz += xprev @ xprev.T + P_s[:, :, t-1]
        # your V_s[:,:,t] matches your compute_A1 loop (t=1..T-1)
        Cxz += X_s[:, t].unsqueeze(1) @ X_s[:, t-1].unsqueeze(0) + V_s[:, :, t]

    Rz = Rz / T
    Uz = Uz / max(T-1, 1)
    Cxz = Cxz / max(T-1, 1)
    YX = YX / T

    out = {"Rz": Rz, "Uz": Uz, "Cxz": Cxz, "YX": YX}

    if U_in is not None:
        p = U_in.shape[0]
        UU = torch.zeros((p, p), device=device, dtype=dtype)
        XU = torch.zeros((m, p), device=device, dtype=dtype)
        XprevU = torch.zeros((m, p), device=device, dtype=dtype)

        for t in range(T):
            ut = U_in[:, t].unsqueeze(1)  # [p,1]
            xt = X_s[:, t].unsqueeze(1)   # [m,1]
            UU += ut @ ut.T
            XU += xt @ ut.T
            if t >= 1:
                xprev = X_s[:, t-1].unsqueeze(1)
                XprevU += xprev @ ut.T

        UU = UU / T
        XU = XU / T
        XprevU = XprevU / max(T-1, 1)

        out.update({"UU": UU, "XU": XU, "XprevU": XprevU})

    return out

# ============================================================
# Composite (linear EM) update for F,B,H together
# ============================================================
@torch.no_grad()
def update_composites_FBH(stats, eps=1e-7):
    """
    Returns:
      F_hat: [m,m]
      B_hat: [m,p] or None
      H_hat: [n,m]
    """
    Rz = stats["Rz"]
    H_hat = stats["YX"] @ torch.linalg.pinv(Rz + eps * torch.eye(Rz.shape[0], device=Rz.device, dtype=Rz.dtype))

    Uz = stats["Uz"]
    Cxz = stats["Cxz"]

    # If control exists, solve for [F B] jointly:
    # [F B] * [[Uz, XprevU],
    #          [XprevU^T, UU]] = [Cxz, XU]
    if "UU" in stats:
        XprevU = stats["XprevU"]
        UU = stats["UU"]
        XU = stats["XU"]

        top = torch.cat([Uz,      XprevU], dim=1)
        bot = torch.cat([XprevU.T, UU],    dim=1)
        S = torch.cat([top, bot], dim=0)

        RHS = torch.cat([Cxz, XU], dim=1)  # [m, m+p]

        # solve: [F B] = RHS * pinv(S)
        S_reg = S + eps * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        FB = RHS @ torch.linalg.pinv(S_reg)

        m = Uz.shape[0]
        p = UU.shape[0]
        F_hat = FB[:, :m]
        B_hat = FB[:, m:m+p]
    else:
        # No control: F_hat = Cxz * pinv(Uz)
        Uz_reg = Uz + eps * torch.eye(Uz.shape[0], device=Uz.device, dtype=Uz.dtype)
        F_hat = Cxz @ torch.linalg.pinv(Uz_reg)
        B_hat = None

    return F_hat, B_hat, H_hat

# ============================================================
# FULL DeCrypt-style EMKF: update factors for F,H,B together
# ============================================================
@torch.no_grad()
def EMKF_FHB_decrypt_style_seq_core(
    sys_model,
    Y_seq,            # [n, T]
    X_true_seq,       # [m, T]
    x0, P0,           # [m,1], [m,m]
    factors_init,
    U_seq=None,       # [p, T] or None
    max_it=3,
    n_sweeps_factor=1,
    update_F=True,
    update_H=True,
    update_B=True,
    H_fixed=None,
    F_fixed=None,
    B_fixed=None,
):
    """
    Single-sequence core.
    Returns:
      out_seq: dict with lists over EM iterations (len=max_it)
      x_T, P_T: last smoothed state/cov for chaining
    """

    # ---------- shapes ----------
    n, T = Y_seq.shape
    m = X_true_seq.shape[0]
    use_control = (U_seq is not None)

    # ---------- safety for fixed matrices ----------
    if not update_F:
        assert F_fixed is not None, "update_F=False requires F_fixed"
    if not update_H:
        assert H_fixed is not None, "update_H=False requires H_fixed"
    if use_control and (not update_B):
        assert B_fixed is not None, "update_B=False with control requires B_fixed"

    # ---------- local factor copies (for THIS sequence) ----------
    T10 = factors_init["T10"].clone()
    T11 = factors_init["T11"].clone()
    T12 = factors_init["T12"].clone()

    D0  = factors_init["D0"].clone()
    D1  = factors_init["D1"].clone()
    D2  = factors_init["D2"].clone()

    if use_control:
        T20 = factors_init["T20"].clone()
        T21 = factors_init["T21"].clone()
        T22 = factors_init["T22"].clone()
    else:
        T20 = T21 = T22 = None

    # ---------- histories over EM iterations ----------
    F_hist = []
    H_hist = []
    B_hist = [] if use_control else None

    T_factors_hist = []
    D_factors_hist = []
    B_factors_hist = [] if use_control else None
    mse_list = []

    # ---------- EM loop ----------
    for it in range(max_it):

        # ===== build composites from factors (or fixed) =====
        F_now = (T10 @ T11 @ T12) if update_F else F_fixed
        H_now = (D0 @ D1 @ D2)    if update_H else H_fixed

        if use_control:
            B_now = (T20 @ T21 @ T22) if update_B else B_fixed
        else:
            B_now = None

        # ===== E-step: RTS smoother =====
        sys_model.InitSequence(x0, P0)

        [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, V_smooth] = S_Test(sys_model,Y_seq.unsqueeze(0),X_true_seq.unsqueeze(0),F=F_now.unsqueeze(0),H=[H_now],
            generate_f=False,generate_h=False,init_x_list=[x0],init_P_list=[P0])
        mse_list.append(float(_mse_avg))
        X_s = X_smooth.squeeze(0)   # [m, T]
        P_s = P_smooth.squeeze(0)   # [m, m, T]
        V_s = V_smooth.squeeze(0)   # [m, m, T]

        # ===== stats =====
        stats = compute_stats_decrypt(X_s, P_s, V_s, Y_seq, U_in=U_seq)

        # ===== composite EM update =====
        # We compute all three, but only use the ones we update.
        F_tmp, B_tmp, H_tmp = update_composites_FBH(stats)

        if update_F:
            F_hat = F_tmp
        if update_H:
            H_hat = H_tmp
        if use_control and update_B:
            B_hat = B_tmp

        # ===== factorize back (paper style) or equivalence mode =====
        if update_F:
            if n_sweeps_factor > 0:
                T10, T11, T12 = factorize_3_als_relu(F_hat, T10, T11, T12, n_sweeps=n_sweeps_factor)
            else:
                I = torch.eye(T10.shape[0], device=T10.device, dtype=T10.dtype)
                T10.copy_(I); T11.copy_(I); T12.copy_(F_hat)

        if update_H:
            if n_sweeps_factor > 0:
                D0, D1, D2 = factorize_3_als_relu(H_hat, D0, D1, D2, n_sweeps=n_sweeps_factor)
            else:
                I_n = torch.eye(D0.shape[0], device=D0.device, dtype=D0.dtype)
                I_m = torch.eye(D2.shape[1], device=D2.device, dtype=D2.dtype)
                D0.copy_(I_n); D1.copy_(I_m); D2.copy_(H_hat)

        if use_control and update_B:
            if n_sweeps_factor > 0:
                T20, T21, T22 = factorize_3_als_relu(B_hat, T20, T21, T22, n_sweeps=n_sweeps_factor)
            else:
                I = torch.eye(T20.shape[0], device=T20.device, dtype=T20.dtype)
                T20.copy_(I); T21.copy_(I); T22.copy_(B_hat)

        # ===== store histories (post-update composites) =====
        F_hist.append(F_now.clone() if not update_F else (T10 @ T11 @ T12).clone())
        H_hist.append(H_now.clone() if not update_H else (D0 @ D1 @ D2).clone())

        if use_control:
            if update_B:
                B_hist.append((T20 @ T21 @ T22).clone())
            else:
                B_hist.append(B_now.clone())

        T_factors_hist.append((T10.clone(), T11.clone(), T12.clone()))
        D_factors_hist.append((D0.clone(), D1.clone(), D2.clone()))
        if use_control:
            B_factors_hist.append((T20.clone(), T21.clone(), T22.clone()))

    # ===== tail for chaining =====
    x_T = X_s[:, -1].unsqueeze(-1).clone()
    P_T = P_s[:, :, -1].clone()

    out_seq = {
        "F_list": F_hist,
        "H_list": H_hist,
        "B_list": B_hist,
        "T_factors": T_factors_hist,
        "D_factors": D_factors_hist,
        "B_factors": B_factors_hist,
        "mse_list":  mse_list

    }

    return out_seq, x_T, P_T



@torch.no_grad()
def EMKF_FHB_decrypt_style_batch(sys_model,Y,X_true,x_0, P_0,factors_init,U_in=None,max_it=3,n_sweeps_factor=1,init_x_list=None,init_P_list=None,update_F=True,
    update_H=True,update_B=True,H_fixed=None,F_fixed=None,B_fixed=None,F_true=None,H_true=None):
    N_seq, n, T = Y.shape
    last_x_list, last_P_list = [], []

    use_control = (U_in is not None)

    # safety for fixed matrices
    if not update_F:
        assert F_fixed is not None, "update_F=False requires F_fixed"
    if not update_H:
        assert H_fixed is not None, "update_H=False requires H_fixed"
    if use_control and (not update_B):
        assert B_fixed is not None, "update_B=False with control requires B_fixed"

    hist = {
        "F_list": [],
        "H_list": [],
        "T_factors": [],
        "D_factors": [],
        "B_list": [] if use_control else None,
        "B_factors": [] if use_control else None,
        "mse_state_list": [],
    }
    # accumulators for printing (mean across sequences)
    sum_mse_state = torch.zeros(max_it, device=device)
    sum_mse_F = torch.zeros(max_it, device=device) if (F_true is not None and update_F) else None
    sum_mse_H = torch.zeros(max_it, device=device) if (H_true is not None and update_H) else None


    for j in range(N_seq):
        # initials per sequence
        if init_x_list is not None:
            x0_j = init_x_list[j]
            P0_j = init_P_list[j]
        else:
            x0_j = x_0
            P0_j = P_0

        # ensure same device as Y
        x0_j = x0_j
        P0_j = P0_j

        U_j = U_in[j] if use_control else None

        out_j, x_T, P_T = EMKF_FHB_decrypt_style_seq_core(sys_model=sys_model,Y_seq=Y[j],X_true_seq=X_true[j],x0=x0_j,P0=P0_j,factors_init=factors_init,
        U_seq=U_j,max_it=max_it,n_sweeps_factor=n_sweeps_factor,update_F=update_F,update_H=update_H,update_B=update_B,H_fixed=H_fixed,F_fixed=F_fixed,B_fixed=B_fixed)

        hist["F_list"].append(out_j["F_list"])
        hist["H_list"].append(out_j["H_list"])
        hist["T_factors"].append(out_j["T_factors"])
        hist["D_factors"].append(out_j["D_factors"])
        hist["mse_state_list"].append(out_j.get("mse_list", None))

        if use_control:
            hist["B_list"].append(out_j["B_list"])
            hist["B_factors"].append(out_j["B_factors"])

        last_x_list.append(x_T)
        last_P_list.append(P_T)

        # accumulate mean state MSE
        mse_list = out_j["mse_list"]  # length max_it
        for it in range(max_it):
            sum_mse_state[it] += float(mse_list[it])

        # accumulate mean F/H MSE (single true matrices)
        if sum_mse_F is not None:
            for it in range(max_it):
                F_hat_it = out_j["F_list"][it]
                sum_mse_F[it] += torch.mean((F_hat_it - F_true) ** 2).double().item()

        if sum_mse_H is not None:
            for it in range(max_it):
                H_hat_it = out_j["H_list"][it]
                sum_mse_H[it] += torch.mean((H_hat_it - H_true) ** 2).double().item()

        # ===== PRINT =====
    mean_state = sum_mse_state / N_seq
    mean_state_db = 10.0 * torch.log10(mean_state + 1e-12)
    print("\n=== Mean STATE MSE (over sequences) per EM iter ===")
    for it in range(max_it):
        print(f"Iter {it:02d}: {mean_state_db[it].item():.3f} dB")

    if sum_mse_F is not None:
        mean_F = sum_mse_F / N_seq
        mean_F_db = 10.0 * torch.log10(mean_F + 1e-12)
        print("\n=== Mean F MSE (over sequences) per EM iter ===")
        for it in range(max_it):
            print(f"Iter {it:02d}: {mean_F_db[it].item():.3f} dB")

    if sum_mse_H is not None:
        mean_H = sum_mse_H / N_seq
        mean_H_db = 10.0 * torch.log10(mean_H + 1e-12)
        print("\n=== Mean H MSE (over sequences) per EM iter ===")
        for it in range(max_it):
            print(f"Iter {it:02d}: {mean_H_db[it].item():.3f} dB")

    return hist, last_x_list, last_P_list