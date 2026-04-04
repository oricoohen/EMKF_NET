# ======================================================
# Rolling LLT Kalman + EM(Q,R) + 1-step Forecast (THESIS STYLE)
# - F and H are FIXED (LLT)
# - EM learns Q and R on each rolling window
# - Forecast uses x_T|T (filtered) (you can switch to smoothed if you want)
# - Reports error metrics + optional simple trading like you had
# ======================================================

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import random
import collections

# -----------------------
# Repro
# -----------------------
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# -----------------------
# Device / dtype
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# -----------------------
# Settings
# -----------------------
ticker = "SPY"
start_date = "2018-01-01"
end_date   = "2019-01-01"

TAU = 20                 # rolling window length
max_em_it = 30           # EM iterations per window

k_pct = 0.05             # threshold for trading (optional)
k = k_pct / 100.0

# LLT model dims
m = 2  # state: [level, slope]
n = 1  # measurement: price level

# FIXED (thesis style)
F_fixed = torch.tensor([[1.0, 1.0],
                        [0.0, 1.0]], device=device, dtype=dtype)

H_fixed = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)

I_m = torch.eye(m, device=device, dtype=dtype)

# init x0, P0 (fixed)
x0_default = torch.zeros(m, 1, device=device, dtype=dtype)
P0_default = 1.0 * torch.eye(m, device=device, dtype=dtype)

# initial Q,R guesses (EM will update)
Q_init = 0.5 * torch.eye(m, device=device, dtype=dtype)
R_init = 0.5 * torch.eye(n, device=device, dtype=dtype)

# -----------------------
# Helpers: KF + RTS + EM(Q,R)
# -----------------------
def kalman_filter(y: torch.Tensor, F: torch.Tensor, H: torch.Tensor,
                  Q: torch.Tensor, R: torch.Tensor,
                  x0: torch.Tensor, P0: torch.Tensor):
    """
    y: (T, 1) tensor
    Returns dict with filtered/predicted states/covs + K gains
    """
    T = y.shape[0]

    x_pred = torch.zeros((T, m, 1), device=device, dtype=dtype)
    P_pred = torch.zeros((T, m, m), device=device, dtype=dtype)
    x_filt = torch.zeros((T, m, 1), device=device, dtype=dtype)
    P_filt = torch.zeros((T, m, m), device=device, dtype=dtype)
    K_list = torch.zeros((T, m, n), device=device, dtype=dtype)

    x_prev = x0
    P_prev = P0

    for t in range(T):
        # Predict
        xp = F @ x_prev
        Pp = F @ P_prev @ F.T + Q

        # Update
        S = H @ Pp @ H.T + R                       # (1,1)
        S_inv = torch.linalg.inv(S)
        K = Pp @ H.T @ S_inv                       # (m,1)

        innov = y[t:t+1].T - (H @ xp)              # (1,1)
        xf = xp + K @ innov

        # Joseph form for numerical stability
        KH = K @ H                                 # (m,m)
        Pf = (I_m - KH) @ Pp @ (I_m - KH).T + K @ R @ K.T

        # store
        x_pred[t] = xp
        P_pred[t] = Pp
        x_filt[t] = xf
        P_filt[t] = Pf
        K_list[t] = K

        x_prev = xf
        P_prev = Pf

    return {
        "x_pred": x_pred, "P_pred": P_pred,
        "x_filt": x_filt, "P_filt": P_filt,
        "K": K_list
    }


def rts_smoother(F: torch.Tensor, kf_out: dict):
    """
    RTS smoother.
    Returns x_smooth, P_smooth, J (smoother gains)
    """
    x_pred = kf_out["x_pred"]
    P_pred = kf_out["P_pred"]
    x_filt = kf_out["x_filt"]
    P_filt = kf_out["P_filt"]

    T = x_filt.shape[0]
    x_smooth = torch.zeros_like(x_filt)
    P_smooth = torch.zeros_like(P_filt)
    J = torch.zeros((T, m, m), device=device, dtype=dtype)

    # init at last
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T-2, -1, -1):
        # J_t = P_t|t F^T (P_{t+1|t})^{-1}
        Pp_next_inv = torch.linalg.inv(P_pred[t+1])
        Jt = P_filt[t] @ F.T @ Pp_next_inv

        x_smooth[t] = x_filt[t] + Jt @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + Jt @ (P_smooth[t+1] - P_pred[t+1]) @ Jt.T
        J[t] = Jt

    return x_smooth, P_smooth, J


def smoothed_lag_cov(F: torch.Tensor, kf_out: dict, P_smooth: torch.Tensor, J: torch.Tensor):
    """
    Compute P_lag[t] = Cov(x_t, x_{t-1} | y_1:T) for t=1..T-1
    Uses a standard backward recursion.
    NOTE: If you only care about R, you can skip this; Q needs it.
    """
    x_pred = kf_out["x_pred"]
    P_pred = kf_out["P_pred"]
    x_filt = kf_out["x_filt"]
    P_filt = kf_out["P_filt"]
    K = kf_out["K"]

    T = P_filt.shape[0]
    P_lag = torch.zeros((T, m, m), device=device, dtype=dtype)  # P_lag[t] valid for t>=1

    if T < 2:
        return P_lag

    # base case (commonly used practical init)
    # P_{T-1,T-2|T} approx = (I - K_{T-1}H) F P_{T-2|T-2}
    KH_last = (K[T-1] @ H_fixed)  # (m,m)
    P_lag[T-1] = (I_m - KH_last) @ F @ P_filt[T-2]

    # recursion backward
    for t in range(T-2, 0, -1):
        # P_{t,t-1|T} = P_{t|t} J_{t-1}^T + J_t (P_{t+1,t|T} - F P_{t|t}) J_{t-1}^T
        P_lag[t] = P_filt[t] @ J[t-1].T + J[t] @ (P_lag[t+1] - F @ P_filt[t]) @ J[t-1].T

    return P_lag


def em_update_QR(y: torch.Tensor, F: torch.Tensor, H: torch.Tensor,
                 x_smooth: torch.Tensor, P_smooth: torch.Tensor, P_lag: torch.Tensor,
                 eps: float = 1e-6):
    """
    EM M-step updates for Q and R with fixed F,H.
    y: (T,1)
    x_smooth: (T,m,1)
    P_smooth: (T,m,m)
    P_lag: (T,m,m) where P_lag[t] = Cov(x_t, x_{t-1}|Y) for t>=1
    """
    T = y.shape[0]

    # Precompute expectations
    Exx = torch.zeros((T, m, m), device=device, dtype=dtype)
    for t in range(T):
        Exx[t] = P_smooth[t] + x_smooth[t] @ x_smooth[t].T

    Exx_lag = torch.zeros((T, m, m), device=device, dtype=dtype)  # E[x_t x_{t-1}^T]
    for t in range(1, T):
        Exx_lag[t] = P_lag[t] + x_smooth[t] @ x_smooth[t-1].T

    # ---- Q update ----
    # Q = (1/(T-1)) Σ_{t=1..T-1} E[(x_t - F x_{t-1})(x_t - F x_{t-1})^T]
    Q_num = torch.zeros((m, m), device=device, dtype=dtype)
    for t in range(1, T):
        term = Exx[t] - F @ Exx_lag[t].T - Exx_lag[t] @ F.T + F @ Exx[t-1] @ F.T
        Q_num += term
    Q_new = Q_num / max(T-1, 1)

    # stabilize: enforce symmetry + PSD-ish
    Q_new = 0.5 * (Q_new + Q_new.T)
    Q_new = Q_new + eps * torch.eye(m, device=device, dtype=dtype)

    # ---- R update ----
    # R = (1/T) Σ E[(y_t - H x_t)(y_t - H x_t)^T]
    #   = (1/T) Σ [ (y_t - H xhat_t)(...)^T + H P_t H^T ]
    R_num = torch.zeros((n, n), device=device, dtype=dtype)
    for t in range(T):
        resid = y[t:t+1].T - (H @ x_smooth[t])  # (1,1)
        R_num += resid @ resid.T + H @ P_smooth[t] @ H.T
    R_new = R_num / T

    R_new = 0.5 * (R_new + R_new.T)
    R_new = R_new + eps * torch.eye(n, device=device, dtype=dtype)

    return Q_new, R_new


# -----------------------
# Data
# -----------------------
data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    progress=True
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

px = data["Adj Close"].dropna().copy()
dates = px.index
z = px.values.astype(np.float64)

if len(z) < TAU + 2:
    raise ValueError(f"Need at least TAU+2={TAU+2} points, got {len(z)}")

# -----------------------
# Rolling predict
# -----------------------
pred_price_list = []
true_price_list = []
dates_list = []
signal_list = []

# equity (optional)
equity_str = 1.0
equity_bh  = 1.0
equity_rand = 1.0

false_buy = false_sell = true_buy = true_sell = 0
hold_buy = hold_sell = 0

def random_trading_decision():
    return 1 if random.random() < 0.5 else -1

for t in range(TAU, len(z)):
    win_start = t - TAU
    win_end   = t  # exclusive
    win_prices = z[win_start:win_end]  # length TAU

    # thesis style: often use price level; keep as-is (no normalization)
    y_win = torch.tensor(win_prices, device=device, dtype=dtype).view(TAU, 1)

    # init per-window EM
    Q = Q_init.clone()
    R = R_init.clone()
    x0 = x0_default.clone()
    P0 = P0_default.clone()

    # EM iterations
    last_ll = None
    for it in range(max_em_it):
        # E-step: KF + RTS
        kf_out = kalman_filter(y_win, F_fixed, H_fixed, Q, R, x0, P0)
        x_smooth, P_smooth, J = rts_smoother(F_fixed, kf_out)
        P_lag = smoothed_lag_cov(F_fixed, kf_out, P_smooth, J)

        # M-step: update Q,R
        Q_new, R_new = em_update_QR(y_win, F_fixed, H_fixed, x_smooth, P_smooth, P_lag)

        # simple convergence on parameter change
        dQ = torch.norm(Q_new - Q) / (torch.norm(Q) + 1e-12)
        dR = torch.norm(R_new - R) / (torch.norm(R) + 1e-12)
        Q, R = Q_new, R_new

        if max(dQ.item(), dR.item()) < 1e-3:
            break

    # -------- Forecast (thesis style) --------
    # Use LAST FILTERED state x_{T-1|T-1} (or use smoothed x_smooth[-1] if you prefer)
    x_last = kf_out["x_filt"][-1].detach()          # (m,1)
    x_next = F_fixed @ x_last                       # (m,1)
    pred_price = (H_fixed @ x_next)[0, 0].item()    # scalar

    today_price = float(z[t - 1])
    tomorrow_price = float(z[t])

    pred_ret = (pred_price - today_price) / (today_price + 1e-12)
    real_ret = (tomorrow_price - today_price) / (today_price + 1e-12)

    # Optional trading (keep your style but FIX the bug)
    sig_label = "hold"
    if pred_ret > k:
        sig_label = "buy"
        equity_str *= (1.0 + real_ret)
        if real_ret > 0: true_buy += 1
        else:            false_buy += 1
    elif pred_ret < -k:
        sig_label = "sell"
        equity_str *= (1.0 - real_ret)
        if real_ret < 0: true_sell += 1
        else:            false_sell += 1
    else:
        # hold: do nothing
        if real_ret > 0: hold_buy += 1
        else:            hold_sell += 1

    # Random baseline
    decision = random_trading_decision()
    if decision == 1:
        equity_rand *= (1.0 + real_ret)
    else:
        equity_rand *= (1.0 - real_ret)

    # Buy & Hold baseline
    equity_bh *= (1.0 + real_ret)

    pred_price_list.append(pred_price)
    true_price_list.append(tomorrow_price)
    dates_list.append(dates[t])
    signal_list.append(sig_label)

# -----------------------
# Report
# -----------------------
results_df = pd.DataFrame({
    "Date": dates_list,
    "TrueClose": true_price_list,
    "PredClose": pred_price_list,
    "Signal": signal_list,
})

# Error metrics (thesis-style evaluation is usually RMSE/MAE/MAPE)
y_true = np.array(true_price_list, dtype=np.float64)
y_pred = np.array(pred_price_list, dtype=np.float64)

mse  = np.mean((y_true - y_pred)**2)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100.0

print("\n" + "=" * 70)
print(f"{ticker} | Rolling LLT + EM(Q,R) + 1-step Forecast (THESIS STYLE)")
print(f"Window TAU={TAU} | EM iters<= {max_em_it} | k={k_pct:.3f}% (optional trading)")
print("=" * 70)
print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.3f}%")
print("-" * 70)
print(f"Strategy final multiple: {equity_str:.4f}  Return: {(equity_str-1)*100:+.2f}%")
print(f"Buy&Hold final multiple: {equity_bh:.4f}  Return: {(equity_bh-1)*100:+.2f}%")
print(f"Random   final multiple: {equity_rand:.4f}  Return: {(equity_rand-1)*100:+.2f}%")
print("-" * 70)
print("Signals:", collections.Counter(signal_list))
print("true buy:", true_buy, "false buy:", false_buy)
print("true sell:", true_sell, "false sell:", false_sell)
print("hold buy:", hold_buy, "hold sell:", hold_sell)

out_csv = "spy_rolling_thesisstyle_emQR_forecast.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(results_df.tail(10))
