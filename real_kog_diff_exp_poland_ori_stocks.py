import numpy as np
import pandas as pd
import torch
import yfinance as yf
import collections

from Simulations.Linear_sysmdl import SystemModel
from Smoothers.RTS_Smoother_test import S_Test
from emkf.main_emkf_func import EMKF_H_analitic, EMKF_FH_analytic  # your function

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# ======================================================
# SETTINGS
# ======================================================
ticker = "SPY"
start_date = "2019-01-01"
end_date   = "2020-01-01"

TAU = 20
max_em_it = 30

k_pct = 0.005
k = k_pct / 100.0   # threshold on percentage return

# Drift-LLT model dims for RETURNS
m = 2  # state: [mu (expected return), beta (drift slope)]
n = 1  # measurement: realized return

H_fixed = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)  # y_t = mu_t + noise

F0 = torch.tensor([[1.0, 1.0],
                   [0.0, 1.0]], device=device, dtype=dtype)        # mu_{t+1} = mu_t + beta_t

# fixed Q/R (not estimated)
Q = 1e-4 * torch.eye(m, device=device, dtype=dtype)
R = 1e-3 * torch.eye(n, device=device, dtype=dtype)

# init x0, P0
x0_default = torch.zeros(m, 1, device=device, dtype=dtype)
P0_default = 1.0 * torch.eye(m, device=device, dtype=dtype)

# ======================================================
# DATA
# ======================================================
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
print(z)
SCALE = 100.0
# percentage (simple) returns: ret[t] = (p_{t+1}-p_t)/p_t
ret = (z[1:] - z[:-1]) / (z[:-1] + 1e-12)     # length = len(z)-1
ret_dates = dates[1:]                          # date of p_{t+1}
print('oriiiiiiiiiiiiiiiiiiiiiiiiiiiii', ret)
if len(ret) < TAU + 1:
    raise ValueError(f"Need at least TAU+1={TAU+1} returns, got {len(ret)}")

# ======================================================
# ROLLING PREDICT + TRADE (1-day position)
# ======================================================
pred_ret_list = []
true_ret_list = []
dates_list = []
signal_list = []

equity_str = 1.0
equity_bh  = 1.0
equity_orc = 1.0

# warm-starts
F_prev = F0.clone()
x0_prev = x0_default.clone()
P0_prev = P0_default.clone()

false_buy = false_sell = true_buy = true_sell = hold_buy = hold_sell = 0

# window uses returns indices: [t-TAU .. t-1], predict return at t
for t in range(TAU, len(ret)):
    win_start = t - TAU
    win_end = t  # exclusive

    win_rets = ret[win_start:win_end] * SCALE # length TAU
    Y = torch.tensor(win_rets, device=device, dtype=dtype).view(1, 1, TAU)

    X_dummy = torch.zeros((1, m, TAU), device=device, dtype=dtype)

    sys_model = SystemModel(F_prev, Q, H_fixed, R, TAU, TAU)
    sys_model.InitSequence(x0_prev, P0_prev)

    # EM: update F (and generate_f True), keep H fixed
    F_matrices, _, last_x_list, last_P_list = EMKF_FH_analytic(
        sys_model, [F_prev], [H_fixed], Q, R, Y,
        x0_prev, P0_prev, X_dummy,
        max_it=max_em_it,
        generate_f=True, generate_h=True,
        init_x_list=None, init_P_list=None,
        update_F=True, update_H=False
    )

    F_hat = F_matrices[0][-1].detach().clone()
    xT_s  = last_x_list[0].detach().clone()
    PT_s  = last_P_list[0].detach().clone()

    # 1-step-ahead forecast of RETURN
    x_next = F_hat @ xT_s
    pred_ret = (H_fixed @ x_next)[0, 0].item()/ SCALE

    # realized RETURN at time t (already percentage return)
    real_ret = float(ret[t])

    # STRATEGY (1-day position)
    if pred_ret > k:
        sig_label = "buy"
        equity_str *= (1.0 + real_ret)
        if real_ret > 0: true_buy += 1
        else: false_buy += 1
    elif pred_ret < -k:
        sig_label = "sell"
        equity_str *= (1.0 - real_ret)  # short proxy
        if real_ret < 0: true_sell += 1
        else: false_sell += 1
    else:
        sig_label = "hold"
        equity_str *= 1.0
        if real_ret > 0: hold_buy += 1
        else: hold_sell += 1

    # Buy & Hold (always long)
    equity_bh *= (1.0 + real_ret)

    # Oracle (perfect direction)
    if real_ret > 0:
        equity_orc *= (1.0 + real_ret)
    elif real_ret < 0:
        equity_orc *= (1.0 - real_ret)
    else:
        equity_orc *= 1.0

    pred_ret_list.append(pred_ret)
    true_ret_list.append(real_ret)
    dates_list.append(ret_dates[t])  # date aligned to this return
    signal_list.append(sig_label)

    # warm-start next window
    F_prev = F_hat
    x0_prev = xT_s
    P0_prev = PT_s

# ======================================================
# REPORT
# ======================================================
results_df = pd.DataFrame({
    "Date": dates_list,
    "TrueRet": true_ret_list,
    "PredRet": pred_ret_list,
    "Signal": signal_list,
})

print("\n" + "=" * 60)
print(f"{ticker} | Rolling EMKF (returns) + 1-step Forecast + Trading")
print(f"Window TAU={TAU} | k={k_pct:.2f}% | EM iters={max_em_it}")
print("=" * 60)

print(f"Strategy final multiple: {equity_str:.4f}  Return: {(equity_str-1)*100:+.2f}%")
print(f"Oracle   final multiple: {equity_orc:.4f}  Return: {(equity_orc-1)*100:+.2f}%")
print(f"Buy&Hold final multiple: {equity_bh:.4f}  Return: {(equity_bh-1)*100:+.2f}%")
print("=" * 60)

print("Signals:", collections.Counter(signal_list))

out_csv = "spy_rolling_emkf_returns_forecast_trading.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(results_df.tail(10))
print('true buy:', true_buy)
print('false buy:', false_buy)
print('true sell:', true_sell)
print('false sell:', false_sell)
print('hold buy:', hold_buy)
print('hold sell:', hold_sell)
