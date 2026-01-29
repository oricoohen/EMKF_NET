import numpy as np
import pandas as pd
import os
seed = 1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(seed)
import torch
import yfinance as yf
import collections
from Simulations.Linear_sysmdl import SystemModel
from Smoothers.RTS_Smoother_test import S_Test
from emkf.main_emkf_func import EMKF_H_analitic, EMKF_FH_analytic  # your function
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def random_trading_decision():
    # returns +1 for BUY, -1 for SELL
    return 1 if random.random() < 0.5 else -1

# ================ ======================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# ======================================================
# SETTINGS
# ======================================================
ticker = "SPY"
start_date = "2018-01-01"
end_date   = "2019-01-01"

TAU = 20
max_em_it = 30

k_pct = 0.05
k = k_pct / 100.0

# LLT model dims (1 trend)
m = 2  # state: [level, slope]
n = 1  # measurement: price level

H_fixed = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)

F0 = torch.tensor([[1.0, 1.0],
                   [0.0, 1.0]], device=device, dtype=dtype)


# fixed Q/R (not estimated)
# Q = 1e-4 * torch.eye(m, device=device, dtype=dtype)
# R = 1e-3 * torch.eye(n, device=device, dtype=dtype)
Q = 0.1 * torch.eye(m, device=device, dtype=dtype)
R = 0.1* torch.eye(n, device=device, dtype=dtype)
# initial H guess




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

# initialize from the last observed real price BEFORE first prediction
x0_default = torch.tensor(    [[z[TAU - 1]],[0.1]],device=device,dtype=dtype)
# init x0, P0
# x0_default = torch.zeros(m, 1, device=device, dtype=dtype)
P0_default = 1.0 * torch.eye(m, device=device, dtype=dtype)


if len(z) < TAU + 2:
    raise ValueError(f"Need at least TAU+2={TAU+2} points, got {len(z)}")

# ======================================================
# ROLLING PREDICT + TRADE (your desired style: no pos memory)
# ======================================================
pred_price_list = []
true_price_list = []
dates_list = []

signal_list = []

equity_str = 1.0
equity_bh  = 1.0
equity_orc = 1.0
equity_rand = 1.0
# warm-starts for EM initialization
F_prev = F0.clone()
x0_prev = x0_default.clone()
P0_prev = P0_default.clone()
false_buy = 0
false_sell = 0
true_buy = 0
true_sell = 0
hold_buy = 0
hold_sell = 0
sig_label = "hold"
# Iterate windows: window [t-TAU .. t-1], predict day t
# Trade from day (t-1) -> t
for t in range(TAU, len(z)):
    win_start = t - TAU
    win_end = t  # exclusive

    win_prices = z[win_start:win_end]  # length TAU

    # window normalization
    # mu = float(win_prices.mean())
    # sig = float(win_prices.std() + 1e-12)
    # y_win_norm = (win_prices - mu) / sig  # normalized level series
#################################
    y_win_norm = win_prices.copy()
    #############################
    # Build tensors for EMKF_H_analitic
    Y = torch.tensor(y_win_norm, device=device, dtype=dtype).view(1, 1, TAU)

    # IMPORTANT: X passed to S_Test inside your EM code expects state-dim.
    # We give dummy with correct shape [N_seq, m, T]
    X_dummy = torch.zeros((1, m, TAU), device=device, dtype=dtype)

    # SystemModel for this window (H will be overwritten inside EM iterations)
    sys_model = SystemModel(F_prev, Q, H_fixed, R, TAU, TAU)
    sys_model.InitSequence(x0_prev, P0_prev)


    # Run EM (H only)
    F_matrices, _, last_x_list, last_P_list =EMKF_FH_analytic(sys_model,[F_prev], [H_fixed], Q, R, Y, x0_prev, P0_prev, X_dummy,
                                                    max_it=max_em_it, generate_f=True, generate_h=True,init_x_list=None, init_P_list=None,  update_F=True, update_H=False)

    # Extract last estimates
    F_hat = F_matrices[0][-1].detach().clone()  # [2,2]
    print("Estimated F:\n", F_hat)
    xT_s = last_x_list[0].detach().clone()  # [2,1]
    PT_s = last_P_list[0].detach().clone()  # [2,2]

    # 1-step-ahead forecast (normalized)
    x_next = F_hat @ xT_s
    y_next_norm = (H_fixed @ x_next)[0, 0].item()
#########################################
    # unnormalize to price
    # pred_price = (y_next_norm * sig) + mu
#3####################################
    pred_price = y_next_norm
    #######################################
    # today and tomorrow prices (USD)
    today_price = float(z[t - 1])
    tomorrow_price = float(z[t])

    pred_ret = (pred_price - today_price) / (today_price + 1e-12)
    real_ret = (tomorrow_price - today_price) / (today_price + 1e-12)

    # STRATEGY (your requested: immediate action, no position memory)
    if pred_ret > k:
        sig_label = "buy"
        equity_str *= (1.0 + real_ret)
        if real_ret > 0:
            true_buy += 1
        else:
            false_buy += 1
    elif pred_ret < -k:
        sig_label = "sell"
        equity_str *= (1.0 - real_ret)
        if real_ret < 0:
            true_sell += 1
        else:
            false_sell += 1
    else:
        if sig_label  == "buy":
            equity_str *= (1.0 + real_ret)
        elif sig_label == "sell":
            equity_str *= (1.0 - real_ret)
        # sig_label = "hold"
        # equity_str *= 1.0
        if real_ret > 0:
            hold_buy += 1
        else:
            hold_sell += 1

    # decision = random_trading_decision()
    # if decision == 1:  # BUY
    #     equity_rand *= (1.0 + real_ret)
    # else:  # SELL
    #     equity_rand *= (1.0 - real_ret)
    # # Buy & Hold
    equity_bh *= (1.0 + real_ret)

    # Oracle (your requested: sign-based, no threshold)
    if real_ret > 0:
        equity_orc *= (1.0 + real_ret)
    elif real_ret < 0:
        equity_orc *= (1.0 - real_ret)
    else:
        equity_orc *= 1.0

    # store
    pred_price_list.append(pred_price)
    true_price_list.append(tomorrow_price)
    dates_list.append(dates[t])
    signal_list.append(sig_label)

    # warm-start next window
    F_prev = F_hat
    # true initial price for this window
    x0_prev = torch.tensor([[y_win_norm[0]],  [0.]],device=device,dtype=dtype)


# ======================================================
# REPORT
# ======================================================
results_df = pd.DataFrame({
    "Date": dates_list,
    "TrueClose": true_price_list,
    "PredClose": pred_price_list,
    "Signal": signal_list,
})

eps = 1e-12

# Convert to numpy for metrics
y_true = np.asarray(true_price_list, dtype=np.float64)
y_pred = np.asarray(pred_price_list, dtype=np.float64)

# 1) Plain price MSE (units: dollars^2)
mse_price = np.mean((y_pred - y_true) ** 2)

# 2) Relative-to-price MSE (dimensionless)
rel_err = (y_pred - y_true) / (y_true + eps)
mse_rel = np.mean(rel_err ** 2)

# Optional: RMSE + Relative RMSE for readability
rmse_price = np.sqrt(mse_price)
rmse_rel = np.sqrt(mse_rel)

print(f"\nMSE (price): {mse_price:.6f} | RMSE (price): {rmse_price:.6f} USD")
print(f"MSE (relative): {mse_rel:.8f} | RMSE (relative): {rmse_rel:.6f} (≈ {rmse_rel*100:.3f}%)")







print("\n" + "=" * 60)
print(f"{ticker} | Rolling EMKF_H_analitic (H-only) + 1-step Forecast + Trading")
print(f"Window TAU={TAU} | k={k_pct:.2f}% | EM iters={max_em_it}")
print("=" * 60)

print(f"Strategy final multiple: {equity_str:.4f}  Return: {(equity_str-1)*100:+.2f}%")
print(f"Oracle   final multiple: {equity_orc:.4f}  Return: {(equity_orc-1)*100:+.2f}%")
print(f"Buy&Hold final multiple: {equity_bh:.4f}  Return: {(equity_bh-1)*100:+.2f}%")
print("=" * 60)

print("Signals:", collections.Counter(signal_list))

out_csv = "spy_rolling_emkfH_analytic_forecast_trading.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(results_df.tail(10))
print('true buy:',true_buy)
print('false buy:',false_buy)
print('true sell:',true_sell)
print('false sell:',false_sell)
print('hold buy:',hold_buy)
print('hold sell:',hold_sell)
print(f"Random Strategy final multiple: {equity_rand:.4f}  Return: {(equity_rand-1)*100:+.2f}%")


# ======================================================
# PLOT: True vs Predicted
# ======================================================
plt.figure()
plt.plot(dates_list, true_price_list, label="True Price")
plt.plot(dates_list, pred_price_list, label="Predicted Price")
plt.title(f"{ticker} True vs Predicted (1-step) | TAU={TAU} | EM iters={max_em_it}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
