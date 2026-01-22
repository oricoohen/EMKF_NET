import pandas as pd
import numpy as np
import yfinance as yf
import torch

from Simulations.Linear_sysmdl import SystemModel
from emkf.second_main_emkf_paper_func import EMKF_FHB_decrypt_style_batch

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda")

# ===============================
# Download ONE YEAR of data
# ===============================
ticker = "BTC-USD"
data = yf.download(
    ticker,
    start="2021-01-01",
    end="2022-01-01",
    interval="1d",
    auto_adjust=False,
    progress=True
)
btc = data.copy()
# ---- FIX: flatten MultiIndex columns (yfinance sometimes returns this) ----
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)


cols = ["Open", "Adj Close", "High", "Low", "Volume"]
btc = btc[cols].copy()
btc = btc.apply(pd.to_numeric, errors="coerce").dropna()

assert not btc.isna().any().any()
assert (btc["High"].to_numpy() >= btc["Low"].to_numpy()).all()
assert (btc["Volume"].to_numpy() >= 0).all()

print(btc.head())
print(f"Total days: {len(btc)}")


def compute_sma(series, window):
    return series.rolling(window).mean()

btc = btc.dropna()

print(f"After SMA: {len(btc)} days")

# ===============================
# NORMALIZATION (global, once)
# ===============================
mu = btc[cols].mean()
sig = btc[cols].std() + 1e-8

# normalize OHLCV
btc[cols] = (btc[cols] - mu) / sig




# ======================================================
# ROLLING WINDOW APPROACH
# ======================================================
TAU = 50  # training window size


# Storage for BOTH methods
pred_A = []      # method A: use x_T (KF/RTS estimate)
pred_B = []      # method B: use y_T (raw last observation)
true_list = []   # true price
mse_A = []
mse_B = []
dates_list = []
pred_naive = []     # baseline: tomorrow = today
mse_naive = []

# State-space dimensions
m = 5  # OHLCV
n = 5
p = 1  # SMA control

# Initial F, B (will be updated)
F_prev = torch.eye(m, device=device) * 0.9

print(f"\n{'='*60}")
print(f"ROLLING WINDOW PREDICTION (step through {len(btc) - TAU} days)")
print(f"{'='*60}\n")

for window_idx in range(len(btc) - TAU):
    # ====== Get training window ======
    # Days: window_idx to window_idx+TAU-1 (50 days)
    train_start = window_idx
    train_end = window_idx + TAU

    X_window = btc[cols].iloc[train_start:train_end].values.T  # [5, 50]

    # Convert to torch (add batch dimension)
    X_batch = torch.tensor(X_window[np.newaxis, :, :], dtype=torch.float32, device=device)  # [1, 5, 50]

    # ====== System Model Setup ======
    T = TAU
    H_true = torch.eye(n, m, device=device)
    Q = 0.5 * torch.eye(m, device=device)  # Increased for stability
    R = 0.5 * torch.eye(n, device=device)  # Increased for stability

    x0 = torch.zeros(m, 1, device=device)
    P0 = 0.5 * torch.eye(m, device=device)  # Increased for stability

    sys_model = SystemModel(F_prev, Q, H_true, R, T, T)
    sys_model.InitSequence(x0, P0)

    # ====== Factored Initialization (warm-start from previous) ======
    I_m = torch.eye(m, device=device)

    factors_init = {
        "T10": I_m.clone(),
        "T11": I_m.clone(),
        "T12": F_prev.clone(),  # Initialize from previous F

        "D0": torch.eye(5, device=device),
        "D1": torch.eye(5, device=device),
        "D2": torch.eye(5, device=device),
    }

    # ====== EMKF Training (only 1 sequence) ======
    hist, last_x_list, last_P_list= EMKF_FHB_decrypt_style_batch(
            sys_model=sys_model,
            Y=X_batch,
            X_true=X_batch,
            x_0=x0,
            P_0=P0,
            factors_init=factors_init,
            U_in=None,
            max_it=10,  # Reduced to prevent instability
            n_sweeps_factor=1,
            update_F=True,
            update_B=False,
            update_H=False,
            H_fixed=H_true,
        )

    # Extract learned F and B
    F_learned = hist["F_list"][0][-1]
    # B_learned = hist["B_list"][0][-1]
    # last_x_list is a python list of length N_seq. Here N_seq=1.
    xT = last_x_list[0]  # (5,1) last smoothed state from RTS

    # ======================================================
    # NEXT-DAY PREDICTION (two methods)
    # ======================================================


    # METHOD A: use smoothed state x_T from EMKF

    x_next_A = F_learned @ xT
    pred_price_A_norm = x_next_A[1, 0].item()

    # METHOD B: use raw last observation y_T (which equals x observation since H=I)
    yT = X_batch[0, :, -1:].clone()  # (5,1)
    x_next_B = F_learned @ yT
    pred_price_B_norm = x_next_B[1, 0].item()

    # ====== True Price (next day, index = train_end) ======

    true_price_idx = train_end
    if true_price_idx < len(btc):
        pred_date = btc.index[true_price_idx]

        true_price_norm = float(btc["Adj Close"].iloc[true_price_idx])
        mu_ac = float(mu["Adj Close"])
        sig_ac = float(sig["Adj Close"])

        true_price = true_price_norm * sig_ac + mu_ac
        pred_A_price = pred_price_A_norm * sig_ac + mu_ac
        mse_a = (pred_A_price - true_price) ** 2
        pred_A.append(pred_A_price)
        # ===== NAIVE BASELINE: predict tomorrow = today =====
        today_price_norm = float(btc["Adj Close"].iloc[train_end - 1])  # last day in the window
        today_price = today_price_norm * sig_ac + mu_ac
        pred_naive.append(pred_naive_price)
        mse_naive.append(mse_n)

        pred_naive_price = today_price
        mse_n = (pred_naive_price - true_price) ** 2

        true_list.append(true_price)
        mse_A.append(mse_a)
        dates_list.append(pred_date)

        # warm start for next window
        F_prev = F_learned.detach().clone()
        # B_prev = B_learned.detach().clone()

        if (window_idx + 1) % 10 == 0:
            print(
                f"Day {window_idx + 1:3d}: True={true_price:8.2f}, "
                f"A={pred_A_price:8.2f} (RMSE={np.sqrt(mse_a):7.2f}), "
                f"Naive={pred_naive_price:8.2f} (RMSE={np.sqrt(mse_n):7.2f})"
            )
results_df = pd.DataFrame({
    "Date": dates_list,
    "True": true_list,
    "Pred_A_xT": pred_A,
    "Pred_Naive": pred_naive,
    "MSE_A": mse_A,
    "MSE_Naive": mse_naive,
    "RMSE_A": [np.sqrt(x) for x in mse_A],
    "RMSE_Naive": [np.sqrt(x) for x in mse_naive],

})
print(results_df.tail())

print("\nSUMMARY")
print(f"Avg RMSE A (x_T):   {np.mean(results_df['RMSE_A']):.4f}")
print(f"Avg RMSE Naive:     {np.mean(results_df['RMSE_Naive']):.4f}")

results_df.to_csv("rolling_window_predictions_two_methods.csv", index=False)




