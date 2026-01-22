import pandas as pd
import numpy as np
import yfinance as yf
import torch


from Simulations.Linear_sysmdl import SystemModel
from Smoothers.RTS_Smoother_test import S_Test
from emkf.second_main_emkf_paper_func import EMKF_FHB_decrypt_style_batch
# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda")
# ===============================
# Download data (paper style)
# ===============================
tickers = ["BTC-USD", "ETH-USD"]

data = yf.download(
    tickers,
    start="2019-07-07",
    end="2022-05-05",
    interval="1d",
    auto_adjust=False,
    progress=True
)


# data = yf.download(
#     tickers,
#     start="2018-01-01",
#     end="2023-01-01",
#     interval="1d",
#     auto_adjust=False,
#     progress=True
# )

btc = data.xs("BTC-USD", level=1, axis=1)
eth = data.xs("ETH-USD", level=1, axis=1)

# Paper features (order matters!)
cols = ["Open", "Adj Close", "High", "Low", "Volume"]
btc = btc[cols]
eth = eth[cols]

# Sanity checks (paper assumes clean data)
assert not btc.isna().any().any()
assert (btc["High"] >= btc["Low"]).all()
assert (btc["Volume"] >= 0).all()

btc.to_csv("BTC_USD_OHLCV.csv")
eth.to_csv("ETH_USD_OHLCV.csv")

print(btc.head())
print("BTC shape:", btc.shape)
# ===============================
# Control input: SMA
# ===============================
SMA_WINDOW = 10   # reasonable + paper-consistent

def compute_sma(series, window):
    return series.rolling(window).mean()

btc["u"] = compute_sma(btc["Adj Close"], SMA_WINDOW)
eth["u"] = compute_sma(eth["Adj Close"], SMA_WINDOW)

# Drop first rows where SMA undefined
btc = btc.dropna()
eth = eth.dropna()

print("After SMA:")
print(btc.head())


# ===============================
# Sliding window (tau = 50)
# ===============================
TAU = 50

def make_windows(df, tau):
    X_list = []
    U_list = []

    values = df[cols].values          # [T, 5]
    u_vals = df["u"].values            # [T]

    for t in range(len(df) - tau):
        X = values[t:t+tau].T          # [5, tau]
        U = u_vals[t:t+tau][None, :]   # [1, tau]

        X_list.append(X)
        U_list.append(U)

    return np.stack(X_list), np.stack(U_list)

X_btc, U_btc = make_windows(btc, TAU)
X_eth, U_eth = make_windows(eth, TAU)

print("BTC windows:", X_btc.shape, U_btc.shape)
print("ETH windows:", X_eth.shape, U_eth.shape)

# ===============================
# Train / Test split (paper)
# ===============================
def split_train_test(df, X, U):
    dates = df.index[TAU:]   # window-aligned dates

    train_idx = dates < "2021-01-01"
    test_idx  = dates >= "2021-01-01"

    X_train, U_train = X[train_idx], U[train_idx]
    X_test,  U_test  = X[test_idx],  U[test_idx]

    return X_train, U_train, X_test, U_test

Xtr_btc, Utr_btc, Xte_btc, Ute_btc = split_train_test(btc, X_btc, U_btc)
Xtr_eth, Utr_eth, Xte_eth, Ute_eth = split_train_test(eth, X_eth, U_eth)

print("BTC train:", Xtr_btc.shape, "test:", Xte_btc.shape)
print("ETH train:", Xtr_eth.shape, "test:", Xte_eth.shape)




# ======================================================
# PICK ASSET (BTC or ETH)
# ======================================================
X_train = torch.tensor(Xtr_btc, dtype=torch.float32, device=device)#[N_train, 5, 50]
U_train = torch.tensor(Utr_btc, dtype=torch.float32, device=device)#[N_train, 1, 50]

X_test  = torch.tensor(Xte_btc, dtype=torch.float32, device=device)#[N_test, 5, 50]
U_test  = torch.tensor(Ute_btc, dtype=torch.float32, device=device)#[N_test, 1, 50]

# ======================================================
# STATE-SPACE DIMENSIONS
# ======================================================
m = 5   # state dimension (OHLCV)
n = 5   # observation dimension
p = 1   # control (SMA)

T = X_train.shape[-1]
N_seq = X_train.shape[0]

# ======================================================
# INITIAL MODEL (INTENTIONALLY WRONG F)
# ======================================================
F_init = torch.eye(m, device=device) * 0.9
B_init =  0.5 * torch.ones(m, p, device=device)  # Random initial B
H_true = torch.eye(n, m, device=device)

Q = 1e-10 * torch.eye(m, device=device)
R = 1e-2* torch.eye(n, device=device)

x0 = torch.zeros(m, 1, device=device)#[5,1]
P0 =1e-2*  torch.eye(m, device=device)#[5,5]

# ======================================================
# SYSTEM MODEL
# ======================================================
sys_model = SystemModel(F_init, Q, H_true, R, T, T)
sys_model.InitSequence(x0, P0)
# Add B matrix to system model
sys_model.B = B_init

# ======================================================
# FACTORED INITIALIZATION (PAPER STYLE)
# F = T10 @ T11 @ T12
# ======================================================
I_m = torch.eye(5, device=device)
I_p = torch.eye(1, device=device)

factors_init = {
    # F factors
    "T10": I_m.clone(),
    "T11": I_m.clone(),
    "T12": F_init.clone(),

    # B factors: [5x1] = [5x5] @ [5x1] @ [1x1]
    "T20": I_m.clone(),                             # [5x5]
    "T21": torch.randn(m, 1, device=device),       # [5x1]
    "T22": torch.ones(1, 1, device=device),        # [1x1]

    # H fixed
    "D0": torch.eye(5, device=device),
    "D1": torch.eye(5, device=device),
    "D2": torch.eye(5, device=device),
}


# ======================================================
# BASELINE: RTS WITH WRONG F
# ======================================================
print("\n--- RTS BASELINE (WRONG F) ---")
_, mse_rts, mse_rts_db, _, _, _ = S_Test(
    sys_model,
    X_test,
    X_test,
    F=[F_init for _ in range(X_test.shape[0])],
    H=[H_true for _ in range(X_test.shape[0])]
)

print(f"RTS MSE: {mse_rts_db.item():.3f} dB")

# ======================================================
# DECRYPT-STYLE EMKF (FACTORED)
# ======================================================
print("\n--- DeCrypt-style EMKF ---")

hist, x_last, p_last = EMKF_FHB_decrypt_style_batch(
    sys_model=sys_model,
    Y=X_train,
    X_true=X_train,
    x_0=x0,
    P_0=P0,
    factors_init=factors_init,
    U_in=U_train,
    max_it=50,
    n_sweeps_factor=1,
    update_F=True,
    update_B=True,        # ✅ MUST BE TRUE
    update_H=False,
    H_fixed=H_true,
)

# ======================================================
# FINAL LEARNED F (AVERAGE OVER SEQUENCES)
# ======================================================
F_final = torch.stack([hist["F_list"][j][-1] for j in range(N_seq)]).mean(dim=0)
B_final = torch.stack([hist["B_list"][j][-1] for j in range(N_seq)]).mean(dim=0)


print("\nLearned F:")
print(F_final)
print("\nLearned B:")
print(B_final)
# ======================================================
# TEST WITH LEARNED F
# ======================================================
print("\n--- RTS WITH LEARNED F ---")
_, mse_dec, mse_dec_db, _, _, _ = S_Test(
    sys_model,
    X_test,
    X_test,
    F=[F_final for _ in range(X_test.shape[0])],
    H=[H_true for _ in range(X_test.shape[0])]
)

print(f"DeCrypt EMKF MSE: {mse_dec_db.item():.3f} dB")
print(f"Improvement over RTS: {(mse_rts_db - mse_dec_db).item():.3f} dB")