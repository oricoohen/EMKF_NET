import pandas as pd
import numpy as np
import torch


from Simulations.Linear_sysmdl import SystemModel
from Smoothers.RTS_Smoother_test import S_Test
from emkf.second_main_emkf_paper_func import EMKF_FHB_decrypt_style_batch

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda")

# ===============================
# Generate synthetic data (no download needed)
# ===============================
print("Generating synthetic financial data...")
np.random.seed(42)

# Simulate realistic OHLCV data
T_total = 1816
opens = 10000 + np.cumsum(np.random.randn(T_total) * 100)
closes = opens + np.random.randn(T_total) * 50
highs = np.maximum(opens, closes) + np.abs(np.random.randn(T_total) * 50)
lows = np.minimum(opens, closes) - np.abs(np.random.randn(T_total) * 50)
volumes = np.random.randint(100000000, 300000000, T_total)

btc = pd.DataFrame({
    'Open': opens,
    'Adj Close': closes,
    'High': highs,
    'Low': lows,
    'Volume': volumes
})

print("BTC data shape:", btc.shape)
print(btc.head())

# ===============================
# Control input: SMA
# ===============================
SMA_WINDOW = 10

def compute_sma(series, window):
    return series.rolling(window).mean()

btc["u"] = compute_sma(btc["Adj Close"], SMA_WINDOW)
btc = btc.dropna()

print("After SMA:")
print(btc.head())

# ===============================
# Sliding window (tau = 50)
# ===============================
TAU = 50

cols = ["Open", "Adj Close", "High", "Low", "Volume"]

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

print("BTC windows:", X_btc.shape, U_btc.shape)

# ===============================
# Train / Test split
# ===============================
split_idx = int(len(X_btc) * 0.6)

X_train = X_btc[:split_idx]
U_train = U_btc[:split_idx]
X_test = X_btc[split_idx:]
U_test = U_btc[split_idx:]

print("Train:", X_train.shape, "Test:", X_test.shape)

# ======================================================
# CONVERT TO TORCH
# ======================================================
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
U_train = torch.tensor(U_train, dtype=torch.float32, device=device)

X_test  = torch.tensor(X_test, dtype=torch.float32, device=device)
U_test  = torch.tensor(U_test, dtype=torch.float32, device=device)

# ======================================================
# STATE-SPACE DIMENSIONS
# ======================================================
m = 5   # state dimension (OHLCV)
n = 5   # observation dimension
p = 1   # control (SMA)

T = X_train.shape[-1]
N_seq = X_train.shape[0]

print(f"Dimensions: m={m}, n={n}, p={p}, T={T}, N_seq={N_seq}")

# ======================================================
# INITIAL MODEL (INTENTIONALLY WRONG F)
# ======================================================
F_init = torch.eye(m, device=device) * 0.9
B_init = torch.randn(m, p, device=device) * 0.1  # Random initial B
H_true = torch.eye(n, m, device=device)

Q = 1e-4 * torch.eye(m, device=device)
R = 1e-3 * torch.eye(n, device=device)

x0 = torch.zeros(m, 1, device=device)
P0 = torch.eye(m, device=device)

# ======================================================
# SYSTEM MODEL
# ======================================================
print("Creating system model...")
sys_model = SystemModel(F_init, Q, H_true, R, T, T)
sys_model.InitSequence(x0, P0)
# Add B matrix to system model
sys_model.B = B_init

print("✓ System model created")

# ======================================================
# FACTORED INITIALIZATION (PAPER STYLE)
# F = T10 @ T11 @ T12
# B = T20 @ T21 @ T22 where [5x1] = [5x5] @ [5x1] @ [1x1]
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

print("✓ Factors initialized")
print(f"  T20 shape: {factors_init['T20'].shape}")
print(f"  T21 shape: {factors_init['T21'].shape}")
print(f"  T22 shape: {factors_init['T22'].shape}")

# Verify B factorization
B_test = factors_init['T20'] @ factors_init['T21'] @ factors_init['T22']
print(f"  B test shape: {B_test.shape} (should be [5, 1])")

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
print("\n--- DeCrypt-style EMKF (Training) ---")

hist, x_last, p_last = EMKF_FHB_decrypt_style_batch(
    sys_model=sys_model,
    Y=X_train,
    X_true=X_train,
    x_0=x0,
    P_0=P0,
    factors_init=factors_init,
    U_in=U_train,
    max_it=3,
    n_sweeps_factor=1,
    update_F=True,
    update_B=True,        # ✅ MUST BE TRUE
    update_H=False,
    H_fixed=H_true,
)

print("✓ EMKF training completed")

# ======================================================
# FINAL LEARNED F (AVERAGE OVER SEQUENCES)
# ======================================================
F_final = torch.stack([hist["F_list"][j][-1] for j in range(N_seq)]).mean(dim=0)

print("\nLearned F (averaged):")
print(F_final)

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

print("\n✅ COMPLETE - Script executed successfully with control input!")

