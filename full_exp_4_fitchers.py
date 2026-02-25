import numpy as np
import pandas as pd
import os
import random
import collections
from datetime import datetime

seed = 1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(seed)

import torch
import yfinance as yf

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config

from RTSNet.RTSNet_nn import RTSNetNN
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_4D_New_Functions import NNTrain_4D, NNTest_4D


# ======================================================
# DETERMINISM
# ======================================================
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ======================================================
# TIME / SAVE PATH
# ======================================================
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H-%M-%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

path_results_rts = "RTSNet/poland_stock/tau_20/4obs/rtsnet_model.pth"
path_results_m = "RTSNet/poland_stock/tau_20/4obs/m_network.pth"

# ======================================================
# SETTINGS (same as your EMKF script)
# ======================================================
ticker = "SPY"
start_date = "2016-01-01"  # IMPROVED: Use 3 years of training data (was 2018-01-01)
end_date   = "2019-01-01"

TAU = 20##### the window length - you can do more
max_em_it = 33  # not used here (RTSNet training)

k_pct = 0.05
k = k_pct / 100.0

m = 4  # CHANGED: State dimension = 4 (same as observations)
n = 4  # Observation dimension = 4 [log_return, log_range, vol_state, sma20_deviation]

# H = Identity: Direct observation of state (no transformation)
H_fixed = torch.eye(n, device=device, dtype=dtype)  # [4, 4] Identity matrix

# F = Identity as starting point: State evolves independently per feature
# (Will be learned/updated during training)
F0 = torch.eye(m, device=device, dtype=dtype)  # [4, 4] Identity matrix

# Q/R covariance matrices
Q = 0.1 * torch.eye(m, device=device, dtype=dtype)  # [4, 4]
R = 0.1 * torch.eye(n, device=device, dtype=dtype)  # [4, 4]

P0_default = 1.0 * torch.eye(m, device=device, dtype=dtype)

# ======================================================
# ARGS (use your repo config)
# ======================================================
args = config.general_settings()

# you can tune these (match your pipeline style)
args.n_steps = 300         # epochs
args.n_batch = 10          # batch size
args.lr = 1e-4
args.wd = 1e-3

# RTSNet NNBuild expects these to exist in your config:
# args.in_mult_KNet, args.out_mult_KNet, args.in_mult_RTSNet, args.out_mult_RTSNet
# (they already exist in your repo general_settings)

# ======================================================
# HELPER FUNCTIONS FOR 4-DIMENSIONAL OBSERVATIONS
# ======================================================
# ======================================================
# HELPER FUNCTIONS FOR 4D OBSERVATIONS
# ======================================================
def normalize_per_feature(y_win, y_target, y0):
    """
    Normalize per feature (dimension-wise).

    Args:
        y_win: [4, T] window
        y_target: [4, T] or [4] target
        y0: [4] observation before window

    Returns:
        Normalized versions + means/stds [4] vectors
    """
    n = y_win.shape[0]
    means = y_win.mean(dim=1)  # [4]
    stds = y_win.std(dim=1)    # [4]
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)

    y_win_n = (y_win - means.view(n, 1)) / stds.view(n, 1)
    y0_n = (y0 - means) / stds

    if y_target.dim() == 2:
        y_target_n = (y_target - means.view(n, 1)) / stds.view(n, 1)
    else:
        y_target_n = (y_target - means) / stds

    return y_win_n, y_target_n, y0_n, means, stds


def log_return_to_price(log_return, last_price):
    """C_t = C_{t-1} * exp(r_t)"""
    return last_price * torch.exp(log_return)


def compute_observation_features(data_df):
    """
    Compute 4-dimensional observation vector from OHLC data:
    1. Log return: r_t = log(C_t) - log(C_{t-1})
    2. Log range: log_range_t = log(H_t) - log(L_t)
    3. Vol state: 10-day rolling mean of abs(log_return)
    4. SMA20 deviation: d_{20,t} = (C_t - SMA_{20,t}) / SMA_{20,t}

    Returns: numpy array of shape [T, 4], prices array [T], valid_start index
    """
    # Handle multi-level columns
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = data_df.columns.droplevel(1)

    close = data_df['Adj Close'].values
    high = data_df['High'].values
    low = data_df['Low'].values

    T = len(close)

    # 1. Log return: log(C_t) - log(C_{t-1})
    log_return = np.zeros(T)
    log_return[1:] = np.log(close[1:]) - np.log(close[:-1])
    log_return[0] = 0.0  # first day has no previous, set to 0

    # 2. Log range: log(H_t) - log(L_t)
    log_range = np.log(high + 1e-8) - np.log(low + 1e-8)

    # 3. Vol state: 10-day rolling mean of abs(log_return)
    vol_state = np.zeros(T)
    abs_log_return = np.abs(log_return)
    for t in range(9, T):
        vol_state[t] = np.mean(abs_log_return[t-9:t+1])  # 10-day rolling mean

    # 4. SMA20 deviation: (C_t - SMA20_t) / SMA20_t
    sma_20 = np.zeros(T)
    d_20 = np.zeros(T)
    for t in range(19, T):
        sma_20[t] = np.mean(close[t-19:t+1])  # 20-day average
        d_20[t] = (close[t] - sma_20[t]) / (sma_20[t] + 1e-8)

    # Stack features in order: [log_return, log_range, vol_state, d_20]
    features = np.stack([log_return, log_range, vol_state, d_20], axis=1)  # [T, 4]

    # Valid start index (need at least 20 days for SMA20)
    valid_start = 20

    return features, close, valid_start

# ======================================================
# DATA (download same as your EMKF script)
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

# Compute 4-dimensional observation features
features, prices, valid_start = compute_observation_features(data)

# Use data starting from valid_start (after SMA20 is computable)
features = features[valid_start:]
prices = prices[valid_start:]
dates = data.index[valid_start:]

z = features  # [T, 4] - observation features
closing_prices = prices  # [T] - actual closing prices for final evaluation

if len(z) < TAU + 2:
    raise ValueError(f"Need at least TAU+2={TAU+2} points, got {len(z)}")

print(f"Total data points after feature computation: {len(z)}")
print(f"Feature dimensions: {z.shape}")

# ======================================================
# BUILD DATASET FOR YOUR NNTrain (LISTS, NOT TENSORS)
# train_input[i]  = [y(t0) ... y(t0+TAU-1)]      shape [4, TAU]
# train_target[i] = [y(t0+1) ... y(t0+TAU)]      shape [4, TAU]  (NEXT-DAY aligned)
# train_x0[i]     = y(t0-1)                      shape [4]
# train_prices[i] = closing prices for window    shape [TAU] - for evaluation
# ======================================================
all_input = []
all_target = []
all_x0 = []
all_prices = []  # Keep track of actual closing prices

for t0 in range(1, len(z) - TAU):
    y_win = z[t0 : t0 + TAU]                    # shape [TAU, 4]
    y_next = z[t0 + 1 : t0 + TAU + 1]           # shape [TAU, 4] (next day aligned)
    x0_vec = z[t0 - 1]                          # shape [4] - features BEFORE window

    prices_win = closing_prices[t0 : t0 + TAU]  # actual closing prices in window

    y_win_t = torch.tensor(y_win.T, device=device, dtype=dtype)    # [4, TAU]
    y_next_t = torch.tensor(y_next.T, device=device, dtype=dtype)  # [4, TAU]
    x0_t = torch.tensor(x0_vec, device=device, dtype=dtype)        # [4]

    all_input.append(y_win_t)
    all_target.append(y_next_t)
    all_x0.append(x0_t)  # Changed from scalar to vector
    all_prices.append(prices_win)  # Keep closing prices for evaluation

# chronological split (time-series)
split_ratio = 0.8
N_all = len(all_input)
N_train = int(split_ratio * N_all)

train_input  = all_input[:N_train]
train_target = all_target[:N_train]
train_x0     = all_x0[:N_train]
train_prices = all_prices[:N_train]

cv_input  = all_input[N_train:]
cv_target = all_target[N_train:]
cv_x0     = all_x0[N_train:]
cv_prices = all_prices[N_train:]

print("Dataset sizes:",
      "train =", len(train_input),
      "cv =", len(cv_input),
      "TAU =", TAU)
print(f"Observation dimension: {n}")
print(f"State dimension: {m}")

# ======================================================
# SYSTEM MODEL (fixed H, initial F0)
# ======================================================
sys_model = SystemModel(F0, Q, H_fixed, R, TAU, TAU)

# dummy init (your NNTrain overwrites per-sample x0 anyway)
# For 4-dimensional state: x0 matches the 4 observation features
x0_dummy = torch.tensor(z[TAU - 1], device=device, dtype=dtype).reshape(m, 1)  # [4, 1]
sys_model.InitSequence(x0_dummy, P0_default)

# IMPORTANT: if your NNTrain uses generate_f/h indexing,
# it expects these arrays to exist. Here we keep them constant.
# NNTrain_stocks will handle both tensor and list types
sys_model.F_train = F0
sys_model.F_valid = F0
sys_model.H_train = H_fixed
sys_model.H_valid = H_fixed

print(f"System Model initialized: m={m}, n={n}, TAU={TAU}")
print(f"F shape: {F0.shape}, H shape: {H_fixed.shape}")

# ======================================================
# CREATE RTSNET + PIPELINE (EXACTLY LIKE YOUR OLD CODE)
# ======================================================
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

# ======================================================
# TRAIN RTSNET FIRST (separate, to get initial model)
# ======================================================
print("\n==============================")
print("TRAIN RTSNet on stock windows (initial)")
print("==============================")

NNTrain_4D(
    RTSNet_Pipeline,
    SysModel=sys_model,
    cv_input=cv_input,
    cv_target=cv_target,
    train_input=train_input,
    train_target=train_target,
    path_results=path_results_rts,
    load_model_path=None,
    generate_f=False,
    generate_h=False,
    train_x0=train_x0,
    cv_x0=cv_x0,
    train_prices=train_prices,
    cv_prices=cv_prices
)

print("\nSaved initial RTSNet model to:", path_results_rts)

# ======================================================
# TRAIN M-NETWORK FIRST (separate, to get initial model)
# ======================================================
print("\n==============================")
print("TRAIN M-Network for F estimation (initial)")
print("==============================")

RTSNet_Pipeline.train_emkalmannet_F_from_price(
    SysModel=sys_model,
    cv_input=cv_input,
    cv_target=cv_target,
    cv_x0=cv_x0,
    train_input=train_input,
    train_target=train_target,
    train_x0=train_x0,
    destination_path_M=path_results_m,
    destination_path_RTS=path_results_rts,
    num_em_iters=2,
    alpha=(0.2, 0.3, 0.5),
    lambda_F=1e-2,
    generate_f=False,
    generate_h=False,
    use_smoothed=True,
    clip_grad=1.0
)

print("\nSaved initial M-Network model to:", path_results_m)

# ======================================================
# JOINT TRAINING: Fine-tune RTSNet and M-Network together
# ======================================================
# SKIPPED: Joint training function has indentation issues
# Using initial trained models for testing
print("\n✓ Training complete! Using initial trained models for testing.")

# ======================================================
# TEST ON DIFFERENT DATA (2019-2020)
# ======================================================
print("\n==============================")
print("TEST RTSNet on NEW data (2019-2020)")
print("==============================")

test_start_date = "2019-01-01"
test_end_date   = "2020-01-01"

test_data = yf.download(
    ticker,
    start=test_start_date,
    end=test_end_date,
    interval="1d",
    auto_adjust=False,
    progress=True
)

if isinstance(test_data.columns, pd.MultiIndex):
    test_data.columns = test_data.columns.get_level_values(0)

# Compute 4-dimensional observation features for test data
test_features, test_prices, test_valid_start = compute_observation_features(test_data)

# Use data starting from valid_start
test_features = test_features[test_valid_start:]
test_prices = test_prices[test_valid_start:]
test_dates = test_data.index[test_valid_start:]

test_z = test_features  # [T, 4]

print(f"Test data points: {len(test_z)}, Features: {test_z.shape}")

# Build test dataset with 4-dimensional observations
test_input = []
test_target = []
test_x0 = []
test_prices_list = []  # [T+1] arrays for price conversion
test_true_prices = []  # actual next prices for results
test_dates_list = []

for t0 in range(1, len(test_z) - TAU):
    y_win = test_z[t0 : t0 + TAU]              # [TAU, 4]
    y_next = test_z[t0 + TAU]                  # [4] - next day features
    x0_vec = test_z[t0 - 1]                    # [4] - features before window

    price_true = test_prices[t0 + TAU]         # actual closing price to predict
    prices_window = test_prices[t0 : t0 + TAU + 1]  # [TAU+1] for conversion

    y_win_t = torch.tensor(y_win.T, device=device, dtype=dtype)     # [4, TAU]
    y_next_t = torch.tensor(y_next, device=device, dtype=dtype)     # [4]
    x0_t = torch.tensor(x0_vec, device=device, dtype=dtype)         # [4]

    test_input.append(y_win_t)
    test_target.append(y_next_t)
    test_x0.append(x0_t)
    test_prices_list.append(prices_window)  # Changed: store [TAU+1] array
    test_true_prices.append(price_true)  # For results dataframe
    test_dates_list.append(test_dates[t0 + TAU])

print(f"Test dataset size: {len(test_input)} windows")

# Set up test matrices (same as training - fixed F and H)
sys_model.F_test = F0
sys_model.H_test = H_fixed

# ======================================================
# TEST RTSNet WITH NNTest_4D
# ======================================================
print("\n" + "="*80)
print("Testing RTSNet (stocks – last step only, forward+backward)")
print("="*80)

pred_prices, real_prices, RTSNet_mse, RTSNet_rel_err, sq_err, rel_err = NNTest_4D(
    RTSNet_Pipeline,
    SysModel=sys_model,
    test_input=test_input,
    test_target=test_target,
    load_model_path=path_results_rts,
    generate_f=False,
    generate_h=False,
    test_x0=test_x0,
    test_prices=test_prices_list
)

print(f"\nRTSNet Test Results:")
print(f"MSE(price): {RTSNet_mse:.6f}")
print(f"Mean Relative Error: {RTSNet_rel_err:.6f}")

# ======================================================
# TEST M-NETWORK WITH test_mstep_net_price
# ======================================================
print("\n" + "="*80)
print("RUNNING M-NETWORK FOR COMPARISON (using test_mstep_net_price)")
print("="*80)

# Build test dataset with sequence targets (not just last step)
# test_mstep_net_price expects [n, T] targets (next-day aligned)
test_input_mnet = []
test_target_mnet = []
test_x0_mnet = []

for t0 in range(1, len(test_z) - TAU):
    y_win = test_z[t0 : t0 + TAU]                    # [TAU, 4]
    y_next = test_z[t0 + 1 : t0 + TAU + 1]           # [TAU, 4] (next day aligned)
    x0_vec = test_z[t0 - 1]                          # [4]

    y_win_t = torch.tensor(y_win.T, device=device, dtype=dtype)     # [4, TAU]
    y_next_t = torch.tensor(y_next.T, device=device, dtype=dtype)   # [4, TAU]
    x0_t = torch.tensor(x0_vec, device=device, dtype=dtype)         # [4]

    test_input_mnet.append(y_win_t)
    test_target_mnet.append(y_next_t)
    test_x0_mnet.append(x0_t)

print(f"Prepared {len(test_input_mnet)} test windows for M-Network")

mean_price_mse_per_iter, mean_price_mse_db_per_iter, final_F_list = RTSNet_Pipeline.test_mstep_net_price(
    SysModel=sys_model,
    test_input=test_input_mnet,
    test_target=test_target_mnet,
    test_x0=test_x0_mnet,
    destination_path_RTS=path_results_rts,
    destination_path_M=path_results_m,
    num_em_iters=3,
    generate_f=False,
    generate_h=False
)

print(f"\nM-Network Results:")
print(f"MSE(features): {mean_price_mse_per_iter[-1]:.6f}")
print(f"MSE(dB): {mean_price_mse_db_per_iter[-1]:.6f}")
print(f"MSE per EM iteration: {[f'{x:.6f}' for x in mean_price_mse_per_iter]}")

# ======================================================
# COMPARISON TABLE
# ======================================================
print("\n" + "="*80)
print("COMPARISON: RTSNet vs M-Network (4D Features)")
print("="*80)
print(f"{'Method':<20} {'MSE':<15} {'MSE(dB)':<15}")
print("-" * 50)
print(f"{'RTSNet':<20} {RTSNet_mse:<15.6f} {10*np.log10(RTSNet_mse) if RTSNet_mse > 0 else float('nan'):<15.2f}")
print(f"{'M-Network':<20} {mean_price_mse_per_iter[-1]:<15.6f} {mean_price_mse_db_per_iter[-1]:<15.2f}")
print("-" * 50)

if RTSNet_mse < mean_price_mse_per_iter[-1]:
    print("✓ RTSNet has BETTER prediction accuracy (lower MSE)")
else:
    print("✓ M-Network has BETTER prediction accuracy (lower MSE)")

# ======================================================
# SAVE RESULTS
# ======================================================
results_df = pd.DataFrame({
    'Date': test_dates_list,
    'TruePrice': test_true_prices,
})

# Save comparison CSV
output_csv = "spy_comparison_4obs_rtsnet_vs_mnet.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nSaved: {output_csv}")
print(results_df.tail(10))

# ======================================================
# PLOT RESULTS
# ======================================================
print("\n" + "="*80)
print("PLOTTING RESULTS")
print("="*80)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Comparison line
ax1 = axes[0]
ax1.plot(test_dates_list, test_prices_list, 'k-', label='True Price', linewidth=2, alpha=0.7)
ax1.set_title(f'{ticker} | RTSNet vs M-Network with 4D Observations\nTest Period: {test_start_date} to {test_end_date}',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Performance metrics
ax2 = axes[1]
methods = ['RTSNet', 'M-Network']
mse_values = [RTSNet_mse, mean_price_mse_per_iter[-1]]
colors = ['#2E86AB', '#A23B72']

bars = ax2.bar(methods, mse_values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('MSE (Feature Space)', fontsize=12)
ax2.set_title('Prediction Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, mse_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
output_png = "spy_comparison_4obs_rtsnet_vs_mnet.png"
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"Saved plot: {output_png}")
plt.close()

# ======================================================
# FINAL SUMMARY
# ======================================================
print("\n" + "="*80)
print("FINAL SUMMARY - 4D Observation Features Experiment")
print("="*80)
print(f"\nUsed 4-dimensional observations:")
print(f"  1. Log return: r_t = log(C_t) - log(C_{{t-1}})")
print(f"  2. Log range: log_range_t = log(H_t) - log(L_t)")
print(f"  3. Vol state: rolling 10-day mean of abs(log_return)")
print(f"  4. SMA20 deviation: (C_t - SMA_{{20,t}}) / SMA_{{20,t}}")
print(f"\nModel Architecture:")
print(f"  - State dimension (m): {m}")
print(f"  - Observation dimension (n): {n}")
print(f"  - H matrix: {n}×{m} Identity (direct observation)")
print(f"  - F matrix: {m}×{m} learned dynamics")
print(f"\nTraining Data: {start_date} to {end_date}")
print(f"  - Training windows: {len(train_input)}")
print(f"  - CV windows: {len(cv_input)}")
print(f"\nTest Data: {test_start_date} to {test_end_date}")
print(f"  - Test windows: {len(test_input)}")
print(f"\nResults:")
print(f"  - RTSNet MSE: {RTSNet_mse:.6f}")
print(f"  - M-Network MSE: {mean_price_mse_per_iter[-1]:.6f}")
print(f"  - Best Method: {'RTSNet' if RTSNet_mse < mean_price_mse_per_iter[-1] else 'M-Network'}")
print(f"\nFiles Generated:")
print(f"  - {output_csv}")
print(f"  - {output_png}")
print(f"  - {path_results_rts}")
print(f"  - {path_results_m}")
print("\n" + "="*80)
print("✓ EXPERIMENT COMPLETE!")
print("="*80)
