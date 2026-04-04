import pandas as pd
import numpy as np
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
from Pipelines.Pipeline_ERTS import train_joint_rtsnet_and_mnet_em2_batch5


# ======================================================
# DETERMINISM
# ======================================================
random.seed(seed)
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

path_results_rts = "RTSNet/poland_stock/tau_5/rtsnet_model_n2_price_logret.pth"
path_results_m = "RTSNet/poland_stock/tau_5/m_network_n2_price_logret.pth"

# ======================================================
# SETTINGS (same as your EMKF script)
# ======================================================
ticker = "SPY"
start_date = "2017-01-01"
end_date   = "2019-09-01"

TAU = 5
max_em_it = 2

k_pct = 0.05
k = k_pct / 100.0

# -------------------------------------------------------
# Choose observation dimension:
#   n=1  →  y_t = [price_t]                   (original)
#   n=2  →  y_t = [price_t, trend_t]          (new)
# -------------------------------------------------------
n = 2   # set to 1 for original, 2 for 2D observations

m = 2

if n == 1:
    # ORIGINAL: scalar price observation
    H_fixed = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)  # [1, 2]
    path_results_rts = "../RTSNet/poland_stock/tau_5/rtsnet_model_x0_f_I.pth"
    path_results_m   = "../RTSNet/poland_stock/tau_5/m_network_x0_f_I_final_loss.pth"
elif n == 2:
    # NEW: [price, trend] observation — H=I so state=[price,trend] is observed directly
    H_fixed = torch.eye(m, device=device, dtype=dtype)                 # [2, 2]
    path_results_rts = "RTSNet/poland_stock/tau_5/rtsnet_model_n2_price_trend.pth"
    path_results_m   = "RTSNet/poland_stock/tau_5/m_network_n2_price_trend.pth"

F0 = torch.tensor([[1.0, 0.0],
                   [0.0, 1.0]], device=device, dtype=dtype)

Q = 0.1 * torch.eye(m, device=device, dtype=dtype)
R = 0.1 * torch.eye(n, device=device, dtype=dtype)

P0_default = 1.0 * torch.eye(m, device=device, dtype=dtype)

# ======================================================
# ARGS (use your repo config)
# ======================================================
args = config.general_settings()

# you can tune these (match your pipeline style)
args.n_steps = 200         # epochs
args.n_batch = 5          # batch size
args.lr = 1e-4
args.wd = 1e-3

# RTSNet NNBuild expects these to exist in your config:
# args.in_mult_KNet, args.out_mult_KNet, args.in_mult_RTSNet, args.out_mult_RTSNet
# (they already exist in your repo general_settings)

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

px = data["Adj Close"].dropna().copy()
dates = px.index
z = px.values.astype(float)  # Use native Python float

if len(z) < TAU + 2:
    raise ValueError(f"Need at least TAU+2={TAU+2} points, got {len(z)}")

# ======================================================
# BUILD DATASET
# ======================================================
all_input = []
all_target = []
all_x0 = []

K = 5  # lookback for trend computation

for t0 in range(K, len(z) - TAU):
    win_prices  = z[t0 : t0 + TAU]           # length TAU
    next_prices = z[t0 + 1 : t0 + TAU + 1]   # length TAU (next-day aligned)

    # x0: price before window + linear trend (no leakage, ends at t0-1)
    price_before = float(z[t0 - 1])
    past  = z[t0 - K : t0]                   # length K, ends at z[t0-1]
    trend0 = float((past[-1] - past[0]) / (K - 1))
    x0_vector = torch.tensor([price_before, trend0], device=device, dtype=dtype)  # [2]

    if n == 1:
        # ---- ORIGINAL: scalar price observation [1, TAU] ----
        y_win_t  = torch.tensor(win_prices,  device=device, dtype=dtype).view(1, TAU)
        y_next_t = torch.tensor(next_prices, device=device, dtype=dtype).view(1, TAU)

    elif n == 2:
        # ---- NEW: [price, trend] observation [2, TAU] ----
        # trend at each step t = (z[t0+t] - z[t0+t-K]) / (K-1)   (no leakage: uses only past)
        win_trend  = []
        next_trend = []
        for ti in range(TAU):
            # trend for window step ti:  past K prices ending at z[t0+ti-1]
            t_start = t0 + ti - K
            t_end   = t0 + ti          # slice [t_start : t_end], length K
            p_past  = z[t_start : t_end]
            win_trend.append((p_past[-1] - p_past[0]) / (K - 1))
            # trend for next step ti:   past K prices ending at z[t0+ti]
            p_past_n = z[t_start + 1 : t_end + 1]
            next_trend.append((p_past_n[-1] - p_past_n[0]) / (K - 1))

        win_trend_arr  = np.array(win_trend,  dtype=float)
        next_trend_arr = np.array(next_trend, dtype=float)

        y_win_t  = torch.zeros(2, TAU, device=device, dtype=dtype)
        y_win_t[0, :]  = torch.tensor(win_prices,     device=device, dtype=dtype)
        y_win_t[1, :]  = torch.tensor(win_trend_arr,  device=device, dtype=dtype)

        y_next_t = torch.zeros(2, TAU, device=device, dtype=dtype)
        y_next_t[0, :] = torch.tensor(next_prices,    device=device, dtype=dtype)
        y_next_t[1, :] = torch.tensor(next_trend_arr, device=device, dtype=dtype)

    all_input.append(y_win_t)
    all_target.append(y_next_t)
    all_x0.append(x0_vector)

# chronological split
split_ratio = 0.8
N_all   = len(all_input)
N_train = int(split_ratio * N_all)

train_input  = all_input[:N_train]
train_target = all_target[:N_train]
train_x0     = all_x0[:N_train]

cv_input  = all_input[N_train:]
cv_target = all_target[N_train:]
cv_x0     = all_x0[N_train:]

print("Dataset sizes:", "train =", len(train_input), "cv =", len(cv_input), "TAU =", TAU)
print(f"Observation dimension: n={n}  ({'price only' if n==1 else 'price + trend'})")
print(f"State dimension: m={m} (price + trend)")

# ======================================================
# SYSTEM MODEL (fixed H=I, initial F=I)
# ======================================================
sys_model = SystemModel(F0, Q, H_fixed, R, TAU, TAU)

# dummy init (your NNTrain overwrites per-sample x0 anyway)
# x0 = [price, trend] where trend is computed from past K differences (no leakage)
dummy_price = z[K + TAU - 1]
dummy_past = z[K - 1 : K + TAU - 1]  # K prices ending just before the first window
dummy_trend = float((dummy_past[-1] - dummy_past[0]) / (K - 1))
x0_dummy = torch.tensor([[dummy_price], [dummy_trend]], device=device, dtype=dtype)
sys_model.InitSequence(x0_dummy, P0_default)

# IMPORTANT: if your NNTrain uses generate_f/h indexing,
# it expects these arrays to exist. Here we keep them constant.
# NNTrain_stocks will handle both tensor and list types
sys_model.F_train = F0
sys_model.F_valid = F0
sys_model.H_train = H_fixed
sys_model.H_valid = H_fixed

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

RTSNet_Pipeline.NNTrain_stocks(
    sys_model,
    cv_input,
    cv_target,
    train_input,
    train_target,
    path_results=path_results_rts,
    load_model_path=None,
    generate_f=False,
    generate_h=False,
    train_x0=train_x0,
    cv_x0=cv_x0
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
    alpha=(0.05, 0.15, 0.85),
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
print("\n==============================")
print("JOINT TRAINING: RTSNet + M-Network (EM2, Batch5)")
print("==============================")

# Create output paths for jointly trained models
path_results_rts_joint = f"RTSNet/poland_stock/tau_5/rtsnet_joint_model_n{n}.pth"
path_results_m_joint   = f"RTSNet/poland_stock/tau_5/m_network_joint_n{n}.pth"


train_joint_rtsnet_and_mnet_em2_batch5(
    RTSNet_Pipeline,
    sys_model,
    train_input, train_target, train_x0,
    cv_input, cv_target, cv_x0,
    path_results_rts,      # Load initial RTSNet
    path_results_m,        # Load initial M-Network
    path_results_rts_joint,  # Save joint RTSNet
    path_results_m_joint,    # Save joint M-Network
    batch_size=5,
    num_em_iters=2,
    lambda_F=1e-3,
    clip_grad=1.0,
    lr_rts=1e-4,
    lr_m=1e-4,
    wd_rts=1e-5,
    wd_m=1e-5
)

print("\nSaved jointly trained RTSNet model to:", path_results_rts_joint)
print("Saved jointly trained M-Network model to:", path_results_m_joint)

# Update paths to use jointly trained models for testing
path_results_rts = path_results_rts_joint
path_results_m = path_results_m_joint
# path_results_rts = path_results_rts
# path_results_m = path_results_m

print("\n✓ Training complete! Using jointly trained models for testing.")

# ======================================================
# TEST ON DIFFERENT DATA (2019-2020)
# ======================================================
print("\n==============================")
print("TEST RTSNet on NEW data (2019-2020)")
print("==============================")

test_start_date = "2018-07-01"
test_end_date   = "2019-01-01"

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

test_px = test_data["Adj Close"].dropna().copy()
test_dates = test_px.index
test_z = test_px.values.astype(float)  # Use native Python float

# Build test dataset
test_input = []
test_target = []
test_x0 = []
test_dates_list = []

K = 5  # same as training

for t0 in range(K, len(test_z) - TAU):
    win_prices  = test_z[t0 : t0 + TAU]
    next_prices = test_z[t0 + 1 : t0 + TAU + 1]

    # x0: price before window + linear trend (no leakage)
    price_before = float(test_z[t0 - 1])
    past  = test_z[t0 - K : t0]
    trend0 = float((past[-1] - past[0]) / (K - 1))
    x0_vector = torch.tensor([price_before, trend0], device=device, dtype=dtype)

    if n == 1:
        # ---- ORIGINAL: scalar price observation ----
        y_win_t  = torch.tensor(win_prices,  device=device, dtype=dtype).view(1, TAU)
        y_next_t = torch.tensor(next_prices, device=device, dtype=dtype).view(1, TAU)

    elif n == 2:
        # ---- NEW: [price, trend] observation ----
        win_trend  = []
        next_trend = []
        for ti in range(TAU):
            t_start = t0 + ti - K
            t_end   = t0 + ti
            p_past  = test_z[t_start : t_end]
            win_trend.append((p_past[-1] - p_past[0]) / (K - 1))
            p_past_n = test_z[t_start + 1 : t_end + 1]
            next_trend.append((p_past_n[-1] - p_past_n[0]) / (K - 1))

        win_trend_arr  = np.array(win_trend,  dtype=float)
        next_trend_arr = np.array(next_trend, dtype=float)

        y_win_t  = torch.zeros(2, TAU, device=device, dtype=dtype)
        y_win_t[0, :]  = torch.tensor(win_prices,     device=device, dtype=dtype)
        y_win_t[1, :]  = torch.tensor(win_trend_arr,  device=device, dtype=dtype)

        y_next_t = torch.zeros(2, TAU, device=device, dtype=dtype)
        y_next_t[0, :] = torch.tensor(next_prices,    device=device, dtype=dtype)
        y_next_t[1, :] = torch.tensor(next_trend_arr, device=device, dtype=dtype)

    test_input.append(y_win_t)
    test_target.append(y_next_t)
    test_x0.append(x0_vector)
    test_dates_list.append(test_dates[t0 + TAU])

print(f"Test dataset size: {len(test_input)} windows")

# Set up test matrices (same as training - fixed F and H)
sys_model.F_test = F0
sys_model.H_test = H_fixed

# Call NNTest_stocks_last
pred_prices, real_prices, mse_price, rel_err_mean, sq_err_arr, rel_err_arr = RTSNet_Pipeline.NNTest_stocks_last(
    sys_model,
    test_input,
    test_target,
    load_model_path=path_results_rts,
    generate_f=False,
    generate_h=False,
    test_x0=test_x0
)

print(f"\nTest Results:")
print(f"MSE(price): {mse_price.item():.6f}")
print(f"RMSE(price): {torch.sqrt(mse_price).item():.6f}")
print(f"Mean Relative Error: {rel_err_mean.item():.6f}")

# ======================================================
# NOW RUN EMKF ANALYTIC FOR COMPARISON
# ======================================================
print("\n" + "="*80)
print("RUNNING EMKF ANALYTIC FOR COMPARISON")
print("="*80)

from emkf.main_emkf_func import EMKF_FH_analytic

emkf_pred_prices = []
emkf_dates = []

# Warm-starts for EM – reset at start of test period, then carry over between windows
F_prev_emkf = F0.clone()
P0_prev_emkf = P0_default.clone()
# x0 for the first window comes from test_x0[0] (price + trend, no leakage)
x0_prev_emkf = test_x0[0].view(m, 1).detach().clone()

# test_z is the raw price array for the test period (built above the test loop)
# test_input[idx] = test_z[t0 : t0+TAU],  true next price = test_z[t0+TAU]
# The loop starts at t0=K so test_input[idx] maps to test_z[idx+K : idx+K+TAU]
# and the true next-day price is test_z[idx+K+TAU]

for idx in range(len(test_input)):
    t0 = idx + K
    win_prices = test_z[t0 : t0 + TAU]
    true_next_price = float(test_z[t0 + TAU])

    if n == 1:
        # ---- ORIGINAL: 1D observation ----
        Y = torch.tensor(win_prices, device=device, dtype=dtype).view(1, 1, TAU)

    elif n == 2:
        # ---- NEW: [price, trend] 2D observation ----
        win_trend = []
        for ti in range(TAU):
            t_start = t0 + ti - K
            t_end   = t0 + ti
            p_past  = test_z[t_start : t_end]
            win_trend.append((p_past[-1] - p_past[0]) / (K - 1))
        win_trend_arr = np.array(win_trend, dtype=float)
        Y = torch.zeros(1, 2, TAU, device=device, dtype=dtype)
        Y[0, 0, :] = torch.tensor(win_prices,    device=device, dtype=dtype)
        Y[0, 1, :] = torch.tensor(win_trend_arr, device=device, dtype=dtype)

    X_dummy = torch.zeros((1, m, TAU), device=device, dtype=dtype)

    if idx == 0:
        print(f"[EMKF SANITY] y_T price: {win_prices[-1]:.4f}  "
              f"true y_Tp1: {test_target[idx][0, -1].item():.4f}  "
              f"x0: {x0_prev_emkf.squeeze().tolist()}")

    sys_model_emkf = SystemModel(F_prev_emkf, Q, H_fixed, R, TAU, TAU)
    sys_model_emkf.InitSequence(x0_prev_emkf, P0_prev_emkf)

    F_matrices, _, last_x_list, last_P_list = EMKF_FH_analytic(
        sys_model_emkf, [F_prev_emkf], [H_fixed], Q, R, Y, x0_prev_emkf, P0_prev_emkf, X_dummy,
        max_it=max_em_it,
        generate_f=True,
        generate_h=True,
        init_x_list=None,
        init_P_list=None,
        update_F=True,
        update_H=False
    )

    F_hat_emkf = F_matrices[0][-1].detach().clone()
    xT_s_emkf  = last_x_list[0].detach().clone()

    # Price prediction: always use first row of H*F*x_T
    x_next_emkf     = F_hat_emkf @ xT_s_emkf
    pred_price_emkf = (H_fixed[0:1, :] @ x_next_emkf)[0, 0].item()

    if idx < 3:
        print(f"\n[EMKF DEBUG] Window {idx}  t0={t0}")
        print(f"  win_prices first 3: {win_prices[:3].tolist()}  last 3: {win_prices[-3:].tolist()}")
        print(f"  x0_prev_emkf: {x0_prev_emkf.squeeze().tolist()}")
        print(f"  F_hat diagonal: [{F_hat_emkf[0,0].item():.4f}, {F_hat_emkf[1,1].item():.4f}]")
        print(f"  xT_s_emkf: {xT_s_emkf.squeeze().tolist()}")
        print(f"  pred_price_emkf: {pred_price_emkf:.4f}")
        print(f"  true_next_price:  {true_next_price:.4f}")
        print(f"  error: {abs(pred_price_emkf - true_next_price):.4f}")

    emkf_pred_prices.append(pred_price_emkf)
    emkf_dates.append(test_dates_list[idx])

    # Warm-start: carry F and P to next window; x0 from test_x0 for alignment
    F_prev_emkf  = F_hat_emkf
    P0_prev_emkf = last_P_list[0].detach().clone()
    # Use smoothed state as warm-start x0 for next window
    x0_prev_emkf = xT_s_emkf.detach().clone()  # [m, 1]

    if idx % 50 == 0:
        print(f"  EMKF: Processed {idx + 1}/{len(test_input)} windows")
        print(f"  F_hat:\n    {F_hat_emkf.cpu().numpy()}")

print(f"EMKF Analytic: {len(emkf_pred_prices)} predictions")

# ======================================================
# NOW TEST M-NETWORK USING PIPELINE FUNCTION
# ======================================================
print("\n" + "="*80)
print("RUNNING M-NETWORK FOR COMPARISON (using test_mstep_net_price)")
print("="*80)

# Use the SAME test data as RTSNet (no need to recreate)
# test_input, test_target, test_x0 are already created above and used by RTSNet

print(f"Using {len(test_input)} test windows (same as RTSNet)")

# Set F_test for SysModel (test_mstep_net_price expects it)
sys_model_mnet = SystemModel(F0, Q, H_fixed, R, TAU, TAU)
sys_model_mnet.F_test = F0  # Will be updated by M-network during testing
sys_model_mnet.H = H_fixed
sys_model_mnet.H_test = H_fixed
sys_model_mnet.InitSequence(x0_dummy, P0_default)
# Call the pipeline's test function
mean_price_mse_per_iter, mean_price_mse_db_per_iter, final_F_list,pred = RTSNet_Pipeline.test_mstep_net_price(
    SysModel=sys_model_mnet,
    test_input=test_input,  # Same as RTSNet
    test_target=test_target,  # Same as RTSNet
    test_x0=test_x0,  # Same as RTSNet
    destination_path_RTS=path_results_rts,
    destination_path_M=path_results_m,
    num_em_iters=2,
    generate_f=False,
    generate_h=False
)

# Extract M-Network predictions from test_mstep_net_price results
print("\nExtracting M-Network predictions from test results...")

# The pred variable contains a list of dicts with denormalized predictions
mnet_pred_prices = []
mnet_dates = []

for pred_dict in pred:
    # Extract denormalized prediction for y_{T+1}
    y_pred_Tp1 = pred_dict["y_pred_Tp1"]
    if torch.is_tensor(y_pred_Tp1):
        pred_price = y_pred_Tp1.item() if y_pred_Tp1.numel() == 1 else y_pred_Tp1[0].item()
    else:
        pred_price = float(y_pred_Tp1)

    mnet_pred_prices.append(pred_price)

# Print F matrix samples
for i in range(0, len(final_F_list), 50):
    print(f"  M-Network: Window {i + 1}/{len(final_F_list)}")
    print(f"  M-Network Final F matrix at window {i + 1}:")
    print(f"    {final_F_list[i].cpu().numpy()}")

print(f"M-Network: {len(mnet_pred_prices)} predictions")

# Compute M-Network metrics (use same ground truth as RTSNet)
mnet_pred_tensor = torch.tensor(mnet_pred_prices, device=device, dtype=dtype)
# Use real_prices which is the ground truth from RTSNet test
mnet_true_tensor = real_prices[:len(mnet_pred_prices)]

mnet_sq_err = (mnet_pred_tensor - mnet_true_tensor) ** 2
mnet_mse = torch.mean(mnet_sq_err)
mnet_rmse = torch.sqrt(mnet_mse)
mnet_mae = torch.mean(torch.abs(mnet_pred_tensor - mnet_true_tensor))
mnet_rel_err = (mnet_pred_tensor - mnet_true_tensor) / (mnet_true_tensor + 1e-12)
mnet_rel_err_mean = torch.mean(mnet_rel_err)

print(f"\nM-Network Results:")
print(f"MSE(price): {mnet_mse.item():.6f}")
print(f"RMSE(price): {mnet_rmse.item():.6f}")
print(f"MAE(price): {mnet_mae.item():.6f}")
print(f"Mean Relative Error: {mnet_rel_err_mean.item():.6f}")

# Convert EMKF predictions to tensor for comparison (use same ground truth as RTSNet)
emkf_pred_tensor = torch.tensor(emkf_pred_prices, device=device, dtype=dtype)
# Use real_prices which is the ground truth from RTSNet test
emkf_true_tensor = real_prices[:len(emkf_pred_prices)]

# Compute EMKF metrics
emkf_sq_err = (emkf_pred_tensor - emkf_true_tensor) ** 2
emkf_mse = torch.mean(emkf_sq_err)
emkf_rmse = torch.sqrt(emkf_mse)
emkf_mae = torch.mean(torch.abs(emkf_pred_tensor - emkf_true_tensor))
emkf_rel_err = (emkf_pred_tensor - emkf_true_tensor) / (emkf_true_tensor + 1e-12)
emkf_rel_err_mean = torch.mean(emkf_rel_err)

print(f"\nEMKF Results:")
print(f"MSE(price): {emkf_mse.item():.6f}")
print(f"RMSE(price): {emkf_rmse.item():.6f}")
print(f"MAE(price): {emkf_mae.item():.6f}")
print(f"Mean Relative Error: {emkf_rel_err_mean.item():.6f}")

# ======================================================
# NAIVE BASELINE: Tomorrow = Today
# ======================================================
print("\n" + "="*80)
print("NAIVE BASELINE: Tomorrow's price = Today's price")
print("="*80)

# Naive prediction: predict tomorrow's price = today's price (use same test windows)
naive_pred_prices = []
for i in range(len(test_input)):
    # Get the last price in the window
    today_price = test_input[i][0, -1].item()  # Last price in window
    naive_pred_prices.append(today_price)  # Predict tomorrow = today

# Compute Naive baseline metrics (use same ground truth as RTSNet)
naive_pred_tensor = torch.tensor(naive_pred_prices, device=device, dtype=dtype)
naive_true_tensor = real_prices[:len(naive_pred_prices)]

naive_sq_err = (naive_pred_tensor - naive_true_tensor) ** 2
naive_mse = torch.mean(naive_sq_err)
naive_rmse = torch.sqrt(naive_mse)
naive_mae = torch.mean(torch.abs(naive_pred_tensor - naive_true_tensor))
naive_rel_err = (naive_pred_tensor - naive_true_tensor) / (naive_true_tensor + 1e-12)
naive_rel_err_mean = torch.mean(naive_rel_err)

print(f"Naive Baseline Results:")
print(f"MSE(price): {naive_mse.item():.6f}")
print(f"RMSE(price): {naive_rmse.item():.6f}")
print(f"MAE(price): {naive_mae.item():.6f}")
print(f"Mean Relative Error: {naive_rel_err_mean.item():.6f}")

print(f"\nM-Network Results:")
print(f"MSE(price): {mnet_mse.item():.6f}")
print(f"RMSE(price): {mnet_rmse.item():.6f}")
print(f"MAE(price): {mnet_mae.item():.6f}")
print(f"Mean Relative Error: {mnet_rel_err_mean.item():.6f}")
print(f"MSE per EM iteration: {mean_price_mse_per_iter.tolist()}")
print(f"MSE(dB) per EM iteration: {mean_price_mse_db_per_iter.tolist()}")

# ======================================================
# COMPARISON TABLE (ALL FOUR METHODS)
# ======================================================
print("\n" + "="*80)
print("COMPARISON: RTSNet vs EMKF Analytic vs M-Network vs Naive Baseline")
print("="*80)
print(f"{'Metric':<25} {'RTSNet':<18} {'EMKF Analytic':<18} {'M-Network':<18} {'Naive':<18}")
print("-"*97)
print(f"{'MSE':<25} {mse_price.item():<18.6f} {emkf_mse.item():<18.6f} {mnet_mse.item():<18.6f} {naive_mse.item():<18.6f}")
print(f"{'RMSE':<25} {torch.sqrt(mse_price).item():<18.6f} {emkf_rmse.item():<18.6f} {mnet_rmse.item():<18.6f} {naive_rmse.item():<18.6f}")
rtsnet_mae = torch.mean(torch.abs(pred_prices - real_prices)).item()
print(f"{'MAE':<25} {rtsnet_mae:<18.6f} {emkf_mae.item():<18.6f} {mnet_mae.item():<18.6f} {naive_mae.item():<18.6f}")
print(f"{'Mean Relative Error':<25} {rel_err_mean.item():<18.6f} {emkf_rel_err_mean.item():<18.6f} {mnet_rel_err_mean.item():<18.6f} {naive_rel_err_mean.item():<18.6f}")
print("-"*97)

# Determine best prediction
pred_results = [
    ("RTSNet", mse_price.item()),
    ("EMKF Analytic", emkf_mse.item()),
    ("M-Network", mnet_mse.item()),
    ("Naive Baseline", naive_mse.item())
]
best_pred = min(pred_results, key=lambda x: x[1])
print(f"✓ {best_pred[0]} has BEST prediction accuracy (MSE: {best_pred[1]:.6f})")
print("="*80)

# ======================================================
# TRADING SIMULATION ON TEST DATA (RTSNet)
# ======================================================
print("\n" + "="*80)
print("TRADING SIMULATION - RTSNet")
print("="*80)

signal_list = []
equity_str = 1.0
equity_bh  = 1.0
equity_orc = 1.0

false_buy = 0
false_sell = 0
true_buy = 0
true_sell = 0
hold_buy = 0
hold_sell = 0
sig_label = "hold"

for i in range(len(pred_prices)):
    pred_price = pred_prices[i].item()
    true_price = real_prices[i].item()

    # Current price is test_z[i+TAU] (last day of window + 1)
    today_price = test_input[i][0, -1].item()
    tomorrow_price = true_price

    pred_ret = (pred_price - today_price) / (today_price + 1e-12)
    real_ret = (tomorrow_price - today_price) / (today_price + 1e-12)

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
        if sig_label == "buy":
            equity_str *= (1.0 + real_ret)
        elif sig_label == "sell":
            equity_str *= (1.0 - real_ret)

        if real_ret > 0:
            hold_buy += 1
        else:
            hold_sell += 1

    equity_bh *= (1.0 + real_ret)

    if real_ret > 0:
        equity_orc *= (1.0 + real_ret)
    elif real_ret < 0:
        equity_orc *= (1.0 - real_ret)

    signal_list.append(sig_label)

rtsnet_equity_str = equity_str
rtsnet_equity_bh = equity_bh
rtsnet_equity_orc = equity_orc
rtsnet_signals = signal_list.copy()
rtsnet_true_buy = true_buy
rtsnet_false_buy = false_buy
rtsnet_true_sell = true_sell
rtsnet_false_sell = false_sell
rtsnet_hold_buy = hold_buy
rtsnet_hold_sell = hold_sell

print(f"Strategy final multiple: {equity_str:.4f}  Return: {(equity_str-1)*100:+.2f}%")
print(f"Oracle   final multiple: {equity_orc:.4f}  Return: {(equity_orc-1)*100:+.2f}%")
print(f"Buy&Hold final multiple: {equity_bh:.4f}  Return: {(equity_bh-1)*100:+.2f}%")
print("Signals:", collections.Counter(signal_list))
print(f'true buy: {true_buy}, false buy: {false_buy}')
print(f'true sell: {true_sell}, false sell: {false_sell}')
print(f'hold buy: {hold_buy}, hold sell: {hold_sell}')

# ======================================================
# TRADING SIMULATION ON TEST DATA (EMKF)
# ======================================================
print("\n" + "="*80)
print("TRADING SIMULATION - EMKF Analytic")
print("="*80)

emkf_signal_list = []
emkf_equity_str = 1.0
emkf_equity_bh = 1.0
emkf_equity_orc = 1.0

emkf_false_buy = 0
emkf_false_sell = 0
emkf_true_buy = 0
emkf_true_sell = 0
emkf_hold_buy = 0
emkf_hold_sell = 0
emkf_sig_label = "hold"

for i in range(len(emkf_pred_prices)):
    pred_price_emkf = emkf_pred_prices[i]
    true_price_emkf = emkf_true_tensor[i].item()

    # FIXED: today_price is the last price in the window, tomorrow is the target
    today_price = test_input[i][0, -1].item()  # Last price in window
    tomorrow_price = true_price_emkf           # Next day (target)

    pred_ret = (pred_price_emkf - today_price) / (today_price + 1e-12)
    real_ret = (tomorrow_price - today_price) / (today_price + 1e-12)

    if pred_ret > k:
        emkf_sig_label = "buy"
        emkf_equity_str *= (1.0 + real_ret)
        if real_ret > 0:
            emkf_true_buy += 1
        else:
            emkf_false_buy += 1
    elif pred_ret < -k:
        emkf_sig_label = "sell"
        emkf_equity_str *= (1.0 - real_ret)
        if real_ret < 0:
            emkf_true_sell += 1
        else:
            emkf_false_sell += 1
    else:
        if emkf_sig_label == "buy":
            emkf_equity_str *= (1.0 + real_ret)
        elif emkf_sig_label == "sell":
            emkf_equity_str *= (1.0 - real_ret)

        if real_ret > 0:
            emkf_hold_buy += 1
        else:
            emkf_hold_sell += 1

    emkf_equity_bh *= (1.0 + real_ret)

    if real_ret > 0:
        emkf_equity_orc *= (1.0 + real_ret)
    elif real_ret < 0:
        emkf_equity_orc *= (1.0 - real_ret)

    emkf_signal_list.append(emkf_sig_label)

print(f"Strategy final multiple: {emkf_equity_str:.4f}  Return: {(emkf_equity_str-1)*100:+.2f}%")
print(f"Oracle   final multiple: {emkf_equity_orc:.4f}  Return: {(emkf_equity_orc-1)*100:+.2f}%")
print(f"Buy&Hold final multiple: {emkf_equity_bh:.4f}  Return: {(emkf_equity_bh-1)*100:+.2f}%")
print("Signals:", collections.Counter(emkf_signal_list))
print(f'true buy: {emkf_true_buy}, false buy: {emkf_false_buy}')
print(f'true sell: {emkf_true_sell}, false sell: {emkf_false_sell}')
print(f"Hold sell: {emkf_hold_sell}")

# ======================================================
# TRADING SIMULATION - M-Network
# ======================================================
print("\n" + "="*80)
print("TRADING SIMULATION - M-Network")
print("="*80)

mnet_equity_str = 1.0
mnet_equity_bh  = 1.0
mnet_equity_orc = 1.0

mnet_false_buy = 0
mnet_false_sell = 0
mnet_true_buy = 0
mnet_true_sell = 0
mnet_hold_buy = 0
mnet_hold_sell = 0
mnet_sig_label = "hold"

for i in range(len(mnet_pred_prices)):
    pred_price_mnet = mnet_pred_prices[i]
    true_price_mnet = mnet_true_tensor[i].item()

    # FIXED: today_price is the last price in the window, tomorrow is the target
    today_price = test_input[i][0, -1].item()  # Last price in window
    tomorrow_price = true_price_mnet           # Next day (target)

    pred_ret = (pred_price_mnet - today_price) / (today_price + 1e-12)
    real_ret = (tomorrow_price - today_price) / (today_price + 1e-12)

    if pred_ret > k:
        mnet_sig_label = "buy"
        mnet_equity_str *= (1.0 + real_ret)
        if real_ret > 0:
            mnet_true_buy += 1
        else:
            mnet_false_buy += 1
    elif pred_ret < -k:
        mnet_sig_label = "sell"
        mnet_equity_str *= (1.0 - real_ret)
        if real_ret < 0:
            mnet_true_sell += 1
        else:
            mnet_false_sell += 1
    else:
        if mnet_sig_label == "buy":
            mnet_equity_str *= (1.0 + real_ret)
        elif mnet_sig_label == "sell":
            mnet_equity_str *= (1.0 - real_ret)
        if real_ret > 0:
            mnet_hold_buy += 1
        else:
            mnet_hold_sell += 1

    mnet_equity_bh *= (1.0 + real_ret)

    if real_ret > 0:
        mnet_equity_orc *= (1.0 + real_ret)
    elif real_ret < 0:
        mnet_equity_orc *= (1.0 - real_ret)
    else:
        mnet_equity_orc *= 1.0

print(f"Strategy final multiple: {mnet_equity_str:.4f}  Return: {(mnet_equity_str-1)*100:+.2f}%")
print(f"Oracle   final multiple: {mnet_equity_orc:.4f}  Return: {(mnet_equity_orc-1)*100:+.2f}%")
print(f"Buy&Hold final multiple: {mnet_equity_bh:.4f}  Return: {(mnet_equity_bh-1)*100:+.2f}%")
print(f"Signals: buy: {sum([1 for i in range(len(mnet_pred_prices)) if (mnet_pred_prices[i] - test_z[TAU+i])/(test_z[TAU+i]+1e-12) > k])}, sell: {sum([1 for i in range(len(mnet_pred_prices)) if (mnet_pred_prices[i] - test_z[TAU+i])/(test_z[TAU+i]+1e-12) < -k])}")
print(f"True buy: {mnet_true_buy}, false buy: {mnet_false_buy}")
print(f"True sell: {mnet_true_sell}, false sell: {mnet_false_sell}")
print(f"Hold buy: {mnet_hold_buy}")
print(f"Hold sell: {mnet_hold_sell}")

# ======================================================
# TRADING COMPARISON TABLE (ALL THREE METHODS)
# ======================================================
print("\n" + "="*80)
print("TRADING PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Metric':<25} {'RTSNet':<20} {'EMKF Analytic':<20} {'M-Network':<20}")
print("-"*85)
print(f"{'Strategy Return':<25} {(rtsnet_equity_str-1)*100:>+18.2f}% {(emkf_equity_str-1)*100:>+18.2f}% {(mnet_equity_str-1)*100:>+18.2f}%")
print(f"{'Oracle Return':<25} {(rtsnet_equity_orc-1)*100:>+18.2f}% {(emkf_equity_orc-1)*100:>+18.2f}% {(mnet_equity_orc-1)*100:>+18.2f}%")
print(f"{'Buy&Hold Return':<25} {(rtsnet_equity_bh-1)*100:>+18.2f}% {(emkf_equity_bh-1)*100:>+18.2f}% {(mnet_equity_bh-1)*100:>+18.2f}%")
print("-"*85)
print(f"{'True Buy':<25} {rtsnet_true_buy:<20} {emkf_true_buy:<20} {mnet_true_buy:<20}")
print(f"{'False Buy':<25} {rtsnet_false_buy:<20} {emkf_false_buy:<20} {mnet_false_buy:<20}")
print(f"{'True Sell':<25} {rtsnet_true_sell:<20} {emkf_true_sell:<20} {mnet_true_sell:<20}")
print(f"{'False Sell':<25} {rtsnet_false_sell:<20} {emkf_false_sell:<20} {mnet_false_sell:<20}")
print("="*80)

# Determine winner
methods = [
    ("RTSNet", rtsnet_equity_str),
    ("EMKF Analytic", emkf_equity_str),
    ("M-Network", mnet_equity_str)
]
winner = max(methods, key=lambda x: x[1])
print(f"✓ {winner[0]} is BEST (Strategy Return: {(winner[1]-1)*100:+.2f}%)")
print("="*80)

# ======================================================
# REPORT + SAVE RESULTS (ALL THREE METHODS)
# ======================================================
# Collect M-Network signals for CSV
mnet_signal_list = []
for i in range(len(mnet_pred_prices)):
    pred_price_mnet = mnet_pred_prices[i]
    today_price = float(test_z[TAU + i])
    pred_ret = (pred_price_mnet - today_price) / (today_price + 1e-12)
    if pred_ret > k:
        mnet_signal_list.append("buy")
    elif pred_ret < -k:
        mnet_signal_list.append("sell")
    else:
        mnet_signal_list.append("hold")

results_df = pd.DataFrame({
    "Date": test_dates_list,
    "TrueClose": real_prices.tolist(),
    "RTSNet_Pred": pred_prices.tolist(),
    "EMKF_Pred": emkf_pred_prices[:len(test_dates_list)],
    "MNet_Pred": mnet_pred_prices[:len(test_dates_list)],
    "Naive_Pred": naive_pred_prices[:len(test_dates_list)],  # Added Naive
    "RTSNet_Signal": rtsnet_signals,
    "EMKF_Signal": emkf_signal_list[:len(test_dates_list)],
    "MNet_Signal": mnet_signal_list[:len(test_dates_list)]
})

print("\n" + "=" * 80)
print(f"{ticker} | Comparison: RTSNet vs EMKF vs M-Network vs Naive Baseline")
print(f"Test Period: {test_start_date} to {test_end_date}")
print(f"Window TAU={TAU} | k={k_pct:.2f}%")
print("=" * 80)

print("\nPREDICTION ACCURACY:")
print(f"{'Method':<20} {'MSE':<15} {'RMSE':<15} {'MAE':<15}")
print("-" * 65)
print(f"{'RTSNet':<20} {mse_price.item():<15.6f} {torch.sqrt(mse_price).item():<15.6f} {rtsnet_mae:<15.6f}")
print(f"{'EMKF Analytic':<20} {emkf_mse.item():<15.6f} {emkf_rmse.item():<15.6f} {emkf_mae.item():<15.6f}")
print(f"{'M-Network':<20} {mnet_mse.item():<15.6f} {mnet_rmse.item():<15.6f} {mnet_mae.item():<15.6f}")
print(f"{'Naive Baseline':<20} {naive_mse.item():<15.6f} {naive_rmse.item():<15.6f} {naive_mae.item():<15.6f}")

print("\nTRADING PERFORMANCE:")
print(f"{'Method':<20} {'Strategy Return':<20} {'Oracle Return':<20} {'Buy&Hold Return':<20}")
print("-" * 80)
print(f"{'RTSNet':<20} {(rtsnet_equity_str-1)*100:>+18.2f}% {(rtsnet_equity_orc-1)*100:>+18.2f}% {(rtsnet_equity_bh-1)*100:>+18.2f}%")
print(f"{'EMKF Analytic':<20} {(emkf_equity_str-1)*100:>+18.2f}% {(emkf_equity_orc-1)*100:>+18.2f}% {(emkf_equity_bh-1)*100:>+18.2f}%")
print(f"{'M-Network':<20} {(mnet_equity_str-1)*100:>+18.2f}% {(mnet_equity_orc-1)*100:>+18.2f}% {(mnet_equity_bh-1)*100:>+18.2f}%")
print("=" * 80)
print("Note: Naive baseline (tomorrow = today) used for prediction comparison only, no trading strategy.")
print("=" * 80)

out_csv = "spy_comparison_rtsnet_vs_emkf_vs_mnet_vs_naive.csv"
results_df.to_csv(out_csv, index=False)
print("\nSaved:", out_csv)
print(results_df.tail(10))

# ======================================================
# VISUALIZATION: RTSNet vs EMKF vs M-Network vs Naive Comparison
# ======================================================
fig = plt.figure(figsize=(18, 12))

# Plot 1: Time series comparison (ALL FOUR METHODS)
ax1 = plt.subplot(2, 4, 1)
plt.plot(test_dates_list, real_prices.tolist(), label="True Price", linewidth=2, color='black', alpha=0.7)
plt.plot(test_dates_list, pred_prices.tolist(), label="RTSNet", linewidth=1.5, linestyle='--', alpha=0.8)
emkf_aligned = emkf_pred_prices[:len(test_dates_list)]
plt.plot(test_dates_list, emkf_aligned, label="EMKF Analytic", linewidth=1.5, linestyle=':', alpha=0.8)
mnet_aligned = mnet_pred_prices[:len(test_dates_list)]
plt.plot(test_dates_list, mnet_aligned, label="M-Network", linewidth=1.5, linestyle='-.', alpha=0.8)
naive_aligned = naive_pred_prices[:len(test_dates_list)]
plt.plot(test_dates_list, naive_aligned, label="Naive", linewidth=1, linestyle=':', alpha=0.5, color='gray')
plt.title(f"{ticker} Price Predictions\n{test_start_date} to {test_end_date}")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: RTSNet Error
ax2 = plt.subplot(2, 4, 2)
rtsnet_errors = pred_prices - real_prices
rtsnet_errors_list = rtsnet_errors.tolist()
plt.plot(test_dates_list, rtsnet_errors_list, label="RTSNet Error", color='blue', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title(f"RTSNet Prediction Error\nRMSE: {torch.sqrt(mse_price).item():.4f}")
plt.xlabel("Date")
plt.ylabel("Error ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 3: EMKF Error
ax3 = plt.subplot(2, 4, 3)
emkf_errors_tensor = emkf_pred_tensor[:len(test_dates_list)] - real_prices
emkf_errors_list = emkf_errors_tensor.tolist()
plt.plot(test_dates_list, emkf_errors_list, label="EMKF Error", color='orange', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title(f"EMKF Prediction Error\nRMSE: {emkf_rmse.item():.4f}")
plt.xlabel("Date")
plt.ylabel("Error ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 4: M-Network Error
ax4 = plt.subplot(2, 4, 4)
mnet_errors_tensor = mnet_pred_tensor[:len(test_dates_list)] - real_prices
mnet_errors_list = mnet_errors_tensor.tolist()
plt.plot(test_dates_list, mnet_errors_list, label="M-Net Error", color='green', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title(f"M-Network Prediction Error\nRMSE: {mnet_rmse.item():.4f}")
plt.xlabel("Date")
plt.ylabel("Error ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 5: Scatter - RTSNet
ax5 = plt.subplot(2, 4, 5)
plt.scatter(real_prices.tolist(), pred_prices.tolist(), alpha=0.5, s=20, color='blue')
min_price = torch.min(real_prices).item()
max_price = torch.max(real_prices).item()
plt.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect')
plt.xlabel("True Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"RTSNet\nMSE = {mse_price.item():.4f}")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Scatter - EMKF
ax6 = plt.subplot(2, 4, 6)
emkf_pred_aligned = emkf_pred_tensor[:len(test_dates_list)].tolist()
real_aligned = real_prices.tolist()
plt.scatter(real_aligned, emkf_pred_aligned, alpha=0.5, s=20, color='orange')
plt.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect')
plt.xlabel("True Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"EMKF Analytic\nMSE = {emkf_mse.item():.4f}")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 7: Scatter - M-Network
ax7 = plt.subplot(2, 4, 7)
mnet_pred_aligned = mnet_pred_tensor[:len(test_dates_list)].tolist()
plt.scatter(real_aligned, mnet_pred_aligned, alpha=0.5, s=20, color='green')
plt.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect')
plt.xlabel("True Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"M-Network\nMSE = {mnet_mse.item():.4f}")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 8: Error distribution comparison (ALL THREE)
ax8 = plt.subplot(2, 4, 8)
plt.hist(rtsnet_errors_list, bins=30, alpha=0.4, label=f"RTSNet (σ={torch.std(rtsnet_errors).item():.3f})",
         edgecolor='black', color='blue')
plt.hist(emkf_errors_list, bins=30, alpha=0.4, label=f"EMKF (σ={torch.std(emkf_errors_tensor).item():.3f})",
         edgecolor='black', color='orange')
plt.hist(mnet_errors_list, bins=30, alpha=0.4, label=f"M-Net (σ={torch.std(mnet_errors_tensor).item():.3f})",
         edgecolor='black', color='green')
plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
plt.xlabel("Prediction Error ($)")
plt.ylabel("Frequency")
plt.title("Error Distribution Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("spy_comparison_rtsnet_vs_emkf_vs_mnet_vs_naive.png", dpi=150, bbox_inches='tight')
print("\nSaved plot: spy_comparison_rtsnet_vs_emkf_vs_mnet_vs_naive.png")
plt.show()

# ======================================================
# FINAL COMPREHENSIVE SUMMARY
# ======================================================
print("\n" + "="*80)
print("FINAL SUMMARY - ALL RESULTS")
print("="*80)

print("\n1. M-NETWORK TRAINING")
print("-"*80)
print(f"M-Network trained and saved to: {path_results_m}")
print(f"Training completed successfully")
print(f"M-Network learns F matrix dynamics from price data")

print("\n2. PREDICTION ACCURACY COMPARISON")
print("-"*80)
print(f"{'Method':<20} {'MSE':<15} {'RMSE':<15} {'MAE':<15} {'Rel Error':<15}")
print("-"*80)
print(f"{'RTSNet':<20} {mse_price.item():<15.6f} {torch.sqrt(mse_price).item():<15.6f} {rtsnet_mae:<15.6f} {rel_err_mean.item():<15.6f}")
print(f"{'EMKF Analytic':<20} {emkf_mse.item():<15.6f} {emkf_rmse.item():<15.6f} {emkf_mae.item():<15.6f} {emkf_rel_err_mean.item():<15.6f}")
print(f"{'M-Network':<20} {mnet_mse.item():<15.6f} {mnet_rmse.item():<15.6f} {mnet_mae.item():<15.6f} {mnet_rel_err_mean.item():<15.6f}")
print("-"*80)
# Determine best prediction
pred_methods_summary = [
    ("RTSNet", mse_price.item()),
    ("EMKF Analytic", emkf_mse.item()),
    ("M-Network", mnet_mse.item())
]
best_pred_summary = min(pred_methods_summary, key=lambda x: x[1])
second_best = sorted(pred_methods_summary, key=lambda x: x[1])[1]
improvement_pred = ((second_best[1] - best_pred_summary[1]) / second_best[1]) * 100
print(f"WINNER (Prediction): {best_pred_summary[0]} ({improvement_pred:.2f}% better than {second_best[0]})")

print("\n3. TRADING PERFORMANCE COMPARISON")
print("-"*80)
print(f"{'Method':<20} {'Strategy':<15} {'Oracle':<15} {'Buy&Hold':<15}")
print("-"*80)
print(f"{'RTSNet':<20} {(rtsnet_equity_str-1)*100:>+13.2f}% {(rtsnet_equity_orc-1)*100:>+13.2f}% {(rtsnet_equity_bh-1)*100:>+13.2f}%")
print(f"{'EMKF Analytic':<20} {(emkf_equity_str-1)*100:>+13.2f}% {(emkf_equity_orc-1)*100:>+13.2f}% {(emkf_equity_bh-1)*100:>+13.2f}%")
print(f"{'M-Network':<20} {(mnet_equity_str-1)*100:>+13.2f}% {(mnet_equity_orc-1)*100:>+13.2f}% {(mnet_equity_bh-1)*100:>+13.2f}%")
print("-"*80)
# Determine best trading
trade_methods_summary = [
    ("RTSNet", rtsnet_equity_str),
    ("EMKF Analytic", emkf_equity_str),
    ("M-Network", mnet_equity_str)
]
best_trade_summary = max(trade_methods_summary, key=lambda x: x[1])
second_best_trade = sorted(trade_methods_summary, key=lambda x: x[1], reverse=True)[1]
diff_trade = (best_trade_summary[1] - second_best_trade[1]) * 100
print(f"WINNER (Trading): {best_trade_summary[0]} (+{diff_trade:.2f}% return advantage over {second_best_trade[0]})")

print("\n4. SIGNAL QUALITY")
print("-"*80)
print(f"{'Method':<20} {'True Buy':<12} {'False Buy':<12} {'True Sell':<12} {'False Sell':<12}")
print("-"*80)
print(f"{'RTSNet':<20} {rtsnet_true_buy:<12} {rtsnet_false_buy:<12} {rtsnet_true_sell:<12} {rtsnet_false_sell:<12}")
print(f"{'EMKF Analytic':<20} {emkf_true_buy:<12} {emkf_false_buy:<12} {emkf_true_sell:<12} {emkf_false_sell:<12}")
print(f"{'M-Network':<20} {mnet_true_buy:<12} {mnet_false_buy:<12} {mnet_true_sell:<12} {mnet_false_sell:<12}")

print("\n5. FILES GENERATED")
print("-"*80)
print(f"✓ RTSNet model: {path_results_rts}")
print(f"✓ M-Network model: {path_results_m}")
print(f"✓ Results CSV: spy_comparison_rtsnet_vs_emkf_vs_mnet.csv")
print(f"✓ Comparison plot: spy_comparison_rtsnet_vs_emkf_vs_mnet.png")

print("\n" + "="*80)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nDataset: {ticker}")
print(f"Training Period: {start_date} to {end_date}")
print(f"Test Period: {test_start_date} to {test_end_date}")
print(f"Window Size (TAU): {TAU}")
print(f"M-Network EM Iterations: 2")  # Updated from 3
print(f"EMKF Analytic EM Iterations: {max_em_it}")
print("\n" + "="*80)

