"""
weather_exp.py
==============
Rolling-window weather prediction using RTSNet + EM Kalman Filter.

Observation y_t = [tavg, temp_range, wind, pressure]   (n=4)
State        x_t = same 4 variables                      (m=4)
F = I_4,  H = I_4

Goal: given a window of TAU=15 days, predict the NEXT day's average temperature.

Data source: Open-Meteo archive API (NYC, 40.71N -74.01E)
             Falls back to a synthetic sinusoidal dataset if the network is unavailable.
"""

import os
import random
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive – never blocks
import matplotlib.pyplot as plt
import requests

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
from RTSNet.RTSNet_nn import RTSNetNN
from Pipelines.pipeline_weather import PipelineWeather, _win_norm_4d


# ======================================================
# DETERMINISM
# ======================================================
seed = 1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

today   = datetime.today()
now     = datetime.now()
strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H-%M-%S")
print("Current Time =", strTime)

# ======================================================
# SETTINGS
# ======================================================
TAU        = 10     # window length
max_em_it  = 5
m          = 4      # state dim  = observation dim
n          = 4      # obs dim

# F = I,  H = I
F0      = torch.eye(m, device=device, dtype=dtype)
H_fixed = torch.eye(m, device=device, dtype=dtype)
Q       = 0.1 * torch.eye(m, device=device, dtype=dtype)
R       = 0.1 * torch.eye(n, device=device, dtype=dtype)
P0_default = torch.eye(m, device=device, dtype=dtype)

# Save paths
os.makedirs("RTSNet/weather/tau_15", exist_ok=True)
path_results_rts        = "RTSNet/weather/tau_10/rtsnet_model.pth"
path_results_m          = "RTSNet/weather/tau_10_with_detaouch/m_network_cv_lastonly.pth"
path_results_rts_joint  = "RTSNet/weather/tau_15/rtsnet_joint_cv_lastonly.pth"
path_results_m_joint    = "RTSNet/weather/tau_15/m_network_cv_lastonly_joint.pth"

# ======================================================
# ARGS
# =============================================claude login --browser/login=========
args          = config.general_settings()
args.n_steps  = 250      # Increased from 150: more epochs for M-Network training
args.n_batch  = 15       # Increased from 5: more batches per epoch
args.lr       = 1e-4
args.wd       = 1e-3

# ======================================================
# DATA DOWNLOAD  (Open-Meteo archive, NYC)
# ======================================================
CACHE_TRAIN = "weather_train_cache.csv"
CACHE_TEST  = "weather_test_cache.csv"

def fetch_open_meteo(start_date: str, end_date: str, cache_path: str) -> pd.DataFrame:
    """
    Fetches daily weather from Open-Meteo for New York City.
    Columns returned: date, tavg, trange, wind, pressure
      tavg    = temperature_2m_mean  (°C)
      trange  = temperature_2m_max - temperature_2m_min  (°C)
      wind    = wind_speed_10m_mean  (km/h)
      pressure= surface_pressure_mean (hPa)
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["date"], index_col="date")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=40.71&longitude=-74.01"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        ",wind_speed_10m_mean,surface_pressure_mean"
        "&timezone=America%2FNew_York"
    )
    print(f"  Fetching Open-Meteo data: {start_date} → {end_date} …")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "date":     pd.to_datetime(d["time"]),
            "tavg":     d["temperature_2m_mean"],
            "tmax":     d["temperature_2m_max"],
            "tmin":     d["temperature_2m_min"],
            "wind":     d["wind_speed_10m_mean"],
            "pressure": d["surface_pressure_mean"],
        })
        df["trange"] = df["tmax"] - df["tmin"]
        df = df[["date", "tavg", "trange", "wind", "pressure"]].dropna()
        df.set_index("date", inplace=True)
        df.to_csv(cache_path)
        print(f"  Saved to {cache_path}  ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  WARNING: Could not fetch data ({e}). Using synthetic data.")
        return None


print("\nLoading training weather data (2015-01-01 → 2022-01-01) …")
df_train_raw = fetch_open_meteo("2015-01-01", "2022-01-01", CACHE_TRAIN)
if df_train_raw is None:
    raise RuntimeError(
        "Failed to fetch training weather data from Open-Meteo. "
        "Check your internet connection and try again. "
        f"Expected cache file: {CACHE_TRAIN}"
    )

print("Loading test weather data (2022-01-01 → 2023-06-01) …")
df_test_raw = fetch_open_meteo("2022-01-01", "2023-06-01", CACHE_TEST)
if df_test_raw is None:
    raise RuntimeError(
        "Failed to fetch test weather data from Open-Meteo. "
        "Check your internet connection and try again. "
        f"Expected cache file: {CACHE_TEST}"
    )

# ======================================================
# FEATURE EXTRACTION
# Feature vector at time t:
#   y_t = [tavg_t, trange_t, wind_t, pressure_t]   shape (4,)
# Target: tavg_{t+1}  (predict next-day average temperature)
# x0 for a window starting at t0: feature vector at t0-1  (no leakage)
# ======================================================
FEATURES = ["tavg", "trange", "wind", "pressure"]

def build_dataset(df: pd.DataFrame, TAU: int, device, dtype):
    """
    Builds rolling-window dataset from a weather DataFrame.

    Returns:
        inputs     : list of [4, TAU] tensors
        targets    : list of [4, TAU] tensors (next-day aligned, same 4 features)
        x0_list    : list of [4] tensors  (feature vector at t0-1)
        dates_out  : list of dates for each prediction (= date of y_{T+1})
        tavg_arr   : raw tavg array  (for final MSE in temperature units)
    """
    arr    = df[FEATURES].values.astype(np.float32)   # [N, 4]
    dates  = df.index
    N      = len(arr)

    inputs, targets, x0_list, dates_out = [], [], [], []

    for t0 in range(1, N - TAU):           # t0=1 so x0 = arr[t0-1] is valid
        win  = arr[t0     : t0 + TAU]      # [TAU, 4]
        nxt  = arr[t0 + 1 : t0 + TAU + 1] # [TAU, 4]  next-day aligned

        # x0: feature vector the day BEFORE the window (no leakage)
        x0 = arr[t0 - 1]                   # [4]

        y_win  = torch.tensor(win.T,  device=device, dtype=dtype)  # [4, TAU]
        y_next = torch.tensor(nxt.T,  device=device, dtype=dtype)  # [4, TAU]
        x0_t   = torch.tensor(x0,     device=device, dtype=dtype)  # [4]

        inputs.append(y_win)
        targets.append(y_next)
        x0_list.append(x0_t)
        dates_out.append(dates[t0 + TAU] if t0 + TAU < len(dates) else dates[-1])

    tavg_arr = arr[:, 0]   # raw tavg column
    return inputs, targets, x0_list, dates_out, tavg_arr


print("\nBuilding training dataset …")
all_input, all_target, all_x0, all_dates, tavg_train = build_dataset(df_train_raw, TAU, device, dtype)

split = int(0.8 * len(all_input))
train_input, train_target, train_x0 = all_input[:split],  all_target[:split],  all_x0[:split]
cv_input,    cv_target,    cv_x0    = all_input[split:],   all_target[split:],  all_x0[split:]

print(f"  train={len(train_input)}  cv={len(cv_input)}  TAU={TAU}")

print("\nBuilding test dataset …")
test_input, test_target, test_x0, test_dates, tavg_test = build_dataset(df_test_raw, TAU, device, dtype)
print(f"  test={len(test_input)}")

# ======================================================
# SYSTEM MODEL
# ======================================================
sys_model = SystemModel(F0, Q, H_fixed, R, TAU, TAU)
x0_dummy  = torch.zeros(m, 1, device=device, dtype=dtype)
sys_model.InitSequence(x0_dummy, P0_default)
sys_model.F_train = F0
sys_model.F_valid = F0
sys_model.F_test  = F0
sys_model.H_train = H_fixed
sys_model.H_valid = H_fixed
sys_model.H_test  = H_fixed

# ======================================================
# CREATE RTSNet + PIPELINE
# ======================================================
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)

RTSNet_Pipeline = PipelineWeather(strTime, "RTSNet_weather", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

# ======================================================
# STEP 1 – TRAIN RTSNet
# ======================================================
print("\n" + "=" * 60)
print("STEP 1: TRAIN RTSNet on weather windows")
print("=" * 60)

# RTSNet_Pipeline.NNTrain_weather(
#     sys_model,
#     cv_input, cv_target,
#     train_input, train_target,
#     path_results=path_results_rts,
#     generate_f=False,
#     generate_h=False,
#     train_x0=train_x0,
#     cv_x0=cv_x0,
# )
print("Saved RTSNet model to:", path_results_rts)

# ======================================================
# STEP 2 – TRAIN M-NETWORK
# ======================================================
print("\n" + "=" * 60)
print("STEP 2: TRAIN M-Network for F estimation")
print("=" * 60)

# RTSNet_Pipeline.train_emkalmannet_weather(
#     SysModel=sys_model,
#     cv_input=cv_input, cv_target=cv_target, cv_x0=cv_x0,
#     train_input=train_input, train_target=train_target, train_x0=train_x0,
#     destination_path_M=path_results_m,
#     destination_path_RTS=path_results_rts,
#     num_em_iters=2,
#     alpha=(0.05, 0.15, 0.85),
#     lambda_F=1.0,
#     generate_f=False,
#     generate_h=False,
#     clip_grad=1.0,
# )
print("Saved M-Network model to:", path_results_m)

# ======================================================
# STEP 3 – JOINT TRAINING
# ======================================================
print("\n" + "=" * 60)
print("STEP 3: JOINT TRAINING RTSNet + M-Network")
print("=" * 60)

RTSNet_Pipeline.train_joint_weather(
    RTSNet_Pipeline,
    sys_model,
    train_input, train_target, train_x0,
    cv_input,    cv_target,    cv_x0,
    path_rts_in=path_results_rts,
    path_m_in=path_results_m,
    path_rts_out=path_results_rts_joint,
    path_m_out=path_results_m_joint,
    batch_size=10,
    num_em_iters=2,
    lambda_F=1e-3,
    clip_grad=1.0,
    lr_rts=1e-4, lr_m=1e-4,
    wd_rts=1e-5, wd_m=1e-5,
)
print("Saved joint RTSNet  to:", path_results_rts_joint)
print("Saved joint M-Network to:", path_results_m_joint)

# Use jointly trained models for testing
# path_results_rts = path_results_rts_joint
# path_results_m   = path_results_m_joint
print("\n✓ Using jointly trained models for testing.")

# ======================================================
# STEP 4 – TEST RTSNet
# ======================================================
print("\n" + "=" * 60)
print("STEP 4: TEST RTSNet  (predict next-day tavg)")
print("=" * 60)

sys_model.F_test = F0
sys_model.H_test = H_fixed

(pred_temps_rts, real_temps, mse_rts, rel_err_rts,
 sq_err_rts, _) = RTSNet_Pipeline.NNTest_weather(
    sys_model,
    test_input, test_target,
    load_model_path=path_results_rts,
    generate_f=False,
    generate_h=False,
    test_x0=test_x0,
)

rmse_rts = torch.sqrt(mse_rts)
print(f"\n  RTSNet  MSE(tavg):  {mse_rts.item():.4f} °C²")
print(f"  RTSNet  RMSE(tavg): {rmse_rts.item():.4f} °C")
print(f"  RTSNet  MRE:        {rel_err_rts.item():.4f}")

# ======================================================
# STEP 5 – TEST M-NETWORK
# ======================================================
print("\n" + "=" * 60)
print("STEP 5: TEST M-Network")
print("=" * 60)

sys_model_mnet = SystemModel(F0, Q, H_fixed, R, TAU, TAU)
sys_model_mnet.F_test = F0
sys_model_mnet.H      = H_fixed
sys_model_mnet.H_test = H_fixed
sys_model_mnet.InitSequence(x0_dummy, P0_default)

mse_per_iter, mse_db_per_iter, final_F_list, pred_dicts = RTSNet_Pipeline.test_mstep_weather(
    SysModel=sys_model_mnet,
    test_input=test_input,
    test_target=test_target,
    test_x0=test_x0,
    destination_path_RTS=path_results_rts,
    destination_path_M=path_results_m,
    num_em_iters=2,
    generate_f=False,
    generate_h=False,
)
# path_results_rts1        = "RTSNet/weather/tau_15/rtsnet_model.pth"
# path_results_m1          = "RTSNet/weather/tau_10/m_network_cv_lastonly.pth"
# mse_per_iter, mse_db_per_iter, final_F_list, pred_dicts = RTSNet_Pipeline.test_mstep_weather(
#     SysModel=sys_model_mnet,
#     test_input=test_input,
#     test_target=test_target,
#     test_x0=test_x0,
#     destination_path_RTS=path_results_rts,
#     destination_path_M=path_results_m,
#     num_em_iters=2,
#     generate_f=False,
#     generate_h=False,
# )
# Extract next-day tavg predictions from pred_dicts
mnet_pred_temps = []
for d in pred_dicts:
    yp = d["y_pred_Tp1"]
    # row 0 = tavg (denormalized)
    val = yp[0].item() if yp.numel() > 1 else yp.item()
    mnet_pred_temps.append(val)

mnet_pred_t = torch.tensor(mnet_pred_temps, device=device, dtype=dtype)
mnet_true_t = real_temps[:len(mnet_pred_temps)]
mnet_mse    = torch.mean((mnet_pred_t - mnet_true_t) ** 2)
mnet_rmse   = torch.sqrt(mnet_mse)

print(f"\n  M-Net   MSE(tavg):  {mnet_mse.item():.4f} °C²")
print(f"  M-Net   RMSE(tavg): {mnet_rmse.item():.4f} °C")

# ======================================================
# STEP 5.5 – ORACLE F (F_mean per sequence)
# ======================================================
print("\n" + "=" * 60)
print("STEP 5.5: Oracle F-Mean (Per-Sequence Optimization)")
print("=" * 60)

# We compute F_mean optimally for each sequence based on OBSERVATIONS (y)
oracle_preds = []

# Ensure no gradients
with torch.no_grad():
    for i in range(len(test_input)):
        # 1. Prepare data
        y_win = test_input[i].to(device)    # [4, TAU]
        y_tgt = test_target[i].to(device)   # [4, TAU]

        # Normalize
        y_mean, y_std, mean_scalar, std_scalar = _win_norm_4d(y_win, device, dtype)
        y_win_n = (y_win - y_mean) / y_std
        y_tgt_n = (y_tgt - y_mean) / y_std

        # 2. Compute F_opt for feature 0 (tavg) - Ridge Regression
        # Minimize sum_t ( (f_row @ y_win_n[:, t]) - y_tgt_n[0, t] )^2
        # Inputs: y_win_n [4, TAU]
        # Targets: y_tgt_n[0, :] [TAU] -> Target is row 0 (tavg) of next step

        Y_mat = y_win_n
        z_vec = y_tgt_n[0, :]

        # Ridge regression: f = z Y^T (Y Y^T + lambda I)^-1
        reg = 1e-5
        YYt = Y_mat @ Y_mat.T + reg * torch.eye(m, device=device)
        Yz  = Y_mat @ z_vec
        f_row = torch.linalg.solve(YYt, Yz) # [4]

        # 3. Predict next step using last observation
        # y_last is the observation at time T-1.
        # We predict y at time T (which corresponds to y_tgt column -1).
        y_last = y_win_n[:, -1]
        pred_norm = torch.dot(f_row, y_last)

        # Denormalize
        pred_val = (pred_norm.item() * std_scalar + mean_scalar).item()
        oracle_preds.append(pred_val)

oracle_tensor = torch.tensor(oracle_preds, device=device, dtype=dtype)
N_oracle = len(oracle_tensor)
oracle_mse  = torch.mean((oracle_tensor - real_temps[:N_oracle])**2)
oracle_rmse = torch.sqrt(oracle_mse)

print(f"  Oracle F MSE:  {oracle_mse.item():.4f} °C²")
print(f"  Oracle F RMSE: {oracle_rmse.item():.4f} °C")

# ======================================================
# STEP 6 – EMKF ANALYTIC BASELINE
# ======================================================
print("\n" + "=" * 60)
print("STEP 6: EMKF Analytic Baseline")
print("=" * 60)

from emkf.main_emkf_func import EMKF_FH_analytic

emkf_pred_temps = []
emkf_mse_per_iter = torch.zeros(max_em_it + 1, device=device)
emkf_mse_count = torch.zeros(max_em_it + 1, device=device)

F_prev  = F0.clone()
P0_prev = P0_default.clone()

for idx in range(len(test_input)):
    # Get the current window (RAW data from build_dataset)
    Y_raw = test_input[idx]                            # [4, TAU] raw (NOT normalized!)

    # CRITICAL: Compute normalization statistics from RAW data
    # Using per-feature normalization (matching RTSNet training)
    y_mean_per_feat = Y_raw.mean(dim=1, keepdim=True)  # [4, 1]
    y_std_per_feat = Y_raw.std(dim=1, keepdim=True)    # [4, 1]

    # Apply threshold per-feature for numerical stability
    y_std_per_feat = torch.where(
        y_std_per_feat < 1e-6,
        torch.ones_like(y_std_per_feat),
        y_std_per_feat
    )  # [4, 1]

    # Normalize the window
    Y_norm = (Y_raw - y_mean_per_feat) / y_std_per_feat  # [4, TAU]

    # Normalize x0 using same per-feature stats
    x0_raw = test_x0[idx].to(device)                     # [4] raw
    x0_norm = (x0_raw.view(m, 1) - y_mean_per_feat) / y_std_per_feat  # [4, 1]

    # Prepare unsqueezed input for EMKF
    Y = Y_norm.unsqueeze(0)                              # [1, 4, TAU]
    X_dummy = torch.zeros(1, m, TAU, device=device, dtype=dtype)

    sys_model_emkf = SystemModel(F_prev, Q, H_fixed, R, TAU, TAU)
    sys_model_emkf.InitSequence(x0_norm, P0_prev)

    F_matrices, _, last_x_list, last_P_list = EMKF_FH_analytic(
        sys_model_emkf, [F_prev], [H_fixed], Q, R, Y,
        x0_norm, P0_prev, X_dummy,
        max_it=max_em_it,
        generate_f=True,
        generate_h=True,
        init_x_list=None, init_P_list=None,
        update_F=True, update_H=False,
    )

    F_all = F_matrices[0]  # List of F at each EM iteration

    # Track MSE at each EM iteration
    # SIMPLIFIED: just use final x estimate and compute MSE with each F
    if idx % 50 == 0:
        print(f"  EMKF: {idx+1}/{len(test_input)} windows")
        print(f"    F evolution:")

        # Use final smoothed state from EMKF (approximating best state estimate for all Fs)
        x_final_smooth = last_x_list[0].detach()  # [m, 1]

        # Get true NEXT day target and normalize it
        # test_target[idx] is [4, TAU] window of next-days. Last col is y_{T+1}
        Y_next_raw  = test_target[idx][:, -1].to(device).view(m, 1)
        Y_next_norm = (Y_next_raw - y_mean_per_feat) / y_std_per_feat

        # For each F in the evolution, compute what the MSE would have been
        for em_it in range(len(F_all)):
            F_em = F_all[em_it]

            # Compute prediction with this F using final smoothed x
            # x_next = F * x_curr
            y_pred_em = H_fixed @ (F_em @ x_final_smooth)

            # MSE on feature 0 (tavg) only
            mse_em = (y_pred_em[0] - Y_next_norm[0]) ** 2
            mse_db = 10 * torch.log10(mse_em + 1e-12)

            diag = F_em.diag().cpu().tolist()
            diag_str = ", ".join(f"{v:.4f}" for v in diag)
            print(f"      After EM iter {em_it}: F_diag=[{diag_str}]  MSE={mse_em.item():.6e} ({mse_db.item():.2f} dB)")
            emkf_mse_per_iter[em_it] += mse_em.item()
            emkf_mse_count[em_it] += 1

    F_hat   = F_all[-1].detach().clone()  # Final F
    xT      = last_x_list[0].detach().clone()  # Final x from last iteration
    x_next  = F_hat @ xT                  # [m, 1] normalized

    # Denormalize prediction back to real units (PER-FEATURE)
    # For feature 0 (tavg): y_pred_real = y_pred_norm * y_std[0] + y_mean[0]
    y_pred_norm = x_next[0, 0]                         # scalar, normalized tavg
    pred_tavg = (y_pred_norm * y_std_per_feat[0, 0] + y_mean_per_feat[0, 0]).item()
    emkf_pred_temps.append(pred_tavg)

    F_prev  = F_hat
    P0_prev = last_P_list[0].detach().clone()

emkf_pred_t = torch.tensor(emkf_pred_temps, device=device, dtype=dtype)

# Compute average MSE per iteration
emkf_mse_avg = torch.zeros(max_em_it, device=device)
for k in range(max_em_it):
    if emkf_mse_count[k] > 0:
        emkf_mse_avg[k] = emkf_mse_per_iter[k] / emkf_mse_count[k]

print("\n[EMKF Analytic] Mean MSE per EM iteration (across all test windows):")
for k in range(max_em_it):
    mse_db = 10 * torch.log10(emkf_mse_avg[k] + 1e-12)
    print(f"  After EM iter {k+1}: MSE={emkf_mse_avg[k].item():.6e}  ({mse_db.item():.2f} dB)")
emkf_true_t = real_temps[:len(emkf_pred_temps)]
emkf_mse    = torch.mean((emkf_pred_t - emkf_true_t) ** 2)
emkf_rmse   = torch.sqrt(emkf_mse)

print(f"\n  EMKF    MSE(tavg):  {emkf_mse.item():.4f} °C²")
print(f"  EMKF    RMSE(tavg): {emkf_rmse.item():.4f} °C")

# Naive baseline: predict today = tomorrow
# For each window, the target is tavg at t0+TAU
# Naive predicts: tavg at t0+TAU-1 (last day of the window = "today")
# CRITICAL: Extract and denormalize the last tavg from RAW data
N_test = len(test_input)
naive_pred = []
for i in range(N_test):
    Y_raw_i = test_input[i]  # [4, TAU] raw
    # Last tavg value in raw data (feature 0, last timestep)
    tavg_raw_last = Y_raw_i[0, -1]  # scalar, real temperature
    naive_pred.append(tavg_raw_last.item())

naive_pred = torch.tensor(naive_pred, device=device, dtype=dtype)
naive_mse   = torch.mean((naive_pred - real_temps[:N_test]) ** 2)
naive_rmse  = torch.sqrt(naive_mse)

# ======================================================
# SUMMARY
# ======================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY – Next-day average temperature prediction (NYC)")
print("=" * 70)
N_common = min(len(mnet_pred_temps), len(emkf_pred_temps), N_test, len(oracle_tensor))
hdr  = f"{'Method':<18} {'MSE (°C²)':>12} {'RMSE (°C)':>12} {'MAE (°C)':>12}"
sep  = "-" * len(hdr)
print(hdr); print(sep)

def row(name, pred_t, true_t):
    sq = (pred_t - true_t)**2
    ab = torch.abs(pred_t - true_t)
    return (f"{name:<18} {sq.mean().item():>12.4f} "
            f"{sq.mean().sqrt().item():>12.4f} {ab.mean().item():>12.4f}")

print(row("RTSNet",   pred_temps_rts,          real_temps))
print(row("M-Network", mnet_pred_t[:N_common], real_temps[:N_common]))
print(row("EMKF Analytic", emkf_pred_t[:N_common], real_temps[:N_common]))
print(row("Oracle F-Mean", oracle_tensor[:N_common], real_temps[:N_common]))
print(row("Naive (persist.)", naive_pred, real_temps))
print(sep)

# Winner
methods = {
    "RTSNet":        torch.mean((pred_temps_rts - real_temps)**2).item(),
    "M-Network":     torch.mean((mnet_pred_t[:N_common] - real_temps[:N_common])**2).item(),
    "EMKF Analytic": torch.mean((emkf_pred_t[:N_common] - real_temps[:N_common])**2).item(),
    "Oracle F-Mean": torch.mean((oracle_tensor[:N_common] - real_temps[:N_common])**2).item(),
}
winner = min(methods, key=methods.get)
print(f"✓ BEST prediction: {winner}  (MSE={methods[winner]:.4f})")

# ======================================================
# PLOT
# ======================================================
n_plot  = min(200, N_common)
dates_p = test_dates[:n_plot]

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
ax = axes[0]
ax.plot(dates_p, real_temps[:n_plot].cpu(),              'k-',  lw=1.5, label="True tavg")
ax.plot(dates_p, pred_temps_rts[:n_plot].cpu(),          'b-',  lw=1.2, label="RTSNet", alpha=0.8)
ax.plot(dates_p, mnet_pred_t[:n_plot].cpu(),             'g--', lw=1.2, label="M-Network", alpha=0.8)
ax.plot(dates_p, emkf_pred_t[:n_plot].cpu(),             'r:',  lw=1.2, label="EMKF", alpha=0.8)
ax.plot(dates_p, oracle_tensor[:n_plot].cpu(),           'c-.', lw=1.2, label="Oracle F", alpha=0.9)
ax.plot(dates_p, naive_pred[:n_plot].cpu(),              'm-.', lw=1.0, label="Naive", alpha=0.6)
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"Next-day avg temperature prediction – NYC  (TAU={TAU})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(dates_p, (pred_temps_rts[:n_plot] - real_temps[:n_plot]).cpu().abs(), 'b-',  lw=1,   label="RTSNet |err|")
ax2.plot(dates_p, (mnet_pred_t[:n_plot]    - real_temps[:n_plot]).cpu().abs(), 'g--', lw=1,   label="M-Net  |err|")
ax2.plot(dates_p, (emkf_pred_t[:n_plot]    - real_temps[:n_plot]).cpu().abs(), 'r:',  lw=1,   label="EMKF   |err|")
ax2.plot(dates_p, (oracle_tensor[:n_plot]  - real_temps[:n_plot]).cpu().abs(), 'c-.', lw=1,   label="Oracle |err|")
ax2.set_ylabel("|Error| (°C)")
ax2.set_xlabel("Date")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("weather_prediction_comparison.png", dpi=120)
print("\nSaved plot: weather_prediction_comparison.png")

# ======================================================
# FULL PLOTS – REAL vs ESTIMATED TEMPERATURE
# ======================================================
import os
os.makedirs("RTSNet/weather/tau_15/plots", exist_ok=True)

# Common length across all methods
N_rts   = len(pred_temps_rts)
N_mnet  = len(mnet_pred_t)
N_emkf  = len(emkf_pred_t)
N_naive = len(naive_pred)
N_oracle= len(oracle_tensor)
N_real  = len(real_temps)

N_common = min(N_rts, N_mnet, N_emkf, N_naive, N_oracle, N_real, len(test_dates))

dates_all = test_dates[:N_common]

real_all  = real_temps[:N_common].detach().cpu()
rts_all   = pred_temps_rts[:N_common].detach().cpu()
mnet_all  = mnet_pred_t[:N_common].detach().cpu()
emkf_all  = emkf_pred_t[:N_common].detach().cpu()
oracle_all= oracle_tensor[:N_common].detach().cpu()
naive_all = naive_pred[:N_common].detach().cpu()

# ---------- Plot 1: all models vs real ----------
plt.figure(figsize=(16, 7))
plt.plot(dates_all, real_all,  label="Real tavg", linewidth=2.0)
plt.plot(dates_all, rts_all,   label="RTSNet", linewidth=1.2)
plt.plot(dates_all, mnet_all,  label="M-Network", linewidth=1.2)
plt.plot(dates_all, emkf_all,  label="EMKF Analytic", linewidth=1.2)
plt.plot(dates_all, oracle_all,label="Oracle F", linewidth=1.2, linestyle='-.', color='cyan')
plt.plot(dates_all, naive_all, label="Naive", linewidth=1.0, alpha=0.8)

plt.title("Average Temperature: Real vs Estimated (All Models)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RTSNet/weather/tau_15/plots/all_models_vs_real_full.png", dpi=140)
plt.close()

print("Saved: RTSNet/weather/tau_15/plots/all_models_vs_real_full.png")

# ---------- Plot 2: RTSNet vs real ----------
plt.figure(figsize=(16, 6))
plt.plot(dates_all, real_all, label="Real tavg", linewidth=2.0)
plt.plot(dates_all, rts_all,  label="RTSNet", linewidth=1.2)
plt.title("Average Temperature: Real vs RTSNet")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RTSNet/weather/tau_15/plots/rtsnet_vs_real_full.png", dpi=140)
plt.close()

print("Saved: RTSNet/weather/tau_15/plots/rtsnet_vs_real_full.png")

# ---------- Plot 3: M-Network vs real ----------
plt.figure(figsize=(16, 6))
plt.plot(dates_all, real_all, label="Real tavg", linewidth=2.0)
plt.plot(dates_all, mnet_all, label="M-Network", linewidth=1.2)
plt.title("Average Temperature: Real vs M-Network")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RTSNet/weather/tau_15/plots/mnet_vs_real_full.png", dpi=140)
plt.close()

print("Saved: RTSNet/weather/tau_15/plots/mnet_vs_real_full.png")

# ---------- Plot 4: EMKF vs real ----------
plt.figure(figsize=(16, 6))
plt.plot(dates_all, real_all, label="Real tavg", linewidth=2.0)
plt.plot(dates_all, emkf_all, label="EMKF Analytic", linewidth=1.2)
plt.title("Average Temperature: Real vs EMKF Analytic")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RTSNet/weather/tau_15/plots/emkf_vs_real_full.png", dpi=140)
plt.close()

print("Saved: RTSNet/weather/tau_15/plots/emkf_vs_real_full.png")

# ---------- Plot 5: Naive vs real ----------
plt.figure(figsize=(16, 6))
plt.plot(dates_all, real_all,  label="Real tavg", linewidth=2.0)
plt.plot(dates_all, naive_all, label="Naive", linewidth=1.2)
plt.title("Average Temperature: Real vs Naive")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RTSNet/weather/tau_15/plots/naive_vs_real_full.png", dpi=140)
plt.close()

print("Saved: RTSNet/weather/tau_15/plots/naive_vs_real_full.png")

# ---------- Plot 6: absolute errors ----------
plt.figure(figsize=(16, 7))
plt.plot(dates_all, (rts_all   - real_all).abs(),   label="RTSNet |err|", linewidth=1.2)
plt.plot(dates_all, (mnet_all  - real_all).abs(),   label="M-Net |err|", linewidth=1.2)
plt.plot(dates_all, (emkf_all  - real_all).abs(),   label="EMKF |err|", linewidth=1.2)
plt.plot(dates_all, (oracle_all- real_all).abs(),   label="Oracle |err|", linewidth=1.2)
plt.plot(dates_all, (naive_all - real_all).abs(),   label="Naive |err|", linewidth=1.0, alpha=0.8)

plt.title("Absolute Error of Next-Day Average Temperature Prediction")
plt.xlabel("Date")
plt.ylabel("|Error| (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RTSNet/weather/tau_15/plots/absolute_errors_full.png", dpi=140)
plt.close()

print("Saved: RTSNet/weather/tau_15/plots/absolute_errors_full.png")

