"""
weather_exp_temp.py
===================
Rolling-window weather prediction: tavg is a HIDDEN state dimension.

State  x_t = [tavg, trange, wind, pressure]   (m=4)
Obs    y_t = [trange, wind, pressure]           (n=3)   tavg is NOT observed
F = I_4,  H = [[0,1,0,0],[0,0,1,0],[0,0,0,1]]

Goal: given a window of TAU=10 days of observations (no tavg), estimate the
hidden state and predict the NEXT day's average temperature (tavg).

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
from Pipelines.pipeline_weather_temp import PipelineWeather, _win_norm_4d, _win_norm_3d
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test


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
max_em_it  = 3
m          = 4      # state dim: [tavg, trange, wind, pressure]
n          = 3      # obs dim:   [trange, wind, pressure]  (tavg is hidden)

F0      = torch.eye(m, device=device, dtype=dtype)
# H selects trange, wind, pressure — drops tavg (row 0 of state)
H_fixed = torch.zeros(n, m, device=device, dtype=dtype)
H_fixed[0, 1] = 1.0   # obs[0] = state[1] = trange
H_fixed[1, 2] = 1.0   # obs[1] = state[2] = wind
H_fixed[2, 3] = 1.0   # obs[2] = state[3] = pressure
Q          = 0.05 * torch.eye(m, device=device, dtype=dtype)
R          = 0.05 * torch.eye(n, device=device, dtype=dtype)
P0_default = torch.eye(m, device=device, dtype=dtype)

# Save paths
os.makedirs("../RTSNet/old_attempts/weather/temptau_10", exist_ok=True)
path_results_rts        = "../RTSNet/old_attempts/weather_temp/tau_10/rtsnet_model.pth"
path_results_m          = "../RTSNet/old_attempts/weather_temp/tau_10_/m_network.pth"
path_results_rts_joint  = "../RTSNet/old_attempts/weather_temp/tau_10/rtsnet_joint.pth"
path_results_m_joint    = "../RTSNet/old_attempts/weather_temp/tau_10/m_network_joint.pth"

# ======================================================
# ARGS
# =============================================claude login --browser/login=========
args          = config.general_settings()
args.n_steps  = 200      # Increased from 150: more epochs for M-Network training
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
STATE_FEATURES = ["tavg", "trange", "wind", "pressure"]  # m=4
OBS_FEATURES   = ["trange", "wind", "pressure"]           # n=3  (tavg hidden)

def build_dataset(df: pd.DataFrame, TAU: int, device, dtype):
    """
    Builds rolling-window dataset from a weather DataFrame for STATE-BASED learning.

    DATA LAYOUT:
    ============
    State:  x_t = [tavg, trange, wind, pressure]          (m=4, includes hidden tavg)
    Obs:    y_t = [trange, wind, pressure]                (n=3, tavg is HIDDEN)
    H matrix: (n×m) extracts obs from state:  y = H @ x   (drops component 0)

    DATASET SHAPES:
    ===============
    For each window starting at time t0:
    - inputs[i]       : [n=3, TAU]  ← Observation window (y_t, ..., y_{t+TAU-1})
    - targets[i]      : [n=3, TAU]  ← Next-day obs (y_{t+1}, ..., y_{t+TAU})
    - x0_list[i]      : [m=4]       ← Initial state at t0-1 (for RTSNet init)
    - state_wins[i]   : [m=4, TAU]  ← True state window (for LOSS computation)
    - next_tavg[i]    : float       ← Ground truth tavg at t0+TAU (prediction target)

    LOSS COMPUTATION:
    =================
    Loss = MSE(x_smooth[0, :], x_true[0, :])  ← Compare TAVG only (component 0)
    (Not observation loss, but STATE loss on hidden tavg)

    Returns:
        inputs      : list of [n=3, TAU] obs tensors
        targets     : list of [n=3, TAU] next-day obs tensors
        x0_list     : list of [m=4] state vectors (for init)
"""
    arr_state = df[STATE_FEATURES].values.astype(np.float32)  # [N, 4]
    arr_obs   = df[OBS_FEATURES].values.astype(np.float32)    # [N, 3]
    dates     = df.index
    N         = len(arr_state)

    inputs, targets, x0_list = [], [], [],

    for t0 in range(1, N - TAU - 1):   # -1 ensures t0+TAU < N (need next-day tavg)
        win_obs   = arr_obs  [t0     : t0 + TAU]       # [TAU, 3]
        win_state = arr_state[t0     : t0 + TAU]       # [TAU, 4]
        x0_state  = arr_state[t0 - 1]                  # [4]

        y_win       = torch.tensor(win_obs.T,   device=device, dtype=dtype)  # [3, TAU]
        x_state_win = torch.tensor(win_state.T, device=device, dtype=dtype)  # [4, TAU]
        x0_t        = torch.tensor(x0_state,    device=device, dtype=dtype)  # [4]

        inputs.append(y_win)
        targets.append(x_state_win)
        x0_list.append(x0_t)

    return inputs, targets, x0_list


print("\nBuilding training dataset …")
(all_input, all_target, all_x0) = build_dataset(df_train_raw, TAU, device, dtype)

split = int(0.8 * len(all_input))
train_input,    train_target,    train_x0    = all_input[:split],  all_target[:split],  all_x0[:split]
cv_input,       cv_target,       cv_x0       = all_input[split:],  all_target[split:],  all_x0[split:]


print(f"  train={len(train_input)}  cv={len(cv_input)}  TAU={TAU}")

print("\nBuilding test dataset …")
(test_input, test_target, test_x0) = build_dataset(df_test_raw, TAU, device, dtype)
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

# RTSNet_Pipeline.train_joint_weather(
#     sys_model,
#     train_input, train_target, train_x0,
#     cv_input,    cv_target,    cv_x0,
#     path_rts_in=path_results_rts,
#     path_m_in=path_results_m,
#     path_rts_out=path_results_rts_joint,
#     path_m_out=path_results_m_joint,
#     batch_size=10,
#     num_em_iters=2,
#     lambda_F=1e-3,
#     clip_grad=1.0,
#     lr_rts=1e-4, lr_m=1e-4,
#     wd_rts=1e-5, wd_m=1e-5,
# )
print("Saved joint RTSNet  to:", path_results_rts_joint)
print("Saved joint M-Network to:", path_results_m_joint)

# Use jointly trained models for testing
# path_results_rts = path_results_rts_joint
# path_results_m   = path_results_m_joint
print("\n✓ Using jointly trained models for testing.")

# ======================================================
# Training Complete
# ======================================================
print("\n" + "=" * 60)
print("✓ ALL TRAINING COMPLETE!")
print("=" * 60)

sys_model.F_test = F0
sys_model.H_test = H_fixed

(mse_rts, rel_err_rts,
 sq_err_r,rel_err_list) = RTSNet_Pipeline.NNTest_weather(
    sys_model,
    test_input, test_target,
    load_model_path=path_results_rts,
    test_x0=test_x0,)

rmse_rts = torch.sqrt(mse_rts)
print(f"\n  RTSNet  MSE(tavg):  {mse_rts.item():.4f} °C²")
print(f"  RTSNet  RMSE(tavg): {rmse_rts.item():.4f} °C")
print(f"  RTSNet  MRE:        {rel_err_rts.item():.4f}")

# ======================================================
# STEP 5 – NAIVE PREDICTION (x_0 persistence)
# ======================================================
print("\n" + "=" * 60)
print("STEP 5: Naive Prediction (x_pred = x_0)")
print("=" * 60)

naive_sq_err = []
for i in range(len(test_input)):
    # Naive: predicted tavg is tavg from initial state x0 (persistence)
    x_true = test_target[i]
    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
    tavg_pred = (test_x0[i][0] - x_mean[0]) / x_std[0]
    tavg_true = ((x_true - x_mean) / x_std)[0, :]

    loss_i = torch.mean((tavg_pred - tavg_true) ** 2)
    naive_sq_err.append((loss_i) )
naive_mse = torch.tensor(naive_sq_err).mean()
naive_rmse = torch.sqrt(naive_mse)

print(f"  Naive   MSE(tavg):  {naive_mse.item():.4f} °C²")
print(f"  Naive   RMSE(tavg): {naive_rmse.item():.4f} °C")

# Compare
print("\n" + "-" * 60)
print(f"{'Method':<15} {'MSE':<10} {'RMSE':<10}")
print("-" * 60)
print(f"{'RTSNet':<15} {mse_rts.item():<10.4f} {rmse_rts.item():<10.4f}")
print(f"{'Naive (x0)':<15} {naive_mse.item():<10.4f} {naive_rmse.item():<10.4f}")
print("-" * 60)

# ======================================================
# STEP 6 – TEST M-NETWORK
# ======================================================
print("\n" + "=" * 60)
print("STEP 6: TEST M-Network")
print("=" * 60)

mse_per_iter, mse_db_per_iter, final_F_list, pred_dicts = RTSNet_Pipeline.test_mstep_weather(
    SysModel=sys_model,
    test_input=test_input,
    test_target=test_target,
    test_x0=test_x0,
    destination_path_RTS=path_results_rts,
    destination_path_M=path_results_m,
    num_em_iters=2,
    print_F_every=50,
)

# Extract tavg predictions from pred_dicts
# Compare M-Net on the same quantity it returns: full-window normalized x[0, :]
mnet_sq_err = []

for d in pred_dicts:
    x_pred_norm = d["x_pred_norm"]   # [m, T]
    x_true_norm = d["x_true_norm"]   # [m, T]

    pred_tavg_norm = x_pred_norm[0, :]
    true_tavg_norm = x_true_norm[0, :]

    mse_i = torch.mean((pred_tavg_norm - true_tavg_norm) ** 2)
    mnet_sq_err.append(mse_i)

if mnet_sq_err:
    mnet_mse = torch.stack(mnet_sq_err).mean().to(device)
    mnet_rmse = torch.sqrt(mnet_mse)

    print(f"\n  M-Net   MSE(tavg):  {mnet_mse.item():.4f}")
    print(f"  M-Net   RMSE(tavg): {mnet_rmse.item():.4f}")
else:
    print("\n  M-Net: No predictions generated")

# ======================================================
# STEP 6B – TEST EMKF (Existing EMKF_FH_analytic)
# ======================================================
print("\n" + "=" * 60)
print("STEP 6B: TEST EMKF (Classical EM Kalman Filter)")
print("=" * 60)

from emkf.main_emkf_func import EMKF_FH_analytic, compute_A1, compute_A2

emkf_sq_err = []

for idx in range(len(test_input)):
    y_win = test_input[idx].to(device)   # [n, TAU]
    x_true = test_target[idx].to(device) # [m, TAU]

    # Normalize
    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
    y_win_n = (y_win - x_mean[1:]) / x_std[1:]
    x_true_n = (x_true - x_mean) / x_std

    T = y_win.size(-1)
    x0_raw = test_x0[idx].to(device)
    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

    # Prepare inputs for EMKF_FH_analytic
    Y = y_win_n.unsqueeze(0)  # [1, n, T]
    X_true_input = x_true_n.unsqueeze(0)  # [1, m, T]

    # Run EMKF with F and H learning
    F_matrices, H_matrices, last_x_list, last_P_list, smooth_x_list = EMKF_FH_analytic(
        sys_model,
        [F0],  # F_init_list
        [H_fixed],  # H_init_list
        Q, R,
        Y,  # observations
        x0_norm.squeeze(),  # x_0
        P0_default,  # P_0
        X_true_input,  # X_true
        max_it=2,  # num_em_iters
        generate_f=True,
        generate_h=False,
        update_F=True,
        update_H=False,
        init_x_list=[x0_norm.squeeze()],
        init_P_list=[P0_default],
    )

    # Extract full smoothed trajectory [m, T]
    x_sm_final = smooth_x_list[0]

    # Compute MSE on tavg (normalized) over full window
    pred_tavg_norm = x_sm_final[0, :]
    true_tavg_norm = x_true_n[0, :]

    mse_emkf_win = torch.mean((pred_tavg_norm - true_tavg_norm) ** 2)
    emkf_sq_err.append(mse_emkf_win.item())

if emkf_sq_err and not all(torch.isnan(torch.tensor(emkf_sq_err))):
    emkf_mse = torch.tensor([e for e in emkf_sq_err if not np.isnan(e)]).mean()
    emkf_rmse = torch.sqrt(emkf_mse)

    print(f"\n  EMKF    MSE(tavg):  {emkf_mse.item():.4f}")
    print(f"  EMKF    RMSE(tavg): {emkf_rmse.item():.4f}")
    use_emkf = True
else:
    print("\n  EMKF: No valid results")
    emkf_mse = torch.tensor(float('nan'))
    emkf_rmse = torch.tensor(float('nan'))
    use_emkf = False


# ======================================================
# STEP 7 – COMPARISON TABLE
# ======================================================
print("\n" + "=" * 60)
print("COMPARISON: All Methods")
print("=" * 60)
print(f"{'Method':<20} {'MSE':<12} {'RMSE':<12}")
print("-" * 60)
print(f"{'RTSNet':<20} {mse_rts.item():<12.4f} {rmse_rts.item():<12.4f}")
if mnet_sq_err:
    print(f"{'M-Network':<20} {mnet_mse.item():<12.4f} {mnet_rmse.item():<12.4f}")
if use_emkf:
    print(f"{'EMKF':<20} {emkf_mse.item():<12.4f} {emkf_rmse.item():<12.4f}")
print(f"{'Naive (x0)':<20} {naive_mse.item():<12.4f} {naive_rmse.item():<12.4f}")
print("-" * 60)

