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
# matplotlib.use("Agg")   # non-interactive – never blocks
import matplotlib.pyplot as plt
import requests

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
from RTSNet.RTSNet_nn import RTSNetNN
from Pipelines.pipeline_weather_temp import PipelineWeather, _win_norm_4d, _win_norm_3d
from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test
from emkf.main_emkf_func import EMKF_FH_analytic, compute_A1, compute_A2

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
os.makedirs("RTSNet/weather/temptau_10", exist_ok=True)
path_results_rts        = "RTSNet/weather_temp/tau_10/rtsnet_model.pth"
path_results_m          = "RTSNet/weather_temp/tau_10_/m_network.pth"
path_results_rts_joint  = "RTSNet/weather_temp/tau_10/rtsnet_joint.pth"
path_results_m_joint    = "RTSNet/weather_temp/tau_10/m_network_joint.pth"

# ======================================================
# ARGS
# =====================================================
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
    Builds NON-OVERLAPPING consecutive-window dataset from a weather DataFrame for STATE-BASED learning.

    DATA LAYOUT:
    ============
    State:  x_t = [tavg, trange, wind, pressure]          (m=4, includes hidden tavg)
    Obs:    y_t = [trange, wind, pressure]                (n=3, tavg is HIDDEN)
    H matrix: (n×m) extracts obs from state:  y = H @ x   (drops component 0)

    DATASET SHAPES (NON-OVERLAPPING):
    ==================================
    For each non-overlapping window of length TAU:
    - Sequence 0: days 0 to TAU-1
    - Sequence 1: days TAU to 2*TAU-1
    - Sequence 2: days 2*TAU to 3*TAU-1
    - etc.

    For each sequence:
    - inputs[i]       : [n=3, TAU]  ← Observation window (y_t, ..., y_{t+TAU-1})
    - targets[i]      : [m=4, TAU]  ← True state window (for LOSS computation)
    - x0_list[i]      : [m=4]       ← Initial state at t0-1 (for RTSNet init)

    LOSS COMPUTATION:
    =================
    Loss = MSE(x_smooth[0, :], x_true[0, :])  ← Compare TAVG only (component 0)
    (Not observation loss, but STATE loss on hidden tavg)

    Returns:
        inputs      : list of [n=3, TAU] obs tensors
        targets     : list of [m=4, TAU] state tensors
        x0_list     : list of [m=4] state vectors (for init)
"""
    arr_state = df[STATE_FEATURES].values.astype(np.float32)  # [N, 4]
    arr_obs   = df[OBS_FEATURES].values.astype(np.float32)    # [N, 3]
    dates     = df.index
    N         = len(arr_state)

    inputs, targets, x0_list = [], [], []

    # Create non-overlapping consecutive sequences
    t0 = 1  # Start from day 1 (day 0 used for x0)
    while t0 + TAU <= N:
        win_obs   = arr_obs  [t0     : t0 + TAU]       # [TAU, 3]
        win_state = arr_state[t0     : t0 + TAU]       # [TAU, 4]
        x0_state  = arr_state[t0 - 1]                  # [4] - state at day before window

        y_win       = torch.tensor(win_obs.T,   device=device, dtype=dtype)  # [3, TAU]
        x_state_win = torch.tensor(win_state.T, device=device, dtype=dtype)  # [4, TAU]
        x0_t        = torch.tensor(x0_state,    device=device, dtype=dtype)  # [4]

        inputs.append(y_win)
        targets.append(x_state_win)
        x0_list.append(x0_t)

        # Move to next non-overlapping window
        t0 += TAU

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
 sq_err_rts, rel_err_list, mse_rts_denorm, sq_err_rts_denorm, rts_preds) = RTSNet_Pipeline.NNTest_weather(
    sys_model,
    test_input, test_target,
    load_model_path=path_results_rts,
    test_x0=test_x0,)

rmse_rts = torch.sqrt(mse_rts)
rmse_rts_denorm = torch.sqrt(mse_rts_denorm)
print(f"\n  RTSNet  MSE(tavg):  {mse_rts.item():.4f} (normalized)")
print(f"  RTSNet  MSE(tavg):  {mse_rts_denorm.item():.4f} °C² (denormalized)")
print(f"  RTSNet  RMSE(tavg): {rmse_rts_denorm.item():.4f} °C")
print(f"  RTSNet  MRE:        {rel_err_rts.item():.4f}")


# ======================================================
# STEP 5 – NAIVE PREDICTION (x_0 persistence)
# ======================================================
print("\n" + "=" * 60)
print("STEP 5: Naive Prediction (x_pred = x_0, in real units)")
print("=" * 60)

naive_sq_err_denorm = []
for i in range(len(test_input)):
    x_true = test_target[i].to(device)
    x0_raw = test_x0[i].to(device)
    # Naive: constant tavg over window, compare against true tavg in raw units
    tavg_pred = x0_raw[0]
    tavg_true = x_true[0, :]
    loss_i = torch.mean((tavg_pred - tavg_true) ** 2)
    naive_sq_err_denorm.append(loss_i.item())

naive_mse_denorm = torch.tensor(naive_sq_err_denorm).mean()
naive_rmse_denorm = torch.sqrt(naive_mse_denorm)

print(f"  Naive   MSE(tavg):  {naive_mse_denorm.item():.4f} °C²")
print(f"  Naive   RMSE(tavg): {naive_rmse_denorm.item():.4f} °C")

# ======================================================
# STEP 5B – CLASSICAL RTS SMOOTHER BASELINE
# ======================================================
print("\n" + "=" * 60)
print("STEP 5B: Classical RTS Smoother (F=F_0, H=H_fixed)")
print("=" * 60)

rts_classic_sq_err_denorm = []
all_rts_classic_smooth_x = []

with torch.no_grad():
    for j in range(len(test_input)):
        y_win  = test_input[j].to(device)
        x_true = test_target[j].to(device)
        T      = y_win.size(-1)

        # Get normalization stats
        x_mean, x_std, x_mean0, x_std0 = _win_norm_4d(x_true, device, dtype)
        y_win_n = (y_win - x_mean[1:]) / x_std[1:]
        x_true_n = (x_true - x_mean) / x_std

        # Normalize x0
        x0_raw = test_x0[j].to(device)
        x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

        # Run classical Kalman Filter forward + RTS backward
        # S_Test expects lists of tensors, not single tensors
        [_mse_arr, _mse_avg, _mse_db, X_smooth, P_smooth, V_smooth] = S_Test(
            sys_model,
            [y_win_n],              # test_input as list
            [x_true_n],             # test_target as list (normalized for loss)
            F=[F0],                 # F as list
            H=[H_fixed],            # H as list
            generate_f=False,       # Don't use j//10 indexing
            generate_h=False,
            init_x_list=[x0_norm.squeeze()],
            init_P_list=[P0_default],
        )

        x_rts_smooth = X_smooth[0]  # [m, T] - extract first (only) sequence

        # Denormalize to real units
        x_rts_denorm = x_rts_smooth.clone()
        for i in range(m):
            x_rts_denorm[i, :] = x_rts_smooth[i, :] * x_std[i] + x_mean[i]
        all_rts_classic_smooth_x.append(x_rts_denorm.detach().cpu())

        # Compute denormalized tavg MSE
        pred_tavg_real = x_rts_denorm[0, :]
        true_tavg_real = x_true[0, :]
        mse_i_denorm = torch.mean((pred_tavg_real - true_tavg_real) ** 2)
        rts_classic_sq_err_denorm.append(mse_i_denorm.item())

rts_classic_mse_denorm = torch.tensor(rts_classic_sq_err_denorm).mean()
rts_classic_rmse_denorm = torch.sqrt(rts_classic_mse_denorm)

print(f"  RTS(classic) MSE(tavg):  {rts_classic_mse_denorm.item():.4f} °C²")
print(f"  RTS(classic) RMSE(tavg): {rts_classic_rmse_denorm.item():.4f} °C")
#########################delete after diagnostics#########################
# DIAGNOSTIC: Check if RTS(classic) collapses to Naive for hidden tavg
print("\n  DIAGNOSTIC: Checking if RTS(classic) ≈ Naive (x0 persistence)")
print("  ─" * 40)
num_diagnostic_windows = min(3, len(test_input))
for diag_idx in range(num_diagnostic_windows):
    x_rts_denorm = all_rts_classic_smooth_x[diag_idx]  # [m, T] in real units
    x_true = test_target[diag_idx].to(device)
    x0_raw = test_x0[diag_idx].to(device)

    rts_tavg_traj = x_rts_denorm[0, :].numpy()  # [T]
    naive_tavg_const = float(x0_raw[0].item())  # scalar, constant

    # Max abs difference between RTS tavg trajectory and naive constant
    diff_from_naive = np.abs(rts_tavg_traj - naive_tavg_const)
    max_diff = np.max(diff_from_naive)
    mean_diff = np.mean(diff_from_naive)

    print(f"  Window {diag_idx}: max|RTS_tavg - x0| = {max_diff:.6e}, mean|diff| = {mean_diff:.6e}")

if num_diagnostic_windows > 0:
    all_diffs = []
    for diag_idx in range(num_diagnostic_windows):
        x_rts_denorm = all_rts_classic_smooth_x[diag_idx]
        x0_raw = test_x0[diag_idx].to(device)
        naive_tavg_const = float(x0_raw[0].item())
        rts_tavg_traj = x_rts_denorm[0, :].numpy()
        all_diffs.extend(np.abs(rts_tavg_traj - naive_tavg_const).tolist())

    if np.max(all_diffs) < 1e-5:
        print("\n  ✓ CONFIRMED: RTS(classic) equals Naive because:")
        print("    - tavg is HIDDEN (not in H matrix)")
        print("    - F = I has no coupling from observed states")
        print("    - KF/RTS cannot update hidden tavg beyond initial x0")
        print("    - Therefore RTS(classic) tavg = x0 persistence = Naive")
    else:
        print(f"\n  ⚠ RTS(classic) differs from Naive (max diff: {np.max(all_diffs):.6e})")
print("  ─" * 40)


# ======================================================
# STEP  – TEST M-NETWORK
# ======================================================
print("\n" + "=" * 60)
print("STEP 6: TEST M-Network")
print("=" * 60)

mse_per_iter, mse_db_per_iter, final_F_list, pred_dicts, mse_per_iter_denorm, mse_db_per_iter_denorm = RTSNet_Pipeline.test_mstep_weather(
    SysModel=sys_model,
    test_input=test_input,
    test_target=test_target,
    test_x0=test_x0,
    destination_path_RTS=path_results_rts,
    destination_path_M=path_results_m,
    num_em_iters=2,
    print_F_every=50,
)

# Compute denormalized M-Net MSE
mnet_sq_err_denorm = []

for idx, d in enumerate(pred_dicts):
    x_pred_denorm = d["x_pred_denorm"]  # [m, T]
    x_true_denorm = d["x_true_denorm"]  # [T] (tavg only)
    pred_tavg_denorm = x_pred_denorm[0, :]
    mse_i_denorm = torch.mean((pred_tavg_denorm - x_true_denorm) ** 2)
    mnet_sq_err_denorm.append(mse_i_denorm.item())

if mnet_sq_err_denorm:
    mnet_mse_denorm = torch.tensor(mnet_sq_err_denorm).mean()
    mnet_rmse_denorm = torch.sqrt(mnet_mse_denorm)
    print(f"\n  M-Net   MSE(tavg):  {mnet_mse_denorm.item():.4f} °C²")
    print(f"  M-Net   RMSE(tavg): {mnet_rmse_denorm.item():.4f} °C")
else:
    print("\n  M-Net: No predictions generated")
    mnet_mse_denorm = torch.tensor(float('nan'))
    mnet_rmse_denorm = torch.tensor(float('nan'))

# ======================================================
# STEP 6B – TEST EMKF (Existing EMKF_FH_analytic)
# ======================================================
print("\n" + "=" * 60)
print("STEP 6B: TEST EMKF (Classical EM Kalman Filter)")
print("=" * 60)

emkf_sq_err_denorm = []
all_emkf_smooth_x = []
all_emkf_mse_per_iter = []

for idx in range(len(test_input)):
    y_win = test_input[idx].to(device)
    x_true = test_target[idx].to(device)

    # Normalize
    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
    y_win_n = (y_win - x_mean[1:]) / x_std[1:]
    x_true_n = (x_true - x_mean) / x_std

    T = y_win.size(-1)
    x0_raw = test_x0[idx].to(device)
    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

    # Prepare inputs for EMKF_FH_analytic
    Y = y_win_n.unsqueeze(0)
    X_true_input = x_true_n.unsqueeze(0)

    # Run EMKF with F and H learning
    F_matrices, H_matrices, last_x_list, last_P_list, smooth_x_list, mean_mse_per_iter = EMKF_FH_analytic(
        sys_model,
        [F0],
        [H_fixed],
        Q, R,
        Y,
        x0_norm.squeeze(),
        P0_default,
        X_true_input,
        max_it=3,
        generate_f=True,
        generate_h=False,
        update_F=True,
        update_H=False,
        init_x_list=[x0_norm.squeeze()],
        init_P_list=[P0_default],
    )

    # Extract full smoothed trajectory [m, T]
    x_sm_final = smooth_x_list[0]  # [m, T] in NORMALIZED units

    # Denormalize to real units
    # x_sm_final is in the normalized space: (x_real - x_mean) / x_std
    # So denormalize: x_real = x_sm_final * x_std + x_mean
    x_sm_denorm = x_sm_final.clone()
    for i in range(m):
        x_sm_denorm[i, :] = x_sm_final[i, :] * x_std[i] + x_mean[i]
    all_emkf_smooth_x.append(x_sm_denorm.detach().cpu())

    # Compute denormalized tavg MSE
    pred_tavg_real = x_sm_denorm[0, :]
    true_tavg_real = x_true[0, :]  # Raw, denormalized
    mse_i_denorm = torch.mean((pred_tavg_real - true_tavg_real) ** 2)
    emkf_sq_err_denorm.append(mse_i_denorm.item())

    # Accumulate MSE per iteration
    all_emkf_mse_per_iter.append(mean_mse_per_iter.detach().cpu())

if emkf_sq_err_denorm and not all(torch.isnan(torch.tensor(emkf_sq_err_denorm))):
    emkf_mse_denorm = torch.tensor([e for e in emkf_sq_err_denorm if not np.isnan(e)]).mean()
    emkf_rmse_denorm = torch.sqrt(emkf_mse_denorm)

    print(f"\n  EMKF    MSE(tavg):  {emkf_mse_denorm.item():.4f} °C²")
    print(f"  EMKF    RMSE(tavg): {emkf_rmse_denorm.item():.4f} °C")

    # Print mean MSE per EM iteration across all sequences
    if all_emkf_mse_per_iter:
        mean_mse_stack = torch.stack([m.float() for m in all_emkf_mse_per_iter])
        mean_mse_per_iteration = mean_mse_stack.mean(dim=0)
        print(f"\n  EMKF Mean MSE per EM iteration (averaged across all sequences):")
        for it, mse_val in enumerate(mean_mse_per_iteration):
            print(f"    Iteration {it+1}: MSE = {mse_val.item():.6f}")

    # DIAGNOSTIC: Check if EMKF with F fixed matches Naive
    print("\n  DIAGNOSTIC: Checking if EMKF(F fixed) ≈ Naive")
    print("  ─" * 40)
    num_diagnostic_windows = min(3, len(test_input))
    for diag_idx in range(num_diagnostic_windows):
        x_emkf_denorm = all_emkf_smooth_x[diag_idx]
        x_true = test_target[diag_idx].to(device)
        x0_raw = test_x0[diag_idx].to(device)

        emkf_tavg_traj = x_emkf_denorm[0, :].numpy()
        naive_tavg_const = float(x0_raw[0].item())

        # Check if EMKF differs from naive
        diff_from_naive = np.abs(emkf_tavg_traj - naive_tavg_const)
        max_diff_from_naive = np.max(diff_from_naive)
        mean_diff = np.mean(diff_from_naive)

        print(f"  Window {diag_idx}: max|EMKF - x0| = {max_diff_from_naive:.6e}, mean|diff| = {mean_diff:.6e}")

    print("  ─" * 40)

    use_emkf = True
else:
    print("\n  EMKF: No valid results")
    emkf_mse_denorm = torch.tensor(float('nan'))
    emkf_rmse_denorm = torch.tensor(float('nan'))
    use_emkf = False

# ======================================================
# PLOTTING UTILITY: Glued Predictions
# ======================================================

def plot_glued_temperature_predictions(
    start_idx,
    num_windows,
    rts_preds,
    rts_classic_preds,
    mnet_preds,
    emkf_preds,
    test_target,
    test_x0,
    save_path="glued_predictions.png",
    tau=TAU,
    device=device,
    dtype=dtype,
    show_plot=True,
):
    """
    Glues consecutive non-overlapping windows into one continuous plot.

    Args:
        start_idx: Starting window index (0-based)
        num_windows: Number of windows to glue together
        rts_preds: RTSNet predictions list [m, T] per window (denormalized)
        rts_classic_preds: Classical RTS predictions list [m, T] per window (denormalized)
        mnet_preds: M-Network predictions list of dicts (from test_mstep_weather)
        emkf_preds: EMKF smoothed trajectories list [m, T] per window (denormalized)
        test_target: List of test state windows [m, TAU]
        test_x0: List of initial states [m]
        save_path: Path to save the figure
        tau: Window length (default TAU from globals)
        device: Device (default device from globals)
        dtype: Torch dtype (default dtype from globals)
        show_plot: Whether to display plot with plt.show() (default True)

    Returns:
        dict with glued trajectories in real °C
    """

    # Initialize lists for gluing
    glued_true = []
    glued_rtsnet = []
    glued_rts_classic = []
    glued_mnet = []
    glued_emkf = []
    glued_naive = []

    # Glue each window
    for w_idx in range(start_idx, start_idx + num_windows):
        if w_idx >= len(test_target):
            break

        x_true = test_target[w_idx].to(device)
        x0_raw = test_x0[w_idx].to(device)

        # Get normalization stats for this window
        x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)

        # TRUE trajectory (raw, already denormalized)
        true_tavg_real = x_true[0, :]
        glued_true.append(true_tavg_real.detach().cpu().numpy())

        # RTSNet prediction (denormalized)
        if w_idx < len(rts_preds):
            rts_pred_denorm = rts_preds[w_idx]
            rts_tavg_real = rts_pred_denorm[0, :]
            glued_rtsnet.append(rts_tavg_real.numpy())

        # Classical RTS prediction (denormalized)
        if w_idx < len(rts_classic_preds):
            rts_classic_denorm = rts_classic_preds[w_idx]
            rts_classic_tavg_real = rts_classic_denorm[0, :]
            glued_rts_classic.append(rts_classic_tavg_real.numpy())

        # M-Network prediction
        if w_idx < len(mnet_preds):
            mnet_dict = mnet_preds[w_idx]
            if "x_pred_denorm" in mnet_dict:
                mnet_pred_denorm = mnet_dict["x_pred_denorm"]
                mnet_tavg_real = mnet_pred_denorm[0, :]
                glued_mnet.append(mnet_tavg_real.numpy())

        # EMKF prediction (denormalized)
        if w_idx < len(emkf_preds):
            emkf_smooth = emkf_preds[w_idx]
            emkf_tavg_real = emkf_smooth[0, :]
            glued_emkf.append(emkf_tavg_real.numpy())

        # Naive prediction (constant x0[0] over the window)
        x0_tavg_raw = x0_raw[0].item()
        naive_tavg_real = np.full(tau, x0_tavg_raw)
        glued_naive.append(naive_tavg_real)

    # Concatenate all windows
    true_curve = np.concatenate(glued_true) if glued_true else None
    rtsnet_curve = np.concatenate(glued_rtsnet) if glued_rtsnet else None
    rts_classic_curve = np.concatenate(glued_rts_classic) if glued_rts_classic else None
    mnet_curve = np.concatenate(glued_mnet) if glued_mnet else None
    emkf_curve = np.concatenate(glued_emkf) if glued_emkf else None
    naive_curve = np.concatenate(glued_naive) if glued_naive else None

    if true_curve is None:
        print("  ERROR: No true data to plot!")
        return None

    total_days = len(true_curve)
    x_axis = np.arange(total_days)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x_axis, true_curve, 'k-', linewidth=2, label='True', marker='o', markersize=3, alpha=0.7)
    ax.plot(x_axis, naive_curve, 'gray', linewidth=1.5, label='Naive (x0)', linestyle='--', alpha=0.7)

    if rts_classic_curve is not None:
        ax.plot(x_axis, rts_classic_curve, 'purple', linewidth=1.5, label='RTS (classic)', linestyle=':', alpha=0.7)

    if rtsnet_curve is not None:
        ax.plot(x_axis, rtsnet_curve, 'b-', linewidth=1.5, label='RTSNet', alpha=0.7)

    if mnet_curve is not None:
        ax.plot(x_axis, mnet_curve, 'r-', linewidth=1.5, label='M-Network', alpha=0.7)

    if emkf_curve is not None:
        ax.plot(x_axis, emkf_curve, 'g-', linewidth=1.5, label='EMKF', alpha=0.7)

    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Average Temperature (°C)', fontsize=11)
    ax.set_title(f'Glued Predictions: Windows {start_idx}–{start_idx + num_windows - 1} ({total_days} days)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to file
    abs_path = os.path.abspath(save_path)
    plt.savefig(abs_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {abs_path}")

    # Display plot if requested
    if show_plot:
        try:
            plt.show(block=False)
        except Exception as e:
            print(f"  Note: plt.show() not available ({e})")

    plt.close()

    return {
        'true': true_curve,
        'rtsnet': rtsnet_curve,
        'rts_classic': rts_classic_curve,
        'mnet': mnet_curve,
        'emkf': emkf_curve,
        'naive': naive_curve,
    }


# ======================================================
# STEP 7 – PLOT GLUED PREDICTIONS
# ======================================================
print("\n" + "=" * 60)
print("STEP 7: Generate Glued 120-Day Prediction Plot (12 windows)")
print("=" * 60)

# Calculate how many windows we can actually plot
max_windows = min(len(test_target), len(rts_preds), len(all_rts_classic_smooth_x),
                  len(pred_dicts), len(all_emkf_smooth_x))
num_plot_windows = min(12, max_windows)  # Changed from 6 to 12 (120 days)

glued_results = plot_glued_temperature_predictions(
    start_idx=0,
    num_windows=num_plot_windows,
    rts_preds=rts_preds,
    rts_classic_preds=all_rts_classic_smooth_x,
    mnet_preds=pred_dicts,
    emkf_preds=all_emkf_smooth_x,
    test_target=test_target,
    test_x0=test_x0,
    save_path="glued_predictions_120days.png",  # Changed filename
)

if glued_results is not None:
    print(f"✓ Generated {glued_results['true'].shape[0]}-day continuous curve plot")
    # Auto-open the plot file
    import subprocess
    import sys
    plot_path = os.path.abspath("glued_predictions_120days.png")
    try:
        if sys.platform == 'win32':
            os.startfile(plot_path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', plot_path])
        else:
            subprocess.Popen(['xdg-open', plot_path])
        print(f"  ✓ Opening plot: {plot_path}")
    except Exception as e:
        print(f"  ⚠ Auto-open failed ({e})")
        print(f"  Plot was still saved here:")
        print(f"  {plot_path}")

# ======================================================
# STEP 8 – FINAL COMPARISON TABLE (Denormalized Only)
# ======================================================

print("\n" + "=" * 60)
print("COMPARISON: All Methods (Denormalized MSE in °C²)")
print("=" * 60)
print(f"{'Method':<25} {'MSE (°C²)':<18} {'RMSE (°C)':<18}")
print("-" * 61)
print(f"{'RTSNet':<25} {mse_rts_denorm.item():<18.4f} {rmse_rts_denorm.item():<18.4f}")
print(f"{'RTS (classic)':<25} {rts_classic_mse_denorm.item():<18.4f} {rts_classic_rmse_denorm.item():<18.4f}")
if not torch.isnan(mnet_mse_denorm):
    print(f"{'M-Network':<25} {mnet_mse_denorm.item():<18.4f} {mnet_rmse_denorm.item():<18.4f}")
if use_emkf:
    print(f"{'EMKF':<25} {emkf_mse_denorm.item():<18.4f} {emkf_rmse_denorm.item():<18.4f}")
print(f"{'Naive (x0)':<25} {naive_mse_denorm.item():<18.4f} {naive_rmse_denorm.item():<18.4f}")
print("-" * 61)

