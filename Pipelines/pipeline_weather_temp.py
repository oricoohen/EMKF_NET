"""
pipeline_weather.py
===================
Pipeline for rolling-window weather prediction using RTSNet + EM Kalman Filter.

Observation y_t = [tavg, trange, wind, pressure]  (n=4, m=4, F=I, H=I)
Target: predict next-day tavg (row 0 of the observation vector).

Per-window normalization: each of the 4 feature rows is normalized
by its own mean/std over the window (same logic as _win_norm in Pipeline_ERTS.py).
"""

import os
import random

import torch
import torch.nn as nn

from emkf.AI_M_step import DeltaF_MStepNet
from Smoothers.RTS_Smoother_test import S_Test


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def _win_norm_4d(y_win, device, dtype):
    """
    Per-feature-row normalization for a [4, TAU] window.
    Returns:
        y_mean      : [4, 1]  – broadcast-compatible
        y_std       : [4, 1]  – broadcast-compatible (floored at 1e-6)
        y_mean_row0 : scalar  – mean of feature 0 (tavg)
        y_std_row0  : scalar  – std  of feature 0 (tavg)
    """
    y_mean = y_win.mean(dim=1, keepdim=True)           # [4, 1]
    y_std  = y_win.std(dim=1, keepdim=True)            # [4, 1]
    y_std  = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
    return y_mean, y_std, y_mean[0, 0], y_std[0, 0]


def _win_norm_3d(y_win, device, dtype):
    """
    Per-feature-row normalization for a [3, TAU] observation window (n=3, no tavg).
    Returns:
        y_mean      : [3, 1]  – broadcast-compatible
        y_std       : [3, 1]  – broadcast-compatible (floored at 1e-6)
        y_mean_rows : [3]     – means of each feature
        y_std_rows  : [3]     – stds of each feature
    """
    y_mean = y_win.mean(dim=1, keepdim=True)           # [3, 1]
    y_std  = y_win.std(dim=1, keepdim=True)            # [3, 1]
    y_std  = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
    return y_mean, y_std, y_mean.squeeze(1), y_std.squeeze(1)


# -----------------------------------------------------------------------
# PipelineWeather  (mirrors Pipeline_ERTS but weather-specific)
# -----------------------------------------------------------------------

class PipelineWeather:
    """
    Minimal pipeline that wraps RTSNet + M-step training/testing for weather.
    Mirrors the structure of Pipeline_ERTS so the experiment script can call
    the same high-level methods.
    """

    def __init__(self, strTime, modelName, dataName):
        self.strTime   = strTime
        self.modelName = modelName
        self.dataName  = dataName
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn   = nn.MSELoss()

    # ------------------------------------------------------------------
    def setssModel(self, SysModel):
        self.SysModel = SysModel

    def setModel(self, model, args):
        self.model = model.to(self.device)

    def setTrainingParams(self, args):
        self.N_steps       = args.n_steps
        self.N_B           = args.n_batch
        self.learningRate  = args.lr
        self.weightDecay   = args.wd
        self.optimizer     = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weightDecay,
        )
        self.M_model     = None   # set by setMModel
        self.M_optimizer = None

    def setMModel(self, m_model, lr=1e-4, wd=1e-5):
        self.M_model     = m_model.to(self.device)
        self.M_optimizer = torch.optim.Adam(
            self.M_model.parameters(), lr=lr, weight_decay=wd
        )

    # ------------------------------------------------------------------
    # RTSNet training  (NNTrain_weather)
    # ------------------------------------------------------------------
    def NNTrain_weather(self, SysModel, cv_input, cv_target,
                        train_input, train_target, path_results,
                        load_model_path=None,
                         train_x0=None, cv_x0=None,):
        """
        Train RTSNet smoother on weather windows (state-based loss).

        train_input[i]     : [3, TAU]   observation window (n=3, tavg hidden)
        train_target[i]    : [3, TAU]   next-day-aligned observations
        train_x_state[i]   : [4, TAU]   true full state window (for loss)
        train_x0[i]        : [4]        state vector one day before window

        Loss: Compares **predicted state** x_smooth to **true state** x_true
              weighted MSE: sum_t w_t * MSE(x_smooth_t, x_true_t)
              + 2 * MSE on final step (double weight for last prediction)
        """
        device = self.device
        dtype  = train_input[0].dtype
        m, n   = SysModel.m, SysModel.n

        self.N_E  = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_lin   = torch.empty([self.N_steps], device=device)
        MSE_cv_dB    = torch.empty([self.N_steps], device=device)
        MSE_tr_lin   = torch.empty([self.N_steps], device=device)
        MSE_tr_dB    = torch.empty([self.N_steps], device=device)
        MSE_tr_batch = torch.empty([self.N_B],     device=device)

        if load_model_path is not None:
            self.model = torch.load(load_model_path, map_location=device,
                                    weights_only=False).to(device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learningRate, weight_decay=self.weightDecay)

        self.model.train()
        best_cv_dB  = 1e9
        nan_streak  = 0

        for ti in range(self.N_steps):

            # ---- TRAIN ----
            self.model.train()
            self.optimizer.zero_grad()
            batch_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)

            for j in range(self.N_B):

                self.model.init_hidden()
                idx = random.randint(0, self.N_E - 1)
                if  idx == 0:#del
                    idx =1
                y_win  = train_input[idx]    # [3, TAU] observations y
                x_true = train_target[idx]   # [4, TAU] true state x: x[0]=tavg(hidden), x[1:]=y(obs)
                T      = y_win.size(-1)

                # Compute normalization stats from TRUE STATE x [4, TAU]
                x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                # x_mean[0] =  train_x0[idx][0]
                # x_std[0]  = 4
                prev_x_mean, prev_x_std, _, _ = _win_norm_4d(train_target[idx-1].detach(), device, dtype)  # del
                x_mean[0] = prev_x_mean[0]  # Use the mean of the previous target for normalization del
                x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                # Normalize observations using x[1:] stats (since y = x[1:])
                y_win_n  = (y_win - x_mean[1:]) / x_std[1:]

                # F selection
                F = SysModel.F_train[0].to(device) \
                    if isinstance(SysModel.F_train, list) \
                    else SysModel.F_train.to(device)
                SysModel.F = F
                self.model.update_F(F)
                SysModel.T = T

                # x0: normalize using stats from TRUE STATE x
                x0_raw = train_x0[idx].to(device)
                x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std
                SysModel.m1x_0 = x0_norm

                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.init_hidden()

                # Forward pass (preserve graph)
                x_fwd = torch.stack(
                    [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                    dim=1)   # [m, T]

                # Backward smoothing
                xs = [None] * T
                xs[T - 1] = x_fwd[:, T - 1]
                self.model.InitBackward(xs[T - 1])
                xs[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                x_sm = torch.stack(xs, dim=1)   # [m, T] smoothed state

                # Normalize true state for loss comparison
                x_true_n = (x_true - x_mean) / x_std

                loss = torch.tensor(0.0, device=device, dtype=dtype)
                for t in range(T):
                    # Compare only tavg (component 0) of predicted vs true state
                    loss = loss + self.loss_fn(x_sm[0, t], x_true_n[0, t])

                batch_loss_sum = batch_loss_sum + loss
                MSE_tr_batch[j] = loss.detach()


            (batch_loss_sum / self.N_B).backward()

            bad = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in self.model.parameters()
            )
            if bad:
                print(f"  [epoch {ti}] NaN/Inf grad – skipped")
                nan_streak += 1
                if nan_streak >= 3:
                    print("  Early stop (3 bad batches).")
                    torch.save(self.model, path_results)
                    return
                self.model.zero_grad(set_to_none=True)
                continue
            nan_streak = 0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            MSE_tr_lin[ti] = MSE_tr_batch.mean()
            MSE_tr_dB[ti]  = 10 * torch.log10(MSE_tr_lin[ti])

            # ---- CV ----
            self.model.eval()
            with torch.no_grad():
                cv_batch = torch.empty([self.N_CV-1], device=device)#del -1
                for j in range(self.N_CV):
                    if j == 0:
                        continue  # Skip first CV sample to avoid data leakage in normalization (since it uses true state stats) del
                    y_win  = cv_input[j]
                    x_true = cv_target[j]  # [4, TAU] true state x
                    T      = y_win.size(-1)

                    # Compute normalization stats from TRUE STATE x
                    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                    prev_x_mean, prev_x_std, _, _ = _win_norm_4d(cv_target[j-1], device, dtype)  # del
                    x_mean[0] = prev_x_mean[0]  # Use the mean of the previous target for normalization del
                    x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                    # x_mean[0] = cv_x0[j][0]
                    # x_std[0] = 4
                    y_win_n  = (y_win - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats

                    F = SysModel.F_valid[0].to(device) \
                        if isinstance(SysModel.F_valid, list) \
                        else SysModel.F_valid.to(device)
                    SysModel.F = F
                    self.model.update_F(F)

                    x0_raw = cv_x0[j].to(device)
                    # Normalize x0 using stats from TRUE STATE x
                    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std
                    SysModel.m1x_0 = x0_norm
                    self.model.InitSequence(x0_norm, T)
                    self.model.init_hidden()

                    x_fwd = torch.stack(
                        [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                        dim=1)
                    xs = [None] * T
                    xs[T - 1] = x_fwd[:, T - 1]
                    self.model.InitBackward(xs[T - 1])
                    xs[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                    x_sm = torch.stack(xs, dim=1)

                    cv_loss = torch.tensor(0.0, device=device, dtype=dtype)
                    x_true_n = (x_true - x_mean) / x_std
                    for t in range(T):
                        # Compare only tavg (component 0) of predicted vs true state
                        cv_loss  = cv_loss +  self.loss_fn(x_sm[0, t], x_true_n[0, t])

                    cv_batch[j-1] = cv_loss#del

                MSE_cv_lin[ti] = cv_batch.mean()
                MSE_cv_dB[ti]  = 10 * torch.log10(MSE_cv_lin[ti])

                if MSE_cv_dB[ti] < best_cv_dB:
                    best_cv_dB = MSE_cv_dB[ti]
                    os.makedirs(os.path.dirname(path_results) or ".", exist_ok=True)
                    torch.save(self.model, path_results)

            print(f"  [epoch {ti:03d}] train={MSE_tr_dB[ti].item():.3f} dB  "
                  f"cv={MSE_cv_dB[ti].item():.3f} dB  best={best_cv_dB:.3f} dB")

        print(f"Saved RTSNet model to: {path_results}")

    # ------------------------------------------------------------------
    # RTSNet test  (NNTest_weather)
    # ------------------------------------------------------------------
    def NNTest_weather(self, SysModel, test_input, test_target,
                       load_model_path, test_x0=None):
        """
        Test RTSNet on weather test windows.

        For each test window:
        - run RTSNet forward filtering and backward smoothing,
        - compare the smoothed hidden temperature component x[0, t]
          against the true hidden temperature over the full window,
        - compute one per-window MSE and one per-window relative error.
        - compute BOTH normalized and denormalized (real °C) MSE.

        The final reported metrics are the averages over all test windows.

        Returns:
            mse: normalized MSE (average over windows)
            rel_err_mean: normalized relative error (average over windows)
            sq_err: per-window normalized squared errors [N_T]
            rel_err: per-window normalized relative errors [N_T]
            mse_denorm: denormalized MSE in (°C)² (average over windows)
            sq_err_denorm: per-window denormalized squared errors [N_T]
            rts_preds: list of denormalized x_pred predictions [m, T] per window (NEW)
        """
        device = self.device
        dtype  = torch.float32
        m      = SysModel.m
        N_T    = len(test_input)

        self.model = torch.load(load_model_path, weights_only=False,
                                map_location=device).eval()

        sq_err     = torch.empty(N_T-1, device=device, dtype=dtype)#del -1
        rel_err    = torch.empty(N_T-1, device=device, dtype=dtype)#del -1
        sq_err_denorm = torch.empty(N_T-1, device=device, dtype=dtype)  # denormalized MSE#del -1
        rts_preds = []  # NEW: store denormalized x_pred per window

        with torch.no_grad():
            for j in range(N_T):
                if j == 0:
                    continue  # Skip first test sample to avoid data leakage in normalization (since it uses true state stats) del
                y_win  = test_input[j].to(device)    # [3, TAU] observations y
                x_true = test_target[j].to(device)   # [4, TAU] true state x
                T      = y_win.size(-1)

                # Compute normalization stats from TRUE STATE x [4, TAU]
                x_mean, x_std, x_mean0, x_std0 = _win_norm_4d(x_true, device, dtype)
                prev_x_mean, prev_x_std, _, _ = _win_norm_4d(test_target[j-1], device, dtype)  # del
                x_mean[0] = prev_x_mean[0]  # Use the mean of the previous target for normalization del
                x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                # x_mean[0] =  test_x0[j][0]
                # x_std[0]  = 4
                y_win_n = (y_win - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats

                F = SysModel.F_test[0].to(device) \
                    if isinstance(SysModel.F_test, list) \
                    else SysModel.F_test.to(device)
                SysModel.F = F
                self.model.update_F(F)

                x0_raw = test_x0[j].to(device)
                # Normalize x0 using stats from TRUE STATE x
                x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std
                self.model.InitSequence(x0_norm, T)
                self.model.init_hidden()

                x_fwd = torch.stack(
                    [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                    dim=1)
                xs = [None] * T
                xs[T - 1] = x_fwd[:, T - 1]
                self.model.InitBackward(xs[T - 1])
                xs[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                x_sm = torch.stack(xs, dim=1)   # [m, T]

                # Normalized values
                true_tavg_norm = ((x_true - x_mean) / x_std)[0, :]  # [T]
                pred_tavg_norm = x_sm[0, :]  # [T]

                # per-window loss over the whole window (normalized)
                sq_err[j-1] = torch.mean((pred_tavg_norm - true_tavg_norm) ** 2)#del -1

                # relative error over the whole window
                rel_err[j-1] = torch.mean(
                    torch.abs(pred_tavg_norm - true_tavg_norm) / (torch.abs(true_tavg_norm) + 1e-8)#del -1
                )

                # NEW: Denormalized (real °C) values
                true_tavg_denorm = x_true[0, :]  # Real °C (already raw)
                pred_tavg_denorm = pred_tavg_norm * x_std[0] + x_mean[0]  # Denormalize

                # per-window denormalized loss (real °C)
                sq_err_denorm[j-1] = torch.mean((pred_tavg_denorm - true_tavg_denorm) ** 2)#del -1

                # NEW: Denormalize full x_sm trajectory and store
                x_sm_denorm = x_sm.clone()
                for i in range(m):
                    x_sm_denorm[i, :] = x_sm[i, :] * x_std[i] + x_mean[i]
                rts_preds.append(x_sm_denorm.detach().cpu())

        mse = sq_err.mean()
        mse_db = 10 * torch.log10(mse)
        rel_err_mean = rel_err.mean()
        mse_denorm = sq_err_denorm.mean()

        print(f"  RTSNet MSE(tavg): {mse.item():.4f} (normalized)")
        print(f"  RTSNet MSE(tavg) [dB]: {mse_db.item():.4f}")
        print(f"  RTSNet MSE(tavg): {mse_denorm.item():.4f} °C² (denormalized)")
        print(f"  RTSNet RelErr: {rel_err_mean.item():.4f}")

        return mse, rel_err_mean, sq_err, rel_err, mse_denorm, sq_err_denorm, rts_preds

    # ------------------------------------------------------------------
    # M-step training  (train_emkalmannet_weather)
    # ------------------------------------------------------------------
    def train_emkalmannet_weather(
        self, SysModel,
        cv_input, cv_target, cv_x0,
        train_input, train_target, train_x0,
        destination_path_M, destination_path_RTS,
        num_em_iters=2, alpha=(0.05, 0.15, 0.85),
        lambda_F=1e-2,
        lambda_f_loss=10.0,
        f_loss=False,
        generate_f=False, generate_h=False,
        clip_grad=1.0,
        train_x_state=None, cv_x_state=None,
    ):
        device    = self.device
        dtype     = train_input[0].dtype
        m, n      = SysModel.m, SysModel.n
        self.N_E  = len(train_input)
        self.N_CV = len(cv_input)

        # Load & freeze RTSNet
        self.model = torch.load(destination_path_RTS, map_location=device,
                                weights_only=False).to(device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.M_model is None:
            self.M_model = DeltaF_MStepNet(m=m, n=n, d_hidden=256).to(device)
            self.M_optimizer = torch.optim.Adam(
                self.M_model.parameters(), lr=1e-4, weight_decay=1e-5)

        model_mstep  = self.M_model.train()
        best_cv_loss = 1e18
        batch_size   = 10

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.M_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        for epoch in range(self.N_steps):

            # ---- TRAIN ----
            model_mstep.train()
            train_loss_sum = 0.0
            grad_norm_sum = 0.0
            max_grad_norm_epoch = 0.0
            clip_hit_count = 0
            grad_bad_count = 0
            dF_norm_sum = 0.0
            dF_norm_count = 0

            for j in range(self.N_B):
                self.M_optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device, dtype=dtype)

                for _ in range(batch_size):
                    idx   = random.randint(0, self.N_E - 1)
                    if idx == 0:
                        continue  # Skip first batch to avoid data leakage in normalization (since it uses true state stats) del
                    y_win = train_input[idx].to(device)   # [3, TAU] observations y
                    x_true = train_target[idx].to(device)  # [4, TAU] true state x
                    T     = int(y_win.size(-1))

                    # Compute normalization stats from TRUE STATE x
                    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                    prev_x_mean, prev_x_std, _, _ = _win_norm_4d(train_target[idx-1], device, dtype)  # del
                    x_mean[0] = prev_x_mean[0]  # Use the mean of the previous target for normalization del
                    x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                    # x_mean[0] = train_x0[idx][0]
                    # x_std[0] = 4
                    y_win_n = (y_win - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats
                    x_true_n = (x_true - x_mean) / x_std

                    F_base = SysModel.F_train[0].to(device) \
                        if isinstance(SysModel.F_train, list) \
                        else SysModel.F_train.to(device)
                    H = SysModel.H.to(device)

                    x0_raw = train_x0[idx].to(device)
                    # Normalize x0 using stats from TRUE STATE x
                    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    F_current  = F_base.clone()
                    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

                    for em_iter in range(num_em_iters):
                        # Pass F to RTSNet – no detach so RTSNet sees F's grad
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.prior_Sigma = prior_P
                        self.model.init_hidden()


                        x_fwd = torch.stack(
                            [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                            dim=1)  # [m, T]
                        xs = [None] * T
                        xs[T - 1] = x_fwd[:, T - 1]
                        self.model.InitBackward(xs[T - 1])
                        xs[T -  2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                        # NO detach – grad flows from loss back through x_sm into RTSNet
                        x_sm = torch.stack(xs, dim=1)  # [m, T]

                        # Sufficient stats over T pairs (include x0 as first x_prev)
                        x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)  # [m, T]
                        x_curr_full = x_sm                                         # [m, T]

                        A1      = (x_curr_full @ x_prev_full.T) / T
                        A2      = (x_prev_full @ x_prev_full.T) / T
                        delta_x = x_curr_full - F_current @ x_prev_full
                        S_delta = (delta_x @ delta_x.T) / T
                        C_delta = (delta_x @ x_prev_full.T) / T
                        nu      = y_win_n - H @ F_current @ x_prev_full
                        S_nu    = (nu @ nu.T) / T

                        # z_in: detach only the F feature (not stats) so net input is stable
                        z_in = torch.cat([
                            A1.detach().reshape(-1),
                            A2.detach().reshape(-1),
                            S_delta.detach().reshape(-1),
                            S_nu.detach().reshape(-1),
                            C_delta.detach().reshape(-1),
                            F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)

                        dF        = model_mstep(z_in).view(m, m)
                        dF_norm_sum += dF.detach().norm().item()
                        dF_norm_count += 1
                        F_current = F_current + dF  # no detach – full grad chain

                        reg_F       = lambda_F * (dF ** 2).sum()


                        # Loss on state x, feature 0
                        resid_x_iter = x_sm[0, :] - x_true_n[0, :]
                        loss_x_iter = (resid_x_iter ** 2).mean()
                        iter_loss = alpha[em_iter] * (loss_x_iter + reg_F)
                        total_loss = total_loss + iter_loss
                        # if j == 2:
                        #     print(f"[LOSS ITER] loss_x_iter={loss_x_iter.item():.4f}")
                    # Final RTS pass with last F – no detach so F grad flows into final loss
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0_norm.clone(), T)
                    self.model.prior_Sigma = prior_P
                    self.model.init_hidden()

                    x_fwd2 = torch.stack(
                        [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                        dim=1)
                    xs2 = [None] * T
                    xs2[T - 1] = x_fwd2[:, T - 1]
                    self.model.InitBackward(xs2[T - 1])
                    xs2[T -  2] = self.model(None, x_fwd2[:, T - 2], x_fwd2[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        xs2[t] = self.model(None, x_fwd2[:, t], x_fwd2[:, t + 1], xs2[t + 2])
                    x_sm2 = torch.stack(xs2, dim=1)  # NO detach

                    # Final loss on state x, feature 0
                    resid_x_final = x_sm2[0, :] - x_true_n[0, :]
                    loss_x_final = (resid_x_final ** 2).mean()
                    total_loss = total_loss + alpha[-1] * loss_x_final
                    # print(
                    #     f"iter_loss={loss_x_iter.item():.4f} "
                    #     f"final_loss={loss_x_final.item():.4f}"
                    # )
                    # Optional: F loss computed from the data constraint
                    # F_true is the matrix that satisfies: F_true @ x_true[:, -2] = x_true[:, -2]
                    if f_loss:
                        x_prev = x_true_n[:, -2].view(m, 1)
                        denom = (x_prev.T @ x_prev) + 1e-9
                        F_true = (x_prev @ x_prev.T) / denom  # [m, m]
                        loss_f = torch.norm(F_true - F_current, p='fro')
                        total_loss = total_loss + lambda_f_loss * loss_f

                    batch_loss = batch_loss + total_loss / batch_size

                batch_loss.backward()

                grads = [p.grad for p in model_mstep.parameters() if p.grad is not None]
                if grads:
                    grad_sq = sum((g.detach().norm() ** 2 for g in grads))
                    grad_norm = torch.sqrt(grad_sq).item()
                else:
                    grad_norm = 0.0
                    print("problem")

                has_bad_grad = any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads)
                if has_bad_grad:
                    grad_bad_count += 1

                pre_clip_norm = grad_norm
                if clip_grad > 0:
                    clipped_total_norm = torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), clip_grad)
                    if float(clipped_total_norm) > clip_grad:
                        clip_hit_count += 1
                self.M_optimizer.step()
                train_loss_sum += batch_loss.detach().item()
                grad_norm_sum += pre_clip_norm
                if pre_clip_norm > max_grad_norm_epoch:
                    max_grad_norm_epoch = pre_clip_norm

            train_avg = train_loss_sum / max(self.N_B, 1)

            # ---- CV ----
            model_mstep.eval()
            cv_loss_sum = 0.0
            with torch.no_grad():
                # hard_sequences = []
                # easy_sequences = []

                # HARD_TH = 10.0
                # EASY_TH = 1.0
                for j in range(self.N_CV):
                    if j == 0:
                        continue  # Skip first CV sample to avoid data leakage in normalization (since it uses true state stats) del
                    y_win = cv_input[j].to(device)   # [3, TAU] observations y
                    x_true = cv_target[j].to(device)  # [4, TAU] true state x
                    T     = int(y_win.size(-1))

                    # Compute normalization stats from TRUE STATE x
                    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                    prev_x_mean, prev_x_std, _, _ = _win_norm_4d(cv_target[j-1], device, dtype)  # del
                    x_mean[0] = prev_x_mean[0]  # Use the mean of
                    x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                    # x_mean[0] = cv_x0[j][0]
                    # x_std[0] = 4
                    y_win_n = (y_win - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats
                    x_true_n = (x_true - x_mean) / x_std

                    F_base = SysModel.F_valid[0].to(device) \
                        if isinstance(SysModel.F_valid, list) \
                        else SysModel.F_valid.to(device)
                    H = SysModel.H.to(device)

                    x0_raw = cv_x0[j].to(device)
                    # Normalize x0 using stats from TRUE STATE x
                    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)


                    F_current = F_base.clone()
                    for em_iter in range(num_em_iters):
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.prior_Sigma = prior_P
                        self.model.init_hidden()

                        x_fwd = torch.stack(
                            [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                            dim=1)
                        xs = [None] * T
                        xs[T - 1] = x_fwd[:, T - 1]
                        self.model.InitBackward(xs[T - 1])
                        xs[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                        x_sm = torch.stack(xs, dim=1)

                        x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)
                        x_curr_full = x_sm
                        A1      = (x_curr_full @ x_prev_full.T) / T
                        A2      = (x_prev_full @ x_prev_full.T) / T
                        delta_x = x_curr_full - F_current @ x_prev_full
                        S_delta = (delta_x @ delta_x.T) / T
                        C_delta = (delta_x @ x_prev_full.T) / T
                        nu      = y_win_n - H @ F_current @ x_prev_full
                        S_nu    = (nu @ nu.T) / T

                        z_in = torch.cat([
                            A1.detach().reshape(-1),
                            A2.detach().reshape(-1),
                            S_delta.detach().reshape(-1),
                            S_nu.detach().reshape(-1),
                            C_delta.detach().reshape(-1),
                            F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)

                        dF        = model_mstep(z_in).view(m, m)
                        F_current = F_current + dF

                    # Mirror train objective: final RTS pass with last F_current.
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0_norm.clone(), T)
                    self.model.prior_Sigma = prior_P
                    self.model.init_hidden()


                    x_fwd2 = torch.stack(
                        [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                        dim=1)
                    xs2 = [None] * T
                    xs2[T - 1] = x_fwd2[:, T - 1]
                    self.model.InitBackward(xs2[T - 1])
                    xs2[T - 2] = self.model(None, x_fwd2[:, T - 2], x_fwd2[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        xs2[t] = self.model(None, x_fwd2[:, t], x_fwd2[:, t + 1], xs2[t + 2])
                    x_sm2 = torch.stack(xs2, dim=1)

                    # CV loss must match final train loss
                    resid_x_cv = x_sm2[0, :] - x_true_n[0, :]
                    cv_l = (resid_x_cv ** 2).mean()
                    cv_loss_sum += cv_l.item()
                    # loss_val = cv_l.item()#del
                    #
                    # if loss_val > HARD_TH:
                    #     hard_sequences.append({
                    #         "idx": j,
                    #         "loss": loss_val,
                    #         "x_true": x_true_n.detach().cpu().clone(),
                    #         "y": y_win_n.detach().cpu().clone(),
                    #         "x0": x0_norm.detach().cpu().clone(),
                    #     })
                    #
                    # elif loss_val < EASY_TH:
                    #     easy_sequences.append({
                    #         "idx": j,
                    #         "loss": loss_val,
                    #         "x_true": x_true_n.detach().cpu().clone(),
                    #         "y": y_win_n.detach().cpu().clone(),
                    #         "x0": x0_norm.detach().cpu().clone(),
                    #     })


            cv_avg = cv_loss_sum / self.N_CV
            scheduler.step(cv_avg)
            # print(f"[DEBUG] hard_sequences={len(hard_sequences)} easy_sequences={len(easy_sequences)}")
            # torch.save(hard_sequences, "hard_sequences.pt")
            # torch.save(easy_sequences, "easy_sequences.pt")
            if cv_avg < best_cv_loss:
                best_cv_loss = cv_avg
                os.makedirs(os.path.dirname(destination_path_M) or ".", exist_ok=True)
                torch.save(model_mstep, destination_path_M)

            cur_lr = self.M_optimizer.param_groups[0]['lr']
            mean_grad = grad_norm_sum / max(self.N_B, 1)
            mean_dF = dF_norm_sum / max(dF_norm_count, 1)
            print(f"  [M-net epoch {epoch:03d}] train={train_avg:.4f}  "
                  f"cv={cv_avg:.4f}  best={best_cv_loss:.4f}  lr={cur_lr:.2e}  "
                  f"grad_mean={mean_grad:.3e} grad_max={max_grad_norm_epoch:.3e}  "
                  f"clip_hits={clip_hit_count}/{self.N_B} bad_grads={grad_bad_count}  "
                  f"dF_mean={mean_dF:.3e}")

        print(f"Saved M-Network to: {destination_path_M}")

    def compute_F_opt_full_sequence(self,x_seq, ridge=1e-6):
        """
        x_seq: [m, T]

        Solves:
            F_opt = argmin_F sum_t ||F x(t-1) - x(t)||^2
        """
        m_local, T_local = x_seq.shape

        X_prev = x_seq[:, :-1]  # [m, T-1]
        X_next = x_seq[:, 1:]  # [m, T-1]

        try:
            F_opt = torch.linalg.lstsq(X_prev.T, X_next.T).solution.T
        except RuntimeError:
            XXt = X_prev @ X_prev.T
            reg = ridge * torch.eye(m_local, device=x_seq.device, dtype=x_seq.dtype)
            F_opt = (X_next @ X_prev.T) @ torch.linalg.inv(XXt + reg)

        return F_opt
    def train_emkalmannet_weather_rts(
            self, SysModel,
            cv_input, cv_target, cv_x0,
            train_input, train_target, train_x0,
            destination_path_M,
            num_em_iters=2, alpha=(0.05, 0.15, 0.85),
            lambda_F=1e-2,
            lambda_f_loss=10.0,
            f_loss=True,
            clip_grad=1.0,
    ):
        device = self.device
        dtype = train_input[0].dtype
        m, n = SysModel.m, SysModel.n
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)


        if self.M_model is None:
            self.M_model = DeltaF_MStepNet(m=m, n=n, d_hidden=256).to(device)
            self.M_optimizer = torch.optim.Adam(
                self.M_model.parameters(), lr=1e-4, weight_decay=1e-5)

        model_mstep = self.M_model.train()
        best_cv_loss = 1e18
        batch_size = 10

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.M_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        for epoch in range(self.N_steps):

            # ---- TRAIN ----
            model_mstep.train()
            train_loss_sum = 0.0
            grad_norm_sum = 0.0
            max_grad_norm_epoch = 0.0
            clip_hit_count = 0
            grad_bad_count = 0
            dF_norm_sum = 0.0
            dF_norm_count = 0

            for j in range(self.N_B):
                self.M_optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device, dtype=dtype)

                for _ in range(batch_size):
                    idx   = random.randint(0, self.N_E - 1)
                    y_win = train_input[idx].to(device)   # [3, TAU] observations y
                    x_true = train_target[idx].to(device)  # [4, TAU] true state x
                    T     = int(y_win.size(-1))

                    # Compute normalization stats from TRUE STATE x
                    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                    x_mean[0] = train_x0[idx][0]
                    x_std[0] = 4
                    y_win_n = (y_win - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats
                    x_true_n = (x_true - x_mean) / x_std
                    F_opt = self.compute_F_opt_full_sequence(x_true_n)
                    F_base = SysModel.F_train[0].to(device) \
                        if isinstance(SysModel.F_train, list) \
                        else SysModel.F_train.to(device)
                    H = SysModel.H.to(device)

                    x0_raw = train_x0[idx].to(device)
                    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    F_current  = F_base.clone()
                    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

                    for em_iter in range(num_em_iters):
                        # Use classical RTS with current F
                        [mse_arr, mse_avg, mse_db, X_smooth, P_smooth, V_smooth] = S_Test(
                            SysModel,
                            [y_win_n],
                            [x_true_n],
                            F=[F_current],
                            H=[H],
                            generate_f=False,
                            generate_h=False,
                            init_x_list=[x0_norm.squeeze()],
                            init_P_list=[prior_P],
                        )
                        x_sm = X_smooth[0]  # [m, T]

                        # Sufficient stats
                        x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)
                        x_curr_full = x_sm

                        A1      = (x_curr_full @ x_prev_full.T) / T
                        A2      = (x_prev_full @ x_prev_full.T) / T
                        delta_x = x_curr_full - F_current @ x_prev_full
                        S_delta = (delta_x @ delta_x.T) / T
                        C_delta = (delta_x @ x_prev_full.T) / T
                        nu      = y_win_n - H @ F_current @ x_prev_full
                        S_nu    = (nu @ nu.T) / T

                        z_in = torch.cat([
                            A1.detach().reshape(-1),
                            A2.detach().reshape(-1),
                            S_delta.detach().reshape(-1),
                            S_nu.detach().reshape(-1),
                            C_delta.detach().reshape(-1),
                            F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)

                        dF = 100*model_mstep(z_in).view(m, m)
                        dF_norm_sum += dF.detach().norm().item()
                        dF_norm_count += 1
                        F_current = F_current + dF

                        reg_F = lambda_F * (dF ** 2).sum()

                        # Loss on state x, feature 0
                        resid_x_iter = x_sm[0, :] - x_true_n[0, :]
                        loss_x_iter = (resid_x_iter ** 2).mean()
                        iter_loss = alpha[em_iter] * (loss_x_iter + reg_F)
                        # total_loss = total_loss + iter_loss

                    # Final RTS pass with last F
                    [mse_arr, mse_avg, mse_db, X_smooth2, P_smooth2, V_smooth2] = S_Test(
                        SysModel,
                        [y_win_n],
                        [x_true_n],
                        F=[F_current],
                        H=[H],
                        generate_f=False,
                        generate_h=False,
                        init_x_list=[x0_norm.squeeze()],
                        init_P_list=[prior_P],
                    )
                    x_sm2 = X_smooth2[0]

                    # Final loss
                    resid_x_final = x_sm2[0, :] - x_true_n[0, :]
                    loss_x_final = (resid_x_final ** 2).mean()
                    # total_loss = total_loss + alpha[-1] * loss_x_final

                    # loss_f = torch.norm(F_current - F_opt, p='fro') ** 2
                    # total_loss = total_loss +  loss_f
                    loss_f = torch.norm(F_current[0] - F_opt[0], p='fro') ** 2
                    total_loss = total_loss +  loss_f

                    batch_loss = batch_loss + total_loss / batch_size

                batch_loss.backward()

                grads = [p.grad for p in model_mstep.parameters() if p.grad is not None]
                if grads:
                    grad_sq = sum((g.detach().norm() ** 2 for g in grads))
                    grad_norm = torch.sqrt(grad_sq).item()
                else:
                    grad_norm = 0.0

                has_bad_grad = any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads)
                if has_bad_grad:
                    grad_bad_count += 1

                pre_clip_norm = grad_norm
                if clip_grad > 0:
                    clipped_total_norm = torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), clip_grad)
                    if float(clipped_total_norm) > clip_grad:
                        clip_hit_count += 1
                self.M_optimizer.step()
                train_loss_sum += batch_loss.detach().item()
                grad_norm_sum += pre_clip_norm
                if pre_clip_norm > max_grad_norm_epoch:
                    max_grad_norm_epoch = pre_clip_norm

            train_avg = train_loss_sum / max(self.N_B, 1)

            # ---- CV ----
            model_mstep.eval()
            cv_loss_sum = 0.0
            with torch.no_grad():
                for j in range(self.N_CV):
                    # ...existing code...
                    y_win = cv_input[j].to(device)
                    x_true = cv_target[j].to(device)
                    T = int(y_win.size(-1))

                    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                    x_mean[0] =  cv_x0[j][0]
                    x_std[0] = 4
                    y_win_n = (y_win - x_mean[1:]) / x_std[1:]
                    x_true_n = (x_true - x_mean) / x_std
                    # direct transition-fit losses on the true sequence
                    X_prev_true = x_true_n[:, :-1]  # [m, T-1]
                    X_next_true = x_true_n[:, 1:]  # [m, T-1]

                    F_opt = self.compute_F_opt_full_sequence(x_true_n)
                    F_base = SysModel.F_valid[0].to(device) \
                        if isinstance(SysModel.F_valid, list) \
                        else SysModel.F_valid.to(device)
                    H = SysModel.H.to(device)
                    fit_loss_F_base = torch.sum((F_base @ X_prev_true - X_next_true) ** 2)
                    fit_loss_F_opt = torch.sum((F_opt @ X_prev_true - X_next_true) ** 2)
                    x0_raw = cv_x0[j].to(device)
                    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    F_current = F_base.clone()
                    for em_iter in range(num_em_iters):
                        # Use classical RTS with current F
                        [mse_arr, mse_avg, mse_db, X_smooth, P_smooth, V_smooth] = S_Test(
                            SysModel,
                            [y_win_n],
                            [x_true_n],
                            F=[F_current],
                            H=[H],
                            generate_f=False,
                            generate_h=False,
                            init_x_list=[x0_norm.squeeze()],
                            init_P_list=[prior_P],
                        )
                        x_sm = X_smooth[0]

                        x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)
                        x_curr_full = x_sm
                        A1      = (x_curr_full @ x_prev_full.T) / T
                        A2      = (x_prev_full @ x_prev_full.T) / T
                        delta_x = x_curr_full - F_current @ x_prev_full
                        S_delta = (delta_x @ delta_x.T) / T
                        C_delta = (delta_x @ x_prev_full.T) / T
                        nu      = y_win_n - H @ F_current @ x_prev_full
                        S_nu    = (nu @ nu.T) / T

                        z_in = torch.cat([
                            A1.detach().reshape(-1),
                            A2.detach().reshape(-1),
                            S_delta.detach().reshape(-1),
                            S_nu.detach().reshape(-1),
                            C_delta.detach().reshape(-1),
                            F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)

                        dF = 100*model_mstep(z_in).view(m, m)
                        # print('df',dF)
                        F_current = F_current + dF
                    # # Final RTS pass with last F
                    # [mse_arr, mse_avg, mse_db, X_smooth2, P_smooth2, V_smooth2] = S_Test(
                    #     SysModel,
                    #     [y_win_n],
                    #     [x_true_n],
                    #     F=[F_current],
                    #     H=[H],
                    #     generate_f=False,
                    #     generate_h=False,
                    #     init_x_list=[x0_norm.squeeze()],
                    #     init_P_list=[prior_P],
                    # )
                    # x_sm2 = X_smooth2[0]
                    #
                    # resid_x_cv = x_sm2[0, :] - x_true_n[0, :]
                    # cv_l = (resid_x_cv ** 2).mean()
                    # # cv_loss_sum += cv_l.item()
                    # cv_f_loss = torch.norm(F_current - F_opt, p='fro') ** 2
                    # cv_loss_sum += cv_f_loss.item()
                    # Final RTS pass with learned F_current
                    [mse_arr, mse_avg, mse_db, X_smooth2, P_smooth2, V_smooth2] = S_Test(
                        SysModel,
                        [y_win_n],
                        [x_true_n],
                        F=[F_current],
                        H=[H],
                        generate_f=False,
                        generate_h=False,
                        init_x_list=[x0_norm.squeeze()],
                        init_P_list=[prior_P],
                    )
                    x_sm2 = X_smooth2[0]

                    resid_x_cv = x_sm2[0, :] - x_true_n[0, :]
                    cv_l = (resid_x_cv ** 2).mean()

                    # RTS loss if we use the oracle F_opt
                    [mse_arr_opt, mse_avg_opt, mse_db_opt, X_smooth_opt, P_smooth_opt, V_smooth_opt] = S_Test(
                        SysModel,
                        [y_win_n],
                        [x_true_n],
                        F=[F_opt],
                        H=[H],
                        generate_f=False,
                        generate_h=False,
                        init_x_list=[x0_norm.squeeze()],
                        init_P_list=[prior_P],
                    )
                    x_sm_opt = X_smooth_opt[0]

                    resid_x_cv_opt = x_sm_opt[0, :] - x_true_n[0, :]
                    cv_l_opt = (resid_x_cv_opt ** 2).mean()
                    fit_loss_F_current = torch.sum((F_current @ X_prev_true - X_next_true) ** 2)
                    # cv_f_loss = torch.norm(F_current - F_opt, p='fro') ** 2
                    cv_f_loss = torch.norm(F_current[0] - F_opt[0], p='fro') ** 2
                    cv_loss_sum += cv_f_loss.item()

                    if j == 0:
                        print(
                            f"[CV DEBUG] "
                            f"F_loss={cv_f_loss.item():.6f} | "
                            f"RTS_loss(F_current)={cv_l.item():.6f} | "
                            f"RTS_loss(F_opt)={cv_l_opt.item():.6f}"
                        )
                    #     print(
                    #         f"[CV FIT DEBUG] "
                    #         f"fit_loss(F_base)={fit_loss_F_base.item():.6f} | "
                    #         f"fit_loss(F_current)={fit_loss_F_current.item():.6f} | "
                    #         f"fit_loss(F_opt)={fit_loss_F_opt.item():.6f}"
                    #     )
                    #     print("F_opt:")
                    #     print(F_opt.detach().cpu())
                    #     print("F_current:")
                    #     print(F_current.detach().cpu())
            cv_avg = cv_loss_sum / self.N_CV
            scheduler.step(cv_avg)
            if cv_avg < best_cv_loss:
                best_cv_loss = cv_avg
                os.makedirs(os.path.dirname(destination_path_M) or ".", exist_ok=True)
                torch.save(model_mstep, destination_path_M)

            cur_lr = self.M_optimizer.param_groups[0]['lr']
            mean_grad = grad_norm_sum / max(self.N_B, 1)
            mean_dF = dF_norm_sum / max(dF_norm_count, 1)
            print(f"  [M-net (RTS) epoch {epoch:03d}] train={train_avg:.4f}  "
                  f"cv={cv_avg:.4f}  best={best_cv_loss:.4f}  lr={cur_lr:.2e}  "
                  f"grad_mean={mean_grad:.3e} grad_max={max_grad_norm_epoch:.3e}  "
                  f"clip_hits={clip_hit_count}/{self.N_B} bad_grads={grad_bad_count}  "
                  f"dF_mean={mean_dF:.3e}")

        print(f"Saved M-Network (trained with RTS) to: {destination_path_M}")
        # ------------------------------------------------------------------
        # M-step test  (test_mstep_weather)
        # ------------------------------------------------------------------

    def test_mstep_weather_rts(
            self, SysModel, test_input, test_target, test_x0,
            destination_path_M,
            num_em_iters=2,
            print_F_every=50,
    ):
        device = self.device
        m = SysModel.m
        N_T = len(test_input)


        model_mstep = torch.load(destination_path_M, weights_only=False,
                                 map_location=device).eval()


        # Track MSE per iteration across ALL sequences
        mse_per_iter = torch.zeros(num_em_iters + 1, device=device)
        count_per_iter = torch.zeros(num_em_iters + 1, device=device)
        mse_per_iter_denorm = torch.zeros(num_em_iters + 1, device=device)  # NEW: denormalized MSE

        final_F_list = []
        preds_out = []

        with torch.no_grad():
            for j in range(N_T):
                if j == 0:
                    continue  # Skip first test sample to avoid data leakage in normalization (since it uses true state stats) del
                y_win = test_input[j].to(device)  # [3, TAU] observations y
                x_true = test_target[j].to(device)  # [4, TAU] true state x
                T = y_win.size(-1)

                # Compute normalization stats from TRUE STATE x
                x_mean, x_std, x_mean0, x_std0 = _win_norm_4d(x_true, device, y_win.dtype)
                x_mean[0] =  test_x0[j][0]
                x_std[0]  = 4
                y_win_n = (y_win - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats
                x_true_n = (x_true - x_mean) / x_std
                F_opt = self.compute_F_opt_full_sequence(x_true_n)
                F_base = SysModel.F_test[0].to(device) \
                    if isinstance(SysModel.F_test, list) \
                    else SysModel.F_test.to(device)
                H = SysModel.H.to(device)

                x0_raw = test_x0[j].to(device)
                # Normalize x0 using stats from TRUE STATE x
                x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

                prior_P = SysModel.m2x_0.clone().detach().to(device) \
                    if hasattr(SysModel, "m2x_0") \
                    else torch.eye(m, device=device, dtype=y_win.dtype)

                F_current = F_base.clone()

                # Track F evolution and MSE per iteration for this sequence
                F_evolution = [F_current.clone()]
                mse_evolution = []
                mse_evolution_denorm = []  # NEW: track denormalized MSE

                for em_iter in range(num_em_iters):


                    [mse_arr, mse_avg, mse_db, X_smooth, P_smooth, V_smooth] = S_Test(
                        SysModel,
                        [y_win_n],
                        [x_true_n],
                        F=[F_current],
                        H=[H],
                        generate_f=False,
                        generate_h=False,
                        init_x_list=[x0_norm.squeeze()],
                        init_P_list=[prior_P],
                    )
                    x_sm = X_smooth[0]  # [m, T]

                    x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)
                    x_curr_full = x_sm
                    A1 = (x_curr_full @ x_prev_full.T) / T
                    A2 = (x_prev_full @ x_prev_full.T) / T
                    delta_x = x_curr_full - F_current @ x_prev_full
                    S_delta = (delta_x @ delta_x.T) / T
                    C_delta = (delta_x @ x_prev_full.T) / T
                    nu = y_win_n - H @ F_current @ x_prev_full
                    S_nu = (nu @ nu.T) / T

                    z_in = torch.cat([
                        A1.detach().reshape(-1),
                        A2.detach().reshape(-1),
                        S_delta.detach().reshape(-1),
                        S_nu.detach().reshape(-1),
                        C_delta.detach().reshape(-1),
                        F_current.detach().reshape(-1),
                    ], dim=0).view(1, -1)

                    dF = model_mstep(z_in).view(m, m)
                    F_current = F_current + 100*dF
                    F_evolution.append(F_current.clone())
                    f_mse_iter = torch.norm(F_current - F_opt, p='fro') ** 2
                    # Compute MSE for this iteration on state x, feature 0 (normalized)
                    x_pred_iter = x_sm[0, :]
                    true_tavg_norm = x_true_n[0, :]
                    mse_iter = ((x_pred_iter - true_tavg_norm) ** 2).mean()

                    # NEW: Compute denormalized MSE
                    x_pred_iter_denorm = x_pred_iter * x_std[0] + x_mean[0]
                    true_tavg_denorm = x_true[0, :]
                    mse_iter_denorm = ((x_pred_iter_denorm - true_tavg_denorm) ** 2).mean()

                    mse_evolution.append(mse_iter.item())
                    mse_evolution_denorm.append(mse_iter_denorm.item())

                    # Accumulate for global statistics
                    mse_per_iter[em_iter] += mse_iter.item()
                    mse_per_iter_denorm[em_iter] += mse_iter_denorm.item()
                    count_per_iter[em_iter] += 1

                # Final RTS pass with last F_current (to match train/CV final prediction)


                [mse_arr, mse_avg, mse_db, X_smooth2, P_smooth2, V_smooth2] = S_Test(
                    SysModel,
                    [y_win_n],
                    [x_true_n],
                    F=[F_current],
                    H=[H],
                    generate_f=False,
                    generate_h=False,
                    init_x_list=[x0_norm.squeeze()],
                    init_P_list=[prior_P],
                )
                x_sm2 = X_smooth2[0]

                # Denormalize full x prediction [m, T]
                x_sm2_denorm = x_sm2.clone()
                for i in range(m):
                    x_sm2_denorm[i, :] = x_sm2[i, :] * x_std[i] + x_mean[i]
                f_mse_final = torch.norm(F_current - F_opt, p='fro') ** 2
                # print(f"      F error vs F_opt: {f_mse_final.item():.6e}")
                preds_out.append({
                    "seq_index": j,
                    "x_pred_norm": x_sm2.detach().cpu(),
                    "x_true_norm": x_true_n.detach().cpu(),
                    "x_pred_denorm": x_sm2_denorm.detach().cpu(),  # NEW: full [m, T]
                    "x_true_denorm": x_true[0, :].detach().cpu(),  # tavg only
                })

                # Final test MSE on state x, feature 0 (normalized)
                pred_tavg_norm = x_sm2[0, :]
                true_tavg_norm = x_true_n[0, :]
                mse_final = ((pred_tavg_norm - true_tavg_norm) ** 2).mean()

                # NEW: Final denormalized MSE
                pred_tavg_denorm = x_sm2_denorm[0, :]
                true_tavg_denorm = x_true[0, :]
                mse_final_denorm = ((pred_tavg_denorm - true_tavg_denorm) ** 2).mean()

                mse_per_iter[num_em_iters] += mse_final.item()
                mse_per_iter_denorm[num_em_iters] += mse_final_denorm.item()
                count_per_iter[num_em_iters] += 1

                final_F_list.append(F_current.detach().clone())

                # Print F evolution every print_F_every windows
                # if j % print_F_every == 0:
                #     print(f"  [test window {j:04d}/{N_T}]")
                #     for it in range(num_em_iters):
                #         print(
                #             f"    After EM iter {it + 1}: MSE={mse_evolution[it]:.6e} (norm) "
                #             f"/ {mse_evolution_denorm[it]:.6e} (denorm) "
                #             f"({10 * torch.log10(torch.tensor(mse_evolution[it]) + 1e-12):.2f} dB)"
                #         )
                #         print(f"      Updated F matrix:\n{F_evolution[it + 1].cpu().numpy()}")

        # Average MSE per iteration (only where computed)
        mean_mse = torch.zeros(num_em_iters + 1, device=device)
        mean_mse_denorm = torch.zeros(num_em_iters + 1, device=device)  # NEW
        for k in range(num_em_iters + 1):
            if count_per_iter[k] > 0:
                mean_mse[k] = mse_per_iter[k] / count_per_iter[k]
                mean_mse_denorm[k] = mse_per_iter_denorm[k] / count_per_iter[k]  # NEW

        mean_mse_db = 10 * torch.log10(mean_mse + 1e-12)
        mean_mse_db_denorm = 10 * torch.log10(mean_mse_denorm + 1e-12)  # NEW

        print("\n[M-Network TEST] Mean MSE per EM iteration (across all test windows):")
        for k in range(num_em_iters + 1):
            print(f"  After EM iter {k + 1}: MSE={mean_mse[k].item():.6e} (norm) / "
                  f"{mean_mse_denorm[k].item():.6e} °C² (denorm)  "
                  f"({mean_mse_db[k].item():.2f} dB / {mean_mse_db_denorm[k].item():.2f} dB)")

        return mean_mse, mean_mse_db, final_F_list, preds_out, mean_mse_denorm, mean_mse_db_denorm
    # ------------------------------------------------------------------
    # M-step test  (test_mstep_weather)
    # ------------------------------------------------------------------
    def test_mstep_weather(
        self, SysModel, test_input, test_target, test_x0,
        destination_path_RTS, destination_path_M,
        num_em_iters=2,
        print_F_every=50,
    ):
        device = self.device
        m      = SysModel.m
        N_T    = len(test_input)

        self.model  = torch.load(destination_path_RTS, weights_only=False,
                                 map_location=device).eval()
        model_mstep = torch.load(destination_path_M,  weights_only=False,
                                 map_location=device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Track MSE per iteration across ALL sequences
        mse_per_iter = torch.zeros(num_em_iters+1, device=device)
        count_per_iter = torch.zeros(num_em_iters+1, device=device)
        mse_per_iter_denorm = torch.zeros(num_em_iters+1, device=device)  # NEW: denormalized MSE

        final_F_list = []
        preds_out = []

        with torch.no_grad():
            for j in range(N_T):
                if j == 0:
                    continue  # Skip first test sample to avoid data leakage in normalization (since it uses true state stats) del
                y_win  = test_input[j].to(device)   # [3, TAU] observations y
                x_true = test_target[j].to(device)  # [4, TAU] true state x
                T      = y_win.size(-1)

                # Compute normalization stats from TRUE STATE x
                x_mean, x_std, x_mean0, x_std0 = _win_norm_4d(x_true, device, y_win.dtype)
                prev_x_mean, prev_x_std, _, _ = _win_norm_4d(test_target[j-1], device, y_win.dtype)  # del
                x_mean[0] = prev_x_mean[0]  # Use the mean of
                x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                # x_mean[0] =  test_x0[j][0]
                # x_std[0]  = 4
                y_win_n  = (y_win  - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats
                x_true_n = (x_true - x_mean) / x_std

                F_base = SysModel.F_test[0].to(device) \
                    if isinstance(SysModel.F_test, list) \
                    else SysModel.F_test.to(device)
                H = SysModel.H.to(device)

                x0_raw = test_x0[j].to(device)
                # Normalize x0 using stats from TRUE STATE x
                x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std

                prior_P = SysModel.m2x_0.clone().detach().to(device) \
                    if hasattr(SysModel, "m2x_0") \
                    else torch.eye(m, device=device, dtype=y_win.dtype)

                F_current = F_base.clone()

                # Track F evolution and MSE per iteration for this sequence
                F_evolution = [F_current.clone()]
                mse_evolution = []
                mse_evolution_denorm = []  # NEW: track denormalized MSE

                for em_iter in range(num_em_iters):
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0_norm.clone(), T)
                    self.model.prior_Sigma = prior_P
                    self.model.init_hidden()


                    x_fwd = torch.stack(
                        [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                        dim=1)
                    xs = [None] * T
                    xs[T - 1] = x_fwd[:, T - 1]
                    self.model.InitBackward(xs[T - 1])
                    xs[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                    x_sm = torch.stack(xs, dim=1)

                    x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)
                    x_curr_full = x_sm
                    A1      = (x_curr_full @ x_prev_full.T) / T
                    A2      = (x_prev_full @ x_prev_full.T) / T
                    delta_x = x_curr_full - F_current @ x_prev_full
                    S_delta = (delta_x @ delta_x.T) / T
                    C_delta = (delta_x @ x_prev_full.T) / T
                    nu      = y_win_n - H @ F_current @ x_prev_full
                    S_nu    = (nu @ nu.T) / T

                    z_in = torch.cat([
                        A1.detach().reshape(-1),
                        A2.detach().reshape(-1),
                        S_delta.detach().reshape(-1),
                        S_nu.detach().reshape(-1),
                        C_delta.detach().reshape(-1),
                        F_current.detach().reshape(-1),
                    ], dim=0).view(1, -1)

                    dF        = model_mstep(z_in).view(m, m)
                    F_current = F_current + dF
                    F_evolution.append(F_current.clone())

                    # Compute MSE for this iteration on state x, feature 0 (normalized)
                    x_pred_iter = x_sm[0, :]
                    true_tavg_norm = x_true_n[0, :]
                    mse_iter = ((x_pred_iter - true_tavg_norm) ** 2).mean()

                    # NEW: Compute denormalized MSE
                    x_pred_iter_denorm = x_pred_iter * x_std[0] + x_mean[0]
                    true_tavg_denorm = x_true[0, :]
                    mse_iter_denorm = ((x_pred_iter_denorm - true_tavg_denorm) ** 2).mean()

                    mse_evolution.append(mse_iter.item())
                    mse_evolution_denorm.append(mse_iter_denorm.item())

                    # Accumulate for global statistics
                    mse_per_iter[em_iter] += mse_iter.item()
                    mse_per_iter_denorm[em_iter] += mse_iter_denorm.item()
                    count_per_iter[em_iter] += 1

                # Final RTS pass with last F_current (to match train/CV final prediction)
                self.model.update_F(F_current)
                self.model.InitSequence(x0_norm.clone(), T)
                self.model.prior_Sigma = prior_P
                self.model.init_hidden()


                x_fwd2 = torch.stack(
                    [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                    dim=1
                )
                xs2 = [None] * T
                xs2[T - 1] = x_fwd2[:, T - 1]
                self.model.InitBackward(xs2[T - 1])
                xs2[T - 2] = self.model(None, x_fwd2[:, T - 2], x_fwd2[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    xs2[t] = self.model(None, x_fwd2[:, t], x_fwd2[:, t + 1], xs2[t + 2])
                x_sm2 = torch.stack(xs2, dim=1)

                # Denormalize full x prediction [m, T]
                x_sm2_denorm = x_sm2.clone()
                for i in range(m):
                    x_sm2_denorm[i, :] = x_sm2[i, :] * x_std[i] + x_mean[i]

                preds_out.append({
                    "seq_index": j,
                    "x_pred_norm": x_sm2.detach().cpu(),
                    "x_true_norm": x_true_n.detach().cpu(),
                    "x_pred_denorm": x_sm2_denorm.detach().cpu(),  # NEW: full [m, T]
                    "x_true_denorm": x_true[0, :].detach().cpu(),  # tavg only
                })

                # Final test MSE on state x, feature 0 (normalized)
                pred_tavg_norm = x_sm2[0, :]
                true_tavg_norm = x_true_n[0, :]
                mse_final = ((pred_tavg_norm - true_tavg_norm) ** 2).mean()

                # NEW: Final denormalized MSE
                pred_tavg_denorm = x_sm2_denorm[0, :]
                true_tavg_denorm = x_true[0, :]
                mse_final_denorm = ((pred_tavg_denorm - true_tavg_denorm) ** 2).mean()

                mse_per_iter[num_em_iters] += mse_final.item()
                mse_per_iter_denorm[num_em_iters] += mse_final_denorm.item()
                count_per_iter[num_em_iters] += 1

                final_F_list.append(F_current.detach().clone())

                # Print F evolution every print_F_every windows
                if j % print_F_every == 0:
                    print(f"  [test window {j:04d}/{N_T}]")
                    for it in range(num_em_iters):
                        print(
                            f"    After EM iter {it + 1}: MSE={mse_evolution[it]:.6e} (norm) "
                            f"/ {mse_evolution_denorm[it]:.6e} (denorm) "
                            f"({10 * torch.log10(torch.tensor(mse_evolution[it]) + 1e-12):.2f} dB)"
                        )
                        print(f"      Updated F matrix:\n{F_evolution[it + 1].cpu().numpy()}")

        # Average MSE per iteration (only where computed)
        mean_mse = torch.zeros(num_em_iters+1, device=device)
        mean_mse_denorm = torch.zeros(num_em_iters+1, device=device)  # NEW
        for k in range(num_em_iters+1):
            if count_per_iter[k] > 0:
                mean_mse[k] = mse_per_iter[k] / count_per_iter[k]
                mean_mse_denorm[k] = mse_per_iter_denorm[k] / count_per_iter[k]  # NEW

        mean_mse_db = 10 * torch.log10(mean_mse + 1e-12)
        mean_mse_db_denorm = 10 * torch.log10(mean_mse_denorm + 1e-12)  # NEW

        print("\n[M-Network TEST] Mean MSE per EM iteration (across all test windows):")
        for k in range(num_em_iters+1):
            print(f"  After EM iter {k+1}: MSE={mean_mse[k].item():.6e} (norm) / "
                  f"{mean_mse_denorm[k].item():.6e} °C² (denorm)  "
                  f"({mean_mse_db[k].item():.2f} dB / {mean_mse_db_denorm[k].item():.2f} dB)")

        return mean_mse, mean_mse_db, final_F_list, preds_out, mean_mse_denorm, mean_mse_db_denorm

    # -----------------------------------------------------------------------
    # Joint training
    # -----------------------------------------------------------------------

    def train_joint_weather(
        self,
        SysModel,
        train_input, train_target, train_x0,
        cv_input,    cv_target,    cv_x0,
        path_rts_in,  path_m_in,
        path_rts_out, path_m_out,
        batch_size=5,
        num_em_iters=2,
        lambda_F=1e-3,
        clip_grad=1.0,
        alpha=(0.05, 0.1, 0.85),
        lr_rts=1e-4, lr_m=1e-4,
        wd_rts=1e-5, wd_m=1e-5,
    ):
        device = self.device
        dtype  = train_input[0].dtype
        m, n   = SysModel.m, SysModel.n
        N_E    = len(train_input)
        N_CV   = len(cv_input)

        self.N_E  = N_E
        self.N_CV = N_CV

        self.model = torch.load(path_rts_in, weights_only=False,
                                map_location=device).to(device).train()
        model_mstep = torch.load(path_m_in, weights_only=False,
                                 map_location=device).to(device).train()

        rts_opt = torch.optim.Adam(self.model.parameters(), lr=lr_rts, weight_decay=wd_rts)
        m_opt   = torch.optim.Adam(model_mstep.parameters(), lr=lr_m, weight_decay=wd_m)
        rts_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            rts_opt, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )
        m_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            m_opt, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )

        best_cv = float("inf")

        for epoch in range(self.N_steps):

            # ---- TRAIN ----
            self.model.train(); model_mstep.train()
            train_loss_sum = 0.0
            rts_grad_norm_sum = 0.0
            m_grad_norm_sum = 0.0
            max_rts_grad_norm = 0.0
            max_m_grad_norm = 0.0
            rts_clip_hits = 0
            m_clip_hits = 0
            bad_grad_count = 0
            dF_norm_sum = 0.0
            dF_norm_count = 0

            for j in range(self.N_B):
                rts_opt.zero_grad(); m_opt.zero_grad()
                batch_loss = torch.tensor(0.0, device=device, dtype=dtype)

                for _ in range(batch_size):
                    idx   = random.randint(0, N_E - 1)
                    if idx == 0:
                        continue  # Skip first batch to avoid data leakage in normalization (since it uses true state stats) del
                    y_win = train_input[idx].to(device)   # [3, TAU] observations y
                    x_true = train_target[idx].to(device)  # [4, TAU] true state x
                    T     = y_win.size(-1)

                    # Compute normalization stats from TRUE STATE x
                    x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                    prev_x_mean, prev_x_std, _, _ = _win_norm_4d(train_target[idx-1], device, dtype)  # del
                    x_mean[0] = prev_x_mean[0]  # Use the mean of
                    x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                    # x_mean[0] = train_x0[idx][0]
                    # x_std[0] = 4
                    y_win_n = (y_win - x_mean[1:]) / x_std[1:]  # Normalize obs using x[1:] stats
                    y_tgt_n = (x_true - x_mean) / x_std

                    F_base = SysModel.F.clone().detach().to(device)
                    H      = SysModel.H.to(device)

                    x0_raw  = train_x0[idx].to(device)
                    x0_norm = (x0_raw.view(m, 1) - x_mean) / x_std  # [m,1] normalized with state stats

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)


                    F_current  = F_base.clone()
                    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

                    for em_iter in range(num_em_iters):
                        # Joint: NO detach on F → grad flows into RTSNet through update_F
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.prior_Sigma = prior_P
                        self.model.init_hidden()


                        x_fwd = torch.stack(
                            [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                            dim=1)
                        xs = [None] * T
                        xs[T - 1] = x_fwd[:, T - 1]
                        self.model.InitBackward(xs[T - 1])
                        xs[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                        # NO detach – joint grad flows through x_sm into RTSNet
                        x_sm = torch.stack(xs, dim=1)

                        x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)
                        x_curr_full = x_sm

                        A1      = (x_curr_full @ x_prev_full.T) / T
                        A2      = (x_prev_full @ x_prev_full.T) / T
                        delta_x = x_curr_full - F_current @ x_prev_full
                        S_delta = (delta_x @ delta_x.T) / T
                        C_delta = (delta_x @ x_prev_full.T) / T
                        nu      = y_win_n - H @ F_current @ x_prev_full
                        S_nu    = (nu @ nu.T) / T

                        # z_in = torch.cat([
                        #     A1.reshape(-1),
                        #     A2.reshape(-1),
                        #     S_delta.reshape(-1),
                        #     S_nu.reshape(-1),
                        #     C_delta.reshape(-1),
                        #     F_current.reshape(-1),
                        # ], dim=0).view(1, -1)
                        z_in = torch.cat([
                            A1.detach().reshape(-1),
                            A2.detach().reshape(-1),
                            S_delta.detach().reshape(-1),
                            S_nu.detach().reshape(-1),
                            C_delta.detach().reshape(-1),
                            F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)
                        dF  = model_mstep(z_in).view(m, m)
                        dF_norm_sum += dF.detach().norm().item()
                        dF_norm_count += 1
                        F_current = F_current + dF

                        reg_F  = lambda_F * (dF ** 2).sum()

                        # Loss on state x, feature 0
                        resid_x_iter = x_sm[0, :] - y_tgt_n[0, :]
                        loss_x_iter = (resid_x_iter ** 2).mean()
                        iter_loss = alpha[min(em_iter, len(alpha) - 2)] * (loss_x_iter + reg_F)
                        total_loss = total_loss + iter_loss

                    # Final RTS pass – NO detach, joint grad flows into RTSNet
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0_norm.clone(), T)
                    self.model.prior_Sigma = prior_P
                    self.model.init_hidden()


                    x_fwd2 = torch.stack(
                        [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                        dim=1)
                    xs2 = [None] * T
                    xs2[T - 1] = x_fwd2[:, T - 1]
                    self.model.InitBackward(xs2[T - 1])
                    xs2[T - 2] = self.model(None, x_fwd2[:, T - 2], x_fwd2[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        xs2[t] = self.model(None, x_fwd2[:, t], x_fwd2[:, t + 1], xs2[t + 2])
                    x_sm2 = torch.stack(xs2, dim=1)  # NO detach

                    # Final loss on state x, feature 0
                    resid_x_final = x_sm2[0, :] - y_tgt_n[0, :]
                    loss_x_final = (resid_x_final ** 2).mean()
                    total_loss = total_loss + alpha[-1] * loss_x_final

                    batch_loss = batch_loss + total_loss / batch_size

                batch_loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), clip_grad)
                rts_opt.step(); m_opt.step()
                train_loss_sum += batch_loss.detach().item()
                # batch_loss.backward()

                # rts_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                # m_grads = [p.grad for p in model_mstep.parameters() if p.grad is not None]
                #
                # if rts_grads:
                #     rts_grad_norm = torch.sqrt(sum((g.detach().norm() ** 2 for g in rts_grads))).item()
                # else:
                #     rts_grad_norm = 0.0
                #
                # if m_grads:
                #     m_grad_norm = torch.sqrt(sum((g.detach().norm() ** 2 for g in m_grads))).item()
                # else:
                #     m_grad_norm = 0.0
                #
                # bad_rts_grad = any(torch.isnan(g).any() or torch.isinf(g).any() for g in rts_grads)
                # bad_m_grad = any(torch.isnan(g).any() or torch.isinf(g).any() for g in m_grads)
                #
                # if bad_rts_grad or bad_m_grad:
                #     bad_grad_count += 1
                #     rts_opt.zero_grad(set_to_none=True)
                #     m_opt.zero_grad(set_to_none=True)
                #     print(f"  [JOINT epoch {epoch:03d} batch {j:03d}] bad grads: RTS={bad_rts_grad} M={bad_m_grad}")
                #     continue
                #
                # rts_grad_norm_sum += rts_grad_norm
                # m_grad_norm_sum += m_grad_norm
                # max_rts_grad_norm = max(max_rts_grad_norm, rts_grad_norm)
                # max_m_grad_norm = max(max_m_grad_norm, m_grad_norm)
                #
                # if clip_grad > 0:
                #     rts_clip_val = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                #     m_clip_val = torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), clip_grad)
                #
                #     if float(rts_clip_val) > clip_grad:
                #         rts_clip_hits += 1
                #     if float(m_clip_val) > clip_grad:
                #         m_clip_hits += 1
                #
                # rts_opt.step()
                # m_opt.step()
                # train_loss_sum += batch_loss.detach().item()

            train_avg = train_loss_sum / self.N_B

            # ---- CV ----
            self.model.eval(); model_mstep.eval()
            cv_loss_sum = 0.0
            with torch.no_grad():
                for j in range(N_CV):
                    if j == 0:
                        continue  # Skip first CV sample to avoid data leakage in normalization (since it uses true state stats) del
                    y_win = cv_input[j].to(device)
                    y_tgt = cv_target[j].to(device)
                    T     = y_win.size(-1)

                    x_mean, x_std, _, _ = _win_norm_4d(y_tgt, device, dtype)
                    prev_x_mean, prev_x_std, _, _ = _win_norm_4d(cv_target[j-1], device, dtype)  # del
                    x_mean[0] = prev_x_mean[0]  # Use the mean of
                    x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
                    # x_mean[0] = cv_x0[j][0]
                    # x_std[0] = 4
                    y_win_n = (y_win - x_mean[1:]) / x_std[1:]
                    y_tgt_n = (y_tgt - x_mean) / x_std

                    F_current = SysModel.F.clone().detach().to(device)
                    H         = SysModel.H.to(device)
                    x0_raw    = cv_x0[j].to(device)
                    x0_norm   = (x0_raw.view(m, 1) - x_mean) / x_std
                    prior_P   = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    for em_iter in range(num_em_iters):
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.prior_Sigma = prior_P
                        self.model.init_hidden()


                        x_fwd = torch.stack(
                            [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                            dim=1)
                        xs = [None] * T
                        xs[T - 1] = x_fwd[:, T - 1]
                        self.model.InitBackward(xs[T - 1])
                        xs[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                        for t in range(T - 3, -1, -1):
                            xs[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], xs[t + 2])
                        x_sm = torch.stack(xs, dim=1)

                        x_prev_full = torch.cat([x0_norm, x_sm[:, :-1]], dim=1)
                        x_curr_full = x_sm
                        A1      = (x_curr_full @ x_prev_full.T) / T
                        A2      = (x_prev_full @ x_prev_full.T) / T
                        delta_x = x_curr_full - F_current @ x_prev_full
                        S_delta = (delta_x @ delta_x.T) / T
                        C_delta = (delta_x @ x_prev_full.T) / T
                        nu      = y_win_n - H @ F_current @ x_prev_full
                        S_nu    = (nu @ nu.T) / T

                        z_in = torch.cat([
                            A1.reshape(-1), A2.reshape(-1),
                            S_delta.reshape(-1), S_nu.reshape(-1),
                            C_delta.reshape(-1), F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)

                        dF        = model_mstep(z_in).view(m, m)
                        F_current = F_current + dF

                    # Mirror train objective: final RTS pass with last F_current.
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0_norm.clone(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = prior_P

                    x_fwd2 = torch.stack(
                        [self.model(y_win_n[:, t], None, None, None) for t in range(T)],
                        dim=1)
                    xs2 = [None] * T
                    xs2[T - 1] = x_fwd2[:, T - 1]
                    self.model.InitBackward(xs2[T - 1])
                    xs2[T - 2] = self.model(None, x_fwd2[:, T - 2], x_fwd2[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        xs2[t] = self.model(None, x_fwd2[:, t], x_fwd2[:, t + 1], xs2[t + 2])
                    x_sm2 = torch.stack(xs2, dim=1)

                    # CV loss must match train_emkalmannet_weather final CV loss
                    resid_x_cv = x_sm2[0, :] - y_tgt_n[0, :]
                    cv_l = (resid_x_cv ** 2).mean()
                    cv_loss_sum += cv_l.item()

            cv_avg = cv_loss_sum / self.N_CV

            rts_sched.step(cv_avg)
            m_sched.step(cv_avg)

            if cv_avg < best_cv:
                best_cv = cv_avg
                os.makedirs(os.path.dirname(path_rts_out) or ".", exist_ok=True)
                os.makedirs(os.path.dirname(path_m_out)   or ".", exist_ok=True)
                torch.save(self.model, path_rts_out)
                torch.save(model_mstep,    path_m_out)

            mean_rts_grad = rts_grad_norm_sum / max(self.N_B, 1)
            mean_m_grad = m_grad_norm_sum / max(self.N_B, 1)
            mean_dF = dF_norm_sum / max(dF_norm_count, 1)

            cur_lr_rts = rts_opt.param_groups[0]['lr']
            cur_lr_m = m_opt.param_groups[0]['lr']

            print(
                f"  [JOINT epoch {epoch:03d}] train={train_avg:.4f}  "
                f"cv={cv_avg:.4f}  best={best_cv:.4f}  "
                f"lr_rts={cur_lr_rts:.2e} lr_m={cur_lr_m:.2e}  "
                f"rts_grad_mean={mean_rts_grad:.3e} rts_grad_max={max_rts_grad_norm:.3e}  "
                f"m_grad_mean={mean_m_grad:.3e} m_grad_max={max_m_grad_norm:.3e}  "
                f"rts_clip_hits={rts_clip_hits}/{self.N_B}  "
                f"m_clip_hits={m_clip_hits}/{self.N_B}  "
                f"bad_grads={bad_grad_count}  dF_mean={mean_dF:.3e}"
            )

        print(f"Saved joint RTSNet   to: {path_rts_out}")
        print(f"Saved joint M-Network to: {path_m_out}")

