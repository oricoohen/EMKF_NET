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
                        load_model_path=None, generate_f=False,
                        generate_h=False, train_x0=None, cv_x0=None):
        """
        Train RTSNet smoother on weather windows.

        train_input[i]  : [4, TAU]   observation window
        train_target[i] : [4, TAU]   next-day-aligned window  (y_{t+1} … y_{t+TAU})
        train_x0[i]     : [4]        feature vector one day before the window (no leakage)

        Loss: weighted MSE   sum_t w_t * MSE(H*F*x_smooth_t, y_{t+1})
              + 2 * MSE on y_{T+1}  (double weight for last prediction)
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
                y_win  = train_input[idx]    # [4, TAU]
                y_next = train_target[idx]   # [4, TAU]
                T      = y_win.size(-1)

                y_mean, y_std, _, _ = _win_norm_4d(y_win, device, dtype)
                y_win_n  = (y_win  - y_mean) / y_std
                y_next_n = (y_next - y_mean) / y_std

                # F selection
                F = SysModel.F_train[0].to(device) \
                    if isinstance(SysModel.F_train, list) \
                    else SysModel.F_train.to(device)
                SysModel.F = F
                self.model.update_F(F)
                SysModel.T = T

                # x0: normalize with per-feature stats
                x0_raw  = train_x0[idx].to(device)   # [4]
                x0_norm = (x0_raw.view(m, 1) - y_mean) / y_std  # [4, 1]
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
                x_sm = torch.stack(xs, dim=1)   # [m, T]

                # Loss
                HF = SysModel.H @ F
                w  = torch.arange(1, T + 1, device=device, dtype=dtype)
                w  = w / w.sum()

                loss = torch.tensor(0.0, device=device, dtype=dtype)
                for t in range(T):
                    y_pred_t = HF @ x_sm[:, t]
                    # Loss only on feature 0 (tavg)
                    loss = loss + w[t] * self.loss_fn(y_pred_t[0], y_next_n[0, t])

                # Double weight on last step, feature 0 only
                y_pred_last = HF @ x_sm[:, -1]
                loss = loss + 2.0 * self.loss_fn(y_pred_last[0], y_next_n[0, -1])

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
                cv_batch = torch.empty([self.N_CV], device=device)
                for j in range(self.N_CV):
                    y_win  = cv_input[j]
                    y_next = cv_target[j]
                    T      = y_win.size(-1)

                    y_mean, y_std, _, _ = _win_norm_4d(y_win, device, dtype)
                    y_win_n  = (y_win  - y_mean) / y_std
                    y_next_n = (y_next - y_mean) / y_std

                    F = SysModel.F_valid[0].to(device) \
                        if isinstance(SysModel.F_valid, list) \
                        else SysModel.F_valid.to(device)
                    SysModel.F = F
                    self.model.update_F(F)

                    x0_raw  = cv_x0[j].to(device)
                    x0_norm = (x0_raw.view(m, 1) - y_mean) / y_std
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

                    HF = SysModel.H @ F
                    w  = torch.arange(1, T + 1, device=device, dtype=dtype)
                    w  = w / w.sum()

                    cv_loss = torch.tensor(0.0, device=device, dtype=dtype)
                    for t in range(T):
                        y_pred_t = HF @ x_sm[:, t]
                        # Loss only on feature 0 (tavg)
                        cv_loss  = cv_loss + w[t] * self.loss_fn(y_pred_t[0], y_next_n[0, t])
                    y_pred_last = HF @ x_sm[:, -1]
                    cv_loss = cv_loss + 2.0 * self.loss_fn(y_pred_last[0], y_next_n[0, -1])

                    cv_batch[j] = cv_loss

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
                       load_model_path, generate_f=False,
                       generate_h=False, test_x0=None):
        """
        Test RTSNet: predict next-day tavg (feature row 0) for each test window.

        Returns:
            pred_temps : [N_T]  predicted tavg (°C, denormalized)
            real_temps : [N_T]  true     tavg (°C, denormalized)
            mse, rel_err_mean, sq_err_arr, rel_err_arr
        """
        device = self.device
        dtype  = torch.float32
        m      = SysModel.m
        N_T    = len(test_input)

        self.model = torch.load(load_model_path, weights_only=False,
                                map_location=device).eval()

        pred_temps = torch.empty(N_T, device=device, dtype=dtype)
        real_temps = torch.empty(N_T, device=device, dtype=dtype)
        sq_err     = torch.empty(N_T, device=device, dtype=dtype)
        rel_err    = torch.empty(N_T, device=device, dtype=dtype)

        with torch.no_grad():
            for j in range(N_T):
                y_win  = test_input[j].to(device)    # [4, TAU]
                y_true = test_target[j].to(device)   # [4, TAU]
                T      = y_win.size(-1)

                y_mean, y_std, y_mean0, y_std0 = _win_norm_4d(y_win, device, dtype)
                y_win_n = (y_win - y_mean) / y_std

                F = SysModel.F_test[0].to(device) \
                    if isinstance(SysModel.F_test, list) \
                    else SysModel.F_test.to(device)
                SysModel.F = F
                self.model.update_F(F)

                x0_raw  = test_x0[j].to(device)
                x0_norm = (x0_raw.view(m, 1) - y_mean) / y_std
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

                # Predict y_{T+1} then denorm row 0 (tavg)
                x_last   = x_sm[:, -1].view(m, 1)
                y_pred_n = (SysModel.H @ (F @ x_last))   # [4, 1]
                # denorm: tavg uses row-0 stats
                pred_tavg = (y_pred_n[0, 0] * y_std0 + y_mean0).item()

                # true next-day tavg = last column of y_true, row 0
                true_tavg = (y_true[0, -1] * y_std0 + y_mean0).item() \
                    if False else y_true[0, -1].item()
                # y_true is already in original units (not normalized in dataset)
                true_tavg = test_target[j][0, -1].item()

                pred_temps[j] = pred_tavg
                real_temps[j] = true_tavg
                sq_err[j]     = (pred_tavg - true_tavg) ** 2
                rel_err[j]    = abs(pred_tavg - true_tavg) / (abs(true_tavg) + 1e-9)

        mse          = sq_err.mean()
        rel_err_mean = rel_err.mean()
        print(f"  RTSNet MSE(tavg): {mse.item():.4f}  RelErr: {rel_err_mean.item():.4f}")
        return pred_temps, real_temps, mse, rel_err_mean, sq_err, rel_err

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
                    y_win = train_input[idx].to(device)
                    y_nxt = train_target[idx].to(device)
                    T     = int(y_win.size(-1))

                    y_mean, y_std, _, _ = _win_norm_4d(y_win, device, dtype)
                    y_win_n = (y_win - y_mean) / y_std
                    y_nxt_n = (y_nxt - y_mean) / y_std

                    F_base = SysModel.F_train[0].to(device) \
                        if isinstance(SysModel.F_train, list) \
                        else SysModel.F_train.to(device)
                    H = SysModel.H.to(device)

                    x0_raw  = train_x0[idx].to(device)              # [m]
                    x0_norm = (x0_raw.view(m, 1) - y_mean) / y_std  # [m,1] per-feature norm

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    w = torch.arange(1, T + 1, device=device, dtype=dtype)
                    w = w / (w.sum() + 1e-12)

                    F_current  = F_base.clone()
                    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

                    for em_iter in range(num_em_iters):
                        # Pass F to RTSNet – no detach so RTSNet sees F's grad
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.init_hidden()
                        self.model.prior_Sigma = prior_P

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
                            A1.reshape(-1),
                            A2.reshape(-1),
                            S_delta.reshape(-1),
                            S_nu.reshape(-1),
                            C_delta.reshape(-1),
                            F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)

                        dF        = model_mstep(z_in).view(m, m)
                        dF_norm_sum += dF.detach().norm().item()
                        dF_norm_count += 1
                        F_current = F_current + dF  # no detach – full grad chain

                        reg_F       = lambda_F * (dF ** 2).sum()
                        HF_iter     = H @ F_current

                        # Loss only on feature 0 (tavg)
                        resid       = torch.stack(
                            [(HF_iter @ x_sm[:, t].view(m, 1))[0] - y_nxt_n[0, t]
                             for t in range(T)], dim=1)
                        loss_y_iter = (w * (resid ** 2).mean(dim=0)).sum()
                        iter_loss   = alpha[min(em_iter, len(alpha) - 2)] * (loss_y_iter + reg_F)
                        total_loss  = total_loss + iter_loss

                    # Final RTS pass with last F – no detach so F grad flows into final loss
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
                    xs2[T -  2] = self.model(None, x_fwd2[:, T - 2], x_fwd2[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        xs2[t] = self.model(None, x_fwd2[:, t], x_fwd2[:, t + 1], xs2[t + 2])
                    x_sm2 = torch.stack(xs2, dim=1)  # NO detach

                    HF_final   = H @ F_current

                    # Loss only on feature 0 (tavg)
                    mse_t2     = torch.stack(
                        [(HF_final @ x_sm2[:, t].view(m, 1))[0] - y_nxt_n[0, t]
                         for t in range(T)], dim=1)
                    loss_y2      = (w * (mse_t2 ** 2).mean(dim=0)).sum()

                    resid_last   = (HF_final @ x_sm2[:, -1].view(m, 1))[0] - y_nxt_n[0, -1]
                    loss_last    = (resid_last ** 2).mean()
                    loss_final   = alpha[-1] * (loss_y2 + 2.0 * loss_last)
                    total_loss   = total_loss + loss_final

                    # Optional: F loss computed from the data constraint
                    # F_true is the matrix that satisfies: F_true @ y_nxt[:, -2] = y_nxt[:, -2]
                    if f_loss:
                        y_prev = y_nxt_n[:, -2].view(m, 1)
                        denom = (y_prev.T @ y_prev) + 1e-9
                        F_true = (y_prev @ y_prev.T) / denom  # [m, m]
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
                    y_win = cv_input[j].to(device)
                    y_nxt = cv_target[j].to(device)
                    T     = int(y_win.size(-1))

                    y_mean, y_std, _, _ = _win_norm_4d(y_win, device, dtype)
                    y_win_n = (y_win - y_mean) / y_std
                    y_nxt_n = (y_nxt - y_mean) / y_std

                    F_base = SysModel.F_valid[0].to(device) \
                        if isinstance(SysModel.F_valid, list) \
                        else SysModel.F_valid.to(device)
                    H = SysModel.H.to(device)

                    x0_raw  = cv_x0[j].to(device)
                    x0_norm = (x0_raw.view(m, 1) - y_mean) / y_std

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    w = torch.arange(1, T + 1, device=device, dtype=dtype)
                    w = w / (w.sum() + 1e-12)

                    F_current = F_base.clone()
                    for em_iter in range(num_em_iters):
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.init_hidden()
                        self.model.prior_Sigma = prior_P

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

                    HF_cv      = H @ F_current

                    # Loss only on feature 0 (tavg)
                    resid_cv   = torch.stack(
                        [(HF_cv @ x_sm2[:, t].view(m, 1))[0] - y_nxt_n[0, t]
                         for t in range(T)], dim=1)
                    cv_l       = (w * (resid_cv ** 2).mean(dim=0)).sum()

                    resid_last = (HF_cv @ x_sm2[:, -1].view(m, 1))[0] - y_nxt_n[0, -1]
                    loss_l     = (resid_last ** 2).mean()
                    cv_loss_sum += (cv_l + 2.0 * loss_l).item()

            cv_avg = cv_loss_sum / self.N_CV
            scheduler.step(cv_avg)

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

    # ------------------------------------------------------------------
    # M-step test  (test_mstep_weather)
    # ------------------------------------------------------------------
    def test_mstep_weather(
        self, SysModel, test_input, test_target, test_x0,
        destination_path_RTS, destination_path_M,
        num_em_iters=2, generate_f=False, generate_h=False,
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

        final_F_list = []
        preds_out    = []

        with torch.no_grad():
            for j in range(N_T):
                y_win  = test_input[j].to(device)
                y_next = test_target[j].to(device)
                T      = y_win.size(-1)

                y_mean, y_std, y_mean0, y_std0 = _win_norm_4d(y_win, device, y_win.dtype)
                y_win_n  = (y_win  - y_mean) / y_std
                y_next_n = (y_next - y_mean) / y_std

                F_base = SysModel.F_test[0].to(device) \
                    if isinstance(SysModel.F_test, list) \
                    else SysModel.F_test.to(device)
                H = SysModel.H.to(device)

                x0_raw  = test_x0[j].to(device)
                x0_norm = (x0_raw.view(m, 1) - y_mean) / y_std  # [m,1] per-feature norm

                prior_P = SysModel.m2x_0.clone().detach().to(device) \
                    if hasattr(SysModel, "m2x_0") \
                    else torch.eye(m, device=device, dtype=y_win.dtype)

                F_current = F_base.clone()

                # Track F evolution and MSE per iteration for this sequence
                F_evolution = [F_current.clone()]
                mse_evolution = []

                for em_iter in range(num_em_iters):
                    self.model.update_F(F_current)
                    self.model.InitSequence(x0_norm.clone(), T)
                    self.model.init_hidden()
                    self.model.prior_Sigma = prior_P

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
                    F_evolution.append(F_current.clone())

                    # Compute MSE for this iteration (on y_{T+1})
                    HF_iter = H @ F_current
                    y_pred_iter = (HF_iter @ x_sm[:, -1].view(m, 1)).view(-1)
                    mse_iter = (y_pred_iter[0] - y_next_n[0, -1]) ** 2
                    mse_evolution.append(mse_iter.item())

                    # Accumulate for global statistics
                    mse_per_iter[em_iter] += mse_iter.item()
                    count_per_iter[em_iter] += 1

                # Final prediction
                HF_final  = H @ F_current
                y_pred_n  = (HF_final @ x_sm[:, -1].view(m, 1)).view(-1)
                pred_tavg = y_pred_n[0] * y_std0 + y_mean0
                true_tavg = y_next[0, -1]

                final_F_list.append(F_current.detach().clone())

                mse_final = (y_pred_n[0] - y_next_n[0, -1]) ** 2
                mse_evolution.append(mse_final.item())
                mse_per_iter[em_iter+1] += mse_iter.item()
                count_per_iter[em_iter+1] += 1
                preds_out.append({
                    "seq_index":       j,
                    "y_pred_Tp1":      pred_tavg.unsqueeze(0).detach().cpu(),
                    "y_true_Tp1":      true_tavg.unsqueeze(0).detach().cpu(),
                    "y_pred_Tp1_norm": y_pred_n.detach().cpu(),
                    "y_true_Tp1_norm": y_next_n[:, -1].detach().cpu(),
                })

                # Print F evolution every print_F_every windows
                if j % print_F_every == 0:
                    print(f"  [test window {j:04d}/{N_T}]")
                    for it, (F_evo, mse_evo) in enumerate(zip(F_evolution, mse_evolution)):
                        print(
                            f"    After EM iter {it}: MSE (denorm)={mse_evo:.6e} ({10 * torch.log10(torch.tensor(mse_evo) + 1e-12):.2f} dB)")
                        print(f"      F matrix:\n{F_evo.cpu().numpy()}")

                    # --- NEW COMPARISONS ---
                    # 1. F_real: F * y_next[-2] = y_next[-1]
                    vec_next = y_next_n[:, -1].view(m, 1)  # y_{T+1}
                    vec_prev = y_next_n[:, -2].view(m, 1)  # y_{T} (same as y_win[-1])
                    denom = (vec_prev.T @ vec_prev) + 1e-9
                    F_real = (vec_next @ vec_prev.T) / denom

                    # 2. F_win: F * y_win[-2] = y_win[-1]
                    vec_win_last = y_win_n[:, -1].view(m, 1)  # y_{T}
                    vec_win_prev = y_win_n[:, -2].view(m, 1)  # y_{T-1}
                    denom_win = (vec_win_prev.T @ vec_win_prev) + 1e-9
                    F_win = (vec_win_last @ vec_win_prev.T) / denom_win

                    # Calculate MSE for these Fs on the prediction task (prev -> next)
                    # Use denormalized Tavg (index 0)
                    true_tavg_val_scalar = (vec_next[0, 0] * y_std0 + y_mean0).item()

                    # Pred with F_real (denormalized)
                    pred_real_norm = F_real @ vec_prev
                    pred_real_tavg = (pred_real_norm[0, 0] * y_std0 + y_mean0).item()
                    mse_F_real = (pred_real_tavg - true_tavg_val_scalar) ** 2

                    # Pred with F_win (denormalized)
                    pred_win_norm = F_win @ vec_prev
                    pred_win_tavg = (pred_win_norm[0, 0] * y_std0 + y_mean0).item()
                    mse_F_win = (pred_win_tavg - true_tavg_val_scalar) ** 2

                    print(
                        f"    [Comparison] F_real MSE (1-step denorm): {mse_F_real:.6e} ({10 * torch.log10(torch.tensor(mse_F_real) + 1e-12):.2f} dB)")
                    print(f"      F_real matrix:\n{F_real.cpu().numpy()}")
                    print(
                        f"    [Comparison] F_win  MSE (1-step denorm): {mse_F_win:.6e} ({10 * torch.log10(torch.tensor(mse_F_win) + 1e-12):.2f} dB)")
                    print(f"      F_win matrix:\n{F_win.cpu().numpy()}")




        # Average MSE per iteration (only where computed)
        mean_mse = torch.zeros(num_em_iters+1, device=device)
        for k in range(num_em_iters+1):
            if count_per_iter[k] > 0:
                mean_mse[k] = mse_per_iter[k] / count_per_iter[k]

        mean_mse_db = 10 * torch.log10(mean_mse + 1e-12)

        print("\n[M-Network TEST] Mean MSE per EM iteration (across all test windows):")
        for k in range(num_em_iters+1):
            print(f"  After EM iter {k+1}: MSE={mean_mse[k].item():.6e}  ({mean_mse_db[k].item():.2f} dB)")

        return mean_mse, mean_mse_db, final_F_list, preds_out

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

        best_cv = float("inf")

        for epoch in range(self.N_steps):

            # ---- TRAIN ----
            self.model.train(); model_mstep.train()
            train_loss_sum = 0.0

            for j in range(self.N_B):
                rts_opt.zero_grad(); m_opt.zero_grad()
                batch_loss = torch.tensor(0.0, device=device, dtype=dtype)

                for _ in range(batch_size):
                    idx   = random.randint(0, N_E - 1)
                    y_win = train_input[idx].to(device)
                    y_tgt = train_target[idx].to(device)
                    T     = y_win.size(-1)

                    y_mean, y_std, _, _ = _win_norm_4d(y_win, device, dtype)
                    y_win_n = (y_win - y_mean) / y_std
                    y_tgt_n = (y_tgt - y_mean) / y_std

                    F_base = SysModel.F.clone().detach().to(device)
                    H      = SysModel.H.to(device)

                    x0_raw  = train_x0[idx].to(device)
                    x0_norm = (x0_raw.view(m, 1) - y_mean) / y_std  # [m,1] per-feature norm

                    prior_P = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    w = torch.arange(1, T + 1, device=device, dtype=dtype)
                    w = w / (w.sum() + 1e-12)

                    F_current  = F_base.clone()
                    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

                    for em_iter in range(num_em_iters):
                        # Joint: NO detach on F → grad flows into RTSNet through update_F
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.init_hidden()
                        self.model.prior_Sigma = prior_P

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

                        z_in = torch.cat([
                            A1.reshape(-1),
                            A2.reshape(-1),
                            S_delta.reshape(-1),
                            S_nu.reshape(-1),
                            C_delta.reshape(-1),
                            F_current.detach().reshape(-1),
                        ], dim=0).view(1, -1)

                        dF  = model_mstep(z_in).view(m, m)
                        F_current = F_current + dF

                        reg_F  = lambda_F * (dF ** 2).sum()
                        HF_iter  = H @ F_current

                        # Loss only on feature 0 (tavg)
                        resid  = torch.stack(
                            [(HF_iter @ x_sm[:, t].view(m, 1))[0] - y_tgt_n[0, t]
                             for t in range(T)], dim=1)
                        loss_y_iter = (((w**2) * (resid ** 2).mean(dim=0)).sum()) * 2
                        iter_loss   = alpha[min(em_iter, len(alpha) - 2)] * (loss_y_iter + reg_F)
                        total_loss  = total_loss + iter_loss

                    # Final RTS pass – NO detach, joint grad flows into RTSNet
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
                    x_sm2 = torch.stack(xs2, dim=1)  # NO detach

                    HF_final   = H @ F_current

                    # Loss only on feature 0 (tavg)
                    mse_t2     = torch.stack(
                        [(HF_final @ x_sm2[:, t].view(m, 1))[0] - y_tgt_n[0, t]
                         for t in range(T)], dim=1)
                    loss_y2      = (((w**2) * (mse_t2 ** 2).mean(dim=0)).sum()) * 2

                    resid_last   = (HF_final @ x_sm2[:, -1].view(m, 1))[0] - y_tgt_n[0, -1]
                    loss_last    = (resid_last ** 2).mean()
                    final_loss   = alpha[-1] * (loss_y2 + 2.0 * loss_last)
                    total_loss   = total_loss + final_loss

                    batch_loss = batch_loss + total_loss / batch_size

                batch_loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), clip_grad)
                rts_opt.step(); m_opt.step()
                train_loss_sum += batch_loss.detach().item()

            train_avg = train_loss_sum / self.N_B

            # ---- CV ----
            self.model.eval(); model_mstep.eval()
            cv_loss_sum = 0.0
            with torch.no_grad():
                for j in range(N_CV):
                    y_win = cv_input[j].to(device)
                    y_tgt = cv_target[j].to(device)
                    T     = y_win.size(-1)

                    y_mean, y_std, _, _ = _win_norm_4d(y_win, device, dtype)
                    y_win_n = (y_win - y_mean) / y_std
                    y_tgt_n = (y_tgt - y_mean) / y_std

                    F_current = SysModel.F.clone().detach().to(device)
                    H         = SysModel.H.to(device)
                    x0_raw    = cv_x0[j].to(device)
                    x0_norm   = (x0_raw.view(m, 1) - y_mean) / y_std
                    prior_P   = SysModel.m2x_0.clone().detach().to(device) \
                        if hasattr(SysModel, "m2x_0") \
                        else torch.eye(m, device=device, dtype=dtype)

                    w = torch.arange(1, T + 1, device=device, dtype=dtype)
                    w = w / (w.sum() + 1e-12)

                    for em_iter in range(num_em_iters):
                        self.model.update_F(F_current)
                        self.model.InitSequence(x0_norm.clone(), T)
                        self.model.init_hidden()
                        self.model.prior_Sigma = prior_P

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

                    HF_cv = H @ F_current

                    # Loss only on feature 0 (tavg)
                    resid_cv = torch.stack(
                        [(HF_cv @ x_sm2[:, t].view(m, 1))[0] - y_tgt_n[0, t]
                         for t in range(T)], dim=1)
                    cv_l = (((w**2) * (resid_cv ** 2).mean(dim=0)).sum()) * 2

                    resid_last = (HF_cv @ x_sm2[:, -1].view(m, 1))[0] - y_tgt_n[0, -1]
                    loss_l = (resid_last ** 2).mean()
                    cv_loss_sum += (cv_l + 2.0 * loss_l).item()

            cv_avg = cv_loss_sum / self.N_CV

            if cv_avg < best_cv:
                best_cv = cv_avg
                os.makedirs(os.path.dirname(path_rts_out) or ".", exist_ok=True)
                os.makedirs(os.path.dirname(path_m_out)   or ".", exist_ok=True)
                torch.save(self.model, path_rts_out)
                torch.save(model_mstep,    path_m_out)

            print(f"  [JOINT epoch {epoch:03d}] train={train_avg:.4f}  "
                  f"cv={cv_avg:.4f}  best={best_cv:.4f}")

        print(f"Saved joint RTSNet   to: {path_rts_out}")
        print(f"Saved joint M-Network to: {path_m_out}")

