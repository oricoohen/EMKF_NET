
import copy

import torch
import torch.nn as nn
import time
import random
from Plot import Plot_extended as Plot
from RTSNet.PsmoothNN import PsmoothNN  # Ensure the PsmoothNN class is correctly imported
from RTSNet.PsmoothNN_combined import PsmoothFromPnot, PNotSmoothNN
import torch.nn as nn
from emkf.main_emkf_func_AI import EMKF_F_Mstep
from Simulations.Lorenz_Atractor.parameters import getJacobian
from emkf.AI_M_step import DeltaF_MStepNet
from emkf.AI_M_step_for_h import DeltaH_MStepNet
import math
import os
device =torch.device("cuda")


def NNTrain_4D(self, SysModel,cv_input, cv_target,train_input, train_target,path_results,load_model_path=None,generate_f=True, generate_h=False,
            train_x0=None, cv_x0=None, train_prices=None, cv_prices=None):
    """
    Inputs:
        SysModel:
            Contains the fixed observation model H, initial F (learned/updated in training),
            and dimensions m (state) and n (measurement).

        train_input / cv_input:
            List of sequences (rolling windows) of measurements.
            Each element is a Tensor of shape [n, T] (usually n=1 for a single stock price).
            Example: y_window = [y(t0), ..., y(t0+T-1)].

        train_target / cv_target:
            List of "next-day" target sequences aligned to each window.
            Each element is a Tensor containing the true next measurements:
                y_next = [y(t0+1), ..., y(t0+T)]
            Shape should match the loss indexing:
                - If you use y_next[:, t] for t=0..T-1, then y_next is [n, T]
                - If you store [y(t0), ..., y(t0+T)] (length T+1), then use y_next[:, t+1]

        train_x0 / cv_x0:
            List of scalars (one per window) giving the measurement BEFORE the window:
                y(t0-1)
            Used to build the initial state:
                x0 = [ y(t0-1) , 0.5 ]   (fixed momentum)

    Training objective:
        For each t in 0..T-1, predict the next-day measurement:
            y_pred(t+1|t) = H F x_forward(t)
        and minimize a weighted MSE:
            loss = sum_t w_t * MSE(y_pred(t+1|t), y_true(t+1))
        with increasing weights w_t so the last prediction gets the most weight.
    """
    self.N_E = len(train_input)
    self.N_CV = len(cv_input)


    self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
    self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)

    MSE_train_linear_batch = torch.empty([self.N_B], device=self.device)
    self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
    self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)

    if load_model_path is not None:
        print("loading model_and keep training them")
        self.model = torch.load(load_model_path, map_location=self.device, weights_only=False).to(self.device).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learningRate,
                                          weight_decay=self.weightDecay)

    # Training Mode
    self.model.train()

    ##############
    ### Epochs ###
    ##############
    self.MSE_cv_dB_opt = 1000
    self.MSE_cv_idx_opt = 0
    nan_streak = 0

    for ti in range(0, self.N_steps):

        ###############################
        ### Training Sequence Batch ###
        ###############################
        self.model.train()
        self.optimizer.zero_grad()

        Batch_Optimizing_LOSS_sum = 0

        for j in range(0, self.N_B):

            self.model.init_hidden()
            n_e = random.randint(0, self.N_E - 1)
            y_next_day = train_target[n_e]        # [n, T] where n=4
            y_training = train_input[n_e]         # [n, T] where n=4
            prices = train_prices[n_e]            # [T+1] closing prices

            # =========================================================
            # PER-FEATURE NORMALIZATION (each dimension independently)
            # =========================================================
            n_dim = y_training.shape[0]  # should be 4
            means = y_training.mean(dim=1)  # [4]
            stds = y_training.std(dim=1)    # [4]
            stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)

            y_training_norm = (y_training - means.view(n_dim, 1)) / stds.view(n_dim, 1)
            y_next_day_norm = (y_next_day - means.view(n_dim, 1)) / stds.view(n_dim, 1)

            if generate_f is True:  ####if we train with different f
                index = n_e // 10
                SysModel.F = SysModel.F_train[index]
                self.model.update_F(SysModel.F)
            else:
                # Use the first (and only) F matrix when not varying F
                if isinstance(SysModel.F_train, list):
                    SysModel.F = SysModel.F_train[0]
                else:
                    SysModel.F = SysModel.F_train
                self.model.update_F(SysModel.F)


            SysModel.T = y_training_norm.size()[-1]

            # =========================================================
            # PER-SEQUENCE x0: x0 IS the observation y0 (not [price, 0.5])
            # Normalize using per-feature statistics
            # =========================================================
            y0 = train_x0[n_e]  # [4] observation before window
            y0_norm = (y0 - means) / stds  # per-feature normalization
            x0 = y0_norm.view(SysModel.m, 1)  # [4, 1]
            SysModel.m1x_0 = x0

            # Init Hidden State
            self.model.InitSequence(SysModel.m1x_0, SysModel.T)
            self.model.init_hidden()

            # FIXED: Forward pass - use list comprehension to preserve computation graph
            x_out_training_forward_list = [self.model(y_training_norm[:, t], None, None, None)
                                           for t in range(SysModel.T)]
            x_out_training_forward = torch.stack(x_out_training_forward_list, dim=1)  # [m, T]

            # FIXED: Backward smoothing - use list to preserve computation graph
            x_out_training_list = [None] * SysModel.T
            x_out_training_list[SysModel.T - 1] = x_out_training_forward[:, SysModel.T - 1]
            self.model.InitBackward(x_out_training_list[SysModel.T - 1])

            if SysModel.T >= 2:
                x_out_training_list[SysModel.T - 2] = self.model(None,
                                                                  x_out_training_forward[:, SysModel.T - 2],
                                                                  x_out_training_forward[:, SysModel.T - 1],
                                                                  None)
            for t in range(SysModel.T - 3, -1, -1):
                x_out_training_list[t] = self.model(None,
                                                    x_out_training_forward[:, t],
                                                    x_out_training_forward[:, t + 1],
                                                    x_out_training_list[t + 2])

            x_out_training = torch.stack(x_out_training_list, dim=1)  # [m, T]

            # =========================================================
            # LOSS: Predict ONLY log return (first dimension) for next-day price
            # We minimize MSE on log(C_{t+1}) - log(C_t) prediction
            # =========================================================
            HF = SysModel.H @ SysModel.F  # [4, 4]

            # Predict next observations (normalized) - shape [4, T]
            y_pred_norm_list = [HF @ x_out_training[:, t] for t in range(SysModel.T)]
            y_pred_norm = torch.stack(y_pred_norm_list, dim=1)  # [4, T]

            # Extract ONLY the log return dimension (index 0)
            log_return_pred_norm = y_pred_norm[0, :]  # [T]
            log_return_true_norm = y_next_day_norm[0, :]  # [T]

            # Denormalize log returns
            log_return_pred = log_return_pred_norm * stds[0] + means[0]  # [T]
            log_return_true = log_return_true_norm * stds[0] + means[0]  # [T]

            # Loss ONLY on log return prediction (minimizes price prediction error)
            rtsnet_loss = self.loss_fn(log_return_pred, log_return_true)

            # =========================================================
            # MINI-BATCH: Accumulate loss across all sequences in batch
            # DON'T call backward inside loop!
            # =========================================================
            Batch_Optimizing_LOSS_sum += rtsnet_loss  # keep gradient graph alive
            MSE_train_linear_batch[j] = rtsnet_loss.detach().item()  # log without gradient

        # =========================================================
        # MINI-BATCH: Single backward on accumulated batch loss
        # =========================================================
        Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
        Batch_Optimizing_LOSS_mean.backward()  # single backward for entire batch

        # Gradient check
        bad_grad = False
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                bad_grad = True
                break

        if bad_grad:
            print("NaN/Inf gradients → batch skipped")
            nan_streak += 1
            if nan_streak >= 3:
                print("Stopping training (3 consecutive bad batches).")
                # Save the best model found so far before early exit
                if self.MSE_cv_idx_opt < ti and hasattr(self, 'best_model_state'):
                    os.makedirs(os.path.dirname(path_results) if os.path.dirname(path_results) else '.', exist_ok=True)
                    torch.save(self.best_model_state, path_results)
                    print(f"Saved best model from epoch {self.MSE_cv_idx_opt} to {path_results}")
                return
            self.model.zero_grad(set_to_none=True)
            continue

        nan_streak = 0

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()  # Single update per mini-batch

        # Logging
        self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
        self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

        #################################
        ### Validation Sequence Batch ###
        #################################
        self.model.eval()
        with torch.no_grad():
            MSE_cv_linear_batch = torch.empty([self.N_CV], device=self.device)

            for j in range(0, self.N_CV):
                y_cv = cv_input[j]                    # [n, T_test] where n=4
                y_next_day_cv = cv_target[j]          # [n, T_test]
                prices_cv = cv_prices[j]              # [T_test+1]

                # =========================================================
                # PER-FEATURE NORMALIZATION (same as training)
                # =========================================================
                n_dim = y_cv.shape[0]  # should be 4
                means = y_cv.mean(dim=1)  # [4]
                stds = y_cv.std(dim=1)    # [4]
                stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)

                y_cv_norm = (y_cv - means.view(n_dim, 1)) / stds.view(n_dim, 1)
                y_next_day_cv_norm = (y_next_day_cv - means.view(n_dim, 1)) / stds.view(n_dim, 1)

                SysModel.T_test = y_cv_norm.size()[-1]

                if generate_f is True:  ####if we valid with different f
                    index = j // 10
                    SysModel.F = SysModel.F_valid[index]
                    self.model.update_F(SysModel.F)
                else:
                    # Use the first (and only) F matrix when not varying F
                    if isinstance(SysModel.F_valid, list):
                        SysModel.F = SysModel.F_valid[0]
                    else:
                        SysModel.F = SysModel.F_valid
                    self.model.update_F(SysModel.F)

                if generate_h is True:  ####if we valid with different h
                    index = j // 10
                    SysModel.H = SysModel.H_valid[index]
                    # Note: update_H not available in base RTSNet
                else:
                    # Use the first (and only) H matrix when not varying H
                    if isinstance(SysModel.H_valid, list):
                        SysModel.H = SysModel.H_valid[0]
                    else:
                        SysModel.H = SysModel.H_valid

                # x0 (CV): observation y0 normalized per-feature
                y0_cv = cv_x0[j]  # [4]
                y0_cv_norm = (y0_cv - means) / stds
                x0 = y0_cv_norm.view(SysModel.m, 1)  # [4, 1]

                SysModel.m1x_0 = x0
                self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                self.model.init_hidden()

                # FIXED: Forward pass - use list comprehension (consistency in no_grad context)
                x_out_cv_forward_list = [self.model(y_cv_norm[:, t], None, None, None)
                                         for t in range(SysModel.T_test)]
                x_out_cv_forward = torch.stack(x_out_cv_forward_list, dim=1)  # [m, T_test]

                # FIXED: Backward pass - use list comprehension (SMOOTHING IN CV!)
                x_out_cv_list = [None] * SysModel.T_test
                x_out_cv_list[SysModel.T_test - 1] = x_out_cv_forward[:, SysModel.T_test - 1]
                self.model.InitBackward(x_out_cv_list[SysModel.T_test - 1])

                if SysModel.T_test >= 2:
                    x_out_cv_list[SysModel.T_test - 2] = self.model(None,
                                                                     x_out_cv_forward[:, SysModel.T_test - 2],
                                                                     x_out_cv_forward[:, SysModel.T_test - 1],
                                                                     None)
                for t in range(SysModel.T_test - 3, -1, -1):
                    x_out_cv_list[t] = self.model(None,
                                                  x_out_cv_forward[:, t],
                                                  x_out_cv_forward[:, t + 1],
                                                  x_out_cv_list[t + 2])

                x_out_cv = torch.stack(x_out_cv_list, dim=1)  # [m, T_test]

                # =========================================================
                # CV LOSS: Predict ONLY log return (first dimension) for next-day price
                # We minimize MSE on log(C_{t+1}) - log(C_t) prediction
                # =========================================================
                HF = SysModel.H @ SysModel.F

                # Predict next observations - shape [4, T_test]
                y_pred_norm_list = [HF @ x_out_cv[:, t] for t in range(SysModel.T_test)]
                y_pred_norm = torch.stack(y_pred_norm_list, dim=1)  # [4, T_test]

                # Extract ONLY the log return dimension (index 0)
                log_return_pred_norm = y_pred_norm[0, :]  # [T_test]
                log_return_true_norm = y_next_day_cv_norm[0, :]  # [T_test]

                # Denormalize log returns
                log_return_pred = log_return_pred_norm * stds[0] + means[0]  # [T_test]
                log_return_true = log_return_true_norm * stds[0] + means[0]  # [T_test]

                # Loss ONLY on log return prediction
                cv_loss = self.loss_fn(log_return_pred, log_return_true)
                MSE_cv_linear_batch[j] = cv_loss.item()

            # Average CV
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, path_results)

        ########################
        ### Training Summary ###
        ########################
        print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
              self.MSE_cv_dB_epoch[ti], "[dB]")

        if (ti > 1):
            d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
            d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

        print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch,
            self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

def NNTest_4D(self, SysModel, test_input, test_target, load_model_path,
                       generate_f=False, generate_h=False, test_x0=None, test_prices=None):

    tp = torch.float32
    print("Testing RTSNet (stocks – last step only, forward+backward)")

    self.N_T = len(test_input)

    # Load trained RTSNet
    self.model = torch.load(load_model_path, weights_only=False).eval()

    pred_prices = torch.empty(self.N_T, device=self.device, dtype=tp)
    real_prices = torch.empty(self.N_T, device=self.device, dtype=tp)
    sq_err_arr = torch.empty(self.N_T, device=self.device, dtype=tp)
    rel_err_arr = torch.empty(self.N_T, device=self.device, dtype=tp)
    rel_err_arr_abs = torch.empty(self.N_T, device=self.device, dtype=tp)

    with torch.no_grad():
        for j in range(0, self.N_T):

            # --------------------------------------------------
            # Window + target + prices
            # --------------------------------------------------
            y_win = test_input[j]  # [4, TAU]
            y_true = test_target[j]  # [4] next observation
            prices = test_prices[j]  # [TAU+1] closing prices

            T = y_win.size(-1)
            SysModel.T_test = T

            # --------------------------------------------------
            # Per-feature normalization
            # --------------------------------------------------
            n_dim = y_win.shape[0]  # should be 4
            means = y_win.mean(dim=1)  # [4]
            stds = y_win.std(dim=1)    # [4]
            stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)

            y_win_norm = (y_win - means.view(n_dim, 1)) / stds.view(n_dim, 1)

            # --------------------------------------------------
            # F / H selection (same logic as your codebase)
            # --------------------------------------------------
            if generate_f is True:
                index = j // 10
                SysModel.F = SysModel.F_test[index]
                self.model.update_F(SysModel.F)
            else:
                # Use the first (and only) F matrix when not varying F
                if isinstance(SysModel.F_test, list):
                    SysModel.F = SysModel.F_test[0]
                else:
                    SysModel.F = SysModel.F_test
                self.model.update_F(SysModel.F)

            if generate_h is True:
                index = j // 10
                SysModel.H = SysModel.H_test[index]
                self.model.update_H(SysModel.H)

            # --------------------------------------------------
            # x0: observation y0 normalized per-feature
            # --------------------------------------------------
            y0 = test_x0[j]  # [4]
            y0_norm = (y0 - means) / stds
            x0 = y0_norm.view(SysModel.m, 1)  # [4, 1]
            SysModel.m1x_0 = x0

            # --------------------------------------------------
            # Init sequence
            # --------------------------------------------------
            self.model.InitSequence(SysModel.m1x_0, T)
            self.model.init_hidden()

            # --------------------------------------------------
            # Forward pass
            # ASSUMPTION: T >= 2 always
            # --------------------------------------------------
            x_fwd_list = [self.model(y_win_norm[:, t], None, None, None) for t in range(T)]
            x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]

            # --------------------------------------------------
            # Backward smoothing - ALWAYS smooth
            # --------------------------------------------------
            x_smooth_list = [None] * T
            x_smooth_list[T - 1] = x_fwd[:, T - 1]
            self.model.InitBackward(x_smooth_list[T - 1])
            x_smooth_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
            for t in range(T - 3, -1, -1):
                x_smooth_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_smooth_list[t + 2])

            x_smooth = torch.stack(x_smooth_list, dim=1)  # [m, T]

            # --------------------------------------------------
            # LAST prediction ONLY: y_{T+1} from x_T (smoothed)
            # Predict observation, denormalize, convert to price
            # --------------------------------------------------
            x_last = x_smooth[:, T - 1].view(SysModel.m, 1)  # x_T

            y_pred_norm = (SysModel.H @ (SysModel.F @ x_last)).squeeze()  # [4]
            y_pred = y_pred_norm * stds + means  # [4]

            # Convert log return to price
            last_price = prices[-2]  # C_T (last closing price in window)
            pred_price = last_price * torch.exp(y_pred[0])  # C_{T+1} = C_T * exp(r_{T+1})
            true_price = prices[-1]  # actual C_{T+1}

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------
            pred_prices[j] = pred_price
            real_prices[j] = true_price

            sq_err_arr[j] = (pred_price - true_price) ** 2
            rel_err_arr[j] = (pred_price - true_price) / true_price
            rel_err_arr_abs[j] = abs((pred_price - true_price) / true_price)

    mse_price = torch.mean(sq_err_arr)
    rel_err_mean = torch.mean(rel_err_arr_abs)

    print("MSE(price):", mse_price.item())
    print("Mean relative error:", rel_err_mean.item())

    return (pred_prices, real_prices, mse_price, rel_err_mean, sq_err_arr, rel_err_arr)

# -------------------------

def train_emkalmannet_F_from_price(self,SysModel,cv_input, cv_target, cv_x0,train_input, train_target, train_x0,destination_path_M,destination_path_RTS,
                                       num_em_iters=3,alpha=(0.05, 0.10, 0.85),lambda_F=1,generate_f=False,generate_h=False,use_smoothed=True,clip_grad=1.0,):
    """
    Train an M-step network to estimate/update F using a frozen RTSNet smoother and a price-domain loss.

    Assumptions (consistent with your NNTrain_stocks):
    - Each sample is a window y_win:      train_input[i]  shape [n, T]
    - Each target is next-day aligned:    train_target[i] shape [n, T]
        i.e., train_target[i][:, t] = y(t0 + t + 1)
    - train_x0[i] is y(t0-1) (scalar), and x0 = [normalized_y(t0-1), 0.5]
    - Per-window normalization: (y - mean)/std for both input and target.
    - RTSNet is used ONLY to compute x_forward / x_smooth given current F.
    - M-net predicts ΔF; we update F_current -> F_next and compute y_pred = H * F_next * x_state.
    - Loss = weighted MSE over t plus regularization on ΔF, unrolled for num_em_iters with alpha weights.
    """

    device = self.device
    dtype = train_input[0].dtype
    m = SysModel.m
    n = SysModel.n

    self.N_E = len(train_input)
    self.N_CV = len(cv_input)

    # -------------------------
    # Load & freeze RTSNet
    # FIXED: Use map_location to avoid CPU/CUDA device mismatches
    self.model = torch.load(destination_path_RTS, map_location=device, weights_only=False).to(device).eval()
    for p in self.model.parameters():
        p.requires_grad_(False)

    batch_size = 10
    # M-step model
    model_mstep = self.M_model.train()

    self.MSE_cv_dB_opt = 1e18

    for epoch in range(self.N_steps):

        # =========================
        # TRAIN
        # =========================
        model_mstep.train()
        train_loss_sum = 0.0
        for j in range(self.N_B):
            self.M_optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
            for _ in range(batch_size):
                # sample one window
                idx = random.randint(0, self.N_E - 1)
                y_win = train_input[idx].to(device)       # [n, T]
                y_next = train_target[idx].to(device)     # [n, T]
                T = int(y_win.size(-1))  # FIXED: ensure T is an int

                # per-window normalization (same as NNTrain_stocks)
                y_mean = y_win.mean()
                y_std = y_win.std()
                # FIXED: Use .item() for safe scalar comparison and ensure y_std is tensor on device
                if float(y_std.item()) < 1e-6:
                    y_std = torch.tensor(1.0, device=device, dtype=dtype)

                y_win_n = (y_win - y_mean) / y_std
                y_next_n = (y_next - y_mean) / y_std

                # choose base F/H (usually fixed for stocks)
                if generate_f:
                    f_index = idx // 10
                    F_base = SysModel.F_train[f_index].to(device)
                else:
                    F_base = SysModel.F_train[0].to(device) if isinstance(SysModel.F_train, list) else SysModel.F_train.to(device)

                if generate_h:
                    h_index = idx // 10
                    H = SysModel.H_train[h_index].to(device)
                else:
                    H = SysModel.H_train[0].to(device) if isinstance(SysModel.H_train, list) else SysModel.H.to(device)

                # x0 from pre-window price, normalized
                x0_raw = float(train_x0[idx])
                # FIXED: Use .item() to extract scalar values safely
                x0_norm = (x0_raw - float(y_mean.item())) / float(y_std.item())
                x0 = torch.empty(m, device=device, dtype=dtype)
                x0[0] = torch.tensor(x0_norm, device=device, dtype=dtype)
                x0[1] = torch.tensor(0.5, device=device, dtype=dtype)
                SysModel.m1x_0 = x0.view(m, 1)

                # init covariance prior for RTSNet (as in your code)
                if hasattr(SysModel, "m2x_0"):
                    prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                else:
                    prior_Sigma = torch.eye(m, device=device, dtype=dtype)

                # M-step K times (num_em_iters iterations)
                # ASSUMPTION: T >= 2 always
                F_current = F_base.clone().detach()
                total_loss = torch.tensor(0.0, device=device, dtype=dtype)

                # increasing weights over t
                w = torch.arange(1, T + 1, device=device, dtype=dtype)
                w = w / (w.sum() + 1e-12)

                for em_iter in range(num_em_iters):

                    # --- E-step: smooth x using frozen RTSNet under F_current ---
                    self.model.update_F(F_current)
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    if hasattr(self.model, "prior_Sigma"):
                        self.model.prior_Sigma = prior_Sigma

                    # Forward pass
                    x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
                    x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]

                    # Backward smoothing - ALWAYS smooth
                    x_sm_list = [None] * T
                    x_sm_list[T - 1] = x_fwd[:, T - 1]
                    self.model.InitBackward(x_sm_list[T - 1])
                    x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
                    x_state = torch.stack(x_sm_list, dim=1)  # [m, T]

                    # IMPORTANT: detach RTS output so only M-net learns
                    x_state = x_state.detach()

                    nu = y_win_n - (H @ x_state)  # [n, T]

                    # Compute M-step statistics
                    A1 = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(1, T):
                        A1 += x_state[:, t].view(m, 1) @ x_state[:, t-1].view(1, m)

                    A2 = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(T-1):
                        A2 += x_state[:, t].view(m, 1) @ x_state[:, t].view(1, m)

                    S_delta_x = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(1, T):
                        delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
                        S_delta_x += delta_x.view(m, 1) @ delta_x.view(1, m)
                    S_delta_x = S_delta_x / max(T-1, 1)

                    S_nu = torch.zeros(n, n, device=device, dtype=dtype)
                    for t in range(T):
                        S_nu += nu[:, t].view(n, 1) @ nu[:, t].view(1, n)
                    S_nu = S_nu / T

                    C_delta_x_xminus = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(1, T):
                        delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
                        C_delta_x_xminus += delta_x.view(m, 1) @ x_state[:, t-1].view(1, m)
                    C_delta_x_xminus = C_delta_x_xminus / max(T-1, 1)

                    # Build feature vector
                    feat = torch.cat([
                        A1.reshape(-1), A2.reshape(-1),
                        S_delta_x.reshape(-1), S_nu.reshape(-1),
                        C_delta_x_xminus.reshape(-1), F_current.reshape(-1)
                    ], dim=0).view(1, -1)  # [1, 5*m^2 + n^2]

                    # predict ΔF, update
                    dF = model_mstep(feat).view(m, m)
                    F_next = F_current + dF

                    # Regularize ΔF for ALL EM iterations
                    reg = lambda_F * torch.mean(dF ** 2)

                    # alpha weighting across EM iterations
                    weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                    total_loss = total_loss + weight * reg

                    # advance F
                    F_current = F_next.detach()

                # --- ONE FINAL RTS with final F for prediction ---
                self.model.update_F(F_current)
                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.init_hidden()
                if hasattr(self.model, "prior_Sigma"):
                    self.model.prior_Sigma = prior_Sigma

                x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
                x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]

                # Smooth - ALWAYS
                x_sm_list = [None] * T
                x_sm_list[T - 1] = x_fwd[:, T - 1]
                self.model.InitBackward(x_sm_list[T - 1])
                x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
                x_state_final = torch.stack(x_sm_list, dim=1)  # [m, T]

                x_state_final = x_state_final.detach()

                # --- price prediction loss: y_hat(t+1|t) = H * F_current * x_state_final(:,t) ---
                HF_final = H @ F_current  # [n, m]
                # FIXED: Use list comprehension to preserve gradients
                y_pred_list = [(HF_final @ x_state_final[:, t].view(m, 1)).view(-1) for t in range(T)]
                y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]

                # weighted MSE over t
                mse_t = (y_pred - y_next_n) ** 2  # [n, T]
                loss_y = torch.sum(w.view(1, T) * mse_t.mean(dim=0, keepdim=True))

                # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
                x_last = x_state_final[:, -1].view(m, 1)
                y_pred_Tp1 = (HF_final @ x_last).view(-1)
                y_true_Tp1 = y_next_n[:, -1]
                loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
                loss_y = loss_y + 2.0 * loss_y_Tp1

                total_loss = total_loss + loss_y
                # FIXED: Accumulate with tensor operations
                batch_loss = batch_loss + (total_loss / float(num_em_iters))
            # FIXED: Single backward call per optimizer step with defensive try/except
            loss = batch_loss / float(batch_size)
            try:
                loss.backward()
            except Exception as e:
                print(f"Warning: backward failed at epoch {epoch} with error: {e}; skipping this batch")
                self.M_optimizer.zero_grad()
                continue
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=float(clip_grad))
            self.M_optimizer.step()

            train_loss_sum += loss.detach().item()

        # =========================
        # VALIDATION
        # =========================
        model_mstep.eval()
        cv_loss_sum = 0.0

        with torch.no_grad():
            for j in range(self.N_CV):
                y_win = cv_input[j].to(device)
                y_next = cv_target[j].to(device)
                T = int(y_win.size(-1))  # FIXED: ensure T is an int

                y_mean = y_win.mean()
                y_std = y_win.std()
                # FIXED: Safe scalar comparison
                if float(y_std.item()) < 1e-6:
                    y_std = torch.tensor(1.0, device=device, dtype=dtype)

                y_win_n = (y_win - y_mean) / y_std
                y_next_n = (y_next - y_mean) / y_std

                if generate_f:
                    f_index = j // 10
                    F_base = SysModel.F_valid[f_index].to(device)
                else:
                    F_base = SysModel.F_valid[0].to(device) if isinstance(SysModel.F_valid, list) else SysModel.F_valid.to(device)

                if generate_h:
                    h_index = j // 10
                    H = SysModel.H_valid[h_index].to(device)
                else:
                    H = SysModel.H_valid[0].to(device) if isinstance(SysModel.H_valid, list) else SysModel.H.to(device)

                x0_raw = float(cv_x0[j])
                # FIXED: Safe scalar conversion
                x0_norm = (x0_raw - float(y_mean.item())) / float(y_std.item())
                x0 = torch.empty(m, device=device, dtype=dtype)
                x0[0] = torch.tensor(x0_norm, device=device, dtype=dtype)
                x0[1] = torch.tensor(0.5, device=device, dtype=dtype)
                SysModel.m1x_0 = x0.view(m, 1)

                if hasattr(SysModel, "m2x_0"):
                    prior_Sigma = SysModel.m2x_0.clone().detach().to(device)
                else:
                    prior_Sigma = torch.eye(m, device=device, dtype=dtype)

                w = torch.arange(1, T + 1, device=device, dtype=dtype)
                w = w / (w.sum() + 1e-12)

                F_current = F_base.clone().detach()
                # FIXED: Use tensor accumulator on device
                total_loss = torch.tensor(0.0, device=device, dtype=dtype)

                # M-step runs num_em_iters times
                for em_iter in range(num_em_iters):
                    self.model.update_F(F_current)
                    self.model.InitSequence(SysModel.m1x_0, T)
                    self.model.init_hidden()
                    if hasattr(self.model, "prior_Sigma"):
                        self.model.prior_Sigma = prior_Sigma

                    # Forward pass
                    x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
                    x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]

                    # Backward smoothing - ALWAYS smooth
                    x_sm_list = [None] * T
                    x_sm_list[T - 1] = x_fwd[:, T - 1]
                    self.model.InitBackward(x_sm_list[T - 1])
                    x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                    for t in range(T - 3, -1, -1):
                        x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
                    x_state = torch.stack(x_sm_list, dim=1)  # [m, T]

                    nu = y_win_n - (H @ x_state)

                    # Compute M-step statistics
                    A1 = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(1, T):
                        A1 += x_state[:, t].view(m, 1) @ x_state[:, t-1].view(1, m)

                    A2 = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(T-1):
                        A2 += x_state[:, t].view(m, 1) @ x_state[:, t].view(1, m)

                    S_delta_x = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(1, T):
                        delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
                        S_delta_x += delta_x.view(m, 1) @ delta_x.view(1, m)
                    S_delta_x = S_delta_x / max(T-1, 1)

                    S_nu = torch.zeros(n, n, device=device, dtype=dtype)
                    for t in range(T):
                        S_nu += nu[:, t].view(n, 1) @ nu[:, t].view(1, n)
                    S_nu = S_nu / T

                    C_delta_x_xminus = torch.zeros(m, m, device=device, dtype=dtype)
                    for t in range(1, T):
                        delta_x = x_state[:, t] - (F_current @ x_state[:, t-1])
                        C_delta_x_xminus += delta_x.view(m, 1) @ x_state[:, t-1].view(1, m)
                    C_delta_x_xminus = C_delta_x_xminus / max(T-1, 1)

                    feat = torch.cat([
                        A1.reshape(-1), A2.reshape(-1),
                        S_delta_x.reshape(-1), S_nu.reshape(-1),
                        C_delta_x_xminus.reshape(-1), F_current.reshape(-1),
                    ], dim=0).view(1, -1)

                    dF = model_mstep(feat).view(m, m)
                    F_next = F_current + dF

                    # FIXED: Regularize for ALL EM iters
                    reg = lambda_F * torch.mean(dF ** 2)
                    weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                    total_loss = total_loss + weight * reg

                    F_current = F_next

                # ONE FINAL RTS with final F for prediction
                self.model.update_F(F_current)
                self.model.InitSequence(SysModel.m1x_0, T)
                self.model.init_hidden()
                if hasattr(self.model, "prior_Sigma"):
                    self.model.prior_Sigma = prior_Sigma

                x_fwd_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
                x_fwd = torch.stack(x_fwd_list, dim=1)  # [m, T]

                # Smooth - ALWAYS
                x_sm_list = [None] * T
                x_sm_list[T - 1] = x_fwd[:, T - 1]
                self.model.InitBackward(x_sm_list[T - 1])
                x_sm_list[T - 2] = self.model(None, x_fwd[:, T - 2], x_fwd[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_sm_list[t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_sm_list[t + 2])
                x_state_final = torch.stack(x_sm_list, dim=1)  # [m, T]

                # Prediction with final F
                HF_final = H @ F_current
                # FIXED: Use list comprehension
                y_pred_list = [(HF_final @ x_state_final[:, t].view(m, 1)).view(-1) for t in range(T)]
                y_pred = torch.stack(y_pred_list, dim=1)  # [n, T]

                mse_t = (y_pred - y_next_n) ** 2
                loss_y = torch.sum(w.view(1, T) * mse_t.mean(dim=0, keepdim=True))

                # EXTRA: Predict y_{T+1} from x_T with DOUBLE weight
                x_last = x_state_final[:, -1].view(m, 1)
                y_pred_Tp1 = (HF_final @ x_last).view(-1)
                y_true_Tp1 = y_next_n[:, -1]
                loss_y_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)
                loss_y = loss_y + 2.0 * loss_y_Tp1

                total_loss = total_loss + loss_y

                cv_loss_sum += (total_loss / float(num_em_iters)).item()

        train_epoch = train_loss_sum / max(1, self.N_B)
        cv_epoch = cv_loss_sum / max(1, self.N_CV)

        # FIXED: Safe comparison with float
        if float(cv_epoch) < float(self.MSE_cv_dB_opt):
            self.MSE_cv_dB_opt = float(cv_epoch)
            torch.save(model_mstep, destination_path_M)

        print(f"[F-MNet via RTSNet] epoch={epoch:03d} train={train_epoch:.6f} cv={cv_epoch:.6f} best_cv={self.MSE_cv_dB_opt:.6f}")


def test_mstep_net_price(self,
        SysModel,
        test_input,  # list: each [n, T]
        test_target,  # list: each [n, T]  (next-day aligned, like your NNTrain_stocks)
        test_x0,  # list: scalar y(t0-1) per window (like your stocks pipeline)
        destination_path_RTS,
        destination_path_M,
        num_em_iters=3,
        generate_f=False,
        generate_h=False,
        use_smoothed=True  # True: use RTS smoothing; False: forward only
):
    """
    Test M-step network for STOCK PRICE prediction.
    - Load frozen RTSNet from destination_path_RTS.
    - Load trained M-step net from destination_path_M.
    - For each test window:
        * normalize window once (mean/std of input window)
        * build x0 = [normalized y(t0-1), 0.5]
        * unroll num_em_iters:
            - smooth x with current F (frozen RTSNet)
            - build z_in features
            - predict ΔF, update F
        * after final F, compute y_pred(t+1|t) = H * F_final * x_state(:,t)
        * compute MSE vs test_target (normalized)
    Returns:
      mean_price_mse_per_iter (tensor [num_em_iters])   # how price error evolves across EM iters
      mean_price_mse_db_per_iter (tensor [num_em_iters])
      final_F_list (list of [m,m])
      (optional) predictions (list of dicts)
    """

    device = self.device
    m = SysModel.m
    n = SysModel.n
    N_T = len(test_input)

    # --- Load and freeze RTSNet ---
    self.model = torch.load(destination_path_RTS, weights_only=False).to(device).eval()
    for p in self.model.parameters():
        p.requires_grad_(False)

    # --- Load M-step net ---
    model_mstep = torch.load(destination_path_M, weights_only=False).to(device).eval()

    # Track mean price MSE per EM iteration
    price_mse_sum_per_iter = torch.zeros(num_em_iters, device=device)

    final_F_list = []
    preds_out = []

    with torch.no_grad():
        for j in range(N_T):

            y_win = test_input[j]  # [n, T]  (assumed already on device)
            y_next = test_target[j]  # [n, T]  (next-day aligned)
            T = y_win.size(-1)

            # Choose base F and H
            if generate_f:
                f_index = j // 10
                F_current = SysModel.F_test[f_index].clone()
            else:
                # common in stocks: one global base F
                F_current = SysModel.F_test[0].clone() if isinstance(SysModel.F_test,
                                                                     list) else SysModel.F_test.clone()

            if generate_h:
                h_index = j // 10
                H = SysModel.H_test[h_index].clone()
                SysModel.H = H
                self.model.update_H(H)
            else:
                H = SysModel.H.clone()

            # ---- Normalize ONCE per window ----
            y_mean = y_win.mean()
            y_std = y_win.std()
            if y_std < 1e-6:
                y_std = torch.tensor(1.0, device=device, dtype=y_win.dtype)

            y_win_n = (y_win - y_mean) / y_std
            y_next_n = (y_next - y_mean) / y_std

            # ---- Build x0 from pre-window price ----
            x0_raw = float(test_x0[j])
            x0_norm = (x0_raw - y_mean.item()) / y_std.item()

            x0 = torch.empty(m, device=device, dtype=y_win.dtype)
            x0[0] = torch.tensor(x0_norm, device=device, dtype=y_win.dtype)
            x0[1] = torch.tensor(0.5, device=device, dtype=y_win.dtype)
            x0 = x0.view(m, 1)

            # prior covariance (same style as your code)
            P0 = SysModel.m2x_0.clone().detach()

            # We also record per-iter price MSE for this sequence
            seq_price_mse_per_iter = torch.zeros(num_em_iters, device=device)

            # ========= M-step K times (num_em_iters iterations) =========
            # ASSUMPTION: T >= 2 always
            for em_iter in range(num_em_iters):

                # --- E-step: RTSNet smoothing under current F ---
                self.model.update_F(F_current)
                self.model.InitSequence(x0.clone().detach(), T)
                self.model.init_hidden()
                self.model.prior_Sigma = P0.clone().detach()

                # Forward pass
                x_forward_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
                x_forward = torch.stack(x_forward_list, dim=1)  # [m, T]

                # Backward smoothing - ALWAYS smooth
                x_smooth_list = [None] * T
                x_smooth_list[T - 1] = x_forward[:, T - 1]
                self.model.InitBackward(x_smooth_list[T - 1])
                x_smooth_list[T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smooth_list[t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth_list[t + 2])
                x_state = torch.stack(x_smooth_list, dim=1)  # [m, T]

                # --- Build z_in features for M-step ---
                x_curr = x_state  # [m, T]
                x_prev = torch.cat([x0, x_curr[:, :-1]], dim=1)  # [m, T]

                A1 = (x_curr @ x_prev.T) / T
                A2 = (x_prev @ x_prev.T) / T

                x_minus = F_current @ x_prev
                delta_x = x_curr - x_minus

                delta_mean = delta_x.mean(dim=1, keepdim=True)
                delta_centered = delta_x - delta_mean
                S_delta_x = (delta_centered @ delta_centered.T) / T

                Hx_curr = H @ x_curr
                nu = y_win_n - Hx_curr

                nu_mean = nu.mean(dim=1, keepdim=True)
                nu_centered = nu - nu_mean
                S_nu = (nu_centered @ nu_centered.T) / T

                C_delta_x_xminus = (delta_x @ x_prev.T) / T

                z_in = torch.cat([
                    A1.reshape(-1),
                    A2.reshape(-1),
                    S_delta_x.reshape(-1),
                    S_nu.reshape(-1),
                    C_delta_x_xminus.reshape(-1),
                    F_current.reshape(-1),
                ], dim=0).view(1, -1)

                # --- M-step: predict ΔF and update ---
                dF = model_mstep(z_in).view(m, m)
                F_next = F_current + dF

                # Update F for next EM iteration
                F_current = F_next

            # ========= ONE FINAL RTS pass with final F for prediction =========
            self.model.update_F(F_current)
            self.model.InitSequence(x0.clone().detach(), T)
            self.model.init_hidden()
            self.model.prior_Sigma = P0.clone().detach()

            # Forward
            x_forward_list = [self.model(y_win_n[:, t], None, None, None) for t in range(T)]
            x_forward = torch.stack(x_forward_list, dim=1)  # [m, T]

            # Smooth - ALWAYS
            x_smooth_list = [None] * T
            x_smooth_list[T - 1] = x_forward[:, T - 1]
            self.model.InitBackward(x_smooth_list[T - 1])
            x_smooth_list[T - 2] = self.model(None, x_forward[:, T - 2], x_forward[:, T - 1], None)
            for t in range(T - 3, -1, -1):
                x_smooth_list[t] = self.model(None, x_forward[:, t], x_forward[:, t + 1], x_smooth_list[t + 2])
            x_state_final = torch.stack(x_smooth_list, dim=1)  # [m, T]

            # --- ONLY predict y_{T+1} from x_T (the last smoothed state) ---
            HF_final = H @ F_current  # [n, m]
            x_last = x_state_final[:, -1].view(m, 1)  # x_T
            y_pred_Tp1 = (HF_final @ x_last).view(-1)  # predict y_{T+1}
            y_true_Tp1 = y_next_n[:, -1]  # y_{T+1} (last element of target)

            # MSE on ONLY this prediction (the only one that matters!)
            mse_Tp1 = torch.mean((y_pred_Tp1 - y_true_Tp1) ** 2)

            # Store this as the final MSE for this sequence
            # (For backward compatibility, store in last iter slot)
            seq_price_mse_per_iter = torch.zeros(num_em_iters, device=device)
            seq_price_mse_per_iter[-1] = mse_Tp1  # Only last slot matters

            # accumulate mean over sequences
            price_mse_sum_per_iter += seq_price_mse_per_iter
            final_F_list.append(F_current.detach().clone())


            # optionally return denormalized predictions from FINAL iteration
            preds_out.append({
                "seq_index": j,
                "y_mean": y_mean.detach().cpu(),
                "y_std": y_std.detach().cpu(),
                # Predictions for y_{T+1} (ONLY)
                "y_pred_Tp1_norm": y_pred_Tp1.detach().cpu(),
                "y_true_Tp1_norm": y_true_Tp1.detach().cpu(),
                "y_pred_Tp1": (y_pred_Tp1 * y_std + y_mean).detach().cpu(),
                "y_true_Tp1": (y_true_Tp1 * y_std + y_mean).detach().cpu(),
            })

    mean_price_mse_per_iter = price_mse_sum_per_iter / float(N_T)
    mean_price_mse_db_per_iter = 10.0 * torch.log10(mean_price_mse_per_iter + 1e-12)

    print("[M-step PRICE TEST] Mean price MSE per EM iteration:")
    for k in range(num_em_iters):
        print(f"  EM iter {k + 1}: mse={mean_price_mse_per_iter[k].item():.6e}  "
              f"({mean_price_mse_db_per_iter[k].item():.2f} dB)")


    return mean_price_mse_per_iter, mean_price_mse_db_per_iter, final_F_list, preds_out



# train_joint function removed - has indentation issues
