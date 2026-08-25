"""
Pipeline for RTSNet training and testing on the 2-D TDOA microphone experiment.
Fixed F, fixed nonlinear h — no per-sequence model updates.
"""

import math
import time
import random

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from emkf.AI_M_step_for_f import DeltaF_MStepNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Pipeline_mic:

    def __init__(self, Time, folderName, modelName):
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self):
        torch.save(self, self.folderName + "pipeline_" + self.modelName + ".pt")

    def setssModel(self, ssModel):
        self.SysModel = ssModel

    def setModel(self, model, args):
        self.args  = args
        self.model = model
        self.model.to(self.device)
        self.use_big_mstep_net = getattr(args, "use_big_mstep_net", False)
        self.mstep_hidden_dim = getattr(
            args, "mstep_hidden_dim",
            512 if self.use_big_mstep_net else 256,
        )
        self.M_model = self._new_mstep_model()

    def _new_mstep_model(self, use_big_mstep_net=None, mstep_hidden_dim=None):
        use_big = self.use_big_mstep_net if use_big_mstep_net is None else use_big_mstep_net
        hidden = self.mstep_hidden_dim if mstep_hidden_dim is None else mstep_hidden_dim
        return DeltaF_MStepNet(
            self.SysModel.m,
            self.SysModel.n,
            d_hidden=hidden,
            use_big_mstep_net=use_big,
        ).to(self.device)

    def _load_or_new_mstep_model(self, load_mnet, use_big_mstep_net=None, mstep_hidden_dim=None):
        use_big = self.use_big_mstep_net if use_big_mstep_net is None else use_big_mstep_net
        hidden = self.mstep_hidden_dim if mstep_hidden_dim is None else mstep_hidden_dim

        if load_mnet is not None:
            try:
                loaded = torch.load(load_mnet, weights_only=False, map_location=self.device).to(self.device)
            except FileNotFoundError:
                if not use_big:
                    raise
                print(f"MNet checkpoint not found: {load_mnet}. Starting a fresh BIG MNet.")
                loaded = None

            if loaded is None:
                self.M_model = self._new_mstep_model(use_big, hidden)
                print(f"Created fresh MNet: big={use_big}, hidden={hidden}")
                return self.M_model

            loaded_big = getattr(loaded, "use_big_mstep_net", False)
            loaded_hidden = getattr(loaded, "d_hidden", 256)
            if loaded_big == use_big and loaded_hidden == hidden:
                loaded.dF_scale = 0.1   # bound ΔF (old checkpoints were saved with 0.3)
                print(f"Loading MNet from: {load_mnet}  (dF_scale forced to {loaded.dF_scale})")
                self.M_model = loaded
                return self.M_model

            print(
                "MNet checkpoint architecture does not match requested "
                f"(checkpoint big={loaded_big}, hidden={loaded_hidden}; "
                f"requested big={use_big}, hidden={hidden}). Starting a fresh MNet."
            )

        self.M_model = self._new_mstep_model(use_big, hidden)
        print(f"Created fresh MNet: big={use_big}, hidden={hidden}")
        return self.M_model

    # ------------------------------------------------------------------
    # Batched (vectorized) helpers — B sequences processed in parallel.
    # Verified numerically equivalent to the sequential path by the
    # equivalence harness (Group 0-3, B in {1,8}, float64).
    # ------------------------------------------------------------------
    def _run_smoother_batched(self, y_seq, F_current, x_0, prior_Sigma, obs_mask=None):
        """Batched forward filter + RTS smoother.
        y_seq [B,n,T], F_current [B,m,m], x_0 [B,m,1] -> x_smooth [B,m,T].
        obs_mask [T] bool or None: masked (False) timesteps use pure F-prediction
        (no measurement update), so the smoother must rely on F across the gap."""
        mdl = self.model
        B, mm, T = F_current.shape[0], mdl.m, y_seq.size(-1)
        mdl.update_F(F_current)
        mdl.InitSequence(x_0, T)
        mdl.prior_Sigma = prior_Sigma
        mdl.init_hidden()
        # Build forward states in a list + stack (NOT in-place index writes): a masked
        # step feeds x_fwd[t-1] through bmm, which saves that slice for backward; mutating
        # a shared x_fwd tensor in-place afterwards would bump its version and break autograd.
        x_fwd_cols = []
        for t in range(T):
            if obs_mask is None or obs_mask[t]:
                x_t = mdl(y_seq[:, :, t], None, None, None)                        # [B,m]
            else:
                x_t = torch.bmm(F_current, x_fwd_cols[t - 1].unsqueeze(-1)).squeeze(-1)  # [B,m]
            x_fwd_cols.append(x_t)
        x_fwd = torch.stack(x_fwd_cols, dim=2)                                     # [B,m,T]
        x_smo = torch.empty(B, mm, T, device=self.device)
        x_smo[:, :, T - 1] = x_fwd[:, :, T - 1]
        mdl.InitBackward(x_fwd[:, :, T - 1])
        x_smo[:, :, T - 2] = mdl(None, x_fwd[:, :, T - 2], x_fwd[:, :, T - 1], None)
        for t in range(T - 3, -1, -1):
            x_smo[:, :, t] = mdl(None, x_fwd[:, :, t], x_fwd[:, :, t + 1], x_smo[:, :, t + 2])
        return x_smo

    def _infer_smooth_batched(self, y_seq, F_current, x_0, obs_mask):
        """Batched forward+smoother for INFERENCE (leaves prior_Sigma untouched,
        supports an obs_mask: masked timesteps use pure F-prediction)."""
        mdl = self.model
        B, mm, T = F_current.shape[0], mdl.m, y_seq.size(-1)
        mdl.update_F(F_current)
        mdl.InitSequence(x_0, T)
        mdl.init_hidden()
        x_fwd = torch.empty(B, mm, T, device=self.device)
        for t in range(T):
            if obs_mask is None or obs_mask[t]:
                x_fwd[:, :, t] = mdl(y_seq[:, :, t], None, None, None)
            else:
                x_fwd[:, :, t] = torch.bmm(F_current, x_fwd[:, :, t - 1].unsqueeze(-1)).squeeze(-1)
        x_smo = torch.empty(B, mm, T, device=self.device)
        x_smo[:, :, T - 1] = x_fwd[:, :, T - 1]
        mdl.InitBackward(x_fwd[:, :, T - 1])
        x_smo[:, :, T - 2] = mdl(None, x_fwd[:, :, T - 2], x_fwd[:, :, T - 1], None)
        for t in range(T - 3, -1, -1):
            x_smo[:, :, t] = mdl(None, x_fwd[:, :, t], x_fwd[:, :, t + 1], x_smo[:, :, t + 2])
        return x_smo

    def _mstep_z_batched(self, x_curr, x_0, F_current, y_curr, h, A1_res):
        """Batched M-step sufficient statistics -> z_in [B, D] (detached).
        x_curr [B,m,T], x_0 [B,m,1], F_current [B,m,m], y_curr [B,n,T]."""
        B, mm, T = x_curr.shape[0], self.model.m, x_curr.size(-1)
        tr = lambda z: z.transpose(-1, -2)
        x_prev = torch.empty_like(x_curr)
        x_prev[:, :, 0] = x_0.reshape(B, mm)
        x_prev[:, :, 1:] = x_curr[:, :, :-1]
        A1 = torch.bmm(x_curr, tr(x_prev)) / T
        A2 = torch.bmm(x_prev, tr(x_prev)) / T
        x_minus = torch.bmm(F_current, x_prev)
        delta_x = x_curr - x_minus
        dc = delta_x - delta_x.mean(dim=2, keepdim=True)
        S_delta_x = torch.bmm(dc, tr(dc)) / T
        Hx = torch.cat([h(x_curr[:, :, t:t + 1]) for t in range(T)], dim=2)   # [B,n,T]
        nu = y_curr - Hx
        nu_c = nu - nu.mean(dim=2, keepdim=True)
        S_nu = torch.bmm(nu_c, tr(nu_c)) / T
        C = torch.bmm(delta_x, tr(x_minus)) / T
        A1_input = (A1 - torch.bmm(F_current, A2)) if A1_res else A1
        return torch.cat([
            A1_input.reshape(B, -1).detach(),
            A2.reshape(B, -1).detach(),
            S_delta_x.reshape(B, -1).detach(),
            S_nu.reshape(B, -1).detach(),
            C.reshape(B, -1).detach(),
            F_current.reshape(B, -1).detach(),
        ], dim=1)

    def setTrainingParams(self, args):
        self.N_steps      = args.n_steps
        self.N_B          = args.n_batch
        self.learningRate = args.lr
        self.weightDecay  = args.wd
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weightDecay,
        )
        # RTSNet's long recurrent passes are sensitive to fp16 underflow.
        # Keep full precision unless AMP is explicitly requested by the caller.
        self.use_amp = (
            self.device.type == 'cuda' and getattr(args, 'use_amp', False)
        )
        self.scaler  = GradScaler(self.device.type, enabled=self.use_amp)

    # ─── Training ─────────────────────────────────────────────────────────────
    def NNTrain(self, SysModel, cv_input, cv_target,
                train_input, train_target, path_results,
                load_model_path=None, generate_f=False):

        self.N_E  = len(train_input)
        self.N_CV = len(cv_input)

        MSE_train_linear_batch = torch.empty([self.N_B],   device=self.device)
        MSE_cv_linear_batch    = torch.empty([self.N_CV],  device=self.device)

        self.MSE_cv_linear_epoch    = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch        = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch     = torch.empty([self.N_steps], device=self.device)

        if load_model_path is not None:
            print("Loading model and continuing training ...")
            self.model = torch.load(
                load_model_path, map_location=self.device, weights_only=False
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learningRate,
                weight_decay=self.weightDecay,
            )

        self.model.train()
        self.MSE_cv_dB_opt  = 1000
        self.MSE_cv_idx_opt = 0
        nan_streak = 0

        for ti in range(self.N_steps):

            # ── Training batch ───────────────────────────────────────────────
            self.model.train()
            self.optimizer.zero_grad()
            Batch_LOSS_sum = 0

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                for j in range(self.N_B):
                    n_e        = random.randint(0, self.N_E - 1)
                    if generate_f:
                        index = n_e
                        SysModel.F = SysModel.F_train[index]
                        self.model.update_F(SysModel.F)
                    y_training = train_input[n_e]          # [n, T]
                    SysModel.T = y_training.size(-1)

                    x_out_fwd = torch.empty(
                        SysModel.m, SysModel.T,
                        device=self.device, dtype=y_training.dtype,
                    )
                    x_out = torch.empty(
                        SysModel.m, SysModel.T,
                        device=self.device, dtype=y_training.dtype,
                    )

                    self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                    self.model.init_hidden()

                    # Forward (filter) pass
                    for t in range(SysModel.T):
                        x_out_fwd[:, t] = self.model(y_training[:, t], None, None, None)

                    # Backward (smoother) pass
                    x_out[:, SysModel.T - 1] = x_out_fwd[:, SysModel.T - 1]
                    self.model.InitBackward(x_out[:, SysModel.T - 1])
                    x_out[:, SysModel.T - 2] = self.model(
                        None,
                        x_out_fwd[:, SysModel.T - 2],
                        x_out_fwd[:, SysModel.T - 1],
                        None,
                    )
                    for t in range(SysModel.T - 3, -1, -1):
                        x_out[:, t] = self.model(
                            None,
                            x_out_fwd[:, t],
                            x_out_fwd[:, t + 1],
                            x_out[:, t + 2],
                        )

                    loss = self.loss_fn(x_out, train_target[n_e])
                    Batch_LOSS_sum += loss
                    MSE_train_linear_batch[j] = loss.item()

            Batch_LOSS_mean = Batch_LOSS_sum / self.N_B
            self.scaler.scale(Batch_LOSS_mean).backward()
            self.scaler.unscale_(self.optimizer)

            # NaN / Inf gradient guard
            bad_grad = any(
                p.grad is not None
                and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in self.model.parameters()
            )
            if bad_grad:
                print(f"[Epoch {ti:04d}] NaN/Inf gradients → batch skipped")
                nan_streak += 1
                if nan_streak >= 3:
                    print("Stopping: 3 consecutive bad batches.")
                    break
                self.scaler.update()
                continue
            nan_streak = 0

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti]     = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            # ── Validation ───────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                for j in range(self.N_CV):
                    if generate_f:
                        index = j
                        SysModel.F = SysModel.F_valid[index]
                        self.model.update_F(SysModel.F)
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size(-1)

                    x_cv_fwd = torch.empty(
                        SysModel.m, SysModel.T_test,
                        device=self.device, dtype=y_cv.dtype,
                    )
                    x_cv = torch.empty(
                        SysModel.m, SysModel.T_test,
                        device=self.device, dtype=y_cv.dtype,
                    )

                    self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                    self.model.init_hidden()

                    for t in range(SysModel.T_test):
                        x_cv_fwd[:, t] = self.model(y_cv[:, t], None, None, None)

                    x_cv[:, SysModel.T_test - 1] = x_cv_fwd[:, SysModel.T_test - 1]
                    self.model.InitBackward(x_cv[:, SysModel.T_test - 1])
                    x_cv[:, SysModel.T_test - 2] = self.model(
                        None,
                        x_cv_fwd[:, SysModel.T_test - 2],
                        x_cv_fwd[:, SysModel.T_test - 1],
                        None,
                    )
                    for t in range(SysModel.T_test - 3, -1, -1):
                        x_cv[:, t] = self.model(
                            None,
                            x_cv_fwd[:, t],
                            x_cv_fwd[:, t + 1],
                            x_cv[:, t + 2],
                        )

                    MSE_cv_linear_batch[j] = self.loss_fn(x_cv, cv_target[j]).item()

                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti]     = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt  = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, path_results)

            print(
                ti,
                "MSE Training :", self.MSE_train_dB_epoch[ti].item(), "[dB]",
                "MSE Validation :", self.MSE_cv_dB_epoch[ti].item(), "[dB]",
            )
            if ti > 1:
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv    = self.MSE_cv_dB_epoch[ti]    - self.MSE_cv_dB_epoch[ti - 1]
                print(
                    "  diff MSE Training :", d_train.item(), "[dB]",
                    "  diff MSE Validation :", d_cv.item(), "[dB]",
                )
            print(
                "  Optimal idx:", self.MSE_cv_idx_opt,
                "Optimal :", self.MSE_cv_dB_opt.item()
                if hasattr(self.MSE_cv_dB_opt, "item") else self.MSE_cv_dB_opt,
                "[dB]",
            )

        return [
            self.MSE_cv_linear_epoch,
            self.MSE_cv_dB_epoch,
            self.MSE_train_linear_epoch,
            self.MSE_train_dB_epoch,
        ]

    # ─── Testing ──────────────────────────────────────────────────────────────
    def NNTest(self, SysModel, test_input, test_target, load_model_path,
               generate_f=False):
        print("Testing RTSNet (mic) ...")
        self.N_T = len(test_input)

        self.MSE_test_linear_arr = torch.empty([self.N_T], device=self.device)
        loss_fn = nn.MSELoss(reduction='mean')

        self.model = torch.load(
            load_model_path, map_location=self.device, weights_only=False
        )
        self.model.to(self.device).eval()

        x_out_list = []
        start = time.time()

        with torch.no_grad():
            for j in range(self.N_T):
                if generate_f:
                    index = j
                    SysModel.F = SysModel.F_test[index]
                    self.model.update_F(SysModel.F)
                y_tst = test_input[j]
                SysModel.T_test = y_tst.size(-1)

                x_fwd = torch.empty(SysModel.m, SysModel.T_test, device=self.device)
                x_out = torch.empty(SysModel.m, SysModel.T_test, device=self.device)

                self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                self.model.init_hidden()

                for t in range(SysModel.T_test):
                    x_fwd[:, t] = self.model(y_tst[:, t], None, None, None)

                x_out[:, SysModel.T_test - 1] = x_fwd[:, SysModel.T_test - 1]
                self.model.InitBackward(x_out[:, SysModel.T_test - 1])
                x_out[:, SysModel.T_test - 2] = self.model(
                    None,
                    x_fwd[:, SysModel.T_test - 2],
                    x_fwd[:, SysModel.T_test - 1],
                    None,
                )
                for t in range(SysModel.T_test - 3, -1, -1):
                    x_out[:, t] = self.model(
                        None,
                        x_fwd[:, t],
                        x_fwd[:, t + 1],
                        x_out[:, t + 2],
                    )

                self.MSE_test_linear_arr[j] = loss_fn(x_out, test_target[j]).item()
                x_out_list.append(x_out)

        end = time.time()
        t = end - start

        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg     = 10 * torch.log10(self.MSE_test_linear_avg)
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.test_std_dB = (
            10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg)
            - self.MSE_test_dB_avg
        )

        print(self.modelName + " MSE Test:", self.MSE_test_dB_avg.item(), "[dB]")
        print(self.modelName + " STD Test:", self.test_std_dB.item(), "[dB]")
        print("Inference Time:", t)

        return [
            self.MSE_test_linear_arr,
            self.MSE_test_linear_avg,
            self.MSE_test_dB_avg,
            torch.stack(x_out_list),
            t,
        ]

    def NNTest_3_datasets(self, SysModel, all_test_inputs, all_test_targets,
                          load_model_path, generate_f=True, datasets=3, obs_mask=None):
        """
        Test RTSNet across `datasets` sequential datasets.
        x_0 carries from the last smoothed state of dataset k into dataset k+1.
        SysModel.F_test must be structured as [datasets][group_idx].
        Outputs are ordered [data * N_T + j]: dataset 0 first, then dataset 1, etc.
        """
        N_T = len(all_test_inputs[0])
        m   = SysModel.m

        self.model = torch.load(load_model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device).eval()

        loss_fn = nn.MSELoss(reduction='mean')
        MSE_arr  = torch.empty(N_T * datasets, device=self.device)

        # ---- Vectorized inference: all N_T sequences as one batch ----
        B = N_T
        idx = list(range(B))
        x_smo_per_ds = []   # [datasets] of [B,m,T]

        start = time.time()
        with torch.no_grad():
            x_0 = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(B, m, 1).contiguous()
            for data in range(datasets):
                y_seq  = torch.stack([all_test_inputs[data][j] for j in idx]).to(self.device)    # [B,n,T]
                x_true = torch.stack([all_test_targets[data][j] for j in idx]).to(self.device)   # [B,m,T]
                T = y_seq.size(-1)
                F_k = torch.stack([SysModel.F_test[data][j] for j in idx]).to(self.device)       # [B,m,m]

                self.model.update_F(F_k)
                self.model.InitSequence(x_0, T)
                self.model.init_hidden()

                x_fwd = torch.empty(B, m, T, device=self.device)
                for t in range(T):
                    if obs_mask is None or obs_mask[t]:
                        x_fwd[:, :, t] = self.model(y_seq[:, :, t], None, None, None)
                    else:
                        x_fwd[:, :, t] = torch.bmm(F_k, x_fwd[:, :, t - 1].unsqueeze(-1)).squeeze(-1)

                x_smo = torch.empty(B, m, T, device=self.device)
                x_smo[:, :, T - 1] = x_fwd[:, :, T - 1]
                self.model.InitBackward(x_fwd[:, :, T - 1])
                x_smo[:, :, T - 2] = self.model(None, x_fwd[:, :, T - 2], x_fwd[:, :, T - 1], None)
                for t in range(T - 3, -1, -1):
                    x_smo[:, :, t] = self.model(None, x_fwd[:, :, t], x_fwd[:, :, t + 1], x_smo[:, :, t + 2])

                MSE_arr[data * N_T:(data + 1) * N_T] = ((x_smo - x_true) ** 2).mean(dim=(1, 2))
                x_smo_per_ds.append(x_smo)
                x_0 = x_smo[:, :, -1:].detach()

        # x_out ordered [j * datasets + data] (seq-major), matching the original loop
        x_out_list = [x_smo_per_ds[data][j] for j in range(N_T) for data in range(datasets)]

        end = time.time()
        t = end - start

        MSE_avg    = torch.mean(MSE_arr)
        MSE_dB_avg = 10 * torch.log10(MSE_avg)
        MSE_std    = torch.std(MSE_arr, unbiased=True)
        std_dB     = 10 * torch.log10(MSE_std + MSE_avg) - MSE_dB_avg

        print(self.modelName + f" MSE Test ({datasets}-datasets):", MSE_dB_avg.item(), "[dB]")
        print(self.modelName + f" STD Test ({datasets}-datasets):", std_dB.item(), "[dB]")
        print("Inference Time:", t)
        for data in range(datasets):
            ds_mse = torch.mean(MSE_arr[data * N_T : (data + 1) * N_T])
            print(f"  Dataset {data}: {10 * torch.log10(ds_mse).item():.2f} dB")

        return [MSE_arr, MSE_avg, MSE_dB_avg, torch.stack(x_out_list), t]

    def train_RTS_net_3_datasets(self, SysModel, cv_input, cv_target,
                                train_input, train_target,
                                destination_path_RTS, load_path_RTS,
                                generate_f=True, datasets=3, obs_mask=None):
        """
        Fine-tune RTSNet on `datasets` independent datasets loaded from load_path_RTS.
        train_input[k]          → [N_E, n, T]   for dataset k
        SysModel.F_train[k][g]  → F for dataset k, group g
        Each dataset's sequences start from SysModel.m1x_0 (independent, no chaining).
        Loss is the average MSE across all datasets.
        """
        self.N_E  = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m

        self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        self.MSE_cv_dB_opt  = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            self.model.train()
            optimizer.zero_grad()
            batch_loss = 0.0
            ds_train_loss = [0.0] * datasets

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # ---- Vectorized mini-batch: B sequences in parallel ----
                B = self.N_B
                idx = [random.randint(0, self.N_E - 1) for _ in range(B)]
                x_0 = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(B, m, 1).contiguous()
                prior_Sigma = SysModel.m2x_0.clone().detach().to(self.device)
                sample_loss = 0.0

                for data in range(datasets):
                    y_seq = torch.stack([train_input[data][i] for i in idx]).to(self.device)        # [B,n,T]
                    x_true_seq = torch.stack([train_target[data][i] for i in idx]).to(self.device)  # [B,m,T]
                    F_k = torch.stack([SysModel.F_train[data][i] for i in idx]).to(self.device)     # [B,m,m]
                    x_smo = self._run_smoother_batched(y_seq, F_k, x_0, prior_Sigma, obs_mask)
                    seq_loss = torch.mean((x_smo - x_true_seq) ** 2)
                    sample_loss = sample_loss + seq_loss
                    ds_train_loss[data] += seq_loss.item()
                    x_0 = x_smo[:, :, -1:].detach()

                batch_loss = sample_loss / datasets

            loss = batch_loss   # already mean over the B-sequence batch
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)

            bad_grad = (not torch.isfinite(loss)) or any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in self.model.parameters()
            )
            if bad_grad:
                print(f"[RTSNet-3ds epoch {epoch:03d}] NaN/Inf loss or gradients → step skipped")
                optimizer.zero_grad()
                self.scaler.update()
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.scaler.step(optimizer)
            self.scaler.update()

            # Validation
            self.model.eval()
            cv_loss_sum = 0.0
            ds_cv_loss = [0.0] * datasets
            with torch.no_grad():
                # ---- Vectorized CV: all N_CV sequences as one batch ----
                Bc = self.N_CV
                idxc = list(range(Bc))
                x_0_cv = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(Bc, m, 1).contiguous()
                prior_Sigma_cv = SysModel.m2x_0.clone().detach().to(self.device)
                sample_loss_cv = 0.0

                for data in range(datasets):
                    y_cv = torch.stack([cv_input[data][j] for j in idxc]).to(self.device)
                    x_true_cv = torch.stack([cv_target[data][j] for j in idxc]).to(self.device)
                    F_k_cv = torch.stack([SysModel.F_valid[data][j] for j in idxc]).to(self.device)
                    x_s_cv = self._run_smoother_batched(y_cv, F_k_cv, x_0_cv, prior_Sigma_cv, obs_mask)
                    seq_loss_cv = torch.mean((x_s_cv - x_true_cv) ** 2).item()
                    sample_loss_cv += seq_loss_cv
                    ds_cv_loss[data] += seq_loss_cv
                    x_0_cv = x_s_cv[:, :, -1:].detach()

                cv_loss_sum = sample_loss_cv / datasets

            cv_epoch = cv_loss_sum   # already averaged over the N_CV batch
            cv_dB    = 10 * math.log10(cv_epoch) if cv_epoch > 0 else float('inf')

            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(cv_epoch)
            curr_lr = optimizer.param_groups[0]['lr']
            lr_str  = f"  lr={curr_lr:.2e}" if curr_lr != prev_lr else ""

            train_dB  = 10 * math.log10(max(loss.item(), 1e-10))
            ds_tr_str = "  ".join(
                f"ds{d}={10*math.log10(max(ds_train_loss[d],1e-10)):.2f}dB" for d in range(datasets)
            )
            ds_cv_str = "  ".join(
                f"ds{d}={10*math.log10(max(ds_cv_loss[d],1e-10)):.2f}dB" for d in range(datasets)
            )
            print(
                f"[epoch {epoch:03d}] train={train_dB:.4f}dB  "
                f"cv_dB={cv_dB:.4f}dB  grad_norm={grad_norm.item():.3e}{lr_str}"
            )
            print(f"  train/ds: {ds_tr_str}")
            print(f"  cv/ds:    {ds_cv_str}")

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt  = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(self.model, destination_path_RTS)
                print(f"  → saved best model at epoch {epoch}")

    # ─── F M-step Training ────────────────────────────────────────────────────
    def train_F_mstep_net(self, SysModel, cv_input, cv_target,
                          train_input, train_target,
                          destination_path_M, destination_path_RTS,
                          load_destination_path_M=None, num_em_iters=2,
                          alpha=(0.3, 1.0, 0.85), lambda_F=1e-3, generate_f=True,
                          A1_res=False):
        """
        M-step training: freeze RTSNet, train DeltaF_MStepNet to update F.
        Statistics [A1, A2, S_delta_x, S_nu_y, C_delta_x_xminus, F_current]:
          x_prev[:,0]=x_0, x_prev[:,1:]=x_smooth[:,:-1]  → shape [m,T]
          A1 = x_smooth @ x_prev.T / T
          A2 = x_prev @ x_prev.T / T
          x_minus = F @ x_prev;  delta_x = x_smooth - x_minus  (centered for S)
          S_delta_x = cov(delta_x);  C = delta_x @ x_minus.T / T
          S_nu_y = centered measurement-residual covariance  [n,n]
        """
        self.N_E  = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load and freeze RTSNet
        self.model = torch.load(destination_path_RTS, map_location=self.device, weights_only=False)
        self.model.to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model and optimizer
        model_mstep = self.M_model.train()
        if load_destination_path_M is not None:
            print(f"Loading F M-step model from: {load_destination_path_M}")
            model_mstep = torch.load(load_destination_path_M, map_location=self.device, weights_only=False)
            self.M_model = model_mstep
            model_mstep.train()
        M_optimizer = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt  = 1000.0
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ── Training ────────────────────────────────────────────────────
            model_mstep.train()
            M_optimizer.zero_grad()
            total_loss_batch = 0.0

            x_loss_first_epoch = 0.0
            f_loss_first_epoch = 0.0
            x_loss_em_epoch = [0.0] * num_em_iters
            f_loss_em_epoch = [0.0] * num_em_iters

            for _ in range(self.N_B):
                n_e    = random.randint(0, self.N_E - 1)
                y_seq  = train_input[n_e]   # [n, T]
                x_true = train_target[n_e]  # [m, T]
                T_seq  = y_seq.size(-1)

                if generate_f:
                    f_index = n_e
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                F_current = F_base.clone()

                # ── Initial E-step ───────────────────────────────────────────
                self.model.update_F(F_current)
                self.model.InitSequence(SysModel.m1x_0, T_seq)
                self.model.init_hidden()

                x_fwd = torch.empty(m, T_seq, device=self.device)
                x_s   = torch.empty(m, T_seq, device=self.device)

                for t in range(T_seq):
                    x_fwd[:, t] = self.model(y_seq[:, t], None, None, None)
                x_s[:, T_seq - 1] = x_fwd[:, T_seq - 1]
                self.model.InitBackward(x_s[:, T_seq - 1])
                x_s[:, T_seq - 2] = self.model(None, x_fwd[:, T_seq - 2], x_fwd[:, T_seq - 1], None)
                for t in range(T_seq - 3, -1, -1):
                    x_s[:, t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_s[:, t + 2])

                x_loss_first_epoch += torch.mean((x_s - x_true) ** 2).item()
                f_loss_first_epoch += torch.mean((F_current - F_true) ** 2).item()

                total_loss_seq = 0.0

                for em_iter in range(num_em_iters):
                    # ── Sufficient statistics ────────────────────────────────
                    # x_prev: x_0 in col 0, x_{t-1} in cols 1..T-1  →  [m, T]
                    x_prev = torch.empty_like(x_s)
                    x_prev[:, 0]  = SysModel.m1x_0.view(-1)
                    x_prev[:, 1:] = x_s[:, :-1]

                    A1 = (x_s @ x_prev.T) / T_seq            # [m, m]
                    A2 = (x_prev @ x_prev.T) / T_seq         # [m, m]

                    x_minus  = F_current @ x_prev             # [m, T]
                    delta_x  = x_s - x_minus                  # [m, T]
                    delta_c  = delta_x - delta_x.mean(dim=1, keepdim=True)
                    S_delta_x        = (delta_c @ delta_c.T) / T_seq       # [m, m]
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T_seq      # [m, m]

                    nu_y = torch.stack(
                        [y_seq[:, t] - SysModel.h(x_s[:, t]) for t in range(T_seq)], dim=1
                    )                                                       # [n, T]
                    nu_c   = nu_y - nu_y.mean(dim=1, keepdim=True)
                    S_nu_y = (nu_c @ nu_c.T) / T_seq                       # [n, n]

                    A1_input = (A1 - F_current @ A2) if A1_res else A1
                    z_in = torch.cat([
                        A1_input.detach().reshape(-1),
                        A2.detach().reshape(-1),
                        S_delta_x.detach().reshape(-1),
                        S_nu_y.detach().reshape(-1),
                        C_delta_x_xminus.detach().reshape(-1),
                        F_current.detach().reshape(-1),
                    ], dim=0).reshape(1, -1)                               # [1, 5m²+n²]

                    # ── M-step forward ───────────────────────────────────────
                    deltaF     = model_mstep(z_in)          # [1, m, m]
                    deltaF_vv  = deltaF.squeeze(0)[2:4, 2:4]   # velocity block only
                    F_next     = F_current.clone()
                    F_next[2:4, 2:4] = F_current[2:4, 2:4] + deltaF_vv

                    f_loss = torch.mean((F_next[2:4, 2:4] - F_true[2:4, 2:4]) ** 2)
                    reg    = lambda_F * torch.mean(deltaF_vv ** 2)
                    f_loss_em_epoch[em_iter] += f_loss.item()

                    F_current = F_next

                    # ── E-step with updated F ────────────────────────────────
                    self.model.update_F(F_current)
                    self.model.InitSequence(SysModel.m1x_0, T_seq)
                    self.model.init_hidden()

                    x_fwd = torch.empty(m, T_seq, device=self.device)
                    x_s   = torch.empty(m, T_seq, device=self.device)

                    for t in range(T_seq):
                        x_fwd[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_s[:, T_seq - 1] = x_fwd[:, T_seq - 1]
                    self.model.InitBackward(x_s[:, T_seq - 1])
                    x_s[:, T_seq - 2] = self.model(None, x_fwd[:, T_seq - 2], x_fwd[:, T_seq - 1], None)
                    for t in range(T_seq - 3, -1, -1):
                        x_s[:, t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_s[:, t + 2])

                    x_loss = torch.mean((x_s - x_true) ** 2)
                    x_loss_em_epoch[em_iter] += x_loss.item()

                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]
                    total_loss_seq += weight * (30*f_loss + reg + 1* x_loss)

                total_loss_batch += total_loss_seq

            total_loss_batch = total_loss_batch / float(self.N_B)
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            M_optimizer.step()

            _xl0 = x_loss_first_epoch / self.N_B
            _fl0 = f_loss_first_epoch / self.N_B
            print(f"[epoch {epoch}] FIRST RTS: "
                  f"x_loss={10*math.log10(max(_xl0,1e-10)):.2f} dB  "
                  f"F_loss={10*math.log10(max(_fl0,1e-10)):.2f} dB")
            for em_iter in range(num_em_iters):
                _xl = x_loss_em_epoch[em_iter] / self.N_B
                _fl = f_loss_em_epoch[em_iter]  / self.N_B
                print(f"[epoch {epoch}] MSTEP {em_iter}: "
                      f"x_loss={10*math.log10(max(_xl,1e-10)):.2f} dB  "
                      f"F_loss={10*math.log10(max(_fl,1e-10)):.2f} dB")

            # ── Validation ──────────────────────────────────────────────────
            model_mstep.eval()
            with torch.no_grad():
                total_loss_cv = 0.0
                f_loss_cv_per_iter = [0.0] * num_em_iters  # accumulate CV F_loss per EM iter

                for j in range(self.N_CV):
                    y_cv      = cv_input[j]   # [n, T]
                    x_true_cv = cv_target[j]  # [m, T]
                    T_cv      = y_cv.size(-1)

                    if generate_f:
                        f_idx_cv  = j
                        F_base_cv = SysModel.F_valid[f_idx_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_idx_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()

                    self.model.update_F(F_current_cv)
                    self.model.InitSequence(SysModel.m1x_0, T_cv)
                    self.model.init_hidden()

                    x_f_cv = torch.empty(m, T_cv, device=self.device)
                    x_s_cv = torch.empty(m, T_cv, device=self.device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)
                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    total_loss_seq_cv = 0.0

                    for em_iter in range(num_em_iters):
                        x_prev_cv = torch.empty_like(x_s_cv)
                        x_prev_cv[:, 0]  = SysModel.m1x_0.view(-1)
                        x_prev_cv[:, 1:] = x_s_cv[:, :-1]

                        A1_cv = (x_s_cv @ x_prev_cv.T) / T_cv
                        A2_cv = (x_prev_cv @ x_prev_cv.T) / T_cv

                        x_minus_cv  = F_current_cv @ x_prev_cv
                        delta_x_cv  = x_s_cv - x_minus_cv
                        delta_c_cv  = delta_x_cv - delta_x_cv.mean(dim=1, keepdim=True)
                        S_delta_x_cv        = (delta_c_cv @ delta_c_cv.T) / T_cv
                        C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                        nu_y_cv = torch.stack(
                            [y_cv[:, t] - SysModel.h(x_s_cv[:, t]) for t in range(T_cv)], dim=1
                        )
                        nu_c_cv   = nu_y_cv - nu_y_cv.mean(dim=1, keepdim=True)
                        S_nu_y_cv = (nu_c_cv @ nu_c_cv.T) / T_cv

                        A1_cv_input = (A1_cv - F_current_cv @ A2_cv) if A1_res else A1_cv
                        z_cv = torch.cat([
                            A1_cv_input.reshape(-1),
                            A2_cv.reshape(-1),
                            S_delta_x_cv.reshape(-1),
                            S_nu_y_cv.reshape(-1),
                            C_delta_x_xminus_cv.reshape(-1),
                            F_current_cv.reshape(-1),
                        ], dim=0).reshape(1, -1)

                        dF_cv     = model_mstep(z_cv)
                        dF_mat_cv = dF_cv.squeeze(0)
                        F_next_cv = F_current_cv.clone()
                        F_next_cv[2:4, 2:4] = F_current_cv[2:4, 2:4] + dF_mat_cv[2:4, 2:4]

                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)
                        reg_cv    = lambda_F * torch.mean(dF_mat_cv ** 2)
                        f_loss_cv_per_iter[em_iter] += f_loss_cv.item()
                        F_current_cv = F_next_cv.clone()

                        self.model.update_F(F_current_cv)
                        self.model.InitSequence(SysModel.m1x_0, T_cv)
                        self.model.init_hidden()

                        x_f_cv = torch.empty(m, T_cv, device=self.device)
                        x_s_cv = torch.empty(m, T_cv, device=self.device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)
                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        x_loss_cv = torch.mean((x_s_cv - x_true_cv) ** 2)
                        if em_iter == 0:
                            weight_cv = alpha[0]
                        elif em_iter == 1:
                            weight_cv = alpha[1]
                        else:
                            weight_cv = alpha[2]
                        total_loss_seq_cv += weight_cv * (30 * f_loss_cv.item() + 1*x_loss_cv.item()
                                                          + reg_cv.item())

                    total_loss_cv += total_loss_seq_cv

            cv_epoch = total_loss_cv / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt  = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)

            print(f"[M-step epoch {epoch}] train={total_loss_batch.item():.6f}  "
                  f"cv={cv_epoch:.6f}  best_cv={self.MSE_cv_dB_opt:.6f} (epoch {self.MSE_cv_idx_opt})")
            for _ei in range(num_em_iters):
                _fl_cv = f_loss_cv_per_iter[_ei] / max(1, self.N_CV)
                print(f"  CV MSTEP {_ei}: F_loss={_fl_cv:.6e} ({10*math.log10(max(_fl_cv,1e-10)):.2f} dB)")

    # ─── Joint Training (RTSNet + MNet together) ──────────────────────────────
    def train_joint_F_mstep_net(self, SysModel, cv_input, cv_target,
                                 train_input, train_target,
                                 destination_path_M, destination_path_RTS,
                                 load_destination_path_RTS=None,
                                 load_destination_path_M=None,
                                 num_em_iters=2,
                                 alpha=(0.3, 1.0, 0.85), lambda_F=1e-3, generate_f=True,
                                 A1_res=False):
        """
        Joint training: RTSNet and DeltaF_MStepNet are both updated.
        RTSNet is NOT frozen — gradients flow through both networks.
        Two separate Adam optimizers, one per network.
        Saves best RTSNet to destination_path_RTS and best MNet to destination_path_M.
        """
        self.N_E  = len(train_input)
        self.N_CV = len(cv_input)
        m = SysModel.m

        # Load RTSNet — trainable
        if load_destination_path_RTS is not None:
            print(f"Loading RTSNet from: {load_destination_path_RTS}")
            self.model = torch.load(load_destination_path_RTS, map_location=self.device, weights_only=False)
        self.model.to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        # Load MNet — trainable
        model_mstep = self.M_model
        if load_destination_path_M is not None:
            print(f"Loading MNet from: {load_destination_path_M}")
            model_mstep = torch.load(load_destination_path_M, map_location=self.device, weights_only=False)
            self.M_model = model_mstep
        model_mstep.to(self.device).train()

        rts_optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learningRate, weight_decay=self.weightDecay)
        M_optimizer   = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt  = 1000.0
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            # ── Training ────────────────────────────────────────────────────
            self.model.train()
            model_mstep.train()
            rts_optimizer.zero_grad()
            M_optimizer.zero_grad()
            total_loss_batch = 0.0

            x_loss_first_epoch = 0.0
            f_loss_first_epoch = 0.0
            x_loss_em_epoch = [0.0] * num_em_iters
            f_loss_em_epoch = [0.0] * num_em_iters

            for _ in range(self.N_B):
                n_e    = random.randint(0, self.N_E - 1)
                y_seq  = train_input[n_e]
                x_true = train_target[n_e]
                T_seq  = y_seq.size(-1)

                if generate_f:
                    f_index = n_e
                    F_base = SysModel.F_train[f_index]
                    F_true = SysModel.F_train_TRUE[f_index]
                else:
                    F_base = SysModel.F_train[n_e]
                    F_true = SysModel.F_train_TRUE[n_e]

                F_current = F_base.clone()

                # ── Initial E-step (RTSNet trainable) ───────────────────────
                self.model.update_F(F_current)
                self.model.InitSequence(SysModel.m1x_0, T_seq)
                self.model.init_hidden()

                x_fwd = torch.empty(m, T_seq, device=self.device)
                x_s   = torch.empty(m, T_seq, device=self.device)

                for t in range(T_seq):
                    x_fwd[:, t] = self.model(y_seq[:, t], None, None, None)
                x_s[:, T_seq - 1] = x_fwd[:, T_seq - 1]
                self.model.InitBackward(x_s[:, T_seq - 1])
                x_s[:, T_seq - 2] = self.model(None, x_fwd[:, T_seq - 2], x_fwd[:, T_seq - 1], None)
                for t in range(T_seq - 3, -1, -1):
                    x_s[:, t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_s[:, t + 2])

                x_loss_first_epoch += torch.mean((x_s - x_true) ** 2).detach().item()
                f_loss_first_epoch += torch.mean((F_current - F_true) ** 2).item()

                total_loss_seq = 0.0

                for em_iter in range(num_em_iters):
                    # ── Sufficient statistics ────────────────────────────────
                    x_prev = torch.empty_like(x_s)
                    x_prev[:, 0]  = SysModel.m1x_0.view(-1)
                    x_prev[:, 1:] = x_s[:, :-1]

                    A1 = (x_s @ x_prev.T) / T_seq
                    A2 = (x_prev @ x_prev.T) / T_seq

                    x_minus  = F_current @ x_prev
                    delta_x  = x_s - x_minus
                    delta_c  = delta_x - delta_x.mean(dim=1, keepdim=True)
                    S_delta_x        = (delta_c @ delta_c.T) / T_seq
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T_seq

                    nu_y = torch.stack(
                        [y_seq[:, t] - SysModel.h(x_s[:, t].detach()) for t in range(T_seq)], dim=1
                    )
                    nu_c   = nu_y - nu_y.mean(dim=1, keepdim=True)
                    S_nu_y = (nu_c @ nu_c.T) / T_seq

                    A1_input = (A1 - F_current @ A2) if A1_res else A1
                    z_in = torch.cat([
                        A1_input.detach().reshape(-1),
                        A2.detach().reshape(-1),
                        S_delta_x.detach().reshape(-1),
                        S_nu_y.detach().reshape(-1),
                        C_delta_x_xminus.detach().reshape(-1),
                        F_current.detach().reshape(-1),
                    ], dim=0).reshape(1, -1)

                    # ── M-step ───────────────────────────────────────────────
                    deltaF     = model_mstep(z_in)
                    deltaF_vv  = deltaF.squeeze(0)[2:4, 2:4]   # velocity block only
                    F_next     = F_current.clone()
                    F_next[2:4, 2:4] = F_current[2:4, 2:4] + deltaF_vv

                    f_loss = torch.mean((F_next[2:4, 2:4] - F_true[2:4, 2:4]) ** 2)
                    reg    = lambda_F * torch.mean(deltaF_vv ** 2)
                    f_loss_em_epoch[em_iter] += f_loss.detach().item()

                    F_current = F_next

                    # ── E-step with updated F (RTSNet trainable) ─────────────
                    self.model.update_F(F_current)
                    self.model.InitSequence(SysModel.m1x_0, T_seq)
                    self.model.init_hidden()

                    x_fwd = torch.empty(m, T_seq, device=self.device)
                    x_s   = torch.empty(m, T_seq, device=self.device)

                    for t in range(T_seq):
                        x_fwd[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_s[:, T_seq - 1] = x_fwd[:, T_seq - 1]
                    self.model.InitBackward(x_s[:, T_seq - 1])
                    x_s[:, T_seq - 2] = self.model(None, x_fwd[:, T_seq - 2], x_fwd[:, T_seq - 1], None)
                    for t in range(T_seq - 3, -1, -1):
                        x_s[:, t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_s[:, t + 2])

                    x_loss = torch.mean((x_s - x_true) ** 2)
                    x_loss_em_epoch[em_iter] += x_loss.detach().item()

                    if em_iter == 0:
                        weight = alpha[0]
                    elif em_iter == 1:
                        weight = alpha[1]
                    else:
                        weight = alpha[2]
                    total_loss_seq += weight * (2 * f_loss + 1*x_loss + reg)

                total_loss_batch += total_loss_seq

            total_loss_batch = total_loss_batch / float(self.N_B)
            total_loss_batch.backward()

            # NaN guard
            bad_rts = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in self.model.parameters()
            )
            bad_m = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model_mstep.parameters()
            )
            if bad_rts or bad_m:
                print(f"[Joint epoch {epoch}] NaN/Inf gradients → skipped")
                rts_optimizer.zero_grad()
                M_optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            rts_optimizer.step()
            M_optimizer.step()

            _xl0_j = x_loss_first_epoch / self.N_B
            _fl0_j = f_loss_first_epoch / self.N_B
            print(f"[Joint epoch {epoch}] FIRST RTS: "
                  f"x_loss={10*math.log10(max(_xl0_j,1e-10)):.2f} dB  "
                  f"F_loss={10*math.log10(max(_fl0_j,1e-10)):.2f} dB")
            for em_iter in range(num_em_iters):
                _xl_j = x_loss_em_epoch[em_iter] / self.N_B
                _fl_j = f_loss_em_epoch[em_iter] / self.N_B
                print(f"[Joint epoch {epoch}] MSTEP {em_iter}: "
                      f"x_loss={10*math.log10(max(_xl_j,1e-10)):.2f} dB  "
                      f"F_loss={10*math.log10(max(_fl_j,1e-10)):.2f} dB")

            # ── Validation ──────────────────────────────────────────────────
            self.model.eval()
            model_mstep.eval()
            with torch.no_grad():
                total_loss_cv = 0.0

                for j in range(self.N_CV):
                    y_cv      = cv_input[j]
                    x_true_cv = cv_target[j]
                    T_cv      = y_cv.size(-1)

                    if generate_f:
                        f_idx_cv  = j
                        F_base_cv = SysModel.F_valid[f_idx_cv]
                        F_true_cv = SysModel.F_valid_TRUE[f_idx_cv]
                    else:
                        F_base_cv = SysModel.F_valid[j]
                        F_true_cv = SysModel.F_valid_TRUE[j]

                    F_current_cv = F_base_cv.clone()

                    self.model.update_F(F_current_cv)
                    self.model.InitSequence(SysModel.m1x_0, T_cv)
                    self.model.init_hidden()

                    x_f_cv = torch.empty(m, T_cv, device=self.device)
                    x_s_cv = torch.empty(m, T_cv, device=self.device)

                    for t in range(T_cv):
                        x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)
                    x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                    self.model.InitBackward(x_s_cv[:, T_cv - 1])
                    x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                    for t in range(T_cv - 3, -1, -1):
                        x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                    total_loss_seq_cv = 0.0

                    for em_iter in range(num_em_iters):
                        x_prev_cv = torch.empty_like(x_s_cv)
                        x_prev_cv[:, 0]  = SysModel.m1x_0.view(-1)
                        x_prev_cv[:, 1:] = x_s_cv[:, :-1]

                        A1_cv = (x_s_cv @ x_prev_cv.T) / T_cv
                        A2_cv = (x_prev_cv @ x_prev_cv.T) / T_cv

                        x_minus_cv  = F_current_cv @ x_prev_cv
                        delta_x_cv  = x_s_cv - x_minus_cv
                        delta_c_cv  = delta_x_cv - delta_x_cv.mean(dim=1, keepdim=True)
                        S_delta_x_cv        = (delta_c_cv @ delta_c_cv.T) / T_cv
                        C_delta_x_xminus_cv = (delta_x_cv @ x_minus_cv.T) / T_cv

                        nu_y_cv = torch.stack(
                            [y_cv[:, t] - SysModel.h(x_s_cv[:, t]) for t in range(T_cv)], dim=1
                        )
                        nu_c_cv   = nu_y_cv - nu_y_cv.mean(dim=1, keepdim=True)
                        S_nu_y_cv = (nu_c_cv @ nu_c_cv.T) / T_cv

                        A1_cv_input = (A1_cv - F_current_cv @ A2_cv) if A1_res else A1_cv
                        z_cv = torch.cat([
                            A1_cv_input.reshape(-1),
                            A2_cv.reshape(-1),
                            S_delta_x_cv.reshape(-1),
                            S_nu_y_cv.reshape(-1),
                            C_delta_x_xminus_cv.reshape(-1),
                            F_current_cv.reshape(-1),
                        ], dim=0).reshape(1, -1)

                        dF_cv     = model_mstep(z_cv)
                        dF_mat_cv = dF_cv.squeeze(0)
                        F_next_cv = F_current_cv.clone()
                        F_next_cv[2:4, 2:4] = F_current_cv[2:4, 2:4] + dF_mat_cv[2:4, 2:4]

                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)
                        reg_cv    = lambda_F * torch.mean(dF_mat_cv ** 2)
                        F_current_cv = F_next_cv.clone()

                        self.model.update_F(F_current_cv)
                        self.model.InitSequence(SysModel.m1x_0, T_cv)
                        self.model.init_hidden()

                        x_f_cv = torch.empty(m, T_cv, device=self.device)
                        x_s_cv = torch.empty(m, T_cv, device=self.device)

                        for t in range(T_cv):
                            x_f_cv[:, t] = self.model(y_cv[:, t], None, None, None)
                        x_s_cv[:, T_cv - 1] = x_f_cv[:, T_cv - 1]
                        self.model.InitBackward(x_s_cv[:, T_cv - 1])
                        x_s_cv[:, T_cv - 2] = self.model(None, x_f_cv[:, T_cv - 2], x_f_cv[:, T_cv - 1], None)
                        for t in range(T_cv - 3, -1, -1):
                            x_s_cv[:, t] = self.model(None, x_f_cv[:, t], x_f_cv[:, t + 1], x_s_cv[:, t + 2])

                        x_loss_cv = torch.mean((x_s_cv - x_true_cv) ** 2)
                        if em_iter == 0:
                            weight_cv = alpha[0]
                        elif em_iter == 1:
                            weight_cv = alpha[1]
                        else:
                            weight_cv = alpha[2]
                        total_loss_seq_cv += weight_cv * (2* f_loss_cv.item() + 1*x_loss_cv.item()
                                                          +  reg_cv.item())

                    total_loss_cv += total_loss_seq_cv

            cv_epoch = total_loss_cv / max(1, self.N_CV)

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt  = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(self.model, destination_path_RTS)
                torch.save(model_mstep, destination_path_M)

            print(f"[Joint epoch {epoch}] train={total_loss_batch.item():.6f}  "
                  f"cv={cv_epoch:.6f}  best_cv={self.MSE_cv_dB_opt:.6f} (epoch {self.MSE_cv_idx_opt})")

    # ─── F M-step Testing ─────────────────────────────────────────────────────
    def test_F_mstep_net(self, SysModel, test_input, test_target,
                         destination_path_RTS, destination_path_M,
                         num_em_iters=1, lambda_F=1e-3, generate_f=True,
                         A1_res=False):
        """
        Apply trained DeltaF_MStepNet at test time.
        Returns [MSE_arr, MSE_avg, MSE_dB_avg, x_out_tensor, RunTime]
        to match the NNTest return signature expected by the main.
        """
        N_T = len(test_input)
        m   = SysModel.m

        # Load and freeze RTSNet
        self.model = torch.load(destination_path_RTS, map_location=self.device, weights_only=False)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Load MNet (inference only)
        model_mstep = torch.load(destination_path_M, map_location=self.device, weights_only=False)
        model_mstep.to(self.device).eval()

        loss_fn = nn.MSELoss(reduction='mean')
        MSE_test_linear_arr = torch.empty(N_T, device=self.device)
        x_out_list          = []
        x_loss_per_iter     = torch.zeros(num_em_iters + 1, device=self.device)

        start = time.time()
        with torch.no_grad():
            for j in range(N_T):
                y_seq  = test_input[j]   # [n, T]
                x_true = test_target[j]  # [m, T]
                T_seq  = y_seq.size(-1)

                if generate_f:
                    f_index = j
                    F_base  = SysModel.F_test[f_index]
                else:
                    F_base  = SysModel.F_test[j]

                F_current = F_base.clone()

                # ── Initial E-step ───────────────────────────────────────────
                self.model.update_F(F_current)
                self.model.InitSequence(SysModel.m1x_0, T_seq)
                self.model.init_hidden()

                x_fwd = torch.empty(m, T_seq, device=self.device)
                x_s   = torch.empty(m, T_seq, device=self.device)

                for t in range(T_seq):
                    x_fwd[:, t] = self.model(y_seq[:, t], None, None, None)
                x_s[:, T_seq - 1] = x_fwd[:, T_seq - 1]
                self.model.InitBackward(x_s[:, T_seq - 1])
                x_s[:, T_seq - 2] = self.model(None, x_fwd[:, T_seq - 2], x_fwd[:, T_seq - 1], None)
                for t in range(T_seq - 3, -1, -1):
                    x_s[:, t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_s[:, t + 2])

                x_loss_per_iter[0] += loss_fn(x_s, x_true).item()

                for em_iter in range(num_em_iters):
                    # ── Sufficient statistics ────────────────────────────────
                    x_prev = torch.empty_like(x_s)
                    x_prev[:, 0]  = SysModel.m1x_0.view(-1)
                    x_prev[:, 1:] = x_s[:, :-1]

                    A1 = (x_s @ x_prev.T) / T_seq
                    A2 = (x_prev @ x_prev.T) / T_seq

                    x_minus  = F_current @ x_prev
                    delta_x  = x_s - x_minus
                    delta_c  = delta_x - delta_x.mean(dim=1, keepdim=True)
                    S_delta_x        = (delta_c @ delta_c.T) / T_seq
                    C_delta_x_xminus = (delta_x @ x_minus.T) / T_seq

                    nu_y = torch.stack(
                        [y_seq[:, t] - SysModel.h(x_s[:, t]) for t in range(T_seq)], dim=1
                    )
                    nu_c   = nu_y - nu_y.mean(dim=1, keepdim=True)
                    S_nu_y = (nu_c @ nu_c.T) / T_seq

                    A1_input = (A1 - F_current @ A2) if A1_res else A1
                    z_in = torch.cat([
                        A1_input.reshape(-1),
                        A2.reshape(-1),
                        S_delta_x.reshape(-1),
                        S_nu_y.reshape(-1),
                        C_delta_x_xminus.reshape(-1),
                        F_current.detach().reshape(-1),
                    ], dim=0).reshape(1, -1)

                    # ── M-step (no grad) ─────────────────────────────────────
                    deltaF    = model_mstep(z_in)
                    deltaF_vv = deltaF.squeeze(0)[2:4, 2:4]   # velocity block only
                    F_next    = F_current.clone()
                    F_next[2:4, 2:4] = F_current[2:4, 2:4] + deltaF_vv
                    F_current = F_next

                    # ── E-step with updated F ────────────────────────────────
                    self.model.update_F(F_current)
                    self.model.InitSequence(SysModel.m1x_0, T_seq)
                    self.model.init_hidden()

                    x_fwd = torch.empty(m, T_seq, device=self.device)
                    x_s   = torch.empty(m, T_seq, device=self.device)

                    for t in range(T_seq):
                        x_fwd[:, t] = self.model(y_seq[:, t], None, None, None)
                    x_s[:, T_seq - 1] = x_fwd[:, T_seq - 1]
                    self.model.InitBackward(x_s[:, T_seq - 1])
                    x_s[:, T_seq - 2] = self.model(None, x_fwd[:, T_seq - 2], x_fwd[:, T_seq - 1], None)
                    for t in range(T_seq - 3, -1, -1):
                        x_s[:, t] = self.model(None, x_fwd[:, t], x_fwd[:, t + 1], x_s[:, t + 2])

                    x_loss_per_iter[em_iter + 1] += loss_fn(x_s, x_true).item()

                MSE_test_linear_arr[j] = loss_fn(x_s, x_true).item()
                x_out_list.append(x_s)

        end = time.time()
        t   = end - start

        MSE_test_linear_avg = torch.mean(MSE_test_linear_arr)
        MSE_test_dB_avg     = 10 * torch.log10(MSE_test_linear_avg)
        x_loss_per_iter    /= N_T

        print(f"[F M-step TEST] MSE avg: {MSE_test_dB_avg.item():.2f} dB  time: {t:.1f}s")
        print("[F M-step TEST] x-loss per EM iter:")
        for k in range(num_em_iters + 1):
            v   = x_loss_per_iter[k].item()
            tag = "init" if k == 0 else f"EM {k}"
            print(f"  {tag}: {v:.6e}  ({10 * math.log10(v):.2f} dB)")

        return [
            MSE_test_linear_arr,
            MSE_test_linear_avg,
            MSE_test_dB_avg,
            torch.stack(x_out_list),
            t,
        ]


    def test_F_mstep_net_3_datasets(self, SysModel, all_test_inputs, all_test_targets,
                                     destination_path_RTS, destination_path_M,
                                     num_em_iters=1, generate_f=True, datasets=3,
                                     propagate_F=True, obs_mask=None, A1_res=False):
        """
        Test MNet across `datasets` sequential datasets.
        For each test sequence j, datasets are processed in order 0→datasets-1.
        x_0 and F carry forward from dataset k to dataset k+1.
        F is seeded from SysModel.F_test[0] (false F of dataset 0) then carried.
        SysModel.F_test must be [datasets][group_idx].
        Outputs are ordered [data * N_T + j].
        """
        N_T = len(all_test_inputs[0])
        m   = SysModel.m

        self.model = torch.load(destination_path_RTS, map_location=self.device, weights_only=False)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        model_mstep = torch.load(destination_path_M, map_location=self.device, weights_only=False)
        model_mstep.to(self.device).eval()

        loss_fn         = nn.MSELoss(reduction='mean')
        MSE_arr         = torch.empty(N_T * datasets, device=self.device)
        x_out_list      = []
        x_loss_per_iter = torch.zeros(num_em_iters + 1, device=self.device)
        f_loss_per_iter = torch.zeros(num_em_iters + 1, device=self.device)
        has_F_true      = hasattr(SysModel, 'F_test_TRUE')

        # ---- Vectorized inference: all N_T sequences as one batch ----
        B = N_T
        idx = list(range(B))
        x_s_per_ds = []   # [datasets] of [B,m,T]

        start = time.time()
        with torch.no_grad():
            x_0 = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(B, m, 1).contiguous()
            F_carried = torch.stack([SysModel.F_test[0][j] for j in idx]).to(self.device).clone()  # [B,m,m]

            for data in range(datasets):
                y_seq  = torch.stack([all_test_inputs[data][j] for j in idx]).to(self.device)     # [B,n,T]
                x_true = torch.stack([all_test_targets[data][j] for j in idx]).to(self.device)    # [B,m,T]
                T = y_seq.size(-1)
                F_current = F_carried
                F_true_b = torch.stack([SysModel.F_test_TRUE[data][j] for j in idx]).to(self.device) if has_F_true else None

                # ── Initial E-step ──
                x_s = self._infer_smooth_batched(y_seq, F_current, x_0, obs_mask)
                per_seq = ((x_s - x_true) ** 2).mean(dim=(1, 2))                 # [B]
                x_loss_per_iter[0] += per_seq.sum()
                if has_F_true:
                    f_loss_per_iter[0] += ((F_current[:, 2:4, 2:4] - F_true_b[:, 2:4, 2:4]) ** 2).mean(dim=(1, 2)).sum()

                _F_start_b = F_current.detach().clone()
                _x_init_b  = per_seq.detach()
                _F_em_b    = [None] * num_em_iters
                _x_em_b    = [None] * num_em_iters

                # ── EM iterations ──
                for em_iter in range(num_em_iters):
                    z_in = self._mstep_z_batched(x_s, x_0, F_current, y_seq, SysModel.h, A1_res)
                    deltaF = model_mstep(z_in)
                    deltaF_vv = deltaF[:, 2:4, 2:4]
                    F_next = F_current.clone()
                    F_next[:, 2:4, 2:4] = F_current[:, 2:4, 2:4] + deltaF_vv
                    F_current = F_next

                    if has_F_true:
                        f_loss_per_iter[em_iter + 1] += ((F_current[:, 2:4, 2:4] - F_true_b[:, 2:4, 2:4]) ** 2).mean(dim=(1, 2)).sum()

                    x_s = self._infer_smooth_batched(y_seq, F_current, x_0, obs_mask)
                    per_seq_em = ((x_s - x_true) ** 2).mean(dim=(1, 2))
                    x_loss_per_iter[em_iter + 1] += per_seq_em.sum()
                    _F_em_b[em_iter] = F_current.detach().clone()
                    _x_em_b[em_iter] = per_seq_em.detach()

                # ── print F evolution for first up-to-4 sequences (rows) ──
                for r in range(min(4, B)):
                    _fs = _F_start_b[r, 2:4, 2:4].cpu()
                    _xi_db = 10 * math.log10(max(_x_init_b[r].item(), 1e-10))
                    _ft_str = ""
                    if has_F_true:
                        _ftv = F_true_b[r, 2:4, 2:4].cpu()
                        _fl0 = ((_F_start_b[r, 2:4, 2:4] - F_true_b[r, 2:4, 2:4]) ** 2).mean().item()
                        _ft_str = (f"  F_true=[[{_ftv[0,0]:.4f} {_ftv[0,1]:.4f}][{_ftv[1,0]:.4f} {_ftv[1,1]:.4f}]]"
                                   f"  F_loss={10*math.log10(max(_fl0,1e-10)):.2f}dB")
                    print(f"[TEST seq{r} ds{data}] x_init={_xi_db:.2f}dB"
                          f"  F_start=[[{_fs[0,0]:.4f} {_fs[0,1]:.4f}][{_fs[1,0]:.4f} {_fs[1,1]:.4f}]]{_ft_str}")
                    for _k in range(num_em_iters):
                        _fe = _F_em_b[_k][r, 2:4, 2:4].cpu()
                        _xl_db = 10 * math.log10(max(_x_em_b[_k][r].item(), 1e-10))
                        _fl_str = ""
                        if has_F_true:
                            _fl = ((_F_em_b[_k][r, 2:4, 2:4] - F_true_b[r, 2:4, 2:4]) ** 2).mean().item()
                            _fl_str = f"  F_loss={10*math.log10(max(_fl,1e-10)):.2f}dB"
                        print(f"  EM{_k+1}: F_est=[[{_fe[0,0]:.4f} {_fe[0,1]:.4f}][{_fe[1,0]:.4f} {_fe[1,1]:.4f}]]"
                              f"  x_loss={_xl_db:.2f}dB{_fl_str}")

                MSE_arr[data * N_T:(data + 1) * N_T] = ((x_s - x_true) ** 2).mean(dim=(1, 2))
                x_s_per_ds.append(x_s)
                x_0 = x_s[:, :, -1:].detach()
                if propagate_F:
                    F_carried = F_current.detach()
                else:
                    F_carried = torch.stack([SysModel.F_test[data][j] for j in idx]).to(self.device).clone()

        # x_out ordered [j * datasets + data] (seq-major), matching the original loop
        x_out_list = [x_s_per_ds[data][j] for j in range(N_T) for data in range(datasets)]

        end = time.time()
        t = end - start

        MSE_avg    = torch.mean(MSE_arr)
        MSE_dB_avg = 10 * torch.log10(MSE_avg)
        x_loss_per_iter /= (N_T * datasets)

        f_loss_per_iter /= (N_T * datasets)
        print(f"[F M-step TEST {datasets}-datasets] MSE avg: {MSE_dB_avg.item():.2f} dB  time: {t:.1f}s")
        for k in range(num_em_iters + 1):
            v   = x_loss_per_iter[k].item()
            tag = "init" if k == 0 else f"EM {k}"
            f_str = ""
            if has_F_true:
                fv    = f_loss_per_iter[k].item()
                f_str = f"  F_MSE={fv:.6e} ({10*math.log10(max(fv,1e-10)):.2f} dB)"
            print(f"  {tag}: x={v:.6e} ({10*math.log10(v):.2f} dB){f_str}")
        for data in range(datasets):
            ds_mse = torch.mean(MSE_arr[data * N_T : (data + 1) * N_T])
            print(f"  Dataset {data}: {10 * torch.log10(ds_mse).item():.2f} dB")

        return [MSE_arr, MSE_avg, MSE_dB_avg, torch.stack(x_out_list), t]

    def train_F_mstep_net_3_datasets(self, SysModel, cv_input, cv_target, train_input, train_target,
                        destination_path_M, load_path_RTS, load_mnet, num_em_iters=3, F_init=None,
                        alpha=(0.05, 0.1, 0.85), lambda_F=1e-3, generate_f=True, datasets=3,
                        propagate_F=True, A1_res=False, use_big_mstep_net=None, mstep_hidden_dim=None,
                        obs_mask=None):
        """
        M-step training for F (state transition matrix) across 3 datasets.
        - h is NONLINEAR and stays fixed
        - F is DIVERSE / mismatched and is updated by the M-network
        - Trains M-network to predict ΔF given smoothed states and statistics
        """
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m
        n = SysModel.n

        # Load and freeze RTSNet
        self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # M-step model for F
        self.M_model = self._load_or_new_mstep_model(
            load_mnet,
            use_big_mstep_net=use_big_mstep_net,
            mstep_hidden_dim=mstep_hidden_dim,
        )
        model_mstep = self.M_model.train()
        self.M_optimizer = torch.optim.Adam(model_mstep.parameters(), lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            train_loss_sum = 0.0
            model_mstep.train()
            self.M_optimizer.zero_grad()

            batch_total_loss = 0.0
            batch_x_loss_start = 0.0
            batch_x_loss_em = [0.0] * num_em_iters

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # ---- Vectorized mini-batch: B sequences in parallel ----
                B = self.N_B
                idx = [random.randint(0, self.N_E - 1) for _ in range(B)]
                _dbg = True

                if F_init is not None:
                    F_base = F_init.clone().detach().to(self.device).reshape(1, m, m).expand(B, m, m).contiguous()
                else:
                    F_base = torch.stack([SysModel.F_train[0][i].clone().detach() for i in idx]).to(self.device)
                x_0 = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(B, m, 1).contiguous()
                prior_Sigma = SysModel.m2x_0.clone().detach().to(self.device)
                sample_total_loss = 0.0

                for data in range(datasets):
                    F_current = F_base.clone()
                    y_seq = torch.stack([train_input[data][i] for i in idx]).to(self.device)
                    x_true_seq = torch.stack([train_target[data][i] for i in idx]).to(self.device)
                    T = y_seq.size(-1)
                    F_true = torch.stack([SysModel.F_train_TRUE[data][i] for i in idx]).to(self.device)

                    x_curr = self._run_smoother_batched(y_seq, F_current, x_0, prior_Sigma, obs_mask)
                    x_loss_start = torch.mean((x_curr - x_true_seq) ** 2)
                    batch_x_loss_start += x_loss_start.detach().item()

                    for em_iter in range(num_em_iters):
                        z_in = self._mstep_z_batched(x_curr, x_0, F_current, y_seq, SysModel.h, A1_res)
                        deltaF = model_mstep(z_in)
                        deltaF_vv = deltaF[:, 2:4, 2:4]
                        F_next = F_current.clone()
                        F_next[:, 2:4, 2:4] = F_current[:, 2:4, 2:4] + deltaF_vv
                        F_current = F_next

                        f_loss = torch.mean((F_next[:, 2:4, 2:4] - F_true[:, 2:4, 2:4]) ** 2)
                        reg = lambda_F * torch.mean(deltaF_vv ** 2)

                        x_curr = self._run_smoother_batched(y_seq, F_current, x_0, prior_Sigma, obs_mask)
                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                        batch_x_loss_em[em_iter] += x_loss.detach().item()

                        loss_em = 60 * f_loss + reg + x_loss
                        weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                        sample_total_loss = sample_total_loss + weight * loss_em

                        if _dbg and data == 0:
                            _fe = F_current[0, 2:4, 2:4].detach().cpu()
                            _ft = F_true[0, 2:4, 2:4].detach().cpu()
                            print(f"[ep{epoch:03d} ds0 EM{em_iter+1}] "
                                  f"F_est=[[{_fe[0,0]:.4f} {_fe[0,1]:.4f}][{_fe[1,0]:.4f} {_fe[1,1]:.4f}]] "
                                  f"F_true=[[{_ft[0,0]:.4f} {_ft[0,1]:.4f}][{_ft[1,0]:.4f} {_ft[1,1]:.4f}]] "
                                  f"f={10*math.log10(max(f_loss.item(),1e-10)):.2f}dB "
                                  f"x={10*math.log10(max(x_loss.item(),1e-10)):.2f}dB")

                    if propagate_F:
                        F_base = F_current.detach()
                    else:
                        if F_init is not None:
                            F_base = F_init.clone().detach().to(self.device).reshape(1, m, m).expand(B, m, m).contiguous()
                        else:
                            F_base = torch.stack([SysModel.F_train[0][i].clone().detach() for i in idx]).to(self.device)
                    x_0 = x_curr[:, :, -1:].detach()

                sample_total_loss = sample_total_loss / float(datasets)
                batch_total_loss = sample_total_loss

            loss = batch_total_loss   # already mean over the B-sequence batch
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.M_optimizer)

            bad_grad = (not torch.isfinite(loss)) or any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model_mstep.parameters()
            )
            if bad_grad:
                print(f"[MNet-3ds epoch {epoch:03d}] NaN/Inf loss or gradients → step skipped")
                self.M_optimizer.zero_grad()
                self.scaler.update()
                continue

            torch.nn.utils.clip_grad_norm_(model_mstep.parameters(), max_norm=1.0)
            self.scaler.step(self.M_optimizer)
            self.scaler.update()

            denom = datasets
            avg_x_loss_start = batch_x_loss_start / denom
            avg_x_loss_em = [x / denom for x in batch_x_loss_em]

            em_msg = " ".join([f"x_loss_em{k}={avg_x_loss_em[k]:.6f}" for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] x_loss_start={avg_x_loss_start:.6f} {em_msg} loss_all={loss.item():.6f}")

            # Validation
            model_mstep.eval()
            cv_loss_sum = 0.0
            batch_cv_x_loss_start = 0.0
            batch_cv_x_loss_em = [0.0] * num_em_iters

            with torch.no_grad():
                # ---- Vectorized CV: all N_CV sequences as one batch ----
                B = self.N_CV
                idx = list(range(B))

                if F_init is not None:
                    F_base_cv = F_init.clone().detach().to(self.device).reshape(1, m, m).expand(B, m, m).contiguous()
                else:
                    F_base_cv = torch.stack([SysModel.F_valid[0][j].clone().detach() for j in idx]).to(self.device)
                x_0_cv = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(B, m, 1).contiguous()
                prior_Sigma_cv = SysModel.m2x_0.clone().detach().to(self.device)
                sample_total_loss_cv = 0.0

                for data in range(datasets):
                    F_current_cv = F_base_cv.clone()
                    y_cv = torch.stack([cv_input[data][j] for j in idx]).to(self.device)
                    x_true_cv_seq = torch.stack([cv_target[data][j] for j in idx]).to(self.device)
                    T_cv = y_cv.size(-1)
                    F_true_cv = torch.stack([SysModel.F_valid_TRUE[data][j] for j in idx]).to(self.device)

                    x_curr = self._run_smoother_batched(y_cv, F_current_cv, x_0_cv, prior_Sigma_cv, obs_mask)
                    x_loss_start_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                    batch_cv_x_loss_start += x_loss_start_cv.item()

                    for em_iter in range(num_em_iters):
                        z_cv = self._mstep_z_batched(x_curr, x_0_cv, F_current_cv, y_cv, SysModel.h, A1_res)
                        dF_cv = model_mstep(z_cv)
                        dF_cv_vv = dF_cv[:, 2:4, 2:4]
                        F_next_cv = F_current_cv.clone()
                        F_next_cv[:, 2:4, 2:4] = F_current_cv[:, 2:4, 2:4] + dF_cv_vv
                        F_current_cv = F_next_cv

                        f_loss_cv = torch.mean((F_next_cv[:, 2:4, 2:4] - F_true_cv[:, 2:4, 2:4]) ** 2)
                        reg_cv = lambda_F * torch.mean(dF_cv_vv ** 2)

                        x_curr = self._run_smoother_batched(y_cv, F_current_cv, x_0_cv, prior_Sigma_cv, obs_mask)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        batch_cv_x_loss_em[em_iter] += x_loss_cv.item()

                        loss_em_cv = 60 * f_loss_cv + reg_cv + x_loss_cv
                        weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                        sample_total_loss_cv = sample_total_loss_cv + weight * loss_em_cv

                    x_0_cv = x_curr[:, :, -1:].detach()
                    if propagate_F:
                        F_base_cv = F_current_cv.detach()
                    else:
                        F_base_cv = torch.stack([SysModel.F_valid[data][j].clone().detach() for j in idx]).to(self.device)

                sample_total_loss_cv = sample_total_loss_cv / float(datasets)
                cv_loss_sum = sample_total_loss_cv.item()

            cv_epoch = cv_loss_sum   # already averaged over the N_CV batch
            cv_denom = datasets
            avg_cv_x_loss_start = batch_cv_x_loss_start / cv_denom
            avg_cv_x_loss_em = [x / cv_denom for x in batch_cv_x_loss_em]

            cv_em_msg = " ".join([f"cv_x_loss_em{k}={avg_cv_x_loss_em[k]:.6f}" for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] cv_x_loss_start={avg_cv_x_loss_start:.6f} {cv_em_msg} cv_all={cv_epoch:.6f}")
            print(f"BEST: epoch={self.MSE_cv_idx_opt}  best_cv_loss={self.MSE_cv_dB_opt:.6f}")

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)




    def train_F_mstep_net_3_datasets_joint(self, SysModel, cv_input, cv_target, train_input, train_target,
                                    destination_path_M, destination_path_RTS, load_path_RTS, load_mnet, num_em_iters=3, F_init=None,
                                    alpha=(0.05, 0.1, 0.85), lambda_F=1e-3, generate_f=True, datasets=3,
                                    x_0_train_list=None, x_0_cv_list=None, propagate_F=True, A1_res=False,
                                    use_big_mstep_net=None, mstep_hidden_dim=None, obs_mask=None):
        """
        Joint training for F M-step + RTSNet across 3 datasets.
        - h is NONLINEAR and stays fixed
        - F is updated by the M-network
        """
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m
        n = SysModel.n

        self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.M_model = self._load_or_new_mstep_model(
            load_mnet,
            use_big_mstep_net=use_big_mstep_net,
            mstep_hidden_dim=mstep_hidden_dim,
        )
        model_mstep = self.M_model.train()
        self.optimizer_joint = torch.optim.Adam(
            list(self.model.parameters()) + list(model_mstep.parameters()),
            lr=self.learningRate
        )
        scheduler_joint = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_joint, T_0=20, T_mult=1, eta_min=1e-6
        )

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for epoch in range(self.N_steps):
            model_mstep.train()
            self.model.train()
            self.optimizer_joint.zero_grad()

            batch_total_loss = 0.0
            batch_x_loss_start = 0.0
            batch_x_loss_em = [0.0] * num_em_iters
            batch_f_loss_em = [0.0] * num_em_iters
            batch_reg_em = [0.0] * num_em_iters

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # ---- Vectorized mini-batch: B sequences in parallel ----
                B = self.N_B
                idx = [random.randint(0, self.N_E - 1) for _ in range(B)]
                _dbg = True   # print row-0 F trajectory once per epoch

                # Per-sequence starting F  [B,m,m] (direct gather by index)
                if F_init is not None:
                    F_base = F_init.clone().detach().to(self.device).reshape(1, m, m).expand(B, m, m).contiguous()
                else:
                    F_base = torch.stack([SysModel.F_train[0][i].clone().detach() for i in idx]).to(self.device)

                # Initial state  [B,m,1]
                if x_0_train_list is not None:
                    x_0 = torch.stack([x_0_train_list[i].reshape(m, 1) for i in idx]).to(self.device)
                else:
                    x_0 = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(B, m, 1).contiguous()

                prior_Sigma = SysModel.m2x_0.clone().detach().to(self.device)
                sample_total_loss = 0.0

                for data in range(datasets):
                    F_current = F_base.clone()                                          # [B,m,m]
                    y_seq = torch.stack([train_input[data][i] for i in idx]).to(self.device)      # [B,n,T]
                    x_true_seq = torch.stack([train_target[data][i] for i in idx]).to(self.device)  # [B,m,T]
                    T = y_seq.size(-1)
                    F_true = torch.stack([SysModel.F_train_TRUE[data][i] for i in idx]).to(self.device)  # [B,m,m]

                    x_curr = self._run_smoother_batched(y_seq, F_current, x_0, prior_Sigma, obs_mask)
                    x_loss_start = torch.mean((x_curr - x_true_seq) ** 2)
                    batch_x_loss_start += x_loss_start.detach().item()

                    for em_iter in range(num_em_iters):
                        z_in = self._mstep_z_batched(x_curr, x_0, F_current, y_seq, SysModel.h, A1_res)
                        deltaF = model_mstep(z_in)                                      # [B,m,m]
                        deltaF_vv = deltaF[:, 2:4, 2:4]
                        F_next = F_current.clone()
                        F_next[:, 2:4, 2:4] = F_current[:, 2:4, 2:4] + deltaF_vv
                        F_current = F_next

                        f_loss = torch.mean((F_next[:, 2:4, 2:4] - F_true[:, 2:4, 2:4]) ** 2)
                        reg = lambda_F * torch.mean(deltaF_vv ** 2)
                        batch_f_loss_em[em_iter] += f_loss.detach().item()
                        batch_reg_em[em_iter] += reg.detach().item()

                        x_curr = self._run_smoother_batched(y_seq, F_current, x_0, prior_Sigma, obs_mask)
                        x_loss = torch.mean((x_curr - x_true_seq) ** 2)
                        batch_x_loss_em[em_iter] += x_loss.detach().item()

                        loss_em = 60 * f_loss + 0.5 * reg + x_loss
                        weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                        sample_total_loss = sample_total_loss + weight * loss_em

                        if _dbg and data == 0:
                            _fe = F_current[0, 2:4, 2:4].detach().cpu()
                            _ft = F_true[0, 2:4, 2:4].detach().cpu()
                            print(f"[ep{epoch:03d} ds0 EM{em_iter+1}] "
                                  f"F_est=[[{_fe[0,0]:.4f} {_fe[0,1]:.4f}][{_fe[1,0]:.4f} {_fe[1,1]:.4f}]] "
                                  f"F_true=[[{_ft[0,0]:.4f} {_ft[0,1]:.4f}][{_ft[1,0]:.4f} {_ft[1,1]:.4f}]] "
                                  f"f={10*math.log10(max(f_loss.item(),1e-10)):.2f}dB "
                                  f"x={10*math.log10(max(x_loss.item(),1e-10)):.2f}dB")

                    # propagate_F=True: carry estimated F into next dataset
                    if propagate_F:
                        F_base = F_current.detach()
                    else:
                        if F_init is not None:
                            F_base = F_init.clone().detach().to(self.device).reshape(1, m, m).expand(B, m, m).contiguous()
                        else:
                            F_base = torch.stack([SysModel.F_train[0][i].clone().detach() for i in idx]).to(self.device)
                    x_0 = x_curr[:, :, -1:].detach()                                    # [B,m,1]

                # loss_em means already reduce over the batch (torch.mean over [B,m,T]),
                # so this is the full mini-batch loss — no extra /N_B below.
                sample_total_loss = sample_total_loss / float(datasets)
                batch_total_loss = sample_total_loss

            loss = batch_total_loss   # already mean over the B-sequence batch
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer_joint)

            all_params = list(self.model.parameters()) + list(model_mstep.parameters())
            bad_grad = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in all_params
            )
            if bad_grad:
                print(f"[Joint-3ds epoch {epoch:03d}] NaN/Inf gradients → skipped")
                self.optimizer_joint.zero_grad()
                self.scaler.update()
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            self.scaler.step(self.optimizer_joint)
            self.scaler.update()
            print(f"  grad_norm (before clip)={grad_norm:.3f}")

            denom = datasets   # batched: losses already averaged over the B sequences
            avg_x_loss_start = batch_x_loss_start / denom
            avg_x_loss_em = [x / denom for x in batch_x_loss_em]
            avg_f_loss_em = [x / denom for x in batch_f_loss_em]
            avg_reg_em = [x / denom for x in batch_reg_em]

            em_msg = " ".join([
                f"x{k}={avg_x_loss_em[k]:.4f} "
                f"f{k}={avg_f_loss_em[k]:.4f} "
                f"reg{k}={avg_reg_em[k]:.4f}"
                for k in range(num_em_iters)
            ])
            print(f"[epoch {epoch:03d}] x_loss_start={avg_x_loss_start:.6f} {em_msg} loss_all={loss.item():.6f}")

            # F-summary printed inline per sample (4 samples per epoch) — see above

            # Validation
            model_mstep.eval()
            self.model.eval()
            cv_loss_sum = 0.0
            batch_cv_x_loss_start = 0.0
            batch_cv_x_loss_em = [0.0] * num_em_iters
            batch_cv_f_loss_em = [0.0] * num_em_iters
            batch_cv_reg_em = [0.0] * num_em_iters

            with torch.no_grad():
                # ---- Vectorized CV: all N_CV sequences as one batch ----
                B = self.N_CV
                idx = list(range(B))

                if F_init is not None:
                    F_base_cv = F_init.clone().detach().to(self.device).reshape(1, m, m).expand(B, m, m).contiguous()
                else:
                    F_base_cv = torch.stack([SysModel.F_valid[0][j].clone().detach() for j in idx]).to(self.device)

                if x_0_cv_list is not None:
                    x_0_cv = torch.stack([x_0_cv_list[j].reshape(m, 1) for j in idx]).to(self.device)
                else:
                    x_0_cv = SysModel.m1x_0.clone().detach().to(self.device).reshape(1, m, 1).expand(B, m, 1).contiguous()

                prior_Sigma_cv = SysModel.m2x_0.clone().detach().to(self.device)
                sample_total_loss_cv = 0.0

                for data in range(datasets):
                    F_current_cv = F_base_cv.clone()
                    y_cv = torch.stack([cv_input[data][j] for j in idx]).to(self.device)
                    x_true_cv_seq = torch.stack([cv_target[data][j] for j in idx]).to(self.device)
                    T_cv = y_cv.size(-1)
                    F_true_cv = torch.stack([SysModel.F_valid_TRUE[data][j] for j in idx]).to(self.device)

                    x_curr = self._run_smoother_batched(y_cv, F_current_cv, x_0_cv, prior_Sigma_cv, obs_mask)
                    x_loss_start_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                    batch_cv_x_loss_start += x_loss_start_cv.item()

                    for em_iter in range(num_em_iters):
                        z_cv = self._mstep_z_batched(x_curr, x_0_cv, F_current_cv, y_cv, SysModel.h, A1_res)
                        dF_cv = model_mstep(z_cv)
                        dF_cv_vv = dF_cv[:, 2:4, 2:4]
                        F_next_cv = F_current_cv.clone()
                        F_next_cv[:, 2:4, 2:4] = F_current_cv[:, 2:4, 2:4] + dF_cv_vv
                        F_current_cv = F_next_cv

                        f_loss_cv = torch.mean((F_next_cv[:, 2:4, 2:4] - F_true_cv[:, 2:4, 2:4]) ** 2)
                        reg_cv = lambda_F * torch.mean(dF_cv_vv ** 2)
                        batch_cv_f_loss_em[em_iter] += f_loss_cv.item()
                        batch_cv_reg_em[em_iter] += reg_cv.item()

                        x_curr = self._run_smoother_batched(y_cv, F_current_cv, x_0_cv, prior_Sigma_cv, obs_mask)
                        x_loss_cv = torch.mean((x_curr - x_true_cv_seq) ** 2)
                        batch_cv_x_loss_em[em_iter] += x_loss_cv.item()

                        loss_em_cv = 60 * f_loss_cv + 0.5 * reg_cv + x_loss_cv
                        weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                        sample_total_loss_cv = sample_total_loss_cv + weight * loss_em_cv

                    x_0_cv = x_curr[:, :, -1:].detach()
                    if propagate_F:
                        F_base_cv = F_current_cv.detach()
                    else:
                        F_base_cv = torch.stack([SysModel.F_valid[data][j].clone().detach() for j in idx]).to(self.device)

                sample_total_loss_cv = sample_total_loss_cv / float(datasets)
                cv_loss_sum = sample_total_loss_cv.item()

            cv_epoch = cv_loss_sum   # already averaged over the N_CV batch
            cv_denom = datasets
            avg_cv_x_loss_start = batch_cv_x_loss_start / cv_denom
            avg_cv_x_loss_em = [x / cv_denom for x in batch_cv_x_loss_em]
            avg_cv_f_loss_em = [x / cv_denom for x in batch_cv_f_loss_em]
            avg_cv_reg_em = [x / cv_denom for x in batch_cv_reg_em]

            cv_em_msg = " ".join([
                f"cv_x{k}={avg_cv_x_loss_em[k]:.6f} "
                f"cv_f{k}={avg_cv_f_loss_em[k]:.6f} "
                f"cv_reg{k}={avg_cv_reg_em[k]:.6f}"
                for k in range(num_em_iters)
            ])

            scheduler_joint.step(epoch)
            curr_lr = self.optimizer_joint.param_groups[0]['lr']
            lr_str = f"  lr={curr_lr:.2e}"

            print(f"[epoch {epoch:03d}] cv_x_loss_start={avg_cv_x_loss_start:.6f} {cv_em_msg} cv_all={cv_epoch:.6f}{lr_str}")
            print(f"BEST: epoch={self.MSE_cv_idx_opt}  best_cv_loss={self.MSE_cv_dB_opt:.6f}")

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)
                torch.save(self.model, destination_path_RTS)
