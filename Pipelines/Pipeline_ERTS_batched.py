"""
Batched training pipeline (delete-to-revert add-on).

`Pipeline_ERTS_batched` subclasses the existing `Pipeline_ERTS` and ADDS batched
twins of the two training functions in use:

    - train_RTS_net_3_datasets_batched
    - train_H_mstep_net_3_datasets_joint_batched   (added after RTS is validated)

The originals in Pipeline_ERTS are untouched, so you can A/B them. To revert the
whole feature: delete this file + RTSNet/RTSNet_nn_batched.py and switch the
import in your run script back to `Pipeline_ERTS`.

All N_B samples of a batch run through RTSNet together (one set of GPU ops)
instead of a Python for-loop. The loss reduction is arithmetically identical to
the sequential version (see the equivalence test).
"""

import os
import random
import torch

from Pipelines.Pipeline_ERTS import Pipeline_ERTS, device
from RTSNet.RTSNet_nn import RTSNetNN
from RTSNet.RTSNet_nn_batched import make_batched_from
from RTSNet.RTSNet_nn_with_F import RTSNetNN as RTSNetNN_with_F
from RTSNet.RTSNet_nn_with_F_batched import make_batched_from_F


class Pipeline_ERTS_batched(Pipeline_ERTS):

    # --------------------------------------------------------------- helpers
    def _to_batched_model(self, batch_size):
        """Re-class self.model to the batched twin (shares the same weights)."""
        self.model = make_batched_from(self.model, batch_size)

    def _to_batched_model_F(self, batch_size):
        """Re-class self.model to the F-architecture batched twin (RTSNet_nn_with_F).
        Use this instead of _to_batched_model when the model embeds F (FC8/FC_F_bw)
        rather than H (FC9/FC_H_bw) -- e.g. exp3's train_F_..._joint_batched."""
        self.model = make_batched_from_F(self.model, batch_size)

    def _save_sequential(self, model, SysModel, path):
        """Save a checkpoint as a plain (sequential) RTSNetNN so existing test
        scripts load it unchanged. The trained weights are identical; only the
        class/forward wrapper differs. Falls back to a direct save if the
        architecture args aren't available (e.g. minimal test harnesses)."""
        args = getattr(self, "args", None)
        if args is None:
            torch.save(model, path)
            return
        seq = RTSNetNN()
        seq.NNBuild(SysModel, args)
        seq.to(self.device)
        seq.load_state_dict(model.state_dict())
        torch.save(seq, path)

    def _smooth_batch(self, y_batch, T, m, B):
        """Forward filter + backward smoother for a [B, n, T] batch.
        Returns x_smooth [B, m, T]. Assumes update_H / InitSequence / init_hidden
        and batch_size have already been set by the caller."""
        x_fwd = torch.empty(B, m, T, device=y_batch.device, dtype=y_batch.dtype)
        for t in range(T):
            x_fwd[:, :, t] = self.model(y_batch[:, :, t], None, None, None)

        x_smooth = torch.empty(B, m, T, device=y_batch.device, dtype=y_batch.dtype)
        x_smooth[:, :, T - 1] = x_fwd[:, :, T - 1]
        self.model.InitBackward(x_smooth[:, :, T - 1])
        x_smooth[:, :, T - 2] = self.model(None, x_fwd[:, :, T - 2], x_fwd[:, :, T - 1], None)
        for t in range(T - 3, -1, -1):
            x_smooth[:, :, t] = self.model(None, x_fwd[:, :, t], x_fwd[:, :, t + 1], x_smooth[:, :, t + 2])
        return x_smooth

    @staticmethod
    def _stack_H(H_list_for_dataset, n_e_list, device):
        # H_list_for_dataset[h_index] -> [n, m]; stack over the batch
        return torch.stack(
            [H_list_for_dataset[n_e // 10] for n_e in n_e_list], dim=0
        ).to(device)

    @staticmethod
    def _stack_idx(lst, indices, device):
        # lst[i] -> [.,.]; stack over the batch using explicit indices
        return torch.stack([lst[i] for i in indices], dim=0).to(device)

    @staticmethod
    def _stack_seq(dataset_tensor, n_e_list, device):
        # dataset_tensor[n_e] -> [d, T]; stack over the batch -> [B, d, T]
        return torch.stack([dataset_tensor[n_e] for n_e in n_e_list], dim=0).to(device)

    # ===================================================================== #
    #  Batched twin of train_RTS_net_3_datasets                             #
    # ===================================================================== #
    def train_RTS_net_3_datasets_batched(self, SysModel, cv_input, cv_target, train_input,
                                         train_target, destination_path_RTS, load_path_RTS,
                                         H_init=None, datasets=3):
        """Batched version of train_RTS_net_3_datasets.

        Parallelizes the N_B sample loop; keeps the (sequential) dataset loop and
        time loop. Loss reduction matches the sequential function exactly.
        """
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m

        self.MSE_cv_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_linear_epoch = torch.empty([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch = torch.empty([self.N_steps], device=self.device)

        if load_path_RTS is not None:
            print("loading model_and keep training them")
            self.model = torch.load(load_path_RTS, map_location=self.device,
                                    weights_only=False).to(self.device)

        # Re-class to batched twin and (re)link the optimizer to its parameters.
        self._to_batched_model(self.N_B)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                          weight_decay=self.weightDecay)

        self.model.train()
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        nan_streak = 0

        for ti in range(0, self.N_steps):
            self.model.train()
            self.model.set_batch_size(self.N_B)
            self.optimizer.zero_grad()

            # Draw n_e exactly as the sequential loop does (N_B draws, same order).
            n_e_list = [random.randint(0, self.N_E - 1) for _ in range(self.N_B)]

            x_0 = SysModel.m1x_0.clone().detach().to(self.device)  # [m,1] -> broadcast in InitSequence
            total_loss = 0.0
            dataset_losses = [0.0] * datasets

            for data in range(datasets):
                if H_init is None:
                    H_batch = self._stack_H(SysModel.H_train[data], n_e_list, self.device)
                else:
                    # shared H_init for every sequence -> expand to [B,n,m]
                    H_batch = H_init.to(self.device)
                    if H_batch.dim() == 2:
                        H_batch = H_batch.unsqueeze(0).expand(self.N_B, -1, -1).contiguous()
                y_batch = self._stack_seq(train_input[data], n_e_list, self.device)   # [B,n,T]
                x_target = self._stack_seq(train_target[data], n_e_list, self.device)  # [B,m,T]
                T = y_batch.size(-1)

                self.model.update_H(H_batch)
                self.model.InitSequence(x_0, T)
                self.model.init_hidden()

                x_smooth = self._smooth_batch(y_batch, T, m, self.N_B)

                rtsnet_loss = self.loss_fn(x_smooth, x_target)
                dataset_losses[data] += rtsnet_loss.detach().item()
                total_loss = total_loss + rtsnet_loss
                x_0 = x_smooth[:, :, -1].detach()  # [B,m] continuity across datasets

            loss = total_loss / datasets   # == sequential Batch_Optimizing_LOSS_mean

            loss_msg = " ".join([f"loss_d{k}={dataset_losses[k]:.6f}" for k in range(datasets)])
            print(f"[epoch {ti:03d}] {loss_msg} loss_all={loss.item():.6f}")

            loss.backward()

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
                continue
            nan_streak = 0

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.MSE_train_linear_epoch[ti] = loss.detach()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################
            self.model.eval()
            with torch.no_grad():
                self.model.set_batch_size(self.N_CV)
                cv_idx = list(range(self.N_CV))

                x_0_cv = SysModel.m1x_0.clone().detach().to(self.device)
                cv_total = 0.0
                cv_dataset_losses = [0.0] * datasets
                for data in range(datasets):
                    if H_init is None:
                        H_batch_cv = self._stack_H(SysModel.H_valid[data], cv_idx, self.device)
                    else:
                        H_batch_cv = H_init.to(self.device)
                        if H_batch_cv.dim() == 2:
                            H_batch_cv = H_batch_cv.unsqueeze(0).expand(self.N_CV, -1, -1).contiguous()
                    y_cv = self._stack_seq(cv_input[data], cv_idx, self.device)
                    x_cv_target = self._stack_seq(cv_target[data], cv_idx, self.device)
                    T_cv = y_cv.size(-1)

                    self.model.update_H(H_batch_cv)
                    self.model.InitSequence(x_0_cv, T_cv)
                    self.model.init_hidden()

                    x_s_cv = self._smooth_batch(y_cv, T_cv, m, self.N_CV)

                    cv_loss_curr = self.loss_fn(x_s_cv, x_cv_target)
                    cv_dataset_losses[data] += cv_loss_curr.item()
                    cv_total = cv_total + cv_loss_curr
                    x_0_cv = x_s_cv[:, :, -1].detach()

                self.MSE_cv_linear_epoch[ti] = (cv_total / datasets).detach()
                cv_msg = " ".join([f"cv_d{k}={cv_dataset_losses[k]:.6f}" for k in range(datasets)])
                print(f"[epoch {ti:03d}] {cv_msg} cv_all={self.MSE_cv_linear_epoch[ti].item():.6f}")

                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    self._save_sequential(self.model, SysModel, destination_path_RTS)

            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]",
                  "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]")
            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch,
                self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    # ===================================================================== #
    #  Batched twin of train_H_mstep_net_3_datasets_joint                   #
    # ===================================================================== #
    def _mstep_batch_setup_H(self, SysModel, idx_list, H_init, generate_h):
        """Initial per-sample H_current [B,n,m] (set once, before the dataset loop)."""
        B = len(idx_list)
        if H_init is None:
            if generate_h:
                return self._stack_idx(SysModel.H_train[0], [i // 10 for i in idx_list], self.device).clone().detach()
            return self._stack_idx(SysModel.H_train[0], idx_list, self.device).clone().detach()
        H = H_init.to(self.device)
        if H.dim() == 2:
            H = H.unsqueeze(0).expand(B, -1, -1)
        return H.clone().detach()

    @staticmethod
    def _em_stats(y_curr, x_curr, H_current, T):
        """Batched EM statistics. y_curr [B,n,T], x_curr [B,m,T], H_current [B,n,m].
        Mirrors the sequential per-sample stats exactly."""
        xt = x_curr.transpose(1, 2)                              # [B,T,m]
        A_yx = torch.bmm(y_curr, xt) / T                         # [B,n,m]
        A_xx = torch.bmm(x_curr, xt) / T                         # [B,m,m]
        Hx = torch.bmm(H_current, x_curr)                        # [B,n,T]
        nu = y_curr - Hx                                         # [B,n,T]
        nu_mean = nu.mean(dim=2, keepdim=True)                   # [B,n,1] (over time)
        nu_c = nu - nu_mean
        S_nu = torch.bmm(nu_c, nu_c.transpose(1, 2)) / T         # [B,n,n]
        C_nu_x = torch.bmm(nu, xt) / T                           # [B,n,m]
        return A_yx, A_xx, S_nu, C_nu_x

    def train_H_mstep_net_3_datasets_joint_batched(
            self, SysModel, cv_input, cv_target, train_input, train_target,
            destination_path_M, destination_path_RTS, load_path_RTS, load_mnet,
            num_em_iters=3, H_init=None, alpha=(0.05, 0.1, 0.85), lambda_H=1e-3,
            generate_h=True, datasets=3, x_0_train_list=None, x_0_cv_list=None):
        """Batched version of train_H_mstep_net_3_datasets_joint.

        Parallelizes the N_B sample loop; keeps EM / dataset / time loops sequential.
        Loss reduction matches the sequential function exactly.
        """
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m
        n = SysModel.n

        # Load RTSNet (trainable) and re-class to batched twin.
        self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self._to_batched_model(self.N_B)

        # M-step network (already batch-native).
        self.M_model_H = torch.load(load_mnet, weights_only=False).to(self.device)
        model_mstep = self.M_model_H.train()

        self.optimizer_joint = torch.optim.Adam(
            list(self.model.parameters()) + list(model_mstep.parameters()),
            lr=self.learningRate)

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        weights = [alpha[0] if k == 0 else alpha[1] if k == 1 else alpha[2]
                   for k in range(num_em_iters)]

        def smooth(B, y_batch, H_cur, x0, T):
            self.model.update_H(H_cur)
            self.model.InitSequence(x0, T)
            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(self.device)
            self.model.init_hidden()
            return self._smooth_batch(y_batch, T, m, B)

        for epoch in range(self.N_steps):
            model_mstep.train()
            self.model.train()
            self.model.set_batch_size(self.N_B)
            self.optimizer_joint.zero_grad()

            n_e_list = [random.randint(0, self.N_E - 1) for _ in range(self.N_B)]
            B = self.N_B

            # logging accumulators (averaged over datasets at the end, == seq /denom)
            log_x_start = 0.0
            log_x_em = [0.0] * num_em_iters
            log_h_em = [0.0] * num_em_iters
            log_reg_em = [0.0] * num_em_iters

            if x_0_train_list is not None:
                x_0 = self._stack_idx(x_0_train_list, n_e_list, self.device).reshape(B, m)
            else:
                x_0 = SysModel.m1x_0.clone().detach().to(self.device)  # broadcast in InitSequence

            H_current = self._mstep_batch_setup_H(SysModel, n_e_list, H_init, generate_h)

            total = 0.0
            for data in range(datasets):
                y_batch = self._stack_seq(train_input[data], n_e_list, self.device)   # [B,n,T]
                x_true = self._stack_seq(train_target[data], n_e_list, self.device)   # [B,m,T]
                T = y_batch.size(-1)
                if generate_h:
                    H_true = self._stack_idx(SysModel.H_train_TRUE[data], [i // 10 for i in n_e_list], self.device)
                else:
                    H_true = self._stack_idx(SysModel.H_train_TRUE[data], n_e_list, self.device)

                x_curr = smooth(B, y_batch, H_current, x_0, T)
                y_curr = y_batch
                log_x_start += torch.mean((x_curr - x_true) ** 2).item()

                for em_iter in range(num_em_iters):
                    A_yx, A_xx, S_nu, C_nu_x = self._em_stats(y_curr, x_curr, H_current, T)
                    z_in = torch.cat([
                        A_yx.reshape(B, -1).detach(),
                        A_xx.reshape(B, -1).detach(),
                        S_nu.reshape(B, -1).detach(),
                        C_nu_x.reshape(B, -1).detach(),
                        H_current.reshape(B, -1).detach(),
                    ], dim=1)

                    deltaH = model_mstep(z_in)                  # [B,n,m]
                    H_current = H_current + deltaH
                    h_loss = torch.mean((H_current - H_true) ** 2)
                    reg = lambda_H * torch.mean(deltaH ** 2)
                    log_h_em[em_iter] += h_loss.item()
                    log_reg_em[em_iter] += reg.item()

                    x_curr = smooth(B, y_curr, H_current, x_0, T)
                    x_loss = torch.mean((x_curr - x_true) ** 2)
                    log_x_em[em_iter] += x_loss.item()

                    loss_em = 2 * h_loss + reg + x_loss
                    total = total + weights[em_iter] * loss_em

                x_0 = x_curr[:, :, -1].detach()

            loss = total / float(datasets)   # == sequential batch_total_loss / N_B
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(model_mstep.parameters()), max_norm=1.5)
            self.optimizer_joint.step()

            avg_x_start = log_x_start / datasets
            avg_x_em = [v / datasets for v in log_x_em]
            avg_h_em = [v / datasets for v in log_h_em]
            avg_reg_em = [v / datasets for v in log_reg_em]
            em_msg = " ".join([f"x{k}={avg_x_em[k]:.4f} h{k}={avg_h_em[k]:.4f} reg{k}={avg_reg_em[k]:.4f}"
                               for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] x_loss_start={avg_x_start:.6f} {em_msg} loss_all={loss.item():.6f}")

            # ----------------------------- Validation -----------------------------
            model_mstep.eval()
            self.model.eval()
            self.model.set_batch_size(self.N_CV)
            Bcv = self.N_CV
            cv_idx = list(range(self.N_CV))
            log_cv_x_start = 0.0
            log_cv_x_em = [0.0] * num_em_iters
            log_cv_h_em = [0.0] * num_em_iters
            log_cv_reg_em = [0.0] * num_em_iters
            with torch.no_grad():
                if x_0_cv_list is not None:
                    x_0_cv = self._stack_idx(x_0_cv_list, cv_idx, self.device).reshape(Bcv, m)
                else:
                    x_0_cv = SysModel.m1x_0.clone().detach().to(self.device)

                H_cur_cv = self._mstep_batch_setup_H_valid(SysModel, cv_idx, H_init, generate_h)

                cv_total = 0.0
                for data in range(datasets):
                    y_cv = self._stack_seq(cv_input[data], cv_idx, self.device)
                    x_true_cv = self._stack_seq(cv_target[data], cv_idx, self.device)
                    T_cv = y_cv.size(-1)
                    if generate_h:
                        H_true_cv = self._stack_idx(SysModel.H_valid_TRUE[data], [j // 10 for j in cv_idx], self.device)
                    else:
                        H_true_cv = self._stack_idx(SysModel.H_valid_TRUE[data], cv_idx, self.device)

                    x_curr = smooth(Bcv, y_cv, H_cur_cv, x_0_cv, T_cv)
                    y_curr = y_cv
                    log_cv_x_start += torch.mean((x_curr - x_true_cv) ** 2).item()

                    for em_iter in range(num_em_iters):
                        A_yx, A_xx, S_nu, C_nu_x = self._em_stats(y_curr, x_curr, H_cur_cv, T_cv)
                        z_cv = torch.cat([
                            A_yx.reshape(Bcv, -1), A_xx.reshape(Bcv, -1), S_nu.reshape(Bcv, -1),
                            C_nu_x.reshape(Bcv, -1), H_cur_cv.reshape(Bcv, -1)], dim=1)
                        dH_cv = model_mstep(z_cv)
                        H_cur_cv = H_cur_cv + dH_cv
                        h_loss_cv = torch.mean((H_cur_cv - H_true_cv) ** 2)
                        reg_cv = lambda_H * torch.mean(dH_cv ** 2)
                        log_cv_h_em[em_iter] += h_loss_cv.item()
                        log_cv_reg_em[em_iter] += reg_cv.item()

                        x_curr = smooth(Bcv, y_curr, H_cur_cv, x_0_cv, T_cv)
                        x_loss_cv = torch.mean((x_curr - x_true_cv) ** 2)
                        log_cv_x_em[em_iter] += x_loss_cv.item()
                        cv_total = cv_total + weights[em_iter] * (2 * h_loss_cv + reg_cv + x_loss_cv)

                    x_0_cv = x_curr[:, :, -1].detach()
                    H_cur_cv = H_cur_cv.detach()

                cv_epoch = (cv_total / float(datasets)).item()

            avg_cv_x_start = log_cv_x_start / datasets
            avg_cv_x_em = [v / datasets for v in log_cv_x_em]
            avg_cv_h_em = [v / datasets for v in log_cv_h_em]
            avg_cv_reg_em = [v / datasets for v in log_cv_reg_em]
            cv_em_msg = " ".join([f"cv_x{k}={avg_cv_x_em[k]:.6f} cv_h{k}={avg_cv_h_em[k]:.6f} cv_reg{k}={avg_cv_reg_em[k]:.6f}"
                                  for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] cv_x_loss_start={avg_cv_x_start:.6f} {cv_em_msg} cv_all={cv_epoch:.6f}")
            print(f"BEST: epoch={self.MSE_cv_idx_opt}  best_cv_loss={self.MSE_cv_dB_opt:.6f}")
            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)
                self._save_sequential(self.model, SysModel, destination_path_RTS)

    def _mstep_batch_setup_H_valid(self, SysModel, idx_list, H_init, generate_h):
        """Initial per-sample validation H_current [B,n,m]."""
        B = len(idx_list)
        if H_init is None:
            if generate_h:
                return self._stack_idx(SysModel.H_valid[0], [i // 10 for i in idx_list], self.device).clone().detach()
            return self._stack_idx(SysModel.H_valid[0], idx_list, self.device).clone().detach()
        H = H_init.to(self.device)
        if H.dim() == 2:
            H = H.unsqueeze(0).expand(B, -1, -1)
        return H.clone().detach()

    # ===================================================================== #
    #  Batched twin of joint_train_mnet_rtsnet_3_datasets  (F estimation)    #
    #  Same structure as train_H_mstep_net_3_datasets_joint_batched, but the #
    #  drifting matrix is F (state transition), not H (observation). Reuses  #
    #  all the batched machinery; only the M-net, update_, EM-stats change.  #
    # ===================================================================== #
    @staticmethod
    def _x0_batch(x0, B, m):
        """Coerce an init state to [B, m] (broadcast a shared [m]/[m,1])."""
        if x0.dim() == 2 and x0.shape[-1] == 1:
            return x0.view(-1).unsqueeze(0).expand(B, m).contiguous()
        if x0.dim() == 1:
            return x0.unsqueeze(0).expand(B, m).contiguous()
        return x0  # already [B, m]

    @staticmethod
    def _em_stats_F(y_curr, x_curr, F_current, x0b, h_fn, T):
        """Batched EM statistics for F estimation. Mirrors the per-sample stats in
        joint_train_mnet_rtsnet_3_datasets exactly.
          y_curr [B,n,T], x_curr [B,m,T], F_current [B,m,m], x0b [B,m].
        h_fn maps [B,m] -> [B,n] (batched observation model; linear or non-linear).
        Returns A1,A2,S_delta_x,S_nu,C  ([B,m,m]/[B,n,n]) and delta_x [B,m,T]."""
        x_prev = torch.empty_like(x_curr)
        x_prev[:, :, 0] = x0b                       # x_{-1} = incoming x_0
        x_prev[:, :, 1:] = x_curr[:, :, :-1]        # x_{t-1|T}
        xpt = x_prev.transpose(1, 2)                # [B,T,m]
        A1 = torch.bmm(x_curr, xpt) / T             # [B,m,m]
        A2 = torch.bmm(x_prev, xpt) / T             # [B,m,m]
        x_minus = torch.bmm(F_current, x_prev)      # [B,m,T]  (F @ x_{t-1})
        delta_x = x_curr - x_minus                  # [B,m,T]
        delta_mean = delta_x.mean(dim=2, keepdim=True)
        delta_c = delta_x - delta_mean
        S_delta_x = torch.bmm(delta_c, delta_c.transpose(1, 2)) / T   # [B,m,m]
        Hx = torch.stack([h_fn(x_curr[:, :, t]) for t in range(T)], dim=2)  # [B,n,T]
        nu = y_curr - Hx
        nu_mean = nu.mean(dim=2, keepdim=True)
        nu_c = nu - nu_mean
        S_nu = torch.bmm(nu_c, nu_c.transpose(1, 2)) / T             # [B,n,n]
        C = torch.bmm(delta_x, xpt) / T                             # [B,m,m]
        return A1, A2, S_delta_x, S_nu, C, delta_x

    def _save_sequential_as(self, model, SysModel, path, seq_class):
        """Save the (batched) trained RTSNet as a plain sequential `seq_class`
        instance so existing (F-aware) exp3 test scripts torch.load it unchanged.
        Only weights are copied; falls back to a direct save if args are absent."""
        args = getattr(self, "args", None)
        if args is None:
            torch.save(model, path)
            return
        seq = seq_class()
        seq.NNBuild(SysModel, args)
        seq.to(self.device)
        seq.load_state_dict(model.state_dict())
        # NNBuild set seq.f = SysModel.f; for make_f(F) that is an unpicklable LOCAL
        # closure (f_func) -> torch.save fails. Replace it with the picklable bound
        # f_new via update_F (a placeholder F -- exp3 test scripts call update_F
        # before use, exactly like the sequential joint trainer's saved checkpoints).
        try:
            seq.update_F(torch.eye(seq.m, device=self.device))
        except Exception:
            pass
        torch.save(seq, path)

    def train_F_mstep_net_3_datasets_joint_batched(
            self, SysModel, cv_input, cv_target, train_input, train_target,
            destination_path_M, destination_path_RTS, load_path_RTS, load_mnet=None,
            num_em_iters=3, F_init=None, alpha=(0.05, 0.1, 0.85), lambda_F=1e-3,
            generate_f=True, non_linear_h=False, h_batched=None, datasets=3):
        """Batched, warm-started F twin of train_H_mstep_net_3_datasets_joint_batched.

        JOINTLY trains ONE RTSNet + ONE F-M-net across `datasets` sequential
        datasets. Parallelizes the N_B sample loop with torch.bmm; the EM /
        dataset / time loops stay sequential. Per-batch loss reduction matches the
        sequential joint_train_mnet_rtsnet_3_datasets (total / (datasets * num_em_iters)).

          - RTSNet warm-started from load_path_RTS (trainable),
          - F-M-net (DeltaF_MStepNet) warm-started from load_mnet (else pipeline default),
          - F_init is the shared WRONG F every sequence starts from (default the
            nominal [[0.83,0.2],[0.2,0.83]] the sequential trainer hardcodes),
          - non_linear_h=True requires h_batched: a callable [B,m]->[B,n] (the
            sequential SysModel.h is NOT batch-aware; see exp3_train.py).
        """
        self.N_E = len(train_input[0])
        self.N_CV = len(cv_input[0])
        m = SysModel.m

        # Warm-start the RTSNet from load_path_RTS if it exists; otherwise fall back
        # to the pipeline's freshly-built RTSNet (self.model) and train from scratch.
        # Mirrors the graceful warm-start handling in the nongauss script
        # (_warm(...) -> None). Capture the on-disk class first so we can save a plain
        # sequential ckpt that existing exp3 test scripts load unchanged.
        if load_path_RTS is not None and os.path.exists(load_path_RTS):
            self.model = torch.load(load_path_RTS, weights_only=False).to(self.device).train()
        else:
            if load_path_RTS is not None:
                print(f"[warm start] RTSNet '{load_path_RTS}' not found -> training RTSNet from scratch.")
            self.model = self.model.to(self.device).train()
        # Reconstruct the saved checkpoint with the F architecture explicitly, NOT
        # type(self.model): the exp3 checkpoints are pickled under the plain-H class
        # NAME (RTSNet.RTSNet_nn.RTSNetNN) while carrying F layers (FC8/FC_F_bw), so
        # without a class alias type() reports the wrong (H) class and NNBuild would
        # rebuild the wrong architecture at save time.
        seq_class = RTSNetNN_with_F
        for p in self.model.parameters():
            p.requires_grad_(True)
        self._to_batched_model_F(self.N_B)   # F-architecture twin (FC8/FC_F_bw)

        # F-M-net (batch-native). Warm-start from load_mnet if it exists, else the
        # pipeline default DeltaF_MStepNet (self.M_model).
        if load_mnet is not None and os.path.exists(load_mnet):
            self.M_model = torch.load(load_mnet, weights_only=False).to(self.device)
        elif load_mnet is not None:
            print(f"[warm start] M-net '{load_mnet}' not found -> using fresh DeltaF_MStepNet.")
        model_mstep = self.M_model.train()

        self.optimizer_joint = torch.optim.Adam(
            list(self.model.parameters()) + list(model_mstep.parameters()),
            lr=self.learningRate, weight_decay=self.weightDecay)

        # Shared wrong F everyone starts from (broadcast to [B,m,m] per epoch).
        if F_init is None:
            F_base = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=self.device)
        else:
            F_base = F_init.to(self.device)

        # Batched observation model h_fn: [B,m] -> [B,n].
        if non_linear_h:
            if h_batched is None:
                raise ValueError(
                    "non_linear_h=True needs a batched h (SysModel.h is not "
                    "batch-aware). Pass h_batched: callable([B,m])->[B,n].")
            h_fn = h_batched
        else:
            H_lin = SysModel.H.to(self.device)
            h_fn = lambda xb: xb @ H_lin.T          # [B,m] @ [m,n] -> [B,n]
        self.model.h = h_fn                          # used by the batched forward filter

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        def smooth_F(B, y_batch, F_cur, x0, T):
            self.model.update_F(F_cur)
            self.model.h = h_fn
            self.model.InitSequence(x0, T)
            self.model.prior_Sigma = SysModel.m2x_0.clone().detach().to(self.device)
            self.model.init_hidden()
            return self._smooth_batch(y_batch, T, m, B)

        denom = float(datasets * num_em_iters)

        for epoch in range(self.N_steps):
            model_mstep.train(); self.model.train()
            self.model.set_batch_size(self.N_B)
            self.optimizer_joint.zero_grad()

            n_e_list = [random.randint(0, self.N_E - 1) for _ in range(self.N_B)]
            B = self.N_B

            log_f_em = [0.0] * num_em_iters
            log_x_em = [0.0] * num_em_iters

            x_0 = SysModel.m1x_0.clone().detach().to(self.device)
            F_base_b = F_base.unsqueeze(0).expand(B, m, m).contiguous()

            total = 0.0
            for data in range(datasets):
                y_batch = self._stack_seq(train_input[data], n_e_list, self.device)   # [B,n,T]
                x_true = self._stack_seq(train_target[data], n_e_list, self.device)   # [B,m,T]
                T = y_batch.size(-1)
                if generate_f:
                    F_true = self._stack_idx(SysModel.F_train_TRUE[data], [i // 10 for i in n_e_list], self.device)
                else:
                    F_true = self._stack_idx(SysModel.F_train_TRUE[data], n_e_list, self.device)

                x0b = self._x0_batch(x_0, B, m)
                F_current = F_base_b
                x_curr = None
                for em_iter in range(num_em_iters):
                    x_curr = smooth_F(B, y_batch, F_current, x_0, T)          # [B,m,T]
                    A1, A2, S_dx, S_nu, C, _ = self._em_stats_F(
                        y_batch, x_curr, F_current, x0b, h_fn, T)
                    z_in = torch.cat([
                        A1.reshape(B, -1).detach(), A2.reshape(B, -1).detach(),
                        S_dx.reshape(B, -1).detach(), S_nu.reshape(B, -1).detach(),
                        C.reshape(B, -1).detach(), F_current.reshape(B, -1).detach(),
                    ], dim=1)

                    deltaF = model_mstep(z_in)                                # [B,m,m]
                    F_next = F_current + deltaF
                    f_loss = torch.mean((F_next - F_true) ** 2)
                    reg = lambda_F * torch.mean(deltaF ** 2)
                    x_loss = torch.mean((x_curr - x_true) ** 2)
                    weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                    total = total + weight * (f_loss + reg + x_loss)
                    log_f_em[em_iter] += f_loss.item()
                    log_x_em[em_iter] += x_loss.item()
                    F_current = F_next

                x_0 = x_curr[:, :, -1].detach()          # [B,m] continuity across datasets
                F_base_b = F_current.detach()            # carry F across datasets

            loss = total / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(model_mstep.parameters()), max_norm=1.0)
            self.optimizer_joint.step()

            em_msg = " ".join([f"f{k}={log_f_em[k] / datasets:.4f} x{k}={log_x_em[k] / datasets:.4f}"
                               for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] {em_msg} loss_all={loss.item():.6f}")

            # ----------------------------- Validation -----------------------------
            model_mstep.eval(); self.model.eval()
            self.model.set_batch_size(self.N_CV)
            Bcv = self.N_CV
            cv_idx = list(range(self.N_CV))
            log_cv_f_em = [0.0] * num_em_iters
            log_cv_x_em = [0.0] * num_em_iters
            with torch.no_grad():
                x_0_cv = SysModel.m1x_0.clone().detach().to(self.device)
                F_base_cv = F_base.unsqueeze(0).expand(Bcv, m, m).contiguous()
                cv_total = 0.0
                for data in range(datasets):
                    y_cv = self._stack_seq(cv_input[data], cv_idx, self.device)
                    x_true_cv = self._stack_seq(cv_target[data], cv_idx, self.device)
                    T_cv = y_cv.size(-1)
                    if generate_f:
                        F_true_cv = self._stack_idx(SysModel.F_valid_TRUE[data], [j // 10 for j in cv_idx], self.device)
                    else:
                        F_true_cv = self._stack_idx(SysModel.F_valid_TRUE[data], cv_idx, self.device)

                    x0b_cv = self._x0_batch(x_0_cv, Bcv, m)
                    F_cur_cv = F_base_cv
                    x_curr = None
                    for em_iter in range(num_em_iters):
                        x_curr = smooth_F(Bcv, y_cv, F_cur_cv, x_0_cv, T_cv)
                        A1, A2, S_dx, S_nu, C, _ = self._em_stats_F(
                            y_cv, x_curr, F_cur_cv, x0b_cv, h_fn, T_cv)
                        z_cv = torch.cat([
                            A1.reshape(Bcv, -1), A2.reshape(Bcv, -1), S_dx.reshape(Bcv, -1),
                            S_nu.reshape(Bcv, -1), C.reshape(Bcv, -1), F_cur_cv.reshape(Bcv, -1)], dim=1)
                        dF_cv = model_mstep(z_cv)
                        F_next_cv = F_cur_cv + dF_cv
                        f_loss_cv = torch.mean((F_next_cv - F_true_cv) ** 2)
                        reg_cv = lambda_F * torch.mean(dF_cv ** 2)
                        x_loss_cv = torch.mean((x_curr - x_true_cv) ** 2)
                        weight = alpha[em_iter] if em_iter < len(alpha) else alpha[-1]
                        cv_total = cv_total + weight * (f_loss_cv + reg_cv + x_loss_cv)
                        log_cv_f_em[em_iter] += f_loss_cv.item()
                        log_cv_x_em[em_iter] += x_loss_cv.item()
                        F_cur_cv = F_next_cv

                    x_0_cv = x_curr[:, :, -1].detach()
                    F_base_cv = F_cur_cv.detach()

                cv_epoch = (cv_total / denom).item()

            cv_em_msg = " ".join([f"cv_f{k}={log_cv_f_em[k] / datasets:.4f} cv_x{k}={log_cv_x_em[k] / datasets:.4f}"
                                  for k in range(num_em_iters)])
            print(f"[epoch {epoch:03d}] {cv_em_msg} cv_all={cv_epoch:.6f}")
            print(f"BEST: epoch={self.MSE_cv_idx_opt}  best_cv_loss={self.MSE_cv_dB_opt:.6f}")

            if cv_epoch < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = cv_epoch
                self.MSE_cv_idx_opt = epoch
                torch.save(model_mstep, destination_path_M)
                self._save_sequential_as(self.model, SysModel, destination_path_RTS, seq_class)
