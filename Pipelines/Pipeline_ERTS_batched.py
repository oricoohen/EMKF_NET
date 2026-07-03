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

import random
import torch

from Pipelines.Pipeline_ERTS import Pipeline_ERTS, device
from RTSNet.RTSNet_nn import RTSNetNN
from RTSNet.RTSNet_nn_batched import make_batched_from


class Pipeline_ERTS_batched(Pipeline_ERTS):

    # --------------------------------------------------------------- helpers
    def _to_batched_model(self, batch_size):
        """Re-class self.model to the batched twin (shares the same weights)."""
        self.model = make_batched_from(self.model, batch_size)

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
