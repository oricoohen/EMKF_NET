import time

import torch
import torch.nn as nn
import torch.optim as optim
import math


class BiGRUSmoother(nn.Module):
    def __init__(self, n, m, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=n,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, m)
        )

    def forward(self, y):
        """
        y shape: [N, n, T]
        returns x_hat shape: [N, m, T]
        """

        y = y.permute(0, 2, 1)          # [N, T, n]
        out, _ = self.bigru(y)          # [N, T, 2*hidden]
        x_hat = self.head(out)          # [N, T, m]
        x_hat = x_hat.permute(0, 2, 1)  # [N, m, T]

        return x_hat


def train_bigru_smoother(
    train_input,
    train_target,
    cv_input,
    cv_target,
    n,
    m,
    save_path,
    device,
    epochs=300,
    batch_size=32,
    lr=1e-3,
    hidden_size=128,
    num_layers=2
):
    import os
    import math
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if isinstance(train_input, list):
        train_input = torch.cat(train_input, dim=0)
        train_target = torch.cat(train_target, dim=0)

    if isinstance(cv_input, list):
        cv_input = torch.cat(cv_input, dim=0)
        cv_target = torch.cat(cv_target, dim=0)

    train_input = train_input.to(device)
    train_target = train_target.to(device)
    cv_input = cv_input.to(device)
    cv_target = cv_target.to(device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = BiGRUSmoother(
        n=n,
        m=m,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    N_train = train_input.shape[0]
    best_cv_loss = float("inf")
    best_epoch = -1

    print("BiGRU train:", train_input.shape, train_target.shape)
    print("BiGRU cv:", cv_input.shape, cv_target.shape)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N_train, device=device)

        train_loss_sum = 0.0
        batch_count = 0

        for start in range(0, N_train, batch_size):
            idx = perm[start:start + batch_size]

            y_batch = train_input[idx]      # [B,n,T]
            x_batch = train_target[idx]     # [B,m,T]

            optimizer.zero_grad()

            x_hat = model(y_batch)          # [B,m,T]
            loss = loss_fn(x_hat, x_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            batch_count += 1

        train_loss = train_loss_sum / batch_count

        model.eval()
        with torch.no_grad():
            x_cv_hat = model(cv_input)
            cv_loss = loss_fn(x_cv_hat, cv_target).item()

        print(
            f"[BiGRU {epoch:03d}] "
            f"train={10.0 * math.log10(train_loss):.2f} dB, "
            f"cv={10.0 * math.log10(cv_loss):.2f} dB"
        )

        if cv_loss < best_cv_loss:
            best_cv_loss = cv_loss
            best_epoch = epoch
            torch.save(model, save_path)

    print(
        f"Best BiGRU: epoch={best_epoch}, "
        f"cv={10.0 * math.log10(best_cv_loss):.2f} dB"
    )

    return model


def test_bigru_smoother(
    test_input,
    test_target,
    load_path,
    device,
    seq_by_seq=True,
    return_time=False
):
    """
    test_input can be:
      - tensor [N, n, T]
      - or list of datasets, each [N, n, T]

    seq_by_seq : DEFAULT True -- run ONE sequence per forward pass instead of
        feeding the whole [N, n, T] batch at once. This is the default so that
        EVERY script that evaluates the BiGRU measures it the same way, and on
        the same footing as EMKF / ERTS / RTSNet, which all loop sequence by
        sequence (see S_Test_ext_H, EMKF_H_analitic_f_nonlinear, NNTest,
        test_H_mstep_net). The batched path amortizes one kernel-launch
        sequence over all N sequences and therefore looks ~N x faster for
        implementation reasons only. The estimates are mathematically identical
        either way (a GRU forward is independent across the batch dimension);
        only the wall-time changes. Pass seq_by_seq=False if you explicitly
        want batched throughput.
    return_time : if True, append the measured inference wall-time (seconds,
        forward passes only -- torch.load and the MSE reduction are excluded)
        to the returned tuple. Default False keeps the historical 3-tuple, so
        existing callers are unaffected.
    """

    if isinstance(test_input, list):
        test_input = torch.cat(test_input, dim=0)
        test_target = torch.cat(test_target, dim=0)

    test_input = test_input.to(device)
    test_target = test_target.to(device)

    model = torch.load(load_path, weights_only=False).to(device)
    model.eval()

    _is_cuda = torch.device(device).type == 'cuda'

    def _sync():
        if _is_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        # Always warm up: the first forward pays cuDNN autotune / lazy CUDA
        # init, which is the main reason repeated runs report different times.
        model(test_input[:1])
        _sync()
        _t0 = time.perf_counter()

        if seq_by_seq:
            # one sequence per forward -> per-sequence latency
            x_hat = torch.cat([model(test_input[i:i + 1])
                               for i in range(test_input.shape[0])], dim=0)
        else:
            x_hat = model(test_input)

        _sync()
        elapsed = time.perf_counter() - _t0

        print("test_input shape =", test_input.shape)
        print("test_target shape =", test_target.shape)
        print("x_hat shape =", x_hat.shape)

        assert x_hat.shape == test_target.shape, \
            f"Shape mismatch: x_hat={x_hat.shape}, target={test_target.shape}"

        err = (x_hat - test_target) ** 2

        mse_all = err.mean()
        mse_per_dim = err.mean(dim=(0, 2))  # [m]
        mse_per_t = err.mean(dim=(0, 1))  # [T]
        mse_per_seq = err.mean(dim=(1, 2))  # [N]
        mse_db = 10 * torch.log10(mse_all).item()
        print("MSE all =", mse_all.item(), "dB =", 10 * torch.log10(mse_all).item())
        # print("MSE per state dim =", mse_per_dim)
        # print("MSE per state dim dB =", 10 * torch.log10(mse_per_dim))
        # print("MSE first seq =", mse_per_seq[0].item())
        # print("x_hat min/max =", x_hat.min().item(), x_hat.max().item())
        # print("target min/max =", test_target.min().item(), test_target.max().item())

    print(f"[BiGRU TEST] MSE = {mse_all.item():.6e}, dB = {10 * torch.log10(mse_all).item():.3f}"
          + (f", inference = {elapsed * 1e3:.2f} ms "
             f"({'seq-by-seq' if seq_by_seq else 'batched'})" if return_time else ""))

    if return_time:
        return mse_all, mse_db, x_hat, elapsed
    return mse_all, mse_db, x_hat