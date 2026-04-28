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
    device
):
    """
    test_input can be:
      - tensor [N, n, T]
      - or list of datasets, each [N, n, T]
    """

    if isinstance(test_input, list):
        test_input = torch.cat(test_input, dim=0)
        test_target = torch.cat(test_target, dim=0)

    test_input = test_input.to(device)
    test_target = test_target.to(device)

    model = torch.load(load_path, weights_only=False).to(device)
    model.eval()

    with torch.no_grad():
        x_hat = model(test_input)
        mse = torch.mean((x_hat - test_target) ** 2)
        mse_db = 10.0 * torch.log10(mse)

    print(f"[BiGRU TEST] MSE = {mse.item():.6e}, dB = {mse_db.item():.3f}")

    return mse, mse_db, x_hat