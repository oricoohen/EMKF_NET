import os
import random
import torch
import torch.nn as nn
from Pipelines.pipeline_weather_temp import _win_norm_4d


def train_bigru(
        model,
        train_input,
        train_target,
        train_x0,
        cv_input,
        cv_target,
        cv_x0,
        save_path,
        n_epochs=150,
        batch_size=32,
        lr=1e-3,
        wd=1e-4,
        device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    N_train = len(train_input)
    N_cv = len(cv_input)
    best_cv_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()

        indices = random.sample(range(N_train), k=min(batch_size, N_train))
        y_batch, x0_batch, true_tavg_batch = [], [], []

        for idx in indices:
            if idx == 0:
                continue  # Skip the first sample since it has no previous target for normalization del
            y_win = train_input[idx].to(device)
            x_true = train_target[idx].to(device)
            x0_raw = train_x0[idx].to(device)

            x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
            prev_x_mean, prev_x_std,_,_ = _win_norm_4d(train_target[idx-1], device, dtype)#del
            # x_mean[0] = x0_raw[0].squeeze()
            # x_std[0] = 4
            x_mean[0] = prev_x_mean[0]  # Use the mean of the previous target for normalization del
            x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
            y_win_n = (y_win - x_mean[1:]) / x_std[1:]
            x0_norm = (x0_raw - x_mean.squeeze()) / x_std.squeeze()
            true_tavg_n = ((x_true - x_mean) / x_std)[0, :]

            y_batch.append(y_win_n)
            x0_batch.append(x0_norm)
            true_tavg_batch.append(true_tavg_n)

        y_batch = torch.stack(y_batch)
        x0_batch = torch.stack(x0_batch)
        true_batch = torch.stack(true_tavg_batch)

        optimizer.zero_grad()
        pred = model(y_batch, x0_batch)
        loss = loss_fn(pred, true_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            cv_losses = []
            for j in range(N_cv):
                if j == 0:
                    continue  # Skip the first sample since it has no previous target for normalization del
                y_win = cv_input[j].to(device)
                x_true = cv_target[j].to(device)
                x0_raw = cv_x0[j].to(device)

                x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
                prev_x_mean, prev_x_std,_,_ = _win_norm_4d(cv_target[j-1], device, dtype)#del
                x_mean[0] = prev_x_mean[0]  # Use the mean of the previous target for normalization del
                x_std[0] =  prev_x_std[0]  # Use the std
                # x_mean[0] = x0_raw[0].squeeze()
                # x_std[0] = 4
                y_win_n = (y_win - x_mean[1:]) / x_std[1:]
                x0_norm = (x0_raw - x_mean.squeeze()) / x_std.squeeze()
                true_tavg_n = ((x_true - x_mean) / x_std)[0, :]

                pred_j = model(y_win_n.unsqueeze(0), x0_norm.unsqueeze(0)).squeeze(0)
                cv_losses.append(loss_fn(pred_j, true_tavg_n).item())

            cv_loss = sum(cv_losses) / len(cv_losses)

        if cv_loss < best_cv_loss:
            best_cv_loss = cv_loss
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model, save_path)
            print(f"Saved BiGRU model to: {save_path}")
        if epoch % 25 == 0 or epoch == n_epochs - 1:
            print(
                f"[BiGRU epoch {epoch:03d}] train_loss={loss.item():.4f} cv_loss={cv_loss:.4f} best={best_cv_loss:.4f}")


def test_bigru(
        save_path,
        test_input,
        test_target,
        test_x0,
        device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model = torch.load(save_path, map_location=device, weights_only=False).eval()

    sq_err_denorm = []
    bigru_preds = []

    with torch.no_grad():
        for j in range(len(test_input)):
            if j == 0:
                continue  # Skip the first sample since it has no previous target for normalization del
            y_win = test_input[j].to(device)
            x_true = test_target[j].to(device)
            x0_raw = test_x0[j].to(device)

            x_mean, x_std, _, _ = _win_norm_4d(x_true, device, dtype)
            prev_x_mean, prev_x_std,_,_ = _win_norm_4d(test_target[j-1], device, dtype)#del
            x_mean[0] = prev_x_mean[0]  # Use the mean of
            x_std[0] =  prev_x_std[0]  # Use the std of the previous target for normalization del
            # x_mean[0] = x0_raw[0].squeeze()
            # x_std[0] = 4
            y_win_n = (y_win - x_mean[1:]) / x_std[1:]
            x0_norm = (x0_raw - x_mean.squeeze()) / x_std.squeeze()

            pred_tavg_norm = model(y_win_n.unsqueeze(0), x0_norm.unsqueeze(0)).squeeze(0)

            pred_tavg_denorm = pred_tavg_norm * x_std[0] + x_mean[0]
            true_tavg_denorm = x_true[0, :]
            mse_j = torch.mean((pred_tavg_denorm - true_tavg_denorm) ** 2).item()
            sq_err_denorm.append(mse_j)

            x_pred_denorm = x_true.clone()
            x_pred_denorm[0, :] = pred_tavg_denorm
            bigru_preds.append(x_pred_denorm.detach().cpu())

    mse_denorm = torch.tensor(sq_err_denorm).mean()
    rmse_denorm = torch.sqrt(mse_denorm)

    return mse_denorm, rmse_denorm, bigru_preds