import torch
from Simulations.TDOA_2D.parameters import make_F_block, h, mic_positions, default_thetas_rad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_F_sequence(T, thetas_rad):
    n_blocks = len(thetas_rad)
    block_size = T // n_blocks

    F_seq = []
    block_ids = []

    for t in range(T):
        k = min(t // block_size, n_blocks - 1)
        F_t = make_F_block(thetas_rad[k])
        F_seq.append(F_t)
        block_ids.append(k)

    return F_seq, torch.tensor(block_ids, dtype=torch.long, device=device)

def generate_single_sequence(T, Q, R, m1x_0, thetas_rad=None):
    if thetas_rad is None:
        thetas_rad = default_thetas_rad

    m = m1x_0.shape[0]
    n = R.shape[0]

    F_seq, block_ids = build_F_sequence(T, thetas_rad)

    x = torch.zeros(m, T, device=device)
    y = torch.zeros(n, T, device=device)

    x[:, 0] = m1x_0[:, 0]

    # first observation
    y[:, 0:1] = h(x[:, 0]) + torch.linalg.cholesky(R) @ torch.randn(n, 1, device=device)

    for t in range(T - 1):
        w_t = torch.linalg.cholesky(Q) @ torch.randn(m, 1, device=device)
        x[:, t+1:t+2] = F_seq[t] @ x[:, t:t+1] + w_t

        r_t = torch.linalg.cholesky(R) @ torch.randn(n, 1, device=device)
        y[:, t+1:t+2] = h(x[:, t+1]) + r_t

    return x, y, F_seq, block_ids