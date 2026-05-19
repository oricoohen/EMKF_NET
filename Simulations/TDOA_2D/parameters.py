"""
Parameters for the 2-D TDOA tracking experiment.

State:         x_t = [p_x, p_y, v_x, v_y]^T
Dynamics:      x_{t+1} = F^(k) x_t + w_t,   w_t ~ N(0, Q)
Observations:  y_t^(i) = (||pos_t - m_i|| - ||pos_t - m_1||) / c + r_t^(i)
               stacked into y_t in R^{M-1},   r_t ~ N(0, R)

F^(k) uses rotation-only velocity dynamics:
    [[1, 0, dt,       0      ],
     [0, 1,  0,      dt      ],
     [0, 0, cos(θ), -sin(θ)  ],
     [0, 0, sin(θ),  cos(θ)  ]]

No scaling is applied; velocity magnitude is preserved.
"""

import math
import torch
from torch import autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dimensions ────────────────────────────────────────────────────────────────
m = 4           # state: [p_x, p_y, v_x, v_y]
M_mics  = 4           # number of microphones
n  = M_mics - 1  # TDOA measurements per timestep (= 3)

# ── Physical constants ────────────────────────────────────────────────────────
dt      = 0.05   # time step  (v0=0.5 × dt=0.1 gives ~5 units travel over T=100)
c_sound = 1.0   # speed of sound (normalized)

# ── Microphone positions on the x-axis (y = 0) ───────────────────────────────
# mic_positions[0] is the reference microphone (used as TDOA baseline).
mic_positions = torch.tensor(
    [[-10.0, 0.0], [-5.0, 0.0], [0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
    dtype=torch.float32, device=device,
)  # [M_mics, 2]

# ── Noise covariances ─────────────────────────────────────────────────────────
Q_structure = torch.eye(m, dtype=torch.float32, device=device)   # [m, m] identity
R_structure = torch.eye(n, dtype=torch.float32, device=device)   # [n, n] identity

q2 = 1e-4   # default — main scripts override with their own q2/r2
r2 = 1e-3
Q  = q2 * Q_structure
R  = r2 * R_structure

# ── Initial condition (fixed and known for every sequence) ────────────────────
# y-offset of 2.0 keeps the target above the mic axis so TDOA is well-defined.
v0    = 0.5
m1x_0 = torch.tensor([[3.0], [3.0], [v0], [0.0]],
                      dtype=torch.float32, device=device)  # [4, 1]
m2x_0 = 0.01 * torch.eye(m, dtype=torch.float32, device=device)  # [4, 4]

# ── Default block-wise turning angles (single-trajectory script) ──────────────
default_thetas_deg = [0.0, 20.0, 0.0, -20.0]
default_thetas_rad = [t * math.pi / 180.0 for t in default_thetas_deg]
default_thetas_rad = [0.0, 0.035, 0.0, -0.035]

# ── F-block constructor ───────────────────────────────────────────────────────
def make_F_block(theta_rad: float) -> torch.Tensor:
    """
    4x4 near-constant-velocity transition matrix with turning angle theta.
        theta = 0   straight motion
        theta > 0   left turn
        theta < 0   right turn
    """
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return torch.tensor(
        [[1.0, 0.0,  dt, 0.0],
         [0.0, 1.0, 0.0,  dt],
         [0.0, 0.0,   c,  -s],
         [0.0, 0.0,   s,   c]],
        dtype=torch.float32, device=device,
    )


# Nominal (straight-line) matrix and function used by ERTS / EMKF
F = make_F_block(0.0)

def f(x):
    return torch.matmul(F, x)

# ── TDOA observation function ─────────────────────────────────────────────────
def h(x: torch.Tensor) -> torch.Tensor:
    """
    Nonlinear TDOA observations relative to mic[0] (reference).
    x: [4]    ->  returns [n_obs]       (1-D; matches Lorenz h, used by RTSNet)
    x: [4, 1] ->  returns [n_obs, 1]    (column; used by SystemModel.GenerateSequence)
    """
    col_input = x.dim() >= 2
    pos   = x.reshape(-1)[:2]
    d_ref = torch.norm(pos - mic_positions[0], p=2)
    tdoas = [
        (torch.norm(pos - mic_positions[i], p=2) - d_ref) / c_sound
        for i in range(1, M_mics)
    ]
    y = torch.stack(tdoas)                             # [n_obs]
    return y.unsqueeze(-1) if col_input else y


# ── Analytic Jacobian of h ────────────────────────────────────────────────────
def h_jacobian(x: torch.Tensor) -> torch.Tensor:
    """
    Closed-form Jacobian of h at x.  Returns [n_obs, m_state].

    ∂TDOA_i/∂p = (p - m_i)/d_i − (p - m_ref)/d_ref  (divided by c_sound)
    Velocity components (columns 2, 3) are always zero.
    """
    pos      = x.reshape(-1)[:2]                                  # [2]
    d_ref    = torch.norm(pos - mic_positions[0], p=2)
    unit_ref = (pos - mic_positions[0]) / (d_ref + 1e-8)          # [2]

    rows = []
    for i in range(1, M_mics):
        d_i      = torch.norm(pos - mic_positions[i], p=2)
        unit_i   = (pos - mic_positions[i]) / (d_i + 1e-8)        # [2]
        grad_pos = (unit_i - unit_ref) / c_sound                   # [2]
        rows.append(torch.cat([grad_pos,
                               torch.zeros(2, device=device)]))    # [4]
    return torch.stack(rows)                                       # [n_obs, 4]


# ── General autograd Jacobian (for verification / future use) ─────────────────
def get_jacobian(x: torch.Tensor, g, out_dim: int, in_dim: int) -> torch.Tensor:
    """
    Jacobian of g at x via autograd.  g: R^in_dim → R^out_dim.
    Returns [out_dim, in_dim].
    """
    x_flat = x.reshape(-1).detach()

    def g_flat(y: torch.Tensor) -> torch.Tensor:
        return g(y).reshape(-1)

    J = autograd.functional.jacobian(g_flat, x_flat, create_graph=False)
    return J.view(out_dim, in_dim)


# ── Single-trajectory generator ──────────────────────────────────────────────
def generate_single_traj(
    T: int,
    thetas_rad: list,
    Q_gen: torch.Tensor = None,
    R_gen: torch.Tensor = None,
    x_init: torch.Tensor = None,
) -> tuple:
    """
    One trajectory of length T with equal-size blocks.

    Returns
    -------
    states : [m_state, T]   true states
    obs    : [n_obs,   T]   noisy TDOA observations
    """
    if Q_gen is None:
        Q_gen = Q
    if R_gen is None:
        R_gen = R

    n_blocks   = len(thetas_rad)
    block_size = T // n_blocks

    L_q = torch.linalg.cholesky(Q_gen)   # [m, m]
    L_r = torch.linalg.cholesky(R_gen)   # [n, n]

    states = torch.zeros(m, T, device=device)
    obs    = torch.zeros(n,   T, device=device)
    x      = x_init.reshape(-1).clone() if x_init is not None else m1x_0.reshape(-1).clone()  # [4]

    for t in range(T):
        k   = min(t // block_size, n_blocks - 1)
        F_k = make_F_block(thetas_rad[k])
        x   = F_k @ x + L_q @ torch.randn(m, device=device)
        y   = h(x).reshape(-1) + L_r @ torch.randn(n, device=device)
        states[:, t] = x
        obs[:, t]    = y

    return states, obs


# ── Multi-trajectory batch generator ─────────────────────────────────────────
def generate_multi_traj_batch(
    size: int,
    T: int,
    thetas_rad: list,
    Q_gen: torch.Tensor = None,
    R_gen: torch.Tensor = None,
) -> tuple:
    """
    Generate many trajectories using the SAME block-wise turning sequence.
    Different trajectories differ only by process / measurement noise.
    """
    if Q_gen is None:
        Q_gen = Q
    if R_gen is None:
        R_gen = R

    all_inputs  = torch.zeros(size, n, T, device=device)
    all_targets = torch.zeros(size, m, T, device=device)

    for s in range(size):
        states, obs = generate_single_traj(T, thetas_rad, Q_gen, R_gen)
        all_inputs[s] = obs
        all_targets[s] = states

    return all_inputs, all_targets


# ── Helper factories (for training mains — keep helpers out of scripts) ────────

def make_f(F_matrix: torch.Tensor):
    """Return a dynamics function f(x) = F_matrix @ x."""
    def f_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(F_matrix, x)
    return f_fn


def make_get_F_blockwise(thetas_rad: list, T: int):
    """Return a get_F(t) that gives the block-wise F at time t (for run_ekf_erts)."""
    n_blocks_  = len(thetas_rad)
    block_size_ = T // n_blocks_
    def get_F(t: int) -> torch.Tensor:
        k = min(t // block_size_, n_blocks_ - 1)
        return make_F_block(thetas_rad[k])
    return get_F


def make_get_F_const(theta_rad: float):
    """Return a constant get_F(t) = make_F_block(theta_rad) for all t."""
    F_const = make_F_block(theta_rad)
    def get_F(_t: int) -> torch.Tensor:
        return F_const
    return get_F


def generate_random_F_matrices_tdoa(num_F: int, theta_center: float,
                                    delta_theta: float = 0.05) -> list:
    """
    Generate num_F F matrices randomly perturbed around theta_center.
    Used for generate_f=True training (one F per 10 sequences, like ori_main).
    """
    import random as _random
    return [make_F_block(theta_center + (2 * _random.random() - 1) * delta_theta)
            for _ in range(num_F)]


def make_get_F_from_matrix(F_matrix: torch.Tensor):
    """Return a constant get_F(t) = F_matrix for all t (for run_ekf_erts)."""
    def get_F(_t: int) -> torch.Tensor:
        return F_matrix
    return get_F


def generate_dataset_random_theta(N: int, T: int, theta_true_max: float,
                                  Q_gen: torch.Tensor = None,
                                  R_gen: torch.Tensor = None,
                                  x_init: torch.Tensor = None,
                                  theta_base: float = 0.0):
    """
    Generate N trajectories in groups of 10.
    Each group uses ONE random theta ~ theta_base + Uniform(-theta_true_max/2, +theta_true_max/2)
    as a constant F for all T time steps of those 10 sequences.
    All trajectories start from x_init (defaults to m1x_0 if None).
    N must be a multiple of 10.

    Returns
    -------
    inputs     : [N, n, T]
    targets    : [N, m, T]
    theta_list : list of N//10 floats   (true theta per group)
    F_list     : list of N//10 tensors  (true F per group)
    """
    import random as _random
    if Q_gen is None:
        Q_gen = Q
    if R_gen is None:
        R_gen = R

    n_groups = N // 10
    inputs  = torch.zeros(N, n, T, device=device)
    targets = torch.zeros(N, m, T, device=device)
    theta_list = []
    F_list     = []

    for g in range(n_groups):
        theta = theta_base + (_random.random() - 0.5) * theta_true_max   # theta_base + Uniform(-max/2, +max/2)
        theta_list.append(theta)
        F_list.append(make_F_block(theta))
        for s in range(10):
            idx = g * 10 + s
            states, obs = generate_single_traj(T, [theta], Q_gen, R_gen, x_init=x_init)
            targets[idx] = states
            inputs[idx]  = obs

    return inputs, targets, theta_list, F_list


def generate_false_F_list(theta_true_list: list, theta_false_max: float) -> list:
    """
    For each true theta, produce a false F:
        theta_false = theta_true + Uniform(-theta_false_max/2, +theta_false_max/2)

    Returns a list of F matrices (same length as theta_true_list).
    """
    import random as _random
    F_false_list = []
    for theta_true in theta_true_list:
        delta = (_random.random() - 0.5) * theta_false_max
        F_false_list.append(make_F_block(theta_true + delta))
    return F_false_list
