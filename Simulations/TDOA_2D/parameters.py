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
M_mics  = 3           # number of microphones
n  = M_mics - 1  # TDOA measurements per timestep (= 2)

# ── Physical constants ────────────────────────────────────────────────────────
dt      = 0.1   # time step
c_sound = 1.0   # speed of sound (normalized)

# ── Microphone positions ──────────────────────────────────────────────────────
# mic_positions[0] is the reference microphone (used as TDOA baseline).
mic_positions = torch.tensor(
    [[0.0, 0.0], [10.0, 0.0], [-10.0, 0.0]],
    dtype=torch.float32, device=device,
)  # [M_mics, 2]

# ── Trajectory validity bounds ────────────────────────────────────────────────
PX_MIN, PX_MAX =  -9.0,  9.0   # position x
PY_MIN, PY_MAX =   1.0,  15.0   # position y
V_MAX          =   3.5         # max speed on either axis
MAX_RETRIES    = 100            # per-sequence retry limit
USE_REFLECTION = True           # True: bounce at walls instead of regenerating; False: old retry behavior

# ── Noise covariances ─────────────────────────────────────────────────────────
Q_structure = torch.eye(m, dtype=torch.float32, device=device)   # [m, m] identity
R_structure = torch.eye(n, dtype=torch.float32, device=device)   # [n, n] identity

q2 = 1e-4   # default — main scripts override with their own q2/r2
r2 = 1e-3
Q  = q2 * Q_structure
R  = r2 * R_structure

# ── Initial condition (fixed and known for every sequence) ────────────────────
# p_y starts at the centre of [PY_MIN, PY_MAX] = 3.5 so noise has equal margin
# in both directions before hitting either bound.
v0    = 0.3
m1x_0 = torch.tensor([[0.5], [4], [v0], [v0]],
                      dtype=torch.float32, device=device)  # [4, 1]
m2x_0 = 0.01 * torch.eye(m, dtype=torch.float32, device=device)  # [4, 4]

# ── Default block-wise turning angles (single-trajectory script) ──────────────
# default_thetas_deg = [0.0, 20.0, 0.0, -20.0]
# default_thetas_rad = [t * math.pi / 180.0 for t in default_thetas_deg]
# default_thetas_rad = [0.0, 0.035, 0.0, -0.035]

# ── F-block constructor ───────────────────────────────────────────────────────
VEL_DECAY = 1  # velocity damping per step — prevents unbounded drift over datasets

def make_F_block(theta_rad: float) -> torch.Tensor:
    """
    4x4 damped-rotation transition matrix.
    Velocity is rotated by theta and decayed by VEL_DECAY each step,
    preventing unbounded drift across datasets.
        theta = 0   straight motion (with decay)
        theta > 0   left turn
        theta < 0   right turn
    """
    c = VEL_DECAY * math.cos(theta_rad)
    s = VEL_DECAY * math.sin(theta_rad)
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


# ── Boundary reflection (used when USE_REFLECTION=True) ──────────────────────
def _reflect_state(x: torch.Tensor) -> torch.Tensor:
    """Reflect position and velocity at spatial bounds in-place. Clamps velocity magnitude."""
    if x[0] < PX_MIN:
        x[0] = 2 * PX_MIN - x[0];  x[2] = x[2].abs()
    elif x[0] > PX_MAX:
        x[0] = 2 * PX_MAX - x[0];  x[2] = -x[2].abs()
    if x[1] < PY_MIN:
        x[1] = 2 * PY_MIN - x[1];  x[3] = x[3].abs()
    elif x[1] > PY_MAX:
        x[1] = 2 * PY_MAX - x[1];  x[3] = -x[3].abs()
    x[2].clamp_(-V_MAX, V_MAX)
    x[3].clamp_(-V_MAX, V_MAX)
    return x


# ── Trajectory validity check ────────────────────────────────────────────────
def _traj_valid(states: torch.Tensor) -> bool:
    """
    Returns True if every timestep of states [m, T] satisfies the bounds:
      p_x in [PX_MIN, PX_MAX], p_y in [PY_MIN, PY_MAX], |v_x|,|v_y| <= V_MAX
    """
    px, py, vx, vy = states[0], states[1], states[2], states[3]
    return bool(
        (px >= PX_MIN).all() and (px <= PX_MAX).all() and
        (py >= PY_MIN).all() and (py <= PY_MAX).all() and
        (vx.abs() <= V_MAX).all() and (vy.abs() <= V_MAX).all()
    )


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
    Retries up to MAX_RETRIES times if the trajectory violates the bounds
    defined by PX_MIN/PX_MAX, PY_MIN/PY_MAX, V_MAX.

    Returns
    -------
    states   : [m_state, T]   true states
    obs      : [n_obs,   T]   noisy TDOA observations
    n_retries: int            number of rejected attempts (0 = accepted first try)
    """
    if Q_gen is None:
        Q_gen = Q
    if R_gen is None:
        R_gen = R

    n_blocks   = len(thetas_rad)
    block_size = T // n_blocks
    L_q = torch.linalg.cholesky(Q_gen)
    L_r = torch.linalg.cholesky(R_gen)
    x0  = x_init.reshape(-1).clone() if x_init is not None else m1x_0.reshape(-1).clone()

    vcounts = {'px_low': 0, 'px_high': 0, 'py_low': 0, 'py_high': 0, 'vx': 0, 'vy': 0}

    for attempt in range(MAX_RETRIES + 1):
        states = torch.zeros(m, T, device=device)
        obs    = torch.zeros(n, T, device=device)
        x      = x0.clone()

        for t in range(T):
            k   = min(t // block_size, n_blocks - 1)
            F_k = make_F_block(thetas_rad[k])
            x   = F_k @ x + L_q @ torch.randn(m, device=device)
            if USE_REFLECTION:
                _reflect_state(x)
            y   = h(x).reshape(-1) + L_r @ torch.randn(n, device=device)
            states[:, t] = x
            obs[:, t]    = y

        if _traj_valid(states):
            return states, obs, attempt, vcounts

        # tally which bounds were violated in this attempt
        if (states[0] < PX_MIN).any(): vcounts['px_low']  += 1
        if (states[0] > PX_MAX).any(): vcounts['px_high'] += 1
        if (states[1] < PY_MIN).any(): vcounts['py_low']  += 1
        if (states[1] > PY_MAX).any(): vcounts['py_high'] += 1
        if (states[2].abs() > V_MAX).any(): vcounts['vx'] += 1
        if (states[3].abs() > V_MAX).any(): vcounts['vy'] += 1

    ranked = sorted(vcounts.items(), key=lambda x: x[1], reverse=True)
    ranked_str = '  |  '.join(f"{k}={v}" for k, v in ranked if v > 0)

    # ── pop up the last failed trajectory before crashing ────────────────────
    try:
        import os, matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        import matplotlib.patches as _mpatches

        px = states[0].cpu()
        py = states[1].cpu()
        vx = states[2].cpu()
        vy = states[3].cpu()

        # which timesteps violate each bound
        bad_px = (px < PX_MIN) | (px > PX_MAX)
        bad_py = (py < PY_MIN) | (py > PY_MAX)
        bad_v  = (vx.abs() > V_MAX) | (vy.abs() > V_MAX)
        bad    = bad_px | bad_py | bad_v

        _fig, _axes = _plt.subplots(1, 2, figsize=(14, 6))

        # left: 2D trajectory
        _ax = _axes[0]
        _ax.plot(px.numpy(), py.numpy(), 'b-', linewidth=1.5, label='trajectory')
        _ax.scatter(px[0].item(), py[0].item(), color='green', s=80, zorder=5, label='start')
        if bad.any():
            _ax.scatter(px[bad].numpy(), py[bad].numpy(),
                        color='red', s=40, zorder=6, label='violation')
        _rect = _mpatches.Rectangle(
            (PX_MIN, PY_MIN), PX_MAX - PX_MIN, PY_MAX - PY_MIN,
            linewidth=1.5, edgecolor='red', facecolor='lightyellow',
            linestyle='--', zorder=2, alpha=0.4, label='valid region',
        )
        _ax.add_patch(_rect)
        for _i, _mic in enumerate(mic_positions):
            _ax.scatter(_mic[0].item(), _mic[1].item(), marker='^', color='black', s=80, zorder=7)
            _ax.annotate(f'm{_i}', (_mic[0].item(), _mic[1].item()),
                         textcoords='offset points', xytext=(4, 4), fontsize=7)
        _ax.set_xlabel('p_x'); _ax.set_ylabel('p_y')
        _ax.set_title('Last failed trajectory (red = violated)')
        _ax.legend(fontsize=8); _ax.grid(True, alpha=0.4)

        # right: velocities vs time
        _t = list(range(len(vx)))
        _axes[1].plot(_t, vx.numpy(), label='v_x')
        _axes[1].plot(_t, vy.numpy(), '--', label='v_y')
        _axes[1].axhline( V_MAX, color='red', linestyle=':', linewidth=1.2, label=f'+V_MAX={V_MAX}')
        _axes[1].axhline(-V_MAX, color='red', linestyle=':', linewidth=1.2, label=f'-V_MAX={V_MAX}')
        _axes[1].set_xlabel('time step'); _axes[1].set_ylabel('velocity')
        _axes[1].set_title('v_x / v_y vs time')
        _axes[1].legend(fontsize=8); _axes[1].grid(True, alpha=0.4)

        _fig.suptitle(f'FAILED after {MAX_RETRIES} retries  |  {ranked_str}', fontsize=11, color='red')
        _plt.tight_layout()
        _save = os.path.abspath('failed_trajectory_debug.png')
        _plt.savefig(_save, dpi=150)
        _plt.close(_fig)
        os.startfile(_save)
        print(f"\n  [DEBUG] Failed trajectory plot saved and opened: {_save}")
    except Exception:
        pass  # never let the plot crash hide the real error

    raise RuntimeError(
        f"generate_single_traj: could not produce a valid trajectory in "
        f"{MAX_RETRIES} attempts.\n"
        f"Violation breakdown (most common first): {ranked_str}\n"
        f"Bounds: px=[{PX_MIN},{PX_MAX}], py=[{PY_MIN},{PY_MAX}], |v|<={V_MAX}.\n"
        f"Consider loosening bounds, reducing noise (Q), or reducing theta range."
    )


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

    n_rejected = 0
    total_vc   = {'px_low': 0, 'px_high': 0, 'py_low': 0, 'py_high': 0, 'vx': 0, 'vy': 0}
    for s in range(size):
        states, obs, retries, vc = generate_single_traj(T, thetas_rad, Q_gen, R_gen)
        if retries > 0:
            n_rejected += 1
            for k in total_vc: total_vc[k] += vc[k]
        all_inputs[s] = obs
        all_targets[s] = states

    rejection_rate = n_rejected / size
    if rejection_rate > 0.10:
        ranked = sorted(total_vc.items(), key=lambda x: x[1], reverse=True)
        ranked_str = '  |  '.join(f"{k}={v}" for k, v in ranked if v > 0)
        raise RuntimeError(
            f"generate_multi_traj_batch: {n_rejected}/{size} sequences "
            f"({100*rejection_rate:.1f}%) needed at least one retry — exceeds 10% threshold.\n"
            f"Violation breakdown (most common first): {ranked_str}\n"
            f"Bounds: px=[{PX_MIN},{PX_MAX}], py=[{PY_MIN},{PY_MAX}], |v|<={V_MAX}.\n"
            f"Suggestions: reduce Q noise, reduce theta range, or loosen bounds."
        )

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
                                  theta_base=None):
    """
    Generate N trajectories, each with its own random theta.
    theta for sequence i: theta_base[i] + Uniform(-theta_true_max/2, +theta_true_max/2)

    theta_base: None or float  → all sequences use 0.0 as base
                list of N floats → each sequence i uses theta_base[i] as base

    x_init: None            → all sequences start from m1x_0
            [m] tensor      → all sequences share this single initial state
            [N, m] tensor   → each sequence idx starts from x_init[idx]

    Returns
    -------
    inputs     : [N, n, T]
    targets    : [N, m, T]
    theta_list : list of N floats   (true theta per sequence)
    F_list     : list of N tensors  (true F per sequence)
    """
    import random as _random
    if Q_gen is None:
        Q_gen = Q
    if R_gen is None:
        R_gen = R

    per_seq = (x_init is not None and x_init.dim() == 2)   # [N, m] per-sequence carry

    inputs  = torch.zeros(N, n, T, device=device)
    targets = torch.zeros(N, m, T, device=device)
    theta_list = []
    F_list     = []

    n_rejected = 0
    total_vc   = {'px_low': 0, 'px_high': 0, 'py_low': 0, 'py_high': 0, 'vx': 0, 'vy': 0}
    for i in range(N):
        base  = (theta_base[i] if isinstance(theta_base, list) else 0.0)
        theta = base + (_random.random() - 0.5) * theta_true_max
        theta_list.append(theta)
        F_list.append(make_F_block(theta))
        x0 = x_init[i] if per_seq else x_init
        states, obs, retries, vc = generate_single_traj(T, [theta], Q_gen, R_gen, x_init=x0)
        if retries > 0:
            n_rejected += 1
            for k in total_vc: total_vc[k] += vc[k]
        targets[i] = states
        inputs[i]  = obs

    rejection_rate = n_rejected / N
    if rejection_rate > 0.10:
        ranked = sorted(total_vc.items(), key=lambda x: x[1], reverse=True)
        ranked_str = '  |  '.join(f"{k}={v}" for k, v in ranked if v > 0)
        raise RuntimeError(
            f"generate_dataset_random_theta: {n_rejected}/{N} sequences "
            f"({100*rejection_rate:.1f}%) needed at least one retry — exceeds 10% threshold.\n"
            f"Violation breakdown (most common first): {ranked_str}\n"
            f"Bounds: px=[{PX_MIN},{PX_MAX}], py=[{PY_MIN},{PY_MAX}], |v|<={V_MAX}.\n"
            f"Suggestions: reduce Q noise, reduce theta range, or loosen bounds."
        )

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


# ── Diagonal (independent-axis) acceleration model ────────────────────────────
# vx_{t+1} = alpha_x * vx_t + noise
# vy_{t+1} = alpha_y * vy_t + noise
# alpha > 1: accelerating, alpha < 1: decelerating, alpha = 1: constant velocity.
# No coupling between axes — no circular motion.

def make_F_diag(alpha_x: float, alpha_y: float) -> torch.Tensor:
    """
    4x4 transition matrix with independent per-axis velocity scaling.
    False F uses make_F_diag(1.0, 1.0) (constant velocity).
    """
    return torch.tensor(
        [[1.0, 0.0,     dt, 0.0    ],
         [0.0, 1.0,    0.0,     dt ],
         [0.0, 0.0, alpha_x, 0.0   ],
         [0.0, 0.0,    0.0, alpha_y]],
        dtype=torch.float32, device=device,
    )


def generate_single_traj_diag(
    T: int,
    alpha_x: float,
    alpha_y: float,
    Q_gen: torch.Tensor = None,
    R_gen: torch.Tensor = None,
    x_init: torch.Tensor = None,
) -> tuple:
    """
    Single trajectory of length T using a diagonal F(alpha_x, alpha_y).
    Retries up to MAX_RETRIES times if bounds are violated.

    Returns: states [m, T], obs [n, T], n_retries int
    """
    if Q_gen is None: Q_gen = Q
    if R_gen is None: R_gen = R

    F_use = make_F_diag(alpha_x, alpha_y)
    L_q   = torch.linalg.cholesky(Q_gen)
    L_r   = torch.linalg.cholesky(R_gen)
    x0    = x_init.reshape(-1).clone() if x_init is not None else m1x_0.reshape(-1).clone()

    for attempt in range(MAX_RETRIES + 1):
        states = torch.zeros(m, T, device=device)
        obs    = torch.zeros(n, T, device=device)
        x      = x0.clone()
        for t in range(T):
            x = F_use @ x + L_q @ torch.randn(m, device=device)
            if USE_REFLECTION:
                _reflect_state(x)
            y = h(x).reshape(-1) + L_r @ torch.randn(n, device=device)
            states[:, t] = x
            obs[:, t]    = y
        if _traj_valid(states):
            return states, obs, attempt, F_use

    raise RuntimeError(
        f"generate_single_traj_diag: could not produce a valid trajectory in "
        f"{MAX_RETRIES} attempts with alpha_x={alpha_x:.4f}, alpha_y={alpha_y:.4f}.\n"
        f"Bounds: px=[{PX_MIN},{PX_MAX}], py=[{PY_MIN},{PY_MAX}], |v|<={V_MAX}.\n"
        f"Suggestion: bring alpha closer to 1.0, reduce Q noise, or loosen bounds."
    )


def generate_dataset_diag_accel(
    N: int,
    T: int,
    alpha_x: float,
    alpha_y: float,
    Q_gen: torch.Tensor = None,
    R_gen: torch.Tensor = None,
    x_init=None,
    a_range: float = 0.0,
    b_range: float = 0.0,
) -> tuple:
    """
    Generate N trajectories using diagonal F(alpha_x_i, alpha_y_i) per sequence.

    x_init: None          → all sequences start from m1x_0
            [m] tensor    → all sequences share this initial state
            [N, m] tensor → each sequence i starts from x_init[i]

    a_range, b_range: per-sequence random offset around (alpha_x, alpha_y).
        alpha_x_i = alpha_x + Uniform(-a_range, a_range)
        If 0 (default, test mode): all sequences share exactly alpha_x, alpha_y.

    Returns: inputs [N, n, T], targets [N, m, T], F_list (list of N F matrices)
    """
    import random as _rng
    if Q_gen is None: Q_gen = Q
    if R_gen is None: R_gen = R

    per_seq = (
        x_init is not None
        and isinstance(x_init, torch.Tensor)
        and x_init.dim() == 2
    )

    inputs  = torch.zeros(N, n, T, device=device)
    targets = torch.zeros(N, m, T, device=device)
    F_list  = []
    n_rejected = 0

    for i in range(N):
        ax_i = alpha_x + (_rng.uniform(-a_range, a_range) if a_range > 0 else 0.0)
        ay_i = alpha_y + (_rng.uniform(-b_range, b_range) if b_range > 0 else 0.0)
        x0 = x_init[i] if per_seq else x_init
        states, obs, retries, F_i = generate_single_traj_diag(T, ax_i, ay_i, Q_gen, R_gen, x0)
        if retries > 0:
            n_rejected += 1
        inputs[i]  = obs
        targets[i] = states
        F_list.append(F_i)

    rejection_rate = n_rejected / N
    if rejection_rate > 0.10:
        raise RuntimeError(
            f"generate_dataset_diag_accel: {n_rejected}/{N} sequences "
            f"({100*rejection_rate:.1f}%) needed at least one retry — exceeds 10%.\n"
            f"alpha_x={alpha_x:.4f}, alpha_y={alpha_y:.4f}.\n"
            f"Suggestion: bring alpha closer to 1.0 or loosen bounds."
        )

    return inputs, targets, F_list
