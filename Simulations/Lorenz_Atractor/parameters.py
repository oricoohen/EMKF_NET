import torch
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd

#########################
### Design Parameters ###
#########################
m = 2
n = 2
variance = 0
m1x_0 = torch.ones(m, 1) 
m2x_0 = 0 * 0 * torch.eye(m)

### Decimation
delta_t_gen =  1e-5
delta_t = 0.02
ratio = delta_t_gen/delta_t

# rotation (degrees). set to 0.0 to disable
ROT_DEG = 50.0
theta = torch.deg2rad(torch.tensor(ROT_DEG, dtype=torch.float32))  # radians
c = torch.cos(theta)
s = torch.sin(theta)
Robs = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])    # [2,2]

# def getJacobian(x, g):
#     # x: [m,1]; g(x): [n,1]
#     # build a vector-output wrapper for autograd
#     def g_vec(z_flat):
#         z = z_flat.view(m, 1)
#         out = g(z).view(-1)
#         return out.view(-1)
#
#     x_flat = x.detach().clone().view(-1).requires_grad_(True)  # [m]
#     y = g_vec(x_flat)                                          # [n]
#     J_rows = []
#     for i in range(n):
#         (grad_i,) = autograd.grad(y[i], x_flat, retain_graph=True, allow_unused=False)
#         J_rows.append(grad_i.view(1, m))
#     J = torch.cat(J_rows, dim=0)                               # [n, m]
#     return J


# F = torch.tensor([[0.83, 0.20],
#                   [0.20, 0.83]], dtype=torch.float32)
F = torch.tensor([[0.63, 0.0021],[0.0021, 1.0299]]) # State transition matrix

######################################################
### State evolution function f for Lorenz Atractor ###
######################################################
### f_gen is for dataset generation


def make_f(F_mat: torch.Tensor):
    """Return an f(x) that uses the provided F_mat. Accepts [m] or [m,1]; returns [m,1]."""
    def f_func(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(m, 1)
        return F_mat @ x
    return f_func

# default (true) dynamics function used everywhere unless you override it in main
f = make_f(F)


##################################################
### Observation function h for Lorenz Atractor ###
##################################################
# -----------------------------
# NONLINEAR observation: 2D range–bearing on (x1, x2)
# y = [ r, theta ]^T  with r = sqrt(x1^2 + x2^2), theta = atan2(x2, x1)
# -----------------------------
# def h_nonlinear(x):
#     # accept [m] or [m,1]; return [n,1]
#     x_vec = x.view(-1) if x.dim() == 2 else x  # [2]
#     Rcur = Robs.to(device=x_vec.device, dtype=x_vec.dtype)
#     x_rot = Rcur @ x_vec                        # rotate only for sensing
#
#     r  = torch.sqrt(x_rot[0]**2 + x_rot[1]**2 + 1e-12)
#     th = torch.atan2(x_rot[1], x_rot[0])
#     return torch.stack([r, th]).unsqueeze(1)    # [2,1]

########################################################################
# def h_nonlinear(x, a=None, b=None):
#     xv = x.view(-1)  # [2]
#     if a is None:
#         a = torch.tensor([1.0, 0.4], device=x.device, dtype=x.dtype)
#     if b is None:
#         b = torch.tensor([-0.6, 1.0], device=x.device, dtype=x.dtype)
#     s = torch.sin((a * xv).sum())
#     c = torch.cos((b * xv).sum())
#     return torch.stack([s, c]).unsqueeze(1)  # [2,1]
#
# def H_sin_cos(x, a=None, b=None):
#     xv = x.view(-1)
#     if a is None:
#         a = torch.tensor([1.0, 0.4], device=x.device, dtype=x.dtype)
#     if b is None:
#         b = torch.tensor([-0.6, 1.0], device=x.device, dtype=x.dtype)
#     Ha =  torch.cos((a * xv).sum()) * a   # row 1
#     Hb = -torch.sin((b * xv).sum()) * b   # row 2
#     return torch.stack([Ha, Hb])          # [2,2]



# def getJacobian(x, g=None):
#     return H_sin_cos(x)


##################################################

def h_nonlinear(x, alpha=0.5):
    x = x.view(2,1)
    x1, x2 = x[0,0], x[1,0]
    eps = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    r     = torch.sqrt(x1*x1 + x2*x2 + eps)
    theta = torch.atan2(x2, x1 + eps)
    H = torch.tensor([[1., 1.],
                      [0.25, 1. ]], device=x.device, dtype=x.dtype)
    lin = (H @ x).view(2)
    return lin + alpha*torch.stack([r, theta])



def getJacobian(x,g=None,alpha=0.5, eps=1e-6):
    x = x.view(2,1)
    x1, x2 = x[0,0], x[1,0]
    r = torch.sqrt(x1*x1 + x2*x2 + torch.as_tensor(eps, device=x.device, dtype=x.dtype))
    D = (x1 + eps)*(x1 + eps) + x2*x2

    H_lin = torch.tensor([[1., 1.],
                          [0.25, 1.]], device=x.device, dtype=x.dtype)

    J_nl = torch.stack([
        torch.stack([ x1/r,          x2/r ]),
        torch.stack([-x2/D, (x1+eps)/D])
    ])

    return H_lin + alpha * J_nl















# Keep H NONLINEAR everywhere:
# If other parts of your code import `h`, make it the same as h_nonlinear.
h = h_nonlinear

# -----------------------------
# Noise structures (scaled in main)
# -----------------------------
Q_structure = torch.eye(m)  # process noise base (2x2)
R_structure = torch.eye(n)  # measurement noise base (2x2)
