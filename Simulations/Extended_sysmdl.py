"""# **Class: System Model for Non-linear Cases**

1 Store system model parameters: 
    state transition function f, 
    observation function h, 
    process noise Q, 
    observation noise R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test,
    state dimension m,
    observation dimension n, etc.

2 Generate datasets for non-linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from torch.distributions import MultivariateNormal, Exponential
from functools import partial

DEVICE =torch.device("cuda")

def uniform_two_ranges(a: float, b: float):
    """Sample U in [a,b] or in [-b,-a] (50/50). Requires 0 ≤ a ≤ b."""
    assert 0.0 <= a <= b, "`a` must be non-negative and ≤ `b`"
    x = a + (b - a) * torch.rand(1).item()
    return x if random.random() < 0.5 else -x

def generate_random_F_matrices(num_F: int, delta_ = 0.5):
    """
    Return a list of base F matrices. You can customize this to your liking.
    Currently uses a fixed template (like your code) to keep behavior stable.
    """
    F_mats = []
    # base = torch.tensor([[0.63, 0.0021], [0.0021, 1.0299]],device = DEVICE, dtype=torch.float32)
    base = torch.tensor([[0.83, 0.2],[0.2, 0.83]], device=DEVICE)
    for _ in range(num_F):
        F_i = rotate_F(base.clone(), i=0, j=1, theta=1, many=False, randomit=True)
        F_mats.append(F_i.clone())
    return F_mats

def rotate_F(F, i=0, j=1, theta=0.78, many=True, randomit=True):
    """
    Apply Givens rotation to F (or list of F) in (i,j) plane.
    """
    def apply_rotation(F_single, angle, i, j):
        F_single = F_single.clone()
        n = F_single.shape[0]
        R = torch.eye(n, dtype=F_single.dtype, device=F_single.device)
        c, s = torch.cos(angle), torch.sin(angle)
        R[i, i] = c;   R[i, j] = -s
        R[j, i] = s;   R[j, j] =  c
        return R @ F_single @ R.T

    if not many:
        ang = uniform_two_ranges(0.0, 1) * theta if randomit else theta
        return apply_rotation(F, torch.tensor(ang, dtype=F.dtype, device=F.device), i, j)

    out = []
    for F_i in F:
        ang = uniform_two_ranges(0.0, 1) * theta if randomit else theta
        out.append(apply_rotation(F_i, torch.tensor(ang, dtype=F_i.dtype, device=F_i.device), i, j))
    return out

from functools import partial
import torch

def h_rotate_apply(x, h_base, phi_deg: float):
    """
    y_rot(x) = R(phi_deg) @ h_base(x)
    - Pure torch ops → autograd works through x
    - Top-level function → picklable (OK with DataLoader / torch.save)
    """
    y = h_base(x)                      # shape [2,1] or [2]
    y2 = y.reshape(-1)                 # [2]; reshape works on non-contiguous too
    phi = torch.as_tensor(phi_deg, dtype=y2.dtype, device=y2.device) * (torch.pi / 180.0)
    c = torch.cos(phi)
    s = torch.sin(phi)
    # 4) Rotation matrix R(φ) and apply to y
    R_out = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))  # [2,2] on correct device
    y_rot = R_out @ y2.unsqueeze(-1)                                   # [2,1]
    return y_rot

def make_rotated_h_nonlinear(h_nonlinear, phi_deg: float = 30.0):
    """
    Returns a picklable callable h_rot(x) that internally calls:
        h_rotate_apply(x, h_base=h_nonlinear, phi_deg=phi_deg)
    """
    return partial(h_rotate_apply, h_base=h_nonlinear, phi_deg=phi_deg)


class SystemModel:

    def __init__(self, f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.m = m
        self.Q = Q
        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n
        self.R = R
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test
        self.F = None  # will be set per sequence if you pass F_gen in GeneratBatch
        self.F_T = None

        self.device = self.Q.device
        self.dtype = self.Q.dtype
        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m, device=self.device)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m), device=self.device)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n, device=self.device)
        else:
            self.prior_S = prior_S




    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        # self.m1x_0 = torch.squeeze(m1x_0)
        self.m1x_0 = m1x_0
        self.m2x_0 = torch.squeeze(m2x_0)


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R
    # --- helpers to switch F used by f (if desired) ---

    def update_f(self, F):
        self.F   = F
        self.F_T = F.T
        def f_linear_1d(x: torch.Tensor) -> torch.Tensor:
            # accept [m] or [m,1]; always RETURN [m]
            if x.dim() == 2:
                x_col = x  # [m,1]
            else:
                x_col = x.view(self.m, 1)  # [m,1]
            y_col = F @ x_col  # [m,1]
            return y_col.view(self.m)  # <-- 1-D [m]

        self.f = f_linear_1d
    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):

        dev = self.Q.device
        dt = self.Q.dtype
        # Pre allocate an array for current state
        self.x = torch.empty((self.m, T), dtype=dt, device=dev)
        # Pre allocate an array for current observation
        self.y = torch.empty((self.n, T), dtype=dt, device=dev)

        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        q2 = torch.tensor(0.01,device=self.F.device,dtype=self.F.dtype)
        r2 = torch.tensor(10, device=self.F.device, dtype=self.F.dtype)

        lam_q = torch.rsqrt(q2)
        lam_r = torch.rsqrt(r2)

        lam_vec_q = lam_q.expand(self.m)
        lam_vec_r = lam_r.expand(self.n)

        q_dist = Exponential(lam_vec_q)
        r_dist = Exponential(lam_vec_r)
        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################   
            if torch.equal(Q_gen,torch.zeros_like(Q_gen)):# No noise
                 xt = self.f(self.x_prev)   
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:            
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m], device=dev, dtype=dt)
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                eq = torch.reshape(eq[:], xt.size())
                ############################3
                # eq = (q_dist.sample() - 1.0 / lam_vec_q).reshape_as(xt)
                # eq = (q_dist.sample()).reshape_as(xt)
                ##############################
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            yt = self.h(xt)
            # Observation Noise         
            if self.n == 1: # 1 dim noise
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:  
                mean = torch.zeros([self.n] ,device=dev, dtype=dt)
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], yt.size())       
                #############################3
                # er = (r_dist.sample() - 1.0 / lam_vec_r).reshape_as(yt)
                # er = (r_dist.sample()).reshape_as(yt)
                ##############################
                # Additive Observation Noise
                yt = torch.add(yt,er)
            
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, delta = 0.5, randomInit=False, randomLength=False,F_gen=True,x0_list = None):
        if(randomLength):
            # Allocate Empty list for Input
            self.Input = []
            # Allocate Empty list for Target
            self.Target = []
            # Init Sequence Lengths
            T_tensor = torch.round(900*torch.rand(size)).int()+100 # Uniform distribution [100,1000]
        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T)

        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.empty(size, self.m)

        if F_gen == True:
            F_matrices = generate_random_F_matrices(size//10 +1,delta)
        else:
            F_matrices = F_gen


        for i in range(0, size):
            index_F = i // 10
            self.F = F_matrices[index_F]
            self.F_T = F_matrices[index_F].T
            self.update_f(F_matrices[index_F])
            # Generate Sequence
            ### Generate Examples
            # ---- choose init x0 for this sequence i ----ori
            if x0_list is not None:
                initConditions = x0_list[i]  # expected shape [m,1]
            else:
                initConditions = self.m1x_0  # default

            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
                self.m1x_0_rand[i,:] = torch.squeeze(initConditions)
            self.InitSequence(initConditions, self.m2x_0)
            
            if(randomLength):
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                # Training sequence input
                self.Input.append(self.y)
                # Training sequence output
                self.Target.append(self.x)           
            else:
                self.GenerateSequence(self.Q, self.R, T)
                # Training sequence input
                self.Input[i, :, :] = self.y
                # Training sequence output
                self.Target[i, :, :] = self.x

        return F_matrices