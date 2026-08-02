"""# **Class: System Model for Linear Cases**

1 Store system model parameters: 
    state transition matrix F, 
    observation matrix H, 
    process noise covariance matrix Q, 
    observation noise covariance matrix R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test, etc.

2 Generate dataset for linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from torch.distributions import MultivariateNormal, Exponential
DEVICE =torch.device("cuda")
device = DEVICE
def uniform_two_ranges(a: float, b: float):
    """
    Draws a single sample that is uniform either in  [a , b]
    or in [−b , −a]   (each half chosen with 50 % probability).

    Requires  0 ≤ a ≤ b.
    """
    assert 0.0 <= a <= b, "`a` must be non-negative and ≤ `b`"

    u = torch.rand(1).item()            # uniform in [0,1)
    x = a + (b - a) * u                 # now uniform in [a , b]

    if random.random() < 0.5:           # coin flip → positive half
        return  x                       #  in  [ a ,  b]
    else:                               # negative half
        return -x                       #  in [−b , −a]
def generate_random_F_matrices(num_F, delta_t=0.5, state_dim=2,F_init=None):
    """
    Generate a list of random F that looks like(1,0.1,-0.5,1) matrices for a 2D state (position and velocity).
    Args:
        num_F (int): Number of random F matrices to generate.
        state_dim (int): Dimensionality of the state vector.
        delta_t (float): Time step.
    Returns:
        List[torch.Tensor]: List of random state evolution matrices F.
    """
    F_matrices = []
    # F = torch.tensor([[0.999, 0.1],
    #                   [0.2, 0.8]], device=DEVICE)
    if F_init ==None:
        F = torch.tensor([[0.83, 0.2],
                        [0.2, 0.83]], device=DEVICE)
    else:
        F = F_init
    for k in range(num_F+1):
        # F_i = torch.tensor([[0.63, 0.0021], [0.0021, 1.0299]], device=DEVICE)
        # F_i = torch.tensor([[0.7954, 0.1970], [0.1970, 0.8646]], device=DEVICE)
        if F_init ==None:
            F_i = rotate_F(F, i=0, j=1, theta=0.2,mult=1, many=False, randomit=True)
        else:
            F_i = rotate_F(F[k], i=0, j=1, theta=0.2,mult=1, many=False, randomit=True)
        # F_i = rotate_F(F, i=0, j=1, theta=0.6,mult=1, many=False, randomit=False)
                # F = torch.tensor([[-0.9, 0.],
        #                    [0.05, 1.98]])
#         F[0, 1] = F[0, 1] + uniform_two_ranges(0.0, 1) * delta_t * 0.125  # random
#         F[1, 0] = F[1, 0] + uniform_two_ranges(0.0, 1) * delta_t * 0.125  # Add random coupling
#         F[0, 0] = F[0, 0] + uniform_two_ranges(0.0, 1) * delta_t * 0.125  # random
#         F[1, 1] = F[1, 1] + uniform_two_ranges(0.0, 1) * delta_t * 0.125  # Add random coupling
        # # F = torch.tensor([[0.6, 0.2],
        # #                   [0.2, 0.6]])
        # F[0, 1] = 0.2 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # random
        # F[1, 0] = 0.2 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # Add random coupling
        # F[0, 0] = 0.6 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # random
        # F[1, 1] = 0.6 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # Add random coupling
        F_matrices.append(F_i)
    # print('ori seltaaaaaa', F_i)
    return F_matrices

def build_rotation_matrix_3d(theta_x, theta_y, theta_z, device, dtype):
    cx, sx = torch.cos(theta_x), torch.sin(theta_x)
    cy, sy = torch.cos(theta_y), torch.sin(theta_y)
    cz, sz = torch.cos(theta_z), torch.sin(theta_z)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], device=device, dtype=dtype)

    Ry = torch.tensor([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], device=device, dtype=dtype)

    Rz = torch.tensor([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], device=device, dtype=dtype)

    return Rz @ Ry @ Rx

def rotate_H(H,  theta=0.087, many=True, randomit=False, same_angle=False):
    """
    Rotate H using full 3D rotation like RTSNet paper:
        H_rot = R @ H

    same_angle=False (default): keep the original behavior (per-axis independent
        random angles when randomit=True).
    same_angle=True: draw ONE random angle ~ U(0, theta) and apply the SAME angle
        to all three axes (x=y=z). Only affects the randomit=True paths.
    """
    def apply_rotation(H_single, theta):
        device = H_single.device
        dtype = H_single.dtype

        if randomit and not same_angle:
            theta_x = torch.rand(1, device=device, dtype=dtype) * theta
            theta_y = torch.rand(1, device=device, dtype=dtype) * theta
            theta_z = torch.rand(1, device=device, dtype=dtype) * theta
        else:
            # not random, OR same_angle: use the (already-resolved) scalar on all axes
            theta_x = theta
            theta_y = theta
            theta_z = theta

        R = build_rotation_matrix_3d(theta_x, theta_y, theta_z, device, dtype)

        return R @ H_single

    if not many:
        if randomit:
            if same_angle:
                theta = float(torch.rand(1).item()) * theta   # one angle in [0, theta], all axes
            else:
                theta = uniform_two_ranges(0,1)*theta
        theta = torch.tensor(theta, device=H.device)
        H=apply_rotation(H, theta)
        print(H)
        return H
    else:
        rotated_list = []
        for H_i in H:
            if randomit:
                if same_angle:
                    angle = float(torch.rand(1).item()) * theta   # one angle in [0, theta], all axes
                else:
                    angle = uniform_two_ranges(0.0,1)*theta
                angle = torch.tensor(angle, device=device)
            else:
                angle = torch.tensor(theta, device=device)
            rotated_list.append(apply_rotation(H_i, angle))
            # delta = (F_i- apply_rotation(F_i, angle, i, j)).norm()
            # print("Deviation:", delta.item())
        if rotated_list:
            print(' a sample of the F switched ', rotated_list[-1])
        return torch.stack(rotated_list)
def estimate_H_ls(y, x):
    # y: [n, T]
    # x: [m, T]

    C1 = y @ x.T                     # [n, m]
    C2 = x @ x.T                     # [m, m]

    eps = 1e-6 * torch.eye(x.size(0), device=x.device, dtype=x.dtype)
    C2 = C2 + eps

    H_hat = torch.linalg.solve(C2.T, C1.T).T
    return H_hat
def estimate_Q_R_from_true_data(all_inputs_by_H,
                                all_targets_by_H,
                                all_H_matrices,
                                f,
                                Q_structure=None,
                                R_structure=None,
                                device=None,
                                dtype=torch.float32):
    """
    Estimate:
      1) full Q_hat, R_hat
      2) scalar q2_hat, r2_hat such that
            Q ~= q2_hat * Q_structure
            R ~= r2_hat * R_structure

    Assumptions:
      - all_targets_by_H[d] shape: [N_T, m, T]
      - all_inputs_by_H[d]  shape: [N_T, n, T]
      - all_H_matrices[d] is a list of length N_T,
        each item H_seq shape [n, m]
      - f(x) takes x shape [m,1] and returns [m,1]
    """

    if device is None:
        device = all_targets_by_H[0].device

    m = all_targets_by_H[0].shape[1]
    n = all_inputs_by_H[0].shape[1]

    Q_sum = torch.zeros((m, m), device=device, dtype=dtype)
    R_sum = torch.zeros((n, n), device=device, dtype=dtype)

    count_q = 0
    count_r = 0

    for d in range(len(all_targets_by_H)):
        X = all_targets_by_H[d].to(device)   # [N_T, m, T]
        Y = all_inputs_by_H[d].to(device)    # [N_T, n, T]
        H_list = all_H_matrices[d]           # list length N_T, each [n,m]

        N_T = X.shape[0]
        T = X.shape[2]

        for k in range(N_T):
            Hk = H_list[k].to(device)

            # ---------- R from measurement residuals ----------
            # v_t = y_t - H_t x_t
            for t in range(T):
                x_t = X[k, :, t].unsqueeze(-1)   # [m,1]
                y_t = Y[k, :, t].unsqueeze(-1)   # [n,1]

                y_pred = Hk @ x_t                # [n,1]
                v_t = y_t - y_pred               # [n,1]

                R_sum += v_t @ v_t.T
                count_r += 1

            # ---------- Q from process residuals ----------
            # w_t = x_t - f(x_{t-1})
            for t in range(1, T):
                x_prev = X[k, :, t - 1].unsqueeze(-1)   # [m,1]
                x_t    = X[k, :, t].unsqueeze(-1)       # [m,1]

                x_pred = f(x_prev)                      # [m,1]
                w_t = x_t - x_pred                      # [m,1]

                Q_sum += w_t @ w_t.T
                count_q += 1

    Q_hat_full = Q_sum / max(count_q, 1)
    R_hat_full = R_sum / max(count_r, 1)

    results = {
        "Q_hat_full": Q_hat_full,
        "R_hat_full": R_hat_full,
        "count_q": count_q,
        "count_r": count_r,
    }

    # ------------------------------------------------
    # If you want scalar q2_hat and r2_hat on structure
    # Solve in LS sense:
    #   q2_hat = argmin ||Q_hat_full - q2 * Q_structure||_F^2
    #   r2_hat = argmin ||R_hat_full - r2 * R_structure||_F^2
    # Closed form:
    #   a_hat = <A,S> / <S,S>
    # ------------------------------------------------
    if Q_structure is not None:
        QS = Q_structure.to(device=device, dtype=dtype)
        q2_hat = torch.sum(Q_hat_full * QS) / torch.sum(QS * QS)
        results["q2_hat"] = q2_hat
        results["Q_hat_structured"] = q2_hat * QS

    if R_structure is not None:
        RS = R_structure.to(device=device, dtype=dtype)
        r2_hat = torch.sum(R_hat_full * RS) / torch.sum(RS * RS)
        results["r2_hat"] = r2_hat
        results["R_hat_structured"] = r2_hat * RS

    return results
def generate_random_H_matrices(num_H, obs_dim=2, state_dim=2,theta=0.4, H_init=None, same_angle=False):
    """
    Generate a list of random H matrices using rotation.
    Args:
        num_H (int): Number of random H matrices to generate.
        obs_dim (int): Dimensionality of the observation vector.
        state_dim (int): Dimensionality of the state vector.
        H_init (torch.Tensor or list, optional): Initial H matrix or list of matrices. Defaults to None.
    Returns:
        List[torch.Tensor]: List of random observation matrices H.
    """
    H_matrices = []
    if H_init is None:
        # Default H: start with a base that isn't identity for better diversity
        # H = torch.tensor([[1., 1], [0.25, 1]], device=DEVICE, dtype=torch.float32)
        H= torch.eye(3, device=device)
    else:
        H = H_init

    for k in range(num_H + 1):
        if H_init is None:
            # For square matrices, rotate directly in observation space
            H_i = rotate_H(H,  theta=theta, many=False, randomit=True, same_angle=same_angle)
            # print('ori', H_i)
        else:
            H_i = rotate_H(H[k], theta=theta, many=False, randomit=True, same_angle=same_angle)
        # H_i = torch.tensor([[1., 1], [0.25, 1]], device=DEVICE, dtype=torch.float32)
        H_matrices.append(H_i)
    return H_matrices


def generate_random_H_matrices_old(num_H, obs_dim=2, state_dim=2, H_init=None):
    """
    Generate a list of random H matrices using rotation.
    Args:
        num_H (int): Number of random H matrices to generate.
        obs_dim (int): Dimensionality of the observation vector.
        state_dim (int): Dimensionality of the state vector.
        H_init (torch.Tensor or list, optional): Initial H matrix or list of matrices. Defaults to None.
    Returns:
        List[torch.Tensor]: List of random observation matrices H.
    """
    H_matrices = []
    if H_init is None:
        # Default H: start with a base that isn't identity for better diversity
        H = torch.tensor([[1., 1], [0.25, 1]], device=DEVICE, dtype=torch.float32)
    else:
        H = H_init
    for k in range(num_H + 1):
        if H_init is None:
            # For square matrices, rotate directly in observation space
            H_i = rotate_F(H, i=0, j=1, theta=1, mult=1, many=False, randomit=True)
        else:
            H_i = rotate_F(H[k], i=0, j=1, theta=1,mult=1, many=False, randomit=True)
        # H_i = torch.tensor([[1., 1], [0.25, 1]], device=DEVICE, dtype=torch.float32)
        H_matrices.append(H_i)
    return H_matrices

def change_F(F, mult=0.0001, many=True):

    def apply_change(F_single, mult):

        F_single[0, 1] = F_single[0, 1] + torch.randn((1), device=DEVICE).item()*mult
        F_single[1, 0] = F_single[1, 0] + torch.randn((1), device=DEVICE).item()*mult
        F_single[0, 0] = F_single[0, 0] + torch.randn((1), device=DEVICE).item()*mult
        F_single[1, 1] = F_single[1, 1] + torch.randn((1), device=DEVICE).item()*mult

        return F_single

    if not many:
        return apply_change(F, mult)
    else:
        LIST_F = []
        for F_i in F:
            F_i2=apply_change(F_i,mult)
            LIST_F.append(F_i2)
            #delta = (F_i- F_i2).norm()
            #print("Deviation:", delta.item())

        return LIST_F



def det(F, mult=0.0001, many=True):

    def apply_change1(F_single, mult):
        F_single = torch.tensor([[0.83, 0.1],
                          [0., 0.999]],device=F_single.device, dtype=F_single.dtype)
        F_single[0, 1] = F_single[0, 1] + uniform_two_ranges(0.0, 1) * mult * 0.25  # Add random coupling
        F_single[1, 0] = F_single[1, 0] + uniform_two_ranges(0.0, 1) * mult * 0.25  # Add random coupling
        F_single[0, 0] = F_single[0, 0] + uniform_two_ranges(0.0, 1) * mult * 0.25  # Add random coupling
        F_single[1, 1] = F_single[1, 1] + uniform_two_ranges(0.0, 1) * mult * 0.25  # Add random coupling

        return F_single

    LIST_F = []
    for F_i in F:
        F_i2=apply_change1(F_i,mult)
        LIST_F.append(F_i2)

    return LIST_F







def rotate_F(F, i=0, j=1, theta=0.087,mult=1, many=True, randomit=False):
    """
    Apply Givens rotation to matrix F (or list of matrices) in (i,j) plane.

    Args:
        F (torch.Tensor or list of torch.Tensor): n×n matrix or list of such.
        i, j (int): Indices of rotation plane.
        theta (float): Max rotation angle in radians.
        many (bool): If True, rotate each matrix in list F.
        randomit (bool): If True, use random angle in [0, theta*pi].

    Returns:
        torch.Tensor or list of torch.Tensor: Rotated matrix/matrices.
    """
    def apply_rotation(F_single, theta, i, j):
        n = F_single.shape[0]
        R = torch.eye(n, dtype=F_single.dtype, device=F_single.device)
        R[i, i] = torch.cos(theta)
        R[i, j] = -torch.sin(theta)
        R[j, i] = torch.sin(theta)
        R[j, j] = torch.cos(theta)

        H = R @ F_single @ R.T
        print('DEHIL',H, theta,i,j)
        return H

    if not many:
        if randomit:
            theta = uniform_two_ranges(0,1)*theta
        theta = torch.tensor(theta, device=F.device)
        H=apply_rotation(F, theta, i, j)
        print(H)
        return H
    else:
        rotated_list = []
        for F_i in F:
            if randomit:
                angle = uniform_two_ranges(0.0,1)*theta
                angle = torch.tensor(angle, device=F[0].device)
            else:
                angle = torch.tensor(theta, device=F[0].device)
            rotated_list.append(apply_rotation(F_i, angle, i, j))
            # delta = (F_i- apply_rotation(F_i, angle, i, j)).norm()
            # print("Deviation:", delta.item())
        if rotated_list:
            print(' a sample of the F switched ', rotated_list[-1])
        return torch.stack(rotated_list)





class SystemModel:

    def __init__(self, F, Q, H, R, T, T_test,F_initial_guess=None, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.F_train = None
        self.F_valid = None
        self.F_test = None
        self.F_train_TRUE = None
        self.F_valid_TRUE = None
        self.F_test_TRUE = None
        self.device = device


        self.F_gen = None
        self.F_initial_guess = F_initial_guess

        self.m = self.F.size()[0]
        self.Q = Q


        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.H_train = None
        self.H_valid = None
        self.H_test = None
        self.H_train_TRUE = None
        self.H_valid_TRUE = None
        self.H_test_TRUE = None
        self.H_gen = None
        self.n = self.H.size()[0]
        self.R = R

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m, device=self.device, dtype=self.F.dtype)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m), device=self.device, dtype=self.F.dtype)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            # prior_S should be m×m (state covariance), not n×n
            # RTSNet uses this for hidden state estimation in init_hidden()
            self.prior_S = torch.eye(self.m, device=self.device, dtype=self.H.dtype)
        else:
            self.prior_S = prior_S


    def f(self, x, u=None):
        # State transition: x_next = F*x + B*u (if control exists)
        # print(self.F,'oiriiiiiiiii')
        result = torch.matmul(self.F, x)
        # if self.B is not None and u is not None:
        #     result = result + torch.matmul(self.B, u)
        return result

    def h(self, x):
        return torch.matmul(self.H, x)

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0#initial state
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0#initial P


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T, U=None):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T], device=self.device, dtype=self.F.dtype)
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T], device=self.device, dtype=self.F.dtype)
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        q2 = torch.tensor(0.01, device=self.device, dtype=self.F.dtype)  # Var(q)
        r2 = torch.tensor(0.1, device=self.device, dtype=self.F.dtype)  # Var(r)

        lam_q = 1.0 / torch.sqrt(q2)
        lam_r = 1.0 / torch.sqrt(r2)
        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            # # Get control input at time t if provided
            # ut = None
            # if U is not None:
            #     ut = torch.squeeze(U[:, t])
            #     if ut.dim() == 0:
            #         ut = ut.unsqueeze(0)
            #     ut = ut.unsqueeze(1)  # Make it column vector [p, 1]

            if torch.equal(Q_gen,torch.zeros(self.m,self.m, device=self.device)):# No noise
                xt = self.f(self.x_prev)
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m], device=DEVICE)
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                ##################################ori added
                # lam_vec_q = torch.full((self.m,), lam_q.item(), dtype=xt.dtype, device=xt.device)
                # eq = Exponential(lam_vec_q).sample()  # shape (n,)
                # eq = z - (1.0 / lam_vec_q)
                ###################################
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            # Observation Noise
            if torch.equal(R_gen,torch.zeros(self.n,self.n, device=self.device)):# No noise
                yt = self.H.matmul(xt)
            elif self.n == 1: # 1 dim noise
                yt = self.H.matmul(xt)
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:
                yt = self.H.matmul(xt)
                mean = torch.zeros([self.n], device=DEVICE)
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                ####################################ori added
                # lam_vec_r = torch.full((self.n,), lam_r.item(), dtype=yt.dtype, device=yt.device)
                # er = Exponential(lam_vec_r).sample()  # shape (n,)
                # er = z -(1.0/lam_vec_r)
                ######################################
                er = torch.reshape(er[:], yt.size())
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
    def GenerateBatch(self, size, T,delta=0.5, randomInit=False, randomLength=False,F_gen=True, H_gen=True, x0_list=None,F_init=None, H_init=None):
        if(randomLength):
            # Allocate Empty list for Input
            self.Input = []
            # Allocate Empty list for Target
            self.Target = []
            # Init Sequence Lengths
            T_tensor = torch.round(900*torch.rand(size, device=self.device)).int()+100 # Uniform distribution [100,1000]
        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T, device=self.device, dtype=self.H.dtype)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T, device=self.device, dtype=self.F.dtype)

        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.empty(size, self.m, device=self.device, dtype=self.F.dtype)
        ### Generate Examples
        initConditions = self.m1x_0

        if F_gen ==True :
            F_matrices = generate_random_F_matrices(size//10 + 1,delta,F_init)
        #print('11111111111111', F_matrices)
        else:
            F_matrices = F_gen

        if H_gen == True:
            H_matrices = generate_random_H_matrices(size // 10 + 1, obs_dim=self.n, state_dim=self.m, H_init=H_init)
            print('for ori H matrices', H_matrices[0], H_matrices[1], H_matrices[2])
        else:
            H_matrices = H_gen

        for i in range(0, size):
            index_F =i//10
            # Ensure F/H live on the model device (F_gen, H_gen or x0_list may be on CPU)
            self.F = F_matrices[index_F].to(self.device)
            self.F_T = self.F.T
            self.H = H_matrices[index_F].to(self.device)
            self.H_T = self.H.T
            if x0_list is not None:
                x0_i = x0_list[i]
                # ensure column shape [m,1]
                if x0_i.dim() == 1:
                    x0_i = x0_i.unsqueeze(-1)

                initConditions = x0_i.to(self.device)
            else:
                initConditions = self.m1x_0.to(self.device)


            # Generate Sequence

            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                """ 
                ### Uncomment this if Uniform Distribution for random init 
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
                self.m1x_0_rand[i,:] = torch.squeeze(initConditions)
                """

                ### if Normal Distribution for random init
                distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                initConditions = distrib.rsample()
                self.m1x_0_rand[i,:] = torch.squeeze(initConditions)


            self.InitSequence(initConditions, self.m2x_0)### for sequence generation

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
        print('size',self.Input.size())
        return F_matrices, H_matrices

    def sampling(self, q, r, gain):

        if (gain != 0):
            gain_q = 0.1
            #aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m, device=self.device)
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m, device=self.device) + aq
        Q_gen = torch.transpose(Aq, 0, 1) * Aq

        if (gain != 0):
            gain_r = 0.5
            #ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n, device=self.device)
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n, device=self.device) + ar
        R_gen = torch.transpose(Ar, 0, 1) * Ar

        return [Q_gen, R_gen]
