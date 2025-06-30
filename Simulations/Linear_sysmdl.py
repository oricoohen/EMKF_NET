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




def generate_random_F_matrices(num_F, delta_t=0.5, state_dim=2):
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
    for _ in range(num_F):
        F = torch.tensor([[1, 1],
                           [0.1, 1]])
        # F = torch.eye(state_dim)
        # F[0, 1] = 1 + torch.randn(1).item() * delta_t*0.5  # random
        # F[1, 0] = 0.1 + torch.randn(1).item() * delta_t*0.5  # Add random coupling
        # F[0, 0] = 1 + torch.randn(1).item() * delta_t*0.5  # random
        # F[1, 1] = 1 + torch.randn(1).item() * delta_t*0.5  # Add random coupling
        # F[0, 1] = 1 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # random
        # F[1, 0] = 0.1 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # Add random coupling
        # F[0, 0] = 1 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # random
        # F[1, 1] = 1 + uniform_two_ranges(0.0, 1) * delta_t*0.5  # Add random coupling
        F_matrices.append(F)
    return F_matrices

def change_F(F, mult=0.0001, many=True):

    def apply_change(F_single, mult):

        F_single[0, 1] = F_single[0, 1] + torch.randn(1).item()*mult
        F_single[1, 0] = F_single[1, 0] + torch.randn(1).item()*mult
        F_single[0, 0] = F_single[0, 0] + torch.randn(1).item()*mult
        F_single[1, 1] = F_single[1, 1] + torch.randn(1).item()*mult

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





def rotate_F(F, i=0, j=1, theta=0.2,mult=1, many=True, randomit=True):
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
        R = torch.eye(n, dtype=F_single.dtype)
        R[i, i] = torch.cos(theta)
        R[i, j] = -torch.sin(theta)
        R[j, i] = torch.sin(theta)
        R[j, j] = torch.cos(theta)

        return R @ F_single*mult @ R.T

    if not many:
        if randomit:
            theta = torch.rand(1) * theta * torch.pi

        return apply_rotation(F, theta, i, j)
    else:
        rotated_list = []
        for F_i in F:
            if randomit:
                angle = torch.rand(1) * theta * torch.pi

            else:
                angle = torch.tensor(theta)

            rotated_list.append(apply_rotation(F_i, angle, i, j))
            delta = (F_i- apply_rotation(F_i, angle, i, j)).norm()
            print("Deviation:", delta.item())

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
        self.F_gen = None
        self.F_initial_guess = F_initial_guess

        self.m = self.F.size()[0]
        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.H = H
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
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S
        

    def f(self, x):
        # print(self.F,'oiriiiiiiiii')
        return torch.matmul(self.F, x)
    
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
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            if torch.equal(Q_gen,torch.zeros(self.m,self.m)):# No noise
                xt = self.F.matmul(self.x_prev)
            elif self.m == 1: # 1 dim noise
                xt = self.F.matmul(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:
                xt = self.F.matmul(self.x_prev)
                mean = torch.zeros([self.m])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                # eq = torch.normal(mean, self.q)
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            # Observation Noise
            if torch.equal(R_gen,torch.zeros(self.n,self.n)):# No noise
                yt = self.H.matmul(xt)
            elif self.n == 1: # 1 dim noise
                yt = self.H.matmul(xt)
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:
                yt = self.H.matmul(xt)
                mean = torch.zeros([self.n])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
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
    def GenerateBatch(self, size, T, delta = 0.5, randomInit=False, randomLength=False,F_gen=None):
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

        ### Generate Examples
        initConditions = self.m1x_0

        F_matrices = generate_random_F_matrices(size//10 +1,delta)
        #print('11111111111111', F_matrices) dehil



        for i in range(0, size):
            if F_gen != None:
                index_F =i//10
                self.F = F_matrices[index_F]
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
        return F_matrices

    def sampling(self, q, r, gain):

        if (gain != 0):
            gain_q = 0.1
            #aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m)
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m) + aq
        Q_gen = torch.transpose(Aq, 0, 1) * Aq

        if (gain != 0):
            gain_r = 0.5
            #ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n)
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n) + ar
        R_gen = torch.transpose(Ar, 0, 1) * Ar

        return [Q_gen, R_gen]
