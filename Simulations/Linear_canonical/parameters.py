"""
This file contains the parameters for the simulations with linear canonical model
* Linear State Space Models with Full Information
    # v = 0, -10, -20 dB
    # scaling model dim to 5x5, 10x10, 20x20, etc
    # scalable trajectory length T
    # random initial state
* Linear SS Models with Partial Information
    # observation model mismatch
    # evolution model mismatch
"""

import torch
device =torch.device("cuda")
m = 2 # state dimension = 2, 5, 10, etc.
n = 2 # observation dimension = 2, 5, 10, etc.

##################################
### Initial state and variance ###
##################################
# m1_0 = torch.zeros(m, 1)
# m2_0 = 0 * torch.eye(m)
m1_0 = torch.tensor([[0.5], [0.5]],device=device)
# m1_0 = torch.tensor([[1.1], [1.7]],device=device)
# m1_0 = torch.tensor([[28.84], [22.05]])
# m1_0 = m1_0.view(-1)
m2_0 = torch.eye(m,device=device)

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################
# # F in canonical form
# F = torch.eye(m)
# F[0] = torch.ones(1,m)
#
F = torch.tensor([[1., 1.],
                  [0.25, 1.]],device=device)


#F_initial_guess = torch.eye(m)
F_initial_guess = None

if m == 2:
    # H = I
    H = torch.eye(2,device=device)
else:
    # H in reverse canonical form
    H = torch.zeros(n,n,device=device)
    H[0] = torch.ones(1,n,device=device)
    for i in range(n):
        H[i,n-1-i] = 1

H = torch.tensor([[1., 1.],
                  [0.25, 1.]],device=device)


#######################
### Rotated F and H ###
#######################
#F_rotated = torch.zeros_like(F)
# F_rotated = None
# H_rotated = torch.zeros_like(H)

if(m==2):
    alpha_degree = 10 # rotation angle in degree
    rotate_alpha = torch.tensor([alpha_degree/180*torch.pi])
    cos_alpha = torch.cos(rotate_alpha)
    sin_alpha = torch.sin(rotate_alpha)
    rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                                [sin_alpha, cos_alpha]])

#   F_rotated = torch.mm(F,rotate_matrix)
#     H_rotated = torch.mm(H,rotate_matrix)

###############################################
### process noise Q and observation noise R ###
###############################################
# Noise variance takes the form of a diagonal matrix
Q_structure = torch.eye(m,device=device)
R_structure = torch.eye(n,device=device)