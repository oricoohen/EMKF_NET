"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""
import torch

class KalmanFilter:

    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = SystemModel.m

        self.Q = SystemModel.Q

        self.H = SystemModel.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

    # Predict

    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(torch.matmul(self.F, self.m1x_posterior))

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(torch.matmul(self.H, self.m1x_prior))

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)

        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        if self.dy.dim() == 0: # for linear kinematic, observe-only-postion case
            self.m1x_posterior = self.m1x_prior + torch.squeeze(self.dy * self.KG)
        else:
            self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior,self.m2x_posterior;

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(0, T):
            yt = torch.squeeze(y[:, t])
            xt,sigmat = self.Update(yt)
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)