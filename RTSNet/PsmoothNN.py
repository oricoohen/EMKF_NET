import torch
import torch.nn as nn


class PsmoothNN(nn.Module):
    def __init__(self, m, args):
        super().__init__()
        self.start = 0
        self.m = m  # Dimension of state space
        self.seq_len_input = 1
        self.batch_size = 1
        self.mult = 4  # You mentioned mult=4

        # Define input size for the DNN layer (self.m ** 2) * mult
        self.d_input_FC8 = (self.m ** 2)*self.mult
        self.d_output_FC8 = self.m**2
        # New hidden dimensions for the DNN layers (not using specific constants like 4 and 2, but adjustable)
        self.d_hidden_FC8_1 = self.d_input_FC8
        self.d_hidden_FC8_2 = self.d_input_FC8// 2

        # Fully connected layers for DNN with normalization
        self.FC8 = nn.Sequential(
            nn.Linear(self.d_input_FC8, self.d_hidden_FC8_1),  # Input size: (self.m ** 2) * mult
            nn.LayerNorm(self.d_hidden_FC8_1),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC8_1, self.d_hidden_FC8_2),  # Second hidden layer
            nn.LayerNorm(self.d_hidden_FC8_2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC8_2, self.d_output_FC8),  # Output layer
            nn.LayerNorm(self.d_output_FC8),
            nn.ReLU())

        # GRU for P_smooth
        self.d_input_Psmooth = 2 * (self.m ** 2) # m² for Sigma_bw + m² for SGain ori
        self.d_hidden_Psmooth = self.m ** 2
        self.GRU_Psmooth = nn.GRU(self.d_input_Psmooth, self.d_hidden_Psmooth)


        ####normelize the input in leranable way
        self.layernorm_Psmooth = nn.LayerNorm(self.d_input_Psmooth)  # Normalize both covariance matrices

        # Fully Connected (FC) layer to refine P_smooth
        self.d_input_FC_P = self.d_hidden_Psmooth
        self.d_output_FC_P = self.m ** 2
        self.FC_Psmooth = nn.Linear(self.d_input_FC_P, self.d_output_FC_P)
        self.activation = nn.ReLU()


    def forward(self, Sigma_bw, SGain,F=None):
        """
        Forward pass of the P_smooth estimation
        :param Sigma_bw: Smoothed Sigma from RTSNet [m, m]
        :param SGain: Smoother gain from RTSNet [m, m]
        :return: P_smooth estimate
        """
        # Ensure inputs have correct shape
        Sigma_bw = self.standardize(Sigma_bw.view(1, -1))  # [1, m²#mult]
        SGain = self.standardize(SGain.view(1, 1, -1))  # [1, 1, m²]
        ##reduce dimentions
        Sigma_in = self.FC8(Sigma_bw).view(1, 1, -1)

        # Concatenate and normalize
        in_Psmooth = torch.cat((Sigma_in, SGain), dim=2)  # [1, 1, 2*m²]
        in_Psmooth = self.layernorm_Psmooth(in_Psmooth)  # normalize the input
        if self.start == 0:####if this is the first t step put inside the hiddenstate the P[T]
            self.h_Psmooth = in_Psmooth[:,:,:self.d_hidden_Psmooth].clone()
            self.start = 1
        out_Psmooth, self.h_Psmooth = self.GRU_Psmooth(in_Psmooth, self.h_Psmooth)
        P_smooth = self.FC_Psmooth(out_Psmooth)# [1, 1, m²]
        return P_smooth
    def standardize(self, x, eps=1e-5):
        return (x - x.mean()) / (x.std() + eps)

    def compute_loss(self, P_pred_seq, x_target, x_smooth):
        """
        Compute loss for P_smooth optimization.

        Args:
            P_pred_seq: [m, m, T] - Predicted covariance matrices
            x_target: [m, T] - Ground truth state
            x_smooth: [m, T] - Smoothed state estimate

        Returns:
            Tensor: Loss value
        """
        m, T = x_target.shape
        loss = 0.0

        # Compute empirical covariance and compare with predicted
        for t in range(T):
            # Compute empirical covariance
            err = (x_target[:, t] - x_smooth[:, t]).unsqueeze(1)  # [m, 1]
            P_true = err @ err.T  # [m, m]
            P_pred = P_pred_seq[:, :, t]  # [m, m]

            # Add Frobenius norm loss
            loss += torch.norm(P_pred - P_true, p='fro') ** 2

        return loss / T

    def enforce_covariance_properties(self,P, eps=1e-6):
        """
        Ensure that the covariance matrix P is positive    definite (PSD).

    Args:
        P: A square matrix [m, m] representing the covariance matrix.
        eps: Small constant to ensure positive semi-definiteness if necessary.

    Returns:
        P: Adjusted covariance matrix that is PSD.
    """
        # Check if P is positive semi-definite
        # We will check the eigenvalues to determine if all are >= 0
        P = (P + P.T) / 2  # Ensure P is symmetric
        # Compute eigenvalues and eigenvectors. Use torch.linalg.eigh for symmetric tensors.
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        if torch.any(eigenvalues.real < 0):  # If there are negative eigenvalues
            # Clamp eigenvalues to ensure they are at least eps.
            eigenvalues = torch.clamp(eigenvalues, min=eps)
            # Reconstruct the matrix
            P = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

        return P
