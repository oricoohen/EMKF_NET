import torch


def nonsing_simetric(A, c=1e-10):
    """
    Ensures the matrix A is symmetric and non-singular.
    Args:
        A: [n, n] torch tensor (matrix)
        c: small constant to ensure SPD if needed
    Returns:
        A: [n, n] SPD matrix
    """
    A = (A + A.T) / 2
    eigvals = torch.linalg.eigvals(A).real
    if torch.any(torch.isclose(eigvals, torch.zeros_like(eigvals))):
        A += c * torch.eye(A.shape[0], device=A.device)
    return A



def Ell(H, R, y, x, P):
    """
    Log-likelihood of observations given predictions.
    Args:
        H: [m, n] observation matrix
        R: [m, m] measurement noise covariance
        y: [m, T] measurements
        x: [n, T] predicted state means
        P: [n, n, T] predicted state covariances
    Returns:
        likelihood: scalar
    """
    T = y.shape[1]
    m = y.shape[0]
    # Initialize the likelihood
    log_likelihood = -T * m / 2 * torch.log(torch.tensor(2 * torch.pi))
    for t in range(T):
        S_t = H @ P[:, :, t] @ H.T + R  # [m, m] Innovation covariance
        residual = y[:, t] - H @ x[:, t]  # [m]Measurement residual
        log_likelihood -= 0.5 * (torch.logdet(S_t) + residual.T @ torch.linalg.pinv(S_t) @ residual)
    return log_likelihood




def compute_A1(x_0, x_t, V,n,T):
    """
    Computes A1 = sum_{t=1}^{T-1} (x_t x_{t-1}^T + V[:,:,t])
    Args:
        x_t: [n, T] smoothed states
        V: [n, n, T] cross-covariance tensor
    Returns:
        A1: [n, n] matrix
    """

    A1 = torch.zeros((n, n), dtype=x_t.dtype, device=x_t.device)
    # print("x_t[:, 0] shape:", x_t[:, 0].shape)
    # print("x_0 shape:", x_0.shape)
    # print("unsqueezed shapes:", x_t[:, 0].unsqueeze(0).shape, x_0.shape)
    A1 += x_t[:, 0].unsqueeze(1) @ x_0.unsqueeze(0) + V[:,:,0]
    for t in range(1, T):
        A1 += x_t[:, t].unsqueeze(1) @ x_t[:, t - 1].unsqueeze(0) + V[:,:,t]
    #nonsing_simetric(A1)
    return A1

def compute_A2(x_0, P_0, x_t, P_t,n,T):
    """
    Computes A2 = sum_t(x_{t-1} x_{t-1}^T + P_{t-1})
    Args:
        x_0: [n,1] initial state
        P_0: [n, n] initial covariance
        x_t: [n, T] smoothed states
        P_t: [n, n, T] smoothed covariances
    Returns:
        A2: [n, n] matrix
    """
    # Compute the first term (x_0 * x_0^T + P_0)
    A2 = x_0.unsqueeze(1)  @ x_0.unsqueeze(0)  + P_0
    for t in range(1, T):
        A2 += x_t[:, t - 1].unsqueeze(1) @ x_t[:, t - 1].unsqueeze(0) + P_t[:, :, t - 1]
    #nonsing_simetric(A2)
    return A2



def compute_A3(x_t, P_t):
    """
    Computes A3 = sum_t(x_t x_t^T + P_t)
    Args:
        x_t: [n, T] smoothed states
        P_t: [n, n, T] smoothed covariances
    Returns:
        A3: [n, n] matrix
    """
    T, n = x_t.shape[1], x_t.shape[0]
    A3 = torch.zeros((n, n), dtype=x_t.dtype, device=x_t.device)
    for t in range(T):
        A3 += x_t[:, t].unsqueeze(1) @ x_t[:, t].unsqueeze(0) + P_t[:, :, t]
    return nonsing_simetric(A3)








def mse(x_true, x_est):
    """
    Mean squared error
    Args:
        x_true: [n, T] true values
        x_est:  [n, T] predicted values
    Returns:
        mse: scalar
    """
    return torch.mean((x_true - x_est) ** 2)