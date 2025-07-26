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






def compute_A1(x_0, x_t, V):
    """
    Computes A1 = sum_t(x_t x_{t-1}^T + V_t_tminus1)
    Args:
        x_0: [n,1] initial state
        x_t: [SEQ][n, T] smoothed states
        V: [SEQ]TENSOR(m,m,T)
    Returns:
        A1: [SEQ,n, n] matrix
    """
    SEQ = x_t.shape[0]
    n = x_t.shape[1]
    T = x_t.shape[2]
    A1 = torch.zeros((SEQ,n, n), dtype=x_t.dtype, device=x_t.device)
    for seq in range(SEQ):
        A1[seq,:,:] += x_t[seq,:, 0].unsqueeze(1) @ x_0.T + V[seq][:,:,0]
        for t in range(1, T):
            A1[seq,:,:] += x_t[seq,:, t].unsqueeze(1) @ x_t[seq,:, t - 1].unsqueeze(0) + V[seq][:,:,t]
        #A1[seq,:,:] = nonsing_simetric(A1[seq,:,:])
    return A1

def compute_A2(x_0, P_0, x_t, P_t):
    """
    Computes A2 = sum_{t=0 to T-1} (E[x_t x_t^T])
    which is (x_t|T @ x_{t|T}^T + P_t|T) summed over time.

    Args:
        x_0: [n,1] initial state vector.
        P_0: [n, n] initial covariance matrix.
        x_t: [seq,n, T] smoothed states for a single sequence.
        P_t: [seq, n, n, T] smoothed covariances for a single sequence.
    Returns:
        A2: [seq, n, n] matrix.
    """
    SEQ,n,T =x_t.shape

    A2= torch.zeros((SEQ,n, n), dtype=x_t.dtype, device=x_t.device)
    # CORRECTION 1: Ensure x_0 is a column vector for the outer product.


    # This loop correctly sums the remaining terms (for t=1 to T-1)
    for seq in range(SEQ):
        A2[seq, :, :] = x_0 @ x_0.T + P_0
        for t in range(T - 1):  # loop t from 0 to T-2
            x_t_minus_1 = x_t[seq,:, t].unsqueeze(1)  # select state at t, which corresponds to x_{t} in sum
            P_t_minus_1 = P_t[seq,:, :, t]  # select covariance at t, which corresponds to P_{t} in sum
            A2[seq,:,:] += (x_t_minus_1 @ x_t_minus_1.T) + P_t_minus_1
        #A2[seq,:,:] = nonsing_simetric(A2[seq,:,:])
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