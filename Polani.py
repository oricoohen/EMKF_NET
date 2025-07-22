import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random

plt.style.use('bmh')
#Check whether a matrix is positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

#Force a matrix to be positive definite if its not
def force_SPD(A, c = 1e-10):
    A = (A + A.T)/2
    if np.amin(np.linalg.eigvals(A)) < 0:
        A += (np.abs(np.amin(np.linalg.eigvals(A))) + c)*np.identity(A.shape[0])
    if np.amin(np.linalg.eigvals(A)) == 0:
        A += c*np.identity(A.shape[0])
    return(A)


def KalmanFilter(F, Q, H, R, z, x_0, P_0):
    T = len(z)
    # Check the dimension of the hidden state
    if isinstance(x_0, np.ndarray) == True:
        n = len(x_0)
    else:
        n = 1
    # Check the dimension of the measurements
    if isinstance(z[0], np.ndarray) == True:
        p = len(z[0])
    else:
        p = 1

    # Multidimensional case
    if n > 1 and p > 1:
        x_hat_minus = np.empty((1, n))
        x_hat = np.array([x_0])
        P_minus = np.empty((1, n, n))
        P = np.array([P_0])
        K = np.empty((1, n, p))

        for i in range(T):
            # State extrapolation equation
            x_hat_minus = np.append(x_hat_minus, [F @ x_hat[i]], axis=0)

            # Covariance extrapolation equation
            P_minus = np.append(P_minus, [F @ P[i] @ F.T + Q], axis=0)

            # Kalman gain
            K = np.append(K, [P_minus[i + 1] @ H.T @ np.linalg.inv(H @ P_minus[i + 1] @ H.T + R)], axis=0)

            # State update equation
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] @ (z[i] - H @ x_hat_minus[i + 1])], axis=0)

            # Covariance update equation
            P = np.append(P, [
                (np.identity(n) - K[i + 1] @ H) @ P_minus[i + 1] @ (np.identity(n) - K[i + 1] @ H).T + K[i + 1] @ R @ K[
                    i + 1].T], axis=0)

    # One-dimensional measurements case
    elif n > 1 and p == 1:
        x_hat_minus = np.empty((1, n))
        x_hat = np.array([x_0])
        P_minus = np.empty((1, n, n))
        P = np.array([P_0])
        K = np.empty((1, n, p))

        for i in range(T):
            # State extrapolation equation
            x_hat_minus = np.append(x_hat_minus, [F @ x_hat[i]], axis=0)

            # Covariance extrapolation equation
            P_minus = np.append(P_minus, [F @ P[i] @ F.T + Q], axis=0)

            # Kalman gain
            K = np.append(K, [P_minus[i + 1] @ H.T @ np.linalg.inv(H @ P_minus[i + 1] @ H.T + R)], axis=0)

            # State update equation
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] @ (z[i] - H @ x_hat_minus[i + 1])], axis=0)

            # Covariance update equation
            P = np.append(P, [
                (np.identity(n) - K[i + 1] @ H) @ P_minus[i + 1] @ (np.identity(n) - K[i + 1] @ H).T + R * K[i + 1] @ K[
                    i + 1].T], axis=0)

    # One-dimensional case
    else:
        x_hat_minus = np.empty(1)
        x_hat = np.array([x_0])
        P_minus = np.empty(1)
        P = np.array([P_0])
        K = np.empty(1)

        for i in range(T):
            # State extrapolation equation
            x_hat_minus = np.append(x_hat_minus, [F * x_hat[i]], axis=0)

            # Covariance extrapolation equation
            P_minus = np.append(P_minus, [F ** 2 * P[i] + Q], axis=0)

            # Kalman gain
            K = np.append(K, [P_minus[i + 1] * H / (H ** 2 * P_minus[i + 1] + R)], axis=0)

            # State update equation
            x_hat = np.append(x_hat, [x_hat_minus[i + 1] + K[i + 1] * (z[i] - H * x_hat_minus[i + 1])], axis=0)

            # Covariance update equation
            P = np.append(P, [(1 - K[i + 1] * H) ** 2 * P_minus[i + 1] + K[i + 1] ** 2 * R], axis=0)

    x_hat_minus = np.delete(x_hat_minus, 0, axis=0)
    P_minus = np.delete(P_minus, 0, axis=0)
    K = np.delete(K, 0, axis=0)

    return (x_hat_minus, P_minus, K, x_hat, P)


def KalmanSmoother(F, x_hat_minus, P_minus, x_hat, P):
    T = len(x_hat_minus)
    if isinstance(x_hat[0], np.ndarray) == True:
        n = len(x_hat[0])
    else:
        n = 1

    # Multidimensional case
    if n > 1:
        x_tilde = np.array([x_hat[T]])
        P_tilde = np.array([P[T]])
        S = np.empty((1, n, n))

        for i in reversed(range(T)):
            # Smoothing gain
            S = np.insert(S, 0, [P[i] @ F.T @ np.linalg.inv(P_minus[i])], axis=0)

            # State correction
            x_tilde = np.insert(x_tilde, 0, [x_hat[i] + S[0] @ (x_tilde[0] - x_hat_minus[i])], axis=0)

            # Covariance correction
            P_tilde = np.insert(P_tilde, 0, [P[i] + S[0] @ (P_tilde[0] - P_minus[i]) @ S[0].T], axis=0)

    # One-dimensional case
    else:
        x_tilde = np.array([x_hat[T]])
        P_tilde = np.array([P[T]])
        S = np.empty(1)

        for i in reversed(range(T)):
            # Smoothing gain
            S = np.insert(S, 0, [P[i] * F / P_minus[i]], axis=0)

            # State correction
            x_tilde = np.insert(x_tilde, 0, [x_hat[i] + S[0] * (x_tilde[0] - x_hat_minus[i])], axis=0)

            # Covariance correction
            P_tilde = np.insert(P_tilde, 0, [P[i] + S[0] ** 2 * (P_tilde[0] - P_minus[i])], axis=0)

    S = np.delete(S, len(S) - 1, axis=0)

    return (S, x_tilde, P_tilde)


def Lag1AutoCov(K, S, F, H, P):
    T = len(P) - 1
    if isinstance(F, np.ndarray) == True:
        n = F.shape[0]
    else:
        n = 1

    # Multidimensional case
    if n > 1:
        V = np.array([(np.identity(n) - K[T - 1] @ H) @ F @ P[T - 1]])

        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] @ S[i - 1].T + S[i] @ (V[0] - F @ P[i]) @ S[i - 1].T], axis=0)

    # One-dimensional case
    else:
        V = np.array([(1 - K[T - 1] * H) * F * P[T - 1]])

        for i in reversed(range(1, T)):
            V = np.insert(V, 0, [P[i] * S[i - 1].T + S[i] * (V[0] - F * P[i]) * S[i - 1].T], axis=0)

    return (V)


def ell(H, R, z, x, P):
    T = len(z)
    if isinstance(x[0], np.ndarray) == True:
        n = len(x[0])
    else:
        n = 1
    if isinstance(z[0], np.ndarray) == True:
        p = len(z[0])
    else:
        p = 1

    likelihood = -T * p / 2 * np.log(2 * np.pi)

    # Multidimensional case
    if n > 1 and p > 1:
        for i in range(T):
            likelihood -= 0.5 * (np.log(np.linalg.det(H @ P[i] @ H.T + R)) + (z[i] - H @ x[i]).T @ np.linalg.inv(
                H @ P[i] @ H.T + R) @ (z[i] - H @ x[i]))

    # One-dimensional measurements case
    elif n > 1 and p == 1:
        for i in range(T):
            likelihood -= 0.5 * (
                        np.log(np.linalg.det(H @ P[i] @ H.T + R)) + (z[i] - H @ x[i]) ** 2 / (H @ P[i] @ H.T + R))
        likelihood = likelihood[0][0]

    # One-dimensional case
    else:
        for i in range(T):
            likelihood -= 0.5 * (np.log(H ** 2 * P[i] + R) + (z[i] - H * x[i]) ** 2 / (H ** 2 * P[i] + R))

    return (likelihood)


def EMKF(F_0, Q_0, H_0, R_0, z, xi_0, L_0, max_it=200, tol_likelihood=0.01, tol_params=0.005,
         em_vars=["F", "Q", "H", "R", "xi", "L"]):
    T = len(z)
    if isinstance(xi_0, np.ndarray) == True:
        n = len(xi_0)
    else:
        n = 1
    if isinstance(z[0], np.ndarray) == True:
        p = len(z[0])
    else:
        p = 1

    # Initialization
    F = np.array([F_0])
    Q = np.array([Q_0])
    H = np.array([H_0])
    R = np.array([R_0])
    xi = np.array([xi_0])
    L = np.array([L_0])

    likelihood = np.empty(1)

    # Multidimensional case
    if n > 1 and p > 1:
        A_5 = np.zeros((p, p))
        for j in range(T):
            A_5 += np.outer(z[j], z[j])

        for i in range(max_it):
            if i > 0 and i % 50 == 0:
                print(f"Iteration {i}")
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            # Convergence check for likelihood
            convergence_count = 0
            if i >= 1 and likelihood[i + 1] - likelihood[i] < tol_likelihood:
                convergence_count += 1

            # M-step
            A_1 = np.zeros((n, n))
            A_2 = np.zeros((n, n))
            A_3 = np.zeros((n, n))
            A_4 = np.zeros((p, n))

            for j in range(T):
                A_1 += np.outer(x_tilde[j + 1], x_tilde[j]) + V[j]
                A_2 += np.outer(x_tilde[j], x_tilde[j]) + P_tilde[j]
                A_3 += np.outer(x_tilde[j + 1], x_tilde[j + 1]) + P_tilde[j + 1]
                A_4 += np.outer(z[j], x_tilde[j + 1])

            if "F" in em_vars:
                # Update equation for F
                F = np.append(F, [A_1 @ np.linalg.inv(A_2)], 0)

                # Convergence check for F
                if i >= 1 and np.all(np.abs(F[i + 1] - F[i]) < tol_params):
                    convergence_count += 1
            else:
                F = np.append(F, [F_0], 0)

            if "Q" in em_vars:
                # Update equation for Q
                if "F" in em_vars:
                    Q_i = (A_3 - F[i + 1] @ A_1.T) / T
                else:
                    Q_i = (A_3 - A_1 @ np.linalg.inv(A_2) @ A_1.T) / T

                    # Check whether the updated estimate for Q is positive definite
                if is_pos_def(Q_i) == False:
                    print(f"Q NON-SPD at iteration {i}")
                    Q_i = force_SPD(Q_i)

                Q = np.append(Q, [Q_i], 0)

                # Convergence check for Q
                if i >= 1 and np.all(np.abs(Q[i + 1] - Q[i]) < tol_params):
                    convergence_count += 1
            else:
                Q = np.append(Q, [Q_0], 0)

            if "H" in em_vars:
                # Update equation for H
                H = np.append(H, [A_4 @ np.linalg.inv(A_3)], 0)

                # Convergence check for H
                if i >= 1 and np.all(np.abs(H[i + 1] - H[i]) < tol_params):
                    convergence_count += 1
            else:
                H = np.append(H, [H_0], 0)

            if "R" in em_vars:
                # Update equation for R
                if "H" in em_vars:
                    R_i = (A_5 - H[i + 1] @ A_4.T) / T
                else:
                    R_i = (A_5 - A_4 @ np.linalg.inv(A_3) @ A_4.T) / T
                # Check whether the updated estimate of R is positive definite
                if is_pos_def(R_i) == False:
                    print(f"R_{i} NON-SPD")
                R_i = force_SPD(R_i)

                R = np.append(R, [R_i], axis=0)

                # Convergence check for R
                if i >= 1 and np.all(np.abs(R[i + 1] - R[i]) < tol_params):
                    convergence_count += 1
            else:
                R = np.append(R, [R_0], 0)

            if "xi" in em_vars:
                # Update equation for xi
                xi = np.append(xi, [x_tilde[0]], axis=0)

                # Convergence check for xi
                if i >= 1 and np.all(np.abs(xi[i + 1] - xi[i]) < tol_params):
                    convergence_count += 1
            else:
                xi = np.append(xi, [xi_0], 0)

            if "L" in em_vars:
                # Update equation for Lambda
                L = np.append(L, [P_tilde[0]], axis=0)

                # Convergence check for Lambda
                if i >= 1 and np.all(np.abs(L[i + 1] - L[i]) < tol_params):
                    convergence_count += 1
            else:
                L = np.append(L, [L_0], axis=0)

            if convergence_count == len(em_vars) + 1:
                break

        iterations = i + 1

    # One-dimensional measurements case
    elif n > 1 and p == 1:
        A_5 = 0
        for j in range(T):
            A_5 += z[j] ** 2

        for i in range(max_it):
            if i > 0 and i % 50 == 0:
                print(f"Iteration {i}")
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            convergence_count = 0
            if i >= 1 and likelihood[i + 1] - likelihood[i] < tol_likelihood:
                convergence_count += 1

            # M-step
            A_1 = np.zeros((n, n))
            A_2 = np.zeros((n, n))
            A_3 = np.zeros((n, n))
            A_4 = np.zeros((p, n))

            for j in range(T):
                A_1 += np.outer(x_tilde[j + 1], x_tilde[j]) + V[j]
                A_2 += np.outer(x_tilde[j], x_tilde[j]) + P_tilde[j]
                A_3 += np.outer(x_tilde[j + 1], x_tilde[j + 1]) + P_tilde[j + 1]
                A_4 += z[j] * x_tilde[j + 1]

            if "F" in em_vars:
                F = np.append(F, [A_1 @ np.linalg.inv(A_2)], 0)

                if i >= 1 and np.all(np.abs(F[i + 1] - F[i]) < tol_params):
                    convergence_count += 1
            else:
                F = np.append(F, [F_0], 0)

            if "Q" in em_vars:
                if "F" in em_vars:
                    Q_i = (A_3 - F[i + 1] @ A_1.T) / T
                else:
                    Q_i = (A_3 - A_1 @ np.linalg.inv(A_2) @ A_1.T) / T
                if is_pos_def(Q_i) == False:
                    print(f"Q_{i} NON-SPD")
                Q_i = force_SPD(Q_i)

                Q = np.append(Q, [Q_i], 0)

                if i >= 1 and np.all(np.abs(Q[i + 1] - Q[i]) < tol_params):
                    convergence_count += 1
            else:
                Q = np.append(Q, [Q_0], 0)

            if "H" in em_vars:
                H = np.append(H, [A_4 @ np.linalg.inv(A_3)], 0)

                if i >= 1 and np.all(np.abs(H[i + 1] - H[i]) < tol_params):
                    convergence_count += 1
            else:
                H = np.append(H, [H_0], 0)

            if "R" in em_vars:
                if "H" in em_vars:
                    R_i = float((A_5 - H[i + 1] @ A_4.T) / T)
                else:
                    R_i = float((A_5 - A_4 @ np.linalg.inv(A_3) @ A_4.T) / T)

                R = np.append(R, [R_i], axis=0)
                if i >= 1 and np.abs(R[i + 1] - R[i]) < tol_params:
                    convergence_count += 1
            else:
                R = np.append(R, [R_0], 0)

            if "xi" in em_vars:
                xi = np.append(xi, [x_tilde[0]], axis=0)

                if i >= 1 and np.all(np.abs(xi[i + 1] - xi[i]) < tol_params):
                    convergence_count += 1
            else:
                xi = np.append(xi, [xi_0], 0)

            if "L" in em_vars:
                L = np.append(L, [P_tilde[0]], axis=0)

                if i >= 1 and np.all(np.abs(L[i + 1] - L[i]) < tol_params):
                    convergence_count += 1
            else:
                L = np.append(L, [L_0], axis=0)

            if convergence_count == len(em_vars) + 1:
                break

        iterations = i + 1

    # One-dimensional case
    else:
        A_5 = 0
        for j in range(T):
            A_5 += z[j] ** 2

        for i in range(max_it):
            if i > 0 and i % 50 == 0:
                print(f"Iteration {i}")
            # E-step
            x_hat_minus, P_minus, K, x_hat, P = KalmanFilter(F[i], Q[i], H[i], R[i], z, xi[i], L[i])
            S, x_tilde, P_tilde = KalmanSmoother(F[i], x_hat_minus, P_minus, x_hat, P)
            V = Lag1AutoCov(K, S, F[i], H[i], P)

            likelihood = np.append(likelihood, [ell(H[i], R[i], z, x_hat_minus, P_minus)], axis=0)

            convergence_count = 0
            if i >= 1 and likelihood[i + 1] - likelihood[i] < tol_likelihood:
                convergence_count += 1

            # M-step
            A_1 = 0
            A_2 = 0
            A_3 = 0
            A_4 = 0

            for j in range(T):
                A_1 += x_tilde[j + 1] * x_tilde[j] + V[j]
                A_2 += x_tilde[j] ** 2 + P_tilde[j]
                A_3 += x_tilde[j + 1] ** 2 + P_tilde[j + 1]
                A_4 += z[j] * x_tilde[j + 1]

            if "F" in em_vars:
                F = np.append(F, [A_1 / A_2], 0)

                if i >= 1 and np.abs(F[i + 1] - F[i]) < tol_params:
                    convergence_count += 1
            else:
                F = np.append(F, [F_0], 0)

            if "Q" in em_vars:
                if "F" in em_vars:
                    Q_i = (A_3 - F[i + 1] * A_1) / T
                else:
                    Q_i = (A_3 - A_1 ** 2 / A_2) / T

                Q = np.append(Q, [Q_i], 0)

                if i >= 1 and np.abs(Q[i + 1] - Q[i]) < tol_params:
                    convergence_count += 1
            else:
                Q = np.append(Q, [Q_0], 0)

            if "H" in em_vars:
                H = np.append(H, [A_4 / A_3], 0)

                if i >= 1 and np.abs(H[i + 1] - H[i]) < tol_params:
                    convergence_count += 1
            else:
                H = np.append(H, [H_0], 0)

            if "R" in em_vars:
                if "H" in em_vars:
                    R_i = (A_5 - H[i + 1] * A_4) / T
                else:
                    R_i = (A_5 - A_4 ** 2 / A_3) / T

                R = np.append(R, [R_i], axis=0)
                if i >= 1 and np.all(np.abs(R[i + 1] - R[i]) < tol_params):
                    convergence_count += 1
            else:
                R = np.append(R, [R_0], 0)

            if "xi" in em_vars:
                xi = np.append(xi, [x_tilde[0]], axis=0)

                if i >= 1 and np.abs(xi[i + 1] - xi[i]) < tol_params:
                    convergence_count += 1
            else:
                xi = np.append(xi, [xi_0], 0)

            if "L" in em_vars:
                L = np.append(L, [P_tilde[0]], axis=0)

                if i >= 1 and np.abs(L[i + 1] - L[i]) < tol_params:
                    convergence_count += 1
            else:
                L = np.append(L, [L_0], axis=0)

            if convergence_count == len(em_vars) + 1:
                break

        iterations = i + 1

    likelihood = np.delete(likelihood, 0, axis=0)

    return (F, Q, H, R, xi, L, likelihood, iterations)


def grad(F, Q, H, R, xi, L, z, x, P, V, em_vars=["F", "Q", "H", "R", "xi", "L"]):
    T = len(z)
    n = len(x[0])
    p = len(z[0])

    A_1 = np.zeros((n, n))
    A_2 = np.zeros((n, n))
    A_3 = np.zeros((n, n))
    A_4 = np.zeros((p, n))
    A_5 = np.zeros((p, p))

    for j in range(T):
        A_1 += np.outer(x[j + 1], x[j]) + V[j]
        A_2 += np.outer(x[j], x[j]) + P[j]
        A_3 += np.outer(x[j + 1], x[j + 1]) + P[j + 1]
        A_4 += np.outer(z[j], x[j + 1])
        A_5 += np.outer(z[j], z[j])

    gradient = np.empty(1)

    if "F" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten(A_1 - F @ A_2, order='F'), axis=0)

    if "Q" in em_vars:
        gradient = np.append(gradient,
                             np.ndarray.flatten((T * Q.T - A_3 + 2 * A_1 @ F.T - F @ A_2 @ F.T) / 2, order='F'), axis=0)

    if "H" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten(A_4 - H @ A_3, order='F'), axis=0)

    if "R" in em_vars:
        gradient = np.append(gradient,
                             np.ndarray.flatten((T * R.T - A_5 + 2 * A_4 @ H.T - H @ A_3 @ H.T) / 2, order='F'), axis=0)

    if "xi" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten((x[0] - xi).T, order='F'), axis=0)

    if "L" in em_vars:
        gradient = np.append(gradient, np.ndarray.flatten(np.outer(x[0] - xi, x[0] - xi) + P[0], order='F'), axis=0)

    gradient = np.delete(gradient, 0, axis=0)
    return (gradient)


def HessianApprox(F, Q, H, R, xi, L, z, x, P, V, em_vars=["F", "Q", "H", "R", "xi", "L"], shift=0.5, scale=1000,
                  MC_size=10000):
    T = len(z)
    n = len(x[0])
    p = len(z[0])

    gen = np.random.default_rng(seed=None)

    d = 0
    for var in em_vars:
        d += np.prod(locals()[var].shape)

    Hessian = np.empty((1, d, d))

    for i in range(MC_size):
        if i % 100 == 0:
            print(f"Simulation {i}")
        delta = np.empty(1)

        if "F" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n ** 2) - shift) / scale, axis=0)
            F_per_plus = F + np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
            F_per_minus = F - np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
        else:
            F_per_plus = F
            F_per_minus = F

        if "Q" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n ** 2) - shift) / scale, axis=0)
            Q_per_plus = Q + np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
            Q_per_minus = Q - np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
        else:
            Q_per_plus = Q
            Q_per_minus = Q

        if "H" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, p * n) - shift) / scale, axis=0)
            H_per_plus = H + np.reshape(delta[len(delta) - p * n:], (p, n), order='F')
            H_per_minus = H - np.reshape(delta[len(delta) - p * n:], (p, n), order='F')
        else:
            H_per_plus = H
            H_per_minus = H

        if "R" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, p ** 2) - shift) / scale, axis=0)
            R_per_plus = R + np.reshape(delta[len(delta) - p ** 2:], (p, p), order='F')
            R_per_minus = R - np.reshape(delta[len(delta) - p ** 2:], (p, p), order='F')
        else:
            R_per_plus = R
            R_per_minus = R

        if "xi" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n) - shift) / scale, axis=0)
            xi_per_plus = xi + np.reshape(delta[len(delta) - n:], n, order='F')
            xi_per_minus = xi - np.reshape(delta[len(delta) - n:], n, order='F')
        else:
            xi_per_plus = xi
            xi_per_minus = xi

        if "L" in em_vars:
            delta = np.append(delta, (gen.binomial(1, 0.5, n ** 2) - shift) / scale, axis=0)
            L_per_plus = L + np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
            L_per_minus = L - np.reshape(delta[len(delta) - n ** 2:], (n, n), order='F')
        else:
            L_per_plus = L
            L_per_minus = L

        delta = np.delete(delta, 0, axis=0)

        grad_per_plus = grad(F_per_plus, Q_per_plus, H_per_plus, R_per_plus, xi_per_plus, L_per_plus, z, x, P, V,
                             em_vars)
        grad_per_minus = grad(F_per_minus, Q_per_minus, H_per_minus, R_per_minus, xi_per_minus, L_per_minus, z, x, P, V,
                              em_vars)
        delta_inv = 1 / delta

        grad_diff = grad_per_plus - grad_per_minus
        Hessian = np.append(Hessian,
                            [0.5 * (np.outer(grad_diff / 2, delta_inv) + np.outer(grad_diff / 2, delta_inv).T)], axis=0)

    Hessian = np.delete(Hessian, 0, axis=0)

    return (np.mean(Hessian, axis=0))


def rotate_one(F, theta = 0.78, i=0, j=1):
    """
    Given a list of square numpy arrays F_list, pick one index at random,
    rotate that matrix in the (i,j) plane by a random angle in [0, theta],
    and return a new list with that one replaced by its rotated version.

    Args:
      F_list : List[np.ndarray]
        List of (n×n) matrices.
      theta : float
        Maximum rotation angle in radians.
      i, j : int
        Indices of the rotation plane (default 0,1).

    Returns:
      List[np.ndarray]
        Copy of F_list with exactly one matrix rotated.
    """
    # copy the list so we don't mutate the input
    n = 2
    # pick random angle uniform [0, theta]
    # angle = np.random.uniform(0, 0.78)
    angle = 0.78
    # build Givens rotation
    R = np.eye(n)
    R[i, i] = np.cos(angle)
    R[j, j] = np.cos(angle)
    R[i, j] = -np.sin(angle)
    R[j, i] = np.sin(angle)
    F_i = R@(F)@(R.T)
    # apply R @ F0 @ R^T

    return F_i





T = 100
n = 2
p = 2

state = 1
gen = np.random.default_rng(seed=state)


F_sim2 = np.array([[0.83, 0.2],
                   [0.2, 0.83]])          #   [[1,1],[0.1,1]]


Q_sim2 = 0.01 * np.eye(n)               #   process-noise covariance
H_sim2 = np.array([[1., 1.], [0.25, 1.]])                  #   full observation
R_sim2 = 0.10 * np.eye(p)               #   measurement-noise covariance

xi_sim2 = np.array([0.5, 0.5])          #   x₀ mean
L_sim2  = np.eye(n)

x_sim2 = np.array([gen.multivariate_normal(xi_sim2, L_sim2)])
z_sim2 = np.empty((1, p))

for t in range(T):
    x_sim2 = np.append(x_sim2, [F_sim2 @ x_sim2[t] + gen.multivariate_normal(np.zeros(n), Q_sim2)], axis=0)
    z_sim2 = np.append(z_sim2, [H_sim2 @ x_sim2[t + 1] + gen.multivariate_normal(np.zeros(p), R_sim2)], axis=0)

z_sim2 = np.delete(z_sim2, 0, axis=0)

print(f"F = {F_sim2}")
print(f"Q = {Q_sim2}")
print(f"H = {H_sim2}")
print(f"R = {R_sim2}")
print(f"xi = {xi_sim2}")
print(f"Lambda = {L_sim2}")




x_hat_minus_sim2, P_minus_sim2, K_sim2, x_hat_sim2, P_sim2 = KalmanFilter(F_sim2, Q_sim2, H_sim2, R_sim2, z_sim2,
                                                                          xi_sim2, L_sim2)
S_sim2, x_tilde_sim2, P_tilde_sim2 = KalmanSmoother(F_sim2, x_hat_minus_sim2, P_minus_sim2, x_hat_sim2, P_sim2)

for i in range(n):
    print(mean_squared_error(x_sim2[:, i], x_hat_sim2[:, i]))

for i in range(n):
    print(mean_squared_error(x_sim2[:, i], x_tilde_sim2[:, i]))

print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

iter = 3
for i in range(iter):
# Perturb the true parameters
    F_0_sim2 = F_sim2.copy()
    #for j in range(i+1):
        #F_0_sim2 = rotate_one(F_0_sim2)
    F_0_sim2 = rotate_one(F_0_sim2)
    print('ori_ffffffff',F_0_sim2)
    Q_0_sim2 = 0.01 * np.eye(n)               #   process-noise covariance
    H_0_sim2 = np.array([[1., 1.], [0.25, 1.]])                      #   full observation
    R_0_sim2 = 0.10 * np.eye(p)               #   measurement-noise covariance

    xi_0_sim2 = np.array([0.5, 0.5])          #   x₀ mean
    L_0_sim2  = np.eye(n)





    MLE_sim2 = EMKF(F_0_sim2, Q_0_sim2, H_0_sim2, R_0_sim2, z_sim2, xi_0_sim2, L_0_sim2,em_vars=["F"])

    it = MLE_sim2[7]
    print(it)
    F_MLE_sim2 = MLE_sim2[0][it]
    Q_MLE_sim2 = MLE_sim2[1][it]
    H_MLE_sim2 = MLE_sim2[2][it]
    R_MLE_sim2 = MLE_sim2[3][it]
    xi_MLE_sim2 = MLE_sim2[4][it]
    L_MLE_sim2 = MLE_sim2[5][it]

    print(f"diff F = {F_MLE_sim2 - F_sim2}")
    print(f"diff Q = {Q_MLE_sim2 - Q_sim2}")
    print(f"diff H = {H_MLE_sim2 - H_sim2}")
    print(f"diff R = {R_MLE_sim2 - R_sim2}")
    print(f"diff xi = {xi_MLE_sim2 - xi_sim2}")
    print(f"diff L = {L_MLE_sim2 - L_sim2}")
    print(np.linalg.norm(F_MLE_sim2 - F_sim2, ord='fro'))
    print(np.linalg.norm(Q_MLE_sim2 - Q_sim2, ord='fro'))
    print(np.linalg.norm(H_MLE_sim2 - H_sim2, ord='fro'))
    print(np.linalg.norm(R_MLE_sim2 - R_sim2, ord='fro'))
    print(np.linalg.norm(xi_MLE_sim2 - xi_sim2))
    print(np.linalg.norm(L_MLE_sim2 - L_sim2, ord='fro'))

    plt.figure()
    plt.plot(MLE_sim2[6])
    plt.title("Log-likelihood of the Measurements")
    plt.xlabel("Iteration")
    ell2 = "\ell"
    theta = "\u03B8"
    bold_z = "\mathbf{{z}}"
    plt.ylabel(f"${ell2}$", rotation=0)

    names = ["F", "Q", "H", "R", "xi", "Lambda"]


    print("True F:\n", F_sim2)
    print("Initial guess F_0:\n", F_0_sim2)
    print("MLE F:\n", F_MLE_sim2)
    # SIMULATION STUDY
    F_used = F_sim2


    d = 'True f'
    xm, Pm, _, xf, Pf = KalmanFilter(F_used, Q_sim2, H_sim2, R_sim2,
                                     z_sim2, xi_sim2, L_sim2)
    _, xs, _ = KalmanSmoother(F_used, xm, Pm, xf, Pf)
    mse_vec = np.mean((x_sim2 - xs) ** 2, axis=0)  # one MSE per state

    mse_db_vec  = 10 * np.log10(mse_vec)
    mse_db_avg  = 10 * np.log10(mse_vec.mean())
    print(f"{d:12s} smoother MSE :", mse_db_vec, "(average =", mse_db_avg, ")")



    d = 'wrong version'
    F_used = F_0_sim2
    xm, Pm, _, xf, Pf = KalmanFilter(F_used, Q_sim2, H_sim2, R_sim2,
                                     z_sim2, xi_sim2, L_sim2)
    _, xs, _ = KalmanSmoother(F_used, xm, Pm, xf, Pf)
    mse_vec = np.mean((x_sim2 - xs) ** 2, axis=0)  # one MSE per state
    mse_db_vec  = 10 * np.log10(mse_vec)
    mse_db_avg  = 10 * np.log10(mse_vec.mean())
    print(f"{d:12s} smoother MSE :", mse_db_vec, "(average =", mse_db_avg, ")")





    d = 'emkf f'
    F_used = F_MLE_sim2
    xm, Pm, _, xf, Pf = KalmanFilter(F_used, Q_sim2, H_sim2, R_sim2,
                                     z_sim2, xi_sim2, L_sim2)
    _, xs, _ = KalmanSmoother(F_used, xm, Pm, xf, Pf)
    mse_vec = np.mean((x_sim2 - xs) ** 2, axis=0)  # one MSE per state
    print('msevec',mse_vec)
    print('msevec',mse_vec.mean())
    mse_db_vec  = 10 * np.log10(mse_vec)
    mse_db_avg  = 10 * np.log10( mse_vec.mean())
    print(f"{d:12s} smoother MSE :", mse_db_vec, "(average =", mse_db_avg, ")")
    mean_x = x_sim2.mean(axis=0)
    print(mean_x)
    norm_x = np.linalg.norm(mean_x, ord=2)
    norm_db_per_t = 20 * np.log10(norm_x)
    print(norm_db_per_t)
    print(x_sim2.shape)
