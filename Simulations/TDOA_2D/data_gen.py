import torch
from Simulations.TDOA_2D.utils import generate_single_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_dataset(N, T, Q, R, m1x_0, thetas_rad):
    x_all = []
    y_all = []
    F_all = []
    block_all = []

    for k in range(N):
        x, y, F_seq, block_ids = generate_single_sequence(
            T=T,
            Q=Q,
            R=R,
            m1x_0=m1x_0,
            thetas_rad=thetas_rad
        )
        x_all.append(x)
        y_all.append(y)
        F_all.append(torch.stack(F_seq, dim=0))   # [T, m, m]
        block_all.append(block_ids)               # [T]

    x_all = torch.stack(x_all, dim=0)   # [N, m, T]
    y_all = torch.stack(y_all, dim=0)   # [N, n, T]
    F_all = torch.stack(F_all, dim=0)   # [N, T, m, m]
    block_all = torch.stack(block_all, dim=0)  # [N, T]

    return y_all, x_all, F_all, block_all

def DataGen_TDOA(args, Q, R, m1x_0, thetas_rad, data_file_name, F_file_name, block_file_name):
    train_input, train_target, F_train, block_train = generate_dataset(
        args.N_E, args.T, Q, R, m1x_0, thetas_rad
    )
    cv_input, cv_target, F_cv, block_cv = generate_dataset(
        args.N_CV, args.T, Q, R, m1x_0, thetas_rad
    )
    test_input, test_target, F_test, block_test = generate_dataset(
        args.N_T, args.T_test, Q, R, m1x_0, thetas_rad
    )

    torch.save(
        [train_input, train_target, cv_input, cv_target, test_input, test_target],
        data_file_name
    )

    torch.save([F_train, F_cv, F_test], F_file_name)
    torch.save([block_train, block_cv, block_test], block_file_name)