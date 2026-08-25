"""
3-dataset AI-mnet training for the non-linear-h / linear-F paper experiment.

Trains BOTH:
  1) the regular M-step net  -> train_mstep_net_3_datasets
  2) the joint (end-to-end) M-step net -> end_To_end_m_net_3_datasets

on 3 sequential datasets whose true F drifts by a RANDOM rotation per dataset
(randomit=True), like the current training -- so the mnet generalizes rather than
overfitting to one fixed step. The 3-dataset test
(oriiiiiiii_M_network_AI_emkf_testing_non_linear_h_paper.py) uses a fixed 0.2-rad
drift; training on random drifts of comparable magnitude covers that case.

The wrong/base F is the nominal [[0.83,0.2],[0.2,0.83]] (hardcoded inside the
3-dataset trainers), so the mnet learns to correct nominal -> true, with the SAME
sequential F/x carryover across datasets that the test uses.

Sensor h is the paper one (parameters_OLD): h(x) = H@x + 0.3*(cartesian->polar).
Noise: r2 = 10 (r_10 experiment). Models save under RTSNet/AI_M_step/exp_3/r_1/.
"""
import os
import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Extended_sysmdl import SystemModel, rotate_F
from Simulations.Lorenz_Atractor.parameters_OLD import (
    m1x_0 as m1_0, m2x_0 as m2_0, m, n, F, make_f, h_nonlinear, Q_structure, R_structure
)
from Simulations.utils import DataLoader, DataGen
import Simulations.config as config

from RTSNet.RTSNet_nn_with_F import RTSNetNN
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Baselines.BiGRU_smoother import train_bigru_smoother   # BiGRU baseline (obs -> states)

# ──────────────────────────────────────────────────────────────────────────────
# Keep the non-linear h during data generation. GenerateBatch calls
# SystemModel.update_h(H), which would rebind self.h to a linear h. Patch it to
# only record H/H_T and leave self.h = h_nonlinear intact.
# ──────────────────────────────────────────────────────────────────────────────
import Simulations.Extended_sysmdl as _esm
def _update_h_keep_nonlinear(self, H):
    self.H = H
    self.H_T = H.T
_esm.SystemModel.update_h = _update_h_keep_nonlinear

# Defensive: older RTSNet checkpoints may be pickled as RTSNet.RTSNet_nn.RTSNetNN.
# Point that name at the F-aware class so torch.load reconstructs them correctly.
import RTSNet.RTSNet_nn as _rts_nn_mod
_rts_nn_mod.RTSNetNN = RTSNetNN

# The pre-trained RTSNet checkpoints pickled self.h BY REFERENCE to
# Simulations.Lorenz_Atractor.parameters.h_nonlinear, which is now the 3-D spherical h.
# Rebind that name to our 2-D h_nonlinear BEFORE any torch.load, so loaded RTSNets use
# the matching 2-D observation (otherwise InitSequence -> self.h(x) crashes on 3-D toSpherical).
import Simulations.Lorenz_Atractor.parameters as _lor_params
_lor_params.h_nonlinear = h_nonlinear

# The nonlinear-h path calls getJacobian (imported into Pipeline_ERTS), which hardcodes
# view(-1, m=3). Replace with a dimension-agnostic version for the 2-D model.
import Pipelines.Pipeline_ERTS as _pipe_mod
def _getJacobian_nd(x, g):
    y = x.reshape(-1)
    Jac = torch.autograd.functional.jacobian(g, y)
    return Jac.reshape(-1, y.shape[0])
_pipe_mod.getJacobian = _getJacobian_nd

print("Pipeline Start")
DEVICE = torch.device("cuda")
DTYPE = torch.float32
device = DEVICE
ddtype = DTYPE
torch.backends.cudnn.benchmark = True

today = datetime.today(); now = datetime.now()
strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

# ─────────────────────────── config ───────────────────────────
args = config.general_settings()
args.N_E = 1000
args.N_CV = 100
args.N_T = 50
args.T = 30
args.T_test = 30
args.n_steps = 175
args.n_batch = 10
args.lr = 1e-4
args.wd = 1e-3

torch.manual_seed(1)

cycles = 3            # number of datasets
num_em_iters = 3
theta_rot = 1       # random F drift per dataset (randomit=True) — matches File A's
                      # generate_random_F_matrices (rotate_F(F[k], theta=0.3)). NOT 1.0:
                      # a ~1 rad cumulative drift over-trains the mnet and it overshoots
                      # the test's small 0.2 drift, causing the earlier divergence.

q2 = 0.01
r2 = 10              # r_1 experiment (match the test)
print("q2 =", q2, " r2 =", r2)

Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)
m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

F_nominal = torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE)

# ─────────── build the 3 datasets' TRUE F (VARIED F per group, like the linear file) ───────────
# Extended_sysmdl.GenerateBatch now APPLIES the F_gen list per group (index_F = i//10),
# so each sequence-group gets its OWN random F, chained cumulatively across datasets —
# same F logic as Linear_sysmdl / M_network_training_3datasets_CORRECTED_r1.py. This gives
# the mnet real F diversity instead of just 3 matrices (recorded F == actual data F).
_prev_F = [F_nominal.clone() for _ in range(args.N_E)]   # list of N_E, one per sequence
F_by_dataset = []
for k in range(cycles):
    _prev_F = rotate_F(_prev_F, i=0, j=1, theta=theta_rot, many=True, randomit=True)  # list in -> list out
    F_by_dataset.append([f.clone() for f in _prev_F])

# ─────────────────────────── generate the 3 datasets ───────────────────────────
dataFolderName = 'Simulations/Linear_canonical/paper/exp_3_datasets_r001/'
os.makedirs(dataFolderName, exist_ok=True)

all_train_inputs, all_train_targets = [], []
all_cv_inputs, all_cv_targets = [], []
all_test_inputs, all_test_targets = [], []
all_F_train, all_F_cv, all_F_test = [], [], []

x0_last = None
for k in range(cycles):
    print(f"\n=== Generating dataset {k} (random F drift) ===")
    F_k_list = F_by_dataset[k]
    sys_model_k = SystemModel(make_f(F_k_list[0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
    sys_model_k.InitSequence(m1_0, m2_0)

    dataFile = dataFolderName + f'ds{k}_data.pt'
    dataFileF = dataFolderName + f'ds{k}_F.pt'
    H_gen_k = [torch.eye(n, device=DEVICE, dtype=DTYPE) for _ in range(args.N_E)]

    DataGen(args, sys_model_k, dataFile, dataFileF, delta=1,
            randomInit_train=False, randomInit_cv=False, randomInit_test=False,
            randomLength=False, Test=False, F_gen=F_k_list, H_gen=H_gen_k, x0_list=x0_last)

    [tr_in, tr_tg, cv_in, cv_tg, te_in, te_tg] = torch.load(dataFile, weights_only=True, map_location=DEVICE)
    [F_tr, F_cv, F_te] = torch.load(dataFileF, map_location=DEVICE)

    # align dtype/device
    tr_in = tr_in.to(DEVICE, dtype=DTYPE); tr_tg = tr_tg.to(DEVICE, dtype=DTYPE)
    cv_in = cv_in.to(DEVICE, dtype=DTYPE); cv_tg = cv_tg.to(DEVICE, dtype=DTYPE)
    te_in = te_in.to(DEVICE, dtype=DTYPE); te_tg = te_tg.to(DEVICE, dtype=DTYPE)

    all_train_inputs.append(tr_in); all_train_targets.append(tr_tg)
    all_cv_inputs.append(cv_in);    all_cv_targets.append(cv_tg)
    all_test_inputs.append(te_in);  all_test_targets.append(te_tg)
    all_F_train.append(F_tr); all_F_cv.append(F_cv); all_F_test.append(F_te)

    # carry the true last state forward so datasets chain continuously (like the test)
    X0_tr = tr_tg[:, :, -1].clone(); X0_cv = cv_tg[:, :, -1].clone(); X0_te = te_tg[:, :, -1].clone()
    x0_last = [
        [X0_tr[j].unsqueeze(-1).clone() for j in range(X0_tr.size(0))],
        [X0_cv[j].unsqueeze(-1).clone() for j in range(X0_cv.size(0))],
        [X0_te[j].unsqueeze(-1).clone() for j in range(X0_te.size(0))],
    ]
    print(f"  train {tr_in.shape}  cv {cv_in.shape}  test {te_in.shape}")

assert len(all_train_inputs) == cycles
print("\nData generation complete.")

# ─────────────────────────── system model + pipeline ───────────────────────────
sys_model = SystemModel(make_f(F_by_dataset[0][0]), Q, h_nonlinear, R, args.T, args.T_test, m, n)
sys_model.InitSequence(m1_0, m2_0)

# 3-dataset F structure (lists of `cycles` F-lists). The trainers use *_TRUE.
sys_model.F_train = all_F_train
sys_model.F_valid = all_F_cv
sys_model.F_test = all_F_test
sys_model.F_train_TRUE = all_F_train
sys_model.F_valid_TRUE = all_F_cv
sys_model.F_test_TRUE = all_F_test

RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

# ─────────────────────────── paths (r_001) ───────────────────────────
path_results_False = 'RTSNet/AI_M_step/exp_3/r_10/False_F/'
path_results_wrong_rts = path_results_False + 'best-rts_false.pt'   # frozen wrong-F RTSNet (r_001)

destination_folder = 'RTSNet/AI_M_step/exp_3/r_10/EMKF/False/'
os.makedirs(destination_folder, exist_ok=True)

# NEW names -> nothing existing is overwritten
destination_path_M = destination_folder + 'mnet_r001_3ds.pt'
destination_path_M_laod = None   # optional warm-start checkpoint for the regular mnet (None = fresh)

# joint mnet + joint RTSNet (ONE of each)
destination_path_M_joint   = destination_folder + 'joint_mnet_1m1r_r10_new.pt'
destination_path_RTS_joint = destination_folder + 'joint_rtsnet_1m1r_r10_new.pt'

# ══════════════════════════════════════════════════════════════════════════════
# 3) BiGRU baseline (supervised smoother, observations -> states; no EMKF, no F).
#    Trains on all 3 datasets concatenated (each a 30-step obs->state mapping).
# ══════════════════════════════════════════════════════════════════════════════
# bgru_path = destination_folder + 'bigru_r10_3ds_new.pt'
# print("\n===== BiGRU baseline training (3 datasets) =====")
# train_bigru_smoother(all_train_inputs, all_train_targets, all_cv_inputs, all_cv_targets,
#                      n=n, m=m, save_path=bgru_path, device=device,
#                      epochs=300, batch_size=32, lr=1e-3, hidden_size=128, num_layers=2)
# print("BiGRU saved to:", bgru_path)

# ══════════════════a════════════════════════════════════════════════════════════
# 1) REGULAR M-step net on 3 datasets (RTSNet frozen)
# ══════════════════════════════════════════════════════════════════════════════
print("\n===== REGULAR M-step training (3 datasets, RTSNet frozen) =====")
RTSNet_Pipeline.train_mstep_net_3_datasets(
    SysModel=sys_model,
    cv_input=all_cv_inputs, cv_target=all_cv_targets,
    train_input=all_train_inputs, train_target=all_train_targets,
    destination_path_M=destination_path_M,
    destination_path_RTS=path_results_wrong_rts,
    num_em_iters=num_em_iters, alpha=(0.3, 0.1, 0.85), lambda_F=1e-3,
    generate_f=True, non_linear_h=True, load=destination_path_M_laod, datasets=cycles,
)
print("Regular mnet saved to:", destination_path_M)

# ══════════════════════════════════════════════════════════════════════════════
# 2) JOINT M-step net on 3 datasets: ONE mnet + ONE RTSNet trained TOGETHER,
#    unrolled per dataset as  RTSNet -> mnet -> RTSNet -> mnet -> RTSNet.
#    Warm-started: mnet from the regular mnet above, RTSNet from the wrong-F RTSNet.
# ══════════════════════════════════════════════════════════════════════════════
print("\n===== JOINT M-step training (3 datasets, 1 mnet + 1 RTSNet) =====")
RTSNet_Pipeline.setTrainingParams(args)   # refresh optimizers / default M_model
RTSNet_Pipeline.joint_train_mnet_rtsnet_3_datasets(
    SysModel=sys_model,
    cv_input=all_cv_inputs, cv_target=all_cv_targets,
    train_input=all_train_inputs, train_target=all_train_targets,
    destination_path_M=destination_path_M_joint,
    destination_path_RTS=destination_path_RTS_joint,
    load_rts=path_results_wrong_rts,     # base RTSNet (wrong F)
    load_m=destination_path_M,           # warm-start the joint mnet from the regular mnet
    num_em_iters=num_em_iters, alpha=(0.3, 0.1, 0.85), lambda_F=1e-3,
    generate_f=True, non_linear_h=True, datasets=cycles,
)
print("Joint mnet saved to:", destination_path_M_joint)
print("Joint RTS saved to:", destination_path_RTS_joint)



print("\nALL DONE.")
