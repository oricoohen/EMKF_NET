"""
NON-GAUSSIAN training experiment (contaminated-Gaussian measurement noise).

Add-on copy of main_lor_DT_3_datasets_train_batched.py. The ONLY differences:
  1. The data-generation SystemModel gets a non-Gaussian measurement-noise law
     via sys_model.set_noise_model(...)  (the filters/nets still assume Gaussian
     R -> deliberate mismatch that breaks classical EMKF & BiGRU).
  2. All FOUR learned methods are actually trained here (BiGRU, TRUE RTSNet,
     FALSE RTSNet, joint RTSNet+M-net = EMKalmanNet).
  3. Distinct save paths under RTSNet/lorenz_nongauss_contam/ so nothing existing
     is overwritten.

Switch the noise by editing NOISE_TYPE / NOISE_PARAMS below. r2/q2 stay the
variance knobs exactly as before.
"""
import torch
from datetime import datetime
from Simulations.utils import DataLoader, DataGen, estimate_QR

import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
    f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure, H_design
from Simulations.Linear_sysmdl import rotate_F, rotate_H

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN
from Baselines.BiGRU_smoother import train_bigru_smoother

# batched pipeline (fast)
from Pipelines.Pipeline_ERTS_batched import Pipeline_ERTS_batched as Pipeline

import shutil
import os
print("Pipeline Start - NON-GAUSSIAN EMKF H Estimation (BATCHED)")

DEVICE = torch.device("cuda")
DTYPE = torch.float32
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m1x_0 = m1x_0.to(device)
m2x_0 = m2x_0.to(device)
H_Rotate = H_Rotate.to(device)
H_Rotate_inv = H_Rotate_inv.to(device)
Q_structure = Q_structure.to(device)
R_structure = R_structure.to(device)
H_design = H_design.to(device)

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
print("Current Time =", strTime)

# ============================================================================ #
#  NOISE MODEL  (edit these two lines to switch the non-Gaussian law)           #
#    'contaminated' : impulsive outliers, params {'eps':..,'kappa':..}          #
#    'student_t'    : heavy tails,  params {'nu':..}                            #
#    'exponential'  : skewed,       params {}                                   #
#    'laplace'      : mild tails,   params {}                                   #
#    'gaussian'     : plain Gaussian (back to the original experiment)          #
# ============================================================================ #
# Pick ONE noise law here. Params come from NOISE_PRESETS below (edit to taste).
#   'gaussian'     : plain Gaussian (original experiment)
#   'contaminated' : impulsive outliers. eps=freq, kappa=size. var=(1+eps(kappa^2-1))*r2
#   'student_t'    : heavy tails, SAME covariance as Gaussian. nu=dof (smaller=heavier)
#   'laplace'      : mild heavy tails, same covariance
#   'exponential'  : skewed (KalmanNet-lineage), same covariance
NOISE_TYPE = 'gaussian'
NOISE_PRESETS = {
    'gaussian':     {},
    'contaminated': {'eps': 0.05, 'kappa': 3.0},   # ~1.4x var (easy); 0.05/5->2.2x; 0.1/10->11x (harsh)
    'student_t':    {'nu': 4.0},                    # nu=3 heavier, nu=8 milder
    'laplace':      {},
    'exponential':  {},
}
NOISE_PARAMS = NOISE_PRESETS[NOISE_TYPE if NOISE_TYPE else 'gaussian']

# ---- RAW (un-normalized) noise -- specific to this new_* experiment ---------- #
# NORMALIZE_NOISE = False -> the non-Gaussian measurement noise is used AS IS:
#   NOT debiased and NOT variance-normalized. Params are chosen so the variance is
#   naturally ~1 (e.g. exponential -> Exp(1): mean 1, var 1, so it is skewed AND
#   biased). Every OTHER experiment (e.g. main_lor_DT_3datasets_nongauss_train.py)
#   leaves this at the default True and keeps the standardized noise.
NORMALIZE_NOISE = False

# ============================================================================ #
#  DATA REUSE                                                                   #
#  LOAD_DATA = False -> generate the training data and SAVE it to a dedicated   #
#                       bundle (TRAIN_DATA_PATH below).                          #
#  LOAD_DATA = True  -> skip generation, just load that saved bundle and train. #
# ============================================================================ #
LOAD_DATA = False

# ---- training H drift (per dataset) ----------------------------------------- #
# H_DRIFT_SAME_ANGLE = True  -> one random angle in [0, H_DRIFT_THETA] applied
#                               equally to all 3 axes (x=y=z).
# H_DRIFT_SAME_ANGLE = False -> original: independent random angle per axis.
H_DRIFT_THETA = 0.3
H_DRIFT_SAME_ANGLE = True

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

###################
###  Settings   ###
###################
args = config.general_settings()
args.N_E = 1000
args.N_CV = 150
args.N_T = 100
args.T = 30
args.T_test = 30
args.n_steps = 400
args.n_batch = 15
args.lr = 1e-4        # was 1e-3: RTSNet recursion diverged (NaN) with the aggressive step
args.wd = 1e-3

torch.manual_seed(1)

# Train AND test on 5 datasets. With 5 cycles the cross-dataset x0 chaining at
# random init is what produced the d2/d3 NaN blow-up; the low lr (1e-4) tames it.
# If NaNs still appear, ease the difficulty (H drift 0.3->0.2, noise kappa 3->2),
# NOT the cycle count.
cycles = 5
num_em_iters = 2
# noise q and r  (variance knobs -- unchanged semantics)
r2 = torch.tensor([1], device=device)  # [100, 10, 1, 0.1, 0.01]
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
Q = q2[0] * Q_structure
R = r2[0] * R_structure
print('q2 is:', q2)
print('r2 is:', r2)

# ============================================================================ #
#  CORRELATE_NOISE -- non-diagonal (correlated) measurement-noise covariance     #
#  Standalone flag, INDEPENDENT of NORMALIZE_NOISE. When True, R becomes the      #
#  equicorrelated matrix R = r2 * C, C = (1-rho) I + rho J (unit diagonal, every  #
#  channel PAIR correlated by rho -- "common-mode" noise). Used CONSISTENTLY by   #
#  the generator AND all nets (no mismatch). MUST MATCH THE TEST SCRIPT.          #
#  Intended for GAUSSIAN noise: MultivariateNormal(cov=R) applies the correlation #
#  directly, regardless of NORMALIZE_NOISE.                                       #
# ============================================================================ #
CORRELATE_NOISE = True
R_CORR_RHO = 0.6
if CORRELATE_NOISE:
    C_corr = ((1.0 - R_CORR_RHO) * torch.eye(n, device=device)
              + R_CORR_RHO * torch.ones(n, n, device=device))
    R = r2[0] * C_corr
    print(f"[CORRELATE_NOISE] equicorrelated R, rho={R_CORR_RHO} "
          f"(generator AND all nets share this true R)")

sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
sys_model.InitSequence(m1x_0, m2x_0)

print("\n" + "=" * 80)
print(f"GENERATING {cycles} DATASETS (DIFFERENT H, FIXED F) WITH "
      f"{NOISE_TYPE.upper()} MEASUREMENT NOISE")
print("=" * 80)

# ---- distinct non-Gaussian save paths (nothing existing is touched) ----
# correlated-R runs get their own subtree so plain-{NOISE_TYPE} nets are untouched
_corr_tag = f'_corr{str(R_CORR_RHO).replace(".", "p")}' if CORRELATE_NOISE else ''
_base = f'RTSNet/lorenz_nongauss_{NOISE_TYPE}{_corr_tag}/5datasets/30'
destination_path_rtsnet_full         = f'{_base}/NEW_RTSNet_full.pt'
destination_path_rtsnet_partial      = f'{_base}/NEW_RTSNet_partial.pt'
destination_path_M_joint             = f'{_base}/NEW_M_step_net_joint.pt'
destination_path_rtsnet_partial_joint = f'{_base}/NEW_RTSNet_partial_joint.pt'
destination_path_bigru               = f'{_base}/NEW_bigru_smoother.pt'

# ---- WARM START ------------------------------------------------------------- #
# Warm-start chain (each net starts from an already-stable net, not from scratch):
#   TRUE  RTSNet  <- Gaussian RTSNet_full        (only if STEP 1a is run)
#   FALSE RTSNet  <- the TRUE RTSNet we train     (destination_path_rtsnet_full)
#   joint (EMKalmanNet) RTSNet <- the FALSE RTSNet we train
#                       M-net   <- Gaussian M-net (or fresh if not found)
# WARM_START = False -> original from-scratch behavior.
WARM_START = True
_gauss = 'RTSNet/lorenz_gauss/lorenz_rotated_1/5datasets/final'   # r=1 Gaussian refs
warm_rtsnet_full = f'{_gauss}/RTSNet_full.pt'       # TRUE RTSNet init (STEP 1a) <- r=1
warm_mnet_joint  = f'{_gauss}/M_step_net_joint.pt'  # M-net init (STEP 2)        <- r=1
# warm_rtsnet_partial = f'{_gauss}/RTSNet_partial.pt'
# (previous r=10 refs under _base:)
# warm_rtsnet_full = f'{_base}/RTSNet_full.pt'
# warm_mnet_joint  = f'{_base}/M_step_net_joint.pt'
def _warm(path):
    """Return path for warm start if it exists, else None (-> train from scratch)."""
    if WARM_START and path and os.path.exists(path):
        return path
    if WARM_START and path:
        print(f"[warm start] '{path}' not found -> training this net from scratch.")
    return None

# ---- dedicated place where the generated TRAINING data bundle is stored ----
r2_val = int(r2[0].item())
_noise_tag = (NOISE_TYPE if NOISE_TYPE else 'gaussian')
DATA_DIR = f'Data/train_nongauss_{_noise_tag}'
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_DATA_PATH = f'{DATA_DIR}/bundle_r2{r2_val}_T{args.T}_c{cycles}.pt'

# Storage for all datasets (train, cv, test)
all_train_inputs, all_train_targets = [], []
all_cv_inputs, all_cv_targets = [], []
all_test_inputs, all_test_targets = [], []
all_H_matrices_train, all_H_matrices_cv, all_H_matrices_test = [], [], []
all_H_matrices_train_rotate, all_H_matrices_cv_rotate, all_H_matrices_test_rotate = [], [], []

if LOAD_DATA and os.path.exists(TRAIN_DATA_PATH):
    # ---------------- LOAD pre-generated data (no regeneration) ----------------
    print(f"\n[load_data] Loading pre-generated training data from {TRAIN_DATA_PATH}")
    bundle = torch.load(TRAIN_DATA_PATH, map_location=DEVICE, weights_only=False)
    all_train_inputs  = bundle['train_inputs']
    all_train_targets = bundle['train_targets']
    all_cv_inputs     = bundle['cv_inputs']
    all_cv_targets    = bundle['cv_targets']
    all_test_inputs   = bundle['test_inputs']
    all_test_targets  = bundle['test_targets']
    all_H_matrices_train = bundle['H_train']
    all_H_matrices_cv    = bundle['H_cv']
    all_H_matrices_test  = bundle['H_test']
    print(f"[load_data] Loaded {len(all_train_inputs)} datasets "
          f"(noise='{bundle.get('noise_type')}' {bundle.get('noise_params')}).")
else:
    if LOAD_DATA:
        print(f"\n[load_data] {TRAIN_DATA_PATH} not found -> generating fresh data.")
    x0_last = None
    H_init = None

    for dataset_id in range(cycles):
        print(f"\n{'=' * 80}\nGenerating Dataset {dataset_id}\n{'=' * 80}")

        # Create system model with FIXED F and this H
        sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
        sys_model.InitSequence(m1x_0, m2x_0)
        # >>> the ONLY new line vs the Gaussian experiment <<<
        sys_model.set_noise_model(meas_type=NOISE_TYPE, meas_params=NOISE_PARAMS,
                                  normalize=NORMALIZE_NOISE)
        # H drift per dataset: one random angle <= H_DRIFT_THETA on all 3 axes
        sys_model.set_H_gen(theta=H_DRIFT_THETA, same_angle=H_DRIFT_SAME_ANGLE)

        dataFolderName = 'Simulations/Linear_canonical/paper/exp_3_datasets_H_nongauss/'
        dataFileName = f'dataset_{dataset_id}_data.pt'
        dataFileName_H = f'dataset_{dataset_id}_H.pt'
        os.makedirs(dataFolderName, exist_ok=True)

        print(f"\nGenerating data for dataset {dataset_id}...")
        DataGen(args, sys_model,
                dataFolderName + dataFileName,
                dataFolderName + dataFileName_H,
                fileName_H=dataFolderName + dataFileName_H,
                delta=1,
                randomInit_train=InitIsRandom_train,
                randomInit_cv=InitIsRandom_cv,
                randomInit_test=InitIsRandom_test,
                randomLength=LengthIsRandom,
                Test=False,
                F_gen=False,   # F is FIXED across datasets
                H_gen=True,
                x0_list=x0_last,
                H_init=H_init)

        [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
            dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
        [H_train_mat, H_val_mat, H_test_mat_list] = torch.load(
            dataFolderName + dataFileName_H, map_location=DEVICE)

        H_init = [H_train_mat, H_val_mat, H_test_mat_list]  # For next dataset

        # x0 continuity for next dataset
        X_0_train = train_target[:, :, -1].clone()
        X_0_cv = cv_target[:, :, -1].clone()
        X_0_test = test_target[:, :, -1].clone()
        x0_last = [None, None, None]
        x0_last[0] = [X_0_train[j].unsqueeze(-1).clone() for j in range(X_0_train.size(0))]
        x0_last[1] = [X_0_cv[j].unsqueeze(-1).clone() for j in range(X_0_cv.size(0))]
        x0_last[2] = [X_0_test[j].unsqueeze(-1).clone() for j in range(X_0_test.size(0))]

        print(f"Dataset {dataset_id}: train {train_input.shape}, cv {cv_input.shape}, test {test_input.shape}")

        all_train_inputs.append(train_input)
        all_train_targets.append(train_target)
        all_cv_inputs.append(cv_input)
        all_cv_targets.append(cv_target)
        all_test_inputs.append(test_input)
        all_test_targets.append(test_target)
        all_H_matrices_train.append(H_train_mat)
        all_H_matrices_cv.append(H_val_mat)
        all_H_matrices_test.append(H_test_mat_list)

    # ---- save the generated data to its dedicated place for later reuse ----
    torch.save({
        'train_inputs': all_train_inputs, 'train_targets': all_train_targets,
        'cv_inputs': all_cv_inputs, 'cv_targets': all_cv_targets,
        'test_inputs': all_test_inputs, 'test_targets': all_test_targets,
        'H_train': all_H_matrices_train, 'H_cv': all_H_matrices_cv,
        'H_test': all_H_matrices_test,
        'noise_type': NOISE_TYPE, 'noise_params': NOISE_PARAMS,
        'r2': r2_val, 'T': args.T, 'cycles': cycles,
    }, TRAIN_DATA_PATH)
    print(f"\n[load_data] Saved generated training data -> {TRAIN_DATA_PATH}")

print("\n" + "=" * 80)
print("DATA READY")
print("=" * 80)

assert len(all_train_inputs) == cycles
assert all_train_inputs[0].shape[0] == args.N_E
assert all_cv_inputs[0].shape[0] == args.N_CV
assert all_test_inputs[0].shape[0] == args.N_T
print("Data structure verified.")

# ---- system models for training ----
sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
sys_model.InitSequence(m1x_0, m2x_0)
sys_model.H_train = all_H_matrices_train_rotate
sys_model.H_valid = all_H_matrices_cv_rotate
sys_model.H_test = all_H_matrices_test_rotate
sys_model.H_train_TRUE = all_H_matrices_train
sys_model.H_valid_TRUE = all_H_matrices_cv
sys_model.H_test_TRUE = all_H_matrices_test

# TRUE system model: sees each sequence's correct H (oracle)
sys_model_true = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, H_Rotate)
sys_model_true.InitSequence(m1x_0, m2x_0)
sys_model_true.H_train = all_H_matrices_train
sys_model_true.H_valid = all_H_matrices_cv
sys_model_true.H_test = all_H_matrices_test
sys_model_true.H_train_TRUE = all_H_matrices_train
sys_model_true.H_valid_TRUE = all_H_matrices_cv
sys_model_true.H_test_TRUE = all_H_matrices_test

# Pipeline
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)
print("Pipeline configured (BATCHED)")

for _p in [destination_path_rtsnet_full, destination_path_rtsnet_partial,
           destination_path_rtsnet_partial_joint, destination_path_M_joint,
           destination_path_bigru]:
    os.makedirs(os.path.dirname(_p), exist_ok=True)

H_init_common = H_Rotate  # the single WRONG H every sequence starts from


def _fresh_rtsnet(ssm):
    net = RTSNetNN()
    net.NNBuild(ssm, args)
    return net.to(device)


# ============================================================================ #
# BASELINE - BiGRU smoother (model-free, MSE-trained).                          #
# ============================================================================ #
print("\n[BASELINE] Training BiGRU smoother...")
train_bigru_smoother(
    train_input=all_train_inputs,
    train_target=all_train_targets,
    cv_input=all_cv_inputs,
    cv_target=all_cv_targets,
    n=n, m=m,
    save_path=destination_path_bigru,
    device=device,
    epochs=300, batch_size=32, lr=1e-3, hidden_size=128, num_layers=2,
)

# ============================================================================ #
# STEP 1a - TRUE RTSNet (per-sequence TRUE H, oracle).                          #
# ============================================================================ #
print("\n[STEP 1a] Training TRUE RTSNet (true H per sequence)...")
RTSNet_Pipeline.model = _fresh_rtsnet(sys_model_true)
RTSNet_Pipeline.train_RTS_net_3_datasets_batched(
    sys_model_true,
    all_cv_inputs, all_cv_targets,
    all_train_inputs, all_train_targets,
    destination_path_RTS=destination_path_rtsnet_full,
    load_path_RTS=_warm(warm_rtsnet_full),   # warm-start from the Gaussian TRUE RTSNet
    H_init=None,             # per-sequence TRUE H
    datasets=cycles)

# ============================================================================ #
# STEP 1b - FALSE RTSNet (single WRONG H_init = H_Rotate for every sequence).   #
# ============================================================================ #
print("\n[STEP 1b] Training FALSE RTSNet (wrong H_init=H_Rotate)...")
RTSNet_Pipeline.model = _fresh_rtsnet(sys_model)
RTSNet_Pipeline.train_RTS_net_3_datasets_batched(
    sys_model,
    all_cv_inputs, all_cv_targets,
    all_train_inputs, all_train_targets,
    destination_path_RTS=destination_path_rtsnet_partial,
    load_path_RTS=_warm(destination_path_rtsnet_full),   # warm-start from the TRUE RTSNet we trained
    H_init=H_init_common,    # SAME wrong H for every sequence
    datasets=cycles)

# Ensure STEP 2 has an M-net to load: use the Gaussian M-net if it exists,
# otherwise save the fresh (untrained) M-net and load that.
if _warm(warm_mnet_joint) is not None:
    mnet_to_load = warm_mnet_joint
else:
    torch.save(RTSNet_Pipeline.M_model_H, destination_path_M_joint)
    mnet_to_load = destination_path_M_joint

# ============================================================================ #
# STEP 2 - joint RTSNet + M-net (EMKalmanNet). Warm-starts from FALSE RTSNet.   #
# ============================================================================ #
print("\n[STEP 2] Joint RTSNet + M-net training (EMKalmanNet)...")
RTSNet_Pipeline.train_H_mstep_net_3_datasets_joint_batched(
    SysModel=sys_model,
    cv_input=all_cv_inputs,
    cv_target=all_cv_targets,
    train_input=all_train_inputs,
    train_target=all_train_targets,
    destination_path_M=destination_path_M_joint,
    destination_path_RTS=destination_path_rtsnet_partial_joint,
    load_path_RTS=destination_path_rtsnet_partial,   # warm-start from the FALSE RTSNet we trained
    load_mnet=mnet_to_load,
    num_em_iters=num_em_iters,
    alpha=(0.5, 1., 0.85),
    lambda_H=1e-3,
    generate_h=True,
    H_init=H_init_common,
    datasets=cycles)

print("\n" + "=" * 80)
print(f"TRAINING COMPLETE (noise = {NOISE_TYPE} {NOISE_PARAMS})")
print("=" * 80)
print(f"BiGRU   -> {destination_path_bigru}")
print(f"TRUE    -> {destination_path_rtsnet_full}")
print(f"FALSE   -> {destination_path_rtsnet_partial}")
print(f"Joint   -> {destination_path_rtsnet_partial_joint}")
print(f"M-net   -> {destination_path_M_joint}")
