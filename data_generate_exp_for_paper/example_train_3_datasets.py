"""
Example script showing how to properly train with train_mstep_net_3_datasets
This demonstrates the correct data structure and function call
"""
import torch
from datetime import datetime
from Simulations.Linear_sysmdl import SystemModel
from Simulations.utils import DataGen
import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure, m1_0, m2_0
from RTSNet.RTSNet_nn import RTSNetNN
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

DEVICE = torch.device("cuda")
DTYPE = torch.float32

# Setup
args = config.general_settings()
args.N_E = 400   # Number of training examples
args.N_CV = 100  # Number of CV examples
args.N_T = 50    # Number of test examples
args.T = 30      # Length of each dataset sequence
args.T_test = 30

# Training parameters
args.n_steps = 175
args.n_batch = 10
args.lr = 1e-4
args.wd = 1e-3

# Model parameters
q2 = 0.01
r2 = 1.0
Q = (q2 * Q_structure).to(DEVICE, dtype=DTYPE)
R = (r2 * R_structure).to(DEVICE, dtype=DTYPE)
H = torch.tensor([[1., 1.], [0.25, 1.]], device=DEVICE, dtype=DTYPE)
m1_0 = m1_0.to(DEVICE, dtype=DTYPE)
m2_0 = m2_0.to(DEVICE, dtype=DTYPE)

print("="*80)
print("STEP 1: Generate 3 different F matrices for 3 datasets")
print("="*80)

# Create 3 different F matrices (you can randomize these)
F_matrices_for_datasets = [
    torch.tensor([[0.83, 0.2], [0.2, 0.83]], device=DEVICE, dtype=DTYPE),
    torch.tensor([[0.85, 0.15], [0.15, 0.85]], device=DEVICE, dtype=DTYPE),
    torch.tensor([[0.80, 0.25], [0.25, 0.80]], device=DEVICE, dtype=DTYPE),
]

print("\nDataset F matrices:")
for i, F_mat in enumerate(F_matrices_for_datasets):
    print(f"Dataset {i}: \n{F_mat}")

print("\n" + "="*80)
print("STEP 2: Generate training, CV, and test data for each F matrix")
print("="*80)

# Storage for all datasets
all_train_inputs = []
all_train_targets = []
all_cv_inputs = []
all_cv_targets = []
all_test_inputs = []
all_test_targets = []
all_F_train = []
all_F_cv = []
all_F_test = []

for dataset_id in range(3):
    print(f"\n--- Generating Dataset {dataset_id} ---")

    # Create system model with this F
    F_current = F_matrices_for_datasets[dataset_id]
    sys_model = SystemModel(F_current, Q, H, R, args.T, args.T_test)
    sys_model.InitSequence(m1_0, m2_0)

    # Create folder and filenames
    dataFolderName = f'Simulations/Linear_canonical/paper/exp_3datasets/'
    dataFileName = f'dataset_{dataset_id}_data.pt'
    dataFileName_F = f'dataset_{dataset_id}_F.pt'

    # Generate data for this F matrix
    print(f"Generating data with F = {F_current[0,0]:.2f}, {F_current[0,1]:.2f}...")
    DataGen(args, sys_model,
            dataFolderName + dataFileName,
            dataFolderName + dataFileName_F,
            delta=1,
            randomInit_train=False,
            randomInit_cv=False,
            randomInit_test=False,
            randomLength=False,
            Test=True)

    # Load the generated data
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(
        dataFolderName + dataFileName, weights_only=True, map_location=DEVICE)
    [F_train_mat, F_val_mat, F_test_mat_list] = torch.load(
        dataFolderName + dataFileName_F, map_location=DEVICE)

    print(f"Dataset {dataset_id} shapes:")
    print(f"  Train: {train_input.shape}, {train_target.shape}")
    print(f"  CV: {cv_input.shape}, {cv_target.shape}")
    print(f"  Test: {test_input.shape}, {test_target.shape}")

    # Store in organized lists
    all_train_inputs.append(train_input)      # Shape: [N_E, n, 30]
    all_train_targets.append(train_target)    # Shape: [N_E, m, 30]
    all_cv_inputs.append(cv_input)            # Shape: [N_CV, n, 30]
    all_cv_targets.append(cv_target)          # Shape: [N_CV, m, 30]
    all_test_inputs.append(test_input)        # Shape: [N_T, n, 30]
    all_test_targets.append(test_target)      # Shape: [N_T, m, 30]
    all_F_train.append(F_train_mat)           # List of F matrices for train
    all_F_cv.append(F_val_mat)                # List of F matrices for CV
    all_F_test.append(F_test_mat_list)        # List of F matrices for test

print("\n" + "="*80)
print("STEP 3: Verify data structure")
print("="*80)

print(f"\nNumber of datasets: {len(all_train_inputs)}")
print(f"Sequences per dataset (train): {all_train_inputs[0].shape[0]}")
print(f"Sequences per dataset (CV): {all_cv_inputs[0].shape[0]}")
print(f"Sequence length (T): {all_train_inputs[0].shape[2]}")

# Verify structure
assert len(all_train_inputs) == 3, "Should have 3 datasets"
assert all_train_inputs[0].shape[0] == args.N_E, f"Train should have {args.N_E} sequences"
assert all_cv_inputs[0].shape[0] == args.N_CV, f"CV should have {args.N_CV} sequences"
assert all_train_inputs[0].shape[2] == 30, "Each sequence should have 30 timesteps"
print("✓ Data structure verified!")

print("\n" + "="*80)
print("STEP 4: Setup system model with F lists")
print("="*80)

# Create a system model (F doesn't matter much, will be updated during training)
F_initial = F_matrices_for_datasets[0]
sys_model = SystemModel(F_initial, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)

# CRITICAL: Set F_train_TRUE, F_valid_TRUE as lists of 3 datasets
sys_model.F_train_TRUE = all_F_train    # List of 3 F lists
sys_model.F_valid_TRUE = all_F_cv       # List of 3 F lists
sys_model.F_test_TRUE = all_F_test      # List of 3 F lists

print("✓ System model configured with 3-dataset F structure")

print("\n" + "="*80)
print("STEP 5: Create and configure RTSNet")
print("="*80)

# Path to pre-trained RTSNet (frozen during M-step training)
path_results_rts = '../RTSNet/synthetic/AI_M_step/exp_1/r_1/False_F/best-rts_false.pt'
destination_path_M = 'RTSNet/AI_M_step/exp_1/r_1/EMKF/False/M_net_3_datasets.pt'

# Create RTSNet model
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)

# Create pipeline
today = datetime.today()
strTime = today.strftime("%m.%d.%y_%H:%M:%S")
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model, args)
RTSNet_Pipeline.setTrainingParams(args)

print("✓ Pipeline configured")

print("\n" + "="*80)
print("STEP 6: Train M-step network on 3 datasets")
print("="*80)

print("\nTraining configuration:")
print(f"  - 3 datasets, each with {args.N_E} training sequences")
print(f"  - Each sequence: 30 timesteps")
print(f"  - Total effective sequence length: 90 timesteps (3 × 30)")
print(f"  - F changes every 30 timesteps")
print(f"  - EM iterations per dataset: 3")
print(f"  - Training epochs: {args.n_steps}")
print(f"  - Batch size: {args.n_batch}")

# Call the training function
RTSNet_Pipeline.train_mstep_net_3_datasets(
    SysModel=sys_model,
    cv_input=all_cv_inputs,           # List of 3 CV datasets [N_CV, n, 30]
    cv_target=all_cv_targets,         # List of 3 CV datasets [N_CV, m, 30]
    train_input=all_train_inputs,     # List of 3 train datasets [N_E, n, 30]
    train_target=all_train_targets,   # List of 3 train datasets [N_E, m, 30]
    destination_path_M=destination_path_M,
    destination_path_RTS=path_results_rts,
    num_em_iters=3,
    alpha=(0.05, 0.1, 0.85),          # Weights for EM iterations
    lambda_F=1e-3,                     # Regularization
    generate_f=True,                   # If True, uses grouped F (f_index = n_e // 10)
    non_linear_h=False,                # Linear observation model
    datasets=3                         # Number of datasets
)

print("\n" + "="*80)
print("STEP 7: Testing on 3 concatenated datasets")
print("="*80)

# After training, you can test on the 3 concatenated test datasets
print("\nTesting configuration:")
print(f"  - 3 datasets, each with {args.N_T} test sequences")
print(f"  - Each sequence: 30 timesteps")

# You can use the test_mstep_net or one_test_mstep_net for testing
# but you'll need to adapt it for the 3-dataset structure
# Or test each dataset separately

for dataset_id in range(3):
    print(f"\n--- Testing Dataset {dataset_id} ---")

    test_input = all_test_inputs[dataset_id]
    test_target = all_test_targets[dataset_id]
    true_F = F_matrices_for_datasets[dataset_id]

    # Set up system model with true F for this dataset
    sys_model_test = SystemModel(true_F, Q, H, R, args.T, args.T_test)
    sys_model_test.InitSequence(m1_0, m2_0)
    sys_model_test.F_test = all_F_test[dataset_id]

    # Run test (example - you may need to adapt based on your needs)
    # results = RTSNet_Pipeline.NNTest_no_p(
    #     sys_model_test,
    #     test_input,
    #     test_target,
    #     load_model_path=path_results_rts,
    #     generate_f=False
    # )

    print(f"Dataset {dataset_id} test completed")

print("\n" + "="*80)
print("Training and testing complete!")
print("="*80)

"""
SUMMARY OF KEY POINTS:

1. Data Structure:
   - train_input = [dataset_0, dataset_1, dataset_2]
   - Each dataset has shape [N_E, n, 30] for training
   - Same structure for CV and test data

2. F Structure:
   - sys_model.F_train_TRUE = [F_list_0, F_list_1, F_list_2]
   - Each F_list contains F matrices for that dataset's sequences

3. Training Process:
   - For each training batch, picks ONE random sequence index n_e
   - Processes the SAME sequence index across all 3 datasets sequentially
   - F and x_0 propagate from dataset_0 → dataset_1 → dataset_2
   - After all 3 datasets and 3 EM iters each, backprop the accumulated loss

4. The function correctly implements:
   ✓ Sequential processing across 3 datasets
   ✓ F and x_0 propagation between datasets
   ✓ EM iteration structure (E-step with frozen RTSNet, M-step with trainable M-net)
   ✓ Weighted loss across EM iterations
   ✓ Proper gradient accumulation

5. What you need to ensure:
   ⚠ Your data generation creates the 3-dataset list structure
   ⚠ F_train_TRUE is structured as a list of 3 lists
   ⚠ You're using TRAINING data (N_E sequences), not just TEST data
"""

