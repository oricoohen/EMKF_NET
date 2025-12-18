# Non-Linear Observation Model Setup - Explained

## Overview
The file `M_network_training_3datasets_no_linear_h.py` is now correctly configured for training with:
- **Linear state transition**: `f(x) = F @ x` (where F is a 2x2 matrix)
- **Non-linear observation**: `h(x)` defined in `parameters.py`

## Key Fixes Applied

### 1. SystemModel Constructor
The `SystemModel` class requires these parameters:
```python
SystemModel(f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None)
```

**Before (WRONG):**
```python
sys_model = SystemModel(F_current, Q, h_nonlinear, R, args.T, args.T_test)
```

**After (CORRECT):**
```python
f_current = make_f(F_current)  # Create linear f(x) = F @ x
sys_model = SystemModel(f_current, Q, h_nonlinear, R, args.T, args.T_test, m, n)
sys_model.InitSequence(m1_0, m2_0)
sys_model.update_f(F_current)  # Update F matrix for later reference
```

### 2. What Each Function Does

#### `make_f(F_mat)` - Creates linear state transition
```python
def make_f(F_mat: torch.Tensor):
    def f_func(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(m, 1)
        return F_mat @ x  # Linear: f(x) = F @ x
    return f_func
```
- Takes a matrix F and returns a function `f(x) = F @ x`
- This is **linear** state evolution
- F changes between datasets (Dataset 0: [[0.83, 0.2], [0.2, 0.83]], etc.)

#### `h_nonlinear(x)` - Non-linear observation
From `parameters.py`:
```python
def h_nonlinear(x, alpha=0.3):
    x = x.view(2,1)
    x1, x2 = x[0,0], x[1,0]
    eps = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    r     = torch.sqrt(x1*x1 + x2*x2 + eps)
    theta = torch.atan2(x2, x1 + eps)
    H = torch.tensor([[1., 1.],
                      [0.25, 1.]], device=x.device, dtype=x.dtype)
    lin = (H @ x).view(2)
    return lin + alpha*torch.stack([r, theta])
```
- This is **non-linear**: combines linear term `H @ x` with polar coordinates `(r, θ)`
- The observation is: `y = H @ x + α * [r, θ]` where r and θ are functions of x

### 3. Training Flow

For each dataset (representing 30 timesteps):
1. **Generate data** with specific F matrix
   - State evolves: `x_{t+1} = F @ x_t + process_noise`
   - Observation: `y_t = h_nonlinear(x_t) + measurement_noise`

2. **Train M-network** 
   - Uses RTS smoother (with wrong F_init) to get `x_smooth`
   - Computes statistics from `x_smooth` and observations `y`
   - The key: when computing `ν = y - h(x_smooth)`, it uses `h_nonlinear`
   - M-network learns to predict ΔF to improve F estimate

3. **The `non_linear_h=True` flag**
   In `train_mstep_net_3_datasets()`, this controls how `Hx_curr` is computed:
   ```python
   if non_linear_h:
       # Apply h(x_t) for each t
       y_hat_list = []
       for t in range(T):
           x_t = x_curr[:, t].view(SysModel.m, 1)
           y_t_hat = SysModel.h(x_t)  # Calls h_nonlinear
           y_hat_list.append(y_t_hat.view(-1))
       Hx_curr = torch.stack(y_hat_list, dim=1)
   else:
       # Linear: just H @ x
       H = SysModel.H
       Hx_curr = H @ x_curr
   ```

## Summary

✅ **State transition f**: Linear (F @ x) - changes between datasets  
✅ **Observation h**: Non-linear (H @ x + α * [r, θ]) - same for all datasets  
✅ **Training**: M-network learns to estimate F from non-linear observations  
✅ **Key insight**: The non-linearity in h makes F estimation harder, which is why we need the neural network

## All SystemModel Instances Fixed

1. ✅ Data generation loop (line ~125)
2. ✅ Main training setup (line ~215)
3. ✅ Testing loop (line ~315)

All now use:
- `make_f(F)` to create linear state transition
- `h_nonlinear` for non-linear observations
- Correct parameters: `(f, Q, h, R, T, T_test, m, n)`

