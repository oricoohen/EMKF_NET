# ✅ SUMMARY - Normalization Issue Resolved

## What Happened

**You tried normalization at test time** → Performance got worse ❌

**Why:** Your network was **trained on non-normalized inputs**, so normalizing at test creates a distribution mismatch.

---

## Current Status

✅ **Normalization is DISABLED** (commented out in Pipeline_ERTS.py line ~5811)
✅ **Back to original behavior** (non-normalized inputs)
✅ **Using old architecture** (`from emkf.AI_M_step import DeltaF_MStepNet`)

**Your code will now work as before!**

---

## What We Learned

### **Your Observation Was Correct:**
> "When x_0 is high, MSE is high and EM doesn't help"

This is TRUE! But the solution isn't just normalization.

### **The Real Problem:**
High x_0 → Poor RTSNet smoothing → Biased statistics (A1, A2, etc.) → Bad F estimate

**Chain of failures:**
```
Large x_0 error
   ↓
RTSNet struggles to converge
   ↓
x_smooth has errors
   ↓
A1 = (x_smooth @ x_smooth.T)/T is biased
   ↓
F_estimate from A1, A2 is biased
   ↓
M-network (trained on good stats) fails on bad stats
```

---

## Solutions (Ranked by Effort)

### **1. Better Initial Conditions** ⭐ EASIEST - NO RETRAINING

**Idea:** Estimate x_0 from observations instead of using default

```python
# Add this function to try.py:
def estimate_x0_from_observations(y_seq, H, m, n_obs=5):
    """
    Estimate x_0 from first few observations.
    If H is full rank, use least squares.
    """
    # Use first n_obs observations
    Y = y_seq[:, :n_obs]  # [n, n_obs]
    
    # Simple method: average of H^{-1} y_t
    if H.shape[0] == H.shape[1]:  # Square H
        try:
            H_inv = torch.linalg.inv(H)
            x0_estimates = H_inv @ Y  # [m, n_obs]
            x0 = x0_estimates.mean(dim=1)  # [m]
            return x0
        except:
            pass
    
    # Fallback: least squares
    y0 = y_seq[:, 0]
    x0 = torch.linalg.lstsq(H, y0.unsqueeze(-1)).solution.squeeze()
    return x0

# Then in try.py around line 320, BEFORE calling one_test_mstep_net:
print("\n=== Computing better initial conditions ===")
H = sys_model_2.H
init_x_list = []
for i in range(len(test_input)):
    x0_est = estimate_x0_from_observations(test_input[i], H, sys_model_2.m)
    init_x_list.append(x0_est)
    if i % 20 == 0:
        print(f"Seq {i}: x0_default={torch.norm(sys_model_2.m1x_0).item():.3f}, "
              f"x0_estimated={torch.norm(x0_est).item():.3f}")

# Then use these better initial conditions:
RTSNet_Pipeline.one_test_mstep_net(
    sys_model_2, test_input, test_target,
    destination_path_RTS=path_results_wrong_rts,
    destination_path_M=destination_path_M,
    lambda_F=1e-3,
    generate_f=True,
    init_x_list=init_x_list,  # ← Use estimated initial conditions!
    init_P_list=None,
    non_linear_h=False
)
```

**Expected improvement:** 2-5 dB better when x_0 is far from default!

---

### **2. Switch to Simple Architecture** ⭐⭐ MEDIUM EFFORT

You have 3 architectures available:

```python
# In Pipelines/Pipeline_ERTS.py line 16-18:

# Current (Complex - 998K params, ignores analytical):
from emkf.AI_M_step import DeltaF_MStepNet

# Option A (Simple - 10K params, uses analytical + corrections):
# from emkf.AI_M_step_simple import DeltaF_MStepNet_Simple as DeltaF_MStepNet

# Option B (Simple Normalized - 10K params, scale-invariant):
# from emkf.AI_M_step_simple_normalized import DeltaF_MStepNet_Simple_Normalized as DeltaF_MStepNet
```

**If you switch:**
1. Must retrain the M-network (different architecture)
2. Simple uses analytical as base → More robust
3. Normalized version would help with x_0 problem (but must train with normalization!)

---

### **3. Retrain with Normalized Inputs** ⭐⭐⭐ LONG TERM

To truly fix the x_0 scale problem:

**Step 1:** Switch Pipeline to use normalized architecture:
```python
from emkf.AI_M_step_simple_normalized import DeltaF_MStepNet_Simple_Normalized as DeltaF_MStepNet
```

**Step 2:** Modify training function to normalize (find all places that create z_in in training functions)

**Step 3:** Add normalization at test time (uncomment lines 5811-5817)

**Step 4:** Retrain from scratch

**Result:** Network will be scale-invariant and robust to any x_0!

---

### **4. Test with Analytical EM** 🔬 DIAGNOSTIC

To understand if the problem is the network or the statistics:

```python
# In try.py, after line 320, add this quick test:
print("\n=== Testing Analytical EM (no network) ===")

# Temporarily bypass network
def test_analytical_em_only(sys_model, test_input, test_target, F_init):
    """Test using pure analytical EM solution"""
    N_T = len(test_input)
    x_loss_before_list = []
    x_loss_after_list = []
    f_loss_list = []
    
    for j in range(N_T):
        y_seq = test_input[j]
        x_true = test_target[j]
        F_true = sys_model.F_test_TRUE[j // 10]
        F_current = F_init[j // 10].clone()
        
        # ... (run RTS smoother with F_current)
        # ... (compute statistics A1, A2)
        
        x_loss_before = ...
        
        # Analytical M-step (no network!)
        I = torch.eye(2)
        A2_reg = A2 + 1e-3 * I
        F_analytical = torch.linalg.solve(A2_reg.T, A1.T).T
        
        # ... (run RTS smoother with F_analytical)
        
        x_loss_after = ...
        f_loss = torch.mean((F_analytical - F_true)**2)
        
        print(f"Seq {j}: x_before={10*math.log10(x_loss_before):.1f}dB, "
              f"x_after={10*math.log10(x_loss_after):.1f}dB, "
              f"F_loss={10*torch.log10(f_loss):.1f}dB")

test_analytical_em_only(sys_model_2, test_input, test_target, sys_model_2.F_test)
```

**If analytical EM works well:** Network is the problem → Retrain or use simpler architecture
**If analytical EM also fails:** Statistics are bad → Need better smoothing (initial conditions)

---

## Immediate Action

**Try Solution #1 (Better Initial Conditions)** - I provided the code above!

This should help immediately without any retraining. Just:
1. Add the `estimate_x0_from_observations` function
2. Compute `init_x_list` before calling `one_test_mstep_net`
3. Pass `init_x_list=init_x_list` to the function

**Expected:** Performance improves for sequences with high x_0 errors!

---

## Bottom Line

✅ **Normalization is disabled** - your code works as before
✅ **You found the real problem** - high x_0 causes cascade of failures
❌ **Test-time normalization failed** - training/test mismatch
⭐ **Solution:** Better initial conditions (easiest!) or retrain with normalization (best long-term)

Try the better initial conditions fix I provided above - it's a one-time code addition with no retraining! 🚀

