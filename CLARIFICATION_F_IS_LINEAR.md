# CLARIFICATION: F is Linear - Only h is Non-Linear

## You are 100% CORRECT!

**F is LINEAR** - it's just a matrix multiplication: `x_{t+1} = F @ x_t + noise`

**h is NON-LINEAR** - it's the observation function from `parameters.py`

## Why We Use make_f(F)?

The `SystemModel` class constructor signature requires:
```python
SystemModel(f, Q, h, R, T, T_test, m, n)
```

Where `f` must be a **function**, not a matrix.

So `make_f(F)` simply wraps the linear matrix F as a function:

```python
def make_f(F_mat):
    def f_func(x):
        return F_mat @ x  # Still linear! Just F @ x
    return f_func
```

## What Changed in Your Code

**Before (WRONG):**
```python
sys_model = SystemModel(F_current, Q, h_nonlinear, R, args.T, args.T_test)
# Missing m, n parameters
# Passing matrix F instead of function f
```

**After (CORRECT):**
```python
f_current = make_f(F_current)  # Wrap F as a function (still linear!)
sys_model = SystemModel(f_current, Q, h_nonlinear, R, args.T, args.T_test, m, n)
# Now has m, n parameters
# Passing function f as required
```

## The Physics Hasn't Changed!

- **State evolution**: Still `x_{t+1} = F @ x_t + process_noise` (LINEAR)
- **Observations**: Still `y_t = h_nonlinear(x_t) + measurement_noise` (NON-LINEAR)

The only change is **how we pass F to SystemModel** - it needs to be wrapped as a function `f(x)` for the API.

## Summary

✅ **F is linear** - you were right!  
✅ **h is non-linear** - as you wanted  
✅ **make_f(F)** just wraps F as a function - still does `F @ x`  
✅ **Your physics is unchanged** - only the API wrapper changed  

Sorry for any confusion! The code is now correct.

