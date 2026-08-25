# Batched training plan (N_B parallelization)

Goal: run all `N_B = 15` samples of a batch through RTSNet **together** on the GPU instead of
one Python-loop iteration at a time. Targets the two functions actually in use:

- `train_RTS_net_3_datasets`
- `train_H_mstep_net_3_datasets_joint`

Expected speedup: ~N_B× (≈10–15×) since the per-sample loop is replaced by one batched pass.

## What parallelizes vs. what stays sequential

Parallelized over the batch dim B (=15):
- the `for j in range(N_B)` sample loop  ← the whole win

Stays sequential (cannot change — true recursion / data dependency):
- time loop `t`
- EM-iteration loop (H feeds the next pass)
- dataset loop (`x_0` chains across the 3 datasets)

## Why "option 1" (group by H) was rejected

In the M-step the M-network predicts a different ΔH per sample from per-sample statistics,
so H_current diverges after the first EM iteration. We MUST carry per-sample H as a
`[B, n, m]` tensor and use batched matrix products (`bmm`). Same for `train_RTS_net_3_datasets`
where H differs per sample via the random `n_e`.

## Assumption

All sequences in a dataset share the same length T (uniform), so they stack into `[B, ·, T]`
cleanly. If T ever varies we'd need padding + a time mask (extra work, not in this plan).

---

## Files (all NEW unless marked)

1. `RTSNet/KalmanNet_nn_batched.py`   (NEW) — batched forward/filter base class
2. `RTSNet/RTSNet_nn_batched.py`      (NEW) — batched smoother subclass
3. `Pipelines/Pipeline_ERTS_batched.py` (NEW) — subclass of `Pipeline_ERTS` adding:
     - `train_RTS_net_3_datasets_batched`
     - `train_H_mstep_net_3_datasets_joint_batched`
     - `_load_rtsnet_batched` (weight-transfer helper)
4. your run script — change ONE import / class name to use the batched pipeline.

Revert = delete files 1–3 and undo the one-line import in the run script. Nothing else touched.

---

## Phase 1 — batched model

- Add a `batch_size` set per call (default 1, so B=1 still behaves exactly as today).
- Carry `self.H` / `self.F` as `[B, n, m]` / `[B, m, m]`; `update_H`/`update_F` accept a stacked batch.
- Replace matrix-vector products in `KNet_step` / `RTSNet_step` / gain steps with `torch.bmm`.
- Per-step inputs become `[B, m]` → expanded to `[1, B, feat]` for the GRUs
  (GRUs already use a batch dim, so this is a clean generalization of `expand_dim`).
- Subclass the existing model classes and override only the forward-path math methods,
  so all layer definitions (and therefore parameter names) are reused unchanged.

## Phase 2 — checkpoint compatibility

Checkpoints are full pickled `RTSNetNN` objects. The batched class has identical layers, so:
`batched = RTSNetNN_batched(); batched.NNBuild(...); batched.load_state_dict(old_model.state_dict())`.
Old `.pt` files load with zero conversion friction; saving still produces a loadable model.

## Phase 3 — `train_RTS_net_3_datasets_batched` (do this FIRST — simpler)

- Sample `n_e` once per batch slot → build `[B, n, T]` y, `[B, m, T]` target, `[B, n, m]` H stacks.
- Keep the dataset loop (3) and the time loop (T); batch the rest.
- Single `loss.backward()` over the batched mean.

## Phase 4 — `train_H_mstep_net_3_datasets_joint_batched`

- Same stacking; M-network input becomes `[B, feat]` (a plain MLP batches automatically —
  no M-net change needed).
- EM stats (`A_yx`, `A_xx`, `S_nu`, `C_nu_x`) computed as batched matmuls (`bmm` over `[B, ·, T]`).
- `H_current` is `[B, n, m]`, updated each EM iter from batched ΔH.

## Phase 5 — ACCEPTANCE TEST (this is how we know it "works good")

Numerical-equivalence check before trusting results:
- Fix the data, fix model weights, fix the RNG sequence used to pick `n_e`.
- Run the OLD sequential function and the NEW batched function for 1 epoch.
- Assert the per-epoch training loss matches to ~1e-4. If it matches, the math is identical and
  only speed changed. Then record wall-clock per epoch old vs new.

## Risk notes

- Highest-risk part: the gain-network reshapes (`[1, B, feat]`) and the `bmm` shapes. Phase 5
  catches any mistake by comparing against the sequential ground truth.
- AMP/`torch.autocast` is an optional extra speedup we can add later; independent of this plan.

---
---

# PART 2 — Batched M-step  (`train_H_mstep_net_3_datasets_joint`)

Status of Part 1: DONE + validated. B=1 batched vs sequential = bit-for-bit identical
smoothed states; B=15 ≈ 13× faster per epoch. The batched model
(`RTSNet/RTSNet_nn_batched.py`) is reused as-is here.

## Good news that shrinks the job

- The M-network `DeltaH_MStepNet` ALREADY takes `z_in: [B, d_z]` and returns `[B, n, m]`.
  No change needed. (The sequential code just happened to call it with B=1 and then
  `.view(n, m)`.)
- The batched RTSNet smoother is already built and validated.

So the M-step work is purely: stack the N_B samples and rewrite the EM statistics as
batched matrix products. New code goes ONLY in `Pipeline_ERTS_batched.py` (one new
method). Nothing else is touched. Revert = same as Part 1.

## What stays sequential (unchanged from Part 1)

- time loop `t`, EM-iteration loop, dataset loop (x_0 + H_current chain across datasets).

## What gets batched (the N_B loop)

State/obs over the batch: `x_curr [B,m,T]`, `y_curr [B,n,T]`.
Per-sample H trajectory: `H_current [B,n,m]` (each sample's H diverges through EM — the
exact reason option 1 was rejected).

EM statistics, per sample, via `bmm` (T = seq length):

| sequential (single sample)        | batched                                             |
|-----------------------------------|-----------------------------------------------------|
| `A_yx = (y @ x.T)/T`   `[n,m]`    | `bmm(y, x.transpose(1,2))/T`      `[B,n,m]`         |
| `A_xx = (x @ x.T)/T`   `[m,m]`    | `bmm(x, x.transpose(1,2))/T`      `[B,m,m]`         |
| `Hx = H @ x`           `[n,T]`    | `bmm(H_current, x)`               `[B,n,T]`         |
| `nu = y - Hx`                     | same (broadcast)                                    |
| `nu_mean = nu.mean(1,keepdim)`    | `nu.mean(dim=2, keepdim=True)`    `[B,n,1]`         |
| `S_nu = (nu_c @ nu_c.T)/T` `[n,n]`| `bmm(nu_c, nu_c.transpose(1,2))/T` `[B,n,n]`        |
| `C_nu_x = (nu @ x.T)/T` `[n,m]`   | `bmm(nu, x.transpose(1,2))/T`     `[B,n,m]`         |

`z_in = cat([A_yx, A_xx, S_nu, C_nu_x, H_current] each reshaped to [B,-1], dim=1)` → `[B,d_z]`,
**detached exactly as the sequential code does**. Then `deltaH = model_mstep(z_in)` → `[B,n,m]`,
`H_current = H_current + deltaH`.

## Loss reduction (proven identical, same argument as Part 1)

Each batched loss already embeds the `1/B` average:
- `h_loss = mean((H_next - H_true)^2)` over `[B,n,m]`  = `(1/B) Σ_j h_loss_j`
- `reg    = lambda_H * mean(deltaH^2)` over `[B,n,m]`
- `x_loss = mean((x_smooth - x_true)^2)` over `[B,m,T]`

Accumulate `total += weight_em * (2*h_loss + reg + x_loss)` over em-iters and datasets,
then `loss = total / datasets`. This equals the sequential
`(1/N_B)(1/datasets) Σ_j Σ_data Σ_em weight·loss_em` exactly — so NO extra `/N_B`.

## Per-sample setup to stack (mirrors sequential exactly)

- `n_e_list = [random.randint(0, N_E-1) for _ in range(N_B)]`  (same N_B RNG draws, same order)
- `H_current` init `[B,n,m]`: from `H_init` (shared) else `H_train[0][n_e//10]` per sample;
  initialized ONCE before the dataset loop and carried across datasets (as in sequential).
- `H_true` per dataset `[B,n,m]`: stack `H_train_TRUE[data][n_e//10]`.
- `x_0 [B,m]`: from `x_0_train_list[n_e]` if given else `m1x_0` (broadcast); updated to
  `x_smooth[:,:,-1].detach()` after each dataset.
- `model.prior_Sigma = m2x_0` (shared; init_hidden broadcasts over B).

Validation mirrors this with `j` indices (no RNG), `H_valid` / `H_valid_TRUE`, and the
`H_base_cv` carry-across-datasets pattern.

## Acceptance test (same gate as Part 1)

Sequential `train_H_mstep_net_3_datasets_joint` vs batched, same seed + same initial
RTSNet & M-net weights, 1 epoch:
- B=1: expect bit-for-bit identical `loss`, `deltaH`, `H` trajectory.
- B=N_B: expect tiny float-only drift; report per-EM `x{k}/h{k}/reg{k}` line matches.
Plus post-`optimizer_joint.step()` max|Δparam| across BOTH networks.

## Risk notes

- `.detach()` on the z_in statistics must be reproduced exactly (gradient flows only
  through M-net params + the re-smoothing RTSNet pass, not the statistics).
- `nu_mean` reduces over the time axis (dim=2 in batched), not the batch axis — easy to
  get wrong; the B=1 test catches it.
- Everything else (GRUs, FC, LayerNorm in M-net) is already per-sample correct.
