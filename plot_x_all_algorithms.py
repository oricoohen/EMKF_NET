"""
Plot the estimated state x for every algorithm, one PNG per algorithm.

It loads the x-estimates saved by the two test scripts:
  - main_lor_DT_3datasets_test.py       -> x_estimates/neural_x.pt
  - main_lor_DT_3dataset_analitic_test.py -> x_estimates/analytic_x.pt

Run BOTH of those first, then run this script:
    python plot_x_all_algorithms.py

Output: one figure per algorithm in the folder x_pictures/.
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")  # save to file, no interactive window needed
import matplotlib.pyplot as plt

IN_DIR = "x_estimates"
OUT_DIR = "x_pictures"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load whatever payloads are available
# ------------------------------------------------------------------
payloads = {}
for src, fname in [("neural", "neural_x.pt"), ("analytic", "analytic_x.pt")]:
    path = os.path.join(IN_DIR, fname)
    if os.path.exists(path):
        payloads[src] = torch.load(path, map_location="cpu", weights_only=False)
        print(f"Loaded {path}")
    else:
        print(f"WARNING: {path} not found — run the corresponding test script first.")

# ------------------------------------------------------------------
# The 7 algorithms the user asked for, in order.
# (display name, key inside the payload's 'algos' dict, source payload)
# ------------------------------------------------------------------
SPEC = [
    ("true x",       "true x",       "neural"),    # ground truth (from the neural run)
    ("rts true",     "rts true",     "analytic"),  # analytic RTS smoother, TRUE H
    ("rts false",    "rts false",    "analytic"),  # analytic RTS smoother, initial/false H
    ("rtsnet false", "rtsnet false", "neural"),    # RTSNet, initial/false H
    ("emkf",         "emkf",         "analytic"),  # analytic EMKF
    ("emkalmanet",   "emkalmanet",   "neural"),    # AI EMKF (neural EM)
    ("bgru",         "bgru",         "neural"),     # BiGRU smoother
]


def safe_name(s):
    return s.replace(" ", "_").replace("/", "")


for display_name, key, src in SPEC:
    if src not in payloads:
        print(f"Skipping '{display_name}' — payload '{src}' not loaded.")
        continue
    algos = payloads[src]["algos"]
    if key not in algos:
        print(f"Skipping '{display_name}' — key '{key}' missing in {src} payload.")
        continue

    x = algos[key]                      # [m, cycles*T]
    if hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()
    cycles = payloads[src]["cycles"]
    T_len = payloads[src]["T_len"]
    sample_idx = payloads[src].get("sample_idx", 0)

    plt.figure(figsize=(12, 6))
    for dim in range(x.shape[0]):
        plt.plot(x[dim], label=f"x[{dim}]", linewidth=2)

    # dataset boundaries
    for d in range(1, cycles):
        plt.axvline(d * T_len, linestyle="--", color="gray", alpha=0.5)

    plt.title(f"{display_name} — estimated x vs time (sample {sample_idx}, {cycles} datasets)")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"x_{safe_name(display_name)}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

print("\nDone. Pictures are in the folder:", OUT_DIR)
