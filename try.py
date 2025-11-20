
# Save as `tests/instability_debug.py` and run from project root.

import importlib
import traceback
from importlib import reload
import torch
import math

torch.manual_seed(0)
torch.set_printoptions(profile="short")

# pick one failing config you reported (example)
cfg = dict(use_proj=True, detach_prev=False, reset_hidden=False, detach_hidden_after=False, seed_psmooth_from_input=True)

# reload module
import RTSNet.PsmoothNN_combined as mod
reload(mod)

PNotSmoothNN = mod.PNotSmoothNN
PsmoothFromPnot = mod.PsmoothFromPnot

# helper to report a tensor
def report_tensor(name, x):
    if x is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tuple(x.shape)}, nan={bool(torch.isnan(x).any())}, inf={bool(torch.isinf(x).any())}, min={float(x.min()):.6g}, max={float(x.max()):.6g}")

def debug_one(config):
    device = torch.device("cpu")
    m = 3; n = 2; lr = 1e-3; T = 10
    # small models
    p0 = torch.eye(m)
    net_not = PNotSmoothNN(m, n, p0.clone())
    net_smooth = PsmoothFromPnot(m)
    net_not.to(device); net_smooth.to(device)
    opt = torch.optim.SGD(list(net_not.parameters()) + list(net_smooth.parameters()), lr=lr)

    K_t = torch.randn(m, n, device=device)
    S_gain = torch.randn(m, m, device=device)
    A = torch.randn(m, m, device=device)
    P_target = (A @ A.T).detach()

    P_prev = p0.clone().to(device)
    net_not.train(); net_smooth.train()
    net_not.reset_state(); net_smooth.reset_state()
    if not config['seed_psmooth_from_input']:
        net_smooth.start = 1
        net_smooth.h_Psmooth = torch.zeros(1, 1, net_smooth.d_hidden_Psmooth, device=device)

    # optionally replace projector
    reload(mod)
    if not config['use_proj']:
        mod.enforce_covariance_properties = lambda P, eps=1e-6: P

    # enable anomaly detection to get stack when autograd fails
    torch.autograd.set_detect_anomaly(True)

    for t in range(T):
        try:
            if config['reset_hidden']:
                net_not.reset_state(); net_smooth.reset_state()
                if not config['seed_psmooth_from_input']:
                    net_smooth.start = 1
                    net_smooth.h_Psmooth = torch.zeros(1, 1, net_smooth.d_hidden_Psmooth, device=device)

            net_not.F = torch.randn(m, m, device=device) * 0.1
            P_feed = P_prev if not config['detach_prev'] else P_prev.detach()

            opt.zero_grad()

            # forward P_not
            try:
                P_not = net_not.forward(K_t.to(device), P_feed.to(device))
            except Exception as e:
                print("Exception during net_not.forward():", repr(e))
                traceback.print_exc()
                return
            report_tensor("P_not (after forward)", P_not)

            # forward P_smooth
            try:
                P_smooth = net_smooth.forward(P_not, S_gain.to(device))
            except Exception as e:
                print("Exception during net_smooth.forward():", repr(e))
                traceback.print_exc()
                # print intermediate debug info from net_smooth inputs
                report_tensor("Input to Psmooth (P_not)", P_not)
                return
            report_tensor("P_smooth (after forward)", P_smooth)

            loss = torch.norm(P_smooth - P_target)**2
            print(f"step={t} loss finite={torch.isfinite(loss).item()} loss={float(loss):.6g}")

            # backward with anomaly detection
            try:
                loss.backward()
            except Exception as e:
                print("Exception during backward:", repr(e))
                traceback.print_exc()
                # print grads presence
                for name, p in list(net_not.named_parameters()) + list(net_smooth.named_parameters()):
                    if p.grad is None:
                        continue
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print("bad grad:", name, "nan/inf present")
                return

            # compute grad norm
            total = 0.0
            any_nan = False
            for p in list(net_not.parameters()) + list(net_smooth.parameters()):
                if p.grad is None:
                    continue
                g = p.grad
                if torch.isnan(g).any() or torch.isinf(g).any():
                    any_nan = True
                total += (g.detach().norm()**2).item()
            grad_norm = math.sqrt(total)
            print(f"step={t} grad_norm={grad_norm:.6g} any_nan_grad={any_nan}")

            opt.step()
            P_prev = P_not

        except Exception as e:
            print("Outer exception:", repr(e))
            traceback.print_exc()
            return

if __name__ == "__main__":
    print("Debugging config:", cfg)
    debug_one(cfg)
