import pandas as pd
import numpy as np
import yfinance as yf
import torch
import collections
from Simulations.Linear_sysmdl import SystemModel
from emkf.main_emkf_func import EMKF_FH_analytic

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda")

# ===============================
# Download ONE YEAR of data
# ===============================
ticker = "SPY"
start_date = "2017-01-01"
end_date = "2018-01-01"
data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    progress=True
)
SPY = data.copy()
# ---- FIX: flatten MultiIndex columns (yfinance sometimes returns this) ----
if isinstance(SPY.columns, pd.MultiIndex):
    SPY.columns = SPY.columns.get_level_values(0)

# ===============================
# Build returns + volume-log-diff + momentum-diff
# ===============================
cols_price = ["Open", "Adj Close", "High", "Low", "Volume"]
SPY = SPY[cols_price].copy()
assert SPY.index.is_unique, "SPY index has duplicates!"
# 1) Returns for price columns
btc = SPY[["Open", "Adj Close", "High", "Low"]].pct_change()##presentage difference

# 2) Log-volume diff (volume "return"-like feature)
btc["Volume"] = np.log(SPY["Volume"] + 1.0).diff()

# 3) Momentum over 10 days (mean of returns)
btc["Mom_10"] = btc["Adj Close"].rolling(15).mean()

# 4) Momentum change = diff of momentum  (this is what you asked!)
btc["dMom_10"] = btc["Mom_10"].diff()

# 5) Drop NaNs created by pct_change / diff / rolling
btc = btc.dropna()

# 6) Use dMom_10 as feature (recommended)
cols = ["Open", "Adj Close", "High", "Low", "Volume", "dMom_10"]
# cols = cols_price

print(f"After SMA: {len(btc)} days")


###################################################################################
# dates_list stores tomorrow_date for each step (as in your code)

start_trade_date = btc.index[30 - 1]
end_trade_date = btc.index[-2]

P_start = float(SPY.loc[start_trade_date, "Adj Close"])
P_end   = float(SPY.loc[end_trade_date, "Adj Close"])

bh_return = (P_end - P_start) / P_start
bh_multiple = 1.0 + bh_return

print(f"Buy&Hold (strategy horizon): {start_trade_date.date()} -> {end_trade_date.date()}")
print(f"Buy&Hold multiple={bh_multiple:.4f}  return={bh_return*100:+.2f}%")

###################################################################################


# ======================================================
# ROLLING WINDOW APPROACH
# ======================================================
TAU = 30  # training window size

# k as percentage threshold (example: k=1.0 means 1%)
k_pct = 0.05
k = k_pct / 100.0

# Storage for BOTH methods
pred_A = []      # method A: use x_T (KF/RTS estimate)
true_list = []   # true price
mse_A = []
dates_list = []
pred_naive = []     # baseline: tomorrow = today
mse_naive = []
abs_err_A = []
pct_err_A = []
oracle_actions = []  # "buy"/"sell"/"hold"
oracle_rets = []     # store real returns used

# State-space dimensions
m = len(cols)
n = len(cols)


# Initial F, B (will be updated)
F_prev = torch.eye(m, device=device) * 0.9
H_prev = torch.eye(n, m, device=device)
print(f"\n{'='*60}")
print(f"ROLLING WINDOW PREDICTION (step through {len(btc) - TAU} days)")
print(f"{'='*60}\n")

x0_prev = torch.zeros(m, 1, device=device)

equity_strategy = 1.0
equity_buyhold  = 1.0

pos = 0  # 1=long, 0=flat, -1=short

signal_list = []
pos_list = []
equity_str_list = []
equity_bh_list = []
ret_real_list = []
ret_pred_list = []
oracle_state = 0
oracle_equity = 1.0
arc_buy =0
orc_sell=0
false_buy = 0
false_sell = 0
true_buy = 0
true_sell = 0
hold_buy = 0
hold_sell = 0
for window_idx in range(len(btc) - TAU):
    # ====== Get training window ======
    # Days: window_idx to window_idx+TAU-1 (50 days)
    train_start = window_idx
    train_end = window_idx + TAU#the next day index after the window

    X_window = btc[cols].iloc[train_start:train_end].values  # [50, 5]

    train = X_window  # [50, 5]
    mu_w = train.mean(axis=0)  # shape (5,)
    sig_w = train.std(axis=0) + 1e-8  # shape (5,)

    X_window = (train - mu_w) / sig_w
    # Convert to torch (add batch dimension)
    X_batch = torch.tensor(X_window[np.newaxis, :, :], dtype=torch.float32, device=device).permute(0, 2, 1)  # [1, 5, 50]

    # ====== System Model Setup ======
    T = TAU
    Q = 1e-4 * torch.eye(m, device=device)  # Increased for stability
    R = 1e-4* torch.eye(n, device=device)  # Increased for stability

    x0 = x0_prev.clone()
    P0 = 1e-2 * torch.eye(m, device=device)  # Increased for stability

    sys_model = SystemModel(F_prev, Q, H_prev, R, T, T)
    sys_model.InitSequence(x0, P0)

    # ====== Factored Initialization (warm-start from previous) ======

    F_init_list = [F_prev.clone()]  # one sequence -> list length 1
    H_init_list = [H_prev.clone()]

    F_hist, H_hist, last_x_list, last_P_list = EMKF_FH_analytic(sys_model=sys_model,F_init_list=F_init_list,H_init_list=H_init_list,Q=Q,R=R,Y=X_batch,  # [1, n, T]
        x_0=x0,P_0=P0,X_true=X_batch,  # only for MSE print inside smoother, can pass X_batch
        max_it=30,  # start small (10); EM inside rolling windows is heavy
        generate_f=False,generate_h=False,init_x_list=None,init_P_list=None)

    F_learned = F_hist[0][-1]  # sequence 0, last iter
    H_learned = H_hist[0][-1]
    print('F_PREDICTED:', F_learned)
    print('H_PREDICTED:', H_learned)
    xT = last_x_list[0]  # (m,1) last smoothed state from RTS

    # warm start next window initial state
    x0_prev = xT.detach().clone()

    # ======================================================
    # NEXT-DAY PREDICTION (two methods)
    # ======================================================


    # METHOD A: use smoothed state x_T from EMKF

    x_next_A = F_learned @ xT
    y_next_pred = H_learned @ x_next_A
    pred_diff_A_norm = y_next_pred[1, 0].item()#the precetage of change in adj close price

    # ====== True Price (next day, index = train_end) ======

    tomorrow_price_index = train_end

    if tomorrow_price_index + 1 < len(btc):

        today_date = btc.index[tomorrow_price_index -1]
        tomorrow_date = btc.index[tomorrow_price_index]

        last_price = float(SPY.loc[today_date, "Adj Close"])
        true_price = float(SPY.loc[tomorrow_date, "Adj Close"])

        adj_idx = cols.index("Adj Close")
        mu_ac = float(mu_w[adj_idx])
        sig_ac = float(sig_w[adj_idx])

        pred_ret  = pred_diff_A_norm * sig_ac + mu_ac #prediction diff price of tomorrow in unnormalized scale
        pred_A_price = last_price * (1.0 + pred_ret)  # P_hat_t
        mse_a = (pred_A_price - true_price) ** 2
        mse_A.append(mse_a)
        abs_err = abs(pred_A_price - true_price)
        pct_err = 100.0 * abs_err / (abs(true_price) + 1e-12)  # MAPE per sample
        abs_err_A.append(abs_err)
        pct_err_A.append(pct_err)

        pred_A.append(pred_A_price)
        # ===== NAIVE BASELINE: predict tomorrow = today =====
        pred_naive_price = last_price  # tomorrow = today
        pred_naive.append(pred_naive_price)

        mse_n = (pred_naive_price - true_price) ** 2
        mse_naive.append(mse_n)

        true_list.append(true_price)
        dates_list.append(tomorrow_date)

        # warm start for next window
        F_prev = F_learned.detach().clone()
        H_prev = H_learned.detach().clone()
        # B_prev = B_learned.detach().clone()

        # ======================================================
        # TRADING UPDATE (right after each prediction)
        # ======================================================

        # predicted return (tomorrow vs today)
        pred_ret_trade = pred_ret
        real_ret_trade = (true_price - last_price) / last_price
########################################################
        # ===== ORACLE (perfect knowledge of tomorrow) =====
        oracle_ret = real_ret_trade  # this IS the true tomorrow return

        if oracle_ret >=  0:
            oracle_sig = "buy"
            oracle_equity *= (1.0 + oracle_ret)
            arc_buy+=1
        elif oracle_ret < 0:
            oracle_sig = "sell"
            oracle_equity *= (1.0 - oracle_ret)
            orc_sell +=1
        oracle_actions.append(oracle_sig)
        oracle_rets.append(oracle_ret)
###################################################################
        # signal from predicted return
        # if pred_ret_trade > k:
        #     sig = "buy"
        #     equity_strategy *= (1.0 + real_ret_trade)
        #     if oracle_ret > 0:
        #         true_buy += 1
        #     else:
        #         false_buy += 1
        # elif pred_ret_trade < -k:
        #     sig = "sell"
        #     equity_strategy *= (1.0 - real_ret_trade)
        #     if oracle_ret < 0:
        #         true_sell += 1
        #     else:
        #         false_sell += 1
        # else:
        #     sig = "hold"
        #     equity_strategy *= 1.0
        #     if oracle_ret > 0:
        #         hold_buy += 1
        #     else:
        #         hold_sell += 1
#######################################################################
        # STRATEGY (your requested: immediate action, no position memory)
        if pred_ret > k:
            sig = "buy"
            equity_strategy *= (1.0 + real_ret_trade)
            if real_ret_trade > 0:
                true_buy += 1
            else:
                false_buy += 1
        elif pred_ret < -k:
            sig = "sell"
            equity_strategy *= (1.0 - real_ret_trade)
            if real_ret_trade < 0:
                true_sell += 1
            else:
                false_sell += 1
        else:
            if sig == "buy":
                equity_strategy *= (1.0 + real_ret_trade)
            elif sig == "sell":
                equity_strategy *= (1.0 - real_ret_trade)
            # sig_label = "hold"
            # equity_str *= 1.0
            if real_ret_trade > 0:
                hold_buy += 1
            else:
                hold_sell += 1
#########################################################################
        # # signal from predicted return
        # if pred_ret_trade > k:
        #     sig = "buy"
        # elif pred_ret_trade < -k:
        #     sig = "sell"
        # else:
        #     sig = "hold"
        #
        # # update position according to your rules
        # if pos == 1:
        #     # have stock: buy/hold -> keep; sell -> exit to flat
        #     if sig == "sell":
        #         pos = 0
        # elif pos == 0:
        #     # flat: buy -> long, sell -> short, hold -> stay flat
        #     if sig == "buy":
        #         pos = 1
        #     elif sig == "sell":
        #         pos = -1
        # elif pos == -1:
        #     # short: buy -> flip to long, hold/sell -> stay short
        #     if sig == "buy":
        #         pos = 1
        #
        # # apply overnight PnL (accumulate immediately)
        # if pos == 1:
        #     equity_strategy *= (1.0 + real_ret_trade)
        # elif pos == -1:
        #     equity_strategy *= (1.0 - real_ret_trade)
        # else:
        #     equity_strategy *= 1.0
        #
        # # buy & hold benchmark over the same horizon (always long)
        equity_buyhold *= (1.0 + real_ret_trade)



        # store for later
        signal_list.append(sig)
        # pos_list.append(pos)
        equity_str_list.append(equity_strategy)
        equity_bh_list.append(equity_buyhold)
        ret_real_list.append(real_ret_trade)
        ret_pred_list.append(pred_ret_trade)
        if window_idx % 50 == 0:
            print("today_date:", today_date, "tomorrow_date:", tomorrow_date)
            print("last_price:", last_price, "true_price:", true_price)
            print("real_ret:", real_ret_trade, "pred_ret:", pred_ret_trade)
            print("-" * 40)
        if (window_idx + 1) % 1 == 0:
            print(
                f"Day {window_idx + 1:3d}: True={true_price:8.2f}, "  
                f"A={pred_A_price:8.2f} (RMSE={np.sqrt(mse_a):7.2f}), "
                f"Naive={pred_naive_price:8.2f} (RMSE={np.sqrt(mse_n):7.2f})"
            )
results_df = pd.DataFrame({
    "Date": dates_list,
    "True": true_list,
    "Pred_A_xT": pred_A,
    "Pred_Naive": pred_naive,
    "MSE_A": mse_A,
    "MSE_Naive": mse_naive,
    "RMSE_A": [np.sqrt(x) for x in mse_A],
    "RMSE_Naive": [np.sqrt(x) for x in mse_naive],})

print(collections.Counter(signal_list))
# print(collections.Counter(pos_list))

results_df["Pred_ret_trade"] = ret_pred_list
results_df["Real_ret_trade"] = ret_real_list
results_df["Signal"] = signal_list
# results_df["Position"] = pos_list
results_df["Equity_Strategy"] = equity_str_list
results_df["Equity_BuyHold"] = equity_bh_list

final_strategy = equity_strategy
final_bh = equity_buyhold

print("\n" + "="*60)
print(f"TRADING BACKTEST (accumulated per step) | k = {k_pct:.2f}%")
print("="*60)
print(f"Strategy final multiple: {final_strategy:.4f}  -> Return: {(final_strategy-1)*100:+.2f}%")
print(f"Buy&Hold final multiple: {final_bh:.4f}  -> Return: {(final_bh-1)*100:+.2f}%")
print(f"Difference (Strategy - Buy&Hold): {((final_strategy-final_bh)*100):+.2f}% (in multiples*100)")
print("="*60)


print(results_df.tail())
avg_mse_close = np.mean(mse_A)                 # USD^2
avg_rmse_close = np.sqrt(avg_mse_close)        # USD
avg_mape_close = np.mean(pct_err_A)            # %
median_ape_close = np.median(pct_err_A)        # % (optional, robust)



true_arr = np.array(true_list, dtype=np.float64)
predA_arr = np.array(pred_A, dtype=np.float64)
predN_arr = np.array(pred_naive, dtype=np.float64)

avg_rel_err_A = np.mean(np.abs((predA_arr - true_arr) / true_arr))
avg_rel_err_naive = np.mean(np.abs((predN_arr - true_arr) / true_arr))

print("\nPRICE ACCURACY — Average Relative Error")
print(f"AvgRelErr A (EMKF):   {avg_rel_err_A*100:+.3f}%")
print(f"AvgRelErr Naive:      {avg_rel_err_naive*100:+.3f}%")


results_df.to_csv("rolling_window_predictions_two_methods.csv", index=False)

oracle_counts = collections.Counter(oracle_actions)

print("\n" + "="*60)
print(f"ORACLE BACKTEST (knows tomorrow) | k = {k_pct:.2f}%")
print("="*60)
print(f"Oracle final multiple: {oracle_equity:.4f}  -> Return: {(oracle_equity-1)*100:+.2f}%")
print(f"Oracle counts: {oracle_counts}")
print("="*60)
print('number of oracle buy signals:',arc_buy)
print('number of oracle sell signals:',orc_sell)

print(f"Buy&Hold (strategy horizon): {start_trade_date.date()} -> {end_trade_date.date()}")
print(f"Buy&Hold multiple={bh_multiple:.4f}  return={bh_return*100:+.2f}%")


print('true buy:',true_buy)
print('false buy:',false_buy)
print('true sell:',true_sell)
print('false sell:',false_sell)
print('hold buy:',hold_buy)
print('hold sell:',hold_sell)
