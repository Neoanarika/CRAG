import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Import your CAT implementation
from cat import run_cat_population  # must be in the same directory

def load_inputs(device="cuda" if torch.cuda.is_available() else "cpu"):
    # You likely already load data inside run_cat_population in your setup.
    # If not, adapt this to your environment.
    import pandas as pd
    df = pd.read_csv("test.csv")
    df = df.iloc[:, 1:]  # drop id column if present
    data = torch.tensor(df.to_numpy(), dtype=torch.float32, device=device)
    observed = (data != -1).to(torch.float32)
    data0 = data * observed

    thetas = torch.load("thetas.pt", map_location=device)
    thetas = thetas["thetas"] if isinstance(thetas, dict) else thetas
    thetas = thetas.to(torch.float32).to(device).detach()

    zs = torch.load("zs.pt", map_location=device)
    zs = zs["zs"] if isinstance(zs, dict) else zs
    zs = zs.to(torch.float32).to(device).detach()
    return data0, observed, thetas, zs

def one_run(seed, max_items=30, target_se=0.30, k_randomesque=3, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data0, obs, theta_star, zs = load_inputs(device)
    df_cat = run_cat_population(
        data_with0=data0,
        observed_idtor=obs,
        thetas_full=theta_star,
        zs=zs,
        max_items=max_items,
        target_se=target_se,
        k_randomesque=k_randomesque,
    )
    # Mean RMSE per step & variant for this run
    cur = (df_cat.groupby(["variant","step"], as_index=False)
                 .agg(rmse=("rmse_vs_full","mean")))
    cur["run"] = seed
    return cur

def aggregate_30_runs(n_runs=30, **kwargs):
    curves = []
    for s in range(n_runs):
        print(f"Run {s+1}/{n_runs}")
        curves.append(one_run(seed=s, **kwargs))
    allc = pd.concat(curves, ignore_index=True)
    agg = (allc.groupby(["variant","step"], as_index=False)
                .agg(rmse_mean=("rmse","mean"),
                     rmse_std=("rmse","std"),
                     n_runs=("run","nunique")))
    return agg.sort_values(["variant","step"])

def plot_rmse_with_errorbars(curve_agg, out_png="rmse_30runs.png", out_pdf="rmse_30runs.pdf"):
    # y-axis with padding, no grid
    ymin = float(curve_agg["rmse_mean"].min())
    ymax = float(curve_agg["rmse_mean"].max())
    pad  = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) if ymax != 0 else 1.0))
    y_lo = max(0.0, ymin - pad)
    y_hi = ymax + pad

    plt.figure(figsize=(9,5))
    for variant, sub in curve_agg.groupby("variant"):
        sub = sub.sort_values("step")
        # error bars: ±1 standard deviation across the 30 runs
        plt.errorbar(
            sub["step"], sub["rmse_mean"], yerr=sub["rmse_std"],
            fmt='-o', capsize=3, label=str(variant)
        )
    plt.xlabel("Administered items (step)")
    plt.ylabel("RMSE")
    plt.ylim(y_lo, y_hi)
    # No grid:
    # (do nothing; matplotlib defaults to no grid)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curve_agg = aggregate_30_runs(
        n_runs=50, max_items=30, target_se=0.30, k_randomesque=3, device=device
    )
    curve_agg.to_csv("cat_rmse_30runs_summary.csv", index=False)
    plot_rmse_with_errorbars(curve_agg)

