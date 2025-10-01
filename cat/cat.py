
import torch
import os
import numpy as np
import pandas as pd
from typing import Callable, Dict, List

def sigmoid(x): 
    return 1.0 / (1.0 + torch.exp(-x))

def rasch_p(theta: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # probs for each selected item j at current theta (scalar)
    return sigmoid(theta - b)

def rasch_info(theta: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    p = rasch_p(theta, b)
    return p * (1 - p)

@torch.no_grad()
def mle_update_theta_rasch(theta0, b_sel, x_sel, max_iter=25, tol=1e-4):
    # (keep if you still want pure MLE for comparison)
    theta = theta0.clone()
    for _ in range(max_iter):
        p = torch.sigmoid(theta - b_sel)
        grad = (x_sel - p).sum()
        info = (p * (1 - p)).sum()
        step = grad / (info + 1e-12)
        # cap step to avoid overshoot
        step = torch.clamp(step, -1.0, 1.0)
        new_theta = torch.clamp(theta + step, -6.0, 6.0)
        if torch.abs(new_theta - theta) < tol:
            theta = new_theta
            break
        theta = new_theta
    se = (1.0 / torch.sqrt(torch.clamp(info, min=1e-6))).item()  # slightly larger floor
    return theta, torch.tensor(se, device=theta.device)

@torch.no_grad()
def map_update_theta_rasch(theta0, b_sel, x_sel, mu0=0.0, sigma0=1.0, max_iter=25, tol=1e-4):
    """
    MAP: maximize log posterior = log L + log prior, prior N(mu0, sigma0^2)
    grad = sum(x - p) - (theta - mu0)/sigma0^2
    info_post = sum(p(1-p)) + 1/sigma0^2
    """
    theta = theta0.clone()
    inv_var = 1.0 / (sigma0 ** 2)
    for _ in range(max_iter):
        p = torch.sigmoid(theta - b_sel)
        grad = (x_sel - p).sum() - (theta - mu0) * inv_var
        info_post = (p * (1 - p)).sum() + inv_var
        step = grad / (info_post + 1e-12)
        # cap step and clip theta
        step = torch.clamp(step, -1.0, 1.0)
        new_theta = torch.clamp(theta + step, -6.0, 6.0)
        if torch.abs(new_theta - theta) < tol:
            theta = new_theta
            break
        theta = new_theta
    se = (1.0 / torch.sqrt(torch.clamp(info_post, min=1e-6))).item()
    return theta, torch.tensor(se, device=theta.device)


def select_item_mfi(theta, b_all, asked_mask, observed_mask_for_person, k_randomesque=1):
    # Only among items not asked yet AND observed for this person.
    candidates = (~asked_mask) & observed_mask_for_person.bool()
    if not candidates.any():
        return None  # <- graceful signal (no candidates)
    info = rasch_info(theta, b_all)
    info[~candidates] = -torch.inf
    if k_randomesque <= 1:
        return int(torch.argmax(info).item())
    k = min(k_randomesque, int(candidates.sum().item()))
    topk_idx = torch.topk(info, k=k).indices
    return int(topk_idx[torch.randint(low=0, high=k, size=(1,))].item())

def select_item_random(asked_mask, observed_mask_for_person):
    candidates = (~asked_mask) & observed_mask_for_person.bool()
    if not candidates.any():
        return None  # <- graceful signal
    idxs = torch.nonzero(candidates, as_tuple=False).flatten()
    ridx = torch.randint(low=0, high=idxs.numel(), size=(1,))
    return int(idxs[ridx].item())

def run_cat_for_person(
    person_idx, responses_row, observed_mask_row, b_all, theta_star,
    max_items=30, target_se=0.30, k_randomesque=3, variant="adaptive"
):
    asked = torch.zeros_like(observed_mask_row, dtype=torch.bool, device=observed_mask_row.device)
    # cap by available observed items
    n_avail = int(observed_mask_row.sum().item())
    if n_avail < 3:
        # skip ultra-sparse persons (optional)
        return pd.DataFrame(columns=["pid","step","item","theta_hat","theta_se","variant","rmse_vs_full","bias_vs_full","stop_reason"])
    max_steps = min(max_items, n_avail)

    theta = torch.tensor(0.0, device=b_all.device)
    logs = {k: [] for k in ["pid","step","item","theta_hat","theta_se","variant","rmse_vs_full","bias_vs_full","stop_reason"]}

    for step in range(1, max_steps + 1):
        # pick next item
        if variant == "adaptive":
            j = select_item_mfi(theta, b_all, asked, observed_mask_row.bool(), k_randomesque=k_randomesque)
        else:
            j = select_item_random(asked, observed_mask_row.bool())

        if j is None:
            # no candidates left; record a terminal row and stop
            logs["pid"].append(int(person_idx))
            logs["step"].append(step)
            logs["item"].append(-1)
            logs["theta_hat"].append(theta.item())
            # compute SE from currently asked items (if any)
            sel_idx = torch.nonzero(asked, as_tuple=False).flatten()
            if sel_idx.numel() > 0:
                p = torch.sigmoid(theta - b_all[sel_idx])
                info = (p * (1 - p)).sum().clamp(min=1e-12)
                se = float((1.0 / torch.sqrt(info)).item())
            else:
                se = float("nan")
            logs["theta_se"].append(se)
            logs["variant"].append(variant)
            rmse = torch.sqrt((theta - theta_star)**2).item()
            logs["rmse_vs_full"].append(rmse)
            logs["bias_vs_full"].append((theta - theta_star).item())
            logs["stop_reason"].append("exhausted")
            break

        asked[j] = True
        # update theta on all administered items
        sel_idx = torch.nonzero(asked, as_tuple=False).flatten()
        b_sel = b_all[sel_idx]
        x_sel = responses_row[sel_idx].float()
        #theta, se_t = mle_update_theta_rasch(theta, b_sel, x_sel)
        theta, se_t =  map_update_theta_rasch(theta, b_sel, x_sel)

        rmse = torch.sqrt((theta - theta_star)**2).item()
        bias = (theta - theta_star).item()

        logs["pid"].append(int(person_idx))
        logs["step"].append(step)
        logs["item"].append(int(j))
        logs["theta_hat"].append(theta.item())
        logs["theta_se"].append(float(se_t))
        logs["variant"].append(variant)
        logs["rmse_vs_full"].append(rmse)
        logs["bias_vs_full"].append(bias)
        logs["stop_reason"].append("se" if se_t.item() <= target_se else "")

        if se_t.item() <= target_se:
            break

    return pd.DataFrame(logs)

def run_cat_population(data_with0, observed_idtor, thetas_full, zs,
                       max_items=30, target_se=0.30, k_randomesque=3):
    n_persons, _ = data_with0.shape
    b_all = (-zs).detach()
    rows = []
    for i in range(n_persons):
        resp_row = data_with0[i]
        obs_row  = (observed_idtor[i] == 1)
        if obs_row.sum() < 3:           # skip very sparse people (optional)
            continue
        df_adapt = run_cat_for_person(i, resp_row, obs_row, b_all, thetas_full[i],
                                      max_items=max_items, target_se=target_se,
                                      k_randomesque=k_randomesque, variant="adaptive")
        df_rand  = run_cat_for_person(i, resp_row, obs_row, b_all, thetas_full[i],
                                      max_items=max_items, target_se=target_se,
                                      k_randomesque=k_randomesque, variant="random")
        if not df_adapt.empty: rows.append(df_adapt)
        if not df_rand.empty:  rows.append(df_rand)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

if __name__ == "__main__":
    data = pd.read_csv("test.csv")
    data = data.iloc[:, 1:]
    data = data.to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_withneg1  = torch.tensor(data, dtype=torch.float, device=device)
    data_idtor = (data_withneg1 != -1).to(float)
    data_with0 = data_withneg1 * data_idtor # -1 -> 0
    # Ground truth proxy from your full fit:
    ckpt = torch.load("thetas.pt", map_location=device)
    # if saved as a raw tensor:
    thetas = ckpt if isinstance(ckpt, torch.Tensor) else ckpt["thetas"]
    thetas = thetas.to(torch.float32).to(device)
    theta_star = thetas.detach()
    zs = torch.load("zs.pt", map_location=device)
    zs = zs.to(torch.float32).to(device)
    zs = zs.detach()
    
    # Build and run CAT for everyone (tweak max_items / target_se as needed)
    df_cat = run_cat_population(
        data_with0=data_with0,            # 0/1 (missing also 0, but we gate by observed)
        observed_idtor=data_idtor,        # 1 where observed (original != -1)
        thetas_full=theta_star,           # full fit theta* per person
        zs=zs,                            # item easiness; difficulty b = -zs
        max_items=30,
        target_se=0.30,
        k_randomesque=3
    )   

    # Aggregate to learning curves (mean across persons at each step)
    if not df_cat.empty:
        curve = (
            df_cat
            .groupby(["variant", "step"], as_index=False)
            .agg(
                mean_se   = ("theta_se", "mean"),
                rmse      = ("rmse_vs_full", "mean"),
                mean_bias = ("bias_vs_full", "mean"),
                n_persons = ("pid", "nunique")
            )
            .sort_values(["variant", "step"])
        )
        print(curve.head())
        print(curve.tail())

        curve.to_csv("cat_learning_curves.csv", index=False)
