import numpy as np
import torch
import pickle
import os
import gc
from torch.distributions import Bernoulli
from torch.optim import LBFGS
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict

from tueplots import bundles
bundles.icml2024()

from torchmetrics import AUROC
auroc = AUROC(task="binary")

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)

device = "cuda:0"

def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
    return parameters

def compute_auc(probs, data, train_idtor, test_idtor):
    train_probs = probs[train_idtor.bool()]
    test_probs = probs[test_idtor.bool()]
    train_labels = data[train_idtor.bool()]
    test_labels = data[test_idtor.bool()]
    
    train_auc = auroc(train_probs, train_labels)
    test_auc = auroc(test_probs, test_labels)
    print(f"train auc: {train_auc}")
    print(f"test auc: {test_auc}")
    
    return train_auc, test_auc

def compute_cttcorr(probs, data, train_idtor, test_idtor):
    train_probs  = probs.clone()
    test_probs   = probs.clone()
    train_labels = data.clone()
    test_labels  = data.clone()

    train_mask = ~train_idtor.bool()
    train_probs[train_mask]  = float('nan')
    train_labels[train_mask] = float('nan')

    test_mask = ~test_idtor.bool()
    test_probs[test_mask]   = float('nan')
    test_labels[test_mask]  = float('nan')
    
    train_prob_ctt = torch.nanmean(train_probs, dim=1).detach().cpu().numpy()
    train_label_ctt = torch.nanmean(train_labels, dim=1).detach().cpu().numpy()
    train_mask = ~np.isnan(train_prob_ctt) & ~np.isnan(train_label_ctt)
    train_cttcorr = pearsonr(train_prob_ctt[train_mask], train_label_ctt[train_mask]).statistic
    
    test_prob_ctt = torch.nanmean(test_probs, dim=1).detach().cpu().numpy()
    test_label_ctt = torch.nanmean(test_labels, dim=1).detach().cpu().numpy()
    test_mask = ~np.isnan(test_prob_ctt) & ~np.isnan(test_label_ctt)
    test_cttcorr = pearsonr(test_prob_ctt[test_mask], test_label_ctt[test_mask]).statistic
    
    print(f"train cttcorr: {train_cttcorr}")
    print(f"test cttcorr: {test_cttcorr}")

    return train_cttcorr, test_cttcorr

if __name__ == "__main__":
    with open(f"../data/resmat.pkl", "rb") as f:
        results = pickle.load(f)
        
    # data_withnan, missing=nan
    # data_withneg1, missing=-1
    # data_with0, missing=0
    data_withnan = torch.tensor(results.values, dtype=torch.float, device=device)
    data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
    data_idtor = (data_withneg1 != -1).to(float)
    data_with0 = data_withneg1 * data_idtor # -1 -> 0
    n_test_takers, n_items = data_with0.shape
    scenarios = results.columns.get_level_values("scenario").unique()

    # save dict
    metric_results = defaultdict(dict)

    vis_resmat_dir = "../result/visualize_resmat"
    os.makedirs(vis_resmat_dir, exist_ok=True)

    # overall stats
    print("Number of test takers:", results.shape[0])
    print("Number of items:", results.shape[1])
    print("Number of scenarios:", results.columns.get_level_values("scenario").nunique())

    
    # data_idtor = train_idtor + test_idtor
    # apply random train/test mask to the matrix, and ensure no one row or column is fully masked
    valid_condition = False
    trial = 0
    while not valid_condition:
        train_idtor = torch.bernoulli(data_idtor * 0.8).int()
        test_idtor = data_idtor - train_idtor
        valid_condition = (train_idtor.sum(axis=1) != 0).all() and (train_idtor.sum(axis=0) != 0).all()
        print(f"trial {trial} valid condition: {valid_condition}")
        trial += 1

    # fit z
    B = 50000
    optimized_zs = []
    thetas_nuisance = torch.randn(150, n_test_takers, device=device)
    for i in tqdm(range(0, n_items, B)):
        data_batch = data_with0[:, i:i+B]
        train_idtor_batch = train_idtor[:, i:i+B]
        current_B = data_batch.shape[1]
        z_i = torch.randn(current_B, requires_grad=True, device=device)
        optim_z_i = LBFGS([z_i], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_z_i():
            optim_z_i.zero_grad()
            probs = torch.sigmoid(thetas_nuisance[:, :, None] + z_i[None, None, :])
            loss = -(Bernoulli(probs=probs).log_prob(data_batch)*train_idtor_batch).mean()
            loss.backward()
            return loss
        z_i_optimized = trainer([z_i], optim_z_i, closure_z_i)[0].detach()
        optimized_zs.append(z_i_optimized)
    zs = torch.cat(optimized_zs)

    # fit theta
    thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
    optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_theta():
        optim_theta.zero_grad()
        probs = torch.sigmoid(thetas[:, None] + zs[None, :])
        loss = -(Bernoulli(probs=probs).log_prob(data_with0)*train_idtor).mean()
        loss.backward()
        return loss
    thetas = trainer([thetas], optim_theta, closure_theta)[0]

    # calculate metrics
    probs = torch.sigmoid(thetas[:, None] + zs[None, :])

    train_auc, test_auc = compute_auc(probs, data_with0, train_idtor, test_idtor)
    metric_results["combined_data"]["train_auc"] = train_auc.item()
    metric_results["combined_data"]["test_auc"] = test_auc.item()

    train_cttcorr, test_cttcorr = compute_cttcorr(probs, data_with0, train_idtor, test_idtor)
    metric_results["combined_data"]["train_cttcorr"] = train_cttcorr.item()
    metric_results["combined_data"]["test_cttcorr"] = test_cttcorr.item()

    del optim_theta, thetas, z_i, thetas_nuisance, optim_z_i
    gc.collect()
    torch.cuda.empty_cache()
