Last login: Tue Sep 30 15:42:38 on ttys028
jamesmingliangang@Mac gstar inivdiusal porject % ssh a100azf
Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-141-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

 System information as of Tue Sep 30 08:00:25 AM UTC 2025

  System load:             0.4
  Usage of /:              84.5% of 1.28TB
  Memory usage:            38%
  Swap usage:              68%
  Processes:               591
  Users logged in:         1
  IPv4 address for enp1s0: 95.179.200.201
  IPv6 address for enp1s0: 2001:19f0:7402:ad3:5400:4ff:fef4:a873

  => There are 2 zombie processes.

 * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s
   just raised the bar for easy, resilient and secure K8s cluster deployment.

   https://ubuntu.com/engage/secure-kubernetes-at-the-edge

Expanded Security Maintenance for Applications is not enabled.

146 updates can be applied immediately.
6 of these updates are standard security updates.
To see these additional updates run: apt list --upgradable

27 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm


The list of available updates is more than a week old.
To check for new updates run: sudo apt update

2 updates could not be installed automatically. For more details,
see /var/log/unattended-upgrades/unattended-upgrades.log

*** System restart required ***
Last login: Tue Sep 30 06:45:35 2025 from 203.117.22.140
root@test-bha:~# cd mlexp/
root@test-bha:~/mlexp# ls
CRAG-RAG-Selection                           data                logreg-svrg-main      optimizer_comparison.png  RAG     svrg_istropic_guassian  trained
crag_task_1_and_2_dev_sample30_v4.jsonl.bz2  GStar-Assignment-1  logreg-svrg-main.zip  output.png                reeval  tmp                     wandb
root@test-bha:~/mlexp# cd CRAG-RAG-Selection/
root@test-bha:~/mlexp/CRAG-RAG-Selection# source .venv/bin/activate
(CRAG-RAG-Selection) root@test-bha:~/mlexp/CRAG-RAG-Selection# uv pip install "chonkie[semantic]"
Resolved 26 packages in 281ms
Prepared 6 packages in 75ms
Installed 6 packages in 9ms
 + chonkie==1.3.1
 + markdown-it-py==4.0.0
 + mdurl==0.1.2
 + model2vec==0.6.0
 + pygments==2.19.2
 + rich==14.1.0
(CRAG-RAG-Selection) root@test-bha:~/mlexp/CRAG-RAG-Selection# ls
api_responses  CODE_OF_CONDUCT.md  data  example_data  local_evaluation.py  models   README.md         tokenizer
calibration    CONTRIBUTING.md     docs  LICENSE       mock_api             prompts  requirements.txt  utils
(CRAG-RAG-Selection) root@test-bha:~/mlexp/CRAG-RAG-Selection# tmux
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# cd models
root@test-bha:~/mlexp/CRAG-RAG-Selection/models# ls
dummy_model.py  __pycache__                      rag_llama_baseline.py  sentence-transformers  utils.py
meta-llama      rag_knowledge_graph_baseline.py  README.md              user_config.py         vanilla_llama_baseline.py
root@test-bha:~/mlexp/CRAG-RAG-Selection/models# vi rag_llama_baseline.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection/models# cd ..
root@test-bha:~/mlexp/CRAG-RAG-Selection# ls
api_responses  CODE_OF_CONDUCT.md  data  example_data  local_evaluation.py  models   README.md         tokenizer
calibration    CONTRIBUTING.md     docs  LICENSE       mock_api             prompts  requirements.txt  utils
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# ls
api_responses  CODE_OF_CONDUCT.md  data  example_data  local_evaluation.py  models   README.md         tokenizer
calibration    CONTRIBUTING.md     docs  LICENSE       mock_api             prompts  requirements.txt  utils
root@test-bha:~/mlexp/CRAG-RAG-Selection# cd models
root@test-bha:~/mlexp/CRAG-RAG-Selection/models# ls
dummy_model.py  __pycache__                      rag_llama_baseline.py  sentence-transformers  utils.py
meta-llama      rag_knowledge_graph_baseline.py  README.md              user_config.py         vanilla_llama_baseline.py
root@test-bha:~/mlexp/CRAG-RAG-Selection/models# vi rag_llama_baseline.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection/models# cd ..
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# python local_evaluation.py 
  File "/root/mlexp/CRAG-RAG-Selection/local_evaluation.py", line 87
    to match the score
       ^^^^^
SyntaxError: invalid syntax
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# python local_evaluation.py 
Traceback (most recent call last):
  File "/root/mlexp/CRAG-RAG-Selection/local_evaluation.py", line 14, in <module>
    from loguru import logger
ModuleNotFoundError: No module named 'loguru'
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# vi local_evaluation.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection# ls
api_responses  CODE_OF_CONDUCT.md  data  example_data  local_evaluation.py  models   README.md         tokenizer
calibration    CONTRIBUTING.md     docs  LICENSE       mock_api             prompts  requirements.txt  utils
root@test-bha:~/mlexp/CRAG-RAG-Selection# cd calibration/
root@test-bha:~/mlexp/CRAG-RAG-Selection/calibration# ls
calibration.py
root@test-bha:~/mlexp/CRAG-RAG-Selection/calibration# vi calibration.py 
root@test-bha:~/mlexp/CRAG-RAG-Selection/calibration# ls
calibration.py
root@test-bha:~/mlexp/CRAG-RAG-Selection/calibration# vi calibration.py 

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

"calibration.py" 171L, 6548B                                                                                                                                       53,4           4%
