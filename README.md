# Introduction
This Repo contains the **official implementation** of the following paper:

|Venue|Method|Paper Title|
|----|-----|-----|
||FedHSCC|Mitigating Dimensional Collapse in Heterogeneous Federated Learning via Hierarchical Sparse Covariance Constraint|
and unofficial implementation of the following papers:

|Venue|Method|Paper Title|
|----|-----|-----|
|AISTATS'17|FedAvg|[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)|
|ArXiv'19|FedAvgM|[Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335)|
|MLSys'20|FedProx|[Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)|
|NeurIPS'20|FedNova|[Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481)|
|CVPR'21|MOON|[Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257)|
|ICLR'21|FedAdagrad/Yogi/Adam|[Adaptive Federated Optimization](https://openreview.net/forum?id=LkFG3lB13U5)|
|KDD'21|FedRS|[FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data](http://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf)|
|ICML'22|FedLogitCal|[Federated Learning with Label Distribution Skew via Logits Calibration](https://arxiv.org/abs/2209.00189)|
|ICML'22/ECCV'22|FedSAM|[Generalized Federated Learning via Sharpness Aware Minimization](https://arxiv.org/pdf/2206.02618.pdf)/[Improving Generalization in Federated Learning by Seeking Flat Minima](https://arxiv.org/abs/2203.11834)|
|ICLR'23|FedExp|[FedExP: Speeding up Federated Averaging via Extrapolation](https://openreview.net/forum?id=IPrzNbddXV)|
|ICLR'23|FedDecorr|[Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning](https://arxiv.org/abs/2210.00226)|






# Running Instructions
Shell scripts to reproduce experimental results in our paper are under "run\_scripts" folder. Simply changing the "ALPHA" variable to run under different degree of heterogeneity.

Here are commands that replicate our results:

FedAvg on CIFAR10:
```
bash run_scripts/cifar10_fedavg.sh
```

FedAvg + FedHSCC on CIFAR10:
```
bash run_scripts/cifar10_fedavg_fedhscc.sh
```

Experiments on other methods (FedAvgM, FedProx, MOON) and other datasets (CIFAR100) follow the similar manner.





# Contact
Jianhang Feng (222409252001@zust.edu.cn)

# Acknowledgement
Some of our code is borrowed following projects: [MOON](https://github.com/QinbinLi/MOON), [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench), [SAM(Pytorch)](https://github.com/davda54/sam)



## Modifications from Upstream
This project is  based on Feddecorr by Yujun Shi.
Original repository: https://github.com/bytedance/FedDecorr.git
Original license: LICENSE
We have made the following significant modifications to extend the framework with **HSCC (Hierarchical Sparse Covariance Constraint)** regularization and performance measurement tools.

### 📝 Summary of Changes

| File / Module | Modifications                                                                                                                                     |
| :--- |:--------------------------------------------------------------------------------------------------------------------------------------------------|
| `loss.py` | Added a new class `FedHSCCLoss` implementing hierarchical sparse covariance constraint. Translated Chinese comments to English.                   |
| **All modified approaches** (`fedavg.py`, `fedprox.py`, `moon.py`, …) | Integrated `FedHSCCLoss` into the local training loop of each approach; added per‑round timing functionality (`--measure_time`) where applicable. |
| `main.py` | Added `--measure_time` command‑line argument; removed data distribution visualization code (matplotlib dependency removed).                       |
