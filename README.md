# Learning Balanced and Invariant Representations for Treatment Effect Estimation

## 1 Abstract

Individualized Treatment Effect (ITE) estimation is a fundamental problem in causal inference. 
A major challenge in ITE estimation is \textbf{selection bias}, which arises from the imbalance between the treatment and control group distributions. 
To address this issue, recent neural network–based methods aim to penalize the distributional discrepancy between the two groups, with \textbf{Counterfactual Regression (CFR)} being a representative approach. 
However, CFR and its subsequent variants have been evaluated only on in-distribution (ID) data, i.e., data drawn from the same domain as the training set. 
Consequently, their \textbf{out-of-distribution (OOD) generalizability} remains uncertain, which is crucial since covariate distributions often shift across domains, leading to \textbf{covariate shift}. 
To bridge this gap, we first provide a theoretical analysis of the discrepancy between ID and OOD generalization in CFR. 
Motivated by this analysis, we propose the \textbf{Hierarchical Wasserstein Barycenter Counterfactual Regression (HWBCFR)} framework. 
Specifically, we introduce a \textbf{domain-level barycenter} to capture invariant representations shared across domains, thereby enhancing CFR’s generalization to unseen domains. 
Furthermore, we design a \textbf{group-level barycenter} to align the treatment and control distributions within each domain, mitigating selection bias. 
By alternately optimizing these two barycenters, HWBCFR effectively minimizes the theoretical gap between ID and OOD distributions, leading to improved OOD generalization. 
Empirical results on two real-world datasets demonstrate that HWBCFR consistently outperforms existing baselines.
## 2 Quick Start

Choose a model (e.g., HWBCFR) to run with the following command.

```
    python main.py --lr 0.01 --batchSize 64 --lambda 0.1
```


## 3 Hyper-parameters search range

We tune hyper-parameters according to the following table.

| Hyper-parameter | Explain                                     | Range                                 |
| --------------- | ------------------------------------------- | ------------------------------------- |
| lr              | learning rate                               | \{0.00001, 0.0001, 0.001, 0.01, 0.1\} |
| bs              | batch size of each mini-batch               | \{16, 32, 64, 128\}                   |
| dim_backbone    | the dimensions of representation            | \{32, 64\}                            |
| dim_task        | the dimensions of prediction head           | \{32, 64\}                            |
| alpha           | weight of reconstruction loss               | \{0.000001, 0.00001, 0.0001, 0.001, 0.01\} |
| kappa           | weight of marginal constraint               | \{0.001, 0.01, 0.1, 0.5, 1\}           |
| lambda          | weight proposed OT-based regularization     | \{0.0001, 0.001, 0.01, 0.1, 1\}       |


