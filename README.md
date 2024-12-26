# RAMDA: Regularized (Adaptive) Momentumized Dual Averaging for Training Structured Neural Networks

This repository contains a pytorch implementation of the Regularized Modernized Dual Averaging (RMDA) algorithm and the Regularized Adaptive Modernized Dual Averaging (RAMDA) algorithm for training structred neural network models.
Additionally, we have included [ProxGen](https://proceedings.neurips.cc/paper/2021/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html) (using AdamW as the base algorithm and called ProxAdamW in our package) and [ProxSGD](https://openreview.net/forum?id=HygpthEtvr) (with unit stepsize so that structures can be obtained) in this package.
Details of RAMDA and RMDA can be found in the following papers:

> (RAMDA) [NeurIPS 2024] Zih-Syuan Huang, Ching-pei Lee, [*Regularized Adaptive Momentum Dual Averaging with an Efficient
Inexact Subproblem Solver for Training Structured Neural Networks*](https://arxiv.org/abs/2403.14398).

> (RMDA) [ICLR 2022] Zih-Syuan Huang, Ching-pei Lee, [*Training Structured Neural Networks Through Manifold Identification and Variance Reduction*](https://arxiv.org/abs/2112.02612).


Based on the evidence presented in the papers above, we recommend that users try our RAMDA and RMDA first when training structured neural network models.

When provided with a regularizer and its corresponding proximal operator and subproblem solver, these algorithms can train a neural network model that conforms with the structure induced by the regularizer.
In this repository, we include the Group-LASSO norm and the nuclear norm as illustrating examples of regularizers, but users can replace them with any other regularizer. To do this, you will need to add the desired proximal operator in prox_fns.py and the corresponding solver function (e.g., pgd_solver_nuclear_norm) in solvers.py

## Getting started

You will need to install torch, torchvision.

```
pip install -r requirements.txt
```

## Contact

If you have any questions, please feel free to contact Zih-Syuan Huang at r11922210@csie.ntu.edu.tw.
