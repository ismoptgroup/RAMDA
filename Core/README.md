# Core code for this directory.

This directory contains:
 - optimizer.py: RMDA, RAMDA, ProxSGD, and ProxGen (ProxAdamW) optimizers.
 - prox_fns.py: 1. proximal operators for the group-LASSO-norm regularizer and the nuclear-norm regularizer. 2. Calculating the subproblem objective value.
 - scheduler.py: Stage-wise learning rate scheduler, momentum scheduler and restarting mechanism.
 - solvers.py: Proximal gradient descent (PG) solver for the subproblems with the group-LASSO norm regularization and with the nuclear-norm regularization.
 - group.py: (for the group-LASSO-norm regularization only) Grouping the models into channel-wise, input-wise or unregularized group.

## How to group model weights for structured sparsity?
1. [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): dim is (0,2,3) for the channel-wise grouping and (1,2,3) for the filter-wise grouping.
2. [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html): dim is (0,2) for the channel-wise grouping and (1,2) for the filter-wise grouping.
3. [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html): dim is (0) for the input-wise grouping and dim is (1) for the output-wise grouping.
4. [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html): dim is (0) for the input-wise grouping and dim is (1) for the output-wise grouping.

We leave [BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html), [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html), [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and [Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) layers unregularized. Thus, we don't group these weights.
