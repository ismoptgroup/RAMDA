# logistic regression on MNIST example (Group-LASSO norm)

This directory contains:
 - model.py: Logistic regression trained on the MNIST dataset.
 - prepare.py: model initializations, download datasets and create folders.
 - optimizer_Linear_on_MNIST.sh: Configuration files to run the experiments.
 - run.sh: Running all the experiments.
 - train.py: Training and evaluating logistic regression on the MNIST dataset.

## Quick Start Guide
1. Modify the path argument in both train.py and prepare.py files to specify the location for storing model initializations, datasets, checkpoints, and logs, if necessary.

2. Run all the experiments

```
bash run.sh
``` 
