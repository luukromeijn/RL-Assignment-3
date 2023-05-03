Prerequisites: Tensorflow and basic libraries (like numpy, matplotlib). 

# Single run
Running agent with optimal settings:

    python policybased.py --bootstrapping --baselinesubtraction

Leaving out one or more flags will cause the algorithm to run without bootstrapping and baseline subtraction, respectively.

# Reproducing experiments
E.g. running experiment 1:

    python experiments.py 1

Add other experiments as numbers with whitespace in between. Order not important. The list of experiments are:
1. Hyperparameter search: try and plot learning curves for learning rate $\alpha\in\{0.005, 0.01, 0.05\}$ in combination with entropy regularization $\eta\in\{0, 0.1, 0.2\}$ for the algorithm with bootstrapping and baseline subtraction.
2. Algorithm comparison: compare settings with/without bootstrapping/baseline subtraction.

When no experiment numbers are provided, the script will run all experiments one after another.