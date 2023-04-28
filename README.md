Prerequisites: Tensorflow and basic libraries (like numpy, matplotlib). 

# Single run
Running agent with optimal settings:

    python policybased.py --bootstrapping --baselinesubtraction

Leaving out one or more flags will cause the algorithm to run without bootstrapping and baseline subtraction, respectively.

# Reproducing experiments
E.g. running experiment 1:

    python experiments.py 1

Add other experiments as numbers with whitespace in between. Order not important. The list of experiments are:
1. TODO