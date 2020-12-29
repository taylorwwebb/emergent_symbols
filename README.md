# Emergent Symbols through Binding in External Memory

Code for the paper 'Emergent Symbols through Binding in External Memory'.

The `./scripts` directory contains scripts to reproduce the primary results presented in the paper. There is one script per task. Each script will train and evaluate 10 networks on all generalization regimes for a particular task. Each script requires two arguments:
1. Architecture name (according to the names of the files in the `./models` directory, e.g. `ESBN`, `LSTM`, etc.)
2. Device index (default is `0` on machines with only one device)

For example, to reproduce the results for the ESBN architecture on the same/different discrimination task, run the following command:
```
./scripts/same_diff.sh ESBN 0
```
These scripts use the default values for the simulations reported in the paper (temporal context normalization, convolutional encoder, learning rate, number of training epochs). To reproduce some of the other results in the paper (such as for the models that needed to be trained for longer, or the experiments involving alternative encoder architectures), some of these values will need to be changed by modifying the scripts accordingly.

## Prerequisites

- Python 3
- [NumPy](https://numpy.org/)
- [colorlog](https://github.com/borntyping/python-colorlog)
- [PIL](https://pillow.readthedocs.io/en/3.1.x/installation.html)
- [PyTorch](https://pytorch.org/)

## Authorship

All code was written by [Taylor Webb](https://github.com/taylorwwebb). 
