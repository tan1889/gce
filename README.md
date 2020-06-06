# Greedy Convex Ensemble

This project contains the source code for the following paper. Please quote it if you use the code.

*Tan Nguyen, Nan Ye, Peter Bartlett. "Greedy Convex Ensemble." The 29th International Joint Conference on Artificial Intelligence (IJCAI 2020).*

## Pre-requisites

The code is written in Python 3. In order to run the code, PyTorch, XGBoost, Scikit-learn need to be installed. 
Refer to pytorch.org for how to install PyTorch. XGBoost and Scikit-learn can be installed as follows.

    pip install xgboost
    pip install sklearn

It is recommended to run GCE on a computer with NVIDIA GPU. We used a GTX 1080 Ti GPU card in our experiments.

## Executing The Experiments 

Execute exp_*.py to run the experiments. 
For example, the following command will run all Greedy Convex Ensemble (GCE) experiments on all datasets, 
including hyper-parameters tuning.

    python exp_gce.py 

Similarly, `python exp_xgboost.py`, `python exp_rforest.py`, `python exp_nn.py`, `python exp_ngce.py` 
will run experiments using XGBoost (XGB), Random Forest (RF), Neural Network (NN), Non-greedy Convex Ensemble (NGCE). 

## Executing an Individual Case

Use run_*.py to run an individual case with specific algorithm, dataset, and parameters. 
`run_gce_nn.py` is for execution of GCE, NGCE, and NN. 
XGB and RF use different execution scripts: `run_xgboost.py` and `run_rforest.py`.
Please see the comments in these scripts for the meaning and possible values of each parameter.

The following command will train the model with default value of parameters.

    python run_gce_nn.py

The following command will train the model with some parameters altered from the default value. 
It is possible to alter any/all parameters this way.

    python run_gce_nn.py --dataset msd --algorithm pfw --model NN2 --epoch 5000 --weight-decay 0.0001 --loss l2 --activation relu

Help on the parameters of run_*.py is available using the following commands.

    python run_nn_gce.py --help 
    python run_xgboost.py --help
    python run_rforest.py --help
