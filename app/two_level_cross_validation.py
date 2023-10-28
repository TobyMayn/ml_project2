import matplotlib.pyplot as plt
import numpy as np
import torch
from helper.loadFile import *
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import draw_neural_net, train_neural_net

def setupAnn(hidden_units):
    n_hidden_units = hidden_units      # number of hidden units
    

    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
    return model


def ann(x_train, y_train, x_test, y_test, model):
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000

    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    X_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean

    return mse

# Define K1, K2, and S
K1 = 10  # Number of outer cross-validation folds
K2 = 10 # Number of inner cross-validation folds
S = 3   # Number of different models

# Create the outer cross-validation splits
CV1 = model_selection.KFold(n_splits=K1, shuffle=True)

for (i, (train_index, test_index)) in enumerate(CV1.split(X,y)):
    X_train1 = X[train_index,:] # D_par
    y_train1 = y[train_index] # D_par
    X_test1 = X[test_index,:] # D_test
    y_test1 = y[test_index] # D_test

    CV2 = model_selection.KFold(n_splits=K2, shuffle=True)

    for (j, (train_index, test_index)) in enumerate(CV2.split(X_train1,y_train1)):
        X_train2 = X_train1[train_index,:]
        y_train2 = y_train1[train_index]
        X_test2 = X_train1[test_index,:]
        y_test2 = y_train1[test_index]

        for s in range(S):
            ann_errors = []
            reg_errors = []
            baseline_errors = []
            # Train model on X_train2, y_train2
            # Evaluate model on X_test2, y_test2
            # Save the performance metric
            # Repeat for all models
            match s:
                case 0:
                    for x in range(5):
                        hidden_units = x+1
                        model = setupAnn(hidden_units)
                        ann_errors.append(ann(X_train2, y_train2, X_test2, y_test2, model))

