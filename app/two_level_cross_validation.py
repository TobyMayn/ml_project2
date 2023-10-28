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

def reg(X_train, y_train, X_test, y_test, lambda1):
    M1 = M + 1
    
    X_train = X_train
    
    y_train = y_train.tolist()
    y_train = list(map(lambda x: x[0], y_train))
    y_train = np.array(y_train)
    
    X_test = X_test

    y_test = y_test.tolist()
    y_test = list(map(lambda x: x[0], y_test))
    y_test = np.array(y_test)

    X_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),1)    

    
    w = []

    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    # Compute parameters for current value of lambda and current CV fold
    # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
    lambdaI = np.power(10., lambda1) * np.eye(M1)
    lambdaI[0,0] = 0 # remove bias regularization
    w.append(np.linalg.solve(XtX+lambdaI,Xty).squeeze())
    # Evaluate training and test performance
    return np.power(y_test-X_test @ w[0].T,2).mean(axis=0)

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

y = np.ndarray.transpose(np.asarray([[float(num) for num in doc.col_values(3, 1, 463)]]))

N, M = X.shape

# Define K1, K2, and S
K1 = 5  # Number of outer cross-validation folds
K2 = 5 # Number of inner cross-validation folds
S = 3   # Number of different models
ann_errors = []
reg_errors = []
baseline_errors = []

# Create the outer cross-validation splits
CV1 = model_selection.KFold(n_splits=K1, shuffle=True)

for (i, (train_index, test_index)) in enumerate(CV1.split(X,y)):
    hidden_units = i+1
    lambda1 = i-2
    ann_model = setupAnn(hidden_units)

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
            
            # Train model on X_train2, y_train2
            # Evaluate model on X_test2, y_test2
            # Save the performance metric
            # Repeat for all models
            match s:
                case 0: 
                    ann_errors.append(ann(X_train2, y_train2, X_test2, y_test2, ann_model))
                case 1:
                    reg_errors.append(reg(X_train2, y_train2, X_test2, y_test2, lambda1))
                    
                case 2:
                    #y_train_mean = y_train2.mean()
                    #baseline_errors.append()
                    pass

    

