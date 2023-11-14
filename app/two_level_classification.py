import matplotlib.pyplot as plt
import numpy as np, scipy.stats as st
import torch
from helper.loadFile import *
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import draw_neural_net, train_neural_net, mcnemar

def setupAnn(hidden_units):
    n_hidden_units = hidden_units      # number of hidden units
    

    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    torch.nn.Sigmoid()# no final tranfer function, i.e. "linear output"
                    )
    return model

def reg(X_train, y_train, X_test, y_test, lambda1):
    
    
    X_train = X_train
    
    y_train = y_train.tolist()
    y_train = list(map(lambda x: x[0], y_train))
    y_train = np.array(y_train)
    
    X_test = X_test

    y_test = y_test.tolist()
    y_test = list(map(lambda x: x[0], y_test))
    y_test = np.array(y_test) 
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    w = []

    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    mdl = LogisticRegression(penalty='l2', C=1/np.power(10., lambda1))

    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T

    train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

    #w_est = mdl.coef_[0]
    #coefficient_norm = np.sqrt(np.sum(w_est**2))
    
    return test_error_rate, y_test_est

def ann(x_train, y_train, x_test, y_test, model):
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000

    loss_fn = torch.nn.BCELoss() # notice how this is now a mean-squared-error loss

    # Extract training and test set for current CV fold, convert to tensors
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

    print('\n\tBest loss: {}\n'.format(final_loss))

    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8)

    # Determine errors and errors
    y_test = y_test.type(dtype=torch.uint8)

    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()


    return error_rate, y_test_est

def baseline_seperator(y_test, y_train_mean):
    y_predict = []
    if y_train_mean > 0.5:
        for x in y_test:
            y_predict.append(1)
    else:
        for x in y_test:
            y_predict.append(0)

    return np.ndarray.transpose(np.asarray([y_predict]))

# Pull CHD column from dataset
y = np.ndarray.transpose(np.asarray([[int(num) for num in doc.col_values(10, 1, 463)]]))
# Pull chosen attributes from dataset
new_X = old_X[:, [0,1,2]]
N, M = new_X.shape

# Define K1, K2, and S
K1 = 2  # Number of outer cross-validation folds
K2 = 2 # Number of inner cross-validation folds
S = 3   # Number of different models


ann1_val_err = []
ann2_val_err = []
ann3_val_err = []
ann4_val_err = []
ann5_val_err = []

reg1_val_err = []
reg2_val_err = []
reg3_val_err = []
reg4_val_err = []
reg5_val_err = []


ann_errors = []
reg_errors = []
baseline_errors = []

ann_gen_error = []
reg_gen_error = []
baseline_gen_error = []

best_test_error = []

X_train2 = 0
y_train2 = 0
X_test2 = 0
y_test2 = 0 

# Create the outer cross-validation splits
CV1 = model_selection.KFold(n_splits=K1, shuffle=True)

for (i, (train_index, test_index)) in enumerate(CV1.split(new_X,y)):

    X_train1 = new_X[train_index,:] # D_par
    y_train1 = y[train_index] # D_par
    X_test1 = new_X[test_index,:] # D_test
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
                    ann_model1 = setupAnn(1)
                    ann_model2 = setupAnn(2)
                    ann_model3 = setupAnn(3)
                    ann_model4 = setupAnn(4)
                    ann_model5 = setupAnn(5)

                    min_err = 100000
                    h = 0
                    for i in range(5):
                        temp = 0
                        match i:
                            case 0:
                                error_rate, _ = ann(X_train2, y_train2, X_test2, y_test2, ann_model1)
                                ann1_val_err.append(error_rate)
                            case 1:
                                error_rate, _ = ann(X_train2, y_train2, X_test2, y_test2, ann_model2)
                                ann2_val_err.append(error_rate)
                            case 2:
                                error_rate, _ = ann(X_train2, y_train2, X_test2, y_test2, ann_model3)
                                ann3_val_err.append(error_rate)
                            case 3:
                                error_rate, _ = ann(X_train2, y_train2, X_test2, y_test2, ann_model4)
                                ann4_val_err.append(error_rate)
                            case 4:
                                error_rate, _ = ann(X_train2, y_train2, X_test2, y_test2, ann_model5)
                                ann5_val_err.append(error_rate)

                case 1:
                    min_err = 100000
                    lamb = 0
                    for i in range(5):
                        temp = 0
                        match i:
                            case 0:
                                error_rate, _ = reg(X_train2, y_train2, X_test2, y_test2, -2)
                                reg1_val_err.append(error_rate)
                            case 1:
                                error_rate, _ = reg(X_train2, y_train2, X_test2, y_test2, -1)
                                reg2_val_err.append(error_rate)
                            case 2:
                                error_rate, _ = reg(X_train2, y_train2, X_test2, y_test2, 0)
                                reg3_val_err.append(error_rate)                            
                            case 3:
                                error_rate, _ = reg(X_train2, y_train2, X_test2, y_test2, 1)
                                reg4_val_err.append(error_rate)                            
                            case 4:
                                error_rate, _ = reg(X_train2, y_train2, X_test2, y_test2, 2)
                                reg5_val_err.append(error_rate)                    
                case 2:
                    pass

    #ann_gen_error = sum(len((X_test2[j])+len(y_test2[j]))/(len(X_train1[i])+len(y_train1[i]))*ann_errors for j in range(K2))
    ann1_gen_error = np.mean(ann1_val_err)
    ann2_gen_error = np.mean(ann2_val_err)
    ann3_gen_error = np.mean(ann3_val_err)
    ann4_gen_error = np.mean(ann4_val_err)
    ann5_gen_error = np.mean(ann5_val_err)

    
    #reg_gen_error = sum(len((X_test2[j])+len(y_test2[j]))/(len(X_train1[i])+len(y_train1[i]))*reg_errors for j in range(K2))
    reg1_gen_error = np.mean(reg1_val_err)
    reg2_gen_error = np.mean(reg2_val_err)
    reg3_gen_error = np.mean(reg3_val_err)
    reg4_gen_error = np.mean(reg4_val_err)
    reg5_gen_error = np.mean(reg5_val_err)


    #baseline_gen_error = sum(len((X_test2[j])+len(y_test2[j]))/(len(X_train1[i])+len(y_train1[i]))*baseline_errors for j in range(K2))

    

    ann_m_model = min(ann1_gen_error, ann2_gen_error, ann3_gen_error, ann4_gen_error, ann5_gen_error)

    reg_m_model = min(reg1_gen_error, reg2_gen_error, reg3_gen_error, reg4_gen_error, reg5_gen_error)
    
    if ann_m_model == ann1_gen_error:
        print("ann1")
        ann_model1 = setupAnn(1)       
        best_error_rate, _ = (ann(X_train1, y_train1, X_test1, y_test1, ann_model1))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
    elif ann_m_model == ann2_gen_error:
        print("ann2")
        ann_model2 = setupAnn(2)
        best_error_rate, _ = (ann(X_train1, y_train1, X_test1, y_test1, ann_model2))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
        
    elif ann_m_model == ann3_gen_error:
        print("ann3")
        ann_model3 = setupAnn(3)
        best_error_rate, _ = (ann(X_train1, y_train1, X_test1, y_test1, ann_model3))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
        
    elif ann_m_model == ann4_gen_error:
        print("ann4")
        ann_model4 = setupAnn(4)
        best_error_rate, _ = (ann(X_train1, y_train1, X_test1, y_test1, ann_model4))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
        
    elif ann_m_model == ann5_gen_error:
        print("ann5")
        ann_model5 = setupAnn(5)
        best_error_rate, _ = (ann(X_train1, y_train1, X_test1, y_test1, ann_model5))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))

    
    if reg_m_model == reg1_gen_error:
        print("reg1")
        best_error_rate, _ = (reg(X_train1, y_train1, X_test1, y_test1, -2))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
    elif reg_m_model == reg2_gen_error:
        print("reg2")
        best_error_rate, _ = (reg(X_train1, y_train1, X_test1, y_test1, -1))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
    elif reg_m_model == reg3_gen_error:
        print("reg3")
        best_error_rate, _ = (reg(X_train1, y_train1, X_test1, y_test1, 0))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
    elif reg_m_model == reg4_gen_error:
        print("reg4")
        best_error_rate, _ = (reg(X_train1, y_train1, X_test1, y_test1, 1))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
    elif reg_m_model == reg5_gen_error:
        print("reg5")
        best_error_rate, _ = (reg(X_train1, y_train1, X_test1, y_test1, 2))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
    

    print("baseline")
    y_train_mean = np.mean(y_train1, 0)
    y_predict = baseline_seperator(y_test1, y_train_mean)
    
    best_error_rate = np.sum(y_predict != y_test1) / len(y_test1)
    print("best test error for i = " + str(i + 1) + "is: " + str(best_error_rate))

    print("!!!! LOOP i: " + str(i+1) + "DONE !!!!")


# Statistical analysis of the different models Regression part B, 3

X_train, X_test, y_train, y_test = model_selection.train_test_split(new_X, y, test_size=0.2)

ann_test = setupAnn(4)

_, yhat_ann = ann(X_train, y_train, X_test, y_test, ann_test)

_, yhat_reg = reg(X_train, y_train, X_test, y_test, 4)

yhat_base = baseline_seperator(y_test, np.mean(y_train, 0))

y_true = y_test

alpha = 0.05

yhat_ann = yhat_ann.numpy()
yhat_reg = np.ndarray.transpose(np.asarray([yhat_reg]))

[thetahat_ann_reg, CI_ANN_REG, p_ann_reg] = mcnemar(y_true, yhat_ann[:], yhat_reg[:], alpha=alpha)
[thetahat_reg_base, CI_REG_BASE, p_reg_base] = mcnemar(y_true, yhat_reg[:], yhat_base[:], alpha=alpha)
[thetahat_ann_base, CI_ANN_BASE, p_ann_base] = mcnemar(y_true, yhat_ann[:], yhat_base[:], alpha=alpha)

print("theta_ann_reg = theta_A-theta_B point estimate", thetahat_ann_reg, " CI: ", CI_ANN_REG, "p-value", p_ann_reg)
print("theta_reg_base = theta_A-theta_B point estimate", thetahat_reg_base, " CI: ", CI_REG_BASE, "p-value", p_reg_base)
print("theta_ann_base = theta_A-theta_B point estimate", thetahat_ann_base, " CI: ", CI_ANN_BASE, "p-value", p_ann_base)




