import matplotlib.pyplot as plt
import numpy as np, scipy.stats as st
import torch
from helper.loadFile import *
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.linear_model import lm
from toolbox_02450 import draw_neural_net, train_neural_net

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
    model = lm.LogisticRegression()
    model.fit(X_train, y_train)
    
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
    return np.power(y_test-X_test @ w[0].T,2).mean(axis=0), np.power(y_test-X_test @ w[0].T,2)

def ann(x_train, y_train, x_test, y_test, model):
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000

    loss_fn = torch.nn.BCELoss() # notice how this is now a mean-squared-error loss

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

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


    return error_rate
# Pull CHD column from dataset
y = np.ndarray.transpose(np.asarray([[int(num) for num in doc.col_values(10, 1, 463)]]))
# Pull chosen attributes from dataset
new_X = old_X[:, [0,1,2]]
N, M = X.shape

# Define K1, K2, and S
K1 = 5  # Number of outer cross-validation folds
K2 = 5 # Number of inner cross-validation folds
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
                                error_rate = ann(X_train2, y_train2, X_test2, y_test2, ann_model1)
                                ann1_val_err.append(error_rate)
                            case 1:
                                error_rate = ann(X_train2, y_train2, X_test2, y_test2, ann_model2)
                                ann2_val_err.append(error_rate)
                            case 2:
                                error_rate = ann(X_train2, y_train2, X_test2, y_test2, ann_model3)
                                ann3_val_err.append(error_rate)
                            case 3:
                                error_rate = ann(X_train2, y_train2, X_test2, y_test2, ann_model4)
                                ann4_val_err.append(error_rate)
                            case 4:
                                error_rate = ann(X_train2, y_train2, X_test2, y_test2, ann_model5)
                                ann5_val_err.append(error_rate)

                case 1:
                    pass
                    min_err = 100000
                    lamb = 0
                    for i in range(5):
                        temp = 0
                        match i:
                            case 0:
                                mse, _ = reg(X_train1, y_train1, X_test1, y_test1, -2)
                                reg1_val_err.append(mse)
                            case 1:
                                mse, _ = reg(X_train1, y_train1, X_test1, y_test1, -1)
                                reg2_val_err.append(mse)
                            case 2:
                                mse, _ = reg(X_train1, y_train1, X_test1, y_test1, 0)
                                reg3_val_err.append(mse)                            
                            case 3:
                                mse, _ = reg(X_train1, y_train1, X_test1, y_test1, 1)

                                reg4_val_err.append(mse)                            
                            case 4:
                                mse, _ = reg(X_train1, y_train1, X_test1, y_test1, 2)
                                reg5_val_err.append(mse)                    
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
        best_error_rate = (ann(X_train1, y_train1, X_test1, y_test1, ann_model1))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
    elif ann_m_model == ann2_gen_error:
        print("ann2")
        ann_model2 = setupAnn(2)
        best_error_rate = (ann(X_train1, y_train1, X_test1, y_test1, ann_model2))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
        
    elif ann_m_model == ann3_gen_error:
        print("ann3")
        ann_model3 = setupAnn(3)
        best_error_rate = (ann(X_train1, y_train1, X_test1, y_test1, ann_model3))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
        
    elif ann_m_model == ann4_gen_error:
        print("ann4")
        ann_model4 = setupAnn(4)
        best_error_rate = (ann(X_train1, y_train1, X_test1, y_test1, ann_model4))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))
        
    elif ann_m_model == ann5_gen_error:
        print("ann5")
        ann_model5 = setupAnn(5)
        best_error_rate = (ann(X_train1, y_train1, X_test1, y_test1, ann_model5))
        print("best test error for i = " + str(i+1) + "is: " + str(best_error_rate))

    """
    if reg_m_model == reg1_gen_error:
        print("reg1")
        best_test_error = (reg(X_train1, y_train1, X_test1, y_test1, -2))
        print("best test error for i = " + str(i+1) + "is: " + str(best_test_error))
    elif reg_m_model == reg2_gen_error:
        print("reg2")
        best_test_error = (reg(X_train1, y_train1, X_test1, y_test1, -1))
        print("best test error for i = " + str(i+1) + "is: " + str(best_test_error))
    elif reg_m_model == reg3_gen_error:
        print("reg3")
        best_test_error = (reg(X_train1, y_train1, X_test1, y_test1, 0))
        print("best test error for i = " + str(i+1) + "is: " + str(best_test_error))
    elif reg_m_model == reg4_gen_error:
        print("reg4")
        best_test_error = (reg(X_train1, y_train1, X_test1, y_test1, 1))
        print("best test error for i = " + str(i+1) + "is: " + str(best_test_error))
    elif reg_m_model == reg5_gen_error:
        print("reg5")
        best_test_error = (reg(X_train1, y_train1, X_test1, y_test1, 2))
        print("best test error for i = " + str(i+1) + "is: " + str(best_test_error))
    """

    print("baseline")
    y_train_mean = y_train1.mean()
    mse = np.mean(np.square(y_test1 - y_train_mean))
    best_test_error = mse
    print("best test error for i = " + str(i + 1) + "is: " + str(best_test_error))

    print("!!!! LOOP i: " + str(i+1) + "DONE !!!!")


# Statistical analysis of the different models Regression part B, 3
"""
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

ann_test = setupAnn(4)

_, zAnnA = (ann(X_train, y_train, X_test, y_test, ann_test))

_, zRegA = reg(X_train, y_train, X_test, y_test, 4)

y_train_mean = y_train.mean()
zBaseA = np.square(y_test - y_train_mean)

alpha = 0.05


# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zAnnReg = zAnnA - zRegA

zAnnBase = zAnnA - zBaseA

zRegBase = zRegA - zBaseA

CI_AR = st.t.interval(1-alpha, len(zAnnReg)-1, loc=np.mean(zAnnReg), scale=st.sem(zAnnReg))  # Confidence interval
p_AR = 2*st.t.cdf( -np.abs( np.mean(zAnnReg) )/st.sem(zAnnReg), df=len(zAnnReg)-1)  # p-value

CI_AB = st.t.interval(1-alpha, len(zAnnBase)-1, loc=np.mean(zAnnBase), scale=st.sem(zAnnBase))  # Confidence interval
p_AB = 2*st.t.cdf( -np.abs( np.mean(zAnnBase) )/st.sem(zAnnBase), df=len(zAnnBase)-1)  # p-value

CI_RB = st.t.interval(1-alpha, len(zRegBase)-1, loc=np.mean(zRegBase), scale=st.sem(zRegBase))  # Confidence interval
p_RB = 2*st.t.cdf( -np.abs( np.mean(zRegBase) )/st.sem(zRegBase), df=len(zRegBase)-1)  # p-value

print("CI_AR = " + str(CI_AR) + "p-value= " + str(p_AR))
print("CI_AB = " + str(CI_AB) + "p-value= " + str(p_AB))
print("CI_RB = " + str(CI_RB) + "p-value= " + str(p_RB))
"""




