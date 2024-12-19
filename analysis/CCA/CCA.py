import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from statsmodels.multivariate.cancorr import CanCorr
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import scipy

# a. Get the desired directory to load/save the data
root = tk.Tk()
root.withdraw() 
directory = filedialog.askdirectory()

# b. set up
n_features_in = 11  # input
n_features_out = 3  # output
alpha = 0.1         # Lasso
np.set_printoptions(suppress=True) # suppress scientific notation

# c. load inputs and outputs
# i. INPUTS:

    # height
in_height_file = os.path.normpath(os.path.join(directory,'features\\in_height.csv'))
in_height = np.abs(np.genfromtxt(in_height_file, delimiter=','))

    # weight
in_weight_file = os.path.normpath(os.path.join(directory,'features\\in_weight.csv'))
in_weight = np.abs(np.genfromtxt(in_weight_file, delimiter=','))

    # sex
in_isFemale_file = os.path.normpath(os.path.join(directory,'features\\in_isFemale.csv'))
in_isFemale = np.abs(np.genfromtxt(in_isFemale_file, delimiter=','))

    # vibration test accuracy
in_vbtest_file = os.path.normpath(os.path.join(directory, 'features\\in_vbtest.csv'))
in_vbtest = np.genfromtxt(in_vbtest_file, delimiter=',')

    # proprioception test accuracy
in_proprio_in_file = os.path.normpath(os.path.join(directory,'features\\in_proprio_in.csv'))
in_proprio_in = np.abs(np.genfromtxt(in_proprio_in_file, delimiter=','))

in_proprio_out_file = os.path.normpath(os.path.join(directory,'features\\in_proprio_out.csv'))
in_proprio_out = np.abs(np.genfromtxt(in_proprio_out_file, delimiter=','))

    # responsiveness to error
in_resp_file = os.path.normpath(os.path.join(directory, 'features\\in_resp.csv'))
in_resp = np.genfromtxt(in_resp_file, delimiter=',')

    # dynamic range of motion during gait
in_ROM_in_file = os.path.normpath(os.path.join(directory, 'features\\in_ROM_in.csv'))
in_ROM_in = np.genfromtxt(in_ROM_in_file, delimiter=',')    

in_ROM_out_file = os.path.normpath(os.path.join(directory, 'features\\in_ROM_out.csv'))
in_ROM_out = np.genfromtxt(in_ROM_out_file, delimiter=',')  

    # baseline FPA
in_bFPA_file = os.path.normpath(os.path.join(directory, 'features\\in_bFPA.csv'))
in_bFPA = np.genfromtxt(in_bFPA_file, delimiter=',')

    # feedback condition (binary, with NF as 0)
feedbackCond_csv_file = os.path.normpath(os.path.join(directory, 'features\\feedbackGroups.csv'))
feedbackCond_file = pd.read_csv(feedbackCond_csv_file)
in_cond_fb = np.zeros((36,))
for i in range(1,37):
        if feedbackCond_file.cond[i-1] == 1 or feedbackCond_file.cond[i-1] == 2:
                in_cond_fb[i-1] = 1

# ii. OUTPUTS:
    # RMSE
out_RMSE_file = os.path.normpath(os.path.join(directory, 'features\\out_RMSE.csv'))
out_RMSE = np.genfromtxt(out_RMSE_file, delimiter=',')

    # error ratio (note: error_in = 1 - error_out)
out_errRatio_out_file = os.path.normpath(os.path.join(directory,'features\\out_errRatio_out.csv'))
out_errRatio_out = np.abs(np.genfromtxt(out_errRatio_out_file, delimiter=','))


# d. assemble inputs and outputs into single arrays:
X = np.stack((in_height[1:], in_weight[1:], in_isFemale[1:], 
              in_vbtest[1:], in_proprio_in[1:], in_proprio_out[1:], in_resp[1:,4],
              np.abs(in_ROM_in[1:]), np.abs(in_ROM_out[1:]),
              in_bFPA[1:], in_cond_fb), axis=1) 
Y = np.stack((out_RMSE[1:,4], out_RMSE[1:,5], out_errRatio_out[1:,5]), axis=1)

# e. run Lasso regression
lasso = Lasso(alpha=alpha)
lasso.fit(X, Y)
print(lasso.coef_)
selected_features = np.abs(lasso.coef_) > alpha
print("Selected features:", selected_features)

# f. run CCA on selected features:
X = np.stack((in_proprio_in[1:], in_bFPA[1:], in_cond_fb, in_resp[1:,4]), axis=1) # shape = 36xN
Y = np.stack((out_RMSE[1:,4], out_RMSE[1:,5], out_errRatio_out[1:,5]), axis=1)

# i. leave-one-out pre-validation
SF_rows = np.where(feedbackCond_file.cond == 1)[0]
TF_rows = np.where(feedbackCond_file.cond == 2)[0]
NF_rows = np.where(feedbackCond_file.cond == 0)[0]

Y_pred = np.zeros_like(Y)
X_pred = np.zeros_like(Y)
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    cca = CCA(n_components=min(X.shape[1], Y.shape[1]))
    cca.fit(X_train, Y_train)

    X_pred[test_index], Y_pred[test_index] = cca.transform(X_test, Y_test)

results = []
for i in range(2):
    
    plt.figure(figsize=(6, 6))
    plt.scatter(X_pred[SF_rows, i], Y_pred[SF_rows, i], alpha=0.7, color = '#0f4c5c', label = 'SF', s = 60)
    plt.scatter(X_pred[TF_rows, i], Y_pred[TF_rows, i], alpha=0.7, color = '#5f0f40', label = 'TF', s = 60)
    plt.scatter(X_pred[NF_rows, i], Y_pred[NF_rows, i], alpha=0.7, color = '#e36414', label = 'NF', s = 60)
    plt.title("Canonical Variate")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.ylim([-2.2, 2])
    plt.xlim([-2.5, 1.7])
    plt.legend()
    plt.show()

    r, p = scipy.stats.pearsonr(X_pred[:, i], Y_pred[:, i])
    
    # calculate F-value
    n = len(X_pred)
    dof = n - 2
    f = (r**2 * dof) / (1 - r**2)
    
    results.append({
        'Component': i + 1,
        'Pearson R': r,
        'F-value': f,
        'p-value': p
    })
        
# print pre-validation results
for result in results:
    print(f"Canonical Component {result['Component']}:")
    print(f"  Pearson's R: {result['Pearson R']:.4f}")
    print(f"  F-value: {result['F-value']:.4f}")
    print(f"  p-value: {result['p-value']:.4f}")
    print()

# ii. compute individual feature loadings:
n_components = min(n_features_in, n_features_out)
cca = CCA(n_components=n_components)
X_c, Y_c = cca.fit_transform(X, Y)

print("\nX loadings:")
print(cca.x_loadings_)
print("\nY loadings:")
print(cca.y_loadings_)
# each column = the x loadings for each variate;
# first column (first variate) shows the strength of contributions of each input feature to that variate

# manually calculate p-values for the loadings between the original variables and the canonical variates:
n = X.shape[0]  # sample size

# get standard errors
se_X = np.sqrt((1 - cca.x_loadings_**2) / (n - 2))
se_Y = np.sqrt((1 - cca.y_loadings_**2) / (n - 2))

# get t-values
t_X = cca.x_loadings_ / se_X
t_Y = cca.y_loadings_ / se_Y

# get p-values (two-tailed test)
p_X = 2 * (1 - scipy.stats.t.cdf(np.abs(t_X), n-2))
p_Y = 2 * (1 - scipy.stats.t.cdf(np.abs(t_Y), n-2))


print("\nP-values for X loadings:")
print(str(p_X))

print("\nP-values for Y loadings:")
print(str(p_Y))

# overall model summary:
cca_model = CanCorr(X,Y)
results = cca_model.corr_test()

print("\nSummary:")
print(results.summary())


