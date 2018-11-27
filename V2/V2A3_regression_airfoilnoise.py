
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# V2A3_regression_airfoilnoise.py
# Programmgeruest zu Versuch 2, Aufgabe 3
# to log outputs start with: python V2A3_regression_airfoilnoise.py >V2A3_regression_airfoilnoise.log

import numpy as np
import pandas as pd

from V2A2_Regression import *


# ***** MAIN PROGRAM ********
# (I) Hyper-Parameters
S=3;               # S-fold cross-validation
lmbda=1;           # regularization parameter (lambda>0 avoids also singularities)
K=1;               # K for K-Nearest Neighbors
flagKLinReg = 0;   # if flag==1 and K>=D then do a linear regression of the KNNs to make prediction
deg=1;             # degree of basis function polynomials 
flagSTD=0;         # if >0 then standardize data before training (i.e., scale X to mean value 0 and standard deviation 1)
N_pred=5;          # number of predictions on the training set for testing
x_test_1 = [0,0,0,0,0];   # REPLACE dummy code: define test vector 1
x_test_2 = [0,0,0,0,0];   # REPLACE dummy code: define test vector 2



# In[2]:


# (II) Load data 
fname='../DATA/AirfoilSelfNoise/airfoil_self_noise.xls'
airfoil_data = pd.read_excel(fname,0); # load data as pandas data frame 
T = airfoil_data.values[:,5]           # target values = noise load (= column 5 of data table)
X = airfoil_data.values[:,:5]          # feature vectors (= column 0-4 of data table)
N,D=X.shape                            # size and dimensionality of data set
idx_perm = np.random.permutation(N)    # get random permutation for selection of test vectors 
print("Data set ",fname," has size N=", N, " and dimensionality D=",D)
print("X=",X)
print("T=",T)
print("x_test_1=",x_test_1)
print("x_test_2=",x_test_2)
print("number of basis functions M=", len(phi_polynomial(X[1],deg)))



# In[3]:


# (III) Do least-squares regression with regularization 
print("\n#### Least Squares Regression with regularization lambda=", lmbda, " ####")
lsr = None  # REPLACE dummy code: Create and fit Least-Squares Regressifier using polynomial basis function of degree deg and flagSTD for standardization of data  
print("lsr.W_LSR=",None)    # REPLACE dummy code: print weight vector for least squares regression  
print("III.1) Some predictions on the training data:")
for i in range(N_pred): 
    n=idx_perm[i]
    print("Prediction for X[",n,"]=",X[n]," is y=",None,", whereas true value is T[",n,"]=",T[n])   # REPLACE dummy code: compute prediction for X[n]
print("III.2) Some predicitions for new test vectors:")
print("Prediction for x_test_1 is y=", None)    # REPLACE dummy code: compute prediction for x_test_1
print("Prediction for x_test_2 is y=", None)    # REPLACE dummy code: compute prediction for x_test_2
print("III.3) S=",S,"fold Cross Validation:")
err_abs,err_rel = None, None                   # REPLACE dummy code: do cross validation!! 
print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel) 



# In[4]:


# (IV) Do KNN regression  
print("\n#### KNN regression with flagKLinReg=", flagKLinReg, " ####")
knnr = None                                    # REPLACE dummy code: Create and fit KNNRegressifier
print("IV.1) Some predictions on the training data:")
for i in range(N_pred): 
    n=idx_perm[i]
    print("Prediction for X[",n,"]=",X[n]," is y=",None,", whereas true value is T[",n,"]=",T[n])   # REPLACE dummy code: compute prediction for X[n]
print("IV.2) Some predicitions for new test vectors:")
print("Prediction for x_test_1 is y=", None)    # REPLACE dummy code: compute prediction for x_test_1
print("Prediction for x_test_2 is y=", None)    # REPLACE dummy code: compute prediction for x_test_2
print("IV.3) S=",S,"fold Cross Validation:")
err_abs,err_rel = None, None                   # REPLACE dummy code: do cross validation!! 
print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel) 


