
# coding: utf-8

# In[19]:


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
K=13;               # K for K-Nearest Neighbors
flagKLinReg = 1;   # if flag==1 and K>=D then do a linear regression of the KNNs to make prediction
deg=5;             # degree of basis function polynomials 
flagSTD=1;         # if >0 then standardize data before training (i.e., scale X to mean value 0 and standard deviation 1)
N_pred=5;          # number of predictions on the training set for testing
x_test_1 = [1250,11,0.2,69.2,0.0051];   # REPLACE dummy code: define test vector 1
x_test_2 = [1305,8,0.1,57.7,0.0048];   # REPLACE dummy code: define test vector 2



# In[2]:


# (II) Load data 
fname='./AirfoilSelfNoise/airfoil_self_noise.xls'
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
lsr = LSRRegressifier(lmbda=lmbda,phi=lambda x: phi_polynomial(x,deg),flagSTD=flagSTD)  # REPLACE dummy code: Create and fit Least-Squares Regressifier using polynomial basis function of degree deg and flagSTD for standardization of data  
lsr.fit(X,T)
print("lsr.W_LSR=",lsr.W_LSR)    # REPLACE dummy code: print weight vector for least squares regression  
print("III.1) Some predictions on the training data:")
for i in range(N_pred): 
    n=idx_perm[i]
    print("Prediction for X[",n,"]=",X[n]," is y=",lsr.predict(X[n]),", whereas true value is T[",n,"]=",T[n])   # REPLACE dummy code: compute prediction for X[n]
print("III.2) Some predicitions for new test vectors:")
print("Prediction for x_test_1 is y=", lsr.predict(x_test_1))    # REPLACE dummy code: compute prediction for x_test_1
print("Prediction for x_test_2 is y=", lsr.predict(x_test_2))    # REPLACE dummy code: compute prediction for x_test_2
print("III.3) S=",S,"fold Cross Validation:")
err_abs,err_rel = lsr.crossvalidate(S,X,T)                  # REPLACE dummy code: do cross validation!! 
print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel) 



# In[13]:


lmbdaRange = [1,2,5,10,30,60,100]
degRange = list(range(1,8))
LSRErrors = []

for lmb in lmbdaRange:
    for d in degRange:
        # (III) Do least-squares regression with regularization 
        print("\n#### Least Squares Regression with regularization lambda=", lmb, " deg=",d,  " ####")
        lsr = LSRRegressifier(lmbda=lmb,phi=lambda x: phi_polynomial(x,d),flagSTD=flagSTD)  # REPLACE dummy code: Create and fit Least-Squares Regressifier using polynomial basis function of degree deg and flagSTD for standardization of data  
        lsr.fit(X,T)
        print("lsr.W_LSR=",lsr.W_LSR)    # REPLACE dummy code: print weight vector for least squares regression  
        #print("III.1) Some predictions on the training data:")
        #for i in range(N_pred): 
            #n=idx_perm[i]
            #print("Prediction for X[",n,"]=",X[n]," is y=",lsr.predict(X[n]),", whereas true value is T[",n,"]=",T[n])   # REPLACE dummy code: compute prediction for X[n]
        #print("III.2) Some predicitions for new test vectors:")
        #print("Prediction for x_test_1 is y=", lsr.predict(x_test_1))    # REPLACE dummy code: compute prediction for x_test_1
        #print("Prediction for x_test_2 is y=", lsr.predict(x_test_2))    # REPLACE dummy code: compute prediction for x_test_2
        print("III.3) S=",S,"fold Cross Validation:")
        err_abs,err_rel = lsr.crossvalidate(S,X,T)                  # REPLACE dummy code: do cross validation!!
        LSRErrors.append((lmb, d, err_abs, err_rel))
        print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel) 


# In[36]:


temp = []
for i,e in enumerate(LSRErrors):
    temp.append((e[2][3],i))
temp.sort()
print(temp)

smallest_mean= LSRErrors[4]
smallest_sd= LSRErrors[25]
smallest_min= LSRErrors[4]
smallest_max= LSRErrors[37]
print(smallest_mean)
print(smallest_sd)
print(smallest_min)
print(smallest_max)


# In[20]:


# (IV) Do KNN regression  
print("\n#### KNN regression with flagKLinReg=", flagKLinReg, " ####")
knnr = KNNRegressifier(K,flagKLinReg)                                   # REPLACE dummy code: Create and fit KNNRegressifier
knnr.fit(X,T)
print("IV.1) Some predictions on the training data:")
for i in range(N_pred): 
    n=idx_perm[i]
    print("Prediction for X[",n,"]=",X[n]," is y=",knnr.predict(X[n]),", whereas true value is T[",n,"]=",T[n])   # REPLACE dummy code: compute prediction for X[n]
print("IV.2) Some predicitions for new test vectors:")
print("Prediction for x_test_1 is y=", knnr.predict(x_test_1))    # REPLACE dummy code: compute prediction for x_test_1
print("Prediction for x_test_2 is y=", knnr.predict(x_test_2))    # REPLACE dummy code: compute prediction for x_test_2
print("IV.3) S=",S,"fold Cross Validation:")
err_abs,err_rel = knnr.crossvalidate(S,X,T)                   # REPLACE dummy code: do cross validation!! 
print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel) 



# In[5]:


flagRange = [0,1]
kRange = list(range(1,15))
KNNErrors = []

for f in flagRange:
    for k in kRange:
        # (IV) Do KNN regression  
        print("\n#### KNN regression with flagKLinReg=", f," K=",k, " ####")
        knnr = KNNRegressifier(k,f)                                   # REPLACE dummy code: Create and fit KNNRegressifier
        knnr.fit(X,T)
        #print("IV.1) Some predictions on the training data:")
        #for i in range(N_pred): 
         #   n=idx_perm[i]
         #   print("Prediction for X[",n,"]=",X[n]," is y=",knnr.predict(X[n]),", whereas true value is T[",n,"]=",T[n])   # REPLACE dummy code: compute prediction for X[n]
        #print("IV.2) Some predicitions for new test vectors:")
        #print("Prediction for x_test_1 is y=", knnr.predict(x_test_1))    # REPLACE dummy code: compute prediction for x_test_1
        #print("Prediction for x_test_2 is y=", knnr.predict(x_test_2))    # REPLACE dummy code: compute prediction for x_test_2
        print("IV.3) S=",S,"fold Cross Validation:")
        err_abs,err_rel = knnr.crossvalidate(S,X,T)                   # REPLACE dummy code: do cross validation!! 
        KNNErrors.append((f, k, err_abs, err_rel))
        print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel) 


# In[18]:


temp = []
for i,e in enumerate(KNNErrors):
    temp.append((e[2][3],i))
temp.sort()
#print(temp)

smallest_mean= KNNErrors[25][3]
smallest_sd= KNNErrors[26][3]
smallest_min= KNNErrors[19][3]
smallest_max= KNNErrors[12][3]
print(smallest_mean)
print(smallest_sd)
print(smallest_min)
print(smallest_max)
print(KNNErrors[26])


# ###### a) Vervollständigen Sie das Programmgerüst V2A3_regression_airfoilnoise.py um eine Least-Squares-Regression auf den Daten zu berechnen. Optimieren Sie die HyperParameter um bei einer S = 3-fachen Kreuzvalidierung möglichst kleine Fehlerwerte zu erhalten.

# Welche Bedeutung haben jeweils die Hyper-Parameter lmbda, deg, flagSTD?

# - lmbda: Regularisierungs parameter
# - deg: Ist der Grad er polynomiellen Basisfunktion
# - flagSTD: gibt an ob die Daten standartisiert werden sollen

# Was passiert ohne Skalierung der Daten (flagSTD=0) bei höheren Polynomgraden
# (achten Sie auf die Werte von maxZ)?

# - EXCEPTION DUE TO BAD CONDITION:flagOK= 0  maxZ= 195590361182.93994  eps= 1e-06   
# - der Fehlerwert beim invertieren explodiert ohne die Skalierung

# Geben Sie Ihre optimalen Hyper-Parameter sowie die resultierenden Fehler-Werte
# an.

# - lambda=1, deg=5, flagSTD=1
# - absolute errors (E,sd,min,max)= (2.2241976966993438, 3.6504292455503355, 0.0023069096170331704, 83.63604473632992)
# - relative errors (E,sd,min,max)= (0.018030635646137383, 0.03132168189811296, 1.746150761488692e-05, 0.762838110293237)

# Welche Prognosen ergibt Ihr Modell für die neuen Datenvektoren
# x_test_1=[1250,11,0.2,69.2,0.0051] bzw. x_test_2=[1305,8,0.1,57.7,0.0048]
# ?

# - Prediction for x_test_1 is y= 130.45406757371268
# - Prediction for x_test_2 is y= 133.1060885540002

# Welchen Polynomgrad und wieviele Basisfunktionen verwendet Ihr Modell?

# - Polynomgrad ist 5
# - mit 6 Basisfunktionen

# ###### b) Vervollständigen Sie das Programmgerüst V2A3_regression_airfoilnoise.py um eine KNN-Regression auf den Daten zu berechnen. Optimieren Sie die Hyper-Parameter um bei einer S = 3-fachen Kreuzvalidierung möglichst kleine Fehlerwerte zu erhalten.

# Welche Bedeutung haben jeweils die Hyper-Parameter K und flagKLinReg?

# - K: anzahl der NN die zum Ergebniss beitragen
# - flagKLinReg: gibt an ob die KNN noch in einen LSRegressifier gegeben werden oder ob nur der Durchschnitt der KNN berechnet wird.

# Geben Sie Ihre optimalen Hyper-Parameter sowie die resultierenden Fehler-Werte
# an.

# - flagKLinReg=1, K=13
# - absolute errors (E,sd,min,max)= (3.0858901947930324, 3.0296772825440974, 0.006016325705218151, 32.13059666400561) 
# - relative errors (E,sd,min,max)= (0.02482629194750422, 0.024838560765987942, 4.864074982591945e-05, 0.2851617187841634)

# Welche Prognosen ergibt Ihr Modell für die neuen Datenvektoren
# x_test_1=[1250,11,0.2,69.2,0.0051] bzw. x_test_2=[1305,8,0.1,57.7,0.0048]
# ?

# - Prediction for x_test_1 is y= 126.56192024990972
# - Prediction for x_test_2 is y= 132.35085577388358

# ###### c) Vergleichen Sie die beiden Modelle. Welches liefert die besseren Ergebnisse?

# LSR: absolute errors (E,sd,min,max)= (2.2241976966993438, 3.6504292455503355, 0.0023069096170331704, 83.63604473632992)  
# KNN: absolute errors (E,sd,min,max)= (3.0858901947930324, 3.0296772825440974, 0.006016325705218151, 32.13059666400561)  
# absolute differences:                (−0.861692498,       0.620751963,        −0.003709416, 51.505448072)   
# LSR: relative errors (E,sd,min,max)= (0.018030635646137383, 0.03132168189811296, 1.746150761488692e-05, 0.762838110293237)  
# KNN: relative errors (E,sd,min,max)= (0.02482629194750422, 0.024838560765987942, 4.864074982591945e-05, 0.2851617187841634)
# relative differences:                (−0.006795656,        0.006483121,          −0.000031179, 0.477676392

# - KNN scheint die bessereren Ergebnisse zu liefern, verwendet intern aber auch LSR
