
# coding: utf-8

# In[1]:


# Programmgeruest zu Versuch 1, Aufgabe 4)
import numpy as np
import pandas as pd
from time import clock
from random import randint
from V1A2_Classifier import *


# In[4]:


# (I) Load data 
forestdata  = pd.read_csv('./Foresttypesdata.csv'); # load data as pandas data frame 
classlabels = ['s','h','d','o'];                                      # possible class labels (C=4) 
classidx    = {classlabels[i]:i for i in range(len(classlabels))}     # dict for mapping classlabel to index 
C           = len(classlabels)        # number of classes (Note: K is now the number of nearest-neighbors!!!!!!)
T_txt = forestdata.values[:,0]        # array of class labels of data vectors (class label is first data attribute)
X = forestdata.values[:,1:]           # array of feature vectors (features are remaining attributes)
T = [classidx[t.strip()] for t in T_txt]          # transform text labels 's','h','d','o' to numeric lables 0,1,2,3
N,D=X.shape                           # size and dimensionality of data set
print("Data set 'ForestData' has size N=", N, " and dimensionality D=",D, " and C=", C, " different classes")
print("X[0..9]=\n",X[0:10])
print("T_txt[0..9]=\n",T_txt[0:10])
print("T[0..9]=\n",T[0:10])


# # b)

# In[7]:


# (II) Test KNN-classifier with S-fold cross validation
S_list=[5]                            # parameter S for cross validation; INSERT appropriate values
K_list=[3]                            # number K of nearest neighbors; INSERT appropriate values
accuracy = np.zeros((len(S_list),len(K_list)));   # array to save accuracy of classifier for each value of S and K
for i in range(len(S_list)):
    S=S_list[i]                      # do an S-fold cross validation
    for j in range(len(K_list)):
        K=K_list[j]
        t1=clock()                   # start time
        knnc = FastKNNClassifier(4,K)            # REPLACE! create appropriate KNN classifier (with kd-trees) 
        pE,pCE = knnc.crossvalidate(S,X,T)      # REPLACE! Do S-fold cross validation and get error probabilities / confusion matrix
        t2=clock()                            # end time
        time_comp=t2-t1                       # computing time in seconds
        print( "\nS=",S," fold cross validation using the",K,"-NNClassifier with KD-Trees yields the following results:")
        print( "Classification error probability = ", pE)
        print( "Accuracy = ", 1.0-pE)
        print( "Confusion Error Probabilities p(class i|class j) = \n", pCE)
        print( "Computing time = ", time_comp, " sec")
        accuracy[i,j]=1.0-pE
print( "\naccuracy=\n",accuracy)
print( "\np_classerror=\n",1.0-accuracy)


# # c)

# In[8]:


# (II) Test KNN-classifier with S-fold cross validation
S_list=[1, 2, 3, 5, 10, 100]                            # parameter S for cross validation; INSERT appropriate values
K_list=[2,3,4,5,10,15,20]                            # number K of nearest neighbors; INSERT appropriate values
accuracy = np.zeros((len(S_list),len(K_list)));   # array to save accuracy of classifier for each value of S and K
for i in range(len(S_list)):
    S=S_list[i]                      # do an S-fold cross validation
    for j in range(len(K_list)):
        K=K_list[j]
        t1=clock()                   # start time
        knnc = FastKNNClassifier(4,K)            # REPLACE! create appropriate KNN classifier (with kd-trees) 
        pE,pCE = knnc.crossvalidate(S,X,T)      # REPLACE! Do S-fold cross validation and get error probabilities / confusion matrix
        t2=clock()                            # end time
        time_comp=t2-t1                       # computing time in seconds
        print( "\nS=",S," fold cross validation using the",K,"-NNClassifier with KD-Trees yields the following results:")
        print( "Classification error probability = ", pE)
        print( "Accuracy = ", 1.0-pE)
        print( "Confusion Error Probabilities p(class i|class j) = \n", pCE)
        print( "Computing time = ", time_comp, " sec")
        accuracy[i,j]=1.0-pE
print( "\naccuracy=\n",accuracy)
print( "\np_classerror=\n",1.0-accuracy)


# # d)

# Man könnte die Trainingsdaten als implizierte Parameter verstehen, auch die Art wie ein kd-Tree gebildet wird (Median oder Mittelwert ?) ist eine Art Parameter
