
# coding: utf-8

# # a)

# In[1]:


# Programmgeruest zu Versuch 1, Aufgabe 1
import numpy as np


# In[5]:


def getKNearestNeighbors(x,X,k=1):  # realizes nearest neighbor search of x in database
    """
    compute the k nearest neighbors for a query vector x given a data matrix X
    :param x: the query vector x
    :param X: the N x D data matrix (in each row there is data vector) as a numpy array
    :param k: number of nearest-neighbors to be returned
    :return: return list of k line indixes referring to the k nearest neighbors of x in X
    """
    
    x = np.repeat(x[:, np.newaxis], X.shape[0], axis=1).T
    X = np.sum((X-x)*(X-x), axis=1)
    d = np.argsort(X)
    return d[:k]        # REPLACE! return indexes of k smallest distances     


# # b)

# In[38]:


# ***** MAIN PROGRAM ********

# (i) Generate dummy data 
X = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]);      # data matrix X: list of data vectors (=database) of dimension D=3
x = np.array([1.5,3.6,5.7]);                          # a test data vector
print("Data matrix X=\n",X)
print("Test vector x=",x)

# (ii) Print all Euklidean distances to test vector x
print("Euklidean distances to x: ", [0 for i in range(len(X))])   # REPLACE! compute list of Euklidean distances   

# (iii) Search for k nearest neighbor
k=2
idx_knn = getKNearestNeighbors(x,X,k)                  # get indexes of k nearest neighbors
print("idx_knn=",idx_knn)

# (iv) output results
print("The k Nearest Neighbors of x are the following vectors:")
for i in range(k):
    idx=idx_knn[i]
    print("The", i+1, "th nearest neighbor is: X[",idx,"]=",X[idx]," with distance ", np.linalg.norm(X[idx]-x))



# # c)

# O(N*D)  
# kD-Trees sind schneller 
