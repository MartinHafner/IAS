
# coding: utf-8

# # Aufgabe 1: (10+9+14+12 = 45 Punkte)  
# Thema: Lineare “Least-Squares” Regression mit Regularisierung in Python  
# Gegeben seien Daten {(xn, tn)|n = 1, ..., N } welche ursprünglich von der Parabel f (x ) = w0+
# w1x +w2x  
# 2 mit w0 = 2, w1 = −1, w2 = 3 gesampelt wurden, aber nun mit Rauschen behaftet  
# sind. Zu diesen Daten soll ein lineares Regressionsmodell y = wTφ(x ) mit polynomiellen  
# Basisfunktionen φ bestimmt werden.  
#   
# ##### a) Betrachten Sie das Programmgerüst V2A1_LinearRegression.py aus dem Praktikumsverzeichnis:  

# Erklären Sie kurz in eigenen Worten (jeweils 1-2 Sätze) wozu die Funktionen fun_true(.),
# generateDataSet(.), getDataError(.) und phi_polynomial(.) dienen. Versuchen
# Sie den Python-Code zu verstehen (muss nicht dokumentiert werden).

# - fun_true(X): berechnet für jedes Element vom X das ensprechende y nach der Parabelfunktion y=3*x²-x+2  
# - generateDataset(N,xmin,xmax,sd_noise): erstellt eine N große Liste von x Werten mit dazugehörigen zielwerten (y) die aber mit einem rauschen gemischt werden  
# - getDataError(Y,T): berechnet die Fehlerquadratsumme für T und Y 
# - phi_polynomial(x,deg=1): berechnet den Merkmalsvektor für x bis zum grad deg(standartmäßig 1)

# Von welcher Funktion sind die Original-Daten (xn, tn) gesampelt?

# - fun_true(X) / t=3*x²-x+2

# Wie lauten die Basisfunktionen φj(x ) für j = 1, ...,deg des linearen Modells?

# - φ = x^5 + x^4 + x^3 + x^2 + x + 1

# Welche Rolle hat die Variable lmbda?

# - lmbda ist der Regularisierungsparameter 

# Worin unterscheiden sich die Variablen X,T von X_test,T_test?

# - X,T haben die gleichen Parameter wie X_test, T_test sind aber mit anderen Zufallswerten erstellt worden

# Was stellen im Plot die grünen Kreuze/Punkte, grüne Kurve, rote Kurve dar?

# - grüne Kreuze sind die Lerndaten
# - grüne Punkte sind die Testdaten
# - grüne Kurve ist die Ausgangsfunktion
# - rote Kurve ist die von uns vorhergesagte Funktion

# In[45]:


# V2A1_LinearRegression.py 
# Programmgeruest zu Versuch 2, Aufgabe 1
import numpy as np 
import matplotlib.pyplot as plt

def fun_true(X):                              # compute 1-dim. parable function; X must be Nx1 data matrix
    w2,w1,w0 = 3.0,-1.0,2.0                   # true parameters of parable y(x)=w0+w1*x+w2*x*x
    return w0+w1*X+w2*np.multiply(X,X)        # return function values (same size as X)

def generateDataSet(N,xmin,xmax,sd_noise):    # generate data matrix X and target values T
    X=xmin+np.random.rand(N,1)*(xmax-xmin)    # get random x values uniformly in [xmin;xmax)
    T=fun_true(X);                            # target values without noise
    if(sd_noise>0):
        T=T+np.random.normal(0,sd_noise,X.shape) # add noise 
    return X,T

def getDataError(Y,T):                        # compute data error (least squares) between prediction Y and true target values T
    D=np.multiply(Y-T,Y-T);                   # squared differences between Y and T
    return 0.5*sum(sum(D)); #eine Summe zu viel?   E_D

def phi_polynomial(x,deg=1):                            # compute polynomial basis function vector phi(x) for data x 
    assert(np.shape(x)==(1,)), "currently only 1dim data supported"
    return np.array([x[0]**i for i in range(deg+1)]).T; # returns feature vector phi(x)=[1 x x**2 x**3 ... x**deg]

def predict(x,w):
    temp = np.array(sum([g*(x**i) for i,g in enumerate(w)]))
    return temp



# In[46]:


# (I) generate data 
np.random.seed(10)                            # set seed of random generator (to be able to regenerate data)
N=10                                          # number of data samples
xmin,xmax=-5.0,5.0                            # x limits
sd_noise=10                                   # standard deviation of Guassian noise
X,T           = generateDataSet(N, xmin,xmax, sd_noise)             # generate training data
X_test,T_test = generateDataSet(N, xmin,xmax, sd_noise)             # generate test data
print("X=",X, "T=",T)



# In[47]:


# (II) generate linear least squares model for regression
lmbda=0                                                           # no regression
deg=5                                                             # degree of polynomial basis functions
N,D = np.shape(X)                                                 # shape of data matrix X
N,K = np.shape(T)                                                 # shape of target value matrix T
PHI = np.array([phi_polynomial(X[i],deg).T for i in range(N)])    # generate design matrix
N,M = np.shape(PHI)                                               # shape of design matrix
print("PHI=", PHI)
W_LSR = np.dot(np.linalg.pinv(PHI),T)
print("W_LSR=",W_LSR)



# In[48]:


# (III) make predictions for test data
Y_test = np.array([predict(xt, W_LSR) for xt in X_test])   # REPLACE THIS BY PROGNOSIS FOR TEST DATA X_test! (result should be N x 1 matrix, i.e., one prognosis per row)
Y_learn = np.array([predict(xt, W_LSR) for xt in X])  # REPLACE THIS BY PROGNOSIS FOR TEST DATA X_test! (result should be N x 1 matrix, i.e., one prognosis per row)
print("Y_test=",Y_test)
print("T_test=",T_test)
print("learn data error = ", getDataError(Y_learn,T))
print("test data error = ", getDataError(Y_test,T_test))
print("W_LSR=",W_LSR)
print("mean weight = ", np.mean(np.mean(np.abs(W_LSR))))



# In[49]:


# (IV) plot data
ymin,ymax = -50.0,150.0                     # interval of y data
x_=np.arange(xmin,xmax,0.01)                # densely sampled x values
Y_LSR = np.array([np.dot(W_LSR.T,np.array([phi_polynomial([x],deg)]).T)[0] for x in x_]);   # least squares prediction
Y_true = fun_true(x_).flat

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X.flat,T.flat,c='g',marker='x',s=100)             # plot learning data points (green x)
ax.scatter(X_test.flat,T_test.flat,c='g',marker='.',s=100)   # plot test data points (green .)
ax.plot(x_,Y_LSR.flat, c='r')         # plot LSR regression curve (red)
ax.plot(x_,Y_true, c='g')             # plot true function curve (green)
ax.set_xlabel('x')                    # label on x-axis
ax.set_ylabel('y')                    # label on y-axis
ax.grid()                             # draw a grid
plt.ylim((ymin,ymax))                 # set y-limits
plt.show()                            # show plot on screen


