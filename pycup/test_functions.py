import math
import numpy as np
from . import evaluation_metrics
import random

"""
The benchmark functions are mainly selected from the following references

Reference:
Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
Knowledge-Based Systems, 89, 228–249. https://doi.org/10.1016/j.knosys.2015.07.006

X: parameter array -> np.array, shape = (1, dim)

Here you can use the None0Const to make sure that the minimum fitness value > 0.
It is for the convenience when using the likelihood uncertainty analysis function (in case of divided by 0)
Users can set the Non0Const = 0 by
pycup.test_functions.Non0Const = 0
"""

Non0Const = 1

#dim = 30
#Unimodal benchmark function.
##fmin = 1,  range = [-100,100]
def uni_fun1(X):

    fitness = np.sum(np.power(X,2)) + Non0Const
    result = fitness.reshape(1,-1)

    return fitness,result
##fmin = 0 + Non0Const,  range = [-10,10]
def uni_fun2(X):

    fitness = np.sum(np.abs(X)) + np.prod(np.abs(X)) + Non0Const
    result = fitness.reshape(1,-1)
    return fitness,result
##fmin = 0 + Non0Const,  range = [-100,100]
def uni_fun3(X):

    fitness = 0
    for i in range(len(X)):
        fitness = fitness + np.square(np.sum(X[0:i + 1]))
    fitness = fitness + Non0Const
    result = fitness.reshape(1,-1)
    return fitness,result

##fmin = 0 + Non0Const,  range = [-100,100]
def uni_fun4(X):
    fitness = np.max(np.abs(X)) + Non0Const
    result = fitness.reshape(1, -1)
    return fitness,result
##fmin = 0 + Non0Const,  range = [-30,30]
def uni_fun5(X):
    X_len = len(X)
    fitness = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1)) + Non0Const
    result = fitness.reshape(1,-1)
    return fitness,result
##fmin = 0 + Non0Const, range = [-100,100]
def uni_fun6(X):
    fitness=np.sum(np.square(np.abs(X+0.5))) + Non0Const
    result = fitness.reshape(1, -1)
    return fitness,result
##fmin = 0 + Non0Const,  range = [-1.28,1.28]
def uni_fun7(X):

    v = [i * X[i-1]**4 for i in range(1,len(X)+1)]
    fitness = np.sum(v) + random.random() + Non0Const
    result = fitness.reshape(1, -1)

    return fitness,result

#Multimodal benchmark functions
##fmin = -418.98 * dim,  range = [-500,500]
def mul_fun1(X):

    fitness = np.sum(-1*np.multiply(X,np.sin(np.sqrt(np.abs(X)))))
    result = fitness.reshape(1, -1)

    return fitness,result

##fmin = 0 + Non0Const, range = [-5.12,5.12]
def mul_fun2(X):
    fitness = np.sum(np.multiply(X,X) - 10*np.cos(2*np.pi*X) + 10) + Non0Const
    result = fitness.reshape(1, -1)

    return fitness, result

##fmin = 0 + Non0Const,  range = [-600,600]
def mul_fun3(X):
    a = np.sum(np.power(X,2))/4000
    b = np.prod([np.cos(X[i-1]/np.sqrt(i)) for i in range(1,len(X)+1)])
    fitness = a - b + 1 + Non0Const
    result = fitness.reshape(1, -1)
    return fitness, result

#Fixed-dimension multimodal benchmark functions
##fmin = 0.9684,  range = [-5,5]
def fixed_fun1(X):
    fitness = 4*X[0]**2 - 2.1*X[0]**4 + (X[0]**6)/3 + X[0] * X[1] - 4*X[1]**2 + 4*X[1]**4 + 2
    result = fitness.reshape(1, -1)
    return fitness, result

##fmin = 0.398, range = [-5,5]
def fixed_fun2(X):
    fitness = (X[1] - (5.1/(4*np.pi**2))*X[0]**2 + (5/np.pi)*X[0] - 6)**2 + 10*(1-1/(8*np.pi)*np.cos(X[0])) + 10
    result = fitness.reshape(1, -1)
    return fitness, result

#Time series functions len(TS) = 100, Metric = 1 - NSE
def ts_fun1(X):

    t = np.arange(100)
    obs = np.sin(t) * 2.1 + np.tan(t) * 5.2**3 + random.random()
    res = np.sin(t) * X[0] + np.tan(t) * X[1]**3
    fitness = evaluation_metrics.OneMinusNSE(obs,res)
    result = res.reshape(1, -1)
    return fitness, result


def ts_fun2(X):

    t = np.arange(100)
    obs = np.sin(t) * (2.1 + np.random.randn()) + np.cos(t) * (5.2 + np.random.randn())**3 + random.random()
    res = np.sin(t) * X[0] + np.cos(t) * X[1]**3
    fitness = evaluation_metrics.OneMinusNSE(obs,res)
    result = res.reshape(1, -1)
    return fitness, result


def mv_fun1(Xs):
    X1 = Xs[0]
    fitness1 = 4*X1[0]**2 - 2.1*X1[0]**4 + (X1[0]**6)/3 + X1[0] * X1[1] - 4*X1[1]**2 + 4*X1[1]**4 + 2
    result1 = fitness1.reshape(1, -1)

    X2 = Xs[1]
    fitness2 = (X2[1] - (5.1/(4*np.pi**2))*X2[0]**2 + (5/np.pi)*X2[0] - 6)**2 + 10*(1-1/(8*np.pi)*np.cos(X2[0])) + 10
    result2 = fitness2.reshape(1, -1)

    fitness = [fitness1,fitness2]
    results = [result1,result2]
    return fitness,results


# for MOPSO testing
# dim = 30, lb = np.zeros(30), ub = np.ones(30)
def ZDT1(X):
    ObjV1 = X[0]
    NDIM = len(X)
    gx = 1 + 9 * np.sum(X[1:]) / (NDIM - 1)
    hx = 1 - np.sqrt(np.abs(ObjV1) / gx)
    ObjV2 = gx * hx
    fitness = np.array([ObjV1,ObjV2]).reshape(1,-1)
    result = fitness
    return fitness,result

def ZDT2(X):
    ObjV1 = X[0]
    NDIM = len(X)
    gx = 1 + 9 * np.sum(X[1:]) / (NDIM - 1)
    hx = 1 - (ObjV1/gx)**2
    ObjV2 = gx * hx
    fitness = np.array([ObjV1,ObjV2]).reshape(1,-1)
    result = fitness
    return fitness,result

def ZDT3(X):
    ObjV1 = X[0]
    NDIM = len(X)
    gx = 1 + 9 * np.sum(X[1:]) / (NDIM - 1)
    hx = 1 - np.sqrt(ObjV1/ gx) - (ObjV1/ gx) * np.sin(10*np.pi*ObjV1)
    ObjV2 = gx * hx
    fitness = np.array([ObjV1,ObjV2]).reshape(1,-1)
    result = fitness
    return fitness,result

def ZDT4(X):
    ObjV1 = X[0]
    NDIM = len(X)
    gx = 1 + 10 * (NDIM-1) + np.sum(X[1:]**2 - 10 * np.cos(4 * np.pi * X[1:]))
    hx = 1 - np.sqrt(ObjV1/ gx)
    ObjV2 = gx * hx
    fitness = np.array([ObjV1,ObjV2]).reshape(1,-1)
    result = fitness
    return fitness,result

def ZDT6(X):
    ObjV1 = 1-np.exp(-4*X[0]) * np.power(np.sin(6*np.pi*X[0]),6)
    NDIM = len(X)
    gx = 1+ 9 * np.power((np.sum(X[1:])/(NDIM-1)),0.25)
    hx = 1 - (ObjV1/gx)**2
    ObjV2 = gx * hx
    fitness = np.array([ObjV1,ObjV2]).reshape(1,-1)
    result = fitness
    return fitness,result
