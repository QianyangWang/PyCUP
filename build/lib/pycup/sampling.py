from pyDOE import lhs
import numpy as np
import random
import math

def LHS_sampling(pop,dim,ub,lb):
    samples = lb + (ub-lb) * lhs(dim,pop)
    return samples,lb,ub

def LHS_samplingMV(pop,dims,ubs,lbs):
    samples = []
    for i in range(len(dims)):
        sample = lbs[i] + (ubs[i]-lbs[i]) * lhs(dims[i],pop)
        samples.append(sample)
    return samples,lbs,ubs


def Random_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub

def Random_samplingMV(pop, dims, ubs, lbs):
    samples = []
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        for i in range(pop):
            for j in range(dims[n]):
                X[i, j] = random.random() * (ubs[n][j] - lbs[n][j]) + lbs[n][j]
        samples.append(X)

    return samples, lbs, ubs


def Chebyshev_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    x0=random.random()
    for i in range(pop):
        for j in range(dim):
            Cvalue = np.cos(i*np.arccos(x0))
            X[i, j] = Cvalue*(ub[j] - lb[j]) + lb[j]
            if X[i,j]>ub[j]:
                X[i, j] = ub[j]
            if X[i,j]<lb[j]:
                X[i, j] = lb[j]
            x0 = Cvalue
    return X,lb,ub


def Chebyshev_samplingMV(pop, dims, ubs, lbs):
    samples = []
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0=random.random()
        for i in range(pop):
            for j in range(dims[n]):
                Cvalue = np.cos(i*np.arccos(x0))
                X[i, j] = Cvalue*(ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i,j]>ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i,j]<lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = Cvalue
        samples.append(X)
    return samples,lbs,ubs


def Circle_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    a = 0.5
    b = 0.2
    x0 = random.random()
    for i in range(pop):
        for j in range(dim):
            Cvalue = np.mod(x0 + b - (a / (2 * math.pi)) * np.sin(2 * math.pi * x0), 1)
            X[i, j] = Cvalue * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            x0 = Cvalue

    return X, lb, ub


def Circle_samplingMV(pop, dims, ubs, lbs):
    samples = []
    a = 0.5
    b = 0.2
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0 = random.random()
        for i in range(pop):
            for j in range(dims[n]):
                Cvalue = np.mod(x0 + b - (a / (2 * math.pi)) * np.sin(2 * math.pi * x0), 1)
                X[i, j] = Cvalue * (ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i, j] > ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i, j] < lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = Cvalue
        samples.append(X)
    return samples, lbs, ubs


def Logistic_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    x0 = random.random()
    a = 4
    for i in range(pop):
        for j in range(dim):
            Cvalue = a * x0 * (1 - x0)
            X[i, j] = Cvalue * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            x0 = Cvalue

    return X, lb, ub


def Logistic_samplingMV(pop, dims, ubs, lbs):
    samples = []
    a = 4
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0=random.random()
        for i in range(pop):
            for j in range(dims[n]):
                Cvalue = a * x0 * (1 - x0)
                X[i, j] = Cvalue*(ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i,j]>ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i,j]<lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = Cvalue
        samples.append(X)
    return samples,lbs,ubs


def Piecewise_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    a = 0.5
    x0 = random.random()
    for i in range(pop):
        for j in range(dim):
            if x0 >= 0 and x0 < a:
                Cvalue = x0 / a
            if x0 >= a and x0 < 0.5:
                Cvalue = (x0 - a) / (0.5 - a)
            if x0 >= 0.5 and x0 < 1 - a:
                Cvalue = (1 - a - x0) / (0.5 - a)
            if x0 >= 1 - a and x0 < 1:
                Cvalue = (1 - x0) / a

            X[i, j] = Cvalue * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            x0 = Cvalue
    return X,lb,ub

def Piecewise_samplingMV(pop, dims, ubs, lbs):
    samples = []
    a = 0.5
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0 = random.random()
        for i in range(pop):
            for j in range(dims[n]):
                if x0 >= 0 and x0 < a:
                    Cvalue = x0 / a
                if x0 >= a and x0 < 0.5:
                    Cvalue = (x0 - a) / (0.5 - a)
                if x0 >= 0.5 and x0 < 1 - a:
                    Cvalue = (1 - a - x0) / (0.5 - a)
                if x0 >= 1 - a and x0 < 1:
                    Cvalue = (1 - x0) / a
                X[i, j] = Cvalue * (ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i, j] > ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i, j] < lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = Cvalue
        samples.append(X)
    return samples, lbs, ubs


def Sine_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    x0 = random.random()
    for i in range(pop):
        for j in range(dim):
            Cvalue = np.sin(math.pi * x0)
            X[i, j] = Cvalue * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            x0 = Cvalue

    return X, lb, ub


def Sine_samplingMV(pop, dims, ubs, lbs):
    samples = []
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0=random.random()
        for i in range(pop):
            for j in range(dims[n]):
                Cvalue = np.sin(math.pi * x0)
                X[i, j] = Cvalue*(ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i,j]>ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i,j]<lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = Cvalue
        samples.append(X)
    return samples,lbs,ubs


def Singer_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    u=1.07
    x0 = random.random()
    for i in range(pop):
        for j in range(dim):
            Cvalue = u*(7.86*x0 - 23.31*(x0**2) + 28.75*(x0**3)-13.302875*(x0**4))
            X[i, j] = Cvalue * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            x0 = Cvalue

    return X, lb, ub


def Singer_samplingMV(pop, dims, ubs, lbs):
    u=1.07
    samples = []
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0=random.random()
        for i in range(pop):
            for j in range(dims[n]):
                Cvalue = u*(7.86*x0 - 23.31*(x0**2) + 28.75*(x0**3)-13.302875*(x0**4))
                X[i, j] = Cvalue*(ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i,j]>ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i,j]<lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = Cvalue
        samples.append(X)
    return samples,lbs,ubs


def Sinusoidal_sampling(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    x0 = random.random()

    for i in range(pop):
        for j in range(dim):
            Cvalue = 2.3 * (x0 ** 2) * np.sin(math.pi * x0)
            X[i, j] = Cvalue * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            x0 = Cvalue

    return X, lb, ub


def Sinusoidal_samplingMV(pop, dims, ubs, lbs):
    samples = []
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0=random.random()
        for i in range(pop):
            for j in range(dims[n]):
                Cvalue = 2.3 * (x0 ** 2) * np.sin(math.pi * x0)
                X[i, j] = Cvalue*(ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i,j]>ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i,j]<lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = Cvalue
        samples.append(X)
    return samples,lbs,ubs


def Tent_sampling(pop,dim,lb,ub):
    X = np.zeros([pop, dim])
    a = 0.7
    x0 = random.random()
    for i in range(pop):
        for j in range(dim):
            if x0 < a:
                TentValue = x0 / a
            if x0 >= a:
                TentValue = (1 - x0) / (1 - a)
            X[i, j] = TentValue * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            x0 = TentValue

    return X, lb, ub


def Tent_samplingMV(pop, dims, ubs, lbs):
    samples = []
    a = 0.7
    for n in range(len(dims)):
        X = np.zeros([pop, dims[n]])
        x0=random.random()
        for i in range(pop):
            for j in range(dims[n]):
                if x0 < a:
                    TentValue = x0 / a
                if x0 >= a:
                    TentValue = (1 - x0) / (1 - a)
                X[i, j] = TentValue*(ubs[n][j] - lbs[n][j]) + lbs[n][j]
                if X[i,j]>ubs[n][j]:
                    X[i, j] = ubs[n][j]
                if X[i,j]<lbs[n][j]:
                    X[i, j] = lbs[n][j]
                x0 = TentValue
        samples.append(X)
    return samples,lbs,ubs