import numpy as np
import random
import copy
from . import sampling
from . import save
from . import multi_jobs
import math
from . import progress_bar
from random import choices
from . import Reslib
from .calc_utils import BorderCheck,CalculateFitness,CalculateFitnessMP

Sampling = "LHS"
F = 0.2
Cr = 0.9


def initial(pop, dim, ub, lb):
    """
    lhs sampling based initialization
    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array

    :returns
    X: the updated samples/solutions
    lb: upper boundary
    ub: lower boundary
    """
    try:
        X, lb, ub = eval("sampling.{}_sampling".format(Sampling))(pop=pop, dim=dim, ub=ub, lb=lb)
    except:
        raise KeyError("The selectable sampling strategies are: 'LHS','Random','Chebyshev','Circle','Logistic','Piecewise','Sine','Singer','Sinusoidal','Tent'.")

    return X, lb, ub




def getNonDominationPops(X, fitness,res):
    """
    The fast non-dominant sort function.

    :argument
    pops: samples -> np.array, shape = (pop, dim)
    fits: fitness -> np.array, shape = (pop, 1)
    res: results array -> np.array, shape = (pop, len(result))

    :return
    pops[ranks==0]: Non-dominated samples
    fits[ranks==0]: Fitness array of non-dominated samples
    res[ranks==0]: Results array of non-dominated samples
    """
    pop = X.shape[0]
    nF = fitness.shape[1]  # num. of objective functions
    ranks = np.ones(pop, dtype=np.int32)
    nPs = np.zeros(pop)  # num. of dominant solutions of each individual
    for i in range(pop):
        for j in range(pop):
            if i == j:
                continue
            isDom1 = fitness[i] <= fitness[j]
            isDom2 = fitness[i] < fitness[j]
            # dominance judgement
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
    indices = np.arange(pop)
    rIdices = indices[nPs==0]
    ranks[rIdices] = 0
    return X[ranks==0], fitness[ranks==0],res[ranks==0]


def mutate(X, F, lb, ub):
    pop, dim = X.shape
    mutantX = np.zeros((pop, dim))
    indices = np.arange(pop).tolist()
    for i in range(pop):
        rs = random.sample(indices, 3)
        mutantX[i] = X[rs[0]] + F * (X[rs[1]] - X[rs[2]])
    # Bordercheck
    mutantX = BorderCheck(mutantX,ub,lb,pop,dim)
    return mutantX


def crossover(X, mutantX, Cr):
    pop, dim = X.shape
    choiMuX1 = np.random.rand(pop, dim) < Cr
    choiMuX2 = np.random.randint(0,pop,(pop,dim))  == np.tile(np.arange(dim),(pop,1))
    choiMuX = choiMuX1 | choiMuX2
    choiX = ~ choiMuX
    trialX = mutantX * choiMuX + X * choiX
    return trialX


def nonDominationSort(X, fits):

    pop = X.shape[0]
    nF = fits.shape[1]
    ranks = np.zeros(pop, dtype=np.int32)
    nPs = np.zeros(pop)
    sPs = []
    for i in range(pop):
        iSet = []
        for j in range(pop):
            if i == j:
                continue
            isDom1 = fits[i] <= fits[j]
            isDom2 = fits[i] < fits[j]

            if sum(isDom1) == nF and sum(isDom2) >= 1:
                iSet.append(j)

            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
        sPs.append(iSet)
    r = 0
    indices = np.arange(pop)
    while sum(nPs==0) != 0:
        rIdices = indices[nPs==0]
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1
        r += 1
    return ranks


def crowdingDistanceSort(X, fitness, ranks):

    pop = X.shape[0]
    nF = fitness.shape[1]
    dis = np.zeros(pop)
    nR = ranks.max()
    indices = np.arange(pop)
    for r in range(nR+1):
        rIdices = indices[ranks==r]
        rFits = fitness[ranks==r]
        rSortIdices = np.argsort(rFits, axis=0)
        rSortFits = np.sort(rFits,axis=0)
        fMax = np.max(rFits,axis=0)
        fMin = np.min(rFits,axis=0)
        n = len(rIdices)
        for i in range(nF):
            orIdices = rIdices[rSortIdices[:,i]]
            j = 1
            while n > 2 and j < n-1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j+1,i] - rSortFits[j-1,i]) / (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n-1]] = np.inf
    return dis

def select1(pool, X, fitness,ress, ranks, distances):

    pop, dim = X.shape
    nF = fitness.shape[1]
    newPops = np.zeros((pool, dim))
    newFits = np.zeros((pool, nF))
    if not Reslib.UseResObject:
        l_res = ress.shape[1]
        newRess = np.zeros((pool,l_res))
    else:
        newRess = np.zeros(pool,dtype=object)

    indices = np.arange(pop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)
        idx = compare(idx1, idx2, ranks, distances)
        newPops[i] = X[idx]
        newFits[i] = fitness[idx]
        newRess[i] = ress[idx]
        i += 1
    return newPops, newFits ,newRess


def compare(idx1, idx2, ranks, distances):

    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        if distances[idx1] <= distances[idx2]:
            idx = idx2
        else:
            idx = idx1
    return idx


def record_check(pop, dim, lb, ub,n_obj, a_pop, a_dim, a_lb, a_ub,a_n_obj):
    a = np.sum(lb == a_lb) == len(lb)
    b = np.sum(ub == a_ub) == len(ub)
    check_list = (pop == a_pop, dim == a_dim, a, b, n_obj == a_n_obj)
    if np.sum(check_list) == len(check_list):
        return True
    else:
        return False


def run(pop,dim,lb,ub,MaxIter,n_obj,fun,RecordPath = None,args=()):
    """
    Main function for the algorithm

    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    MaxIter: num. of iterations. int
    n_obj: number of objective functions -> int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    Vmin: lower boundary of the speed -> np.array, shape = (,dim)
    Vmax: upper boundary of the speed -> np.array, shape = (,dim)
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :return:
    paretoPops: pareto solutions -> np.array, shape = (n_pareto, dim)
    paretoFits: fitness of pareto solutions -> np.array, shape = (n_pareto, n_obj)
    paretoRes: results of pareto solutions -> np.array, shape = (n_pareto, len(result))

    Reference:

    """
    print("Current Algorithm: MODE")
    print("Iterations: 1 + {}".format(MaxIter))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    if not RecordPath:
        iter = 0
        hs = []
        hf = []
        hr = []

        X,lb,ub = initial(pop, dim, ub, lb)

        fitness,res = CalculateFitness(X,fun,n_obj,args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=0,n_obj=n_obj,fitness=fitness,res=res)
        record.save()
    else:
        record = save.MOswarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        fitness = record.fitness
        res = record.res
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        a_n_obj = record.n_obj
        same = record_check(pop,dim,lb,ub,n_obj,a_pop,a_dim,a_lb,a_ub,a_n_obj)
        if not same:
            raise ValueError("The pop, dim, lb, ub, and n_obj should be same as the record")
        Progress_Bar.update(len(hf))

    for t in range(iter,MaxIter):

        mutantX = mutate(X, F, lb, ub)
        trialX = crossover(X, mutantX, Cr)
        trialX = BorderCheck(trialX, ub, lb, pop, dim)
        trialFits,trialRes = CalculateFitness(trialX, fun,n_obj=n_obj,args=args)
        M_pops = np.concatenate((X, trialX), axis=0)
        M_fits = np.concatenate((fitness, trialFits), axis=0)
        M_ress = np.concatenate((res, trialRes), axis=0)
        M_ranks = nonDominationSort(M_pops, M_fits)
        distances = crowdingDistanceSort(M_pops, M_fits, M_ranks)

        X, fitness,res = select1(pop, M_pops, M_fits,M_ress, M_ranks, distances)

        hr.append(trialRes)
        hs.append(copy.copy(trialX))
        hf.append(copy.copy(trialFits))
        Progress_Bar.update(len(hf))

        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=t+1,n_obj=n_obj,fitness=fitness,res=res)
        record.save()

    paretoPops = M_pops[M_ranks == 0]
    paretoFits = M_fits[M_ranks == 0]
    paretoRes = M_ress[M_ranks == 0]
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes, method_info="Pareto front")
    raw_saver = save.RawDataSaver(hs, hf, hr, paretoFits=paretoFits, paretoPops=paretoPops,paretoRes=paretoRes,OptType="MO-SWARM")
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return paretoPops, paretoFits,paretoRes


def runMP(pop,dim,lb,ub,MaxIter,n_obj,fun,n_jobs,RecordPath = None,args=()):
    """
    Main function for the algorithm

    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    MaxIter: num. of iterations. int
    n_obj: number of objective functions -> int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    Vmin: lower boundary of the speed -> np.array, shape = (,dim)
    Vmax: upper boundary of the speed -> np.array, shape = (,dim)
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :return:
    paretoPops: pareto solutions -> np.array, shape = (n_pareto, dim)
    paretoFits: fitness of pareto solutions -> np.array, shape = (n_pareto, n_obj)
    paretoRes: results of pareto solutions -> np.array, shape = (n_pareto, len(result))

    Reference:

    """
    print("Current Algorithm: MODE")
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    if not RecordPath:
        iter = 0
        hs = []
        hf = []
        hr = []

        X,lb,ub = initial(pop, dim, ub, lb)

        fitness,res = CalculateFitnessMP(X,fun,n_obj,n_jobs,args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=0,n_obj=n_obj,fitness=fitness,res=res)
        record.save()
    else:
        record = save.MOswarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        fitness = record.fitness
        res = record.res
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        a_n_obj = record.n_obj
        same = record_check(pop,dim,lb,ub,n_obj,a_pop,a_dim,a_lb,a_ub,a_n_obj)
        if not same:
            raise ValueError("The pop, dim, lb, ub, and n_obj should be same as the record")
        Progress_Bar.update(len(hf))

    for t in range(iter,MaxIter):

        mutantX = mutate(X, F, lb, ub)
        trialX = crossover(X, mutantX, Cr)
        trialX = BorderCheck(trialX, ub, lb, pop, dim)
        trialFits,trialRes = CalculateFitnessMP(trialX, fun,n_obj=n_obj,n_jobs=n_jobs,args=args)
        M_pops = np.concatenate((X, trialX), axis=0)
        M_fits = np.concatenate((fitness, trialFits), axis=0)
        M_ress = np.concatenate((res, trialRes), axis=0)
        M_ranks = nonDominationSort(M_pops, M_fits)
        distances = crowdingDistanceSort(M_pops, M_fits, M_ranks)

        X, fitness,res = select1(pop, M_pops, M_fits,M_ress, M_ranks, distances)

        hr.append(trialRes)
        hs.append(copy.copy(trialX))
        hf.append(copy.copy(trialFits))
        Progress_Bar.update(len(hf))
        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=t+1,n_obj=n_obj,fitness=fitness,res=res)
        record.save()

    paretoPops = M_pops[M_ranks == 0]
    paretoFits = M_fits[M_ranks == 0]
    paretoRes = M_ress[M_ranks == 0]
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes, method_info="Pareto front")
    raw_saver = save.RawDataSaver(hs, hf, hr, paretoFits=paretoFits, paretoPops=paretoPops,paretoRes=paretoRes,OptType="MO-SWARM")
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return paretoPops, paretoFits,paretoRes




