import numpy as np
import random
import copy
from . import sampling
from . import save
from . import multi_jobs
import math
from . import progress_bar
from . import Reslib
from .calc_utils import CalculateFitness,CalculateFitnessMP,BorderCheck

Sampling = "LHS"
pc = 0.6
pm = 0.1
etaC = 1
etaM = 1


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


def mutate(X, pm, etaM, lb, ub):
    nPop = X.shape[0]
    dim = X.shape[1]
    for i in range(nPop):
        if np.random.rand() < pm:
            polyMutation(X[i], etaM)
    X = BorderCheck(X,ub=ub,lb=lb,pop = nPop,dim=dim)
    return X

def polyMutation(chr, etaM):

    pos1, pos2 = np.sort(np.random.randint(0,len(chr),2))
    pos2 += 1
    u = np.random.rand()
    if u < 0.5:
        delta = (2*u) ** (1/(etaM+1)) - 1
    else:
        delta = 1-(2*(1-u)) ** (1/(etaM+1))
    chr[pos1:pos2] += delta


def nonDominationSort(X, fitness):

    pop = X.shape[0]
    nF = fitness.shape[1]
    ranks = np.zeros(pop, dtype=np.int32)
    nPs = np.zeros(pop)
    sPs = []
    for i in range(pop):
        iSet = []
        for j in range(pop):
            if i == j:
                continue
            isDom1 = fitness[i] <= fitness[j]
            isDom2 = fitness[i] < fitness[j]

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
        #rPops = X[ranks==r]
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
                    dis[orIdices[j]] += (rSortFits[j+1,i] - rSortFits[j-1,i]) / \
                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n-1]] = np.inf
    return dis


def select1(pool, X, fitness,ress, ranks, distances):

    pop, dim = X.shape
    nF = fitness.shape[1]
    newX = np.zeros((pool, dim))
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
        newX[i] = X[idx]
        newFits[i] = fitness[idx]
        newRess[i] = ress[idx]
        i += 1
    return newX, newFits ,newRess

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

def crossover(X, pc, etaC, lb, ub):

    chrX = X.copy()
    pop = chrX.shape[0]
    dim = chrX.shape[1]
    for i in range(0, pop, 2):
        if np.random.rand() < pc:
            SBX(chrX[i], chrX[i+1], etaC)
    chrX = BorderCheck(chrX,ub=ub,lb=lb,pop=pop,dim=dim)
    return chrX


def SBX(chr1, chr2, etaC):

    pos1, pos2 = np.sort(np.random.randint(0,len(chr1),2))
    pos2 += 1
    u = np.random.rand()
    if u <= 0.5:
        gamma = (2*u) ** (1/(etaC+1))
    else:
        gamma = (1/(2*(1-u))) ** (1/(etaC+1))
    x1 = chr1[pos1:pos2]
    x2 = chr2[pos1:pos2]
    chr1[pos1:pos2], chr2[pos1:pos2] = 0.5*((1+gamma)*x1+(1-gamma)*x2), \
        0.5*((1-gamma)*x1+(1+gamma)*x2)



def optSelect(X, fitness,res, chrPops, chrFits,chrRes):

    pop, dim = X.shape
    nF = fitness.shape[1]

    newX = np.zeros((pop, dim))
    newFits = np.zeros((pop, nF))
    if not Reslib.UseResObject:
        lenRes = res.shape[1]
        newRes = np.zeros((pop, lenRes))
    else:
        newRes = np.zeros(pop,dtype=object)

    MergePops = np.concatenate((X, chrPops), axis=0)
    MergeFits = np.concatenate((fitness, chrFits), axis=0)
    MergeRes = np.concatenate((res, chrRes), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]
    while i + len(rIndices) <= pop:
        newX[i:i + len(rIndices)] = MergePops[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits[rIndices]
        newRes[i:i + len(rIndices)] = MergeRes[rIndices]
        r += 1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]

    if i < pop:
        rDistances = MergeDistances[rIndices]
        rSortedIdx = np.argsort(rDistances)[::-1]
        surIndices = rIndices[rSortedIdx[:(pop - i)]]
        newX[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
        newRes[i:] = MergeRes[surIndices]
    return newX, newFits, newRes


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
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :return:
    paretoPops: pareto solutions -> np.array, shape = (n_pareto, dim)
    paretoFits: fitness of pareto solutions -> np.array, shape = (n_pareto, n_obj)
    paretoRes: results of pareto solutions -> np.array, shape = (n_pareto, len(result))

    Reference:

    """
    print("Current Algorithm: NSGA-II")
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

        ranks = nonDominationSort(X, fitness)
        distances = crowdingDistanceSort(X, fitness, ranks)
        X, fitness,res = select1(pop, X, fitness,res, ranks, distances)
        chrpops = crossover(X, pc, etaC, lb, ub)
        chrpops = mutate(chrpops, pm, etaM, lb, ub)
        chrfits, chrres = CalculateFitness(chrpops,fun,n_obj=n_obj,args=args)
        X, fitness, res = optSelect(X, fitness,res, chrpops, chrfits,chrres)

        hr.append(chrres)
        hs.append(copy.copy(chrpops))
        hf.append(copy.copy(chrfits))
        Progress_Bar.update(len(hf))

        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=t+1,n_obj=n_obj,fitness=fitness,res=res)
        record.save()

    ranks = nonDominationSort(X, fitness)
    paretoPops = X[ranks == 0]
    paretoFits = fitness[ranks == 0]
    paretoRes = res[ranks == 0]
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes,method_info="Pareto front")

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
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :return:
    paretoPops: pareto solutions -> np.array, shape = (n_pareto, dim)
    paretoFits: fitness of pareto solutions -> np.array, shape = (n_pareto, n_obj)
    paretoRes: results of pareto solutions -> np.array, shape = (n_pareto, len(result))

    Reference:

    """
    print("Current Algorithm: NSGA-II")
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

        ranks = nonDominationSort(X, fitness)
        distances = crowdingDistanceSort(X, fitness, ranks)
        X, fitness,res = select1(pop, X, fitness,res, ranks, distances)
        chrpops = crossover(X, pc, etaC, lb, ub)
        chrpops = mutate(chrpops, pm, etaM, lb, ub)
        chrfits, chrres = CalculateFitnessMP(chrpops,fun,n_obj=n_obj,n_jobs=n_jobs,args=args)
        X, fitness, res = optSelect(X, fitness,res, chrpops, chrfits,chrres)

        hr.append(chrres)
        hs.append(copy.copy(chrpops))
        hf.append(copy.copy(chrfits))
        Progress_Bar.update(len(hf))

        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=t+1,n_obj=n_obj,fitness=fitness,res=res)
        record.save()

    ranks = nonDominationSort(X, fitness)
    paretoPops = X[ranks == 0]
    paretoFits = fitness[ranks == 0]
    paretoRes = res[ranks == 0]
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes, method_info="Pareto front")

    raw_saver = save.RawDataSaver(hs, hf, hr, paretoFits=paretoFits, paretoPops=paretoPops,paretoRes=paretoRes,OptType="MO-SWARM")
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return paretoPops, paretoFits,paretoRes


