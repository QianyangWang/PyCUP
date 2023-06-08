import numpy as np
import random
import copy
from . import sampling
from . import save
from . import multi_jobs
import math
from . import progress_bar
from . import Reslib
from .calc_utils import BorderChecker,SortFitness,SortPosition,check_listitem,record_check,record_checkMV
from .calc_utils import CalculateFitness,CalculateFitnessMP,CalculateFitness_MV,CalculateFitnessMP_MV
from collections import Counter
Sampling = "LHS"
EliteOppoSwitch = True
OppoFactor = 0.1
BorderCheckMethod = "rebound"
Mode = 1


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


# non-domination sort considering crowding distances
def rank_distance_sort(X, fitness, ranks, distances):
    # sort according to ranks
    idx = np.argsort(ranks)
    ranks = np.sort(ranks)
    X = X[idx]
    fitness = fitness[idx]
    distances = distances[idx]
    # sort according to distance
    elements = list(set(ranks))
    lengths = [len(ranks[ranks==i]) for i in elements]
    cursor=0
    for l,e in zip(lengths,elements):
        Xe = X[cursor:l]
        fite = fitness[cursor:l]
        dise = distances[cursor:l]
        dis_idx = np.argsort(dise)[::-1]
        Xe = Xe[dis_idx]
        fite = fite[dis_idx]
        X[cursor:l] = Xe
        fitness[cursor:l] = fite
    return X,fitness



def getNonDominationPops(pops, fits,res):
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
    nPop = pops.shape[0]
    nF = fits.shape[1]  # num. of objective functions
    ranks = np.ones(nPop, dtype=np.int32)
    nPs = np.zeros(nPop)  # num. of dominant solutions of each individual
    for i in range(nPop):
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i] <= fits[j]
            isDom2 = fits[i] < fits[j]
            # dominance judgement
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
    r = 0
    indices = np.arange(nPop)
    rIdices = indices[nPs==0]
    ranks[rIdices] = 0
    return pops[ranks==0], fits[ranks==0],res[ranks==0]


def updateArchive(pops, fits, res, archive, arFits, arRes):
    """
    Update the archive according to the current population.

    :argument
    pops: samples -> np.array, shape = (pop, dim)
    fits: fitness -> np.array, shape = (pop, 1)
    res: results array -> np.array, shape = (pop, len(result))
    archive: an archive of non-dominated solutions -> np.array, shape = (archive size, dim)
    arFits: the fitness value of solutions in the archive -> np.array, shape = (archive size, num. obj_fun)
    arRes: the results of solutions in the archive -> np.array, shape = (archive size, len(result))

    :returns
    archive: updated archive
    arFits: fitness of the updated archive
    arRes: results of the updated archive
    """

    nonDomPops, nonDomFits, nonDomRes = getNonDominationPops(pops, fits, res)
    isCh = np.zeros(nonDomPops.shape[0]) >= 1
    nF = fits.shape[1]
    for i in range(nonDomPops.shape[0]):

        isDom1 = nonDomFits[i] >= arFits
        isDom2 = nonDomFits[i] > arFits
        isDom = (np.sum(isDom1, axis=1) == nF) & (np.sum(isDom2, axis=1) >= 1)

        #print((np.sum(nonDomPops[i] == archive,axis=1) < nonDomPops.shape[1]).all())
        notsameflag = np.sum(np.sum(nonDomPops[i] == archive,axis=1) == nonDomPops.shape[1])
        if np.sum(~isDom) >= 1 and notsameflag == 0:
            isCh[i] = True
    if np.sum(isCh) >= 1:
        archive = np.concatenate((archive, nonDomPops[isCh]), axis=0)
        arFits = np.concatenate((arFits, nonDomFits[isCh]), axis=0)
        arRes = np.append(arRes, nonDomRes[isCh], axis=0)

    return archive, arFits, arRes


def getPosition(archive, arFits, M):
    """
    Get the position of the archive.

    :argument
    archive: an archive of non-dominated solutions -> np.array, shape = (archive size, dim)
    arFits: the fitness value of solutions in the archive -> np.array, shape = (archive size, num. obj_fun)
    M: mesh size -> int

    :return
    flags: the position of each particle
    """
    fmin = np.min(arFits, axis=0)
    fmax = np.max(arFits, axis=0)
    grid = (fmax-fmin)/M
    # in case of fmax = fmin -> grid = 0
    for i in range(len(grid)):
        if grid[i] == 0:
            grid[i] = 1
    pos = np.ceil((arFits-fmin)/grid)
    nA, nF = pos.shape
    flags = np.zeros(nA)
    for dim in range(nF-1):
        flags += (pos[:, dim] - 1) * (M**(nF-dim-1))
    flags += pos[:,-1]
    return flags


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

def checkArchive(archive, arFits,arRes, nAr, M):
    """
    Check the size of the archive, if the archive size is too large, reduce it.

    :argument
    archive: an archive of non-dominated solutions -> np.array, shape = (archive size, dim)
    arFits: the fitness value of solutions in the archive -> np.array, shape = (archive size, num. obj_fun)
    arRes: the results of the solutions in the archive.
    nAr: archive size
    M: mesh size

    :returns
    archive: the updated archive
    arFits: the updated fitness
    arRes: the updated results
    """
    if archive.shape[0] <= nAr:
        return archive, arFits,arRes
    else:
        nA = archive.shape[0]
        flags = getPosition(archive, arFits, M)

        counts = Counter(flags).most_common()

        isCh = np.array([True for i in range(nA)])
        indices = np.arange(nA)
        for i in range(len(counts)):
            if counts[i][-1] > 1:

                pn = int((nA-nAr)/nA*counts[i][-1]+0.5)

                gridIdx = indices[flags==counts[i][0]].tolist()
                pIdx = random.sample(gridIdx, pn)
                isCh[pIdx] = False
        archive = archive[isCh]
        arFits = arFits[isCh]
        arRes = arRes[isCh]
        return archive, arFits,arRes


def run(pop, dim, lb, ub, MaxIter,n_obj,nAr,M, fun,RecordPath = None, args=()):
    """
    Main function for the algorithm

    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    MaxIter: num. of iterations. int
    n_obj: number of objective functions -> int
    nAr: archive size -> int
    M: number of mesh grids -> int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    RecordPath: the path of the record file for a hot start -> str
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    hs: Historical samples.
    hf: The fitness of historical samples.
    hr: The results of historical samples.

    Reference:
    single objective version ->
    Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
    Knowledge-Based Systems, 89, 228–249. https://doi.org/10.1016/j.knosys.2015.07.006

    bullets for multi objective modification ->
    1. non-domination sort (used in almost all kinds of MO-algorithms)
    2. corwding distance sort (used also in NSGA2). 1. and 2. were the criteria for sorting the flame population
    3. archive (used also in MOPSO)
    4. flame-based EOBL improvement (modified and proposed by Qianyang Wang in MFO, and converted to a non-domination version
       by Qianyang Wang in MOMFO)


    Usage:
    import pycup as cp
    from pycup.test_functions import ZDT4

    dim = 10
    lb = np.zeros(10)
    ub = np.ones(10)
    n_obj=2
    cp.MOMFO.EliteOppoSwitch = True
    cp.MOMFO.run(90,10,lb,ub,500,2,100,20,ZDT4)
    """
    print("Current Algorithm: MFO")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    checker = BorderChecker(method=BorderCheckMethod)
    if not RecordPath:
        iter = 0
        hs = []
        hf = []
        hr = []
        X, lb, ub = initial(pop, dim, ub, lb)
        fitness,res = CalculateFitness(X, fun,n_obj, args)
        # X t-1
        Xp = copy.copy(X)
        fitnessP = copy.copy(fitness)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))

        # initialize the archive set
        archive, arFits,arRes = getNonDominationPops(X, fitness,res)

        # the first iteration use the sorted X as the flame
        ranks = nonDominationSort(X, fitness)
        distances = crowdingDistanceSort(X, fitness, ranks)
        #Xs, fitnessS = select1(pop, X, fitness, ranks, distances)
        Xs, fitnessS = rank_distance_sort(X, fitness, ranks, distances)
        record = save.MOswarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         X=X,fitness=fitness,iteration=0,
                         Xs=Xs,fitnessS=fitnessS,Xp=Xp,fitnessP=fitnessP,archive=archive,arFits=arFits,arRes=arRes,n_obj=n_obj)
        record.save()

    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        archive = record.archive
        arFits = record.arFits
        arRes = record.arRes
        X = record.X
        Xs = record.Xs
        Xp = record.Xp
        fitnessS = record.fitnessS
        fitness = record.fitness
        fitnessP = record.fitnessP
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        same = record_check(pop,dim,lb,ub,a_pop,a_dim,a_lb,a_ub)
        if not same:
            raise ValueError("The pop, dim, lb, and ub should be same as the record")
        Progress_Bar.update(len(hf))

    for t in range(iter,MaxIter):

        Flame_no = round(pop - t * ((pop - 1) / MaxIter))
        a = -1 + t * (-1) / MaxIter  # linear
        for i in range(pop):
            for j in range(dim):
                if i <= Flame_no-1:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1

                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[i, j]
                else:
                    distance_to_flame = np.abs(Xs[Flame_no-1, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1
                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[Flame_no-1, j]


        X = checker.BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitness(X, fun,n_obj, args)
        archive, arFits,arRes = updateArchive(X, fitness,res, archive, arFits,arRes)
        archive, arFits,arRes = checkArchive(archive, arFits,arRes, nAr, M)
        # merge the flame and moth
        if Mode == 0:
            # Mirjalili's original version
            fitnessM = np.concatenate([fitnessP, fitness], axis=0)
            Xm = np.concatenate([Xp, X], axis=0)
        elif Mode == 1:
            # My version: Remain the historical local optimums, v1 and v2 will be the same in iteration 0 and 1
            fitnessM = np.concatenate([fitnessS,fitness],axis=0)
            Xm = np.concatenate([Xs,X],axis=0)
        else:
            raise NotImplementedError("The mode should be 0 or 1.")
        ranks = nonDominationSort(Xm, fitnessM)
        distances = crowdingDistanceSort(Xm, fitnessM, ranks)
        #Xm, fitnessM = select1(Xm.shape[0], Xm, fitnessM, ranks, distances)
        Xm, fitnessM = rank_distance_sort(Xm, fitnessM, ranks, distances)
        # update the Xs (flame population)
        Xs = Xm[0:Flame_no,:]
        fitnessS = fitnessM[0:Flame_no]

        Xp = copy.copy(X)
        fitnessP = copy.copy(fitness)
        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(Xs.shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = copy.copy(Xs[0:EliteNumber, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)
                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = checker.BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CalculateFitness(XOppo, fun,n_obj, args)
                for j in range(EliteNumber):
                    isDom1 = fitOppo[j] < fitnessS[j]
                    isDom2 = fitOppo[j] <= fitnessS[j]
                    isDom = sum(isDom1) >= 1 and sum(isDom2) == n_obj
                    if isDom:
                        fitnessS[j] = copy.copy(fitOppo[j])
                        Xs[j, :] = copy.copy(XOppo[j, :])

                ranks = nonDominationSort(Xs, fitnessS)
                distances = crowdingDistanceSort(Xs, fitnessS, ranks)
                Xs, fitnessS = rank_distance_sort(Xs, fitnessS, ranks, distances)
                X2file = np.concatenate([X2file, XOppo], axis=0)
                fitness2file = np.concatenate([fitness2file, fitOppo], axis=0)
                res2file = np.concatenate([res2file, resOppo], axis=0)

                archive, arFits, arRes = updateArchive(XOppo, fitOppo, resOppo, archive, arFits, arRes)
                archive, arFits, arRes = checkArchive(archive, arFits, arRes, nAr, M)

        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)


        Progress_Bar.update(len(hf))

        record = save.MOswarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         X=X,fitness=fitness,iteration=t+1,
                         Xs=Xs,fitnessS=fitnessS,Xp=Xp,fitnessP=fitnessP,archive=archive,arFits=arFits,arRes=arRes,n_obj=n_obj)
        record.save()

    paretoPops, paretoFits, paretoRes = getNonDominationPops(archive, arFits, arRes)
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes, method_info="Pareto front")
    raw_saver = save.RawDataSaver(hs, hf, hr, paretoFits=paretoFits, paretoPops=paretoPops,paretoRes=paretoRes,OptType="MO-SWARM" )
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")
    return  hs, hf, hr


def runMP(pop, dim, lb, ub, MaxIter,n_obj,nAr,M, fun,n_jobs,RecordPath = None, args=()):
    """
    Main function for the algorithm

    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    MaxIter: num. of iterations. int
    n_obj: number of objective functions -> int
    nAr: archive size -> int
    M: number of mesh grids -> int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: number of threads for multi-processing optimization -> int
    RecordPath: the path of the record file for a hot start -> str
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    hs: Historical samples.
    hf: The fitness of historical samples.
    hr: The results of historical samples.

    Reference:
    single objective version ->
    Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
    Knowledge-Based Systems, 89, 228–249. https://doi.org/10.1016/j.knosys.2015.07.006

    bullets for multi objective modification ->
    1. non-domination sort (used in almost all kinds of MO-algorithms)
    2. corwding distance sort (used also in NSGA2). 1. and 2. were the criteria for sorting the flame population
    3. archive (used also in MOPSO)
    4. flame-based EOBL improvement (modified and proposed by Qianyang Wang in MFO, and converted to a non-domination version
       by Qianyang Wang in MOMFO)


    Usage:
    import pycup as cp
    from pycup.test_functions import ZDT4

    dim = 10
    lb = np.zeros(10)
    ub = np.ones(10)
    n_obj=2
    cp.MOMFO.EliteOppoSwitch = True
    cp.MOMFO.runMP(90,10,lb,ub,50,2,100,20,ZDT4,5)
    """
    print("Current Algorithm: MFO")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    checker = BorderChecker(method=BorderCheckMethod)
    if not RecordPath:
        iter = 0
        hs = []
        hf = []
        hr = []
        X, lb, ub = initial(pop, dim, ub, lb)
        fitness,res = CalculateFitnessMP(X, fun,n_obj,n_jobs, args)
        # X t-1
        Xp = copy.copy(X)
        fitnessP = copy.copy(fitness)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))

        # initialize the archive set
        archive, arFits,arRes = getNonDominationPops(X, fitness,res)

        # the first iteration use the sorted X as the flame
        ranks = nonDominationSort(X, fitness)
        distances = crowdingDistanceSort(X, fitness, ranks)
        #Xs, fitnessS = select1(pop, X, fitness, ranks, distances)
        Xs, fitnessS = rank_distance_sort(X, fitness, ranks, distances)
        record = save.MOswarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         X=X,fitness=fitness,iteration=0,
                         Xs=Xs,fitnessS=fitnessS,Xp=Xp,fitnessP=fitnessP,archive=archive,arFits=arFits,arRes=arRes,n_obj=n_obj)
        record.save()

    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        archive = record.archive
        arFits = record.arFits
        arRes = record.arRes
        X = record.X
        Xs = record.Xs
        Xp = record.Xp
        fitnessS = record.fitnessS
        fitness = record.fitness
        fitnessP = record.fitnessP
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        same = record_check(pop,dim,lb,ub,a_pop,a_dim,a_lb,a_ub)
        if not same:
            raise ValueError("The pop, dim, lb, and ub should be same as the record")
        Progress_Bar.update(len(hf))

    for t in range(iter,MaxIter):

        Flame_no = round(pop - t * ((pop - 1) / MaxIter))
        a = -1 + t * (-1) / MaxIter  # linear
        for i in range(pop):
            for j in range(dim):
                if i <= Flame_no-1:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1

                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[i, j]
                else:
                    distance_to_flame = np.abs(Xs[Flame_no-1, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1
                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[Flame_no-1, j]


        X = checker.BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitnessMP(X, fun,n_obj,n_jobs, args)
        archive, arFits,arRes = updateArchive(X, fitness,res, archive, arFits,arRes)
        archive, arFits,arRes = checkArchive(archive, arFits,arRes, nAr, M)
        if Mode == 0:
            # Mirjalili's original version
            fitnessM = np.concatenate([fitnessP, fitness], axis=0)
            Xm = np.concatenate([Xp, X], axis=0)
        elif Mode == 1:
            # My version: Remain the historical local optimums, v1 and v2 will be the same in iteration 0 and 1
            fitnessM = np.concatenate([fitnessS,fitness],axis=0)
            Xm = np.concatenate([Xs,X],axis=0)
        else:
            raise NotImplementedError("The mode should be 0 or 1.")
        ranks = nonDominationSort(Xm, fitnessM)
        distances = crowdingDistanceSort(Xm, fitnessM, ranks)
        Xm, fitnessM = rank_distance_sort(Xm, fitnessM, ranks, distances)
        # update the Xs (flame population)
        Xs = Xm[0:Flame_no,:]
        fitnessS = fitnessM[0:Flame_no]

        Xp = copy.copy(X)
        fitnessP = copy.copy(fitness)
        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(Xs.shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = copy.copy(Xs[0:EliteNumber, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)
                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = checker.BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CalculateFitnessMP(XOppo, fun,n_obj,n_jobs, args)
                for j in range(EliteNumber):
                    isDom1 = fitOppo[j] < fitnessS[j]
                    isDom2 = fitOppo[j] <= fitnessS[j]
                    isDom = sum(isDom1) >= 1 and sum(isDom2) == n_obj
                    if isDom:
                        fitnessS[j] = copy.copy(fitOppo[j])
                        Xs[j, :] = copy.copy(XOppo[j, :])

                ranks = nonDominationSort(Xs, fitnessS)
                distances = crowdingDistanceSort(Xs, fitnessS, ranks)
                Xs, fitnessS = rank_distance_sort(Xs, fitnessS, ranks, distances)
                X2file = np.concatenate([X2file, XOppo], axis=0)
                fitness2file = np.concatenate([fitness2file, fitOppo], axis=0)
                res2file = np.concatenate([res2file, resOppo], axis=0)

                archive, arFits, arRes = updateArchive(XOppo, fitOppo, resOppo, archive, arFits, arRes)
                archive, arFits, arRes = checkArchive(archive, arFits, arRes, nAr, M)

        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)

        Progress_Bar.update(len(hf))

        record = save.MOswarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         X=X,fitness=fitness,iteration=t+1,
                         Xs=Xs,fitnessS=fitnessS,Xp=Xp,fitnessP=fitnessP,archive=archive,arFits=arFits,arRes=arRes,n_obj=n_obj)
        record.save()

    paretoPops, paretoFits, paretoRes = getNonDominationPops(archive, arFits, arRes)
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes, method_info="Pareto front")
    raw_saver = save.RawDataSaver(hs, hf, hr, paretoFits=paretoFits, paretoPops=paretoPops,paretoRes=paretoRes,OptType="MO-SWARM" )
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")
    return  hs, hf, hr



