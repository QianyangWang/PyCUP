import numpy as np
import random
import copy
from . import sampling
from . import save
from . import multi_jobs
import math
from collections import Counter
from . import progress_bar
from . import Reslib
from .calc_utils import CalculateFitness,CalculateFitnessMP,BorderCheck

VFactor = 0.3 # velocity factor, users can modify this to control the velocity ranges. Default 0.1 * parameter ranges
EliteOppoSwitch = True # Switch for elite opposition based learning
OppoFactor = 0.1 # proportion of samples to do the elite opposition operation
Sampling = "LHS"

w = 0.9 # inertia factor
c1 = 2  # constant 1
c2 = 2  # constant 2


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


def getGBest(pops, fits, archive, arFits, M):
    """
    Find the global optimal solution from the archive set according to the density.

    :argument
    pops: samples -> np.array, shape = (pop, dim)
    fits: fitness -> np.array, shape = (pop, 1)
    archive: an archive of non-dominated solutions -> np.array, shape = (archive size, dim)
    arFits: the fitness value of solutions in the archive -> np.array, shape = (archive size, num. obj_fun)
    M: mesh size -> int

    :return:
    gBest: the global optimal solution set
    """
    nPop, nChr = pops.shape # nPop: population size, nChr: dim
    nF = fits.shape[1]
    gBest = np.zeros((nPop, nChr))
    flags = getPosition(archive, arFits, M)

    counts = Counter(flags).most_common()
    for i in range(nPop):
        # get the non-dominated solution from the archive
        isDom1 = fits[i] <= arFits
        isDom2 = fits[i] < arFits
        isDom = (np.sum(isDom1, axis=1)==nF) & \
            (np.sum(isDom2, axis=1)>=1)

        isDom = ~isDom
        if np.sum(isDom) == 0:
            gBest[i] = pops[i]
            continue
        elif np.sum(isDom) == 1:
            gBest[i] = archive[isDom]
            continue
        archivePop = archive[isDom]
        archivePopFit = arFits[isDom]
        # obtain the position of particles in the archive set
        aDomFlags = flags[isDom]
        counts = Counter(aDomFlags).most_common()
        minFlag, minCount = counts[-1]
        minFlags = [counts[i][0] for i in range(len(counts))
                    if counts[i][1]==minCount]
        isCh = False
        for minFlag in minFlags:
            isCh = isCh | (aDomFlags == minFlag)
        indices = np.arange(aDomFlags.shape[0])
        chIndices = indices[isCh]

        idx = chIndices[int(np.random.rand()*len(chIndices))]
        gBest[i] = archivePop[idx]

    return gBest


def updatePBest(pBest, pFits, pops, fits):
    """
    Update the particles' best solutions.

    :argument
    pBest: the particles' best solutions
    pFits: the corresponding fitness
    pops: not used
    fits: not used

    :returns:
    pBest: the updated particles' best solutions
    pFits: the updated corresponding fitness
    """
    nPop, nF = fits.shape
    isDom1 = fits <= pFits
    isDom2 = fits < pFits
    isCh = (np.sum(isDom1, axis=1) == nF) & (np.sum(isDom2, axis=1) >= 1)
    if np.sum(isCh) >= 1:
        # update the pBest if the solution dominate the current pBest
        pBest[isCh] = pops[isCh]
        pFits[isCh] = fits[isCh]
    return pBest, pFits


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


def record_check(pop, dim, lb, ub,n_obj, a_pop, a_dim, a_lb, a_ub,a_n_obj):
    a = np.sum(lb == a_lb) == len(lb)
    b = np.sum(ub == a_ub) == len(ub)
    check_list = (pop == a_pop, dim == a_dim, a, b, n_obj == a_n_obj)
    if np.sum(check_list) == len(check_list):
        return True
    else:
        return False



def run(pop,dim,lb,ub,MaxIter,n_obj,nAr,M,fun,Vmin=None,Vmax=None,RecordPath = None,args=()):
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
    M: mesh size -> int
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
    Coello Coello, C. A., & Lechuga, M. S. (2002, May). MOPSO: a proposal for multiple objective particle swarm optimization.
    In Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No.02TH8600) (pp. 1051–1056). IEEE.
    https://doi.org/10.1109/CEC.2002.1004388
    """
    print("Current Algorithm: MOPSO")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: {}".format(MaxIter+1))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    if not Vmin or not Vmax:
        Vmin = -VFactor * (ub-lb)
        Vmax = VFactor * (ub-lb)
    if not RecordPath:
        iter = 0
        hs = []
        hf = []
        hr = []

        X,lb,ub = initial(pop, dim, ub, lb)
        V,Vmin,Vmax = initial(pop, dim, Vmax, Vmin)
        fitness,res = CalculateFitness(X,fun,n_obj,args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))

        # initialize the archive set
        archive, arFits,arRes = getNonDominationPops(X, fitness,res)
        #GbestScore = copy.copy(fitness[0])
        GbestPositon = copy.copy(X)
        Pbest = copy.copy(X)
        fitnessGbest = copy.copy(fitness)
        fitnessPbest = copy.copy(fitness)
        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=0,n_obj=n_obj,
                                    Pbest=Pbest,GbestPositon=GbestPositon,V=V,fitnessPbest=fitnessPbest,fitnessGbest=fitnessGbest,
                                    archive=archive,arFits=arFits,arRes=arRes)
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
        V = record.V
        Pbest = record.Pbest
        fitnessPbest = record.fitnessPbest
        GbestPositon = record.GbestPositon
        fitnessGbest = record.fitnessGbest
        archive = record.archive
        arFits = record.arFits
        arRes = record.arRes
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

        # copy for Levy flight
        for j in range(pop):

            V[j,:] = w*V[j,:] + c1*np.random.random()*(Pbest[j, :] - X[j, :]) + c2*np.random.random()*(GbestPositon[j, :] - X[j, :])

            for ii in range(dim):
               if V[j, ii] < Vmin[ii]:
                   V[j, ii] = Vmin[ii]
               if V[j, ii] > Vmax[ii]:
                   V[j, ii] = Vmax[ii]

            X[j, :] = X[j, :] + V[j, :]

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitness(X, fun,n_obj, args)
        Pbest, fitnessPbest = updatePBest(Pbest, fitnessPbest, X, fitness)
        archive, arFits,arRes = updateArchive(X, fitness,res, archive, arFits,arRes)
        archive, arFits,arRes = checkArchive(archive, arFits,arRes, nAr, M)
        #GbestPositon = getGBest(X, fitness, archive, arFits, M)
        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(archive.shape[0] * OppoFactor))
            if EliteNumber > 0:
                EliteIdx = np.sort(np.array(random.sample(range(0,archive.shape[0]),EliteNumber)))
                XElite = copy.copy(archive[EliteIdx, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)
                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CalculateFitness(XOppo, fun,n_obj, args)
                for i in range(len(EliteIdx)):
                    notsameflag = np.sum(np.sum(XOppo[i] == archive, axis=1) == XOppo.shape[1])
                    isDom1 = fitOppo[i] < arFits[EliteIdx[i]]
                    isDom2 = fitOppo[i] <= arFits[EliteIdx[i]]
                    isDom = sum(isDom1) >= 1 and sum(isDom2) == n_obj
                    if isDom and notsameflag == 0:
                        archive[EliteIdx[i], :] = XOppo[i, :]
                        arFits[EliteIdx[i], :] = fitOppo[i, :]
                X2file = np.concatenate([X2file,XOppo],axis=0)
                fitness2file = np.concatenate([fitness2file,fitOppo],axis=0)
                res2file = np.concatenate([res2file,resOppo],axis=0)
        GbestPositon = getGBest(X, fitness, archive, arFits, M)

        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)
        Progress_Bar.update(len(hf))

        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=t+1,n_obj=n_obj,
                                    Pbest=Pbest,GbestPositon=GbestPositon,V=V,fitnessPbest=fitnessPbest,fitnessGbest=fitnessGbest,
                                    archive=archive,arFits=arFits,arRes=arRes)
        record.save()

    paretoPops, paretoFits,paretoRes = getNonDominationPops(archive, arFits,arRes)
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes, method_info="Pareto front")
    raw_saver = save.RawDataSaver(hs, hf, hr, paretoFits=paretoFits, paretoPops=paretoPops,paretoRes=paretoRes,OptType="MO-SWARM")
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return paretoPops, paretoFits,paretoRes


def runMP(pop, dim, lb, ub, MaxIter,  n_obj, nAr, M,fun,n_jobs, Vmin=None, Vmax=None,RecordPath = None, args=()):
    """
    Main function for the algorithm (multi-processing version)

    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    MaxIter: num. of iterations. int
    n_obj: number of objective functions -> int
    nAr: archive size -> int
    M: mesh size -> int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: number of threads/processes -> int
    Vmin: lower boundary of the speed -> np.array, shape = (,dim)
    Vmax: upper boundary of the speed -> np.array, shape = (,dim)
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :return:
    paretoPops: pareto solutions -> np.array, shape = (n_pareto, dim)
    paretoFits: fitness of pareto solutions -> np.array, shape = (n_pareto, n_obj)
    paretoRes: results of pareto solutions -> np.array, shape = (n_pareto, len(result))
    """
    print("Current Algorithm: MOPSO (Multi-Processing)")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: {}".format(MaxIter + 1))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    if not Vmin or not Vmax:
        Vmin = -VFactor * (ub - lb)
        Vmax = VFactor * (ub - lb)

    if not RecordPath:
        iter = 0
        hs = []
        hf = []
        hr = []

        X, lb, ub = initial(pop, dim, ub, lb)
        V, Vmin, Vmax = initial(pop, dim, Vmax, Vmin)
        fitness, res = CalculateFitnessMP(X, fun, n_obj,n_jobs, args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))

        archive, arFits, arRes = getNonDominationPops(X, fitness, res)
        # GbestScore = copy.copy(fitness[0])
        GbestPositon = copy.copy(X)
        Pbest = copy.copy(X)
        fitnessGbest = copy.copy(fitness)
        fitnessPbest = copy.copy(fitness)
        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=0,n_obj=n_obj,
                                    Pbest=Pbest,GbestPositon=GbestPositon,V=V,fitnessPbest=fitnessPbest,fitnessGbest=fitnessGbest,
                                    archive=archive,arFits=arFits,arRes=arRes)
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
        V = record.V
        Pbest = record.Pbest
        fitnessPbest = record.fitnessPbest
        GbestPositon = record.GbestPositon
        fitnessGbest = record.fitnessGbest
        archive = record.archive
        arFits = record.arFits
        arRes = record.arRes
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

        for j in range(pop):

            V[j, :] = w * V[j, :] + c1 * np.random.random() * (Pbest[j, :] - X[j, :]) + c2 * np.random.random() * (
                        GbestPositon[j, :] - X[j, :])

            for ii in range(dim):
                if V[j, ii] < Vmin[ii]:
                    V[j, ii] = Vmin[ii]
                if V[j, ii] > Vmax[ii]:
                    V[j, ii] = Vmax[ii]

            X[j, :] = X[j, :] + V[j, :]

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitnessMP(X, fun, n_obj,n_jobs, args)
        Pbest, fitnessPbest = updatePBest(Pbest, fitnessPbest, X, fitness)
        archive, arFits, arRes = updateArchive(X, fitness, res, archive, arFits, arRes)
        archive, arFits,arRes = checkArchive(archive, arFits,arRes, nAr, M)
        #GbestPositon = getGBest(X, fitness, archive, arFits, M)
        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(archive.shape[0] * OppoFactor))
            if EliteNumber > 0:
                EliteIdx = np.sort(np.array(random.sample(range(0,archive.shape[0]),EliteNumber)))
                XElite = copy.copy(archive[EliteIdx, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)
                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CalculateFitnessMP(XOppo, fun,n_obj,n_jobs, args)
                for i in range(len(EliteIdx)):
                    notsameflag = np.sum(np.sum(XOppo[i] == archive, axis=1) == XOppo.shape[1])
                    isDom1 = fitOppo[i] < arFits[EliteIdx[i]]
                    isDom2 = fitOppo[i] <= arFits[EliteIdx[i]]
                    isDom = sum(isDom1) >= 1 and sum(isDom2) == n_obj
                    if isDom and notsameflag == 0:
                        archive[EliteIdx[i], :] = XOppo[i, :]
                        arFits[EliteIdx[i], :] = fitOppo[i, :]
                X2file = np.concatenate([X2file, XOppo], axis=0)
                fitness2file = np.concatenate([fitness2file, fitOppo], axis=0)
                res2file = np.concatenate([res2file, resOppo], axis=0)
        GbestPositon = getGBest(X, fitness, archive, arFits, M)

        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)
        Progress_Bar.update(len(hf))

        record = save.MOswarmRecord(pop = pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,X=X,iteration=t+1,n_obj=n_obj,
                                    Pbest=Pbest,GbestPositon=GbestPositon,V=V,fitnessPbest=fitnessPbest,fitnessGbest=fitnessGbest,
                                    archive=archive,arFits=arFits,arRes=arRes)
        record.save()

    paretoPops, paretoFits, paretoRes = getNonDominationPops(archive, arFits, arRes)
    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
        paretoRes = Reslib.ResultDataPackage(l_result=paretoRes, method_info="Pareto front")
    raw_saver = save.RawDataSaver(hs, hf, hr, paretoFits=paretoFits, paretoPops=paretoPops,paretoRes=paretoRes,OptType="MO-SWARM")
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return paretoPops, paretoFits, paretoRes

