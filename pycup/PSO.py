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

VFactor = 0.3 # velocity factor, users can modify this to control the velocity ranges. Default 0.1 * parameter ranges
EliteOppoSwitch = True # Switch for elite opposition based learning
OppoFactor = 0.1 # proportion of samples to do the elite opposition operation
Sampling = "LHS"
BorderCheckMethod = "rebound"

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


def initial_MV(pop, dims, ubs, lbs):
    """
    lhs sampling based initialization for multi-variable functions

    :argument
    pop: population size -> int
    dims: num. parameters list -> [int, ..., int]
    ub: upper boundaries list -> [np.array, ..., np.array]
    lb: lower boundary list -> [np.array, ..., np.array]

    :returns
    Xs: a list of the updated samples/solutions
    lbs: a list of upper boundaries
    ubs: a lower boundaries
    """
    try:
        Xs, lbs, ubs = eval("sampling.{}_samplingMV".format(Sampling))(pop=pop, dims=dims, ubs=ubs, lbs=lbs)
    except:
        raise KeyError("The selectable sampling strategies are: 'LHS','Random','Chebyshev','Circle','Logistic','Piecewise','Sine','Singer','Sinusoidal','Tent'.")

    return Xs, lbs, ubs


def run(pop,dim,lb,ub,MaxIter,fun,Vmin=None,Vmax=None,RecordPath = None,args=()):
    """
    Main function for the algorithm

    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    MaxIter: num. of iterations. int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    Vmin: lower particle velocity boundary -> np.array, default value = -0.1 * (ub - lb), users can define it through
          this argument or modify the pycup.PSO.VFactor
    Vmax: upper particle velocity boundary -> np.array, default value =  0.1 * (ub - lb), users can define it through
          this argument or modify the pycup.PSO.VFactor
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    GbestScore: The best fitness obtained by the algorithm.
    GbestPositon: The sample which obtained the best fitness.
    Curve: The optimization curve
    hs: Historical samples.
    hf: The fitness of historical samples.
    hr: The results of historical samples.

    Reference:
    Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
    Knowledge-Based Systems, 89, 228–249. https://doi.org/10.1016/j.knosys.2015.07.006

    Usage:
    import pycup as cp

    def uni_fun1(X):
        # X for example np.array([1,2,3,...,30])
        fitness = np.sum(np.power(X,2)) + 1 # example: 1.2
        result = fitness.reshape(1,-1) # example ([1.2,])
        return fitness,result

    lb = -100 * np.ones(30)
    ub = 100 * np.ones(30)
    cp.PSO.VFactor = 0.2
    cp.PSO.run(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1)
    """
    print("Current Algorithm: PSO")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    checker = BorderChecker(method=BorderCheckMethod)
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
        fitness,res = CalculateFitness(X,fun,1,args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        fitness,sortIndex = SortFitness(fitness)
        X = SortPosition(X,sortIndex)
        GbestScore = copy.copy(fitness[0])
        GbestPosition = copy.copy(X[0,:])
        Curve = np.zeros([MaxIter,1])
        Pbest = copy.copy(X)
        fitnessPbest = copy.copy(fitness)
        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                        Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()

    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        Pbest = record.Pbest
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
        fitnessPbest = record.fitnessPbest
        V = record.V
        Curve = record.Curve
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

        for j in range(pop):

            V[j,:] = w*V[j,:] + c1*np.random.random()*(Pbest[j, :] - X[j, :]) + c2*np.random.random()*(GbestPosition - X[j, :])

            for ii in range(dim):
               if V[j, ii] < Vmin[ii]:
                   V[j, ii] = Vmin[ii]
               if V[j, ii] > Vmax[ii]:
                   V[j, ii] = Vmax[ii]

            X[j, :] = X[j, :] + V[j, :]


        X = checker.BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitness(X, fun,1, args)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        Pbest = SortPosition(Pbest,sortIndex)
        fitnessPbest = SortPosition(fitnessPbest,sortIndex)
        if not Reslib.UseResObject:
            res = SortPosition(res, sortIndex)
        else:
            res = res[sortIndex].flatten()

        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(X.shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = copy.copy(X[0:EliteNumber, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)
                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = checker.BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CalculateFitness(XOppo, fun,1, args)
                for j in range(EliteNumber):
                    if fitOppo[j] < fitness[j]:
                        fitness[j] = copy.copy(fitOppo[j])
                        X[j, :] = copy.copy(XOppo[j, :])
                fitness, sortIndex = SortFitness(fitness)
                X = SortPosition(X, sortIndex)
                Pbest = SortPosition(Pbest, sortIndex)
                fitnessPbest = SortPosition(fitnessPbest, sortIndex)
                X2file = np.concatenate([X2file,XOppo],axis=0)
                fitness2file = np.concatenate([fitness2file,fitOppo],axis=0)
                res2file = np.concatenate([res2file,resOppo],axis=0)

        for j in range(pop):
            if fitness[j] < fitnessPbest[j]:
                Pbest[j, :] = copy.copy(X[j, :])
                fitnessPbest[j] = copy.copy(fitness[j])
            if fitness[j] < GbestScore[0]:
                GbestScore[0] = copy.copy(fitness[j])
                GbestPosition = copy.copy(X[j, :])


        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)
        Progress_Bar.update(len(hf))
        Curve[t] = GbestScore

        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                         Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()

    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")

    raw_saver = save.RawDataSaver(hs, hf, hr, GbestScore, GbestPosition, Curve)
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hs, hf, hr


def runMP(pop, dim, lb, ub, MaxIter, fun,n_jobs, Vmin = None, Vmax = None,RecordPath = None, args=()):
    """
    Main function for the algorithm (multi-processing version)

    :argument
    pop: population size -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    MaxIter: num. of iterations. int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: num. of threads/processes -> int
    Vmin: lower particle velocity boundary -> np.array, default value = -0.1 * (ub - lb), users can define it through
          this argument or modify the pycup.PSO.VFactor
    Vmax: upper particle velocity boundary -> np.array, default value =  0.1 * (ub - lb), users can define it through
          this argument or modify the pycup.PSO.VFactor
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    GbestScore: The best fitness obtained by the algorithm.
    GbestPositon: The sample which obtained the best fitness.
    Curve: The optimization curve
    hs: Historical samples.
    hf: The fitness of historical samples.
    hr: The results of historical samples.

    Usage:
    import pycup as cp

    def uni_fun1(X):
        # X for example np.array([1,2,3,...,30])
        fitness = np.sum(np.power(X,2)) + 1 # example: 1.2
        result = fitness.reshape(1,-1) # example ([1.2,])
        return fitness,result

    lb = -100 * np.ones(30)
    ub = 100 * np.ones(30)
    cp.PSO.VFactor = 0.2
    cp.PSO.runMP(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1, n_jobs = 5)
    """
    print("Current Algorithm: PSO (Multi-Processing)")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dim))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    checker = BorderChecker(method=BorderCheckMethod)
    if not Vmin or not Vmax:
        Vmin = -VFactor * (ub-lb)
        Vmax = VFactor * (ub-lb)
    if not RecordPath:
        iter = 0
        hs = []
        hf = []
        hr = []

        X, lb, ub = initial(pop, dim, ub, lb)
        V, Vmin, Vmax = initial(pop, dim, Vmax, Vmin)
        fitness, res = CalculateFitnessMP(X, fun,1, n_jobs, args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        GbestScore = copy.copy(fitness[0])
        GbestPosition = copy.copy(X[0, :])
        Curve = np.zeros([MaxIter, 1])
        Pbest = copy.copy(X)
        fitnessPbest = copy.copy(fitness)
        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                        Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        Pbest = record.Pbest
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
        fitnessPbest = record.fitnessPbest
        V = record.V
        Curve = record.Curve
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

        for j in range(pop):

            V[j,:] = w*V[j,:] + c1*np.random.random()*(Pbest[j, :] - X[j, :]) + c2*np.random.random()*(GbestPosition - X[j, :])

            for ii in range(dim):
               if V[j, ii] < Vmin[ii]:
                   V[j, ii] = Vmin[ii]
               if V[j, ii] > Vmax[ii]:
                   V[j, ii] = Vmax[ii]

            X[j, :] = X[j, :] + V[j, :]


        X = checker.BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitnessMP(X, fun,1,n_jobs, args)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        Pbest = SortPosition(Pbest,sortIndex)
        fitnessPbest = SortPosition(fitnessPbest,sortIndex)
        if not Reslib.UseResObject:
            res = SortPosition(res, sortIndex)
        else:
            res = res[sortIndex].flatten()

        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(X.shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = copy.copy(X[0:EliteNumber, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)

                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = checker.BorderCheck(XOppo,ub,lb,EliteNumber,dim)
                fitOppo,resOppo = CalculateFitnessMP(XOppo,fun,1,n_jobs,args)
                for j in range(EliteNumber):
                    if fitOppo[j] < fitness[j]:
                        fitness[j] = copy.copy(fitOppo[j])
                        X[j, :] = copy.copy(XOppo[j,:])
                fitness,sortIndex = SortFitness(fitness)
                X = SortPosition(X, sortIndex)
                Pbest = SortPosition(Pbest, sortIndex)
                fitnessPbest = SortPosition(fitnessPbest, sortIndex)
                X2file = np.concatenate([X2file,XOppo],axis=0)
                fitness2file = np.concatenate([fitness2file,fitOppo],axis=0)
                res2file = np.concatenate([res2file,resOppo],axis=0)

        for j in range(pop):
            if fitness[j] < fitnessPbest[j]:
                Pbest[j, :] = copy.copy(X[j, :])
                fitnessPbest[j] = copy.copy(fitness[j])
            if fitness[j] < GbestScore[0]:
                GbestScore[0] = copy.copy(fitness[j])
                GbestPosition = copy.copy(X[j, :])

        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)
        Curve[t] = GbestScore
        Progress_Bar.update(len(hf))

        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                        Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()

    print("")  # for progress bar

    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")

    raw_saver = save.RawDataSaver(hs, hf, hr, GbestScore, GbestPosition, Curve)
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hs, hf, hr


def run_MV(pop,dims,lbs,ubs,MaxIter,fun,Vmin=None,Vmax=None,RecordPath = None,args=()):
    """
    Main function for the algorithm (multi-variable version)
    See the document for more information.

    :argument
    pop: population size -> int
    dims: num. parameters list -> [int, ..., int]
    ubs: upper boundaries list -> [np.array, ..., np.array]
    lbs: lower boundary list -> [np.array, ..., np.array]
    Vmin: list of lower particle velocity boundaries -> list, default element value = -0.1 * (ub - lb), users can define
          it through this argument or modify the pycup.PSO.VFactor
    Vmax: list of upper particle velocity boundaries -> list, default element value =  0.1 * (ub - lb), users can define
          it through this argument or modify the pycup.PSO.VFactor
    MaxIter: num. of iterations. int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    GbestScore: List of the best fitness obtained by the algorithm.
    GbestPositon: List of the sample which obtained the best fitness.
    Curve: List of the optimization curve
    hss: List of Historical samples.
    hfs: List of The fitness of historical samples.
    hrs: List of the results of historical samples.
    """
    print("Current Algorithm: PSO  (Multi-Variables)")
    print("Num. Variables: {}".format(len(dims)))
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dims))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lbs))
    print("Upper Bnd.:{}".format(ubs))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    checker = BorderChecker(method=BorderCheckMethod)
    num_var = len(dims)
    if not Vmin or not Vmax:
        Vmin = [-VFactor * (ubs[n] - lbs[n]) for n in range(num_var)]
        Vmax = [VFactor * (ubs[n] - lbs[n]) for n in range(num_var)]
    if not RecordPath:
        iter = 0

        hss = [[] for i in range(num_var)]
        hfs = [[] for i in range(num_var)]
        hrs = [[] for i in range(num_var)]

        X,lbs,ubs = initial_MV(pop, dims, ubs, lbs)
        V,Vmin,Vmax = initial_MV(pop, dims, Vmax, Vmin)
        fitness,res = CalculateFitness_MV(X,fun,args)
        for i in range(num_var):
            hrs[i].append(res[i])
            hss[i].append(copy.copy(X[i]))
            hfs[i].append(copy.copy(fitness[i]))
        Progress_Bar.update(len(hfs[0]))
        for n in range(num_var):
            fitness[n],sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n],sortIndex)
        GbestScore = [copy.copy(fitness[n][0]) for n in range(num_var)]
        GbestPosition = [copy.copy(X[n][0,:]) for n in range(num_var)]
        Curve = [np.zeros([MaxIter, 1]) for i in range(num_var)]
        Pbest = [copy.copy(X[n]) for n in range(num_var)]

        fitnessPbest = [copy.copy(fitness[n]) for n in range(num_var)]
        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                        Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()

    else:
        record = save.SwarmRecord.load(RecordPath)
        hss = record.hs
        hfs = record.hf
        hrs = record.hr
        X = record.X
        Pbest = record.Pbest
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
        fitnessPbest = record.fitnessPbest
        V = record.V
        Curve = record.Curve
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        same = record_checkMV(pop,dims,lbs,ubs,a_pop,a_dim,a_lb,a_ub)
        if not same:
            raise ValueError("The pop, dim, lb, and ub should be same as the record")
        Progress_Bar.update(len(hfs[0]))

    for t in range(iter,MaxIter):

        for n in range(num_var):
            for j in range(pop):

                V[n][j,:] = w*V[n][j,:] + c1*np.random.random()*(Pbest[n][j, :] - X[n][j, :]) + c2*np.random.random()*(GbestPosition[n] - X[n][j, :])

                for ii in range(dims[n]):
                   if V[n][j, ii] < Vmin[n][ii]:
                       V[n][j, ii] = Vmin[n][ii]
                   if V[n][j, ii] > Vmax[n][ii]:
                       V[n][j, ii] = Vmax[n][ii]

                X[n][j, :] = X[n][j, :] + V[n][j, :]


        for n in range(num_var):
            X[n] = checker.BorderCheck(X[n], ubs[n], lbs[n], pop, dims[n])
        fitness, res = CalculateFitness_MV(X, fun, args)
        for n in range(num_var):
            fitness[n], sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n], sortIndex)
            Pbest[n] = SortPosition(Pbest[n],sortIndex)
            fitnessPbest[n] = SortPosition(fitnessPbest[n],sortIndex)
            if not Reslib.UseResObject:
                res[n] = SortPosition(res[n], sortIndex)
            else:
                res[n] = res[n][sortIndex].flatten()

        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(pop * OppoFactor))
            if EliteNumber > 0:
                XElite = [copy.copy(X[n][0:EliteNumber, :]) for n in range(num_var)]
                Tlb = [np.min(XElite[n], 0) for n in range(num_var)]
                Tub = [np.max(XElite[n], 0) for n in range(num_var)]

                XOppo = [np.array([random.random() * (Tlb[n] + Tub[n]) - XElite[n][j, :] for j in range(EliteNumber)]) for n in range(num_var)]
                for n in range(num_var):
                    XOppo[n] = checker.BorderCheck(XOppo[n], ubs[n], lbs[n], EliteNumber, dims[n])
                fitOppo, resOppo = CalculateFitness_MV(XOppo, fun, args)
                for j in range(EliteNumber):
                    for n in range(num_var):
                        if fitOppo[n][j] < fitness[n][j]:
                            fitness[n][j] = copy.copy(fitOppo[n][j])
                            X[n][j, :] = copy.copy(XOppo[n][j, :])

                for n in range(num_var):
                    fitness[n], sortIndex = SortFitness(fitness[n])
                    X[n] = SortPosition(X[n], sortIndex)
                    Pbest[n] = SortPosition(Pbest[n], sortIndex)
                    fitnessPbest[n] = SortPosition(fitnessPbest[n], sortIndex)
                    X2file[n] = np.concatenate([X2file[n],XOppo[n]],axis=0)
                    fitness2file[n] = np.concatenate([fitness2file[n],fitOppo[n]],axis=0)
                    res2file[n] = np.concatenate([res2file[n],resOppo[n]],axis=0)

        for n in range(num_var):
            for j in range(pop):
                if fitness[n][j] < fitnessPbest[n][j]:
                    Pbest[n][j, :] = copy.copy(X[n][j, :])
                    fitnessPbest[n][j] = copy.copy(fitness[n][j])
                if fitness[n][j] < GbestScore[n][0]:
                    GbestScore[n][0] = copy.copy(fitness[n][j])
                    GbestPosition[n] = copy.copy(X[n][j, :])
            Curve[n][t] = GbestScore[n]

        for n in range(num_var):
            hrs[n].append(res2file[n])
            hss[n].append(X2file[n])
            hfs[n].append(fitness2file[n])
        Progress_Bar.update(len(hfs[0]))

        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                        Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()

    print("")  # for progress bar
    for n in range(num_var):

        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[n]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(n+1)
        if Reslib.UseResObject:
            hrs[n] = Reslib.ResultDataPackage(l_result=hrs[n], method_info="Algorithm")
        raw_saver = save.RawDataSaver(hss[n], hfs[n], hrs[n], GbestScore[n], GbestPosition[n], Curve[n])
        raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hss, hfs, hrs


def runMP_MV(pop,dims,lbs,ubs,MaxIter, fun,n_jobs,Vmin=None,Vmax=None,RecordPath = None,args=()):
    """
    Main function for the algorithm (multi-processing multi-variable version)
    See the document for more information.

    :argument
    pop: population size -> int
    dims: num. parameters list -> [int, ..., int]
    ubs: upper boundaries list -> [np.array, ..., np.array]
    lbs: lower boundary list -> [np.array, ..., np.array]
    Vmin: list of lower particle velocity boundaries -> list, default element value = -0.1 * (ub - lb), users can define
          it through this argument or modify the pycup.PSO.VFactor
    Vmax: list of upper particle velocity boundaries -> list, default element value =  0.1 * (ub - lb), users can define
          it through this argument or modify the pycup.PSO.VFactor
    MaxIter: num. of iterations. int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: num. of threads/processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    GbestScore: List of the best fitness obtained by the algorithm.
    GbestPositon: List of the sample which obtained the best fitness.
    Curve: List of the optimization curve
    hss: List of Historical samples.
    hfs: List of The fitness of historical samples.
    hrs: List of the results of historical samples.
    """
    print("Current Algorithm: PSO  (Multi-Processing-Multi-Variables)")
    print("Num. Variables: {}".format(len(dims)))
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dims))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lbs))
    print("Upper Bnd.:{}".format(ubs))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    checker = BorderChecker(method=BorderCheckMethod)
    num_var = len(dims)
    if not Vmin or not Vmax:
        Vmin = [-VFactor * (ubs[n]-lbs[n]) for n in range(num_var)]
        Vmax = [VFactor * (ubs[n]-lbs[n]) for n in range(num_var)]
    if not RecordPath:
        iter = 0

        hss = [[] for i in range(num_var)]
        hfs = [[] for i in range(num_var)]
        hrs = [[] for i in range(num_var)]


        X,lbs,ubs = initial_MV(pop, dims, ubs, lbs)
        V,Vmin,Vmax = initial_MV(pop, dims, Vmax, Vmin)
        fitness,res = CalculateFitnessMP_MV(X,fun,n_jobs,args)
        for i in range(num_var):
            hrs[i].append(res[i])
            hss[i].append(copy.copy(X[i]))
            hfs[i].append(copy.copy(fitness[i]))
        Progress_Bar.update(len(hfs[0]))
        for n in range(num_var):
            fitness[n],sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n],sortIndex)
        GbestScore = [copy.copy(fitness[n][0]) for n in range(num_var)]
        GbestPosition = [copy.copy(X[n][0,:]) for n in range(num_var)]
        Curve = [np.zeros([MaxIter, 1]) for i in range(num_var)]
        Pbest = [copy.copy(X[n]) for n in range(num_var)]

        fitnessPbest = [copy.copy(fitness[n]) for n in range(num_var)]
        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                        Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()

    else:
        record = save.SwarmRecord.load(RecordPath)
        hss = record.hs
        hfs = record.hf
        hrs = record.hr
        X = record.X
        Pbest = record.Pbest
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
        fitnessPbest = record.fitnessPbest
        V = record.V
        Curve = record.Curve
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        same = record_checkMV(pop,dims,lbs,ubs,a_pop,a_dim,a_lb,a_ub)
        if not same:
            raise ValueError("The pop, dim, lb, and ub should be same as the record")
        Progress_Bar.update(len(hfs[0]))

    for t in range(iter,MaxIter):
        for n in range(num_var):
            for j in range(pop):

                V[n][j,:] = w*V[n][j,:] + c1*np.random.random()*(Pbest[n][j, :] - X[n][j, :]) + c2*np.random.random()*(GbestPosition[n] - X[n][j, :])

                for ii in range(dims[n]):
                   if V[n][j, ii] < Vmin[n][ii]:
                       V[n][j, ii] = Vmin[n][ii]
                   if V[n][j, ii] > Vmax[n][ii]:
                       V[n][j, ii] = Vmax[n][ii]

                X[n][j, :] = X[n][j, :] + V[n][j, :]


        for n in range(num_var):
            X[n] = checker.BorderCheck(X[n], ubs[n], lbs[n], pop, dims[n])
        fitness, res = CalculateFitnessMP_MV(X, fun,n_jobs, args)
        for n in range(num_var):
            fitness[n], sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n], sortIndex)
            Pbest[n] = SortPosition(Pbest[n],sortIndex)
            fitnessPbest[n] = SortPosition(fitnessPbest[n],sortIndex)
            if not Reslib.UseResObject:
                res[n] = SortPosition(res[n], sortIndex)
            else:
                res[n] = res[n][sortIndex].flatten()

        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(pop * OppoFactor))
            if EliteNumber > 0:
                XElite = [copy.copy(X[n][0:EliteNumber, :]) for n in range(num_var)]
                Tlb = [np.min(XElite[n], 0) for n in range(num_var)]
                Tub = [np.max(XElite[n], 0) for n in range(num_var)]

                XOppo = [np.array([random.random() * (Tlb[n] + Tub[n]) - XElite[n][j, :] for j in range(EliteNumber)]) for n in range(num_var)]
                for n in range(num_var):
                    XOppo[n] = checker.BorderCheck(XOppo[n], ubs[n], lbs[n], EliteNumber, dims[n])
                fitOppo, resOppo = CalculateFitnessMP_MV(XOppo, fun,n_jobs, args)
                for j in range(EliteNumber):
                    for n in range(num_var):
                        if fitOppo[n][j] < fitness[n][j]:
                            fitness[n][j] = copy.copy(fitOppo[n][j])
                            X[n][j, :] = copy.copy(XOppo[n][j, :])

                for n in range(num_var):
                    fitness[n], sortIndex = SortFitness(fitness[n])
                    X[n] = SortPosition(X[n], sortIndex)
                    Pbest[n] = SortPosition(Pbest[n], sortIndex)
                    fitnessPbest[n] = SortPosition(fitnessPbest[n], sortIndex)
                    X2file[n] = np.concatenate([X2file[n],XOppo[n]],axis=0)
                    fitness2file[n] = np.concatenate([fitness2file[n],fitOppo[n]],axis=0)
                    res2file[n] = np.concatenate([res2file[n],resOppo[n]],axis=0)

        for n in range(num_var):
            for j in range(pop):
                if fitness[n][j] < fitnessPbest[n][j]:
                    Pbest[n][j, :] = copy.copy(X[n][j, :])
                    fitnessPbest[n][j] = copy.copy(fitness[n][j])
                if fitness[n][j] < GbestScore[n][0]:
                    GbestScore[n][0] = copy.copy(fitness[n][j])
                    GbestPosition[n] = copy.copy(X[n][j, :])
            Curve[n][t] = GbestScore[n]

        for n in range(num_var):
            hrs[n].append(res2file[n])
            hss[n].append(X2file[n])
            hfs[n].append(fitness2file[n])
        Progress_Bar.update(len(hfs[0]))

        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                        Pbest=Pbest,fitnessPbest=fitnessPbest,V=V)
        record.save()

    print("")  # for progress bar
    for n in range(num_var):

        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[n]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(n+1)
        if Reslib.UseResObject:
            hrs[n] = Reslib.ResultDataPackage(l_result=hrs[n], method_info="Algorithm")
        raw_saver = save.RawDataSaver(hss[n], hfs[n], hrs[n], GbestScore[n], GbestPosition[n], Curve[n])
        raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hss, hfs, hrs






