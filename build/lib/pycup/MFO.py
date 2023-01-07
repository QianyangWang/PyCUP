import numpy as np
import random
import copy
from . import sampling
from . import save
from . import multi_jobs
import math
from . import progress_bar
from . import Reslib
from .calc_utils import BorderCheck,SortFitness,SortPosition,check_listitem,record_check,record_checkMV
from .calc_utils import CalculateFitness,CalculateFitnessMP,CalculateFitness_MV,CalculateFitnessMP_MV

Sampling = "LHS"
EliteOppoSwitch = True
OppoFactor = 0.1


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


def run(pop, dim, lb, ub, MaxIter,  fun,RecordPath = None, args=()):
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

    Notes:
    The pseudo code in the original paper seems to has a little when updating the flame. Here we concatenate the moth
    and flame and sort them to update the flame. By doing this, the global optimal position can always be stored in the
    flame population.

    Usage:
    import pycup as cp

    def uni_fun1(X):
        # X for example np.array([1,2,3,...,30])
        fitness = np.sum(np.power(X,2)) + 1 # example: 1.2
        result = fitness.reshape(1,-1) # example ([1.2,])
        return fitness,result

    lb = -100 * np.ones(30)
    ub = 100 * np.ones(30)
    cp.MFO.run(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1)
    """
    print("Current Algorithm: MFO")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
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
        X, lb, ub = initial(pop, dim, ub, lb)
        fitness,res = CalculateFitness(X, fun,1, args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        #X_previous = copy.copy(X)
        #fitness_previous = copy.copy(fitness)
        #res_previous = copy.copy(res)
        fitnessS, sortIndex = SortFitness(fitness)
        # in the first iteration, the flame is the sorted population
        Xs = SortPosition(X, sortIndex)
        GbestScore = copy.copy(fitnessS[0])
        GbestPosition = np.zeros([1, dim])
        GbestPosition[0, :] = copy.copy(Xs[0, :])
        Curve = np.zeros([MaxIter, 1])
        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        Xs = record.Xs
        fitnessS = record.fitnessS
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
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


        X = BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitness(X, fun,1, args)

        # merge the flame and moth
        fitnessM = np.concatenate([fitnessS,fitness],axis=0)
        Xm = np.concatenate([Xs,X],axis=0)
        fitnessM, sortIndexM = SortFitness(fitnessM)
        Xm = SortPosition(Xm, sortIndexM)
        # update the Xs (flame population)
        Xs = Xm[0:Flame_no,:]
        fitnessS = fitnessM[0:Flame_no]

        """
        Here I used the elite sample to replace the sample in X for saving, because that if 
        fitOppo[j] < fitnessS[j] (the elite is better than the flame), it is better than any
        of the sample in X. This will not affect the actual search process.
        """
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
                XOppo = BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CalculateFitness(XOppo, fun,1, args)
                for j in range(EliteNumber):
                    if fitOppo[j] < fitnessS[j]:
                        fitnessS[j] = copy.copy(fitOppo[j])
                        Xs[j, :] = copy.copy(XOppo[j, :])

                fitnessS, sortIndex = SortFitness(fitnessS)
                Xs = SortPosition(Xs, sortIndex)
                X2file = np.concatenate([X2file, XOppo], axis=0)
                fitness2file = np.concatenate([fitness2file, fitOppo], axis=0)
                res2file = np.concatenate([res2file, resOppo], axis=0)

        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)


        if fitnessS[0] <= GbestScore:
            GbestScore = copy.copy(fitnessS[0])
            GbestPosition[0, :] = copy.copy(Xs[0, :])
        Curve[t] = GbestScore
        Progress_Bar.update(len(hf))

        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()

    print("")  # for progress bar
    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
    raw_saver = save.RawDataSaver(hs, hf, hr, GbestScore, GbestPosition, Curve)
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")
    return GbestScore, GbestPosition, Curve, hs, hf, hr


def runMP(pop, dim, lb, ub, MaxIter, fun, n_jobs,RecordPath = None, args=()):
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
    n_jobs: number of threads/processes -> int
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
    cp.MFO.runMP(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1, n_jobs = 5)
    """
    print("Current Algorithm: MFO")
    print("Elite Opposition:{}".format(EliteOppoSwitch))
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

        X, lb, ub = initial(pop, dim, ub, lb)
        fitness, res = CalculateFitnessMP(X, fun,1,n_jobs, args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        fitnessS, sortIndex = SortFitness(fitness)
        Xs = SortPosition(X, sortIndex)
        GbestScore = copy.copy(fitnessS[0])
        GbestPosition = np.zeros([1, dim])
        GbestPosition[0, :] = copy.copy(Xs[0, :])
        Curve = np.zeros([MaxIter, 1])
        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        Xs = record.Xs
        fitnessS = record.fitnessS
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
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

        Flame_no = round(pop - t * ((pop - 1) / MaxIter))
        a = -1 + t * (-1) / MaxIter
        for i in range(pop):
            for j in range(dim):
                if i <= Flame_no - 1:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1

                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[i, j]
                else:
                    distance_to_flame = np.abs(Xs[Flame_no - 1, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1
                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[Flame_no - 1, j]

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness, res = CalculateFitnessMP(X, fun,1,n_jobs, args)
        fitnessM = np.concatenate([fitnessS, fitness], axis=0)
        Xm = np.concatenate([Xs, X], axis=0)
        fitnessM, sortIndexM = SortFitness(fitnessM)
        Xm = SortPosition(Xm, sortIndexM)
        # update the Xs
        Xs = Xm[0:Flame_no, :]
        fitnessS = fitnessM[0:Flame_no]

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
                XOppo = BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CalculateFitnessMP(XOppo, fun,1,n_jobs, args)
                for j in range(EliteNumber):
                    if fitOppo[j] < fitnessS[j]:
                        fitnessS[j] = copy.copy(fitOppo[j])
                        Xs[j, :] = copy.copy(XOppo[j, :])

                fitnessS, sortIndex = SortFitness(fitnessS)
                Xs = SortPosition(Xs, sortIndex)
                X2file = np.concatenate([X2file, XOppo], axis=0)
                fitness2file = np.concatenate([fitness2file, fitOppo], axis=0)
                res2file = np.concatenate([res2file, resOppo], axis=0)

        hr.append(res2file)
        hs.append(X2file)
        hf.append(fitness2file)

        if fitnessS[0] <= GbestScore:
            GbestScore = copy.copy(fitnessS[0])
            GbestPosition[0, :] = copy.copy(Xs[0, :])
        Curve[t] = GbestScore
        Progress_Bar.update(len(hf))

        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()

    print("")  # for progress bar
    if Reslib.UseResObject:
        hr = Reslib.ResultDataPackage(l_result=hr,method_info="Algorithm")
    raw_saver = save.RawDataSaver(hs, hf, hr, GbestScore, GbestPosition, Curve)
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")
    return GbestScore, GbestPosition, Curve, hs, hf, hr


def run_MV(pop, dims, lbs, ubs, MaxIter, fun,RecordPath = None, args=()):
    """
    Main function for the algorithm (multi-variable version)
    See the document for more information.

    :argument
    pop: population size -> int
    dims: num. parameters list -> [int, ..., int]
    ubs: upper boundaries list -> [np.array, ..., np.array]
    lbs: lower boundary list -> [np.array, ..., np.array]
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
    print("Current Algorithm: MFO (Multi-Variables)")
    print("Num. Variables: {}".format(len(dims)))
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dims))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lbs))
    print("Upper Bnd.:{}".format(ubs))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    num_var = len(dims)
    if not RecordPath:
        iter = 0
        hss = [[] for i in range(num_var)]
        hfs = [[] for i in range(num_var)]
        hrs = [[] for i in range(num_var)]

        X, lbs, ubs = initial_MV(pop, dims, ubs, lbs)
        fitness,res = CalculateFitness_MV(X, fun, args)
        Xs = copy.copy(X)
        fitnessS = copy.copy(fitness)
        for i in range(num_var):
            hrs[i].append(res[i])
            hss[i].append(copy.copy(X[i]))
            hfs[i].append(copy.copy(fitness[i]))
        Progress_Bar.update(len(hfs[0]))

        for n in range(num_var):
            fitnessS[n], sortIndex = SortFitness(fitnessS[n])
            Xs[n] = SortPosition(Xs[n], sortIndex)

        GbestScore = [copy.copy(fitnessS[n][0]) for n in range(num_var)]
        GbestPosition = [np.zeros([1, dims[n]]) for n in range(num_var)]
        for n in range(num_var):
            GbestPosition[n][0, :] = copy.copy(Xs[n][0, :])
        Curve = [np.zeros([MaxIter, 1]) for i in range(num_var)]
        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hss = record.hs
        hfs = record.hf
        hrs = record.hr
        X = record.X
        Xs = record.Xs
        fitnessS = record.fitnessS
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
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

        Flame_no = round(pop - t * ((pop - 1) / MaxIter))
        a = -1 + t * (-1) / MaxIter
        for n in range(num_var):
            for i in range(pop):
                for j in range(dims[n]):
                    if i <= Flame_no-1:
                        distance_to_flame = np.abs(Xs[n][i, j] - X[n][i, j])
                        b = 1
                        r = (a - 1) * random.random() + 1

                        X[n][i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[n][i, j]
                    else:
                        distance_to_flame = np.abs(Xs[n][Flame_no-1, j] - X[n][i, j])
                        b = 1
                        r = (a - 1) * random.random() + 1
                        X[n][i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[n][Flame_no-1, j]

        for n in range(num_var):
            X[n] = BorderCheck(X[n], ubs[n], lbs[n], pop, dims[n])
        fitness, res = CalculateFitness_MV(X, fun, args)

        for n in range(num_var):
            fitnessM = np.concatenate([fitnessS[n],fitness[n]],axis=0)
            Xm = np.concatenate([Xs[n],X[n]],axis=0)
            fitnessM, sortIndexM = SortFitness(fitnessM)
            Xm = SortPosition(Xm, sortIndexM)
            # update the Xs
            Xs[n] = Xm[0:Flame_no,:]
            fitnessS[n] = fitnessM[0:Flame_no]

        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(Xs[0].shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = [copy.copy(Xs[n][0:EliteNumber, :]) for n in range(num_var)]
                Tlb = [np.min(XElite[n], 0) for n in range(num_var)]
                Tub = [np.max(XElite[n], 0) for n in range(num_var)]

                XOppo = [np.array([random.random() * (Tlb[n] + Tub[n]) - XElite[n][j, :] for j in range(EliteNumber)]) for n in range(num_var)]
                for n in range(num_var):
                    XOppo[n] = BorderCheck(XOppo[n], ubs[n], lbs[n], EliteNumber, dims[n])
                fitOppo, resOppo = CalculateFitness_MV(XOppo, fun, args)
                for j in range(EliteNumber):
                    for n in range(num_var):
                        if fitOppo[n][j] < fitness[n][j]:
                            fitnessS[n][j] = copy.copy(fitOppo[n][j])
                            Xs[n][j, :] = copy.copy(XOppo[n][j, :])

                for n in range(num_var):
                    fitnessS[n], sortIndex = SortFitness(fitnessS[n])
                    Xs[n] = SortPosition(Xs[n], sortIndex)
                    X2file[n] = np.concatenate([X2file[n],XOppo[n]],axis=0)
                    fitness2file[n] = np.concatenate([fitness2file[n],fitOppo[n]],axis=0)
                    res2file[n] = np.concatenate([res2file[n],resOppo[n]],axis=0)

        for n in range(num_var):
            hrs[n].append(res2file[n])
            hss[n].append(X2file[n])
            hfs[n].append(fitness2file[n])

        for n in range(num_var):
            if fitnessS[n][0] <= GbestScore[n]:
                GbestScore[n] = copy.copy(fitnessS[n][0])
                GbestPosition[n][0, :] = copy.copy(Xs[n][0, :])
            Curve[n][t] = GbestScore[n]
        Progress_Bar.update(len(hfs[0]))

        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()

    print("")  # for progress bar
    for n in range(num_var):

        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[n]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(n + 1)
        if Reslib.UseResObject:
            hrs[n] = Reslib.ResultDataPackage(l_result=hrs[n], method_info="Algorithm")
        raw_saver = save.RawDataSaver(hss[n], hfs[n], hrs[n], GbestScore[n], GbestPosition[n], Curve[n])
        raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hss, hfs, hrs


def runMP_MV(pop, dims, lbs, ubs, MaxIter, fun,n_jobs,RecordPath = None, args=()):
    """
    Main function for the algorithm (multi-processing multi-variable version)
    See the document for more information.

    :argument
    pop: population size -> int
    dims: num. parameters list -> [int, ..., int]
    ubs: upper boundaries list -> [np.array, ..., np.array]
    lbs: lower boundary list -> [np.array, ..., np.array]
    MaxIter: num. of iterations. int
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: number of threads/processes -> int
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
    print("Current Algorithm: MFO (Multi-Processing-Multi-Variables)")
    print("Num. Variables: {}".format(len(dims)))
    print("Elite Opposition:{}".format(EliteOppoSwitch))
    print("Iterations: 1 (init.) + {}".format(MaxIter))
    print("Dim:{}".format(dims))
    print("Population:{}".format(pop))
    print("Lower Bnd.:{}".format(lbs))
    print("Upper Bnd.:{}".format(ubs))
    Progress_Bar = progress_bar.ProgressBar(MaxIter + 1)
    num_var = len(dims)
    if not RecordPath:
        iter = 0
        hss = [[] for i in range(num_var)]
        hfs = [[] for i in range(num_var)]
        hrs = [[] for i in range(num_var)]
        X, lbs, ubs = initial_MV(pop, dims, ubs, lbs)
        fitness,res = CalculateFitnessMP_MV(X, fun,n_jobs, args)
        Xs = copy.copy(X)
        fitnessS = copy.copy(fitness)
        for i in range(num_var):
            hrs[i].append(res[i])
            hss[i].append(copy.copy(X[i]))
            hfs[i].append(copy.copy(fitness[i]))
        Progress_Bar.update(len(hfs[0]))

        for n in range(num_var):
            fitnessS[n], sortIndex = SortFitness(fitnessS[n])
            Xs[n] = SortPosition(Xs[n], sortIndex)

        GbestScore = [copy.copy(fitnessS[n][0]) for n in range(num_var)]
        GbestPosition = [np.zeros([1, dims[n]]) for n in range(num_var)]
        for n in range(num_var):
            GbestPosition[n][0, :] = copy.copy(Xs[n][0, :])
        Curve = [np.zeros([MaxIter, 1]) for i in range(num_var)]
        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hss = record.hs
        hfs = record.hf
        hrs = record.hr
        X = record.X
        Xs = record.Xs
        fitnessS = record.fitnessS
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
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

        Flame_no = round(pop - t * ((pop - 1) / MaxIter))
        a = -1 + t * (-1) / MaxIter
        for n in range(num_var):
            for i in range(pop):
                for j in range(dims[n]):
                    if i <= Flame_no-1:
                        distance_to_flame = np.abs(Xs[n][i, j] - X[n][i, j])
                        b = 1
                        r = (a - 1) * random.random() + 1

                        X[n][i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[n][i, j]
                    else:
                        distance_to_flame = np.abs(Xs[n][Flame_no-1, j] - X[n][i, j])
                        b = 1
                        r = (a - 1) * random.random() + 1
                        X[n][i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[n][Flame_no-1, j]

        for n in range(num_var):
            X[n] = BorderCheck(X[n], ubs[n], lbs[n], pop, dims[n])
        fitness, res = CalculateFitnessMP_MV(X, fun,n_jobs, args)

        for n in range(num_var):
            fitnessM = np.concatenate([fitnessS[n],fitness[n]],axis=0)
            Xm = np.concatenate([Xs[n],X[n]],axis=0)
            fitnessM, sortIndexM = SortFitness(fitnessM)
            Xm = SortPosition(Xm, sortIndexM)
            # update the Xs
            Xs[n] = Xm[0:Flame_no,:]
            fitnessS[n] = fitnessM[0:Flame_no]

        X2file = copy.copy(X)
        fitness2file = copy.copy(fitness)
        res2file = copy.copy(res)
        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(Xs[0].shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = [copy.copy(Xs[n][0:EliteNumber, :]) for n in range(num_var)]
                Tlb = [np.min(XElite[n], 0) for n in range(num_var)]
                Tub = [np.max(XElite[n], 0) for n in range(num_var)]

                XOppo = [np.array([random.random() * (Tlb[n] + Tub[n]) - XElite[n][j, :] for j in range(EliteNumber)]) for n in range(num_var)]
                for n in range(num_var):
                    XOppo[n] = BorderCheck(XOppo[n], ubs[n], lbs[n], EliteNumber, dims[n])
                fitOppo, resOppo = CalculateFitnessMP_MV(XOppo, fun,n_jobs, args)
                for j in range(EliteNumber):
                    for n in range(num_var):
                        if fitOppo[n][j] < fitness[n][j]:
                            fitnessS[n][j] = copy.copy(fitOppo[n][j])
                            Xs[n][j, :] = copy.copy(XOppo[n][j, :])

                for n in range(num_var):
                    fitnessS[n], sortIndex = SortFitness(fitnessS[n])
                    Xs[n] = SortPosition(Xs[n], sortIndex)
                    X2file[n] = np.concatenate([X2file[n],XOppo[n]],axis=0)
                    fitness2file[n] = np.concatenate([fitness2file[n],fitOppo[n]],axis=0)
                    res2file[n] = np.concatenate([res2file[n],resOppo[n]],axis=0)

        for n in range(num_var):
            hrs[n].append(res2file[n])
            hss[n].append(X2file[n])
            hfs[n].append(fitness2file[n])

        for n in range(num_var):
            if fitnessS[n][0] <= GbestScore[n]:
                GbestScore[n] = copy.copy(fitnessS[n][0])
                GbestPosition[n][0, :] = copy.copy(Xs[n][0, :])
            Curve[n][t] = GbestScore[n]
        Progress_Bar.update(len(hfs[0]))

        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1,
                         Xs=Xs,fitnessS=fitnessS)
        record.save()

    print("")  # for progress bar
    for n in range(num_var):

        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[n]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(n + 1)
        if Reslib.UseResObject:
            hrs[n] = Reslib.ResultDataPackage(l_result=hrs[n], method_info="Algorithm")
        raw_saver = save.RawDataSaver(hss[n], hfs[n], hrs[n], GbestScore[n], GbestPosition[n], Curve[n])
        raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hss, hfs, hrs