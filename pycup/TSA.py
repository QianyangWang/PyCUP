import numpy as np
import random
import copy
from . import sampling
from . import save
from . import multi_jobs
import math
from . import progress_bar

EliteOppoSwitch = True # Switch for elite opposition based learning
OppoFactor = 0.1 # proportion of samples to do the elite opposition operation
Sampling = "LHS"

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


def BorderCheck(X, ub, lb, pop, dim):
    """
    Border check function. If a solution is out of the given boundaries, it will be mandatorily
    moved to the boundary.

    :argument
    X: samples -> np.array, shape = (pop, dim)
    ub: upper boundaries list -> [np.array, ..., np.array]
    lb: lower boundary list -> [np.array, ..., np.array]
    pop: population size -> int
    dims: num. parameters list -> [int, ..., int]

    :return
    X: the updated samples
    """
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


def CaculateFitness(X,fun,args):
    """
    The fitness calculating function.

    :argument
    X: samples -> np.array, shape = (pop, dim)
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    fitness: The calculated fitness value.
    res_l: The simulation/calculation results after concatenate. -> np.array, shape = (pop, len(result))
           For a continuous simulation, the len(result) is equivalent to len(time series)
    """
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    res_l = []
    for i in range(pop):
        fitness[i],res = fun(X[i, :],*args)
        res_l.append(res)
    res_l = np.concatenate(res_l)
    return fitness,res_l


def CaculateFitnessMP(X, fun, n_jobs, args):
    """
    The fitness calculating function for multi-processing tasks.

    :argument
    X: samples -> np.array, shape = (pop, dim)
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: number of threads/processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.
    """
    fitness, res_l = multi_jobs.do_multi_jobs(func=fun, params=X,  n_process=n_jobs, args=args)

    return fitness,res_l


def CaculateFitness_MV(Xs, fun, args):
    """
    The fitness calculating function for multi-variable tasks.

    :argument
    Xs: a list of the updated samples/solutions
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    fitness: A list of calculated fitness values.
    res_l: A list of simulation/calculation results after concatenate.
    """
    num_var = len(Xs)
    pop = Xs[0].shape[0]
    fitness = []
    res_l = []
    temp_f = []
    temp_r = []
    for i in range(pop):
        sample = [Xs[j][i] for j in range(num_var)]
        fit, res = fun(sample, *args)
        res = [i.reshape(1, -1) for i in res]
        fit = [np.array(i).reshape(-1, 1) for i in fit]
        temp_f.append(fit)
        temp_r.append(res)

    for d in range(num_var):
        hf = [temp_f[i][d] for i in range(pop)] # -> [ [],[],[], ... ]
        hf = np.concatenate(hf,axis=0)
        hr = [temp_r[i][d] for i in range(pop)] # -> [ [,,,], [,,,], ... ]
        hr = np.concatenate(hr,axis=0)
        fitness.append(hf)
        res_l.append(hr)

    return fitness, res_l


def CaculateFitnessMP_MV(Xs, fun, n_jobs, args):
    """
    The fitness calculating function for multi-variable & multi-processing tasks.

    :argument
    Xs: a list of the updated samples/solutions
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: number of threads/processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    fitness: A list of calculated fitness values.
    res_l: A list of simulation/calculation results after concatenate.
    """
    fitness, res_l = multi_jobs.do_multi_jobsMV(func=fun, params=Xs, n_process=n_jobs, args=args)

    return fitness, res_l


def SortFitness(Fit):
    """
    Sort the fitness.

    :argument
    Fit: fitness array -> np.array, shape = (pop,1)

    :returns
    fitness: The sorted fitness.
    index: The corresponding sorted index.
    """
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    """
    Sort the sample according to the rank of fitness.

    :argument
    X: samples -> np.array, shape = (pop, dim)
    index: The index of sorted samples according to fitness values. -> np.array, shape = (pop,1)
    """
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def check_listitem(item1, item2):
    s_flags = [item1[i] == item2[i] for i in range(len(item1))]
    length = [len(item1[i]) for i in range(len(item1))]
    same = np.sum(s_flags) == np.sum(length)
    return same


def record_check(pop, dim, lb, ub, a_pop, a_dim, a_lb, a_ub):
    a = np.sum(lb == a_lb) == len(lb)
    b = np.sum(ub == a_ub) == len(ub)
    check_list = (pop == a_pop, dim == a_dim, a, b)
    if np.sum(check_list) == len(check_list):
        return True
    else:
        return False


def record_checkMV(pop, dims, lbs, ubs, a_pop, a_dims, a_lbs, a_ubs):
    p_sflage = pop == a_pop
    d_sflage = np.sum(np.array(dims) == np.array(a_dims)) == len(dims)
    lb_flag = check_listitem(lbs, a_lbs)
    ub_flag = check_listitem(ubs, a_ubs)
    if np.sum([p_sflage, d_sflage, lb_flag, ub_flag]) == 4:
        return True
    else:
        return False


def run(pop, dim, lb, ub, MaxIter, fun, RecordPath = None, args=()):
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
    Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate Swarm Algorithm:
    A new bio-inspired based metaheuristic paradigm for global optimization.
    Engineering Applications of Artificial Intelligence, 90, 103541. https://doi.org/10.1016/j.engappai.2020.103541

    Usage:
    import pycup as cp

    def uni_fun1(X):
        # X for example np.array([1,2,3,...,30])
        fitness = np.sum(np.power(X,2)) + 1 # example: 1.2
        result = fitness.reshape(1,-1) # example ([1.2,])
        return fitness,result

    lb = -100 * np.ones(30)
    ub = 100 * np.ones(30)
    cp.TSA.run(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1)
    """
    print("Current Algorithm: TSA")
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
        fitness,res = CaculateFitness(X, fun,args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        GbestScore = copy.copy(fitness[0])
        GbestPosition = np.zeros([1,dim])
        GbestPosition[0,:] = copy.copy(X[0,:])
        Curve = np.zeros([MaxIter, 1])
        Pos = np.zeros([pop,dim])
        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=0, Pos=Pos)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        Pos = record.Pos
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
        xmin = 1
        xmax = 4
        xr = xmin + random.random()*(xmax-xmin)
        xr = np.fix(xr)

        for i in range(pop):
            for j in range(dim):
                A1 = (random.random() + random.random()) - (2*random.random())/xr
                c2 = random.random()
                if i==0:
                    c3 = random.random()
                    d_pos = np.abs(GbestPosition[0,j] - c2*X[i,j])
                    if c3>=0.5:
                        Pos[i,j] = GbestPosition[0,j] + A1*d_pos
                    else:
                        Pos[i,j] = GbestPosition[0,j] - A1*d_pos
                else:
                    c3 = random.random()
                    d_pos = np.abs(GbestPosition[0,j] - c2*X[i,j])
                    if c3>=0.5:
                        Pos[i,j] = GbestPosition[0,j] + A1*d_pos
                    else:
                        Pos[i,j] = GbestPosition[0,j] - A1*d_pos
                    X[i,j] = (Pos[i,j] + X[i,j])/(2 + random.random())


        X = BorderCheck(X, ub, lb, pop, dim)
        fitness,res = CaculateFitness(X, fun,args)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        res = SortPosition(res, sortIndex)

        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(X.shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = copy.copy(X[0:EliteNumber, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)

                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CaculateFitness(XOppo, fun, args)
                for j in range(EliteNumber):
                    if fitOppo[j] < fitness[j]:
                        fitness[j] = copy.copy(fitOppo[j])
                        X[j, :] = copy.copy(XOppo[j, :])
                        res[j, :] = copy.copy(resOppo[j, :])
                fitness, sortIndex = SortFitness(fitness)
                X = SortPosition(X, sortIndex)
                res = SortPosition(res, sortIndex)

        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))

        if fitness[0] <= GbestScore:
            GbestScore = copy.copy(fitness[0])
            GbestPosition[0,:] = copy.copy(X[0, :])
        Curve[t] = GbestScore

        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1, Pos=Pos)
        record.save()

    print("")  # for progress bar

    raw_saver = save.RawDataSaver(hs, hf, hr, GbestScore, GbestPosition, Curve)
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore,GbestPosition,Curve, hs, hf,hr


def runMP(pop, dim, lb, ub, MaxIter, fun,n_jobs, RecordPath = None, args=()):
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
    cp.TSA.runMP(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1, n_jobs = 5)
    """
    print("Current Algorithm: TSA (Multi-Processing)")
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
        fitness, res = CaculateFitnessMP(X, fun,n_jobs, args)
        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        GbestScore = copy.copy(fitness[0])
        GbestPosition = np.zeros([1, dim])
        GbestPosition[0, :] = copy.copy(X[0, :])
        Curve = np.zeros([MaxIter, 1])
        Pos = np.zeros([pop, dim])
        record = save.SwarmRecord(pop=pop, dim=dim, lb=lb, ub=ub, hf=hf, hs=hs, hr=hr,
                                  GbestPosition=GbestPosition, GbestScore=GbestScore, Curve=Curve, X=X, fitness=fitness,
                                  iteration=0, Pos=Pos)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        X = record.X
        Pos = record.Pos
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
        Curve = record.Curve
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        same = record_check(pop, dim, lb, ub, a_pop, a_dim, a_lb, a_ub)
        if not same:
            raise ValueError("The pop, dim, lb, and ub should be same as the record")
        Progress_Bar.update(len(hf))

    for t in range(iter,MaxIter):

        xmin = 1
        xmax = 4
        xr = xmin + random.random() * (xmax - xmin)
        xr = np.fix(xr)

        for i in range(pop):
            for j in range(dim):
                A1 = (random.random() + random.random()) - (2 * random.random()) / xr
                c2 = random.random()
                if i == 0:
                    c3 = random.random()
                    d_pos = np.abs(GbestPosition[0, j] - c2 * X[i, j])
                    if c3 >= 0.5:
                        Pos[i, j] = GbestPosition[0, j] + A1 * d_pos
                    else:
                        Pos[i, j] = GbestPosition[0, j] - A1 * d_pos
                else:
                    c3 = random.random()
                    d_pos = np.abs(GbestPosition[0, j] - c2 * X[i, j])
                    if c3 >= 0.5:
                        Pos[i, j] = GbestPosition[0, j] + A1 * d_pos
                    else:
                        Pos[i, j] = GbestPosition[0, j] - A1 * d_pos
                    X[i, j] = (Pos[i, j] + X[i, j]) / (2 + random.random())


        X = BorderCheck(X, ub, lb, pop, dim)
        fitness,res = CaculateFitnessMP(X, fun,n_jobs,args)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        res = SortPosition(res, sortIndex)

        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(X.shape[0] * OppoFactor))
            if EliteNumber > 0:
                XElite = copy.copy(X[0:EliteNumber, :])
                Tlb = np.min(XElite, 0)
                Tub = np.max(XElite, 0)
                XOppo = np.array([random.random() * (Tlb + Tub) - XElite[j, :] for j in range(EliteNumber)])
                XOppo = BorderCheck(XOppo, ub, lb, EliteNumber, dim)
                fitOppo, resOppo = CaculateFitnessMP(XOppo, fun, n_jobs, args)
                for j in range(EliteNumber):
                    if fitOppo[j] < fitness[j]:
                        fitness[j] = copy.copy(fitOppo[j])
                        X[j, :] = copy.copy(XOppo[j, :])
                        res[j, :] = copy.copy(resOppo[j, :])
                fitness, sortIndex = SortFitness(fitness)
                X = SortPosition(X, sortIndex)
                res = SortPosition(res, sortIndex)

        hr.append(res)
        hs.append(copy.copy(X))
        hf.append(copy.copy(fitness))
        Progress_Bar.update(len(hf))

        if fitness[0] <= GbestScore:
            GbestScore = copy.copy(fitness[0])
            GbestPosition[0, :] = copy.copy(X[0, :])
        Curve[t] = GbestScore

        record = save.SwarmRecord(pop=pop,dim=dim,lb=lb,ub=ub,hf=hf,hs=hs,hr=hr,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1, Pos=Pos)
        record.save()

    print("")  # for progress bar
    raw_saver = save.RawDataSaver(hs, hf, hr, GbestScore, GbestPosition, Curve)
    raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hs, hf, hr


def run_MV(pop, dims, lbs, ubs, MaxIter, fun, RecordPath = None, args=()):
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
    print("Current Algorithm: TSA (Multi-Variables)")
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
        X,lbs,ubs = initial_MV(pop, dims, ubs, lbs)
        fitness,res = CaculateFitness_MV(X,fun,args)
        for i in range(num_var):
            hrs[i].append(res[i])
            hss[i].append(copy.copy(X[i]))
            hfs[i].append(copy.copy(fitness[i]))
        Progress_Bar.update(len(hfs[0]))
        for n in range(num_var):
            fitness[n], sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n], sortIndex)
        GbestScore = [copy.copy(fitness[n][0]) for n in range(num_var)]
        GbestPosition = [np.zeros([1, dims[n]]) for n in range(num_var)]
        for n in range(num_var):
            GbestPosition[n][0, :] = copy.copy(X[n][0, :])
        Curve = [np.zeros([MaxIter, 1]) for i in range(num_var)]
        Pos = [np.zeros([pop, dims[n]]) for n in range(num_var)]
        record = save.SwarmRecord(pop=pop, dim=dims, lb=lbs, ub=ubs, hf=hfs, hs=hss, hr=hrs,
                                  GbestPosition=GbestPosition, GbestScore=GbestScore, Curve=Curve, X=X, fitness=fitness,
                                  iteration=0, Pos=Pos)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hss = record.hs
        hfs = record.hf
        hrs = record.hr
        X = record.X
        Pos = record.Pos
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
        Curve = record.Curve
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        same = record_checkMV(pop, dims, lbs, ubs, a_pop, a_dim, a_lb, a_ub)
        if not same:
            raise ValueError("The pop, dim, lb, and ub should be same as the record")
        Progress_Bar.update(len(hfs[0]))

    for t in range(iter,MaxIter):

        xmin = 1
        xmax = 4
        xr = xmin + random.random() * (xmax - xmin)
        xr = np.fix(xr)

        for n in range(num_var):
            for i in range(pop):
                for j in range(dims[n]):
                    A1 = (random.random() + random.random()) - (2 * random.random()) / xr
                    c2 = random.random()
                    if i == 0:
                        c3 = random.random()
                        d_pos = np.abs(GbestPosition[n][0, j] - c2 * X[n][i, j])
                        if c3 >= 0.5:
                            Pos[n][i, j] = GbestPosition[n][0, j] + A1 * d_pos
                        else:
                            Pos[n][i, j] = GbestPosition[n][0, j] - A1 * d_pos
                    else:
                        c3 = random.random()
                        d_pos = np.abs(GbestPosition[n][0, j] - c2 * X[n][i, j])
                        if c3 >= 0.5:
                            Pos[n][i, j] = GbestPosition[n][0, j] + A1 * d_pos
                        else:
                            Pos[n][i, j] = GbestPosition[n][0, j] - A1 * d_pos
                        X[n][i, j] = (Pos[n][i, j] + X[n][i, j]) / (2 + random.random())

        for n in range(num_var):
            X[n] = BorderCheck(X[n], ubs[n], lbs[n], pop, dims[n])
        fitness, res = CaculateFitness_MV(X, fun, args)
        for n in range(num_var):
            fitness[n], sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n], sortIndex)
            res[n] = SortPosition(res[n], sortIndex)

        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(pop * OppoFactor))
            if EliteNumber>0:
                XElite = [copy.copy(X[n][0:EliteNumber, :]) for n in range(num_var)]
                Tlb = [np.min(XElite[n], 0) for n in range(num_var)]
                Tub = [np.max(XElite[n], 0) for n in range(num_var)]

                XOppo = [np.array([random.random() * (Tlb[n] + Tub[n]) - XElite[n][j, :] for j in range(EliteNumber)]) for n in range(num_var)]
                for n in range(num_var):
                    XOppo[n] = BorderCheck(XOppo[n], ubs[n], lbs[n], EliteNumber, dims[n])
                fitOppo, resOppo = CaculateFitness_MV(XOppo, fun, args)
                for j in range(EliteNumber):
                    for n in range(num_var):
                        if fitOppo[n][j] < fitness[n][j]:
                            fitness[n][j] = copy.copy(fitOppo[n][j])
                            X[n][j, :] = copy.copy(XOppo[n][j, :])
                            res[n][j, :] = copy.copy(resOppo[n][j, :])
                for n in range(num_var):
                    fitness[n], index = SortFitness(fitness[n])
                    X[n] = SortPosition(X[n], index)
                    res[n] = SortPosition(res[n], index)
        for n in range(num_var):
            hrs[n].append(res[n])
            hss[n].append(copy.copy(X[n]))
            hfs[n].append(copy.copy(fitness[n]))
        Progress_Bar.update(len(hfs[0]))
        for n in range(num_var):
            if fitness[n][0] <= GbestScore[n]:
                GbestScore[n] = copy.copy(fitness[n][0])
                GbestPosition[n][0, :] = copy.copy(X[n][0, :])
            Curve[n][t] = GbestScore[n]

        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1, Pos=Pos)
        record.save()

    print("")  # for progress bar
    for n in range(num_var):


        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[n]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(n+1)

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
    print("Current Algorithm: TSA (Multi-Processing-Multi-Variables)")
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
        X,lbs,ubs = initial_MV(pop, dims, ubs, lbs)
        fitness,res = CaculateFitnessMP_MV(X,fun,n_jobs,args)
        for i in range(num_var):
            hrs[i].append(res[i])
            hss[i].append(copy.copy(X[i]))
            hfs[i].append(copy.copy(fitness[i]))
        Progress_Bar.update(len(hfs[0]))
        for n in range(num_var):
            fitness[n], sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n], sortIndex)
        GbestScore = [copy.copy(fitness[n][0]) for n in range(num_var)]
        GbestPosition = [np.zeros([1, dims[n]]) for n in range(num_var)]
        for n in range(num_var):
            GbestPosition[n][0, :] = copy.copy(X[n][0, :])
        Curve = [np.zeros([MaxIter, 1]) for i in range(num_var)]
        Pos = [np.zeros([pop, dims[n]]) for n in range(num_var)]
        record = save.SwarmRecord(pop=pop, dim=dims, lb=lbs, ub=ubs, hf=hfs, hs=hss, hr=hrs,
                                  GbestPosition=GbestPosition, GbestScore=GbestScore, Curve=Curve, X=X, fitness=fitness,
                                  iteration=0, Pos=Pos)
        record.save()
    else:
        record = save.SwarmRecord.load(RecordPath)
        hss = record.hs
        hfs = record.hf
        hrs = record.hr
        X = record.X
        Pos = record.Pos
        fitness = record.fitness
        GbestScore = record.GbestScore
        GbestPosition = record.GbestPosition
        Curve = record.Curve
        iter = record.iteration
        a_lb = record.lb
        a_ub = record.ub
        a_pop = record.pop
        a_dim = record.dim
        same = record_checkMV(pop, dims, lbs, ubs, a_pop, a_dim, a_lb, a_ub)
        if not same:
            raise ValueError("The pop, dim, lb, and ub should be same as the record")
        Progress_Bar.update(len(hfs[0]))

    for t in range(iter,MaxIter):
        xmin = 1
        xmax = 4
        xr = xmin + random.random() * (xmax - xmin)
        xr = np.fix(xr)

        for n in range(num_var):
            for i in range(pop):
                for j in range(dims[n]):
                    A1 = (random.random() + random.random()) - (2 * random.random()) / xr
                    c2 = random.random()
                    if i == 0:
                        c3 = random.random()
                        d_pos = np.abs(GbestPosition[n][0, j] - c2 * X[n][i, j])
                        if c3 >= 0.5:
                            Pos[n][i, j] = GbestPosition[n][0, j] + A1 * d_pos
                        else:
                            Pos[n][i, j] = GbestPosition[n][0, j] - A1 * d_pos
                    else:
                        c3 = random.random()
                        d_pos = np.abs(GbestPosition[n][0, j] - c2 * X[n][i, j])
                        if c3 >= 0.5:
                            Pos[n][i, j] = GbestPosition[n][0, j] + A1 * d_pos
                        else:
                            Pos[n][i, j] = GbestPosition[n][0, j] - A1 * d_pos
                        X[n][i, j] = (Pos[n][i, j] + X[n][i, j]) / (2 + random.random())


        for n in range(num_var):
            X[n] = BorderCheck(X[n], ubs[n], lbs[n], pop, dims[n])
        fitness, res = CaculateFitnessMP_MV(X, fun,n_jobs, args)
        for n in range(num_var):
            fitness[n], sortIndex = SortFitness(fitness[n])
            X[n] = SortPosition(X[n], sortIndex)
            res[n] = SortPosition(res[n], sortIndex)

        if EliteOppoSwitch:

            EliteNumber = int(np.ceil(pop * OppoFactor))
            if EliteNumber>0:
                XElite = [copy.copy(X[n][0:EliteNumber, :]) for n in range(num_var)]
                Tlb = [np.min(XElite[n], 0) for n in range(num_var)]
                Tub = [np.max(XElite[n], 0) for n in range(num_var)]

                XOppo = [np.array([random.random() * (Tlb[n] + Tub[n]) - XElite[n][j, :] for j in range(EliteNumber)]) for n in range(num_var)]
                for n in range(num_var):
                    XOppo[n] = BorderCheck(XOppo[n], ubs[n], lbs[n], EliteNumber, dims[n])
                fitOppo, resOppo = CaculateFitnessMP_MV(XOppo, fun,n_jobs, args)
                for j in range(EliteNumber):
                    for n in range(num_var):
                        if fitOppo[n][j] < fitness[n][j]:
                            fitness[n][j] = copy.copy(fitOppo[n][j])
                            X[n][j, :] = copy.copy(XOppo[n][j, :])
                            res[n][j, :] = copy.copy(resOppo[n][j, :])
                for n in range(num_var):
                    fitness[n], index = SortFitness(fitness[n])
                    X[n] = SortPosition(X[n], index)
                    res[n] = SortPosition(res[n], index)
        for n in range(num_var):
            hrs[n].append(res[n])
            hss[n].append(copy.copy(X[n]))
            hfs[n].append(copy.copy(fitness[n]))
        Progress_Bar.update(len(hfs[0]))
        for n in range(num_var):
            if fitness[n][0] <= GbestScore[n]:
                GbestScore[n] = copy.copy(fitness[n][0])
                GbestPosition[n][0, :] = copy.copy(X[n][0, :])
            Curve[n][t] = GbestScore[n]

        record = save.SwarmRecord(pop=pop,dim=dims,lb=lbs,ub=ubs,hf=hfs,hs=hss,hr=hrs,
                         GbestPosition=GbestPosition,GbestScore=GbestScore,Curve=Curve,X=X,fitness=fitness,iteration=t+1, Pos=Pos)
        record.save()

    print("")  # for progress bar
    for n in range(num_var):

        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[n]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(n+1)

        raw_saver = save.RawDataSaver(hss[n], hfs[n], hrs[n], GbestScore[n], GbestPosition[n], Curve[n])
        raw_saver.save(save.raw_path)

    print("Analysis Complete!")

    return GbestScore, GbestPosition, Curve, hss, hfs, hrs
