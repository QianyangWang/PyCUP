import numpy as np
from . import multi_jobs
from . import Reslib


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


def CalculateFitness(X, fun,n_obj, args):
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
    fitness = np.zeros([pop, n_obj])
    res_l = []
    for i in range(pop):
        fitness[i], res = fun(X[i, :], *args)
        res_l.append(res)
    if not Reslib.UseResObject:
        res_l = np.concatenate(res_l)
    else:
        res_l = np.array(res_l,dtype=object)
    return fitness, res_l


def CalculateFitness_MV(Xs, fun, args):
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
        if not Reslib.UseResObject:
            res = [i.reshape(1, -1) for i in res]
        fit = [np.array(i).reshape(-1, 1) for i in fit]
        temp_f.append(fit)
        temp_r.append(res)

    for d in range(num_var):
        hf = [temp_f[i][d] for i in range(pop)] # -> [ [],[],[], ... ]
        hf = np.concatenate(hf,axis=0)
        hr = [temp_r[i][d] for i in range(pop)] # -> [ [,,,], [,,,], ... ]
        if not Reslib.UseResObject:
            hr = np.concatenate(hr,axis=0)
        else:
            hr = np.array(hr,dtype=object)
        fitness.append(hf)
        res_l.append(hr)

    return fitness, res_l


def CalculateFitnessMP(X, fun,n_obj, n_jobs, args):
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
    if n_obj == 1:
        fitness, res_l = multi_jobs.do_multi_jobs(func=fun, params=X, n_process=n_jobs, args=args)
    else:
        fitness, res_l = multi_jobs.do_multi_jobsMO(func=fun, params=X, n_process=n_jobs, n_obj=n_obj, args=args)

    return fitness, res_l


def CalculateFitnessMP_MV(Xs, fun, n_jobs, args):
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


def check_listitem(item1,item2):

    s_flags = [item1[i] == item2[i] for i in range(len(item1))]
    s_flags = [np.sum(i) for i in s_flags]
    length = [len(item1[i]) for i in range(len(item1))]
    same = np.sum(s_flags) == np.sum(length)
    return same


def record_check(pop,dim,lb,ub,a_pop,a_dim,a_lb,a_ub):
    a = np.sum(lb==a_lb) == len(lb)
    b = np.sum(ub==a_ub) == len(ub)
    check_list = (pop==a_pop,dim==a_dim,a,b)
    if np.sum(check_list) == len(check_list):
        return True
    else:
        return False


def record_checkMV(pop,dims,lbs,ubs,a_pop,a_dims,a_lbs,a_ubs):
    p_sflage = pop == a_pop
    d_sflage = np.sum(np.array(dims) == np.array(a_dims)) == len(dims)
    lb_flag = check_listitem(lbs,a_lbs)
    ub_flag = check_listitem(ubs,a_ubs)
    if np.sum([p_sflage,d_sflage,lb_flag,ub_flag]) == 4:
        return True
    else:
        return False