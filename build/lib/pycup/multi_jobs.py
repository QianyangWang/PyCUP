import multiprocessing as mp
import numpy as np
from . import progress_bar


def do_multi_jobs(func, params,  n_process,args,pb=False):
    """
    Parallelization function

    :argument
    func: the objective function
    params: the samples/solutions in the sub-population -> np.array, shape = (n_samples, dim)
    n_process: num. of processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.
    pb: the switch for progress bar -> bool, default False

    :return:
    fitnesses: the fitness values of the given sub-population -> np.array, shape = (n_samples, 1)
    res_l: the calculation results of the given sub-population -> np.array, shape = (n_samples, len(result))
    """

    pool = mp.Pool(processes=n_process)  # generate n processes

    results = []
    pop = params.shape[0]
    fitnesses = np.zeros([pop, 1])
    total_calc = pop
    ProgressBar = progress_bar.ProgressBar(total_calc)
    if pb == True:
        ProgressBar.update(0)
    for i in range(pop):
        #results.append(pool.apply_async(func, args = (params[i,:],)))
        results.append(pool.apply_async(func, args=(params[i, :], *args)))
        if pb == True:
            ProgressBar.update(i+1)
    if pb == True:
        print("")
    pool.close()  # close the process pool
    pool.join()  # wait for all the processes finishing their tasks

    ress = []

    for i in range(len(results)):
        fitness,  res = results[i].get()
        fitnesses[i] = fitness
        ress.append(res)
    res_l = np.concatenate(ress)

    return fitnesses,  res_l


def do_multi_jobsMO(func, params,  n_process,n_obj,args,pb=False):
    """
    Parallelization function for multi-objective algorithms

    :argument
    func: the objective function
    params: the samples/solutions in the sub-population -> np.array, shape = (n_samples, dim)
    n_process: num. of processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.
    pb: the switch for progress bar -> bool, default False

    :return:
    fitnesses: the fitness values of the given sub-population -> np.array, shape = (n_samples, 1)
    res_l: the calculation results of the given sub-population -> np.array, shape = (n_samples, len(result))
    """

    pool = mp.Pool(processes=n_process)

    results = []
    pop = params.shape[0]
    fitnesses = np.zeros([pop, n_obj])
    total_calc = pop
    ProgressBar = progress_bar.ProgressBar(total_calc)
    if pb == True:
        ProgressBar.update(0)
    for i in range(pop):
        #results.append(pool.apply_async(func, args = (params[i,:],)))
        results.append(pool.apply_async(func, args=(params[i, :], *args)))
        if pb == True:
            ProgressBar.update(i+1)
    if pb == True:
        print("")
    pool.close()
    pool.join()

    ress = []

    for i in range(len(results)):
        fitness,  res = results[i].get()
        fitnesses[i] = fitness
        ress.append(res)
    res_l = np.concatenate(ress)

    return fitnesses,  res_l


def do_multi_jobsMV(func, params,  n_process,args,pb=False):
    """
    Parallelization function for multi-variable algorithms

    :argument
    func: the objective function
    params: the samples/solutions in the sub-population -> np.array, shape = (n_samples, dim)
    n_process: num. of processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.
    pb: the switch for progress bar -> bool, default False

    :return:
    fitnesses: a list of fitness values of the given sub-population -> [np.array,..., np.array, shape = (n_samples, 1)]
    res_l: a list of the calculation results -> [np.array,..., np.array, shape = (n_samples, len(result))]
    """

    pool = mp.Pool(processes=n_process)

    results = []
    pop = params[0].shape[0]
    num_vars = len(params)
    fitnesses = [np.zeros([pop, 1]) for i in range(num_vars)]
    total_calc = pop
    ProgressBar = progress_bar.ProgressBar(total_calc)
    if pb == True:
        ProgressBar.update(0)
    for i in range(pop):
        results.append(pool.apply_async(func, args=([params[n][i, :] for n in range(num_vars)], *args)))
        if pb == True:
            ProgressBar.update(i+1)
    if pb == True:
        print("")
    pool.close()
    pool.join()

    ress = [[] for i in range(num_vars)]


    for i in range(len(results)):
        fitness,  res = results[i].get()
        for n in range(num_vars):
            fitnesses[n][i] = fitness[n]
            ress[n].append(res[n])

    res_l = [np.concatenate(ress[n],axis=0) for n in range(num_vars)]

    return fitnesses,  res_l



def predict_multi_jobs(func, params,  n_process,args,pb=False):
    """
    Parallelization function

    :argument
    func: the objective function
    params: the samples/solutions in the sub-population -> np.array, shape = (n_samples, dim)
    n_process: num. of processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.
    pb: the switch for progress bar -> bool, default False

    :return:
    fitnesses: the fitness values of the given sub-population -> np.array, shape = (n_samples, 1)
    res_l: the calculation results of the given sub-population -> np.array, shape = (n_samples, len(result))
    """

    pool = mp.Pool(processes=n_process)  # generate n processes

    results = []
    pop = params.shape[0]
    total_calc = pop
    ProgressBar = progress_bar.ProgressBar(total_calc)
    if pb == True:
        ProgressBar.update(0)
    for i in range(pop):
        #results.append(pool.apply_async(func, args = (params[i,:],)))
        results.append(pool.apply_async(func, args=(params[i, :], *args)))
        if pb == True:
            ProgressBar.update(i+1)
    if pb == True:
        print("")
    pool.close()  # close the process pool
    pool.join()  # wait for all the processes finishing their tasks

    ress = []

    for i in range(len(results)):
        res = results[i].get()
        ress.append(res)
    res_l = np.concatenate(ress)

    return res_l