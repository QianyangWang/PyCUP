import math

import numpy as np
from . import sampling
from . import save
from . import multi_jobs
from . import progress_bar
import copy


def SortFitness(Fit):
    fitness = np.sort(Fit,axis=0)
    index = np.argsort(Fit,axis=0)
    return fitness,index


def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

def check_listitem(item1, item2):
    s_flags = [item1[i] == item2[i] for i in range(len(item1))]
    length = [len(item1[i]) for i in range(len(item1))]
    same = np.sum(s_flags) == np.sum(length)
    return same


def record_check(n, dim, lb, ub, a_n, a_dim, a_lb, a_ub):
    a = np.sum(lb == a_lb) == len(lb)
    b = np.sum(ub == a_ub) == len(ub)
    check_list = (n == a_n, dim == a_dim, a, b)
    if np.sum(check_list) == len(check_list):
        return True
    else:
        return False


def record_checkMV(n, dims, lbs, ubs, a_n, a_dims, a_lbs, a_ubs):
    p_sflage = n == a_n
    d_sflage = np.sum(np.array(dims) == np.array(a_dims)) == len(dims)
    lb_flag = check_listitem(lbs, a_lbs)
    ub_flag = check_listitem(ubs, a_ubs)
    if np.sum([p_sflage, d_sflage, lb_flag, ub_flag]) == 4:
        return True
    else:
        return False

# single thread version
def run(n,dim,lb,ub,fun,RecordPath = None,args=()):
    """
    Main function for the algorithm

    :argument
    n: num. iterations -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    hs: Historical samples.
    hf: The fitness of historical samples.
    hr: The results of historical samples.

    Reference:
    Keith, B., & Andrew, B. (1992). The future of distributed models: Model calibration and uncertainty prediction.
    Hydrological Process, 6, 279–298.

    Usage:
    import pycup as cp

    def uni_fun1(X):
	    # X for example np.array([1,2,3,...,30])
        fitness = np.sum(np.power(X,2)) + 1 # example: 1.2
        result = fitness.reshape(1,-1) # example ([1.2,])
        return fitness,result

    lb = -100 * np.ones(30)
    ub = 100 * np.ones(30)
    cp.GLUE.run(n = 10000, dim = 30, lb = lb, ub = ub, fun = uni_fun1)
    """
    print("Current Algorithm: GLUE")
    print("Iterations: {}".format(n))
    print("Dim:{}".format(dim))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    Progress_Bar = progress_bar.ProgressBar(n)
    if not RecordPath:
        # 1. sampling
        hs,lb,ub = sampling.LHS_sampling(pop=n,dim=dim,ub=ub,lb=lb)
        ## hf -> historical fitness
        ## hr -> historical results
        hf = []
        hr = []
        iteration = 0
        record = save.GLUERecord(n=n, dim=dim, lb=lb, ub=ub, hf=hf, hs=hs, hr=hr,iteration=0,mode="GLUE")
        record.save()
    else:
        record = save.GLUERecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        a_dim = record.dim
        a_lb = record.lb
        a_ub = record.ub
        a_n = record.n
        iteration = record.iteration
        mode = record.mode
        same = record_check(n,dim,lb,ub,a_n,a_dim,a_lb,a_ub)
        same_mode = mode == "GLUE"
        if not same and same_mode:
            raise ValueError("The n, dim, lb, ub, and single/multi-processing mode should be same as the record")
        Progress_Bar.update(len(hf))

    for i in range(iteration,n):
        fitness,res = fun(hs[i],*args)
        hf.append(fitness)
        hr.append(res)
        Progress_Bar.update(len(hf))
        record = save.GLUERecord(n=n, dim=dim, lb=lb, ub=ub, hf=hf, hs=hs, hr=hr,iteration=i+1,mode="GLUE")
        record.save()

    hr = [i.reshape(1, -1) for i in hr]
    hr = np.concatenate(hr, axis=0)
    hf = np.array(hf).reshape(-1,1)

    hf, f_index = SortFitness(hf)
    hs = SortPosition(hs,f_index)
    hr = SortPosition(hr,f_index)

    GbestPositon = np.zeros([1, dim])
    GbestScore = copy.copy(hf[0])
    GbestPositon[0, :] = copy.copy(hs[0, :])

    # 2. save
    saver = save.RawDataSaver(hs,hf,hr,bs=GbestScore,bp=GbestPositon,OptType="GLUE")
    saver.save(save.raw_path)
    print("")  # for progress bar
    print("Analysis Complete!")
    return hs,hr,hf


def runMP(n,dim,lb,ub,fun,n_jobs,RecordPath = None,args=()):
    """
    Main function for the algorithm (multi-processing version)

    :argument
    n: num. iterations -> int
    dim: num. parameters -> int
    ub: upper boundary -> np.array
    lb: lower boundary -> np.array
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: number of threads/processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
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
    cp.GLUE.runMP(n = 10000, dim = 30, lb = lb, ub = ub, fun = uni_fun1, n_jobs = 5)
    """
    print("Current Algorithm: GLUE (Multi-Processing)")
    print("Iterations: {}".format(n))
    print("Dim:{}".format(dim))
    print("Lower Bnd.:{}".format(lb))
    print("Upper Bnd.:{}".format(ub))
    n_iter = math.ceil(n/n_jobs)
    Progress_Bar = progress_bar.ProgressBar(n_iter)
    if not RecordPath:
        # 1. sampling
        hs,lb,ub = sampling.LHS_sampling(pop=n,dim=dim,ub=ub,lb=lb)
        hf = []
        hr = []
        ## hf -> historical fitness
        ## hr -> historical results
        iteration = 0
        record = save.GLUERecord(n=n, dim=dim, lb=lb, ub=ub, hf=hf, hs=hs, hr=hr,iteration=0,mode="GLUE-MP")
        record.save()
    else:
        record = save.GLUERecord.load(RecordPath)
        hs = record.hs
        hf = record.hf
        hr = record.hr
        a_dim = record.dim
        a_lb = record.lb
        a_ub = record.ub
        a_n = record.n
        iteration = record.iteration
        mode = record.mode
        same = record_check(n,dim,lb,ub,a_n,a_dim,a_lb,a_ub)
        same_mode = mode == "GLUE-MP"
        if not same and same_mode:
            raise ValueError("The n, dim, lb, ub, and single/multi-processing mode should be same as the record")
        Progress_Bar.update(len(hf))

    for i in range(iteration,n_iter):
        fitness, result = multi_jobs.do_multi_jobs(fun,hs[i*n_jobs:i*n_jobs+n_jobs],n_jobs,args)
        hf.append(fitness)
        hr.append(result)
        Progress_Bar.update(len(hf))
        record = save.GLUERecord(n=n, dim=dim, lb=lb, ub=ub, hf=hf, hs=hs, hr=hr,iteration=i+1,mode="GLUE-MP")
        record.save()

    hf = np.concatenate(hf,axis=0)
    hr = np.concatenate(hr,axis=0)
    hf, f_index = SortFitness(hf)
    hs = SortPosition(hs,f_index)
    hr = SortPosition(hr,f_index)

    GbestPositon = np.zeros([1, dim])
    GbestScore = copy.copy(hf[0])
    GbestPositon[0, :] = copy.copy(hs[0, :])

    # 2. save
    saver = save.RawDataSaver(hs,hf,hr,bs=GbestScore,bp=GbestPositon,OptType="GLUE")
    saver.save(save.raw_path)
    print("")  # for progress bar
    print("Analysis Complete!")
    return hs,hr,hf


def run_MV(n,dims,lbs,ubs,fun,RecordPath = None,args=()):
    """
    Main function for the algorithm (multi-variable version)
    See the document for more information.

    :argument
    n: num. iterations -> int
    dims: num. parameters list -> [int, ..., int]
    ubs: upper boundaries list -> [np.array, ..., np.array]
    lbs: lower boundary list -> [np.array, ..., np.array]
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    hss: List of Historical samples.
    hfs: List of The fitness of historical samples.
    hrs: List of the results of historical samples.
    """
    print("Current Algorithm: GLUE (Multi-Variables)")
    print("Num. Variables: {}".format(len(dims)))
    print("Iterations: {}".format(n))
    print("Dim:{}".format(dims))
    print("Lower Bnd.:{}".format(lbs))
    print("Upper Bnd.:{}".format(ubs))
    Progress_Bar = progress_bar.ProgressBar(n)
    hfs = []
    hrs = []
    if not RecordPath:
        # 1. sampling
        hss,lbs,ubs = sampling.LHS_samplingMV(pop=n,dims=dims,ubs=ubs,lbs=lbs)
        ## hf -> historical fitness
        ## hr -> historical results
        temp_f = []
        temp_r = []
        iteration = 0
        record = save.GLUERecord(n=n, dim=dims, lb=lbs, ub=ubs, hf=temp_f, hs=hss, hr=temp_r,iteration=0,mode="GLUE")
        record.save()
    else:
        record = save.GLUERecord.load(RecordPath)
        hss = record.hs
        temp_f = record.hf
        temp_r = record.hr
        a_dim = record.dim
        a_lb = record.lb
        a_ub = record.ub
        a_n = record.n
        iteration = record.iteration
        mode = record.mode
        same = record_checkMV(n,dims,lbs,ubs,a_n,a_dim,a_lb,a_ub)
        same_mode = mode == "GLUE"
        if not same and same_mode:
            raise ValueError("The n, dim, lb, ub, and single/multi-processing mode should be same as the record")
        Progress_Bar.update(len(temp_f))
    for t in range(iteration,n):
        sample = [hss[j][t] for j in range(len(hss))]
        fitness,res = fun(sample,*args)
        # fitness  [[],[]]
        # res [[,,,],[,,,]]
        res = [i.reshape(1, -1) for i in res]
        fitness = [np.array(i).reshape(-1, 1) for i in fitness]
        temp_f.append(fitness)
        temp_r.append(res)
        Progress_Bar.update(len(temp_f))
        record = save.GLUERecord(n=n, dim=dims, lb=lbs, ub=ubs, hf=temp_f, hs=hss, hr=temp_r,iteration=t+1,mode="GLUE")
        record.save()
    # temp_f [ [[],[]], [[],[]], [[],[]], ... ]
    # temp_res [ [[,,,],[,,,]], [[,,,],[,,,]], [[,,,],[,,,]], ... ]
    for d in range(len(dims)):
        hf = [temp_f[i][d] for i in range(n)] # -> [ [],[],[], ... ]
        hf = np.concatenate(hf,axis=0)
        hr = [temp_r[i][d] for i in range(n)] # -> [ [,,,], [,,,], ... ]
        hr = np.concatenate(hr,axis=0)
        hfs.append(hf)
        hrs.append(hr)

    for i in range(len(hss)):
        hfs[i], f_index = SortFitness(hfs[i])
        hss[i] = SortPosition(hss[i],f_index)
        hrs[i] = SortPosition(hrs[i],f_index)

    # 2. save
    for i in range(len(hss)):

        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[i]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(i+1)

        GbestPositon = np.zeros([1, dims[i]])
        GbestScore = copy.copy(hfs[i][0])
        GbestPositon[0, :] = copy.copy(hss[i][0, :])

        saver = save.RawDataSaver(hss[i],hfs[i],hrs[i],bs=GbestScore,bp=GbestPositon,OptType="GLUE")
        saver.save(save.raw_path)
    print("")  # for progress bar
    print("Analysis Complete!")
    return hss,hrs,hfs


def runMP_MV(n,dims,lbs,ubs,fun,n_jobs,RecordPath = None,args=()):
    """
    Main function for the algorithm (multi-processing multi-variable version)
    See the document for more information.

    :argument
    n: num. iterations -> int
    dims: num. parameters list -> [int, ..., int]
    ubs: upper boundaries list -> [np.array, ..., np.array]
    lbs: lower boundary list -> [np.array, ..., np.array]
    fun: The user defined objective function or function in pycup.test_functions. The function
         should return a fitness value and a calculation result. See pycup.test_functions for
         more information -> function variable
    n_jobs: number of threads/processes -> int
    args: A tuple of arguments. Users can use it for obj_fun's customization. For example, the
          parameter file path and model file path can be stored in this tuple for further use.
          See the document for more details.

    :returns
    hss: List of Historical samples.
    hfs: List of The fitness of historical samples.
    hrs: List of the results of historical samples.
    """
    print("Current Algorithm: GLUE (Multi-Processing-Multi-Variables)")
    print("Num. Variables: {}".format(len(dims)))
    print("Iterations: {}".format(n))
    print("Dim:{}".format(dims))
    print("Lower Bnd.:{}".format(lbs))
    print("Upper Bnd.:{}".format(ubs))
    n_iter = math.ceil(n/n_jobs)
    Progress_Bar = progress_bar.ProgressBar(n_iter)
    hfs = []
    hrs = []
    if not RecordPath:
        # 1. sampling
        hss,lbs,ubs = sampling.LHS_samplingMV(pop=n,dims=dims,ubs=ubs,lbs=lbs)
        temp_f = []
        temp_r = []
        ## hf -> historical fitness
        ## hr -> historical results
        iteration = 0
        record = save.GLUERecord(n=n, dim=dims, lb=lbs, ub=ubs, hf=temp_f, hs=hss, hr=temp_r,iteration=0,mode="GLUE-MP")
        record.save()
    else:
        record = save.GLUERecord.load(RecordPath)
        hss = record.hs
        temp_f = record.hf
        temp_r = record.hr
        a_dim = record.dim
        a_lb = record.lb
        a_ub = record.ub
        a_n = record.n
        iteration = record.iteration
        mode = record.mode
        same = record_checkMV(n,dims,lbs,ubs,a_n,a_dim,a_lb,a_ub)
        same_mode = mode == "GLUE-MP"
        if not same and same_mode:
            raise ValueError("The n, dim, lb, ub, and single/multi-processing mode should be same as the record")
        Progress_Bar.update(len(temp_f))
    for t in range(iteration,n_iter):
        ## hf -> historical fitness
        ## hr -> historical results
        fitness, result = multi_jobs.do_multi_jobsMV(fun, [hss[j][t * n_jobs:t * n_jobs + n_jobs] for j in range(len(hss))], n_jobs, args)
        temp_f.append(fitness)
        temp_r.append(result)
        Progress_Bar.update(len(temp_f))
        record = save.GLUERecord(n=n, dim=dims, lb=lbs, ub=ubs, hf=temp_f, hs=hss, hr=temp_r,iteration=t+1,mode="GLUE-MP")
        record.save()

    for d in range(len(dims)):
        hf = [temp_f[i][d] for i in range(n_iter)]
        hf = np.concatenate(hf,axis=0)
        hr = [temp_r[i][d] for i in range(n_iter)]
        hr = np.concatenate(hr,axis=0)
        hfs.append(hf)
        hrs.append(hr)

    for i in range(len(hss)):
        hfs[i], f_index = SortFitness(hfs[i])
        hss[i] = SortPosition(hss[i],f_index)
        hrs[i] = SortPosition(hrs[i],f_index)

    # 2. save
    for i in range(len(hss)):

        if len(save.raw_pathMV) == len(hss):
            save.raw_path = save.raw_pathMV[i]
        else:
            save.raw_path = "RawResult_Var{}.rst".format(i+1)

        GbestPositon = np.zeros([1, dims[i]])
        GbestScore = copy.copy(hfs[i][0])
        GbestPositon[0, :] = copy.copy(hss[i][0, :])

        saver = save.RawDataSaver(hss[i],hfs[i],hrs[i],bs=GbestScore,bp=GbestPositon,OptType="GLUE")
        saver.save(save.raw_path)
    print("")  # for progress bar
    print("Analysis Complete!")
    return hss,hrs,hfs
