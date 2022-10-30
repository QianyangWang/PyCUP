import numpy as np
from . import save
from scipy.interpolate import interp1d
import statsmodels.api as sm


def SortFitness(Fit):
    fitness = np.sort(Fit,axis=0)
    index = np.argsort(Fit,axis=0)
    return fitness,index


def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def check_ppu(confidence_value):
    """
    :param confidence_value:
    :return: lower and upper confidence value
    """
    if confidence_value > 1:
        out_bounds = 1.0 - confidence_value/100
    else:
        out_bounds = 1.0 - confidence_value
    lower = out_bounds/2
    upper = 1.0 - out_bounds/2

    return lower,upper


def gen_ppu_bound(confidence_value,cum,results_sort):
    lower,upper = check_ppu(confidence_value)
    ppu_line_lower = []
    ppu_line_upper = []
    for i in range(cum.shape[1]):
        all_results = cum[:, i].flatten()
        nearest_l, id_nearest_l = find_nearest(all_results, lower)
        nearest_u, id_nearest_u = find_nearest(all_results, upper)
        ppu_line_lower.append(results_sort[id_nearest_l, i])
        ppu_line_upper.append(results_sort[id_nearest_u, i])
    ppu_line_lower = np.array(ppu_line_lower)
    ppu_line_upper = np.array(ppu_line_upper)
    return ppu_line_lower,ppu_line_upper


def likelihood_uncertainty(raw_saver,threshold,ppu,st_id=0):
    """
    This is an uncertainty analysis function based on the principles of GLUE method, it is a common method for all the
    algorithms in this package, also, it is similar to what you will have in other software (e.g. SWAT-CUP).
    This function will generate a pycup.save.ProcResultSaver object, which contains all the analysis results. The
    save method of this object will also be called. Users can also change the saving path by modifying the
    pycup.save.proc_path variable (see the document).

    :argument
    raw_saver: save.RawDataSaver object (optimization result file)
    threshold: threshold value for extracting the behavioral samples -> float
    ppu: prediction uncertainty level -> float or int, typically 95 (95PPU).
         If the given value < 1.0, it will be treated as value * 100, therefore, 0.95 is also acceptable
    st_id: station index for the simulation results -> int, this is only valid when the result is a 3-dimensional array
           (when you have multi-station results). The uncertainty analysis will be carried out based the data at the
           station that you have selected.

    :return:
    saver: save.ProcResultSaver object

    Usage:
    import pycup as cp
    import pycup.uncertainty_analysis_fun as u_fun

    raw_res = cp.save.RawDataSaver.load(r"RawResult.rst")
    u_fun.likelihood_uncertainty(raw_res,threshold=0.5,ppu=0.95)
    """

    if not(isinstance(raw_saver,save.RawDataSaver)):
        raise ValueError("The given saver object is not a save.RawDataSaver.")

    if raw_saver.opt_type == "GLUE":
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples
    else:
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

        hs = np.concatenate(hs)
        hr = np.concatenate(hr)
        hf = np.concatenate(hf)

    if len(hr.shape) == 3:
        hr = hr[:,:,st_id]
    # 1. sort the samples and fitness (likelihood function values)
    sorted_fitness, index_l = SortFitness(hf) ## ascending according to the fitness
    sorted_samples = SortPosition(hs, index_l)
    sorted_results = SortPosition(hr, index_l)
    best_sample = sorted_samples[0]
    best_fitness = sorted_fitness[0]
    best_result = sorted_results[0]

    # 2. find the behaviour_samples and fitness according to the threshold
    ## the likelihood function would be errors (e.g. RMSE, MAE), of which a lower value denotes a btter performance
    ## for metrics such as NSE or R2, (1-NSE) or (1-R2) should be used.
    behaviour_samples = np.array([sorted_samples[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    behaviour_fitness = np.array([sorted_fitness[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    behaviour_results = np.array([sorted_results[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])

    if len(behaviour_samples) == 0:
        print("No behaviour parameter has been found, please adjust the threshold value")
        return
    else:
        # 3. normalize the fitness (likelihood function values) and calculate the likelihood value (weight)
        ## since a lower value denotes a btter performance, take the reciprocals of the likelihood function values, as
        ## the result, a lower likelihood function value can have a higher weight

        reciprocals = 1/behaviour_fitness
        total_likelihood = np.sum(reciprocals)

        ## for the convenience of the following steps, normalized_weight_arr is used (the weights should
        ## be sorted according to the rank of simulated values to calculate the cumulative weighted)
        normalized_weight = reciprocals / total_likelihood # shape = (num_behaviour_samples,)
        normalized_weight_arr = np.array([normalized_weight.flatten() for i in range(behaviour_results.shape[1])]).T # shape = (num_behaviour_samples,time_steps)

        # 4. calculate the posterior distribution
        sorted_sample_val,sorted_sample_id = np.sort(behaviour_samples,axis=0),np.argsort(behaviour_samples,axis=0)
        id_sample_column = np.array([np.arange(behaviour_samples.shape[1]) for i in range(behaviour_samples.shape[0])])
        normalized_sample_weight = np.array([normalized_weight.flatten() for i in range(behaviour_samples.shape[1])]).T
        normalized_weight_sort = normalized_sample_weight[sorted_sample_id, id_sample_column]
        cum_sample = np.cumsum(normalized_weight_sort, axis=0)

        # 5. sort the results for each time step, manipulate the weight according to the indices, calculate the cumulative weight
        results_sort, id = np.sort(behaviour_results, axis=0), np.argsort(behaviour_results, axis=0)
        id_column = np.array([np.arange(behaviour_results.shape[1]) for i in range(behaviour_results.shape[0])])
        normalized_fit_sort = normalized_weight_arr[id, id_column]
        cum = np.cumsum(normalized_fit_sort, axis=0)

        # 6. calculate the lower and upper uncertainty band, generate the max and min bound
        ppu_line_lower, ppu_line_upper = gen_ppu_bound(ppu,cum,results_sort)
        line_min = results_sort[0, :].flatten()
        line_max = results_sort[-1, :].flatten()

        # 7. calculate the 50% percentile prediction
        median_prediction = get_median_series(behaviour_results)

        saver = save.ProcResultSaver(hs, hf, hr, best_sample, best_fitness, best_result, behaviour_samples, behaviour_fitness, behaviour_results,
                       normalized_weight, sorted_sample_val, cum_sample, results_sort, cum, ppu_line_lower, ppu_line_upper, line_min, line_max,median_prediction)
        saver.save(save.proc_path)
        return saver


def likelihood_uncertaintyMO(raw_saver, n_obj, thresholds, ppu, obj_weights,st_id=0):
    """
    This is an uncertainty analysis function for multi-objective optimization algorithms.
    This function will generate a pycup.save.ProcResultSaver object, which contains all the analysis results. The
    save method of this object will also be called. Users can also change the saving path by modifying the
    pycup.save.proc_path variable (see the document).

    :argument
    raw_saver: save.RawDataSaver object (optimization result file)
    n_obj: num. of objective functions -> int
    thresholds: a list of threshold values for extracting the behavioral samples -> array like, e.g. [0.5, 0.5]
    ppu: prediction uncertainty level -> float or int, typically 95 (95PPU).
         If the given value < 1.0, it will be treated as value * 100, therefore, 0.95 is also acceptable
    obj_weights: a list of weights for the objective functions -> array like, e.g. [0.5, 0.5] means two objective
                 functions with the same weight.

    :return:
    saver: save.ProcResultSaver object

    Usage:
    import pycup as cp
    import pycup.uncertainty_analysis_fun as u_fun

    raw_res = cp.save.RawDataSaver.load(r"RawResult.rst")
    u_fun.likelihood_uncertaintyMO(raw_res,n_obj=2,thresholds=[0.5,0.2],ppu=0.95,obj_weights=[0.5,0.5])
    """

    if not (isinstance(raw_saver, save.RawDataSaver)):
        raise ValueError("The given saver object is not a save.RawDataSaver.")

    if len(obj_weights) != n_obj:
        raise ValueError("The length of objective function weights should be equal to n_obj.")

    if raw_saver.opt_type == "GLUE":
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples
    else:
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

        hs = np.concatenate(hs)
        hr = np.concatenate(hr)
        hf = np.concatenate(hf)


    paretoPops = raw_saver.pareto_samples
    paretoFits = raw_saver.pareto_fitness
    paretoRes = raw_saver.pareto_results

    if len(hr.shape) == 3:
        hr = hr[:,:,st_id]
        paretoRes = paretoRes[:,:,st_id]

    obj_weights = np.array(obj_weights).flatten()

    # 1. find the behaviour_samples and fitness according to the threshold
    ## the likelihood function would be errors (e.g. RMSE, MAE), of which a lower value denotes a btter performance
    ## for metrics such as NSE or R2, (1-NSE) or (1-R2) should be used.
    pop = hf.shape[0]
    behaviour_id = []
    for i in range(pop):
        if np.sum(hf[i] < np.array(thresholds).reshape(1, -1)) == n_obj:
            behaviour_id.append(i)
    behaviour_id = np.array(behaviour_id)
    if len(behaviour_id) == 0:
        print("No behaviour parameter has been found, please adjust the threshold value")
        return
    else:
        behaviour_samples = hs[behaviour_id]
        behaviour_fitness = hf[behaviour_id]
        behaviour_results = hr[behaviour_id]

        # 2. normalize the fitness (likelihood function values) and calculate the likelihood value (weight)
        ## since a lower value denotes a btter performance, take the reciprocals of the likelihood function values, as
        ## the result, a lower likelihood function value can have a higher weight

        reciprocals = 1 / behaviour_fitness
        total_likelihood = np.sum(reciprocals, axis=0)

        ## for the convenience of the following steps, normalized_weight_arr is used (the weights should
        ## be sorted according to the rank of simulated values to calculate the cumulative weighted)
        normalized_weight = reciprocals / total_likelihood  # shape = (num_behaviour_samples,n_obj)
        ## the weighted normalized weight

        w_normalized_weight = np.sum(normalized_weight * obj_weights, axis=1)  # shape = (num_behaviour_samples,)
        normalized_weight_arr = np.array([w_normalized_weight.flatten() for i in range(
            behaviour_results.shape[1])]).T  # shape = (num_behaviour_samples,time_steps)

        # 3. calculate the posterior distribution
        sorted_sample_val, sorted_sample_id = np.sort(behaviour_samples, axis=0), np.argsort(behaviour_samples, axis=0)
        id_sample_column = np.array([np.arange(behaviour_samples.shape[1]) for i in range(behaviour_samples.shape[0])])
        normalized_sample_weight = np.array(
            [w_normalized_weight.flatten() for i in range(behaviour_samples.shape[1])]).T
        normalized_weight_sort = normalized_sample_weight[sorted_sample_id, id_sample_column]
        cum_sample = np.cumsum(normalized_weight_sort, axis=0)

        # 4. sort the results for each time step, manipulate the weight according to the indices, calculate the cumulative weight
        results_sort, id = np.sort(behaviour_results, axis=0), np.argsort(behaviour_results, axis=0)
        id_column = np.array([np.arange(behaviour_results.shape[1]) for i in range(behaviour_results.shape[0])])
        normalized_fit_sort = normalized_weight_arr[id, id_column]
        cum = np.cumsum(normalized_fit_sort, axis=0)
        # 5. calculate the lower and upper uncertainty band, generate the max and min bound
        ppu_line_lower, ppu_line_upper = gen_ppu_bound(ppu, cum, results_sort)
        line_min = results_sort[0, :].flatten()
        line_max = results_sort[-1, :].flatten()

        # 6. calculate the 50% percentile prediction
        median_prediction = get_median_series(behaviour_results)

        saver = save.ProcResultSaver(hs, hf, hr, None, None, None, behaviour_samples, behaviour_fitness,
                                    behaviour_results,
                                    w_normalized_weight, sorted_sample_val, cum_sample, results_sort, cum,
                                    ppu_line_lower, ppu_line_upper, line_min, line_max, median_prediction,paretoPops, paretoFits,
                                    paretoRes)
        saver.save(save.proc_path)
        return saver


def cal_validation_uncertainty(fitness,results,ppu,weights):

    normalized_weight = weights  # shape = (num_behaviour_samples,)
    normalized_weight_arr = np.array([normalized_weight.flatten() for i in range(
        results.shape[1])]).T  # shape = (num_behaviour_samples,time_steps)

    results_sort, id = np.sort(results, axis=0), np.argsort(results, axis=0)
    id_column = np.array([np.arange(results.shape[1]) for i in range(results.shape[0])])
    normalized_fit_sort = normalized_weight_arr[id, id_column]
    cum = np.cumsum(normalized_fit_sort, axis=0)
    ppu_line_lower, ppu_line_upper = gen_ppu_bound(ppu, cum, results_sort)
    line_min = results_sort[0, :].flatten()
    line_max = results_sort[-1, :].flatten()
    if fitness.shape[1] == 1:
        best_idx = np.argmin(fitness)
        best_result = results[best_idx]
    else:
        best_result = None
    median_prediction = get_median_series(results)
    saver = save.ValidationResultSaver(fitness,results,ppu_line_upper,ppu_line_lower,line_max,line_min,best_result,median_prediction)
    saver.save()
    return saver


def cal_prediction_uncertainty(results,ppu,weights):

    normalized_weight = weights  # shape = (num_behaviour_samples,)
    normalized_weight_arr = np.array([normalized_weight.flatten() for i in range(
        results.shape[1])]).T  # shape = (num_behaviour_samples,time_steps)

    results_sort, id = np.sort(results, axis=0), np.argsort(results, axis=0)
    id_column = np.array([np.arange(results.shape[1]) for i in range(results.shape[0])])
    normalized_fit_sort = normalized_weight_arr[id, id_column]
    cum = np.cumsum(normalized_fit_sort, axis=0)
    ppu_line_lower, ppu_line_upper = gen_ppu_bound(ppu, cum, results_sort)
    line_min = results_sort[0, :].flatten()
    line_max = results_sort[-1, :].flatten()
    median_prediction = get_median_series(results)
    saver = save.PredResultSaver(results,ppu_line_upper,ppu_line_lower,line_max,line_min,median_prediction)
    saver.save()
    return saver


def calc_p_factor(upper, lower, obs_x, obs_y):
    x = np.arange(len(upper))
    f_upper = interp1d(x, upper)
    f_lower = interp1d(x, lower)
    y_upper = f_upper(obs_x)
    y_lower = f_lower(obs_x)
    inner = []

    for i in range(len(obs_y)):
        if obs_y[i] > y_lower[i] and obs_y[i] < y_upper[i]:
            inner.append(obs_y[i])
    p_facter = len(inner)/len(obs_y)
    return p_facter


def calc_r_factor(upper, lower,obs):

    dbar = np.average(upper-lower)
    r_factor = dbar/np.std(obs)
    return r_factor


def cal_band_width(upper, lower):
    b = np.average(upper-lower)
    return b


def cal_relative_deviation(upper, lower,obs,idx_obs=None):
    """
    :param upper:
    :param lower:
    :param obs:
    :param idx_obs: the (location) of the observations, this is for the case that the number of observations is less than the
           number of values in the simulated series
    :return:
    """
    if idx_obs is not None:
        upper = upper[idx_obs]
        lower = lower[idx_obs]
    rd = np.average(np.abs(0.5*(upper+lower)-obs)/obs)
    return rd


def sensitivity_analysis(raw_saver,objfun_id=0):
    """
    This is a global sensitivity analysis function based on multi-linear regression method a t-statistic.

    :argument
    raw_saver: save.RawDataSaver object (optimization result file)
    objfun_id: the index of objective function -> int, 0 for single objective function, if you are doing this for a
               multi-objective algorithm (MOPSO), this argument should be the column index of the fitness value that you
               want to analyze.
    """
    if not (isinstance(raw_saver, save.RawDataSaver)):
        raise ValueError("The given saver object is not a save.RawDataSaver.")

    if raw_saver.opt_type == "GLUE":
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples
    elif raw_saver.opt_type == "SWARM":
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples
        hs = np.concatenate(hs)
        hf = np.concatenate(hf)
    else:
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples
        hs = np.concatenate(hs)
        hf = np.concatenate(hf)
    hf = hf[:,objfun_id]
    mlr_mdl = sm.OLS(hf,sm.add_constant(hs))
    res = mlr_mdl.fit()
    print(res.summary())


def get_median_series(behaviour_results):

    median_series = np.percentile(behaviour_results,50,axis=0)
    return median_series