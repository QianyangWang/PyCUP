import numpy as np
from . import save
from scipy.interpolate import interp1d
import statsmodels.api as sm
from . import Reslib

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


def likelihood_uncertainty(raw_saver, threshold, ppu):
    """
    This is an uncertainty analysis function based on the principles of GLUE method, it is a common method for all the
    algorithms in this package, also, it is similar to what you will have in other software (e.g. SWAT-CUP).
    This function will generate a pycup.save.ProcResultSaver object, which contains all the analysis results.
    Users can also change the saving path by modifying the
    pycup.save.proc_path variable (see the document).

    The method basically follows the GLUE method. It assumes the prior probabilities of your behavioral samples are
    all the same, then, the posterior probabilities of them are calculated using your likelihood function values.
    For the LHS sampling based GLUE method, the result of this function can be used to estimate both
    the parameter uncertainty and the prediction uncertainty. For the heuristic algorithms, this method
    is only recommended to be used to estimate the prediction bound. It is not capable of obtaining an
    accurate parameter posterior distribution, although the response surface of the parameter-objective function value
    relationship can also be found. The reason is that the sampling processes of heuristic algorithms are not following
    the same principles of Monte Carlo methods.

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

    if not (isinstance(raw_saver, save.RawDataSaver)):
        raise TypeError("The given saver object is not a save.RawDataSaver.")

    if raw_saver.opt_type == "GLUE":
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

    else:
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

        hs = np.concatenate(hs)
        hf = np.concatenate(hf)
        if Reslib.UseResObject:
            hr = np.concatenate(raw_saver.historical_results.data)
        else:
            hr = np.concatenate(hr)

    # 1. sort the samples and fitness (likelihood function values)
    sorted_fitness, index_l = SortFitness(hf)  ## ascending according to the fitness
    sorted_samples = SortPosition(hs, index_l)
    if not Reslib.UseResObject:
        sorted_results = SortPosition(hr, index_l)
    else:
        sorted_results = hr[index_l].flatten()
    best_sample = sorted_samples[0]
    best_fitness = sorted_fitness[0]
    best_result = sorted_results[0]

    # 2. find the behaviour_samples and fitness according to the threshold
    ## the likelihood function would be errors (e.g. RMSE, MAE), of which a lower value denotes a btter performance
    ## for metrics such as NSE or R2, (1-NSE) or (1-R2) should be used.
    behaviour_samples = np.array(
        [sorted_samples[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    behaviour_fitness = np.array(
        [sorted_fitness[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    if not Reslib.UseResObject:
        behaviour_results = np.array(
            [sorted_results[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    else:
        behaviour_results = np.array(
            [sorted_results[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold], dtype=object)
    if len(behaviour_samples) == 0:
        print("No behaviour parameter has been found, please adjust the threshold value")
        return
    else:
        # 3. normalize the fitness (likelihood function values) and calculate the likelihood value (weight)
        ## since a lower value denotes a btter performance, take the reciprocals of the likelihood function values, as
        ## the result, a lower likelihood function value can have a higher weight

        reciprocals = 1 / behaviour_fitness
        total_likelihood = np.sum(reciprocals)

        ##3.5 for the convenience of the following steps, normalized_weight_arr is used (the weights should
        ## be sorted according to the rank of simulated values to calculate the cumulative weight)
        normalized_weight = reciprocals / total_likelihood  # shape = (num_behaviour_samples,)

        # 4. calculate the posterior distribution
        sorted_sample_val, sorted_sample_id = np.sort(behaviour_samples, axis=0), np.argsort(behaviour_samples, axis=0)
        id_sample_column = np.array([np.arange(behaviour_samples.shape[1]) for i in range(behaviour_samples.shape[0])])
        normalized_sample_weight = np.array([normalized_weight.flatten() for i in range(behaviour_samples.shape[1])]).T
        normalized_weight_sort = normalized_sample_weight[sorted_sample_id, id_sample_column]
        cum_sample = np.cumsum(normalized_weight_sort, axis=0)

        if not Reslib.UseResObject:

            ##3.5 continue
            normalized_weight_arr = np.array([normalized_weight.flatten() for i in range(
                behaviour_results.shape[1])]).T  # shape = (num_behaviour_samples,time_steps)

            # 5. sort the results for each time step, manipulate the weight according to the indices, calculate the cumulative weight
            results_sort, id = np.sort(behaviour_results, axis=0), np.argsort(behaviour_results, axis=0)
            id_column = np.array([np.arange(behaviour_results.shape[1]) for i in range(behaviour_results.shape[0])])
            normalized_fit_sort = normalized_weight_arr[id, id_column]
            cum = np.cumsum(normalized_fit_sort, axis=0)

            # 6. calculate the lower and upper uncertainty band, generate the max and min bound
            ppu_line_lower, ppu_line_upper = gen_ppu_bound(ppu, cum, results_sort)
            line_min = results_sort[0, :].flatten()
            line_max = results_sort[-1, :].flatten()

            # 7. calculate the 50% percentile prediction
            median_prediction = get_median_series(behaviour_results)
        else:
            stations = list(hr[0].keys())

            ppu_line_lower = Reslib.SimulationResult()
            ppu_line_upper = Reslib.SimulationResult()
            line_min = Reslib.SimulationResult()
            line_max = Reslib.SimulationResult()
            median_prediction = Reslib.SimulationResult()

            cum = np.array([Reslib.SimulationResult() for i in range(len(behaviour_results))], dtype=object)
            results_sort = np.array([Reslib.SimulationResult() for i in range(len(behaviour_results))], dtype=object)

            for st in stations:

                ppu_line_lower.add_station(st)
                ppu_line_upper.add_station(st)
                line_min.add_station(st)
                line_max.add_station(st)
                median_prediction.add_station(st)

                for i in range(len(behaviour_results)):
                    cum[i].add_station(st)
                    results_sort[i].add_station(st)

                events = list(hr[0][st].keys())
                for e in events:
                    # ResultDataPackage[0] => the first Simulation object
                    normalized_weight_arr = np.array([normalized_weight.flatten() for i in range(behaviour_results[0][st][e].shape[1])]).T  # shape = (num_behaviour_samples,time_steps)

                    st_e_br = []  # station event behavioral results
                    for i in behaviour_results:
                        data = i[st][e]
                        data = np.array(data).reshape(1, -1)
                        st_e_br.append(data)
                    st_e_br = np.concatenate(st_e_br, axis=0)

                    # 5. sort the results for each time step, manipulate the weight according to the indices, calculate the cumulative weight
                    st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)
                    id_column = np.array(
                        [np.arange(st_e_br.shape[1]) for i in range(st_e_br.shape[0])])
                    normalized_fit_sort = normalized_weight_arr[id, id_column]
                    st_e_cum = np.cumsum(normalized_fit_sort, axis=0)

                    for i in range(len(behaviour_results)):
                        cum[i][st].add_event(e, st_e_cum[i])
                        results_sort[i][st].add_event(e, st_e_results_sort[i])

                    # 6. calculate the lower and upper uncertainty band, generate the max and min bound
                    st_e_ppu_line_lower, st_e_ppu_line_upper = gen_ppu_bound(ppu, st_e_cum, st_e_results_sort)
                    st_e_line_min = st_e_results_sort[0, :].flatten()
                    st_e_line_max = st_e_results_sort[-1, :].flatten()

                    # 7. calculate the 50% percentile prediction
                    st_e_median_prediction = get_median_series(st_e_br)

                    ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                    ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                    line_min[st].add_event(e, st_e_line_min)
                    line_max[st].add_event(e, st_e_line_max)
                    median_prediction[st].add_event(e, st_e_median_prediction)
            cum = Reslib.ResultDataPackage(cum, method_info="Cumulative weights")
            results_sort = Reslib.ResultDataPackage(results_sort, method_info="Sorter results")
            behaviour_results = Reslib.ResultDataPackage(behaviour_results, method_info="Behavioral results")
            hr = Reslib.ResultDataPackage(hr, method_info="Historical results")

        saver = save.ProcResultSaver(hs, hf, hr, best_sample, best_fitness, best_result, behaviour_samples,
                                     behaviour_fitness, behaviour_results,
                                     normalized_weight, sorted_sample_val, cum_sample, results_sort, cum,
                                     ppu_line_lower, ppu_line_upper, line_min, line_max,
                                     median_prediction, uncertainty_method="likelihood")
        saver.save(save.proc_path)
        return saver


def likelihood_uncertaintyMO(raw_saver, n_obj, thresholds, ppu, obj_weights):
    """
    This is an uncertainty analysis function for multi-objective optimization algorithms.
    This function will generate a pycup.save.ProcResultSaver object, which contains all the analysis results.
    Users can also change the saving path by modifying the
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
        raise TypeError("The given saver object is not a save.RawDataSaver.")

    if len(obj_weights) != n_obj:
        raise TypeError("The length of objective function weights should be equal to n_obj.")

    if raw_saver.opt_type == "GLUE":
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

    else:
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

        hs = np.concatenate(hs)
        hf = np.concatenate(hf)
        if Reslib.UseResObject:
            hr = np.concatenate(raw_saver.historical_results.data)
        else:
            hr = np.concatenate(hr)

    paretoPops = raw_saver.pareto_samples
    paretoFits = raw_saver.pareto_fitness
    paretoRes = raw_saver.pareto_results

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
        print("No behaviour parameter has been found, please adjust the threshold value.")
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

        # 3. calculate the posterior distribution
        sorted_sample_val, sorted_sample_id = np.sort(behaviour_samples, axis=0), np.argsort(behaviour_samples, axis=0)
        id_sample_column = np.array([np.arange(behaviour_samples.shape[1]) for i in range(behaviour_samples.shape[0])])
        normalized_sample_weight = np.array(
            [w_normalized_weight.flatten() for i in range(behaviour_samples.shape[1])]).T
        normalized_weight_sort = normalized_sample_weight[sorted_sample_id, id_sample_column]
        cum_sample = np.cumsum(normalized_weight_sort, axis=0)

        if not Reslib.UseResObject:

            ##3.5 continue
            normalized_weight_arr = np.array([w_normalized_weight.flatten() for i in range(
                behaviour_results.shape[1])]).T  # shape = (num_behaviour_samples,time_steps)

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

        else:
            stations = list(hr[0].keys())

            ppu_line_lower = Reslib.SimulationResult()
            ppu_line_upper = Reslib.SimulationResult()
            line_min = Reslib.SimulationResult()
            line_max = Reslib.SimulationResult()
            median_prediction = Reslib.SimulationResult()

            cum = np.array([Reslib.SimulationResult() for i in range(len(behaviour_results))], dtype=object)
            results_sort = np.array([Reslib.SimulationResult() for i in range(len(behaviour_results))], dtype=object)

            for st in stations:

                ppu_line_lower.add_station(st)
                ppu_line_upper.add_station(st)
                line_min.add_station(st)
                line_max.add_station(st)
                median_prediction.add_station(st)

                for i in range(len(behaviour_results)):
                    cum[i].add_station(st)
                    results_sort[i].add_station(st)

                events = list(hr[0][st].keys())
                for e in events:

                    normalized_weight_arr = np.array([w_normalized_weight.flatten() for i in range(behaviour_results[0][st][e].shape[1])]).T  # shape = (num_behaviour_samples,time_steps)

                    st_e_br = []  # station event behavioral results
                    for i in behaviour_results:
                        data = i[st][e]
                        data = np.array(data).reshape(1, -1)
                        st_e_br.append(data)
                    st_e_br = np.concatenate(st_e_br, axis=0)

                    # 5. sort the results for each time step, manipulate the weight according to the indices, calculate the cumulative weight
                    st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)
                    id_column = np.array(
                        [np.arange(st_e_br.shape[1]) for i in range(st_e_br.shape[0])])
                    normalized_fit_sort = normalized_weight_arr[id, id_column]
                    st_e_cum = np.cumsum(normalized_fit_sort, axis=0)

                    for i in range(len(behaviour_results)):
                        cum[i][st].add_event(e, st_e_cum[i])
                        results_sort[i][st].add_event(e, st_e_results_sort[i])

                    # 6. calculate the lower and upper uncertainty band, generate the max and min bound
                    st_e_ppu_line_lower, st_e_ppu_line_upper = gen_ppu_bound(ppu, st_e_cum, st_e_results_sort)
                    st_e_line_min = st_e_results_sort[0, :].flatten()
                    st_e_line_max = st_e_results_sort[-1, :].flatten()

                    # 7. calculate the 50% percentile prediction
                    st_e_median_prediction = get_median_series(st_e_br)

                    ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                    ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                    line_min[st].add_event(e, st_e_line_min)
                    line_max[st].add_event(e, st_e_line_max)
                    median_prediction[st].add_event(e, st_e_median_prediction)
            cum = Reslib.ResultDataPackage(cum, method_info="Cumulative weights")
            results_sort = Reslib.ResultDataPackage(results_sort, method_info="Sorter results")
            behaviour_results = Reslib.ResultDataPackage(behaviour_results, method_info="Behavioral results")
            hr = Reslib.ResultDataPackage(hr, method_info="Historical results")

        # if this is a RawSaver processed by the TOPSIS.TopsisAnalyzer
        if hasattr(raw_saver, "TOPSISidx"):
            best_sample = raw_saver.GbestPosition
            best_fitness = raw_saver.GbestScore
            best_result = paretoRes[raw_saver.TOPSISidx]
        else:
            best_sample = None
            best_fitness = None
            best_result = None
        saver = save.ProcResultSaver(hs, hf, hr, best_sample, best_fitness, best_result, behaviour_samples,
                                     behaviour_fitness,
                                     behaviour_results,
                                     w_normalized_weight, sorted_sample_val, cum_sample, results_sort, cum,
                                     ppu_line_lower, ppu_line_upper, line_min, line_max, median_prediction, paretoPops,
                                     paretoFits,
                                     paretoRes, uncertainty_method="likelihood")
        saver.save(save.proc_path)
        return saver


def frequency_uncertainty(raw_saver, threshold, ppu, intervals=10, approximation=True):
    """
    This is an output/simulation uncertainty analysis function for optimization algorithms.
    This function will generate a pycup.save.ProcResultSaver object, which contains all the analysis results. The
    save method of this object will also be called. Users can also change the saving path by modifying the
    pycup.save.proc_path variable (see the document).

    This method estimates the prediction uncertainty band based on the frequency of the output values. It is only used
    when for the prediction uncertainty analysis of behavioral samples. It is based on the uncertainty estimation method
    used in SWAT-CUP (https://www.youtube.com/watch?v=oMhTLpPJeeU). It calculates the histograms of the output values
    (of the behavioral solutions) and find the user defined ppu values according to the accumulative frequency
    distribution.

    :param raw_saver: save.RawDataSaver object (optimization result file)
    :param threshold: threshold value for extracting the behavioral samples -> float
    :param ppu: prediction uncertainty level -> float or int, typically 95 (95PPU).
         If the given value < 1.0, it will be treated as value * 100, therefore, 0.95 is also acceptable
    :param intervals: the number of intervals when calculating the frequency distribution.
    :param approximation: whether calculate the approximate value of the ppu value or not. -> bool
           The accumulative frequency distribution is not continuous (e.g. 0.9, 0.93, 0.98, 1.00). This option provides
           the estimation of the specific ppu value (e.g. 0.025 and 0.975). If True, the output value will be calculated
           based on the triangle proportionality theorem. If False, the value that locates closest to the ppu bound will
           be used.
    :param st_id:
    :return:
    """
    if not (isinstance(raw_saver, save.RawDataSaver)):
        raise TypeError("The given saver object is not a save.RawDataSaver.")
    if raw_saver.opt_type == "GLUE":
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

    else:
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

        hs = np.concatenate(hs)
        hf = np.concatenate(hf)
        if Reslib.UseResObject:
            hr = np.concatenate(raw_saver.historical_results.data)
        else:
            hr = np.concatenate(hr)

    # 1. sort the samples and fitness (likelihood function values)
    sorted_fitness, index_l = SortFitness(hf)  ## ascending according to the fitness
    sorted_samples = SortPosition(hs, index_l)
    if not Reslib.UseResObject:
        sorted_results = SortPosition(hr, index_l)
    else:
        sorted_results = hr[index_l].flatten()
    best_sample = sorted_samples[0]
    best_fitness = sorted_fitness[0]
    best_result = sorted_results[0]

    # 2. find the behaviour_samples and fitness according to the threshold
    ## the likelihood function would be errors (e.g. RMSE, MAE), of which a lower value denotes a btter performance
    ## for metrics such as NSE or R2, (1-NSE) or (1-R2) should be used.
    behaviour_samples = np.array(
        [sorted_samples[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    behaviour_fitness = np.array(
        [sorted_fitness[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    if not Reslib.UseResObject:
        behaviour_results = np.array(
            [sorted_results[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold])
    else:
        behaviour_results = np.array(
            [sorted_results[i] for i in range(len(sorted_fitness)) if sorted_fitness[i] < threshold], dtype=object)

    if len(behaviour_samples) == 0:
        print("No behaviour parameter has been found, please adjust the threshold value")
        return
    else:
        if not Reslib.UseResObject:
            results_sort = np.sort(behaviour_results, axis=0)
            line_min = results_sort[0, :].flatten()
            line_max = results_sort[-1, :].flatten()
            # 3. frequency statistic
            counts = [np.histogram(results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                      range(results_sort.shape[1])]
            counts = np.concatenate(counts, axis=1)
            sections = [np.histogram(results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                        range(results_sort.shape[1])]
            sections = np.concatenate(sections, axis=1)
            freq = counts / behaviour_samples.shape[0]
            freq = np.concatenate([np.zeros((1, results_sort.shape[1])), freq], axis=0)
            cum_freq = np.cumsum(freq, axis=0)
            ppu_line_lower, ppu_line_upper = gen_frequency_ppu_bound(ppu, cum_freq, sections, approximation)

            # This is for the over boundary problem of the np.histogram api when all the data = 0
            ppu_line_upper = ppu_line_upper.flatten()
            ppu_line_lower = ppu_line_lower.flatten()
            ob_low = np.argwhere(ppu_line_lower < line_min)
            ppu_line_lower[ob_low] = line_min[ob_low]
            ob_up = np.argwhere(ppu_line_upper > line_max)
            ppu_line_upper[ob_up] = line_max[ob_up]

            # 4. calculate the 50% percentile prediction
            median_prediction = get_median_series(behaviour_results)
        else:
            stations = list(hr[0].keys())

            ppu_line_lower = Reslib.SimulationResult()
            ppu_line_upper = Reslib.SimulationResult()
            line_min = Reslib.SimulationResult()
            line_max = Reslib.SimulationResult()
            median_prediction = Reslib.SimulationResult()
            cum_freq = np.array([Reslib.SimulationResult() for i in range(intervals + 1)], dtype=object)
            results_sort = np.array([Reslib.SimulationResult() for i in range(len(behaviour_results))], dtype=object)

            for st in stations:

                ppu_line_lower.add_station(st)
                ppu_line_upper.add_station(st)
                line_min.add_station(st)
                line_max.add_station(st)
                median_prediction.add_station(st)
                for i in range(len(behaviour_results)):
                    results_sort[i].add_station(st)
                for i in range(intervals + 1):
                    cum_freq[i].add_station(st)
                events = list(hr[0][st].keys())
                for e in events:
                    st_e_br = []  # station event behavioral results
                    for i in behaviour_results:
                        data = i[st][e]
                        data = np.array(data).reshape(1, -1)
                        st_e_br.append(data)
                    st_e_br = np.concatenate(st_e_br, axis=0)
                    st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)

                    counts = [np.histogram(st_e_results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                              range(st_e_results_sort.shape[1])]
                    counts = np.concatenate(counts, axis=1)
                    sections = [np.histogram(st_e_results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                                range(st_e_results_sort.shape[1])]
                    sections = np.concatenate(sections, axis=1)
                    freq = counts / behaviour_samples.shape[0]
                    freq = np.concatenate([np.zeros((1, st_e_results_sort.shape[1])), freq], axis=0)
                    st_e_cum_freq = np.cumsum(freq, axis=0)

                    for i in range(len(behaviour_results)):
                        results_sort[i][st].add_event(e, st_e_results_sort[i])
                    for i in range(intervals + 1):
                        cum_freq[i][st].add_event(e, st_e_cum_freq[i])

                    st_e_ppu_line_lower, st_e_ppu_line_upper = gen_frequency_ppu_bound(ppu, st_e_cum_freq, sections,
                                                                                       approximation)
                    st_e_line_min = st_e_results_sort[0, :].flatten()
                    st_e_line_max = st_e_results_sort[-1, :].flatten()

                    # This is for the over boundary problem of the np.histogram api when all the data = 0
                    st_e_ppu_line_lower = st_e_ppu_line_lower.flatten()
                    st_e_ppu_line_upper =  st_e_ppu_line_upper.flatten()
                    ob_low = np.argwhere(st_e_ppu_line_lower<st_e_line_min)
                    st_e_ppu_line_lower[ob_low] = st_e_line_min[ob_low]
                    ob_up = np.argwhere(st_e_ppu_line_upper > st_e_line_max)
                    st_e_ppu_line_upper[ob_up] = st_e_line_max[ob_up]

                    # 4. calculate the 50% percentile prediction
                    st_e_median_prediction = get_median_series(st_e_br)

                    ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                    ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                    line_min[st].add_event(e, st_e_line_min)
                    line_max[st].add_event(e, st_e_line_max)
                    median_prediction[st].add_event(e, st_e_median_prediction)

            cum_freq = Reslib.ResultDataPackage(cum_freq, method_info="Cumulative weights")
            results_sort = Reslib.ResultDataPackage(results_sort, method_info="Sorter results")
            behaviour_results = Reslib.ResultDataPackage(behaviour_results, method_info="Behavioral results")
            hr = Reslib.ResultDataPackage(hr, method_info="Historical results")

        saver = save.ProcResultSaver(hs, hf, hr, best_sample, best_fitness, best_result, behaviour_samples,
                                     behaviour_fitness, behaviour_results,
                                     None, None, None, results_sort, cum_freq, ppu_line_lower, ppu_line_upper, line_min,
                                     line_max, median_prediction,
                                     uncertainty_method="frequency")
        saver.save(save.proc_path)
        return saver


def frequency_uncertaintyMO(raw_saver, n_obj, thresholds, ppu, intervals=10, approximation=True):
    """
    This is an uncertainty analysis function for multi-objective optimization algorithms.
    This function will generate a pycup.save.ProcResultSaver object, which contains all the analysis results.
    Users can also change the saving path by modifying the
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
        raise TypeError("The given saver object is not a save.RawDataSaver.")

    if raw_saver.opt_type == "GLUE":
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

    else:
        hr = raw_saver.historical_results
        hf = raw_saver.historical_fitness
        hs = raw_saver.historical_samples

        hs = np.concatenate(hs)
        hf = np.concatenate(hf)
        if Reslib.UseResObject:
            hr = np.concatenate(raw_saver.historical_results.data)
        else:
            hr = np.concatenate(hr)

    paretoPops = raw_saver.pareto_samples
    paretoFits = raw_saver.pareto_fitness
    paretoRes = raw_saver.pareto_results

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
        print("No behaviour parameter has been found, please adjust the threshold value.")
        return
    else:
        behaviour_samples = hs[behaviour_id]
        behaviour_fitness = hf[behaviour_id]
        behaviour_results = hr[behaviour_id]

        if not Reslib.UseResObject:
            # 2. sort the results for each time step
            results_sort, id = np.sort(behaviour_results, axis=0), np.argsort(behaviour_results, axis=0)

            line_min = results_sort[0, :].flatten()
            line_max = results_sort[-1, :].flatten()

            # 3. frequency statistic
            counts = [np.histogram(results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                      range(results_sort.shape[1])]
            counts = np.concatenate(counts, axis=1)
            sections = [np.histogram(results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                        range(results_sort.shape[1])]
            sections = np.concatenate(sections, axis=1)
            freq = counts / behaviour_samples.shape[0]
            freq = np.concatenate([np.zeros((1, results_sort.shape[1])), freq], axis=0)
            cum_freq = np.cumsum(freq, axis=0)
            ppu_line_lower, ppu_line_upper = gen_frequency_ppu_bound(ppu, cum_freq, sections, approximation)

            # This is for the over boundary problem of the np.histogram api when all the data = 0
            ppu_line_upper = ppu_line_upper.flatten()
            ppu_line_lower = ppu_line_lower.flatten()
            ob_low = np.argwhere(ppu_line_lower < line_min)
            ppu_line_lower[ob_low] = line_min[ob_low]
            ob_up = np.argwhere(ppu_line_upper > line_max)
            ppu_line_upper[ob_up] = line_max[ob_up]

            # 4. calculate the 50% percentile prediction
            median_prediction = get_median_series(behaviour_results)
        else:
            stations = list(hr[0].keys())

            ppu_line_lower = Reslib.SimulationResult()
            ppu_line_upper = Reslib.SimulationResult()
            line_min = Reslib.SimulationResult()
            line_max = Reslib.SimulationResult()
            median_prediction = Reslib.SimulationResult()
            cum_freq = np.array([Reslib.SimulationResult() for i in range(intervals + 1)], dtype=object)
            results_sort = np.array([Reslib.SimulationResult() for i in range(len(behaviour_results))], dtype=object)

            for st in stations:

                ppu_line_lower.add_station(st)
                ppu_line_upper.add_station(st)
                line_min.add_station(st)
                line_max.add_station(st)
                median_prediction.add_station(st)
                for i in range(len(behaviour_results)):
                    results_sort[i].add_station(st)
                for i in range(intervals + 1):
                    cum_freq[i].add_station(st)
                events = list(hr[0][st].keys())
                for e in events:
                    st_e_br = []  # station event behavioral results
                    for i in behaviour_results:
                        data = i[st][e]
                        data = np.array(data).reshape(1, -1)
                        st_e_br.append(data)
                    st_e_br = np.concatenate(st_e_br, axis=0)
                    st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)

                    counts = [np.histogram(st_e_results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                              range(st_e_results_sort.shape[1])]
                    counts = np.concatenate(counts, axis=1)
                    sections = [np.histogram(st_e_results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                                range(st_e_results_sort.shape[1])]
                    sections = np.concatenate(sections, axis=1)
                    freq = counts / behaviour_samples.shape[0]
                    freq = np.concatenate([np.zeros((1, st_e_results_sort.shape[1])), freq], axis=0)
                    st_e_cum_freq = np.cumsum(freq, axis=0)

                    for i in range(len(behaviour_results)):
                        results_sort[i][st].add_event(e, st_e_results_sort[i])
                    for i in range(intervals + 1):
                        cum_freq[i][st].add_event(e, st_e_cum_freq[i])

                    st_e_ppu_line_lower, st_e_ppu_line_upper = gen_frequency_ppu_bound(ppu, st_e_cum_freq, sections,
                                                                                       approximation)
                    st_e_line_min = st_e_results_sort[0, :].flatten()
                    st_e_line_max = st_e_results_sort[-1, :].flatten()

                    # This is for the over boundary problem of the np.histogram api when all the data = 0
                    st_e_ppu_line_lower = st_e_ppu_line_lower.flatten()
                    st_e_ppu_line_upper =  st_e_ppu_line_upper.flatten()
                    ob_low = np.argwhere(st_e_ppu_line_lower<st_e_line_min)
                    st_e_ppu_line_lower[ob_low] = st_e_line_min[ob_low]
                    ob_up = np.argwhere(st_e_ppu_line_upper > st_e_line_max)
                    st_e_ppu_line_upper[ob_up] = st_e_line_max[ob_up]

                    # 4. calculate the 50% percentile prediction
                    st_e_median_prediction = get_median_series(st_e_br)

                    ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                    ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                    line_min[st].add_event(e, st_e_line_min)
                    line_max[st].add_event(e, st_e_line_max)
                    median_prediction[st].add_event(e, st_e_median_prediction)

            cum_freq = Reslib.ResultDataPackage(cum_freq, method_info="Cumulative weights")
            results_sort = Reslib.ResultDataPackage(results_sort, method_info="Sorter results")
            behaviour_results = Reslib.ResultDataPackage(behaviour_results, method_info="Behavioral results")
            hr = Reslib.ResultDataPackage(hr, method_info="Historical results") \
 \
                # if this is a RawSaver processed by the TOPSIS.TopsisAnalyzer
        if hasattr(raw_saver, "TOPSISidx"):
            best_sample = raw_saver.GbestPosition
            best_fitness = raw_saver.GbestScore
            best_result = paretoRes[raw_saver.TOPSISidx]
        else:
            best_sample = None
            best_fitness = None
            best_result = None
        saver = save.ProcResultSaver(hs, hf, hr, best_sample, best_fitness, best_result, behaviour_samples,
                                     behaviour_fitness,
                                     behaviour_results,
                                     None, None, None, results_sort, cum_freq,
                                     ppu_line_lower, ppu_line_upper, line_min, line_max, median_prediction, paretoPops,
                                     paretoFits,
                                     paretoRes, uncertainty_method="frequency")
        saver.save(save.proc_path)
        return saver


def tri_proportionality(results,cum,lower,upper,id_nearest_l,id_nearest_u):
    # for the lower ppu
    if cum[id_nearest_l] < lower:
        id_next = np.argwhere(cum > lower)[0]
        f_l1,f_l2 = cum[id_nearest_l],cum[id_next]
        r_l1,r_l2 = results[id_nearest_l],results[id_next]
        ratio = (lower - f_l1)/(f_l2 - f_l1)
        diff = ratio * (r_l2-r_l1)
        ppu_lower = r_l1 + diff
    elif cum[id_nearest_l] > lower:
        id_last = np.argwhere(cum < lower)[-1]
        f_l1, f_l2 = cum[id_last], cum[id_nearest_l]
        r_l1,r_l2 = results[id_last],results[id_nearest_l]
        ratio = (lower - f_l1)/(f_l2 - f_l1)
        diff = ratio * (r_l2 - r_l1)
        ppu_lower = r_l1 + diff
    else:
        ppu_lower = results[id_nearest_l]

    # for the upper ppu
    if cum[id_nearest_u] < upper:
        f_u1 = cum[id_nearest_u]
        id_next = np.argwhere(cum>upper)[0]
        f_u2 = cum[id_next]
        r_u1, r_u2 = results[id_nearest_u], results[id_next]
        ratio = (f_u2 - upper)/(f_u2 - f_u1)
        diff = ratio * (r_u2 - r_u1)
        ppu_upper = r_u2 - diff
    elif cum[id_nearest_u] > upper:
        id_last = np.argwhere(cum<upper)[-1]
        f_u1 = cum[id_last]
        f_u2 = cum[id_nearest_u]
        r_u1, r_u2 = results[id_last], results[id_nearest_u]
        ratio = (f_u2 - upper)/(f_u2 - f_u1)
        diff = ratio * (r_u2 - r_u1)
        ppu_upper = r_u2 - diff
    else:
        ppu_upper = results[id_nearest_u]
    return ppu_lower,ppu_upper


def gen_frequency_ppu_bound(confidence_value,cum,sections,approximation=True):
    lower,upper = check_ppu(confidence_value)
    ppu_line_lower = []
    ppu_line_upper = []
    for i in range(cum.shape[1]):
        all_results = cum[:, i].flatten()
        nearest_l, id_nearest_l = find_nearest(all_results, lower)
        nearest_u, id_nearest_u = find_nearest(all_results, upper)
        if approximation == False:
            ppu_line_lower.append(sections[id_nearest_l, i])
            ppu_line_upper.append(sections[id_nearest_u, i])
        else:
            ppu_lower,ppu_upper = tri_proportionality(sections[:,i].flatten(),all_results,
                                                      lower,upper,id_nearest_l,id_nearest_u)
            ppu_line_lower.append(ppu_lower)
            ppu_line_upper.append(ppu_upper)
    ppu_line_lower = np.array(ppu_line_lower)
    ppu_line_upper = np.array(ppu_line_upper)
    return ppu_line_lower,ppu_line_upper


def validation_likelihood_uncertainty(vali_raw_saver, ppu):
    if not isinstance(vali_raw_saver, save.ValidationRawSaver):
        raise TypeError("The saver object should be save.ValidationRawSaver")

    if not Reslib.UseResObject:
        results = vali_raw_saver.results
    else:
        results = vali_raw_saver.results.data
    normalized_weight = vali_raw_saver.weights  # shape = (num_behaviour_samples,)
    fitness = vali_raw_saver.fitness

    if not Reslib.UseResObject:
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
    else:
        stations = list(results[0].keys())

        ppu_line_lower = Reslib.SimulationResult()
        ppu_line_upper = Reslib.SimulationResult()
        line_min = Reslib.SimulationResult()
        line_max = Reslib.SimulationResult()
        median_prediction = Reslib.SimulationResult()

        for st in stations:
            ppu_line_lower.add_station(st)
            ppu_line_upper.add_station(st)
            line_min.add_station(st)
            line_max.add_station(st)
            median_prediction.add_station(st)
            events = list(results[0][st].keys())
            for e in events:
                # ResultDataPackage[0] => the first Simulation object
                normalized_weight_arr = np.array(
                    [normalized_weight.flatten() for i in range(results[0][st][e].shape[1])]).T
                st_e_br = []  # station event behavioral results
                for i in results:
                    data = i[st][e]
                    data = np.array(data).reshape(1, -1)
                    st_e_br.append(data)
                st_e_br = np.concatenate(st_e_br, axis=0)
                st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)
                id_column = np.array(
                    [np.arange(st_e_br.shape[1]) for i in range(st_e_br.shape[0])])
                normalized_fit_sort = normalized_weight_arr[id, id_column]
                st_e_cum = np.cumsum(normalized_fit_sort, axis=0)

                st_e_ppu_line_lower, st_e_ppu_line_upper = gen_ppu_bound(ppu, st_e_cum, st_e_results_sort)
                st_e_line_min = st_e_results_sort[0, :].flatten()
                st_e_line_max = st_e_results_sort[-1, :].flatten()

                st_e_median_prediction = get_median_series(st_e_br)

                ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                line_min[st].add_event(e, st_e_line_min)
                line_max[st].add_event(e, st_e_line_max)
                median_prediction[st].add_event(e, st_e_median_prediction)
        results = Reslib.ResultDataPackage(results, method_info="Validation results")

    if fitness.shape[1] == 1:
        best_idx = np.argmin(fitness)
        best_result = results[best_idx]
    else:
        best_result = None

    saver = save.ValidationProcSaver(fitness, results, ppu_line_upper, ppu_line_lower, line_max, line_min, best_result,
                                     median_prediction,
                                     uncertainty_method="likelihood")
    saver.save()
    return saver


def validation_frequency_uncertainty(vali_raw_saver, ppu, intervals=10, approximation=True):
    if not isinstance(vali_raw_saver, save.ValidationRawSaver):
        raise TypeError("The saver object should be save.ValidationRawSaver")

    if not Reslib.UseResObject:
        results = vali_raw_saver.results
    else:
        results = vali_raw_saver.results.data
    fitness = vali_raw_saver.fitness

    if not Reslib.UseResObject:
        results_sort, id = np.sort(results, axis=0), np.argsort(results, axis=0)
        counts = [np.histogram(results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                  range(results_sort.shape[1])]
        counts = np.concatenate(counts, axis=1)
        sections = [np.histogram(results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                    range(results_sort.shape[1])]
        sections = np.concatenate(sections, axis=1)
        freq = counts / results.shape[0]
        freq = np.concatenate([np.zeros((1, results_sort.shape[1])), freq], axis=0)
        cum_freq = np.cumsum(freq, axis=0)
        ppu_line_lower, ppu_line_upper = gen_frequency_ppu_bound(ppu, cum_freq, sections, approximation)
        line_min = results_sort[0, :].flatten()
        line_max = results_sort[-1, :].flatten()

        # This is for the over boundary problem of the np.histogram api when all the data = 0
        ppu_line_upper = ppu_line_upper.flatten()
        ppu_line_lower = ppu_line_lower.flatten()
        ob_low = np.argwhere(ppu_line_lower < line_min)
        ppu_line_lower[ob_low] = line_min[ob_low]
        ob_up = np.argwhere(ppu_line_upper > line_max)
        ppu_line_upper[ob_up] = line_max[ob_up]

        median_prediction = get_median_series(results)
    else:
        stations = list(results[0].keys())

        ppu_line_lower = Reslib.SimulationResult()
        ppu_line_upper = Reslib.SimulationResult()
        line_min = Reslib.SimulationResult()
        line_max = Reslib.SimulationResult()
        median_prediction = Reslib.SimulationResult()
        for st in stations:
            ppu_line_lower.add_station(st)
            ppu_line_upper.add_station(st)
            line_min.add_station(st)
            line_max.add_station(st)
            median_prediction.add_station(st)
            events = list(results[0][st].keys())
            for e in events:
                st_e_br = []  # station event behavioral results
                for i in results:
                    data = i[st][e]
                    data = np.array(data).reshape(1, -1)
                    st_e_br.append(data)
                st_e_br = np.concatenate(st_e_br, axis=0)
                st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)

                counts = [np.histogram(st_e_results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                          range(st_e_results_sort.shape[1])]
                counts = np.concatenate(counts, axis=1)
                sections = [np.histogram(st_e_results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                            range(st_e_results_sort.shape[1])]
                sections = np.concatenate(sections, axis=1)
                freq = counts / results.shape[0]
                freq = np.concatenate([np.zeros((1, st_e_results_sort.shape[1])), freq], axis=0)
                st_e_cum_freq = np.cumsum(freq, axis=0)

                st_e_ppu_line_lower, st_e_ppu_line_upper = gen_frequency_ppu_bound(ppu, st_e_cum_freq, sections,
                                                                                   approximation)
                st_e_line_min = st_e_results_sort[0, :].flatten()
                st_e_line_max = st_e_results_sort[-1, :].flatten()

                # This is for the over boundary problem of the np.histogram api when all the data = 0
                st_e_ppu_line_lower = st_e_ppu_line_lower.flatten()
                st_e_ppu_line_upper = st_e_ppu_line_upper.flatten()
                ob_low = np.argwhere(st_e_ppu_line_lower < st_e_line_min)
                st_e_ppu_line_lower[ob_low] = st_e_line_min[ob_low]
                ob_up = np.argwhere(st_e_ppu_line_upper > st_e_line_max)
                st_e_ppu_line_upper[ob_up] = st_e_line_max[ob_up]

                # 4. calculate the 50% percentile prediction
                st_e_median_prediction = get_median_series(st_e_br)

                ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                line_min[st].add_event(e, st_e_line_min)
                line_max[st].add_event(e, st_e_line_max)
                median_prediction[st].add_event(e, st_e_median_prediction)
        results = Reslib.ResultDataPackage(results, method_info="Validation results")

    if fitness.shape[1] == 1:
        best_idx = np.argmin(fitness)
        best_result = results[best_idx]
    else:
        best_result = None

    saver = save.ValidationProcSaver(fitness, results, ppu_line_upper, ppu_line_lower, line_max, line_min, best_result,
                                     median_prediction,
                                     uncertainty_method="frequency")
    saver.save()
    return saver


def prediction_likelihood_uncertainty(pred_raw_saver, ppu):
    if not isinstance(pred_raw_saver, save.PredRawSaver):
        raise TypeError("The saver object should be save.PredRawSaver")
    normalized_weight = pred_raw_saver.weights  # shape = (num_behaviour_samples,)

    if not Reslib.UseResObject:
        results = pred_raw_saver.results
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
    else:
        results = pred_raw_saver.results.data
        stations = list(results[0].keys())
        ppu_line_lower = Reslib.SimulationResult()
        ppu_line_upper = Reslib.SimulationResult()
        line_min = Reslib.SimulationResult()
        line_max = Reslib.SimulationResult()
        median_prediction = Reslib.SimulationResult()
        for st in stations:
            ppu_line_lower.add_station(st)
            ppu_line_upper.add_station(st)
            line_min.add_station(st)
            line_max.add_station(st)
            median_prediction.add_station(st)
            events = list(results[0][st].keys())
            for e in events:
                # ResultDataPackage[0] => the first Simulation object
                normalized_weight_arr = np.array(
                    [normalized_weight.flatten() for i in range(results[0][st][e].shape[1])]).T
                st_e_br = []  # station event behavioral results
                for i in results:
                    data = i[st][e]
                    data = np.array(data).reshape(1, -1)
                    st_e_br.append(data)
                st_e_br = np.concatenate(st_e_br, axis=0)
                st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)
                id_column = np.array(
                    [np.arange(st_e_br.shape[1]) for i in range(st_e_br.shape[0])])
                normalized_fit_sort = normalized_weight_arr[id, id_column]
                st_e_cum = np.cumsum(normalized_fit_sort, axis=0)

                st_e_ppu_line_lower, st_e_ppu_line_upper = gen_ppu_bound(ppu, st_e_cum, st_e_results_sort)
                st_e_line_min = st_e_results_sort[0, :].flatten()
                st_e_line_max = st_e_results_sort[-1, :].flatten()

                st_e_median_prediction = get_median_series(st_e_br)

                ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                line_min[st].add_event(e, st_e_line_min)
                line_max[st].add_event(e, st_e_line_max)
                median_prediction[st].add_event(e, st_e_median_prediction)
        results = Reslib.ResultDataPackage(results, method_info="Validation results")
    saver = save.PredProcSaver(results, ppu_line_upper, ppu_line_lower, line_max, line_min, median_prediction,
                               uncertainty_method="likelihood")
    saver.save()
    return saver


def prediction_frequency_uncertainty(pred_raw_saver, ppu, intervals=10, approximation=True):
    if not isinstance(pred_raw_saver, save.PredRawSaver):
        raise TypeError("The saver object should be save.PredRawSaver")

    if not Reslib.UseResObject:
        results = pred_raw_saver.results
        results_sort, id = np.sort(results, axis=0), np.argsort(results, axis=0)
        line_min = results_sort[0, :].flatten()
        line_max = results_sort[-1, :].flatten()
        counts = [np.histogram(results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                  range(results_sort.shape[1])]
        counts = np.concatenate(counts, axis=1)
        sections = [np.histogram(results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                    range(results_sort.shape[1])]
        sections = np.concatenate(sections, axis=1)
        freq = counts / results.shape[0]
        freq = np.concatenate([np.zeros((1, results_sort.shape[1])), freq], axis=0)
        cum_freq = np.cumsum(freq, axis=0)
        ppu_line_lower, ppu_line_upper = gen_frequency_ppu_bound(ppu, cum_freq, sections, approximation)

        ppu_line_upper = ppu_line_upper.flatten()
        ppu_line_lower = ppu_line_lower.flatten()
        ob_low = np.argwhere(ppu_line_lower < line_min)
        ppu_line_lower[ob_low] = line_min[ob_low]
        ob_up = np.argwhere(ppu_line_upper > line_max)
        ppu_line_upper[ob_up] = line_max[ob_up]

        median_prediction = get_median_series(results)
    else:
        results = pred_raw_saver.results.data
        results = pred_raw_saver.results.data
        stations = list(results[0].keys())
        ppu_line_lower = Reslib.SimulationResult()
        ppu_line_upper = Reslib.SimulationResult()
        line_min = Reslib.SimulationResult()
        line_max = Reslib.SimulationResult()
        median_prediction = Reslib.SimulationResult()
        for st in stations:
            ppu_line_lower.add_station(st)
            ppu_line_upper.add_station(st)
            line_min.add_station(st)
            line_max.add_station(st)
            median_prediction.add_station(st)
            events = list(results[0][st].keys())
            for e in events:
                st_e_br = []  # station event behavioral results
                for i in results:
                    data = i[st][e]
                    data = np.array(data).reshape(1, -1)
                    st_e_br.append(data)
                st_e_br = np.concatenate(st_e_br, axis=0)
                st_e_results_sort, id = np.sort(st_e_br, axis=0), np.argsort(st_e_br, axis=0)

                counts = [np.histogram(st_e_results_sort[:, i], bins=intervals)[0].reshape(-1, 1) for i in
                          range(st_e_results_sort.shape[1])]
                counts = np.concatenate(counts, axis=1)
                sections = [np.histogram(st_e_results_sort[:, i], bins=intervals)[1].reshape(-1, 1) for i in
                            range(st_e_results_sort.shape[1])]
                sections = np.concatenate(sections, axis=1)
                freq = counts / results.shape[0]
                freq = np.concatenate([np.zeros((1, st_e_results_sort.shape[1])), freq], axis=0)
                st_e_cum_freq = np.cumsum(freq, axis=0)

                st_e_ppu_line_lower, st_e_ppu_line_upper = gen_frequency_ppu_bound(ppu, st_e_cum_freq, sections,
                                                                                   approximation)
                st_e_line_min = st_e_results_sort[0, :].flatten()
                st_e_line_max = st_e_results_sort[-1, :].flatten()

                # This is for the over boundary problem of the np.histogram api when all the data = 0
                st_e_ppu_line_lower = st_e_ppu_line_lower.flatten()
                st_e_ppu_line_upper = st_e_ppu_line_upper.flatten()
                ob_low = np.argwhere(st_e_ppu_line_lower < st_e_line_min)
                st_e_ppu_line_lower[ob_low] = st_e_line_min[ob_low]
                ob_up = np.argwhere(st_e_ppu_line_upper > st_e_line_max)
                st_e_ppu_line_upper[ob_up] = st_e_line_max[ob_up]

                # 4. calculate the 50% percentile prediction
                st_e_median_prediction = get_median_series(st_e_br)

                ppu_line_lower[st].add_event(e, st_e_ppu_line_lower)
                ppu_line_upper[st].add_event(e, st_e_ppu_line_upper)
                line_min[st].add_event(e, st_e_line_min)
                line_max[st].add_event(e, st_e_line_max)
                median_prediction[st].add_event(e, st_e_median_prediction)
        results = Reslib.ResultDataPackage(results, method_info="Validation results")
    saver = save.PredProcSaver(results, ppu_line_upper, ppu_line_lower, line_max, line_min, median_prediction,
                               uncertainty_method="frequency")
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