import numpy as np
from . import save
"""
This is a module for selecting the best solution from the non-dominated solutions (Pareto front)
based on the TOPSIS method (C.L.Hwang and K.Yoon, 1981). This module is only used for the cases
of multi-objective optimization results, for example (NSGA-II, MOPSO, and MODE).

Note: the algorithms this package were designed to minimize the objective function values, therefore,
the ideal solution here is the solution with the lowest objective function values. (np.argmin)
"""


def calc_ideal(data):
    idx_best = np.argmin(data,axis=0)
    idx_worst = np.argmax(data,axis=0)
    idxy = np.arange(data.shape[1])
    A_plus = data[idx_best,idxy]
    A_minus = data[idx_worst,idxy]
    return A_plus,A_minus

def calc_distances(A_plus,A_minus,data):
    D_plus = np.linalg.norm(data-A_plus,axis=1)
    D_minus = np.linalg.norm(data-A_minus,axis=1)
    return D_plus,D_minus


def normalization(data):
    K = np.power(np.sum(pow(data,2),axis = 0),0.5)
    return data/K


def topsis(pareto_front, weights=None):
    """
    This is a function for simple TOPSIS analysis to find the optimal solution from the pareto front.
    :param pareto_front: The 2D-numpy array of the fitness matrix of the non-dominated solutions.
    :param weights: a list/tuple/numpy.ndarray of weights of the decision variables. The summation of this list should
                    equal to 1.0. e.g. [0.5,0.5]
    :return: The optimal solution; the index (location) of it; the TOPSIS score of it; all the TOPSIS scores.
    Usage:
    import pycup
    res = pycup.save.RawDataSaver.load(r"raw00.rst")
    opt_fitness,opt_idx,score,scores = topsis(res.pareto_fitness)
    print(opt_sln)
    print(score)
    print(scores)
    """
    if weights:
        if not isinstance(weights,list) and not isinstance(weights,tuple) and not isinstance(weights,np.ndarray):
            raise TypeError("The weighting vector should be an iterable.")
        if len(weights) != pareto_front.shape[1]:
            raise ValueError("The length of the weighting vector should equal to the number of decision variables.")
        if np.sum(weights) != 1:
            raise ValueError("The summation of the weights should be 1.")
    else:
        weights = np.ones(pareto_front.shape[1]) * 1/pareto_front.shape[1]
    n_data = normalization(pareto_front) * weights
    A_plus, A_minus = calc_ideal(n_data)
    D_plus,D_minus = calc_distances(A_plus,A_minus,n_data)
    scores = D_minus/(D_plus+D_minus)
    idx_best = np.argmax(scores)
    optimal_fitness = pareto_front[idx_best]
    optimal_score = scores[idx_best]
    return optimal_fitness,idx_best,optimal_score,scores


class TopsisAnalyzer:
    """
    This is an object for the formal TOPSIS analysis

    Usage:
    import pycup
    res = pycup.save.RawDataSaver.load(r"raw00.rst")
    top = TopsisAnalyzer(res,[0.8,0.2])
    print(top.OptimalFitness)
    print(top.OptimalFitness)
    # users can change the weight and re-analyze.
    top.weights = [0.5,0.5]
    top.analyze()
    print(top.OptimalFitness)
    print(top.OptimalSolution)
    print(top.OptimalResult)
    """

    def __init__(self,raw_saver,weights=None):
        if not isinstance(raw_saver,save.RawDataSaver):
            raise TypeError("The argument raw_saver should be pycup.save.RawDataSaver object.")
        self._raw_saver = raw_saver
        self._pareto_samples = self._raw_saver.pareto_samples
        self._pareto_fitness = self._raw_saver.pareto_fitness
        self._pareto_results = self._raw_saver.pareto_results
        self.weights = weights
        self._weights_check()
        self.analyze()

    def _weights_check(self):
        if self.weights:
            if not isinstance(self.weights, list) and not isinstance(self.weights, tuple) and not isinstance(self.weights, np.ndarray):
                raise TypeError("The weighting vector should be an iterable.")
            if len(self.weights) != self._pareto_fitness.shape[1]:
                raise ValueError("The length of the weighting vector should equal to the number of decision variables.")
            if np.sum(self.weights) != 1:
                raise ValueError("The summation of the weights should be 1.")
        else:
            self.weights = np.ones(self._pareto_fitness.shape[1]) * 1 / self._pareto_fitness.shape[1]

    def analyze(self):
        n_data = normalization(self._pareto_fitness) * self.weights
        A_plus, A_minus = calc_ideal(n_data)
        D_plus, D_minus = calc_distances(A_plus, A_minus, n_data)
        scores = D_minus / (D_plus + D_minus)
        idx_best = np.argmax(scores)
        self.OptimalFitness = self._pareto_fitness[idx_best]
        self.OptimalTopsisScore = scores[idx_best]
        self.TopsisScores = scores
        self.OptimalSolution = self._pareto_samples[idx_best]
        self.OptimalResult = self._pareto_results[idx_best]
        self.IdxBest = idx_best

    def updateTopsisRawSaver(self,path):
        """
        :param path: The saving path of the updated raw data saver.
        :return:
        NOTE: the new attribute ".TOPSISidx" is the index of the TOPSIS optimum in the pareto_front (pareto_fitness)
        rather than the historical fitness. The user can get the optimum parameter set (solution) by using
        RawDataSaver.pareto_samples[RawDataSaver.TOPSISidx]
        """
        self._raw_saver.GbestPosition = self.OptimalSolution
        self._raw_saver.GbestScore = self.OptimalFitness
        self._raw_saver.TOPSISidx = self.IdxBest
        self._raw_saver.save(path)

