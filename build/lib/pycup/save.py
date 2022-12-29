import pickle
proc_path = None
raw_path = None
val_path = None
pred_path = None
record_path = None
proc_pathMV = []
raw_pathMV = []
val_pathMV = []
pred_pathMV = []


class ProcResultSaver:

    def __init__(self,
                 historical_samples,
                 historical_fitness,
                 historical_results,
                 best_sample,
                 best_fitness,
                 best_result,
                 behaviour_samples,
                 behaviour_fitness,
                 behaviour_results,
                 normalized_weight,
                 sorted_sample_val,
                 cum_sample,
                 result_sort,
                 cum,
                 ppu_line_lower,
                 ppu_line_upper,
                 line_min,
                 line_max,
                 median_prediction,
                 paretoPops = None,
                 paretoFits = None,
                 paretoRes = None
                 ):
        """
        Uncertainty analysis processed result object

        Usage (load the result):
        import pycup as cp
        opt_res = cp.save.ProcResultSaver.load(r"ProcResult.rst")

        from pycup import save
        save.proc_path = r"D:\example02.txt"
        opt_res = save.ProcResultSaver.load(r"D:\example02.txt")
        """
        self.historical_results = HistoricalResults(historical_samples,historical_fitness,historical_results)
        self.best_result = BestResult(best_sample,best_fitness,best_result)
        self.behaviour_results = BehaviourResults(behaviour_samples,behaviour_fitness,behaviour_results,normalized_weight)
        self.posterior_results = PosteriorResults(sorted_sample_val,cum_sample)
        self.uncertain_results = UncertaintyBandResults(result_sort,cum,ppu_line_lower,ppu_line_upper,line_min,line_max)
        self.pareto_result = ParetoResult(paretoPops,paretoFits,paretoRes)
        self.median_prediction = median_prediction

    def save(self,path=proc_path):
        if path is None:
            path = "ProcResult.rst"
        with open(path, 'wb') as f:
            str = pickle.dumps(self,protocol=pickle.HIGHEST_PROTOCOL)
            f.write(str)

    @classmethod
    def load(cls,path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj,ProcResultSaver):
            raise TypeError("The input saver object should be pycup.save.ProcResultSaver.")
        return obj

    @classmethod
    def help(cls):
        print(
            """
            Attributes & sub-attributes:
            .historical_samples: all the behaviour and non-behaviour samples
            .historical_fitness: the calculated fitness of behaviour and non-behaviour samples
            .historical_results: the simulated results of behaviour and non-behaviour samples
            .best_sample: the global optimal sample/solution     
            .best_fitness: the global optimal fitness
            .best_result: the simulation result of the global optimal sample/solution  
            .behaviour_samples: the behavioural samples/solutions according to the given threshold     
            .behaviour_fitness: the behavioural fitness
            .behaviour_results: the behavioural results of the global optimal sample/solution    
            .normalized_weight: the normalized weight of samples array generated by the uncertainty analysis function
            .sorted_sample_val: the sorted sample values in ascending order
            .cum_sample: the cumulative weight (probability) array, the order is same as the sorted normalized weight
            .result_sort: the sorted result value array (for uncertainty band plotting)
            .cum: the cumulative weight (probability) array, the order is same as the sorted behavioural result values
            .ppu_line_lower: the lower PPU boundary
            .ppu_line_upper: the upper PPU boundary
            .line_min: the minimum prediction result line
            .line_max: the maximum prediction result line
            .paretoPops = the pareto optimal samples (only for multi-objective swarm optimization algorithms)  
            .paretoFits = the pareto optimal fitness (only for multi-objective swarm optimization algorithms) 
            .paretoRes = the corresponding simulation results of pareto optimal samples (only for multi-objective swarm optimization algorithms)          
            """
        )


class BestResult:

    def __init__(self,best_sample,best_fitness,best_result):
        self.best_sample = best_sample
        self.best_fitness = best_fitness
        self.best_results = best_result


class ParetoResult:
    def __init__(self,paretoPops,paretoFits,paretoRes):
        self.pareto_samples = paretoPops
        self.pareto_fitness = paretoFits
        self.pareto_results = paretoRes


class BehaviourResults:

    def __init__(self,behaviour_samples,behaviour_fitness,behaviour_results,normalized_weight):
        self.behaviour_samples = behaviour_samples
        self.behaviour_fitness = behaviour_fitness
        self.behaviour_results = behaviour_results
        self.normalized_weight = normalized_weight


class HistoricalResults:

    def __init__(self,historical_samples,historical_fitness,historical_results):
        self.historical_samples = historical_samples
        self.historical_fitness = historical_fitness
        self.historical_results = historical_results


class PosteriorResults:

    def __init__(self,sorted_sample_val,cum_sample):
        self.sorted_sample_val = sorted_sample_val
        self.cum_sample = cum_sample


class UncertaintyBandResults:

    def __init__(self,result_sort,cum,ppu_line_lower,ppu_line_upper,line_min,line_max):
        """
        Attributes:
        1.  result_sort: the sorted behaviour simulation results according to their values at each time step.
            np.array, shape = (n_behaviour_samples,time_steps)
        2.  cum_weight: the cumulative likelihood weights of the sorted simulation results, those two things can be used
            to calculate the 90 or 95 ppu lines.
            np.array, shape = (n_behaviour_samples,time_steps)
        3.  ppu_line_lower & ppu_line_upper: the processed lower and upper uncertainty boundaries.
            list, length = time_steps
        """
        self.result_sort = result_sort
        self.cum_weight = cum
        self.ppu_line_lower = ppu_line_lower
        self.ppu_line_upper = ppu_line_upper
        self.line_min = line_min
        self.line_max = line_max


class RawDataSaver:

    def __init__(self,hs,hf,hr,bs=None,bp=None,c=None, paretoPops=None, paretoFits=None, paretoRes=None,OptType="SWARM"):
        """
        This object is to save the un-processed data, if you only want the optimization result in stead of the uncertainty
        analysis result, this is all you want.
        The historical samples, results, as well as likelihood function values can be extract from it.
        The uncertainty post processing results can be obtained using the function "likelihood_uncertainty"

        Usage:
        from pycup import save
        import pycup as cp
        import numpy as np

        def uni_fun1(X):
            # X for example np.array([1,2,3,...,30])
            fitness = np.sum(np.power(X,2)) + 1 # example: 1.2
            result = fitness.reshape(1,-1) # example ([1.2,])

            return fitness,result

        lb = -100 * np.ones(30)
        ub = 100 * np.ones(30)
        save.raw_path = r"D:\example01.txt"
        cp.SSA.run(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1)
        raw_res = save.RawDataSaver.load(r"D:\example01.txt")
        """
        self.historical_samples = hs
        self.historical_fitness = hf
        self.historical_results = hr
        self.GbestScore = bs
        self.GbestPosition = bp
        self.Curve = c
        self.pareto_samples = paretoPops
        self.pareto_fitness = paretoFits
        self.pareto_results = paretoRes
        self.opt_type = OptType


    def save(self,path=raw_path):
        if path is None:
            path = "RawResult.rst"
        with open(path, 'wb') as f:
            str = pickle.dumps(self,protocol=pickle.HIGHEST_PROTOCOL)
            f.write(str)

    @classmethod
    def help(cls):
        print(
        """
        Attributes & sub-attributes:
        .historical_samples: all the behaviour and non-behaviour samples
        .historical_fitness: the calculated fitness of behaviour and non-behaviour samples
        .historical_results: the simulated results of behaviour and non-behaviour samples
        .GbestScore: the global optimal fitness (only for swarm optimization algorithms)     
        .GbestPosition: the global optimal sample (only for swarm optimization algorithms)     
        .Curve: the optimization curve (only for swarm optimization algorithms)  
        .pareto_samples: the pareto optimal samples (only for multi-objective swarm optimization algorithms)  
        .pareto_fitness: the pareto optimal fitness (only for multi-objective swarm optimization algorithms)  
        .pareto_results: the corresponding simulation results of pareto optimal samples (only for multi-objective swarm optimization algorithms)          
        """
        )

    @classmethod
    def load(cls,path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj,RawDataSaver):
            raise TypeError("The input saver object should be pycup.save.RawDataSaver.")
        return obj


class ValidationResultSaver:

    def __init__(self,fitness,results,ppu_upper,ppu_lower,line_max,line_min,best_result,median_prediction):
        self.fitness = fitness
        self.results = results
        self.ppu_upper = ppu_upper
        self.ppu_lower = ppu_lower
        self.line_max = line_max
        self.line_min = line_min
        self.best_result = best_result
        self.median_prediction = median_prediction

    def save(self):
        if val_path is None:
            path = "ValidationResult.rst"
        else:
            path = val_path
        with open(path, 'wb') as f:
            str = pickle.dumps(self,protocol=pickle.HIGHEST_PROTOCOL)
            f.write(str)

    @classmethod
    def load(cls,path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj,ValidationResultSaver):
            raise TypeError("The input saver object should be pycup.save.ValidationResultSaver.")
        return obj


class PredResultSaver:

    def __init__(self,res,ppu_upper,ppu_lower,line_max,line_min,median_prediction):
        self.result = res
        self.ppu_upper = ppu_upper
        self.ppu_lower = ppu_lower
        self.line_max = line_max
        self.line_min = line_min
        self.median_prediction = median_prediction

    def save(self):

        if pred_path is None:
            path = "PredictionResult.rst"
        else:
            path = pred_path
        with open(path, 'wb') as f:
            str = pickle.dumps(self,protocol=pickle.HIGHEST_PROTOCOL)
            f.write(str)

    @classmethod
    def load(cls,path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj,PredResultSaver):
            raise TypeError("The input saver object should be pycup.save.PredResultSaver.")
        return obj


class RecordSaver:

    def save(self):

        if record_path is None:
            path = "Record.rcd"
        else:
            path = record_path
        with open(path, 'wb') as f:
            str = pickle.dumps(self,protocol=pickle.HIGHEST_PROTOCOL)
            f.write(str)
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj,RecordSaver):
            raise TypeError("The input saver object should be pycup.save.RecordSaver.")
        return obj


class SwarmRecord(RecordSaver):

    def __init__(self, pop, dim, lb, ub, hf, hs, hr, X, fitness, iteration, GbestPosition, GbestScore, Curve,
                 Pbest=None, fitnessPbest=None, V=None,
                 Alpha_pos=None, Alpha_score=None, Beta_pos=None, Beta_score=None, Delta_pos=None, Delta_score=None,
                 Xs=None, fitnessS=None, resS=None,
                 MS=None, CS=None, DS=None,X_new=None,
                 Pos=None
                 ):
        self.pop = pop
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.hf = hf
        self.hs = hs
        self.hr = hr
        self.X = X
        self.fitness = fitness
        self.iteration = iteration
        self.GbestPosition = GbestPosition
        self.GbestScore = GbestScore
        self.Curve = Curve

        self.Pbest = Pbest
        self.fitnessPbest = fitnessPbest
        self.V = V

        self.Alpha_pos = Alpha_pos
        self.Alpha_score = Alpha_score
        self.Beta_pos = Beta_pos
        self.Beta_score = Beta_score
        self.Delta_pos = Delta_pos
        self.Delta_score = Delta_score

        self.Xs = Xs
        self.fitnessS = fitnessS
        self.resS = resS

        self.MS = MS
        self.CS = CS
        self.DS = DS
        self.X_new = X_new
        self.Pos = Pos


class GLUERecord(RecordSaver):

    def __init__(self,n,dim,lb,ub,hf,hs,hr,iteration,mode):
        self.n = n
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.hf = hf
        self.hs = hs
        self.hr = hr
        self.iteration = iteration
        self.mode = mode


class MOswarmRecord(RecordSaver):

    def __init__(self,pop,dim,lb,ub,hf,hs,hr,X,iteration,n_obj,V=None,
                 archive=None,arFits=None,arRes=None,Pbest=None,fitnessPbest=None,fitnessGbest = None,GbestPositon=None,fitness=None,res=None):
        self.pop = pop
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.hf = hf
        self.hs = hs
        self.hr = hr
        self.X = X
        self.fitness = fitness
        self.res = res
        self.iteration = iteration
        self.n_obj = n_obj
        self.V = V
        self.archive = archive
        self.arFits = arFits
        self.arRes = arRes
        self.Pbest = Pbest
        self.fitnessPbest = fitnessPbest
        self.GbestPositon = GbestPositon
        self.fitnessGbest = fitnessGbest