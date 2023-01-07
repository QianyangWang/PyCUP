from . import save
import numpy as np
from . import multi_jobs
from . import uncertainty_analysis_fun as ufun
from . import progress_bar
from . import Reslib
UseTOPSIS = True


class EnsembleValidator:

    def __init__(self,opt_saver,obj_fun,n_obj,args=(), rstpath=None):
        """
        This is a class for users to carry out the ensemble validation process based on behavioral results obtained
        from uncertainty analysis. The argument opt_saver should be given a pycup.save.ProcResultSaver object (uncertainty
        analysis result). By calling the run or runMP method of this EnsembleValidator object, the program will execute the
        objective function for all the behavioral samples.
        :param opt_saver: a pycup.save.ProcResultSaver object
        :param ppu: percentage of prediction uncertainty, typically 95. If the value < 1, for example 0.95, it will be equivalent to 95.
        :param obj_fun: the objective function for validation.
        :param args: the arguments that are expected for the obj_fun
        :param rstpath: the path of the ValidationResultSaver objective, if None, a default path "ValidationResult.rst" will be used.
        """

        self.opt_saver = opt_saver
        self.obj_fun = obj_fun
        self.args = args
        self.rstpath = rstpath
        self.n_obj = n_obj


        if not isinstance(opt_saver,save.ProcResultSaver):
            raise TypeError("The opt_saver object should be pycup.save.ProcResultSaver.")

    def __CalculateFitness(self,X, fun,n_obj, args):
        pop = X.shape[0]
        fitness = np.zeros([pop, n_obj])
        res_l = []
        pb = progress_bar.ProgressBar(pop)
        pb.update(0)
        for i in range(pop):
            fitness[i], res = fun(X[i, :], *args)
            res_l.append(res)
            pb.update(i+1)
        if not Reslib.UseResObject:
            res_l = np.concatenate(res_l)
        else:
            res_l = np.array(res_l, dtype=object)
        return fitness, res_l

    def __CalculateFitnessMP(self,X, fun,n_obj, n_jobs, args):
        if n_obj == 1:
            fitness, res_l = multi_jobs.do_multi_jobs(func=fun, params=X,  n_process=n_jobs, args=args,pb=False)
        else:
            fitness, res_l = multi_jobs.do_multi_jobsMO(func=fun, params=X, n_process=n_jobs,n_obj=n_obj, args=args,pb=False)

        return fitness,res_l

    def run(self):
        print("Current Task: Ensemble Validation")
        X = self.opt_saver.behaviour_results.behaviour_samples
        weights = self.opt_saver.behaviour_results.normalized_weight
        fitness, res = self.__CalculateFitness(X,self.obj_fun,self.n_obj,self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Ensemble validation")
        if self.rstpath is not None:
            save.val_raw_path = self.rstpath
        saver = save.ValidationRawSaver(fitness,res,weights)
        saver.save()
        print("")
        print("Analysis Complete!")
        return saver

    def runMP(self,n_jobs):
        print("Current Task: Ensemble Validation MP")
        X = self.opt_saver.behaviour_results.behaviour_samples
        weights = self.opt_saver.behaviour_results.normalized_weight
        fitness, res = self.__CalculateFitnessMP(X,self.obj_fun,self.n_obj,n_jobs,self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Ensemble validation")
        if self.rstpath is not None:
            save.val_raw_path = self.rstpath
        saver = save.ValidationRawSaver(fitness,res,weights)
        saver.save()
        print("")
        print("Analysis Complete!")
        return saver
    

class EnsemblePredictor:

    def __init__(self,opt_saver,obj_fun,args=(),rstpath=None):
        """
        This is a class for users to carry out the ensemble prediction process based on behavioral results obtained
        from uncertainty analysis. The argument opt_saver should be given a pycup.save.ProcResultSaver object (uncertainty
        analysis result). By calling the run or runMP method of this EnsemblePredictor object, the program will execute the
        objective function for all the behavioral samples.
        :param opt_saver: a pycup.save.ProcResultSaver object
        :param ppu: percentage of prediction uncertainty, typically 95. If the value < 1, for example 0.95, it will be equivalent to 95.
        :param obj_fun: the objective function for prediction. In this function, you should only return the simulation result since that
                        it's a prediction process without observations.
        :param args: the arguments that are expected for the obj_fun
        :param rstpath: the path of the PredResultSaver objective, if None, a default path "PredictionResult.rst" will be used.
        """

        self.opt_saver = opt_saver
        self.obj_fun = obj_fun
        self.args = args
        self.rstpath = rstpath


        if not isinstance(opt_saver,save.ProcResultSaver):
            raise TypeError("The opt_saver object should be pycup.save.ProcResultSaver.")

    def __CalculateResult(self,X, fun, args):

        if len(X.shape) == 1:
            raise ValueError("There is only 1 behaviour result, please modify your threshold in uncertainty analysis.")
        pop = X.shape[0]
        res_l = []
        pb = progress_bar.ProgressBar(pop)
        pb.update(0)
        for i in range(pop):
            res = fun(X[i, :], *args)
            res_l.append(res)
            pb.update(i+1)
        if not Reslib.UseResObject:
            res_l = np.concatenate(res_l)
        else:
            res_l = np.array(res_l, dtype=object)
        return res_l

    def __CalculateResultMP(self,X, fun, n_jobs, args):

        if len(X.shape) == 1:
            raise ValueError("There is only 1 behaviour result, please modify your threshold in uncertainty analysis.")
        res_l = multi_jobs.predict_multi_jobs(func=fun, params=X,  n_process=n_jobs, args=args,pb=False)
        return res_l

    def run(self):
        print("Current Task: Ensemble Prediction")
        X = self.opt_saver.behaviour_results.behaviour_samples
        weights = self.opt_saver.behaviour_results.normalized_weight
        res = self.__CalculateResult(X,self.obj_fun,self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Ensemble prediction")
        if self.rstpath is not None:
            save.pred_path = self.rstpath
        saver = save.PredRawSaver(res,weights)
        saver.save()
        print("")
        print("Analysis Complete!")
        return res,saver

    def runMP(self,n_jobs):
        print("Current Task: Ensemble Prediction MP")
        X = self.opt_saver.behaviour_results.behaviour_samples
        weights = self.opt_saver.behaviour_results.normalized_weight
        res = self.__CalculateResultMP(X,self.obj_fun,n_jobs,self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Ensemble prediction")
        if self.rstpath is not None:
            save.pred_path = self.rstpath
        saver = save.PredRawSaver(res,weights)
        saver.save()
        print("")
        print("Analysis Complete!")
        return res,saver


class Validator:

    def __init__(self, opt_saver, obj_fun,n_obj, args=(), rstpath=None):
        """
        This is a class for users to carry out the validation process based on the best result obtained
        from calibration process. The argument opt_saver should be given a pycup.save.RawDataSaver object (calibration result).
        For calibration results of single-objective algorithms, by calling the run or runMP method of this Validator object,
        the program will execute the objective function for the best solution. For multi-objective algorithms, the program will
        execute the objective function for all the pareto non-dominated solutions, since that typically you cannot judge
        which one in the pareto non-dominated solutions is the best.
        :param opt_saver: a pycup.save.RawDataSaver object
        :param obj_fun: the objective function for validation.
        :param args: the arguments that are expected for the obj_fun
        :param rstpath: the path of the ValidationResultSaver objective, if None, a default path "ValidationResult.rst" will be used.
        """
        self.opt_saver = opt_saver
        self.obj_fun = obj_fun
        self.args = args
        self.rstpath = rstpath
        self.n_obj = n_obj
        if not isinstance(opt_saver, save.RawDataSaver):
            raise TypeError("The opt_saver object should be pycup.save.RawDataSaver.")

    def __CalculateFitness(self, X, fun,n_obj, args):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        pop = X.shape[0]
        fitness = np.zeros([pop, n_obj])
        res_l = []
        pb = progress_bar.ProgressBar(pop)
        pb.update(0)
        for i in range(pop):
            fitness[i], res = fun(X[i,:], *args)
            res_l.append(res)
            pb.update(i+1)
        if not Reslib.UseResObject:
            res_l = np.concatenate(res_l)
        else:
            res_l = np.array(res_l, dtype=object)
        return fitness, res_l

    def __CalculateFitnessMP(self, X, fun,n_obj, n_jobs, args):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        if n_obj == 1:
            fitness, res_l = multi_jobs.do_multi_jobs(func=fun, params=X, n_process=n_jobs, args=args,pb=False)
        else:
            fitness, res_l = multi_jobs.do_multi_jobsMO(func=fun, params=X, n_process=n_jobs,n_obj=n_obj, args=args,pb=False)

        return fitness, res_l

    def run(self):
        print("Current Task: Validation")
        if self.opt_saver.GbestPosition is not None and UseTOPSIS:
            X = self.opt_saver.GbestPosition
        else:
            X = self.opt_saver.pareto_samples
        fitness, res = self.__CalculateFitness(X, self.obj_fun,self.n_obj, self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Validation")
        if self.rstpath is not None:
            save.val_path = self.rstpath
        saver = save.ValidationRawSaver(fitness=fitness,results=res)
        saver.save()
        print("")
        print("Analysis Complete!")
        return saver

    def runMP(self, n_jobs):
        print("Current Task: Validation MP")
        if self.opt_saver.GbestPosition is not None and UseTOPSIS:
            X = self.opt_saver.GbestPosition
        else:
            X = self.opt_saver.pareto_samples
        fitness, res = self.__CalculateFitnessMP(X, self.obj_fun,self.n_obj, n_jobs, self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Validation")
        if self.rstpath is not None:
            save.val_path = self.rstpath
        saver = save.ValidationRawSaver(fitness=fitness,results=res)
        saver.save()
        print("")
        print("Analysis Complete!")
        return saver


class Predictor:

    def __init__(self, opt_saver, obj_fun, args=(), rstpath=None):
        """
        This is a class for users to carry out the prediction process based on the best result obtained
        from calibration process. The argument opt_saver should be given a pycup.save.RawDataSaver object (calibration result).
        For calibration results of single-objective algorithms, by calling the run or runMP method of this Validator object,
        the program will execute the objective function for the best solution. For multi-objective algorithms, the program will
        execute the objective function for all the pareto non-dominated solutions, since that typically you cannot judge
        which one in the pareto non-dominated solutions is the best.
        :param opt_saver: a pycup.save.RawDataSaver object
        :param obj_fun: the objective function for validation.
        :param args: the arguments that are expected for the obj_fun
        :param rstpath: the path of the PredResultSaver objective, if None, a default path "PredictionResult.rst" will be used.
        """
        self.opt_saver = opt_saver
        self.obj_fun = obj_fun
        self.args = args
        self.rstpath = rstpath
        if not isinstance(opt_saver, save.RawDataSaver):
            raise TypeError("The opt_saver object should be pycup.save.RawDataSaver.")

    def __CalculateResult(self,X, fun, args):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        pop = X.shape[0]
        pb = progress_bar.ProgressBar(pop)
        pb.update(0)
        res_l = []
        for i in range(pop):
            res = fun(X[i, :], *args)
            res_l.append(res)
            pb.update(i+1)
        if not Reslib.UseResObject:
            res_l = np.concatenate(res_l)
        else:
            res_l = np.array(res_l, dtype=object)
        return res_l

    def __CalculateResultMP(self,X, fun, n_jobs, args):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        res_l = multi_jobs.predict_multi_jobs(func=fun, params=X,  n_process=n_jobs, args=args,pb=False)
        return res_l

    def run(self):
        print("Current Task: Prediction")
        if self.opt_saver.GbestPosition is not None and UseTOPSIS:
            X = self.opt_saver.GbestPosition
        else:
            X = self.opt_saver.pareto_samples
        res = self.__CalculateResult(X, self.obj_fun, self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Prediction")
        if self.rstpath is not None:
            save.pred_path = self.rstpath
        saver = save.PredRawSaver(res)
        saver.save()
        print("")
        print("Analysis Complete!")
        return saver

    def runMP(self, n_jobs):
        print("Current Task: Prediction MP")
        if self.opt_saver.GbestPosition is not None and UseTOPSIS:
            X = self.opt_saver.GbestPosition
        else:
            X = self.opt_saver.pareto_samples
        res = self.__CalculateResultMP(X, self.obj_fun, n_jobs, self.args)
        if Reslib.UseResObject:
            res = Reslib.ResultDataPackage(l_result=res,method_info="Prediction")
        if self.rstpath is not None:
            save.pred_path = self.rstpath
        saver = save.PredRawSaver(res)
        saver.save()
        print("")
        print("Analysis Complete!")
        return saver