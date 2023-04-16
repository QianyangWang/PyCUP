import warnings
import numpy as np
from . import save
from .NSGA2 import nonDominationSort
import os
from . import Reslib
import glob
import math
import os
import re
from .PESTclasses import PESTparams,PESTobs,PESTbatch,PESTio,PESTsimpair,PESTsimgp,WeightedAverage,WeightType,WeightedAndDropOut,DorpOut
from .PESTclasses import PESTinsReader,PESTresultReader
import warnings
import prettytable as pt
import subprocess
import shutil
import glob
import multiprocessing as mp



class SpotpyDbConverter:
    """
    A soptpy database converter to convert the spotpy database to pycup.save.RawDataSaver
    object or to write a spotpy database using a pycup.save.RawDataSaver object.

    The optimization direction is "Minimize" is default. When the "Maximize" is given,
    the fitness/like values in the soptpy database/pycup saver object will be subtracted
    by 1 (1 - fitness/like). The "grid" optimization direction is currently not supported.

    Usage:

    from pycup.integrate import SpotpyDbConverter
    s = SpotpyDbConverter()

    # pycup.save.RawDataSaver -> spotpy hdf5 database
    s.RawSaver2hdf5("RawResult.rst",r"pycup_result.h5")

    # pycup.save.RawDataSaver -> spotpy csv database
    s.RawSaver2csv("RawResult.rst",r"pycup_result.csv")

    # spotpy csv database -> pycup.save.RawDataSaver
    s.csv2RawSaver("NSGA2.csv","RawResult.rst")

    # spotpy hdf5 database -> pycup.save.RawDataSaver
    s.csv2RawSaver("NSGA2.h5","RawResult.rst")
    """

    def __init__(self,opt_direction="Minimize"):
        if opt_direction != "Minimize" and opt_direction != "Maximize":
            raise ValueError("The optimization direction can only be 'Minimize' or 'Maximize', the 'Grid' option in Spotpy is currently not supported")
        if opt_direction == "Maximize":
            print("CONVERTER NOTE: The minimization process is used in PyCUP, therefore, the obj func value for a maximization process will be subtracted by one when converted to/from Spotpy.")
        self.opt_direction = opt_direction

    def _analyze_fmt(self,path):
        fmt = os.path.splitext(path)[-1]
        return fmt

    def _analyze_chain(self,chain):

        iter_idx0 = np.argwhere(chain==0)
        iter_idx1 = np.argwhere(chain==1)
        iter_idx0 = self._check_continuity(iter_idx0)
        if len(iter_idx1) != 0:
            # heuristic algorithms
            if len(iter_idx0) == len(iter_idx1):
                # algorithms such as nsga-ii
                iterations = len(iter_idx0)
                iter_idx = np.append(iter_idx0,len(chain))
            else:
                # sce-ua
                iterations = len(iter_idx1)+1
                iter_idx = np.insert(iter_idx1,0,0)
                iter_idx = np.append(iter_idx, len(chain))
        else:
            # random sampling methods
            iterations = 1
            iter_idx=[]

        return iter_idx,iterations

    def _check_continuity(self,iter_idx):
        redun = []
        for loc,i in enumerate(iter_idx[1:]):
            if i == iter_idx[loc] + 1:
                redun.append(loc+1)
        iter_idx = np.delete(iter_idx,redun)
        return iter_idx

    def _post(self,headers, data, target_path):
        like_idx = []
        par_idx = []
        sim_idx = []
        idx_chain = -1
        for i, colname in enumerate(headers):
            if colname.startswith("like"):
                like_idx.append(i)
            elif colname.startswith("par"):
                par_idx.append(i)
            elif colname.startswith("simulation"):
                sim_idx.append(i)
            if colname == "chain":
                idx_chain = i
        chain = data[:, idx_chain]
        iter_idx, iter_num = self._analyze_chain(chain)
        if iter_num == 1:
            hs = data[:, par_idx[0]:par_idx[-1] + 1]
            hf = data[:, like_idx[0]:like_idx[-1] + 1]
            if self.opt_direction != "Minimize":
                hf = 1 - hf
            hr = data[:, sim_idx[0]:sim_idx[-1] + 1]
        else:
            hs = [data[iter_idx[i]:iter_idx[i + 1], par_idx[0]:par_idx[-1] + 1] for i in range(iter_num)]
            hf = [data[iter_idx[i]:iter_idx[i + 1], like_idx[0]:like_idx[-1] + 1] for i in range(iter_num)]
            if self.opt_direction != "Minimize":
                hf = [1 - f for f in hf]
            hr = [data[iter_idx[i]:iter_idx[i + 1], sim_idx[0]:sim_idx[-1] + 1] for i in range(iter_num)]
        if len(like_idx) == 1:
            # single objective calibration in spotpy
            GBestScore, best_id = np.min(data[:, like_idx]), np.argmin(data[:, like_idx])
            GBestPosition = data[:, par_idx[0]:par_idx[-1] + 1][best_id]
            if iter_num > 1:
                Curve = [np.min(f) for f in hf]
                for loc in range(len(Curve) - 1):
                    if Curve[loc + 1] > Curve[loc]:
                        Curve[loc + 1] = Curve[loc]
                OptType = "SWARM"
            else:
                Curve = None
                # All the random sampling methods use the marker GLUE here
                OptType = "GLUE"
            saver = save.RawDataSaver(hs=hs, hf=hf, hr=hr, bs=GBestScore, bp=GBestPosition, c=Curve, OptType=OptType)
        else:
            # multi-objective calibration in spotpy
            if iter_num > 1:
                OptType = "MO-SWARM"

                ranks = nonDominationSort(hs[-1], hf[-1])
                paretoPops = hs[-1][ranks == 0]
                paretoFits = hf[-1][ranks == 0]
                paretoRes = hr[-1][ranks == 0]
                saver = save.RawDataSaver(hs=hs, hf=hf, hr=hr, paretoPops=paretoPops, paretoFits=paretoFits,
                                          paretoRes=paretoRes, OptType=OptType)
            else:
                OptType = "GLUE"
                saver = save.RawDataSaver(hs=hs, hf=hf, hr=hr, OptType=OptType)
        if target_path is not None and isinstance(target_path, str):
            saver.save(target_path)
        return saver

    def csv2RawSaver(self,path,target_path=None):
        if self._analyze_fmt(path) == ".csv":
            with open(path) as f:
                db_data = f.readline()
                headers = db_data.split(",")
                data = np.genfromtxt(path,delimiter="," ,skip_header = 1)
            saver = self._post(headers,data,target_path)
            return saver
        else:
            raise TypeError("The given spotpy database is not in a csv format.")

    def RawSaver2csv(self,raw_saver,target_path,station=None,event=None,param_names=None):
        if self._analyze_fmt(target_path) != ".csv":
            raise ValueError("The given target path should be the csv format.")
        if isinstance(raw_saver,str):
            saver = save.RawDataSaver.load(raw_saver)
            self._historical_results = saver.historical_results
        elif isinstance(raw_saver,save.RawDataSaver):
            saver = raw_saver
            self._historical_results = saver.historical_results
        else:
            raise TypeError("The given raw_saver should be the pycup.save.RawDataSaver object or the path of it.")
        if isinstance(saver.historical_results,Reslib.ResultDataPackage):
            if station is not None and event is not None:
                self._historical_results = saver.historical_results[station][event]
            else:
                raise KeyError("When the pycup.ResLib for multi-station/multi-event result is used, the station name and the event name should be given.")
        if saver.opt_type == "SWARM" or saver.opt_type == "MO-SWARM":
            hr = np.concatenate(self._historical_results)
            hf = np.concatenate(saver.historical_fitness)
            hs = np.concatenate(saver.historical_samples)
            pops = [i.shape[0] for i in self._historical_results]
            chains = np.concatenate([np.arange(j) for j in pops]).reshape(-1,1)

        else:
            hr = self._historical_results
            hf = saver.historical_fitness
            hs = saver.historical_samples
            chains = np.zeros(len(hr)).reshape(-1,1)

        data = np.concatenate([hf,hs,hr,chains],axis=1)
        header = []
        for i in range(hf.shape[-1]):
            header.append("like{}".format(i+1))
        if param_names is not None:
            if not isinstance(param_names,list):
                raise TypeError("The argument param_names should be a list")
            if len(param_names) != hs.shape[-1]:
                raise ValueError("The length of the param_names is not equal to the dimension of the parameter space.")
            for j in range(hs.shape[-1]):
                header.append("par{}".format(param_names[j]))
        else:
            for j in range(hs.shape[-1]):
                header.append("par{}".format(j))
        for k in range(hr.shape[-1]):
            header.append("simulation_{}".format(k))
        header.append("chain")
        header = ",".join(header)
        np.savetxt(target_path,data, delimiter=",", header=header,comments="")

    def hdf52RawSaver(self,path,target_path=None):
        if self._analyze_fmt(path) == ".h5" or self._analyze_fmt(path) == ".hdf5":
            import h5py
            with h5py.File(path, "r") as f:
                keys = f.keys()
                for i in keys:
                    db_name = i
                dataset = f[db_name]
                headers = list(dataset.dtype.fields.keys())
                likes = []
                pars = []
                sims = None
                chain = None
                for i in headers:
                    if i.startswith("like"):
                        likes.append(np.array(dataset[i]).reshape(-1, 1))
                    elif i.startswith("par"):
                        pars.append(np.array(dataset[i]).reshape(-1, 1))
                    elif i.startswith("sim"):
                        sims = np.array(dataset[i])
                    elif i.startswith("chain"):
                        chain = np.array(dataset[i]).reshape(-1, 1)
                headers.remove("simulation")
            for i in range(sims.shape[-1]):
                headers.insert(-1,"simulation_{}".format(i))
            likes = np.concatenate(likes, axis=1)
            pars = np.concatenate(pars, axis=1)

            data = np.concatenate([likes, pars, sims, chain], axis=1)
            saver = self._post(headers,data,target_path)
            return saver
        else:
            raise TypeError("The given spotpy database should be in a .h5 or .hdf5 format.")

    def RawSaver2hdf5(self,raw_saver,target_path,station=None,event=None,param_names=None):

        if self._analyze_fmt(target_path) != ".h5" and self._analyze_fmt(target_path) != ".hdf5":
            raise ValueError("The given target path should be the .h5 or .hdf5 format.")
        if isinstance(raw_saver,str):
            saver = save.RawDataSaver.load(raw_saver)
            self._historical_results = saver.historical_results
        elif isinstance(raw_saver,save.RawDataSaver):
            saver = raw_saver
            self._historical_results = saver.historical_results
        else:
            raise TypeError("The given raw_saver should be the pycup.save.RawDataSaver object or the path of it.")
        if isinstance(saver.historical_results,Reslib.ResultDataPackage):
            if station is not None and event is not None:
                self._historical_results = saver.historical_results[station][event]
            else:
                raise KeyError("When the pycup.ResLib for multi-station/multi-event result is used, the station name and the event name should be given.")
        if saver.opt_type == "SWARM" or saver.opt_type == "MO-SWARM":
            hr = np.concatenate(self._historical_results)
            hf = np.concatenate(saver.historical_fitness)
            hs = np.concatenate(saver.historical_samples)
            pops = [i.shape[0] for i in self._historical_results]
            chains = np.concatenate([np.arange(j) for j in pops]).reshape(-1,1)
        else:
            hr = self._historical_results
            hf = saver.historical_fitness
            hs = saver.historical_samples
            chains = np.zeros(len(hr)).reshape(-1,1)

        header = []
        for i in range(hf.shape[-1]):
            header.append("like{}".format(i+1))
        if param_names is not None:
            if not isinstance(param_names,list):
                raise TypeError("The argument param_names should be a list")
            if len(param_names) != hs.shape[-1]:
                raise ValueError("The length of the param_names is not equal to the dimension of the parameter space.")
            for j in range(hs.shape[-1]):
                header.append("par{}".format(param_names[j]))
        else:
            for j in range(hs.shape[-1]):
                header.append("par{}".format(j))
        header.append("simulation")
        header.append("chains")
        self._write2h5(header,hf,hs,hr,chains,target_path)

    def _write2h5(self,header,hf,hs,hr,chains,target_path):

        import tables
        db = tables.open_file(target_path, "w")
        filters = tables.Filters(complevel=9)
        database_name = os.path.basename(target_path).split(".")[0]
        db.create_table("/", database_name,description=self._get_table_def(header,hf,hs,hr),filters=filters)
        table = db.root[database_name]
        sheet_data = []
        for i in range(hf.shape[0]):
            row = []
            row.extend(hf[i].flatten().tolist())
            row.extend(hs[i].flatten().tolist())
            row.append(hr[i].flatten())
            row.append(int(chains[i]))
            sheet_data.append(row)
        for i in sheet_data:
            new_row = table.row
            for h,d in zip(header,i):
                new_row[h] = d
            new_row.append()
        db.close()

    def _get_table_def(self,header,hf,hs,hr):
        import tables
        like_pos = 0
        param_pos = hf.shape[-1]
        sim_pos = param_pos + hs.shape[-1]
        chain_pos = sim_pos
        dtype = np.dtype(np.float32)
        columns = {
            header[i]: tables.Col.from_dtype(dtype, pos=i)
            for i in range(like_pos, sim_pos)
        }

        sim_shape = hr.shape[-1]
        sim_dtype = np.dtype((np.float32, sim_shape))
        columns["simulation"] = tables.Col.from_dtype(sim_dtype, pos=sim_pos)
        chain_pos += 1
        # Add a column chains
        columns["chains"] = tables.UInt16Col(pos=chain_pos)
        return columns


class SpotpySetupConverter:
    """
    A spotpy model setup object converter, the convert method will return an objective function
    for using in pycup optimization process

    Usage:

    import spotpy
    import pycup
    from pycup.integrate import SpotpySetupConverter
    from spotpy.examples.spot_setup_hymod_python import spot_setup # example spotpy setup object
    spot_setup=spot_setup(spotpy.objectivefunctions.rmse)
    s = SpotpySetupConverter()
    s.convert(spot_setup)
    pycup.MFO.run(20,s.dim,s.lb,s.ub,10,s.obj_fun,args=())
    """

    def __init__(self):
        self.lb = None
        self.ub = None
        self.obj_fun = None
        self.dim = None

    def convert(self,spot_setup):
        import spotpy
        self.params = spotpy.parameter.get_parameters_from_setup(spot_setup)
        self._check_param_class()
        self.lb = self._gen_bound("lb")
        self.ub = self._gen_bound("ub")
        self._check_bnd()
        self.dim = len(self.lb)
        self.obj_fun = self._gen_pycup_objfun(spot_setup)

    def _check_param_class(self):
        for p in self.params:
            import spotpy
            if not isinstance(p,spotpy.parameter.Uniform):
                warnings.warn("The distribution of the parameter is not Uniform in the spotpy setup class, while a uniform distribution is assumed in PyCUP. Please check the correctness of the generated boundaries.")

    def _gen_bound(self,type):
        if type == "lb":
            bnd = np.array([p.minbound for p in self.params])
        else:
            bnd = np.array([p.maxbound for p in self.params])
        return bnd

    def _gen_pycup_objfun(self,spot_setup):

        def obj_fun(x,*args):
            sim = spot_setup.simulation(x)
            eva = spot_setup.evaluation()
            if args:
                if len(args)>1:
                    fitness = spot_setup.objectivefunction(sim,eva,params=args)
                else:
                    fitness = spot_setup.objectivefunction(sim, eva, params=args[0])
            else:
                fitness = spot_setup.objectivefunction(sim, eva)
            return fitness,np.array(sim).reshape(1,-1)
        return obj_fun

    def _check_bnd(self):
        if np.sum(self.lb > self.ub) > 0:
            raise ValueError("The boundaries in the setup class is not correctly set.")



class PESTconvertor:

    """
    *Introduction:
    A PEST++ integration class for carrying out a PyCUP calibration based on the PEST++ IO framework.
    By using the .convert() method of this class object, users can obtain the well written calibration
    programming workflow for PyCUP algorithms. It can make the calibration process more model-agnostic.
    This class currently support PEST++ suite with tsproc.exe for a simulation task which is associated
    with time series. Other main features include:
    -- it can recognize the control file and command lines automatically
    -- it supports the generation of single-processing function and two kinds of multi-processing functions
    -- it supports the parameter relationships and observation weights in PEST++
    -- The objective function (metric) in PEST++ is adopted as default, while users can specify the customized evaluation metric
    -- it supports multi-objective function generation

    *Attrs:
    lb: the generated parameter lower boundary
    ub: the generated parameter upper boundary
    pst_path: the pst file path
    workspace: the working directory
    objfun: the generated objective function object which can be used in pycup algorithms
    io_pairs: PEST parameter/result input/output pairs
    obs: PEST observation object
    tar_pars: the recognized calibration target parameters (type = none or log)
    oth_pars: the recognized parameters which will not be computed by the algorithm (type = fixed or tied)
    params: parameter information

    *Main Methods:
    convert: convert a PEST++ project to functions which are needed by PyCUP calibration/validation/prediction
    show_task_info: (use it after the conversion) print a table of PEST++ project descrption
    show_parameter_info: (use it after the conversion) print a table of parameter settings

    *Other methods (will be called automatically by .convert method):
    CallExe1: the cmd calling function designed for multiprocessing
    CallExe2: the cmd calling function designed for single-processing
    gen_call: method for generating the user specified cmd calling function
    gen_objfun: method for generating the user specified objective function
    gen_ResultIO: method for generating the result reading function
    gen_ParamIO: method for generating the parameter reading/updating function

    *Usage:
    import pycup
    from pycup.integrate import PESTconvertor
    con = PESTconvertor(r"D:\PESTXAJ\XAJ")
    con.convert(mpi=True,mpimode="dynamic",evaluation_metric=pycup.evaluation_metrics.OneMinusNSE,n_jobs=5,hide=True)
    con.show_task_info()
    con.show_parameter_info()
    pycup.GLUE.runMP(100,con.dim,con.lb,con.ub,con.objfun,n_jobs=5)
    """

    def __init__(self, pest_dir,bat_fname=None):
        Reslib.UseResObject = True
        if not os.path.isabs(pest_dir):
            pest_dir = os.getcwd() + "\\" + pest_dir
        self.workspace = pest_dir
        self.pst_path = None
        self.tsproc_path = None
        if bat_fname:
            if os.path.isabs(bat_fname):
                self.bat_fname = bat_fname
            else:
                self.bat_fname = self.workspace + "\\" + bat_fname
        else:
            self.bat_fname = None
        self._find_pst()
        if not self.pst_path:
            if self.bat_fname:
                print("The PEST control file (.pst) not found, trying to locate it through tsproc.dat provided in batch file.")
                self._read_bat_tsproc()
                if self.tsproc_path:
                    with open(self.tsproc_path,"r") as f:
                        pst_fname = None
                        lines = f.readlines()
                        for row in lines:
                            if "NEW_PEST_CONTROL_FILE" in row:
                                pst_fname = list(filter(None,row.split(" ")))[-1].strip("\n")
                        pst_path = glob.glob(self.workspace + "\\{}".format(pst_fname))
                    if len(pst_path) == 1:
                        self.pst_path = pst_path[0]
                        print("The PEST control file has been successfully located.")
                    else:
                        print("The PEST control file mentioned in tsproc.dat is not found, trying to generate it using the tsproc.exe command.")
                        cwd = os.getcwd()
                        if pst_fname:
                            fmt = "." + pst_fname.split(".")[-1]
                            command = self.workspace + "\\tsproc.exe" + " " + self.tsproc_path + " " + self.record_path
                            os.chdir(self.workspace)
                            os.system(command)
                            self._find_pst(fmt)
                            subprocess.call(["taskkill", "/F", "/IM", self.workspace + "\\tsproc.exe"],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                            os.chdir(cwd)
                        else:

                            subprocess.call(["taskkill", "/F", "/IM", self.workspace + "\\tsproc.exe"],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                            os.chdir(cwd)
                            raise FileNotFoundError("The PEST control file should be specified in the tsproc.dat")
                        if not self.pst_path:
                            raise FileNotFoundError("The .pst file is not successfully generated, please check the settings in .bat file.")
                        else:
                            print("The .pst file has been generated successfully.")
                else:
                    raise ValueError("There is no relevant tsproc.exe command for generating a .pst file.")
            else:
                raise FileNotFoundError("Conversion Failed: the PEST control file (.pst) could not be found, the batch file was also absent.")

    def convert(self,evaluation_metric=None,hide=True,callfun="subprocess",return_metric=True,mpi=False,n_jobs=None,mpimode="dynamic"):
        """
        The PEST++ conversion method.

        :param evaluation_metric: The evaluation metric computing function. If None,
                                  the summation of the square of weighted residuals sum((w*r)**2),
                                  which is default in PEST++, will be used. See also the PEST++ documentation.
        :param hide: Hide the commandline output, True as default.
        :param callfun: the cmd calling function, currently support subprocess.Popen
        :param return_metric: True as default, if False, the return value of the generated objective function will not contain the evaluation metric.
        :param mpi: Multi-Processing. False as default. If True, the multi-processing version of the objective function will be generated.
        :param mpimode: only valid when mpi == True. "dynamic" or "fixed" are accepted.
                        If "dynamic":
                        The program will create unfixed number of sub-folders and name them according to the pid.
                        After each simulation, the folder will be dynamically deleted. This mode allow users
                        to use a lot of processes during the calibration task.
                        If "fixed":
                        The program will create the fixed number(n_jobs) of sub-folders.
                        During the calibration process, the new simulation result will be overwritten in the
                        current exsisting folder. This mode can avoid the situation that the folder are frequently created
                        and deleted, which is faced with the "dynamic" mode. However, this mode only supports no more than
                        10 processes.

        :param n_jobs: number of processes, this is only valid when the mpimode is "fixed"
        :return: None
        """
        self.n_jobs=n_jobs
        self.mode = mpimode
        self.mpi = mpi
        if mpi and self.mode not in ["fixed","dynamic"]:
            raise ValueError("The multiprocessing function is used, the mpimode should only be 'fixed' or 'dynamic'.")
        if mpimode == "fixed":
            if not n_jobs:
                raise ValueError("When the mpimode is 'fixed', the n_jobs should be given.")
        self.evaluation_metric = evaluation_metric
        self.cmd_hide=hide
        self._analyze_pst()
        # check the weight series of the observation
        if evaluation_metric:
            weight_types = []
            for gp in self.obs:
                weight_types.append(gp.check_weights())
                weight_type_flag1 = sum([issubclass(i, DorpOut) for i in weight_types])
                weight_type_flag2 = sum([issubclass(i, WeightedAverage) for i in weight_types])
                if weight_type_flag1 != len(weight_types) and weight_type_flag2 != len(weight_types):
                    raise RuntimeError("The types of the observation weight series should be uniform.")
        self._gen_boundary()
        self.gen_paramIO()
        self.gen_resultIO()
        self.gen_call(callfun=callfun)
        self.gen_objfun(evaluation_metric,return_metric,mpi)


    def _call_sim(self,X):
        self.UpdateParam(X)
        self.CallExe()
        # a list of observation group results
        lgpres = self.ReadResult()
        return lgpres

    def _call_simMP(self,X):
        pid = os.getpid()
        dirs = os.listdir(self.workspace)
        dst_folder = self.workspace+"\\pycup_process{}".format(pid)
        os.mkdir(dst_folder)
        for f in dirs:
            if os.path.basename(f) != "pestpp.exe"  and os.path.basename(f) != "tsproc.exe" and "pycup_process" not in f:
                shutil.copy(self.workspace +"\\"+ f,dst_folder+"\\"+os.path.basename(f))

        os.chdir(dst_folder)
        self.UpdateParam(X,subfolder="pycup_process{}".format(pid))
        self.CallExe(subfolder="pycup_process{}".format(pid))
        # a list of observation group results
        lgpres = self.ReadResult(subfolder="pycup_process{}".format(pid))
        return pid,lgpres

    def _call_simMPF(self,X,n_jobs):
        # using multiprocessing to allocate the folder id
        pid = mp.current_process().name.split("-")[-1]
        dirs = os.listdir(self.workspace)
        pid = int(pid) % n_jobs + 1
        dst_folder = self.workspace+"\\pycup_process{}".format(pid)
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)
            for f in dirs:
                if os.path.basename(f) != "pestpp.exe"  and os.path.basename(f) != "tsproc.exe" and "pycup_process" not in f:
                    shutil.copy(self.workspace +"\\"+ f,dst_folder+"\\"+os.path.basename(f))
        os.chdir(dst_folder)
        self.UpdateParam(X,subfolder="pycup_process{}".format(pid))
        self.CallExe(subfolder="pycup_process{}".format(pid))
        # a list of observation group results
        lgpres = self.ReadResult(subfolder="pycup_process{}".format(pid))
        #print(len(lgpres[0].idx),len(lgpres[0].res),len(self.obs["i_mod_est"].index))
        return pid,lgpres


    def _rm_dirMP(self,pid):
        os.chdir(self.workspace)
        try:
            # remove the folder.
            shutil.rmtree(self.workspace + "\\" + "pycup_process{}".format(pid))
        except:
            # sometimes the exe program has not been finished completely, close the exe and then remove the folder.
            files = glob.glob(self.workspace + "\\" + "pycup_process{}\\*.exe".format(pid))
            for f in files:
                p = subprocess.call(["taskkill","/F","/IM",f],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            shutil.rmtree(self.workspace + "\\" + "pycup_process{}".format(pid))

    def _closeExeMPF(self,pid):
        files = glob.glob(self.workspace + "\\" + "pycup_process{}\\*.exe".format(pid))
        for f in files:
            p = subprocess.call(["taskkill", "/F", "/IM", f], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    def gen_objfun(self,evaluation_metric=None,return_metric=True,mpi=False):
        if mpi:
            if return_metric:
                if not evaluation_metric:
                    objfun = self._objfunPESTMP
                else:
                    # multi-objective optimization
                    if isinstance(evaluation_metric, list) or isinstance(evaluation_metric, tuple) or isinstance(
                            evaluation_metric, np.ndarray):
                        objfun = self._objfunMOMP
                    else:
                        objfun = self._objfunSingleMP
            else:
                objfun = self._objfunPredMP
        else:
            if return_metric:
                if not evaluation_metric:
                    objfun = self._objfunPEST
                else:
                    # multi-objective optimization
                    if isinstance(evaluation_metric, list) or isinstance(evaluation_metric, tuple) or isinstance(
                            evaluation_metric, np.ndarray):
                        objfun = self._objfunMO
                    else:
                        objfun = self._objfunSingle
            else:
                objfun = self._objfunPred
        self.objfun=objfun
        return objfun

    # The objective function in PEST++
    def _objfunPEST(self,X):
        ori_cwd = os.getcwd()
        lgpres = self._call_sim(X)
        weighted_errs = []
        # Uss the residual (same as PEST++) as the objective function (err = sum((w*r)**2))
        for gp in lgpres:
            for o, r, w in zip(gp.obs, gp.res, gp.wei):
                werr = np.square(w * (r - o))
                weighted_errs.append(werr)
        metric = np.sum(weighted_errs)
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))
        os.chdir(ori_cwd)
        return metric, res_obj

    # multi-objective optimization
    def _objfunMO(self,X):
        ori_cwd = os.getcwd()
        lgpres = self._call_sim(X)
        weight_types = []
        for gp in lgpres:
            weight_types.append(gp.check_weights())
        flag = sum([issubclass(i, DorpOut) for i in weight_types])
        # Only consider the observation dropout, a mean evaluation metric without weight will be calculated
        if flag == len(weight_types):
            metrics = []
            for e_fun in self.evaluation_metric:
                gp_metric = []
                for gp in lgpres:
                    obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                    obs_for_eva = obs[np.argwhere(wei != 0)]
                    res_for_eva = res[np.argwhere(wei != 0)]
                    gp_metric.append(e_fun(obs_for_eva, res_for_eva))
                metrics.append(np.average(gp_metric))
        # Consider both the dropout and the weight, a weighted average evaluation metric will be calculated
        else:
            metrics = []
            for e_fun in self.evaluation_metric:
                gp_metric = []
                for gp in lgpres:
                    obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                    if len(set(gp.wei)) > 1:
                        # Weighted average with dropout
                        obs_for_eva = obs[np.argwhere(wei != 0)]
                        res_for_eva = res[np.argwhere(wei != 0)]
                        gp_metric.append(np.max(wei) * e_fun(obs_for_eva, res_for_eva))
                    else:
                        # Weighted average without dropout
                        gp_metric.append(np.max(wei) * e_fun(obs, res))
                metrics.append(np.sum(gp_metric))
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))
        os.chdir(ori_cwd)
        if len(metrics) > 1:
            return metrics, res_obj
        else:
            return metrics[0], res_obj

    # single-objective optimization
    def _objfunSingle(self,X):
        ori_cwd = os.getcwd()
        lgpres = self._call_sim(X)
        weight_types = []
        for gp in lgpres:
            weight_types.append(gp.check_weights())
        flag = sum([issubclass(i, DorpOut) for i in weight_types])
        # Only consider the observation dropout, a mean evaluation metric without weight will be calculated
        if flag == len(weight_types):
            gp_metric = []
            for gp in lgpres:
                obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                obs_for_eva = obs[np.argwhere(wei != 0)]
                res_for_eva = res[np.argwhere(wei != 0)]
                gp_metric.append(self.evaluation_metric(obs_for_eva, res_for_eva))
            metric = np.average(gp_metric)
        # Consider both the dropout and the weight, a weighted average evaluation metric will be calculated
        else:
            gp_metric = []
            for gp in lgpres:
                obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                if len(set(gp.wei)) > 1:
                    # Weighted average with dropout
                    obs_for_eva = obs[np.argwhere(wei != 0)]
                    res_for_eva = res[np.argwhere(wei != 0)]
                    gp_metric.append(np.max(wei) * self.evaluation_metric(obs_for_eva, res_for_eva))
                else:
                    # Weighted average without dropout
                    gp_metric.append(np.max(wei) * self.evaluation_metric(obs, res))
            metric = np.sum(gp_metric)
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))
        os.chdir(ori_cwd)
        return metric, res_obj

    def _objfunPred(self,X):
        ori_cwd = os.getcwd()
        lgpres = self._call_sim(X)
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))
        os.chdir(ori_cwd)
        return res_obj

    def _objfunSingleMP(self,X):
        if self.mode == "dynamic":
            pid,lgpres = self._call_simMP(X)
        else:
            pid, lgpres = self._call_simMPF(X,self.n_jobs)
        weight_types = []
        for gp in lgpres:
            weight_types.append(gp.check_weights())
        flag = sum([issubclass(i, DorpOut) for i in weight_types])
        # Only consider the observation dropout, a mean evaluation metric without weight will be calculated
        if flag == len(weight_types):
            gp_metric = []
            for gp in lgpres:
                obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                obs_for_eva = obs[np.argwhere(wei != 0)]
                res_for_eva = res[np.argwhere(wei != 0)]
                gp_metric.append(self.evaluation_metric(obs_for_eva, res_for_eva))
            metric = np.average(gp_metric)
        # Consider both the dropout and the weight, a weighted average evaluation metric will be calculated
        else:
            gp_metric = []
            for gp in lgpres:
                obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                if len(set(gp.wei)) > 1:
                    # Weighted average with dropout
                    obs_for_eva = obs[np.argwhere(wei != 0)]
                    res_for_eva = res[np.argwhere(wei != 0)]
                    gp_metric.append(np.max(wei) * self.evaluation_metric(obs_for_eva, res_for_eva))
                else:
                    # Weighted average without dropout
                    gp_metric.append(np.max(wei) * self.evaluation_metric(obs, res))
            metric = np.sum(gp_metric)
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))

        if self.mode == "dynamic":
            self._rm_dirMP(pid)
        else:
            self._closeExeMPF(pid)

        return metric, res_obj


    def _objfunPESTMP(self,X):
        if self.mode == "dynamic":
            pid,lgpres = self._call_simMP(X)
        else:
            pid, lgpres = self._call_simMPF(X,self.n_jobs)
        weighted_errs = []
        # Uss the residual (same as PEST++) as the objective function (err = sum((w*r)**2))
        for gp in lgpres:
            for o, r, w in zip(gp.obs, gp.res, gp.wei):
                werr = np.square(w * (r - o))
                weighted_errs.append(werr)
        metric = np.sum(weighted_errs)
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))
        if self.mode == "dynamic":
            self._rm_dirMP(pid)
        else:
            self._closeExeMPF(pid)
        return metric, res_obj

    def _objfunMOMP(self,X):
        if self.mode == "dynamic":
            pid,lgpres = self._call_simMP(X)
        else:
            pid, lgpres = self._call_simMPF(X,self.n_jobs)
        weight_types = []
        for gp in lgpres:
            weight_types.append(gp.check_weights())
        flag = sum([issubclass(i, DorpOut) for i in weight_types])
        # Only consider the observation dropout, a mean evaluation metric without weight will be calculated
        if flag == len(weight_types):
            metrics = []
            for e_fun in self.evaluation_metric:
                gp_metric = []
                for gp in lgpres:
                    obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                    obs_for_eva = obs[np.argwhere(wei != 0)]
                    res_for_eva = res[np.argwhere(wei != 0)]
                    gp_metric.append(e_fun(obs_for_eva, res_for_eva))
                metrics.append(np.average(gp_metric))
        # Consider both the dropout and the weight, a weighted average evaluation metric will be calculated
        else:
            metrics = []
            for e_fun in self.evaluation_metric:
                gp_metric = []
                for gp in lgpres:
                    obs, res, wei = np.array(gp.obs), np.array(gp.res), np.array(gp.wei)
                    if len(set(gp.wei)) > 1:
                        # Weighted average with dropout
                        obs_for_eva = obs[np.argwhere(wei != 0)]
                        res_for_eva = res[np.argwhere(wei != 0)]
                        gp_metric.append(np.max(wei) * e_fun(obs_for_eva, res_for_eva))
                    else:
                        # Weighted average without dropout
                        gp_metric.append(np.max(wei) * e_fun(obs, res))
                metrics.append(np.sum(gp_metric))
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))

        if self.mode == "dynamic":
            self._rm_dirMP(pid)
        else:
            self._closeExeMPF(pid)

        if len(metrics) > 1:
            return metrics, res_obj
        else:
            return metrics[0], res_obj

    def _objfunPredMP(self,X):
        if self.mode == "dynamic":
            pid,lgpres = self._call_simMP(X)
        else:
            pid, lgpres = self._call_simMPF(X,self.n_jobs)
        res_obj = Reslib.SimulationResult()
        for gp in lgpres:
            res_obj.add_station(gp.gpname)
            res_obj[gp.gpname].add_event("simulation results", np.array(gp.res).reshape(1, -1))
            res_obj[gp.gpname].add_event("observations", np.array(gp.obs).reshape(1, -1))
        if self.mode == "dynamic":
            self._rm_dirMP(pid)
        else:
            self._closeExeMPF(pid)
        return res_obj

    def _gen_boundary(self):
        lb = []
        ub = []
        tar_pars = []
        oth_pars = []
        all_pars = self.param.list_params()
        for p in all_pars:
            if p.type == "fixed" or "tied" in p.type:
                oth_pars.append(p)
            else:
                tar_pars.append(p)
        for p in tar_pars:
            lb.append(p.lb)
            ub.append(p.ub)
        self.tar_pars = tar_pars
        self.oth_pars = oth_pars
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim=len(self.lb)

    def gen_paramIO(self):

        self.UpdateParam = self._paramIO
        return self._paramIO

    def _paramIO(self,X,subfolder=None):
        for tpl in self.io_pairs.fparam:
            if not subfolder:
                tpl_path = self.workspace + "\\"+  tpl
                par_path = self.workspace + "\\" + self.io_pairs.fparam[tpl]
            else:
                tpl_path = self.workspace + "\\"+ subfolder + "\\"+ tpl
                par_path = self.workspace + "\\"+ subfolder + "\\" + self.io_pairs.fparam[tpl]
            with open(tpl_path,"r") as f:
                dlmt = list(filter(None, f.readline().split(" ")))[-1].strip("\n")
                lines = "".join(f.readlines())
                for p, tp in zip(X, self.tar_pars):
                    # designed for user defined interval
                    if tp.gptype == "absolute":
                        t = math.floor(p/tp.gpderinc)
                        p = t * tp.gpderinc
                    lines=re.sub("{}\s*?{}\s*?{}".format(dlmt,tp.name,dlmt),str(p),lines,flags=re.IGNORECASE)
                for to in self.oth_pars:
                    if to.type == "fixed":
                        lines = re.sub("{}\s*?{}\s*?{}".format(dlmt, to.name, dlmt), str(to.ini), lines,flags=re.IGNORECASE)
                    elif "tied" in to.type:
                        tied_par_name = to.type.replace("tied_","")
                        for cand,val in zip(self.tar_pars,X):
                            if cand.name.upper() == tied_par_name.upper():
                                p_val = val * to.ini/cand.ini
                                if to.gptype == "absolute":
                                    t = math.floor(p_val / tp.gpderinc)
                                    p_val = t * tp.gpderinc
                                lines = re.sub("{}\s*?{}\s*?{}".format(dlmt, to.name, dlmt), str(p_val), lines,flags=re.IGNORECASE)

            with open(par_path,"w") as w:
                w.write(lines)

    def gen_resultIO(self):
        self.ReadResult = self._resultIO
        return self._resultIO

    def _resultIO(self,subfolder=None):
        dic_res = PESTsimpair()

        for ins, out in zip(self.io_pairs.fout.keys(), self.io_pairs.fout.values()):
            if not subfolder:
                ins_path = self.workspace + "\\" + ins
                out_path = self.workspace + "\\" + out
            else:
                ins_path = self.workspace + "\\"+ subfolder + "\\" + ins
                out_path = self.workspace + "\\"+ subfolder + "\\" + out
            insReader = PESTinsReader(ins_path)
            resReader = PESTresultReader(out_path,insReader)
            dic_res.add_data(resReader.data_name, resReader.data)
        lgpres = []
        for gp in self.obs:
            gp_result = PESTsimgp(gpname=gp.name)
            #print(gp.index==dic_res.idx)
            # i -> observation name, d -> observation data, dic_res[i] -> simulation result data, w -> observation weight
            for i, d, w in zip(gp.index, gp.data, gp.weights):
                #print(i,dic_res[i])
                gp_result.add_data(i, d, dic_res[i], w)
            lgpres.append(gp_result)
        return lgpres

    def gen_call(self,callfun="subprocess"):
        if callfun == "subprocess":
            if self.mpi:
                CallExe = self.CallExe1
            else:
                CallExe = self.CallExe2
        else:
            raise RuntimeError("The argument callfun only supports 'subprocess' in this version.")
        self.CallExe = CallExe
        return CallExe

    def CallExe1(self,subfolder=None):
        if not subfolder:
            os.chdir(self.workspace)
        else:
            os.chdir(self.workspace + "\\" + subfolder)
        for i in self.cmd:
            if "tsproc.exe" in i:
                if not os.path.isabs(i):
                    i = self.workspace+"\\"+i
            if re.findall(".*?\.py",i):
                if not re.match("python\s+.*?\.py",i):
                    i = "python " + i
            if self.cmd_hide:
                p =subprocess.Popen(i.split(" "), stdout=subprocess.PIPE)
                p.wait()
                p.kill()
            else:
                p =subprocess.Popen(i.split(" "))
                p.wait()
                p.kill()

    def CallExe2(self):

        os.chdir(self.workspace)
        for i in self.cmd:
            verbs = i.split(" ")
            for idx,v in enumerate(verbs):
                if "tsproc.exe" in v:
                    if not os.path.isabs(v):
                        verbs[idx] = self.workspace + "\\" + verbs[idx]
            if self.cmd_hide:
                try:
                    subprocess.run(verbs, stdout=subprocess.PIPE)
                except:
                    subprocess.call(verbs, stdout=subprocess.PIPE)
            else:
                try:
                    subprocess.run(verbs)
                except:
                    subprocess.call(verbs)


    def _analyze_pst(self):
        # ignore the empty row
        def check_empty(content):
            indicators = ["\n","\t"," "]
            nc = []
            for row in content:
                # ignore '++' PEST++ settings
                if sum([bool(s not in indicators) for s in row]) > 0 and not re.match("\+\+.*?",row):
                    nc.append(row)
            return nc
        with open(self.pst_path, "r") as f:
            pst_content = f.readlines()
            idxs = []
            labels = []
            for idx,label in enumerate(pst_content):
                if "*" in label:
                    idxs.append(idx)
                    labels.append(label)
            for l in labels:
                if re.match("\*\sexternal.*?",l):
                    raise NotImplementedError("The convertor currently does not support external parameter/group data files, this will be implemented in the near future.")
            idxs.append(None)
            labels.append(None)
            gp_content = check_empty(pst_content[idxs[labels.index("* parameter groups\n")]+1:idxs[labels.index("* parameter groups\n")+1]])
            par_content = check_empty(pst_content[idxs[labels.index("* parameter data\n")]+1:idxs[labels.index("* parameter data\n")+1]])
            self._analyze_param(gp_content,par_content)

            obs_gp_content = check_empty(pst_content[idxs[labels.index("* observation groups\n")] + 1:idxs[labels.index("* observation groups\n") + 1]])
            obs_data_content = check_empty(pst_content[idxs[labels.index("* observation data\n")] + 1:idxs[labels.index("* observation data\n") + 1]])
            self._analyze_observation(obs_gp_content,obs_data_content)

            batch_content = check_empty(pst_content[idxs[labels.index("* model command line\n")] + 1:idxs[labels.index("* model command line\n") + 1]])
            self._analyze_batch(self.workspace,batch_content)

            input_content = check_empty(pst_content[idxs[labels.index("* model input/output\n")] + 1:idxs[labels.index("* model input/output\n") + 1]])
            self._analyze_input(self.workspace,input_content)

    def _analyze_param(self,gp_content,par_content):
        param_obj = PESTparams(gp_content,par_content)
        self.param = param_obj

    def _analyze_observation(self,obs_gp_content,obs_data_content):
        obs_obj = PESTobs(obs_gp_content,obs_data_content)
        self.obs = obs_obj

    def _analyze_batch(self,workspace,batch_content):
        batch_obj = PESTbatch(workspace,batch_content)
        self.cmd = batch_obj.cmds

    def _analyze_input(self,workspace,input_content):
        input_pairs = PESTio(workspace,input_content)
        self.io_pairs = input_pairs

    def _scan_prj(self):
        self._find_pst()

    def _find_pst(self,fmt=".pst"):
        pst_path = glob.glob(self.workspace+"\\*{}".format(fmt))
        if len(pst_path) == 1:
            self.pst_path = pst_path[0]
        elif len(pst_path) > 1:
            warnings.warn("PEST control file (.pst) recognization failed: There is more than one .pst file in the folder.")
            self.pst_path = None
        else:
            self.pst_path = None

    def _read_bat_tsproc(self):
        with open(self.bat_fname,"r") as f:
            bat_lines = f.readlines()
            for l in bat_lines:
                if "tsproc" in l:
                    cmds = list(filter(None,l.split(" ")))
                    self.tsproc_path = self.workspace + "\\" + cmds[1]
                    self.record_path = self.workspace + "\\" + cmds[2]
                    if len(cmds) > 3:
                        self.context = cmds[3]
                    else:
                        self.context = None
    """"""
    def show_task_info(self):
        info_tbl = pt.PrettyTable()
        info_tbl.title = "PyCUP--PEST++ Project Conversion Results"
        info_tbl.field_names = ["Item","Detailed Information"]
        info_tbl.add_row(["PEST++ Workspace",self.workspace])
        info_tbl.add_row(["PEST++ Control File",self.pst_path])
        info_tbl.add_row(["Parameter Files","\n".join(self.io_pairs.fparam.values())])
        info_tbl.add_row(["Parameter Templates","\n".join(self.io_pairs.fparam.keys())])
        info_tbl.add_row(["Result Files", "\n".join(self.io_pairs.fout.values())])
        info_tbl.add_row(["Result Instructions", "\n".join(self.io_pairs.fout.keys())])
        info_tbl.add_row(["Command Lines","\n".join(self.cmd)])
        print(info_tbl)

    def show_parameter_info(self):
        info_tbl = pt.PrettyTable()
        info_tbl.title = "Calibration Parameter Info"
        info_tbl.field_names = ["Name", "Type", "Ini. Val.", "Lower Bnd.", "Upper Bnd.", "Group", "Calibration"]
        for p in self.tar_pars:
            info_tbl.add_row([p.name, p.type, str(p.ini), str(p.lb), str(p.ub), p.gp, "True"])
        for p in self.oth_pars:
            info_tbl.add_row([p.name, p.type, str(p.ini), "N/A", "N/A", p.gp, "False"])
        print(info_tbl)