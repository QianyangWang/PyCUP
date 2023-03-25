import warnings
import numpy as np
from . import save
from .NSGA2 import nonDominationSort
import os
from . import Reslib
# if this module is used, the spotpy, h5py, and tables should be installed

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
