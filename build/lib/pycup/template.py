import os
import pycup.evaluation_metrics
import multiprocessing as mp


def template_obj_fun(X, execute, fsim, fparam, fres,obs):
    """

    :param X:
    :param execute:
    :param fsim:
    :param fparam:
    :param fres:
    :param obs:
    :return:
    """
    write_param(X,fparam)
    callexe(execute,fsim)
    res = read_result(fres)
    nse = pycup.evaluation_metrics.OneMinusNSE(obs,res)
    return nse,res

def template_obj_fun_mp(X, execute, dir_proj, fnsim,fnparam, fnres,obs,n_jobs):
    """

    :param X:
    :param execute:
    :param dir_proj:
    :param fnparam:
    :param fnres:
    :param obs:
    :param n_jobs:
    :return:
    """
    process_id = mp.current_process().name.split("-")[-1]
    """
    Allocalate the folder id for parallel simulations.
    
    Users can create several folders to simulate parallelly in case of potential conflictions between different
    simulations. By doing this, the program can write the param and read the result separately according to the 
    process-id.
    for example:
        We have 4 processes (n_jobs = 4)
        r'hydro_simulation\\process1\\'
        r'hydro_simulation\\process2\\'
        r'hydro_simulation\\process3\\'
        r'hydro_simulation\\process4\\'
    the process-id is 0013,
    then, its corresponding folder id will be 13 % 4 = 1 + 1 = 2
    the params and results will be written in r'hydro_simulation\\process2\\'
    """
    folder_id = int(process_id) % n_jobs + 1
    process_folder = dir_proj + r"\\process{}\\".format(folder_id)

    f_param = process_folder + fnparam
    write_param(X, f_param)

    f_sim = process_folder + fnsim
    callexe(execute,f_sim)

    f_res = process_folder + fnres
    res = read_result(f_res)

    nse = pycup.evaluation_metrics.OneMinusNSE(obs, res)
    return nse, res


def callexe(f_model, f_sim):

    cmd = f_model + " " + f_sim
    ret = os.system(cmd)

    return ret


def write_param(param,fparam):
    pass


def read_result(fres):
    pass