import pycup as cp
import numpy as np
import params
import multiprocessing as mp
import inp_io
import callSWMM as cswmm
import read_res

outfalls = ['O1']


# update parameters
def update_params(X, inp):
    inp = params.write_p_zero(X[0], inp)
    inp = params.write_n_imperv(X[1], inp)
    inp = params.write_n_perv(X[2], inp)
    inp = params.write_max_rate(X[3], inp)
    inp = params.write_min_rate(X[4], inp)
    inp = params.write_dry_time(X[5], inp)
    inp = params.write_decay(X[6], inp)
    inp = params.write_s_perv(X[7], inp)
    inp = params.write_s_imperv(X[8], inp)
    inp = params.write_conduit_roughness(X[9], inp)
    inp = params.write_ws_width(X[10], inp)
    return inp


# important, your objective function for multi-processing prediction
# Please note, the obj func for the prediction process does not return the fitness value
def objective_function(X, read_start, read_end, read_step):
    # 1. allocate the current process folder
    proj_dir = r"project"
    inp_path = proj_dir + r"\Example4-Post.inp"
    rpt_path = proj_dir + r"\Example4-Post.rpt"
    swmm_out_path = proj_dir + r"\Example4-Post.out"
    fmodel = proj_dir + r"\swmm5.exe"

    # 2. open the inp file and update your parameters
    inp = inp_io.open_inp(inp_path)
    inp = update_params(X, inp)
    inp_io.update_inp(inp, inp_path)

    # 3. call the executable
    cswmm.callexe(fmodel, inp_path, rpt_path, swmm_out_path)

    # 4. read the simulation result
    res, actual_stamps = read_res.read_swmm_result(swmm_out_path, outfalls, read_start, read_end, read_step)
    res_outfall = res[0]

    res_outfall = np.array(res_outfall).reshape(1, -1)
    return  res_outfall


if __name__ == "__main__":
    ## Ensemble prediction (use the same objective function for the convenience)
    saver = cp.save.ProcResultSaver.load(r"ProcResult.rst")

    # create a predictor
    predictor = cp.utilize.EnsemblePredictor(saver,  obj_fun=objective_function,
                                             args=("2007-01-01 00:01:00", "2007-01-01 12:00:00", 60))
    # run prediction--1 process
    predictor.run()

    # load the prediction result saver
    pred_saver = cp.save.PredRawSaver.load(r"PredictionRawSaver.rst")

    # post processing
    cp.uncertainty_analysis_fun.prediction_frequency_uncertainty(pred_saver, ppu=95, intervals=20)
    proc_saver = cp.save.PredProcSaver.load(r"PredictionProcSaver.rst")

    # plot
    cp.plot.plot_uncertainty_band(proc_saver,obsx=None,obsy=None)
