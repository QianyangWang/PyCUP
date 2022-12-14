import pycup as cp
import numpy as np
import params
import multiprocessing as mp
import inp_io
import callSWMM as cswmm
import read_res

outfalls = ['O1']
calibration_data = np.load("calibration_data.npy")

# update parameters
def update_params(X,inp):
    inp = params.write_p_zero(X[0],inp)
    inp = params.write_n_imperv(X[1],inp)
    inp = params.write_n_perv(X[2], inp)
    inp = params.write_max_rate(X[3],inp)
    inp = params.write_min_rate(X[4],inp)
    inp = params.write_dry_time(X[5],inp)
    inp = params.write_decay(X[6],inp)
    inp = params.write_s_perv(X[7],inp)
    inp = params.write_s_imperv(X[8], inp)
    inp = params.write_conduit_roughness(X[9], inp)
    inp = params.write_ws_width(X[10],inp)
    return inp

# important, your objective function for multi-processing calibration
def objective_functionMP(X,read_start,read_end,read_step,n_jobs):
    # 1. allocate the current process folder
    process_id = mp.current_process().name.split("-")[-1]
    folder_id = int(process_id) % n_jobs + 1
    proj_dir = r"process{}".format(folder_id)
    inp_path = proj_dir + r"\Example2-Post.inp"
    rpt_path = proj_dir + r"\Example2-Post.rpt"
    swmm_out_path = proj_dir + r"\Example2-Post.out"
    fmodel = proj_dir + r"\swmm5.exe"

    # 2. open the inp file and update your parameters
    inp = inp_io.open_inp(inp_path)
    inp = update_params(X, inp)
    inp_io.update_inp(inp, inp_path)

    # 3. call the executable
    cswmm.callexe(fmodel, inp_path, rpt_path, swmm_out_path)

    # 4. read the simulation result
    res, actual_stamps = read_res.read_swmm_result(swmm_out_path,outfalls, read_start, read_end, read_step)
    res_outfall = res[0]

    # 5. calculate the fitness value
    fitness = cp.evaluation_metrics.OneMinusNSE(calibration_data,res_outfall)
    res_outfall = np.array(res_outfall).reshape(1,-1)
    return fitness,res_outfall


if __name__ == "__main__":

    # Calibrate a SWMM model (USEPA example 02)
    lb = np.array([0,0.01, 0.05, 30, 1, 1, 0, 2, 0.1, 0.01, 0.2])
    ub = np.array([50, 0.05, 0.8, 250, 30, 10, 50, 10, 4, 0.05, 5.0])
    cp.GWO.EliteOppoSwitch = True
    cp.GWO.OppoFactor = 0.1
    cp.GWO.runMP(45, 11, lb=lb, ub=ub, MaxIter=39, fun=objective_functionMP, n_jobs=2,
             args=("2007-01-01 00:01:00", "2007-01-01 12:00:00", 60,  2))