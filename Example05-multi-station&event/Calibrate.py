import pycup as cp
import numpy as np
import params
import inp_io
import callSWMM as cswmm
import read_res
from pycup import Reslib

outfalls = ['O1',"O2"]
O1Event1=np.load("O1Event1.npy")
O1Event2=np.load("O1Event2.npy")
O2Event1=np.load("O2Event1.npy")
O2Event2=np.load("O2Event2.npy")

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
def objective_function(X):

    proj_dir = r"project"
    inp_path = proj_dir + r"\Example5-Post.inp"
    rpt_path = proj_dir + r"\Example5-Post.rpt"
    swmm_out_path = proj_dir + r"\Example5-Post.out"
    fmodel = proj_dir + r"\swmm5.exe"

    # 2. open the inp file and update your parameters
    inp = inp_io.open_inp(inp_path)
    inp = update_params(X, inp)
    inp_io.update_inp(inp, inp_path)

    # 3. call the executable
    cswmm.callexe(fmodel, inp_path, rpt_path, swmm_out_path)

    # 4. read the simulation result
    res1,_ = read_res.read_swmm_result(swmm_out_path,outfalls,  "2007-01-01 00:10:00", "2007-01-01 23:00:00", 600)
    res_o1e1,res_o2e1 = res1[0],res1[1]

    res2,_ = read_res.read_swmm_result(swmm_out_path,outfalls,  "2007-01-02 00:00:00", "2007-01-02 23:00:00", 600)
    res_o1e2,res_o2e2 = res2[0],res2[1]

    # 5. calculate the fitness, use the average NSE value as the obj func value
    fito1e1 = cp.evaluation_metrics.OneMinusNSE(O1Event1,res_o1e1)
    fito1e2 = cp.evaluation_metrics.OneMinusNSE(O1Event2,res_o1e2)
    fito2e1 = cp.evaluation_metrics.OneMinusNSE(O2Event1, res_o2e1)
    fito2e2 = cp.evaluation_metrics.OneMinusNSE(O2Event2, res_o2e2)
    fitness = np.mean([fito1e1,fito1e2,fito2e1,fito2e2])

    # 6. wrap your results using the Simulation result object
    sim_result = Reslib.SimulationResult()
    sim_result.add_station("O1")
    sim_result.add_station("O2")
    sim_result["O1"].add_event("event1",res_o1e1.reshape(1,-1))
    sim_result["O1"].add_event("event2", res_o1e2.reshape(1, -1))
    sim_result["O2"].add_event("event1", res_o2e1.reshape(1, -1))
    sim_result["O2"].add_event("event2", res_o2e2.reshape(1, -1))

    return fitness,sim_result


if __name__ == "__main__":

    # This setting is important if you want to use the relevant object for multi-station multi-event calibration
    Reslib.UseResObject = True
    # Calibrate a SWMM model (USEPA example 02)
    lb = np.array([0,0.01, 0.05, 30, 1, 1, 0, 2, 0.1, 0.01, 0.2])
    ub = np.array([50, 0.05, 0.8, 250, 30, 10, 50, 10, 4, 0.05, 5.0])
    cp.MFO.OppoFactor = 0.1
    cp.MFO.run(45, 11, lb=lb, ub=ub, MaxIter=39, fun=objective_function)