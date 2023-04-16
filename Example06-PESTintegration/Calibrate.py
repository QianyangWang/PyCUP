import pycup
from pycup.integrate import PESTconvertor



if __name__ == "__main__":
    con = PESTconvertor(r"PESTxaj")
    con.convert(mpi=True,mpimode="fixed",evaluation_metric=pycup.evaluation_metrics.OneMinusNSE,n_jobs=5,hide=True)
    con.show_task_info()
    con.show_parameter_info()
    pycup.GLUE.runMP(50,con.dim,con.lb,con.ub,con.objfun,n_jobs=5)
    pycup.Reslib.UseResObject = True
    res = pycup.save.RawDataSaver.load(r"RawResult.rst")
    print(res.historical_results["i_mod_est"]["simulation results"])
    print(res.historical_results["i_mod_est"]["observations"])