import numpy as np
from pycup import save
import pycup.uncertainty_analysis_fun as u_fun
from pycup import plot
from pycup import Reslib

Reslib.UseResObject = True
data = np.load("O1Event1.npy")
raw_saver = save.RawDataSaver.load("RawResult.rst")

"""
extract and print the result at each station in each event
When the Reslib is used, all the result related data should be given a station name index and an event name
if you want to check the data
"""
print(raw_saver.historical_results["O1"]["event1"])

# export your historical results to the database (xlsx, h5, and hdf5 are supported)
db_writer = Reslib.DataBaseWriter(res_package=raw_saver.historical_results,
                                  params=raw_saver.historical_samples,
                                  fitness=raw_saver.historical_fitness,
                                  fmt="xlsx",
                                  pop=45,
                                  dim=11)
db_writer.write2DB()

# uncertainty analysis
u_fun.frequency_uncertainty(raw_saver,0.2,95)
post_saver = save.ProcResultSaver.load("ProcResult.rst")
plot.plot_opt_curves(raw_saver)
plot.plot_2d_fitness_space(raw_saver,10,slim=False)

# you must specify the station name and the event name for this case
plot.plot_uncertainty_band(post_saver,obsx=np.arange(len(data)),obsy=data,station="O1",event="event1")
