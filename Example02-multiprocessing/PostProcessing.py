import matplotlib.pyplot as plt
import numpy as np
from pycup import save
import pycup.uncertainty_analysis_fun as u_fun
from pycup import plot

data = np.load("calibration_data.npy")
raw_saver = save.RawDataSaver.load("RawResult.rst")
u_fun.frequency_uncertainty(raw_saver,0.5,95)
post_saver = save.ProcResultSaver.load("ProcResult.rst")
plot.plot_opt_curves(raw_saver)
plot.plot_2d_fitness_space(raw_saver,10)

plot.plot_uncertainty_band(post_saver,obsx=np.arange(len(data)),obsy=data)
