import matplotlib.pyplot as plt
import numpy as np
from pycup import save
from pycup import TOPSIS
from pycup import plot

data = np.load("calibration_data.npy")
raw_saver = save.RawDataSaver.load("RawSWMM.rst")
top = TOPSIS.TopsisAnalyzer(raw_saver)
top.updateTopsisRawSaver(r"TopsisRawSWMM.rst")
topsis_saver = save.RawDataSaver.load(r"TopsisRawSWMM.rst")
# Plot the Pareto front with the TOPSIS optimum.
plot.plot_2d_pareto_front(topsis_saver,objfunid1=0,objfunid2=1,topsis_optimum=True,slim=False)

