# Using pycup.integrate to integrate with spotpy

## Potential Dependencies 

1. spotpy (https://github.com/thouska/spotpy)
2. h5py

## From pycup calibration result file to spotpy database

The spotpy database in a csv or hdf5 format is supported. For a csv format:

```python
from pycup.integrate import SpotpyDbConverter
s = SpotpyDbConverter()
s.RawSaver2csv(r"RawResult.rst",r"spotpyres.csv")
res = spotpy.analyser.load_csv_results(r"spotpyres")
print(res["like1"])
```

For a hdf5 format:

```python
from pycup.integrate import SpotpyDbConverter
s = SpotpyDbConverter()
s.RawSaver2hdf5(r"RawResult.rst",r"spotpyres.h5")
res = spotpy.analyser.load_hdf5_results(r"spotpyres")
print(res["like1"])
```

If the multi-station and multi-event data structure in pycup.ResLib has been used, the station name and the event name should be specified, for example:

```python
from pycup.integrate import SpotpyDbConverter
s = SpotpyDbConverter()
s.RawSaver2hdf5(r"RawResult.rst",r"spotpyres.h5",station="station1",event="event1")
```

## From spotpy database to pycup calibration result 

The spotpy calibration result database can be converted to the pycup.save.RawDataSaver object to use the post-processing functionalities in pycup. For a database in hdf5 format:

```python
import pycup
s = SpotpyDbConverter()
s.hdf52RawSaver("spotpyres.h5","RawResult.rst")
saver = pycup.save.RawDataSaver.load("RawResult.rst")
pycup.plot.plot_2d_fitness_space(saver,variable_id=0)
```

For a csv format:

```python
import pycup
s = SpotpyDbConverter()
s.csv2RawSaver("SCEUA_hymod.csv","RawResult.rst")
saver = pycup.save.RawDataSaver.load("RawResult.rst")
pycup.plot.plot_2d_fitness_space(saver,variable_id=0)
```

## Using a spotpy model setup object to start a quick pycup calibration

```python
import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup
spot_setup=spot_setup(spotpy.objectivefunctions.rmse)
s = SpotpySetupConverter()
s.convert(spot_setup)
# the lb, ub, and objective function are stored in the attributes of the converter.
# if you have other arguments required by the objective function, they can be passed through the args. 
pycup.MFO.run(20,s.dim,s.lb,s.ub,10,s.obj_fun,args=())
saver = pycup.save.RawDataSaver.load("RawResult.rst")
saver2 = pycup.uncertainty_analysis_fun.frequency_uncertainty(saver,10,95)
pycup.plot.plot_uncertainty_band(saver2,None,None)
```