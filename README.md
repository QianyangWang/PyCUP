# PyCUP

<img src="https://img.shields.io/badge/Version-0.1.7-brightgreen" /><img src="https://img.shields.io/badge/Language-Python-blue" />	

This is an open-source package designed for (environmental) model calibration and uncertainty analysis. The current version is the very first version, we welcome all comments, suggestions, and improvements.

## v 0.1.7 Update

A new superior algorithm MOMFO (multi-objective moth-flame optimizer)  based on archive and crowding distance non-domination sort, and its elite-opposition improved version (EO-MOMFO) have been designed and provided in the new version. The elite opposition mechanism was modified based on the concept of non-domination and was embedded for updating the flame population of MOMFO. The implemented MOMFO has a similar principle as the current literature, although with some differences in details, while the elite-opposition version is original in this package.

## What does it have

### (1) For model calibration/optimization

1. Single-objective heuristic algorithms including PSO, GWO, MFO, SOA, SCA, SSA, TSA, and WOA.
2. Multi-objective heuristic algorithms including MOPSO, MODE, and NSGA-II.
3. Elite opposition strategy modified heuristic algorithms -- with better optimum search abilities.
4. Statistic based-method LHS-GLUE.
5. Three kinds of algorithm border check mechanisms including Absorb, Random, and Rebound, designed for different problems.

### (2) For sensitivity & uncertainty analysis

1. Likelihood uncertainty estimation used in the GLUE framework for the parameter uncertainty analysis/prediction uncertainty estimation.
2. The frequency based-uncertainty estimation method for the prediction uncertainty estimation.
3. The multi-linear regression method for the all-at-a-time parameter sensitivity based on statmodels.

### (3) Other convenient features

1. Multi-processing calibration.
2. Recording and resuming during the calibration task.
3. Several result plotting functions.
4. A special simulation result object  for multi-station & multi-event results (of environmental models) in pycup.ResLib.

### (4) Package/Tools integration

1. PyCUP can be linked to spotpy database for post-processing, a pycup objective function can also be generated from the spotpy objective function using the module named pycup.integrate.
2. A basic integration with PEST++ IO operations for model-agnostic calibrations. Details and  limitations are provided in the specific documentation. The required objective function for pycup calibration can be easily generated using a PEST++ optimization project with/without a tsproc.exe. The PESTconvertor object in pycup.integrate provides several APIs for reading PEST++ files such as .pst, .ins, and .tpl.

## How to install

​	The project has been uploaded onto the PyPI https://pypi.org/project/pycup/ . Or install the .whl file in the dist folder.

```
pip install pycup
```

## How to use

​	Here is a simple example. For more details, please see the documentation.

```python
import pycup as cp
import numpy as np

def uni_fun1(X):
	# X for example np.array([1,2,3,...,30])
    fitness = np.sum(np.power(X,2)) + 1 # example: 1.2
    result = fitness.reshape(1,-1) # example ([1.2,])

    return fitness,result

lb = -100 * np.ones(30)
ub = 100 * np.ones(30)
cp.SSA.run(pop = 1000, dim = 30, lb = lb, ub = ub, MaxIter = 30, fun = uni_fun1)
```

## Example SWMM (Storm Water Management Model) calibration projects

***IMPORTANT: PLEASE OPEN YOUR IDE (e.g. PYCHARM) OR COMMAND LINE WITH THE ADMINISTRATOR RIGHTS BEFORE EXECUTING THE EXAMPLE PROJECT***

#### Location: https://github.com/QianyangWang/PyCUP

1. The example in folder 'Example01-GLUE' contains an SWMM calibration project using single-processing GLUE. Install the dependencies (for example: pip install swmm-api==0.2.0.18.3, pip install pyswmm). Execute the 'Calibrate.py' to calibrate the model. Then, execute the 'PostProcessing.py' for uncertainty analysis.
2. The example in folder 'Example02-multiprocessing' contains an SWMM calibration project using multi-processing EOGWO.
3. The example in folder 'Example03-multiobjective' contains an SWMM multi-objective calibration project using EOMOPSO. 
4. The example in folder 'Example04-validation&prediction' shows how to use our (Ensemble)Validator/(Ensemble)Predictor objects for the validation and prediction of the model using the calibrated parameter (set).
5. The example in folder 'Example05-multi-station&event' shows how to use the pycup.Reslib.SimulationResult object for the storage of multi-station & multi-event simulation results, as well as the further analysis using them.

<div align=center>
<img src="https://user-images.githubusercontent.com/116932670/209893309-e67c425f-0eff-47b4-a552-b30d717a138b.png">
</div>

## Example PEST++ conversion project (with a Xinanjiang hydrologic model)

1. The example in folder 'Example06-PESTintegration' contains a PEST++ Xinanjiang model calibration project and the python script to run a PyCUP calibration based on it.
