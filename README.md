# PyCUP

​	This is an open-source package designed for (environmental) model calibration and uncertainty analysis. The current version is the very first version, we welcome all comments, suggestions, and improvements.

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

​	Folder 'PyCUPexample01' contains an SWMM calibration project using single-processing GLUE. Install the dependencies (for example: pip install swmm-api==0.2.0.18.3, pip install pyswmm). Execute the 'Calibrate.py' to calibrate the model. Then, execute the 'PostProcessing.py' for uncertainty analysis.
​	Folder 'PyCUPexample02' contains an SWMM calibration project using multi-processing EOGWO.

