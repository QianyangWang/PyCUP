<<<<<<< HEAD
# PyCUP

<img src="https://img.shields.io/badge/Version-0.1.2-brightgreen" /><img src="https://img.shields.io/badge/Language-Python-blue" />	

This is an open-source package designed for (environmental) model calibration and uncertainty analysis. The current version is the very first version, we welcome all comments, suggestions, and improvements.

## v 0.1.2 Update

1. The bug of the algorithm MOPSO when using the function runMP for calibrating modelling softwares has been fixed. (The misused single-processing CalculateFitness function in it)
2. The function named "CaculateFitness" has been replaced by "CalculateFitness".
3. The attached documentation file size has been compressed. Therefore, the .whl file size is smaller now.
4. Most importantly, the TOPSIS method for determining the global optimum of the Pareto front has been provided. By utilizing the pycup.Topsis.TopsisAnalyzer, users can carry out the TOPSIS analysis based on the multi-objective optimization result. The calibration result pycup.save.RawDataSaver can be updated by the TopsisAnalyzer.updateTopsisRawSaver(saveing_path) to obtain the global optimum .GbestPosition and .GbestScore, which are not available before the TOPSIS analysis. For more information, please see the documentation.

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

​	Folder 'PyCUPexample01' contains an SWMM calibration project using single-processing GLUE. Install the dependencies (for example: pip install swmm-api==0.2.0.18.3, pip install pyswmm). ***IMPORTANT:PLEASE OPEN YOUR PYCHARM OR COMMAND LINE WITH THE ADMINISTRATOR RIGHTS BEFORE EXECUTING THE PROJECT***. Execute the 'Calibrate.py' to calibrate the model. Then, execute the 'PostProcessing.py' for uncertainty analysis.
​	Folder 'PyCUPexample02' contains an SWMM calibration project using multi-processing EOGWO.

​	Folder 'PyCUPexample03' contains an SWMM multi-objective calibration project using EOMOPSO. 



<div align=center><img src="https://user-images.githubusercontent.com/116932670/200309723-e7730802-9dc9-4304-b86f-456b07a91b31.jpg" alt="opt_curves" style="zoom:20%;" /><img src="https://user-images.githubusercontent.com/116932670/200312283-e30e1ff0-0679-4bc0-8c94-743486b45a05.jpg" alt="fitness2d" style="zoom:10.1%;" /></div>

<div align=center><img src="https://user-images.githubusercontent.com/116932670/200309801-c0035f68-0d3a-4e28-ad1c-66ded9cd8052.jpg" alt="band" style="zoom: 25%;" /></div>
=======
# PyCUP

​	This is an open-source package designed for (environmental) model calibration and uncertainty analysis. The current version is the very first version, we welcome all comments, suggestions, and improvements.

## v 0.1.1 Update

​	The bug of defining the record_path (the user defined saving path for the RecordSaver object) was fixed. 

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

​	Folder 'PyCUPexample01' contains an SWMM calibration project using single-processing GLUE. Install the dependencies (for example: pip install swmm-api==0.2.0.18.3, pip install pyswmm). ***IMPORTANT:PLEASE OPEN YOUR PYCHARM OR COMMAND LINE WITH THE ADMINISTRATOR RIGHTS BEFORE EXECUTING THE PROJECT***. Execute the 'Calibrate.py' to calibrate the model. Then, execute the 'PostProcessing.py' for uncertainty analysis.
​	Folder 'PyCUPexample02' contains an SWMM calibration project using multi-processing EOGWO.

<img src="https://user-images.githubusercontent.com/116932670/200309723-e7730802-9dc9-4304-b86f-456b07a91b31.jpg" alt="opt_curves" style="zoom:28%;" />
<img src="https://user-images.githubusercontent.com/116932670/200312283-e30e1ff0-0679-4bc0-8c94-743486b45a05.jpg" alt="fitness2d" style="zoom:15%;" />
<img src="https://user-images.githubusercontent.com/116932670/200309801-c0035f68-0d3a-4e28-ad1c-66ded9cd8052.jpg" alt="band" style="zoom: 29%;" />

>>>>>>> b931dcd6faa7460293bd19fec17d5932fefa89aa
