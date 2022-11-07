This is an example project that calibrates an SWMM model using multi-processing EOGWO
*****PLEASE EXECUTE THIS PROJECT UNDER THE ADMINISTRATOR'S RIGHT IN CASE OF POPUP WINDOWS*****
1. The example SWMM model can be downloaded from https://www.epa.gov/water-research/storm-water-management-model-swmm#downloads
2. The calibration data is a generated example data series (not real data).
3. run the Calibrate.py for model calibration, run the PostProcessing.py for uncertainty analysis.
Note: The true calibration process does not necessarily need you to put the model executable into the folder.
          The executable here is for the convenience that you can directly run the example scripts.

Example dependencies:
1. PyCUP
2. NumPy
3. SciPy
4. matplotlib
5. multiprocessing
6. pyswmm (tested on version 1.1.1)
7. swmm-api (0.2.0.18.3)
8. swmm-toolkit (0.8.2)
9. win32api
10. win32event
11. win32process
12. win32con
13. win32com
14. Anaconda 3 environment (tested)

