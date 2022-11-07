import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import pyswmm
from swmm_api import SwmmInput

def open_inp(inp):
    ininp = SwmmInput.read_file(inp)
    return ininp

def update_inp(ininp,outinp):
    ininp.write_file(outinp,encoding='utf-8')