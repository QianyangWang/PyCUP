import math
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
from swmm_api import SwmmInput
from swmm_api.input_file.section_labels import SUBAREAS,INFILTRATION,CONDUITS,LANDUSES,POLLUTANTS,SUBCATCHMENTS



def write_n_imperv(params,ininp,mode=0):
    lis_subcatchments = list(ininp[SUBAREAS].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[SUBAREAS][subcatchment].N_Imperv = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[SUBAREAS][lis_subcatchments[i]].N_Imperv = params[i]
    return ininp

def write_n_perv(params,ininp,mode=0):
    lis_subcatchments = list(ininp[SUBAREAS].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[SUBAREAS][subcatchment].N_Perv = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[SUBAREAS][lis_subcatchments[i]].N_Perv = params[i]
    return ininp


def write_s_perv(params,ininp,mode=0):
    lis_subcatchments = list(ininp[SUBAREAS].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[SUBAREAS][subcatchment].S_Perv = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[SUBAREAS][lis_subcatchments[i]].S_Perv = params[i]
    return ininp


def write_s_imperv(params,ininp,mode=0):
    lis_subcatchments = list(ininp[SUBAREAS].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[SUBAREAS][subcatchment].S_Imperv = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[SUBAREAS][lis_subcatchments[i]].S_Imperv = params[i]
    return ininp


def write_p_zero(params,ininp,mode=0):
    lis_subcatchments = list(ininp[SUBAREAS].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[SUBAREAS][subcatchment].PctZero = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[SUBAREAS][lis_subcatchments[i]].PctZero = params[i]
    return ininp


def write_max_rate(params,ininp,mode=0):
    lis_subcatchments = list(ininp[INFILTRATION].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[INFILTRATION][subcatchment].MaxRate = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[INFILTRATION][lis_subcatchments[i]].MaxRate = params[i]
    return ininp

def write_min_rate(params,ininp,mode=0):
    lis_subcatchments = list(ininp[INFILTRATION].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[INFILTRATION][subcatchment].MinRate = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[INFILTRATION][lis_subcatchments[i]].MinRate = params[i]
    return ininp

def write_decay(params,ininp,mode=0):
    lis_subcatchments = list(ininp[INFILTRATION].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[INFILTRATION][subcatchment].Decay = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[INFILTRATION][lis_subcatchments[i]].Decay = params[i]
    return ininp

def write_ws_width(K,ininp,mode=0):
    lis_subcatchments = list(ininp[SUBCATCHMENTS].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[SUBCATCHMENTS][subcatchment].Width = K * math.sqrt(10000 * ininp[SUBCATCHMENTS][subcatchment].Area)
    else:
        for i in range(len(lis_subcatchments)):
            ininp[SUBCATCHMENTS][lis_subcatchments[i]].Width = K[i] * math.sqrt(10000 * ininp[SUBCATCHMENTS][lis_subcatchments[i]].Area)
    return ininp

def write_dry_time(params,ininp,mode=0):
    lis_subcatchments = list(ininp[INFILTRATION].keys())
    if mode == 0:
        for subcatchment in lis_subcatchments:
            ininp[INFILTRATION][subcatchment].DryTime = params
    else:
        for i in range(len(lis_subcatchments)):
            ininp[INFILTRATION][lis_subcatchments[i]].DryTime = params[i]
    return ininp

def write_conduit_roughness(params,ininp,mode=0):
    lis_conduits = list(ininp[CONDUITS].keys())
    if mode == 0:
        for conduit in lis_conduits:
            ininp[CONDUITS][conduit].Roughness = params
    else:
        for i in range(len(lis_conduits)):
            ininp[CONDUITS][lis_conduits[i]].Roughness = params[i]
    return ininp

