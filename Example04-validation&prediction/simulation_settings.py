import math
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import pyswmm
from swmm_api import SwmmInput
from swmm_api.input_file.section_labels import OPTIONS
import datetime


def write_routing_step(ininp,H,M,S):
    ininp[OPTIONS]["ROUTING_STEP"] = "{}:{}:{}".format(str(H).zfill(2),str(M).zfill(2),str(S).zfill(2))
    return ininp

def write_simulation_period(ininp,start,end,report=True):

    M_start = start.month
    D_start = start.day
    Y_start = start.year
    H_start = start.hour
    MIN_start = start.minute
    S_start = start.second

    M_end = end.month
    D_end  = end.day
    Y_end  = end.year
    H_end = end.hour
    MIN_end = end.minute
    S_end = end.second

    ininp[OPTIONS]["START_DATE"] = "{}/{}/{}".format(str(M_start).zfill(2),str(D_start).zfill(2),str(Y_start))
    ininp[OPTIONS]["START_TIME"] = "{}:{}:{}".format(str(H_start).zfill(2), str(MIN_start).zfill(2), str(S_start).zfill(2))
    ininp[OPTIONS]["END_DATE"] = "{}/{}/{}".format(str(M_end).zfill(2), str(D_end).zfill(2), str(Y_end))
    ininp[OPTIONS]["END_TIME"] = "{}:{}:{}".format(str(H_end).zfill(2), str(MIN_end).zfill(2), str(S_end).zfill(2))

    if report == True:
        ininp[OPTIONS]["REPORT_START_DATE"] = "{}/{}/{}".format(str(M_start).zfill(2), str(D_start).zfill(2), str(Y_start))
        ininp[OPTIONS]["REPORT_START_TIME"] = "{}:{}:{}".format(str(H_start).zfill(2), str(MIN_start).zfill(2),str(S_start).zfill(2))
        # The REPORT_END_DATE and REPORT_END_TIME is not used in the version SWMM 5.1.13
        #ininp[OPTIONS]["REPORT_END_DATE"] = "{}/{}/{}".format(str(M_end).zfill(2), str(D_end).zfill(2), str(Y_end))
        #ininp[OPTIONS]["REPORT_END_TIME"] = "{}:{}:{}".format(str(H_end).zfill(2), str(MIN_end).zfill(2), str(S_end).zfill(2))
    return ininp


def write_outfall_free_boundary(ininp,excludes=None):

    for i in ininp["OUTFALLS"]:

        if excludes:
            if i not in excludes:
                ininp["OUTFALLS"][i].Type = "FREE"
                ininp["OUTFALLS"][i].Data = math.nan
        else:
            ininp["OUTFALLS"][i].Type = "FREE"
            ininp["OUTFALLS"][i].Data = math.nan

    return ininp

def write_outfall_ts_boundary(ininp,data,excludes=None):
    for i in ininp["OUTFALLS"]:
        if excludes:
            if i not in excludes:
                ininp["OUTFALLS"][i].Type = "TIMESERIES"
                ininp["OUTFALLS"][i].Data = data[i]
        else:
            ininp["OUTFALLS"][i].Type = "TIMESERIES"
            ininp["OUTFALLS"][i].Data = data[i]
    return ininp

def read_outfall_elev(ininp,excludes=None):
    data = {}
    for i in ininp["OUTFALLS"]:
        if excludes:
            if i not in excludes:
                elev = ininp["OUTFALLS"][i].Elevation
                data[i] = elev
        else:
            elev = ininp["OUTFALLS"][i].Elevation
            data[i] = elev
    return data

