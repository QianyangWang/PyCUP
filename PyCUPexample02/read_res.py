import os
import numpy as np
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
from swmm.toolkit.shared_enum import SystemAttribute,NodeAttribute
import datetime
from pyswmm import Output

def get_date_ranges(start, end, step):
    list_days = []
    while start <= end:
        list_days.append(start)
        start += datetime.timedelta(seconds=step)
    return list_days


def read_swmm_result(path, nodes, start, end, step):
    if type(start) == str:
        try:
            start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            end = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        except:
            start = datetime.datetime.strptime(start, "%Y/%m/%d %H:%M:%S")
            end = datetime.datetime.strptime(end, "%Y/%m/%d %H:%M:%S")

    time_stamps = get_date_ranges(start, end, step)
    res = []
    with Output(path) as out:
        for node in nodes:
            ts = out.node_series(node, NodeAttribute.TOTAL_INFLOW, start, end)

            node_series = []
            for s in time_stamps[0:-1]:
                node_value = ts[s]
                node_series.append(node_value)
            node_series = np.array(node_series)
            res.append(node_series)
    actual_stamps = time_stamps[0:-1]
    return res, actual_stamps