import numpy as np
import pandas as pd
UseResObject = False

DataBaseDirs = []




class SimulationResult:

    def __init__(self):
        self.stations = {}

    def add_station(self,station_name):
        station = StationObject(station_name)
        self.stations[station_name] = station
        return self

    def remove_station(self,station_name):
        self.stations.pop(station_name)

    def list_stations(self):
        s = ",".join(self.stations.keys())
        print(s)
        return s

    def keys(self):
        return self.stations.keys()

    def values(self):
        return self.stations.values()

    def __getitem__(self, item):
        return self.stations[item]

    def __len__(self):
        return len(self.stations)

    def __add__(self,other_results):
        if not isinstance(other_results,SimulationResult):
            raise TypeError("The added object should be a SimulationResult object.")
        new_result = SimulationResult()
        new_result.stations.update(self.stations)
        new_result.stations.update(other_results.stations)
        return new_result
    def __repr__(self):
        return "SimulationResult object => indexing: [str(station name)][str(event name)]"


class StationObject:

    def __init__(self,station_name):
        self.name = station_name
        self.events = {}

    def add_event(self,event_name,event_data):
        event = EventData(event_name,event_data)
        self.events[event_name] = event

    def list_events(self):
        s = ",".join(self.events.keys())
        print(s)
        return s

    def keys(self):
        return self.events.keys()

    def values(self):
        return self.events.values()

    def __getitem__(self, item):
        return self.events[item].data

    def __repr__(self):
        return "StationObject(name='{}',events={})".format(self.name,self.events)

    def __len__(self):
        return len(self.events)


class EventData:

    def __init__(self,event_name,event_data):
        self.name = event_name
        self.data = event_data
        self._check_data()
        self.shape = event_data.shape

    def _check_data(self):
        self.data = np.array(self.data).reshape(1,-1)

    def __repr__(self):
        return "EventData(name='{}',data=[{}...{}])".format(self.name,self.data[:,0],self.data[:,-1])

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)



class DataBaseWriter:

    def __init__(self,res_package,params=None,fitness=None,dir="SimulationResults",fmt="xlsx",pop=None,dim=None,lb=None,ub=None,max_iter=None,eobl_fraction=None):
        self.dir = dir
        self.fmt = fmt
        self.path = dir+"."+fmt
        self.pop = pop
        self.dim=dim
        self.lb=lb
        self.ub=ub
        self.max_iter=max_iter
        self.eobl_fraction = eobl_fraction
        self.res_pkg = res_package
        self.params = params
        self.fitness=fitness
        self.method_info = res_package.method

    def write2DB(self):
        if self.fmt == "xlsx":
            self._write2excel(self.res_pkg.data,self.params,self.fitness)
        elif self.fmt == "hdf5" or self.fmt == "h5":
            self._write2hdf5(self.res_pkg.data,self.params,self.fitness)

    def _write2excel(self,results,params=None,fitness=None):
        writer = pd.ExcelWriter(self.path)
        if self.method_info == "Algorithm":
            algorithm_sheets = self._algorithm_res2df(results)
            for st in algorithm_sheets.keys():
                algorithm_sheets[st].to_excel(writer, sheet_name=st, encoding="utf-8", index=False)
            if params is not None:
                df_param = self._algorithm_params2df(params)
                df_param.to_excel(writer, sheet_name="params", encoding="utf-8", index=False)
            if fitness is not None:
                df_fitness = self._algorithm_fitness2df(fitness)
                df_fitness.to_excel(writer, sheet_name="obj fun value", encoding="utf-8", index=False)
        else:
            # For GLUE, validation/prediction results
            glue_sheets = self._general_res2df(results)
            for st in glue_sheets.keys():
                glue_sheets[st].to_excel(writer, sheet_name=st, encoding="utf-8", index=False)
            if params is not None:
                df_param = self._general_params2df(params)
                df_param.to_excel(writer, sheet_name="params", encoding="utf-8", index=False)
            if fitness is not None:
                df_fitness = self._general_fitness2df(fitness)
                df_fitness.to_excel(writer, sheet_name="obj fun value", encoding="utf-8", index=False)
        info_page = self._info_page2df()
        info_page.to_excel(writer,sheet_name="info page",encoding="utf-8",index=False)
        writer.save()

    def _write2hdf5(self,results,params=None,fitness=None):
        writer = pd.HDFStore(self.path,mode="w",complevel=6)
        if self.method_info == "Algorithm":
            algorithm_sheets = self._algorithm_res2df(results)
            for st in algorithm_sheets.keys():
                writer.append(key=st,value=algorithm_sheets[st])
            if params is not None:
                df_param = self._algorithm_params2df(params)
                writer.append(key="params",value=df_param)
            if fitness is not None:
                df_fitness = self._algorithm_fitness2df(fitness)
                writer.append(key="obj fun value", value=df_fitness)
        else:
            # For GLUE, validation/prediction results
            glue_sheets = self._general_res2df(results)
            for st in glue_sheets.keys():
                writer.append(key=st,value=glue_sheets[st])
            if params is not None:
                df_param = self._general_params2df(params)
                writer.append(key="params", value=df_param)
            if fitness is not None:
                df_fitness = self._general_fitness2df(fitness)
                writer.append(key="obj fun value", value=df_fitness)
        info_page = self._info_page2df()
        writer.append(key="info page",value=info_page)
        writer.close()


    def _algorithm_res2df(self,results):
        # scan stations from the first row of the simulation result
        stations = results[0][0].keys()
        # 1 station 1 sheet
        sheets = {}
        for station in stations:
            # 1 iteration 1 column section in db
            l_iteration_data = []

            # scan events
            events = results[0][0][station].keys()
            event_lengths = [results[0][0][station][e].shape[-1] for e in events]
            event_labels = []
            for idx_e, e in enumerate(events):
                for l in range(event_lengths[idx_e]):
                    event_labels.append(e)
            event_labels = pd.DataFrame(event_labels, columns=["event name"])

            for n_iter,iteration in enumerate(results):
                l_sample_data = []
                # num of rows/samples in each iteration (1 sample 1 column in db)
                for r in range(len(results[n_iter])):
                    l_event_data = []
                    for e in events:
                        event_data = np.array(iteration[r][station][e]).reshape(-1, 1)
                        l_event_data.append(event_data)
                    l_event_data = np.concatenate(l_event_data, axis=0)
                    l_sample_data.append(l_event_data)
                l_sample_data = np.concatenate(l_sample_data, axis=1)
                l_iteration_data.append(l_sample_data)
            station_data = np.concatenate(l_iteration_data, axis=1)

            sample_labels = []
            for i in range(len(results)):
                for j in range(results[i].shape[0]):
                    sample_labels.append("s{}_{}".format(i, j))
            station_data = pd.DataFrame(station_data, columns=sample_labels)
            sheet_data = pd.concat([event_labels, station_data], axis=1)
            sheets[station] = sheet_data
        return sheets

    def _general_res2df(self,results):
        # scan stations from the first row of the simulation result
        stations = results[0].keys()
        sheets = {}
        # 1 station 1 sheet
        for station in stations:
            # 1 iteration 1 column section in db
            # for glue only 1 iteration
            l_iteration_data = []

            # scan events
            events = results[0][station].keys()
            event_lengths = [results[0][station][e].shape[-1] for e in events]
            event_labels = []
            for idx_e, e in enumerate(events):
                for l in range(event_lengths[idx_e]):
                    event_labels.append(e)
            event_labels = pd.DataFrame(event_labels, columns=["event name"])

            l_sample_data = []
            # num of rows/samples in each iteration (1 sample 1 column in db)
            for r in range(len(results)):
                l_event_data = []
                for e in events:
                    event_data = np.array(results[r][station][e]).reshape(-1, 1)
                    l_event_data.append(event_data)
                l_event_data = np.concatenate(l_event_data, axis=0)
                l_sample_data.append(l_event_data)
            l_sample_data = np.concatenate(l_sample_data, axis=1)
            l_iteration_data.append(l_sample_data)
            station_data = np.concatenate(l_iteration_data, axis=1)

            sample_labels = []
            for i in range(len(results)):
                sample_labels.append("s_{}".format(i))
            station_data = pd.DataFrame(station_data, columns=sample_labels)
            sheet_data = pd.concat([event_labels, station_data], axis=1)
            sheets[station]=sheet_data
        return sheets

    def _algorithm_params2df(self,params):
        # scan param shape
        n_params = params[0].shape[1]
        sample_labels = []
        n_pop = [i.shape[0] for i in params]
        for i in range(len(params)):
            for j in range(n_pop[i]):
                sample_labels.append("s{}_{}".format(i,j))
        sample_labels = pd.DataFrame(sample_labels,columns=["sample id"])
        param_labels = ["x_{}".format(i) for i in range(n_params)]
        all_samples = np.concatenate(params,axis=0)
        all_samples = pd.DataFrame(all_samples,columns=param_labels)
        param_data = pd.concat([sample_labels,all_samples],axis=1)

        return param_data


    def _general_params2df(self,params):
        n_params = params.shape[1]
        n_pop = params.shape[0]
        sample_labels = ["s_{}".format(i) for i in range(n_pop)]
        sample_labels = pd.DataFrame(sample_labels, columns=["sample id"])
        param_labels = ["x_{}".format(i) for i in range(n_params)]
        all_samples = pd.DataFrame(params,columns=param_labels)
        param_data = pd.concat([sample_labels,all_samples],axis=1)

        return param_data

    def _algorithm_fitness2df(self,fitness):
        n_pop = [i.shape[0] for i in fitness]
        n_iter = len(fitness)
        n_dim = fitness[0].shape[1]
        fitness = np.concatenate(fitness,axis=0)
        sample_labels=[]
        for i in range(n_iter):
            for j in range(n_pop[i]):
                sample_labels.append("s{}_{}".format(i,j))
        sample_labels = pd.DataFrame(sample_labels, columns=["sample id"])
        fitness = pd.DataFrame(fitness,columns=["fitness_{}".format(i) for i in range(n_dim)])
        sheet_data = pd.concat([sample_labels,fitness],axis=1)

        return sheet_data


    def _general_fitness2df(self,fitness):
        n_pop = fitness.shape[0]
        sample_labels = ["s_{}".format(i) for i in range(n_pop)]
        sample_labels = pd.DataFrame(sample_labels, columns=["sample id"])
        fitness = pd.DataFrame(fitness,columns=["fitness_{}".format(i) for i in range(fitness.shape[1])])
        sheet_data = pd.concat([sample_labels,fitness],axis=1)

        return sheet_data

    def _info_page2df(self):
        info = []
        info_labels = []
        info.append(self.method_info)
        info_labels.append("method_info")
        if self.pop is not None:
            info.append(self.pop)
            info_labels.append("pop")
        if self.dim is not None:
            info.append(self.dim)
            info_labels.append("dim")
        if self.max_iter is not None:
            info.append(self.max_iter)
            info_labels.append("max_iter")
        if self.eobl_fraction is not None:
            info.append(self.eobl_fraction)
            info_labels.append("EOBL fraction")
        info = np.array(info).reshape(1,-1)
        info_page = pd.DataFrame(info,columns=info_labels)
        if self.lb is not None:
            lbdf = pd.DataFrame(self.lb.reshape(-1,1),columns=["lb"])
            info_page = pd.concat([info_page,lbdf],axis=1)
        if self.ub is not None:
            ubdf = pd.DataFrame(self.ub.reshape(-1,1),columns=["ub"])
            pd.concat([info_page,ubdf],axis=1)
            info_page = pd.concat([info_page,ubdf],axis=1)

        return info_page


class ResultDataPackage:
    def __init__(self,l_result,method_info):
        self.data = l_result
        self.method = method_info

    def __getitem__(self, item):
        if type(item) == str:
            if self.method == "Algorithm":
                station_objects = []
                for i in self.data:
                    iter_data = [j[item] for j in i]
                    station_objects.append(iter_data)
                stations = StationDataPackage(station_objects,self.method)
                return stations
            else:
                stations = StationDataPackage([i[item] for i in self.data],self.method)
                return stations
        elif type(item) == int:
            return self.data[item]
        else:
            raise KeyError("The index should be the station name or the sample index.")


    def list_stations(self):
        if self.method == "Algorithm":
            stations = self.data[0][0].keys()
            return stations
        else:
            stations = self.data[0].keys()
            return stations

    def __repr__(self):
        return "ResultDataPackage object => indexing: [str(station name)] or [int(sample index)]"


class StationDataPackage:

    def __init__(self,station_objects,method_info):
        self.data = station_objects
        self.method = method_info

    def __getitem__(self, item):

        if self.method == "Algorithm":
            event_objects = []
            for i in self.data:
                iter_data = np.concatenate([j[item].reshape(1,-1) for j in i],axis=0)
                event_objects.append(iter_data)
            return event_objects
        else:
            # For GLUE or validation & prediction
            res_data = np.concatenate([row[item].reshape(1, -1) for row in self.data], axis=0)
            return res_data

    def list_events(self):

        if self.method == "Algorithm":
            events = self.data[0][0].keys()
            return events
        else:
            events = self.data[0].keys()
            return events

    def __repr__(self):
        return "StationDataPackage object => indexing: [str(event name)]"

