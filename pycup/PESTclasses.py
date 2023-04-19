import re
import numpy as np
import warnings
import prettytable as pt
import os


class PESTparams:

    """
    *Introduction:
    A top class for storing parameter information

    *Browse the information:
    PESTparams[parameter group name] to visit a parameter group
    PESTparams[parameter group name][parameter name] to visit a specific parameter
    """

    def __init__(self,gp_info,param_info):
        self._cursor = -1
        self.gp_info = gp_info
        self.param_info = param_info
        self.param_gps = {}
        self.scan_gp_info()
        self.scan_param_info()
        self.length = len(self.param_gps.keys())

    def scan_gp_info(self):
        for i in self.gp_info:
            settings = list(filter(None,i.split(" ")))
            name = settings[0]
            type = settings[1]
            derinc = float(settings[2])
            new_gp = PESTparamgp(name,type,derinc)
            self.param_gps[name] = new_gp
    def scan_param_info(self):
        for i in self.param_info:
            name,partype,_,ini,lb,ub,gp,_,_,_ = list(filter(None,i.split(" ")))
            self.param_gps[gp].add_par(name,partype,float(ini),float(lb),float(ub),gp)

    def list_params(self):
        pars = []
        for gp in self:
            for par in gp:
                pars.append(par)
        return pars

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.param_gps[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._cursor + 1 < self.length:
            self._cursor += 1
            return list(self.param_gps.values())[self._cursor]
        else:
            self._cursor = -1
            raise StopIteration

    def __repr__(self):
        return "PESTparams({})".format(self.param_gps)


class PESTparamgp:

    # PEST parameter group management class

    def __init__(self,name,type,derinc):
        self._cursor = -1
        self.name = name
        self.pars = {}
        self.length = len(self.pars.keys())
        self.type = type
        self.derinc = derinc

    def add_par(self,name,type,ini,lb,ub,gp):
        par = PESTpar(name,type,ini,lb,ub,gp,self.type,self.derinc)
        self.pars[name]=par
        self.length = len(self.pars.keys())

    def get_lb(self):
        lb = np.min([i.lb for i in self.pars])
        self.lb = lb
        return lb

    def get_ub(self):
        ub = np.max([i.ub for i in self.pars])
        self.ub = ub
        return ub

    def __len__(self):
        return self.length

    def __repr__(self):
        return "PESTparamgp(name='{}',pars='{}')".format(self.name,self.pars)

    def __getitem__(self, item):
        return self.pars[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._cursor + 1 < self.length:
            self._cursor += 1
            return list(self.pars.values())[self._cursor]
        else:
            self._cursor = -1
            raise StopIteration


class PESTpar:

    # PEST parameter setting management class

    def __init__(self,name,type,ini,lb,ub,gp,gptype,gpderinc):
        self.name = name
        self.type = type
        self.gptype = gptype
        self.gpderinc = gpderinc
        self.ini = ini
        self.lb = lb
        self.ub = ub
        self.gp =gp
        if self.type == "log":
            warnings.warn("The parameter {} is a log-transformed parameter, the log-transform operation will not be considered in PyCUP.".format(self.name))

    def __repr__(self):
        return "PESTpar(name='{}',type='{}',ini={},lb={},ub={},gp='{}')".format(self.name,self.type,self.ini,self.lb,self.ub,self.gp)


class PESTobs:

    # observation data management class for storing observation groups

    def __init__(self,obs_gp_content,obs_data_content):
        self._cursor = -1
        self.gp_info = obs_gp_content
        self.data_info = obs_data_content
        self.obs_gps = {}
        self.scan_gp_info()
        self.length = len(self.obs_gps.keys())
        self.scan_series_info()

    def scan_gp_info(self):
        for i in self.gp_info:
            obs_gp = i.strip("\n")
            self.obs_gps[obs_gp] = PESTobsgp(obs_gp)

    def scan_series_info(self):
        for i in self.data_info:
            idx,val,weight,gp = list(filter(None,i.split(" ")))
            gp = gp.strip("\n")
            self.obs_gps[gp].add_data(idx,weight,val)
        for g in self:
            g.to_array()

    def __getitem__(self, item):
        return self.obs_gps[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._cursor + 1 < self.length:
            self._cursor += 1
            return list(self.obs_gps.values())[self._cursor]
        else:
            self._cursor = -1
            raise StopIteration

    def __repr__(self):
        return "PESTobs({})".format(self.obs_gps)


class PESTobsgp:

    # observation group management class for storing observation data, index, and weights

    def __init__(self,name):
        self.name = name
        self.data = []
        self.weights = []
        self.index = []
        self.dic_data = {}
        self.dic_weight = {}

    def add_data(self,idx,wei,n):
        self.index.append(idx)
        self.weights.append(float(wei))
        self.data.append(float(n))
        self.dic_data[idx]=float(n)
        self.dic_weight[idx]=float(wei)

    def __getitem__(self, item):
        if isinstance(item,int):
            return self.data[item]
        elif isinstance(item,str):
            return self.dic_data[item],self.dic_weight[item]

    def __repr__(self):
        return "PESTobsgroup(name='{}',data,weights,index)".format(self.name)

    def to_array(self):
        self.data = np.array(self.data)
        self.weights = np.array(self.weights)

    def check_weights(self):
        if len(set(self.weights)) == 1:
            return WeightedAverage
        elif np.sum(self.weights==0) + np.sum(self.weights==1) == len(self.weights):
            return DorpOut
        elif len(set(self.weights)) == 2 and np.any(self.weights==0):
            return WeightedAndDropOut
        else:
            raise RuntimeError("When the customize evaluation metric has been used, PyCUP only supports "
                               "the weight of observations like:\n"
                               "Type1: The series which only contains 0 and 1 for the dropout of the selected observations.\n"
                               "Type2: The series which is uniform for a weighted average calculation, such as [0.3,0.3,0.3].\n")

class PESTbatch:

    # PEST commandline storing class

    def __init__(self,workspace,batch_content):
        self.workspace = workspace
        self.batch_content = batch_content
        self.cmds = []
        ori_cwd = os.getcwd()
        os.chdir(self.workspace)
        self.scan_cmds()
        os.chdir(ori_cwd)

    def scan_cmds(self):
        cmds = self.batch_content
        for fn in cmds:
            fn = fn.strip("\n")
            if ".bat" in fn and len(list(filter(None,fn.split(" "))))==1:
                self.recur_batch(fn)
            else:
                self.cmds.append(fn)

    def recur_batch(self,fn):
        with open(fn) as f:
            lines = f.readlines()
            for r in lines:
                content = r.strip("\n")
                if ".bat" in content and len(list(filter(None,content.split(" "))))==1:
                    self.recur_batch(content)
                else:
                    self.cmds.append(content)


class PESTio:

    # PEST parameter/result input/output information storing class

    def __init__(self,workspace,input_content):
        self.workspace = workspace
        self.input_content = input_content
        self.fparam = {}
        self.fout = {}
        self.scan_input()

    def scan_input(self):
        for fpair in self.input_content:
            fs = list(filter(None,fpair.split(" ")))
            fs[-1] = fs[-1].strip("\n")
            with open(self.workspace + "\\" + fs[0]) as f:
                ftype = f.readline()
                if "pif" in ftype:
                    self.fout[fs[0]]=fs[1]
                elif "ptf" in ftype:
                    self.fparam[fs[0]] = fs[1]

    def list_tpl(self):
        return list(self.fparam.keys())

    def list_fparam(self):
        return list(self.fparam.values())

    def list_fins(self):
        return list(self.fout.keys())

    def list_fout(self):
        return list(self.fout.values())


class PESTsimpair:

    # PEST simulation result management class

    def __init__(self):
        self.idx = []
        self.res = []
        self.dict_res = {}

    def add_data(self,idx,res):
        self.idx.extend(idx)
        self.res.extend(res)
        for i,j in zip(idx,res):
            self.dict_res[i]=j

    def __getitem__(self, item):
        return self.dict_res[item]


class PESTsimgp:

    # PEST simulation result-observation data management class

    def __init__(self,gpname):
        self.gpname = gpname
        self.obs = []
        self.res = []
        self.wei = []
        self.idx = []

    def add_data(self,idx,obs,res,weight):
        self.obs.append(obs)
        self.res.append(res)
        self.wei.append(weight)
        self.idx.append(idx)

    """"""
    def __repr__(self):
        tbl = pt.PrettyTable()
        tbl.title = "Sim. Result Group {} Info.".format(self.gpname)
        tbl.field_names = ["Index", "Observed Data", "Simulation Result", "Weight"]
        for i,o,s,w in zip(self.idx,self.obs,self.res,self.wei):
            tbl.add_row([i,str(o),str(s),str(w)])
        print(tbl)
        return ""


    def check_weights(self):
        wei = np.array(self.wei)
        if len(set(wei)) == 1:
            return WeightedAverage
        elif np.sum(wei==0) + np.sum(wei==1) == len(wei):
            return DorpOut
        elif len(set(wei)) == 2 and np.any(wei==0):
            return WeightedAndDropOut



class WeightType:
    pass

class WeightedAverage(WeightType):
    pass

class DorpOut(WeightType):
    pass

class WeightedAndDropOut(WeightedAverage):
    pass


class PESTinsReader:

    def __init__(self,inspath):
        with open(inspath,"r") as f:
            lines = f.readlines()
        ins_content = [i.strip("\n") for i in lines]
        commands = []
        headline=True
        for i in ins_content:
            if not i.isspace():
                if not headline:
                    commands.append(i)
                else:
                    self.dlmt = i.split(" ")[-1]
                    headline =False
        # after recognizing the delimiter
        self.commands = []
        for i in commands:
            self._scan_line(i)
        self.cmd_obj = []
        for vb in self.commands:
            obj = self.to_cmd_obj(vb)
            self.cmd_obj.append(obj)

    def _scan_line(self,line):
        verbs = []
        v = ""
        idpdt = False
        in_delimiter = False
        ndlmt = 0
        for id,i in enumerate(line):
            if not i.isspace():
                idpdt = True
                v += i
                if i == self.dlmt:
                    ndlmt += 1
                    if ndlmt == 2:
                        ndlmt = 0
                        in_delimiter = False
                    elif ndlmt == 1:
                        in_delimiter = True
                if id == len(line)-1:
                    verbs.append(v)
            else:
                if not in_delimiter:
                    if idpdt:
                        verbs.append(v)
                        idpdt = False
                        v = ""
                else:
                    v += i
        self.commands.extend(verbs)

    def to_cmd_obj(self,verb):
        if re.match("l[0-9]+",verb):
            return PESTLineAdvance(verb)
        elif re.match("{}.*?{}".format(self.dlmt,self.dlmt),verb):
            return PESTDelimiter(verb,self.dlmt)
        elif verb == "w":
            return PESTwhitespace(verb)
        elif re.match("!.*?!",verb):
            return PESTNoneFixedData(verb)
        elif re.match("\[.*?][0-9]+:[0-9]+",verb):
            return PESTFixedData(verb)
        elif re.match("\(.*?\)[0-9]+:[0-9]+",verb):
            return PESTSemiFixedData
        elif verb == "&":
            return PESTcontinue(verb)
        elif re.match("t[0-9]+",verb):
            return PESTtab(verb)
        else:
            raise NotImplementedError("The command {} is currently not supproted".format(verb))


class PESTinscmd:

    def __init__(self,verb):
        self.cmd = verb

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"],kwargs["linecursor"]
        return rowcursor,linecursor


class PESTLineAdvance(PESTinscmd):

    def __init__(self,verb):
        super().__init__(verb)

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"], kwargs["linecursor"]
        rows = int(self.cmd[1:])
        rowcursor += rows
        linecursor = 0
        return rowcursor,linecursor


class PESTNoneFixedData(PESTinscmd):

    def __init__(self,verb):
        super().__init__(verb)
        self.dataname = re.findall("!(.*?)!", verb)[0]

    def return_data(self,content,rowcursor,linecursor):
        res = re.findall("\s*?(-?\d+\.?\d*?E?-?\d*?)\s*?", content)[0]
        self.strdata = res
        rowcursor, linecursor = self.update_cursor(rowcursor=rowcursor,linecursor=linecursor)
        if res.isdigit():
            return int(res),rowcursor,linecursor
        else:
            return float(res),rowcursor,linecursor

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"], kwargs["linecursor"]
        linecursor += len(self.strdata)
        return rowcursor,linecursor


class PESTSemiFixedData(PESTinscmd):

    def __init__(self,verb):
        super().__init__(verb)
        raw_markers = re.findall("\(.*?\)(\d+:\d+)", verb)
        markers = raw_markers[0].split(":")
        self.start = int(markers[0])
        self.end = int(markers[1])
        self.dataname = re.findall("\((.*?)\)\d+:\d+", verb)[0]

    def return_data(self,content,rowcursor,linecursor):
        res = re.findall("\s*?(-?\d+\.?\d*?E?-?\d*?)\s*?", content)[0]
        self.strdata = res
        rowcursor, linecursor = self.update_cursor(rowcursor=rowcursor,linecursor=linecursor)
        if res.isdigit():
            return int(res),rowcursor,linecursor
        else:
            return float(res),rowcursor,linecursor

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"], kwargs["linecursor"]
        linecursor = self.end
        return rowcursor,linecursor


class PESTFixedData(PESTinscmd):

    def __init__(self,verb):
        super().__init__(verb)
        raw_markers = re.findall("\[.*?](\d+:\d+)", verb)
        markers = raw_markers[0].split(":")
        self.start = int(markers[0])
        self.end = int(markers[1])
        self.dataname = re.findall("\[(.*?)]\d+:\d+", verb)[0]

    def return_data(self,content,rowcursor,linecursor):
        res = re.findall("\s*?(-?\d+\.?\d*?E?-?\d*?)\s*?", content)[0]
        self.strdata = res
        rowcursor, linecursor = self.update_cursor(rowcursor=rowcursor,linecursor=linecursor)
        if res.isdigit():
            return int(res),rowcursor,linecursor
        else:
            return float(res),rowcursor,linecursor

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"], kwargs["linecursor"]
        linecursor = self.end
        return rowcursor,linecursor


class PESTDelimiter(PESTinscmd):

    def __init__(self,verb,dlmt):
        super().__init__(verb)
        self.keyword = re.findall("{}(.*?){}".format(dlmt,dlmt),verb)

    def update_cursor(self,**kwargs):
        rowcursor, linecursor,contents = kwargs["rowcursor"], kwargs["linecursor"],kwargs["contents"]
        rowcursor, linecursor = self.skip_to(rowcursor,linecursor,contents)
        return rowcursor,linecursor

    def skip_to(self,rowcursor, linecursor,contents):
        idx = contents[rowcursor].find(self.keyword,beg=linecursor,end=len(contents[rowcursor]))
        if idx == -1:
            rowcursor += 1
            linecursor = 0
            rowcursor, linecursor = self.skip_to(rowcursor,linecursor,contents)
        else:
            linecursor = idx+ len(self.keyword)
        return rowcursor,linecursor


class PESTwhitespace(PESTinscmd):

    def __init__(self,verb):
        super().__init__(verb)

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"], kwargs["linecursor"]
        linecursor += 1
        return rowcursor,linecursor


class PESTtab(PESTinscmd):

    def __init__(self,verb):
        super().__init__(verb)
        self.count = int(verb[1:])

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"], kwargs["linecursor"]
        linecursor += self.count
        return rowcursor,linecursor


class PESTcontinue(PESTinscmd):
    def __init__(self,verb):
        super().__init__(verb)

    def update_cursor(self,**kwargs):
        rowcursor, linecursor = kwargs["rowcursor"], kwargs["linecursor"]
        return rowcursor,linecursor


class PESTresultReader:

    def __init__(self,respath,insreader):
        with open(respath,"r") as f:
            lines = f.readlines()
        self.raw_results = lines
        self.rowcursor = -1
        self.linecursor = 0
        self.insreader = insreader
        self.data_name = []
        self.data = []
        self.read_result()


    def read_result(self):
        for cmd in self.insreader.cmd_obj:
            if isinstance(cmd,PESTDelimiter):
                self.rowcursor,self.linecursor = cmd.update_cursor(rowcursor=self.rowcursor,linecursor=self.linecursor,contents=self.raw_results)
            elif isinstance(cmd,PESTwhitespace) or isinstance(cmd,PESTtab) or isinstance(cmd,PESTLineAdvance) or isinstance(cmd,PESTcontinue):
                self.rowcursor, self.linecursor = cmd.update_cursor(rowcursor=self.rowcursor, linecursor=self.linecursor)
            # data object
            else:
                data,self.rowcursor, self.linecursor = cmd.return_data(self.raw_results[self.rowcursor][self.linecursor:],self.rowcursor,self.linecursor)
                name = cmd.dataname
                if name != "dum":
                    self.data.append(data)
                    self.data_name.append(name)





if __name__ == "__main__":
    ins = PESTinsReader(r"D:\PESTEXAMPLES\pestpp_v5_data_release_barf\mf6_freyberg\freyberg6.lst.ins")
    print(ins.cmd_obj)



