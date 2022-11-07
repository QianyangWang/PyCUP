import win32api
import win32event
import win32process
import win32con
from win32com.shell.shell import ShellExecuteEx
from win32com.shell import shellcon

def callexe(f_model, f_sim,f_rpt,fout):
    param = " ".join([f_sim,f_rpt,fout])
    process_info = ShellExecuteEx(nShow=win32con.SW_HIDE,
                                  fMask=shellcon.SEE_MASK_NOCLOSEPROCESS,
                                  lpVerb='runas',
                                  lpFile=f_model,
                                  lpParameters=param)

    win32event.WaitForSingleObject(process_info['hProcess'], -1)
    ret = win32process.GetExitCodeProcess(process_info['hProcess'])


    return  ret

