# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:08:12 2020

@author: Juan Felipe
"""
from logging import raiseExceptions
import os
import glob
import shutil

home= 'C:\\Users\\vargasju\\PhD\\experiments\\2p\\'
os.chdir(home)
experiment=['T4T5_STRF_glucla_rescues']#['test_L3_newfilters'] #'Mi1_glucl_mi1_suff'
type_experiment=['STRF']#['']##'FFF_data'
experimenter=['jv','jv','jv']

for idx,exp in enumerate(experiment):
    os.chdir(home)
    print(exp)
    search_str= home+exp+'\\processed\\'+'*'+ experimenter[idx]+'*'+'\\**\\'+'*_'+ experimenter[idx]+'_*'
    list_f=glob.glob(search_str)
    target=home+exp+'\\processed\\files' #\\files' # '+type_experiment[idx] +'\\
    os.chdir(target)
    try:
        shutil.rmtree(target+'\\old_run')
    except WindowsError:
        pass
    finally:
        pass   
    os.mkdir('old_run')
    if len(list_f)>0:
        for old in glob.glob(target+'\\*_jv_*'):         
            shutil.move(old,target+'\\old_run')
        for file in list_f:
            shutil.move(file,target)
    else:
        print ('no new pickle files. break to prevent mistake-erasing')
        


