# -*- coding: utf-8 -*-
"""
Created on 22.04.21 

@author: Jfelipeco. plots T4T5 traces

"""

import numpy as np
import glob
import pandas as pd
import sima
import pathlib
import matplotlib.pyplot as plt
import os
from skimage import io
import copy
import cPickle
#from preprocessing import produce_Tseries_list # eventually, transfer this function to process_mov_core script
import sys
#code_path = r'D:\progresources\2panalysis\Helpers'
code_path= r'U:\Dokumente\GitHub\IIIcurrentIII2p_analysis_development\2panalysis\Helpers' #Juan desktop
sys.path.insert(0, code_path) 

from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize,getMicRelativeTime,getrastersPerFrame
import ROI_mod
import core_functions
from core_functions import saveWorkspace
import process_mov_core as pmc

home_path='C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan
experiment='t4t5_freqtunning_8dir_Mi1 glucl overexpression'#['t4t5_shits_p9silencing']#['T4T5_r57c10gal4_panneuronal_silencing']#['t4t5_microscope_comparison']# ['Mi1_Glucl_acute_rescue']#['Miris_data_piece']
experimenter=['*_jv_*']#['*_jv_*']

data_dir = home_path+experiment+'\\'+'processed\\results\\edges\\files'

datasets_to_load = os.listdir(data_dir)

for dataset in datasets_to_load:

    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    exp_id=curr_rois[0].id
    analysis_type=curr_rois[0].analysis_info['analysis_type']
    T_series=curr_rois[0].analysis_info['analysis_type']
    figure_save_dir = home_path + experiment +'\\' + 'summary\\'+ exp_id + '\\' + T_series + '\\figures\\cycle_01\\'  

    #extract saving directory

    pmc.plot_roi_traces(curr_rois,analysis_type,figure_save_dir)