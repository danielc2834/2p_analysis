# -*- coding: utf-8 -*-
"""
Created on Tue Nov  28 14:27:35 2023

@author: Juan Felipe, Chrisschna. It uses functions created by Burak Gur. 
Modified versions of functions from burak Gur are included in the script 
"""
#%%
from __future__ import print_function, division 
import os
import glob
import numpy as np
from scipy.signal.windows import gaussian
from PIL import Image
from scipy.fftpack import fft2,fftshift,ifftshift,ifft2  
import sys
import matplotlib.pyplot as plt
import tifffile
# from caiman.source_extraction.cnmf import params as params
# from caiman.source_extraction.cnmf.params import CNMFParams
import yaml
import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the root window

# Ask the user to select the config file
config_path = filedialog.askopenfilename(title="Select Configuration File", filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])

# Ensure the user selected a file
if not config_path:
    raise ValueError("No configuration file selected.")

#load config file 
# config_path = r'C:\Users\ptrimbak\Work_PT\2p_data\241129_L2_zoomed\50ms\config.yaml'
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Validate configuration parameters
required_keys = ['code_path','home_path', 'experiment',  'warnings_filter', 'warnings_category']
for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required configuration parameter: {key}")

# Use configuration parameters
cwd = os.getcwd()
code_path = config.get('code_path', os.path.join(cwd, '2panalysis', 'Helpers'))
sys.path.insert(0, code_path)

from core_functions import *
from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize,getMicRelativeTime
from preprocessing_func import *

#%%
#  preliminary data processing functions. list of tasks: create ROI instances, get a stimulus trace,        

# initialization variables
#neuropile='Lobula_plate' # this is important. depending on different neuropiles, the bleedthorugh filtering needs different filters

home_path=  config.get('home_path') #r'C:\Users\ptrimbak\Work_PT\2p_data\241129_L2_zoomed' #desktop office Juan
experiment = config.get('experiment')
experimenter=config.get('experimenter')

mot_corr=config.get('mot_corr') #perform motion correction without filtering out bleedthorugh
scale_values=config.get('scale_values')
bleedtrough_filtering=config.get('bleedtrough_filtering') #this is to be set true when the bleedthrough from stimulation light interferes with motion correction. both filtering and motion correction are done
fft_filter_treshold=config.get('fft_filter_treshold')
displacement=config.get('displacement')#[y,x] max displacement values, this is an input for the motion correction function of sima
produce_video_boolean=config.get('produce_video_boolean') # deprecated
produce_tif_stack=config.get('produce_tif_stack')
export_average=config.get('export_average')
produce_individual_tifs=config.get('produce_individual_tifs')
auto_ROI=config.get('auto_ROI') # deprecated
#%%
Motion_correction_algorithm = config.get('Motion_correction_algorithm') # "Caiman" or "chris"

channel=config.get('channel')

os.chdir(home_path) 

# TODO register the displacement used to do motion correction

for iidx,exp in enumerate(experiment):
    experimenter_code=experimenter[iidx]    
    Tseries_paths=produce_Tseries_list(home_path, exp,experimenter_code)    
    for Tser in Tseries_paths:
        print(Tser)
        os.chdir(Tser)
        # file_list,processed_files=produce_tif_list(Tser,mot_corr,channel=channel)
        # scanner, frame_period,im_perframe,samp_rate=extract_xml_info(Tser)
        if bleedtrough_filtering==True:
            exists=len(glob.glob('stack_FFT_filtered_Mcorr*'))>0
            #exists=False #uncomment if needed
            if exists==False:
                number_cycles=find_cycles(Tser)
                cycle_list=filter_cycles(Tser,number_cycles)
                FFT_2d,im_size=calculate_spatial_FFT(cycle_list)
                xmlFile=os.path.split((os.path.split(Tser)[0]))[1]
                xmlFile=Tser+ xmlFile +'.xml'
                _,pixel_size,_=getPixelSize(xmlFile)
                
                
                filtered_data_FFT=filter_bleedthrough_gaussianstripe(cycle_list,FFT_2d,im_size,'Lobula_plate',pixel_size,treshold=fft_filter_treshold)
                backtransformed_data=backtransform(filtered_data_FFT)
                os.chdir(Tser)
                dataset=motion_correction_with_array(backtransformed_data,Tser,fft_filter_treshold,displacement=displacement)
            else:
                dataset = sima.ImagingDataset.load('TserMC_bandstop_filtered.sima')
                
                
                
        elif Motion_correction_algorithm == "Caiman":
            dataset=preprocess_dataset(Tser,mot_corr, usecaiman=True)
        else:
            dataset = preprocess_dataset(Tser,mot_corr, usecaiman=False)
            # processed_files=export_tiffs(Tser,dataset,file_list,processed_files,mot_corr,produce_tif_stack=produce_tif_stack,max_displ=displacement)
        if auto_ROI==True:
            number_cycles=find_cycles(Tser)
            cycle_list=filter_cycles(Tser,number_cycles)
            # get pixel sizes, adjust them to fit in the 5 to 10 um range # this needs still adjustment
            xmlFile=os.path.split((os.path.split(Tser)[0]))[1]
            xmlFile=Tser+ xmlFile +'.xml'
            X_size,Y_size,pixel_area=pixel_size=getPixelSize(xmlFile) # these values are in um units
            lower_limit=int(round(1.5/pixel_area)) # the limits are hardcoded here 1 to 3 um^2
            higherlimit=int(round(7.5/pixel_area)) # TODO check how the separation between epochs works with a single
            find_clusters_STICA_JF_BG(dataset, lower_limit, higherlimit,len(cycle_list[1]),components=80)
            #
            
            #TODO adapt the calcium analysis script with your functions, TODO adapt the code to deal with multiple cycle recordings
        
        # if produce_video_boolean:
        #     produce_video(processed_files,mot_corr)
        

        # #TODO. 
        
        # file_list=[]
        # processed_files=[]
        # scanner=[]
        # frame_period=[]
        # im_perframe=[]
        # samp_rate=[]
        # dataset=[]
        # processed_files=[]
        