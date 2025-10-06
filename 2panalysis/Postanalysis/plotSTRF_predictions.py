
import os
from time import time
from tkinter import Grid
import cPickle
import STRF_utils as RF
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import NMF
code_path= r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis\helpers' #Juan desktop
sys.path.insert(0, code_path) 
import post_analysis_core as pac
import process_mov_core as pmc
import ROI_mod
from matplotlib.gridspec import GridSpec
from meansetSTRF import spatio_temporal_set

#initialize 
initialDirectory = 'C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan

experiment = 'T4T5_STRF_glucla_rescues'
data_dir = initialDirectory+experiment+'\\'+'processed\\files'
results_save_dir = initialDirectory+experiment+'\\'+'processed\\results\\STRF'
checkpoint_dir = initialDirectory+experiment+'\\'+'processed\\results\\STRF_checkpoints'
z_scores_dir = initialDirectory+experiment+'\\'+'processed\\results\\Z_scores'
ind_RF_dir = initialDirectory+experiment+'\\'+'processed\\results\\ind_RFs'
summary_dir = os.path.abspath('C:\\Users\\vargasju\\PhD\\experiments\\2p\\T4T5_STRF_glucla_rescues\\RFs')
stimulus_dir = os.path.abspath('C:\\Users\\vargasju\\PhD\\experiments\\2p\\T4T5_STRF_glucla_rescues\\stimuli_arrays')
plot_mapping_hist = False
calculate_av_SxT = True
simple_response_prediction = False
genotypes = ['control_homocygous']#['Tm3_Homocygous'] #['control_heterozygous', 'control_homocygous','L5rescue',  'Mi1rescue']
luminances_treats = ['lum_1'] #['lum_0.1','lum_0.25','lum_1'] 
restrictions = ['spatial_restriction','temporal_restriction','unrestricted']
stim_type = '3max'
colors=[(195.0/256.0,170.0/256.0,109.0/256.0),(150.0/256.0,150.0/256.0,150.0/256.0),(89.0/256.0,125.0/256.0,191.0/256.0),(100.0/256.0,100.0/256.0,100.0/256.0)]
regularization = [1]#[0.0001,0.01,0.1,1]
# import rois
datasets_to_load = os.listdir(data_dir)
curr_rois = []
final_rois = []
mapping_rois = []
included_count = 0
excluded_count = 0

#%%
for dataset in datasets_to_load:
    final_rois = []
    if 'DriftingStripe' in dataset:
        continue    
    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    print('working on Mean spatiotemporal RFs %s'%(workspace['final_rois'][0].experiment_info['FlyID']))
    print('%s rois'%(len(workspace['final_rois'])))
    
    
    for ij,roi in enumerate(workspace['final_rois']):

        RF.plot_predictions(roi)

