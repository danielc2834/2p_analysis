#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:42:35 2020

@author: burakgur , Jfelipeco
"""


# from asyncore import compact_traceback
import copy
#from _curses import ra
from logging import exception
from re import L
import time
from matplotlib import image
from numpy.core.defchararray import index
from numpy.core.numeric import roll
from numpy.lib.type_check import imag
# import sima
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import _pickle as cPickle # For Python 3.X
import random
import warnings

from skimage import io
from scipy import interpolate
from scipy.stats import pearsonr
from itertools import permutations

from skimage import filters
from scipy import ndimage
from scipy.ndimage import rotate
from matplotlib.backends.backend_pdf import PdfPages

from Helpers import ROI_mod
from Helpers import summary_figures as sf
from Helpers.xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize,getMicRelativeTime,getrastersPerFrame
from Helpers.core_functions import readStimOut, readStimInformation, getEpochCount, divide_all_epochs, divide_trials_1epoch, divideEpochs
from Helpers.post_analysis_core import run_matplotlib_params
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import fclusterdata
import tifffile
import cv2


def whole_stim_experiment(rois):
    stim_info = rois[0].stim_info 
    stim_coords = stim_info['trial_coordinates']
    for roi in rois:
        # Generate stimulation trace for whole experiment
        stim_trace_all = roi.stim_info['epoch_trace_frames']
        stim_trace=np.zeros([5,len(stim_trace_all)])
        for idx2, stim_property in enumerate(list(roi.stim_info['meta']['epoch_infos']['epoch_1'].keys())[1:6]):
            curr_stim = np.zeros((1,len(stim_trace_all)))[0]
            for idx, e in enumerate(stim_trace_all):
                try:
                    curr_stim[idx] = roi.stim_info['meta']['epoch_infos'][list(stim_coords)[int(e-1)]][stim_property]
                except:
                    curr_stim[idx] = np.mean(roi.stim_info['meta']['epoch_infos'][list(stim_coords)[int(e-2)]][stim_property],roi.stim_info['meta']['epoch_infos'][list(stim_coords)[int(e)]][stim_property])
                stim_trace[idx2,:]=curr_stim
        
        roi.whole_stim_experiment = stim_trace
    return rois

    
def compute_correlation_image(video):
    
    xdim = video.shape[1]
    ydim = video.shape[2]
    window = 6
    factor = window/2
    corr_image = np.zeros(video.shape[1:])
    pval_image = np.zeros(video.shape[1:])
    for ix in range(xdim- window):
        for iy in range(ydim-window):
            pix_x = ix+factor
            pix_y = iy+factor
            curr_pix_trace = video[:,pix_x,pix_y]
            neighbors = video[:,pix_x-factor:pix_x+factor, 
                              pix_y-factor:pix_y+factor]
            neighbors_trace = neighbors.mean(axis=1).mean(axis=1)
            curr_coeff, pval = pearsonr(curr_pix_trace,neighbors_trace)
            corr_image[pix_x,pix_y] = curr_coeff
            pval_image[pix_x,pix_y] = pval
    return corr_image, pval_image
    

def get_stim_xml_paramsPyStim(t_series_path, stimInputDir):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    t_series_path : str
        Path to the T series folder for retrieving stimulus related information and
        xml file which contains imaging parameters.
    
    stimInputDir : str
        Path to the folder where stimulus input information is located.
        
    Returns
    =======
    stimulus_information : list 
        Stimulus related information is stored here.
    trialCoor : list
        Start, end coordinates for each trial for each epoch
    frameRate : float
        Image acquisiton rate.
    depth :
        Z axis value of the imaging dataset.

    """

    # Finding the xml file and retrieving relevant information
    
    xmlPath = os.path.join(t_series_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    imagetimes = getMicRelativeTime(xmlFile)
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Stimulus information
    stimOutPath = os.path.join(t_series_path, '*.pickle')
    stimOutFile = (glob.glob(stimOutPath))[0]

    load_path = open(stimOutFile, 'rb')
    stimInfo = cPickle.load(load_path)

    randomization_condition = int(stimInfo['meta']['randomization_condition'])
    epochDur = stimInputData['Stimulus.duration']
    epochDur = [float(sec) for sec in epochDur]
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount <= 1:
        trialCoor = 0
        trialCount = 0
    elif randomization_condition == 1:
        (trialCoor, trialCount, _) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=randomization_condition,
                                                 framePeriod=framePeriod,
                                                 trialDiff=0.20,
                                                 overlappingFrames=0,
                                                 firstEpochIdx=0,
                                                 epochColumn=3,
                                                 imgFrameColumn=7,
                                                 incNextEpoch=True,
                                                 checkLastTrialLen=True)
    else:
        (trialCoor, trialCount) = divide_all_epochs(rawStimData, epochCount, 
                                                    framePeriod, trialDiff=0.20,
                                                    epochColumn=3, imgFrameColumn=7,
                                                    checkLastTrialLen=True)
     
    stimulus_information ={}
    stimulus_data = stimInputData
    stimulus_information['epoch_dur'] = epochDur
    stimulus_information['random'] = randomization_condition
    stimulus_information['output_data'] = rawStimData
    stimulus_information['frame_timings'] = imagetimes
    if randomization_condition==0:
        stimulus_information['baseline_epoch'] = 0  
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus non random, baseline epoch selected as 0th epoch\n')
    elif randomization_condition == 2:
        stimulus_information['baseline_epoch'] = None
        stimulus_information['baseline_duration'] = None
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus all random, no baseline epoch present\n')
    elif randomization_condition == 1:
        stimulus_information['baseline_epoch'] = 0 
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 1
        
    stimulus_information['epoch_dir'] = \
            np.asfarray(stimulus_data['Stimulus.stimrot.mean'])
    epoch_speeds = np.asfarray(stimulus_data['Stimulus.stimtrans.mean'])
    stimulus_information['epoch_frequency'] = \
        epoch_speeds/np.asfarray(stimulus_data['Stimulus.spacing'])
    stimulus_information['epochs_duration'] =\
         np.asfarray(stimulus_data['Stimulus.duration'])
    stimulus_information['epoch_number'] =  \
        np.asfarray(stimulus_data['EPOCHS'][0])
    stimulus_information['stim_type'] =  \
        np.asfarray(stimulus_data['Stimulus.stimtype'])
    stimulus_information['input_data'] = stimInputData
    stimulus_information['stim_name'] = stimType.split('\\')[-1]
    stimulus_information['trial_coordinates'] = trialCoor
    
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth}
        
    return stimulus_information, imaging_information

def get_stim_xml_params(t_series_path,original_stimDir,Tseries_len,cplusplus=False): #from sebastians code
    """ Gets the required stimulus and imaging parameters.
    Parameters, uses txt _stimulus_output files 
    ==========
    t_series_path : str
        Path to the T series folder for retrieving stimulus related information and
        xml file which contains imaging parameters.
    
    stimInputDir : str
        Path to the folder where stimulus input information is located.
        
    Returns
    =======
    stimulus_information : list 
        Stimulus related information is stored here.
    trialCoor : list
        Start, end coordinates for each trial for each epoch
    frameRate : float
        Image acquisiton rate.
    depth :
        Z axis value of the imaging dataset.

    """
    # Finding the xml file and retrieving relevant information
    
    xmlFile = f'{t_series_path}/{os.path.basename(t_series_path)}.xml'
    stimOuputFile = f'{t_series_path}/{os.path.basename(t_series_path)}_stim_output.txt'
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    screen_dim= 80 #screen dimensions in degrees. caution. this is hardcoded
    imagetimes = getMicRelativeTime(xmlFile)
    #imagetimes = imagetimes[49:-1] # Seb: when imaging with cycle00001 = 50 frames  CHECK POINT
    #print('50 frames cycle00001 in data')
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Stimulus output information
    
    #stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
    #stimOutFile = (glob.glob(stimOutPath))[0]
    (stimType, rawStimData,raw_stimkeys) = readStimOut(stimOuputFile,Tseries_len, 
                                          skipHeader=3,cplusplus=cplusplus) # Seb: skipHeader = 3 for _stimulus_ouput from 2pstim
    #TODO continue here
    # Stimulus information
    (stimInputFile,stimInputData) = readStimInformation(stimType=stimType,
                                                      original_stimDir=original_stimDir)

    try:
        isRandom = int(stimInputData['RANDOMIZATION_MODE'])
        
    except:
        isRandom = int(stimInputData['randomize'][0])
    
    for ix,entry in enumerate(stimInputData['duration']):
        try:
            subepoch=stimInputData['subepoch'][ix]
        except:
            subepoch=1
        if entry==-1 and stimInputData['stimtype'][ix]=='G':
            stimInputData['duration'][ix]=stimInputData['sWavelength'][ix]/stimInputData["velocity"][ix]
            
        #calculate stim duration


        elif entry == 1 and stimInputData['stimtype'][ix]=='ADS' and subepoch==2:
            x_component=np.abs(np.cos((np.deg2rad(stimInputData['angle'][ix]))))
            y_component=np.abs(np.sin((np.deg2rad(stimInputData['angle'][ix]))))
            if x_component>y_component:
                hypothenuse=screen_dim/x_component
            elif x_component<y_component:
                hypothenuse=screen_dim/y_component
            else:
                hypothenuse=np.sqrt(screen_dim**2)
            barwidth=hypothenuse+10
            stimInputData['duration'][ix]=2*(barwidth/stimInputData['velocity'][ix])
            stimInputData['duration'][ix]=stimInputData['duration'][ix]+stimInputData['tau'][ix]
        elif entry == 1 and stimInputData['stimtype'][ix]=='ADS' and subepoch==1:
            x_component=np.abs(np.cos((np.deg2rad(stimInputData['angle'][ix]))))
            y_component=np.abs(np.sin((np.deg2rad(stimInputData['angle'][ix]))))
            if x_component>y_component:
                hypothenuse=screen_dim/x_component
            elif x_component<y_component:
                hypothenuse=screen_dim/y_component
            else:
                hypothenuse=np.sqrt(screen_dim**2)
            barwidth=hypothenuse+10
            stimInputData['duration'][ix]=barwidth/stimInputData['velocity'][ix]
            stimInputData['duration'][ix]=stimInputData['duration'][ix]+stimInputData['tau'][ix]
        elif entry == 1 and stimInputData['stimtype'][ix]=='driftingstripe' and subepoch==2:
            x_component=np.abs(np.cos((np.deg2rad(stimInputData['angle'][ix]))))
            y_component=np.abs(np.sin((np.deg2rad(stimInputData['angle'][ix]))))
            if x_component>y_component:
                hypothenuse=screen_dim/x_component
            elif x_component<y_component:
                hypothenuse=screen_dim/y_component
            else:
                hypothenuse=np.sqrt(screen_dim**2)
            barwidth=hypothenuse+10
            stimInputData['duration'][ix]=2*(barwidth/stimInputData['velocity'][ix])
            stimInputData['duration'][ix]=stimInputData['duration'][ix]+stimInputData['tau'][ix]
        # elif entry == 8 and stimInputData['stimtype'][ix]=='driftingstripe' and subepoch==1:
        #     x_component=np.abs(np.cos((np.deg2rad(stimInputData['angle'][ix]))))
        #     y_component=np.abs(np.sin((np.deg2rad(stimInputData['angle'][ix]))))
        #     if x_component>y_component:
        #         hypothenuse=screen_dim/x_component
        #     elif x_component<y_component:
        #         hypothenuse=screen_dim/y_component
        #     else:
        #         hypothenuse=np.sqrt(screen_dim**2)
        #     barwidth=hypothenuse+10
        #     stimInputData['duration'][ix]=barwidth/stimInputData['velocity'][ix]
        #     stimInputData['duration'][ix]=stimInputData['duration'][ix]+stimInputData['tau'][ix]    

    epochDur = stimInputData['duration']
    epochDur = [float(sec) for sec in epochDur]
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    
    # !!!!!!!!temporal change
    
    #fg_info=stimInputData['fg']
    #bg_info=stimInputData['bg']
    
    
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount == 1: #and stimInputData['EPOCHS'] == 1: # this applies to whitenoise stimuli
        # trialCoor = {0:[[int(rawStimData[0][-1]),-1],[int(rawStimData[0][-1]),-1]]}
        # trialCount = 1
        (trialCoor, trialCount) = divide_trials_1epoch(rawStimData)
    
    elif isRandom == 1:
        (trialCoor, trialCount, _) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=isRandom,
                                                 framePeriod=framePeriod,
                                                 trialDiff=0.20,
                                                 overlappingFrames=0,
                                                 firstEpochIdx=0,
                                                 epochColumn=3,
                                                 imgFrameColumn=7,
                                                 incNextEpoch=True,
                                                 checkLastTrialLen=True)
    else:
        (trialCoor, trialCount) = divide_all_epochs(rawStimData, epochCount, 
                                                    framePeriod, trialDiff=0.20,
                                                    epochColumn=3, imgFrameColumn=7,
                                                    checkLastTrialLen=True)
     
    
    # Transfering all data from input file to stimulus_information
    
    stimulus_data = stimInputData
    # stimulus_information ={}
    # stimulus_information = dict(stimulus_information.items() + stimulus_data.items())
    stimulus_information = stimulus_data.copy()


    # Adding more information
    stimulus_information['epoch_dur'] = epochDur # Seb: consider to delete this line. Redundancy
    stimulus_information['random'] = isRandom # Seb: consider to delete this line. Redundancy
    stimulus_information['output_data'] = rawStimData
    #
    ##!!! temporal comment. uncomment
    
    #stimulus_information['fg']=fg_info
    #stimulus_information['bg']=bg_info
    
    #!!!!!
    
    # # Seb: matching imagetimes with the first recorded frame with stimulus on the screen
    # first_frame = int(stimulus_information['output_data'][0][-1]) -1
    # last_frame = len(imagetimes)
    # imagetimes = imagetimes[first_frame:last_frame] 

    stimulus_information['frame_timings'] = imagetimes
    stimulus_information['input_data'] = stimInputData # Seb: consider to delete this line. Redundancy
    stimulus_information['stim_name'] = stimType.split('/')[-1]
    stimulus_information['trial_coordinates'] = trialCoor

    if isRandom==0:
        stimulus_information['baseline_epoch'] = 0  
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus non random, baseline epoch selected as 0th epoch\n')
    elif isRandom == 2:
        stimulus_information['baseline_epoch'] = None
        stimulus_information['baseline_duration'] = None
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus all random, no baseline epoch present\n')
    elif isRandom == 1:
        stimulus_information['baseline_epoch'] = 0 
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 1
    
    #Seb: commented

    # stimulus_information['epoch_dir'] = \
    #         np.asfarray(stimulus_data['Stimulus.stimrot.mean'])
    # epoch_speeds = np.asfarray(stimulus_data['Stimulus.stimtrans.mean'])
    # stimulus_information['epoch_frequency'] = \
    #     epoch_speeds/np.asfarray(stimulus_data['Stimulus.spacing'])
    # stimulus_information['epochs_duration'] =\
    #      np.asfarray(stimulus_data['Stimulus.duration'])
    # stimulus_information['epoch_number'] =  \
    #     np.asfarray(stimulus_data['EPOCHS'][0])
    # stimulus_information['stim_type'] =  \
    #     np.asfarray(stimulus_data['Stimulus.stimtype'])


    # Keeping imaging information
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth,}
    
    #subsample raw stim-data to get one datapoint per frame and store this:

    frame_idx=np.concatenate(([0],np.where(np.diff(rawStimData[:,-1])==1)[0]+1))
    stimulus_information['output_data_downsampled'] = rawStimData[frame_idx,:]
    stimulus_information['output_data_downsampled']=pd.DataFrame(stimulus_information['output_data_downsampled'],columns=raw_stimkeys)
    raw_stim_trace=[]
    
    raw_stim_out_df=pd.DataFrame(rawStimData,columns=raw_stimkeys) #temporary object
    
    # make a raw_stimulus trace to calculate correlation between raw response and stim
    for microscope_frame in stimulus_information['output_data_downsampled']['data'].astype(int):
        subset=raw_stim_out_df.loc[raw_stim_out_df['data']==microscope_frame]
        if np.array(subset['epoch'])[0]==np.array(subset['epoch'])[-1]:
            raw_stim_trace.append(np.array(subset['epoch'])[0])
        elif np.array(subset['epoch'])[0]!=np.array(subset['epoch'])[-1]:
            raw_stim_trace.append(np.array(subset['epoch'])[-1])
    # make a vector that matches epochs and directions

    if isRandom == 1 and stimulus_information['stim_name']=='exp_random_ONedges_20dirs_20degPerS.txt':
        stimulus_information['output_data_downsampled'].loc[stimulus_information['output_data_downsampled']['epoch']==0,'theta']=np.nan
    stimulus_information['output_data_downsampled']['epoch']=raw_stim_trace
    stimulus_information['output_data_downsampled']['interlude']=stimulus_information['output_data_downsampled']['epoch']==0
    stimulus_information['output_data_downsampled']['frame']=(stimulus_information['output_data_downsampled']['data'].astype(int))
    stimulus_information['output_data_downsampled']=stimulus_information['output_data_downsampled'].set_index('frame')
    return stimulus_information, imaging_information

def separate_trials_video(time_series,stimulus_information,frameRate):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : list
        Start, end coordinates for each trial for each epoch
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    dff_baseline_dur_frame: int
        Duration of baseline before the stimulus for using in dF/F calculation.
        
    Returns
    =======
    wholeTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-

    """
    trialCoor = stimulus_information['trial_coordinates']
    mov_xDim = time_series.shape[1]
    mov_yDim = time_series.shape[2]
    wholeTraces_allTrials = {}
    respTraces_allTrials = {}
    baselineTraces_allTrials = {}
    if len(trialCoor) > 1:
        for iEpoch in trialCoor:
            currentEpoch = trialCoor[iEpoch]         
            current_epoch_dur = stimulus_information['duration'][iEpoch]
            trial_numbers = len(currentEpoch)
            trial_lens = []
            base_lens = []
            for curr_trial_coor in currentEpoch:
                current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
                trial_lens.append(current_trial_length)
                
                baselineStart = curr_trial_coor[1][0]
                baselineEnd = curr_trial_coor[1][1]
                
                base_lens.append(baselineEnd - baselineStart)
            median_len = np.median(trial_lens)       
            trial_len =  min(trial_lens) # Juan changed this from -4 to -1, then changed it from -1 to 0
            resp_len = int(round(frameRate * current_epoch_dur))+1
    #        resp_len = int(round(frameRate * 3))
            base_len = min(base_lens)
            
            if stimulus_information['random']==2:
                wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                                trial_numbers))
                respTraces_allTrials[iEpoch] = None
                baselineTraces_allTrials[iEpoch] = None
                
            elif stimulus_information['random'] == 1:
                wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                                trial_numbers))
                respTraces_allTrials[iEpoch] = np.zeros(shape=(resp_len,mov_xDim,mov_yDim,
                                trial_numbers))
                baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                                trial_numbers))
            else:
                wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                                trial_numbers))
                respTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                                trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials\
                                    [stimulus_information['baseline_epoch']])[0]
                baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                                trial_numbers))
    
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if stimulus_information['random']==2:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    raw_signal = time_series[trialStart:trialStart+trial_len, : , :] ### it was before trialStart:trialEnd
                    
                    wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                
                elif stimulus_information['random'] == 1:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    raw_signal = time_series[trialStart:trialStart+trial_len, : , :]
                
                    currentResp = time_series[respStart:epochEnd, : , :]
                    #        dffTraces_allTrials[iEpoch].append(dFF[:trial_len,:,:])
                    wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                    respTraces_allTrials[iEpoch][:,:,:,trial_num]= currentResp[:resp_len,:,:]
                    baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:base_len,:,:]
                else:
                    
                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    if iEpoch == stimulus_information['baseline_epoch']:
                        baseline_signal = time_series[trialStart:trialEnd, : , :]
                    raw_signal = time_series[trialStart:trialStart+trial_len, : , :]
                    if iEpoch==7 and trial_num==1:
                        aaa='aaaaa'
                    wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                    respTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                    baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= baseline_signal[:base_len,:,:]
                
            print('Epoch %d completed \n' % iEpoch)
    
    elif len(trialCoor) == 1 and stimulus_information['EPOCHS'] == 1: # intended for white noise stimuli
        try:
            wholeTraces_allTrials = {0:time_series[trialCoor[0][0][0]:-1]}
            respTraces_allTrials = {0:time_series[trialCoor[0][0][0]:-1]}
        except:
            #wholeTraces_allTrials = {0:time_series[trialCoor[0][0][0][0]:-1]}
            #respTraces_allTrials = {0:time_series[trialCoor[0][0][0][0]:-1]}
            pass
            #wholeTraces_allTrials
    return (wholeTraces_allTrials, respTraces_allTrials, baselineTraces_allTrials)

def calculate_pixel_SNR(baselineTraces_allTrials,respTraces_allTrials,
                  stimulus_information,frameRate,SNR_mode ='Estimate'):
    """ Calculates the pixel-wise signal-to-noise ratio (SNR). Equation taken from
    Kouvalainen et al. 1994 (see calculation of SNR true from SNR estimated). 
    
    Parameters
    ==========
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    SNR_mode : not implemented yet
    
        
        
    Returns
    =======
    
    SNR_max_matrix : np array
        An m x n array with pixel-wise SNR.

    """
    
    mov_xDim = np.shape(baselineTraces_allTrials[1])[1]
    mov_yDim = np.shape(baselineTraces_allTrials[1])[2]
    total_epoch_numbers = len(baselineTraces_allTrials)
    
    
#    total_background_dur = stimulus_information['epochs_duration'][0]
    SNR_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    for iPlot, iEpoch in enumerate(baselineTraces_allTrials):
        
        trial_numbers = np.shape(baselineTraces_allTrials[iEpoch])[3]
        currentBaseTrace = baselineTraces_allTrials[iEpoch][:,:,:,:]
        currentRespTrace =  respTraces_allTrials[iEpoch][:,:,:,:]
        
        noise_std = currentBaseTrace.std(axis=0).mean(axis=2)
        resp_std = currentRespTrace.std(axis=0).mean(axis=2)
        
        signal_std = resp_std - noise_std
        # SNR calculation taken from
        curr_SNR_true = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
#        curr_SNR = (signal_std/noise_std) 
        SNR_matrix[:,:,iPlot] = curr_SNR_true
        
       
    SNR_matrix[np.isnan(SNR_matrix)] = np.nanmin(SNR_matrix) # change nan values with min values
    
    SNR_max_matrix = SNR_matrix.max(axis=2) # Take max SNR for every pixel for every epoch

    return SNR_max_matrix

def calculate_pixel_max(respTraces_allTrials,stimulus_information):
    
    """ Calculates the pixel-wise maximum responses for each epoch. Returns max 
    epoch indices as well but adjusting the indices considering that baseline is 0.
    
    Parameters
    ==========
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
        
    Returns
    =======
    
    MaxResp_matrix_without_edge : np array
        An array with pixel-wise maxiumum responses for epochs other than edges
        that are normally used as probe stimuli. Edge maximums are set to -100 so that
        they're never maximum.
    
    MaxResp_matrix_all_epochs : np array
        An array with pixel-wise maxiumum responses for every epoch.
        
    maxEpochIdx_matrix_without_edge : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    maxEpochIdx_matrix_all : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    """
    epoch_adjuster = stimulus_information['epoch_adjuster']
    
    mov_xDim = np.shape(respTraces_allTrials[1])[1]
    mov_yDim = np.shape(respTraces_allTrials[1])[2]
    total_epoch_numbers = len(respTraces_allTrials)
    
    # Create an epoch-wise maximum response list
    maxResp = {}
    meanResp ={}
    # Create an array with m x n x nEpochs
    MaxResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    MeanResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    
    for index, iEpoch in enumerate(respTraces_allTrials):
        
        # Find maximum of pixels after trial averaging
        curr_max =  np.nanmax(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
        curr_mean = np.nanmean(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
        
        maxResp[iEpoch] = curr_max
        MaxResp_matrix[:,:,index] = curr_max
        
        meanResp[iEpoch] = curr_mean
        MeanResp_matrix[:,:,index] = curr_mean
        
    
    # Make an additional one with edge and set edge maximum to 0 in the main array
    # This is to avoid assigning pixels to edge temporal frequency if they respond
    # max in the edge epoch but not in one of the grating epochs.
    MaxResp_matrix_all_epochs = copy.deepcopy(MaxResp_matrix)
    MeanResp_matrix_all_epochs = copy.deepcopy(MeanResp_matrix)

    
    # Finding pixel-wise max epochs
    maxEpochIdx_matrix_all = np.argmax(MaxResp_matrix_all_epochs,axis=2) 
    maxEpochIdx_matrix_all_mean = np.argmax(MeanResp_matrix_all_epochs,axis=2) 
    
    # To assign numbers like epoch numbers
    maxEpochIdx_matrix_all = maxEpochIdx_matrix_all + epoch_adjuster
    maxEpochIdx_matrix_all_mean = maxEpochIdx_matrix_all_mean + epoch_adjuster
    
    
    return MaxResp_matrix_all_epochs, maxEpochIdx_matrix_all, \
           MeanResp_matrix_all_epochs, maxEpochIdx_matrix_all_mean
            


def create_DSI_image(stimulus_information, maxEpochIdx_matrix_all,max_resp_matrix_all,
                     MaxResp_matrix_all_epochs):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
   stimulus_information : list 
        Stimulus related information is stored here.
        
    maxEpochIdx_matrix_all : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    max_resp_matrix_all : np array
        An array with pixel-wise maxiumum responses for the experiment.
        
    MaxResp_matrix_all_epochs : np array
        An array with pixel-wise maxiumum responses for every epoch.
        
    

     
    Returns
    =======
    
    DSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    
    """
    
    DSI_image = copy.deepcopy(max_resp_matrix_all) # copy it for keeping nan value
    for iEpoch, current_epoch_type in enumerate (stimulus_information['stim_type']):
        
        if (stimulus_information['random']) and (iEpoch ==0):
            continue
        
        current_pixels = (maxEpochIdx_matrix_all == iEpoch) & \
                            (~np.isnan(max_resp_matrix_all))
        current_freq = stimulus_information['epoch_frequency'][iEpoch]
        if ((current_epoch_type != 50) and (current_epoch_type != 61) and\
            (current_epoch_type != 46)) or (current_freq ==0):
            DSI_image[current_pixels] = 0
            continue
        current_dir = stimulus_information['epoch_dir'][iEpoch]
        required_epoch_array = \
            (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
            (stimulus_information['epoch_frequency'] == current_freq) & \
            (stimulus_information['stim_type'] == current_epoch_type)
            
        opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in \
                              enumerate(required_epoch_array) if epoch][0]
        opposite_dir_epoch = opposite_dir_epoch # To find the real epoch number without the baseline
        # Since a matrix will be indexed which doesn't have any information about the baseline epoch
        
    
        opposite_response_trace = MaxResp_matrix_all_epochs\
            [:,:,opposite_dir_epoch-stimulus_information['epoch_adjuster']] 
            
        DSI_image[current_pixels] =(np.abs(((max_resp_matrix_all[current_pixels] - \
                                      opposite_response_trace[current_pixels])\
                                        /(max_resp_matrix_all[current_pixels] + \
                                          opposite_response_trace[current_pixels]))))
#    DSI_image[(DSI_image>1)] = 0
#    DSI_image[(DSI_image<-1)] = 0
    
    return DSI_image

def create_CSI_image(stimulus_information, frameRate,respTraces_allTrials, 
                     DSI_image):
    """ Makes pixel-wise plot of CSI

    Parameters
    ==========
   stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
            
    DSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    

     
    Returns
    =======
    
    CSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    
    """
    # Image dimensions
    mov_xDim = np.shape(DSI_image)[0]
    mov_yDim = np.shape(DSI_image)[1]
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    epochDur= stimulus_information['epochs_duration']
    
    if len(edge_epochs) == 2: # 2 edges exist 
        
        ON_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        OFF_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        CSI_image = np.zeros(shape=(mov_xDim,mov_yDim))
        half_dur_frames = int((round(frameRate * epochDur[edge_epochs[0]]))/2)
        
        for index, epoch in enumerate(edge_epochs):
            
            
            OFF_resp[:,:,index] = np.nanmax(np.nanmean(\
                    respTraces_allTrials[epoch]\
                    [:half_dur_frames,:,:,:],axis=3),axis=0)
            ON_resp[:,:,index] = np.nanmax(np.nanmean(\
                   respTraces_allTrials[epoch]\
                   [half_dur_frames:,:,:,:],axis=3),axis=0)
        
        
        CSI_image[DSI_image>0] = (ON_resp[:,:,0][DSI_image>0] - OFF_resp[:,:,0][DSI_image>0])/(ON_resp[:,:,0][DSI_image>0] + OFF_resp[:,:,0][DSI_image>0])
        CSI_image[DSI_image<0] = (ON_resp[:,:,1][DSI_image<0] - OFF_resp[:,:,1][DSI_image<0])/(ON_resp[:,:,1][DSI_image<0] + OFF_resp[:,:,1][DSI_image<0])
        
    # It shouldn't be below -1 or above +1 if it is not noise
    CSI_image[(CSI_image>1)] = 0
    CSI_image[(CSI_image<-1)] = 0    
    
    return CSI_image


def plot_pixel_maps(im1, im2, im3, im4, exp_ID, depth, save_fig = False,
                    save_dir = None):

    """ Plots 4 images in a figure. Normally used with mean, max, DSI, CSI 
    images
    
    Parameters
    ==========
    mean_image : numpy array
        Mean image of the video.
    
    max_resp_matrix_all : numpy array
        Maximum responses
        
    DSI_image : numpy array
        Mean image of the video.
        
    CSI_image : numpy array
        Mean image of the video.
        
    exp_ID : str
    
    depth : int or float
        

    """
    plt.close('all')
    
    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True,
                            facecolor='k', edgecolor='w',figsize=(16, 10))
    
    
    depthstr = 'Z: %d' % depth
    figtitle = 'Summary ' + depthstr
    
    # Mean image
    fig1.suptitle(figtitle,fontsize=12)
    sns.heatmap(im1,ax=ax1[0][0],cbar_kws={'label': 'dF/F'},cmap='viridis')
    #    sns.heatmap(layer_masks,alpha=.2,cmap='Set1',ax=ax2[0],cbar=False)
    #    sns.heatmap(BG_mask,alpha=.1,ax=ax2[0],cbar=False)
    ax1[0][0].axis('off')
    ax1[0][0].set_title('Mean image')
    
    # Max responses
    sns.heatmap(im2,cbar_kws={'label': 'SNR'},ax=ax1[0][1])
    #sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax2[1])
    ax1[0][1].axis('off')
    ax1[0][1].set_title('SNR image')
    # DSI
    sns.heatmap(im3,cbar_kws={'label': 'DSI'},cmap = 'inferno',ax=ax1[1][0])
    #sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax2[1])
    ax1[1][0].axis('off')
    ax1[1][0].set_title('DSI (Blue: --> Red: <--)')
    
    #CSI
    sns.heatmap(im4,cbar_kws={'label': 'CSI'},cmap = 'inferno',ax=ax1[1][1],vmax=1,vmin=-1)
    
    ax1[1][1].axis('off')    
    ax1[1][1].set_title('CSI (Dark:OFF, Red:ON)')
    
    if save_fig:
        # Saving figure
        save_name = 'summary_%s' % (exp_ID)
        # os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('Pixel maps saved')

def generate_avg_movie(dataDir, stimulus_information, 
                           wholeTraces_allTrials_video):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    dataDir: str
        Path into the directory where the motion corrected dataset with selected
        masks is present.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    wholeTraces_allTrials_video : list containing np arrays
        Epoch list of time traces including all trials in the form of:
        baseline epoch - stimulus epoch - baseline epoch
            
    
        
    Returns
    =======
    
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
    """
    print('Generating averaged movie...\n')
    # Directory for where to save the cluster movie
    selected_movie_dir = os.path.join(dataDir,'processed.sima')
    mov_xDim = np.shape(wholeTraces_allTrials_video[1])[1]
    mov_yDim = np.shape(wholeTraces_allTrials_video[1])[2]    
    
    epochs_to_use = range(len(stimulus_information['epoch_frequency']))
    if stimulus_information['random'] == 1:
        epochs_to_use = np.delete(epochs_to_use,stimulus_information['baseline_epoch'])
    epoch_frames = np.zeros(shape=np.shape(epochs_to_use))
    
    # Generating and filling the movie array movie array
    for index, epoch in enumerate(epochs_to_use):
        epoch_frames[index] = np.shape(wholeTraces_allTrials_video[epoch])[0]
        
    avg_movie = np.zeros(shape=(int(epoch_frames.sum()),1,mov_xDim,mov_yDim,1))
    
    startFrame = 0
    for index, epoch in enumerate(epochs_to_use):
        if index>0:
            startFrame =  endFrame 
        endFrame = startFrame + epoch_frames[index]
        avg_movie[int(startFrame):int(endFrame),0,:,:,0] = \
            wholeTraces_allTrials_video[epoch].mean(axis=3)
            
    
    # Create a sima dataset and export the cluster movie
    b = sima.Sequence.create('ndarray',avg_movie)
    average_dataset = sima.ImagingDataset([b],None)
    average_dataset.export_frames([[[os.path.join(selected_movie_dir,'avg_vid.tif')]]],
                                      fill_gaps=True,scale_values=True)
    print('Averaged movie generated...\n')
    
    return average_dataset


def find_clusters_STICA(cluster_dataset, area_min, area_max):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
        
    area_min : int
        Minimum area of a cluster in pixels
        
    area_max : int
        Maximum area of a cluster in pixels
        
    

     
    Returns
    =======
    
    clusters : sima.ROI.ROIList
        A list of ROIs.
        
    all_clusters_image: numpy array
        A numpy array that contains the masks.
    """    
    print('\n-->Segmentation running...')
    segmentation_approach = sima.segment.STICA(channel = 0,components=45,mu=0.1)
    segmentation_approach.append(sima.segment.SparseROIsFromMasks(
            min_size=area_min,smooth_size=3))
    #segmentation_approach.append(sima.segment.MergeOverlapping(threshold=0.90))
    #segmentation_approach.append(sima.segment.SmoothROIBoundaries(tolerance=0.1,n_processes=(nCpu - 1)))
    size_filter = sima.segment.ROIFilter(lambda roi: roi.size >= area_min and \
                                         roi.size <= area_max)
#    circ_filter = sima.segment.CircularityFilter(circularity_threhold=0.7)
    segmentation_approach.append(size_filter)
#    segmentation_approach.append(circ_filter)
    start1 = time.time()
    clusters = cluster_dataset.segment(segmentation_approach, 'auto_ROIs')
    initial_cluster_num = len(clusters)
    end1 = time.time()
    time_passed = end1-start1
    print('Clusters found in %d minutes\n' % \
          round(time_passed/60) )
    print('Number of initial clusters: %d\n' % initial_cluster_num)
    
    
    # Generating an image with all clusters
    data_xDim = cluster_dataset.frame_shape[1]
    data_yDim = cluster_dataset.frame_shape[2]
    all_clusters_image = np.zeros(shape=(data_xDim,data_yDim))
    all_clusters_image[:] = np.nan
    for index, roi in enumerate(clusters):
        curr_mask = np.array(roi)[0,:,:]
        all_clusters_image[curr_mask] = index+1        
        
    return clusters, all_clusters_image

def get_layers_bg_mask(dataDir):
    """ Gets the masks of pre-selected (with roibuddy) layers.

    Parameters
    ==========
    dataDir: str
        Path into the directory where the motion corrected dataset with selected
        masks is present.
    
  
    Returns
    =======
    
    layer_masks_bool: np array
        A boolean image of where the masks are located
        
    BG_mask: np array
        A boolean image of background mask
    
    """                       
                                    
    dataset = sima.ImagingDataset.load(dataDir)
    roiKeys = dataset.ROIs.keys()
    roiKeyNo = 0
    rois_layer = dataset.ROIs[roiKeys[roiKeyNo]]
    layer_masks = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    layer_masks_bool = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    layer_masks[:] = np.nan
    
    BG_mask = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    BG_mask[:] = np.nan
    for index, roi in enumerate(rois_layer):
        curr_mask = np.array(roi)[0,:,:]
        roi_label = roi.label
        
        if roi_label == 'Layer1':
            
            L1_mask = curr_mask
            layer_masks[curr_mask] = 1
            layer_masks_bool[curr_mask] = 1
            print("Layer 1 mask found\n")
        elif roi_label == 'Layer2':
            L2_mask = curr_mask
            layer_masks[curr_mask] = 2
            layer_masks_bool[curr_mask] = 1
            print("Layer 2 mask found\n")
        elif roi_label == 'Layer3':
             
            L3_mask = curr_mask
            layer_masks[curr_mask] = 3
            layer_masks_bool[curr_mask] = 1
            print("Layer 3 mask found\n")
        elif roi_label == 'Layer4':
            L4_mask = curr_mask
            layer_masks[curr_mask] = 4
            layer_masks_bool[curr_mask] = 1
            print("Layer 4 mask found\n")
        elif roi_label == 'LobDen':
            LD_mask = curr_mask
    #        layer_masks[curr_mask] = 3
    #        layer_masks_bool[curr_mask] = 1
        elif roi_label == 'MedDen':
            MD_mask = curr_mask
    #        layer_masks[curr_mask] = 4
    #        layer_masks_bool[curr_mask] = 1
        elif roi_label == 'BG':
            BG_mask = curr_mask
            print("BG mask found\n")
        else:
            print("ROI doesn't match to criteria: ")
            print(roi_label)      

    return layer_masks_bool, BG_mask                              
                                
                        
def select_regions(image_to_select_from, image_cmap ="gray",pause_t=7,
                   ask_name=True,roi_type=None):
    """ Enables user to select rois from a given image using roipoly module.

    Parameters
    ==========
    image_to_select_from : numpy.ndarray
        An image to select ROIs from
    
    roi_type: string
        if Different kinds of categories are selected 
        (incl: to select and inclusion ROI) (cat: to select category ROIs)
        (None: to select any kind of arbitrary ROI type)
        default is None

    Returns
    =======
    
    """
    import warnings 
    plt.close('all')
    stopsignal = 0
    roi_number = 0
    roi_masks = []
    mask_names = []
    
    im_xDim = np.shape(image_to_select_from)[0]
    im_yDim = np.shape(image_to_select_from)[1]
    mask_agg = np.zeros(shape=(im_xDim,im_yDim))
    iROI = 0
    plt.style.use("dark_background")
    while (stopsignal==0):

        
        # Show the image
        fig = plt.figure()
        plt.imshow(image_to_select_from, interpolation='nearest', cmap=image_cmap)
        plt.colorbar()
        plt.imshow(mask_agg, alpha=0.3,cmap = 'tab20b')
        if  roi_type=='Incl':
            plt.title("Select inclusion zone %d" % roi_number)
            
        if roi_type=='cat':
            plt.title("Select category ROI %d" % roi_number)
        else:
            plt.title("Select ROI: ROI%d" % roi_number)
        plt.show(block=False)
        
        
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        plt.waitforbuttonpress()
        plt.pause(pause_t)
        #plt.close()
        if ask_name:
            mask_name = raw_input("\nEnter the ROI name:\n>> ")
        else:
            mask_name = iROI
        curr_mask = curr_roi.get_mask(image_to_select_from)
        if len(np.where(curr_mask)[0]) ==0 :
            warnings.warn('ROI empty.. discarded.')
            iROI = iROI - 1
            if iROI==0:
                print('select again')
            continue
        mask_names.append(mask_name)
        
        
        roi_masks.append(curr_mask)
        
        mask_agg[curr_mask] += 1
                
        roi_number += 1
        if roi_type!='Incl':
            signal = raw_input("\nPress k for exiting program, otherwise press enter")
        else:
            signal = 'k'
        if (signal == 'k'):
            stopsignal = 1
        
    
    return roi_masks, mask_names

def clusters_restrict_size_regions(rois,cluster_1d_max_size_pixel,
                                cluster_1d_min_size_pixel,
                                cluster_region_bool=None,  
                                otsu_thresholded_mask=None,no_filter=False): #Juan edit: otsu=None
    """
    """    
    
    # Getting rid of clusters based on pre-defined regions and size
    passed_rois  = []
    ROI_mod.calcualte_mask_1d_size(rois)
    if no_filter==False:
        for roi in rois:
            # Check if mask is within the pre-defined regions
            if otsu_thresholded_mask is  not None:
                otsu_inclusion_points =  np.where(roi.mask * otsu_thresholded_mask)[0]
            if cluster_region_bool is not None:
                mask_inclusion_points =  np.where(roi.mask * cluster_region_bool)[0]
                if (mask_inclusion_points.size == np.where(roi.mask)[0].size) and\
                    (otsu_inclusion_points.size>(roi.mask.sum()/2)): 
                    pass 
            # Check if mask is within the size restrictions
            if ((roi.x_size < cluster_1d_max_size_pixel) & (roi.y_size < cluster_1d_max_size_pixel) & \
                (roi.x_size > cluster_1d_min_size_pixel) & (roi.y_size > cluster_1d_min_size_pixel)): #Juan edit. changed indentation
                    
                    passed_rois.append(roi)
    elif no_filter==True:
        passed_rois=rois            
            
    # Generating an image with masks
    data_xDim = np.shape(rois[0].mask)[0]
    data_yDim = np.shape(rois[0].mask)[1]
    
    passed_rois_image = np.zeros(shape=(data_xDim,data_yDim))
    passed_rois_image[:] = np.nan
    for index, roi in enumerate(passed_rois):
        passed_rois_image[roi.mask.astype(bool)] = index+1
    
    print('Clusters excluded based on layers...')
    
    all_pre_selected_mask = np.zeros(shape=(data_xDim,data_yDim))

    pre_selected_roi_indices = np.arange(len(passed_rois))
    pre_selected_roi_indices_copy = np.arange(len(passed_rois))
    
    for index, roi in enumerate(passed_rois):
        all_pre_selected_mask[roi.mask.astype(bool)] += 1
        
    # Getting rid of overlapping clusters
    if no_filter==False:
        while len(np.where(all_pre_selected_mask>1)[0]) != 0:
            
            for index, roi_idx in enumerate(pre_selected_roi_indices):
                
                if pre_selected_roi_indices[index] != -1:
                    curr_mask = passed_rois[roi_idx].mask
                    non_intersection_matrix = \
                        (all_pre_selected_mask[curr_mask] == 1)
                    
                    if len(np.where(non_intersection_matrix)[0]) == 0: 
                        # get rid of cluster if it doesn't have any non overlapping part
                        pre_selected_roi_indices[index] = -1
                        all_pre_selected_mask[curr_mask] -= 1
                        
                    elif (len(np.where(non_intersection_matrix)[0]) != len(all_pre_selected_mask[curr_mask])): 
                        # get rid of cluster if it has any overlapping part
                        pre_selected_roi_indices[index] = -1
                        all_pre_selected_mask[curr_mask] -= 1
                else:
                    continue
    

    # To retrieve some clusters if there are no overlaps 
    for iRep in range(100):
        
        for index, roi in enumerate(pre_selected_roi_indices):
            if pre_selected_roi_indices[index] == -1:
        #        print(index)
                curr_mask = passed_rois[pre_selected_roi_indices_copy[index]].mask
                non_intersection_matrix = (all_pre_selected_mask[curr_mask] == 0)
                if (len(np.where(non_intersection_matrix)[0]) == len(all_pre_selected_mask[curr_mask])):
                    # If there's no cluster here add the cluster back
                    print('cluster added back')
                    pre_selected_roi_indices[index] = pre_selected_roi_indices_copy[index]
                    all_pre_selected_mask[curr_mask] += 1
    
    separated_roi_indices = pre_selected_roi_indices[pre_selected_roi_indices != -1]
    sep_masks_image = np.zeros(shape=(data_xDim,data_yDim))
    sep_masks_image[:] = np.nan
    separated_rois = []
    
    for index, sep_clus_idx in enumerate(separated_roi_indices):
        sep_masks_image[passed_rois[sep_clus_idx].mask.astype(bool)] = index+1
        
        separated_rois.append(passed_rois[sep_clus_idx])
    
    print('Clusters separated...')
    print('Cluster pass ratio: %.2f' % (float(len(separated_rois))/\
                                        float(len(rois))))
    print('Total clusters: %d'% len(separated_rois))
    
    return separated_rois, sep_masks_image
        


def separate_trials_ROI_v3(time_series,rois,stimulus_information,
                           frameRate, df_method, df_use = True, plotting=False,
                           max_resp_trial_len = 'max'):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces. Adds responses and whole traces into the ROI_bg 
    instances.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch. Trials consist of 
        previous baseline epoch _ stimulus epoch _ following baseline epoch
        (if there is a baseline presentation)
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.
    
    rois : list
        A list of ROI_bg instances.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    df_method : str
        Method for calculating dF/F defined in the ROI_bg class.
        
    plotting: bool
        If the user wants to visualize the masks and the traces for clusters.
        
    Returns
    =======
    wholeTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
            
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    """
    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    baselineTraces_allTrials_ROIs = {}
    all_clusters_dF_whole_trace = np.zeros(shape=(len(rois),
                                                  np.shape(time_series)[0]))
    
    # dF/F calculation
    for iROI, roi in enumerate(rois):
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method,moving_avg = True, bins = 3)
        all_clusters_dF_whole_trace[iROI,:] = roi.df_trace
        
        if df_use:
            roi.base_dur = [] # Initialize baseline duration here (not good practice...)
        if plotting:
            plt.figure(figsize=(8, 7))
            grid = plt.GridSpec(8, 1, wspace=0.4, hspace=0.3)
            plt.subplot(grid[:7,0])
            roi.showRoiMask()
            plt.subplot(grid[7:8,0])
            plt.plot(roi.df_trace)
            plt.title('Cluster %d %s:' % (iROI, roi))
            
            
            plt.waitforbuttonpress()
            plt.close('all')
            
    trialCoor = stimulus_information['trial_coordinates']
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        resp_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            base_len = baselineEnd - baselineStart
            
            base_lens.append(base_len) 
            
            resp_start = curr_trial_coor[0][0]+base_len
            resp_end = curr_trial_coor[0][1]-base_len
            resp_lens.append(resp_end-resp_start)
        
        trial_len =  min(trial_lens)
        resp_len = min(resp_lens)
        base_len = min(base_lens)
        
        if not((max_resp_trial_len == 'max') or \
               (current_epoch_dur < max_resp_trial_len)):
            resp_len = int(round(frameRate * max_resp_trial_len))+1
            
        wholeTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs[iEpoch] = {}
        baselineTraces_allTrials_ROIs[iEpoch] = {}
   
        for iCluster, roi in enumerate(rois):
            
            if stimulus_information['random']:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            else:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [stimulus_information['baseline_epoch']]\
                                     [iCluster])[0]
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(int(frameRate*1.5),
                                   trial_numbers))
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if stimulus_information['random']:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_resp = roi.df_trace[respStart:epochEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                    
                            
                    wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:base_len]
                else:
                    
                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        
                    
                    if iEpoch == stimulus_information['baseline_epoch']:
                        baseline_trace = roi_whole_trace[:base_len]
                        baseline_trace = baseline_trace[-int(frameRate*1.5):]
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= baseline_trace
                    else:
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster]\
                            [:,trial_num]= baselineTraces_allTrials_ROIs\
                            [stimulus_information['baseline_epoch']][iCluster]\
                            [:,trial_num]
                    
                    wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
    
                    
    for iEpoch in trialCoor:
        for iCluster, roi in enumerate(rois):
            
            # Appending trial averaged responses to roi instances only if 
            # df is used
            if df_use:
                if not stimulus_information['random']:
                    if iEpoch > 0 and iEpoch < len(trialCoor)-1:
                        
                        wt = np.concatenate((wholeTraces_allTrials_ROIs[iEpoch-1][iCluster].mean(axis=1),
                                            wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1),
                                            wholeTraces_allTrials_ROIs[iEpoch+1][iCluster].mean(axis=1)),
                                            axis =0)
                        roi.base_dur.append(len(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster].mean(axis=1))) 
                    else:
                        wt = wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1)
                        roi.base_dur.append(0) 
                else:
                    wt = wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1)
                    base_dur = frameRate * stimulus_information['baseline_duration']
                    roi.base_dur.append(int(round(base_dur)))
                    
                roi.appendTrace(wt,iEpoch, trace_type = 'whole')
                roi.appendTrace(respTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1),
                                  iEpoch, trace_type = 'response' )
                    
                    
                
        
    if df_use:
        print('Trial separation for ROIs completed')
    else:
        print('Trial separation not done (df not calculated)')
    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs, 
            baselineTraces_allTrials_ROIs, all_clusters_dF_whole_trace) 

def calculate_background_ROIs(bg_mask,rois,in_phase=False):
    """
    introduce a background ROI into the roi instances

    parameters
    ===========

    bg_mask: background mask (2d bolean numpy array)

    rois: roi_bg class instance

    In phase: (bool) if True this is intended to compensate  background from the ultima
              that is structured in the y axis. this chooses a background region spanning the same places
              in y as the roi.
    """

    for roi in rois:
        if in_phase:
            y_position=np.where(roi.mask,)[0]
            filter_bg=np.zeros(np.shape(roi.mask))
            filter_bg[y_position,:]=1
            roi.bg_mask = bg_mask*filter_bg
            if np.sum(roi.bg_mask)==0:
                raise Exception ('there is no overlap between background region and roi in y axis')
        else:
            roi.bg_mask = bg_mask

def separate_trials_ROI_v4(time_series,rois,stimulus_information,frameRate, 
                           df_method = None, df_use = True,
                           max_resp_trial_len = 'max',mov_av=False,add_prev_traces=False,analysis_type=None):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces. Adds responses and whole traces into the ROI_bg 
    instances.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch. Trials consist of 
        previous baseline epoch _ stimulus epoch _ following baseline epoch
        (if there is a baseline presentation)
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.
    
    rois : list
        A list of ROI_bg instances.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    df_method : str
        Method for calculating dF/F defined in the ROI_bg class.
        
    plotting: bool
        If the user wants to visualize the masks and the traces for clusters.
        
    Returns
    =======
    wholeTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
            
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    """
    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    baselineTraces_allTrials_ROIs = {}
    
    # 
    for iROI, roi in enumerate(rois):
        #if iROI==22:
        #    'flag'

        if add_prev_traces: #juan edit # this part is not in use anymore. deprecated
            roi.appendPrevTraces()  
        #roi.mask=np.where(roi.mask,1,np.nan)
        local_mask=np.where(roi.mask,1,np.nan)
        roi.original_trace = np.nanmean(time_series*local_mask[np.newaxis,:,:],axis=(1,2))
        roi.BG_trace = np.sum(time_series*roi.bg_mask[np.newaxis,:,:],axis=(1,2))/np.sum(roi.bg_mask)
        roi.raw_trace = roi.original_trace-roi.BG_trace       
        roi.calculateDf(stimulus_information,df_method,mov_av,3)
        #plt.figure()
        #plt.plot(roi.raw_trace)
        #plt.title(roi.category)
        if df_use:
            roi.base_dur = [] # Initialize baseline duration here for upcoming analysis
    print('\n Background subtraction done...')
    print('df/f method: %s'%(df_method))
    plt.close('all')
    trialCoor = stimulus_information['trial_coordinates']
    # Trial averaging by loooping through epochs and trials
    if 'STRF' in analysis_type:
        for roi in rois:
            #period = roi.imaging_info['FramePeriod']  
            start_of_stim_frame = roi.stim_info['trial_coordinates'][0][0][0]
            #len_subepoch = roi.stim_info['tau'][0]
            #subepoch_frames = int(np.floor(len_subepoch/period))

            trace = roi.df_trace[start_of_stim_frame:]
            roi.white_noise_response = trace
        return None,None,None
    
    if 'Frozen' in analysis_type:
        for roi in rois:    
            #period = roi.imaging_info['FramePeriod']  
            start_of_stim_frame = roi.stim_info['trial_coordinates'][0][0][0]
            #len_subepoch = roi.stim_info['tau'][0]
            #subepoch_frames = int(np.floor(len_subepoch/period))

            trace = roi.df_trace[start_of_stim_frame:]
            roi.white_noise_response = trace
        
        return None,None,None

    for iEpoch in trialCoor:
        
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        resp_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            # current_trial_length = curr_trial_coor[1]-curr_trial_coor[0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            base_len = baselineEnd - baselineStart
            
            base_lens.append(base_len) 
            
            resp_start = curr_trial_coor[0][0]+base_len
            resp_end = curr_trial_coor[0][1]-base_len
            resp_lens.append(resp_end-resp_start)
        
        trial_len =  min(trial_lens)
        resp_len = min(resp_lens)
        base_len = min(base_lens)
        
        if not((max_resp_trial_len == 'max') or \
               (current_epoch_dur < max_resp_trial_len)):
            resp_len = int(round(frameRate * max_resp_trial_len))+1
            
        wholeTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs[iEpoch] = {}
        baselineTraces_allTrials_ROIs[iEpoch] = {}
   
        for iCluster, roi in enumerate(rois):
            if iCluster==0 and iEpoch==2:
                'flaggg'
            # Baseline epoch is presented only when random value = 0 and 1 
            if stimulus_information['random'] == 1:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            elif stimulus_information['random'] == 0:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [stimulus_information['baseline_epoch']]\
                                     [iCluster])[0]
                #baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(int(frameRate*1.5),
                #                   trial_numbers))
            else:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if stimulus_information['random'] == 1:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_resp = roi.df_trace[respStart:epochEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                            
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:base_len]
                elif stimulus_information['random'] == 0:
                    
                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        
                    
                    # if iEpoch == stimulus_information['baseline_epoch']:
                    #     baseline_trace = roi_whole_trace[:base_len]
                    #     baseline_trace = baseline_trace[-int(frameRate*1.5):]
                    #     #baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= baseline_trace
                    # else:
                    #     baselineTraces_allTrials_ROIs[iEpoch][iCluster]\
                    #         [:,trial_num]= baselineTraces_allTrials_ROIs\
                    #         [stimulus_information['baseline_epoch']][iCluster]\
                    #         [:,trial_num]
                    
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        
                else:
                    # If the sequence is all random the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                    
                    try:
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]

                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace
                        respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= new_trace


    
    for iCluster, roi in enumerate(rois):
        # if (roi.experiment_info['expected_polarity']=='OFF' and df_method=='postpone'): # get rid of first trial in order to pair up the baseline trial correctly to the response trial
        #     wholeTraces_allTrials_ROIs[0][iCluster]= wholeTraces_allTrials_ROIs[0][iCluster][:,1:]
        for iEpoch in trialCoor:        #Juan edit (switched order of loops)
            #calculate df/f if traces come from FFF stimulation and df calculation has been postponed

 

                #calculate df_f if it was postponed
                #TODO, extract sampling rate and calculate 1 or 2 seconds
                # if roi.experiment_info['expected_polarity']=='ON':
                #     if iEpoch==0:
                #         f0= np.nanmean(wholeTraces_allTrials_ROIs[0][iCluster][-20:,:len(trialCoor[iEpoch])],axis=0) #TODO make the slice fixed in time
                #     else:
                #         f0=f0[:len(trialCoor[iEpoch])]
                #     wholeTraces_allTrials_ROIs[iEpoch][iCluster]=(wholeTraces_allTrials_ROIs[iEpoch][iCluster]-f0[np.newaxis,:])/f0[np.newaxis,:]
                # elif roi.experiment_info['expected_polarity']=='OFF':
                #     if iEpoch==0:

                #         f0= np.nanmean(wholeTraces_allTrials_ROIs[1][iCluster][-20:,:wholeTraces_allTrials_ROIs[iEpoch][iCluster].shape[1]],axis=0)
                #     else:
                #         f0=f0
                #     wholeTraces_allTrials_ROIs[iEpoch][iCluster]=(wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,:len(f0)]-f0[np.newaxis,:])/f0[np.newaxis,:]
                # else:
                #     raise Exception('expected polarity not found')
                
            # Appending trial averaged responses to roi instances only if 
            # df is used

            if df_use:
                if stimulus_information['random'] == 0:

                    if iEpoch > 0 and iEpoch < len(trialCoor)-1:
                        
                        wt = np.concatenate((np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1),
                                            np.nanmean(wholeTraces_allTrials_ROIs[iEpoch+1][iCluster],axis=1)),
                                            axis =0)
                        roi.base_dur.append(len(np.nanmean(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster],axis=1)))
                    else:
                        if (roi.experiment_info['expected_polarity']=='OFF' and df_method=='postpone' and iEpoch==0) : # get rid of first trial where no response is expected
                            wholeTraces_allTrials_ROIs[0][iCluster]= wholeTraces_allTrials_ROIs[0][iCluster][:,1:]

                        wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)

                        #if df_f was postponed, calculate df/f with the trial averaged traces

                        # if (stimulus_information['stim_name']=='LocalCircle_5sec_220deg_0degAz_0degEl_Sequential_LumDec_LumInc.txt'\
                        #     or stimulus_information['stim_name']=='LocalCircle_5sec_120deg_0degAz_0degEl_Sequential_LumDec_LumInc_10sec.txt'\
                        #     or stimulus_information['stim_name']=='LocalCircle_5sec_120deg_0degAz_0degEl_Sequential_LumDec_LumInc.txt')\
                        #     and df_method=='postpone':     
                                                        
                        #     f0_interval=int(round(roi.imaging_info['frame_rate']*1))
                            
                        #     if roi.experiment_info['expected_polarity']=='ON':
                        #         if iEpoch==0:
                        #             f0= np.nanmean(wt[-1*f0_interval:],axis=0)
                        #         #wt= (wt-f0)/f0
                        #     elif roi.experiment_info['expected_polarity']=='OFF':
                                
                        #         if iEpoch==0:
                        #             wt1= np.nanmean(wholeTraces_allTrials_ROIs[1][iCluster],axis=1)
                        #             f0= np.nanmean(wt1[-1*f0_interval:],axis=0)
                        #         #wt= (wt-f0)/f0  

                        roi.base_dur.append(0) 

                elif stimulus_information['random'] == 1:
                    wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)
                    base_dur = frameRate * stimulus_information['baseline_duration']
                    roi.base_dur.append(int(round(base_dur)))
                else:

                    #### this is trial averaging
                    wt = np.nanmean(wholeTraces_allTrials_ROIs[iEpoch][iCluster],axis=1)
                #### this is trial averaging
                roi.appendTrace(wt,iEpoch, trace_type = 'whole')
                roi.appendTrace(np.nanmean(respTraces_allTrials_ROIs[iEpoch][iCluster],axis=1),
                                  iEpoch, trace_type = 'response' )
                if np.any(np.isnan(respTraces_allTrials_ROIs[iEpoch][iCluster])):
                    'flaaag'
                roi.appendTrace(wholeTraces_allTrials_ROIs[iEpoch][iCluster],iEpoch, trace_type = 'raw')
                    
                
        
    if df_use:
        print('Traces are stored in ROI objects.')
    else:
        print('No trace is stored in objects.')
    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs, 
            baselineTraces_allTrials_ROIs)                 
        
def plot_roi_properties(images, properties, colormaps,underlying_image,vminmax,
                        exp_ID,depth,save_fig = False, save_dir = None,
                        figsize=(10, 6),alpha=0.5):
    """ 
    Parameters
    ==========
    
        
    Returns
    =======

    """
    plt.close('all')
    run_matplotlib_params()    
    #plt.style.use('dark_background')
    #plt.style.use('seaborn')
    total_n_images = len(images)
    col_row_n = math.ceil(math.sqrt(total_n_images))
    
    fig1, ax1 = plt.subplots(ncols=col_row_n, nrows=col_row_n, sharex=True, 
                             sharey=True,figsize=figsize)
    depthstr = 'Z: %d' % depth
    figtitle = 'ROIs summary: ' + depthstr
    fig1.suptitle(figtitle,fontsize=16)
    
    for idx, ax in enumerate(ax1.reshape(-1)): 
        if idx >= total_n_images:
            ax.axis('off')
        else:
            sns.heatmap(underlying_image,cmap='gray',ax=ax,cbar=False)
            
            sns.heatmap(images[idx],alpha=alpha,cmap = colormaps[idx],ax=ax,
                        cbar=True,cbar_kws={'label': properties[idx]},
                        vmin = vminmax[idx][0], vmax=vminmax[idx][1])
            ax.axis('off')
    
    if save_fig:
        # Saving figure
        # save_name = 'ROI_props_%s' % (exp_ID)
        # os.chdir(save_dir)
        plt.savefig(f'{save_dir}_ROI_props.png', bbox_inches='tight',dpi=300)
        print('ROI property images saved')
        

def plot_roi_masks(roi_image, underlying_image,n_roi1,exp_ID,
                       save_fig = False, save_dir = None,alpha=0.5):
    """ Plots two different cluster images underlying an another common image.
    Parameters
    ==========
    first_clusters_image : numpy array
        An image array where clusters (all from segmentation) have different 
        values.
    
    second_cluster_image : numpy array
        An image array where clusters (the final ones) have different values.
        
    underlying_image : numpy array
        An image which will be underlying the clusters.
        
    Returns
    =======

    """

    plt.close('all')
    #plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(ncols=1, nrows=1,facecolor='k', edgecolor='w',
                             figsize=(5, 5))
    
    # All clusters
    sns.heatmap(underlying_image,cmap='gray',ax=ax1,cbar=False)
    sns.heatmap(roi_image,alpha=alpha,cmap = 'tab20b',ax=ax1,
                cbar=False)
    
    ax1.axis('off')
    ax1.set_title('ROIs n=%d' % n_roi1)
    
    if save_fig:
        # Saving figure
        save_name = 'ROIs_%s' % (exp_ID)
        # os.chdir(save_dir)
        plt.savefig('%s.png'% (save_name), bbox_inches='tight')
        print('ROI images saved')


def plot_raw_responses_stim(responses, rawStimData, exp_ID, save_fig =False, 
                            save_dir = None, ax_to_plot =None):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    responses : n x m numpy array
        Response traces along the row dimension. (n ROIs, m time points)
    
    rawStimData : numpy array
        Raw stimulus output data where the frames and stim values are stored.
        
    Returns
    =======
    

    """
    
    adder = np.linspace(0, np.shape(responses)[0]*4, 
                        np.shape(responses)[0])[:,None]
    scaled_responses = responses + adder
    
    # Finding stimulus
    
    stim_frames = rawStimData[:,7]  # Frame information
    stim_vals = rawStimData[:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]
    # Make normalized values of stimulus values for plotting
    
    stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
        *np.max(scaled_responses)/3
    stim_df = pd.DataFrame(stim_vals,columns=['Stimulus'],dtype='float')
    
    resp_df = pd.DataFrame(np.transpose(scaled_responses),dtype='float')
    
    if ax_to_plot is None:
        ax = resp_df.plot(legend=False,alpha=0.8,lw=0.5)
    else:
        ax = ax_to_plot
        resp_df.plot(legend=False,alpha=0.8,lw=0.5,ax=ax_to_plot)
        
    stim_df.plot(dashes=[2, 1],ax=ax,color='w',alpha=.8,lw=2)
    plt.title('Responses (N:%d)' % np.shape(responses)[0])
    
    if save_fig:
        # Saving figure
        save_name = 'ROI_traces%s' % (exp_ID)
        # os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('All traces figure saved')

 

def calculate_SNR_Corr(base_traces_all_roi, resp_traces_all_roi,
                       rois, epoch_to_exclude = None):
    """ Calculates the signal-to-noise ratio (SNR). Equation taken from
    Kouvalainen et al. 1994 (see calculation of SNR true from SNR estimated).
    Also calculates the correlation between the first and the last trial to 
    estimate the reliability of responses.
    
    
    Parameters
    ==========
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    rois : list
        A list of ROI_bg instances.
        
    epoch_to_exclude : int 
        Default: None
        Epoch number to exclude when calculating corr and SNR
        
        
    Returns
    =======
    
    SNR_max_matrix : np array
        SNR values for all ROIs.
        
    Corr_matrix : np array
        SNR values for all ROIs.
        
    """
    total_epoch_numbers = len(base_traces_all_roi)
    
    SNR_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))
    Corr_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))
    
    for iROI, roi in enumerate(rois):
        
        for iEpoch, iEpoch_index in enumerate(base_traces_all_roi):
            
            if iEpoch_index == epoch_to_exclude:
                SNR_matrix[iROI,iEpoch] = 0
                Corr_matrix[iROI,iEpoch] = 0
                continue
            
            trial_numbers = np.shape(resp_traces_all_roi[iEpoch_index][iROI])[1]
            
            
            ##### juan comment ####
            #currentBaseTrace = base_traces_all_roi[iEpoch_index][iROI][:,:]
            ### end #####
            
            currentRespTrace =  resp_traces_all_roi[iEpoch_index][iROI][:,:]
            
            # Reliability between all possible combinations of trials
            perm = permutations(range(trial_numbers), 2) 
            coeff =[]
            for iPerm, pair in enumerate(perm):
                curr_coeff, pval = pearsonr(currentRespTrace[:-2,pair[0]],
                                            currentRespTrace[:-2,pair[1]])
                coeff.append(curr_coeff)
                
            coeff = np.array(coeff).mean()
            
            #### Juan comment ####
            #noise_std = currentBaseTrace.std(axis=0).mean(axis=0)
            #### end

            resp_std = currentRespTrace.std(axis=0).mean(axis=0)

            ##### Juan comment ####
            #signal_std = resp_std - noise_std
            ### end ###

            # SNR calculation taken from
            
            ###### juan comment ###
            #curr_SNR_true = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
            #SNR_matrix[iROI,iEpoch] = curr_SNR_true
            #### end ####
            
            Corr_matrix[iROI,iEpoch] = coeff
        try:
            roi.SNR
        except:
            roi.SNR=[]
        finally:
            pass
        try:
            roi.reliability
        except:
            roi.reliability=[]
        finally:
            pass           

        
        roi.SNR.append(np.nanmax(SNR_matrix[iROI,:]))
        roi.SNR.append(None)
        

        roi.reliability.append(np.nanmax(Corr_matrix[iROI,:])) #Juan edit: turned this variables into lists, for multiple cycles
    
     
    SNR_max_matrix = np.nanmax(SNR_matrix,axis=1) 
    Corr_matrix = np.nanmax(Corr_matrix,axis=1)
    
    return SNR_max_matrix, Corr_matrix

def append_snr_reliability(rois,SNR_rois,corr_rois,name_snr='SNR',name_rel='reliability'): #juan function
    localSNR=SNR_rois.copy()
    localrel=corr_rois.copy()
    for i, roi in enumerate(rois):
        roi.append_SNR_Reliability(name_snr,name_rel,localSNR[i],localrel[i])

def plot_df_dataset(df, properties, save_name = 'ROI_plots_%s', 
                    exp_ID=None, save_fig = False, save_dir=None):
    """ Plots a variable against 3 other variables

    Parameters
    ==========
   
     
    Returns
    =======
    
    
    """
    plt.close('all')
    colors = run_matplotlib_params()    
    if len(properties) < 5:
        dim1= len(properties)
        dim2 = 1
    elif len(properties)/5.0 >= 1.0:
        dim1 = 5
        dim2 = int(np.ceil(len(properties)/5.0))
        
    fig1, ax1 = plt.subplots(ncols=dim1, nrows=dim2,figsize=(10, 3))
    axs = ax1.flatten()
    
    for idx, ax in enumerate(axs):
        try:
            sns.distplot(df[properties[idx]],ax=ax,color=plt.cm.Dark2(3),rug=True,
                         hist=False)
        except:
            continue
    
    if save_fig:
            # Saving figure
            save_name = 'ROI_plots_%s' % (exp_ID)
            os.chdir(save_dir)
            plt.savefig('%s.png'% save_name, bbox_inches='tight',dpi=300)
            print('ROI properties saved')
    return None


def run_roi_transfer(transfer_data_path, transfer_type,experiment_info=None,
                     imaging_info=None):
    '''
    
    Updates:
        25/03/2020 - Removed transfer types of 11 steps and AB steps since they
        are redundant with minimal type
    '''
    load_path = open(transfer_data_path, 'rb')
    workspace = cPickle.load(load_path)
    rois = workspace['final_rois']
    
    if transfer_type == 'luminance_gratings' or \
        transfer_type == 'lum_con_gratings' :
        
        properties = ['CSI', 'CS','PD','DSI','category',
                      'analysis_params']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'stripes_OFF_delay_profile':
        
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS='OFF')
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
        
    elif transfer_type == 'stripes_ON_delay_profile':
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS='ON')

        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))

    elif ((transfer_type == 'stripes_ON_vertRF_transfer') or \
          (transfer_type == 'stripes_ON_horRF_transfer') or \
          (transfer_type == 'stripes_OFF_vertRF_transfer') or \
          (transfer_type == 'stripes_OFF_horRF_transfer')):
        properties = ['corr_fff', 'max_response','category','analysis_params']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'ternaryWN_elavation_RF'):
        properties = ['corr_fff', 'max_response','category','analysis_params',
                      'reliability','SNR']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'gratings_transfer_rois_save'):
        
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        
    elif (transfer_type == 'luminance_edges_OFF' ):
        if (('R64G09' in rois[0].experiment_info['Genotype']) or \
         ('T5' in rois[0].experiment_info['Genotype'])):
            CS = 'OFF'
            warnings.warn('Transferring only T5 neurons')
        else:
            CS = None
            warnings.warn('NO CS selected since genotype is not found to be T4-5')
        
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS=CS)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'luminance_edges_ON'):
        
        if (('R64G09' in rois[0].experiment_info['Genotype']) or \
         ('T4' in rois[0].experiment_info['Genotype'])):
            CS = 'ON'
            warnings.warn('Transferring only T4 neurons')
        else:
            CS = None
            warnings.warn('NO CS selected since genotype is not found to be T4-5')
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS=CS)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
   
    elif transfer_type == 'STF_1':
        properties = ['CSI', 'CS','PD','DSI','category','analysis_params']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'minimal' :
        print('Transfer type is minimal... Transferring just masks, categories and if present RF maps...\n')
        properties = ['category','analysis_params','RF_maps','RF_map',
                      'RF_center_coords','RF_map_norm']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    else:
        raise NameError('Invalid ROI transfer type')
        
        
   
    return transferred_rois

def run_analysis(analysis_params, rois,experiment_conditions,
                 imaging_information,summary_save_dir, cycle, expected_polarity, mean_image,
                 save_fig=True,fig_save_dir = None, 
                 exp_ID=None,df_method=None,keep_prev=False,**kwargs): #edited by Juan
    """
    asd
    """
    
    analysis_type = analysis_params['analysis_type'][0]
    figtitle = 'Summary: %s Gen: %s | Age: %s | Z: %d' % \
           (experiment_conditions['MovieID'], #.split('-')[0] juan commented
            experiment_conditions['treatment'], experiment_conditions['Age'],
            imaging_information['depth'])

    if cycle==1:
        'flag'
    
    for roi in rois:
      if 'STRF' in analysis_type or 'Frozen_noise' in analysis_type:
        roi.setSourceImage(mean_image)
      else:
        roi.calculate_reliability()
        roi.calculate_stim_signal_correlation()
        roi.findMaxResponse_all_epochs(cycle,polarity=expected_polarity) #polarity is relevant for FFF stimuli. if inverted neurons are present or stimulus is different it should be None
        roi.setSourceImage(mean_image)
    
    if analysis_type == 'A_B_steps':
        
        rois = ROI_mod.analyze_A_B_step(rois,int_rate = 10)
        
        if ("3OFF" not in rois[0].stim_name) and \
            ("varyingDur" not in rois[0].stim_name):
            roi_image = ROI_mod.get_masks_image(rois)
            fig, fig2 = sf.make_exp_summary_AB_steps(figtitle,rois,
                                                      roi_image,
                                                      experiment_conditions['MovieID'],
                                                      summary_save_dir)
            f_n = 'Summary_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig.savefig('%s.png'% f_n, bbox_inches='tight',
                       transparent=False,dpi=300)
            
            f_n = 'Summary_%s_lum_con' % (exp_ID)
            os.chdir(fig_save_dir)
            fig2.savefig('%s.png'% f_n, bbox_inches='tight',
                       transparent=False,dpi=300)
    
        
    elif analysis_type == '11_steps_luminance':
        
        rois = ROI_mod.analyze_luminance_steps(rois,int_rate = 10)
        roi_image = ROI_mod.get_masks_image(rois)
        fig = sf.make_exp_summary_luminance_steps(figtitle,rois,
                                                  roi_image,
                                                  experiment_conditions['MovieID'],
                                                  summary_save_dir)
        f_n = 'Summary_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f_n, bbox_inches='tight',
                   transparent=False,dpi=300)
        
    elif analysis_type == 'luminance_gratings':
        
        if ('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype']) and \
            (not('1D' in rois[0].stim_name)):
            for roi in rois:
              roi.calculate_DSI_PD(method='PDND')
            
        rois = ROI_mod.analyze_luminance_gratings(rois)
        # rois = ROI_mod.analyze_luminance_gratings_1Hz(rois)
        run_matplotlib_params()
        mean_TFL = np.mean([np.array(roi.tfl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        mean = pd.DataFrame({})
        for n, i in enumerate(mean_TFL):
            mean_single = pd.DataFrame({n:i[2]}, index=[1])
            mean = pd.concat([mean, mean_single], axis=1)
            # mean.concat([mean, mean-single], axis=1)
            # print(i)
        ax=sns.heatmap(mean, cmap='coolwarm',center=0,
                    #    xticklabels=np.array(rois[0].tfl_map.columns.levels[1]).astype(float),
                    #    yticklabels=np.array(rois[0].tfl_map.index),
                       cbar_kws={'label': 'Delta F/F'})
        ax.invert_yaxis()
        plt.title('TFL map')
        
        fig = plt.gcf()
        f0_n = 'Summary_TFL_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig(f'{exp_ID}_summary_TFL.png', bbox_inches='tight',
                    transparent=False,dpi=300)
        
    elif analysis_type == 'lum_con_gratings':
        
        rois = ROI_mod.analyze_lum_con_gratings(rois)
        run_matplotlib_params()
        mean_CL = np.mean([np.array(roi.cl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        
        ax=sns.heatmap(mean_CL, cmap='coolwarm',center=0,
                       xticklabels=np.array(rois[0].cl_map.columns.levels[1]).astype(float),
                       yticklabels=np.array(rois[0].cl_map.index),
                       cbar_kws={'label': 'Delta F/F'})
        ax.invert_yaxis()
        plt.title('CL map')
        
        fig = plt.gcf()
        f0_n = 'Summary_CL_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False,dpi=300)
        
    elif analysis_type == '8D_10dps_stripes_RF':
        
        rois = ROI_mod.map_RF_adjust_stripe_time(rois,screen_props = {'45':74, '135':72,
                                                   '225':74,'315':72,
                                                   '0':53,'180':53,
                                                   '90':78,'270':78},
                                               delay_use=False)
    
        # random.shuffle(rois_plot)
        
        fig1 = ROI_mod.plot_RFs(rois, number=len(rois), f_w =5,cmap='coolwarm',
                                center_plot = True, center_val = 0.95)
        
        fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop=None, cmap='tab20b',
                              ylab='ROI num',lims=(1,len(rois)))
        fig3 = ROI_mod.plot_RF(rois[random.randint(0,len(rois)-1)],
                                   cmap1='coolwarm',cmap2='inferno')
        f3_n = 'BT_roi_example_%s' % (exp_ID)
        fig3.savefig('%s.png'% f3_n, bbox_inches='tight',
                   transparent=False,dpi=300)
                
        
        if save_fig:
           # Saving figure 
           f1_n = 'RF_examples_%s' % (exp_ID)
           f2_n = 'RF_on_screen_%s' % (exp_ID)
           
           os.chdir(fig_save_dir)
           fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
                       transparent=False,dpi=300)
           
    
    elif (analysis_type == '2D_edges_find_rois_delay_profile_save') or\
        (analysis_type == '2D_edges_find_save'):
        
        for roi in rois:
          roi.calculate_DSI_PD(method='PDND') 
          roi.calculate_CSI(frameRate= imaging_information['frame_rate'])
          
        if analysis_type == '2D_edges_find_rois_delay_profile_save':
            rois = ROI_mod.generate_time_delay_profile_2Dedges(rois) # TODO commented by Juan but don't know if oimportant

    elif (analysis_type =='1-dir_ON_OFF'):
        for roi in rois:
          roi.calculate_CSI(frameRate=imaging_information['frame_rate'])
        
    elif ((analysis_type == 'luminance_edges_OFF' ) or\
          (analysis_type == 'luminance_edges_ON' ) or\
           analysis_type == '1-dir_edge_1pol') :
        
        if (('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype']) and\
             analysis_type != '1-dir_edge_1pol')  :
            #map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
            print ('flag for empty if statement')

        if 'contrasts' in rois[0].stim_info['stim_name']:
            variable2Analyze='contrast'
        elif 'luminances' in rois[0].stim_info['stim_name']:
            variable2Analyze='luminance'
        elif 'speeds' in rois[0].stim_info['stim_name']:
            variable2Analyze='velocity'
        #elif '8D' in rois[0].stim_info['stim_name']:        
        #    variable2Analyze = 'direction'
        else:
            print('no analysis made, variable to analize not found')

        rois = ROI_mod.analyze_lum_Cont_speed_edges(rois,variable2Analyze,analysis_type,reliability_filter=kwargs['reliability_filter'],int_rate = 11)
              
        
        #roi_image = ROI_mod.generate_colorMasks_properties(rois, 'slope')
        #fig = sf.make_exp_summary_luminance_edges(figtitle,rois,
        #                                          roi_image,
        #                                          experiment_conditions['MovieID'],
        #                                          summary_save_dir)
        
        
        #slope_data = ROI_mod.data_to_list(rois, ['slope'])['slope']
        #rangecolor= np.max(np.abs([np.min(slope_data),np.max(slope_data)]))
        
        # if 'RF_map' in rois[0].__dict__:
        #     fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='slope',
        #                                              cmap='PRGn',
        #                                              ylab='Lum sensitivity',
        #                                              lims=(-rangecolor,
        #                                                    rangecolor))
        #     f2_n = 'Slope_on_screen_%s' % (exp_ID)
        #     os.chdir(fig_save_dir)
        #     fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
        #                    transparent=False,dpi=300)
        # else:
        #     print('No RF found for the ROI.')
        
        # f1_n = 'Summary_%s' % (exp_ID)
        # os.chdir(fig_save_dir)
        # fig.savefig('%s.png'% f1_n, bbox_inches='tight',
        #                transparent=False,dpi=300)
          
    elif analysis_type == '5sFFF_analyze_save':
        # if df_method=='postpone':
        #     rois = ROI_mod.conc_traces(rois, interpolation = True, int_rate = 10,df_method='postpone')
        rois = ROI_mod.conc_traces(rois, interpolation = True, int_rate = 10,df_method=df_method)
        roi_traces = []        
        roi_conc_traces = []
        for roi in rois:
          roi_traces.append(roi.df_trace) #roi.df_trace[roi.start_id[0]:-roi.start_id[1]]
          roi_conc_traces.append(roi.conc_trace)
        
        stim_trace  = rois[0].stim_trace
        raw_stim = rois[0].stim_info['output_data']
        fig = sf.make_exp_summary_FFF(figtitle,
                             rois[0].source_image,
                             ROI_mod.get_masks_image(rois),
                             roi_traces,raw_stim,stim_trace,
                             roi_conc_traces,save_fig,
                             experiment_conditions['MovieID'],
                             summary_save_dir)
        f1_n = '5sFFF_summary_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig(f"{exp_ID}_5sFFF_summary.png",
                       transparent=False,dpi=300)
        
    elif analysis_type == '8D_edges_find_rois_save' or analysis_type == '4D_edges':
        
        for roi in rois:
          roi.calculate_DSI_PD(method='Mazurek')
          roi.calculate_CSI(frameRate=imaging_information['frame_rate'])
    
    
        rois = ROI_mod.map_RF_adjust_edge_time(rois, save_path = summary_save_dir, edges=True, delay_use = True)
               
        #rois = ROI_mod.map_RF_adjust_edge_time(rois,summary_save_dir,edges=True,
                                               #delay_degrees=9.6,
                                               #delay_use=False,
                                               #edge_props = {'45':114, '135':114,'225':114,'315':114,
                                               #     '0':80,'180':80,
                                               #     '90':80,'270':80})


    elif analysis_type == '12_dir_random_driftingstripe':
        
        for roi in rois:
          roi.calculate_DSI_PD(method='Mazurek')


        ####Important: Juan temporarily commented the receptive field mapping. not urgent 
        # to make it work (TODO)

        # rois = ROI_mod.map_RF_adjust_edge_time(rois,edges=True,
        #                                        delay_degrees=9.6,
        #                                        delay_use=True,
        #                                        edge_props = {'45':71, 
        #                                                      '135':71,
        #                                                      '225':71,'315':71, 
        #                                                      '0':51,'180':51, 
        #                                                      '90':75,'270':75})
        copy_rois = copy.deepcopy(rois)
                
        #plot_reliability_n = 0.7
        #rois_plot = [roi for roi in copy_rois if roi.reliability>plot_reliability_n]
        # random.shuffle(rois_plot)
        
        # fig1 = ROI_mod.plot_RFs(rois_plot, number=20, f_w =5,cmap='coolwarm',
        #                         center_plot = True, center_val = 0.95)
        
        # fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='PD')
        # try:
        #     fig3 = ROI_mod.plot_RF(rois_plot[random.randint(0,len(rois_plot))],
        #                            cmap1='coolwarm',cmap2='inferno')
        #     f3_n = 'BT_roi_example_%s' % (exp_ID)
        #     fig3.savefig('%s.png'% f3_n, bbox_inches='tight',
        #                transparent=False,dpi=300)
        # except:
        #     print('No roi above the reliability %.2f threshold' % plot_reliability_n)
                
        
        # if save_fig:
        #    # Saving figure 
        #    f1_n = 'RF_examples_%s' % (exp_ID)
        #    f2_n = 'RF_on_screen_%s' % (exp_ID)
           
        #    os.chdir(fig_save_dir)
        #    fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
        #                transparent=False,dpi=300)
        #    fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
        #                transparent=False,dpi=300)
           
           
    elif (analysis_type == 'stripes_OFF_delay_profile') or \
        (analysis_type == 'stripes_ON_delay_profile'):
        rois=ROI_mod.generate_time_delay_profile_combined(rois)
        
        
        if int(len(rois)/3) >10:
            f_w = 10
        else:
            f_w=int(len(rois)/3) 
            
        fig1 = ROI_mod.plot_delay_profile_examples(rois,number=None,f_w=None)
        
        if save_fig:
            # Saving figure 
            save_name = 'DelayProfiles_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig1.savefig('%s.png'% save_name, bbox_inches='tight',
                        transparent=False)
        
        filt_rois = ROI_mod.filter_delay_profile_rois(rois,Rsq_t = 0)
        data_to_extract = ['resp_delay_deg', 'resp_delay_fits_Rsq','PD']
        filt_data = ROI_mod.data_to_list(filt_rois, data_to_extract)
        mean_rsq = map(np.min,filt_data['resp_delay_fits_Rsq'])
        deg = filt_data['resp_delay_deg']
        pref_dir = filt_data['PD']
        pd_S = map(str,list(map(int,pref_dir)))
        
        df_l = {}
        df_l['deg'] = deg
        df_l['mean_rsq'] = mean_rsq
        df_l['pd'] = pd_S
        df = pd.DataFrame.from_dict(df_l) 
        
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        ax=sns.jointplot(x=deg, y=mean_rsq, kind="kde", color=plt.cm.Dark2(3))
        
        ax.plot_joint(plt.scatter, c='w', s=10, linewidth=0.5, marker="o",
                       alpha=.0)
        sns.scatterplot('deg', 'mean_rsq', hue='pd',data=df,alpha=.8,s=30,
                        palette=[plt.cm.Dark2(4), plt.cm.Dark2(2)])
        ax.set_axis_labels(xlabel='Degrees (circ)',ylabel='R^2')
        
        if save_fig:
            # Saving figure 
            save_name = 'DelayProfiles_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig1.savefig('%s.png'% save_name, bbox_inches='tight',
                        transparent=False,dpi=300)
            ax.savefig('Deg_vs_Rsq.png', bbox_inches='tight',
                        transparent=False,dpi=300)
            
    elif (analysis_type == 'stripes_ON_horRF_save_rois'):
        rois, all_rfs_sorted,max_epoch_traces  = \
            ROI_mod.generate_RF_profile_stripes(rois)
            
        ax = sns.heatmap(all_rfs_sorted)
        ax.set_xticklabels(rois[1].RF_profile_coords)
    elif ((analysis_type == 'stripes_ON_vertRF_transfer') or \
          (analysis_type == 'stripes_ON_horRF_transfer') or \
          (analysis_type == 'stripes_OFF_vertRF_transfer') or \
          (analysis_type == 'stripes_OFF_horRF_transfer')):
        
        rois = ROI_mod.generate_RF_map_stripes(rois, screen_w = 60)
        roi_traces = list(map(lambda roi: roi.df_trace, rois))
        roi_RF = list(map(lambda roi: roi.i_stripe_resp, rois))
        raw_stim = rois[0].stim_info['output_data']
        if (analysis_type == 'stripes_ON_vertRF_transfer'):
            figtitle = figtitle + '| ON_vert_RF'
        elif (analysis_type == 'stripes_ON_horRF_transfer'):
            figtitle = figtitle + '| ON_hor_RF'
        elif (analysis_type == 'stripes_OFF_vertRF_transfer'):
            figtitle = figtitle + '| OFF_vert_RF'
        elif (analysis_type == 'stripes_OFF_horRF_transfer'):
            figtitle = figtitle + '| OFF_hor_RF'
        for roi in rois:
            back_projected = np.tile(roi.i_stripe_resp, 
                                     (len(roi.i_stripe_resp),1) )
            if (analysis_type == 'stripes_ON_vertRF_transfer'):
                roi.vert_RF_ON_trace = roi.i_stripe_resp
                rotated = rotate(back_projected+1, angle=90)
                rotated= rotated-1
                roi.vert_RF_ON = rotated
                roi.vert_RF_ON_gauss = roi.stripe_gauss_profile
            elif (analysis_type == 'stripes_ON_horRF_transfer'):
                roi.hor_RF_ON_trace = roi.i_stripe_resp
                roi.hor_RF_ON = back_projected
                roi.hor_RF_ON_gauss = roi.stripe_gauss_profile
            elif (analysis_type == 'stripes_OFF_vertRF_transfer'):
                roi.vert_RF_OFF_trace = roi.i_stripe_resp
                rotated = rotate(back_projected+1, angle=90)
                rotated= rotated-1
                roi.vert_RF_OFF = rotated
                roi.vert_RF_OFF_gauss = roi.stripe_gauss_profile
            elif (analysis_type == 'stripes_OFF_horRF_transfer'):
                roi.hor_RF_OFF_trace = roi.i_stripe_resp
                roi.hor_RF_OFF = back_projected
                roi.hor_RF_OFF_gauss = roi.stripe_gauss_profile
        
        
        
        fig = sf.make_exp_summary_stripes(figtitle,analysis_params,
                                 rois[0].source_image,
                                 ROI_mod.get_masks_image(rois),
                                 roi_traces,raw_stim,roi_RF,save_fig,
                                 experiment_conditions['MovieID'],
                                 summary_save_dir)
                
        if save_fig:
            # Saving figure 
            save_name = 'RF_summary_%s' % (exp_ID)
            os.chdir(fig_save_dir)
            fig.savefig('%s.png'% save_name, bbox_inches='tight',
                        transparent=False)
    elif analysis_type == 'moving_gratings':
        #map(lambda roi: roi.calculateTFtuning_BF(), rois)
        #ROI_mod.apply_filters(rois,'NA',cycle,kwargs['CSI_filter'],kwargs['reliability_filter'],kwargs['position_filter'],kwargs['direction_filter'])
        
        for roi in rois:
          roi.calculate_freq_powers(cycle)
             
        # TODO interpolate traces for later plotting
        required_epochs=np.where(np.array(rois[0].stim_info['stimtype'])=='G')[0]
        ROI_mod.interpolation_alingment_epochs(rois,required_epochs,reliability_filter=kwargs['reliability_filter'],CSI_filter=0.45,int_rate=11,grating=True)

        #  Summary of current experiment
        #data_to_extract = ['BF', 'SNR', 'reliability', 'uniq_id','exp_ID', 
        #                   'stim_name']
        # roi_data = ROI_mod.data_to_list(rois, data_to_extract)
        # roi_traces = list(map(lambda roi: roi.df_trace, rois))
        # bf_image = ROI_mod.generate_colorMasks_properties(rois, 'BF')
        # rois_df = pd.DataFrame.from_dict(roi_data)
        # raw_stim = rois[0].stim_info['input_data']
        # stim_info = rois[0].stim_info
        # fig0=sf.make_exp_summary_TF(figtitle, 'aa',
        #                  rois[0].source_image,ROI_mod.get_masks_image(rois), 
        #                  roi_traces,raw_stim, bf_image,
        #                  rois_df, rois,stim_info, save_fig,
        #                  experiment_conditions['MovieID'],summary_save_dir)
        
        
        # if save_fig:
        #    # Saving figure 
        #    f0_n = 'Summary_%s' % (exp_ID)
        #    os.chdir(fig_save_dir)
        #    fig0.savefig('%s.png'% f0_n, bbox_inches='tight',
        #                transparent=False,dpi=300)
        #    if "RF_map" in rois[0].__dict__.keys():
        #        fig1 = ROI_mod.plot_RF_centers_on_screen(rois,prop='BF',
        #                                          cmap='inferno',
        #                                          ylab='BF (Hz)',
        #                                          lims=(0,1.5))
               
        #        f1_n = 'BF_on_screen_%s' % (exp_ID)
        #        fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
        #                    transparent=False,dpi=300)
        #return ROIs 
        'flag'

    # elif analysis_type == 'gratings_transfer_rois_save':
    #     map(lambda roi: roi.calculateTFtuning_BF(), rois)
    #     # Summary of current experiment
    #     data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id',
    #                        'PD', 'exp_ID', 'stim_name']
    #     roi_data = ROI_mod.data_to_list(rois, data_to_extract)
    #     roi_traces = list(map(lambda roi: roi.df_trace, rois))
    #     bf_image = ROI_mod.generate_colorMasks_properties(rois, 'BF')
    #     rois_df = pd.DataFrame.from_dict(roi_data)
    #     raw_stim = rois[0].stim_info['input_data']
    #     stim_info = rois[0].stim_info
    #     fig0=sf.make_exp_summary_TF(figtitle, 'aa',
    #                      rois[0].source_image,ROI_mod.get_masks_image(rois), 
    #                      roi_traces,raw_stim, bf_image,
    #                      rois_df, rois,stim_info, save_fig,
    #                      experiment_conditions['MovieID'],summary_save_dir)
    
    #     fig1 = ROI_mod.plot_RFs(rois, number=20, f_w =5,cmap='coolwarm',
    #                             center_plot = True, center_val = 0.95)
    #     fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='BF',
    #                                              cmap='inferno',
    #                                              ylab='BF (Hz)',
    #                                              lims=(0,1.5))
        
    #     fig3 = ROI_mod.plot_RF_centers_on_screen(rois,prop='PD')
        
    #     if save_fig:
    #        # Saving figure 
    #        f0_n = 'Summary_%s' % (exp_ID)
    #        f1_n = 'RF_examples_%s' % (exp_ID)
    #        f2_n = 'BF_on_screen_%s' % (exp_ID)
    #        f3_n = 'PD_on_screen_%s' % (exp_ID)
    #        os.chdir(fig_save_dir)
    #        fig0.savefig('%s.png'% f0_n, bbox_inches='tight',
    #                    transparent=False,dpi=300)
    #        fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
    #                    transparent=False,dpi=300)
    #        fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
    #                    transparent=False,dpi=300)
           
    #        fig3.savefig('%s.png'% f3_n, bbox_inches='tight',
    #                    transparent=False,dpi=300)
    
    #     for bf in np.unique(roi_data['BF']):
    #         curr_rois = np.array(rois)[np.array(roi_data['BF'])==bf]
    #         fig5 = ROI_mod.plot_RF_centers_on_screen(curr_rois,prop='BF',
    #                                    cmap='inferno',
    #                                    ylab='BF (Hz)',
    #                                    lims=(0,1.5))
    #         f5_n = 'BF_%.2f_onscreen' % (bf)
    #         os.chdir(fig_save_dir)
    #         fig5.savefig('%s.png'% f5_n, bbox_inches='tight',
    #                  transparent=False,dpi=300)
            
    # elif analysis_type == 'STF_1':
    #     rois = ROI_mod.create_STF_maps(rois)
    #     # Summary of current experiment
    #     run_matplotlib_params()
    #     mean_STF = np.mean([np.array(roi.stf_map) for roi in rois],axis=0)
    #     fig = plt.figure(figsize = (5,5))
    #     plt.subplot(211)
    #     plt.title('STF map')
    #     ax=sns.heatmap(mean_STF, cmap='coolwarm',center=0,
    #                    xticklabels=np.array(rois[0].stf_map.columns.levels[1]).astype(int),
    #                    yticklabels=np.array(rois[0].stf_map.index),
    #                    cbar_kws={'label': '$\Delta F/F$'})
    #     ax.invert_yaxis()
    #     ax.invert_xaxis()
        
    #     plt.subplot(212)
    #     plt.title('STF map normalized')
    #     stf_map_norm=np.mean([np.array(roi.stf_map_norm) for roi in rois],axis=0)
        
        
    #     ax1=sns.heatmap(stf_map_norm, cmap='coolwarm',center=0,
    #                    xticklabels=np.array(rois[0].stf_map.columns.levels[1]).astype(int),
    #                    yticklabels=np.array(rois[0].stf_map.index),
    #                    cbar_kws={'label': 'zscore'})
    #     ax1.invert_yaxis()
    #     ax1.invert_xaxis()
    #     fig = plt.gcf()
    #     f0_n = 'Summary_STF_%s' % (exp_ID)
    #     os.chdir(fig_save_dir)
    #     fig.savefig('%s.png'% f0_n, bbox_inches='tight',
    #                 transparent=False,dpi=300)
        
    elif analysis_type == 'shifted_STRF':
        
        ROI_mod.reverse_correlation_analysis_JF(rois,kwargs['expdir'])
    
    elif analysis_type == 'Frozen_noise':
        
        for roi in rois:
          roi.trial_average_noise()
          roi.calculate_reliability()
        
        ROI_mod.append_RF_test_trace(rois,kwargs['expdir'])
        
        # calculate reliability
        # do trial average and store it  

    elif analysis_type == 'shifted_STRF_temporal':
        ROI_mod.reverse_correlation_analysis_JF(rois,kwargs['stim_dir'], test_frames=[[0,250]]) #for 17ms update or [[0, 250]] for 50ms update
        # ROI_mod.reverse_correlation_analysis_JF(rois,kwargs['stim_dir']) #for 17ms update or [[0, 250]] for 50ms update
        # test_frames=[[0, 307], [924, 1231], [1848, 2155]]
        # test_frames=[[0,1200],[3600, 4800],[7200,8400]]    
    elif analysis_type == 'chirp':
        foo = 1  
        
    return rois
            



def generate_pixel_maps(time_series,trialCoor,stimulus_information,frameRate,smooth=True,sigma=0.75):
    """
    """
    
    ## Generating pixel maps
    smooth_time_series = filters.gaussian(time_series, sigma=sigma)
    (wholeTraces_allTrials_smooth, respTraces_allTrials_smooth, baselineTraces_allTrials_smooth) = \
        separate_trials_video(smooth_time_series, trialCoor, stimulus_information,
                                                            frameRate)
    
    # Calculate maximum response
    MaxResp_matrix_all_epochs,  maxEpochIdx_matrix_all, \
    MeanResp_matrix_all_epochs, maxEpochIdx_matrix_all_mean = calculate_pixel_max(respTraces_allTrials_smooth,
                                                          stimulus_information)
    max_resp_matrix_all_mean = np.nanmax(MeanResp_matrix_all_epochs, axis=2)
    
    SNR_image = calculate_pixel_SNR(baselineTraces_allTrials_smooth,
                                    respTraces_allTrials_smooth,
                                    stimulus_information, frameRate,
                                    SNR_mode='Estimate')
    # DSI and CSI 
    DSI_image = create_DSI_image(stimulus_information, maxEpochIdx_matrix_all_mean,
                                                               max_resp_matrix_all_mean, MeanResp_matrix_all_epochs)
    mean_image = time_series.mean(0)
    CSI_image = create_CSI_image(stimulus_information,frameRate, 
                                 respTraces_allTrials_smooth, DSI_image)
    
    
    
    return SNR_image, DSI_image, CSI_image


    
    


def generate_roi_masks_image(roi_masks,im_shape):
    # Generating an image with all clusters
    all_rois_image = np.zeros(shape=im_shape)
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    return all_rois_image
    

def organize_extraction_params(extraction_type,
                               current_t_series=None,current_exp_ID=None,
                               alignedDataDir=None,
                               stimInputDir=None,
                               use_other_series_roiExtraction = None,
                               use_avg_data_for_roi_extract = None,
                               roiExtraction_tseries=None,
                               transfer_data_n = None,
                               transfer_data_store_dir = None,
                               transfer_type = None,
                               analysis_type=None,
                               imaging_information=None,
                               experiment_conditions=None,
                               stimuli=None,
                               thresholds=None): #Juan made some modifications here
    
    extraction_params = {}
    extraction_params['type'] = extraction_type
    if extraction_type == 'SIMA-STICA':
        extraction_params['stim_input_path'] = stimInputDir
        if use_other_series_roiExtraction:
            series_used = roiExtraction_tseries
        else:
            series_used = current_t_series
        extraction_params['series_used'] = series_used
        extraction_params['series_path'] = \
            os.path.join(current_exp_ID, 
                                  series_used)
        extraction_params['area_max_micron'] = 4
        extraction_params['area_min_micron'] = 1
        extraction_params['cluster_max_1d_size_micron'] = 4
        extraction_params['cluster_min_1d_size_micron'] = 1
        extraction_params['extraction_reliability_threshold'] = 0.4
        extraction_params['use_trial_avg_video'] = \
            use_avg_data_for_roi_extract
    elif extraction_type == 'transfer':
        transfer_data_path = os.path.join(transfer_data_store_dir,
                                          transfer_data_n)
        extraction_params['transfer_data_path'] = transfer_data_path
        extraction_params['transfer_type']=transfer_type
        extraction_params['imaging_information']= imaging_information
        extraction_params['experiment_conditions'] = experiment_conditions
        extraction_params['stimuli']=stimuli
        extraction_params['thresholds']=thresholds
        extraction_params['analysis_type']=analysis_type
    return extraction_params

def refine_rois(rois, cat_bool, extraction_params,roi_1d_max_size_pixel,
                roi_1d_min_size_pixel,use_otsu=True, 
                mean_image=None,otsu_mask=None):
        
        if use_otsu:
            otsu_threshold_Value = filters.threshold_otsu(mean_image[otsu_mask])
            otsu_thresholded_mask = mean_image > otsu_threshold_Value
        else:
            otsu_thresholded_mask = cat_bool>-1
        
        [refined_rois, roi_image] = \
            clusters_restrict_size_regions(rois,cat_bool,roi_1d_max_size_pixel,
                 roi_1d_min_size_pixel,otsu_thresholded_mask)
        return refined_rois, roi_image
def select_properties_plot(rois , analysis_type):
    
    if analysis_type == 'gratings_transfer_rois_save' or\
        (analysis_type == 'STF_1'):
        
        
        properties = ['PD', 'DSI', 'CS','BF']
        colormaps = ['hsv', 'viridis', 'PRGn', 'inferno']
        
        if (analysis_type == 'STF_1'):
            vminmax = [(0,360), (0, 1), (-1, 1), (0, 1)]
        else:
            vminmax = [(0,360), (0, 2), (-1, 1), (0, 1.5)]
            
        
        data_to_extract = ['DSI', 'CSI', 'SNR', 'reliability','BF']
        
    elif ((analysis_type == '8D_10dps_stripes_RF') or\
         (analysis_type == '11_steps_luminance') or\
         (analysis_type == 'A_B_steps')   ):
        max_d = ROI_mod.data_to_list(rois, ['max_response'])
        max_snr = ROI_mod.data_to_list(rois, ['SNR'])
        properties = ['reliability', 'max_response' ,'SNR','reliability']
        colormaps = ['inferno', 'viridis','inferno','inferno']
        vminmax = [(0, 1), (0, np.max(max_d['max_response'])),
                   (0, np.max(max_snr['SNR'])),(0, 1)]
        data_to_extract = ['reliability', 'max_response' ,'SNR','reliability']
    
    elif ((analysis_type == 'luminance_edges_OFF' ) or\
          (analysis_type == 'luminance_edges_ON' )):
        
        properties = ['SNR', 'slope','reliability']
        colormaps = ['viridis', 'PRGn', 'viridis']
        vminmax = [(0, 3), (-1, 2), (0, 1)]
        data_to_extract = ['CSI', 'slope', 'reliability']
    
    elif (analysis_type == '5sFFF_analyze_save'):
        
        max_d = ROI_mod.data_to_list(rois, ['max_response'])
        properties = ['corr_fff', 'max_response']
        colormaps = ['PRGn', 'viridis']
        vminmax = [(-1,1), (0, np.max(max_d['max_response']))]
        data_to_extract = ['corr_fff', 'max_response']
        
    elif ((analysis_type == 'stripes_ON_vertRF_transfer') or \
          (analysis_type == 'stripes_ON_horRF_transfer') or \
          (analysis_type == 'stripes_OFF_vertRF_transfer') or \
          (analysis_type == 'stripes_OFF_horRF_transfer')):
        
        max_d = ROI_mod.data_to_list(rois, ['max_response'])
        max_snr = ROI_mod.data_to_list(rois, ['SNR'])
        properties = ['corr_fff', 'max_response' ,'SNR','reliability']
        colormaps = ['PRGn', 'viridis','inferno','inferno']
        vminmax = [(-1,1), (0, np.max(max_d['max_response'])),
                   (0, np.max(max_snr['SNR'])),(0, 1)]
        data_to_extract = ['stripe_gauss_fwhm', 'max_response' ,'SNR','reliability']
    
    elif ((analysis_type == 'stripes_ON_delay_profile') or \
          (analysis_type == 'stripes_OFF_delay_profile')):
        properties = ['PD', 'SNR', 'CS','reliability']
        colormaps = ['hsv', 'viridis', 'PRGn', 'viridis']
        vminmax = [(0,360), (0, 25), (-1, 1), (0, 1)]
        data_to_extract = ['DSI', 'CSI', 'resp_delay_deg', 'reliability']
    
    elif analysis_type=='2D_edges_find_save':
        max_snr = ROI_mod.data_to_list(rois, ['SNR'])
        max_DSI = ROI_mod.data_to_list(rois, ['DSI'])
        max_CSI = ROI_mod.data_to_list(rois, ['CSI'])
        properties = ['SNR','reliability','DSI','PD','CS']
        colormaps = ['viridis','viridis','viridis','RdBu','viridis',]
        vminmax = [(0, np.max(max_snr['SNR'])),
                  (0, 1),(0, np.max(max_DSI['DSI'])), 
                  (-1,1), (0,np.max(max_CSI['CSI'])),
                  (-1,1)]
        data_to_extract=['SNR','reliability','DSI','PD','CS']
    
    elif analysis_type=='1-dir_ON_OFF':
        max_CSI = ROI_mod.data_to_list(rois, ['CSI'])
        properties = ['CSI_ON','CSI_OFF']
        colormaps = ['viridis','viridis']
        vminmax = [(0, np.max(max_CSI['CSI'])),
            (0, np.max(max_CSI['CSI'])),]
        data_to_extract=['CSI_ON','CSI_OFF']

    elif analysis_type=='8D_edges_find_rois_save' or analysis_type=='4D_edges' :
        #max_snr = ROI_mod.data_to_list(rois, ['SNR'])
        #max_DSI = ROI_mod.data_to_list(rois, ['DSI'])
        max_CSI = ROI_mod.data_to_list(rois, ['CSI'])
        max_DSI_ON=ROI_mod.data_to_list(rois, ['DSI_ON'])
        max_DSI_OFF=ROI_mod.data_to_list(rois, ['DSI_OFF'])
        


        #properties = ['SNR','reliability','DSI_ON','DSI_OFF','CSI','PD_ON','PD_OFF']
        #colormaps = ['viridis','viridis','viridis','viridis','viridis','gist_rainbow','gist_rainbow']
        # vminmax = [(0, np.max(max_snr['SNR'])),
        #           (0, 1), 
        #           (0, np.max(max_DSI_ON['DSI_ON'])),
        #           (0, np.max(max_DSI_OFF['DSI_OFF'])),
        #           (0, np.max(max_CSI['CSI'])),
        #           (0,359),(0,359)]
        properties = ['DSI_ON','DSI_OFF','CSI_ON','CSI_OFF','PD_ON','PD_OFF']
        colormaps = ['viridis','viridis','viridis','viridis','hsv','hsv']
        vminmax = [(0, np.max(max_DSI_ON['DSI_ON'])),
                  (0, np.max(max_DSI_OFF['DSI_OFF'])),
                  (0, np.max(max_CSI['CSI'])),
                  (0, np.max(max_CSI['CSI'])),
                  (0,359),(0,359)]
        data_to_extract=['DSI_ON','DSI_OFF','CSI_ON','CSI_OFF','PD_ON','PD_OFF']
        #data_to_extract=['SNR','reliability','DSI_ON','DSI_OFF','CSI','PD_ON','PD_OFF']
    
    elif analysis_type== '12_dir_random_driftingstripe':
        max_CSI = ROI_mod.data_to_list(rois, ['CSI_ON'])
        max_DSI=ROI_mod.data_to_list(rois, ['DSI'])
        properties = ['DSI','CSI_ON','PD_ON']
        colormaps = ['viridis','viridis','hsv']
        vminmax = [(0, 1), (0,1), (0,359)]
        data_to_extract=['DSI','CSI_ON','PD_ON']
    else:
        # properties = ['SNR','reliability']
        # colormaps = ['viridis','viridis']
        # vminmax = [(0, 3),  (0, 1)]
        # data_to_extract = ['SNR', 'reliability']
        return None, None, None, None
        
    return properties, colormaps, vminmax, data_to_extract

def run_ROI_selection(extraction_params,stack,image_to_select=None,**kwargs):
    """

    """
    #Categories can be used to classify ROIs depending on their location
    # # # plt.close('all')
    # # # plt.style.use("default")
    # # # print('\n\nSelect categories and background')
    # # # [cat_masks, cat_names] = select_regions(image_to_select, 
    # # #                                          image_cmap="viridis",
    # # #                                          pause_t=8)
    
    # have to do different actions depending on the extraction type
    if extraction_params['type'] == 'manual':
        print('\n\nSelect ROIs')
        [roi_masks, roi_names] = select_regions(image_to_select, 
                                                image_cmap="viridis",
                                                pause_t=4.5,
                                                ask_name=False)
        all_rois_image = generate_roi_masks_image(roi_masks,
                                                  np.shape(image_to_select))
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, None
            
    ##### deprecated #############
    #lif extraction_params['type'] == 'SIMA-STICA': 
     #  # Need the time series and information about the video to be extracted
    #   (time_series, stimulus_information,imaging_information) = \
    #       pre_processing_movie (extraction_params['series_path'],
    #                             extraction_params['stim_input_path'])
    
        
        # A trial averaged version of the video can be used for extraction 
        # since it may decrease the noise. Yet it can introduce artifacts
        # between the epochs
   #    if extraction_params['use_trial_avg_video']:
    #       (avg_video, _, _) = \
   #            separate_trials_video(time_series,stimulus_information,
   #                                  imaging_information['frame_rate'])
   #        sima_dataset = generate_avg_movie(extraction_params['series_path'], 
    #                                         stimulus_information,
    #                                         avg_video)
    #   else:
    #       movie = np.zeros(shape=(time_series.shape[0],1,time_series.shape[1],
    #                           time_series.shape[2],1))
    #       movie[:,0,:,:,0] = time_series
    #       b = sima.Sequence.create('ndarray',movie)
    #       sima_dataset = sima.ImagingDataset([b],None)
    #   
        # We need a certain range of areas for rois 
    #   area_max_micron = extraction_params['area_max_micron']
    #   area_min_micron = extraction_params['area_min_micron']
    #   area_max = int(math.pow(math.sqrt(area_max_micron) / \
    #                           imaging_information['pixel_size'], 2))
    #   area_min = int(math.pow(math.sqrt(area_min_micron) / \
    #                           imaging_information['pixel_size'], 2))
    #   [roi_masks, all_rois_image] = find_clusters_STICA(sima_dataset,
    #                                                     area_min,
     #                                                    area_max)
    #   threshold_dict = {'SNR': 0.75,'reliability': 0.4}
        
    #   return cat_masks, cat_names, roi_masks, all_rois_image, None, threshold_dict
    
    elif extraction_params['type'] == 'transfer':
        
        rois = run_roi_transfer(extraction_params['transfer_data_path'],
                                extraction_params['transfer_type'],
                                experiment_info=extraction_params['experiment_conditions'],
                                imaging_info=extraction_params['imaging_information'])
        
        return cat_masks, cat_names, None, None, rois, None
    
    elif extraction_params['type'] == 'cluster analysis MH_JF':
        dataDir=kwargs['dataDir']
        stimPaths=kwargs['stim_paths']
        stim_types=kwargs['stim_types']
        input_stim_dir=kwargs['stim_dir']
        cplusplus=kwargs['cplusplus']
        categories=kwargs['cat_names']
        category_names=kwargs['cat_names']
        I_zone=kwargs['I_zone']
        fig_dir=kwargs['fig_dir']
        clustering_mode=extraction_params['clustering_type']
        Tser=kwargs['dataDir']
        Tser_index=None
        
        # check if we have a valid stimuli in any of the cycles recorded
        for ix,ind_stim in enumerate(stim_types):
            ind_stim=ind_stim.split('\n')[0]
            ind_stim=ind_stim.split(' ')[0]
            if (ind_stim== 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt'\
                or ind_stim=='DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_8Dirs_optimized.txt'\
                or ind_stim=='DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_8Dirs_optimized_fullcontrast.txt'\
                or ind_stim== 'DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_8Dirs_optimized_80_3s.txt'
                or ind_stim== 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_ONEDGEFIRST_80sec.txt'\
                or ind_stim=='Drifting_stripesJF_8dir.txt'\
                or ind_stim=='Dirfting_edges_JF_8dirs_ON_first.txt'\
                or ind_stim== 'Mapping_for_MIexp.txt'\
                or ind_stim== 'DriftingStripe_4sec_0deg.txt'\
                or ind_stim== 'DriftingStripe_4sec_180degsec.txt'\
                or ind_stim== 'DriftingStripe_4sec_180deg.txt'
                or ind_stim== 'DriftingDriftStripe_Sequential_LumDec_8D_15degoffset_ONEDGEFIRST.txt'\
                or ind_stim== 'DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                or ind_stim== 'mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt')\
                or 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D' in ind_stim:
                Tser_index=ix
                # stack=stack[ix]
                def_ix=ix
        if Tser_index==None:
            raise Exception('no appropiate stimulus found for clustering analysis')
        # crop the stack to get rid of cell bodies use te inclusion zone for this:

        


        (stimulus_information, imaging_information) = \
                    get_stim_xml_params(dataDir, stimPaths[def_ix],input_stim_dir,stack.shape[0],cplusplus=cplusplus) 
        #X_size,Y_size,pixel_area=pixel_size=getPixelSize(xmlFile) 
        # # # [I_zone, _] = select_regions(image_to_select, 
        # # #                                      image_cmap="viridis",
        # # #                                      pause_t=7,ask_name=True,roi_type='Incl')
        #Categories can be used to classify ROIs depending on their location
        # # # plt.close('all')
        # # # plt.style.use("default")
        # # # print('\n\nSelect categories and background')
        # # # [cat_masks, cat_names] = select_regions(image_to_select, 
        # # #                                      image_cmap="viridis",
        # # #                                      pause_t=9,roi_type='cat')
        
        roi_masks, all_rois_image, polarity_list, background, metadata = run_cluster_analysis_MH(dataDir,fig_dir,stack,stimulus_information, imaging_information,category_names,I_zone,clustering_mode=clustering_mode,Tser=Tser)
        return {'background':background, 'roi_masks':roi_masks, 'all_rois_image':all_rois_image,'polarities': polarity_list},metadata

    else:
       raise TypeError('ROI selection type not understood.') 
    
    
    
    
    
    
def interpolate_data_dyuzak(stimtimes, stimframes100hz, dsignal, imagetimes, freq):
    """Interpolates the stimulus frame numbers (*stimframes100hz*), signal
    traces (*dsignal*) by using the
    stimulus time (*stimtimes*)  and the image time stamps (*imagetimes*)
    recorded. Interpolation is done to a frequency (*freq*) defined by the
    user.
    recorded in

    Parameters
    ----------
    stimtimes : 1D array
        Stimulus time stamps obtained from stimulus_output file (with the
        rate of ~100Hz)
    stimframes100hz : 1D array
        Stimulus frame numbers through recording (with the rate of ~100Hz)
    dsignal : mxn 2D array
        Fluorescence responses of each ROI. Axis m is the number of ROIs while
        n is the time points of microscope recording with lower rate (10-15Hz)
    imagetimes : 1D array
        The time stamps of the image frames with the microscope recording rate
    freq : int
        The desired frequency to interpolate

    Returns
    -------
    newstimtimes : 1D array
        Stimulus time stamps with the rate of *freq*
    dsignal : mxn 2D array
        Fluorescence responses of each ROI with the rate of *freq*
    imagetimes : 1D array
        The time stamps of the image frames with the rate of *freq*
    """
    
    # Interpolation of stimulus frames and responses to freq

    # Creating time vectors of original 100 Hz(x) and freq Hz sampled(xi)
    # x = vector with 100Hz rate, xi = vector with user input rate (freq)
    x = np.linspace(0,len(stimtimes),len(stimtimes))
    xi = np.linspace(0,len(stimtimes),
                     np.round(int((np.max(stimtimes)-np.min(stimtimes))*freq)+1))

    # Get interpolated stimulus times for 20Hz
    # stimtimes and x has same rate (100Hz)
    # and newstimtimes is interpolated output of xi vector
    newstimtimes = np.interp(xi, x, stimtimes)
    newstimtimes =  np.array(newstimtimes,dtype='float32')

    # Get interpolated stimulus frame numbers for 20Hz
    # Below stimframes is a continuous function with stimtimes as x and
    # stimframes100Hz as y values
    stimframes = interpolate.interp1d(stimtimes,stimframes100hz,kind='nearest')
    # Below interpolated stimulus times are given as x values to the stimtimes
    # function to find interpolated stimulus frames (y value)
    stimframes = stimframes(newstimtimes)
    stimframes = stimframes.astype('int')

    #Get interpolated responses for 20Hz
    dsignal1 = np.empty(shape=(len(dsignal),
                               len(newstimtimes)),dtype=dsignal.dtype)
    dsignal=np.interp(newstimtimes, imagetimes, dsignal)
   

    return (newstimtimes, dsignal, stimframes)
    
#%% Juan additions

def find_cycles(path):
    tif_path=path+'\\*Cycle*.tif'
    tif_list=glob.glob(tif_path)
    cycles=[]
    for tif in tif_list:
        cycle_num=tif.split('\\')[-1]
        cycle_num=cycle_num.split('_')[1]
        cycles.append(cycle_num)
    
    cycles,lenghtCycles=np.unique(np.array(cycles),return_counts=True)
    number_cycles=len(cycles)
    return number_cycles,lenghtCycles

# def import_cat_ROIs(dataset,mot_corr=True,ROIs_label='categories'):
#     """
#     takes ROIs selected with the sima ROIbuddy (Kaifosh et al., 2014) and  
#     returns a list of category ROIs (for example, background and inclusion masks)
#     and a list of category names correcponding to the category ROIs 
    
#     args:

#     dataset: sima.imagingDataset object 
#     mot_corr: indicates if data is motion corrected
#     ROIs_label: string. This is a dictionary key to extract a sima.roilist object 
#                 example: dataset.ROIs[ROIs_label] is an roi list
        
#     returns:
#     """
#     # if mot_corr==True:
#     #     sima_string='TserMC.sima'
#     # else:
#     #     sima_string='Tser.sima' 

#     #os.chdir(Tser)
#     Manual_ROIs=dataset.ROIs[ROIs_label]
#     ROI_list=[]
#     """
#     put ROI masks in numpy array
#     """
#     check=0
#     tag_list=[]
#     for idx,element in enumerate(Manual_ROIs):
#         if 'BG' in element.tags:
#             BG=np.squeeze(np.array(element))
#             #background=np.reshape(np.array(element),(dataset.frame_shape[1],dataset.frame_shape[2],1))
#             #background.astype(int)
#             BG=BG.astype(bool)
#             bg_index=idx
#             check=1
#         elif 'BG' not in element.tags:
#             /region=np.squeeze(np.array(element))
#             /region.astype(int)
#             ROI_list.append(region)
#             tags=element.tags
#             if len(tags)==0:
#                 tag_list.append(['No_tag'])
#             else:
#                 tag_list.append([tags])
#     if check==0:
#         raise Exception('no Background for %s' %(dataDir))    
#     #np.delete(ROI_list,bg_index,axis=2)
#     # """
#     # save ROIs as pickle file
#     # """
        
#     #ROI_dict={'tags':tag_list, 'ROIs':ROI_list,'background':background}
        
#     # with open('manual_ROIs.pkl', 'wb') as handle:
#     #    pickle.dump(ROI_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#     # """
#     #   produce an image with the ROIs
#     # """
#     # avg=mpimg.imread('Average_im.tif')
#     # plt.figure()
#     # plt.imshow(avg,'gray')
#     # ROI_list=np.where(ROI_list==0,np.nan,ROI_list)       
#     # colors=['Wistia','coolwarm','hsv','PRGn','Accent','Pastel1','PuOr']
#     # count=0
    
#     # for frame in range(len(ROI_list[0,0,:])-1):
        
#     #     plt.imshow(ROI_list[:,:,frame],colors[count],alpha=0.4)
#     #     count=count+1
#     #     if count==7:
#     #         count=0

#     # plt.savefig('ROIs_manual.tif')
#     # plt.close()
#     return  BG,tag_list, ROI_list  #this is equivalent to cat_names, cat_masks in Buraks code

def find_stimfile_check_stim(exp,Tser,number_of_cycles): #repair this function. relying on time is no good
    """
    looks for stimulus files in Tseries folder
    
    returns:
        stimtype(list): string with the name of stimuli type inside stim_files
        stim_names(list): stimfiles names 
    """
    #####date sorting stopped for unreliability

    # path_raw=home_path+'\\'+ exp +'\\' + 'raw\\'
    # path=[]
    # path=dataDir+current_t_series+'.xml' 
    # date=current_exp_ID.split('_')[0]
    # xml_time=pathlib.Path(glob.glob(path)[0])
    # xml_time=pathlib.Path.stat(xml_time)[8]
    # ##search for correct stim_file
    # stim_path= '*stim*_%s_jv*\\'%(date)
    # stim_path=path_raw + stim_path
    # stim_path=stim_path + '_stimulus_output*'
    # stimfiles=glob.glob(stim_path)# !!!!!sorted
    # for number_stim,stim in enumerate(stimfiles):
    #     stim_p=pathlib.Path(stim)
    #     stimtime=pathlib.Path.stat(stim_p)[8]
    #     if '2021_3_8' in stim : # this corrects a time delay observed in sepcific dates
    #         stimtime=stimtime-3600
    #     #stim.split('.txt')[0].split('_')[-3:] #in case we need repairing this code
    #     difference=stimtime-xml_time
    #     correct_stimulus=abs(difference)<30
    #     if correct_stimulus:     
    #         Stim_content= open(stim, "r")
    #         line=Stim_content.readlines()
    #         stimtype=[line[1].split(r'/')[-1]]
    #         Stim_content.close()
    #         #stim_name=[os.path.split(stim)] #TODO check if the slice is [0] or [1]
    #         stim_name=[stim]
    #         if number_of_cycles==1: 
    #             return stimtype, stim_name
    #         elif number_of_cycles>1:
    #             slice_number=number_of_cycles-1
    #             extra_names=np.array(stimfiles[number_stim-slice_number:number_stim])
    #             stim_names=np.append(extra_names,stim_name)
    #             mult_stim_types=np.array([])
    #             for cycle in extra_names:
    #                 Stim_content_add= open(cycle, "r")
    #                 line_add=Stim_content_add.readlines()
    #                 stimtype_add=[line_add[1].split(r'/')[-1]]
    #                 Stim_content_add.close()
    #                 mult_stim_types=np.append(mult_stim_types,stimtype_add)
    #             mult_stim_types=np.append(mult_stim_types,stimtype)
    #             original_stims=[]
    #             #return mult_stim_types, stim_names

    stim_names=sorted(glob.glob(Tser +'\\*stim*'),key=os.path.basename)
    stim_types=[]
    if len(stim_names)<number_of_cycles:
        raise Exception('at least 1 stim missing for %s'%(Tser))
    for stim in stim_names:
        Stim_content= open(stim, "r")
        line=Stim_content.readlines()
        stim_type=[line[1].split(r'/')[-1]][0]
        stim_types.append(stim_type)
    return stim_types, stim_names

def load_Tseries(dataDir,number_of_cycles,lenghtCycles,file_string='_motCorr.tif'):
    """
    loads Tseries movies, splited according to cycles of recording

    args:
        datadir: Tseries folder path

        number_of_cycles: list. number of recording cycles in the current Tseries

        lenght Cycles: list. number of frames for each cycle 

        file_string: name of movie file
    
    returns:
        mean image: np array. average image of the Tseries
        Tserlist: list. contains numpy arrays for every recording cycle
        
    """
    tif_location=dataDir + '\\'+file_string # if error here, check MC or non MC
    time_series = io.imread(glob.glob(tif_location)[-1])
    mean_image = time_series.mean(0)
    #time_series = np.asarray(time_series) #make sure that the shape here is correct. it should be time in axis 0
    Tserlist=[]
    lenght_prev_cycles=0
    #TODO put time_series in sliceable format
    for cycle in range(number_of_cycles):
        currentNumberFrames=lenghtCycles[cycle]
        series_slice=time_series[lenght_prev_cycles:lenght_prev_cycles+currentNumberFrames,:,:]
        Tserlist.append(series_slice)
        lenght_prev_cycles+=currentNumberFrames
    return mean_image, Tserlist

def extract_xml_info(dataDir):
    """
    extract relevant metadata from xml file within Tseries folder
    
    arg:
        path (string): path for Tseries folder
        
        
    returns:
        imaging_info(dict): key-value pairs correspond to Tseries imaging properties:
                            FramePeriod, rastersPerFrame, pixel size, pixel area, layer position([x,y,z])
    """
    #os.chdir(Tser)
    xmlFile=os.path.split(os.path.split(dataDir)[0])[1]
    #xmlFile=dataDir+'\\'+ xmlFile +'.xml'
    xmlFile=dataDir+'\\'+ '*.xml'
    xmlFile=glob.glob(xmlFile)[0]
    imaging_info={}

    imaging_info['FramePeriod']=getFramePeriod(xmlFile)
    imaging_info['micRelTimes']=getMicRelativeTime(xmlFile)
    imaging_info['rastersPerFrame']=getrastersPerFrame(xmlFile)
    imaging_info['frame_rate']=1/imaging_info['FramePeriod']
    imaging_info['x_size'], imaging_info['y_size'], imaging_info['pixelArea']=getPixelSize(xmlFile)
    imaging_info['layerPosition']=getLayerPosition(xmlFile)
    return imaging_info

##########Pradeep modified ##############

def dataframe_to_dict(dfs, filenames):
    """
    Convert a pandas DataFrame to a dictionary.
    Parameters:
    df (pd.DataFrame): DataFrame with columns 'index', 'axis-1', 'axis-2'.
    Returns:
    dict: Dictionary with keys as unique 'index' and values as lists of [axis-1, axis-2] pairs.
    """
    # Initialize an empty dictionary
    result_dict = {}
    
    for df, filename in zip(dfs, filenames):
        roi_set_name=os.path.split(filename)[-1].split('.csv')[0]
        if roi_set_name not in result_dict:
            result_dict[roi_set_name] = {}
    
    # Iterate through the DataFrame
        for idx, row in df.iterrows():
            if row['index'] not in result_dict[roi_set_name]:
                result_dict[roi_set_name][row['index']] = []

            # Append the [axis-1, axis-2] pair to the innermost list
            result_dict[roi_set_name][row['index']].append([row['axis-1'], row['axis-0']])
    return result_dict

class ROI_polygon:
    def __init__(self,polygon,key,path,visualise = False):
        self.polygon = polygon
        self.tag = key
        self.path = path
        #self.tags = self.dict.keys()
        #self.polygons = self.dict.values()
        self.average_image = tifffile.imread(os.path.join(path, '_motavg.tif'))  
        self.mask = self.create_mask_from_polygon()
        self.visualise = self.visualize(toplot = visualise)
        # self.subset = self.subs(atags = None, neg_tags = None)
        
    def create_mask_from_polygon(self):
        #mask = []
        #for k in self.dict.keys():
        pts = np.array(self.polygon, np.int32)
        # Fill the polygon in the mask
        mask = np.zeros_like(self.average_image)
        mask = cv2.fillPoly(mask, [pts], color=(255, 0, 0))
        return mask
            
    def visualize(self, toplot = False):
        if toplot == True:
            #for mask in self.masks:
            cv2.imshow("img", 10*cv2.resize(self.mask, None, fx=5, fy=5))  #visualise the masks
            cv2.waitKey(0)
            # Closing all open windows
            cv2.destroyAllWindows()
        else:
            pass


def import_clusters(Tser,manual=False,FFT_filtered=False,video=None):
    #this function uses bits of Buraks code
    # if FFT_filtered==False:
    #     try:
    #         sima_folder=glob.glob(Tser+'TserMC.sima')[0] #MC is in all motion corrected SIMA objects
    #     except:
    #         sima_folder=glob.glob(Tser+'Tser.sima')[0]
    #     finally:
    #         pass
    # else:
    #     sima_folder=glob.glob(Tser+'TserMC_bandstop_filtered.sima')[0]
    # dataset=sima.ImagingDataset.load(sima_folder)
    # if manual==False:
    #     clusters=dataset.ROIs['auto_ROIs']
    # else:
    #     try:
    #         clusters=dataset.ROIs['Manual']
    #     except:
    #         try:
    #             clusters=dataset.ROIs['manual'] #if key error, then rois ned to be selected
    #         except:
    #             clusters=dataset.ROIs['Manual_rois']
    #     finally:
    #         pass
        # Generating an image with all clusters
    if manual == False:
        raise Exception ("Sima dependency deprecated and removed, please use napari or roipoly or imageJ to draw manual ROIs.")
    else:    
        dfs = []
        filenames = []
        for csvfile in glob.glob(Tser+'/rois*.csv'):
            dfs.append(pd.read_csv(csvfile))
            filenames.append(csvfile)
        dictionaries = dataframe_to_dict(dfs,filenames)
        clusters = []
        for layer in dictionaries.keys():
            for polygon in dictionaries[layer].keys():        
                    clusters.append(ROI_polygon(dictionaries[layer][polygon], layer, Tser))
        check=0
        tag_list=[]
        for idx,element in enumerate(clusters):
            print (element.tag)
            if 'BG' in element.tag:                
                BG=np.squeeze(np.array(element.mask))
                background=np.array(element.mask)
                background.astype(int)
                BG=BG.astype(bool)
                bg_index=idx
                check=1
                # clusters=clusters.subset(tags=None, neg_tags=['BG'])
                clusters.pop(bg_index)
            elif 'background' in element.tag:
                BG=np.squeeze(np.array(element))
                background=np.reshape(np.array(element),(dataset.frame_shape[1],dataset.frame_shape[2],1))
                background.astype(int)
                BG=BG.astype(bool)
                bg_index=idx
                check=1
                clusters=clusters.subset(tags=None, neg_tags=['background'])
        if check==0:
            #dataDir=Tser
            raise Exception ('no background')
            BG=extract_im_background_forclustering(video,video)         
        # else:
        #     #_,BG=extract_im_background_forclustering(video,video)
            
        for idx,element in enumerate(clusters):
            tags=element.tag
            if len(tags)==0:
                tag_list.append(['No_tag'])
                raise Exception('no tag')
            else:
                tag_list.append([tags])
        
    data_xDim = clusters[0].mask.shape[0]
    data_yDim = clusters[0].mask.shape[1]
    all_clusters_image = np.zeros(shape=(data_xDim,data_yDim))
    all_clusters_image[:] = np.nan
    for index, roi in enumerate(clusters):
        curr_mask = np.array(clusters[index].mask).astype(bool)
        all_clusters_image[curr_mask.astype(int)] = index+1    
           
    if manual==True:  
        return BG,tag_list,clusters, all_clusters_image 
    else:
        raise Exception ('SIMA auto_roi and SIMA overall is deprecated')
        #return dataset, clusters, all_clusters_image


def plot_roi_traces(rois,cycle,analysis_type,summary_save_dir):
    #try to create raw_traces_folder


    #clear any old traces. 


    if analysis_type[cycle]=='8D_edges_find_rois_save' or analysis_type[cycle]=='12_dir_random_driftingstripe' or analysis_type[cycle]=='1-dir_ON_OFF':
        ROI_mod.plot_traces_edges(rois,summary_save_dir)
        #TODO. plot all raw traces in the same plot
    elif analysis_type[cycle]=='1-dir_edge_1pol':
        ROI_mod.plot_tracesEdges_against_variables(rois,summary_save_dir)
    elif analysis_type[cycle]=='moving_gratings':
        ROI_mod.plot_tracesEdges_against_variables(rois,summary_save_dir)
        ROI_mod.plot_psds(rois,summary_save_dir)
    else:
        print('raw trace plotting not implemented for this analysis type')


def filter_overlaps_auto_ROIs_with_quantification(quant_to_treshold,rois,save_dir=None,save_fig=True):
    # this function is to be used with ROIs that have been processed, such that quantifications 
    # can be used as inclusion/exclusion criteria 

    # in case of overlap, The ROI with the highest quantity is used (por ej contrast selectivity, reliability, etc) 
    

    # this function is based on buraks function pmc.clusters_restrict_size_regions()

    #passed_rois  = []
    eliminated_rois =[]
    property_list= ROI_mod.data_to_list(rois, quant_to_treshold)
    #ROI_mod.calcualte_mask_1d_size(rois)
    data_xDim = np.shape(rois[0].mask)[0]
    data_yDim = np.shape(rois[0].mask)[1]
    rois_image = np.zeros(shape=(data_xDim,data_yDim))
    rois_image[:] = np.nan
    copy_rois=rois #this is a copy of the original ROI object before it is modified in upcoming steps
    for index, roi in enumerate(rois):
        rois_image[roi.mask] = index+1

    all_pre_selected_mask = np.zeros(shape=(data_xDim,data_yDim))

    #pre_selected_roi_indices = np.arange(len(rois))
    #pre_selected_roi_indices_copy = np.arange(len(rois))
    
    for index, roi in enumerate(rois):
        all_pre_selected_mask[roi.mask] += 1

    mask_array=np.zeros((data_xDim,data_yDim,len(rois))) 

    # fill up the mask array

    for index, roi in enumerate(rois):
        mask_array[:,:,index]=roi.mask #this mask will be used to find overlaps

    while len(np.where(all_pre_selected_mask>1)[0]) != 0:
        
        # find a list with the locations of overlaps. deal with the places where there are the most overlaps first

        overlap_locs=np.where((all_pre_selected_mask==np.max(all_pre_selected_mask)) & (all_pre_selected_mask>1))
        

        overlap_idx1=overlap_locs[0][0]
        overlap_idx2=overlap_locs[1][0]

        # find the indices of ROIs in that overlap 

        rois_in_overlap=np.where(mask_array[overlap_idx1,overlap_idx2,:]==1)[0] # if the masks are boolean True instead of 1

        #within the overlapping rois, find the one with the best quantity

        evaluated_property=np.zeros(len(rois_in_overlap))
        for iroi,O_roi in enumerate(rois_in_overlap):
            current_roi=rois[O_roi]
            evaluated_property[iroi]=(getattr(current_roi,quant_to_treshold[0]))

            # alternative: evaluated_property= property_list[O_roi]
        #find the index of the maximum quantification
      
        idx_of_max=np.where(evaluated_property==np.max(evaluated_property))

        #find the ROI with best quantity and take it out of the overlaping list
        best_roi_idx=rois_in_overlap[idx_of_max] #TODO check if best roi is eliminated from list
        rois_in_overlap=np.delete(rois_in_overlap,np.s_[idx_of_max])
        # eliminate the other ROIs from the mask matrix and from the roi list. 
        for O_roi in rois_in_overlap:
            eliminated_rois.append(rois[O_roi])
        rois=np.array(rois)
        rois=np.delete(rois,np.s_[rois_in_overlap])
            #del property_list[O_roi]         
            
        
        mask_array=np.delete(mask_array,rois_in_overlap,axis=2)
        #redo the all_pre_selected_mask object
        all_pre_selected_mask=np.sum(mask_array,axis=2)
     
    #loop through eliminated rois, to rescue the ones with little overlaps

    #first organize the eliminated rois in decreasing quantity order
    # (to give priority to the high quantity ones):
    print('passed %s rois / %s'%(len(rois),len(copy_rois)))
    values = np.array(list(map(lambda roi : roi.__dict__[quant_to_treshold[0]], eliminated_rois)))
    descending_idxs=np.flip(np.argsort(values),0)
    reordered_eliminated_ROIs=np.array(eliminated_rois)[descending_idxs]
    rec_rois=0
    for ix,el_roi in enumerate(reordered_eliminated_ROIs):        
        el_mask=el_roi.mask
        overlap_finder=el_mask+all_pre_selected_mask
        overlap_idx=np.where(overlap_finder>1)
        if len(overlap_idx[0])<3:
            # eliminate overlaps form eliminated mask
            el_roi.mask[overlap_idx]=0
            # append back the ROI in the rois object
            rois=np.append(rois,el_roi)
            rec_rois+=1
    print('recovered %s rois' %(rec_rois))
    
    # pass ROIs through and additional reliability filter. JF commented out for now
    rois_to_erase=[]
    all_pre_selected_mask[:]=0

    # for index, roi in enumerate(rois):
    #     if roi.reliability[0]<0.5:
    #         rois_to_erase.append(index)
    # rois_to_erase=np.array(rois_to_erase)
    # rois=np.delete(rois,np.s_[rois_to_erase])
    # print('filtered out %s rois' %(len(rois_to_erase)))
    
    # make a figure with the resulting ROIs
    if save_fig==True:
        count=0
        for index, roi in enumerate(rois):
            count+=1
            all_pre_selected_mask[roi.mask]=count
        all_pre_selected_mask[all_pre_selected_mask==0]=np.nan
        plot_roi_masks(all_pre_selected_mask,rois[0].source_image,len(rois),rois[0].experiment_info['FlyID'],save_fig=save_fig,
                                save_dir=save_dir,alpha=0.4)    
    return rois           

def extract_im_background_forclustering(non_cropped_stack,I_zone):
    ''' takes an average projection of a Tseries cycle, blurs it and thresholds it
        (Otsus method) the resulting mask represents the background of the video'''
    # non_cropped_stac = tifffile.imread(non_cropped_stack)
    def rolling_otsus(window_size,stack): #Deprecated
        image=np.zeros((stack.shape[1:]))
        for step in range(stack.shape[0]):
            if stack.shape[0]-step >= window_size:
                #local_slice=ndimage.gaussian_filter(np.abs(stack[step:step+window_size,:,:].max(0)-stack[step:step+window_size,:,:].min(0)),1.5)
                local_slice=ndimage.gaussian_filter(stack[step:step+window_size,:,:].mean(0),1.5)
                local_treshold=filters.threshold_otsu(local_slice)
                local_otsus=local_slice>local_treshold
        image=image + local_otsus 
        return image>0
  
    # use the I_zone to exclude areas of the image that could interfere with the calculation
    # to do this, the image needs to be flatened and nans eliminated
    mean_im=np.mean(non_cropped_stack[0:-1], axis=0)
    #mean_im=ndimage.gaussian_filter(mean_im,1.5)
    # mean_image_cropped=cropped_stack.mean(0)
    std_im=non_cropped_stack.std(0)

    # crop the image based on a provided mask, this mask should include desired regions as well as dark pixels
    resulting_image_mean=mean_im*I_zone
    resulting_image_mean=np.where(resulting_image_mean>0,mean_im,np.nan)
    # resulting_image_mean=resulting_image_mean.reshape((1,-1))
    #resulting_image_std=std_im*I_zone
    #resulting_image_std=np.where(resulting_image_std>0,std_im,np.nan)
    #resulting_image_std=resulting_image_std.reshape((1,-1))

    # eliminate Nan values and calculate otsus treshold to find the foreground in the image
    resulting_image_mean=resulting_image_mean[~np.isnan(resulting_image_mean)].reshape((1,-1))
    otsu_threshold_Value_mean=filters.threshold_otsu(resulting_image_mean)
    #resulting_image_std=resulting_image_std[~np.isnan(resulting_image_std)].reshape((1,-1))
    #otsu_threshold_Value_std=filters.threshold_otsu(resulting_image_std)
    
    # calculate foreground
    FG=np.where(mean_im>otsu_threshold_Value_mean,mean_im,0)
    FG_image=(FG*I_zone).astype(bool)
    FG=np.where(mean_im>otsu_threshold_Value_mean,1,0)
    FG=(FG*I_zone).astype(bool)
    # calculate a first step background
    first_step_BG=np.where(mean_im>otsu_threshold_Value_mean,0,mean_im)
    mean_image_otsu_1st_BG=np.ndarray.flatten(first_step_BG)
    non_zero_values_1st_BG=mean_image_otsu_1st_BG[np.nonzero(mean_image_otsu_1st_BG)]

    #calculate a second step background
    final_BG=np.where(mean_im<np.percentile(non_zero_values_1st_BG, 10),1,0)
    
    #gaussian blur and otsus thresholding in the cropped image
    #gaussian_filtered=ndimage.gaussian_filter(mean_image_cropped,1.5)
    #otsu_threshold_Value_cropped=filters.threshold_otsu(gaussian_filtered)

    # yen_treshold_value=filters.threshold_yen(gaussian_filtered)
    # min_tresh=filters.threshold_minimum(gaussian_filtered)
    # foreground_min=gaussian_filtered>min_tresh
    # foreground_yen=gaussian_filtered>yen_treshold_value
    #foreground_adaptive=gaussian_filtered>
    #foreground_otsus=np.where(gaussian_filtered>otsu_threshold_Value,1,0) 
    
    # True background extraction with whole image (not cropped)
    # gaussian_filtered=ndimage.gaussian_filter1d(mean_image,1.5)
    # otsu_threshold_Value_wholeim=filters.threshold_otsu(gaussian_filtered)
    # mean_image_otsu_background=np.where(mean_image>otsu_threshold_Value_wholeim,0,mean_image) 
    # mean_image_otsu_b=np.ndarray.flatten(mean_image_otsu_background)
    # non_zero_b=mean_image_otsu_b[np.nonzero(mean_image_otsu_b)]
    # background=np.where(mean_image<np.percentile(non_zero_b, 10),1,0) #check if second treshold is necessary
    
    
    #foreground_otsus= rolling_otsus(20,cropped_stack)

    #mean_image_otsu_background = mean_image_otsu_background[np.nonzero(mean_image_otsu_b)]
    # second tresholding step
    return FG_image,FG, final_BG

def calculate_reliability_per_pixel(stack,stimulus_information,imaging_informations):
    return 'not impemented'

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def produce_feature_table_forClustering(stack,foreground,response_treshold,savedir,categories=None,cat_names=None,min_CSI=0.2,stimulus_information=None, imaging_information=None,apply_CSI_treshold=False,manually_clasiffy_layers=False,apply_otsus=True,orderofedges='off_on',Tser=None):
    
    ''' takes a video with a recording and calculates CSI pixel wise
     then it tresholds every pixel with a minumum CSI value and 
     returns a mask of the cs pixels'''
    manually_clasiffy_layers=False
    parameters={'used manual-rois for finding prefered direction':manually_clasiffy_layers}
    #stack=video[cycle]
    try:
        index_for_sequence=np.where(np.array(stimulus_information['stimtype'])=='driftingstripe')[0][0]
    except:
        try:
            index_for_sequence=np.where(np.array(stimulus_information['stimtype'])=='ADS')[0][0]
        except:
            index_for_sequence=np.where(np.array(stimulus_information['stimtype'])=='G')[0][0]
    if stimulus_information['polarity'][index_for_sequence] == 0: #this is for older stim, ideally should identify Stimulus.polarity in these old stimuli (stimulus_information['fg'][index_for_sequence] == 1 and stimulus_information['bg'][index_for_sequence]==0):
        orderofedges='on_off'
    elif stimulus_information['polarity'][index_for_sequence] == 1: # (stimulus_information['fg'][index_for_sequence] == 0 and stimulus_information['bg'][index_for_sequence]==1):
        orderofedges='off_on'
    # get stimulus information
    
    #check if we have the correct stimulus
    local_stack_copy=copy.deepcopy(stack)
    #local_stack_copy=local_stack_copy*foreground[np.newaxis,:,:]
    image_size=foreground.shape
    if stimulus_information is None:
        # Tseries=stack[cycle]
        # (stimulus_information, imaging_information) = \
        #                 get_stim_xml_params(dataDir, stimOutputFile,original_stimDir,cplusplus=cplusplus)
        raise Exception('there is no stim information')
    # get the trial separated video. do a video trial average per epoch
    (wholeTraces_allTrials_video,_,_) = \
                separate_trials_video(local_stack_copy,stimulus_information,
                imaging_information['frame_rate'])
    
    
    ## find the correct cardinal slices (means: corresponding to cardinal direction epochs)

     
    cardinal_slices=[]
    for idx,angle in enumerate(stimulus_information['angle']):
        if stimulus_information['stimtype'][idx]=='driftingstripe' :
            #if (angle== 0.0) or (angle == 90.0) or (angle==180.0) or (angle == 270.0):
            cardinal_slices.append(idx)
        if (stimulus_information['stimtype'][idx]=='ADS' or stimulus_information['stimtype'][idx]=='G') and ('15degoffset' in stimulus_information['stim_name'] or '0degoffset'): # ADS stands for arbitrary drifting stripes .in this stimulus, the cardinal dirs are shifted
            #if (angle== 330.0) or (angle == 60.0) or (angle==150.0) or (angle == 240.0):
            cardinal_slices.append(idx)
    
    cardinal_slices=np.array(cardinal_slices)    
    number_of_epochs=len(cardinal_slices)
    angles=np.array(stimulus_information['angle'])[cardinal_slices]
    maximum_responses_on=np.zeros((len(cardinal_slices),image_size[0],image_size[1]))
    maximum_responses_off=np.zeros((len(cardinal_slices),image_size[0],image_size[1]))
    timing_max_on=np.zeros((len(cardinal_slices),image_size[0],image_size[1]),dtype=int)
    timing_max_off=np.zeros((len(cardinal_slices),image_size[0],image_size[1]),dtype=int)
    all_differences_ON_OFF=np.zeros((len(cardinal_slices),image_size[0],image_size[1]),dtype=float)
    all_differences_OFF_ON=np.zeros((len(cardinal_slices),image_size[0],image_size[1]),dtype=float)
    ### end of patch


    ##### test: try to see if pref_dir can be calculated in a different way
    # this creates a projection of the video for every cardinal direction of movement, where the maximum is shown
    maxima_across_polarities=np.zeros((len(cardinal_slices),image_size[0],image_size[1]),dtype=float)
    maxima_across_polarities[:]=np.nan
    all_categories=np.zeros(image_size)
    
    if categories==None:
        all_categories=np.ones_like(all_categories)*foreground
    else:
        for count,mask in enumerate(categories):
            all_categories=all_categories+mask
            all_categories=np.where(all_categories>1,0,all_categories)
    if manually_clasiffy_layers and categories!=None:
        pref_dir_mat=np.zeros(image_size)
        for label,mask in zip(cat_names,categories):
            if label[0] == 'LPA':
                pref_dir_mat[np.squeeze(np.array(mask))]= np.where(angles==stimulus_information['angle'][cardinal_slices[-1]])[0][0] + 1
            elif label[0] == 'LPB':
                pref_dir_mat[np.squeeze(np.array(mask))]= np.where(angles==stimulus_information['angle'][cardinal_slices[1]])[0][0] + 1
            elif label[0] == 'LPC':
                pref_dir_mat[np.squeeze(np.array(mask))]= np.where(angles==stimulus_information['angle'][cardinal_slices[0]])[0][0] + 1
            elif label[0] == 'LPD':
                pref_dir_mat[np.squeeze(np.array(mask))]= np.where(angles==stimulus_information['angle'][cardinal_slices[2]])[0][0] + 1
            else:
                raise Exception ('unlabeled category')
    
    
    else:
        for index,epoch in enumerate(cardinal_slices):
            average=np.mean(wholeTraces_allTrials_video[epoch],axis=3)
            maxima_across_polarities[index,:,:]=np.max(average,axis=0)
        pref_dir_mat=np.argmax(maxima_across_polarities,axis=0)
        pref_dir_mat=pref_dir_mat+1
        pref_dir_mat=pref_dir_mat#*all_categories 
    ### end of test

    # include only relevant pixels
    responsive_pixels=np.zeros(image_size,dtype=float)
    responsive_pixels[:]=np.nan
    for idx,slices in enumerate(cardinal_slices):
        
        average=np.mean(wholeTraces_allTrials_video[slices],axis=3)
        std=np.std(average,axis=0)
        maxima_temp= np.max(average,axis=0)
        average=np.mean(average,axis=0)
        #curr_indices=np.where(pref_dir_mat==curr_slice)
        responsive_pixels[pref_dir_mat==idx+1]= maxima_temp[pref_dir_mat==idx+1]>(average[pref_dir_mat==idx+1]+(response_treshold*std[pref_dir_mat==idx+1]))
    
    # # check if stuff worked
    # pref_dir_mat_visual=copy.deepcopy(pref_dir_mat)
    # plt.figure()
    # grid=plt.GridSpec(1, 2, wspace=0.1, hspace=0.2)
    # plt.subplot(grid[0])
    # plt.imshow(pref_dir_mat_visual*foreground)
    # plt.title('pixesl wo response filtering')
    # plt.subplot(grid[1])
    # plt.title('pixels with response filtering')
    # plt.imshow(pref_dir_mat_visual*foreground*responsive_pixels)
    
    counter=0
    for index,epoch in enumerate(cardinal_slices): # enumerate(cardinal_slices):
        #if index==3:
            #print('aa')
        if stimulus_information['stimtype'][epoch]=='driftingstripe' \
            or (stimulus_information['stimtype'][epoch]=='ADS' and stimulus_information['subepoch'][epoch]==2)\
            or stimulus_information['stimtype'][epoch]=='G':
            average=np.mean(wholeTraces_allTrials_video[epoch],axis=3)
            #plt.figure()
            #plt.plot(np.mean(average,axis=(1,2)))
            ### temporal monkey patch for miris data!!!
            #cutoff=[45,58,45,50,40,40,45,45][index]            
            ### end of patch
            
            cutoff=len(average[:,0,0])//2

            if orderofedges=='off_on':
                current_on_half=average[cutoff:,:,:]
                current_off_half=average[:cutoff,:,:]
            elif orderofedges=='on_off':
                current_off_half=average[cutoff:,:,:]
                current_on_half=average[:cutoff,:,:]                


            ###### Juan Commented out next 2 lines. deprecated. 
            #current_on_half=average[:cutoff,:,:] #### Only for cplusplus stimuli!! careful
            #current_off_half=average[cutoff:,:,:]   ###### Only for cplusplus stimuli!! careful
            
            #extract_maximum on and off peaks
            maximum_responses_off[index,:,:]=np.max(current_off_half,axis=0)
            maximum_responses_on[index,:,:]=np.max(current_on_half,axis=0)
            np.argmax(current_off_half,axis=0,out=timing_max_on[index,:,:])
            np.argmax(current_on_half,axis=0,out=timing_max_off[index,:,:]) 
            #calculate max difference for all epochs
            all_differences_ON_OFF[index,:,:]=maximum_responses_on[index,:,:]-maximum_responses_off[index,:,:]
            all_differences_OFF_ON[index,:,:]=maximum_responses_off[index,:,:]-maximum_responses_on[index,:,:]
            # extract the ON and OFF halves of every trace to later 
            # reconcatenate ON and OFF traces
            if counter==0:
                on_averaged_video=current_on_half
                off_averaged_video=current_off_half
            else:
                on_averaged_video=np.concatenate((on_averaged_video,current_on_half),axis=0) 
                off_averaged_video=np.concatenate((off_averaged_video,current_off_half),axis=0)
            counter=+1
    #create a timing matrix, with the timing of every maximum in every epoch

    #TODO create a superimposed image of ON and OFF maxima
    #plt.imshow(np.max(on_averaged_video,axis=0))
    #plt.imshow(np.max(off_averaged_video,axis=0),cmap='gray_r',alpha=0.6)
    #find the maximum of each pixel in the array of maxima
    
    plt.figure()
    ceil=int(np.ceil(all_differences_ON_OFF.shape[0]/2))
    if ceil==0:
        grid = plt.GridSpec(1, 1, wspace=0.1, hspace=0.3)
    else:
        grid = plt.GridSpec(ceil, 2, wspace=0.1, hspace=0.3)

    for h in range(number_of_epochs):
        to_plot=all_differences_ON_OFF
        title= 'ON-OFF maxima' + ' epoch %s %sdeg' %(h+1,angles[h])
        plt.subplot(grid[h])
        ax=plt.imshow(to_plot[h,:,:],cmap='seismic')
        plt.axis('off')
        plt.title(title,fontsize=8)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        cbar=plt.colorbar()
        plt.clim(-1,1)
        cbar.ax.tick_params(labelsize=8) 
        
    total_maximum_on=np.max(maximum_responses_on,axis=0)
    total_maximum_off=np.max(maximum_responses_off,axis=0)
    # find which pixels are on and which OFF by comparing the total maximum matrices
    
    CS_mat=(total_maximum_on-total_maximum_off)>0 #true values are ON pixels, False values are off pixels
    
    #calculate CSI for all epochs after knowing which pixel is on and which OFF
    # # #  CSI_mat=np.zeros(all_differences_ON_OFF.shape)
    ##test 
    # calculate CS_mat and CSI by comparing on and off values within the pref dir
    CS_mat2=np.zeros(image_size)
    maxima_pref_dir_on=np.zeros((number_of_epochs,image_size[0],image_size[1]))
    maxima_pref_dir_off=np.zeros((number_of_epochs,image_size[0],image_size[1]))
    for pref_dir in range(number_of_epochs):
        temp1=np.where(pref_dir_mat==(pref_dir+1),1,0)
        maxima_pref_dir_on[pref_dir,:,:]=maximum_responses_on[pref_dir,:,:]*temp1
        maxima_pref_dir_off[pref_dir,:,:]=maximum_responses_off[pref_dir,:,:]*temp1
    maxima_pref_dir_on=np.sum(maxima_pref_dir_on,axis=0)
    maxima_pref_dir_off=np.sum(maxima_pref_dir_off,axis=0)
    CS_mat2=((maxima_pref_dir_on)-(maxima_pref_dir_off))

    #alternative: choose the slice for every pixel that maximizes the differences between ON and OFF
    # for now, this code is not used for anything else
    all_differences_ON_OFF_max= np.argmax(np.abs(all_differences_ON_OFF),axis=0)[np.newaxis,:,:]
    _,idx1,idx2=np.where(~np.isnan(all_differences_ON_OFF_max))
    CS_mat3=all_differences_ON_OFF[all_differences_ON_OFF_max.flatten(),idx1,idx2]
    CS_mat3=CS_mat3.reshape((image_size))

    # CS_mat2 has a lot of zeros. that complicates performing tresholding. to fix this, overwrite CS_mat2 with
    # the values from CS_mat wherever CS_mat2 is 0

    #CS_mat2=np.where(CS_mat2==0,CS_mat,CS_mat2)

    ## test continues. to calculate CSI use CS_mat2 and maxima
    CSI_mat_on=np.where(CS_mat2>0,CS_mat2/maxima_pref_dir_on,0)
    CSI_mat_off=np.where(CS_mat2<0,np.abs(CS_mat2)/maxima_pref_dir_off,0)
    if apply_CSI_treshold:
        CSI_tresholded_mask_on=CSI_mat_on>min_CSI
        CSI_tresholded_mask_off=CSI_mat_off>min_CSI
    else:
        CSI_tresholded_mask_on=CS_mat2>0
        CSI_tresholded_mask_off=CS_mat2<0
    parameters.update({'apply CSI treshold':apply_CSI_treshold,'min_CSI (if treshold applied)':min_CSI})
    gaussian_filtered=ndimage.gaussian_filter(np.abs(CS_mat2),1.5)
    second_treshold=filters.threshold_otsu(gaussian_filtered)     
    second_foreground=np.where(gaussian_filtered>second_treshold,1,0) 
    second_foreground=(foreground+second_foreground)>0
    



    ##### Juan test. use 2 otsus tresholds to maximize the number of pixels condidered
    second_foreground=((foreground + second_foreground)>0) ### test temporary !!!
    ### end of test

    #deprecated if apply_otsus==False:
    #    second_foreground=all_categories 
    
    ##
    # check if stuff worked
    # pref_dir_mat_visual=copy.deepcopy(pref_dir_mat)
    # plt.figure()
    # grid=plt.GridSpec(1, 2, wspace=0.1, hspace=0.2)
    # plt.subplot(grid[0])
    # plt.imshow(pref_dir_mat_visual*foreground)
    # plt.title('pixesl wo response filtering')
    # plt.subplot(grid[1])
    # plt.title('pixels with response filtering')
    # plt.imshow(pref_dir_mat_visual*foreground*responsive_pixels)

    #experiment (calculate foreground with the maxima):
    
    #ON_OFF_abs_diff=np.abs(total_maximum_on-total_maximum_off)

    # plt.figure()
    # plt.imshow(CS_mat2,cmap='seismic')
    # plt.colorbar()
    # plt.clim(-1,1)
    # plt.imshow(np.where(gaussian_filtered>second_treshold,1,0),cmap='gray_r',alpha=0.3)
    # plt.title('Otsus foreground from absolute diference between polarity maxima')
    plt.figure()
    plt.imshow(CS_mat2,cmap='seismic')
    plt.axis('off')
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.clim(-1,1)
    plt.imshow(foreground,cmap='gray_r',alpha=0.3)
    plt.axis('off')
    plt.title('Otsus foreground from mean_projection')

    # # # location_matrix_on=maximum_responses_on==total_maximum_on[np.newaxis,:,:]
    # # # location_matrix_off=maximum_responses_off==total_maximum_off[np.newaxis,:,:]
    # multiply matrices by the foreground and filter for contrast selectivity 
    # #(afterwards check for repeated pixels)

    # # # CSI_tresholded_mask_on=np.zeros((image_size))
    # # # CSI_tresholded_mask_off=np.zeros((image_size))

    # TODO modify this first if clustering does not work (calculate CSI within the pref_dir epoch)!!!! TODO TODO
    #if apply_CSI_treshold:
        #first calculate the CSI_mat using the location matrix
        # # # CSI_mat_on=np.where(location_matrix_on,(all_differences_ON_OFF)/(maximum_responses_on),0)
        # # # CSI_mat_off=np.where(location_matrix_off,(all_differences_OFF_ON)/(maximum_responses_off),0)
        
        
    plt.figure()
    # # # ceil=int(np.ceil(all_differences_ON_OFF.shape[0]/2))
    grid = plt.GridSpec(1,2, wspace=0.1, hspace=0.3)
    # # # for h in range(number_of_epochs):
    #to_plot=CSI_mat_on*CS_mat2[np.newaxis,:,:]
    title= 'CSI_ON' #+ ' epoch %s %s deg' %(h+1,angles[h])
    plt.subplot(grid[1])
    to_plot=np.where(CSI_mat_on*foreground*(CS_mat2>0)>0,CSI_mat_on*foreground*(CS_mat2>0),np.nan)
    ax=plt.imshow(to_plot,cmap='cool')
    plt.axis('off')
    plt.title(title,fontsize=9)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.clim(0,1)
    cbar.ax.tick_params(labelsize=8) 

    # # # plt.figure()
    # # # ceil=int(np.ceil(all_differences_ON_OFF.shape[0]/2))
    # # # grid = plt.GridSpec(ceil, 2, wspace=0.1, hspace=0.3)
    # # # for h in range(number_of_epochs):
    # # # to_plot=CSI_mat_off*~CS_mat2[np.newaxis,:,:]
    title= 'CSI_OFF' #+ ' epoch %s %s deg' %(h+1,angles[h])
    plt.subplot(grid[0])
    to_plot=np.where(CSI_mat_off*foreground*(CS_mat2<0)>0,CSI_mat_off*foreground*(CS_mat2<0),np.nan)
    ax=plt.imshow(to_plot,cmap='cool')
    plt.title(title,fontsize=9)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.clim(0,1)
    cbar.ax.tick_params(labelsize=8) 

        # # # CSI_mat_on=np.sum(CSI_mat_on,axis=0)
        # # # CSI_mat_off=np.sum(CSI_mat_off,axis=0)
        # # # CSI_tresholded_mask_on=CSI_mat_on>min_CSI
        # # # CSI_tresholded_mask_off=CSI_mat_off>min_CSI

        #CSI_tresholded_mask_on[CS_mat]=((total_maximum_on[CS_mat]-total_maximum_off[CS_mat])/total_maximum_on[CS_mat])>0.2
        #CSI_tresholded_mask_off[~CS_mat]=(total_maximum_off[~CS_mat]-total_maximum_on[~CS_mat])/total_maximum_off[~CS_mat]>0.2
        # # # CSI_tresholded_mask_on[CS_mat]=CSI_mat_on[CS_mat]>0.2
        # # # CSI_tresholded_mask_off[~CS_mat]=CSI_mat_off[~CS_mat]>0.2
        # # # location_matrix_on=location_matrix_on*(second_foreground*CSI_tresholded_mask_on)[np.newaxis,:,:]
        # # # location_matrix_off=location_matrix_off*(second_foreground*CSI_tresholded_mask_off)[np.newaxis,:,:]
        ## TODO TODO check if the CSI tresh mask is correct. 
    if apply_CSI_treshold:
        plt.figure()
        grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.3)
        plt.subplot(grid[0,0])
        plt.imshow(CSI_tresholded_mask_on*foreground,cmap='seismic')
        plt.axis('off')
        plt.title('ON pixels CSI>%f %f pixels' %(min_CSI,np.sum(CSI_tresholded_mask_on*foreground)))
        plt.subplot(grid[1,0])
        plt.imshow(CSI_tresholded_mask_off*foreground,cmap='seismic')
        plt.axis('off')
        plt.title('OFF pixels CSI>%f %f pixels' %(min_CSI,np.sum(CSI_tresholded_mask_off*foreground)))
        #TODO save figures
    # # # else:
    # # #     location_matrix_on=location_matrix_on*(second_foreground*CS_mat)[np.newaxis,:,:]
    # # #     location_matrix_off=location_matrix_off*(second_foreground*~CS_mat)[np.newaxis,:,:]
    # # #     CSI_tresholded_mask_on=CS_mat
    # # #     CSI_tresholded_mask_off=~CS_mat
    # # #     plt.figure()
    # # #     grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.3)
    # # #     plt.subplot(grid[0,0])
    # # #     plt.imshow(CS_mat*second_foreground)
    # # #     plt.title('ON pixels no CSI tresh %f pixels' %(np.sum(CS_mat*second_foreground)))
    # # #     plt.subplot(grid[1,0])
    # # #     plt.imshow(~CS_mat*second_foreground)
    # # #     plt.title('OFF pixels no CSI tresh %f' %(np.sum(~CS_mat*second_foreground)))
    #TODO save figures

    #check for duplicates and eliminate pixels with duplicates of pref dir TODO se if this sanity check makes sense
    # if (np.sum(location_matrix_on)>foreground.shape[1]*foreground.shape[0] or\
    #     np.sum(location_matrix_off)>foreground.shape[1]*foreground.shape[0]):
    #     # find the ambiguous pixel and erase it from foreground
    #     raise Exception('one pixel has more than one pref dir. not implemented')
    
    ### make a feature dataframe
    ### the matrices called location_matrix and timing max have all the necesary information:
    #indices 1 and 2 have the y,x locations and index 0 corresponds to the epoch.
    # a value of true in [0,0,0] means that for pixel [0,0] the pref direction is the  epoch [0]
    
    # with these 3 indices the actual timing inside the pref direction epoch is extracted from the 
    #response time matrix 

    # # # prefered_direction_bool_on=np.where(location_matrix_on==True)
    # # # prefered_direction_bool_off=np.where(location_matrix_off==True)
    
    # # # ####for testing
    # # # test_mat_pref_dir_on=np.zeros((image_size))
    # # # test_mat_pref_dir_off=np.zeros((image_size))
    # # # sanity_check_on=np.sum(location_matrix_on,axis=0)
    # # # sanity_check_off=np.sum(location_matrix_off,axis=0)
    # # # for pair_id in range(len(prefered_direction_bool_on[0])):
    # # #     id0=prefered_direction_bool_on[1][pair_id]
    # # #     id1=prefered_direction_bool_on[2][pair_id]
    # # #     value=prefered_direction_bool_on[0][pair_id]+1
    # # #     test_mat_pref_dir_on[id0,id1]=value
    # # # for pair_id in range(len(prefered_direction_bool_off[0])):
    # # #     id0=prefered_direction_bool_off[1][pair_id]
    # # #     id1=prefered_direction_bool_off[2][pair_id]
    # # #     value=prefered_direction_bool_off[0][pair_id]+1
    # # #     test_mat_pref_dir_off[id0,id1]=value
    plt.figure()
    grid = plt.GridSpec(1, 3, wspace=0.1, hspace=0.3)
    # # # plt.subplot(grid[0])
    # # # plt.imshow(test_mat_pref_dir_on)
    # # # plt.title('pref_dir_on')
    # # # plt.subplot(grid[1])
    # # # plt.imshow(test_mat_pref_dir_off)
    # # # plt.title('pref_dir_off')
    # # #plt.subplot(grid[2])
    plt.subplot(grid[0])    
    to_plot=(np.where(((pref_dir_mat)*foreground)>0),(pref_dir_mat)*foreground,np.nan)
    plt.imshow((pref_dir_mat)*foreground,cmap='jet')#*second_foreground*(CSI_tresholded_mask_on+CSI_tresholded_mask_off))
    plt.axis('off')
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.title('pref_dir_calculated_from begining')
    plt.subplot(grid[1])    
    to_plot=(np.where(((pref_dir_mat)*foreground*CSI_tresholded_mask_on)>0),(pref_dir_mat)*foreground,np.nan)
    plt.imshow((pref_dir_mat)*foreground*CSI_tresholded_mask_on,cmap='jet')#*second_foreground*(CSI_tresholded_mask_on+CSI_tresholded_mask_off))
    plt.axis('off')
    plt.title('prefdir ON CSI_treshold %f %s'%(min_CSI,apply_CSI_treshold))   
    plt.colorbar(fraction=0.03, pad=0.02)  
    plt.subplot(grid[2])    
    plt.imshow((pref_dir_mat)*foreground*CSI_tresholded_mask_off,cmap='jet')#*second_foreground*(CSI_tresholded_mask_on+CSI_tresholded_mask_off))
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.title('prefdir OFF CSI_treshold %f %s'%(min_CSI,apply_CSI_treshold))     
    plt.axis('off')
    ### end of test

    # create empty lists that will be filled up with indexes, timing, and pref direction of each pixel
    #number_of_valid_pixels=len(prefered_direction_bool_on[0])
    number_of_valid_pixels_on=np.sum(foreground*CSI_tresholded_mask_on*responsive_pixels)
    number_of_valid_pixels_on = int(number_of_valid_pixels_on)
    location_on=np.where(foreground*CSI_tresholded_mask_on*responsive_pixels)
    location_off=np.where(foreground*CSI_tresholded_mask_off*responsive_pixels)
    # check np.unique prefdirmat
    pix_index_1_on=np.zeros(int(number_of_valid_pixels_on),dtype=int)
    pix_index_2_on=np.zeros(number_of_valid_pixels_on,dtype=int)
    pref_dir_on=np.zeros(number_of_valid_pixels_on,dtype=int)
    response_time_on=np.zeros(number_of_valid_pixels_on,dtype=int)
    polarity=np.zeros(number_of_valid_pixels_on,dtype=int)
    ON_vs_OFF_diff_on=np.zeros(number_of_valid_pixels_on,dtype=float)
    
    pref_angle_on=np.zeros(number_of_valid_pixels_on,dtype=int)
    ### this next line is for testing
    test_fig= np.zeros(image_size,dtype=int)
    test_fig2= np.zeros(image_size,dtype=float)
    ###pause in testing
    # use the indices 
    for i in range(number_of_valid_pixels_on):
        pix_index_1_on[i]=int(location_on[0][i])
        pix_index_2_on[i]=int(location_on[1][i])
        pref_dir_on[i]=int(pref_dir_mat[location_on[0][i],location_on[1][i]])-1
        #pref_angle_on[i]=angles[pref_dir_on[i]]
        response_time_on[i]=timing_max_on[int(pref_dir_on[i]),int(pix_index_1_on[i]),int(pix_index_2_on[i])]
        polarity[i]=1
        #include in the feature table the difference between ON and OFF response
        ON_vs_OFF_diff_on[i]=CS_mat2[pix_index_1_on[i],pix_index_2_on[i]]
        ### this next line is for testing
        test_fig[pix_index_1_on[i],pix_index_2_on[i]]=response_time_on[i]
        test_fig2[pix_index_1_on[i],pix_index_2_on[i]]=ON_vs_OFF_diff_on[i]
        ###pause in testing
    ### this next line is for testing
    # plt.figure()
    # ceil=int(np.ceil(number_of_epochs/2))
    # grid = plt.GridSpec(ceil+1, 2, wspace=0.1, hspace=0.1)
    # plt.subplot(grid[0])
    # plt.title('timing on')
    # plt.imshow(test_fig,cmap='cool')
    # plt.colorbar(fraction=0.03, pad=0.02)
    # for ixx in range(1,number_of_epochs+1):
    #     plt.subplot(grid[ixx])
    #     plt.imshow(timing_max_off[ixx-1,:,:]*CSI_tresholded_mask_on*foreground,cmap='cool')
    #     plt.colorbar(fraction=0.03, pad=0.02)
    # plt.figure()
    # plt.imshow(test_fig2,cmap='cool')
    # plt.colorbar(fraction=0.03, pad=0.02)
    # plt.title('difference between on and off response for on pixs')
    ###pause in testing

    # # # number_of_valid_pixels=len(prefered_direction_bool_off[0])
    number_of_valid_pixels_off=np.sum(foreground*CSI_tresholded_mask_off*responsive_pixels)
    number_of_valid_pixels_off = int(number_of_valid_pixels_off)
    pix_index_1_off=np.zeros(number_of_valid_pixels_off,dtype=int)
    pix_index_2_off=np.zeros(number_of_valid_pixels_off,dtype=int)
    pref_dir_off=np.zeros(number_of_valid_pixels_off,dtype=int)
    response_time_off=np.zeros(number_of_valid_pixels_off,dtype=int)
    pref_angle_off=np.zeros(number_of_valid_pixels_off,dtype=int)
    polarity=np.zeros(number_of_valid_pixels_off)
    ON_vs_OFF_diff_off=np.zeros(number_of_valid_pixels_off,dtype=float)
    ### this next line is for testing
    test_fig=np.zeros(image_size,dtype=int)
    test_fig2= np.zeros(image_size,dtype=float)
    ###pause in testing
    for i2 in range(number_of_valid_pixels_off):
        pix_index_1_off[i2]=int(location_off[0][i2])
        pix_index_2_off[i2]=int(location_off[1][i2])
        pref_dir_off[i2]=int(pref_dir_mat[location_off[0][i2],location_off[1][i2]])-1
        #pref_angle_off[i2]=angles[pref_dir_off[i2]]
        response_time_off[i2]=timing_max_off[pref_dir_off[i2],pix_index_1_off[i2],pix_index_2_off[i2]]    
        #include in the feature table the difference between ON and OFF response
        ON_vs_OFF_diff_off[i2]=CS_mat2[pix_index_1_off[i2],pix_index_2_off[i2]]
        polarity[i2]=0
        ### this next line is for testing
        test_fig[pix_index_1_off[i2],pix_index_2_off[i2]]=response_time_off[i2]
        test_fig2[pix_index_1_off[i2],pix_index_2_off[i2]]=ON_vs_OFF_diff_off[i2]
        ###pause in testing

    ### this next line is for testing
    # plt.figure()
    # ceil=int(np.ceil(number_of_epochs/2))
    # grid = plt.GridSpec(ceil+1, 2, wspace=0.2, hspace=0.1)
    # plt.subplot(grid[0])
    # plt.title('timing off')
    # plt.imshow(test_fig)
    # for ixx in range(1,number_of_epochs+1):
    #     plt.subplot(grid[ixx])
    #     plt.imshow(timing_max_off[ixx-1,:,:]*CSI_tresholded_mask_off*foreground)
    # plt.figure()
    # plt.imshow(test_fig2)
    # plt.title('difference between on and off response for off pixs')
    ##pause in testing
    feature_dataframe_on=pd.DataFrame({'idx1':pix_index_1_on,'idx2':pix_index_2_on,'timing':response_time_on, 'pref_dir':pref_dir_on,'response_difference':ON_vs_OFF_diff_on})
    feature_dataframe_off=pd.DataFrame({'idx1':pix_index_1_off,'idx2':pix_index_2_off,'timing':response_time_off, 'pref_dir':pref_dir_off,'response_difference':ON_vs_OFF_diff_off})
    overall_feature_dataframe=pd.concat([feature_dataframe_on,feature_dataframe_off]).reset_index(drop=True)
    features={'on':feature_dataframe_on,'off':feature_dataframe_off,'overall':overall_feature_dataframe}
    metadata={'minCSI':min_CSI, 'apply_CSI_treshold':apply_CSI_treshold, 'manually_clasiffy_layers':manually_clasiffy_layers, 'apply_otsus':apply_otsus}
    # # #find the maximum on and off responses accross the whole trial averaged video
    # # #use those maximums to calculate a CSI
    # # on_maximum=np.max(on_averaged_video,axis=0)
    # # off_maximum=np.max(off_averaged_video,axis=0)

    # # #consider applying a reliability filter here

    
    # # CSI_mat=(on_maximum-off_maximum)
    # # CSI_mat2=copy.deepcopy(CSI_mat)
    # # #TODO change following lines for np.where
    
    # # #CSI_mat=np.where(CSI_mat>0,CSI_mat/on_maximum,CSI_mat)
    # # CSI_mat[CSI_mat>0]=CSI_mat2[CSI_mat2>0]/on_maximum[CSI_mat2>0]
    # # #CSI_mat=np.where(CSI_mat<0,CSI_mat/off_maximum,CSI_mat)
    # # CSI_mat[CSI_mat2<0]=CSI_mat2[CSI_mat2<0]/off_maximum[CSI_mat2<0]
    # # CS_mask_on=np.where(CSI_mat>min_CSI,1,0)
    # # CS_mask_off=np.where(CSI_mat<(min_CSI*-1),1,0)
    # # CS_mask={'on':CS_mask_on,'off':CS_mask_off,'general':np.abs(CSI_mat<(-1*min_CSI))}
    
    # # #plt.imshow(CS_mask)
    # # #plt.imshow(Background)
    # # #save the masks in the raw data folder

    # # return CS_mask # this is going to be stored in the ROI object
    plt.figure()
    plt.imshow(CS_mat2,cmap='seismic')
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.clim(-1,1)
    plt.title('ON-OFF responses in the pref direction')
    plt.axis('off')
    Tser=Tser.split('\\')[-2]
    save_str=savedir+'\\clustering_features_edges_stim_%s.pdf' %(Tser)
    if number_of_epochs>4:
        multipage(save_str, figs=None, dpi=600)
    else:
        multipage(save_str, figs=None, dpi=600)
    return features,metadata,parameters
    


            
def import_cat_ROIs(dataset,mot_corr=True,ROIs_label='categories',time_series=None,ignore_background=True):
    """
    takes ROIs selected with the sima ROIbuddy (Kaifosh et al., 2014) and  
    returns a list of category ROIs (for example, background and inclusion masks)
    and a list of category names correcponding to the category ROIs 
    
    args:
        
    returns:
    """
    # if mot_corr==True:
    #     sima_string='TserMC.sima'
    # else:
    #     sima_string='Tser.sima' 

    #os.chdir(Tser)
    Manual_ROIs=[]
    ROI_list=[]
    dfs = []
    filenames = []
    for csvfile in glob.glob(dataset+'/rois*.csv'):
        dfs.append(pd.read_csv(csvfile))
        filenames.append(csvfile)
    dictionaries = dataframe_to_dict(dfs,filenames)
    for layer in dictionaries.keys():
        for polygon in dictionaries[layer].keys():        
                Manual_ROIs.append(ROI_polygon(dictionaries[layer][polygon], layer, dataset))
    
    """
    put ROI masks in numpy array
    """
    check=0
    tag_list=[]
    
    for idx,element in enumerate(Manual_ROIs):
        if 'BG' in element.tag:
            BG=np.squeeze(np.array(element.mask))
            #background=np.reshape(np.array(element),(dataset.frame_shape[1],dataset.frame_shape[2],1))
            #background.astype(int)
            BG=BG.astype(bool)
            bg_index=idx
            check=1
        elif 'BG' not in element.tag:
            region=np.squeeze(np.array(element.mask))
            region.astype(int)
            ROI_list.append(region)
            tags=element.tag
            if len(tags)==0:
                tag_list.append('No_tag')
            else:
                tag_list.append(tags)
    #### note by Juan. background is detected automatically now commented next 2 lines
    if check==0:
        if ignore_background:
            if time_series is not None:
                for idx,k in enumerate(Manual_ROIs):
                    if k.tag == "rois_I_Zone":
                        I_zone = np.squeeze(np.array(k.mask))
                        I_zone.astype(int)
                        BG=extract_im_background_forclustering(time_series,I_zone)
                    else:
                        pass
            else:   
                BG=None
        else:
            raise Exception('no Background roi')    
    

    #np.delete(ROI_list,bg_index,axis=2)
   
   
    # """
    # save ROIs as pickle file
    # """
        
    #ROI_dict={'tags':tag_list, 'ROIs':ROI_list,'background':background}
        
    # with open('manual_ROIs.pkl', 'wb') as handle:
    #    pickle.dump(ROI_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # """
    #   produce an image with the ROIs
    # """
    # avg=mpimg.imread('Average_im.tif')
    # plt.figure()
    # plt.imshow(avg,'gray')
    # ROI_list=np.where(ROI_list==0,np.nan,ROI_list)       
    # colors=['Wistia','coolwarm','hsv','PRGn','Accent','Pastel1','PuOr']
    # count=0
    
    # for frame in range(len(ROI_list[0,0,:])-1):
        
    #     plt.imshow(ROI_list[:,:,frame],colors[count],alpha=0.4)
    #     count=count+1
    #     if count==7:
    #         count=0

    # plt.savefig('ROIs_manual.tif')
    # plt.close()
    
    return  BG,tag_list, ROI_list, I_zone  #this is equivalent to cat_names, cat_masks in Buraks code


def find_double_contrast_edgeresponse(curr_rois):
    #checks if T4T5 rois respond to both ON and OFF edges
    #labels the ROI object according to that
    print('NotImplemented')

    # loop trough epochs and find the maximum ON and OFF responses

def extract_null_dir_response(rois):
    '''searches for the response to the opposite direction of where the maximum measured response is'''
    for roi in rois:
        
        # index_of_max_response_ON=np.where(roi.max_response_all_epochs_ON==np.max(roi.max_response_all_epochs_ON))[0]
        # direction_of_max_response_ON=roi.direction_vector[index_of_max_response_ON]
        # index_of_max_response_OFF=np.where(roi.max_response_all_epochs_OFF==np.max(roi.max_response_all_epochs_OFF))[0]
        # direction_of_max_response_OFF=roi.direction_vector[index_of_max_response_OFF]
        
        #find the direction of max response
        direction_of_max_response_ON=roi.direction_vector[roi.max_resp_idx_ON]
        direction_of_max_response_OFF=roi.direction_vector[roi.max_resp_idx_OFF]
        
        if direction_of_max_response_ON< 180:
            null_dir_ON=float(direction_of_max_response_ON + 180)       
        else:
            null_dir_ON=float(direction_of_max_response_ON - 180)
        if direction_of_max_response_OFF< 180:
            null_dir_OFF=float(direction_of_max_response_OFF + 180)
        else:
            null_dir_OFF=float(direction_of_max_response_OFF - 180)

        idx_of_null_ON=np.where(np.array(roi.direction_vector)==null_dir_ON)[0][0]
        roi.norm_null_dir_resp_ON=roi.max_resp_all_epochs_ON[idx_of_null_ON]/roi.max_response_ON
        idx_of_null_OFF=np.where(np.array(roi.direction_vector)==null_dir_OFF)[0][0]
        roi.norm_null_dir_resp_OFF=roi.max_resp_all_epochs_OFF[idx_of_null_OFF]/roi.max_response_OFF
    return rois
        #do the same for off

def normalize_stack(stack):
    #normalize if needed
    if np.max(stack)>1:
        stack=np.divide(stack.astype(float),np.max(stack)) #double check if this is working correctly
        #mean_image= stack.mean(0)
    return stack

def run_cluster_analysis_MH(dataDir,fig_dir,stack,stimulus_information,imaging_information,category_names,I_zone,response_treshold=0,manually_clasiffy_layers=False,apply_otsus=True,clustering_mode='timing&contrast',Tser=None):

    pixel_area=imaging_information['pixel_size']**2 # units: um^2/pixel
    # ROI_size limits (hardcoded for T4T5, based on Miriam Henning's original code)
    lower_size_limit=int(round(0.7/pixel_area)) # the limits (1um^2) are hardcoded here (the units are um^2). they were chosen by Miri based on confocal measurements
    #lower_size_limit=5 # min size was changed!!
    # TODO TODO choose the minimum size as a function of the resolution of the microscope. at around 5-6zoom
    
    #lower_size_limit=5 # minimum number of pixels, a relevant individual terminal could be even represented in one pixel, a higher number is chosen to avoid spurious results
    higher_size_limit=np.ceil(round(6.25/pixel_area)) #was 6.25# these limits are within the range calculated by Maisak... ()
    
    #manual category selection is default, automatic not yet implemented
    # manually select category ROIs
    # [cat_masks, cat_names] = select_regions(mean_image, 
    #                                     image_cmap="viridis",
    #                                     pause_t=8)
    stack_to_normalize=copy.deepcopy(stack)
    norm_stack=normalize_stack(stack_to_normalize)
    
    #categories,I_zone,cropped_stack=crop_stack(I_zone,categories,norm_stack) # DEPRECATED
    
    
    FG_image,foreground,background=extract_im_background_forclustering(norm_stack,I_zone)
    #subtract the background to alleviate artifacts from stimulation bleedthrough
    background_trace=(np.sum(norm_stack*background[np.newaxis,:,:],axis=(1,2))/np.sum(background))[:,np.newaxis,np.newaxis]
    #background_trace=(norm_stack*background/background)[:]
    stack_subtracted=norm_stack.astype(float)-background_trace
    
    
    # stimulus chekpoint
    if len(np.unique(np.array(stimulus_information['angle'])))>=3 \
        and (np.sum(np.array(stimulus_information['stimtype'])=='driftingstripe')>=4 \
            or np.sum(np.array(stimulus_information['stimtype'])=='ADS')>=4\
            or np.sum(np.array(stimulus_information['stimtype'])=='G')>=4):
        
    # if (stimulus_information['stim_name']=='DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_ONEDGEFIRST_80sec.txt'):
    # if (stimulus_information['stim_name']=='DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt' or \
    #     stimulus_information['stim_name']=='DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_8Dirs_optimized.txt'\
    #     or stimulus_information['stim_name']== 'DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_4Dirs_optimized.txt'\
    #     or stimulus_information['stim_name']=='DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_8Dirs_optimized_fullcontrast.txt'\
    #     or stimulus_information['stim_name']== 'DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_8Dirs_optimized_80_3s.txt'
    #     or stimulus_information['stim_name']== 'DriftingEdge_LumDecLumInc_1period_20degPerSec_90deg_BlankRand_8Dirs_optimized_80_3s.txt'):
        print('correct stimulus, proceeding to cluster extraction') 
    elif np.sum(np.array(stimulus_information['stimtype'])=='driftingstripe')==1:
        print('cluster extraction with single direction')
    else:
        raise Exception('not implemented. stimulus does not match')

    # build a feature table to perform clustering on #TODO maybe include a lonelypixel filter and a response treshold
    #save_path=dataDir + 'preprocessing'
    save_path=fig_dir
    features,metadata,extra_params=produce_feature_table_forClustering(stack_subtracted,foreground,response_treshold,save_path,cat_names=category_names,min_CSI=0,stimulus_information= stimulus_information, imaging_information=imaging_information,manually_clasiffy_layers=manually_clasiffy_layers,apply_otsus=apply_otsus,Tser=Tser)
    clusters, all_rois_image,polarity_list,features_for_clustering =hierarchical_clustering_analysis(stack_subtracted,foreground.shape,features,lower_size_limit,higher_size_limit,use_pref_all_info=True,saving_path=save_path,clustering_mode=clustering_mode)
    metadata.update({'min_ROIsize_pixels':lower_size_limit,'max_ROIsize_pixels':higher_size_limit,'features used clustering':features_for_clustering})#,'max_clusters_Hierchachical_clustering':max_clusters})
    metadata.update(extra_params)
    #print(metadata)
    return clusters, all_rois_image, polarity_list, background, metadata

def hierarchical_clustering_analysis(stack,imsize,features,lower_size_limit,higher_size_limit,use_pref_all_info=True,saving_path=None,CSI_filtered=False,clustering_mode='contrast_dif'):
    
    original_size=stack[0,:,:].shape
    #TODO put in here the cluster size tresholds
    ROI_list=[]
    roi_sizes=[]
    polarity_list=[]
    pref_direction=[]
    all_rois_image_on=np.zeros(imsize)
    all_rois_image_on[:]=np.nan
    all_rois_image_off=np.zeros(imsize)
    all_rois_image_off[:]=np.nan
    all_rois_image=np.zeros(imsize)
    all_rois_image[:]=np.nan
    roi_num_on=0
    roi_num_off=0
    roi_num=0
    overall_feature_table={}
    for polarity in ['off','on']: #features.keys():
        feature_df=features[polarity]
        if polarity=='off':
            aa='aaa'
        for layer in np.unique(feature_df['pref_dir']):
            feature_df_layer=feature_df.loc[feature_df['pref_dir']==layer]
            if len(feature_df_layer) <= 1:
                print ( 'direction or layer %s has no valid clusters' %(layer))
                continue # no clusters can be calculated with 1 datapoint
            
            #feature_df_layer=feature_df
            #feature_df_layer=feature_df_layer[['idx1', 'idx2', 'timing' ,'response_difference']] #before it included timing
            #pref_dir=feature_df_layer['pref_angle'][0].values() #TODO add actual pref direction to the feature table

            if clustering_mode=='contrast_dif':
                feature_df_layer=feature_df_layer[['idx1', 'idx2', 'response_difference']] 
                clustering_features='position and response difference ON-OFF'
            elif clustering_mode=='timing':
                feature_df_layer=feature_df_layer[['idx1', 'idx2', 'timing' ]] 
                clustering_features='position and time of response'
            elif clustering_mode=='timing&contrast':
                feature_df_layer=feature_df_layer[['idx1', 'idx2', 'timing', 'response_difference' ]] 
                clustering_features='position, time of response and response difference ON-OFF'
            feature_df_layer=feature_df_layer.to_numpy(dtype=float,copy=True)
            max_cluster_number=feature_df_layer.shape[0] 
            valid_clusters_vector=np.zeros(max_cluster_number)
            valid_clusters_vector[:]=np.nan             
            
            for ix,i in enumerate(range(1,max_cluster_number+1)):
                #perform clustering operation as done by Miriam
                clusters=fclusterdata(feature_df_layer,i,criterion='maxclust',metric='euclidean',method='average')
                # clusters is a vector asigning an observation (index) to a cluster number (value). i.e clusters[observation]=cluster number
                
                # now count the number of clusters within our size limits (first calculate number of pixels per cluster)
                cluster_number,counts =np.unique(clusters,return_counts=True)
                #cluster_area=counts
                index_of_validClusters_1=np.where(counts>lower_size_limit)[0]#[0]
                
                # index_of_validClusters_2= np.where(counts<higher_size_limit)
                # index_of_validClusters=np.intersect1d(index_of_validClusters_1,index_of_validClusters_2)                    
                # valid_clusters_vector[ix]=len(index_of_validClusters)
                # valid_clusters_vector[ix]=len(index_of_validClusters_1)
                index_of_validClusters_2=[]
                #TODO filter based on span instead
                copy_of_feature_df_layer=copy.deepcopy(feature_df_layer)
                copy_of_feature_df_layer[:,2]=clusters
                for clust_index,indiv_cluster in enumerate(cluster_number):
                    cluster_location=np.where(copy_of_feature_df_layer[:,2]==indiv_cluster)[0]
                    cluster_subset=copy_of_feature_df_layer[cluster_location]
                    index_tuples=list(zip(cluster_subset[:,0],cluster_subset[:,1]))
                    maximum_distance=np.nanmax(cdist(index_tuples,index_tuples,'euclidean'))
                    if maximum_distance<np.sqrt(higher_size_limit):
                        index_of_validClusters_2.append(clust_index)
                    # for each cluster, find the spatial components and find the maximum
                    # distance within, then include in valid clusters if it is less than 
                
                index_of_validClusters_2=np.array(index_of_validClusters_2)
                #index_of_validClusters_2=[]
                index_of_validClusters=np.intersect1d(index_of_validClusters_1,index_of_validClusters_2)                    
                valid_clusters_vector[ix]=len(index_of_validClusters)
            
            ##indentation here.... careful
            # if len(valid_clusters_vector)==0:
            #     break
            # if  np.nanmax(valid_clusters_vector)==0:
            #     break
            cluster_number,counts=[],[]
            optimal_maxnumber_of_clusters=np.where(valid_clusters_vector==np.nanmax(valid_clusters_vector))[0]
            
            #repeat the clustering with the optimal condition, and repeat the size filtering
            final_clusters=fclusterdata(feature_df_layer,optimal_maxnumber_of_clusters[-1],criterion='maxclust',metric='euclidean',method='average') 
            # append the cluster identity to the timing array (replace timing with cluster identity)
            pixel_cluster_identity_matrix=copy.deepcopy(feature_df_layer)
            pixel_cluster_identity_matrix[:,2]=final_clusters
            pixel_cluster_identity_matrix=pixel_cluster_identity_matrix[:,0:3]
            #make pandas dataframe with the cluster identity matrix
            cluster_df=pd.DataFrame(data=pixel_cluster_identity_matrix,index=range(0,len(pixel_cluster_identity_matrix[:,0])),columns=['x','y','cluster'])
            cluster_df=cluster_df.astype(int)
            cluster_number,counts=np.unique(final_clusters,return_counts=True)
            #exclude invalid clusters, repeat the same process done in the for loop above
            index_valid_clusters_1=np.where(counts>lower_size_limit)[0] ### lower_size_limit
            index_validClusters_2=[]
            #### old version: index_valid_clusters_2=np.where(counts<higher_size_limit)
            for clust_index,indiv_cluster in enumerate(cluster_number):
                cluster_location=np.where(pixel_cluster_identity_matrix[:,2]==indiv_cluster)[0]
                cluster_subset=pixel_cluster_identity_matrix[cluster_location]
                index_tuples=list(zip(cluster_subset[:,0],cluster_subset[:,1]))
                maximum_distance=np.nanmax(cdist(index_tuples,index_tuples,'euclidean'))
                if maximum_distance<np.sqrt(higher_size_limit): #higher_size_limit
                    index_validClusters_2.append(clust_index)
                # # for each cluster, find the spatial components and find the maximum
                # distance within, then include in valid clusters if it is less than 
            index_valid_clusters_2=np.array(index_of_validClusters_2)
            #index_valid_clusters_2=[]
            index_valid_clusters_3=np.intersect1d(index_valid_clusters_1,index_valid_clusters_2)
            if len(index_valid_clusters_3)==0:
                continue
            valid_cluster_id=cluster_number[index_valid_clusters_3]
            
            final_cluster_df=cluster_df[cluster_df['cluster'].isin(valid_cluster_id)]
            
            #make ROIs list an array. first initialize the mask, then fill it up
            current_mask=np.zeros(original_size).astype(int)            

            for counter,cluster_number in enumerate(np.unique(final_cluster_df['cluster'])):
                current_mask[:]=0
                subset=final_cluster_df.loc[final_cluster_df['cluster']==cluster_number]
                subset_x=subset['x'].values
                subset_y=subset['y'].values
                for x,y in zip(subset_x,subset_y):
                    current_mask[(x,y)]=1
                roi_num=roi_num+1
                if polarity=='on':
                    roi_num_on=roi_num_on+1
                else:
                    roi_num_off=roi_num_off+1
                ## append mask and mask properties to the corresponding lists  
                # if (x_span==False and y_span==False):
                current_mask=current_mask.astype(int)#current_mask=current_mask.astype(bool)
                ROI_list.append(copy.deepcopy(current_mask)) # check if append work
                polarity_list.append(polarity)
                roi_sizes.append(np.nansum(current_mask))
                for x,y in zip(subset_x,subset_y):
                    if polarity=='on':
                        all_rois_image_on[(x,y)]=counter+1
                    else:
                        all_rois_image_off[(x,y)]=counter+1
                    all_rois_image[(x,y)]=counter+1
#                all_rois_image[0:int(np.sqrt(higher_size_limit)),0]=0
#                all_rois_image[0:lower_size_limit,1]=0
    #show figure
    ROIs_im_dict={'on':all_rois_image_on,'off':all_rois_image_off,'all':all_rois_image}
    #plt.close('all')
    plt.figure()
    grid = plt.GridSpec(1, 3, wspace=0.2, hspace=0.2)
    for idx,types in enumerate(ROIs_im_dict.keys()):
        plt.subplot(grid[idx])
        if types=='on':
            num=roi_num_on
        elif types=='all':
            num=roi_num
        else:
            num=roi_num_off
        
        plt.title('%s %s Rois'  %(types,num))        
        plt.imshow(stack.mean(0),cmap='viridis')
        plt.imshow(ROIs_im_dict[types],cmap='prism',alpha=0.8)        
    roi_list=np.array(ROI_list)
    if saving_path is not None:
        if clustering_mode=='contrast_dif':
            try:
               os.mkdir(saving_path + '\\contrast_dif_clustmode')
               str_file=saving_path + '\\contrast_dif_clustmode'
            except:
                str_file=saving_path + '\\contrast_dif_clustmode'
            finally:
                pass
        elif clustering_mode=='timing&contrast':
            try:
               os.mkdir(saving_path + '\\timing_and_contrast_clustmode')
               str_file=saving_path + '\\timing_and_contrast_clustmode'
            except:
                str_file=saving_path + '\\timing_and_contrast_clustmode'
            finally:
                pass
        elif clustering_mode=='timing':
            try:
               os.mkdir(saving_path + '\\timing_clustmode')
               str_file=saving_path + '\\timing_clustmode'
            except:
                str_file=saving_path + '\\timing_clustmode'
            finally:
                pass
        #str_file=os.path.split(saving_path)[0]+'\\detected_clusters_%s-%s.pdf' %(higher_size_limit,lower_size_limit)
        str_file=str_file+'\\detected_clusters_%s-%s_pixels.pdf' %(higher_size_limit,lower_size_limit)
        plt.savefig(str_file)
        plt.close('all')

    return roi_list, all_rois_image, polarity_list,clustering_features#optimal_maxnumber_of_clusters[-1]


def find_CS_clusters(I_zone,foreground,CS_mask,norm_stack,treshold_mask,lower_size_limit,higher_size_limit,categories):
    
    # filter the stack with the background mask and the treshold mask:
    use_cats=True #temporary thing for testing
    local_stack=copy.deepcopy(norm_stack)
    #background=np.array(background,dtype=bool)
    local_stack=local_stack*foreground[np.newaxis,:,:]*I_zone[np.newaxis,:,:]
    max_stack=np.nanmax(local_stack,axis=0)
    response_mask=max_stack>treshold_mask
    max_stack=max_stack*response_mask
    #max_stack=np.where(max_stack!=0,max_stack,0)

    #loop through polarities. each polarity should have a contrast selectivity mask 
    ROI_list=[]
    roi_sizes=[]
    polarity_list=[]
    all_rois_image=np.zeros(norm_stack.shape[1:])
    all_rois_image[:]=np.nan
    if use_cats==True:
        for ix1,category in enumerate(categories):
            cat_mask=np.squeeze(np.array(category))
            for polarity in ['on','off']:
                
                polarity=polarity
                CS_filter=CS_mask[polarity]
                pol_specific_max=copy.deepcopy(max_stack)*CS_filter*cat_mask
                #pol_specific_max=np.where(max_stack!=0,pol_specific_max,0)
                copy_nan_pol_specific_max=np.where(pol_specific_max==0,np.nan,pol_specific_max)
                timing_indices=np.where(local_stack==copy_nan_pol_specific_max[np.newaxis,:,:])
                #timing indices will contain 3 linear arrays containing indices for 3 dimensions: t,x,y. we're
                #intrerested in keeping the timing information (dimension 0)
                t_x_y_reponse_array=np.zeros((len(timing_indices[0]),3))
                for ix2 in range(3):
                    t_x_y_reponse_array[:,ix2]=timing_indices[ix2]

                # # calculate the euclidean distance matrix
                # euclidean_distance=pdist(timingAndPosition_reponse_array, metric='euclidean')
                # #perform clustering including size constraints (hierarchical aglomerative clustering)
                # linkage(euclidean_distance,method='average')

                # the timing and position response array is a nxm matrix(n=#valid pixels matrix, m=3). m is the axis containing x,y,timing of max response
                max_cluster_number=t_x_y_reponse_array.shape[0]
                valid_clusters_vector=np.zeros(max_cluster_number)
                valid_clusters_vector[:]=np.nan 
                for ix,i in enumerate(range(1,max_cluster_number+1)):
                    #perform clustering operation as done by Miriam
                    clusters=fclusterdata(t_x_y_reponse_array,i,criterion='maxclust',metric='euclidean',method='average')
                    # cluesters is a vector asigning an observation (index) to a cluster number (value). i.e clusters[observation]=cluster number
                    
                    # now count the number of clusters within our size limits (first calculate number of pixels per cluster)
                    cluster_number,counts =np.unique(clusters,return_counts=True)
                    #cluster_area=counts
                    index_of_validClusters_1=np.where(counts>lower_size_limit)#[0]
                    index_of_validClusters_2= np.where(counts<higher_size_limit)
                    index_of_validClusters=np.intersect1d(index_of_validClusters_1,index_of_validClusters_2)                    
                    valid_clusters_vector[ix]=len(index_of_validClusters)
                    cluster_number,counts=[],[]
                optimal_maxnumber_of_clusters=np.where(valid_clusters_vector==np.nanmax(valid_clusters_vector))[0]
                #repeat the clustering with the optimal condition, and repeat the size filtering
                final_clusters=fclusterdata(t_x_y_reponse_array,optimal_maxnumber_of_clusters[0],criterion='maxclust',metric='euclidean',method='average') 
                ##final_clusters=fclusterdata(t_x_y_reponse_array,593,criterion='maxclust',metric='euclidean',method='average') 

                # append the cluster identity to the timing array (replace timing with cluster identity)
                pixel_cluster_identity_matrix=copy.deepcopy(t_x_y_reponse_array)
                pixel_cluster_identity_matrix[:,0]=final_clusters
                #make pandas dataframe with the cluster identity matrix
                cluster_df=pd.DataFrame(data=pixel_cluster_identity_matrix,index=range(0,len(pixel_cluster_identity_matrix[:,0])),columns=['cluster','x','y'])
                cluster_df=cluster_df.astype(int)
                cluster_number,counts=np.unique(final_clusters,return_counts=True)
                #exclude invalid clusters
                index_valid_clusters_1=np.where(counts>lower_size_limit)
                index_valid_clusters_2=np.where(counts<higher_size_limit)
                index_valid_clusters_3=np.intersect1d(index_valid_clusters_1,index_valid_clusters_2)
                valid_cluster_id=cluster_number[index_valid_clusters_3]
                #TODO filter according to full span of ROI (eliminate ROIs that are too stretched)

                
                final_cluster_df=cluster_df[cluster_df['cluster'].isin(valid_cluster_id)]
                
                #make ROIs list an array. first initialize the mask, then fill it up
                current_mask=np.zeros(norm_stack.shape[1:])
                
            
                for counter,cluster_number in enumerate(np.unique(final_cluster_df['cluster'])):
                    current_mask[:]=np.nan
                    subset=final_cluster_df.loc[final_cluster_df['cluster']==cluster_number]
                    subset_x=subset['x'].values
                    subset_y=subset['y'].values
                    for x,y in zip(subset_x,subset_y):
                        current_mask[(x,y)]=1
                    
                    #filter according to full span of ROI (eliminate ROIs that are too stretched)
                    x_span=(np.max(subset_x)-np.min(subset_x))>round(np.sqrt(higher_size_limit))
                    y_span=(np.max(subset_y)-np.min(subset_y))>round(np.sqrt(higher_size_limit))
                    if (x_span==False and y_span==False):
                        ROI_list.append(copy.deepcopy(current_mask)) # check if append work
                        polarity_list.append(polarity)
                        roi_sizes.append(np.nansum(current_mask))
                    for x,y in zip(subset_x,subset_y):
                        all_rois_image[(x,y)]=counter+1
                    #plt.figure()
                    #plt.imshow(current_mask)
                 

    else:
        for polarity in ['on','off']:
            countt=0
            polarity=polarity
            CS_filter=CS_mask[polarity]
            pol_specific_max=copy.deepcopy(max_stack)*CS_filter*I_zone
            pol_specific_max=np.where(max_stack!=0,pol_specific_max,np.nan)
            timing_indices=np.where(local_stack==pol_specific_max[np.newaxis,:,:])
            #timing indices will contain 3 linear arrays containing indices for 3 dimensions: t,x,y. we're
            #intrerested in keeping the timing information (dimension 0)
            t_x_y_reponse_array=np.zeros((len(timing_indices[0]),3))
            for ix in range(3):
                t_x_y_reponse_array[:,ix]=timing_indices[ix]

            # # calculate the euclidean distance matrix
            # euclidean_distance=pdist(timingAndPosition_reponse_array, metric='euclidean')
            # #perform clustering including size constraints (hierarchical aglomerative clustering)
            # linkage(euclidean_distance,method='average')

            # the timing and position response array is a nxm matrix(n=#valid pixels matrix, m=3). m is the axis containing x,y,timing of max response
            max_cluster_number=t_x_y_reponse_array.shape[0]
            valid_clusters_vector=np.zeros(max_cluster_number)
            valid_clusters_vector[:]=np.nan 
            for ix,i in enumerate(range(1,max_cluster_number+1)):
                #perform clustering operation as done by Miriam
                clusters=fclusterdata(t_x_y_reponse_array,i,criterion='maxclust',metric='euclidean',method='average')
                # cluesters is a vector asigning an observation (index) to a cluster number (value). i.e clusters[observation]=cluster number
                
                # now count the number of clusters within our size limits (first calculate number of pixels per cluster)
                cluster_number,counts =np.unique(clusters,return_counts=True)
                #cluster_area=counts
                index_of_validClusters_1=np.where(counts>lower_size_limit)#[0]
                index_of_validClusters_2= np.where(counts<higher_size_limit)
                index_of_validClusters=np.intersect1d(index_of_validClusters_1,index_of_validClusters_2)
                valid_clusters_vector[ix]=len(index_of_validClusters)
                cluster_number,counts=[],[]
            optimal_maxnumber_of_clusters=np.where(valid_clusters_vector==np.nanmax(valid_clusters_vector))[0]
            #repeat the clustering with the optimal condition, and repeat the size filtering
            final_clusters=fclusterdata(t_x_y_reponse_array,optimal_maxnumber_of_clusters[0],criterion='maxclust',metric='euclidean',method='average') 
            #final_clusters=fclusterdata(t_x_y_reponse_array,593,criterion='maxclust',metric='euclidean',method='average') 

            # append the cluster identity to the timing array (replace timing with cluster identity)
            pixel_cluster_identity_matrix=copy.deepcopy(t_x_y_reponse_array)
            pixel_cluster_identity_matrix[:,0]=final_clusters
            #make pandas dataframe with the cluster identity matrix
            cluster_df=pd.DataFrame(data=pixel_cluster_identity_matrix,index=range(0,len(pixel_cluster_identity_matrix[:,0])),columns=['cluster','x','y'])
            cluster_df=cluster_df.astype(int)
            cluster_number,counts=np.unique(final_clusters,return_counts=True)
            #exclude invalid clusters
            index_valid_clusters_1=np.where(counts>lower_size_limit)
            index_valid_clusters_2=np.where(counts<higher_size_limit)
            index_valid_clusters_3=np.intersect1d(index_valid_clusters_1,index_valid_clusters_2)
            valid_cluster_id=cluster_number[index_valid_clusters_3]
            
            
            final_cluster_df=cluster_df[cluster_df['cluster'].isin(valid_cluster_id)]
            
            #make ROIs list an array. first initialize the mask, then fill it up
            current_mask=np.zeros(norm_stack.shape[1:])
            
            
            for counter,cluster_number in enumerate(np.unique(final_cluster_df['cluster'])):
                current_mask[:]=np.nan
                subset=final_cluster_df.loc[final_cluster_df['cluster']==cluster_number]
                subset_x=subset['x'].values
                subset_y=subset['y'].values
                for x,y in zip(subset_x,subset_y):
                    current_mask[(x,y)]=1
                #filter according to full span of ROI (eliminate ROIs that are too stretched)
                x_span=(np.max(subset_x)-np.min(subset_x))>round(np.sqrt(higher_size_limit))
                y_span=(np.max(subset_y)-np.min(subset_y))>round(np.sqrt(higher_size_limit))
                # plt.figure()
                # plt.imshow(current_mask)
                if (x_span==False and y_span==False):
                    ROI_list.append(copy.deepcopy(current_mask)) # check if append work
                    polarity_list.append(polarity)
                    roi_sizes.append(np.nansum(current_mask))
                    for x,y in zip(subset_x,subset_y):
                        all_rois_image[(x,y)]=counter+1
                else:
                    countt=+1


    plt.figure()
    plt.imshow(norm_stack.mean(0),cmap='viridis')
    plt.imshow(all_rois_image,cmap='prism',alpha=0.8)        
    roi_list=np.array(ROI_list)
    return roi_list, polarity_list, all_rois_image

def produce_treshold_mask(stack,response_treshold=4):
    '''produces a mask that includes the pixels that have a measurable response 
    within a time series '''

    #CS_mask_gen=CS_mask['general']
    mean_image=stack.mean(0)
    SD_projection=np.std(stack,axis=0) #confirm axis
    treshold_mask=mean_image+(response_treshold*SD_projection)
    return treshold_mask

def crop_stack(I_zone,categories,stack):
    #Uses an ROI (I_zone) to crop the stack in order to exclude cell bodies, 
    # for now, the function on chops in the y axis    
    
    I_zone_border=np.where(I_zone)
    I_zone_border_y=np.max(I_zone_border[0])
    I_zone_border_x=np.max(I_zone_border[1])
    cropped_stack=stack[:,0:I_zone_border_y+1,0:I_zone_border_x+1]

    ###temp monkey patch
    #cropped_stack=copy.deepcopy(stack)
    
    I_zone=I_zone[0:I_zone_border_y+1,0:I_zone_border_x+1]
    for idx,cat in enumerate(categories):
        cat=np.squeeze(np.array(cat))
        categories[idx]=cat[0:I_zone_border_y+1,0:I_zone_border_x+1] 
        #categories[idx]=cat
    return categories,I_zone,cropped_stack

    
# %%
