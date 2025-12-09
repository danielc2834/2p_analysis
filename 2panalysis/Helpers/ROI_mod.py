#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:09:54 2019

@author: burakgur
"""
import chunk
from csv import excel_tab
from logging import exception, raiseExceptions
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#from numpy.linalg.linalg import _raise_linalgerror_nonposdef
import seaborn as sns
import pandas as pd
# import sima # Commented out by seb
import copy
#import scipy
from scipy import signal
# from roipoly import RoiPoly #commented by Juan
from scipy.optimize import curve_fit,Bounds,least_squares
from scipy.stats import linregress
from skimage import filters
from scipy.stats import pearsonr
from scipy import fft
from scipy.signal import blackman
from scipy.signal import detrend
import Helpers.post_analysis_core as pac
import Helpers.process_mov_core as pmc
# import cPickle
import pickle
import cmath
from itertools import permutations
#from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as patches
import glob
import scipy.ndimage
import imageio
from scipy.ndimage import map_coordinates
from skimage.draw import line
from skimage.measure import profile_line
from scipy.spatial import distance
import cv2
from Helpers.core_functions import readStimOut
# from numba import njit
import time
from scipy.signal import fftconvolve

# @njit
# def rev_corr_step(stim,trace_val,STRF):
     
#     return STRF + (stim*trace_val)


# @njit 
# def compute_trace_value(stim_chunk,STRF):

#     return  np.sum(stim_chunk*STRF)

def compute_STRF(trace_indexes_vals, stim_frame_vals, trace, stimulus, snippet, STA_array):
    
    trace_indexes_vals = trace_indexes_vals-1
    initial_pad = np.zeros((snippet,stimulus.shape[1],stimulus.shape[2]))
    for ix in range(len(trace_indexes_vals)):
        current_stim_frame = stim_frame_vals[ix]
        if current_stim_frame == -1:
            continue

        start_idx = current_stim_frame - snippet
        end_idx = current_stim_frame + 10

        if start_idx < 0:# or end_idx > stimulus.shape[0]:
            pad = initial_pad [-np.abs(start_idx):,:,:]
            stim_chunk = np.concatenate((pad,stimulus[:end_idx, :, :]),axis=0)
        else:
            stim_chunk = stimulus[start_idx:end_idx, :, :]
        STA_array = rev_corr_step(stim_chunk,trace[trace_indexes_vals[ix]],STA_array)
        
    return STA_array

# @njit
def compute_STRF_prediction(trace_indexes_vals, excluded, stim_frame_vals, trace, stimulus, snippet, STRF):
    
    #initialize output
    strf_prediction = np.zeros(trace.shape[0])
    strf_prediction[:] = np.nan 
    copy_trace = np.zeros(trace.shape[0])
    copy_trace[:] = np.nan 
    initial_pad = np.zeros((snippet,stimulus.shape[1],stimulus.shape[2]))
    
    
    for ix in range(len(trace_indexes_vals)):
        current_stim_frame = stim_frame_vals[ix]

        start_idx = current_stim_frame - snippet
        end_idx = current_stim_frame 

        if excluded[ix]==1:
            continue    
        if start_idx < 0:# or end_idx > stimulus.shape[0]:
            pad = initial_pad [-np.abs(start_idx):,:,:]
            stim_chunk = np.concatenate([pad,stimulus[:end_idx, :, :]],axis=0)
        else:
            stim_chunk = stimulus[start_idx:end_idx, :, :] 

        # predict a time point 
        strf_prediction[trace_indexes_vals[ix]]= compute_trace_value (stim_chunk,STRF) 
        copy_trace[trace_indexes_vals[ix]] = trace[trace_indexes_vals[ix]]

    return strf_prediction,copy_trace

def optimized_trace_prediction(stimulus_preprocessed, restricted_RF):

    kernel = restricted_RF[:30,:,:]
    kernel_pad = np.zeros((np.shape(stimulus_preprocessed)[0]-30,kernel.shape[1],kernel.shape[2]))
    kernel = np.concatenate([kernel_pad,kernel],axis=0)
    predicted_trace = scipy.fftpack.fft(stimulus_preprocessed,axis=0,overwrite_x=True)*scipy.fftpack.fft(kernel,axis=0,overwrite_x=True)
    predicted_trace = scipy.fftpack.ifft(predicted_trace,axis=0,overwrite_x=True)
    #predicted_trace = fftconvolve(stimulus_preprocessed, kernel, mode='same', axes=0)
    predicted_trace = np.sum(predicted_trace,axis = (1,2))
    #predicted_trace = convolve1d(np.flatten(stimulus),np.flatten(kernel))

    return predicted_trace


class ROI_bg: 
    """A region of interest from an image sequence """
    
    def __init__(self,Mask = None, experiment_info = None,imaging_info = None): 
        """ 
        Initialized with a mask and optionally with experiment and imaging
        information
        """
        if (Mask is None):
            raise TypeError('ROI_bg: ROI must be initialized with a mask (numpy array)')
        if (experiment_info is not None):
            self.experiment_info = experiment_info
        if (imaging_info is not None):
            self.imaging_info = imaging_info
   
        self.mask = Mask
        self.uniq_id = id(self) # Generate a unique ID everytime 
        
    def __str__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)
    
    def __repr__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)
    
    def setCategory(self,Category):
        self.category = Category
        
    def set_z_depth(self,depth):
        self.z_depth = depth
        
    def setSourceImage(self, Source_image):
        
        if np.shape(Source_image) == np.shape(self.mask):
            self.source_image = Source_image
        else:
            raise TypeError('ROI_bg: source image dimensions has to match with\
                            ROI mask.')

    def set_extraction_type(self,extraction_type):
        self.extraction_type = extraction_type
    
    def showRoiMask(self, cmap = 'Pastel2',source_image = None):
        
        if (source_image is None):
            source_image = self.source_image
        curr_mask = np.array(copy.deepcopy(self.mask),dtype=float)
        curr_mask[curr_mask==0] = np.nan
        sns.heatmap(source_image,alpha=0.8,cmap = 'gray',cbar=False)
        sns.heatmap(curr_mask, alpha=0.6,cmap = cmap,cbar=False)
        plt.axis('off')
        plt.title(self)
    def calculateDf(self,stimulus_information,method='mean',moving_avg = False, bins = 4):
        try:
            self.raw_trace
        except NameError:
            raise NameError('ROI_bg: for deltaF calculations, a raw trace \
                            needs to be provided: a.raw_trace')
        if method=='begining': #first 5 seconds of recording
            frames=np.array(stimulus_information['output_data_downsampled'].index)
            mean_val=mean_of_initial5secs(self.raw_trace,frames)
            df_trace = (self.raw_trace-mean_val)/(mean_val)
            self.baseline_method = method
        elif method=='mean':
            trace=copy.deepcopy(self.raw_trace)
            df_trace = (trace-np.mean(trace))/np.mean(trace)
            self.baseline_method = method
        elif method=='rolling_mean':

            if 'T4T5' in self.experiment_info['Genotype']:
                shift=int(np.ceil(0.45*self.imaging_info['frame_rate'])) # trace is shifted to account for a delay in visualization of response delay was calculated by Bgur
                trace=copy.deepcopy(np.roll(self.raw_trace,-shift))
            else:
                trace=copy.deepcopy(self.raw_trace)
            
            window_size=30
            period=self.imaging_info['FramePeriod']
            window=np.round(window_size/period)
            trace_b=pd.Series(np.where(trace<((np.std(trace)*2)+np.mean(trace)),trace,np.nan)) #np.floor(window/2).astype(int)

            mean_trace=trace_b.rolling(window.astype(int),min_periods=np.floor(window/3).astype(int),center=True).mean()
            if np.all(np.isnan(mean_trace)):
                raise Exception ('all trace is Nan')
            df_trace=(trace-mean_trace)/mean_trace
            self.baseline_method='rolling average data<(2*std)+mean window 20secs'
        elif method=='baseline_epoch':
            stim=np.array(stimulus_information['output_data_downsampled']['epoch'])
            frames=np.array(stimulus_information['output_data_downsampled'].index)
            if 'T4T5' in self.experiment_info['Genotype']:
                shift=int(np.ceil(0.45*self.imaging_info['frame_rate'])) # trace is shifted to account for a delay in visualization of response delay was calculated by Bgur
                trace=copy.deepcopy(np.roll(self.raw_trace,-shift))
            else:
                trace=copy.deepcopy(self.raw_trace)
            # exclude for now the initial frames (with no sitmuli)
            signal_l=trace
            signal_0=signal_l[:frames[0]-1]
            signal_0[:]=np.mean(signal_0)
            signal_1=signal_l[frames[0]-1:]
            mean_trace=compute_local_means(stim, signal_1,mean_=True)
            mean_trace=np.concatenate([signal_0,mean_trace])
            df_trace=(trace-mean_trace)/mean_trace
            self.baseline_method= 'initial_period_with_no_stim'
        elif method=='convolution':
            trace=copy.deepcopy(self.raw_trace)
            period=self.imaging_info['FramePeriod']
            window_size=60 # make a rolling average for every minute of recording 60secs
            window=np.round(window_size/period)
            kernel=np.divide(np.ones(window,dtype=float),float(window))
            traces_mean=ndimage.convolve1d(trace,kernel,axis=0,mode='reflect')
            df_trace=np.divide(trace-traces_mean,traces_mean)
        #################################################
        ###### method ==  'postpone is deprecated' not recomended. it is kept around for security checking older analysis that were done that way
        #################################################
        elif method=='postpone': #hold on the df calculation to do it based on stimulus
            df_trace=self.raw_trace
            self.baseline_method = method
            #print('df/f postponed')
            
            
        elif method == 'tau_subepoch' and self.stim_info['EPOCHS'] == 1: # use the initial part of the stimulus (named tau is stimfile; I know, bad name) which shows a pre-epoch that depends on the stimulus
            
            if self.stim_info['EPOCHS'] == 1: 
                period = self.imaging_info['FramePeriod']  
                len_subepoch = self.stim_info['tau'][0]
                subepoch_frames = int(np.floor(len_subepoch/period))
                baseline = self.raw_trace[self.stim_info['trial_coordinates'][0][0][0] - subepoch_frames: self.stim_info['trial_coordinates'][0][0][0]]
                df_trace = np.divide(self.raw_trace - np.mean(baseline), np.mean(baseline))
                self.imaging_info['baseline_mean']=np.mean(baseline)
            elif self.stim_info['Trials'] > 1:
                period = self.imaging_info['FramePeriod']
                baseline = np.zeros_like(self.raw_trace)
                len_subepoch = self.stim_info['tau'][0]
                subepoch_frames = int(np.floor(len_subepoch/period))
                baseline[0:self.stim_info['trial_coordinates'][0][0][0]-subepoch_frames] = np.mean(self.raw_trace[self.stim_info['trial_coordinates'][0][0][0]-subepoch_frames:self.stim_info['trial_coordinates'][0][0][0]])
                
                for trial in range(len(self.stim_info['trial_coordinates'][0])):
                    
                    len_subepoch = self.stim_info['tau'][0]
                    subepoch_frames = int(np.floor(len_subepoch/period))
                    baseline[self.stim_info['trial_coordinates'][0][trial][0] - subepoch_frames:] = np.mean(self.raw_trace[self.stim_info['trial_coordinates'][0][trial][0] - subepoch_frames: self.stim_info['trial_coordinates'][0][trial][0]])
                
                df_trace = np.divide(self.raw_trace - baseline, np.mean(baseline))



            

        elif method == '0_epoch_1sbefore': # often prefered for FFF stims where one epoch is the baseline of others
            period = self.imaging_info['FramePeriod']  
            trial_counts = []
            #baseline = np.zeros_like(self.raw_trace)
            df_trace = np.zeros_like(self.raw_trace)
            for key in self.stim_info['trial_coordinates']:
                trial_counts.append(len(self.stim_info['trial_coordinates'][key]))
            for i in range(trial_counts[0]):
                for key in self.stim_info['trial_coordinates']:
                    try:
                        boundaries = self.stim_info['trial_coordinates'][key][i][0]
                    except IndexError:
                        continue
                    boundaries_baseline = self.stim_info['trial_coordinates'][0][i][0] 
                    boundaries_baseline = [boundaries_baseline[1]-int(1/period),boundaries_baseline[1]] # get the last second of the baseline epoch
                    base_vals = (np.mean(self.raw_trace[boundaries_baseline[0]:boundaries_baseline[1]]))
                    vals = self.raw_trace[boundaries[0]:boundaries[1]]
                    df_trace[boundaries[0]:boundaries[1]] = (vals - base_vals) / base_vals
            #calculate df for the first 5 seconds (no stim period) 
            df_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]] = (self.raw_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]] - np.mean(self.raw_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]])) / np.mean(self.raw_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]])
            # eliminate any remaining  
            df_trace = np.where(df_trace == 0 ,np.nan, df_trace)
            df_trace = df_trace[~np.isnan(df_trace)]
        
        elif method == 'gaussian filter':
            'pending'
        
        else:
            raise Exception('Tau df calculation not yet implemented for more than 1 epoch')
        if moving_avg: # de-trending step
            self.df_trace = movingaverage(df_trace, bins)
        else:
            self.df_trace = df_trace

        


        return self.df_trace
            
    def plotDF(self, line_w = 1, adder = 0,color=plt.cm.Dark2(0)):
        
        plt.plot(self.df_trace+adder, lw=line_w, alpha=.8,color=color)
       
        try:
            self.stim_info['output_data']
            stim_frames = self.stim_info['output_data'][:,7]  # Frame information
            stim_vals = self.stim_info['output_data'][:,3] # Stimulus value
            uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
            stim_vals = stim_vals[uniq_frame_id]
            # Make normalized values of stimulus values for plotting
            
            stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
                *np.max(self.df_trace+adder)
            plt.plot(stim_vals,'--', lw=1, alpha=.6,color='k')
        except KeyError:
            print('No raw stimulus information found')
        
    def appendPrevTraces(self):

        """
        in the cases where more than one stimulation was done for the same t series, 
        the first stimulation paradigm and the corresponding traces are stored in a seperate
        attribute for further use, and to avoid identity mixing among traces from different stimulations

        """
        try:
            self.prevstim_data
        except AttributeError:
            raise Exception('to append previous stimulation traces a prevstim_data dict is needed')
        
        # if self.prevstim_data['stim_info']['stim_name'] == self.stim_info['stim_name']:
        #     raise Exception ('previous stim data and current data are the same')
        try:
            self.resp_trace_all_epochs
        except AttributeError:
            raise Exception('no previous traces')
        # if self.prevstim_data['stim_info']['epochs']==len(resp_trace_all_epochs):
        self.prevstim_data['prev_resp_traces']=copy.deepcopy(self.resp_trace_all_epochs)
        
        ### TODO fix this next lines! maybe just erase everything except 2 or 3 important ones
        del self.resp_trace_all_epochs
        del self.raw_trace
        del self.stim_name
        del self.whole_trace_all_epochs
        del self.df_trace
        # else:
        #     raise Exception('epochs dont coincide between prevstim info and traces')

    def appendPrevResponseProperties(self):
        self.RespProperties={}
        try:
            self.RespProperties['CS']=self.CS
        except:
            pass
        try:
            if self.CS=='OFF':
                self.RespProperties['CSI']=self.CSI_OFF
            elif self.CS=='ON':  
                self.RespProperties['CSI']=self.CSI_ON
        except: 
            pass
        try:
            if self.CS=='OFF':
                self.RespProperties['DSI']=self.DSI_OFF
            elif self.CS=='ON':  
                self.RespProperties['DSI']=self.DSI_ON
        except:
            pass

    def appendPrevanalysisParams(self):
        try:
            self.appendPrevanalysisParams
        except: 
            pass
    def appendTrace(self, trace, epoch_num, trace_type = 'whole'):
        
        if trace_type == 'whole':
            try:
                self.whole_trace_all_epochs
            except AttributeError:
                self.whole_trace_all_epochs = {}
                
            
            self.whole_trace_all_epochs[epoch_num] = trace
        
        elif trace_type == 'response':
            try:
                self.resp_trace_all_epochs
            except AttributeError:
                self.resp_trace_all_epochs = {}
                
            self.resp_trace_all_epochs[epoch_num] = trace
        
        elif trace_type == 'raw':
            try:
                self.whole_traces_all_epochsTrials
            except AttributeError:
                self.whole_traces_all_epochsTrials={}
            
            self.whole_traces_all_epochsTrials[epoch_num] = trace
            
    def appendprevStimInfo(self, Stim_info ,raw_stim_info = None):

        try:
            self.prevstim_data
        except AttributeError:
            self.prevstim_data={}
        finally:
            pass
        try: 
            self.stim_info
        except:
            raise Exception('no previous stimuli found')
        if self.stim_info['stim_name']==Stim_info['stim_name']:
            raise Exception ('prev stim is the same as current')
        else:
            self.prevstim_data['stim_info']=copy.deepcopy(self.stim_info)
            del self.stim_info

    def appendStimInfo(self, Stim_info ,raw_stim_info = None):
        
        self.stim_info = Stim_info
        self.stim_name = Stim_info['stim_name']
        
        if (raw_stim_info is not None):
            # This part is now stored in the stim_info already but keeping it 
            # for backward compatibility.
            self.raw_stim_info = raw_stim_info
           
    def findMaxResponse_all_epochs(self,cycle,polarity=None):
        
        try:
            self.resp_trace_all_epochs
        except AttributeError:
            raise AttributeError('ROI_bg: for finding maximum responses \
                            "resp_trace_all_epochs" has to be appended by \
                            appendTrace() method ')
            
        #obtain analysis type( depending on that, more than one maximum may be needed)
        
        analysis_type=self.experiment_info['analysis_type']

        if analysis_type[cycle]=='8D_edges_find_rois_save' or analysis_type[cycle]=='4D_edges' or analysis_type[cycle]=='1-dir_ON_OFF': 
            
            self.max_resp_all_epochs_ON = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) #Seb: epochs_number --> EPOCHS
            self.max_resp_all_epochs_ON[:] = np.nan
            
            self.max_resp_all_epochs_OFF = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) 
            self.max_resp_all_epochs_OFF[:] = np.nan

            #self.response_quality_ON=np.empty(shape=(int(self.stim_info['EPOCHS']),1))
            #self.response_quality_OFF=np.empty(shape=(int(self.stim_info['EPOCHS']),1))
            self.max_resp_all_epochs_ON=np.zeros((int(self.stim_info['EPOCHS'])))
            self.max_resp_all_epochs_ON[:]=np.nan
            self.max_resp_all_epochs_OFF=np.zeros((int(self.stim_info['EPOCHS'])))
            self.max_resp_all_epochs_OFF[:]=np.nan
            self.trace_STD_OFF=np.zeros((int(self.stim_info['EPOCHS'])))
            self.trace_STD_OFF[:]=np.nan
            self.trace_STD_ON=np.zeros((int(self.stim_info['EPOCHS'])))
            self.trace_STD_ON[:]=np.nan
            ### following measurements are not in use
            self.baseline_STD_OFF=np.zeros((int(self.stim_info['EPOCHS'])))
            self.baseline_STD_OFF[:]=np.nan
            self.baseline_STD_ON=np.zeros((int(self.stim_info['EPOCHS'])))
            self.baseline_STD_ON[:]=np.nan
            self.baseline_mean_ON=np.zeros((int(self.stim_info['EPOCHS'])))
            self.baseline_mean_ON[:]=np.nan
            self.mean_of_overall_baseline=np.zeros((int(self.stim_info['EPOCHS'])))
            self.mean_of_overall_baseline[:]=np.nan
            self.baseline_mean_OFF=np.zeros((int(self.stim_info['EPOCHS'])))
            self.baseline_mean_OFF[:]=np.nan
            self.overall_baseline={}
            #### until here
            self.shifted_trace_={}
            plt.close('all')
            for epoch_idx in self.resp_trace_all_epochs.keys():
                if self.stim_info['stimtype'][epoch_idx]!='driftingstripe' and self.stim_info['stimtype'][epoch_idx]!='ADS' and self.stim_info['stimtype'][epoch_idx]!='G':
                    continue
                stim_response_timedelay= 0.45 # this is a hardcoded delay between the stimulus 
                                         #and the peak of a t4t5 axon, Burak gur calculated this number (seconds)
                # the shift is necessary for the stimulus: 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt'
                shift=int(stim_response_timedelay*self.imaging_info['frame_rate'])+1 # calculate number of frames that include the delay time required, since this is float, approximate to nearest higher int

                if (self.stim_info['stim_name'] == 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt'):
                    # for this stimulus type there is a brigth flash and then the first edge is off. hopefully it is not like this for other stimuli
                    shifted_trace=np.roll(self.resp_trace_all_epochs[epoch_idx],-shift)
                    shifted_trace_OFF=shifted_trace[:len(shifted_trace)//2]
                    shifted_trace_ON=shifted_trace[len(shifted_trace)//2:]
                elif self.stim_info['stim_name'] == 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_ONEDGEFIRST_80sec.txt'\
                                or self.stim_info['stim_name'] == 'Mapping_for_MIexp.txt'\
                                or self.stim_info['stim_name']=='Drifting_stripesJF_8dir.txt' \
                                or 'DriftingStripe_4sec' in self.stim_info['stim_name']\
                                or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_15degoffset_ONEDGEFIRST.txt'\
                                or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                                or self.stim_info['stim_name'] =='mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                                or self.stim_info['stim_name'] == 'Dirfting_edges_JF_8dirs_ON_first.txt'\
                                or self.stim_info['stim_name'] == '15deg_50_Dirfting_edges_JF_8dirs_ON_first.txt'\
                                or self.stim_info['stim_name'] == '15deg_Dirfting_edges_JF_8dirs_ON_first.txt'\
                                or self.stim_info['stim_name'] == '30deg_50_Dirfting_edges_JF_8dirs_ON_first.txt'\
                                or self.stim_info['stim_name'] == '30deg_Dirfting_edges_JF_8dirs_ON_first.txt'\
                                or self.stim_info['stim_name'] == '1_Dirfting_edges_JF_8dirs_ON_first.txt':
                    shifted_trace=self.resp_trace_all_epochs[epoch_idx]
                    shifted_trace_OFF=shifted_trace[len(shifted_trace)//2:]
                    shifted_trace_ON=shifted_trace[:len(shifted_trace)//2]
                peak_interval=int(round(0.5*self.imaging_info['frame_rate'])) #TODO this will be used to exclude the 1s interval containing the maximum response from the mean calculation
                self.trace_STD_ON[epoch_idx]=np.std(shifted_trace_ON)
                self.trace_STD_OFF[epoch_idx]=np.std(shifted_trace_OFF)
                
                self.max_resp_all_epochs_OFF[epoch_idx] = np.nanmax(shifted_trace_OFF)
                location_off_peak=np.where(shifted_trace_OFF==self.max_resp_all_epochs_OFF[epoch_idx])[0][0]

                self.max_resp_all_epochs_ON[epoch_idx] = np.nanmax(shifted_trace_ON)

                location_on_peak=np.where(shifted_trace_ON==self.max_resp_all_epochs_ON[epoch_idx])[0][0]

                # the range_x variables produce indices that map to the response traces, excluding the location of the peak response
                # this is used to calculate a baseline mean and standard deviation
                
                if location_on_peak<=shift:
                    range_ON=np.array(range(location_on_peak+peak_interval,len(shifted_trace_ON)-(2*peak_interval)+1))
                else:
                    range_ON=np.append(np.array(range(0,location_on_peak-peak_interval)),np.array(range(location_on_peak+peak_interval,len(shifted_trace_ON))))
                    #just in case the range ON turns into a float type (this happens if the peak is to close to the end of the trace)
                    range_ON=range_ON.astype(int)

                if location_off_peak<=shift:
                    range_OFF=np.array(range(location_off_peak+peak_interval,len(shifted_trace_OFF)-(2*peak_interval)+1))
                else:
                    range_OFF=np.append(np.array(range(0,location_off_peak-peak_interval)),np.array(range(location_off_peak+peak_interval,len(shifted_trace_OFF))))
                    #just in case the range OFF turns into a float type (this happens if the peak is to close to the end of the trace)
                    range_OFF=range_OFF.astype(int)
                
                
                self.overall_baseline_loc=np.append(shifted_trace_ON[range_ON],shifted_trace_OFF[range_OFF])
                overall_baseline_loc=copy.deepcopy(self.overall_baseline_loc)


                if self.baseline_method == 'postpone': #if df/f has not been calculated, this calculates df/f using the baseline trace (trace excluding the response)
                    # self.max_resp_all_epochs_ON[epoch_idx]=(self.max_resp_all_epochs_ON[epoch_idx]-np.mean(self.baseline_mean_ON[epoch_idx]))/np.mean(self.baseline_mean_ON[epoch_idx])
                    # self.max_resp_all_epochs_OFF[epoch_idx]=(self.max_resp_all_epochs_OFF[epoch_idx]-np.mean(self.baseline_mean_OFF[epoch_idx]))/np.mean(self.baseline_mean_OFF[epoch_idx])
                    # self.baseline_STD_ON[epoch_idx]=(self.baseline_mean_ON[epoch_idx]-np.mean(self.baseline_mean_ON[epoch_idx]))/np.mean(self.baseline_mean_ON[epoch_idx])
                    # self.baseline_STD_OFF[epoch_idx]=(self.baseline_mean_OFF[epoch_idx]-np.mean(self.baseline_mean_OFF[epoch_idx]))/np.mean(self.baseline_mean_OFF[epoch_idx])
                    # self.resp_trace_all_epochs[epoch_idx]=(self.resp_trace_all_epochs[epoch_idx]-np.mean(overall_baseline))/np.mean(overall_baseline)
                    # self.whole_trace_all_epochs[epoch_idx]=(self.whole_trace_all_epochs[epoch_idx]-np.mean(overall_baseline))/np.mean(overall_baseline)
                    

                    self.max_resp_all_epochs_ON[epoch_idx]=(self.max_resp_all_epochs_ON[epoch_idx]-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.max_resp_all_epochs_OFF[epoch_idx]=(self.max_resp_all_epochs_OFF[epoch_idx]-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.baseline_mean_ON[epoch_idx]=(self.baseline_mean_ON[epoch_idx]-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.baseline_mean_OFF[epoch_idx]=(self.baseline_mean_OFF[epoch_idx]-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.resp_trace_all_epochs[epoch_idx]=(self.resp_trace_all_epochs[epoch_idx]-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.whole_trace_all_epochs[epoch_idx]=(self.whole_trace_all_epochs[epoch_idx]-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.shifted_trace_[epoch_idx]=(shifted_trace-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.overall_baseline[epoch_idx]=(self.overall_baseline_loc-np.mean(overall_baseline_loc))/np.mean(overall_baseline_loc)
                    self.mean_of_overall_baseline[epoch_idx]=np.mean(self.overall_baseline_loc)

                    self.baseline_STD_ON[epoch_idx]=np.std(shifted_trace_ON[range_ON])
                    self.baseline_mean_ON[epoch_idx]=np.mean(shifted_trace_ON[range_ON])
                    self.baseline_STD_OFF[epoch_idx]=np.std(shifted_trace_OFF[range_OFF])
                    self.baseline_mean_OFF[epoch_idx]=np.mean(shifted_trace_OFF[range_OFF])


                    if self.stim_info['stim_name'] == 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt':
                         # for this stimulus type there is a brigth flash and then the first edge is off. hopefully it is not like this for other stimuli
                        shifted_trace=np.flip(np.roll(np.flip(self.resp_trace_all_epochs[epoch_idx],0),shift),0)
                        shifted_trace_OFF=shifted_trace[:len(shifted_trace)//2]
                        shifted_trace_ON=shifted_trace[len(shifted_trace)//2:]

                    else:
                        shifted_trace=self.resp_trace_all_epochs[epoch_idx]
                        shifted_trace_OFF=shifted_trace[len(shifted_trace)//2:]
                        shifted_trace_ON=shifted_trace[:len(shifted_trace)//2]
                    self.baseline_STD_ON[epoch_idx]=np.std(shifted_trace_ON[range_ON])
                    self.baseline_STD_OFF[epoch_idx]=np.std(shifted_trace_OFF[range_OFF])
                    #self.overall_baseline= np.append(shifted_trace_ON[range_ON],shifted_trace_OFF[range_OFF])


                    self.baseline_method=='mean of trace excluding response'
                    #
                    #plt.figure()
                    #plt.plot(self.whole_trace_all_epochs[epoch_idx])
                        
            
            self.max_response_ON = np.nanmax(self.max_resp_all_epochs_ON)
            self.max_response_OFF = np.nanmax(self.max_resp_all_epochs_OFF)
            self.max_resp_idx_ON = np.nanargmax(self.max_resp_all_epochs_ON)
            self.max_resp_idx_OFF = np.nanargmax(self.max_resp_all_epochs_OFF)
            #reliability=np.zeros(len(self.reliability_all_epochs))
            # for idx,epoch in enumerate(self.reliability_all_epochs):
                # reliability[epoch]=
            
            #use the preferred directions of the ON and OFF responses to set the reliability for every ROI
            self.reliability_PD_ON= self.reliability_all_epochs_ON[self.max_resp_idx_ON]
            self.reliability_PD_OFF= self.reliability_all_epochs_OFF[self.max_resp_idx_OFF]
            self.reliability_PD = [self.reliability_PD_OFF,self.reliability_PD_ON][np.argmax([self.max_response_OFF,self.max_response_ON])]
            # for the ON and OFF prefered epochs, extract information relevant to exclude noisy ROIs. calculate evoked-non-evoked ratio divided by std of baseline signal
            
            #these measurements are not curently being used: 
        
            #self.peak_baseline_ON=np.max(self.overall_baseline[self.max_resp_idx_ON])
            # self.peak_baseline_OFF=np.max(self.overall_baseline[self.max_resp_idx_OFF])
            # self.mean_overall_baseline_ON=np.mean(self.overall_baseline[self.max_resp_idx_ON])
            # self.mean_overall_baseline_OFF=np.mean(self.overall_baseline[self.max_resp_idx_OFF])
            # self.std_overall_baseline_ON=np.std(self.overall_baseline[self.max_resp_idx_ON])
            # self.std_overall_baseline_OFF=np.std(self.overall_baseline[self.max_resp_idx_OFF])

            # these quantities are not being used. deprecated:
            # self.peak_to_peak_response_baseline_comparison_ON=(self.max_response_ON-np.max(self.overall_baseline[self.max_resp_idx_ON]))/self.std_overall_baseline_ON
            # self.peak_to_peak_response_baseline_comparison_OFF=(self.max_response_OFF-np.max(self.overall_baseline[self.max_resp_idx_OFF]))/self.std_overall_baseline_OFF
            # self.peak_to_mean_response_baseline_comparison_OFF=(self.max_response_ON-self.mean_overall_baseline_OFF)/self.std_overall_baseline_OFF
            # self.peak_to_mean_response_baseline_comparison_ON=(self.max_response_OFF-self.mean_overall_baseline_ON)/self.std_overall_baseline_ON

            
        elif (self.stim_info['stim_name'] == 'LocalCircle_5secON_5sec_OFF_120deg_10sec.txt' or\
            self.stim_info['stim_name'] == '2_LocalCircle_5secON_5sec_OFF_120deg_10sec.txt' or\
            self.stim_info['stim_name'] =='LocalCircle_5sec_120deg_0degAz_0degEl_Sequential_LumDec_LumInc_10sec.txt' or\
            self.stim_info['stim_name'] =='LocalCircle_5sec_220deg_0degAz_0degEl_Sequential_LumDec_LumInc.txt') or\
            analysis_type[cycle]== 'Flashes'    :
        
            if self.baseline_method == 'postpone':
                # calculate Df/f
                if polarity==None:
                    print(self.stim_trace_correlation)
                    if self.stim_trace_correlation>0:
                        self.polarity='ON'
                        polarity='ON'
                    elif self.stim_trace_correlation<0:
                        self.polarity='OFF'
                        polarity='OFF'
                    else:
                        polarity=None
                        self.discard_flag=1
                if polarity=='OFF':
                    baseline=np.mean(self.whole_trace_all_epochs[1][-40:])
                    self.whole_trace_all_epochs_df={}
                    self.whole_trace_all_epochs_df[0]=(self.whole_trace_all_epochs[0]-baseline)/baseline
                    self.whole_trace_all_epochs_df[1]=(self.whole_trace_all_epochs[1]-baseline)/baseline
                    self.reliability_OFF= self.reliability_all_epochs[0]
                    concatenated_trace=np.concatenate((self.whole_trace_all_epochs_df[0],self.whole_trace_all_epochs_df[1]))
                    self.df_baseline='ON'

                elif polarity=='ON':
                    self.whole_trace_all_epochs_df={}
                    baseline=np.mean(self.whole_trace_all_epochs[0][-40:])
                    self.whole_trace_all_epochs_df[0]=(self.whole_trace_all_epochs[0]-baseline)/baseline
                    self.whole_trace_all_epochs_df[1]=(self.whole_trace_all_epochs[1]-baseline)/baseline
                    self.reliability_ON= self.reliability_all_epochs[1]
                    concatenated_trace=np.concatenate((self.whole_trace_all_epochs_df[0],self.whole_trace_all_epochs_df[1]))

                    self.df_baseline='OFF'
            

                else:
                    raise  Exception ('expected polarity should be "ON" or "OFF"')
            
            self.max_resp_all_epochs = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) #Seb: epochs_number --> EPOCHS
            
            self.max_resp_all_epochs[:] = np.nan
            
            for epoch_idx in self.whole_trace_all_epochs:
                self.max_resp_all_epochs[epoch_idx] = np.nanmax(self.whole_trace_all_epochs[epoch_idx])
            
            self.max_response = np.nanmax(self.max_resp_all_epochs)
            self.max_resp_idx = np.nanargmax(self.max_resp_all_epochs)
        
        elif self.stim_info['stim_name'] == 'Gratings_sine_30sw_TF_0.2_to_4hz_3sec_GRAY_Xsec_moving_right_left_90sec.txt':
            # calculate df/f the baseline here is the gray interlude.
            aaa='aaa'      # it is already done somewhere else, in the calculate df function :)      

        elif self.stim_info['stim_name'] == 'exp_random_ONedges_20dirs_20degPerS.txt':
            #find the maximum for every trial
            # 
            #calculate df/f if postponed
            # it is already done somewhere else, in the calculate df function :)  
            if self.baseline_method == 'postpone':
                print('df/f not calculated, for random drift stripes it is not defined yet')
                ## it is already done somewhere else, in the calculate df function :)  
            self.max_response_alltrialsEpochs= {}  
            self.timing_max_response_alltrialsEpochs={}
            self.max_resp_all_epochs=np.zeros(len(self.whole_traces_all_epochsTrials.keys()))

            for ix,epoch in enumerate(self.whole_traces_all_epochsTrials.keys()):
                #slice out the intermediate epoch. 
                base_duration=self.base_dur[ix]
                alltrials=self.whole_traces_all_epochsTrials[epoch][base_duration:,:]
                self.max_response_alltrialsEpochs[epoch]=np.nanmax(alltrials,axis=0)
                self.timing_max_response_alltrialsEpochs[epoch]=np.argmax(alltrials,axis=0)*self.imaging_info['FramePeriod']
                 # find the average max responses across trials 
                self.max_resp_all_epochs[ix]= np.nanmax(self.whole_trace_all_epochs[epoch][self.base_dur[epoch-1]:])
            self.max_response = np.nanmax(self.max_resp_all_epochs)
            self.max_resp_idx = np.nanargmax(self.max_resp_all_epochs)
        else:
            self.max_resp_all_epochs = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) #Seb: epochs_number --> EPOCHS
            
            self.max_resp_all_epochs[:] = np.nan
            
            for epoch_idx in self.resp_trace_all_epochs:
                self.max_resp_all_epochs[epoch_idx] = np.nanmax(self.resp_trace_all_epochs[epoch_idx])
            
            self.max_response = np.nanmax(self.max_resp_all_epochs)
            self.max_resp_idx = np.nanargmax(self.max_resp_all_epochs)
    
    def calculate_DSI_PD(self,method='PDND'):
        '''Calcuates DSI and PD of an ROI '''
        # NOTE:code for methods PDND and vector is not updated. update before using
        # for now it only works with contrast specific responses
        try:
            self.max_resp_all_epochs
            self.max_resp_idx
            self.stim_info
            self.max_response               
        except AttributeError:
            #raise TypeError('ROI_bg: for finding DSI an ROI needs\
                                 #max_resp_all_epochs and stim_info')
            self.max_resp_all_epochs_ON
        #finally:
            #pass
        def find_opp_epoch(self, current_dir, current_freq, current_epoch_type): # deprecated
            required_epoch_array = \
                    (np.array(self.stim_info['direction']) == (current_dir*-1)) & \
                    (np.array(self.stim_info['velocity']) == current_freq) & \
                    (np.array(self.stim_info['stimtype']) == current_epoch_type)  
            
            return np.where(required_epoch_array)[0]
        
        #find ON_OFF max responses:
        epochDur= self.stim_info['duration']
        
        # self.ON_responses=[]
        # self.OFF_responses=[]
        # for epoch in self.whole_trace_all_epochs.keys():
        #     #find maxon and max_off
        #     half_dur_frames = int((round(self.imaging_info['frame_rate'] * epochDur[epoch]))/2)
        #     self.ON_responses.append(np.nanmax(self.whole_trace_all_epochs[epoch][half_dur_frames:]))
        #     self.OFF_responses.append(np.nanmax(self.whole_trace_all_epochs[epoch][:half_dur_frames]))
        # self.ON_max_resp_idx=np.nanargmax(self.ON_responses)
        # self.OFF_max_resp_idx=np.nanargmax(self.OFF_responses)


        if method == 'PDND':
            ### atention. next line is a correction of direction specifically for this stim
            raise Exception ('the code for this mehtod is not up to date')
            self.stim_info['direction']=np.array(self.stim_info['direction'])*-1

            # Finding the maximum response epoch properties
            idx_of_motion=np.where(np.array(self.stim_info['direction'])==1.0)[0]

            direction_vector=np.array(self.stim_info['bar.orientation'])
            idx_45=np.where(direction_vector==45.)[0]
            idx_135=np.where(direction_vector==135.)[0]
            direction_vector[idx_45]=135.
            direction_vector[idx_135]=45.

            direction_vector[idx_of_motion]=direction_vector[idx_of_motion]+180
            self.direction_vector=direction_vector
            current_dir=direction_vector[self.max_resp_idx]
            #current_dir = self.stim_info['direction'][self.max_resp_idx]
            current_freq = self.stim_info['velocity'][self.max_resp_idx]
            current_epoch_type = self.stim_info['stimtype'][self.max_resp_idx]
            
            if current_freq == 0:
                warn('ROI %s -- max response is not in a moving epoch...' % self.uniq_id)
                moving_epochs = np.where(self.stim_info['epoch_frequency']>0)[0]
                # Find the moving epoch with max response
                idx = np.nanargmax(self.max_resp_all_epochs[moving_epochs])
                max_epoch = moving_epochs[idx]
                max_resp = self.max_resp_all_epochs[max_epoch]
                
            else:
                
                max_epoch = self.max_resp_idx
                max_resp = self.max_response
            # Calculating the DSI
            
            opposite_dir_epoch = find_opp_epoch(self,current_dir, current_freq,
                                               current_epoch_type)
            if direction_vector[max_epoch]>=180:
                opposite_dir_epoch=np.where(direction_vector==direction_vector[max_epoch]-180)[0]
            elif direction_vector[max_epoch]<180:
                opposite_dir_epoch=np.where(direction_vector==direction_vector[max_epoch]+180)[0]
            
            DSI = (max_resp - self.max_resp_all_epochs[opposite_dir_epoch])/\
                (max_resp + self.max_resp_all_epochs[opposite_dir_epoch])
            self.DSI = DSI[0][0]
            
            self.PD = current_dir
            

            
        elif method =='vector':
            raise Exception ('the code for this mehtod is not up to date')
            ### atention. next line is a correction of direction specifically for this stim
            
            self.stim_info['direction']=np.array(self.stim_info['direction'])*-1 #before this: -1 was for downwards movement (down,leftdown,etc) and for left movement. 1 was for everything going up and for rightwards movement 
         
            direction_vector=np.array(self.stim_info['bar.orientation'])
            ### reorganize orientation vector (this is required to transform the polar axis into one that goes from 0 to 135 degrees counterclockwise (right now is clockwise))

            idx_45=np.where(direction_vector==45.)[0]
            idx_135=np.where(direction_vector==135.)[0]
            direction_vector[idx_45]=135.
            direction_vector[idx_135]=45.

            idx_of_motion=np.where(np.array(self.stim_info['direction'])==1.0)[0]
            direction_vector[idx_of_motion]=direction_vector[idx_of_motion]+180
            self.direction_vector=direction_vector #store this vector. this will replace self.stim_info['direction'] in a prev version of the code

            dirs = direction_vector
            # self.stim_info['direction'][self.stim_info['baseline_epoch']+1:] 
            ON_resps = self.max_resp_all_epochs_ON
            OFF_resps =self.max_resp_all_epochs_OFF
            #resps = self.max_resp_all_epochs

            # Functions work with radians so convert
            xs_on= np.transpose(ON_resps)*np.cos(np.radians(dirs))
            ys_on = np.transpose(ON_resps)*np.sin(np.radians(dirs))
            
            xs_off = np.transpose(OFF_resps)*np.cos(np.radians(dirs))
            ys_off = np.transpose(OFF_resps)*np.sin(np.radians(dirs))

            # xs= np.transpose(resps)*np.cos(np.radians(dirs))
            # ys = np.transpose(resps)*np.sin(np.radians(dirs))
            # x = (xs).sum()
            # y = (ys).sum()
            # DSI_vector = [x, y]
            # cosine_angle = np.dot(DSI_vector, [1,0]) / (np.linalg.norm(DSI_vector) * np.linalg.norm([1,0]))
            

            x_on = (xs_on).sum()
            y_on = (ys_on).sum()
            x_off = (xs_off).sum()
            y_off = (ys_off).sum()

            DSI_vector_on = [x_on, y_on]
            DSI_vector_off = [x_off, y_off]

            cosine_angle_on = np.dot(DSI_vector_on, [1,0]) / (np.linalg.norm(DSI_vector_on) * np.linalg.norm([1,0]))
            cosine_angle_off = np.dot(DSI_vector_off, [1,0]) / (np.linalg.norm(DSI_vector_off) * np.linalg.norm([1,0]))
            # origin = [0], [0] # origin point
            # for idx,direction in enumerate(dirs):
            #     plt.quiver(origin[0],origin[1], xs[0][idx],ys[0][idx],
            #                color=plt.cm.Dark2(idx),
            #                label=str(direction),scale=2)
            # plt.quiver(origin[0],origin[1], x, y, color='r',scale=6)
            # plt.legend()
            
            angle_on=np.degrees(np.arccos(cosine_angle_on))
            angle_off=np.degrees(np.arccos(cosine_angle_off))

            # angle = np.degrees(np.arccos(cosine_angle))
            #if y<0:
                #angle = 360 - angle
            #continue here
            if y_on<0:
                angle_on = 360 - angle_on
            
            if y_off<0:
                angle_off = 360 - angle_off            

            #self.DSI  = np.linalg.norm(DSI_vector)/np.max(resps)
            #self.PD = angle

            self.DSI_ON= np.linalg.norm(DSI_vector_on)/np.max(ON_resps)
            self.DSI_OFF= np.linalg.norm(DSI_vector_off)/np.max(OFF_resps)
            self.PD_ON = angle_on
            self.PD_OFF = angle_off

        elif method=='Mazurek':
            # this method has been described by Mazurek et al.,  doi: 2014 10.3389/fncir.2014.00092

            #extract responses to the different epochs/subepochs

            #print(direction_vector)
            ### reorganize orientation vector (this is required to transform the polar axis into one that goes from 0 to 135 degrees counterclockwise (right now is clockwise))

            # idx_45=np.where(direction_vector==45.)[0]
            # idx_135=np.where(direction_vector==135.)[0]
            # direction_vector[idx_45]=135.
            # direction_vector[idx_135]=45.
            try: # ideally an stim file should define Stimulus.angle!!
                direction_vector=self.stim_info['angle']
            except:
                direction_vector=np.sort(np.unique(self.stim_info['output_data_downsampled']['theta']))
                direction_vector=direction_vector[~np.isnan(direction_vector)]
                # print('stimulus.angle not defined')
                # self.stim_info['direction']=np.array(self.stim_info['direction']) #before this: -1 was for downwards movement (down,leftdown,etc) and for left movement. 1 was for everything going up and for rightwards movement 
                # direction_vector=np.array(self.stim_info['bar.orientation'])
                # direction_vector=direction_vector*self.stim_info['direction']
                # direction_vector=np.where(direction_vector>0.,direction_vector+180,-direction_vector)
            
            
            #idx_of_motion=np.where(np.array(self.stim_info['direction'])==1.0)[0]
            #print(np.array(self.stim_info['direction']))
            #direction_vector[idx_of_motion]=direction_vector[idx_of_motion]+180
            self.direction_vector=direction_vector #store this vector. this will replace self.stim_info['direction'] in a prev version of the code

            #print(direction_vector)
            dirs = direction_vector
            dirs = np.radians(dirs)
            # self.stim_info['direction'][self.stim_info['baseline_epoch']+1:] 
            try: # check if the stimulus has 2 polarities
                self.max_resp_all_epochs_ON
                self.max_resp_all_epochs_OFF
                pols=2
            except:
                pols=1
            if pols==2:
                resps_on  = np.zeros(len(self.max_resp_all_epochs_ON),dtype='c16')
                resps_off = np.zeros(len(self.max_resp_all_epochs_OFF),dtype='c16')
                exp_on=np.zeros(len(self.max_resp_all_epochs_ON),dtype='c16')
                exp_off=np.zeros(len(self.max_resp_all_epochs_ON),dtype='c16')
                for ix,value in enumerate(self.max_resp_all_epochs_ON):
                    exp_on[ix]=cmath.exp(complex(0,dirs[ix]))
                    resps_on[ix]=value
                for ix,value in enumerate(self.max_resp_all_epochs_OFF):
                    exp_off[ix]=cmath.exp(complex(0,dirs[ix]))
                    resps_off[ix]=value
                # calculate the vector 
                print('2 polarities analized')
                self.DSI_ON  = np.abs(np.sum(resps_on*exp_on)/ np.sum(resps_on))
                self.DSI_OFF = np.abs(np.sum(resps_off*exp_off)/ np.sum(resps_off))
                self.PD_ON   = np.angle(np.sum(resps_on*exp_on)/ np.sum(resps_on), deg=True)
                if self.PD_ON <0:
                    self.PD_ON= 360+self.PD_ON
                self.PD_OFF  = np.angle(np.sum(resps_off*exp_off)/ np.sum(resps_off),  deg=True)
                if self.PD_OFF <0:
                    self.PD_OFF= 360+self.PD_OFF
            
                if self.max_response_ON>self.max_response_OFF:
                    self.dir_max_resp=self.direction_vector[self.max_resp_idx_ON]
                elif self.max_response_ON<self.max_response_OFF:
                    self.dir_max_resp=self.direction_vector[self.max_resp_idx_OFF]

            else: 
                print('one polarity analized')
                resps=np.zeros(len(self.max_resp_all_epochs),dtype='c16')
                exp_=np.zeros(len(self.max_resp_all_epochs),dtype='c16')
                for ix,value in enumerate(self.max_resp_all_epochs):
                    exp_[ix]=cmath.exp(complex(0,dirs[ix]))
                    resps[ix]=value
                self.DSI  = np.abs(np.sum(resps*exp_)/ np.sum(resps))
                self.PD   = np.angle(np.sum(resps*exp_)/ np.sum(resps), deg=True)
                if self.PD <0:
                    self.PD= 360+self.PD
    def append_SNR_Reliability(self,name_snr,name_rel,SNR_val,corr_val):     #Juan made function
        """
        takes in a  Signal to noise ratio value and a reliability value 
        and appends it to An ROI instance

        names are user input, in case that more than one SNR value wants 
        to be added to the same ROI instance, for example when multiple recording 
        cycles or Tseries use the same ROI
        """
        if self.SNR is not None:
            self.SNR.update({name_snr:SNR_val})
            self.SNR.reliability.update({name_rel:corr_val})
        else:
            self.SNR={name_snr:SNR_val}
            self.reliability={name_rel:corr_val}

    def trial_average_noise(self):

        # check length of trials
        lengths = []
        traces = []
        for trial in self.stim_info['trial_coordinates'][0]:
            lengths.append(trial[1]-trial[0])
            traces.append(self.df_trace[trial[0],trial[1]])
        minlen = np.min (np.array(lengths))

        trials_arr = np.zeros(minlen,len(self.stim_info['trial_coordinates'][0]))
        for ix,trace in enumerate(traces):
            trials_arr[ix,:] = trace[:minlen]
        self.whole_traces_all_epochsTrials = {0:trials_arr}

    def calculate_reliability(self, epoch_to_exclude = None):
        """ calculates the average pariwise correlation between trials to 
        estimate the reliability of responses.
        
        
        Parameters
        ==========
        self: an ROI_bg instance which must include:

            self.whole_trace_all_epochsTrials: dict including all trials and epochs as defined by 
                                               the stimulus epochs.

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
        # extract the type of stimulus. for drifting edges we need a reliability for ON and for OFF

        try:
            self.whole_traces_all_epochsTrials
        except AttributeError:
            raise exception ('no raw trial traces in ROI instance, traces needed to compute reliability')

        self.reliability_all_epochs={}
        self.reliability_all_epochs_ON={}
        self.reliability_all_epochs_OFF={}
        # if epoch_to_exclude!=None:
        #     copy_traces=copy.deepcopy(self.whole_trace_all_epochsTrials)
        #     del copy_traces[epoch_to_exclude]
        # else:
        copy_traces=copy.deepcopy(self.whole_traces_all_epochsTrials)
            
        for iEpoch, iEpoch_index in enumerate(copy_traces):
            
            trial_numbers = np.shape(copy_traces[iEpoch_index])[1]
            currentRespTrace =  copy_traces[iEpoch_index][:,:]            
            # if (np.isnan(currentRespTrace[1]))[-1]:
            #     currentRespTrace = []
            
            #     for validx, values in enumerate(copy_traces[iEpoch_index]):
            #         #appends = np.array(filter(lambda v: v==v, values))
            #         appends = values[~np.isnan(values)]
            #         currentRespTrace.append(appends)
            #     currentRespTrace = np.asarray(currentRespTrace, dtype=np.float32)
            #     trial_numbers = np.shape(currentRespTrace)[1]
            # Reliability between all possible combinations of trials

            if trial_numbers<2:
                perm = [trial_numbers]
            else:    
                perm = permutations(range(trial_numbers), 2) 
            coeff =[]

            if self.stim_info['stim_name']=='DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt':
                # note: for this stimulus there is a flash and then the off edge goes first in the first trial 
                coeff_all_trace=[]
                coeff_on=[]
                coeff_off=[]
                for _, pair in enumerate(perm):
                    stim_response_timedelay= 0.45 # this is a hardcoded delay between the stimulus 
                                         #and the peak of a t4t5 axon, Burak gur calculated this number (seconds)
                    shift=int(stim_response_timedelay*self.imaging_info['frame_rate'])+1 # calculate number of frames that include the delay time required, since this is float, approximate to nearest higher int

                    shifted_trace1=np.roll(currentRespTrace[:,pair[0]],-shift,axis=0)
                    shifted_trace2=np.roll(currentRespTrace[:,pair[1]],-shift,axis=0)
                    
                    # checknant1 = np.argwhere(np.isnan(zip(shifted_trace1, shifted_trace2)))
                    checknan = ~(np.isnan(shifted_trace1) | np.isnan(shifted_trace2))

                    if False in checknan:
                        shifted_trace1 = shifted_trace1[checknan]
                        shifted_trace2 = shifted_trace2[checknan]

                    if len(shifted_trace1)!=len(shifted_trace2):
                        if np.abs(len(shifted_trace1)-len(shifted_trace2))>=2:
                            continue
                        elif len(shifted_trace1)>len(shifted_trace2):
                            shifted_trace1=shifted_trace1[:len(shifted_trace2)]
                        elif len(shifted_trace2)>len(shifted_trace1):
                            shifted_trace2=shifted_trace2[:len(shifted_trace1)]
                    if 'contrasts' in self.stim_info['stim_name'] or 'luminances' in self.stim_info['stim_name'] or 'speeds' in self.stim_info['stim_name']:
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                    else:
                        curr_coeff_OFF,_ = pearsonr(shifted_trace1[:len(shifted_trace1)//2],
                                                    shifted_trace2[:len(shifted_trace1)//2])
                        curr_coeff_ON,_ = pearsonr(shifted_trace1[len(shifted_trace1)//2:],
                                                    shifted_trace2[len(shifted_trace1)//2:])
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                        
                        coeff_on.append(curr_coeff_ON)
                        coeff_off.append(curr_coeff_OFF)
                    coeff_all_trace.append(curr_coeff_general)

                self.reliability_all_epochs_ON[iEpoch_index] = np.array(coeff_on).mean()
                self.reliability_all_epochs_OFF[iEpoch_index] = np.array(coeff_off).mean()
                self.reliability_all_epochs[iEpoch_index] = np.array(curr_coeff_general).mean()
        
            elif self.stim_info['stim_name']=='DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_ONEDGEFIRST_80sec.txt'\
                        or self.stim_info['stim_name']=='Mapping_for_MIexp.txt' \
                        or self.stim_info['stim_name']=='Drifting_stripesJF_8dir.txt' \
                        or 'DriftingStripe_4sec' in self.stim_info['stim_name']\
                        or self.stim_info['stim_name']=='DriftingDriftStripe_Sequential_LumDec_8D_15degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name']== 'mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name']== 'Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '15deg_50_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '15deg_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '30deg_50_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '30deg_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name']== '1_Dirfting_edges_JF_8dirs_ON_first.txt':
                # in this stimulus there is a 5 sec dark period followed by a bright edge
                coeff_all_trace=[]
                coeff_on=[]
                coeff_off=[]
                for _, pair in enumerate(perm):
                    if self.stim_info['stim_name']=='DriftingDriftStripe_Sequential_LumDec_8D_15degoffset_ONEDGEFIRST'\
                        or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name']=='Drifting_stripesJF_8dir.txt' \
                        or self.stim_info['stim_name']=='1_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '15deg_50_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '15deg_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '30deg_50_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] == '30deg_Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name']=='Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name'] =='mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt':
                        stim_response_timedelay=0
                        shift=0
                    else:
                        stim_response_timedelay= 0.45 # this is a hardcoded delay between the stimulus 
                                         #and the peak of a t4t5 axon, Burak gur calculated this number (seconds) this is to shift the traces when there is no waiting time (tau) betweeen epochs
                        shift=int(stim_response_timedelay*self.imaging_info['frame_rate'])+1 # calculate number of frames that include the delay time required, since this is float, approximate to nearest higher int

                    if pair == 1:
                        shifted_trace1=np.roll(currentRespTrace[:],-shift,axis=0)
                        shifted_trace2=np.roll(currentRespTrace[:],-shift,axis=0)
                        print('Watch Out: There is only one trial. Calculated realiability will be 1 or gibberish. Check the frames you capture when recording the data. Calculate carefully the time required to run the stimulus.')
                    else:    
                        shifted_trace1=np.roll(currentRespTrace[:,pair[0]],-shift,axis=0)
                        shifted_trace2=np.roll(currentRespTrace[:,pair[1]],-shift,axis=0)
                    checknan = ~(np.isnan(shifted_trace1) | np.isnan(shifted_trace2))

                    if False in checknan:
                        shifted_trace1 = shifted_trace1[checknan]
                        shifted_trace2 = shifted_trace2[checknan]
                        if len(shifted_trace1) == 0:
                            continue
                        elif len (shifted_trace2) == 0:
                            continue
                    if len(shifted_trace1)!=len(shifted_trace2):
                        if np.abs(len(shifted_trace1)-len(shifted_trace2))>=2:
                            continue
                        elif len(shifted_trace1)>len(shifted_trace2):
                            shifted_trace1=shifted_trace1[:len(shifted_trace2)]
                        elif len(shifted_trace2)>len(shifted_trace1):
                            shifted_trace2=shifted_trace2[:len(shifted_trace1)]
                    if 'contrasts' in self.stim_info['stim_name'] or 'luminances' in self.stim_info['stim_name'] or 'speeds' in self.stim_info['stim_name']:
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                    else:
                        shifted_trace1 = np.squeeze(shifted_trace1)
                        shifted_trace2 = np.squeeze(shifted_trace2)
                        curr_coeff_ON,_ = pearsonr(shifted_trace1[:len(shifted_trace1)//2],
                                                    shifted_trace2[:len(shifted_trace1)//2])
                        curr_coeff_OFF,_ = pearsonr(shifted_trace1[len(shifted_trace1)//2:],
                                                    shifted_trace2[len(shifted_trace1)//2:])
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                        coeff_on.append(curr_coeff_ON)
                        coeff_off.append(curr_coeff_OFF)
                    coeff_all_trace.append(curr_coeff_general)

                self.reliability_all_epochs_ON[iEpoch_index] = np.array(coeff_on).mean()
                self.reliability_all_epochs_OFF[iEpoch_index] = np.array(coeff_off).mean()
                self.reliability_all_epochs[iEpoch_index] = np.array(curr_coeff_general).mean()
            elif "Chirp" in self.stim_info['stim_name']:
                continue
            else:
                # calculate reliability independent for every epoch
                for _, pair in enumerate(perm):
                    currentRespTrace = np.nan_to_num(currentRespTrace, nan=0)
                    try:
                        curr_coeff,_ = pearsonr(currentRespTrace[:-2,pair[0]],
                                                    currentRespTrace[:-2,pair[1]])
                    except:
                        curr_coeff = 0
                    coeff.append(curr_coeff)
                    
                self.reliability_all_epochs[iEpoch_index] = np.array(coeff).mean()
        aaa='aaa' 

    def calculate_stim_signal_correlation(self):
        '''uses self.raw_trace and raw stim trace to calculate correlation
            this is an indication of the polarity of a neuron for fff stimuli in particular'''
        if self.experiment_info['analysis_type'][0]=='5sFFF_analyze_save':
            start_id=np.array(self.stim_info['output_data_downsampled']['data']).astype(int)[0]-1
            self.stim_trace_correlation=pearsonr(np.array(self.stim_info['output_data_downsampled']['epoch']),self.raw_trace[start_id:])[0]

    def calculate_CSI(self, frameRate = None):
        
        try:
            self.max_response_ON
            if self.experiment_info['analysis_type'][0]=='8D_edges_find_rois_save'\
                or self.experiment_info['analysis_type'][0]=='4D_edges' \
                or self.experiment_info['analysis_type'][0]=='1-dir_ON_OFF':    
                moving_edges=True

        except AttributeError:
            moving_edges=False 
        finally:
            pass
        
        try:
            self.resp_trace_all_epochs
            self.stim_info
            
        except AttributeError:

            raise TypeError('ROI_bg: for finding CSI an ROI needs\
                                resp_trace_all_epochs and stim_info')
        
        if moving_edges==True:
            if self.max_response_OFF< self.max_response_ON:
                off_pd_response=self.max_resp_all_epochs_OFF[self.max_resp_idx_ON]
                self.CSI = np.abs((self.max_response_ON-off_pd_response)/(self.max_response_ON))
                self.CS = 'ON'
            else:
                on_pd_response=self.max_resp_all_epochs_ON[self.max_resp_idx_OFF]
                self.CSI = np.abs((self.max_response_OFF-on_pd_response)/(self.max_response_OFF))
                self.CS = 'OFF'
        else:
            # Find edge epochs
            edge_epochs = np.where(np.array(self.stim_info['stimtype'])=='driftingstripe')[0]
            if len(edge_epochs)==0:
                edge_epochs = np.where(np.array(self.stim_info['stimtype'])=='ADS')[0]

            epochDur= self.stim_info['duration']
            
            self.edge_response = np.max(self.max_resp_all_epochs[edge_epochs])
            # Find the edge epoch with max response
            idx = np.nanargmax(self.max_resp_all_epochs[edge_epochs])
            max_edge_epoch = edge_epochs[idx]
            
            
            raw_trace = self.resp_trace_all_epochs[max_edge_epoch]
            trace = raw_trace
            # Filtering to decrease noise in max detection
    #        b, a = signal.butter(3, 0.3, 'low')
    #        trace = signal.filtfilt(b, a, raw_trace)
            
            half_dur_frames = int((round(self.imaging_info['frame_rate'] * epochDur[max_edge_epoch]))/2)
            
            OFF_resp = np.nanmax(trace[:half_dur_frames])
            ON_resp = np.nanmax(trace[half_dur_frames:])


            CSI = (ON_resp-OFF_resp)/(ON_resp+OFF_resp)
            
            self.CSI = np.abs(copy.deepcopy(CSI))
            if CSI >0:
                self.CS = 'ON'
            else:
                self.CS = 'OFF'
                
            
        
    def calculateTFtuning_BF(self):
        self.stim_info['velocity']=np.array(self.stim_info['velocity'])
        self.stim_info['sWavelength']=np.array(self.stim_info['sWavelength'])
        self.stim_info['epoch_frequency']=  np.divide(self.stim_info['velocity'],self.stim_info['sWavelength'])
        self.stim_info['stimtype'] = np.array(self.stim_info['stimtype'])
        self.stim_info['angle']=np.array(self.stim_info['angle'])
        grating_epochs = np.where((self.stim_info['stimtype'] == 'TFgrating') &\
                                   (self.stim_info['epoch_frequency'] > 0))[0]
        
         
        # try:
        #     self.stim_info['orientation']=np.array(self.stim_info['orientation'])
        #     directionTimesOrientation= self.stim_info['orientation']*self.stim_info['direction']
        # except KeyError:
        #     print('no orientation found, skipping')
        # finally:
        #     pass
        # # If there are no grating epochs
        if grating_epochs.size==0:
            raise ValueError('ROI_bg: No grating epoch' )
            
        max_grating_epoch=np.nanargmax(self.max_resp_all_epochs[grating_epochs])      
        max_grating_epoch=grating_epochs[max_grating_epoch]            
        
        current_dir = self.stim_info['direction'][max_grating_epoch]
        current_epoch_type = self.stim_info['stimtype'][max_grating_epoch]
        
        # Finding all same direction moving grating epochs
        required_epoch_array = \
                (self.stim_info['direction'] == current_dir) & \
                (self.stim_info['stimtype'] == current_epoch_type)& \
                (self.stim_info['epoch_frequency'] > 0) 
        opposite_epoch_array = \
                (self.stim_info['direction'] == ((current_dir+180) % 360)) & \
                (self.stim_info['stimtype'] == current_epoch_type)& \
                (self.stim_info['epoch_frequency'] > 0) 
                
        self.TF_curve_stim = self.stim_info['epoch_frequency'][required_epoch_array]
        
        self.ND_TF_curve_stim = self.stim_info['epoch_frequency'][opposite_epoch_array]
        # Get it as integer indices
        req_epochs_PD = np.where(required_epoch_array)[0]
        self.TF_curve_resp = self.max_resp_all_epochs[req_epochs_PD]
        
        req_epochs_ND = np.where(opposite_epoch_array)[0]
        self.ND_TF_curve_resp = self.max_resp_all_epochs[req_epochs_ND]
        
        self.BF = self.stim_info['epoch_frequency'][max_grating_epoch]
    
    def define_independent_variable_gratings(self,cycle=0):

        """ defines how to calculate or what to extract as indepnedent variable for specific analyses later on"""

        self.independent_var = self.analysis_params['independent_var']
        
        if self.analysis_params['independent_var'] == 'frequency':

            self.independent_var_vals = np.where(np.array(self.stim_info['stimtype'])=='G',
                        np.divide(self.stim_info['velocity'],self.stim_info['sWavelength']),
                        np.nan)                  
        elif self.analysis_params['independent_var'] == 'angle':
            self.independent_var = 'orientation'
            self.independent_var_vals = np.where(np.array(self.stim_info['velocity'])>0,
                        self.stim_info[self.independent_var],np.nan)  


    def calculate_freq_powers(self,cycle,per_epoch=True):
        ''' for grating stimuli. Use fourier transform. especifically PSD to find the power of signal at 
            especific frequencies, for different epochs'''

        try:
            self.rejected
        except: 
            self.rejected = False

        if self.rejected==False:

            self.define_independent_variable_gratings(cycle)

            signal_trace=self.df_trace[self.stim_info['output_data_downsampled'].index[0]:] # discard the initial 5 seconds of recording
            sampling_frequency=self.imaging_info['frame_rate']
            stim=self.stim_info['output_data_downsampled']['epoch']
            durations=self.stim_info['duration']
            window=np.sum(durations)*sampling_frequency # the window of analysis for frequencies is the duration of a trial
            #TODO compute the psd per epoch!
            if per_epoch==True:
                self.signal_psd_dict={}
                self.stim_psd_dict={}
                freq_power_epochs_=[]
                freq_power_stims_=[]
                self.stim_traces={}
                self.power_measured_at=np.zeros_like(durations)
                self.power_measured_at[:]=np.nan
                self.deltaf_measurement=np.zeros_like(durations)
                self.deltaf_measurement[:] = np.nan
                for epoch in self.resp_trace_all_epochs.keys():
                        independt_var=self.independent_var_vals[epoch]
                        signal_trace=self.resp_trace_all_epochs[epoch]
                        dur=durations[epoch]
                        window=len(signal_trace)
                        stim=np.repeat(0,len(signal_trace))
                        self.stim_traces[epoch]=generate_sinusoid_fromstim(stim,np.array([self.independent_var_vals[epoch]]),sampling_frequency,multiple=False)
                        self.stim_traces[epoch]=(self.stim_traces[epoch]-np.mean(self.stim_traces[epoch]))/np.mean(self.stim_traces[epoch])
                        self.stim_psd_dict[epoch],meas_power_stim,_,_=compute_and_plot_psd(self.stim_traces[epoch], sampling_frequency,  np.array([independt_var]), np.array([dur]),window)
                        #_,_,_=compute_and_plot_psd((self.stim_traces[epoch]-np.mean(self.stim_traces[epoch]))/np.mean(self.stim_traces[epoch]), sampling_frequency,  np.array([independt_var]), np.array([dur]),window)
                        freq_power_stims_.append(meas_power_stim[0])
                        self.signal_psd_dict[epoch],meas_power,self.power_measured_at[epoch],self.deltaf_measurement[epoch]=compute_and_plot_psd(signal_trace, sampling_frequency, np.array([independt_var]), np.array([dur]),window)
                        freq_power_epochs_.append(meas_power[0])
                        #self.independent_var_vals[epoch]=xmeas_power[0]

                self.freq_power_epochs=np.zeros_like(durations)
                self.freq_power_epochs[:]=np.nan
                self.freq_power_epochs[1:]=freq_power_epochs_
                self.freq_power_stims=np.zeros_like(durations)
                self.freq_power_stims[:]=np.nan
                self.freq_power_stims[1:]=freq_power_stims_
            else:
                stim_trace=generate_sinusoid_fromstim(stim,self.independent_var_vals,sampling_frequency,multiple=True)
                #stim_trace=(stim_trace-np.mean(stim_trace))/np.mean(stim_trace)
                self.stim_psd_dict,self.freq_power_stims,_,_=compute_and_plot_psd(stim_trace, sampling_frequency, self.independent_var_vals, durations,window)
                self.signal_psd_dict,self.freq_power_epochs,self.power_measured_at,self.deltaf_measurement=compute_and_plot_psd(signal_trace, sampling_frequency, self.independent_var_vals, durations,window)
      
    def clear_prev_atributes(self,cycle,attributes_to_retain):
        # Create a set for efficient membership testing
        
        attrs_to_retain_set = set(attributes_to_retain)
        all_attrs = vars(self)
        # Remove attributes not in attrs_to_retain
        try:
            self.cycle1_reliability
            rel_cycle1_exists = True
        except:
            self.cycle1_reliability={}
            rel_cycle1_exists = False
        try:
            self.cycle1_traceinfo
            traceinf_cycle1_exists = True
        except:
            self.cycle1_traceinfo = {}
            traceinf_cycle1_exists = False
        for attr in list(all_attrs):  # We need to make a copy of keys to avoid RuntimeError
            if cycle > 0:
                
                if ('reliability' in attr) and rel_cycle1_exists == False: # reliability is a special variable that should be preserved for the first cycle
                    self.cycle1_reliability.update({attr:getattr(self, attr)})
                # elif (cycle>1 and 'reliability' in attr) and rel_cycle1_exists == False:
                #     self.cycle1_reliability = getattr(self, attr)
                elif (('stim' in attr) or ('df_trace' in attr)) and traceinf_cycle1_exists == False:
                    self.cycle1_traceinfo.update({attr:getattr(self, attr)})
            if attr not in attrs_to_retain_set:
                delattr(self, attr)


def find_opp_epoch_roi(roi, current_dir, current_freq, current_epoch_type):
            required_epoch_array = \
                    (roi.stim_info['epoch_dir'] == ((current_dir+180) % 360)) & \
                    (roi.stim_info['epoch_frequency'] == current_freq) & \
                    (roi.stim_info['stimtype'] == current_epoch_type)  
            
            return np.where(required_epoch_array)[0]
def low_pass(trace, frame_rate, crit_freq=3,plot=False):
    """ Applies a 3rd order butterworth low pass filter for getting rid of noise
    """
    wn_norm = crit_freq / (frame_rate/2) # critical frequency normalized to the nyquist frequency
    b, a = signal.butter(3, wn_norm, 'lowpass')
    filt_trace = signal.filtfilt(b, a, trace)
    
    if plot:
        fig1, ax1 = plt.subplots(2, 1, sharex=True, sharey=False)
        ax1[0].plot(trace,lw=0.4)
        ax1[1].plot(filt_trace,lw=0.4)
    
    return filt_trace

def High_pass(trace,frame_rate,crit_freq=0.3,plot=False):
    """ Applies a 3rd order butterworth low pass filter for getting rid of noise
    """

    wn_norm = crit_freq / (frame_rate/2) # critical frequency normalized to the nyquist frequency
    b, a = signal.butter(3, wn_norm, 'highpass')
    filt_trace = signal.filtfilt(b, a, trace)

    
    
    if plot:
        fig1, ax1 = plt.subplots(2, 1, sharex=True, sharey=False)
        ax1[0].plot(trace,lw=0.4)
        ax1[1].plot(filt_trace,lw=0.4)
    
    return filt_trace

def low_pass(trace,frame_rate,crit_freq=5,plot=False):
    """ Applies a 3rd order butterworth low pass filter for getting rid of noise
    """

    wn_norm = crit_freq / (frame_rate/2) # critical frequency normalized to the nyquist frequency
    b, a = signal.butter(3, wn_norm, 'lowpass')
    filt_trace = signal.filtfilt(b, a, trace)

    if plot:
        fig1, ax1 = plt.subplots(2, 1, sharex=True, sharey=False)
        ax1[0].plot(trace,lw=0.4)
        ax1[1].plot(filt_trace,lw=0.4)
    
    return filt_trace

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
            
def fit_poly(trace, x = None, order=3):
    """ Fits a polynomial and return the fitted trace """
    if x is None:
        x = range(len(trace))
        
    z = np.polyfit(x, trace, order)
    p = np.poly1d(z)
    
    return p(x)
    


def calcualte_mask_1d_size(rois):
    """ Finds the coordinates of the masks and determins x and y size in pixels
    """
    
    for roi in rois:
        x = list(map(lambda coords : coords[0], np.argwhere(roi.mask)))
        y = list(map(lambda coords : coords[1], np.argwhere(roi.mask)))
        roi.x_size = np.max(x) - np.min(x) + 1
        roi.y_size = np.max(y) - np.min(y) + 1

def find_opp_epoch(roi, current_dir, current_freq, current_epoch_type):
            required_epoch_array = \
                    (roi.stim_info['epoch_dir'] == ((current_dir+180) % 360)) & \
                    (roi.stim_info['epoch_frequency'] == current_freq) & \
                    (roi.stim_info['stimtype'] == current_epoch_type)  
            
            return np.where(required_epoch_array)[0]
def ROI_find(rois, roi_prop, value):
    """ Finds in a list of rois, rois that has a certain value of a desired
        property and returns as another list.
    """
    
    
    return [roi for roi in rois if roi.__dict__[roi_prop] == value]

def generate_ROI_instances(roi_masks, category_masks, category_names, source_im,
                           experiment_info = None, imaging_info =None,mapping_mode=False):
    """ Generates ROI_bg instances and adds the category information.

    Parameters
    ==========
    roi_masks : list
        A list of ROI masks in the form of numpy arrays.
        
    category_masks: list
        A list of category masks in the form of numpy arrays.
        
    category_names: list
        A list of category names.
        
    source_im : numpy array
        An array containing a representation of the source image where the 
        ROIs are found.
    
    Returns
    =======
    
    rois : list 
        A list containing instances of ROI_bg
    """

    roi_masks = list(map(lambda roi : np.array(roi)[:,:], roi_masks))
        
    # Generate instances of ROI_bg from the masks
    rois = list(map(lambda mask : ROI_bg(mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info), roi_masks))

    def assign_region(roi, category_masks, category_names):
        """ Finds which layer the current mask is in"""
        for iLayer, category_mask in enumerate(category_masks):
            if np.sum(roi.mask*category_mask):
            # if np.sum((roi.mask.astype(bool))*(category_mask.astype(bool))):
                # if 'layer' in category_names[iLayer]:
                roi.setCategory(category_names[iLayer])
                # else:
                #     continue

        try:
            roi.category
        except AttributeError:
            roi.category=['No_category']
    # Add information            
    for roi in rois:
        #if mapping_mode==False:
        assign_region(roi, category_masks, category_names)
        roi.setSourceImage(source_im)
    #TODO maybe filter out uncategorized rois  
    return rois

def generate_ROI_instances_categorized(roi_masks, category_names, source_im,
                           experiment_info = None, imaging_info =None):
    """ Generates ROI_bg instances and adds the category information.

    Parameters
    ==========
    roi_masks : list
        A list of ROI masks in the form of numpy arrays.
        
          
    category_names: list
        A list of category names.
        
    source_im : numpy array
        An array containing a representation of the source image where the 
        ROIs are found.
    
    Returns
    =======
    
    rois : list 
        A list containing instances of ROI_bg
    """

    #if type(roi_masks) == sima.ROI.ROIList:
    roi_masks = list(map(lambda roi : roi.mask, roi_masks))
        
    # Generate instances of ROI_bg from the masks
    rois = list(map(lambda mask : ROI_bg(mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info), roi_masks))

    def assign_region(roi, category_name):
        """ asigns category"""
        roi.setCategory(category_name)
    
    # Add information            
    for index,roi in enumerate(rois):
        assign_region(roi, category_names[index])
        roi.setSourceImage(source_im)
        
    return rois


def get_masks_image(rois):
    """ Generates an image of masks.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    
    Returnsf
    =======
    
    roi_data_dict : np array
        A numpy array with masks depicted in different integers
    """   
    
    roi_masks_image = np.zeros((rois[0].mask.shape))

    for ix,roi in enumerate(rois):

        roi_masks_image +=  ix*roi.mask

    # roi_masks_image = np.array(map(lambda idx_roi_pair : \
    #                          idx_roi_pair[1].mask.astype(float) * (idx_roi_pair[0]+1), 
    #                          list(enumerate(rois)))).sum(axis=0)
    # roi_masks_image = list(roi_masks_image)
    
    roi_masks_image[roi_masks_image==0] = np.nan
    
    
    return roi_masks_image

def generate_colorMasks_properties(rois, prop = 'BF',cycle=0):
    """ Generates images of masks depending on DSI CSI Rel and BF

    TODO: Is it possible to generate something independent?
    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    
    Returns
    =======
    
    roi_data_dict : np array
        A numpy array with masks depicted in different integers
    """  
    if prop == 'BF':
        BF_image = np.zeros(np.shape(rois[0].mask))
        
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            BF_image = BF_image + (curr_mask * roi.BF)
        BF_image[BF_image==0] = np.nan
        
        return BF_image
    elif prop == 'CS':
        CSI_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_CS = roi.CS
            if curr_CS == 'OFF':
                curr_CSI = roi.CSI * -1
            else:
                curr_CSI = roi.CSI
            curr_mask = roi.mask.astype(int)
            CSI_image = CSI_image + (curr_mask * curr_CSI)
        CSI_image[CSI_image==0] = np.nan
        return CSI_image
    elif prop =='DSI':
        DSI_image  = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_DSI = roi.DSI               
            curr_mask = roi.mask.astype(int)
            DSI_image = DSI_image + (curr_mask * curr_DSI)
        DSI_image[DSI_image==0] = np.nan
        return DSI_image
    elif prop =='PD':
        PD_image  = np.full(np.shape(rois[0].mask),np.nan)
        alpha_image  = np.full(np.shape(rois[0].mask),np.nan)
        for roi in rois:
            PD_image[roi.mask.astype(bool)] = roi.PD
            alpha_image[roi.mask.astype(bool)] = roi.PD
        
        return PD_image
    
    elif prop == 'reliability':
        Corr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            Corr_image = Corr_image + (curr_mask * roi.reliability[cycle])
            
        Corr_image[Corr_image==0] = np.nan
        return Corr_image
    
    elif prop == 'SNR':
        snr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            snr_image = snr_image + (curr_mask * roi.SNR[cycle])
            
        snr_image[snr_image==0] = np.nan
        return snr_image
    elif prop == 'corr_fff':
        Corr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            Corr_image = Corr_image + (curr_mask * roi.corr_fff)
            
        Corr_image[Corr_image==0] = np.nan
        return Corr_image
    elif prop == 'max_response':
        max_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            max_image = max_image + (curr_mask * roi.max_response)
            
        max_image[max_image==0] = np.nan
        return max_image
    elif prop == 'slope':
        
        slope_im = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            slope_im = slope_im + (curr_mask * roi.slope)
            
        slope_im[slope_im==0] = np.nan
        return slope_im

    elif prop =='DSI_ON':
        DSI_image  = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            if roi.CS=='ON':
                curr_DSI = roi.DSI_ON                
                curr_mask = roi.mask.astype(int)
                DSI_image = DSI_image + (curr_mask * curr_DSI)
        DSI_image[DSI_image==0] = np.nan
        return DSI_image

    elif prop =='DSI_OFF':
        DSI_image  = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            if roi.CS=='OFF':
                curr_DSI = roi.DSI_OFF                
                curr_mask = roi.mask.astype(int)
                DSI_image = DSI_image + (curr_mask * curr_DSI)
        DSI_image[DSI_image==0] = np.nan
        return DSI_image

    elif prop =='PD_ON':
        PD_image  = np.full(np.shape(rois[0].mask),np.nan)
        alpha_image  = np.full(np.shape(rois[0].mask),np.nan)
        for roi in rois:
            if roi.CS=='ON':
                try:
                    PD_image[roi.mask.astype(bool)] = roi.PD_ON
                    alpha_image[roi.mask.astype(bool)] = roi.DSI_ON
                except:
                    PD_image[roi.mask.astype(bool)] = roi.PD
                    alpha_image[roi.mask.astype(bool)] = roi.DSI    
        return PD_image

    elif prop =='PD_OFF':
        PD_image  = np.full(np.shape(rois[0].mask),np.nan)
        alpha_image  = np.full(np.shape(rois[0].mask),np.nan)
        for roi in rois:
            if roi.CS=='OFF':
                PD_image[roi.mask.astype(bool)] = roi.PD_OFF
                alpha_image[roi.mask.astype(bool)] = roi.DSI_OFF

        return PD_image
    
    elif prop == 'CSI_ON':
        CSI_image = np.full(np.shape(rois[0].mask),np.nan) 
        alpha_image  = np.full(np.shape(rois[0].mask),np.nan)
        for roi in rois:
            if roi.CS=='ON':
                CSI_image[roi.mask.astype(bool)] = roi.CSI
                alpha_image[roi.mask.astype(bool)] = roi.CSI
        return CSI_image

    elif prop == 'CSI_OFF':
        CSI_image = np.full(np.shape(rois[0].mask),np.nan) 
        alpha_image  = np.full(np.shape(rois[0].mask),np.nan)
        for roi in rois:
            if roi.CS=='OFF':
                CSI_image[roi.mask.astype(bool)] = roi.CSI
                alpha_image[roi.mask.astype(bool)] = roi.CSI
        return CSI_image

    else:
        raise TypeError('Property %s not available for color mask generation' % prop)
        return 0
    

def data_to_list(rois, data_name_list):
    """ Generates a dictionary with desired variables from ROIs.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    data_name_list: list
        A list of strings with desired variable names. The variables should be 
        written as defined in the ROI_bg class. 
        
    Returns
    =======
    
    roi_data_dict : dictionary 
        A dictionary with keys as desired data variable names and values as
        list of data.
    """   
    class my_dictionary(dict):  
  
        # __init__ function  
        def __init__(self):  
            self = dict()  
              
        # Function to add key:value  
        def add(self, key, value):  
            self[key] = value  
    
    roi_data_dict = my_dictionary()
    
    # Generate an empty dictionary
    for key in data_name_list:
        roi_data_dict.add(key, [])
    
    # Loop through ROIs and get the desired data            
    for iROI, roi in enumerate(rois):
        for key, value in roi_data_dict.items():
            if key in roi.__dict__.keys():
                value.append(roi.__dict__[key])
            else:
                value.append(np.nan)
    return roi_data_dict

def threshold_ROIs(rois, threshold_dict):
    """ Thresholds given ROIs and returns the ones passing the threshold.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    threshold_dict: dict
        A dictionary with desired ROI_bg property names that will be 
        thresholded as keys and the corresponding threshold values as values. 
    
    Returns
    =======
    
    thresholded_rois : list 
        A list containing instances of ROI_bg which pass the thresholding step.
    """
    # If there is no threshold
    if threshold_dict is None:
        print('No threshold used.')
        return rois
    vars_to_threshold = threshold_dict.keys()
    
    roi_data_dict = data_to_list(rois, vars_to_threshold)
    
    pass_bool = np.ones((1,len(rois)))
    
    for key, value in threshold_dict.items():
        
        if type(value) == tuple:
            if value[0] == 'b':
                pass_bool = \
                    pass_bool * (np.array(roi_data_dict[key]).flatten() > value[1])
                
            elif value[0] == 's':
                pass_bool = \
                    pass_bool * (np.array(roi_data_dict[key]).flatten() < value[1])
            else:
                raise TypeError("Tuple first value not understood: should be 'b' for bigger than or 's' for smaller than")
                
        else:
            pass_bool = pass_bool * (np.array(roi_data_dict[key]).flatten() > value)
    
    pass_indices = np.where(pass_bool)[1]
    
    thresholded_rois = []
    for idx in pass_indices:
        thresholded_rois.append(rois[idx])
    
    return thresholded_rois
    
    

    
def calculate_distance_from_region(rois):
    """
    """
    
    # Show the image
    fig = plt.figure()
    rois_image = get_masks_image(rois)
    
    plt.imshow(rois[0].source_image, interpolation='nearest', cmap='gray')
    plt.imshow(rois_image, alpha=0.5,cmap = 'tab20b')
    
    
    plt.title("Select a region")
    plt.show(block=False)
   
    
    # Draw ROI
    curr_roi = RoiPoly(color='r', fig=fig) #commented out by Juan
    plt.waitforbuttonpress()
    plt.pause(10)
    curr_mask = curr_roi.get_mask(rois_image)
    distance_mask=scipy.ndimage.morphology.distance_transform_edt(1-curr_mask)

    distances = list(map(lambda roi : np.min(distance_mask[roi.mask]), rois))
    
      
    for i, roi in enumerate(rois):
            roi.distance = distances[i]
    
    return distances,distance_mask


def calculate_edge_timing(rois):
    """
    """

    for roi in rois:
        # Find edge epochs
        edge_epochs = np.where(roi.stim_info['stimtype'] == 50)[0]
        # Find the edge epoch with max response
        idx = np.nanargmax(roi.max_resp_all_epochs[edge_epochs])
        max_edge_epoch = edge_epochs[idx]
        raw_trace = roi.resp_trace_all_epochs[max_edge_epoch]

        frameRate_approx = len(raw_trace)/8.0
        # Considering edge epochs are presented for 8s (change this later on for robustness)
        roi.edge_peak_t = np.argmax(raw_trace)/frameRate_approx

    edge_timings = list(map(lambda roi: roi.edge_peak_t, rois))


    return edge_timings
    

def make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm', plot_x='reliability',plot_y='CSI'):
    
    from post_analysis_core import run_matplotlib_params
    
    curr_roi_props = rois_df[rois_df['uniq_id']==roi.uniq_id]
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors = run_matplotlib_params()
    
    try:
        CS = roi.CS
    except AttributeError :
        CS = ''
    if roi.CS =='OFF':
        color = colors[3]
    else:
        color = colors[2]
    # Constructing the plot backbone
    fig = plt.figure(figsize=(9, 12))
    grid = GridSpec(7, 8, wspace=2, hspace=1)
    
    # Mask
    plt.subplot(grid[:3,:3])
    roi.showRoiMask(cmap='PiYG')
    plt.title('%s PD: %d CS: %s' % (roi.uniq_id,roi.PD,CS))
    
    # plt.subplot(grid[:3,:3])
    # screen = np.zeros(np.shape(roi.RF_map))
    # screen[np.isnan(roi.RF_map)] = -0.1
    # curr_RF = np.full(np.shape(roi.RF_map), np.nan)
    # curr_RF[roi.RF_map_norm>0.99] = 1
    # # curr_RF[roi.RF_map_norm>0.99] = roi.__dict__['BF']
    # plt.imshow(roi.RF_map,alpha=1,cmap=cmap)   
    # plt.imshow(curr_RF,alpha=1,cmap='Greens_r',vmin=0,vmax=2,label='RF center')        
    
    # ax = plt.gca()
    # ax.set_title('ROI %s \nPD: %d CS: %s' % (roi.uniq_id,roi.PD,CS))
    # ax.axis('off')
    # ax.set_xlim(((np.shape(roi.RF_map)[0]-60)/2,(np.shape(roi.RF_map)[0]-60)/2+60))
    # ax.set_ylim(((np.shape(roi.RF_map)[0]-60)/2+60,(np.shape(roi.RF_map)[0]-60)/2))
    
    # Raw Trace
    plt.subplot(grid[0,3:])
    roi.plotDF(line_w=0.5,color = color)
    plt.title('Raw trace')
    
    # First property
    plt.subplot(grid[1:3,3:5])
    sns.scatterplot(x=plot_x, y=plot_y,alpha=.8,color='grey',
                    data =rois_df,legend=False,size=10)
    sns.scatterplot(x=plot_x, y=plot_y,color=color,
                    data =curr_roi_props,legend=False,size=7)
    
    plt.xlim(0, rois_df[plot_x].max()+0.3)
    plt.ylim(0, rois_df[plot_y].max()+0.3)
    
    
    # Tuning curve
    ax = plt.subplot(grid[1:3,5:])
    plt.plot(roi.TF_curve_stim,roi.TF_curve_resp,'-o',
             color = color,lw=3,markersize=10)
    
    ax.set(xscale="log")
    ax.set_title('TF tuning')
    ax.set_ylim(0,np.max(roi.TF_curve_resp)+0.2)
    ax.set_xlim((ax.get_xlim()[0],10)) 
    # Plot edge epochs
    grating_epochs = np.where(((roi.stim_info['stimtype'] == 61) | \
                             (roi.stim_info['stimtype'] == 46))\
                             & (roi.stim_info['epoch_frequency'] >0))[0]
    unique_freqs = np.unique(roi.stim_info['epoch_frequency'][grating_epochs])
    if len(unique_freqs) > 8:
        unique_freqs = unique_freqs[:8]
        
    adder =  4/len(unique_freqs)/2 + 1
    modn = len(unique_freqs)/2
    col_counter = np.repeat(range(2),len(unique_freqs)/2) *4
    
    
    for idx, freq in enumerate(unique_freqs):
        epochs = grating_epochs[roi.stim_info['epoch_frequency']\
                                [grating_epochs] == freq]
        ax = plt.subplot(grid[(3 + np.mod(idx,modn)):\
                              (3 + np.mod(idx,modn))+adder,col_counter[idx]\
                              :col_counter[idx]+4])
        
        max_epoch = epochs[np.argmax(roi.max_resp_all_epochs[epochs])]

        current_dir = roi.stim_info['epoch_dir'][max_epoch]
        current_freq = roi.stim_info['epoch_frequency'][max_epoch]
        current_epoch_type = roi.stim_info['stimtype'][max_epoch]
        opp_epoch = find_opp_epoch(roi, current_dir, current_freq, current_epoch_type)
        opp_epoch = opp_epoch[0]
        epochs_to_plot =[max_epoch, opp_epoch]
        
        for epoch in epochs_to_plot:
            if epoch==max_epoch:
                curr_color = color
            else:
                curr_color = colors[1]
                
            label_str = ('Dir: %d' % roi.stim_info['epoch_dir'][epoch])
            ax.plot(roi.whole_trace_all_epochs[epoch],label=label_str,lw=2,
                    color = curr_color)
            try :
                base_dur = roi.base_dur[epochs[0]]
                ax.plot([base_dur,base_dur],
                        [-0.5,np.ceil(roi.max_response)],'r')
                base_end = len(roi.resp_trace_all_epochs[epochs[0]]) +base_dur
                ax.plot([base_end,base_end],
                        [-0.5,np.ceil(roi.max_response)],'r')
     
            except AttributeError:
                print('No baseline duration found')
           
        ax.set_title('%.2f Hz'%freq)
        ax.set_ylim(-0.5,np.around(np.nanmax(roi.max_resp_all_epochs[grating_epochs]),1))
    fig.tight_layout()
    return fig
            
def calculate_distance_between_rois(rois):
    """
    Calculates the distance (in micrometers) between different ROIs.
    :param rois:
    :return distance_matrix:
    """
    distance_matrix = np.zeros(shape=(len(rois),len(rois)))
    for i, roi in enumerate(rois):
        distance_mask = scipy.ndimage.morphology.distance_transform_edt(1 - roi.mask)
        distance_mask = distance_mask*roi.imaging_info['pix_size']

        distance_matrix[i,:] = np.array(map(lambda roi: np.min(distance_mask[roi.mask]), rois))

    return distance_matrix

def fit_gauss(x,y):
    
    from scipy.optimize import curve_fit
    from scipy import asarray as exp
    
    
    n = len(x)                         
    mean = sum(x*y)/n                   
    sigma = sum(y*(x-mean)**2)/n        
    
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
    
    return gaus(x,*popt), popt, pcov

def map_RF(rois,edges=True,screen_dim = 60):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    """
    from scipy.ndimage.interpolation import rotate
    from scipy.stats import zscore
   
    screen_coords = np.linspace(0, screen_dim, num=screen_dim, endpoint=True) # degree of the screen

    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and (roi.stim_name.find('LumDecLumInc') != -1):
            # cut the trace len to half for ON and OFF epochs
            trace_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
        pad = np.ceil((np.sqrt(2)*screen_dim-screen_dim)/2)
        dim = screen_dim+2*pad
        all_RFs = []

        
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                full_trace = signal.filtfilt(b, a,roi.resp_trace_all_epochs[epoch_idx])
                raw_trace = np.full((trace_len,),np.nan)
               
                if edges:
                    if roi.CS =='OFF':
                        raw_trace = full_trace[:trace_len]
                    else:

                        raw_trace[:trace_len] =\
                            full_trace[trace_len-np.mod(len(full_trace),2):
                                       trace_len-np.mod(len(full_trace),2)+trace_len]
                
                normalized = (raw_trace - min(raw_trace)) / \
                             (max(raw_trace) - min(raw_trace))
                # standardized = zscore(curr_response)
                resp_trace = raw_trace
                edge_speed = \
                    roi.stim_info['input_data']['Stimulus.stimtrans.mean'][epoch_idx]
                delay_frames = \
                    np.around(9.6/float(edge_speed) * roi.imaging_info['frame_rate'],0)
                resp_trace = np.roll(resp_trace,-int(delay_frames))
                roi_t_v = np.linspace(0, screen_dim, num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                
                
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]

                back_projected = np.tile(i_resp, (len(i_resp),1) )
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, angle=np.mod(90-curr_direction,360))
                rotated[rotated==1000] = np.nan
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                rot_dim = len(rotated)
                idx1 = int((dim-rot_dim)/2)
                idx2 = int((dim-rot_dim)/2+rot_dim)
                curr_RF[idx1 : idx2,idx1 : idx2] = rotated
                all_RFs.append(curr_RF)
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]


    return rois

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def map_RF_adjust_edge_time(rois,save_path,edges=True,delay_degrees=9.6,delay_use=False,
                            edge_props = {'45':113, '135':113,'225':113,'315':113,
                                          '0':80,'180':80,
                                          '90':80,'270':80},mapping_mode=False):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    
    Added the perspective correction by measuring the extend of the screen in
    all directions, following is the measurement from 31/01/2020:
        Horizontal (90 and 270 degrees direction) : 75 degrees
        Vertical (0 and 180) : 51 degrees
        Diagonals (45, 135, 225, 315) : 71 degrees
        
    Also added the delay that was measured as 9.6 degrees by the experiments
    that I have done. (delay between a 20dps edge and the center of RF probed
    with standing stripes)

    JF: corrected some calculations

    """
    from scipy.ndimage import rotate
    from scipy.stats import zscore
    directions=(len(rois[0].direction_vector))
    dim = int(np.max(list(edge_props.values())))*np.sqrt(2)
    dim = round(dim)
    centers_on= np.full((int(dim), int(dim),directions),np.nan)
    centers_off= np.full((int(dim), int(dim),directions),np.nan)
    centers_count_ON=np.zeros(directions)
    centers_count_OFF=np.zeros(directions)
    center_cat=[]
    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and \
        (roi.stim_name.find('LumDecLumInc') != -1 \
                or roi.stim_name == 'DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                or roi.stim_name == 'mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                or roi.stim_name == 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt'\
                or roi.stim_name == 'Drifting_stripesJF_8dir.txt'):
            # cut the trace len to half for ON and OFF epochs
            half_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
            
        # We need a common dimension for mapping all images and this should be 
        # the turned version of the maximum covered distance in the screen.
        # The turned version is maximum after a 45 degree turn (or for other
        # angles that have 45 deg in mod 90).
        # dim = round(np.max(list(edge_props.values())))*np.sqrt(2) 
        dim = round(dim)
        all_RFs = []
        roi.delayed_RF_traces ={}
        roi.non_delayed_RF_traces ={}
        max_vals=[]
        full_traces=[]
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 'ADS' or epoch_type== 'driftingstripe' or epoch_type=='G':
                edge_speed = \
                    roi.stim_info['velocity'][epoch_idx] #Seb: what is store in this sitmulus input column?
                
                
                
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = copy.deepcopy(roi.whole_trace_all_epochs[epoch_idx])
                whole_t = signal.filtfilt(b, a,whole_t)
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = int((len(whole_t)-resp_len)/2)
                #base_t = whole_t[:base_len]
                #base_activity = np.mean(base_t)
                
                if delay_use:
                    delay_frames = \
                        np.around(delay_degrees/float(edge_speed) * roi.imaging_info['frame_rate'],0)
                    delayed_trace = np.roll(whole_t,-int(delay_frames))
                    roi.delay_used_in_RF = delay_degrees
                    full_trace =delayed_trace[base_len:base_len+resp_len]
                    
                    roi.delayed_RF_traces[epoch_idx] = full_trace
                    roi.non_delayed_RF_traces[epoch_idx] = whole_t[base_len:base_len+resp_len]
                    
                else:
                    roi.delay_used_in_RF = 0
                    full_trace =whole_t[base_len:base_len+resp_len]
                    
                #JF edit. detrend trace. the filtering led to the baseline going above 0:
                full_traces.append(detrend(full_trace,type='constant'))
                max_vals.append(np.nanmax(detrend(full_trace,type='constant')))
        max_val=np.max(max_vals)
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            full_trace=full_traces[epoch_idx]
            curr_RF = np.full((int(dim), int(dim)),np.nan)
            if epoch_type == 'ADS' or epoch_type== 'driftingstripe' or epoch_type=='G':
                
                curr_direction = roi.stim_info['angle'][epoch_idx]
                try:
                    degrees_covered = edge_props[str(int(curr_direction))]
                    frames_needed = int(np.around((degrees_covered/float(edge_speed))\
                        * roi.imaging_info['frame_rate'],0))
                    tau_frames=int(np.around(roi.stim_info['tau'][epoch_idx]*roi.imaging_info['frame_rate'],0))
                except KeyError:
                    raise KeyError('Edge direction not found: %s degs' % str(int(curr_direction)))

                # if edges and roi.stim_info['polarity'][epoch_idx]==0: #PRADEEP from miriam's code (stimuli from Juan)
                #     for cs in ['ON','OFF']:
                #         if roi.CS =='OFF':   #for OFF edge, the response is the second half of the trace
                #             if int(curr_direction) in [0, 180, 90, 270]:
                #                 raw_trace = full_trace[len(full_trace)-round(5.25*roi.imaging_info['frame_rate']):]
                #             else:
                #                 raw_trace = full_trace[len(full_trace)-round(6.9*roi.imaging_info['frame_rate']):]    
                            
                #         else: #for ON edge, the response is the second half of the trace
                #             if int(curr_direction) in [0, 180, 90, 270]:
                #                 raw_trace = full_trace[:round(5.25*roi.imaging_info['frame_rate'])]
                #             else:
                #                 raw_trace = full_trace[:round(6.9*roi.imaging_info['frame_rate'])]    


                # elif edges and roi.stim_info['polarity'][epoch_idx]== 1 :  #'ONEDGEFIRST' not in roi.stim_info['stim_name']:
                #     if roi.CS =='OFF':
                #         raw_trace = full_trace[:frames_needed]
                #     else:

                #         raw_trace = full_trace[half_len:half_len+frames_needed]        

                if tau_frames>0 and epoch_type=='ADS': # JF edit
                    if roi.CS =='OFF':
                        # raw_trace = full_trace[frames_needed:half_len+frames_needed-tau_frames]
                        raw_trace = full_trace[:frames_needed-tau_frames]
                    elif roi.CS =='ON':
                        raw_trace = full_trace[:frames_needed]
                # elif epoch_type=='G' and roi.stim_info['polarity'][epoch_idx]==0:
                #     if roi.CS =='OFF':
                #         raw_trace = full_trace[half_len:half_len+frames_needed]
                #     elif roi.CS =='ON':
                #         raw_trace = full_trace[:frames_needed]                    
                # Standardize responses so that DS responses dominate less JF: we want the opposite to get less noisy RFs
                #sd = np.std(raw_trace) 
                
                
                if roi.CS=='OFF': # JF edit: in direction selective neurons, one cannot give the same weight to responsive and unresponsive directions
                    normalized = raw_trace/max_val #scipy.stats.zscore(raw_trace) #JF edit # (raw_trace - np.mean(raw_trace))/sd
                elif roi.CS=='ON':
                    normalized = raw_trace/max_val
                resp_trace = normalized                

                # Need to map to the screen
                diagonal_dir = str(int(roi.direction_vector[epoch_idx])) #JF edit
                
                degree_needed = degrees_covered
                diag_dir_covered = edge_props[diagonal_dir]
                deg_frame=roi.stim_info['velocity'][epoch_idx]/roi.imaging_info['frame_rate']
                loc_vector=roi_t_v = np.array(range(len(resp_trace)))*deg_frame # Juan edit
                    
                screen_coords = np.linspace(0, degree_needed, 
                                            num=degree_needed, endpoint=True) 
                # roi_t_v = np.linspace(0, degree_needed, 
                #                       num=len(resp_trace), endpoint=True) 
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                #diagonal_dir = str(int(np.mod(curr_direction+90,360)))
                back_projected = np.tile(i_resp, (diag_dir_covered,1))
                back_projected[np.isnan(back_projected)] = 0
                # 
                #jf-edit the rotation of a plane containing a single direction map should be the same as the 
                # stimulus direction as long as 0 deg is right, 90 deg is up, etc
                
                rotated = rotate(back_projected+1, 
                                 angle=roi.stim_info['angle'][epoch_idx]) # check if correct
                rotated[rotated==0] = np.nan
                rotated = rotated-1

                idx1_1 = int((dim-rotated.shape[0])/2)
                idx1_2 = int((dim-rotated.shape[0])/2+rotated.shape[0])
                
                idx2_1 = int((dim-rotated.shape[1])/2)
                idx2_2 = int((dim-rotated.shape[1])/2+rotated.shape[1])
                
                
                curr_RF[idx1_1 : idx1_2,idx2_1 : idx2_2] = rotated
                all_RFs.append(curr_RF)
                
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        # superimpose a 60 deg circle (for the case that the mapping is done with a circular mask)
        arr = np.full((roi.RF_map_norm.shape[0], roi.RF_map_norm.shape[1]), np.nan)
        center = roi.RF_map_norm.shape[1] // 2
        radius = 25
        y, x = np.indices((roi.RF_map_norm.shape[0], roi.RF_map_norm.shape[1]))
        circle = (x - center)**2 + (y - center)**2
        arr[np.where((circle <= (radius + 0.5)**2) & (circle >= (radius - 0.5)**2))] = 1
        if (roi.RF_center_coords[0] - center) ** 2 + (roi.RF_center_coords[1] - center) ** 2 >= radius ** 2:
            roi.Center_position_filter= False
        else:
            roi.Center_position_filter= True
        #TODO calculate a gaussian fit and test significance
        def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
            x, y = xy
            xo = float(xo)
            yo = float(yo)
            a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
            b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
            c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
            g = offset + amplitude * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
            return g.ravel()
        import lmfit
        from lmfit.lineshapes import gaussian2d
        # Create a meshgrid of x and y coordinates
        x = np.linspace(0, roi.RF_map_norm.shape[1] - 1, roi.RF_map_norm.shape[1])
        y = np.linspace(0, roi.RF_map_norm.shape[0] - 1, roi.RF_map_norm.shape[0])
        x, y = np.meshgrid(x, y)

        # Create a mask to filter out NaN values
        mask = ~np.isnan(roi.RF_map_norm)

        # Filter the data and coordinates using the mask
        x_filtered = x[mask]
        y_filtered = y[mask]
        data_filtered = roi.RF_map_norm[mask]
        error = np.sqrt(data_filtered+1)
        # Flatten the filtered coordinates
        xy_filtered = (x_filtered, y_filtered)
        model = lmfit.models.Gaussian2dModel()
# Initial guess for the parameters
        params = model.guess(data_filtered, x_filtered, y_filtered)

        # Fit the model to the filtered data
        result = model.fit(data_filtered, params, x=x_filtered, y=y_filtered, weights=1/error)

        # Print the fitting results
        # print(result.rsquared)

        # Generate the fitted Gaussian
        fitted_gaussian = gaussian2d(x, y, **result.best_values).reshape(roi.RF_map_norm.shape)

        # Plot the original data and the fitted Gaussian
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(roi.RF_map_norm, cmap='viridis', extent=[0, roi.RF_map_norm.shape[1], 0, roi.RF_map_norm.shape[0]])
        plt.colorbar()
        plt.title('Original RF_map_norm')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 2, 2)
        plt.imshow(fitted_gaussian, cmap='viridis', extent=[0, roi.RF_map_norm.shape[1], 0, roi.RF_map_norm.shape[0]])
        plt.colorbar()
        plt.title(f'Fitted 2D Gaussian: R^2 = {result.rsquared}')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.savefig(f'')
        # plt.show()

    #     if mapping_mode==False:
    #         plt.figure()
    #         plt.imshow(roi.RF_map_norm,cmap='viridis')
    #         plt.colorbar()
    #         plt.clim(0,1) 
    #         plt.imshow(arr,cmap='gray')
    #     if roi.CS=='ON':
    #         index=roi.max_resp_idx_ON
    #     else:
    #         index=roi.max_resp_idx_OFF
    #     plt.title('roi: %s CS: %s CSI:%s cat: %s PD: %s' %(roi.uniq_id, roi.CS, roi.CSI,roi.category,roi.direction_vector[index]))
    # value_dict={'rois_layerA':1,'rois_layerB':2,'rois_layerC':3,'rois_layerD':4}
    # for roi in rois:
    #     if roi.category== ['No_category'] :
    #         continue 
    #     if roi.CS=='ON':
    #         centers_on[roi.RF_center_coords[0]-3:roi.RF_center_coords[0]+3,roi.RF_center_coords[1]-3:roi.RF_center_coords[1]+3,roi.max_resp_idx_ON]=value_dict[roi.category]
    #         centers_count_ON[roi.max_resp_idx_ON]+=1
    #     else:    
    #         centers_off[roi.RF_center_coords[0]-3:roi.RF_center_coords[0]+3,roi.RF_center_coords[1]-3:roi.RF_center_coords[1]+3,roi.max_resp_idx_OFF]=value_dict[roi.category]
    #         centers_count_OFF[roi.max_resp_idx_OFF]+=1
    
    # for ix in range(directions):
    #     plt.figure()
    #     plt.imshow(centers_on[:,:,ix],cmap='jet')
    #     plt.colorbar()
    #     plt.clim(0,4)
    #     plt.imshow(arr,cmap='gray')
    #     plt.title('dir:%s, rois recovered: %son'%(rois[0].direction_vector[ix],centers_count_ON[ix]))
    #     plt.figure()        
    #     plt.imshow(centers_off[:,:,ix],cmap='jet')
    #     plt.colorbar()
    #     plt.clim(0,4)
    #     plt.imshow(arr,cmap='gray')
    #     plt.title('dir:%s, rois recovered: %soff'%(rois[0].direction_vector[ix],centers_count_OFF[ix]))
    multipage(save_path +'\\RF_backprojection_roi.pdf', figs=None, dpi=200)
    plt.close('all')
    return rois

def calculate_roi_delay(roi):
    
    delays_s = []
    delays_f = []
    for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
        if epoch_type == 50:
            
            curr_dir = roi.stim_info['epoch_dir'][epoch_idx]
            curr_freq = roi.stim_info['epoch_frequency'][epoch_idx]
            curr_type = roi.stim_info['stimtype'][epoch_idx]
            
            opp_epoch = find_opp_epoch_roi(roi,curr_dir,curr_freq,curr_type)
            
            b, a = signal.butter(3, 0.2, 'low')
            trace1 = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
            trace2 = signal.filtfilt(b, a,roi.whole_trace_all_epochs[int(opp_epoch)])
            trace2 = np.flip(trace2)
            
            delay_in_frames = np.abs(np.argmax(trace2) - np.argmax(trace1))
            delays_f.append(delay_in_frames)
            delay_in_s = delay_in_frames/roi.imaging_info['frame_rate']
            delays_s.append(delay_in_s)
        else:
            delays_f.append(None)
            delays_s.append(None)
    return delays_f, delays_s
            
    

                
                
def map_RF_adjust_stripe_time(rois,screen_props = {'45':74, '135':72,
                                                   '225':74,'315':72,
                                                   '0':53,'180':53,
                                                   '90':78,'270':78},
                              delay_use=False):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    
    Added the perspective correction by measuring the extend of the screen in
    all directions, following is the measurement from 02/2020:
        Horizontal (90 and 270 degrees direction) : 78 degrees
        Vertical (0 and 180) : 53 degrees
        Diagonals (45, 225) : 74 degrees
        Diagonals (135, 315) : 72 degrees
        
    # Delay correction not good as currently implemented here. Non DS neurons 
    # do not need delay correction
        
    """
    from scipy.ndimage.interpolation import rotate

    for i, roi in enumerate(rois):
        
        
                
        dim = int(np.max(screen_props.values())*np.sqrt(2))
        all_RFs = []
        
        (delays_f, delays_s) = calculate_roi_delay(roi)
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                stripe_speed = \
                    roi.stim_info['input_data']['Stimulus.stimtrans.mean'][epoch_idx] #Seb: what is store in this sitmulus input column?
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]
                try:
                    degrees_covered = screen_props[str(int(curr_direction))]
                    frames_needed = int(np.around((degrees_covered/float(stripe_speed))\
                        * roi.imaging_info['frame_rate'],0))
                except KeyError:
                    raise KeyError('Edge direction not found: %s degs' % str(int(curr_direction)))
                
                
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
                whole_t = whole_t -np.min(whole_t)
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                
                roi.delay_used = delay_use
                if delay_use:
                    delayed_trace = np.roll(whole_t,
                                            -int(delays_f[epoch_idx]))
                    raw_trace =delayed_trace[base_len:base_len+resp_len]
                else:
                    raw_trace =whole_t[base_len:base_len+resp_len]
                    
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
                normalized = (raw_trace - base_activity)/sd
                resp_trace = normalized
            
                # Need to map to the screen
                diagonal_dir = str(int(np.mod(curr_direction+90,360)))
                
                degree_needed = degrees_covered
                diag_dir_covered = screen_props[diagonal_dir]
                    
                    
                screen_coords = np.linspace(0, degree_needed, 
                                            num=degree_needed, endpoint=True)
                roi_t_v = np.linspace(0, degree_needed, 
                                      num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                diagonal_dir = str(int(np.mod(curr_direction+90,360)))
                back_projected = np.tile(i_resp, (diag_dir_covered,1))
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, 
                                 angle=np.mod(90-curr_direction,360))
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                idx1_1 = int((dim-rotated.shape[0])/2)
                idx1_2 = int((dim-rotated.shape[0])/2+rotated.shape[0])
                
                idx2_1 = int((dim-rotated.shape[1])/2)
                idx2_2 = int((dim-rotated.shape[1])/2+rotated.shape[1])
                
                
                curr_RF[idx1_1 : idx1_2,idx2_1 : idx2_2] = rotated
                all_RFs.append(curr_RF)
                
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
    return rois


def map_RF_v2(rois,edges=True,screen_dim = 60,delay=9.6):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    """
    from scipy.ndimage.interpolation import rotate
    from scipy.stats import zscore
   
    screen_coords = np.linspace(0, screen_dim, num=screen_dim, endpoint=True) # degree of the screen

    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and (roi.stim_name.find('LumDecLumInc') != -1):
            # cut the trace len to half for ON and OFF epochs
            trace_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
        pad = np.ceil((np.sqrt(2)*screen_dim-screen_dim)/2)
        dim = screen_dim+2*pad
        all_RFs = []
        all_RFs_no_delay = []

        
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                curr_RF_nd=np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                
                
                full_trace =whole_t[base_len:base_len+resp_len]
                raw_trace = np.full((trace_len,),np.nan)
               
                if edges:
                    if roi.CS =='OFF':
                        raw_trace = full_trace[:trace_len]
                    else:

                        raw_trace[:trace_len] =\
                            full_trace[trace_len-np.mod(len(full_trace),2):
                                       trace_len-np.mod(len(full_trace),2)+trace_len]
                
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
                normalized = (raw_trace - base_activity)/sd
                                
                # normalized = (raw_trace - min(raw_trace)) / \
                #              (max(raw_trace) - min(raw_trace))
                # standardized = zscore(curr_response)
                resp_trace = normalized
                
                # Fixing the delay of responses
                edge_speed = \
                    roi.stim_info['input_data']['Stimulus.stimtrans.mean'][epoch_idx] #Seb: what is store in this sitmulus input column?
                delay_frames = \
                    np.around(delay/float(edge_speed) * roi.imaging_info['frame_rate'],0)
                resp_trace = np.roll(resp_trace,-int(delay_frames))
                roi.resp_delay = delay
                resp_trace[-int(delay_frames):] = np.min(resp_trace)
                
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]
                # Need to map to the screen
                roi_t_v = np.linspace(0, screen_dim, 
                                      num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                back_projected = np.tile(i_resp, (len(i_resp),1) )
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, 
                                 angle=np.mod(90-curr_direction,360))
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                rot_dim = len(rotated)
                idx1 = int((dim-rot_dim)/2)
                idx2 = int((dim-rot_dim)/2+rot_dim)
                curr_RF[idx1 : idx2,idx1 : idx2] = rotated
                all_RFs.append(curr_RF)
                
                # Store the non-delayed original ones for a back up
                roi_t_v_nd = np.linspace(0, screen_dim, num=len(normalized), endpoint=True)
                i_resp_nd = np.interp(screen_coords, roi_t_v_nd, normalized)
                back_projected_nd = np.tile(i_resp_nd, (len(i_resp_nd),1) )
                back_projected_nd[np.isnan(back_projected_nd)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated_nd = rotate(back_projected_nd+1, 
                                    angle=np.mod(90-curr_direction,360))
                rotated_nd[rotated_nd==0] = np.nan
                rotated_nd = rotated_nd-1
                rot_dim_nd = len(rotated_nd)
                idx1_nd = int((dim-rot_dim_nd)/2)
                idx2_nd = int((dim-rot_dim_nd)/2+rot_dim_nd)
                curr_RF_nd[idx1_nd : idx2_nd,idx1_nd : idx2_nd] = rotated_nd
                all_RFs_no_delay.append(curr_RF_nd)
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
        roi.RF_maps_no_delay = all_RFs_no_delay
        roi.RF_map_no_delay = np.mean(roi.RF_maps_no_delay, axis=0)
        roi.RF_center_coords_no_delay =\
            np.argwhere(roi.RF_map_no_delay==np.nanmax(roi.RF_map_no_delay))[0]


    return rois

def map_RF_no_delay(rois,edges=True,screen_dim = 60):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    """
    from scipy.ndimage.interpolation import rotate
   
    screen_coords = np.linspace(0, screen_dim, num=screen_dim, endpoint=True) # degree of the screen

    for i, roi in enumerate(rois):
        lens = [len(v) for v in roi.resp_trace_all_epochs.values()]
        if edges and (roi.stim_name.find('LumDecLumInc') != -1):
            # cut the trace len to half for ON and OFF epochs
            trace_len = int(np.ceil(min(lens)/2.0)) 
        elif edges:
            print('Stimulus do not contain ON and OFF edges in single epoch.')
            
        pad = np.ceil((np.sqrt(2)*screen_dim-screen_dim)/2)
        dim = screen_dim+2*pad
        all_RFs = []
        
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 50:
                curr_RF = np.full((int(dim), int(dim)),np.nan)
                b, a = signal.butter(3, 0.2, 'low')
                whole_t = signal.filtfilt(b, a,roi.whole_trace_all_epochs[epoch_idx])
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                
                full_trace =whole_t[base_len:base_len+resp_len]
                raw_trace = np.full((trace_len,),np.nan)
               
                if edges:
                    if roi.CS =='OFF':
                        raw_trace = full_trace[:trace_len]
                    else:

                        raw_trace[:trace_len] =\
                            full_trace[trace_len-np.mod(len(full_trace),2):
                                       trace_len-np.mod(len(full_trace),2)+trace_len]
                
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
                normalized = (raw_trace - base_activity)/sd
                resp_trace = normalized
                
                
                curr_direction = roi.stim_info['epoch_dir'][epoch_idx]
                # Need to map to the screen
                roi_t_v = np.linspace(0, screen_dim, 
                                      num=len(resp_trace), endpoint=True)
                i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
                back_projected = np.tile(i_resp, (len(i_resp),1) )
                back_projected[np.isnan(back_projected)] = 0
                # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
                # 0 degrees is upwards so 90-curr_dir
                # 
                rotated = rotate(back_projected+1, 
                                 angle=np.mod(90-curr_direction,360))
                rotated[rotated==0] = np.nan
                rotated = rotated-1
                rot_dim = len(rotated)
                idx1 = int((dim-rot_dim)/2)
                idx2 = int((dim-rot_dim)/2+rot_dim)
                curr_RF[idx1 : idx2,idx1 : idx2] = rotated
                all_RFs.append(curr_RF)
                
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
    return rois


# Gaussian 2d fit
def twoDgaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitTwoDgaussian(data):
    from scipy import optimize
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(twoDgaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p



def plot_RF(roi,cmap1='inferno',cmap2='viridis',center_plot = False, 
            center_val = 0.95):
    plt.close('all')
    colors = pac.run_matplotlib_params()
    fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(5, 4))
    ax = ax1.flatten()
    for idx , curr_axis in enumerate(ax):
        # Mean image

        if idx == 4:
  
            
            center_RF = copy.deepcopy(roi.RF_map_norm)
            center_RF[center_RF<center_val] =np.nan
            sns.heatmap(roi.RF_map_norm, cmap=cmap1, ax=ax[idx], cbar=False)
            if center_plot:
                sns.heatmap(center_RF,ax=ax[idx], cbar=False,alpha=.5,
                            cmap='Greens')
            
            
            
            # ax[idx].matshow(BT_map, cmap='viridis')
            #
            # BT_map = BT_map[]
            # BT_map[np.isnan(BT_map)] = np.nanmin(BT_map)
            # params = fitTwoDgaussian(BT_map)
            # fit = twoDgaussian(*params)
            # ax[idx].contour(fit(*np.indices(BT_map.shape)))

            
            ax[idx].axis('off')
            try:
                ax[idx].set_title('BT-RF - PD: {PD}'.format(PD = int(roi.PD)))
            except AttributeError:
                ax[idx].set_title('BT-RF - {cat}'.format(cat = roi.category))
            continue
        elif idx > 3:
            plt_idx = idx-1
        else:
            plt_idx = idx
        sns.heatmap(roi.RF_maps[plt_idx], ax=ax[idx], cmap=cmap2,cbar=False,
                    vmin=np.nanmin(roi.RF_maps),
                        vmax=np.nanmax(roi.RF_maps))
        
       
        
        epoch_dir = roi.stim_info['epoch_dir'][plt_idx+roi.stim_info['epoch_adjuster']]
        ax[idx].set_title(f'{str(int(epoch_dir))}circ')
        ax[idx].axis('off')


    return fig1

def plot_RFs(rois, number=None, f_w =None,cmap='inferno',
             center_plot = False, center_val = 0.95):
    import random
    plt.close('all')
    colors = pac.run_matplotlib_params()
    if (number == None) or (f_w==None):
        f_w = 5
        if len(rois)>10:
            number=10
        else:
            number = len(rois)
    elif number > len(rois):
        number = len(rois)
    f_w = f_w*2
    # Randomize ROIs
    copy_rois = copy.deepcopy(rois)
    # random.shuffle(copy_rois)
        
        
    
    if number <= f_w/2:
        dim1= number
        dim2 = 1
    elif number/float(f_w/2) > 1.0:
        dim1 = f_w/2
        dim2 = int(np.ceil(number/float(f_w/2)))
    fig1, ax1 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1, dim2))
    ax = ax1.flatten()
    for idx, roi in enumerate(rois):
        if idx == number:
            break
        
        center_RF = copy.deepcopy(roi.RF_map_norm)
        center_RF[center_RF<center_val] =np.nan
        ax[idx].imshow(roi.RF_map_norm, cmap=cmap)
        if center_plot:
            ax[idx].imshow(center_RF,alpha=.5,
                        cmap='Greens')
        ax[idx].axis('off')
        ax[idx].set_xlim(((np.shape(roi.RF_map_norm)[1]-75)/2,(np.shape(roi.RF_map_norm)[0]-75)/2+75))
        ax[idx].set_ylim(((np.shape(roi.RF_map_norm)[0]-51)/2+51,(np.shape(roi.RF_map_norm)[0]-51)/2))
        try:
            ax[idx].set_title('PD: {pd}'.format(pd=int(roi.PD)),fontsize='xx-small')
        except AttributeError:
            a=0
    try:
        for ax_id in range(len(ax)-idx-1):
            ax[ax_id+idx].axis('off')
    except:
        a =1
    return fig1


def plot_RF_centers_on_screen(rois,prop = 'PD',cmap='hsv',
                              ylab='PD (circ)',lims=(0,360),rel_t=0):

    plt.close('all')
    colors = pac.run_matplotlib_params()

    screen = np.zeros(np.shape(rois[0].RF_map))
    rfs = np.full(np.shape(rois[0].RF_map), np.nan)
    
    screen[np.isnan(rois[0].RF_map)] = -0.1
    plt.imshow(screen, cmap='binary', alpha=.3)
    for idx, roi in enumerate(rois):
        if roi.reliability < rel_t:
            continue
        curr_RF = np.full(np.shape(roi.RF_map), np.nan)
        # curr_RF[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        # rfs[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        
        if prop == None:
            curr_RF[roi.RF_map_norm>0.95] = idx+1
        else:
            curr_RF[roi.RF_map_norm>0.95] = roi.__dict__[prop]
                
        
        plt.imshow(curr_RF,alpha=.5,cmap=cmap,vmin =lims[0],vmax=lims[1])
   
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(ylab)

    # sns.heatmap(screen,cmap='binary',alpha=.3,cbar=False)
    
    ax = plt.gca()
    ax.set_xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    ax.set_ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    # ax.set_xlabel('Screen $^\circ$')
    # ax.set_ylabel('$^\circ$')
    
    ax.axis('off')


    fig = ax.get_figure()
    return fig

def plot_RF_centers_on_screen_smooth(rois,prop = 'PD',cmap='hsv',
                              ylab='PD (circ)',lims=(0,360),rel_t=0):
    import scipy.ndimage as ndimage
    plt.close('all')
    colors = pac.run_matplotlib_params()

    screen = np.zeros(np.shape(rois[0].RF_map))
    rfs = np.full(np.shape(rois[0].RF_map), np.nan)
    
    screen[np.isnan(rois[0].RF_map)] = -0.1
    plt.imshow(screen, cmap='binary', alpha=.3)
    for roi in rois:
        if roi.reliability < rel_t:
            continue
        curr_RF = np.full(np.shape(roi.RF_map), np.nan)
        curr_RF_form = np.full((roi.RF_map.shape[0],roi.RF_map.shape[1]-1), np.nan)
        # curr_RF[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        # rfs[roi.RF_center_coords[0]-1:roi.RF_center_coords[0]+1,
        #     roi.RF_center_coords[1]-1:roi.RF_center_coords[1]+1] = roi.__dict__[prop]
        
        curr_RF[roi.RF_map_norm>0.95] = roi.__dict__[prop]
        curr_RF[np.isnan(curr_RF)] = 0
        
        curr_RF_form[ np.diff(curr_RF) != 0] = roi.__dict__[prop]
        
        img = ndimage.gaussian_filter(curr_RF, sigma=(5, 5), order=0)
        img[img==0] = np.nan
        plt.imshow(curr_RF_form,alpha=.5,cmap=cmap,vmin =lims[0],vmax=lims[1])
   
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(ylab)

    # sns.heatmap(screen,cmap='binary',alpha=.3,cbar=False)
    
    ax = plt.gca()
    ax.set_xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    ax.set_ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    # ax.set_xlabel('Screen $^\circ$')
    # ax.set_ylabel('$^\circ$')
    
    ax.axis('off')


    fig = ax.get_figure()
    return fig

def generate_time_delay_profile_2Dedges(rois):
    
    for roi in rois:
        epochDur= roi.stim_info['duration']
        max_epoch = roi.max_resp_idx
        roi.edge_start_loc = roi.stim_info['input_data']['Stimulus.stimtrans.amp'][max_epoch] #Seb: what is store in this sitmulus input column?
        roi.edge_speed = roi.stim_info['input_data']['velocity'][max_epoch] #Seb: what is store in this sitmulus input column?
        half_dur_frames = int((round(roi.imaging_info['frame_rate'] * epochDur[max_epoch]))/2)
        trace = roi.resp_trace_all_epochs[roi.max_resp_idx]
        OFF_resp = trace[:half_dur_frames]
        ON_resp = trace[half_dur_frames:]
        if roi.CS =='ON':
            roi.two_d_edge_profile = ON_resp
        elif roi.CS =='OFF':
            roi.two_d_edge_profile = OFF_resp
    return rois


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
def fit_1d_gauss(data_x, data_y):
    
    p0 = [np.max(data_y), np.argmax(data_y), 1]
    coeff, pcov = curve_fit(gauss, data_x, data_y, p0=p0)
    fit_trace = gauss(data_x, *coeff)
    
    residuals = data_y- fit_trace
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_y-np.mean(data_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return fit_trace, r_squared, coeff
def generate_RF_map_stripes(rois, screen_w = 80):
    
    screen_coords = np.linspace(0, screen_w, num=screen_w, endpoint=True) # degree of the screen
    for roi in rois:
        roi_t_v_stripe = np.linspace(0, 60, num=len(roi.max_resp_all_epochs[1:]), endpoint=True)
        
        roi.i_stripe_resp = np.interp(screen_coords, roi_t_v_stripe, 
                                  np.transpose(roi.max_resp_all_epochs[1:])[0])
        try:
            fit_trace, r_squared, coeff = fit_1d_gauss(screen_coords, roi.i_stripe_resp)
        except RuntimeError:
            print('Fit parameters not found... ROI fitting stopped {s}'.format(s=roi))
            roi.discard = True
            roi.stripe_gauss_profile = None
            roi.stripe_gauss_coeff = None
            roi.stripe_r_squared = None
            roi.stripe_gauss_fwhm = None
            continue
        roi.stripe_gauss_profile = fit_trace
        roi.stripe_gauss_coeff = coeff
        roi.stripe_gauss_fwhm = 2.355 * coeff[2]
        roi.stripe_r_squared = r_squared
        
    return rois

def generate_RF_profile_stripes(rois):
    
    for roi in rois:
        rf_profile = np.zeros(np.shape(roi.max_resp_all_epochs[1:]))
        for epoch_idx, epoch_type in enumerate(roi.stim_info['stimtype']):
            if epoch_type == 57:
                
                whole_t = roi.whole_trace_all_epochs[epoch_idx]
                whole_t = whole_t -np.min(whole_t)+1
                resp_len = len(roi.resp_trace_all_epochs[epoch_idx])
                base_len = (len(whole_t)-resp_len)/2
                base_t = whole_t[:base_len]
                base_activity = np.mean(base_t)
                resp_trace =whole_t[base_len:base_len+resp_len]
                # Standardize responses so that DS responses dominate less
                sd = np.sqrt(np.sum(np.square(resp_trace))/(len(resp_trace)-1))
                normalized = (resp_trace - base_activity)/sd
                resp_trace = normalized
                rf_profile[epoch_idx-roi.stim_info['epoch_adjuster']]=\
                    np.max(resp_trace)
        roi.RF_profile = rf_profile
        roi.RF_profile_coords = \
            np.array(roi.stim_info['input_data']['Stimulus.stimtrans.amp'][1:],
                     float) #Seb: what is store in this sitmulus input column?
            
    a=np.array(map(lambda roi : roi.RF_profile, rois))
    all_rfs_sorted = list(a[:,:,0])
    all_rfs_sorted.sort(key=lambda trace : np.argmax(trace))
        
    lens=map(lambda roi : len(roi.resp_trace_all_epochs[roi.max_resp_idx]), 
          rois)
    max_traces =\
        map(lambda roi : roi.resp_trace_all_epochs[roi.max_resp_idx][:np.min(lens)], 
          rois)
    max_epoch_traces = np.vstack(max_traces)
    
    
    
    return rois, all_rfs_sorted,max_epoch_traces

def generate_time_delay_profile_combined(rois,screen_deg = 60):
    # TODO: Hard coded edge width bad practice!!!
    

    screen_coords = np.linspace(0, screen_deg, num=screen_deg, endpoint=True) # degree of the screen
    
    
    
    # Edge is presented in the full screen and not just 60 degrees of the visual field

    for roi in rois:
        
        roi_t_v_stripe = np.linspace(0, screen_deg, num=len(roi.max_resp_all_epochs[1:]), endpoint=True)
        
        
        diff = np.abs(int(roi.edge_start_loc)) - 40 -screen_deg/2
        start_frame = int(np.around((diff/float(roi.edge_speed)) * roi.imaging_info['frame_rate']))
        end_frame = start_frame + int(np.ceil((60/float(roi.edge_speed) * roi.imaging_info['frame_rate'])))
        roi_t_v_edge = \
            np.linspace(0, screen_deg, 
                        num=len(roi.two_d_edge_profile[start_frame:end_frame]),
                        endpoint=True)
        
        i_edge = np.interp(screen_coords, roi_t_v_edge, 
                           roi.two_d_edge_profile[start_frame:end_frame])
        roi.i_stripe_resp = np.interp(screen_coords, roi_t_v_stripe, 
                                  np.transpose(roi.max_resp_all_epochs[1:])[0])
        if roi.PD == 90: # Rotate the response if PD is 90 since  
            roi.i_edge_resp = i_edge[::-1]
        else:
            roi.i_edge_resp = i_edge
        try:
            fit_trace, r_squared, coeff = fit_1dtt_gauss(screen_coords, roi.i_edge_resp)
        except RuntimeError:
            print('Fit parameters not found... discarding {s}'.format(s=roi))
            roi.discard = True
            roi.edge_gauss_profile = None
            roi.edge_r_squared = None
            roi.edge_gauss_coeff = None
            roi.resp_delay_deg = None
            roi.resp_delay_fits_Rsq = None
            continue
        roi.edge_gauss_profile = fit_trace
        roi.edge_r_squared = r_squared
        roi.edge_gauss_coeff = coeff
        
        try:
            fit_trace, r_squared, coeff = fit_1d_gauss(screen_coords, roi.i_stripe_resp)
        except RuntimeError:
            print('Fit parameters not found... discarding {s}'.format(s=roi))
            roi.discard = True
            roi.stripe_gauss_profile = None
            roi.stripe_r_squared = None
            roi.stripe_gauss_coeff = None
            roi.resp_delay_deg = None
            roi.resp_delay_fits_Rsq = None
            continue
        roi.stripe_gauss_profile = fit_trace
        roi.stripe_r_squared = r_squared
        roi.stripe_gauss_coeff = coeff
        
        roi.resp_delay_deg = np.abs(np.argmax(roi.stripe_gauss_profile) - np.argmax(roi.edge_gauss_profile))
        roi.resp_delay_fits_Rsq = np.array([roi.edge_r_squared,roi.stripe_r_squared])
        roi.discard = False
        
    return rois
        
def filter_delay_profile_rois(rois,Rsq_t = 0.5):
    filtered_rois = []
    for roi in rois:
        if roi.discard:
            continue
        elif np.where(roi.resp_delay_fits_Rsq < Rsq_t)[0].size>0:
            continue
        else:
            filtered_rois.append(roi)
   
    return filtered_rois
def plot_delay_profile_examples(rois, number=None,f_w=None,lw=1.3,alpha=.7,
                                colors = None):
    import random
    plt.close('all')

    colorss = pac.run_matplotlib_params()
    plt.rcParams["axes.titlesize"] = 'x-small'
    
    rois_to_plot = []
    for idx, roi in enumerate(rois):
        if roi.discard:
           
            continue
        if np.where(roi.resp_delay_fits_Rsq < 0.5)[0].size>0:
            
            continue
        rois_to_plot.append(roi)
    # Randomize ROIs
    random.shuffle(rois)
        
    fig1, ax1 = plt.subplots(ncols=3, nrows=2,figsize=(10, 5))
    axs = ax1.flatten()
    
    for idx, ax in enumerate(axs):
        if idx == number:
            break
        try:
            aa = rois_to_plot[idx]
        except IndexError:
            break
        ax.plot(rois_to_plot[idx].edge_gauss_profile, '--k',lw=lw,alpha=alpha)
        ax.plot(rois_to_plot[idx].stripe_gauss_profile,'--k',lw=lw,alpha=alpha)
        ax.plot(rois_to_plot[idx].i_edge_resp, label='edge',lw=lw,alpha=alpha,
                color = colors[0])
        ax.plot(rois_to_plot[idx].i_stripe_resp, label='stripe',lw=lw,alpha=alpha,
                color = colors[1])
        limy_one= ax.get_ylim()[1]
        limy_zero = ax.get_ylim()[0]
        ax.plot([np.argmax(rois_to_plot[idx].edge_gauss_profile),
                  np.argmax(rois_to_plot[idx].edge_gauss_profile)],
                 [limy_zero,limy_one],'--',color = colors[0])
        
        ax.plot([np.argmax(rois_to_plot[idx].stripe_gauss_profile),
                  np.argmax(rois_to_plot[idx].stripe_gauss_profile)],
                 [limy_zero,limy_one],'--',color = colors[1])
        ax.set_ylim((limy_zero,limy_one))
        ax.set_xlabel('Screen (circ)')
        ax.set_ylabel('Delta F/F')
       
        
        rsq1=round(rois_to_plot[idx].resp_delay_fits_Rsq[0],2)
        rsq2=round(rois_to_plot[idx].resp_delay_fits_Rsq[1],2)
        ax.set_title(f"Delay: {rois_to_plot[idx].resp_delay_deg}circ Rsq {rsq1},{rsq2} PD {rois_to_plot[idx].PD} CS {roi.CS}")
        ax.legend()
    fig1.tight_layout()
    return fig1
            
def transfer_masks(rois, properties,experiment_info = None, 
                   imaging_info =None,CS=None,transfer_traces=False): # edited by Juan
    """
    Generates new roi instances
    """
    new_rois = []
    for roi in rois:
        if  imaging_info is None:
            new_roi = ROI_bg(roi.mask, experiment_info = experiment_info,
                                    imaging_info=roi.imaging_info)
        else:
            new_roi = ROI_bg(roi.mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info)
       
        new_roi.uniq_id=roi.uniq_id
        if CS != None:
            if not(roi.CS == CS):
                continue
                
        for prop in properties:
            # Note: Copy here is required otherwise it will just assign the pointer
            # which is dangerous if you want to use both rois in a script
            # that uses this function.
            try:
                new_roi.__dict__[prop] = copy.deepcopy(roi.__dict__[prop])
            except KeyError:
                print('Property:-{pr}- not found... Skipping property for this ROI\n'.format(pr=prop))
                continue
        if transfer_traces==True:
            prevstim_data={'prev_resp_traces':roi.resp_trace_all_epochs,'stim_info':roi.stim_info,
                            'extraction_params':roi.extraction_params}
            new_roi.prevstim_data=prevstim_data
        new_rois.append(new_roi)
    print('ROI transfer successful.')
    return new_rois
        
def interpolate_signal(signal, sampling_rate, int_rate,stim_duration = 10,new_time = None):
    """
    """

    #juan: corrected interpolation
    

    period=1/float(sampling_rate)
    timeV=  np.linspace(0,(len(signal)+1)*period,num=len(signal))#(len(signal)+1)*period
    #timeV = np.linspace(0,len(signal),len(signal))
    # Create an interpolated time vector in the desired interpolation rate
    #timeVI = np.linspace(0,len(signal),int(np.ceil((len(signal)/sampling_rate)*int_rate)))
        
    if new_time is None:
        steps = (len(signal)+1)*period*int_rate
        new_time=np.linspace(1/float(int_rate),stim_duration,int(steps)) #careful if you change int_rate. slightly hardcoded line


    return new_time,np.interp(new_time, timeV, signal)
    
def conc_traces(rois, interpolation = True, int_rate = 10,df_method=None):
    """
    Concatanates and interpolates traces.
    
    #edited by Juan: introduced df_method 'postpone' 
    """
    for roi in rois:
        conc_trace = []
        stim_trace = []
        for idx, epoch in enumerate(range(roi.stim_info['EPOCHS'])): #Seb: epochs_number --> EPOCHS
            curr_stim = np.zeros((1,len(roi.whole_trace_all_epochs[epoch])))[0]
            #curr_stim = np.zeros((1,len(roi.whole_trace_all_epochs[epoch])))[0]
 
            curr_stim = curr_stim + idx
            stim_trace=np.append(stim_trace,curr_stim,axis=0)
            conc_trace=np.append(conc_trace,roi.whole_trace_all_epochs[epoch],axis=0) ####HERE
            #conc_trace=np.append(conc_trace,roi.whole_trace_all_epochs[epoch],axis=0) ####HERE


        roi.conc_trace = conc_trace
        roi.stim_trace = stim_trace
        
        # if df_method=='postpone': # warning: buggy. this applies only to Full field flash stimulation for now
        #     if roi.experiment_info['expected_polarity']=='OFF':
        #         roi.conc_trace= (roi.conc_trace - np.mean(roi.conc_trace[len(roi.conc_trace)/2:-2]))/np.mean(roi.conc_trace[len(roi.conc_trace)/2:-2])
        #     elif roi.experiment_info['expected_polarity']=='ON':
        #         roi.conc_trace= (roi.conc_trace - np.mean(roi.conc_trace[0:len(roi.conc_trace)/2-2]))/np.mean(roi.conc_trace[0:len(roi.conc_trace)/2-2])


        # Calculating correlation
        curr_coeff, pval = pearsonr(roi.conc_trace,roi.stim_trace)
        roi.corr_fff = curr_coeff
        roi.corr_pval = pval
        if interpolation:
            roi.int_con_trace = interpolate_signal(roi.conc_trace, 
                                                   roi.imaging_info['frame_rate'], 
                                                   int_rate)
            roi.int_stim_trace = interpolate_signal(stim_trace, 
                                                    roi.imaging_info['frame_rate'], 
                                                   int_rate)
            roi.int_rate = int_rate
            
    return rois

def calculate_correlation(rois,stim_type = None):
    """
    Calculate pearson's correlation between responses and stimulus.
    
    """
    
    
    for roi in rois:
        if stim_type == '11LuminanceSteps':
            stim_to_correlate = roi.luminances
            resp = np.mean(roi.lum_resp_traces_interpolated,axis=1)
        elif stim_type == 'AB_steps':
            stim_to_correlate = roi.epoch_contrast_A_steps[1:]
            resp = roi.a_step_responses[1:]
            
            base_mean = roi.resp_traces_interpolated.mean(axis=0)[:50].mean()
            resp_mean = roi.resp_traces_interpolated.mean(axis=0)[70:100].mean()
            
            roi.mean_base_resp_diff = resp_mean - base_mean
        elif stim_type =='LuminanceEdges':
            fr = roi.imaging_info['frame_rate']
            stim_to_correlate = roi.luminances
            resp = []
            for trace in roi.resp_trace_all_epochs.values():
                resp.append(trace[int(-fr*4):].max())
            
        else:
            raise NameError('Stimulus type not found.')
            

        # Calculating correlation
        curr_coeff, pval = pearsonr(resp,stim_to_correlate)
        roi.correlation = curr_coeff
        roi.corr_pval = pval
        
    return rois

def find_inverted(rois,stim_type = None):
    """
    Calculate pearson's correlation between responses and stimulus.
    
    """
    
    
    for roi in rois:
        if stim_type == '1Hz_luminance_gratings':
            fr = roi.imaging_info['frame_rate']
            baseline_frames_total = roi.stim_info['baseline_duration'] * fr
            baseline_frames_needed = int(baseline_frames_total-(roi.stim_info['baseline_duration']/2.0 * fr))
            
            baseline_m = roi.whole_trace_all_epochs[1][baseline_frames_needed:int(baseline_frames_total)].mean()
            
            response_m = roi.resp_trace_all_epochs[1].mean()
            
            diff = response_m - baseline_m
            if diff>0:
                roi.inverted = 0
            else:
                roi.inverted = 1
        
                
            
        else:
            raise NameError('Stimulus type not found.')
            

        
        
    return rois



def transfer_props(roi, data_roi,transfer_props):
    """ 
    Transfers data from one roi to another one.
    """ 
    for prop in transfer_props:
        roi.__dict__[prop] = data_roi.__dict__[prop]
    return roi

def create_STF_maps(rois):
    
    """ Creates the spatiotemporal frequency maps """
    
    
    for roi in rois:
        roi_dict = {}
        pd_epochs = roi.stim_info['epoch_dir'][1:]==roi.PD
        roi_dict['SF'] = \
            np.array(roi.stim_info['input_data']['Stimulus.spacing'][1:]).astype(float)[pd_epochs]
        roi_dict['TF'] = roi.stim_info['epoch_frequency'][1:][pd_epochs]
            
        
        roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[1:]))[pd_epochs]
        
        df_roi = pd.DataFrame.from_dict(roi_dict)
        stf_map = df_roi.pivot(index='TF',columns='SF')
        roi.stf_map= stf_map
        roi.stf_map_norm=(stf_map-stf_map.mean())/stf_map.std()
        
        roi.BF = roi.stim_info['epoch_frequency'][roi.max_resp_idx]
    return rois
        
def plot_stf_map(roi,rois_df,plot_x='reliability',plot_y='CSI'):
    
    curr_roi_props = rois_df[rois_df['uniq_id']==roi.uniq_id]

    plt.close('all')
    from post_analysis_core import run_matplotlib_params
    # Constructing the plot backbone, selecting colors
    colors = run_matplotlib_params()
    try:
        CS = roi.CS
    except AttributeError :
        CS = ''
        color = colors[2]
    if roi.CS =='OFF':
        color = colors[3]
    else:
        color = colors[2]
    fig = plt.figure(figsize=(9, 12))
    
    # Mask
    plt.subplot(221)
    roi.showRoiMask(cmap='PiYG')
    plt.title('%s PD: %d CS: %s' % (roi.uniq_id,roi.PD,roi.CS))
    
    plt.subplot(222)
    sns.scatterplot(x=plot_x, y=plot_y,alpha=.8,color='grey',
                    data =rois_df,legend=False,size=20)
    sns.scatterplot(x=plot_x, y=plot_y,color=color,
                    data =curr_roi_props,legend=False,size=20)
    
    plt.xlim(0, rois_df[plot_x].max()+0.3)
    plt.ylim(0, rois_df[plot_y].max()+0.3)
    
    
    # Constructing the plot backbone
    plt.subplot(223)
    plt.title('STF map')
    ax=sns.heatmap(roi.stf_map, cmap='coolwarm',center=0,
                   xticklabels=np.array(roi.stf_map.columns.levels[1]).astype(int),
                   yticklabels=np.array(roi.stf_map.index),
                   cbar_kws={'label': 'Delta F/F'})
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.subplot(224)
    plt.title('STF map normalized')
    ax1=sns.heatmap(roi.stf_map_norm, cmap='coolwarm',center=0,
                   xticklabels=np.array(roi.stf_map_norm.columns.levels[1]).astype(int),
                   yticklabels=np.array(roi.stf_map_norm.index),
                   cbar_kws={'label': 'zscore'})
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    fig.tight_layout()
    
    return fig
def analyze_lum_Cont_speed_edges(rois,variab,analys_type,reliability_filter=0,int_rate = 11):
    
    #epoch_dirs = rois[0].stim_info['epoch_dir']
    # epoch_dirs_no_base= \
    #     np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']

    if variab == 'luminance':
        #epoch_file_l = np.array(rois[0].stim_info['input_data']['lum'],float)
        #epoch_file_c = np.array(rois[0].stim_info['input_data']['contrast'],float)
        x_variab= rois[0].stim_info['bg'] # for this stim luminance is encoded in bg rather than fg
    elif variab == 'contrast':
        luminances=np.concatenate([np.array(rois[0].stim_info['fg'])[np.newaxis,:],np.array(rois[0].stim_info['bg'])[np.newaxis,:]],axis=0)
        x_variab= (np.nanmax(luminances,axis=0)-np.nanmin(luminances,axis=0))/(np.nanmax(luminances,axis=0)+np.nanmin(luminances,axis=0)) # michelson contrast calculation
    elif variab == 'velocity':
        x_variab= rois[0].stim_info['velocity']
    # if 'OFF' in rois[0].stim_name:
    #     epoch_luminances= epoch_file_l * (1-epoch_file_c)
    # elif 'ON' in rois[0].stim_name:
    #     epoch_luminances= epoch_file_l * (1+epoch_file_c)
        
    for roi in rois:

        
        # It needs to handle multiple types of same stimulus 
        if not('1-dir' in analys_type):
            # epoch_dirs = rois[0].stim_info['epoch_dir']
            # epoch_dirs_no_base= \
            # np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
            # curr_pref_dir = \
            #     np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
            # req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
            raise Exception ('case not implemented')
        else:
            try:
                reliability_val=roi.cycle1_reliability['reliability_PD_ON']
                roi.independent_var=variab
                roi.independent_var_vals=x_variab

            except:
                raise Exception ('no reliability value from cycle0 available for filtering and performing analysis')
            req_epochs = np.where(np.array(rois[0].stim_info['stimtype'])=='driftingstripe')[0]
            if len(req_epochs)==0:
                req_epochs= np.where(np.array(rois[0].stim_info['stimtype'])=='ADS')[0]
            # if reliability_val>reliability_filter:
            #     roi.rejected=False                
            # else:
            #     roi.rejected=True
            #     continue
            # if len(np.unique(roi.stim_info['angle']))!=1:
            #     raise Exception('more than one direction in stim')
            # else:
            #     if np.unique(roi.stim_info['angle'])[0]!= roi.dir_max_resp:
            #         roi.rejected=True
            #         continue
            #     elif np.unique(roi.stim_info['angle'])[0]== roi.dir_max_resp:
            #         roi.rejected=False
            # if roi.Center_position_filter==True:
            #     roi.rejected=False
            # else:
            #     roi.rejected=True
            #     continue
            roi.interpolated_traces_epochs={} # this dict is to be filled up with mean traces for the epoch 
            roi.interpolated_time={}
            
            for idx,  epoch in enumerate(req_epochs):
                    stim_duration=roi.stim_info['duration'][epoch]
                    roi.interpolated_time[epoch],roi.interpolated_traces_epochs[epoch]=\
                            interpolate_signal(roi.resp_trace_all_epochs[epoch],
                            roi.imaging_info['frame_rate'],
                            int_rate,stim_duration=stim_duration)
                    
        # after interpolation, align the interpolated traces for future plotting        
    align_traces(rois) 
   
        
    return rois

def align_traces_per_epoch(rois,key,grating):
    # Get the first array
    check=False
    passed_ROIs=[]
    for roi in rois:
        # if roi.rejected==False: PRADEEP
        passed_ROIs.append(roi)
    if grating==False:
        if key==1:
            'flag'
        for roi in passed_ROIs:
            ref_array = copy.deepcopy(roi.interpolated_traces_epochs[key])
            max_index = np.argmax(ref_array) 
            if max_index<(len(ref_array)/3)-1 or max_index>(len(ref_array)*2/3)-1 or check==True:
                pass
            else:
                # Shift the reference array so that its maximum value is in the middle
                shift_ref = len(ref_array) // 2 - max_index
                ref_array = np.roll(ref_array, shift_ref)
                check=True
                break
        if check==False:
            for roi in passed_ROIs:
                limit=int(np.ceil(0.45*roi.imaging_info['frame_rate']))
                ref_array = copy.deepcopy(roi.interpolated_traces_epochs[key])
                max_index = np.argmax(ref_array) 
                if max_index>len(ref_array)-limit or max_index<limit or check==True:
                    pass
                else:
                    # Shift the reference array so that its maximum value is in the middle
                    shift_ref = len(ref_array) // 2 - max_index
                    ref_array = np.roll(ref_array, shift_ref)
                    check=True
                    break   

    else: 
        ref_array = copy.deepcopy(passed_ROIs[0].interpolated_traces_epochs[key])
    len_array=[]
    aligned_arrays = []
    for roi in passed_ROIs: # code for producing an array with aligned traces
        len_array.append(len(roi.interpolated_traces_epochs[key]))
    ref_array=ref_array[:np.min(np.array(len_array))]
    for roi in passed_ROIs:
        array = copy.deepcopy(roi.interpolated_traces_epochs[key][:np.min(np.array(len_array))])
        shift = np.argmax(scipy.signal.correlate(ref_array, array,mode='same')) 
        
        # Create a new array filled with nan
        #aligned_array = np.full_like(array, np.nan)

        # Shift the array and place it into the new array
        rolled_array = np.roll(array, shift - len(array) // 2)
        #start = max(0, -shift + len(array) // 2)
        #end = min(len(array), len(array) - shift + len(array) // 2)
        #aligned_array [start:end]= rolled_array[start:end]
        roi.interpolated_traces_epochs[key] = rolled_array
    #return np.array(aligned_arrays)

def align_traces(rois,grating=False):

    # Get the keys from the first ROI
    keys = rois[0].resp_trace_all_epochs.keys()

    
    for key in keys:
       align_traces_per_epoch(rois, key,grating)

    #return aligned
def analyze_A_B_step(rois,int_rate = 10):
    
    
    
    epoch_file_l = np.array(rois[0].stim_info['input_data']['lum'],float)
    epoch_file_c = np.array(rois[0].stim_info['input_data']['contrast'],float)
    
    epoch_lum_BG = epoch_file_l[0]
    epoch_luminance_A_steps = epoch_file_l * (1+epoch_file_c)
    epoch_luminance_B_steps = epoch_file_l * (1-epoch_file_c)
    epoch_contrast_A_steps = \
        (epoch_luminance_A_steps -epoch_lum_BG)/epoch_lum_BG
    epoch_contrast_B_steps = \
        (epoch_luminance_B_steps -epoch_luminance_A_steps)/epoch_luminance_A_steps
        
    epoch_contrast_B_steps_BGweber = \
        (epoch_luminance_B_steps -epoch_luminance_A_steps)/epoch_lum_BG
    
    for roi in rois:
        roi.epoch_lum_BG = epoch_lum_BG
        roi.epoch_luminance_A_steps = epoch_luminance_A_steps
        roi.epoch_luminance_B_steps = epoch_luminance_B_steps
        roi.epoch_contrast_A_steps = epoch_contrast_A_steps
        roi.epoch_contrast_B_steps = epoch_contrast_B_steps
        roi.epoch_contrast_B_steps_BGweber = epoch_contrast_B_steps_BGweber
                        
        min_len = np.array(map(len, roi.whole_trace_all_epochs.values())).min()
        traces = np.zeros((len(roi.whole_trace_all_epochs),min_len))
        ex_trace = roi.whole_trace_all_epochs[1][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((len(roi.whole_trace_all_epochs),int_len))
        mat_idx = 0
        
        roi.a_step_responses = np.full(np.shape(epoch_luminance_A_steps),np.nan)
        roi.b_step_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)
        
        roi.a_step_baseline_responses = np.full(np.shape(epoch_luminance_A_steps),np.nan)
        roi.b_step_baseline_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)
        
        roi.a_to_b_step_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)
        
        roi.a_step_responses = np.full(np.shape(epoch_luminance_A_steps),np.nan)
        roi.b_step_responses = np.full(np.shape(epoch_luminance_B_steps),np.nan)

        for epoch in roi.whole_trace_all_epochs:
            whole_t = roi.whole_trace_all_epochs[epoch][:min_len]
            resp_len = len(roi.resp_trace_all_epochs[epoch])
            base_len = (len(roi.whole_trace_all_epochs[epoch])-resp_len)/2
            base_t = \
                whole_t[base_len-(int(2*roi.imaging_info['frame_rate'])):base_len]
            base_activity = np.mean(base_t)
            resp_t = roi.resp_trace_all_epochs[epoch]
            A_t = resp_t[:int(len(resp_t)/2)]
            B_t = resp_t[int(len(resp_t)/2):]
            
            roi.a_step_responses[epoch] = np.max(A_t) - base_activity
            roi.b_step_responses[epoch] = np.max(B_t) - base_activity
            
            
            late_a = int(0.50 * roi.imaging_info['frame_rate'])
            early_b = int(1 * roi.imaging_info['frame_rate'])
            roi.a_step_baseline_responses[epoch] = \
                np.nanmean(A_t[-late_a:]) - base_activity
            roi.b_step_baseline_responses[epoch] = \
                np.nanmean(B_t[-late_a:]) - base_activity
                
            roi.a_to_b_step_responses[epoch] = np.nanmax(B_t[:early_b]) - np.nanmean(A_t[-late_a:])
            
            traces[mat_idx,:] = whole_t
            
            int_traces[mat_idx,:] = interpolate_signal(whole_t, 
                                     roi.imaging_info['frame_rate'],int_rate)
            mat_idx +=1
            
        roi.int_rate = int_rate
        roi.resp_traces = traces
        roi.resp_traces_interpolated = int_traces
        roi.AB_resp_max = traces.max()
        
        
    return rois

def sep_trial_compute_df(rois,df_method=None,df_base_dur=None,
                         max_resp_trial_len='max',filtering=False,
                         cf = 1):
  
    
    for roi in rois:
        fps = roi.imaging_info['frame_rate']
        if filtering:
            filtered = low_pass(roi.raw_trace.copy(), fps, 
                                        crit_freq=cf,plot=False)
            roi.raw_trace = filtered
        
        frameRate = roi.imaging_info['frame_rate']
        trialCoor = roi.stim_info['trial_coordinates']
        roi.whole_trace_all_epochs = {}
        roi.resp_trace_all_epochs = {}
        roi.base_dur = []
        # Trial averaging by loooping through epochs and trials
        for iEpoch in trialCoor:
            currentEpoch = trialCoor[iEpoch]
            current_epoch_dur = roi.stim_info['epochs_duration'][iEpoch]
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
                
       
            # Baseline epoch is presented only when random value = 0 and 1 
            if roi.stim_info['random'] == 1:
                wholeTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs = np.zeros(shape=(base_len,
                                                         trial_numbers))
            elif roi.stim_info['random'] == 0:
                wholeTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [roi.stim_info['baseline_epoch']])[0]
                baselineTraces_allTrials_ROIs = \
                    np.zeros(shape=(int(frameRate*1.5),trial_numbers))
            else:
                wholeTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs = None
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if roi.stim_info['random'] == 1:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    
                    
                    if df_method == None:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                        
                    elif df_method == 'Baseline_epoch':
                        base_f_len = int(frameRate * df_base_dur)
                        base_resp = roi.raw_trace[baselineEnd-base_f_len:baselineEnd].mean()
                        roi_whole_trace = (roi.raw_trace[trialStart:trialEnd] - base_resp)/base_resp
                        roi_resp = (roi.raw_trace[respStart:epochEnd] - base_resp)/base_resp
                    
                    try:
                        wholeTraces_allTrials_ROIs[:,trial_num]= roi_whole_trace[:trial_len]
                    except ValueError:
                        new_trace = np.full((trial_len,),np.nan)
                        new_trace[:len(roi_whole_trace)] = roi_whole_trace.copy()
                        wholeTraces_allTrials_ROIs[:,trial_num]= new_trace
                            
                    respTraces_allTrials_ROIs[:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[:,trial_num]= roi_whole_trace[:base_len]
                elif roi.stim_info['random'] == 0:
                    a=1 
                    #TODO: FIX HERE
            wt = np.nanmean(wholeTraces_allTrials_ROIs,axis=1)
            roi.base_dur.append(df_base_dur)
            roi.appendTrace(wt,iEpoch, trace_type = 'whole')
            roi.appendTrace(np.nanmean(respTraces_allTrials_ROIs,axis=1),
                              iEpoch, trace_type = 'response' )
    return rois
        

def analyze_luminance_steps(rois,int_rate = 10):
    
    
    try:
        epoch_file_l = np.array(rois[0].stim_info['input_data']['lum'],float)
        epoch_file_c = np.array(rois[0].stim_info['input_data']['contrast'],float)
        epoch_luminances= epoch_file_l * (1-epoch_file_c)
    except:
        epoch_luminances= np.array(rois[0].stim_info['input_data']['bg'],float)
        
    for roi in rois:
        roi.luminances = epoch_luminances
        min_len = np.array(map(len, roi.resp_trace_all_epochs.values())).min()
        traces = np.zeros((len(roi.resp_trace_all_epochs),min_len))
        ex_trace = roi.resp_trace_all_epochs[0][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((len(roi.resp_trace_all_epochs),int_len))
        mat_idx = 0
        
        
        for epoch in roi.resp_trace_all_epochs:
            curr_trace = roi.resp_trace_all_epochs[epoch][:min_len]
            fps = roi.imaging_info['frame_rate']
            curr_trace = low_pass(curr_trace, fps, crit_freq=1,plot=False)
            traces[mat_idx,:] = curr_trace
            
            int_traces[mat_idx,:] = interpolate_signal(curr_trace, 
                                     roi.imaging_info['frame_rate'],int_rate)
            mat_idx +=1
            
        roi.int_rate = int_rate
        roi.lum_resp_traces = traces
        roi.lum_resp_traces_interpolated = int_traces
        roi.lum_resp_max = traces.max()
        
        
    return rois


def analyze_luminance_gratings(rois):
    
    # Seb: if this variable does NOT exist in the stim file, made it 0 (= one single direction)
    if 'epoch_dir' in rois[0].stim_info:
        epoch_dirs = rois[0].stim_info['epoch_dir']
    else:
        epoch_dirs = np.ndarray.tolist(np.zeros(rois[0].stim_info['EPOCHS']))

    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']
    epoch_luminances= np.array(rois[0].stim_info['input_data']['lum'],float)
        
    for roi in rois:
        roi_dict = {}
        
        #Seb commented this out.
        # if not('1D' in roi.stim_name):
        #     curr_pref_dir = \
        #         np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
        #     req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        # else:
        #     req_epochs = (epoch_types != 11)
        req_epochs = [e == rois[0].stim_info['stimtype'][-1] for e in epoch_types] # Seb: selecting epochs of interest based on the name of the last epoch in the stimulus input file
        if rois[0].stim_info['stimtype'][0] != rois[0].stim_info['stimtype'][1]:
            req_epochs[0] = False
        roi_dict['luminance'] = epoch_luminances[req_epochs]
        # Seb: if this variable does NOT exist in the stim file, create it from others
        if 'epoch_frequency' in rois[0].stim_info:
            roi_dict['TF'] = roi.stim_info['epoch_frequency'][req_epochs]
            
        else:
            temp_TF_list = []
            for i, value in enumerate(roi.stim_info['velocity']):
                if value == 0:
                    temp_TF_list.append(0)
                else:
                    temp_TF = value/roi.stim_info['sWavelength'][i]
                    temp_TF_list.append(temp_TF)

            temp_TF_list = np.array(temp_TF_list)
            roi.stim_info['epoch_frequency'] = temp_TF_list
            roi_dict['TF'] = temp_TF_list[req_epochs]
            
            
        roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[req_epochs]))
        
        df_roi = pd.DataFrame.from_dict(roi_dict)
        
        
        # tfl_map = df_roi.pivot(index='TF',columns='luminance')
        # roi.tfl_map= tfl_map
        # roi.tfl_map_norm=(tfl_map-tfl_map.mean())/tfl_map.std()
        
        
        roi.tfl_map= df_roi
        
        
        roi.BF = roi.stim_info['epoch_frequency'][roi.max_resp_idx] 
        
        
        
        conc_trace = []
        for epoch in np.argwhere((roi.stim_info['epoch_frequency'] == 1))[1:]:
            
            conc_trace=np.append(conc_trace,
                                 roi.whole_trace_all_epochs[float(epoch)],axis=0)
        roi.oneHz_conc_resp = conc_trace
        
    return rois

def analyze_luminance_gratings_1Hz(rois,int_rate = 10):
    
    # Seb: if this variable does NOT exist in the stim file, made it 0 (= one single direction)
    if 'epoch_dir' in rois[0].stim_info:
        epoch_dirs = rois[0].stim_info['epoch_dir']
    else:
        epoch_dirs = np.ndarray.tolist(np.zeros(rois[0].stim_info['EPOCHS']))
    
    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']
    epoch_luminances= np.array(rois[0].stim_info['input_data']['lum'],float)
        
    for roi in rois:

        # Seb: commented this out
        # if not('1D' in roi.stim_name):
        #     curr_pref_dir = \
        #         np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
        #     req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        # else:
        #     req_epochs = (epoch_types != 11)

        req_epochs = [e == rois[0].stim_info['stimtype'][-1] for e in epoch_types] # Seb: selecting epochs of interest based on the name of the last epoch in the stimulus input file
        if rois[0].stim_info['stimtype'][0] == rois[0].stim_info['stimtype'][1]:
            req_epochs[0] = False

        
        roi.luminances = epoch_luminances[req_epochs]            
        conc_trace = []
        roi.power_at_hz = np.zeros_like(roi.luminances)
        roi.base_power = np.zeros_like(roi.luminances)
        roi.baselines = np.zeros_like(roi.luminances)
        fr = roi.imaging_info['frame_rate']
        
        
        min_len = np.array(map(len, roi.resp_trace_all_epochs.values())).min()
        traces = np.zeros((np.sum(req_epochs),min_len))
        
        ex_trace = roi.resp_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len]
        int_len = len(interpolate_signal(ex_trace, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces =np.zeros((np.sum(req_epochs),int_len))
        
        
        min_len_wholeT = np.array(map(len, roi.whole_trace_all_epochs.values())).min()
        traces_wholeT = np.zeros((np.sum(req_epochs),min_len_wholeT))
        
        ex_trace_wholeT = roi.whole_trace_all_epochs[np.where(req_epochs)[0][0]][:min_len_wholeT]
        int_len_wholeT = len(interpolate_signal(ex_trace_wholeT, 
                                     roi.imaging_info['frame_rate'],int_rate))
        int_traces_wholeT =np.zeros((np.sum(req_epochs),int_len_wholeT))
        mat_idx = 0
        for idx,epoch in enumerate(np.where(req_epochs)[0]):
            curr_freq = roi.stim_info['epoch_frequency'][epoch]
            curr_resp = roi.resp_trace_all_epochs[epoch]
            
            two_sec = int(2 * roi.imaging_info['frame_rate'])
            curr_whole = roi.whole_trace_all_epochs[epoch][two_sec:-1-two_sec]
            bg_resp = roi.whole_trace_all_epochs[epoch][two_sec:-two_sec+int(roi.imaging_info['frame_rate'])]
            bg_resp_mean = bg_resp.mean()
            roi.baselines[idx] = curr_resp[int(fr):].mean()-bg_resp_mean
            
            # Fourier analysis of baseline responses
            N = len(curr_whole)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_whole)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_whole-curr_whole.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=roi.luminances[idx]))
            # plt.legend()
            base_p = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs(xf-(1.0/6)))
            roi.base_power[idx] = base_p[req_idx]
            
            
            # Fourier analysis of sinusodial responses
            N = len(curr_resp)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_resp)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_resp-curr_resp.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=roi.luminances[idx]))
            # plt.legend()
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs(xf-curr_freq))
            roi.power_at_hz[idx] = power[req_idx]
            
            # Concatenate trace
            conc_trace=np.append(conc_trace,
                                 roi.whole_trace_all_epochs[float(epoch)][two_sec:],axis=0)
            
            # Interpolation
            curr_trace = roi.resp_trace_all_epochs[epoch][:min_len]
            traces[mat_idx,:] = curr_trace
            int_traces[mat_idx,:] = interpolate_signal(curr_trace, 
                                 roi.imaging_info['frame_rate'],int_rate)
            
            curr_trace_wt = roi.whole_trace_all_epochs[epoch][:min_len_wholeT]
            traces_wholeT[mat_idx,:] = curr_trace_wt
            int_traces_wholeT[mat_idx,:] = interpolate_signal(curr_trace_wt, 
                                 roi.imaging_info['frame_rate'],int_rate)
            
            mat_idx +=1
            
       
        roi.int_rate = int_rate
        roi.grating_resp_traces = traces
        roi.grating_resp_traces_interpolated = int_traces
        
        roi.grating_whole_traces = traces_wholeT
        roi.grating_whole_traces_interpolated = int_traces_wholeT
             
        # plt.legend()
        # plt.title(roi.experiment_info['Genotype'])
        # plt.xlabel('Hz')
        # plt.ylabel('Signal')
        # plt.waitforbuttonpress()
        # plt.close('all')
        roi.conc_resp = conc_trace
        
        X = roi.luminances
        Y = roi.power_at_hz
        Z = roi.base_power
        
        roi.slope = linregress(X, np.transpose(Y))[0]
        roi.basePower_slope = linregress(X, np.transpose(Z))[0]
        roi.base_slope = linregress(X, np.transpose(roi.baselines))[0]
        
    return rois

def analyze_luminance_freq_gratings(rois):
    '''CURRENTLY NOT WORKING '''
    
    
    epoch_dirs = rois[0].stim_info['epoch_dir']
    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']
    epoch_luminances= np.array(rois[0].stim_info['input_data']['lum'],float)
        
    for roi in rois:

        
        if not('1D' in roi.stim_name):
            curr_pref_dir = \
                np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
            req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        else:
            req_epochs = (epoch_types != 11)
            
        roi_dict = {}
        roi_dict['luminance'] = epoch_luminances[req_epochs]
        roi_dict['TF'] = roi.stim_info['epoch_frequency'][req_epochs]
        
        roi.luminances = epoch_luminances[req_epochs]            
        
        roi.power_at_hz = np.zeros_like(roi.luminances)
        for idx,epoch in enumerate(np.where(req_epochs)[0]):
            curr_freq = roi.stim_info['epoch_frequency'][epoch]
            if curr_freq > roi.imaging_info['frame_rate']//2:
                roi.power_at_hz[idx] = 0
                continue
                
            curr_resp = roi.resp_trace_all_epochs[epoch]
            
            # Fourier analysis
            N = len(curr_resp)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_resp)
            
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft(curr_resp*w)
            
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs(xf-curr_freq))
            roi.power_at_hz[idx] = power[req_idx]
            
            # Concatenate trace
        
        roi_dict['power'] = roi.power_at_hz[idx]
        df_roi = pd.DataFrame.from_dict(roi_dict)
        tfl_map = df_roi.pivot(index='TF',columns='luminance')
        roi.tfl_map= tfl_map
        
        roi.BF = roi.stim_info['epoch_frequency'][req_epochs][np.argmax(roi.power_at_hz)]
        
        
        conc_trace = []
        for epoch in np.argwhere((roi.stim_info['epoch_frequency'] == 1)):
            
            conc_trace=np.append(conc_trace,
                                 roi.whole_trace_all_epochs[float(epoch)],axis=0)
        roi.oneHz_conc_resp = conc_trace
        
        X = roi.luminances
        # Y = roi.power_at_hz[]
        
        roi.slope = linregress(X, np.transpose(Y))[0]
        
    return rois

def analyze_lum_con_gratings(rois):
    
    epoch_luminances = np.array(rois[0].stim_info['input_data']['lum'],float)
    epoch_contrasts = np.array(rois[0].stim_info['input_data']['contrast'],float)
    
    
    for roi in rois:
        roi_dict = {}
        
        roi_dict['luminance'] = epoch_luminances[1:]
        roi_dict['contrast'] = epoch_contrasts[1:]
            
        
        roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[1:]))
        
        df_roi = pd.DataFrame.from_dict(roi_dict)
        cl_map = df_roi.pivot(index='contrast',columns='luminance')
        roi.cl_map= cl_map
        roi.cl_map_norm=(cl_map-cl_map.mean())/cl_map.std()
                
        
    return rois

# def multiple_direction_gratings(rois):

#     epoch_luminances = np.array(rois[0].stim_info['input_data']['lum'],float)
#     epoch_contrasts = np.array(rois[0].stim_info['input_data']['contrast'],float)
#     epoch_directions = np.array(rois[0].stim_info['input_data']['angle'],float)
    
#     for roi in rois:
#         roi_dict = {}
        
#         roi_dict['luminance'] = epoch_luminances[1:]
#         roi_dict['contrast'] = epoch_contrasts[1:]
            
        
#         roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[1:]))
        
#         df_roi = pd.DataFrame.from_dict(roi_dict)
#         cl_map = df_roi.pivot(index='contrast',columns='luminance')
#         roi.cl_map= cl_map
#         roi.cl_map_norm=(cl_map-cl_map.mean())/cl_map.std()
                
        
#     return rois

def import_png_snippet(png_list,indices):
    
    im0 = imageio.imread(png_list[0])
    snippet = np.zeros(im0[:,:,0].shape)
    snippet = np.repeat(snippet[np.newaxis,:,:],len(indices),axis=0)
    for i,ix in enumerate(indices):
        snippet[i,:,:] = imageio.imread(png_list[ix])[:,:,0]
    snippet = np.repeat(snippet,2,axis=1)
    return snippet

def reverse_correlation_analysis_JF(rois,stimpath,cf=4,highPassfiltering=True,
                                 poly_fitting=False,t_window=1.5,
                                 stim_up_rate=20,test_frames=None,
                                 predict_edges=True,skip_steps = False):
    """ Reverse correlation analysis 
        no interpolation analysis 
        
    """
    
    # find if stimulus is shifted or not
    # stimtype = rois[0].stim_name.split('_')
    # stimtype = list(filter(None, stimtype)) # get rid of empty slices
    # shift = int(stimtype[-1][0])
    # square_size = int(stimtype[-3].split('deg')[0])
    stimtype = rois[0].stim_name.split('.')[0]
    stim_dataframe = pd.DataFrame(rois[0].stim_info['output_data'], columns = ['entry','rel_time','boutInd','epoch','delay_count','delay','stim_frame','mic_frame'])
    stim_dataframe = stim_dataframe[['stim_frame','rel_time','mic_frame','delay']]
    if np.all(stim_dataframe['stim_frame']==0): # some flies in the grating stim have a damaged outfile
        stim_dataframe['stim_frame'] = np.array(list(stim_dataframe.index))
    # get the stimulus information
    #rois[0].stim_info
    #freq = stim_up_rate # The update rate of stimulus frames is 20Hz
    stim_up_rate = 1/float(rois[0].stim_info['duration'][0])
    snippet = int(t_window*stim_up_rate)

    # check if the stimulus manages to cover all the signal length. (in some test the stim can be shorter)
    starting_signal_frame=stim_dataframe.iloc[0]['mic_frame'] 
    # stim_dataframe['mic_frame'] = stim_dataframe['mic_frame'] - starting_signal_frame
    # stim_dataframe['rel_time'] = stim_dataframe['rel_time'] - stim_dataframe['rel_time'].iloc[0]
    
    last_stim_frame_presented = int(np.max(stim_dataframe['stim_frame'])) #int(stim_dataframe.iloc[-1]['stim_frame'])
    print('max_stim_frame %s' %(last_stim_frame_presented))
    
    if 'grating' in stimtype:
        # put the tiff images of a stim into an array
        stim_path = os.chdir(stimpath+'\\pics_grating_noise')
        png_list = glob.glob('*.png')
        im0 = imageio.imread(png_list[0])
        stimulus = None
        stim_up_rate = 60
        len_stim = len(png_list)
        if last_stim_frame_presented>len_stim:  #chop the signal if too long   ---adition for a specific instance231114_f1-->    #or int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])<len(rois[0].white_noise_response):
            signal_end_index = int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])
            
        elif last_stim_frame_presented<len_stim: # chop the stimulus if too long
            signal_end_index=-1
            png_list = png_list[:last_stim_frame_presented]
        else:
            signal_end_index=-1

    else:
        #stim_path = glob.glob(stimpath+'\\stimuli_arrays\\'+ stimtype.split('.')[0] + '.npy')
        #if len(stim_path)>1:
        #    raise Exception('put a single array in the array folder')
        stim_available=False
        
        
        
        #for stim_ in os.listdir(stimpath+'\\stimuli_arrays\\'):
        #    
            
            # if 'random_moving_WN_5degbox_50msUpdate_20degpers' in stim_:
            #     stim_path = glob.glob(stimpath+'\\stimuli_arrays\\'+ stim_)
            #     stim_available=True
            #     break
            # elif 'maxdur' in stim_:
            #     stim_path = glob.glob(stimpath+'\\stimuli_arrays\\'+ stim_)
            #     stim_available=True
                
        png_list = None
        #shift_bool = shift>0
        sampling_rate = rois[0].imaging_info['FramePeriod']
        #jf: read the stimulus array
        #stim_path = glob.glob(stimpath+'\\stimuli_arrays\\*.npy')


        if '01lum' in stimtype:
            stim_lum = 0.1
            #stimtype = stimtype.replace('01lum',"")
        elif '025lum' in stimtype:
            stim_lum = 0.25
            #stimtype = stimtype.replace('025lum',"")
        else:
            stim_lum = 1
        
        if '5degbox' in stimtype:
            stimtype_load = 'whithened_multiplespeeds_mov_random_moving_WN_5degbox_50msUpdate_20degpers'
            stim_size = '5'
        elif '8degbox' in stimtype:
            stimtype_load = 'whithened_multiplespeeds_mov_random_moving_WN_8degbox_50msUpdate_20degpers'
            stim_size = '8'
        elif '10degbox' in stimtype:
            stimtype_load = 'whithened_multiplespeeds_mov_random_moving_WN_10degbox_50msUpdate_20degpers'
            stim_size = '10'
    
        stimtype_load = 'vector_fly_ternary_wn_50ms'
        stim_size = 'NA'
    
        #### temporal fix.... some input arrays 
        stim_path = glob.glob(os.path.join(stimpath,'*.npy'))[0]
        stimulus = np.load(stim_path)
        stimulus = np.repeat(stimulus.reshape(3600,1,1),2,axis=1)
        stimulus = np.repeat(stimulus,2,axis=2)
        stimulus = np.where(stimulus==0,-1,stimulus)
        stimulus = np.where(stimulus==0.5,0,stimulus)
        #stimulus = stimulus.astype('float16')
        # if 'multiplespeeds_mov_random_moving_WN_10degbox_50msUpdate_20degpers' != stimtype: # temporal mod ... the 10 deg stim already incorporates this change
        #     stimulus = np.where(stimulus==0,-1,stimulus)
        # else:
        #     stimulus = np.where(stimulus==0,1,stimulus)

        # check if we can make stim smaller:
        len_stim = len(stimulus)
        if last_stim_frame_presented>len_stim:  #chop the signal if too long   ---adition for a specific instance231114_f1-->    #or int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])<len(rois[0].white_noise_response):
            signal_end_index = int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])
        
        elif last_stim_frame_presented<len_stim: # chop the stimulus if too long
            signal_end_index=-1
        
            stimulus = stimulus[:last_stim_frame_presented,:,:]
        
        else:
            signal_end_index=-1

    #  we should start doing reverse correlation only when enough time has passed such that enpugh stim frames are presented
    fps = rois[0].imaging_info['frame_rate']
    initial_frame = int(stim_dataframe.loc[stim_dataframe['stim_frame']==0].iloc[0]['mic_frame']) #int(np.ceil(t_window*2*rois[0].imaging_info['frame_rate']))

    #initialize variables that would be used for plotting
    stamax = []
    stamin = []
    stamax_y = []
    stamax_x = []
    stamin_y = []
    stamin_x = []
    #hold_out_prediction = True
    # if hold_out_prediction: 
    #     # use 20% of the trace for prediction-testing the RFs
    #     # find the lenght of 20% of trace and pick the indices

    #     if signal_end_index == -1:
    #         len_test_trace = int(0.2*len(range(initial_frame,len(rois[0].white_noise_response))))
    #         available_indices = np.array(range(initial_frame,len(rois[0].white_noise_response)-len_test_trace))
    #     else:
    #         len_test_trace = int(0.2*len(range(initial_frame,signal_end_index)))
    #         available_indices = np.array(range(initial_frame,signal_end_index-len_test_trace))
        

        # np.random.seed(20)
        # test_start_index = np.random.choice(available_indices,len(rois))

        # #temporal change!!!
        # test_start_index = np.repeat([len(rois[0].white_noise_response)-len_test_trace],len(rois))


    # np.random.seed(20)
    # test_start_index = np.random.choice(available_indices,len(rois))
            
    # np.random.seed(20)
    # test_start_index = np.random.choice(available_indices,len(rois))


    # initialize prediction vectors
    train_corr = []
    test_corr = []
    for ix,roi in enumerate(rois):
     
        # initialize prediction vectors
        train_corr = []
        test_corr = []
    for ix,roi in enumerate(rois):

        columns = [
        'train_corr',
        'test_corr',
        'train_corr_M',
        'test_corr_M',
        'roilist',
        'polarity_list',
        'frozen_reliability',
        'id',
        'roi',
        'genotype',
        'treatment',
        'DS',
        'DSI',
        'pref_dir'
        'edges_reliability',
        'center_vect',
        'stim_size',
        'stim_lum',
        'category'
        ]

        # Initialize the DataFrame with the specified columns
        predictions_df = pd.DataFrame(columns=columns)
        plt.close('all')
        #initialize vector map
        vector_map = np.ones_like(stimulus[0,:,:]) 
        vector_map[:] = np.nan
        fig_map = plt.figure()
        gs = GridSpec(1, 1)
        ax_map = fig_map.add_subplot(gs[0])
        color_quiver = {'LPA':'g','LPB': 'b','LPC':'y','LPD':'r'}
        ax_map.imshow(vector_map,extent=[0,240, 0, 240])
    #plt.show()
    #ax_map.set_ylim(240, 0)
    for ix,roi in enumerate(rois):
        if ix < 3 :  #PRADEEP
            continue
        print(stimtype)
        # if ix != 63:
        #     print('skipped')
        #     continue
        # if ix != 29:
        #      continue
        print('roi %s out of %s'%(ix+1,len(rois)))
        if skip_steps == False:
            roi.STRF_data = {}
        
            #print('roi %s out of %s'%(ix+1,len(rois)))

            #filter for reliability:
            #reliab = roi.cycle1_reliability['reliability_PD_ON'] if roi.CS == 'ON' else roi.cycle1_reliability['reliability_PD_OFF']
            
            # if reliab<0.5:
            #     roi.STRF_data['status'] = 'excluded'
            #     print('roi_skipped')
            #     continue
            # # filter for contrast selectivity:
            # # if roi.CSI<0.4:
            # #     roi.STRF_data['status'] = 'excluded'
            # #     print('roi_skipped')
            # #     continue
            # else:
            roi.STRF_data['status'] = 'included'
            
            # keep the trace used for future reference
            
            roi.strf_trace = roi.df_trace#roi.white_noise_response[0:int(signal_end_index)]
                      
            trace = copy.deepcopy(roi.df_trace) 
        
            # filter signal components with period higher than 33 seconds to detrend and eliminate slow oscillations
            
            #trace = High_pass(trace,fps,crit_freq=0.01,plot=False) 
            #trace = low_pass(trace,fps,crit_freq=5,plot=False) #crit_freq in Hz
            #trace = ((trace - np.mean(trace))/np.mean(trace)) #+ roi.imaging_info['baseline_mean']
            trace = trace[:stim_dataframe.iloc[-1]['mic_frame'].astype(int)+1]
            #roi.STRF_data['test_trace'] = trace[test_start_index[ix]:test_start_index[ix] + len_test_trace]
            
            ################################
            ####### record the test indices
            ################################
            # if test_frames is None:
            #     extended_indices = [test_start_index[ix], test_start_index[ix] + len_test_trace]
            #     roi.STRF_data['test_indices'] = [test_start_index[ix], test_start_index[ix] + len_test_trace]
            #     test_fr_bool = False
            # else:
            test_fr_bool = True    
            extended_indices = []
            for pair in test_frames:
                extended_indices.append(np.arange(pair[0],pair[1]))
            test_frames_Squeezed = np.array(extended_indices).flatten()
            roi.STRF_data['test_indices'] = test_frames
            
        
            #if skip_steps == False:
            time_i = time.time()
            roi.STRF_data['strf'], roi.STRF_data['mean_strf'],roi.STRF_data['frozen_mic_frames'] = apply_reverse_correlation(trace,roi,stim_up_rate,stim_dataframe,snippet,initial_frame,stimulus=stimulus,png_list=png_list,frozen_indices=test_frames_Squeezed)#indices=response_idxs
            plt.figure()
            plt.plot(roi.STRF_data['strf'][:,0,0])
            time_j = time.time() - time_i
            print('rev_corr took %s s'%(time_j))
            #_,roi.STRF_data['z_score'],_ = calculate_Zscore_STRF(roi,control=False)
            #roi.STRF_data['robust_z_score'] = calculate_robust_Zscore(roi)
        else:
            print('rev correlation skipped')    
            test_fr_bool = True
            trace = copy.deepcopy(roi.df_trace)
            trace = trace[:stim_dataframe.iloc[-1]['mic_frame'].astype(int)+1]

        # find the center of the RF
        scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1])
        #z_blur = filters.gaussian(roi.STRF_data['robust_z_score'], sigma = int(5/scaling_fact))
        #roi.STRF_data['Zmax_index'] = [np.where(z_blur==np.max(z_blur))[0][0],np.where(z_blur==np.max(z_blur))[1][0]]
        "Pradeep uncommented this"
        # try:                  
        #     roi.center_vect
        #     print('using preexisting center vect')
        #     print(roi.center_vect)
        # except:
        #     print('no preexisting center vect')
        #     roi.center_vect =  roi.STRF_data['Zmax_index'] # copy the center vector to avoid overwriting it
        #     print(roi.center_vect)
        # copy_center_vect = copy.deepcopy(roi.center_vect)
        
        # roi.center_vect_specific = roi.STRF_data['Zmax_index'] # this center vector is calculated for this specific recording
        
        # _,square_mask = apply_square_mask(roi.STRF_data['strf'][0:snippet,:,:],roi,15,indices = roi.center_vect,only_mask=True)
        # d1,d2 = np.where(np.squeeze(square_mask) == 1)
        # restricted_stim = stimulus[:,np.min(d1):np.max(d1),np.min(d2):np.max(d2)]

        # roi.STRF_data['reduced_strf'], roi.STRF_data['reduced_mean_strf'] = apply_reverse_correlation(trace,roi,stim_up_rate,stim_dataframe,snippet,initial_frame,stimulus=restricted_stim,png_list=png_list,extended_indices=test_frames_Squeezed)
        "Pradeep uncommented this"
        shiftpath = os.path.join(stimpath,'RFs',stimtype.split('.')[0])
        try:
            os.mkdir(shiftpath)
        except:
            pass
        savepath=shiftpath+'\\%s'%(roi.experiment_info['FlyID'])

        try:                
            os.mkdir(savepath)
        except: 
            pass
        prediction_path = os.path.join(savepath,'prediction')
        try:
            os.mkdir(prediction_path)
        except:
            pass
        ###

        if 'control' in roi.experiment_info['treatment']:
            if stim_lum != 0.1 and stim_lum !=0.25:
                scramble = True
            else:
                scramble = False
        else:
            scramble = False

        if skip_steps == False:
            if test_fr_bool: # if test_frames is a list, then we have repeated instances of frozen noise
                test_trace, frozen_reliability = Trial_av_frozen_noise(trace,stim_dataframe,test_frames)
                roi.STRF_data['frozen_stim_resp_reliability'] = frozen_reliability
                #roi.center_vect =  roi.STRF_data['Zmax_index'] #[np.where(np.abs(roi.STRF_data['mean_strf'])==np.max(np.abs(roi.STRF_data['mean_strf'])))[0][0],np.where(np.abs(roi.STRF_data['mean_strf'])==np.max(np.abs(roi.STRF_data['mean_strf'])))[1][0]]
                #time0 = time.time()
                #train_, test_ = STRF_response_prediction(roi,ix,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,n_epochs=1,held_out_frames = extended_indices,test_frames = test_frames,test_trace=test_trace,max_index=copy_center_vect)
                #time1 = time.time() - time0
                print('prediction started')
                timea = time.time()
                train_, test_ = optimized_STRF_response_prediction(prediction_path,roi,ix,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,indices=None,held_out_frames = roi.STRF_data['frozen_mic_frames'],test_frames = test_frames,test_trace=test_trace,max_index=copy_center_vect)
                time2 = time.time() - timea

                print('timeprediction_%s' %(time2))

                if scramble==True:
                    print('shuffled RF_prediction started')
                    train_scrambled, test_scrambled,_scrambledim = optimized_STRF_response_prediction(prediction_path,roi,ix,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,held_out_frames = roi.STRF_data['frozen_mic_frames'],test_frames = test_frames,test_trace=test_trace,max_index=copy_center_vect,scramble = True)
                    print('computation_finished')
                print('frozen_reliability %s'%(frozen_reliability))
            else:
                #roi.center_vect = roi.STRF_data['Zmax_index'] #[np.where(np.abs(roi.STRF_data['mean_strf'])==np.max(np.abs(roi.STRF_data['mean_strf'])))[0][0],np.where(np.abs(roi.STRF_data['mean_strf'])==np.max(np.abs(roi.STRF_data['mean_strf'])))[1][0]]
                train_, test_ = optimized_STRF_response_prediction(prediction_path,roi,ix,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,held_out_frames = roi.STRF_data['frozen_mic_frames'],test_trace=None,max_index=copy_center_vect)
                if scramble == True:
                    train_scrambled, test_scrambled = optimized_STRF_response_prediction(prediction_path,roi,ix,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,n_epochs=1,held_out_frames = roi.STRF_data['frozen_mic_frames'],test_frames = test_frames,test_trace=None,max_index=copy_center_vect,scramble = True)
        else:
            print('prediction skipped')
            test_trace, frozen_reliability = Trial_av_frozen_noise(trace,stim_dataframe,test_frames)
            train_, test_ = roi.STRF_data['prediction_corr_train'][0] , roi.STRF_data['prediction_corr_test'][0]
            if scramble==True:
                train_scrambled, test_scrambled = roi.STRF_data['shuffled_prediction_corr_train'][0] , roi.STRF_data['shuffled_prediction_corr_test'][0] 
        
        ##### temporal comment 
#        if predict_edges:
#            predict_edgeResponse_withSTRF(rois,stim_path)


        #train_mean_subtracted = roi.STRF_data['train_prediction'] - np.nanmean(roi.STRF_data['train_prediction'])
        #softplus_params = fit_softplus_non_linearity(roi.STRF_data['train_set_trace'], roi.STRF_data['train_prediction'])
        #mean_subtracted = roi.STRF_data['test_prediction'] - np.nanmean(roi.STRF_data['test_prediction'])
        #roi.STRF_data['rectified_prediction'] = softplus_non_linearity(softplus_params,roi.STRF_data['test_prediction'])
        
        # plot again the rectified prediction vs the real trace 

        row_data = {
            'train_corr': train_,
            'test_corr': test_,
            'train_corr_M': train_scrambled if scramble else None,
            'test_corr_M': test_scrambled if scramble else None,
            'roilist': ix,
            'polarity_list': roi.CS,
            'frozen_reliability' : frozen_reliability,
            'id': roi.experiment_info['FlyID'],
            'roi': ix,
            'genotype': roi.experiment_info['Genotype'],
            'treatment': roi.experiment_info['treatment'],
            'DS' :roi.DSI_ON if roi.CS == 'ON' else roi.DSI_OFF,
            'DSI': roi.CSI,
            'pref_dir': roi.PD_ON if roi.CS == 'ON' else roi.PD_OFF,
            'edges_reliability': roi.cycle1_reliability['reliability_PD'],
            'center_vect': roi.center_vect,
            'stim_lum': stim_lum,
            'stim_size': stim_size,
            'category': roi.category
        }
        
        # Append the row to the DataFrame
        
        predictions_df=predictions_df.append(row_data, ignore_index=True)
        #plot prediction
        predictions_df.to_csv(savepath +'\\%spredictions_df.csv'%(roi.experiment_info['FlyID']))    

        # fig_pred = plt.figure()
        # gs = GridSpec (1,2)
        # ax = fig_pred.add_subplot(gs[0])
        # ax.plot(roi.white_noise_response[roi.STRF_data['test_indices'][0]:roi.STRF_data['test_indices'][1]],color='black')
        # ax.plot(roi.STRF_data['strf_prediction'][roi.STRF_data['test_indices'][0]:roi.STRF_data['test_indices'][1]],color='blue')
        # ax.title.set_text('roi%s zoomed prediction'%(ix))
        
        # ax = fig_pred.add_subplot(gs[1])
        # ax.plot(roi.white_noise_response,color='black')
        # ax.plot(roi.STRF_data['strf_prediction'],color='blue')
        # ax.title.set_text('roi%s correlation %s'%(ix,roi.STRF_data['prediction_corr'][0]))
        # plt.savefig(prediction_path +'\\prediction_roi%s.pdf'%(ix))
        # plt.savefig(prediction_path +'\\prediction_roi%s.jpg'%(ix))
        
        # prediction_df = (roi.STRF_data['strf_prediction']-np.nanmean(roi.STRF_data['strf_prediction']))/np.nanmean(roi.STRF_data['strf_prediction'])

        # fig_pred = plt.figure()
        # gs = GridSpec (1,2)
        # ax = fig_pred.add_subplot(gs[0])
        # ax.plot(roi.white_noise_response[roi.STRF_data['test_indices'][0]:roi.STRF_data['test_indices'][1]],color='black')
        # ax.plot(roi.STRF_data['strf_prediction'][roi.STRF_data['test_indices'][0]:roi.STRF_data['test_indices'][1]],color='blue')
        # ax.title.set_text('roi%s zoomed prediction'%(ix))
        
        # ax = fig_pred.add_subplot(gs[1])
        # ax.plot(roi.white_noise_response,color='black')
        # ax.plot(roi.STRF_data['strf_prediction'],color='blue')
        # ax.title.set_text('roi%s correlation %s'%(ix,roi.STRF_data['prediction_corr'][0]))
        # plt.savefig(prediction_path +'\\prediction_dfed_roi%s.pdf'%(ix))
        # plt.savefig(prediction_path +'\\prediction_dfed_roi%s.jpg'%(ix))
        
        # have a check on how correlations would look like with scrambled traces
        # if 'control' in roi.experiment_info['treatment']:
           
        #         response_idxs = np.where(np.abs(trace)>((np.std(trace)*2)+np.mean(trace)))[0]
        #         trace_shuffled = copy.deepcopy(trace)        
        #         np.random.seed(8)
        #         np.random.shuffle(trace_shuffled)
        #         response_idxs_shuffled = np.where(trace_shuffled>((np.std(trace)*2)+np.mean(trace)))[0]
        #         roi.STRF_data['strf_null'], roi.STRF_data['mean_strf_null'] = apply_reverse_correlation(trace_shuffled,roi,stim_up_rate,stim_dataframe,snippet,initial_frame,stimulus=stimulus,png_list=png_list,test_indices=roi.STRF_data['test_indices'])#,indices=response_idxs_shuffled
        #         _,roi.STRF_data['z_score_null'],_ = calculate_Zscore_STRF(roi,control=True)
        #         #STRF_response_prediction(roi,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,n_epochs=1,control=True,held_out_frames = roi.STRF_data['test_indices'])

        pdir = (roi.PD_ON if roi.CS == 'ON' else roi.PD_OFF) 
        pdircopy = (roi.PD_ON if roi.CS == 'ON' else roi.PD_OFF)
        pdir = np.deg2rad(pdir)
        pdir2 = roi.dir_max_resp
        pdir2 = np.deg2rad(pdir2)

        pdy_map = np.cos(pdir)
        pdx_map = np.sin(pdir)
        scaler = roi.DSI_OFF if roi.CS == 'OFF' else roi.DSI_ON
        vec_len_for_map = int(10.0/scaling_fact) * scaler
        end_vect_map = np.array(np.round(vec_len_for_map*np.array([pdx_map,pdy_map]))).astype(int)
  
        #ax.quiver(roi.center_vect[0], roi.center_vect[1], pdx, pdy, angles='xy', scale_units='xy', scale=vec_len, color='black')
        ###############
        #####plot vector map
        ###############
        try:    
            color_q = color_quiver[roi.category[0]]
        except:
            color_q = 'k'
        #fig, ax = plt.subplots()
        #ax_map.imshow(vector_map,extent=[0,240, 0, 240])
        dSI = roi.DSI_ON if roi.CS == 'ON' else roi.DSI_OFF
        #if roi.strf_data['frozen_stim_resp_reliability']>0.4 and dSI>0.3:
        ax_map.quiver(copy_center_vect[1], 240-copy_center_vect[0], end_vect_map[1], end_vect_map[0],color = color_q,width=0.004,angles='uv')#headaxislength = 3,headlength = 3, headwidth = 2, width = 1)#,color_q,angles = 'x,y') 
        plt.draw()
        plt.savefig(savepath +'\\vector_map.pdf')

        # plot Z_score
        
        fig = plt.figure()    
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])
        imZ = ax.imshow(roi.STRF_data['robust_z_score'],cmap='PuBuGn',vmax=np.max(roi.STRF_data['robust_z_score']),vmin=0)
        cbar=fig.colorbar(imZ, ax=ax)
        plt.savefig(savepath +'\\Zscore_roi%s.jpg'%(ix))
        plt.close(fig)
        fig = plt.figure()    
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])
        imZ = ax.imshow(z_blur,cmap='PuBuGn',vmax=np.max(z_blur),vmin=0)
        cbar=fig.colorbar(imZ, ax=ax)
        plt.savefig(savepath +'\\Zscore_5degstdgaussFilter_roi%s.jpg'%(ix))
        
        # plot STA detail
        plt.close(fig)
        plt.ioff()
        fig=plt.figure()                
        gs = GridSpec(5, 6) 

        for ix2,i in enumerate(range(roi.STRF_data['strf'][:-10,:,:].shape[0])):
            ax = fig.add_subplot(gs[ix2])
            peakval=np.max(np.abs(roi.STRF_data['strf']))
            im = ax.imshow(roi.STRF_data['strf'][i,:,:],cmap='RdBu',vmax=peakval,vmin=-peakval)#, extent = [0,80,0,80])
            ax.set_axis_off()
            cbar=fig.colorbar(im, ax=ax)
            # cbar.ax.yaxis.set_major_formatter(formatter)
            plt.rcParams.update({'font.size' : 8})
            
            cbar.ax.tick_params(labelsize=5)
            
            if ix2 == 0:
                ax.title.set_text('-%ss_%s'%(t_window-i*0.05,roi.category[0]))
            elif ix2 == 1:
                ax.title.set_text('-%ss_%s'%(t_window-i*0.05,roi.CS))
            elif ix2 == 2:
                ax.title.set_text('-%ss_CS%.2f'%(t_window-i*0.05,roi.CSI))
            elif ix2 == 3: 
                ax.title.set_text('-%ss_Zscore%.2f'%(t_window-i*0.05,roi.STRF_data['z_score']))
            elif ix2 == 4:
                 ax.title.set_text('-%ss_Vect_len%.2f'%(t_window-i*0.05,(roi.DSI_ON if roi.CS=='ON' else roi.DSI_OFF)))
            elif ix2 == 5:
                 ax.title.set_text('-%ss_PD%.2f'%(t_window-i*0.05,(roi.PD_ON if roi.CS=='ON' else roi.PD_OFF)))
            elif ix2 == 6:
                 ax.title.set_text('-%ss_MaxRspDir%.2f'%(t_window-i*0.05,(roi.dir_max_resp)))
            elif ix2 == 7:
                 ax.title.set_text('-%ss_corrs%.2f,%.2f'%(t_window-i*0.05,train_, test_))
            else:
                ax.title.set_text('-%ss'%(t_window-i*0.05))
            #fig.suptitle('%s' %(roi.CS))
            #plt.tight_layout()
            
            # if shift_bool:
            #     shiftpath = stimpath+'\\RFs\\shifted%s_%sdeg'%(shift,square_size)
            # else:



        # get the index of the max and min values 
        #index_of_max = [np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[2][0]]
        #index_of_min = [np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[2][0]]
        
        index_of_max = [np.where(roi.STRF_data['mean_strf']==np.max(roi.STRF_data['mean_strf']))[0][0],np.where(roi.STRF_data['mean_strf']==np.max(roi.STRF_data['mean_strf']))[1][0]]
        index_of_min = [np.where(roi.STRF_data['mean_strf']==np.min(roi.STRF_data['mean_strf']))[0][0],np.where(roi.STRF_data['mean_strf']==np.min(roi.STRF_data['mean_strf']))[1][0]]

        #index_of_max = [np.where(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])==np.max(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])))[1][0],np.where(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])==np.max(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])))[2][0]]
        #index_of_min = [np.where(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])==np.min(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])))[1][0],np.where(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])==np.min(apply_circular_mask(roi,20,indices=roi.STRF_data['Zmax_index'])))[2][0]]

        # get the index of the maximum Z-score
        
        #roi.center_vect = roi.STRF_data['Zmax_index']

        # scaling factor for pixel size
        scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1]) # 80 is the size of the screen in degrees
        vec_len = int(15.0/scaling_fact)
        #radius = int(10.0/scaling_fact)
        extend =  int(10/scaling_fact)
        arrow_len = int(20.0/scaling_fact)
        arrow_width = int(5.0/scaling_fact)
        # create prefdir vector (15deg length scaling)
        
        # map the correct polar coordinates of the vector to the x and y components of the strf (the coordinates are rotated in the image presentation, so a correction is needed)
        pdy = np.cos(pdir)
        pdx = -np.sin(pdir)
        pdy2 = np.cos(pdir2)
        pdx2 = -np.sin(pdir2)
       
        # calculate min max values in the neighborhood of the max value

        # if roi.cs == 'ON':
        #     roi.center_vect = index_of_max    
        # else:
        #     roi.center_vect = index_of_min    

        end_vect = np.array(copy_center_vect) + np.array(np.round(vec_len*np.array([pdx,pdy]))).astype(int)
        end_vect2 = np.array(copy_center_vect) + np.array(np.round(vec_len*np.array([pdx2,pdy2]))).astype(int)
        ax.plot([copy_center_vect[1],end_vect[1]],[copy_center_vect[0],end_vect[0]],color='black',linewidth = 1)
        ax.plot([copy_center_vect[1],end_vect2[1]],[copy_center_vect[0],end_vect2[0]],color='blue',linewidth = 1)
        

        #ax.set_xlim(0, 240)
        #ax.set_ylim(0, 240)
        #headaxislength = 3, headlength = 3, headwidth = 2, width = 1)
        
        # ax_map.quiverkey(quiv_plot, X=0.1, Y=1.1, U=10,
        #         label='Quiver key, length = 10', labelpos='E')

        # maxmask = np.nan((roi.STRF_data['strf'].shape[1],roi.STRF_data['strf'].shape[2]))
        #minmask =  np.nan((roi.STRF_data['strf'].shape[1],roi.STRF_data['strf'].shape[2]))
               
        # 1st approach for spatiotemporal representation:
        # this approach consists of stacking the values along the line between max and min value 
        # across time 
        # create an empty sxt image lookalike to test where is the prefdir
        pref_dir_mark = np.array(copy_center_vect) - np.array(np.round(5*np.array([pdx,pdy]))).astype(int)
        pref_dir_mark2 = np.array(copy_center_vect) - np.array(np.round(5*np.array([pdx,pdy]))).astype(int)
        pref_dir_mark3 = np.array(copy_center_vect) - np.array(np.round(1*np.array([pdx,pdy]))).astype(int)
        
        ax.plot([copy_center_vect[1],pref_dir_mark[1]],[copy_center_vect[0],pref_dir_mark[0]],color='red',linewidth = 1)
        plt.draw()
        
        plt.savefig(savepath +'\\STRF_roi%s.jpg'%(ix))
        plt.close(fig)
        stamax.append(np.max(roi.STRF_data['strf']))
        stamin.append(np.min(roi.STRF_data['strf']))
        # STRF_lookalike = np.zeros_like(roi.STRF_data['strf'])
        # STRF_lookalike[:] = 0

        # try:
        #     STRF_lookalike[0:10,pref_dir_mark[0],pref_dir_mark[1]] = 1
        #     len_mark = arrow_len
        # except: 
        #     try:
        #         STRF_lookalike[0:10,pref_dir_mark2[0],pref_dir_mark2[1]] = 1
        #         len_mark = arrow_width
        #     except: 
        #         STRF_lookalike[0:10,pref_dir_mark3[0],pref_dir_mark3[1]] = 1
        #         len_mark = 1
        # #figure out how the prefered direction maps to the SxT representation
        # # if ix == 7:
        # #     'flag'
        # SxT_pdir, x_lims, ylims = stack_minmaxLine(STRF_lookalike,roi.center_vect,end_vect,pdir2,extend=int(5/scaling_fact))
        # SxT_pdir = np.where(SxT_pdir > 0,1,0)
        # try:
        #     roi.pos_pdir_marker =  ((SxT_pdir.shape[0]//2 - len_mark) - np.where(SxT_pdir == 1) [0][0]) # if marker is negetive, dir is down and visceversa
        #     no_arrow=False
        # except IndexError:
        #     roi.pos_pdir_marker = 0
        #     no_arrow=True

        SxT_representation, x_lims, ylims = stack_minmaxLine(roi.STRF_data['strf'],end_vect,copy_center_vect,pdir,extend)
        line_size = np.sqrt(((x_lims[0]-x_lims[1])**2) +((ylims[0]-ylims[1])**2))
        print('line')
        print(line_size)
        line_size = line_size*scaling_fact
        roi.line_size_pref_dir = line_size
        SxT_representation2, x_lims2, ylims2 = stack_minmaxLine(roi.STRF_data['strf'],end_vect2,copy_center_vect,pdir2,extend)
        line_size2 = np.sqrt(((x_lims2[0]-x_lims2[1])**2) + ((ylims2[0]-ylims2[1])**2))
        line_size2 = line_size2*scaling_fact
        roi.line_size_Max_resp = line_size2
        # if roi.pos_pdir_marker < 0:
        #     SxT_representation2 = np.flip(SxT_representation2,axis=0) 
        roi.STRF_data['SxT_representation_pdir'] = SxT_representation
        roi.STRF_data['SxT_representation_MaxRespDir'] = SxT_representation2

        #SxT_representation = np.repeat (SxT_representation,4,axis=1)
        fig2=plt.figure()  
        
        gs = GridSpec(1, 2)
        ax = fig2.add_subplot(gs[0])
        SxT_representation_2plot = np.repeat(SxT_representation,3,axis=1)
        time0_marker = 10*3
        peakval = np.nanmax(np.abs(SxT_representation))
        if np.isnan(line_size) or np.isposinf(line_size) or np.isneginf(line_size):
            im = ax.imshow(SxT_representation_2plot ,cmap='RdBu',vmax=peakval,vmin=-peakval)#,extent=[-3,1,0,line_size])
        else:
            im = ax.imshow(SxT_representation_2plot ,cmap='RdBu',vmax=peakval,vmin=-peakval)#,extent=[-3,1,0,line_size],aspect ='auto')
        plt.plot([90, 90], [0, SxT_representation_2plot.shape[0]-1], linestyle='--', color='black', linewidth=2)
        cbar=fig2.colorbar(im, ax=ax)
        ax.title.set_text('sxTRF values across pref dir axis \n %s %s'%(roi.CS,line_size))

        ax2 = fig2.add_subplot(gs[1])
        SxT_representation_2plot = np.repeat(SxT_representation2,3,axis=1)
        peakval = np.nanmax(np.abs(SxT_representation2))
        if np.isnan(line_size2) or np.isposinf(line_size2) or np.isneginf(line_size2):
            im2 = ax.imshow(SxT_representation_2plot ,cmap='RdBu',vmax=peakval,vmin=-peakval)#,extent=[-3,1,0,line_size])
        else:
            im2 = ax2.imshow(SxT_representation_2plot ,cmap='RdBu',vmax=peakval,vmin=-peakval)#,extent=[-3,1,0,line_size2],aspect ='auto')
        plt.plot([90, 90], [0, SxT_representation_2plot.shape[0]-1], linestyle='--', color='black', linewidth=2)
        cbar=fig2.colorbar(im2, ax=ax2)
        ax2.title.set_text('sxTRF values dirMaxResp axis %s'%(line_size2))        

        # if no_arrow==False:
        #     arrow = patches.FancyArrow(10,40,0,vec_len/2*-1,length_includes_head = True, head_width=5,head_length=10,fc='black',ec='black')
        #     plt.gca().add_patch(arrow)
        plt.savefig(savepath +'\\STRF_roi%s_SxT.jpg'%(ix))
        plt.close(fig2)


    
        #for ix,roi in enumerate(rois):

        # get the index of the max and min values 
        index_of_max = [np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[2][0]]
        index_of_min = [np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[2][0]]

        # fit a double gaussian

        half_sec = int(0.5*stim_up_rate)

        # plot average sta:
        fig=plt.figure()               
        gs = GridSpec(1, 3)
        
        ax = fig.add_subplot(gs[0])
        im = ax.imshow(np.mean(roi.STRF_data['strf'][-half_sec:-1,:,:],axis=0),cmap='RdBu',vmax=np.max(np.abs(roi.STRF_data['strf'])),vmin=-np.max(np.abs(roi.STRF_data['strf'])))
        # plot the line of the spatiotemporal representation
        ax.plot(x_lims,ylims,color='black',linewidth = 1)
        ax.plot(x_lims2,ylims2,color='blue',linewidth = 1)
        roi.STRF_data['lims for sxt'] = {'x_lims': x_lims,'y_lims':ylims}
        # plot a 20 deg circle arround the center 
        yi, xi = np.indices((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]))
        circle = (xi - copy_center_vect[1])**2 + (yi - copy_center_vect[0])**2
        circ_array = np.full((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]), np.nan)
        circ_array[np.where((circle <= (vec_len + 0.5)**2) & (circle >= (vec_len - 0.5)**2))] = 1
        ax.imshow(circ_array)
        ax.title.set_text('last.5sec_%s_%.2f_PD_%s_\nMrespDir_%s'%(roi.CS,roi.CSI,pdircopy,roi.dir_max_resp))
        cbar=fig.colorbar(im, ax=ax)
       
        ax1 = fig.add_subplot(gs[1])
        im = ax1.imshow(np.mean(roi.STRF_data['strf'][-half_sec*2:-half_sec,:,:],axis=0),cmap='RdBu',vmax=np.max(np.abs(roi.STRF_data['strf'])),vmin=-np.max(np.abs(roi.STRF_data['strf'])))
        ax1.title.set_text('2ndlast.5sec_%s_Z%s'%(roi.category[0], roi.STRF_data['z_score']))
        cbar=fig.colorbar(im, ax=ax1)
        
        ax2 = fig.add_subplot(gs[2])
        im = ax2.imshow(np.mean(roi.STRF_data['strf'][:half_sec,:,:],axis=0),cmap='RdBu',vmax=np.max(np.abs(roi.STRF_data['strf'])),vmin=-np.max(np.abs(roi.STRF_data['strf'])))
        ax2.title.set_text('noise.5sec_%s_Z%s'%(roi.category[0], roi.STRF_data['z_score']))
        cbar=fig.colorbar(im, ax=ax2)
        plt.savefig(savepath +'\\%sroi_mean_SRF.jpg'%(ix))
        plt.close(fig)
       
        ON_subset = predictions_df.loc[predictions_df['polarity_list']=='ON']
        OFF_subset = predictions_df.loc[predictions_df['polarity_list']=='OFF']


    plt.close('all')
    fig1 = plt.figure()
    plt.hist(np.array(ON_subset['train_corr']),bins=40,range=(-1,1))
    plt.title('corr_seen_on')
    fig5 = plt.figure()
    plt.hist(np.array(ON_subset['test_corr']),bins=40,range=(-1,1))
    plt.title('corr_unseen_on')

    fig2 = plt.figure()
    plt.hist(np.array(ON_subset['frozen_reliability']),bins=40,range=(-1,1))
    plt.title('frozen_reliability_ON')

    fig3 = plt.figure()
    plt.hist(np.array(OFF_subset['train_corr']),bins=40,range=(-1,1))
    plt.title('corr_seen_off')
    fig6 = plt.figure()
    plt.hist(np.array(OFF_subset['test_corr']),bins=40,range=(-1,1))
    plt.title('corr_unseen_off')
    
    fig4 = plt.figure()
    plt.hist(np.array(OFF_subset['frozen_reliability']),bins=40,range=(-1,1))
    plt.title('frozen_reliability_OFF')
    
    pmc.multipage(savepath +'\\correlation_hists.pdf')
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    if scramble:
        #try:
        fig = plt.figure()
        plt.hist(np.array(ON_subset['train_corr_M']),bins=40,range=(-1,1))
        plt.title('corr_seen_on_scrambleed')
        fig2=plt.figure()
        plt.hist(np.array(ON_subset['test_corr_M']),bins=40,range=(-1,1))
        plt.title('corr_unseen_on_scrambleed')

        fig1 = plt.figure()
        plt.hist(np.array(OFF_subset['train_corr_M']),bins=40,range=(-1,1))
        plt.title('corr_seen_off_scrambleed')
        fig3=plt.figure()
        plt.hist(np.array(OFF_subset['test_corr_M']),bins=40,range=(-1,1))
        plt.title('corr_unseen_off_scrambleed')
        pmc.multipage(savepath +'\\correlation_hists_scrambled.pdf')
        plt.close(fig)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

        #save predictions df
         
    

    
    return rois


def Trial_av_frozen_noise(trace,stim_dataframe,test_frames,calculate_reliability = True):
    
    '''Use given indices to trial average the responses to frozen noise'''
    
    len_traces=[]
    test_traces=[]
    test_frames = [[0, 1200], [3600, 4800], [7200, 8400],[10800,12000]]
    for trial in test_frames:
        index_1 = int(stim_dataframe.loc[stim_dataframe['stim_frame'].astype(int)==trial[0]].iloc[0]['mic_frame'])
        index_2 = int(stim_dataframe.loc[stim_dataframe['stim_frame'].astype(int)==trial[1]].iloc[0]['mic_frame']+1)
        test_traces.append(trace[index_1:index_2])
        len_traces.append(len(trace[index_1:index_2]))
    minlen_traces = np.min(len_traces)
    stacked_traces = np.zeros((len(test_frames),minlen_traces))
    
    for i,trial in enumerate(test_traces):
        stacked_traces[i,:] = test_traces[i][:minlen_traces]
    #test_traces = np.hstack(test_traces)

    if calculate_reliability == True:
        perm = permutations(range(stacked_traces.shape[0]), 2) 
        correlations = []
        for comp in perm:
            correlations.append(pearsonr(stacked_traces[comp[0],:], stacked_traces[comp[1],:])[0])
        correlations = np.array(correlations)
        return np.nanmean(stacked_traces,axis=0), np.nanmean(correlations)
    else:
        
        return np.nanmean(stacked_traces,axis=0)

def fit_gaussian2D(rois):
    print('not yet implemented')

def gaussian_2d(x, y, A, x0, y0, sigma_x, sigma_y, C):
    """2D Gaussian function for fitting."""
    return A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))) + C


def stack_minmaxLine(images, max_index, min_index,pref_dir,extend=1):
    """
    Extracts lines of pixel values from each image in a stack of images.

    Parameters:
    images (numpy.ndarray): A 3D array of shape (n, x, y) representing a stack of images.
    max_index (tuple): A tuple (x_max, y_max) indicating the indices of the maximum value.
    min_index (tuple): A tuple (x_min, y_min) indicating the indices of the minimum value.

    Returns:
    numpy.ndarray: A 2D array where each row represents the line of pixel values 
                   from max_index to min_index for each image.
    """
    
    
    
    # if (extend!= 1 and (np.cos(pref_dir) != 0 and np.sin(pref_dir) != 0)):
    #     extend = int(extend/2)
    #     pixel_range = range(-extend, extend+1)
    #     local_min_index = [0,0]
    #     local_max_index = [0,0]
    #     start_x, start_y, end_x, end_y,slope,intercept = calculate_extended_line ( min_index[0], min_index[1], max_index[0], max_index[1],images[0,:,:])
    #     for ix,shift in enumerate(range(-extend, extend+1)):
    #         #calculate the sine and move the line in a single axis (y)
    #         local_min_index[1] = extend*np.sin(pref_dir) + min_index[0]
    #         local_max_index[1] = extend*np.sin(pref_dir) + max_index[0]        
    #         local_min_index[0] = max_index[1]
    #         local_max_index[0] = max_index[0]
    #         start_x, start_y, end_x, end_y = calculate_extended_line ( min_index[0], min_index[1], max_index[0], max_index[1],images[0,:,:])
    #         start_x, start_y, end_x, end_y = int(round(start_x)),int(round(start_y)),int(round(end_x)),int(round(end_y))

    start_x, start_y, end_x, end_y = calculate_extended_line ( min_index[0], min_index[1], max_index[0], max_index[1],images[0,:,:])
    if distance.euclidean(np.array([start_x,start_y]),np.array([min_index[0],min_index[1]])) < distance.euclidean(np.array([start_x,start_y]),np.array([max_index[0],max_index[1]])):
        start_y,start_x, end_y, end_x,  = int(round(start_x)),int(round(start_y)),int(round(end_x)),int(round(end_y))
    else:
        start_y, start_x, end_y, end_x = int(round(end_x)),int(round(end_y)),int(round(start_x)),int(round(start_y))
    
    x_line, y_line = line(start_x, start_y, end_x, end_y)
    line_stack = np.zeros((len(x_line),len(range(-extend, extend+1)),images.shape[0]))
    line_stack[:] = np.nan
    #orig_start_x, orig_start_y, orig_end_x, orig_end_y = int(round(start_x)),int(round(start_y)),int(round(end_x)),int(round(end_y))
    
    for im_ix in range(images.shape[0]):
        local_stack = profile_line(images[im_ix,:,:],(start_y,start_x),(end_y,end_x),linewidth=extend,mode = 'constant',cval = np.nan) #reduce_func = np.mean
        mean_stack = local_stack #np.mean (local_stack,axis=0)
        if im_ix == 0:
            #initialize array
            line_stack = np.zeros((len(mean_stack),images.shape[0]))
        line_stack[:,im_ix]=mean_stack


    
   #  orig_x_line, orig_y_line = line(start_x, start_y, end_x, end_y)
    
    # for ix,shift in enumerate(range(-extend, extend+1)):
    #     local_min_index = np.zeros((2))
    #     local_max_index = np.zeros((2))
        
    #     # if extend == 0:
    #     #     line_values = map_coordinates(images[ix], [x_line, y_line], order=1)
    #     #     line_stack[:,ix,im_ix] = line_values            
           
    #     #    continue
    #     if np.cos(pref_dir) == 0: #shift in x only 
    #         local_min_index[1] = extend + min_index[1]
    #         local_max_index[1] = extend + max_index[1]
    #     elif np.sin(pref_dir) == 0: # shift in y only:
    #         local_min_index[0] = extend + min_index[0]
    #         local_max_index[0] = extend + max_index[0]
    #     else: 
    #         #calculate the sine and move the line in a single axis (y)
    #         local_min_index[0] = extend*np.sin(pref_dir) + min_index[0]
    #         local_max_index[0] = extend*np.sin(pref_dir) + max_index[0]
            
    #         # Generate the coordinates for the line between max and min indices for interpolation
    #         start_x, start_y, end_x, end_y,_,_ = calculate_extended_line (local_min_index[0], local_min_index[1], local_max_index[0], local_max_index[1],images[0,:,:],pref_dir)

    #         start_x, start_y, end_x, end_y = int(round(start_x)),int(round(start_y)),int(round(end_x)),int(round(end_y))
    #         x_line, y_line = line(start_x, start_y, end_x, end_y)
    #         n, x, y = images.shape
            
    #         #figure our the needed padding to keep alignment
    #         pad_left,pad_right = calculate_padding (slope,intercept,extend*np.sin(pref_dir),images[0,:,:])
    #         pad_left = np.abs(orig_start_y-orig_end_y) - np.abs(start_y-end_y)
    #         pad_right = np.abs(orig_start_x-orig_end_x) - np.abs(start_x-end_x)
            
    #         for im_ix in range(images.shape[0]):
    #             # Extract the line of pixel values using interpolation
    #             if im_ix == 0:
    #                 line_values = map_coordinates(images[ix], [x_line, y_line], order=1)
    #                 image_stack = np.zeros(images.shape[0],len(line_values))
    #                 image_stack[im_ix,:] = line_values
    #             else:
    #                 line_values = map_coordinates(images[ix], [x_line, y_line], order=1)
    #                 image_stack[im_ix,:] = line_values
            
    #         if pad_left>0:
    #             image_stack = np.pad(image_stack, ((pad_left, 0),(0,0)), 'constant', constant_values=np.nan)
    #         elif pad_left<0:
    #             #chop linestack on the left 
    #             image_stack = image_stack[:,(pad_left*-1)-1:]
    #         if pad_right>0:
    #             image_stack = np.pad(image_stack, ((0, pad_right),(0,0)), 'constant', constant_values=np.nan)
    #         elif pad_right<0:
    #             image_stack = image_stack[:,:pad_right]
           
    #         line_stack[:,ix,:]= image_stack


    return line_stack, [start_x,end_x],[start_y,end_y]

# def calculate_padding(slope,intercept,shift,image):


def calculate_extended_line(x_1, y_1, x_2, y_2,image):
    """
    Calculate the endpoints of a line passing through (x_min, y_min) and (x_max, y_max)
    and extending to the edges of an image of size (img_width, img_height).
    """
    
    # x_Max_ix = np.argmax(np.array([x_1,x_2]))
    # x_Min_ix = np.argmin(np.array([x_1,x_2]))
    x_min = x_1
    y_min = y_1
    x_max = x_2
    y_max = y_2
    # Calculate slope and intercept of the line
    if x_max != x_min:  # Avoid division by zero
        slope = (y_max - y_min) / float(x_max - x_min)
        intercept = y_min - (slope * x_min)

        # Find intersections with the image frame
        y_at_x0 = intercept  # y at x = 0
        y_at_xmax = slope * (image.shape[0] - 1) + intercept  # y at x = img_width
        x_at_y0 = -intercept / slope  # x at y = 0, avoid division by zero
        x_at_ymax = ((image.shape[1] - 1) - intercept) / slope # x at y = img_height
        
        poss_vals = np.array([[y_at_x0, y_at_xmax],[x_at_y0, x_at_ymax]])
        condition1 = 0<=np.array([[y_at_x0, y_at_xmax],[x_at_y0, x_at_ymax]])
        condition2 = np.array([[y_at_x0, y_at_xmax],[x_at_y0, x_at_ymax]])<image.shape[1]
        indices = np.where(condition1 & condition2)
        initvals = np.array([[0, image.shape[0] - 1], [0, image.shape[1] - 1]])
        points = []
        for y,x in zip(indices[0],indices[1]):
            if y == 0:
                points.append ([poss_vals[y,x],initvals[y,x]])
            else:
                points.append ([initvals[y,x],poss_vals[y,x]])
       
    else: 
        points = [[0,x_min],[image.shape[1] - 1,x_min]]
             

    return (points[0][1], points[0][0], points[1][1], points[1][0]) #start_x, start_y, end_x, end_y

def calculate_Zscore_STRF(roi,control = False):
    
    if control:
        strf_mat = roi.STRF_data['strf_null']
    else:
        strf_mat = roi.STRF_data['strf']
    #calculate the Zscores of the values in an strf 
    z_vals = (strf_mat-(np.mean(strf_mat)))/np.std(strf_mat)
    max_Z_val = np.max(z_vals) if roi.CS =='ON' else np.abs(np.min(z_vals))
    max_Z_val_coords = np.argmax(z_vals) if roi.CS =='ON' else np.argmin(z_vals)
    roi.STRF_data['max_Z_val_polarity'] = max_Z_val
    roi.STRF_data['max_z_val_abs'] = np.max(np.abs(z_vals)) 
    roi.STRF_data['max_z_val_abs_coords'] = np.argmax(np.abs(z_vals))
    return z_vals, max_Z_val, max_Z_val_coords

def calculate_robust_Zscore(roi):
    # the deviation and the median are calculated within the time window where no response is expected (first half of reconstruction)
    #mad is median absolute deviation
    mad = np.median(np.abs(roi.STRF_data['strf'][0:15,:,:]- np.median(roi.STRF_data['strf'][0:15,:,:], axis=0)[np.newaxis,:,:]),axis=0)
    # Generate the index arrays for the second and third dimensions

    x, y = np.meshgrid(np.arange(roi.STRF_data['strf'].shape[1]), np.arange(roi.STRF_data['strf'].shape[2]), indexing='ij')

    
    robust_z_score = 0.6745*np.abs((roi.STRF_data['strf'][:-10,:,:][np.argmax(np.abs(roi.STRF_data['strf'][:-10,:,:]),axis=0),x,y]) - np.median(roi.STRF_data['strf'][0:15,:,:],axis=0))/mad
    return robust_z_score

def calculate_saliency_With_RZscore(roi,max_index):

    #center = apply_circular_mask(roi,20,indices = max_index)
    #surround = apply_circular_mask(roi,20,indices = max_index,surround=True)
    only_mask_center = np.sum(apply_circular_mask(roi.STRF_data['strf'],roi,15,indices = max_index,only_mask=True))
    only_mask_surround = np.sum(apply_circular_mask(roi.STRF_data['strf'],roi,15,indices = max_index,surround=True,only_mask=True))


    av_zscore = np.sum(only_mask_center*roi.STRF_data['robust_z_score'])
    av_surround_zscore = np.sum(only_mask_surround*roi.STRF_data['robust_z_score'])

    return av_zscore - av_surround_zscore

# def find_STRF_maxMin_line(roi):

#     max_pos = np.unravel_index(np.argmax(roi.mean_strf), roi.mean_strf.shape)
#     min_pos = np.unravel_index(np.argmin(roi.mean_strf), roi.mean_strf.shape)
#     # Create a line from min to max
#     line_y, line_x = np.linspace(min_pos[0], max_pos[0], num=100), np.linspace(min_pos[1], max_pos[1], num=100)
#     # Use interpolation to sample the image
#     line_values = map_coordinates(roi.mean_strf, np.vstack((line_y, line_x)))
#     # calculate the length of the resulting line 
#     line_lenght = np.sqrt((min_pos[0]-max_pos[0])^2 + (min_pos[1]-max_pos[1])^2)

def apply_reverse_correlation(trace,roi,stim_up_rate,stim_dataframe,snippet,initial_frame,stimulus=None,indices=None,png_list=None,frozen_indices=None):
    
        
    #jf: commented low_pass_filtering, we are still interested in frequencies close to the nyquist frequency
    #filtered = low_pass(filtered, fps, crit_freq=cf,plot=False) #juan: consider changing critical frequency to nyquist freq

    # loop through the signal timepoints and calculate the reverse correlation for each point in the previous 2 seconds 
    
    #first initialize the STA array. it should have a time dimension and as many spatial dimensions as the stimuli
    if indices is not None:
        trace_indexes = indices[indices > initial_frame]
    else :
        trace_indexes = range(initial_frame,len(trace))


    if stimulus is not None:
        sta_dims = stimulus[0,:,:].shape
    else: 
        sta_dims = np.repeat(imageio.imread(png_list[0])[:,:,0],2,axis=0).shape

    STA_array = np.zeros(sta_dims).astype(float)
    STA_array = np.expand_dims(STA_array,axis = 0)
    STA_array = np.repeat(STA_array,snippet+10,axis = 0)
    ##extra dim added to stack snippets and then average them
    # STA_array = np.expand_dims(STA_array,axis = 0)
    # STA_array = np.repeat(STA_array,len(trace_indexes),axis = 0)
    #print(STA_array.shape)

    trace_indexes = np.array(trace_indexes)

    #translate frozen stim indexes into microscope frames
    frozen_mic_frames = stim_dataframe[stim_dataframe['stim_frame'].isin(frozen_indices)]['mic_frame']
    frozen_mic_frames = np.unique(frozen_mic_frames)

    # exclude frozen frames from index list
    trace_indexes = np.setdiff1d(trace_indexes,frozen_mic_frames) #frozen indices should be a comprehensive list of indices. not just the edges of the range
    
    # take out the 1.5 seconds after a missing frame or a delayed stimulus frame 
    missing_frames,trashed1 = identify_mising_frames_in_outputfile(stim_dataframe)
    trace_indexes = np.setdiff1d(trace_indexes,missing_frames)
    

    delayed_frames,trashed2 = find_delayed_frames(stim_dataframe,treshold=0.01,mic_framerate=roi.imaging_info['frame_rate'])
    trace_indexes = np.setdiff1d(trace_indexes,delayed_frames)

    print('trashed frames_apprx: %s' %(trashed1 + trashed2))


    if len(trace_indexes)==0:
        raise Exception ('error with the indices') 

    # pad the stim array at the end
    pad = np.zeros((11,stimulus.shape[1],stimulus.shape[2]))
    stimulus = np.concatenate([stimulus,pad])

    if frozen_indices is not None:
        test_indices = frozen_indices # test indices are the indexes of stimuli that cannot be used 
    else:
        test_indices = np.array(range(frozen_indices[0],frozen_indices[1]))
    not_used_frames = []

    # create an array of the stim_frames that are presented at time deltat=0 with respect to the microscope frames
    stim_dataframe_reduced = stim_dataframe.drop_duplicates(subset=['mic_frame'],keep='last')
    stim_dataframe_reduced = stim_dataframe_reduced[stim_dataframe_reduced['mic_frame'].isin(trace_indexes)]
    stim_frame_vals = stim_dataframe_reduced['stim_frame'].astype(int).values
    trace_indexes_vals = stim_dataframe_reduced['mic_frame'].astype(int).values

    #assert np.array_equal(trace_indexes, trace_indexes_vals)
    # filter the dataframe based on the remaining mic frames available 

    # create an optimized function for STRF computing 
    print('computing reverse_correlation')


    STA_array = compute_STRF(trace_indexes_vals, stim_frame_vals, trace, stimulus, snippet, STA_array)

    print('finished_computing')
    ###########
    ### previous implementation: not optimized for speed
    ###########
    # for ix,i in enumerate(trace_indexes):
     
    #     current_stim_frame = int(stim_dataframe.loc[stim_dataframe['mic_frame'] == i].iloc[-1]['stim_frame']) # previously it was iloc[0]['stim_frame']
    #     #last_stim_frame = int(stim_dataframe.iloc[-1]['stim_frame'])
    #     #list_of_stim_indices = np.array(range(current_stim_frame-snippet+1,current_stim_frame+1))

    #     if ix%1000==0:
    #         print('round %s of %s' %(ix+1,len(trace_indexes))) 

    #     stim_chunk = stimulus[current_stim_frame-snippet:current_stim_frame+10,:,:]
    #     stim_chunk = stim_chunk*trace[i]
    #     STA_array  += stim_chunk
    

    #print('lost frames: %s'%(lost_count))
    #print ('timing errors: %s'%(delay_count))
    #strf = np.mean(STA_array,axis=0)#/np.max(STA_array)
    strf = STA_array/(len(trace_indexes)-len(test_indices))
    mean_strf = np.mean(strf[int(len(strf))-15:int(len(strf))-10,:,:],axis=0)
    plt.figure()
    plt.plot(np.array(not_used_frames))
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    #formatter.set_powerlimits((-1,1))
    return strf, mean_strf,frozen_mic_frames


def STRF_response_prediction(roi,roi_name,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=None,t_window=40,n_epochs=1,control=False,held_out_frames=None,test_frames=None,test_trace=None,max_index=None,scramble=False):

    """
    predict neuron response based on its STA (receptive field and the random stimulus)
    t_window should be the same as used to calculate sta

    for now it's implemented for 1 repetition of stimulus

    """

    snippet = t_window
    # if type_ == 'train':
    #     if held_out_frames is not None:
    #         trace_indexes = range(held_out_frames[0],held_out_frames[1]+1)
    #     else:
    #         trace_indexes = range(initial_frame,len(trace))
    #     snippet = t_window
    # elif type_ == 'test':
    #     if held_out_frames is not None:
    #         trace_indexes = np.array(range(0,held_out_frames[0])) 
    #         trace_indexes2 = np.array(range(held_out_frames[1],len(trace)))
    #         trace_indexes = np.concatenate(trace_indexes,trace_indexes2)
    #     else:
    #         trace_indexes = range(initial_frame,len(trace))

    trace_indexes = range(initial_frame,len(trace))
    #for roi in rois:
    strf_prediction = np.zeros(trace.shape[0])
    strf_prediction[:] = np.nan 
    copy_trace = np.zeros(trace.shape[0])
    copy_trace[:] = np.nan 
    frame_lost_label = -1


    restricted_RF = apply_circular_mask(roi,15)
    
    restricted_RF = apply_circular_mask(roi,15)
    
    if scramble:
        scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1])
        # radius_scaled = 30/scaling_fact
        # slice_0_size = int(radius_scaled)/2
        # slice_1_size = int(radius_scaled)/2
        # if max_index[0]>=210 or max_index[0]<=30:
        #     if max_index[0]>=210:
        #         distance_1 = max_index[0] - 210
        #         distance_0 = 0
        #     else:
        #         distance_0 = max_index[0] - 30
        #         distance_1 = 0
        # else:
        #     distance_0 = 0
        #     distance_1 = 0
        # if max_index[1]>=210 or max_index[1]<=30:
        #     if max_index[1]>=180:
        #         distance_j =  max_index[0] - 210
        #         distance_i = 0
        #     else:
        #         distance_i = max_index[1] - 30
        #         distance_j = 0
        # else:
        #     distance_i = 0
        #     distance_j = 0
        copy_RF = copy.deepcopy(roi.STRF_data['strf'][0:snippet,:,:])
        #square_region = copy_RF[:,max_index[0]-slice_0_size-distance_0:max_index[0]+slice_0_size-distance_1, max_index[1]-slice_1_size-distance_i:max_index[1]+slice_1_size-distance_j]
        #flipped =  square_region[:,::-1,::-1] #np.rot90(square_region, k=1, axes=(1,2))       
        scrambled = scramble_spatialaxes(copy_RF,roi,seed=42)
        #copy_RF[:,max_index[0]-slice_0_size-distance_0:max_index[0]+slice_0_size-distance_1, max_index[1]-slice_1_size-distance_i:max_index[1]+slice_1_size-distance_j] = scrambled
        #mean_scrambleed = np.mean(restricted_RF[20:30,:,:],axis = 0) 
        #max_index = [np.where(np.abs(mean_scrambleed)==np.max(np.abs(mean_scrambleed)))[0][0],np.where(np.abs(mean_scrambleed)==np.max(np.abs(mean_scrambleed)))[1][0]]
        restricted_RF = apply_circular_mask(scrambled,roi,15,indices = max_index)
    else:
        restricted_RF = apply_circular_mask(roi.STRF_data['strf'][0:snippet,:,:],roi,15,indices = max_index)

    for ix,i in enumerate(trace_indexes):
        if i == 4680:
            'pepe'
        try:
            index = int(stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0]['mic_frame'])
        except IndexError: #this happens when frames are skipped
            frame_lost_label = 0
            strf_prediction[i]=np.nan
            #lost_count += 1
            print ('frame %s is lost'%(i))
        if frame_lost_label >=0 and frame_lost_label < int(np.ceil(3*roi.imaging_info['frame_rate'])):
            frame_lost_label += 1
            continue
        elif frame_lost_label == int(np.ceil(3*roi.imaging_info['frame_rate'])):
            frame_lost_label = -1

        # retrieve the real value for comparison
        #copy_trace[i] = trace[i]
        if i == 60:
            'aaa'
        if ix%1000==0:
            print('Prediction_round %s of %s' %(ix+1,len(trace_indexes)))

        current_stim_frame = int(stim_dataframe.loc[stim_dataframe['mic_frame'] == i].iloc[-1]['stim_frame'])
        list_of_stim_indices = np.array(range(current_stim_frame-snippet,current_stim_frame+1))

        if np.any(list_of_stim_indices<0):
            stim_chunk_1 = np.zeros((np.sum(list_of_stim_indices<0),stimulus.shape[1],stimulus.shape[2]))
            remain = snippet - np.sum(list_of_stim_indices<0)
            if remain>0:
                stim_chunk_2 = copy.deepcopy(stimulus[current_stim_frame:current_stim_frame+remain,:,:])
                stim_chunk = np.concatenate((stim_chunk_1,stim_chunk_2),axis = 0)
            else:
                stim_chunk = stim_chunk_1
        else:
            stim_chunk = copy.deepcopy(stimulus[current_stim_frame-snippet:current_stim_frame,:,:])

        ### check that the snippet contents are timed correctly otherwise create delays in the data to compensate
        ## this seems to be necesary since some frames are staying longer than expected on the screen
        index = stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[-1].name
        delay_treshold = ((1/float(stim_up_rate))+((1/float(stim_up_rate)/10)))
        if np.any(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>=delay_treshold):
            error_ix = np.where(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>delay_treshold)[0]
            for error in error_ix:
                delay_signature = int(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])[error]/(1/float(stim_up_rate)))
                stim_chunk[error: error + delay_signature,:,:] = stim_chunk[error-1,:,:]
                remaining_len = len(stim_chunk[error + delay_signature:,0,0])
                stim_chunk [error + delay_signature:,:,:] = stim_chunk [error:error+remaining_len,:,:] 
                #delay_count += 1

        strf_prediction[i]= np.sum(stim_chunk*restricted_RF) # prediction giving no weight to the different times in STRF

    #prediction=np.zeros(random_stimulus.shape[0]-t_window)
    # for idx,time_point in enumerate(range(offset, random_stimulus.shape[0])):
    #     stimpreceed = random_stimulus[time_point-t_window:time_point,:,0] # this is valid for stimuli with only one spatial axis of variation (for example when you present bars across elevation or azimuth only)
    #     strf_prediction[idx] = np.sum(stimpreceed*curr_sta)
    #     #pred[key] = prediction
    # #TODO figure out what's the right time vector for the raw trace
    # roi.sta_prediction= prediction
    # time_ax_pred=np.arange(0,(10000-t_window),0.05)
    # time_ax_dfTrace=roi.stim_info['frame_timings']
    # #interpolate prediction and data to 10hz
    # interp_time_ax=np.linspace(0,len(prediction)*0.05,int((len(prediction)*0.05)/0.1))
    # interp_prediction=np.interp(interp_time_ax,time_ax_pred,roi.sta_prediction)
    # interp_trace=np.interp(interp_time_ax,time_ax_dfTrace,roi.df_trace)

    #TODO slice the trace and prediction and exclude nans
    # if control == False:
    #     roi.STRF_data['strf_prediction_%s'%(type_)] = strf_prediction

    #     roi.STRF_data['prediction_corr_%s'%(type_)] = scipy.stats.pearsonr(copy_trace[~np.isnan(strf_prediction)],strf_prediction[~np.isnan(strf_prediction)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
    # else:
    #     roi.STRF_data['strf_prediction_cont'] = strf_prediction
    #     roi.STRF_data['prediction_corr_cont'] = scipy.stats.pearsonr(roi.STRF_data['strf_prediction'][initial_frame:],trace[initial_frame:]) 

    ### evaluate the simple prediction with the train and test traces 
 
    # temporal modification- rectifying non-linearity
    #strf_prediction = np.where(strf_prediction>0,strf_prediction,0)
    ####
    if test_trace is None:
        roi.STRF_data['prediction_corr_test'] = scipy.stats.pearsonr(trace[held_out_frames[0]:][~np.isnan(strf_prediction[held_out_frames[0]:])],strf_prediction[held_out_frames[0]:][~np.isnan(strf_prediction[held_out_frames[0]:])]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
        strf_prediction= copy.deepcopy(strf_prediction)#/(np.nanmax(strf_prediction[held_out_frames[0]:]))
        roi.STRF_data['strf_prediction'] = strf_prediction
        roi.STRF_data['prediction_corr_train'] = scipy.stats.pearsonr(trace[:held_out_frames[0]][~np.isnan(strf_prediction[:held_out_frames[0]])],strf_prediction[:held_out_frames[0]][~np.isnan(strf_prediction[:held_out_frames[0]])]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
    else:
        predicted_test_trace = Trial_av_frozen_noise(strf_prediction,stim_dataframe,test_frames,calculate_reliability = False)
        plt.figure()
        plt.plot(predicted_test_trace)
        plt.title('raw predicted response test set')
        #predicted_test_trace = (predicted_test_trace-np.nanmean(predicted_test_trace))#/np.nanmean(predicted_test_trace)
        held_out_frames = np.concatenate(held_out_frames)
        included_indices = np.setdiff1d(trace_indexes, held_out_frames)

        train_set_prediction = strf_prediction[included_indices]
        plt.figure()
        plt.plot(train_set_prediction)
        plt.title('raw predicted response train set')
        #train_set_prediction = train_set_prediction-np.mean(train_set_prediction)
        train_set_trace = trace[included_indices]

    if scramble == False:
        roi.STRF_data['strf_prediction'] = strf_prediction
        roi.STRF_data['train_prediction'] = train_set_prediction
        roi.STRF_data['test_prediction'] = predicted_test_trace
        roi.STRF_data['train_set_trace'] = train_set_trace
        roi.STRF_data['test_set_trace'] = test_trace

        
        roi.STRF_data['prediction_corr_test'] = scipy.stats.pearsonr(test_trace[~np.isnan(predicted_test_trace)],predicted_test_trace[~np.isnan(predicted_test_trace)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
        roi.STRF_data['prediction_corr_train'] = scipy.stats.pearsonr(train_set_trace[~np.isnan(train_set_prediction)],train_set_prediction[~np.isnan(train_set_prediction)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]

        ####################
        ####### plot prediction
        ####################
            
        fig = plt.figure()
        gs = GridSpec(2,5)

        ax1 = fig.add_subplot(gs[0,2])
        ax2 = fig.add_subplot(gs[1,:])
        timing_vector = np.arange(len(roi.STRF_data['test_prediction']))/roi.imaging_info['frame_rate']
        ax2.plot(timing_vector,(roi.STRF_data['test_set_trace']/np.nanmax(roi.STRF_data['test_set_trace'])), color = 'grey',label='response')
        Minus_bg = roi.STRF_data['test_prediction']-np.nanmean(roi.STRF_data['test_prediction'])
        prediction_normalized = Minus_bg/(np.nanmax(Minus_bg))
        ax2.plot(timing_vector,(prediction_normalized-1), color = 'black',label='prediction')
        ax2.legend()
        ax2.set_xlabel('seconds')
        ax2.set_title( '%s_corr_seen: %.2f corr_unsees: %.2f reliab %.2f' %(roi.CS,roi.STRF_data['prediction_corr_train'][0],roi.STRF_data['prediction_corr_test'][0],roi.STRF_data['frozen_stim_resp_reliability']))
        
        max_val = np.nanmax(np.abs(restricted_RF))
        
        im = ax1.imshow(np.mean(restricted_RF[restricted_RF.shape[0]//2:,:,:],axis=0),cmap='RdBu',vmax=max_val,vmin=-max_val)
        cbar=fig.colorbar(im, ax=ax1,shrink=0.5)
        ax1.axis('off')
        #roi.STRF_data['prediction'] = strf_prediction
        return roi.STRF_data['prediction_corr_train'][0] , roi.STRF_data['prediction_corr_test'][0]

    else:
        roi.STRF_data['shuffled_strf_prediction'] = strf_prediction
        roi.STRF_data['shuffled_train_prediction'] = train_set_prediction
        roi.STRF_data['shuffled_test_prediction'] = predicted_test_trace
        roi.STRF_data['shuffled_train_set_trace'] = train_set_trace
        roi.STRF_data['shuffled_test_set_trace'] = test_trace

        roi.STRF_data['shuffled_prediction_corr_test'] = scipy.stats.pearsonr(test_trace[~np.isnan(predicted_test_trace)],predicted_test_trace[~np.isnan(predicted_test_trace)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
        roi.STRF_data['shuffled_prediction_corr_train'] = scipy.stats.pearsonr(train_set_trace[~np.isnan(train_set_prediction)],train_set_prediction[~np.isnan(train_set_prediction)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]

        return roi.STRF_data['shuffled_prediction_corr_train'][0] , roi.STRF_data['shuffled_prediction_corr_test'][0], restricted_RF


def optimized_STRF_response_prediction(savepath,roi,roi_name,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=None,t_window=40,indices=None,held_out_frames=None,test_frames=None,test_trace=None,max_index=None,scramble=False):
    """
    predict neuron response based on its STA (receptive field and the random stimulus)
    t_window should be the same as used to calculate sta

    for now it's implemented for 1 repetition of stimulus
    """
    snippet = t_window
    
    if indices is not None:
        trace_indexes = indices[indices > initial_frame]
    else :
        trace_indexes = range(initial_frame,len(trace))    

    
    # mask the receptive field to decrease the roles of irrelevant pixels
    if scramble:
        scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1])
        copy_RF = copy.deepcopy(roi.STRF_data['strf'][0:snippet,:,:])
        #square_region = copy_RF[:,max_index[0]-slice_0_size-distance_0:max_index[0]+slice_0_size-distance_1, max_index[1]-slice_1_size-distance_i:max_index[1]+slice_1_size-distance_j]
        #flipped =  square_region[:,::-1,::-1] #np.rot90(square_region, k=1, axes=(1,2))       
        scrambled = scramble_spatialaxes(copy_RF,roi,seed=42)
        #copy_RF[:,max_index[0]-slice_0_size-distance_0:max_index[0]+slice_0_size-distance_1, max_index[1]-slice_1_size-distance_i:max_index[1]+slice_1_size-distance_j] = scrambled
        #mean_scrambleed = np.mean(restricted_RF[20:30,:,:],axis = 0) 
        #max_index = [np.where(np.abs(mean_scrambleed)==np.max(np.abs(mean_scrambleed)))[0][0],np.where(np.abs(mean_scrambleed)==np.max(np.abs(mean_scrambleed)))[1][0]]
        restricted_RF = apply_circular_mask(scrambled,roi,15,indices = max_index)
    else:
        restricted_RF = copy.deepcopy(apply_circular_mask(roi.STRF_data['strf'][0:snippet,:,:],roi,15,indices = max_index))

    # create vectors to perform calculations per frame later on
    # create an array of the stim_frames that are presented at time deltat=0 with respect to the microscope frames
    stim_dataframe_reduced = stim_dataframe.drop_duplicates(subset=['mic_frame'],keep='last')
    stim_dataframe_reduced = stim_dataframe_reduced[stim_dataframe_reduced['mic_frame'].isin(trace_indexes)]
    stim_frame_vals = stim_dataframe_reduced['stim_frame'].astype(int).values
    trace_indexes_vals = stim_dataframe_reduced['mic_frame'].astype(int).values    
     
    # track missing and delayed frames 
    excluded = np.zeros_like(trace_indexes_vals)

    # take out the 1.5 seconds after a missing frame or a delayed stimulus frame 
    missing_frames = replace_mising_delayed_frames(stim_dataframe,stimulus,mic_framerate=roi.imaging_info['frame_rate'])
    #corr_stim_copy = copy.deepcopy(corrected_stim)
    for ix__,element in enumerate(trace_indexes):
         if element in missing_frames:
            excluded[ix__]=1

    # crop the stimulus and the RF to reduce computational load 
    chopped_RF,squaremask = apply_square_mask(restricted_RF,roi,15,indices=max_index,return_subset=True,only_mask=False)
    # chopped_RF = chopped_RF.astype('float32')
    # d1,d2 = np.where(np.squeeze(squaremask) == 1)
    # corr_stim_copy = corr_stim_copy[:,np.min(d1):np.max(d1),np.min(d2):np.max(d2)]
    # time1 = time.time()
    # strf_prediction2 = optimized_trace_prediction(corr_stim_copy, chopped_RF)
    # print('timefft %s'%(time.time()-time1))
    # interp_time = stim_dataframe_reduced['rel_time'].values - stim_dataframe_reduced['rel_time'].values[0]
    # _,strf_prediction2 = interpolate_signal(strf_prediction2,20,11,0.1*trace[97:].shape[0],new_time=interp_time)
    # #real_trace = interpolate_signal(strf_prediction2,20,11,0.1*trace[97:].shape[0],new_time=interp_time)
    
    #chopped_RF,squaremask = apply_square_mask(restricted_RF,roi,15,indices=max_index,return_subset=True,only_mask=False)
    d1,d2 = np.where(np.squeeze(squaremask) == 1)
    corr_stim_copy = stimulus[:,np.min(d1):np.max(d1),np.min(d2):np.max(d2)]

    time1 = time.time()
    strf_prediction,copy_trace = compute_STRF_prediction(trace_indexes_vals, excluded, stim_frame_vals, trace, corr_stim_copy, snippet, chopped_RF)
    # print(time.time()-time1)
    print('prediction took %ss'%(time.time()-time1))
    print('computation finished')

    if test_trace is None:
        roi.STRF_data['prediction_corr_test'] = scipy.stats.pearsonr(trace[held_out_frames[0]:][~np.isnan(strf_prediction[held_out_frames[0]:])],strf_prediction[held_out_frames[0]:][~np.isnan(strf_prediction[held_out_frames[0]:])]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
        strf_prediction= copy.deepcopy(strf_prediction)#/(np.nanmax(strf_prediction[held_out_frames[0]:]))
        roi.STRF_data['strf_prediction'] = strf_prediction
        roi.STRF_data['prediction_corr_train'] = scipy.stats.pearsonr(trace[:held_out_frames[0]][~np.isnan(strf_prediction[:held_out_frames[0]])],strf_prediction[:held_out_frames[0]][~np.isnan(strf_prediction[:held_out_frames[0]])]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
    else:
        predicted_test_trace = Trial_av_frozen_noise(strf_prediction,stim_dataframe,test_frames,calculate_reliability = False)
        fig1 = plt.figure()
        plt.plot(predicted_test_trace)
        plt.title('raw predicted response test set')
        #predicted_test_trace = (predicted_test_trace-np.nanmean(predicted_test_trace))#/np.nanmean(predicted_test_trace)
        #held_out_frames = np.concatenate(held_out_frames)
        included_indices = np.setdiff1d(trace_indexes_vals, held_out_frames).astype(int)

        train_set_prediction = strf_prediction[included_indices]
        fig2 = plt.figure()
        plt.plot(train_set_prediction)
        plt.title('raw predicted response train set')
        #train_set_prediction = train_set_prediction-np.mean(train_set_prediction)
        train_set_trace = trace[included_indices]

    if scramble == False:
        roi.STRF_data['strf_prediction'] = strf_prediction
        roi.STRF_data['train_prediction'] = train_set_prediction
        roi.STRF_data['test_prediction'] = predicted_test_trace
        roi.STRF_data['train_set_trace'] = train_set_trace
        roi.STRF_data['test_set_trace'] = test_trace

        
        roi.STRF_data['prediction_corr_test'] = scipy.stats.pearsonr(test_trace[~np.isnan(predicted_test_trace)],predicted_test_trace[~np.isnan(predicted_test_trace)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
        roi.STRF_data['prediction_corr_train'] = scipy.stats.pearsonr(train_set_trace[~np.isnan(train_set_prediction)],train_set_prediction[~np.isnan(train_set_prediction)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]

        ####################
        ####### plot prediction
        ####################
            
        fig = plt.figure()
        gs = GridSpec(2,5)

        ax1 = fig.add_subplot(gs[0,2])
        ax2 = fig.add_subplot(gs[1,:])
        timing_vector = np.arange(len(roi.STRF_data['test_prediction']))/roi.imaging_info['frame_rate']
        ax2.plot(timing_vector,(roi.STRF_data['test_set_trace']/np.nanmax(roi.STRF_data['test_set_trace'])), color = 'grey',label='response')
        Minus_bg = roi.STRF_data['test_prediction']-np.nanmean(roi.STRF_data['test_prediction'])
        prediction_normalized = Minus_bg/(np.nanmax(Minus_bg))
        ax2.plot(timing_vector,(prediction_normalized-1), color = 'black',label='prediction')
        ax2.legend()
        ax2.set_xlabel('seconds')
        ax2.set_title( '%s_corr_seen: %.2f corr_unsees: %.2f reliab %.2f' %(roi.CS,roi.STRF_data['prediction_corr_train'][0],roi.STRF_data['prediction_corr_test'][0],roi.STRF_data['frozen_stim_resp_reliability']))
        
        max_val = np.nanmax(np.abs(restricted_RF))
        
        im = ax1.imshow(np.mean(restricted_RF[restricted_RF.shape[0]//2:,:,:],axis=0),cmap='RdBu',vmax=max_val,vmin=-max_val)
        cbar=fig.colorbar(im, ax=ax1,shrink=0.5)
        ax1.axis('off')
        #roi.STRF_data['prediction'] = strf_prediction
        pmc.multipage(savepath + '\\%s_roi.pdf'%(roi_name),figs= [fig1,fig2,fig])
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig)

        return roi.STRF_data['prediction_corr_train'][0] , roi.STRF_data['prediction_corr_test'][0]

    else:
        roi.STRF_data['shuffled_strf_prediction'] = strf_prediction
        roi.STRF_data['shuffled_train_prediction'] = train_set_prediction
        roi.STRF_data['shuffled_test_prediction'] = predicted_test_trace
        roi.STRF_data['shuffled_train_set_trace'] = train_set_trace
        roi.STRF_data['shuffled_test_set_trace'] = test_trace

        roi.STRF_data['shuffled_prediction_corr_test'] = scipy.stats.pearsonr(test_trace[~np.isnan(predicted_test_trace)],predicted_test_trace[~np.isnan(predicted_test_trace)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
        roi.STRF_data['shuffled_prediction_corr_train'] = scipy.stats.pearsonr(train_set_trace[~np.isnan(train_set_prediction)],train_set_prediction[~np.isnan(train_set_prediction)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]

        return roi.STRF_data['shuffled_prediction_corr_train'][0] , roi.STRF_data['shuffled_prediction_corr_test'][0], restricted_RF
    


def fit_softplus_non_linearity(trace,prediction):
    '''build after the work of leong et al., 2016
       this code uses a training trace (groundtruth) and a predicted trace
       to fit a softplus nonlinearity in order to account for neuronal rectification

       the fit is done using minimal square error minimization 
    '''

    # Define bounds
     
    bounds_ = Bounds([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    lower_bounds = [0, 0, 0, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
    bounds = (lower_bounds, upper_bounds)
    initial_guesses = [1,0,1,0,2]
    optimal_params = least_squares(residuals,initial_guesses,args=(trace,prediction))#,bounds=bounds)
    return optimal_params.x

def softplus_non_linearity(params,x):
    ''' build after the work of leong et al., 2016
        https://doi.org/10.1523/JNEUROSCI.1272-16.2016'''
    a,b,c,d,k = params
    return c* np.log(1+np.exp(a*x+b))**k  + d

def residuals(params, trace, prediction):
    model_vals = softplus_non_linearity(params, trace)
    return model_vals - prediction 

def ReLu (x,a,b):

    return a*np.max(0,x) + b

def apply_circular_mask(strf,roi,radius,indices=None,surround=False,only_mask=False):
    
    """filters out the area beyond radius from the point where STRF is absolute highest
    inputs:
        roi: roi object containing STRF information
        radius: the radius of the area to be included
    
    output: masked STRF
    """

    scaling_fact = 80.0/(strf.shape[1])
    radius_scaled = radius/scaling_fact
    # if indices is None:
    #     indices = np.where(np.abs(strf)==np.max(np.abs(strf)))
    #     indices = indices[1][0], indices[2][0]
    # else:
    indices = indices[0], indices[1]
    yi, xi = np.indices((strf.shape[1], strf.shape[2]))
    circle = (xi - indices[1])**2 + (yi - indices[0])**2
    circ_array = np.full((strf.shape[1], strf.shape[2]), 0)
    if surround == False:
        circ_array[np.where((circle <= (radius_scaled)**2))] = 1
    else:
        circ_array[np.where((circle >= (radius_scaled)**2))] = 1

    if only_mask == False:
        return strf * circ_array[np.newaxis,:,:]
    else: 
        return circ_array[np.newaxis,:,:]
    # multiply the mask with the RF


def apply_square_mask(strf, roi, half_size, indices=None, surround=False, only_mask=False,return_subset=True):
    """
    Filters out the area beyond a square region centered at the point where STRF is absolute highest.
    
    Inputs:
        strf (numpy.ndarray): 3D array to be masked.
        roi: ROI object containing STRF information (not used directly in this function).
        half_size (float): Half the length of the side of the square region to be included.
        indices (tuple, optional): Tuple of (y, x) indices to center the square. If None, centers at the max absolute value in strf.
        surround (bool, optional): If False, masks inside the square. If True, masks outside the square.
        only_mask (bool, optional): If False, returns the masked STRF. If True, returns only the mask.
        
    Output:
        numpy.ndarray: Masked STRF or the mask itself.
    """
    
    # Scaling factor based on the original function
    scaling_fact = 80.0 / strf.shape[1]
    half_size_scaled = half_size / scaling_fact
    
    # Determine the center indices
    if indices is None:
        # Find the indices where the absolute value of strf is maximum
        max_idx = np.unravel_index(np.argmax(np.abs(strf)), strf.shape)
        center_y, center_x = max_idx[1], max_idx[2]
    else:
        center_y, center_x = indices[0], indices[1]
    
    # Create a grid of indices
    yi, xi = np.indices((strf.shape[1], strf.shape[2]))
    
    # Define the square mask
    square = (
        (np.abs(xi - center_x) <= half_size_scaled) &
        (np.abs(yi - center_y) <= half_size_scaled)
    )
    
    # Initialize the mask array with zeros
    mask_array = np.full((strf.shape[1], strf.shape[2]), 0)
    
    if not surround:
        # Mask inside the square
        mask_array[square] = 1
    else:
        # Mask outside the square
        mask_array[~square] = 1  # Invert the square mask
    
    # if not only_mask:
    #     # Apply the mask to the STRF
    #     return strf * mask_array[np.newaxis, :, :]
   
    if return_subset:
        # Create a copy of `strf` to avoid modifying the original data
        d1,d2 = np.where(np.squeeze(mask_array) == 1)
        subset_strf = copy.deepcopy(strf[:,np.min(d1):np.max(d1),np.min(d2):np.max(d2)])
        return subset_strf, mask_array

    else:
        # Return only the mask
        return mask_array[np.newaxis, :, :]
    


def predict_edgeResponse_withSTRF(rois,stims_path):
    '''convolve a receptive field reconstruction with an aribitrary stimulus video to produce a
        prediction of a neurons response to that video'''
    mainpath = os.path.split(stims_path)[0]
    video_dir = os.path.join(mainpath,'pics_driftingstripe','resized')
    struct_dir = glob.glob(os.path.join(video_dir,'*.txt'))[0]
    video = load_video(video_dir)
    
    
    _, rawstructure,_ = readStimOut(struct_dir,video.shape[0], 
                                          skipHeader=3)
    rawstructure = pd.DataFrame(rawstructure,columns=['entry','rel_time','epochs_presented','epoch','xpos','ypos','stim_frame','mic_frame'])
    epoch_structure  = rawstructure['epoch'] # the  column contains the epoch information 
    

    print(len(epoch_structure)==video.shape[0])
    current_epoch_structure_df = pd.DataFrame(rois[0].cycle1_traceinfo['stim_info']['output_data'],columns=['entry','rel_time','epochs_presented','epoch','xpos','ypos','stim_frame','mic_frame'])
    #single_trial_structure_df = current_epoch_structure_df.loc[current_epoch_structure_df<=8].drop_duplicates()['epochs_presented']

    # obtain the structure of the trials in the real data

    single_trial_structure_df = create_trial_column(current_epoch_structure_df)
    
   
    # for trial in np.unique(single_trial_structure_df['trials']):
    #     #check length
    #     lengths.append(len(single_trial_structure_df.loc[single_trial_structure_df['trials']==trial]))
    # minlen = np.min(lengths)
    
    trial_ixs = np.zeros((2,len(np.unique(single_trial_structure_df['trial']))))
    
    # get the microscope frames corresponding to the trials and average
    
    for trial in np.unique(single_trial_structure_df['trial']):
        trial_ixs[0,trial] = single_trial_structure_df.loc[single_trial_structure_df['trial']==trial]['mic_frame'].iloc[0]
        trial_ixs[1,trial] = single_trial_structure_df.loc[single_trial_structure_df['trial']==trial]['mic_frame'].iloc[-1]
    trial_ixs = trial_ixs.astype(int)
    
    ####################
    # perform trial average
    ###################
    
    
    for roi in rois:
        if roi.STRF_data['status'] == 'excluded':
            continue
        # trial average the df_trace
        trial_traces = []
        for trial in np.unique(single_trial_structure_df['trial']):
            trial_traces.append(roi.cycle1_traceinfo['df_trace'].iloc[trial_ixs[0,trial]:trial_ixs[1,trial]].values)
    
        lengths = []

        for trial in np.unique(single_trial_structure_df['trial']):
            #check length
            if trial == np.unique(single_trial_structure_df['trial'])[-1]:
                len_last = len(trial_traces[trial])
            lengths.append(len(trial_traces[trial]))
        minlen = np.min(lengths)

        # initialize array with trials
        trial_array = np.zeros((len(np.unique(single_trial_structure_df['trial'])),minlen))
        trial_array[:] = np.nan
        
        for trial in trial_traces:
            trial_array[trial,:] = trial[0:minlen]

        roi.trial_averaged_edge_trace = np.nanmean(trial_array, axis = 0)

    ############
    ### reasemble stimulus video to resemble the stimulation that produced the trace
    ############
    reduced_stim_DF=single_trial_structure_df.loc[single_trial_structure_df['trial']==0]
    real_epoch_structure = single_trial_structure_df.loc[single_trial_structure_df['trial']==0].drop_duplicates(subset=['epoch'],keep='first')

    #structure of ref video
    frame_structure_ix1 = rawstructure.drop_duplicates(subset=['epoch'],keep='first')['entry'].values
    frame_structure_ix2 = rawstructure.drop_duplicates(subset=['epoch'],keep='last')['entry'].values

    video_ensemble = []

    for epoch in np.unique(real_epoch_structure):
        

        video_ensemble.append(video[frame_structure_ix1[epoch]:frame_structure_ix2[epoch],:,:])

    video_ensemble = np.concatenate(video_ensemble,axis=0)

    # run prediction code
    for ix, roi in enumerate(rois):

        if roi.STRF_data['status'] == 'excluded':
            continue
        train_, test_ = STRF_response_prediction(roi,ix,roi.trial_averaged_edge_trace,60,reduced_stim_DF,0,stimulus=video_ensemble,t_window=30,n_epochs=1,held_out_frames = None,test_frames = None,test_trace=None,max_index=copy_center_vect)
    # plot prediction and real trace

    # plot an arrow indicating direction

def create_trial_column(stim_dataframe):
    # Initialize trial counter
    trial_counter = 0
    pattern_start_value = stim_dataframe['epoch'][0]  # Assuming the pattern starts with the first value

    # Initialize the trial column with zeros
    stim_dataframe['trial'] = 0

    # Iterate through the DataFrame to assign trial numbers
    for i in range(1, len(stim_dataframe)):
        # Check if the pattern starts again
        if stim_dataframe['epoch'][i] == pattern_start_value and stim_dataframe['epoch'][i - 1] != pattern_start_value:
            trial_counter += 1
        stim_dataframe.at[i, 'trial'] = trial_counter
    return stim_dataframe


def load_video(video_dir):
    '''use a stimulus outputfile from an roi to reconstruct the timecourse of an 8-dir
        edges trial. for this, this function uses a saved video of the 8_dir edges'''
    
    # calculate number of frames per epoch.
    video_path = glob.glob(video_dir + '\\*.npy')
    
    # save video structure in text file (link frames to epoch) 
    file_exists = len(video_path)>0
    if file_exists:
        video = np.load(video_path[0])
    else:
        resize_save_video(video_dir,dimensions=[240,240])
        video = np.load(video_path[0])
    return video
    # reorganize according to trial structure in the current experiment, and return this 


def resize_save_video(folder_path,savepath,dimensions=[240,240]):
    '''takes in a video, resizes it to a desired dimension, then it saves it back as a np array
        .made with chatgpt'''

    # Get a list of image file names in the folder, sorted by file name
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    
    # Initialize an empty list to store the images
    images = []

    # Read each image file and append to the images list
    for ix,file_name in enumerate(image_files):
        image_path = os.path.join(folder_path, file_name)
        image = image = cv2.imread(image_path)[:,:,:]
        #image = np.sum(np.array(image),axis=2)
        #image = np.where(image>0,1,0).astype(np.uint8)
        if ix == 500:
            'aaa'
        #image = np.resize(image, (dimensions[0], dimensions[1]))
        #print(image.shape)
        image = cv2.resize(image, (dimensions[0], dimensions[1]), interpolation=cv2.INTER_AREA)
        image = np.sum(image, axis=2)/np.max(image)
        image = np.where(image==0,-1,image)
        images.append(image)
    
    # Convert the list of images into a 3D numpy array
    # Assuming all images have the same shape (height, width, channels)
    images_array = np.array(images)
    np.save(os.path.join(savepath,'resized_array.npy'),images_array)
    



def create_STRFs(rois, f_w=None,number=None,cmap='coolwarm'):
    import random
    SEED = 54378
    np.random.seed(SEED)
    
    plt.close('all')
    colors = pac.run_matplotlib_params()
    if (number == None) or (f_w==None):
        f_w = 5
        if len(rois)>10:
            number=10
        else:
            number = len(rois)
    elif number > len(rois):
        number = len(rois)
    f_w = f_w*2
    # Randomize ROIs
    copy_rois = copy.deepcopy(rois)
    random.shuffle(copy_rois)
    max_n=np.array(map(lambda roi : np.max(roi.sta), rois)).max()
    min_n=np.array(map(lambda roi : np.max(roi.sta), rois)).min()
        
    
    if number <= f_w/2:
        dim1= number
        dim2 = 1
    elif number/float(f_w/2) > 1.0:
        dim1 = f_w/2
        dim2 = int(np.ceil(number/float(f_w/2)))
    fig1, ax1 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1, dim2))
    ax = ax1.flatten()
    for idx, roi in enumerate(rois):
        if idx == number:
            break
        
        sns.heatmap(roi.sta.T, cmap=cmap, ax=ax[idx], cbar=False,vmax=max_n,
                    center=0)
        ax[idx].axis('off')
    for axs in ax:
        axs.axis('off')
    
    return fig1

def analyze_gratings(rois):
    
    a=1
    return rois

def find_edge_resp_decay(rois):
    """ For luminance edges to find decay from peak response"""
    for roi in rois:
        
        curr_trace = roi.edge_whole_traces_interpolated[roi.max_resp_idx-1][20:]
        max_resp = np.max(curr_trace)
        max_idx = np.argmax(curr_trace)
        last_resp = curr_trace[-1]
        
        decay_df = ((max_resp - last_resp) / max_resp)/(len(curr_trace)-max_idx) * 10
        roi.decay_strength = decay_df
    return rois


def analyze_gratings_general(rois):
    # IMPORTANT INFO
    # Seb: function name was changed from: analyze_luminance_gratings >>> analyze_gratings_general
    # The idea is to make one singl function handling all type of grating stimulation

   
    

    # Seb: if this variable does NOT exist in the stim file, made it 0 (= one single direction)
    if 'direction' in rois[0].stim_info:
        epoch_dirs = rois[0].stim_info['direction']
        mult_directions=True
    else:
        epoch_dirs = np.ndarray.tolist(np.zeros(rois[0].stim_info['EPOCHS']))

    epoch_dirs_no_base= \
        np.delete(epoch_dirs,rois[0].stim_info['baseline_epoch'])
    epoch_types = rois[0].stim_info['stimtype']
    epoch_luminances= np.array(rois[0].stim_info['input_data']['lum'],float)
    if epoch_types[1] == 'noisygrating':
        epoch_SNR= np.array(rois[0].stim_info['input_data']['SNR'],float)

        
    for roi in rois:
        roi_dict = {}


        prefered_direction=roi.PD #Juan: find prefered direction in ROI, and use this to filter the responses to the prefered dir.
        prefered_direction_filter=roi.stim_info['direction']== prefered_direction
        #prefered_direction_filter=np.where(prefered_direction_filter==True) #here we get key values that can be used for specific freqs in traces dict
        

        #Seb commented this out.
        # if not('1D' in roi.stim_name):
        #     curr_pref_dir = \
        #         np.unique(epoch_dirs_no_base)[np.argmin(np.abs(np.unique(epoch_dirs_no_base)-roi.PD))]
        #     req_epochs = (epoch_dirs==curr_pref_dir) & (epoch_types != 11)
        # else:
        #     req_epochs = (epoch_types != 11)
        req_epochs = [e == rois[0].stim_info['stimtype'][-1] for e in epoch_types] # Seb: selecting epochs of interest based on the name of the last epoch in the stimulus input file
        if int(rois[0].stim_info['random']) == 1:
            req_epochs[0] = False # Seb: first epoch is for the baseline, not for analyzing any response
        if rois[0].stim_info['stimtype'][0] != rois[0].stim_info['stimtype'][1]:
            req_epochs[0] = False
        if rois[0].stim_info['stimtype'][0] == 'circle':
            req_epochs[0] = False
        
        # Specific variable based on the typ of stimulation for buiding a future heat map between this variable and TF
        if epoch_types[1] == 'lumgrating':
            roi_dict['luminance'] = epoch_luminances[req_epochs]
            roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[req_epochs]))
            variable_name = 'lum'

        elif epoch_types[1] == 'noisygrating':
            roi_dict['SNR'] = epoch_SNR[req_epochs]
            roi_dict['deltaF'] = np.array(map(float,roi.max_resp_all_epochs[req_epochs]))
            variable_name = 'SNR'

        # if epoch_types[1] == 'TFgrating':
        #     roi.frequencies=roi.stim_info['epoch_frequency'][prefered_direction_filter]
        # else:
        #     roi.frequencies=[1]
        #Seb: fft analysis 
        #epochs_roi_data= roi.whole_trace_all_epochs # try also just with roi.resp_trace_all_epochs
        epochs_roi_data = roi.resp_trace_all_epochs
        
        #loop through freqs, calculate FFT for every trace that correspons to the keys in filter
        #for key,frequency in zip(prefered_direction_filter,frequencies)
        m_acumm=[]
        amp_fft_norm={}
        amp_fft=[]
        frequencies=[]
        indices=[]
        for idx, epoch in enumerate(epochs_roi_data):
            if prefered_direction_filter[idx] == False:
                continue
            frequencies.append(roi.stim_info['epoch_frequency'][idx])
            current_freq=roi.stim_info['epoch_frequency'][idx]
            curr_trace = epochs_roi_data[epoch]
            N = len(curr_trace) # frames or total number of points (aka sample rate)
                
            # FFT and power spectra calculations
            period = 1.0 / roi.imaging_info['frame_rate']
            yf = fft(curr_trace)
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = np.blackman(N)
            ywf = fft((curr_trace-curr_trace.mean())*w)

            # nhz sinusoidal as reference
            Lx = N/roi.imaging_info['frame_rate'] # Duration in seconds
            f = current_freq * np.rint(Lx) # 1hz 
            amp = 0.5 # desired amplitude
            x = np.arange(N)
            y = amp * np.sin(2 * np.pi * f * x / N)
            yf_ref = fft(y) #fft values ref
            # yf_theo_ref = 2.0*np.abs(fft_values_ref/N)
            # mitigate spectral leakage
            w = np.blackman(N)
            ywf_ref = fft((y-y.mean())*w)

            # Locating max peak of the reference frequency
            ref_trace_power = 2.0/N * np.abs(ywf_ref[1:N//2])
            m = max(ref_trace_power)
            m_acumm.append(m)
            max_idx = [i for i in range(len(ref_trace_power)) if ref_trace_power[i] == m][0]
            # Locating amplitude of the desired frequency in the response trace
            response_trace_power = 2.0/N * np.abs(ywf[1:N//2])
            temp_respose_amp = response_trace_power[max_idx]
            amp_fft.append(temp_respose_amp)
            amp_fft_norm[current_freq]=temp_respose_amp#/m #normalization to reference 
            indices.append(max_idx)
                # plotting the spectrum
                # x_freqs = xf[1:N//2]

                # plt.plot(x_freqs, response_trace_power, label='fft values')
                # plt.plot(x_freqs, ref_trace_power)
                # plt.plot(x_freqs[max_idx],temp_respose_amp, 'ro')
                # plt.show()
                # plt.close()

        #save the amplitude of the components and the respective frequencies in 

        # Saving the amplitude of the 1hz component 
        roi.fft_amp = np.array(amp_fft)
        roi.fft_amp_norm=amp_fft_norm
        roi.fft_ref_vals=np.array(m_acumm)
        roi.frequencies_fft=np.array(frequencies)
        #roi.fft_amp_norm=np.array(amp_fft)/np.array(m_acumm)
        # Seb: if this variable does NOT exist in the stim file, create it from others

        #comented by Juan: info already in frequencies_fft
        # if 'epoch_frequency' in rois[0].stim_info:
        #     roi_dict['TF'] = roi.stim_info['epoch_frequency'][req_epochs]
            
        # else:
        #     temp_TF_list = []
        #     for i, value in enumerate(roi.stim_info['velocity']):
        #         #Seb: if statement to take care of non-grating epoch
        #         if value == 0.0 and roi.stim_info['sWavelength'][i] == 0.0:
        #             temp_TF = 0.0
        #             temp_TF_list.append(temp_TF)
        #             continue
        #         temp_TF = value/roi.stim_info['sWavelength'][i]
        #         temp_TF_list.append(temp_TF)

        #     temp_TF_list = np.array(temp_TF_list)
        #     roi.stim_info['epoch_frequency'] = temp_TF_list
        #     roi_dict['TF'] = temp_TF_list[req_epochs]


        # Creating a pandas dataframe for future heat map   
        df_roi = pd.DataFrame.from_dict(roi_dict)
        if epoch_types[1] == 'lumgrating':
            tfl_map = df_roi.pivot(index='TF',columns='luminance')
        elif epoch_types[1] == 'noisygrating':
            tfl_map = df_roi.pivot(index='TF',columns='SNR')
 
        #roi.tfl_map= tfl_map
        #roi.tfl_map_norm=(tfl_map-tfl_map.mean())/tfl_map.std()
        #roi.BF = roi.stim_info['epoch_frequency'][roi.max_resp_idx] 
        
        
        
        conc_trace = []
        for epoch in np.argwhere((roi.stim_info['epoch_frequency'] == 1))[1:]:
            
            conc_trace=np.append(conc_trace,
                                 roi.whole_trace_all_epochs[float(epoch)],axis=0)
        roi.oneHz_conc_resp = conc_trace
        
    return rois
#%% Juan additions

def plot_individual_STRFs(rois,current_exp_ID,save_path,cmap='coolwarm',save=True,properties=['PD','CS','SNR','CSI','reliability']):
    
    max_n=np.array(map(lambda roi : np.max(roi.sta), rois)).max()
    for idx,roi in enumerate(rois):
        prop_dict={}
        for prop in properties:
        # Note: Copy here is required otherwise it will just assign the pointer
        # which is dangerous if you want to use both rois in a script
        # that uses this function.
            try:
                prop_dict[str(prop)] = copy.deepcopy(roi.__dict__[prop])
            except KeyError:
                print('Property:-{pr}- not found... Skipping property for this ROI\n'.format(pr=prop))
                prop_dict[str(prop)]=None
                continue
        plt.close('all')
        fig1,ax1=plt.subplots(ncols=1, nrows=1,figsize=(10,8))
        #ax = ax1.flatten()
        
        sns.heatmap(roi.sta.T, cmap=cmap, ax=ax1, cbar=True,vmax=max_n,
                    center=0)
        ax1.axis('off')
        textstr = '\n'.join((
        r'$\mathrm{PD}=%.2f$' % (prop_dict['PD'], ),
        r'$\mathrm{CS}=%s$' % (prop_dict['CS'], ),
        r'$\mathrm{CSI}=%.2f$' % (prop_dict['CSI'], ),
        r'$\mathrm{SNR}=%.2f$' % (prop_dict['SNR'][0], ),
        r'$\mathrm{Rel}=%.2f$' % (prop_dict['reliability'][0], ),))
        Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=Textprops)

        if save==True:
            os.chdir(save_path)
            try:
                os.mkdir('STRFs')
                os.chdir('STRFs')
            except:
                os.chdir('STRFs')
            finally:
                pass
                    
            f1_n = 'STRFs_%s_%s_PD%s_CS%s' % (idx,current_exp_ID,int(prop_dict['PD']),prop_dict['CS'])
            fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                transparent=False,dpi=300)

def plot_individual_STRFs_with_edges(rois,current_exp_ID,save_path,
                                    cmap='coolwarm',save=True,
                                    properties=['PD','CS','SNR','CSI','reliability','DSI'],
                                    alpha=0.5):
    
    max_n=np.array(map(lambda roi : np.max(roi.sta), rois)).max()
    print('plotting for: %s' %(rois[0].experiment_info['FlyID']))
    for idx,roi in enumerate(rois):
        prop_dict={}
        for prop in properties:
        # Note: Copy here is required otherwise it will just assign the pointer
        # which is dangerous if you want to use both rois in a script
        # that uses this function.
            try:
                prop_dict[str(prop)] = copy.deepcopy(roi.__dict__[prop])
            except KeyError:
                print('Property:-{pr}- not found... Skipping property for this ROI\n'.format(pr=prop))
                prop_dict[str(prop)]=None
                continue
        plt.close('all')
        # prev fig1,ax1=plt.subplots(ncols=1, nrows=1,figsize=(10,8))
        fig1=plt.figure(figsize=(40,12))
        gs = GridSpec(3, 2,width_ratios=np.array([3,2,3]))
        #ax = ax1.flatten()
        period = 1.0 / roi.imaging_info['frame_rate']
        directions=np.unique(np.array(rois[0].prevstim_data['stim_info']['direction']).astype(int))
        for idx,direction in enumerate(directions):
            ax1=fig1.add_subplot(gs[idx,1])
            #direction=roi.prevstim_data['stim_info']['direction'][idx]
            trace=roi.prevstim_data['prev_resp_traces'][idx]
            time_ax= np.arange(len(trace))*period
            ax1.plot(time_ax,trace)
            ax1.set_title('Dir: %s'% (direction))
            ax1.set_ylabel('D'+'f/f')
            ax1.set_xlabel('it{ t }'+ '(s)')
            stim_trace=np.zeros(len(trace))
            stim_trace[len(stim_trace)//2:]=1
            max_val=np.max(trace)
            stim_trace=stim_trace+max_val+0.2
            ax1.plot(time_ax,stim_trace,linestyle='--')
            if idx==0:
                textstr = '\n'.join((
                r'$\mathrm{PD}=%.2f$' % (prop_dict['PD'], ),
                r'$\mathrm{CS}=%s$' % (prop_dict['CS'], ),
                r'$\mathrm{CSI}=%.2f$' % (prop_dict['CSI'], ),
                r'$\mathrm{SNR}=%.2f$' % (prop_dict['SNR'][0], ),
                r'$\mathrm{Rel}=%.2f$' % (prop_dict['reliability'][0], ),
                r'$\mathrm{DSI}=%.2f$' % (prop_dict['DSI'][0], )))
                Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=Textprops)

        ax1=fig1.add_subplot(gs[:,2])
        sns.heatmap(roi.sta.T, cmap=cmap, ax=ax1, cbar=True,vmax=max_n,
                    center=0)
        ax1.axis('off')
        
        ax1=fig1.add_subplot(gs[0,0])
        sns.heatmap(roi.source_image,cmap='gray',ax=ax1,cbar=False)
        sns.heatmap(roi.mask,alpha=alpha,cmap = 'tab20b',ax=ax1,
                    cbar=False)
        ax1.axis('off')

        #plot prediction together with df trace, 

        ax1=fig1.add_subplot(gs[1,0])
        x_axis='aa'
        ax1.plot(roi.raw_trace)
        ax1.plot(roi.sta_prediction)
        #TODO format

        # textstr = '\n'.join((
        # r'$\mathrm{PD}=%.2f$' % (prop_dict['PD'], ),
        # r'$\mathrm{CS}=%s$' % (prop_dict['CS'], ),
        # r'$\mathrm{CSI}=%.2f$' % (prop_dict['CSI'], ),
        # r'$\mathrm{SNR}=%.2f$' % (prop_dict['SNR'][0], ),
        # r'$\mathrm{Rel}=%.2f$' % (prop_dict['reliability'][0], ),))
        # Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
        # verticalalignment='top', bbox=Textprops)

        if save==True:
            os.chdir(save_path)
            try:
                os.mkdir('STRFs')
                os.chdir('STRFs')
            except:
                os.chdir('STRFs')
            finally:
                pass
                    
            f1_n = 'STRFs_ROI%s-%s_%s_PD%s_CS%s' % (roi.uniq_id,idx,current_exp_ID,int(prop_dict['PD']),prop_dict['CS'])
            fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                transparent=False,dpi=300)
        plt.close('all')        
        # 
def summarize_properties_RFs():
    pass
    # 1st: count ON ROIs, OFF ROIs
    # make a distribution of SNR  (based on CS)
    # make a distribution of reliability (based on CS)
    # make a distribution of 
    # make a bar plot of CS and a bar plot of PD        

def plot_traces_freq_epochs(idx,pickle_file_path,save_path,plot_prev=False):
    """
        plot's the traces in the save path for every epoch and every ROI 
        within the given dataset 
        for now, it only works with edges and frequency gratings (given that 
        certain properties are present)
        
    """
    
    load_path = os.path.join(pickle_file_path)
    load_path = open(load_path, 'rb')
    workspace = pickle.load(load_path)
    curr_rois = workspace['final_rois']     
    epoch_types = curr_rois[0].stim_info['stimtype']
    req_epochs = np.array([e == curr_rois[0].stim_info['stimtype'][-1] for e in epoch_types]) # Seb: selecting epochs of interest based on the name of the last epoch in the stimulus input file
    #req_epochs = np.where(req_epochs==True)[0]
    print(curr_rois[0].experiment_info['FlyID'])
    if int(curr_rois[0].stim_info['random']) == 1:
        req_epochs[0] = False # Seb: first epoch is for the baseline, not for analyzing any response
    if curr_rois[0].stim_info['stimtype'][0] != curr_rois[0].stim_info['stimtype'][1]:
        req_epochs[0] = False
    if curr_rois[0].stim_info['stimtype'][0] == 'circle':
        req_epochs[0] = False
    for roi_i,roi in enumerate(curr_rois):
        if roi.stim_info['stimtype'][-1] == 'TFgrating':
            TF=True
            #extract relevant data
            BF=roi.BF
            rel=roi.reliability[1]
            SNR= roi.SNR[1]
        else:
            rel=roi.reliability[0]
            SNR= roi.SNR[0]
            TF=False
        if plot_prev==True:
            rel0=roi.reliability[0]
            SNR0= roi.SNR[0]
            rel1=roi.reliability[1]
            SNR1= roi.SNR[1]
        DSI=roi.DSI
        CSI=roi.CSI
        CS=roi.CS
        PD=roi.PD
        treatment=roi.experiment_info['treatment']

        flyID=roi.experiment_info['FlyID'] 
        flyID=flyID.replace('\\','-')
        dirs=np.array(roi.stim_info['direction'])
        directions=np.unique(dirs[(req_epochs)])
        prefered_direction=roi.PD
        prefered_direction_filter=np.array(roi.stim_info['direction'])== prefered_direction
        epochs_roi_data = roi.resp_trace_all_epochs
        
        if plot_prev==False:
            fig,ax=plt.subplots(nrows=len(directions),ncols=np.sum(prefered_direction_filter),figsize=(30,16))

            if TF:

                textstr = '\n'.join((
                r'$\mathrm{PD}=%.2f$' % (PD, ),
                r'$\mathrm{CS}=%s$' % (CS, ),
                r'$\mathrm{CSI}=%.2f$' % (CSI, ),
                r'$\mathrm{SNR}=%.2f$' % (SNR, ),
                r'$\mathrm{Rel}=%.2f$' % (rel, ),
                r'$\mathrm{BF}=%.2f$' % (BF, )))
                Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax[0,0].text(0.05, 0.95, textstr, transform=ax[0,0].transAxes, fontsize=12,
                verticalalignment='top', bbox=Textprops)
            else:
                textstr = '\n'.join((
                r'$\mathrm{PD}=%.2f$' % (PD, ),
                r'$\mathrm{CS}=%s$' % (CS, ),
                r'$\mathrm{CSI}=%.2f$' % (CSI, ),
                r'$\mathrm{SNR}=%.2f$' % (SNR, ),
                r'$\mathrm{Rel}=%.2f$' % (rel, ),))
                Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=12,
                verticalalignment='top', bbox=Textprops)

            #save_path= os.path.join(save_path, flyID)
            save_path1=save_path + '\\traces_ID%s_%s_PD%s_CS%s.png'%(idx,flyID,PD,CS)


            for idx,curr_direction in enumerate(directions):
            # idx will be used to plot across directios
                direction_filter=np.where(roi.stim_info['direction']== curr_direction)[0]

                for idx1,epoch in enumerate(direction_filter):
                    trace=epochs_roi_data[epoch]
                    if TF == True:
                        frequency=roi.stim_info['epoch_frequency'][epoch]
                    period = 1.0 / roi.imaging_info['frame_rate']
                    time_ax= np.arange(len(trace))*period
                    if TF == True:
                        ax[idx,idx1].plot(time_ax,trace)
                        ax[idx,idx1].set_title('%s Hz dir: %s'%(frequency,curr_direction))
                        ax[idx,idx1].set_ylabel('\u0394'+'f/f')
                        ax[idx,idx1].set_xlabel('it{ t }'+ '(s)')
                    
                    else:
                        ax[idx].plot(time_ax,trace)
                        ax[idx].set_title(' dir: %s'%(curr_direction))
                        ax[idx].set_ylabel('\u0394'+'f/f')
                        ax[idx].set_xlabel('it{ t }'+ '(s)')

        else:
        
            fig,ax=plt.subplots(nrows=len(directions),ncols=np.sum(prefered_direction_filter)+1,figsize=(40,16))

            if TF:

                textstr = '\n'.join((
                r'$\mathrm{PD}=%.2f$' % (PD, ),
                r'$\mathrm{CS}=%s$' % (CS, ),
                r'$\mathrm{CSI}=%.2f$' % (CSI, ),
                r'$\mathrm{SNR 0}=%.2f$' % (SNR0, ),
                r'$\mathrm{Rel 0}=%.2f$' % (rel0, ),
                r'$\mathrm{SNR 1}=%.2f$' % (SNR1, ),
                r'$\mathrm{Rel 1}=%.2f$' % (rel1, ),                
                r'$\mathrm{BF}=%.2f$' % (BF, )))
                Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax[0,0].text(0.05, 0.95, textstr, transform=ax[0,0].transAxes, fontsize=12,
                verticalalignment='top', bbox=Textprops)
            else:
                textstr = '\n'.join((
                r'$\mathrm{PD}=%.2f$' % (PD, ),
                r'$\mathrm{CS}=%s$' % (CS, ),
                r'$\mathrm{CSI}=%.2f$' % (CSI, ),
                r'$\mathrm{SNR}=%.2f$' % (SNR, ),
                r'$\mathrm{Rel}=%.2f$' % (rel, ),))
                Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=15,
                verticalalignment='top', bbox=Textprops)

            #save_path= os.path.join(save_path, flyID)
            save_path1=save_path + '\\traces_roi%sID%s_%s_PD%s_CS%s.png'%(roi.uniq_id,idx,flyID,PD,CS)


            for idx,curr_direction in enumerate(directions):
            # idx will be used to plot across directios
                direction_filter=np.where(roi.stim_info['direction']== curr_direction)[0]
                direction_filter1=np.where(roi.prevstim_data['stim_info']['direction']== curr_direction)[0]
                #TODO evaluate if the following line makes sense
                trace1=roi.prevstim_data['prev_resp_traces'][idx]
                period = 1.0 / roi.imaging_info['frame_rate']
                time_ax1= np.arange(len(trace1))*period
                stim_trace=np.zeros(len(trace1))
                stim_trace[len(stim_trace)//2:]=1
                max_val=np.max(trace1)
                stim_trace=stim_trace+max_val+0.2
                if idx==0:
                    ax[idx,idx].plot(time_ax1,trace1)
                    ax[idx,idx].set_title('edges dir: %s'%(curr_direction),fontsize=15)
                    ax[idx,idx].set_ylabel('D'+'f/f',fontsize=15)
                    ax[idx,idx].set_xlabel('it{ t }'+ '(s)',fontsize=15)
                    ax[idx,idx].plot(time_ax1,stim_trace,linestyle='--',color='gray')

                else:
                    ax[idx,0].plot(time_ax1,trace1)
                    ax[idx,0].set_title('edges dir: %s'%(curr_direction),fontsize=15)
                    ax[idx,0].set_ylabel('D'+'f/f',fontsize=15)
                    ax[idx,0].set_xlabel('it{ t }'+ '(s)',fontsize=15)
                    ax[idx,0].plot(time_ax1,stim_trace,linestyle='--',color='gray')
                for idx1,epoch in enumerate(direction_filter):
                    trace=epochs_roi_data[epoch]
                    if TF == True:
                        frequency=roi.stim_info['epoch_frequency'][epoch]
                    period = 1.0 / roi.imaging_info['frame_rate']
                    time_ax= np.arange(len(trace))*period
                    if TF == True:
                        ax[idx,idx1+1].plot(time_ax,trace)
                        ax[idx,idx1+1].set_title('%s Hz dir: %s'%(frequency,curr_direction),fontsize=15)
                        ax[idx,idx1+1].set_ylabel('D'+'f/f',fontsize=15)
                        ax[idx,idx1+1].set_xlabel('it{ t }'+ '(s)',fontsize=15)
                    
                    # else: #Juan: this seems innecesary here
                    #     ax[idx,idx1+1].plot(time_ax,trace)
                    #     ax[idx,idx1+1].set_title(' dir: %s'%(curr_direction))
                    #     ax[idx,idx1+1].set_ylabel('\u0394'+'f/f')
                    #     ax[idx,idx1+1].set_xlabel('$\it{ t }$'+ '(s)')
        fig.suptitle('%s ID: %s-%s treatment: %s'%(roi.uniq_id,roi.experiment_info['FlyID'],idx,roi.experiment_info['treatment']), fontsize=16)
        #print(roi.experiment_info['FlyID'])
        plt.savefig(save_path1)
        plt.close('all')
    print('plotted %s ROIs'%(roi_i+1))

def plot_tracesEdges_against_variables(rois,save_path):
    # get all traces per epoch and plot them together in dict
    try:
        os.mkdir(save_path + 'aligned_traces')
    except:
        pass
    finally:
        pass
    old=glob.glob(save_path + 'aligned_traces'+'\\*.pdf')
    for entry in old:
        os.remove(entry)
    
    all_traces_rois={}
    
    passed_ROIs=[]
    
    for roi in rois:
        if roi.rejected==False:
            passed_ROIs.append(roi)
    epochs=passed_ROIs[0].interpolated_traces_epochs.keys()
    max_vals_array=np.zeros((len(epochs),len(passed_ROIs)))
    x_variable=np.array(passed_ROIs[0].independent_var_vals)
    x_variable=x_variable[~np.isnan(x_variable)]
    var_name=passed_ROIs[0].independent_var
    for epoch in epochs:
        all_traces_rois[epoch]=np.zeros((len(passed_ROIs[0].interpolated_traces_epochs[epoch]),len(passed_ROIs)))
        for idx,roi in enumerate(passed_ROIs):
            all_traces_rois[epoch][:,idx]=roi.interpolated_traces_epochs[epoch]

    for ix,roi in enumerate(passed_ROIs):
        depend_var=np.concatenate(roi.max_resp_all_epochs)
        max_vals_array[:,ix]= depend_var[~np.isnan(depend_var)]
        
    time_vectors=passed_ROIs[0].interpolated_time
    # make plot scaffold
    fig1 = plt.figure(figsize=(40,12))
    gs = GridSpec(2, int(np.ceil(float(len(epochs))/2)),wspace=0.4,hspace=0.2)
    #plot all traces
    for ix,epoch in enumerate(epochs):
        ax1=fig1.add_subplot(gs[ix])
        ax1.plot(time_vectors[epoch][:,np.newaxis],all_traces_rois[epoch],linewidth=0.5,color='grey')
        #plot mean on top
        ax1.plot(time_vectors[epoch],np.mean(all_traces_rois[epoch],axis=1),color='k',linewidth=1.5)
        plt.title('%s_%s_%spassedRois/%s'%(var_name,x_variable[ix],len(passed_ROIs),len(rois)))
        ax1.set_ylim([-1.5,np.max(max_vals_array)+1])
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('Df/f')
    # put max values per epoch in an array
    #first sort the values
    inds=np.argsort(x_variable)
    fig2= plt.figure()
    plt.plot(x_variable[inds],max_vals_array[inds],linewidth=0.5,color='grey')
    plt.plot(x_variable[inds],np.mean(max_vals_array[inds],axis=1),linewidth=1.5,color='k',marker='o',markersize=8)
    yerr = np.std(max_vals_array[inds],axis=1) / np.sqrt(max_vals_array.shape[1])
    plt.errorbar(x_variable[inds],np.mean(max_vals_array[inds],axis=1),yerr,color='k',)
    plt.xlabel(passed_ROIs[0].independent_var)
    plt.ylabel('mean Df/f')
    plt.title ('tunning curve : %s %srois/ %s' %(passed_ROIs[0].independent_var,len(passed_ROIs),len(rois)))
    save_path=save_path + 'aligned_traces' +'\\var_%s.pdf' %(var_name)
    multipage(save_path, figs=None, dpi=200)
    #save the results

def plot_psds(rois,save_path,multiple=True):
    plt.close('all')
    try:
        os.mkdir(save_path + 'psd')
    except:
        pass
    finally:
        pass
    old=glob.glob(save_path + 'psd'+'\\*.pdf')
    for entry in old:
        os.remove(entry)
    #first plot the PSDs
    passed_ROIs=[]
    for roi in rois:
        if roi.rejected==False:
            passed_ROIs.append(roi)
    if multiple==True:
        epochs=passed_ROIs[0].interpolated_traces_epochs.keys()
        max_vals_array=np.zeros((len(epochs),len(passed_ROIs)))

        
        var_name=passed_ROIs[0].independent_var
        fig1=plt.figure()
        gs = GridSpec(2, (len(epochs)),wspace=0.4,hspace=0.2)
        #plot all traces
        for ix,epoch in enumerate(epochs):
            freqs=passed_ROIs[0].signal_psd_dict[epoch]['freqs']
            ind1=[freqs>=0]
            freqs=freqs[ind1[0]]
            ind2=[freqs<5]
            freqs=freqs[ind2[0]]
            psds=np.zeros((len(passed_ROIs),len(freqs)))
            stim_psds=np.zeros((len(passed_ROIs),len(freqs)))
            x_variable=roi.independent_var_vals
            tunning_xvals=passed_ROIs[0].power_measured_at
            tunning_yvals=np.zeros((len(passed_ROIs),len(passed_ROIs[0].power_measured_at)))
            tunning_yvals[:]=np.nan
            for idx,roi in enumerate(passed_ROIs):
                Temp=roi.signal_psd_dict[epoch]['psd'][ind1[0]]
                Temp=Temp[ind2[0]]
                psds[idx,:]=Temp
                temp2=roi.stim_psd_dict[epoch]['psd'][ind1[0]]
                temp2=temp2[ind2[0]]
                stim_psds[idx,:]=temp2
                tunning_yvals[idx,:]=roi.freq_power_epochs
            ax1=fig1.add_subplot(gs[0,ix])
            ax2=fig1.add_subplot(gs[1,ix])
            ax1.plot(freqs[:,np.newaxis],np.transpose(psds),linewidth=0.5,color='grey')
            ax1.plot(freqs,np.mean(psds,axis=0),color='k',linewidth=1.5)
            ax2.plot(freqs[:,np.newaxis],np.transpose(stim_psds),linewidth=0.5,color='grey')
            ax2.plot(freqs,np.mean(stim_psds,axis=0),color='k',linewidth=1.5)
            #ax2.set_xticks([range(4)])
            ax1.set_title('frequency_%s_passedRois%s'%(x_variable[epoch],len(passed_ROIs)))
            ax2.set_title('frequency_%s_stim'%(x_variable[epoch]))
            #ax1.set_xticks([range(4)])
            #ax1.set_ylim([-0.5,np.max(max_vals_array)+1])
            ax1.set_xlabel('frequencies')
            ax1.set_ylabel(' fourier transform (/Hz)')
        multipage(save_path + '\\fft_plots_perepoch.pdf')
        plt.close('all')

        #plot tunning curve for frequencies
        # 
        tunning_xvals=tunning_xvals[1:]
        tunning_yvals=tunning_yvals[:,1:]
        plt.figure()
        plt.plot(tunning_xvals[:,np.newaxis],np.transpose(tunning_yvals),linewidth=0.5,color='grey')     
        plt.plot(tunning_xvals[:,np.newaxis],np.mean(tunning_yvals,axis=0),linewidth=1.5,color='k',marker='o',markersize=8)
        yerr = np.std(tunning_yvals,axis=0) / np.sqrt(tunning_yvals.shape[1])
        plt.errorbar(tunning_xvals,np.mean(tunning_yvals,axis=0),yerr,color='k')
        plt.title('tunning curve based on freq components')
        plt.ylabel('FFT value /lenght Hz')
        plt.xlabel('frequency (Hz)')
        multipage(save_path + '\\FFT_freq_tunning.pdf')

        # you need a second gridspec for this. where you plot the psd values at specific freqs
        # but maybe this is sorta irrelevant here. 
    else:
    
        freqs=passed_ROIs[0].signal_psd_dict['freqs']
        psds=np.zeros((len(passed_ROIs),len(freqs)))
        psds_stim=np.zeros((len(passed_ROIs),len(freqs)))
        for idx,roi in enumerate(passed_ROIs):
            psds[idx,:]=roi.signal_psd_dict['psd']
            psds_stim[idx,:]=roi.stim_psd_dict['psd']
        psd_mean=np.mean(psds,axis=0)
        psd_stim_mean=np.mean(psds_stim,axis=0)
        fig1= plt.figure()
        gs = GridSpec(1,2,wspace=0.4,hspace=0.2)
        ax1=fig1.add_subplot(gs[0])
        ax2=fig1.add_subplot(gs[1])  

        ax1.plot(freqs, np.transpose(psds),color='grey',linewidth=0.5)
        ax1.plot(freqs, psd_mean,color='k',linewidth=1.5)
        ax1.set_xlim([0, 5])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD (/Hz)') #power spectral density
        ax1.set_title ('power spectral density %s / %s rois window %s ' %(len(passed_ROIs),len(rois),roi.signal_psd_dict['window']))
        
        ax2.plot(freqs, psd_stim_mean,color='k',linewidth=1.5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD (/Hz)') #power spectral density
        ax2.set_title ('power spectral density stim ')
        ax2.set_xlim([0, 5])

    # on second place, plot the extracted power vs variable, sort of a second tunning curve
    x_variable=passed_ROIs[0].independent_var_vals
    max_vals_array=np.zeros((len(passed_ROIs),len(x_variable)))
    max_vals_stim=np.zeros((len(passed_ROIs),len(x_variable)))
    for idx,roi in enumerate(passed_ROIs):
        max_vals_array[idx,:]=roi.freq_power_epochs
        max_vals_stim[idx,:]=roi.freq_power_stims
    x_variable=x_variable[~np.isnan(x_variable)]
    max_vals_array=max_vals_array[:,1:]
    max_vals_stim=max_vals_stim[:,1:]
    max_vals_divided=max_vals_array/max_vals_stim
    inds=np.argsort(x_variable)
    
    fig2= plt.figure()
    gs = GridSpec(1,3,wspace=0.4,hspace=0.2)
    #plot all traces
    ax1=fig2.add_subplot(gs[0])
    ax2=fig2.add_subplot(gs[1])
    ax3=fig2.add_subplot(gs[2])

    ax1.plot(x_variable[inds],np.transpose(max_vals_array[:,inds]),linewidth=0.5,color='grey')
    ax1.plot(x_variable[inds],np.mean(max_vals_array[:,inds],axis=0),linewidth=1.5,color='k',marker='o',markersize=8)
    yerr = np.std(max_vals_array[:,inds],axis=0) / np.sqrt(max_vals_array.shape[1])
    ax1.errorbar(x_variable[inds],np.mean(max_vals_array[:,inds],axis=0),yerr,color='k')
    ax1.set_xlabel(passed_ROIs[0].independent_var + '(Hz)')
    ax1.set_ylabel('Fourier component (/lenght Hz')
    ax1.set_title ('tunning curve : %s %srois/ %s' %(passed_ROIs[0].independent_var,len(passed_ROIs),len(rois)))
    
    ax2.plot(x_variable[inds],np.mean(max_vals_stim[:,inds],axis=0),linewidth=1.5,color='k',marker='o',markersize=8)
    ax2.set_xlabel(passed_ROIs[0].independent_var + 'Hz')
    ax2.set_ylabel('Fourier component (/lenght Hz)')
    ax2.set_title ('psd curve of stim at analyzed freqs')
    ax2.set_ylim([0,5])
    
    ax3.plot(x_variable[inds],np.transpose(max_vals_divided[:,inds]),linewidth=0.5,color='grey')
    ax3.plot(x_variable[inds],np.mean(max_vals_divided[:,inds],axis=0),linewidth=1.5,color='k',marker='o',markersize=8)
    yerr = np.std(max_vals_divided[:,inds],axis=0) / np.sqrt(max_vals_divided.shape[1])
    ax3.errorbar(x_variable[inds],np.mean(max_vals_divided[:,inds],axis=0),yerr,color='k',)
    ax3.set_xlabel(passed_ROIs[0].independent_var + '(Hz)')
    ax3.set_ylabel('Fourier component (/lenght Hz')
    ax3.set_title ('tunning curve : %s %srois/ %s' %(passed_ROIs[0].independent_var,len(passed_ROIs),len(rois)))


    save_path=save_path + '\\psd' +'\\psd_var_%s.pdf' %(passed_ROIs[0].independent_var)
    multipage(save_path)

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def plot_traces_edges(Rois,save_path):

    try:
        os.mkdir(save_path + 'mean_raw_traces')
    except:
        pass
    finally:
        pass
    old=glob.glob(save_path + 'mean_raw_traces'+'\\*.png')
    for entry in old:
        os.remove(entry)
    # load_path = os.path.join(pickle_file_path)
    # load_path = open(load_path, 'rb')
    # workspace = cPickle.load(load_path)
    # curr_rois = workspace['final_rois']
    curr_rois=Rois
    valid_epochs=np.array(curr_rois[0].stim_info['stimtype'])=='driftingstripe'
    if np.sum(valid_epochs)==0:
        valid_epochs=np.array(curr_rois[0].stim_info['stimtype'])=='ADS'        
    if np.sum(valid_epochs)==0:
        valid_epochs=np.array(curr_rois[0].stim_info['stimtype'])=='G'
        number_epochs=np.sum(valid_epochs)
        directions=np.array(curr_rois[0].direction_vector)[valid_epochs]
    elif np.sum(valid_epochs)==1: 
        directions=np.array(curr_rois[0].stim_info['angle'])[valid_epochs]
    else:
        number_epochs=np.sum(valid_epochs)
        directions=np.array(curr_rois[0].direction_vector)[valid_epochs]
    if np.sum(valid_epochs)==0:
        print('no traces plotted, no valid epochs')
        return 

    plt.style.use('seaborn-v0_8-talk')
    
    for iroi,roi in enumerate(curr_rois):
        if np.sum(valid_epochs)==1: 
            roi.direction_vector=directions
        #for idx,curr_direction in enumerate(directions):
        #find epoch location
        #direction_filter=np.where(np.array(roi.direction_vector)== curr_direction)[0]
        plt.close('all')
        # get ROI mask image
        # extract number of epochs to plot and set Gridsepc
       
        number_of_epochs=sum((np.array(roi.stim_info['stimtype'])=='driftingstripe'))
        epochs=np.where((np.array(roi.stim_info['stimtype'])=='driftingstripe'))[0]
        if number_of_epochs==0:
            number_of_epochs=sum((np.array(roi.stim_info['stimtype'])=='ADS'))
            epochs=np.where((np.array(roi.stim_info['stimtype'])=='ADS'))[0]         
        fig1=plt.figure(figsize=(40,12))
        if number_of_epochs>1:
            gs = GridSpec(2, int(round(number_epochs/2)+2),wspace=0.4,hspace=0.2)
        else:
            gs = GridSpec(2,2,wspace=0.4,hspace=0.2)
        fig1.patch.set_facecolor('white')
        ax1=fig1.add_subplot(gs[0,0:2])

        # plot roi in pos 0,0
        mean_image=roi.source_image  
        mask=roi.mask 
        sns.heatmap(mean_image,cmap='gray',ax=ax1,cbar=False)
        mask=np.where(mask==False,np.nan,mask)
        sns.heatmap(mask,alpha=0.2,cmap = 'Set1',ax=ax1,
            cbar=False)
        
        #plt.style.use("dark_background")

        ax1.axis('off')
        ax1.set_title('ROIs n=%d' % iroi)

        


        # plot vector in pos 0,1
        if np.sum(valid_epochs)>1:
            ax1 = fig1.add_subplot(gs[1,0],polar=True)
            PD_ON=roi.PD_ON
            PD_OFF=roi.PD_OFF
            DSI_ON=roi.DSI_ON
            DSI_OFF=roi.DSI_OFF
            color= 'black'
            
            if 'A' in list(roi.category[0])[0]:
                color='tab:green'
            elif 'B' in list(roi.category[0])[0]:
                color='tab:blue'
            elif 'C' in list(roi.category[0])[0]:
                color='tab:red'
            elif 'D' in list(roi.category[0])[0]:
                color='gold'
            else:
                color= 'black'

            ax1.quiver(0,0, np.radians(PD_ON),DSI_ON,color=color,
            scale=1,angles="xy",scale_units='xy',alpha=1.)
            ax1.quiver(0,0, np.radians(PD_OFF),DSI_OFF,color='gray',
            scale=1,angles="xy",scale_units='xy',alpha=0.3)
            ax1.set_ylim(0,1.2)
            ax1.set_xlabel(' DSI_ON=%.2f, DSI_OFF=%.2f,PD_ON=%.2f, PD_OFF=%.2f' % (DSI_ON,DSI_OFF,PD_ON,PD_OFF))
            ax1.set_yticks([0.5,1])
            #plot response to every direction in the polar plot normalized to the sum of all responses
            #resp_ON=np.array(map(lambda ix:roi.max_resp_all_epochs_ON[ix],range(0,len(roi.max_resp_all_epochs_ON))))
            #resp_OFF=np.array(map(lambda ix:roi.max_resp_all_epochs_OFF[epoch],range(0,len(roi.max_resp_all_epochs_ON))))
            roi.direction_vector=np.array(roi.direction_vector)

            ax1=fig1.add_subplot(gs[1,1], polar=True)
            #ax1.set_ylim(0,1)
            #ax1.set_xlabel('DSI_ON=%.2f, DSI_OFF=%.2f,PD_ON=%.2f, PD_OFF=%.2f' % (DSI_ON,DSI_OFF,PD_ON,PD_OFF))
            #ax1.set_yticks([1,2,4])
            resp_ON=np.array(list(map(lambda ix: roi.max_resp_all_epochs_ON[ix],np.arange(0,len(roi.max_resp_all_epochs_ON)))))
            resp_OFF=np.array(list(map(lambda ix: roi.max_resp_all_epochs_OFF[ix],np.arange(0,len(roi.max_resp_all_epochs_OFF)))))
            directions=np.sort(roi.direction_vector)
            sort_ix=np.argsort(roi.direction_vector)
            resp_ON=resp_ON[sort_ix]
            resp_ON=np.append(resp_ON,resp_ON[0])
            resp_ON=np.where(resp_ON<0,resp_ON*0,resp_ON)
            resp_ON=resp_ON#/np.max(resp_ON)
            resp_OFF=resp_OFF[sort_ix]
            resp_OFF=np.append(resp_OFF,resp_OFF[0])
            resp_OFF=np.where(resp_OFF<0,resp_OFF*0,resp_OFF)
            resp_OFF=resp_OFF#/np.max(resp_OFF)
            directions=np.append(directions,360)
            #convert directions to radians
            directions=np.radians(directions)
            ax1.plot(directions,resp_ON,color=color)
            ax1.plot(directions,resp_OFF,color='gray',alpha=0.3)
            #labels and plot parameters
            #put ROI summary info in the plot

        period = 1.0 / roi.imaging_info['frame_rate']
        counter=1
        subplot_loc1=0
        if np.sum(valid_epochs)>1:
            for iepoch,epoch in enumerate (epochs):
                counter=counter+1
                if counter==len(epochs)//2+2:
                    counter=2
                    subplot_loc1=1
                curr_direction=roi.direction_vector[iepoch]
                subplot_loc2=counter
                ax1=fig1.add_subplot(gs[subplot_loc1,subplot_loc2],polar=False)
                ax1.set_ylim(-0.5,8)
                trace=roi.resp_trace_all_epochs[epoch]
                time_ax= np.arange(len(trace))*period
                ax1.plot(time_ax,trace)
                ax1.set_title('Dir: %s'% (curr_direction))
                ax1.set_ylabel('D'+'f/f')
                if subplot_loc1==1:
                    ax1.set_xlabel('it{ t }'+ '(s)')
                stim_trace=np.zeros(len(trace))
                stim_trace[len(stim_trace)//2:]=1
                max_val=np.max(trace)
                stim_trace=stim_trace+5
                ax1.plot(time_ax,stim_trace,linestyle='--')
        else:
            counter=counter+1
            if counter==len(epochs)//2+2:
                counter=2
                subplot_loc1=1
            curr_direction=roi.direction_vector[0]
            subplot_loc2=counter
            ax1=fig1.add_subplot(gs[1,0],polar=False)
            ax1.set_ylim(-0.5,4)
            trace=roi.resp_trace_all_epochs[1]
            time_ax= np.arange(len(trace))*period
            ax1.plot(time_ax,trace)
            ax1.set_title('Dir: %s'% (curr_direction))
            ax1.set_ylabel('D'+'f/f')
            if subplot_loc1==1:
                ax1.set_xlabel('it{ t }'+ '(s)')
            stim_trace=np.zeros(len(trace))
            stim_trace[len(stim_trace)//2:]=1
            max_val=np.max(trace)
            stim_trace=stim_trace+5
            ax1.plot(time_ax,stim_trace,linestyle='--')
        
        if np.sum(valid_epochs)>1:   
            textstr = '\n'.join((
            r'$\mathrm{PD_ON}=%.2f$' % (roi.PD_ON, ),
            r'$\mathrm{PD_OFF}=%.2f$' % (roi.PD_OFF, ),
            r'$\mathrm{CS}=%s$' % (roi.CS, ),        
            r'$\mathrm{CSI}=%.2f$' % (roi.CSI, ),
            #r'$\mathrm{SNR}=%.2f$' % (roi.SNR[0], ),
            r'$\mathrm{Rel_ON}=%.2f$' % (roi.reliability_PD_ON, ),
            r'$\mathrm{Rel_OFF}=%.2f$' % (roi.reliability_PD_OFF,),
            r'$\mathrm{traces_sifted}=%.2f$' % (False, ))) # this is just to indicate that the real analisis is done on a version
                                                        # of the traces that is shfited 0.45 seconds to the left.
            Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=Textprops)
        else:
            textstr = '\n'.join((
            #r'$\mathrm{PD_ON}=%.2f$' % (roi.PD_ON, ),
            #r'$\mathrm{PD_OFF}=%.2f$' % (roi.PD_OFF, ),
            r'$\mathrm{CS}=%s$' % (roi.CS, ),        
            r'$\mathrm{CSI}=%.2f$' % (roi.CSI, ),
            #r'$\mathrm{SNR}=%.2f$' % (roi.SNR[0], ),
            r'$\mathrm{Rel_ON}=%.2f$' % (roi.reliability_PD_ON, ),
            r'$\mathrm{Rel_OFF}=%.2f$' % (roi.reliability_PD_OFF,),
            r'$\mathrm{traces_sifted}=%.2f$' % (False, ))) # this is just to indicate that the real analisis is done on a version
                                                        # of the traces that is shfited 0.45 seconds to the left.
            Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=Textprops)
        try:
            save_str=f"{save_path}/mean_raw_traces/roi_{iroi}_{roi.category[0]}_averaged_traces"
        except:
            save_str=f"{save_path}/mean_raw_traces/roi_{iroi}_no_cat_averaged_traces"
        finally:
            pass
        plt.savefig(save_str)
        plt.close()
    #plot traces in remaining positions


        # for n_,epoch in enumerate(roi.resp_trace_all_epochs.keys()):
        #     if req_epochs[n_] == False:
        #         continue
        #     if stim_info['stimtype'][n_]=='gratings_TF': #TODO check if correct name
        #         current_freq= roi.stim_info['epoch_frequency']
                
                #TODO create folder for roi
                #TODO put epoch_value in name 
                #TODO make plots for the number of directions (include pref dir, PD and DSI labels)
                
            #take stim trace of epoch
            #take trace of epoch
            #plot epoch
            #save plot with value of epoch in name

    #take the epoch divided traces from an roi. plot in an roi per roi basis
def create_data_matrix(rois, save_path):
    df_list = []
    df_dirs= rois[0].stim_info['output_data_downsampled'][['theta','interlude']]
    
    for roi in rois:
        # Create a DataFrame initialized with 'df_trace'
        
        df = pd.DataFrame({'data_roi_%s'%(roi.uniq_id): roi.df_trace,'frame':range(1,len(roi.df_trace)+1)})
        df=df.set_index('frame')
        df_list.append(df)

    # Concatenate all DataFrames together
    df_final = pd.concat(df_list, axis=1)
    df_final=df_final.join(df_dirs)
    # Save DataFrame to CSV
    name1,name2=save_path.split('\\')[-1],save_path.split('\\')[-2]
    save_path_o=save_path+'\\DatMat_%s_%s_.csv'%(name1,name2)
    df_final.to_csv(save_path_o)

    return df_final

def compute_and_plot_psd(signal_trace, sampling_frequency, target_frequencies, duration,window):
    # Compute the frequencies and PSD using Welch's method
    #window=1000#int(window)
    #frequencies, psd = signal.welch(signal_trace, fs=sampling_frequency,window='blackman',noverlap=0, nperseg=duration*sampling_frequency,detrend='linear')
    
    frequencies = np.fft.fftfreq(len(signal_trace), 1/sampling_frequency)
    fft_vals = np.fft.fft(signal_trace)
    psd = np.abs(fft_vals) #just normal fourier values ** 2) #psd normalized by number of samples
    psd = psd/(len(signal_trace)/sampling_frequency)# the aim here is to normalize 
    #signal_trace.tofile(r'C:\\Users\\vargasju\\PhD\\experiments\\2p\\aaa.csv',sep=',')
    # Plot the PSD in the range of 0-5Hz
    fig1=plt.figure()
    gs = GridSpec(1, 2,wspace=0.4,hspace=0.2)
    ax1=fig1.add_subplot(gs[0])
    ax2=fig1.add_subplot(gs[1])
    ax1.plot(frequencies[frequencies>=0], psd[frequencies>=0],color='blue')
    #plt.xlim([0, 5])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (/Hz)') #power spectral density
    ax1.set_title('psd %s Hz'%(target_frequencies))

    ax2.plot(np.array(range(len(signal_trace)))/sampling_frequency,signal_trace,color='blue')
    #plt.xlim([0, 5])
    ax2.set_xlabel('time (S)')
    ax2.set_ylabel('df/f') #power spectral density
    ax2.set_title('signal %s Hz'%(target_frequencies))

    target_psd=np.zeros(len(target_frequencies))
    analized_freq=np.zeros(len(target_frequencies))
    deltaf_epoch=np.zeros(len(target_frequencies))
    for ix,target_frequency in enumerate(target_frequencies):
        if np.isnan(target_frequency):
            analized_freq[ix] = np.nan
            # Extract the PSD at the target frequency
            target_psd[ix] = np.nan
            deltaf_epoch[ix] = np.nan
            continue
        # Find the closest frequency in the computed frequencies to the target frequency
        target_index = np.abs(frequencies - target_frequency).argmin()
        analized_freq[ix]=frequencies[target_index]
        deltaf_epoch[ix]=frequencies[1]
        # Extract the PSD at the target frequency
        target_psd[ix] = psd[target_index]

        #print('The PSD at %sHz is %s '%(target_frequency,target_psd))

    return {'freqs':frequencies,'psd':psd,'window':window,'detrend':'linear'},target_psd,analized_freq,deltaf_epoch


def generate_sinusoid_fromstim(stim, frequencies, fps, multiple=True):
    # Initialize the grating signal with the same length as the stimulus
    stim_grating = np.zeros_like(stim, dtype=float) + 0.5

    # For each unique value in the stimulus (excluding 0)
    for value in np.unique(stim):
        if value != 0:
            # Get the frequency for this value
            frequency = frequencies[int(value)]

            # Generate a sinusoid at this frequency
            sinusoid = 0.5 * np.sin(2 * np.pi * frequency * np.arange(len(stim)) / fps) + 0.5

            coord1=np.where(np.diff((stim == value).astype(int))>0)[0]
            coord2=np.where(np.diff((stim == value).astype(int))<0)[0]
            # Add the sinusoid to the grating signal at the positions where the stimulus equals this value
            
            for ix in range(len(coord1)):
                if ix==len(coord1)-1 and len(coord1)>len(coord2):
                    stim_grating[coord1[ix]:]=sinusoid[:len(stim_grating[coord1[ix]:])]
                else:
                    stim_grating[coord1[ix]:coord2[ix]-1] = sinusoid[:len(stim_grating[coord1[ix]:coord2[ix]-1])]
        if value ==0 and multiple==False:
            frequency = frequencies[int(value)]
            stim_grating = 0.5 * np.sin(2 * np.pi * frequency * np.arange(len(stim)) / fps) + 0.5

    return stim_grating
# %%

def interpolation_alingment_epochs(rois,required_epochs,reliability_filter=None,CSI_filter=None,int_rate=15.5,grating=False):
    plt.close('all')
    for roi in rois:
        if roi.rejected==True:
            continue
        else:
            roi.interpolated_traces_epochs={} # this dict is to be filled up with mean traces for the epoch 
            roi.interpolated_time={}
            
            for idx,  epoch in enumerate(required_epochs):
                    stim_duration=roi.stim_info['duration'][epoch]
                    roi.interpolated_time[epoch],roi.interpolated_traces_epochs[epoch]=\
                            interpolate_signal(roi.resp_trace_all_epochs[epoch],
                                    roi.imaging_info['frame_rate'],
                                    int_rate,stim_duration=stim_duration)
                        
                # after interpolation, align the interpolated traces for future plotting        
    align_traces(rois,grating=grating) 

def apply_filters(rois,ori_of_stim,cycle,CSI_filter,reliability_filter,position_filter,direction_filter):
        # put a label on rois that should be rejected based on filters 
        
    for roi in rois:       
        if cycle==0:
            # reliability_val=roi.reliability_PD_ON
            
            CSI_val=roi.CSI
        else:
            try:
                # reliability_val=roi.cycle1_reliability['reliability_PD_ON']
                CSI_val=roi.CSI
            except:
                raise Exception ('no reliability or CSI value from cycle0 available for filtering and performing analysis')
        #print(reliability_val)
        # if reliability_filter is not None:
        #     if reliability_val>reliability_filter:
        #         roi.rejected=False                
        #     else:
        #         roi.rejected=True
        #         continue
        if CSI_filter is not None:
            if CSI_val>CSI_filter and reliability_filter is None:
                roi.rejected=False                
            # elif CSI_val>CSI_filter and reliability_filter is not None and reliability_val>reliability_filter:
            #     roi.rejected=False
            else:
                roi.rejected=True
                continue
        if cycle==0:
            if direction_filter==True:
                try:
                    ori_of_stim
                except: 
                    raise Exception ('no predefined orientation of stim')
                if ori_of_stim!= roi.dir_max_resp:
                    roi.rejected=True
                    continue
                else:
                    roi.rejected=False                
        else:
            if direction_filter==True:
                if len(np.unique(roi.stim_info['angle']))!=1:
                    raise Exception('more than one direction in stim')
                else:
                    if np.unique(roi.stim_info['angle'])[0]!= roi.dir_max_resp:
                        roi.rejected=True
                        continue
                    else:
                        roi.rejected=False
        if position_filter==True:
            if roi.Center_position_filter==True:
                roi.rejected=False
            else:
                roi.rejected=True
                continue
#def compute_local_mean(stim, signal, frames,frame_rate):
    # frames=frames-1
    # # Initialize the output array with zeros
    # loc_mean = np.zeros_like(signal)
    # # Initialize mean_value variable
    # start_index = 0
    # check=0
    # first=0
    # # Iterate over the stim array
    # for i in range(len(stim)):
    #     # Check if the current value in stim is zero
    #     if stim[i] != 0 and check==0:
    #         # Compute the index of the middle of the current stretch of zeros
    #         mid_index = start_index + int(round(0.5*frame_rate)) #(i - start_index) // 4 # exclude the first quarter of the trace, to avoid including flash responses
    #         signal_indices = frames[start_index:i]
    #         calculation_indices=frames[mid_index:i]
    #         check=1
    #         if first==0:
    #             mean_value = np.mean(signal[start_index:signal_indices[0]])
    #             loc_mean[0:signal_indices[0]] = mean_value
    #             first=1
    #             start_index=i
    #     elif i<len(stim)-1 and stim[i+1] == 0 and check==1 and stim[i]!=0:
    #         mean_value = np.mean(signal[signal_indices])
    #         # Get the corresponding indices in the signal array
    #         loc_mean[signal_indices[0]:i] = mean_value
    #         # Update the start index for the next stretch of zeros
    #         start_index = i     
    #         check=0        
    #     if i==len(stim)-1 and stim[i]==0:
    #         signal_indices=frames[start_index:]
    #         mean_value = np.mean(signal[signal_indices])
    #         # Get the corresponding indices in the signal array
    #         loc_mean[signal_indices] = mean_value
    #         # Update the start index for the next stretch of zeros
    #     elif i==len(stim)-1 and stim[i]!=0:
    #         loc_mean[i]=loc_mean[i-1]


def mean_of_initial5secs(signal, frames):
    signal_l=copy.deepcopy(signal)
    signal_0=signal_l[:frames[0]-1]
    signal_0[:]=np.mean(signal_0)
    signal_1=signal_l[frames[0]-1:]

    # Initialize the output array with zeros
    #loc_mean = np.zeros_like(signal_1)
    # Find the indices where stim changes from non-zero to zero
    #down_change_indices = np.where(np.diff((stim).astype(int)) <0)[0]
    #up_change_indices = np.where(np.diff((stim).astype(int)) >0)[0]
    # Initialize the start index of the current stretch of zeros
    #start_index = 0
    # Iterate over the change indices
    #for ix in range(len(up_change_indices)):
        # Compute the index of the middle of the current stretch of zeros
    #    mid_index = start_index + int(round(1.5*framerate)) # here 1.5 is as an interval during the the interstim duration where the baseline is not yet meant ot be calculated

        # Get the corresponding indices in the signal array
        #signal_indices = frames[mid_index:up_change_indices[ix]]

        # Compute the mean of the signal over the second half of the current stretch of zeros
     #   mean_value = np.mean(signal[mid_index:up_change_indices[ix]])

        # Set the corresponding values in the loc_mean array
    #    if len(down_change_indices)<len(up_change_indices) and ix==range(len(down_change_indices))[-1]:
    #        loc_mean[start_index:] = mean_value
     #   elif ix==range(len(up_change_indices))[-1]:
    ##        loc_mean[start_index:] = np.mean(signal_1[start_index:])
    #    else:
    #        loc_mean[start_index:down_change_indices[ix]] = mean_value
    #        start_index = down_change_indices[ix] 
        # Update the start index for the next stretch of zeros
            
    return np.mean(signal_0)#np.concatenate([signal_0,loc_mean])

def compute_local_means(stim, trace, use_second_half=False,mean_=True):
    """
    Compute the local means array based on the stim and trace arrays.

    Parameters:
    - stim (list[int]): The stim array.
    - trace (list[float]): The trace array.
    - use_second_half (bool, optional): If True, the mean is computed using only the second half of the zeros epoch.
                                        If False, the mean is computed using the entire epoch of zeros.
                                        Defaults to False.

    Returns:
    - local_means (list[float]): The computed local means array.
    """
    
    local_means = np.zeros_like(trace)
    unit_start = 0

    for i in range(1, len(stim)):
        if stim[i] != 0 and stim[i-1] == 0:
            zero_indices = [j for j in range(unit_start, i) if stim[j] == 0]
            
            if use_second_half:
                start_idx = zero_indices[len(zero_indices) // 2]
            else:
                start_idx = zero_indices[0]
            if mean_==True:
                current_mean = np.mean(trace[start_idx:i])
            else:
                current_mean = np.min(trace[start_idx:i])
        if stim[i] == 0 and stim[i-1] != 0:
            local_means[unit_start:i] = current_mean
            unit_start = i

    if i == len(stim)-1:
        local_means[unit_start:] = current_mean
    
    return local_means

def append_RF_test_trace(rois,p_path):
    
    """ takes the information from a frozen stim analysis and joins it with the correct roi
        
    """
        
    os.chdir(p_path +'\\processed')

    flyID = rois.experimental_info['flyID']
    for folder_ in os.listdir():
        if flyID in folder_:
            # open pickle add the stuff
            os.chdir(folder_)
            for root,dirs,pick in os.walk(os.getcwd()):
                if 'random_moving_WN' in pick:
                    #load_pickle
                    'bla'
        os.chdir(p_path +'\\processed')
    
    for orig_roi in prev_rois:
        for curr_roi in rois:
            #test if the id is the same 
            # if yes, append the new info
            'bla'

    return None

def scramble_spatialaxes(arr,roi, seed=None):
    """
    Scramble the elements of a 3D NumPy array independently along axis 0 for each (j, k).

    Parameters:
    arr (np.ndarray): A 3D NumPy array with shape (d0, d1, d2).
    seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    np.ndarray: A new 3D NumPy array with the same shape as `arr`, 
               with elements along axis 0 shuffled independently for each (j, k).
    """
    if arr.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")
    
    if seed is not None:
        np.random.seed(seed)

    # get the mask of the relevant area of the RF
    #circ_mask = np.squeeze(apply_circular_mask(arr,roi,15,only_mask=True))
    mask = np.ones((arr.shape[1],arr.shape[2]))
    d1, d2 = np.where(mask==1)#arr.shape
        
    # Generate a single permutation for axis 1 and axis 2
    perm1 = np.random.permutation(d1)
    perm2 = np.random.permutation(d2)

    # Apply the permutations to axes 1 and 2 consistently across axis 0
    scrambled = np.zeros_like(arr)
    scrambled[:,d1,d2] = arr[:, perm1, perm2]
    #scrambled = scrambled[:, :, perm2]
    
    return scrambled

def predict_w_scrambled_trace(rois,stimpath,test_frames = [[0,1200],[3600,4800],[7200,8400],[10800,12000]],t_window = 1.5):

    ''' temporary function used to make trace predictions using STRF reconstructions when the 
        rest of the analysis is already done'''
    
    stimtype = rois[0].stim_name.split('.')[0]
    stim_dataframe = pd.DataFrame(rois[0].stim_info['output_data'], columns = ['entry','rel_time','boutInd','epoch','delay_count','delay','stim_frame','mic_frame'])
    stim_dataframe = stim_dataframe[['stim_frame','rel_time','mic_frame','delay']]
    if np.all(stim_dataframe['stim_frame']==0): # some flies in the grating stim have a damaged outfile
        stim_dataframe['stim_frame'] = np.array(list(stim_dataframe.index))
    
    # get the stimulus information
    #rois[0].stim_info
    #freq = stim_up_rate # The update rate of stimulus frames is 20Hz
    stim_up_rate = 1/float(rois[0].stim_info['frame_duration'])
    snippet = int(t_window*stim_up_rate)

    # check if the stimulus manages to cover all the signal length. (in some test the stim can be shorter)
    starting_signal_frame=stim_dataframe.iloc[0]['mic_frame'] 
    stim_dataframe['mic_frame'] = stim_dataframe['mic_frame'] - starting_signal_frame
    stim_dataframe['rel_time'] = stim_dataframe['rel_time'] - stim_dataframe['rel_time'].iloc[0]
    
    last_stim_frame_presented = int(np.max(stim_dataframe['stim_frame'])) #int(stim_dataframe.iloc[-1]['stim_frame'])
    print('max_stim_frame %s' %(last_stim_frame_presented))

    fps = rois[0].imaging_info['frame_rate']
    initial_frame = int(stim_dataframe.loc[stim_dataframe['stim_frame']==0].iloc[0]['mic_frame']) #int(np.ceil(t_window*2*rois[0].imaging_info['frame_rate']))
    
    ## load stim
    if '01lum' in stimtype or '025lum' in stimtype:
        if '5degbox' in stimtype:
            stimtype_load = 'multiplespeeds_mov_random_moving_WN_5degbox_50msUpdate_20degpers'
        else:
            stimtype_load = 'multiplespeeds_mov_random_moving_WN_8degbox_50msUpdate_20degpers'
    else:
        stimtype_load = stimtype
    
    stim_path = glob.glob(stimpath+'\\stimuli_arrays\\'+ stimtype_load + '.npy')[0]
    #stimulus = np.load(stim_path)
    #stimulus = np.flip(stimulus, axis = 2) # and for the left rigth flip introduced by the screen
    #stimulus = np.where(stimulus==0,-1,stimulus)

    # check if we can make stim smaller:
    len_stim = len(stimulus)
    if last_stim_frame_presented>len_stim:  #chop the signal if too long   ---adition for a specific instance231114_f1-->    #or int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])<len(rois[0].white_noise_response):
        signal_end_index = int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])
    
    elif last_stim_frame_presented<len_stim: # chop the stimulus if too long
        signal_end_index=-1
    
        stimulus = stimulus[:last_stim_frame_presented,:,:]
    
    else:
        signal_end_index=-1


    # arange the frozen frames indices for analysis 

    extended_indices = rois[0].STRF_data['test_indices'] 
    
    # initialize list for storing prediction values
    train_corr_on_SC = []
    test_corr_on_SC = []
    train_corr_off_SC = []
    test_corr_off_SC = []


    for ix,roi in enumerate(rois):

        roi.strf_trace = roi.white_noise_response[0:int(signal_end_index)]
        
        
        trace = copy.deepcopy(roi.strf_trace) 
      
        # filter signal components with period higher than 33 seconds to detrend and eliminate slow oscillations
        
        trace = High_pass(trace,fps,crit_freq=0.03,plot=False) 
        trace = low_pass(trace,fps,crit_freq=5,plot=False) #crit_freq in Hz

        train_scrambled, test_scrambled = STRF_response_prediction(roi,ix,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,n_epochs=1,held_out_frames = extended_indices,test_frames = test_frames,test_trace=None,max_index=copy_center_vect,scramble = True)
        
        print('cors:%s , %s' %(train_scrambled,test_scrambled))
        if roi.CS =='ON':
            
            train_corr_on_SC.append(train_scrambled)
            test_corr_on_SC.append(test_scrambled)        
        else: 
            
            train_corr_off_SC.append(train_scrambled)
            test_corr_off_SC.append(test_scrambled)

        
    shiftpath = os.path.join(stimpath,'RFs',stimtype.split('.')[0])
    savepath=shiftpath+'\\%s'%(rois[0].experiment_info['FlyID'])
    
    plt.figure()
    plt.hist(np.array(train_corr_on_SC))
    plt.title('corr_seen_on_scrambleed')
    plt.figure()
    plt.hist(np.array(test_corr_on_SC))
    plt.title('corr_unseen_on_scrambleed')

    plt.figure()
    plt.hist(np.array(train_corr_off_SC))
    plt.title('corr_seen_off_scrambleed')
    plt.figure()
    plt.hist(np.array(test_corr_off_SC))
    plt.title('corr_unseen_off_scrambleed')

    pmc.multipage(savepath +'\\correlation_hists_scrambled.pdf')



# %%
def identify_mising_frames_in_outputfile(stim_dataframe,stim_framerate=20):

    """ finds instances of missing microscope frames  in the stim output file """
    stim_dataframe_reduced = stim_dataframe.drop_duplicates(subset=['mic_frame'],keep='last')
    indices = np.diff(stim_dataframe_reduced['mic_frame'])==2#[0]
    indices = np.concatenate([np.array([False]),indices])
    indices = stim_dataframe_reduced.loc[indices]['mic_frame'].values.astype(int)
    
    if len(indices) == 0:
        return indices,0
    
    lenix = enumerate(indices)
    for index in lenix:
        curr_ix = index[1]
        extra_values = np.array(range(curr_ix+1,curr_ix+1+int(1.5*stim_framerate)))
        indices = np.concatenate([indices,extra_values])

    if np.any(np.diff(stim_dataframe['mic_frame'])>2):
        raise Exception('consecutive frames missing')

    # take out those indices from the index list

    return indices, len(indices)

def replace_mising_delayed_frames(stim_dataframe,stimulus,mic_framerate=20,treshold = 0.01):

    """ finds instances of missing microscope frames  in the stim output file """

    stim_dataframe_reduced = stim_dataframe.drop_duplicates(subset=['mic_frame'],keep='last')
    indices = np.diff(stim_dataframe_reduced['mic_frame'])==2#[0]
    indices = np.concatenate([np.array([False]),indices])
    indices = stim_dataframe_reduced.loc[indices]['mic_frame'].values.astype(int)


    lenix = enumerate(indices)
    for index in lenix:
        curr_ix = index[1]
        extra_values = np.array(range(curr_ix+1,curr_ix+1+int(1.5*mic_framerate)))
        indices = np.concatenate([indices,extra_values])

    indices = np.unique(indices)

    try:
        delayed_frames = stim_dataframe.loc[stim_dataframe['delay']>(0.05+treshold)]['mic_frame'].values.astype(int)
        delayed_frames = delayed_frames.astype(int)
        real_delayed_frames = copy.deepcopy(delayed_frames)
        lenix = enumerate(delayed_frames)
        for index in lenix:
            curr_ix = index[1]
            extra_values = range(curr_ix+1,curr_ix+1+int(1.5*mic_framerate))
            delayed_frames = np.concatenate([delayed_frames,np.array(extra_values)])
    
        indices = np.sort(np.unique(np.concatenate([indices,delayed_frames])))

    except KeyError:
        raise Exception('no delay columns available in Stimfile')
        
    #stim_corrected = copy.deepcopy(stimulus)
    #stim_corrected[indices,:,:] = 0

    if np.any(np.diff(stim_dataframe['mic_frame'])>2):
        raise Exception('consecutive frames missing')

    return indices#stim_corrected


def find_delayed_frames(stim_dataframe,mic_framerate=20,treshold=20):
    """ finds instances of delayed microscope frames in the stim output file"""
    
    try:
        delayed_frames = stim_dataframe.loc[stim_dataframe['delay']>(0.05+treshold)]['mic_frame'].values.astype('int')
        delayed_frames = delayed_frames.astype(int)
        real_delayed_frames = copy.deepcopy(delayed_frames)
        lenix = enumerate(delayed_frames)
        if len(delayed_frames) == 0:
            return delayed_frames,0
        for index in lenix:
            curr_ix = index[1]
            extra_values = range(curr_ix+1,curr_ix+1+int(1.5*mic_framerate))
            delayed_frames = np.concatenate([delayed_frames,np.array(extra_values)])
            delayed_frames =np.unique(delayed_frames)
    except KeyError:
        print('no delay columns available')
    
    return delayed_frames, len(delayed_frames)
        