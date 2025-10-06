# utilities for the analysis of spatial-temporal receptive fields

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
from scipy import stats
from scipy.ndimage import map_coordinates
from skimage.draw import line
from skimage.measure import profile_line
from skimage import filters
import glob
import scipy.ndimage
from scipy.ndimage import zoom
import imageio
import os
import copy
from scipy import signal
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
import sys
from scipy.fftpack import fft,fftshift,ifft

code_path= r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis\helpers' #Juan desktop
sys.path.insert(0, code_path) 
import post_analysis_core as pac
import process_mov_core as pmc
import ROI_mod
from meansetSTRF import spatio_temporal_set


def create_corrDF_scaffold(restrictions,genotypes,luminances):
    
    
    mean_SxT = {}
    correlation_prediction_train = {}
    correlation_prediction_test = {}
    for genotype in genotypes:
        mean_SxT[genotype] = {}
        correlation_prediction_train[genotype] = {}
        correlation_prediction_test[genotype] = {}
        for lum_treat in luminances:
            mean_SxT[genotype][lum_treat] = {}
            correlation_prediction_train[genotype][lum_treat] = {}
            correlation_prediction_test[genotype][lum_treat] = {}
            for polarity in ['ON','OFF']:
                mean_SxT[genotype][lum_treat][polarity] = spatio_temporal_set(genotype,polarity,lum_treat)
                correlation_prediction_train[genotype][lum_treat][polarity] = {}
                correlation_prediction_test[genotype][lum_treat][polarity] = {}
                for restriction in restrictions:
                    correlation_prediction_train[genotype][lum_treat][polarity][restriction] = {}
                    correlation_prediction_test[genotype][lum_treat][polarity][restriction] = {}
                    for NLin in ['Relu','None']:
                        correlation_prediction_train[genotype][lum_treat][polarity][restriction][NLin]=[]

    return     mean_SxT,correlation_prediction_train,correlation_prediction_test


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

def calculate_STRF_prediction_distributions(STRF_df):
    
    # extract the values required (Z_scores) and Correlation vals
    # first for Z_scores

    treatments = np.unique(STRF_df['treatment'])
    for pol in np.unique(STRF_df['CS']):
        
        fig = plt.figure()
        gs = GridSpec(1,2)
        sns.set_context("paper",font_scale=1.5)
        sns.set_style("ticks") 

        pol_subset = STRF_df.loc[STRF_df['CS'] == pol]

        Z_scores = STRF_df['Z_score'].values
        Z_scores_null = STRF_df['Z_score_null'].values

        ax = gs[0]
        ax.hist(Z_scores,bins=np.linspace(0,12,0.5),histtype=u'step',density=True,label='Z',color='black',lw=2)
        ax.hist(Z_scores_null,bins=np.linspace(0,12,0.5),histtype=u'step',density=True,label='Z_null',color='grey',lw=2)


        ax = gs[1]
        prediction_corr = STRF_df['predicted_trace_corr'].values
        prediction_corr_null = STRF_df['predicted_trace_corr_null'].values

        ax.hist(prediction_corr,bins=np.linspace(0,20,1),histtype=u'step',density=True,label='prediction_corr',color='black',lw=2)
        ax.hist(prediction_corr_null,bins=np.linspace(0,20,1),histtype=u'step',density=True,label='corr_null',color='grey',lw=2)

        plt.title('%s' %(pol))

    
    return None


def gaussian_2d(X, A, x0, y0, sigma_x, sigma_y, C):
    """2D Gaussian function for fitting

        inputs: 
        X: should be a (len(datapoints),2) matrix; 
            where the second dimension represents the spatial dimensions x,y
            
        A,x0,y0,sigma_x,sigma_y,c: are the parameters of the gaussian function"""
        
    x = X[:,0]
    y = X[:,1]

    return A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))) + C


def reverse_correlation_analysis_JF(rois,stimpath,cf=4,highPassfiltering=True,
                                 poly_fitting=False,t_window=1.5,
                                 stim_up_rate=20):
    """ Reverse correlation analysis 
        no interpolation analysis 
        
    """
    
    # find if stimulus is shifted or not
    # stimtype = rois[0].stim_name.split('_')
    # stimtype = list(filter(None, stimtype)) # get rid of empty slices
    # shift = int(stimtype[-1][0])
    # square_size = int(stimtype[-3].split('deg')[0])
    stimtype = rois[0].stim_name
    stim_dataframe = pd.DataFrame(rois[0].stim_info['output_data'], columns = ['entry','rel_time','boutInd','epoch','xpos','ypos','stim_frame','mic_frame'])
    stim_dataframe = stim_dataframe[['stim_frame','rel_time','mic_frame']]
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
    
    last_stim_frame_presented = int(stim_dataframe.iloc[-1]['stim_frame'])
    
    
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
        for stim_ in os.listdir(stimpath+'\\stimuli_arrays\\'):
            if 'random_moving_WN_5degbox_50msUpdate_20degpers' in stim_:
                stim_path = glob.glob(stimpath+'\\stimuli_arrays\\'+ stim_)
                stim_available=True
            else:
                stim_available=False
        png_list = None
        #shift_bool = shift>0
        sampling_rate = rois[0].imaging_info['FramePeriod']
        #jf: read the stimulus array
        stim_path = glob.glob(stimpath+'\\stimuli_arrays\\*.npy')
        
        for stim_file in stim_path:
            #if stim_file.split('\\')[-1].split('.npy')[0] == rois[0].stim_name.split('.txt')[0]:
            #if int(stim_file.split('_')[-2][0]) == shift and int(stimtype[-3].split('deg')[0]) == square_size:
                stimulus = np.load(stim_file)
                stimulus = np.flip(stimulus, axis = 1) # flips are needed to account for the mirror up-down flip
                stimulus = np.flip(stimulus, axis = 2) # and for the left rigth flip introduced by the screen
                
                break
        if stim_available == False:
            raise Exception ('there is no appropiate stimulus array in the exp folder')
        
        # check if we can make stim smaller:
        len_stim = len(stimulus)
        if last_stim_frame_presented>len_stim:  #chop the signal if too long   ---adition for a specific instance231114_f1-->    #or int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])<len(rois[0].white_noise_response):
            signal_end_index = int(stim_dataframe.loc[stim_dataframe['stim_frame']==len(stimulus)-1]['mic_frame'].iloc[-1])
        
        elif last_stim_frame_presented<len_stim: # chop the stimulus if too long
            signal_end_index=-1
        
            stimulus = stimulus[:last_stim_frame_presented,:,:]
        
        else:
            signal_end_index=-1
            #re-escale stimarray:
            # Define a mapping from original unique values to new scale values

        # conditions = [
        #     stimulus == -1.0, 
        #     stimulus == -0.7529411764705882, 
        #     stimulus == -0.5058823529411764
        # ]
        # Corresponding values to map to
        # values = [-0.5, 0, 0.5]
        # Apply mapping
        # stimulus = np.select(conditions, values)

    #temporary change
    # stimulus = stimulus[:int(len(stimulus)/2),:,:]



    #temporary:
    #stimulus = zoom(stimulus, [1,0.5,0.5], order=1)
    #####
    #  we should start doing reverse correlation only when enough time has passed such that enpugh stim frames are presented
    fps = rois[0].imaging_info['frame_rate']
    initial_frame = int(np.ceil(t_window*2*rois[0].imaging_info['frame_rate']))

    #initialize variables that would be used for plotting
    stamax = []
    stamin = []
    stamax_y = []
    stamax_x = []
    stamin_y = []
    stamin_x = []

    for ix,roi in enumerate(rois):
        roi.STRF_data = {}
        print('computing STRF') 
        print('roi %s out of %s'%(ix+1,len(rois)))
        if ix == 3:
            'aaa'
        #filter for reliability:
        reliab = roi.cycle1_reliability['reliability_PD_ON'] if roi.CS == 'ON' else roi.cycle1_reliability['reliability_PD_OFF']
        if reliab<0.4:
            roi.STRF_data['status'] = 'excluded'
            print('roi_skipped')
            continue
        # filter for contrast selectivity:
        if roi.CSI<0.4:
            roi.STRF_data['status'] = 'excluded'
            print('roi_skipped')
            continue
                 
        roi.STRF_data['status'] = 'excluded'
        trace = roi.white_noise_response[0:int(signal_end_index)]
        # filter signal components with period higher than 10 seconds to detrend
        
        trace = High_pass(trace,fps,crit_freq=0.1,plot=False) 
        #trace = low_pass(trace,fps,crit_freq=3,plot=False) 
    
        apply_reverse_correlation(trace,roi,stim_up_rate,stim_dataframe,snippet,initial_frame,stimulus=stimulus,png_list=png_list)#indices=response_idxs
    
    
        response_idxs = np.where(np.abs(trace)>((np.std(trace)*2)+np.mean(trace)))[0]
        trace_shuffled = copy.deepcopy(trace)        
        np.random.seed(8)
        np.random.shuffle(trace_shuffled)
        response_idxs_shuffled = np.where(trace_shuffled>((np.std(trace)*2)+np.mean(trace)))[0]

        roi.STRF_data['strf_control'], roi.mean_strf_control = apply_reverse_correlation(trace_shuffled,roi,fps,stim_dataframe,stimulus,snippet,initial_frame)#,indices=response_idxs_shuffled
        _,roi.STRF_data['z_score'],_ = calculate_Zscore_STRF(roi,control=False)
        _,roi.STRF_data['z_score_control'],_ = calculate_Zscore_STRF(roi,control=True)

        STRF_response_prediction(roi,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,n_epochs=1)

        # plot STA detail
        plt.close('all')
        plt.ioff()
        fig=plt.figure()                
        gs = GridSpec(5, 6) 

        for i in range(roi.STRF_data['strf'].shape[0]):
            ax = fig.add_subplot(gs[i])
            im = ax.imshow(roi.STRF_data['strf'][i,:,:],cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
            ax.set_axis_off()
            cbar=fig.colorbar(im, ax=ax)
            # cbar.ax.yaxis.set_major_formatter(formatter)
            cbar.ax.tick_params(labelsize=3)
            if i == 0:
                ax.title.set_text('-%ss_%s'%(t_window-i*0.05,roi.category[0]))
            elif i == 1:
                ax.title.set_text('-%ss_%s'%(t_window-i*0.05,roi.CS))
            elif i == 2:
                ax.title.set_text('-%ss_CS%.2f'%(t_window-i*0.05,roi.CSI))
            elif i == 3: 
                ax.title.set_text('-%ss_Zscore%.2f'%(t_window-i*0.05,roi.STRF_data['z_score']))
            elif i == 4:
                 ax.title.set_text('-%ss_Vect_len%.2f'%(t_window-i*0.05,(roi.DSI_ON if roi.CS=='ON' else roi.DSI_OFF)))
            elif i == 5:
                 ax.title.set_text('-%ss_PD%.2f'%(t_window-i*0.05,(roi.PD_ON if roi.CS=='ON' else roi.PD_OFF)))
            elif i == 6:
                 ax.title.set_text('-%ss_MaxRspDir%.2f'%(t_window-i*0.05,(roi.dir_max_resp)))
            else:
                ax.title.set_text('-%ss'%(t_window-i*0.05))
            #fig.suptitle('%s' %(roi.CS))
            #plt.tight_layout()
            
            # if shift_bool:
            #     shiftpath = stimpath+'\\RFs\\shifted%s_%sdeg'%(shift,square_size)
            # else:
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


        # get the index of the max and min values 
        index_of_max = [np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[2][0]]
        index_of_min = [np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[2][0]]
        
        # scaling factor for pixel size
        scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1])
        vec_len = int(15.0/scaling_fact)
        #radius = int(10.0/scaling_fact)
        extend =  int(5/scaling_fact)
        # create prefdir vector (15deg length scaling)
        
        pdir = (roi.PD_ON if roi.CS == 'ON' else roi.PD_OFF) 
        pdircopy = (roi.PD_ON if roi.CS == 'ON' else roi.PD_OFF)
        pdir = np.deg2rad(pdir)
        pdir2 = roi.dir_max_resp
        pdir2 = np.deg2rad(pdir2)

        # map the correct polar coordinates of the vector to the x and y components of the strf (the coordinates are rotated in the image presentation, so a correction is needed)
        pdy = np.cos(pdir)
        pdx = -np.sin(pdir)
        pdy2 = np.cos(pdir2)
        pdx2 = -np.sin(pdir2)


        # 
        center_vect = index_of_max if roi.CS == 'ON' else index_of_min
        end_vect = np.array(center_vect) + np.array(np.round(vec_len*np.array([pdx,pdy]))).astype(int)
        end_vect2 = np.array(center_vect) + np.array(np.round(vec_len*np.array([pdx2,pdy2]))).astype(int)

        #ax.quiver(center_vect[0], center_vect[1], pdx, pdy, angles='xy', scale_units='xy', scale=vec_len, color='black')
        ax.plot([center_vect[1],end_vect[1]],[center_vect[0],end_vect[0]],color='black',linewidth = 1)
        ax.plot([center_vect[1],end_vect2[1]],[center_vect[0],end_vect2[0]],color='blue',linewidth = 1)
        plt.draw()
        # maxmask = np.nan((roi.STRF_data['strf'].shape[1],roi.STRF_data['strf'].shape[2]))
        #minmask =  np.nan((roi.STRF_data['strf'].shape[1],roi.STRF_data['strf'].shape[2]))
        
        plt.savefig(savepath +'\\STRF_roi%s.jpg'%(ix))
        stamax.append(np.max(roi.STRF_data['strf']))
        stamin.append(np.min(roi.STRF_data['strf']))
        
        # 1st approach for spatiotemporal representation:
        # this approach consists of stacking the values along the line between max and min value 
        # across time 
        
        SxT_representation, x_lims, ylims = stack_minmaxLine(roi.STRF_data['strf'],center_vect,end_vect,pdir,extend)
        SxT_representation2, x_lims2, ylims2 = stack_minmaxLine(roi.STRF_data['strf'],center_vect,end_vect2,pdir2,extend)
        roi.STRF_data['SxT_representation_pdir'] = SxT_representation
        roi.STRF_data['SxT_representation_MaxRespDir'] = SxT_representation2

        #SxT_representation = np.repeat (SxT_representation,4,axis=1)
        fig2=plt.figure()  
        
        gs = GridSpec(1, 2)
        ax = fig2.add_subplot(gs[0])
        SxT_representation_2plot = np.repeat(SxT_representation,3,axis=1)
        im = ax.imshow(SxT_representation_2plot ,cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
        cbar=fig2.colorbar(im, ax=ax)
        ax.title.set_text('sxTRF values across pref dir axis')

        ax = fig2.add_subplot(gs[1])
        SxT_representation_2plot = np.repeat(SxT_representation2,3,axis=1)
        im = ax.imshow(SxT_representation_2plot ,cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
        cbar=fig2.colorbar(im, ax=ax)
        ax.title.set_text('sxTRF values dirMaxResp axis')        

        plt.savefig(savepath +'\\test_STRF_roi%s_SxTrepminmax.jpg'%(ix))


    # stamax=np.max(stamax)
    # stamin=np.min(stamin)
    
        #for ix,roi in enumerate(rois):

        # get the index of the max and min values 
        index_of_max = [np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.max(roi.STRF_data['strf']))[2][0]]
        index_of_min = [np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[1][0],np.where(roi.STRF_data['strf']==np.min(roi.STRF_data['strf']))[2][0]]


        # fit a double gaussian



        # plot average sta:
        fig=plt.figure()               
        gs = GridSpec(1, 2)
        ax = fig.add_subplot(gs[0])
        im = ax.imshow(np.mean(roi.STRF_data['strf'][int(len(roi.STRF_data['strf'])/2):,:,:],axis=0),cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
        # plot the line of the spatiotemporal representation
        ax.plot(ylims,x_lims,color='black',linewidth = 1)
        ax.plot(ylims2,x_lims2,color='blue',linewidth = 1)
        # plot a 20 deg circle arround the center 
        yi, xi = np.indices((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]))
        circle = (xi - center_vect[1])**2 + (yi - center_vect[0])**2
        circ_array = np.full((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]), np.nan)
        circ_array[np.where((circle <= (vec_len + 0.5)**2) & (circle >= (vec_len - 0.5)**2))] = 1
        ax.imshow(circ_array)
        ax.title.set_text('2ndhalf_%s_%.2f_PD_%s_\nMrespDir_%s'%(roi.CS,roi.CSI,pdircopy,roi.dir_max_resp))
        cbar=fig.colorbar(im, ax=ax)
        ax1 = fig.add_subplot(gs[1])
        im = ax1.imshow(np.mean(roi.STRF_data['strf'][:int(len(roi.STRF_data['strf'])/2),:,:],axis=0),cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
        ax1.title.set_text('1sthalf_%s_Z%s'%(roi.category[0],roi.STRF_data['z_score']))
        cbar=fig.colorbar(im, ax=ax1)
        plt.savefig(savepath +'\\%sroi_mean_SRF.jpg'%(ix))
        plt.close('all')
       
        # fig=plt.figure()
        # gs = GridSpec(2, 2) 
        # #plt.close('all')
        # for i in range(2):            
        #     if i == 0 :
        #         #collapse y
        #         local_strf_min = np.transpose(roi.STRF_data['strf'][:,index_of_min[0],:])
        #         local_strf_max = np.transpose(roi.STRF_data['strf'][:,index_of_max[0],:])
        #         plot_string = 'across_x'
        #     else:
        #         #collapse x                
        #         local_strf_min = np.transpose(roi.STRF_data['strf'][:,:,index_of_min[1]])
        #         local_strf_max = np.transpose(roi.STRF_data['strf'][:,:,index_of_max[1]])
        #         plot_string = 'across_y'
        #     ax = fig.add_subplot(gs[i,0])
        #     ax1 = fig.add_subplot(gs[i,1])
        #     local_strf_min = np.repeat(local_strf_min,4,axis=1)
        #     im1 = ax.imshow(local_strf_min,cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
        #     cbar = fig.colorbar(im1, ax=ax)
        #     local_strf_max = np.repeat(local_strf_max,4,axis=1)
        #     im2 = ax1.imshow(local_strf_max,cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
        #     cbar = fig.colorbar(im2, ax=ax1)
            
        #     ax.title.set_text('SxT_min %s roi_%s CS_%s CSI_%s'%(plot_string,ix,roi.CS,roi.CSI))
        #     ax1.title.set_text('SxT_max %s roi_%s CS_%s CSI_%s'%(plot_string,ix,roi.CS,roi.CSI))
        
        # plt.savefig(savepath +'\\%s_SxT.jpg'%(ix))

    #pmc.multipage(stimpath+'\\RFs\\STRF_%s_shifted_%s_%s.pdf' %(roi.experiment_info['FlyID'],shift_bool,square_size))
    return rois



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
    start_x, start_y, end_x, end_y = int(round(start_x)),int(round(start_y)),int(round(end_x)),int(round(end_y))
    x_line, y_line = line(start_x, start_y, end_x, end_y)
    line_stack = np.zeros((len(x_line),len(range(-extend, extend+1)),images.shape[0]))
    line_stack[:] = np.nan
    #orig_start_x, orig_start_y, orig_end_x, orig_end_y = int(round(start_x)),int(round(start_y)),int(round(end_x)),int(round(end_y))
    
    for im_ix in range(images.shape[0]):
        local_stack = profile_line(images[im_ix,:,:],(start_x,start_y),(end_x,end_y),linewidth=extend,mode = 'constant',cval = np.nan) #reduce_func = np.mean
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
    x_max = x_1
    y_max = y_1
    x_min = x_2
    y_min = y_2
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
        strf_mat = roi.STRF_data['strf_control']
    else:
        strf_mat = roi.STRF_data['strf']
    #calculate the Zscores of the values in an strf 
    z_vals = (strf_mat-(np.mean(strf_mat)))/np.std(strf_mat)
    max_Z_val = np.max(np.abs(z_vals)) 
    max_Z_val_coords = np.argmax(np.abs(z_vals))
    return z_vals, max_Z_val, max_Z_val_coords

# def find_STRF_maxMin_line(roi):

#     max_pos = np.unravel_index(np.argmax(roi.mean_strf), roi.mean_strf.shape)
#     min_pos = np.unravel_index(np.argmin(roi.mean_strf), roi.mean_strf.shape)
#     # Create a line from min to max
#     line_y, line_x = np.linspace(min_pos[0], max_pos[0], num=100), np.linspace(min_pos[1], max_pos[1], num=100)
#     # Use interpolation to sample the image
#     line_values = map_coordinates(roi.mean_strf, np.vstack((line_y, line_x)))
#     # calculate the length of the resulting line 
#     line_lenght = np.sqrt((min_pos[0]-max_pos[0])^2 + (min_pos[1]-max_pos[1])^2)

def apply_reverse_correlation(trace,roi,stim_up_rate,stim_dataframe,snippet,initial_frame,stimulus=None,indices=None,png_list=None):
    
        
    #jf: commented low_pass_filtering, we are still interested in frequencies close to the nyquist frequency
    #filtered = low_pass(filtered, fps, crit_freq=cf,plot=False) #juan: consider changing critical frequency to nyquist freq

    # loop through the signal timepoints and calculate the reverse correlation for each point in the previous 2 seconds 
    
    #first initialize the STA array. it should have a time dimension and as many spatial dimensions as the stimuli
    if indices is not None:
        trace_indexes = indices[indices > initial_frame]
    else:
        trace_indexes = range(initial_frame,len(trace))
    
    if stimulus is not None:
        sta_dims = stimulus[0,:,:].shape
    else: 
        sta_dims = np.repeat(imageio.imread(png_list[0])[:,:,0],2,axis=0).shape

    STA_array = np.zeros(sta_dims).astype(float)
    STA_array = np.expand_dims(STA_array,axis = 0)
    STA_array = np.repeat(STA_array,snippet,axis = 0)
    ##extra dim added to stack snippets and then average them
    # STA_array = np.expand_dims(STA_array,axis = 0)
    # STA_array = np.repeat(STA_array,len(trace_indexes),axis = 0)
    print(STA_array.shape)
    
    # check correct indexes 

    if len(trace_indexes)>0:
        frame_lost_label = -1
        lost_count = 0
        delay_count = 0
        for ix,i in enumerate(trace_indexes):
            # get signal location
            try:
                index = int(stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0]['mic_frame'])
            except IndexError: #this happens when frames are skipped
                frame_lost_label = 0
                lost_count += 1
                print ('frame %s is lost'%(i))
            if frame_lost_label >=0 and frame_lost_label < int(np.ceil(1*roi.imaging_info['frame_rate'])):
                frame_lost_label += 1
                continue
            elif frame_lost_label == int(np.ceil(1*roi.imaging_info['frame_rate'])):
                frame_lost_label = -1
            
            if ix%1000==0:
                print('round %s of %s' %(ix+1,len(trace_indexes))) 
            current_stim_frame = int(stim_dataframe.loc[stim_dataframe['mic_frame'] == i].iloc[0]['stim_frame'])
            if stimulus is None:
                stim_chunk = import_png_snippet(png_list,range(current_stim_frame-snippet,current_stim_frame))            
            else:
                stim_chunk = stimulus[current_stim_frame-snippet:current_stim_frame,:,:]
            
            ### check that the snippet contents are timed correctly otherwise create delays in the data to compensate
            ## this seems to be necesary since some frame are staying longer than expected on the screen
            index = stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0].name
            delay_treshold = ((1/float(stim_up_rate))+((1/float(stim_up_rate)/10)))
            if np.any(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>=delay_treshold):
                error_ix = np.where(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>delay_treshold)[0]
                for error in error_ix:
                    delay_signature = int(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])[error]/(1/float(stim_up_rate)))
                    stim_chunk[error: error + delay_signature,:,:] = stim_chunk[error-1,:,:]
                    remaining_len = len(stim_chunk[error + delay_signature:,0,0])
                    stim_chunk [error + delay_signature:,:,:] = stim_chunk [error:error+remaining_len,:,:] 
                    delay_count += 1
            stim_chunk = stim_chunk*trace[i]
            STA_array  += stim_chunk
    print('lost frames: %s'%(lost_count))
    print ('timing errors: %s'%(delay_count))
    #strf = np.mean(STA_array,axis=0)#/np.max(STA_array)
    strf = STA_array/len(trace_indexes)
    mean_strf = np.mean(strf[int(len(strf)/2):,:,:],axis=0)
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    #formatter.set_powerlimits((-1,1))
    roi.STRF_data['strf'], roi.STRF_data['mean_strf'] = strf, mean_strf


def STRF_response_prediction(roi,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=None,t_window=40,n_epochs=1):

    """
    predict neuron response based on its STA (receptive field and the random stimulus)
    t_window should be the same as used to calculate sta

    for now it's implemented for 1 repetition of stimulus

    """

    trace_indexes = range(initial_frame,len(trace))
    snippet = t_window
    

    #for roi in rois:
    strf_prediction = np.zeros(trace.shape)
    curr_sta=roi.STRF_data['strf']
    offset=t_window
    frame_lost_label = -1
    for ix,i in enumerate(trace_indexes):

        try:
            index = int(stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0]['mic_frame'])
        except IndexError: #this happens when frames are skipped
            frame_lost_label = 0
            strf_prediction[ix]=np.nan
            #lost_count += 1
            print ('frame %s is lost'%(i))
        if frame_lost_label >=0 and frame_lost_label < int(np.ceil(1*roi.imaging_info['frame_rate'])):
            frame_lost_label += 1
            continue
        elif frame_lost_label == int(np.ceil(1*roi.imaging_info['frame_rate'])):
            frame_lost_label = -1
    
        if ix%1000==0:
            print('round %s of %s' %(ix+1,len(trace_indexes))) 
        current_stim_frame = int(stim_dataframe.loc[stim_dataframe['mic_frame'] == i].iloc[0]['stim_frame'])
        stim_chunk = stimulus[current_stim_frame-snippet:current_stim_frame,:,:]
        
        ### check that the snippet contents are timed correctly otherwise create delays in the data to compensate
        ## this seems to be necesary since some frames are staying longer than expected on the screen
        index = stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0].name
        delay_treshold = ((1/float(stim_up_rate))+((1/float(stim_up_rate)/10)))
        if np.any(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>=delay_treshold):
            error_ix = np.where(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>delay_treshold)[0]
            for error in error_ix:
                delay_signature = int(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])[error]/(1/float(stim_up_rate)))
                stim_chunk[error: error + delay_signature,:,:] = stim_chunk[error-1,:,:]
                remaining_len = len(stim_chunk[error + delay_signature:,0,0])
                stim_chunk [error + delay_signature:,:,:] = stim_chunk [error:error+remaining_len,:,:] 
                #delay_count += 1

        strf_prediction[i]= np.sum(stim_chunk*roi.STRF_data['strf']) # prediction giving no weight to the different times in STRF

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
    roi.STRF_data['strf_prediction'] = strf_prediction
    roi.STRF_data['prediction_corr'] = scipy.stats.pearsonr(roi.STRF_data['strf_prediction'][initial_frame:],trace[initial_frame:]) 
    roi.STRF_data['prediction'] = strf_prediction
    #return rois

def create_dataframe_forSTRF(curr_rois,independent_vars=None,mapping=False):
    data = []

    # Define the possible values of independent_var
    


    # Loop through the roi objects
    for roi in curr_rois:
        if roi.STRF_data['status']=='excluded':
            continue

        try:
            z_null = roi.STRF_data['z_score_null']
        except:
            z_null = None
        lum_val = roi.stim_name[0:2]
        
        if lum_val == '01':
            luminance = 0.1
        elif lum_val == '02':
            luminance = 0.25
        else:
             lum_val = 1
        
        reliability_on = roi.cycle1_reliability['reliability_PD_ON']
        reliability_off = roi.cycle1_reliability['reliability_PD_OFF']
        # Create a dictionary with the relevant attributes
        entry = {
            'uniq_id': roi.uniq_id,
            'Z_score': roi.STRF_data['z_score'],            
            'luminance_stim': lum_val,
            'FlyId': roi.experiment_info['FlyID'],  # Include FlyId attribute
            'treatment': roi.experiment_info['treatment'],
            'category': roi.category[0],
            'reliability_ON': reliability_on,
            'reliability_OFF': reliability_off,
            'DSI_ON': getattr(roi, 'DSI_ON', np.nan),
            'DSI_OFF': getattr(roi, 'DSI_OFF', np.nan),
            'CS': getattr(roi, 'CS', np.nan),
            'CSI': getattr(roi, 'CSI',np.nan),
            'Z_score_null': z_null

        } #'predicted_trace_corr': roi.STRF_data['prediction_corr'],
          # 'predicted_trace': roi.STRF_data['strf_prediction'],

        # Append the dictionary to the data list

        data.append(entry)
    # Convert the data list to a pandas DataFrame
    df = pd.DataFrame(data)

    return df            

def plot_NMF_segregated_components(roi,components_xy,components_t,savedir,type = 'pos'):
    
    """
    plots the first n spatial and temporal components of a NMF decomposition 
    from an STRF

    input:
    components_xy: list of arrays: array(x,y,components)
    components_t: list of arrays : array(temporalweights,components)

    """

    # fig = plt.figure()
    # gs = GridSpec (4,components_xy.shape[2])
    for i in range(len(components_xy)):
        if np.any(components_xy[i] < 0): 
            min_val = np.min(components_xy[i])
            max_val = -1*min_val
            type_ = 'neg_comps_NMF'
            colmap = 'PRGn'#'Purples'
            multiplier = 1 #-1
        else:
            max_val =  np.max(components_xy[i])
            min_val = max_val*-1
            type_ = 'pos_comps_NMF'
            colmap = 'PRGn'#'Greens'
            multiplier = 1

        plot_spatial_temporal_components(components_xy[i],components_t[i],savedir,type = type_,max_val=max_val,min_val=min_val,comap=colmap,multiplier=multiplier)

    return None

def plot_spatial_temporal_components(components_xy,components_t,savedir,type = 'mixed',max_val=None,min_val=None,comap = 'PRGn',multiplier=1):

    """
    plots the first n spatial and temporal components of a svd decomposition 
    from an STRF

    input:
    components_xy: array(x,y,components)
    components_t: array(temporalweights,components)

    """
    components_xy = components_xy*multiplier
    if max_val is None and min_val is None:
        max_val = np.max(components_xy)
        min_val = np.min(components_xy)

    fig = plt.figure(figsize=(15,10))
    gs = GridSpec (2,components_xy.shape[2])

    for i in range(components_xy.shape[2]):
        
        plt.rcParams.update({'font.size' : 10})
        
        ax_xy = fig.add_subplot(gs[0,i])
        ax_t = fig.add_subplot(gs[1,i])
        
        im = ax_xy.imshow(components_xy[:,:,i],cmap=comap,vmax=max_val,vmin=min_val)
        ax_xy.set_axis_off()
        cbar=fig.colorbar(im, ax=ax_xy, aspect = 8)
        cbar.ax.tick_params(labelsize=10)
        ax_t.plot(components_t[:,i])

        ax_xy.title.set_text('xy_component_%s'%(i))
        ax_t.title.set_text('t_component_%s'%(i))
    #pmc.multipage(savedir + 'type' + 'components.pdf')

def fit_component_gaussians(roi,list_components_xy):

    """fits as many gaussians as entries in the 3rd axis for every item
        in the input
    
    input:
    list_components_xy: list of arrays array(x,y,components) (normally 2 arrays one negative one positive)
    
    roi: instance of the roi_bg class

    """
    # initialize output list
    output_ = []
    
    #make x,y coordinate tuple 

    xcoords = np.linspace(0, list_components_xy[0].shape[0], list_components_xy[0].shape[0],endpoint=False)
    ycoords = np.linspace(0, list_components_xy[0].shape[1], list_components_xy[0].shape[1],endpoint=False)
    
    x_grid, y_grid = np.meshgrid(xcoords, ycoords)

    
    # Flatten the grid
    x_data_flat = x_grid.ravel()
    y_data_flat = y_grid.ravel()
    
    input_mat = np.concatenate((x_data_flat[:,np.newaxis],y_data_flat[:,np.newaxis]),axis=1)

    for item in len(list_components_xy):
        for comp in range(item.shape[2]):
            z = item[:,:,comp]
            # take the item and convert coor
            z_data_flat = z.ravel(z)
            
            # find location of max component, this is location guess
            indices = np.where(z==np.max(np.abs(z)))

            # User-defined initial guesses for gaussian_params: [Amplitude, x_center, y_center, sigma_x, sigma_y, offset]
            
            initial_guesses = [1, indices[1], indices[0], 5, 5, 0]

    popt, pcov = curve_fit(gaussian_2d, (y_data_flat, x_data_flat), z_data_flat, p0=initial_guesses)

def apply_circular_mask(roi,radius,indices=None):
    
    """filters out the area beyond radius from the point where STRF is absolute highest
    inputs:
        roi: roi object containing STRF information
        radius: the radius of the area to be included
    
    output: masked STRF
    """

    scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1])
    radius_scaled = radius/scaling_fact
    if indices is None:
        indices = np.where(np.abs(roi.STRF_data['strf'])==np.max(np.abs(roi.STRF_data['strf'])))
        indices = indices[1][0], indices[2][0]
    else:
        indices = indices[0], indices[1]
    yi, xi = np.indices((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]))
    circle = (xi - indices[1])**2 + (yi - indices[0])**2
    circ_array = np.full((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]), 0)
    circ_array[np.where((circle <= (radius_scaled)**2))] = 1

    return roi.STRF_data['strf'] * circ_array[np.newaxis,:,:]
    # multiply the mask with the RF

def _flatten_strf(dims,mat):
    """take a 3d strf matrix and make it flat into 
    a 2d space,time matrix"""

    flattened_mat = np.zeros((dims[1]*dims[2],dims[0]))
    
    for time_slice in range(dims[0]):
        flattened_mat[:,time_slice] = mat[time_slice,:,:].flatten()
    return flattened_mat

def svd_flattenedSTRF(roi,masked_array,components=5):
    
    dims = roi.STRF_data['strf'][0:40,:,:].shape
    
    flattened_mat = _flatten_strf (dims,masked_array[0:40,:,:])
      
    U,S,Vt = scipy.linalg.svd(flattened_mat, full_matrices=False)
    
    #U should contain temporal components size (dims[1]*dims[1],dims[0])

    first_10x_components = U[:,0:components]
    first_10t_components = np.transpose(Vt[0:components,:])

    # un-flatten spatial components
    components_xy = np.zeros((dims[1],dims[2],components))
    for component in range(0,components):
        components_xy[:,:,component] = np.reshape(first_10x_components[:,component],(dims[1],dims[2]))

    return components_xy, first_10t_components

def segregated_NMf(roi,radius,regularization = 0.0001, components =5):
    
    """ segregates a STRF in positive and negatively correlated arrays 
        and performs non-negative-matrix multiplication on them independently, 
        then it concatenates the components obtained
        inputs:
        roi: roi object with strf data 
        masked_array: strf passed through a mask or full strf array
        regularization: parameter to control sparseness of the NMF result
    """

    masked_array = apply_circular_mask(roi,radius)
    dims = roi.STRF_data['strf'][:,:,:].shape #roi.STRF_data['strf'].shape*(2/3):
    flattened_mat = _flatten_strf (dims,masked_array[:,:,:]) #roi.STRF_data['strf'].shape*(2/3):

    model = NMF(n_components=components, init = 'nndsvd', random_state=0, alpha = regularization)

    H_xy_pos = model.fit_transform(np.where(flattened_mat>0,flattened_mat,0))
    W_t_pos = model.components_

    H_xy_neg = model.fit_transform(np.where(flattened_mat<0,flattened_mat*-1,0),0)
    W_t_neg = model.components_

    H_xy_neg = H_xy_neg*-1

    # un-flatten spatial components
    components_xy_pos = np.zeros((dims[1],dims[2],components))
    components_xy_neg = np.zeros((dims[1],dims[2],components))
 
    for component in range(0,components):
        components_xy_pos[:,:,component] = H_xy_pos[:,component].reshape((dims[1],dims[2]))
        components_xy_neg[:,:,component] = H_xy_neg[:,component].reshape((dims[1],dims[2]))


    return [components_xy_neg,components_xy_pos],[W_t_neg.transpose(),W_t_pos.transpose(),]

def produce_spatiotemporal_view(roi,save_path,ix):
    ''' produce a spatiotemporal representation of a 3d signal trigered average
        (in the roi object is called strf 
        
        this uses the maximum absolute value of the strf to produce a 2d timexspace
        representation along the max response direction'''
    
    #create a vector that represents the prefered direction
    scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1]) # 80 is the screen size in degrees
    vec_len = int(15.0/scaling_fact)
    arrow_len = int(20.0/scaling_fact)
    arrow_width = int(5.0/scaling_fact)
    pdir = roi.dir_max_resp 
    # map some polar coordinates to the strf such that the direction of movement coincides with a mapping where 0deg is rigth, and 90 deg is up
    pdx = np.cos(np.deg2rad(pdir))
    pdy = -np.sin(np.deg2rad(pdir))

    index_of_max = [np.where(np.abs(roi.STRF_data['strf'])==np.max(np.abs(roi.STRF_data['strf'])))[0][0],np.where(np.abs(roi.STRF_data['strf'])==np.max(np.abs(roi.STRF_data['strf'])))[1][0]]
    center_vect = index_of_max            
    end_vect= np.array(center_vect) + np.array(np.round(vec_len*np.array([pdx,pdy]))).astype(int)
    pref_dir_mark = np.array(center_vect) - np.array(np.round(arrow_len*np.array([pdx,pdy]))).astype(int)
    pref_dir_mark2 = np.array(center_vect) - np.array(np.round(arrow_width*np.array([pdx,pdy]))).astype(int)
    # take the axis of the vector and sample to the sides of the vector to create a space*time 
    # representation
    STRF_lookalike = np.zeros_like(roi.STRF_data['strf'])
    STRF_lookalike[:] = 0

    try:
        STRF_lookalike[0:10,pref_dir_mark[0],pref_dir_mark[1]] = 1
        len_mark = arrow_len
    except: 
        STRF_lookalike[0:10,pref_dir_mark2[0],pref_dir_mark2[1]] = 1
        len_mark = arrow_width

    #figure out how the prefered direction maps to the SxT representation
    SxT_pdir, x_lims, ylims = stack_minmaxLine(STRF_lookalike,center_vect,end_vect,pdir,extend=int(5/scaling_fact))
    SxT_pdir = np.where(SxT_pdir > 0,1,0)
    roi.pos_pdir_marker =  ((SxT_pdir.shape[0]//2 - len_mark) - np.where(SxT_pdir == 1) [0][0]) # if marker is negetive, dir is down and visceversa

    

    roi.SxT_representation,_,_ = stack_minmaxLine(roi.STRF_data['strf'],center_vect,end_vect,pdir,extend=int(5/scaling_fact))
    # if  roi.pos_pdir_marker is less than zero, that means that the leading side of the neuron is on top. flip it so every pdir goes from bottom to top
    if roi.pos_pdir_marker < 0:
        roi.SxT_representation = np.flip(roi.SxT_representation,axis=0) 

    # scaler=StandardScaler()
    # scaler.fit(roi.SxT_representation)
    # roi.SxT_representation = scaler.transform(roi.SxT_representation)

    # plot SxT representation
    plt.figure()
    fig,ax = plt.subplots()
    ax.imshow(np.repeat(roi.SxT_representation,2,axis=1) ,cmap='PRGn',vmax=np.max(roi.STRF_data['strf']),vmin=np.min(roi.STRF_data['strf']))
    ax.title.set_text('sxTRF values dirMaxResp axis\n polarity_%s' %(roi.CS)) 
    arrow = patches.FancyArrow(10,40,0,vec_len/2*-1,length_includes_head = True, head_width=5,head_length=10,fc='black',ec='black')
    plt.gca().add_patch(arrow)
    #save
    stimtype = roi.stim_info['stim_name'].split('.txt')[0]
    fly = roi.experiment_info['FlyID']
    save_str = os.path.join(save_path,stimtype,fly)
    try:
        os.mkdir(save_str)
    except:
        pass 
    
    plt.savefig(save_str +'\\STRF_roi%s_SxT.jpg'%(ix))


    #return roi.SxT_representation, [x_lims, ylims]

def predict_signal_correlation(roi,stimulus,type_ = 'test',t_window = 3,restriction='unrestricted'):
    '''
    use The STRF to predict the df/f signal
    if type == 'test' use test data for this, otherwise use training data

    time_window is the lenght of the time window used to apply reverse correlation,
    default is 3 seconds

    '''
    #get test indices and traces
    #test_indices = roi.STRF_data['test_indices']
    #test_trace = roi.STRF_data['test_trace']
    fps = roi.imaging_info['frame_rate']
    trace = copy.deepcopy(roi.strf_trace)
    stim_up_rate = 1/float(roi.stim_info['frame_duration'])

    stimtype = roi.stim_name
    stim_dataframe = pd.DataFrame(roi.stim_info['output_data'], columns = ['entry','rel_time','boutInd','epoch','xpos','ypos','stim_frame','mic_frame'])
    stim_dataframe = stim_dataframe[['stim_frame','rel_time','mic_frame']]
    initial_frame = int(np.ceil(t_window*2*roi.imaging_info['frame_rate']))
    snippet = int(t_window*stim_up_rate)
    
    # set the microscope frame count to 0
    
    starting_signal_frame=stim_dataframe.iloc[0]['mic_frame'] 
    stim_dataframe['mic_frame'] = stim_dataframe['mic_frame'] - starting_signal_frame
    stim_dataframe['rel_time'] = stim_dataframe['rel_time'] - stim_dataframe['rel_time'].iloc[0]

    # filter signal components with period higher than 10 seconds to detrend
    
    #trace1 = High_pass(trace,fps,crit_freq=0.03,plot=False)
    trace1 = low_pass(trace,fps,crit_freq=3,plot=False)
    # use the STRF to predict the test and train signals
    # TODO continue here:

    #if type_ == 'test': #use heldout frames to test strf prediction
    return STRF_response_prediction(roi,trace1,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,held_out_frames = roi.STRF_data['test_indices'],type_=type_,restriction = restriction)
    # elif type_ == 'train': #use the frames used for reverse correlation for prediction
    #     STRF_response_prediction(roi,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,n_epochs=1,held_out_frames = roi.STRF_data['test_indices'],type_='train',restrict_space=restrict_space,restrict_time=restrict_space)


def load_stimulus(stim_path,stim_type):
    ''' finds the correct stimulus to load and loads it 
    
    '''
    
    for stim_ in os.listdir(stim_path):
        if '10max' in stim_ or '3max' in stim_:
            locs = glob.glob( stim_path + '\\stimulus_10maxdur_5degbox_1.0deg_step.npy')
        elif 'random_moving' in stim_:
            locs = glob.glob( stim_path + '\\random_moving_WN_5degbox_50msUpdate_20degpers.npy')
            break
    stimulus = np.load(locs[0])
    stimulus = np.flip(stimulus, axis = 1) # flips are needed to account for the mirror up-down flip
    stimulus = np.flip(stimulus, axis = 2) # and for the left rigth flip introduced by the screen
    
    return stimulus

def STRF_response_prediction(roi,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=None,t_window=40,control=False,held_out_frames=None,type_='train',restriction='unrestricted'):

    """
    predict neuron response based on its STA (receptive field and the random stimulus)
    t_window should be the same as used to calculate sta

    for now it's implemented for 1 repetition of stimulus

    """

    # create the structure for prediction analysis storage, this will distinguish between correlations obtained from restricted or unrestricted 
    # RFs

    # try:
    #     roi.STRF_prediction_analysis
    # except:
    #     roi.STRF_prediction_analysis = {}
    
    # create string for restriction type:

    # if restriction == 'spatially_restricted':
    #     restriction_str = 'spatially_restricted'
    # elif restrict_time is not None:
    #     restriction_str = 'temporally_restricted'        
    # elif restrict_space is not None:
    #     restriction_str = 'spatially_restricted'
    # else:
    #     restriction_str = 'unrestricted'

    fps = roi.imaging_info['frame_rate']
    trace = copy.deepcopy(roi.strf_trace)
    stim_up_rate = 1/float(roi.stim_info['frame_duration'])

    stimtype = roi.stim_name
    stim_dataframe = pd.DataFrame(roi.stim_info['output_data'], columns = ['entry','rel_time','boutInd','epoch','xpos','ypos','stim_frame','mic_frame'])
    stim_dataframe = stim_dataframe[['stim_frame','rel_time','mic_frame']]
    initial_frame = int(np.ceil(t_window*2*roi.imaging_info['frame_rate']))
    snippet = int(t_window*stim_up_rate)

    starting_signal_frame=stim_dataframe.iloc[0]['mic_frame'] 
    stim_dataframe['mic_frame'] = stim_dataframe['mic_frame'] - starting_signal_frame
    stim_dataframe['rel_time'] = stim_dataframe['rel_time'] - stim_dataframe['rel_time'].iloc[0]


    restriction_str = restriction
    # take a copy of STRF
    localSTRf = copy.deepcopy(roi.STRF_data['strf'])

    # Z_normalize the STRF (consider using chrises standartdeviation fix)
    fps = roi.imaging_info['frame_rate']
    trace = low_pass(trace,fps,crit_freq=3,plot=False)    
    trace = High_pass(trace,fps,crit_freq=0.1,plot=False)
    if restriction == 'spatially_restricted':
        localSTRf = apply_circular_mask(roi,15)
    if restriction == 'temporally_restricted':
        # restrict to the last 1.5 secs (!!! warning: hardcoded)
        localSTRf = localSTRf[0:localSTRf.shape[0]//2,:,:] = 0
    elif restriction == 'spatiotemporally_restricted':
        localSTRf = localSTRf[0:localSTRf.shape[0]//2,:,:] = 0
        localSTRf = apply_circular_mask(roi,15)

    localSTRf = (localSTRf-np.mean(localSTRf))/ np.std(localSTRf)
    # create the structure for prediction analysis storage, this will distinguish between correlations obtained from restricted or unrestricted 
    # RFs 

    try:
        roi.STRF_prediction_analysis[restriction_str]
    except:
        roi.STRF_prediction_analysis[restriction_str] = {}
    
    #distinguish between train and test sets and analyze accordingly

    if type_ == 'test':
        if held_out_frames is not None:
            trace_indexes = range(held_out_frames[0],held_out_frames[1]-1)
        else:
            trace_indexes = range(initial_frame,len(trace))
        snippet = t_window
    elif type_ == 'train':
        if held_out_frames is not None:
            trace_indexes = np.array(range(initial_frame,held_out_frames[0])) 
            trace_indexes2 = np.array(range(held_out_frames[1],len(trace)))
            trace_indexes = np.concatenate([trace_indexes,trace_indexes2])
        else:
            trace_indexes = range(initial_frame,len(trace))
        snippet = t_window

    #for roi in rois:
    trace_indexes = np.array(trace_indexes).astype(int)
    strf_prediction = np.zeros(trace.shape[0])
    strf_prediction[:] = np.nan 
    copy_trace = np.zeros(trace.shape[0])
    copy_trace[:] = np.nan 
    curr_sta=roi.STRF_data['strf']
    offset=t_window
    frame_lost_label = -1
    for ix,i in enumerate(trace_indexes):

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
        copy_trace[i] = trace[i]
        
        if ix%1000==0:
            print('Prediction_round %s of %s' %(ix+1,len(trace_indexes))) 
        current_stim_frame = int(stim_dataframe.loc[stim_dataframe['mic_frame'] == i].iloc[0]['stim_frame'])
        stim_chunk = stimulus[current_stim_frame-snippet:current_stim_frame,:,:]
        
        ### check that the snippet contents are timed correctly otherwise create delays in the data to compensate
        ## this seems to be necesary since some frames are staying longer than expected on the screen
        index = stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0].name
        delay_treshold = ((1/float(stim_up_rate))+((1/float(stim_up_rate)/10)))
        if np.any(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>=delay_treshold):
            error_ix = np.where(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>delay_treshold)[0]
            for error in error_ix:
                delay_signature = int(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])[error]/(1/float(stim_up_rate)))
                stim_chunk[error: error + delay_signature,:,:] = stim_chunk[error-1,:,:]
                remaining_len = len(stim_chunk[error + delay_signature:,0,0])
                stim_chunk [error + delay_signature:,:,:] = stim_chunk [error:error+remaining_len,:,:] 
                #delay_count += 1

        strf_prediction[i]= np.sum(stim_chunk*localSTRf) # prediction giving no weight to the different times in STRF

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
    strf_prediction_df = strf_prediction/np.nanmax(strf_prediction) #(strf_prediction - np.nanmean(strf_prediction))/np.nanmean(strf_prediction)

    if control == False:
        roi.STRF_prediction_analysis[restriction_str]['strf_prediction_%s'%(type_)] = strf_prediction_df
        roi.STRF_prediction_analysis[restriction_str]['strf_prediction_%s_relu'%(type_)] = _Relu_func(strf_prediction_df,1,0)
        roi.STRF_prediction_analysis[restriction_str]['prediction_corr_%s'%(type_)] = scipy.stats.pearsonr(copy_trace[~np.isnan(copy_trace)],strf_prediction_df[~np.isnan(copy_trace)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
        roi.STRF_prediction_analysis[restriction_str]['prediction_corr_%s_relu'%(type_)] = scipy.stats.pearsonr(copy_trace[~np.isnan(copy_trace)],roi.STRF_prediction_analysis[restriction_str]['strf_prediction_%s_relu'%(type_)][~np.isnan(copy_trace)]) #[trace_indexes[0]:trace_indexes[-1]],trace[trace_indexes[0]:trace_indexes[-1]]
    else:
        roi.STRF_prediction_analysis['strf_prediction_cont'] = strf_prediction
        roi.STRF_prediction_analysis['prediction_corr_cont'] = scipy.stats.pearsonr(roi.STRF_data['strf_prediction'][initial_frame:],trace[initial_frame:]) 

    #roi.STRF_data['prediction'] = strf_prediction
    #return rois

    # apply simple non-linearity

    return roi.STRF_prediction_analysis[restriction_str]['prediction_corr_%s'%(type_)], roi.STRF_prediction_analysis[restriction_str]['prediction_corr_%s_relu'%(type_)]

def _Relu_func(trace,a,b):
    
    # passes prediction trace through a rectified non-linearity (ReLu)
    scaffold = np.zeros((len(trace),2))
    scaffold[:,0] = np.where(~np.isnan(trace),a*trace + b,np.nan)
    scaffold[:,1] = np.where(~np.isnan(trace),b,np.nan)

    return np.nanmax(scaffold,axis = 1)


def extract_temporal_kernel(roi,savedir):

    """ uses the STRF (STA) estimation pixels around the maximum value of the STA to 
     calculate a temporal kernel for the receptive field 
      this is done after Sridar et al., 2024 (gollisch lab)"""


    local_STRF = roi.STRF_data['strf']#[2*(roi.STRF_data['strf'].shape[0])//3:,:,:]

    # get the median Image and peakvalues per pixel
    STRF_median = np.median(local_STRF, axis=0)
    # Generate the index arrays for the second and third dimensions
    x, y = np.meshgrid(np.arange(240), np.arange(240), indexing='ij')
    peak_values_STRF = local_STRF[np.argmax(np.abs(local_STRF),axis=0),x,y]

    z_score_strf = np.abs((local_STRF[np.argmax(np.abs(local_STRF),axis=0),x,y]) - np.mean(local_STRF,axis=0))/np.std(local_STRF,axis=0)


    # calculate the 6x 1.4826x absolute median deviation as an estimator of the Standard deviation
    #mad = np.median(np.abs(peak_values_STRF-STRF_median))
    mad = np.median(np.abs(local_STRF-STRF_median[np.newaxis,:,:]),axis=0)
    robust_deviation_index = np.abs(peak_values_STRF-mad)
    robust_deviation_filter = 6*1.4826*np.median(mad)
    robust_z_score = 0.6745*np.abs((local_STRF[np.argmax(np.abs(local_STRF),axis=0),x,y]) - np.median(local_STRF,axis=0))/mad

    # alternatively, create Z score filter


    # plot for checking
    fig = plt.figure(figsize=(12,6))
    
    gs = GridSpec(1,3)

    ax = plt.subplot(gs[0])
    max_= np.abs(np.max(z_score_strf))
    im = plt.imshow(z_score_strf,cmap='PuBuGn',vmax=max_,vmin=0)
    ax.title.set_text('z_score_strf')
    cbar=fig.colorbar(im, ax=ax,shrink=0.5)
    ax.axis('off')

    ax = plt.subplot(gs[1])
    max_ = np.abs(np.max(robust_z_score))
    im = plt.imshow(robust_z_score,cmap='PuBuGn',vmax=max_,vmin=0)
    ax.title.set_text('robust_z_score')
    cbar=fig.colorbar(im, ax=ax,shrink=0.5)
    ax.axis('off')

    ax = plt.subplot(gs[2])
    max_ = np.abs(np.max(mad))
    im = plt.imshow(mad,cmap='PuBuGn',vmax=max_,vmin=0)
    ax.title.set_text('mad')
    cbar=fig.colorbar(im, ax=ax,shrink=0.5)
    ax.axis('off')


    Z_deviation_filter = 4

    # find all pixels in the RF that go beyond 6 Z score



    # create a filter of all pixels whose peak intensity exceeds the robust_deviation
    max_proy_strf = np.max(np.abs(roi.STRF_data['strf'] - STRF_median[np.newaxis,:,:]), axis=0)
    Z_pix_filter = np.where(robust_z_score>=Z_deviation_filter,1,0)
    Med_dev_pix_filter = np.where(max_proy_strf>=robust_deviation_filter,1,0)
    
    Z_valid_pixs = np.sum(Z_pix_filter)
    meddev_valid_pixs = np.sum(Med_dev_pix_filter)


    if Z_valid_pixs == 0:
        roi.STRF_data['status'] = 'excluded'
        return 'invalid'
    
    filtered_strf = Z_pix_filter[np.newaxis,:,:]*roi.STRF_data['strf']

    roi.STRF_data['temporal_kernel'] = np.sum(filtered_strf,axis=(1,2))/Z_valid_pixs

    # plot the filtered pixels 
    fig = plt.figure(figsize=(40,6))
    
    gs = GridSpec(1,5)
    ax = plt.subplot(gs[0])
    max_= np.abs(np.max(roi.STRF_data['strf']))
    im = plt.imshow(np.mean(roi.STRF_data['strf'][2*(roi.STRF_data['strf'].shape[0])//3:,:,:],axis=0),cmap='PRGn',vmin=-max_,vmax=max_)
    ax.title.set_text('mean of 0.5s')
    ax.axis('OFF')   

    ax = plt.subplot(gs[1])
    im = plt.imshow(np.mean(filtered_strf[2*(roi.STRF_data['strf'].shape[0])//3:,:,:],axis=0),cmap='PRGn',vmin=-max_,vmax=max_)
    ax.title.set_text('filtered mean of 0.5s')
    ax.axis('OFF')

    ax = plt.subplot(gs[2])
    im = plt.plot(roi.STRF_data['temporal_kernel'])
    ax.title.set_text('temporal kernel')
    plt.subplots_adjust(wspace=0.3)
    
    # find the pixel with the max Z_score
    blurred_z = filters.gaussian(robust_z_score, sigma=2)
    index_Z_max = np.array([np.where(blurred_z==np.max(blurred_z))[0][0],np.where(blurred_z==np.max(blurred_z))[1][0]])

    extract_spatial_kernel(roi,index_Z_max,radius=20)

    ax = plt.subplot(gs[3])
    max_ = np.abs(np.max(roi.STRF_data['spatial_kernel_crop']))
    im = plt.imshow(roi.STRF_data['spatial_kernel_crop'],cmap='PRGn',vmin=-max_,vmax=max_)
    ax.title.set_text('spatial kernel')
    plt.subplots_adjust(wspace=0.3)
    ax.axis('OFF')
    pac.multipage(savedir)

    ax = plt.subplot(gs[4])
    max_ = np.abs(np.max(roi.STRF_data['spatial_kernel']))
    im = plt.imshow(roi.STRF_data['spatial_kernel'],cmap='PRGn',vmin=-max_,vmax=max_)
    ax.title.set_text('spatial kernel')
    plt.subplots_adjust(wspace=0.3)
    ax.axis('OFF')
    pac.multipage(savedir)

    plt.close('all')
    # calculate average time course from these pixels
    return 'valid'

def extract_spatial_kernel(roi,index_Z_max,radius):

    """ uses the STRF (STA) estimation pixels around the maximum value of the STA to 
    calculate a spatial kernel for the receptive field 
    this is done after Sridar et al., 2024 (gollisch lab)"""

    
    # get the temporal kernel 
    temporal_weights = roi.STRF_data['temporal_kernel'][:,np.newaxis,np.newaxis]#/np.max(np.abs(roi.STRF_data['temporal_kernel']))

    # project the 3d strf based on the temporal filter 
    #roi.STRF_data['spatial_kernel'] = np.sum(temporal_weights*roi.STRF_data['strf'],axis=0)#/roi.STRF_data['strf'].shape[0]

    # crop 40 degrees

    scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1])
    radius_scaled = radius/scaling_fact
  
    yi, xi = np.indices((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]))
    circle = (xi - index_Z_max[1])**2 + (yi - index_Z_max[0])**2
    circ_array = np.full((roi.STRF_data['strf'].shape[1], roi.STRF_data['strf'].shape[2]), 0)
    circ_array[np.where((circle <= (radius_scaled)**2))] = 1
    
    roi.STRF_data['spatial_kernel'] = np.sum(temporal_weights*roi.STRF_data['strf'],axis=0)
    roi.STRF_data['spatial_kernel_crop'] = np.sum(temporal_weights*roi.STRF_data['strf'],axis=0)*circ_array


def predict_response_w_kernels(roi,stimulus,type_ = 'test',t_window = 1,restriction='unrestricted'):
    '''
    use The STRF to predict the df/f signal
    if type == 'test' use test data for this, otherwise use training data

    time_window is the lenght of the time window used to apply reverse correlation,
    default is 3 seconds

    '''


    #get test indices and traces
    #test_indices = roi.STRF_data['test_indices']
    #test_trace = roi.STRF_data['test_trace']
    fps = roi.imaging_info['frame_rate']
    trace = copy.deepcopy(roi.strf_trace)
    stim_up_rate = 1/float(roi.stim_info['frame_duration'])

    stimtype = roi.stim_name
    stim_dataframe = pd.DataFrame(roi.stim_info['output_data'], columns = ['entry','rel_time','boutInd','epoch','xpos','ypos','stim_frame','mic_frame'])
    stim_dataframe = stim_dataframe[['stim_frame','rel_time','mic_frame']]
    initial_frame = int(np.ceil(t_window*2*roi.imaging_info['frame_rate']))
    snippet = int(t_window*stim_up_rate)
    
    # set the microscope frame count to 0
    
    starting_signal_frame=stim_dataframe.iloc[0]['mic_frame'] 
    stim_dataframe['mic_frame'] = stim_dataframe['mic_frame'] - starting_signal_frame
    stim_dataframe['rel_time'] = stim_dataframe['rel_time'] - stim_dataframe['rel_time'].iloc[0]

    # filter signal components with period higher than 10 seconds to detrend
    
    trace1 = High_pass(trace,fps,crit_freq=0.03,plot=False)
    
    # use the STRF to predict the test and train signals
    # TODO continue here:

    #if type_ == 'test': #use heldout frames to test strf prediction
    return STRF_response_prediction_w_kernels(roi,trace1,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,held_out_frames = roi.STRF_data['test_indices'],type_=type_)
    # elif type_ == 'train': #use the frames used for reverse correlation for prediction
    #     STRF_response_prediction(roi,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=stimulus,t_window=snippet,n_epochs=1,held_out_frames = roi.STRF_data['test_indices'],type_='train',restrict_space=restrict_space,restrict_time=restrict_space)


def STRF_response_prediction_w_kernels(roi,trace,stim_up_rate,stim_dataframe,initial_frame,stimulus=None,t_window=40,n_epochs=1,held_out_frames = None,type_= 'test'):

    """
    predict neuron response based on its STA (receptive field and the random stimulus)
    t_window should be the same as used to calculate sta

    for now it's implemented for 1 repetition of stimulus

    """
    #localSTRf = copy.deepcopy(roi.STRF_data['strf'])

    try:
        roi.STRF_prediction_analysis
    except:
        roi.STRF_prediction_analysis = {}
    

    #localSTRf = (localSTRf-np.mean(localSTRf))/ np.std(localSTRf)
    # create the structure for prediction analysis storage, this will distinguish between correlations obtained from restricted or unrestricted 
    # RFs 

    try:
        roi.STRF_prediction_analysis['kernel_pred']
    except:
        roi.STRF_prediction_analysis['kernel_pred'] = {}
    
    #distinguish between train and test sets and analyze accordingly

    # if type_ == 'test':
    #     if held_out_frames is not None:
    #         trace_indexes = range(held_out_frames[0],held_out_frames[1]-1)
    #     else:
    #         trace_indexes = range(initial_frame,len(trace))
    #     snippet = t_window
    # elif type_ == 'train':
    #     if held_out_frames is not None:
    #         trace_indexes = np.array(range(initial_frame,held_out_frames[0])) 
    #         #trace_indexes2 = np.array(range(held_out_frames[1],len(trace)))
    #         #trace_indexes = np.concatenate([trace_indexes,trace_indexes2])
    #     else:
    #         trace_indexes = range(initial_frame,len(trace))
    #     snippet = t_window


    # take a copy of STRF

    trace_indexes = range(initial_frame,len(trace))
    snippet = t_window
    # fft_kernel = fft(roi.STRF_data['temporal_kernel'])
    # corrected_fft_kernel=np.zeros_like(fft_kernel)
    # corrected_fft_kernel[30:-1] = fft_kernel[31:]
    # corrected_fft_kernel[0:29] = fft_kernel[1:30]
    # corrected_fft_kernel[0] = corrected_fft_kernel[1]
    
    # fourier_backkernel = ifft(corrected_fft_kernel)
    # plt.figure()
    # plt.plot(fourier_backkernel)
    # plt.figure()
    # plt.plot(roi.STRF_data['temporal_kernel'])
    
    #roi.STRF_data['temporal_kernel'] = fourier_backkernel
    #for roi in rois:
    time_convolution = np.zeros(stimulus[:len(trace),:,:].shape)
    #alternative_ = np.zeros((len(stimulus[:len(trace),:,:])))
    
    offset=t_window
    frame_lost_label = -1
    for ix,i in enumerate(trace_indexes):

        try:
            index = int(stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0]['mic_frame'])
        except IndexError: #this happens when frames are skipped
            frame_lost_label = 0
            time_convolution[i,:,:]=np.nan
            #lost_count += 1
            print ('frame %s is lost'%(i))
        if frame_lost_label >=0 and frame_lost_label < int(np.ceil(1*roi.imaging_info['frame_rate'])):
            frame_lost_label += 1
            continue
        elif frame_lost_label == int(np.ceil(1*roi.imaging_info['frame_rate'])):
            frame_lost_label = -1
    
        if ix%1000==0:
            print('round %s of %s' %(ix+1,len(trace_indexes))) 
        current_stim_frame = int(stim_dataframe.loc[stim_dataframe['mic_frame'] == i].iloc[0]['stim_frame'])
        stim_chunk = stimulus[current_stim_frame-snippet:current_stim_frame,:,:]
        
        ### check that the snippet contents are timed correctly otherwise create delays in the data to compensate
        ## this seems to be necesary since some frames are staying longer than expected on the screen
        index = stim_dataframe.loc[stim_dataframe['mic_frame'].astype(int)==i].iloc[0].name
        delay_treshold = ((1/float(stim_up_rate))+((1/float(stim_up_rate)/10)))
        
        if np.any(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>=delay_treshold):
            error_ix = np.where(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])>delay_treshold)[0]
            for error in error_ix:
                delay_signature = int(np.diff(stim_dataframe.loc[1+(index-snippet):index+1]['rel_time'])[error]/(1/float(stim_up_rate)))
                stim_chunk[error: error + delay_signature,:,:] = stim_chunk[error-1,:,:]
                remaining_len = len(stim_chunk[error + delay_signature:,0,0])
                stim_chunk [error + delay_signature:,:,:] = stim_chunk [error:error+remaining_len,:,:] 
                #delay_count += 1


        # perform one convolution step in the temporal domain
        kernel_slice = len(roi.STRF_data['temporal_kernel'])-t_window
        #alternative_[i] = np.sum(np.sum(stim_chunk*roi.STRF_data['spatial_kernel'][np.newaxis,:,:],axis=(1,2))*roi.STRF_data['temporal_kernel'][kernel_slice:])
        time_convolution[i,:,:]= np.sum(stim_chunk*roi.STRF_data['temporal_kernel'][kernel_slice:,np.newaxis,np.newaxis],axis=0) 

        # use the spatial filter to multiply the result and integrate

    linear_activation = np.sum(time_convolution*roi.STRF_data['spatial_kernel'][np.newaxis,:,:],axis=(1,2))
    linear_activation = (linear_activation - np.mean(linear_activation))/np.mean(linear_activation)
    #alternative = (alternative_-np.mean(alternative_))/np.mean(alternative_)
    #rectified_act = np.where(linear_activation>0,linear_activation,0)
    
    # slice the trace and prediction and exclude nans
    trace_train = trace[initial_frame:held_out_frames[0]-2]
    trace_test = trace[held_out_frames[0]-2:held_out_frames[1]-2]
    

    roi.STRF_prediction_analysis['kernel_pred'] = {}
    roi.STRF_prediction_analysis['kernel_pred']['trace'] = linear_activation #(linear_activation - np.mean(linear_activation))/np.mean(linear_activation)
    roi.STRF_prediction_analysis['kernel_pred']['corr_train'] = scipy.stats.pearsonr(linear_activation[initial_frame:held_out_frames[0]-2],trace_train) 
    roi.STRF_prediction_analysis['kernel_pred']['corr_test'] = scipy.stats.pearsonr(linear_activation[held_out_frames[0]-2:held_out_frames[1]-2],trace_test) 

    #return rois


# def plot_predictions(roi):

#     """takes prediction traces and plots them """
#     'aaa'