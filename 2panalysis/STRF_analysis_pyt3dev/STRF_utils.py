# utilities for the analysis of spatial-temporal receptive fields

from calendar import c
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
from scipy.ndimage import map_coordinates
from skimage.draw import line
from skimage.measure import profile_line
import glob
import scipy.ndimage
from scipy.ndimage import zoom
import imageio
import os
import copy
from scipy import signal
import seaborn as sns
from sklearn.decomposition import NMF

import sys

code_path= r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis\helpers' #Juan desktop
sys.path.insert(0, code_path) 
import post_analysis_core as pac
import process_mov_core as pmc
import ROI_mod

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

def fit_double_gaussian2D(rois):


    print('not yet implemented')

def gaussian_2d(x, y, A, x0, y0, sigma_x, sigma_y, C):
    """2D Gaussian function for fitting."""
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

def plot_NMF_segregated_components(components_xy,components_t,savedir,type = 'pos'):
    
    """
    plots the first n spatial and temporal components of a svd decomposition 
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
        else:
            max_val =  np.max(components_xy[i])
    
    for i in range(len(components_xy)):
        if np.any(components_xy[i]) < 0: 
            type_ = 'neg_comps_NMF'
        else:
            type_ = 'pos_comps_NMF'

        plot_spatial_temporal_components(components_xy[i],components_t[i],savedir,type = type_,max_val=max_val,min_val=min_val)

    return None

def plot_spatial_temporal_components(components_xy,components_t,savedir,type = 'mixed',max_val=None,min_val=None):

    """
    plots the first n spatial and temporal components of a svd decomposition 
    from an STRF

    input:
    components_xy: array(x,y,components)
    components_t: array(temporalweights,components)

    """
    if max_val is None and min_val is None:
        max_val = np.max(components_xy)
        min_val = np.min(components_xy)

    fig = plt.figure(figsize=(15,10))
    gs = GridSpec (2,components_xy.shape[2])

    for i in range(components_xy.shape[2]):
        
        plt.rcParams.update({'font.size' : 10})
        
        ax_xy = fig.add_subplot(gs[0,i])
        ax_t = fig.add_subplot(gs[1,i])
        
        im = ax_xy.imshow(components_xy[:,:,i],cmap='PRGn',vmax=max_val,vmin=min_val)
        ax_xy.set_axis_off()
        cbar=fig.colorbar(im, ax=ax_xy, aspect = 8)
        cbar.ax.tick_params(labelsize=10)
        ax_t.plot(components_t[:,i])

        ax_xy.title.set_text('xy_component_%s'%(i))
        ax_t.title.set_text('t_component_%s'%(i))
    #pmc.multipage(savedir + 'type' + 'components.pdf')

def fit_component_gaussians(components_xy):

    """fits as many gaussians as entries in the 3rd axis of the input
    
    input:
    components_xy: array(x,y,components)
    
    """

def apply_circular_mask(roi,radius):
    
    """filters out the area beyond radius from the point where STRF is absolute highest
    inputs:
        roi: roi object containing STRF information
        radius: the radius of the area to be included
    
    output: masked STRF
    """

    scaling_fact = 80.0/(roi.STRF_data['strf'].shape[1])
    radius_scaled = radius/scaling_fact
    indices = np.where(np.abs(roi.STRF_data['strf'])==np.max(np.abs(roi.STRF_data['strf'])))
    indices = indices[1][0], indices[2][0]
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
    
    dims = roi.STRF_data['strf'].shape
    
    flattened_mat = _flatten_strf (dims,masked_array)
      
    U,S,Vt = scipy.linalg.svd(flattened_mat, full_matrices=False)
    
    #U should contain temporal components size (dims[1]*dims[1],dims[0])

    first_10x_components = U[:,0:components]
    first_10t_components = np.transpose(Vt[0:components,:])

    # un-flatten spatial components
    components_xy = np.zeros((dims[1],dims[2],components))
    for component in range(0,components):
        components_xy[:,:,component] = np.reshape(first_10x_components[:,component],(dims[1],dims[2]))

    return components_xy, first_10t_components

def segregated_NMf(roi,masked_array,regularization = 0.0001, components =5):
    
    """ segregates a STRF in positive and negatively correlated arrays 
        and performs non-negative-matrix multiplication on them independently, 
        then it concatenates the components obtained
        inputs:
        roi: roi object with strf data 
        masked_array: strf passed through a mask or full strf array
        regularization: parameter to control sparseness of the NMF result
    """
    dims = roi.STRF_data['strf'].shape
    flattened_mat = _flatten_strf (dims,masked_array)

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

