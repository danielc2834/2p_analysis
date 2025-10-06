'''Functions related to analysis of Calcium Imaging recordings'''
################################################ IMPORTS ################################################
import os, pickle, glob, tifffile, cv2, copy, warnings, preprocessing_params, re, random, math
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.fftpack import fft2,fftshift,ifftshift,ifft2  
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
import caiman as cm 
from caiman.motion_correction import MotionCorrect
from roipoly import RoiPoly
from Helpers.xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize, getMicRelativeTime
from Helpers import ROI_mod
from Helpers import summary_figures as sf
import pandas as pd
from itertools import islice
import seaborn as sns
from scipy.ndimage import rotate
################################################ Plotting raw traces ################################################
def run_matplotlib_params():
    plt.style.use('default')
    plt.style.use('seaborn-v0_8-talk')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)    
    plt.rcParams["axes.titlesize"] = 'medium'
    plt.rcParams["axes.labelsize"] = 'small'
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams["legend.fontsize"] = 'small'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["figure.titleweight"] = 'bold'
    plt.rcParams["figure.titlesize"] = 'medium'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['legend.loc'] = 'upper right'
    
    c_dict = {}
    c_dict['dark_gray'] = np.array([77,77,77]).astype(float)/255
    c_dict['light_gray'] = np.array([186,186,186]).astype(float)/255
    c_dict['green1'] = np.array([102,166,30]).astype(float)/255
    c_dict['green2']=np.array([179,226,205]).astype(float)/255
    c_dict['green3'] = np.array([27,158,119]).astype(float)/255
    c_dict['orange']  = np.array([201,102,47]).astype(float)/255
    c_dict['red']  = np.array([228,26,28]).astype(float)/255
    c_dict['magenta']  = np.array([231,41,138]).astype(float)/255
    c_dict['purple']  = np.array([117,112,179]).astype(float)/255
    c_dict['yellow'] = np.array([255,255,51]).astype(float)/255
    c_dict['brown'] = np.array([166,86,40]).astype(float)/255
    
    c_dict['L3_'] = np.array([102,166,30]).astype(float)/255 # Dark2 Green
    c_dict['Tm9'] = np.array([27,158,119]).astype(float)/255 # Dark2 Weird green
    c_dict['L1_'] = np.array([230,171,2]).astype(float)/255 # Dark2 Yellow
    c_dict['L2_'] = np.array([55,126,184]).astype(float)/255 # blue
    c_dict['Mi1'] = np.array([166,118,29]).astype(float)/255 # Dark2 dark yellow
    c_dict['Mi4'] = np.array([217,95,2]).astype(float)/255 # Dark2 orange
    c_dict['Tm3'] = np.array([231,41,138]).astype(float)/255 # Dark2 magenta
    
    

    
    c = []
    c.append(c_dict['dark_gray'])
    c.append(c_dict['light_gray'])
    c.append(c_dict['green1']) # Green
    c.append(c_dict['orange']) # Orange
    c.append(c_dict['red']) # Red
    c.append(c_dict['magenta']) # magenta
    c.append(c_dict['purple'])# purple
    c.append(c_dict['green2']) # Green
    c.append(c_dict['yellow']) # Yellow
    c.append(c_dict['brown']) # Brown
    
    
    return c, c_dict


def plot_roi_traces(rois,analysis_type,summary_save_dir):
    #try to create raw_traces_folder


    #clear any old traces. 


    if analysis_type=='8D_edges_find_rois_save' or analysis_type=='12_dir_random_driftingstripe' or analysis_type=='1-dir_ON_OFF':
        ROI_mod.plot_traces_edges(rois,summary_save_dir)
        #TODO. plot all raw traces in the same plot
    elif analysis_type=='1-dir_edge_1pol':
        ROI_mod.plot_tracesEdges_against_variables(rois,summary_save_dir)
    elif analysis_type=='moving_gratings':
        ROI_mod.plot_tracesEdges_against_variables(rois,summary_save_dir)
        ROI_mod.plot_psds(rois,summary_save_dir)
    else:
        print('raw trace plotting not implemented for this analysis type')
        
        
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
        save_name = 'ROI_props_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
        print('ROI property images saved')
        
def divideEpochs(rawStimData, epochCount, isRandom, framePeriod, trialDiff=0.20, overlappingFrames=0, firstEpochIdx=0, epochColumn=3, imgFrameColumn=7, incNextEpoch=True, checkLastTrialLen=True):
    """divides the raw trace into epochs
    
    Parameters
    ---------
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    isRandom : int

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    overlappingFrames : int

    firstEpochIdx : int
        Default: 0

        Index of the first epoch.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    incNextEpoch :
    checkLastTrialLen :

    Returns
    ---------
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [Z, D]] where first term is the trial beginning
        (first of first) and end (second of first), and second term is the
        baseline start (first of second) and end (second of second) for that
        trial.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).

    isRandom :
    """
    trialDiff = float(trialDiff)
    firstEpochIdx = int(firstEpochIdx)
    overlappingFrames = int(overlappingFrames)
    trialCoor = {}
    fullEpochSeq = []
    if isRandom == 0:
        fullEpochSeq = range(epochCount)
        # if the key is zero, that means isRandom is 0
        # this is for compatibitibility and
        # to make unified workflow with isRandom == 1
        trialCoor[0] = []
    elif isRandom == 1:
        # in this case fullEpochSeq is just a list of dummy values
        # important thing is its length
        # it's set to 3 since trials will be sth like: 0, X
        # if incNextEpoch is True, then it will be like : 0,X,0
        fullEpochSeq = range(2)
        for epoch in range(1, epochCount):
            # add epoch numbers to the dictionary
            # do not add the first epoch there
            # since it is not the exp epoch
            # instead it is used for baseline and inc coordinates
            trialCoor[epoch] = []
    if incNextEpoch:
        # add the first epoch
        fullEpochSeq.append(firstEpochIdx)
    elif not incNextEpoch:
        pass
    # min and max img frame numbers for each and every trial
    # first terms in frameBaselineCoor are the trial beginning and end
    # second terms are the baseline start and end for that trial
    currentEpochSeq = []
    frameBaselineCoor = [[0, 0], [0, 0]]
    nextMin = 0
    baselineMax = 0
    for line in rawStimData:
        if (len(currentEpochSeq) == 0 and
                len(currentEpochSeq) < len(fullEpochSeq)):
            # it means it is the very beginning of a trial block.
            # in the very first trial,
            # min frame coordinate cannot be set by nextMin.
            # this condition satisfies this purpose.
            currentEpochSeq.append(int(line[epochColumn]))
            if frameBaselineCoor[0][0] == 0:
                frameBaselineCoor[0][0] = int(line[imgFrameColumn])
                frameBaselineCoor[1][0] = int(line[imgFrameColumn])
        elif (len(currentEpochSeq) != 0 and len(currentEpochSeq) < len(fullEpochSeq)):
            # only update the current epoch list
            # already got the min coordinate of the trial
            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))
            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for the very first trial
                    frameBaselineCoor[1][1] = (int(line[imgFrameColumn])- overlappingFrames)
        elif len(currentEpochSeq) == len(fullEpochSeq):
            if nextMin == 0:
                nextMin = int(line[imgFrameColumn]) + overlappingFrames
            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))
            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for all the trials except the very first trial
                    baselineMax = int(line[imgFrameColumn]) - overlappingFrames
        else:
            frameBaselineCoor[0][1] = (int(line[imgFrameColumn])- overlappingFrames)
            if frameBaselineCoor[0][1] > 0:
                if isRandom == 0:
                    # if the key is zero, that means isRandom is 0
                    # this is for compatibitibility and
                    # to make unified workflow isRandom == 1
                    trialCoor[0].append(frameBaselineCoor)
                elif isRandom == 1:
                    # get the epoch number
                    # epoch no should be the 2nd term in currentEpochSeq
                    expEpoch = currentEpochSeq[1]
                    trialCoor[expEpoch].append(frameBaselineCoor)
            # this is just a safety check
            # towards the end of the file, the number of epochs might
            # not be enough to form a trial block
            # so if the max img frame coordinate is still 0
            # it means this not-so-complete trial will be discarded
            # only complete trials are appended to trial coordinates
            # if it has a max frame coord, it is safe to say
            # it had nextMin in frameBaselineCoor
            # print(currentEpochSeq)
            currentEpochSeq = []
            currentEpochSeq.append(firstEpochIdx)
            # each time currentEpochSeq resets means that
            # one trial block is complete
            # adding firstEpochIdx is necessary
            # otherwise currentEpochSeq will shift by 1
            # after every trial cycle
            # now that the frame coordinates are stored
            # can reset it
            # and add min coordinate for the next trial
            # then add the max baseline coordinate for the next trial
            frameBaselineCoor = [[0, 0], [0, 0]]
            frameBaselineCoor[0][0] = nextMin
            frameBaselineCoor[1][0] = nextMin
            frameBaselineCoor[1][1] = baselineMax
            nextMin = 0
            baselineMax = 0
    # @TODO: no need to separate isRandoms, make a unified for loop
    if checkLastTrialLen:
        if isRandom == 0:
            lenFirstTrial = trialCoor[0][0][0][1] - trialCoor[0][0][0][0]
            lenLastTrial = trialCoor[0][-1][0][1] - trialCoor[0][-1][0][0]
            if ((lenFirstTrial - lenLastTrial) * framePeriod) >= trialDiff:
                trialCoor[0].pop(-1)
                print("Last trial is discarded since the length was too short")
        elif isRandom == 1:
            for epoch in trialCoor:
                delSwitch = False
                lenFirstTrial = (trialCoor[epoch][0][0][1] - trialCoor[epoch][0][0][0])
                lenLastTrial = (trialCoor[epoch][-1][0][1] - trialCoor[epoch][-1][0][0])
                if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                    delSwitch = True
                if delSwitch:
                    print("Last trial of epoch " + str(epoch) + " is discarded since the length was too short")
                    trialCoor[epoch].pop(-1)
    trialCount = []
    if isRandom == 0:
        # there is only a single key in the trialCoor dict in this case
        trialCount.append(len(trialCoor[0]))
    elif isRandom == 1:
        epochTrial = []
        for epoch in trialCoor:
            epochTrial.append(len(trialCoor[epoch]))
        # in this case first element in trialCount is min no of trials
        # second element is the max no of trials
        trialCount.append(min(epochTrial))
        trialCount.append(max(epochTrial))
    return trialCoor, trialCount, isRandom

def get_stim_xml_params(xml_path,original_stimDir,Tseries_len): #from sebastians code
    """ Gets the required stimulus and imaging parameters
    
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
    #  Finding the frame period (1/FPS) and layer position
    xmlFile = f'{xml_path}/{os.path.basename(xml_path)}.xml'
    stimOuputFile = f'{xml_path}/{os.path.basename(xml_path)}_stim_output.txt'
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
    (stimType, rawStimData,raw_stimkeys) = readStimOut(stimOuputFile,Tseries_len, skipHeader=3) # Seb: skipHeader = 3 for _stimulus_ouput from 2pstim
    #TODO continue here
    # Stimulus information
    (stimInputFile,stimInputData) = readStimInformation(stimType=stimType,original_stimDir=original_stimDir)
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

def readStimOut(stimOutFile,Tseries_len,skipHeader):
    """Read and get the stimulus output data.

    Parameters
    ==========
    stimOutFile : str
        Stimulus output file path.

    skipHeader : int, optional
        Default: 1

        Number of lines to be skipped from the beginning of the stimulus
        output file.

    Returns
    =======
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file.

    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.
    """
    # skip the first line since it is a file path
    
    # if cplusplus==True:
    #     skipHeader=1
    #     Stim_content=open(stimOutFile, "r")
    #     line=Stim_content.readlines()
    #     if 'nothing' in line[0] or 'Nothing' in line[0]:
    #         skipHeader=2
    #     rawStimData = np.genfromtxt(stimOutFile, dtype='float',
    #                                skip_header=skipHeader)#,delimiter=',') #Juan comment: to deal with c++ outputfiles # Seb: delimiter = ',' for _stimulus_ouput from 2pstim
    # else:
    Stim_content=open(stimOutFile, "r")
    keys=Stim_content.readlines()
    keys=keys[2].split('\n')[0].split(',')
    rawStimData = np.genfromtxt(stimOutFile, dtype='float',
                                skip_header=skipHeader,delimiter=',')
    # also get the file path
    # do not mix it with numpy
    # only load and read the first line
    
    #erase any extra entries (entries that go beyond the number of frames), update made by Juan
    framefilter=rawStimData[:,-1]<=Tseries_len
    framefilter=np.sum(framefilter)
    rawStimData=rawStimData[:np.sum(framefilter),:]
    ##
    
    stimType = "stimType"
    with open(stimOutFile, 'r') as infile:
        lines_gen = islice(infile, 2) # Seb: the stimType is printed in line'2' of the _stimulus_ouput from 2pstim
        for line in lines_gen:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            line = re.sub(' ', '', line)
            stimType = line
            stimType=[line.split('\\')[-1]][0] #Juan this line deals with c++ stims
            
            # break #Seb: commented this out

    return stimType, rawStimData, keys

def readStimInformation(stimType, original_stimDir):
    """
    Parameters
    ==========
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file. *Required* if `gui` is FALSE and `stimInputDir`
        is given a non-default value.

    stimInputDir : str
        Path of the directory to look for the stimulus input.

    Returns
    =======
    stimInputFile : str
        Path to the stimulus input file (which contains stimulus parameters,
        not the output file).

    stimInputData : dict
        Lines of the stimulus generator file. Keys are the first terms of the
        line (the parameter names), values of the key is the rest of the line
        (a list).
    
    isRandom : int
        A that changes how epochs are randomized 

    Notes
    =====
    This function does not return the values below anymore:

    - epochDur : list
        Duration of the epochs as in the stimulus input file.

    - stimTransAmp : list

    """
    stimType = stimType.split('/')[-1] # Sen: \\ to / 
    if 'Search' in stimType:
        stimType=stimType.split('Search_')[-1]
    stimInputFile = glob.glob(os.path.join(original_stimDir, stimType))[0] #Juan: names changed
    

    # Seb: commnted out this and replaced for waht is below
    # flHandle = open(stimInputFile, 'r')
    # stimInputData = {}
    # for line in flHandle:
    #     line = re.sub('\n', '', line)
    #     line = re.sub('\r', '', line)
    #     line = line.split('\t')
    #     stimInputData[line[0]] = line[1:]

    
    ###################################### Seb version ###################################
    # To avoid that extra invisible 'tabs' end up in the stimInputFile dictionary
    stimInputData = {}
    with open(stimInputFile) as file:
        for line in file:
            curr_list = line.split()

            if not curr_list:
                 continue
                    
            key = curr_list.pop(0)
                
            if len(curr_list) == 1 and not "Stimulus." in key:
                try:
                    stimInputData[key] = int(curr_list[0])
                except ValueError:
                    stimInputData[key] = curr_list[0]
                continue 
                    
            if key.startswith("Stimulus."):
                key = key[9:]
                    
                if key.startswith("stimtype"):
                    stimInputData[key] = list(map(str, curr_list))
                else: 
                    stimInputData[key] = list(map(float, curr_list))

#########################################################################################

#        epochDur = stimInputData['Stimulus.duration']
    
#        stimTransAmp = stimInputData['Stimulus.stimtrans.amp']
#
#        epochDur = [float(sec) for sec in epochDur]
#        stimTransAmp = [float(deg) for deg in stimTransAmp]

    return stimInputFile, stimInputData

def getEpochCount(rawStimData, epochColumn=3):
    """Get the total epoch number.

    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    Returns
    =======
    epochCount : int
        Total number of epochs.
    """
    # get the max epoch count from the rawStimData
    # 4th column is the epoch number
    # add plus 1 since the min epoch no is zero
    
    # BG edit: Changed the previous epoch extraction, which uses the maximum 
    # number + 1 as the epoch number, to a one finding the unique values and 
    # taking the length of it
    epochCount = np.shape(np.unique(rawStimData[:, epochColumn]))[0]
    print("Number of epochs = " + str(epochCount))

    return epochCount

def divide_trials_1epoch(arr):
    # Input validation
    if not isinstance(arr, np.ndarray) or arr.shape[1] != 8:
        raise ValueError("Input must be a NumPy array with shape (n, 8)")
    
    # Initialize the output list
    output = []
    
    # Get unique values in the third column, preserving order
    unique_vals, indices = np.unique(arr[:, 2], return_index=True)
    indices = np.sort(indices)  # Ensure indices are sorted
    
    if len(unique_vals) == 1:
        return {0:[[int(arr[0][-1]),-1],[int(arr[0][-1]),-1]]}, 1
        # trialCount = 1

    # Iterate through unique values
    for i in range(len(indices)):
        uniq_val = unique_vals[i]
        start_index = indices[i]
        if (i + 1) < len(indices):
            end_index = indices[i + 1] - 1  # -1 to adjust for the next start
        else:
            end_index = -1  # Handle the last segment
            end_index = -1
        # Extract the start and end value from the eighth column
        start_val = int(arr[start_index, 7])
        end_val = int(arr[end_index, 7] if end_index != -1 else arr[-1, 7])
        
        output.append([start_val, end_val])
    return {0:output}, unique_vals

def divide_all_epochs(rawStimData, epochCount, framePeriod, trialDiff=0.20,
                      epochColumn=3, imgFrameColumn=7,checkLastTrialLen=True):
    """
    
    Finds all trial and epoch beginning and end frames
    
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [X, Y]] where X is the trial beginning and Y 
        is the trial end.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).
    """
    trialDiff = float(trialDiff)
    trialCoor = {}
    
    for epoch in range(0, epochCount):
        
        trialCoor[epoch] = []

    previous_epoch = []
    for line in rawStimData:
        
        current_epoch = int(line[epochColumn])
        
        if (not(previous_epoch == current_epoch )): # Beginning of a new epoch trial
            
            
            if (not(previous_epoch==[])): # If this is after stim start (which is normal case)
                epoch_trial_end_frame = previous_frame
                trialCoor[previous_epoch].append([[epoch_trial_start_frame, epoch_trial_end_frame], 
                                            [epoch_trial_start_frame, epoch_trial_end_frame]])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                previous_epoch = int(line[epochColumn])
                
            else:
                previous_epoch = int(line[epochColumn])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                
        previous_frame = int(line[imgFrameColumn])
        
    if checkLastTrialLen:
        for epoch in trialCoor:
            delSwitch = False
            lenFirstTrial = (trialCoor[epoch][0][0][1]
                             - trialCoor[epoch][0][0][0])
            lenLastTrial = (trialCoor[epoch][-1][0][1]
                            - trialCoor[epoch][-1][0][0])
    
            if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                delSwitch = True
    
            if delSwitch:
                print("Last trial of epoch " + str(epoch)
                      + " is discarded since the length was too short")
                trialCoor[epoch].pop(-1)
                
    trialCount = []
    epochTrial = []
    for epoch in trialCoor:
        epochTrial.append(len(trialCoor[epoch]))
    # in this case first element in trialCount is min no of trials
    # second element is the max no of trials
    trialCount.append(min(epochTrial))
    trialCount.append(max(epochTrial))
       

    return trialCoor, trialCount

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
    
def run_analysis(analysis_params, rois,experiment_conditions,
                 imaging_information,summary_save_dir, cycle, expected_polarity, mean_image,
                 save_fig=True,fig_save_dir = None, 
                 exp_ID=None,df_method=None,keep_prev=False,**kwargs): #edited by Juan
    """
    asd
    """
    
    analysis_type = analysis_params['analysis_type']
    figtitle = 'Summary: %s Gen: %s | Age: %s | Z: %d' % \
           (experiment_conditions['MovieID'], #.split('-')[0] juan commented
            experiment_conditions['treatment'], experiment_conditions['Age'],
            imaging_information['depth'])

    # if cycle==1:
    #     'flag'
    
    for roi in rois:
        if 'STRF' in analysis_type or 'Frozen_noise' in analysis_type:
            roi.setSourceImage(mean_image)
        else:
            roi.calculate_reliability()
            roi.calculate_stim_signal_correlation()
            roi.findMaxResponse_all_epochs() #polarity is relevant for FFF stimuli. if inverted neurons are present or stimulus is different it should be None
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
        run_matplotlib_params()
        mean_TFL = np.mean([np.array(roi.tfl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        
        ax=sns.heatmap(mean_TFL, cmap='coolwarm',center=0,
                    xticklabels=np.array(rois[0].tfl_map.columns.levels[1]).astype(float),
                    yticklabels=np.array(rois[0].tfl_map.index),
                    cbar_kws={'label': '$\Delta F/F$'})
        ax.invert_yaxis()
        plt.title('TFL map')
        
        fig = plt.gcf()
        f0_n = 'Summary_TFL_%s' % (exp_ID)
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False,dpi=300)
        
    elif analysis_type == 'lum_con_gratings':
        
        rois = ROI_mod.analyze_lum_con_gratings(rois)
        run_matplotlib_params()
        mean_CL = np.mean([np.array(roi.cl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        
        ax=sns.heatmap(mean_CL, cmap='coolwarm',center=0,
                    xticklabels=np.array(rois[0].cl_map.columns.levels[1]).astype(float),
                    yticklabels=np.array(rois[0].cl_map.index),
                    cbar_kws={'label': '$\Delta F/F$'})
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
    elif analysis_type == '5sFFF_analyze_save':
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
        fig.savefig('%s.png'% f1_n,
                    transparent=False,dpi=300)
        
    elif analysis_type == '8D_edges_find_rois_save' or analysis_type == '4D_edges':
        
        for roi in rois:
            roi.calculate_DSI_PD(method='Mazurek')
            roi.calculate_CSI(frameRate=imaging_information['frame_rate'])
    
    
        rois = ROI_mod.map_RF_adjust_edge_time(rois, save_path = summary_save_dir, edges=True, delay_use = True)

    elif analysis_type == '12_dir_random_driftingstripe':
        
        for roi in rois:
            roi.calculate_DSI_PD(method='Mazurek')

    elif (analysis_type == 'stripes_OFF_delay_profile') or \
        (analysis_type == 'stripes_ON_delay_profile'):
        rois=ROI_mod.generate_time_delay_profile_combined(rois)
        
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
        ax.set_axis_labels(xlabel='Degrees ($^\circ$)',ylabel='$R^2$')
        
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
        for roi in rois:
            roi.calculate_freq_powers()
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

    elif analysis_type == 'shifted_STRF_temporal':
        ROI_mod.reverse_correlation_analysis_JF(rois,kwargs['stim_dir'], test_frames=[[0,250]]) #for 17ms update or [[0, 250]] for 50ms update
    return rois

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