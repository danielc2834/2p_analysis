# -*- coding: utf-8 -*-
"""
Created on Tue Nov  28 14:27:35 2023

@author: Juan Felipe, Chrisschna. It uses functions created by Burak Gur. 
Modified versions of functions from burak Gur are included in the script 
"""
import os
import glob
import cv2
import tifffile
from scipy.signal.windows import gaussian
from PIL import Image
from scipy.fftpack import fft2,fftshift,ifftshift,ifft2  
import re
import time
from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize,getMicRelativeTime
import numpy as np
import matplotlib.pyplot as plt
import logging
import caiman as cm 
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
# from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.params import CNMFParams


"Motion correction parameters" #modify the following according to your exp

max_shifts = (6, 6)  # default(6, 6) maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  #default(48, 48) create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  #default(24, 24) overlap between pathes (size of patch strides+overlaps)
max_deviation_rigid = 3   #default(3) maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = True # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
nonneg_movie=False

#params
opts_dict = {
        'max_shifts' : max_shifts, 
        'strides' : strides,
        'overlaps' : overlaps,    
        'max_deviation_rigid' : max_deviation_rigid,      
        'pw_rigid' : pw_rigid,
        'shifts_opencv' : shifts_opencv,
        'border_nan' : border_nan, 
        'nonneg_movie': nonneg_movie,
}

#create a single options variable to pass into caiman motion correction algorithm
opts = CNMFParams(params_dict=opts_dict)

def produce_tif_list(Tser,mot_corr,channel='2'):
    """
    make list of tif files with Tseries path
    
    arg:
        mot_corr(bool): boolean telling if motion correction is to be performed
        Tser (string): path for Tseries folder
        
    returns:
        list: tif file names list
    """
    ch_string='*ch'+channel+'*ome*.tif'
    files=sorted(glob.glob(os.path.join(Tser,ch_string)))
    if mot_corr:
        processed_files=sorted(glob.glob(os.path.join(Tser,'mot_corr*.tif'))) #if motion corrected files exists then take them out from list    
        filesc=sorted(glob.glob(os.path.join(Tser,'NMcorr*.tif')))
    else:
        processed_files=sorted(glob.glob(os.path.join(Tser,'NMcorr*.tif')))
        filesc=sorted(glob.glob(os.path.join(Tser,'mot_corr*.tif')))
    file_list=sorted(list(set(files)-set(processed_files)-set(filesc)))
    return file_list,processed_files
    # os.chdir(Tser_path[Tser])

def extract_xml_info(Tser): #Juan: function dropped. opted for buraks version
    """
    extract relevant metadata from xml file within Tseries folder
    
    arg:
        Tser (string): path for Tseries folder
        
        
    returns:
        scanner(string): describes recording scanner. resonant or galvo
        frame_period(float): duration of frame uptake
        im_perframe(int): number of imaging frames averaged for one aquired frame  
        samp_rate(float):sampling oeriod taking into account the averaging during aquisition
    """
    os.chdir(Tser)
    sttr=os.path.split(os.path.split(Tser)[0])[1]
    sttr=sttr +'.xml'
    content = open(sttr).read();
    scanner=re.findall('(.*)key="activeMode" (.*)', content)[0][1]
    scanner=str(scanner.split('"')[1])
    frame_period = re.findall('(.*)key="framePeriod" (.*)', content) 
    if scanner=='ResonantGalvo':    
        frame_period=float(frame_period[1][1].split('"')[1])
    else:
        frame_period=float(frame_period[0][1].split('"')[1])
    im_perframe=re.findall('(.*)key="rastersPerFrame" (.*)', content)
    im_perframe=int(im_perframe[0][1].split('"')[1])
    samp_rate=im_perframe*frame_period
    return frame_period,im_perframe,samp_rate

def motionStabilize(stack, path):
    # use mean image of 40 frames as template
    im1 =  np.mean(stack[0:-1], axis = 0).astype("float32")
    template=cv2.GaussianBlur((im1),(3,3),0)
    # cv2.imshow("Template", 10*cv2.resize(template, None, fx=5, fy=5))
    # cv2.imshow("image", 10*cv2.resize(im1, None, fx=5, fy=5))
    # cv2.waitKey(0)

    # Find size of image
    sz = im1.shape

    # Motion correction
    motCorr=[]
    count=-1
    LOG_EVERY_N = 100
    for image in stack:
        count+=1
        if (count % LOG_EVERY_N) == 0:
            print(count)

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION# cv2.MOTION_EUCLIDEAN #

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 10000

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-20

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        try:
            image_t=image.astype(np.float32)
            # _, image_t = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
            # Run the ECC algorithm. The results are stored in warp_matrix.
            (cc, warp_matrix) = cv2.findTransformECC (template,cv2.GaussianBlur(image_t,(3,3),0),warp_matrix, warp_mode, criteria)

            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                # Use warpPerspective for Homography
                image_aligned = cv2.warpPerspective (image_t, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else :
                # Use warpAffine for Translation, Euclidean and Affine
                image_aligned = cv2.warpAffine(image_t, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            # _, image_t = cv2.threshold(image_aligned, 70, 255, cv2.THRESH_TOZERO)   
        except:
            # image_aligned = image_t
            print('no significant motion')
            
        motCorr.append(image_aligned.astype(np.uint16))
        # motCorr.append(image_t.astype(np.uint16))
    
        # Show results
        """cv2.imshow("Image 1", cv2.resize(10*template, None, fx=5, fy=5))
        cv2.imshow("Image 2", cv2.resize(10*((image/256).astype('uint8')), None, fx=5, fy=5))
        cv2.imshow("Aligned Image 2", cv2.resize(10*((image_aligned/256).astype('uint8')), None, fx=5, fy=5))
        cv2.waitKey(0)"""
    mot_avg = np.mean(np.array(motCorr), axis = 0)
    tifffile.imwrite(path.split('.tif')[0]+'_motCorr.tif', np.array(motCorr))
    tifffile.imwrite(path.split('.tif')[0]+'_motavg.tif', mot_avg)

def motionStabilizeCaiman(movie, opts, path):
    # create a motion correction object
    mc = MotionCorrect(movie, **opts.get_group('motion'))

    # save the motion corrected object as a memory mappable file (not on disk but into memory)
    mc.motion_correct(save_movie=True)

    # load motion corrected movie from the memory (previous line)
    m_rig = cm.load(mc.mmap_file)
    # fix borders
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
    mot_avg = mc.total_template_rig
    # save the motion corrected movie on the disk with the name "_motCorr"
    # m_random = cm.movie(m_rig)
    m_rig.save(path.split('.tif')[0]+'_motCorr.tif')
    tifffile.imwrite(path.split('.tif')[0]+'_motavg.tif', mot_avg)

def motionStabilize_tifsInFolder(path, usecaiman=bool):
    
    image_list= []
    cm_list=[]

    for filename in glob.glob(path+'/*.ome.tif'):
            image_data = tifffile.imread(filename, key=0)
            cm_images = filename
            cm_list.append(cm_images)
            image_list.append(image_data)

    image_stack = np.stack(image_list)
    tifffile.imsave(path.split('.tif')[0]+'_motStack.tif', np.array(image_stack))

    # Supress warnings. But can be modified by setting logging.ERROR to logging.WARNING
    logging.basicConfig(format= "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.ERROR)

    # load all the .ome.tif files as a movie 
    if usecaiman==True:
        single_movie = cm.load(cm_list[0], fr=10)
        # according to docs motion percived better when downsampled but we can skip this step
        downsample_ratio = .2 
        motionStabilizeCaiman(single_movie, opts, path)
    elif usecaiman==False:
        motionStabilize(image_stack, path)
    elif usecaiman!=bool:
        print('Provide a boolean for usecaiman argument. It can only take "True" or "False" values.')
    else:    
        print('Error. Make sure you provide the usecaiman an argument')

def preprocess_dataset(Tser,mot_corr, usecaiman = True):
    os.chdir(Tser)
    data_exists=os.path.isdir(Tser) != 0
    if data_exists and mot_corr:
        mot_corr_exists=os.path.isfile(os.path.split('.tif')[0]+'_motCorr.tif') != 0
        if mot_corr_exists:
            # dataset = sima.ImagingDataset.load('TserMC.sima')
            print('Motion corrected file exists. Did not overwrite. Check the path and analysis folder.')        
        else:
#            mc_approach= sima.motion.PlaneTranslation2D(max_displacement=[15,30])
            #  mc_approach = sima.motion.HiddenMarkov2D(granularity='plane', max_displacement=displacement)
            #  dataset=mc_approach.correct(sequence,'TserMC.sima')
            motionStabilize_tifsInFolder(Tser, usecaiman)
    elif mot_corr and data_exists==False:
        # dataset = sima.ImagingDataset(sequence, 'Tser.sima')
        # #mc_approach= sima.motion.PlaneTranslation2D(max_displacement=[15,30])
        # mc_approach = sima.motion.HiddenMarkov2D(granularity='plane', max_displacement=displacement)
        print('Tifs not found')
        # dataset=mc_approach.correct(sequence,'TserMC.sima')
    elif mot_corr==False and data_exists:
        # dataset = sima.ImagingDataset.load('Tser.sima')
        motionStabilize_tifsInFolder(Tser, usecaiman)
    elif mot_corr==False and data_exists==False:
        # dataset = sima.ImagingDataset(sequence, 'Tser.sima')
        print('The folder you provided is empty or the path is incorrect.')

def get_stim_xml_params(t_series_path, stimInputDir):
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
    imagetimes = imagetimes[49:-1] # Seb: when imaging with cycle00001 = 50 frames
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Stimulus output information
    
    stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile, 
                                          skipHeader=3) # Seb: skipHeader = 3 for _stimulus_ouput from 2pstim
    
    # Stimulus information
    (stimInputFile,stimInputData) = readStimInformation(stimType=stimType,
                                                      stimInputDir=stimInputDir)
    isRandom = int(stimInputData['randomize'][0])
    epochDur = stimInputData['duration']
    epochDur = [float(sec) for sec in epochDur]
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount <= 1:
        trialCoor = 0
        trialCount = 0
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
                             'depth' : depth}
        
    return stimulus_information, imaging_information

def find_clusters_STICA_JF_BG(cluster_dataset, area_min, area_max,cycle_01_lenght,components=45):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
        
    area_min : int
        Minimum area of a cluster in pixels
        
    area_max : int
        Maximum area of a cluster in pixels
        
    
    components: int 
        Number of principal components used to segment 
     
    Returns
    =======
    
    clusters : sima.ROI.ROIList
        A list of ROIs.
        
    all_clusters_image: numpy array
        A numpy array that contains the masks.
    """   
    try:
        inclusion_zone=np.array(dataset.ROIs['I_Zone'][0]) # juan edit. this is a nparray with a mask that corresponds to the neuropile
                                                                   # every ROI outside this zone will be eliminated
    except:
        inclusion_zone=np.array(dataset.ROIs['I_zone'][0])
    finally:
        pass
         #print('please create inclusion zone and run script once more %s'%(Tser))
    print('\n-->Segmentation running...')
    segmentation_approach = sima.segment.STICA(channel = 0,components=components,mu=0.1)
    segmentation_approach.append(sima.segment.SparseROIsFromMasks(
            min_size=area_min,smooth_size=3))
    inclusion_filter=sima.segment.ROIFilter(lambda roi: np.sum(np.squeeze(np.array(roi))*np.squeeze(inclusion_zone))>2)
    size_filter = sima.segment.ROIFilter(lambda roi: roi.size >= area_min and \
                                         roi.size <= area_max)
    # use subsample (1st cycle) of the initial dataset to do segmentation upon
    # try:
    #     dataset2=sima.ImagingDataset(cluster_dataset[0,0:cycle_01_lenght,:,:,:].sequences, 'ROIsDS.sima')
    # except WindowsError:
    #     dataset2=sima.ImagingDataset.load('ROIsDS.sima')
    # finally:
    #     pass
    dataset2=cluster_dataset

    segmentation_approach.append(size_filter)
    segmentation_approach.append(inclusion_filter)

    start1 = time.time()
    #clusters = dataset2.segment(segmentation_approach, 'auto_ROIs')
    clusters = dataset2.segment(segmentation_approach, 'auto_ROIs')
    initial_cluster_num = len(clusters)
    end1 = time.time()
    time_passed = end1-start1
    print('Clusters found in %d minutes\n' % \
          round(time_passed/60) )
    print('Number of initial clusters: %d\n' % initial_cluster_num)
    
    #cluster_dataset.add_ROIs(dataset2.ROIs['auto_ROIs'],label='auto_ROIs')

  
#%% functions for bleedthrough filtering 
# in case bleedtrough filtering is needed
def find_cycles(path):
    tif_path=path+'\\*Cycle*.tif'
    tif_list=glob.glob(tif_path)
    cycles=[]
    for tif in tif_list:
        cycle_num=tif.split('\\')[-1]
        cycle_num=cycle_num.split('_')[1]
        cycles.append(cycle_num)
    cycles=np.unique(np.array(cycles))
    number_cycles=len(cycles)
    return number_cycles

def filter_cycles(path,number_cycles):
    cycle_list={}
    for cycle in range(1,number_cycles+1):
        #find tifs for each cycle
        cycle_list[cycle]=glob.glob(path +'\\'+'*_Cycle*'+str(cycle)+'_ch*')
    return cycle_list  
    
def calculate_spatial_FFT(cycle_list):
    
    FFT_2d={}
    tif_array={}
    for cycle in cycle_list.keys():
        #find image size
        im=Image.open(cycle_list[cycle][0])
        size = np.array(im.size)
        tif_array[cycle]=np.zeros((size[1],size[0],len(cycle_list[cycle]))) 
        for n_tif,tif in enumerate(cycle_list[cycle]): 
            im=Image.open(cycle_list[cycle][n_tif])
            im=np.array(im)
            #im=im/np.max(im) #take this out when testing is done
            tif_array[cycle][:,:,n_tif]=im
        FFT_2d[cycle]=fftshift(fft2(tif_array[cycle],axes=(0,1)),axes=(0,1)) #fftshift shifts the dc component to the center of the image
    return FFT_2d,size #this should be 2d numpy arrays in the dict 

def filter_bleedthrough_gaussianstripe(data_cycle_list,FFT_2d,im_size,neuropile,pixel_size,treshold=1):
    height= im_size[1]
    width= im_size[0]
    height_um=height*pixel_size
    critical_wavelenght=height_um/2
    middle=[height//2,width//2]
    if neuropile=='medulla':
        #treshold=54 #hardcoded treshold for the filter (40um wavelength filter low pass)
        treshold=treshold #Before it was 3 #Mi9_glucldef 202010601_f2 added for specific recordings that are still noise with frist treshold
    if neuropile=='Lobula_plate':
        treshold=treshold #Before it was 3
    wavelenght_vector=np.linspace(height_um,height_um/middle[0],middle[0])
    diff=np.abs(wavelenght_vector-treshold)
    #idx_of_treshold=np.where(diff==np.min(diff))[0][0]+1
    idx_of_treshold=treshold
    print('filteridx %s'%(idx_of_treshold))
    rectangle_filter=np.ones((im_size[1],im_size[0],1))
    gaussian_filter=gaussian(im_size[0],1)
    gaussian_filter=(gaussian_filter*-1)+1
    rectangle_filter[:,:,0]=np.broadcast_to(gaussian_filter,(im_size[1],im_size[0]))
    center=gaussian(6,5)
    rectangle_filter[middle[0]-idx_of_treshold:middle[0]+idx_of_treshold,:]=1
    im=plt.imshow(np.squeeze(rectangle_filter))
    plt.savefig('filter.png')
    plt.close()
    # if neuropile=='medulla':
    #     rectangle_filter[middle[0]-10:middle[0]+10,:]=1
    # elif neuropile=='Lobula_plate':
    #     rectangle_filter[middle[0]-3:middle[0]+3,:]=1
    # else:
    #     raise Exception('not implemented for this neuropile')
    filtered_data_FFT={}
    for cycle in data_cycle_list.keys():
        FFT_for_cycle=FFT_2d[cycle]
        #final_filter=np.broadcast_to(rectangle_filter,(height,width,data_FFT_2d[cycle].shape[2]))
        FFT_for_cycle=FFT_for_cycle*rectangle_filter
        filtered_data_FFT[cycle]=FFT_for_cycle
    return filtered_data_FFT

def backtransform(filtered_data_FFT):
    #os.chdir(data_path)
    complete_array=[]
    for cycle in filtered_data_FFT.keys():
        iFFT_for_cycle=filtered_data_FFT[cycle]
        iFFT_for_cycle=np.abs(ifft2(ifftshift(iFFT_for_cycle,axes=(0,1)),axes=(0,1)))
        iFFT_for_cycle=iFFT_for_cycle.astype(np.int16)
        filtered_data_FFT[cycle]=iFFT_for_cycle
    for cycle in filtered_data_FFT.keys():
        if cycle==1:
            complete_array.append(filtered_data_FFT[cycle])
            complete_array=np.array(complete_array[0])
        else:
            complete_array= np.concatenate((complete_array,filtered_data_FFT[cycle]),axis=2)
    complete_array=complete_array[:,np.newaxis,:,:,np.newaxis]
    complete_array=np.transpose(complete_array,(3,1,0,2,4))
    return complete_array #filtered_data_FFT 
    
def motion_correction_with_array(array,Tser,fft_filter_treshold,keep_original=False,displacement=[30,30]):
    ch_corr=array    
    #sima.Sequence.join()
    os.chdir(Tser)
    exists=len(glob.glob('stack_FFT_filtered_Mcorr*'))>0
    if keep_original==True:
        preMC_dataset=sima.ImagingDataset.load('Tser.sima')
        sequence_base=[preMC_dataset.sequence]
        sequence=[sima.Sequence.create('ndarray', ch_corr)]
        #under construction
    if exists==False:
        sequence=[sima.Sequence.create('ndarray', ch_corr)]
        #load non_corrected dataset to estimate correction
        #preMC_dataset=sima.ImagingDataset.load('Tser.sima')        
        #mc_approach = sima.motion.HiddenMarkov2D(granularity='plane') #this maximum displacement was adjusted individually for some flies to make better alignment
        #estimation=mc_approach.estimate(preMC_dataset)
        #mc_approach = sima.motion.HiddenMarkov2D(granularity='plane', max_displacement=estimation+10)
        mc_approach = sima.motion.HiddenMarkov2D(granularity='plane', max_displacement=displacement)
        if len(glob.glob('TserMC_bandstop_filtered.sima'))==0:
            dataset=mc_approach.correct(sequence,'TserMC_bandstop_filtered.sima')
        else:
            dataset=sima.ImagingDataset.load('TserMC_bandstop_filtered.sima')
        dataset.export_frames([[['stack_FFT_filtered_Mcorr_%s_%s.tif' %(displacement,fft_filter_treshold)]]],fmt='TIFF16')

    else:
        print('done for Tser:%s'%(Tser))
        dataset = sima.ImagingDataset.load('TserMC_bandstop_filtered.sima')
    if export_average==True:
        dataset.export_averages(['Average_MC_bandstop_filtered.tif'],fmt='TIFF16',scale_values=scale_values)
    return dataset
