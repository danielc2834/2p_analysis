import os
from skimage import io
import glob
import pickle
import re
import numpy as np
import warnings
import seaborn as sns
from matplotlib import pyplot as plt
from roipoly import RoiPoly, MultiRoi
if python_version < 3.0:
    import tkFileDialog as filedialog
    import cPickle # For Python 2.X
elif python_version > 3.0:
    from tkinter import filedialog
    import _pickle as cPickle # For Python 3.X

# import PyROI
import ROI_mod #added by Juan
from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize, getMicRelativeTime
from post_analysis_core import run_matplotlib_params
import summary_figures as sf

def preProcessMovie(data_path):
    

    # Generate necessary directories for figures
    current_t_series=os.path.basename(data_path)
    
    # Load movie, get stimulus and imaging information
    try:
        movie_path = os.path.join(data_path, 'motCorr.sima',
                                  '{t_name}_motCorr.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
    except IOError:
        movie_path = os.path.join(data_path, '{t_name}_motCorr.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
    
    frames_imaged = time_series.shape[0]
    ## Get stimulus and xml information
    imaging_info = processXmlInfo(data_path)
    stim_info = processPyStimInfo(data_path,frames_imaged)
    
    
    return time_series, stim_info, imaging_info

def processXmlInfo(data_path):
    """
        Extracts the stimulus and imaging parameters. 
    """
    # Finding the xml file and retrieving relevant information
    xmlPath = os.path.join(data_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    imagetimes = getMicRelativeTime(xmlFile)
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)

    imaging_info = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth, 'frame_timings':imagetimes}
    
    return imaging_info
    
def processPyStimInfo(data_path, frames_imaged):

    # Stimulus information
    paths = os.path.join(data_path, '*.pickle')
    stim_file_path = (glob.glob(paths))[0]

    load_var = open(stim_file_path, 'rb')
    stim_info = pickle.load(load_var)

    stim_info['processed'] = {}

    # Epoch coordinates
    (stim_info['processed']['trial_coordinates'], \
        stim_info['processed']['epoch_trace_frames']) = \
        getEpochTrialCoords(stim_info,frames_imaged)

    return stim_info

def getEpochTrialCoords(stim_info, frames_imaged):
    """
        Finds the epoch changing coordinates.

        Returns
        =======
        epoch coordinates : dict
            Each key is the epoch name. 
            Trial coordinates are given as frames: [[baseStart, trialStart, trialEnd]]
            Note: Frames are mapped onto 0 index (1st frame is -> 0) due to Python indexing. Movie will be mapped like this too.


    """

    epoch_coordinates = {}
    epoch_trace = np.array(stim_info['stimulus_epoch'])
    frame_trace = np.array(stim_info['imaging_frame'])-1 # Map frames to 0 beginning index

    frame_changes = np.where(np.insert(np.diff(frame_trace),0,1))[0]

    epoch_trace_frames = np.ones([frames_imaged])
    frame_trace_frames = range(frames_imaged)

    # Find the epochs corresponding to the imaged frames
    # If a frame contains more than 1 epoch than assign it to the longest appearing epoch
    for iFrame in range(frames_imaged):
        frame_mask = frame_trace == iFrame
        epoch_trace_frames[iFrame] = np.bincount(epoch_trace[frame_mask].astype(int)).argmax()


    epoch_trace_frames = epoch_trace_frames[:frames_imaged]

    # In case of randomization is 1 or 3, then we will have coordinates with
    # [[baseStart, trialStart, trialEnd, baseEnd]]
    randomization = stim_info['meta']['randomization_condition']
    if (randomization == 1) or (randomization == 3):
        base_epoch_n = 1
        base_coords_beg = np.where(np.diff(np.array(epoch_trace_frames==base_epoch_n).astype(int))==1)[0] + 1
        base_coords_beg = np.insert(base_coords_beg,0,0) # Add the start

        base_coords_end = np.where(np.diff(np.array(epoch_trace_frames==base_epoch_n).astype(int))==-1)[0]
        base_reps = np.min([len(base_coords_beg),len(base_coords_end)])
        base_coords_beg = base_coords_beg[:base_reps]
        base_coords_end = base_coords_end[:base_reps]
        base_length = np.min(base_coords_end - base_coords_beg) + 1
    else:
        base_epoch_n = None
        base_length = 0
    
    for curr_epoch in stim_info['meta']['epoch_infos'].keys():
        curr_epoch_n = int(re.search(r'\d+', curr_epoch).group())
        epoch_coordinates[curr_epoch] = []
        # Don't take the baseline epoch
        if curr_epoch_n == base_epoch_n:
            for baseTrial in range(base_reps):
                epoch_coordinates[curr_epoch].append([frame_trace_frames[base_coords_beg[baseTrial]],
                     frame_trace_frames[base_coords_end[baseTrial]]])
            continue

        # Find the trial start and ends
        epoch_beg = np.where(np.diff(np.array(epoch_trace_frames==curr_epoch_n).astype(int))==1)[0] + 1

        # If this is the first presented epoch then add a 0 to the epoch_beg
        # This condition will only work when the first epoch is not a baseline epoch 
        # meaning that randomization is not 1 or 3
        if epoch_trace_frames[0] == curr_epoch_n:
            epoch_beg = np.insert(epoch_beg,0,0) # Add the start
        epoch_end = np.where(np.diff(np.array(epoch_trace_frames==curr_epoch_n).astype(int))==-1)[0]
        completed_trial_n = np.min([len(epoch_beg),len(epoch_end)]) # Don't use if the trial ended prematurely
        
        
        # Arrange it in a list
        for iTrial in range(completed_trial_n):  
            base_beg = frame_trace_frames[epoch_beg[iTrial]-base_length]
            
            trial_beg = frame_trace_frames[epoch_beg[iTrial]]
            trial_end = frame_trace_frames[epoch_end[iTrial]]      
            
            # Don't use the trial if it is not followed by a full baseline
            if epoch_end[iTrial]+base_length > len(frame_trace_frames):
                continue
            else:
                base_end = frame_trace_frames[epoch_end[iTrial]+base_length]

            epoch_coordinates[curr_epoch].append([base_beg, trial_beg,trial_end,base_end])
    
    return epoch_coordinates, epoch_trace_frames
            
def organizeExtractionParams(extraction_type,
                               current_t_series=None,current_exp_ID=None,
                               alignedDataDir=None,
                               stimInputDir=None,
                               use_other_series_roiExtraction = None,
                               use_avg_data_for_roi_extract = None,
                               roiExtraction_tseries=None,
                               transfer_data_n = None,
                               transfer_data_store_dir = None,
                               transfer_type = None,
                               imaging_info=None,
                               experiment_conditions=None):
    
    extraction_params = {}
    extraction_params['type'] = extraction_type
    if extraction_type == 'SIMA-STICA':
        if use_other_series_roiExtraction:
            series_used = roiExtraction_tseries
        else:
            series_used = current_t_series
        extraction_params['series_used'] = series_used
        extraction_params['series_path'] = \
            os.path.join(alignedDataDir, current_exp_ID, 
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
        extraction_params['imaging_information']= imaging_info
        extraction_params['experiment_conditions'] = experiment_conditions
        
        
    return extraction_params

def selectManualROIs(image_to_select_from, image_cmap ="gray",pause_t=7,
                   ask_name=True):
    """ Enables user to select rois from a given image using roipoly module.

    Parameters
    ==========
    image_to_select_from : numpy.ndarray
        An image to select ROIs from
    
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
        curr_agg = mask_agg.copy()
        curr_agg[curr_agg==0] = np.nan
        plt.imshow(curr_agg, alpha=0.3,cmap = 'Accent')
        plt.title("Select ROI: ROI%d" % roi_number)
        plt.show(block=False)
       
        
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        if ask_name:
            mask_name = raw_input("\nEnter the ROI name:\n>> ")
        else:
            mask_name = iROI
        curr_mask = curr_roi.get_mask(image_to_select_from)
        if len(np.where(curr_mask)[0]) ==0 :
            warnings.warn('ROI empty.. discarded.') 
            continue
        mask_names.append(mask_name)
        
        
        roi_masks.append(curr_mask)
        
        mask_agg[curr_mask] += 1
        
        
        
        roi_number += 1
        signal = raw_input("\nPress k for exiting program, otherwise press enter")
        if (signal == 'k'):
            stopsignal = 1
        
    
    return roi_masks, mask_names

def generateROIsImage(roi_masks,im_shape):
    # Generating an image with all clusters
    all_rois_image = np.zeros(shape=im_shape)
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    return all_rois_image
    
def transferROIs(transfer_data_path, transfer_type,experiment_info=None,
                     imaging_info=None,transfer_traces=False): #edited by Juan
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
                                          imaging_info =imaging_info,transfer_traces=transfer_traces) #modified by juan
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'stripes_OFF_delay_profile':
        
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS='OFF')
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
        
    elif transfer_type == 'stripes_ON_delay_profile':
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = PyROI.transfer_masks(rois, properties,
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
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'ternaryWN_elavation_RF'):
        properties = ['corr_fff', 'max_response','category','analysis_params',
                      'reliability','SNR','DSI','CSI','CS','PD']
        transferred_rois = ROI_mod.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,transfer_traces=transfer_traces)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'gratings_transfer_rois_save'):
        
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = PyROI.transfer_masks(rois, properties,
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
        transferred_rois = PyROI.transfer_masks(rois, properties,
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
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS=CS)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
   
    elif transfer_type == 'STF_1':
        properties = ['CSI', 'CS','PD','DSI','category','analysis_params']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'minimal' :
        print('Transfer type is minimal... Transferring just masks, categories and if present RF maps...\n')
        properties = ['category','analysis_params','RF_maps','RF_map',
                      'RF_center_coords','RF_map_norm']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    else:
        raise NameError('Invalid ROI transfer type')
        
        
   
    return transferred_rois

def selectROIs(extraction_params, image_to_select=None):
    """

    """
    # Categories can be used to classify ROIs depending on their location
    # Backgroud mask (named "bg") will be used for background subtraction
    plt.close('all')
    plt.style.use("default")
    print('\n\nSelect categories and background')
    [cat_masks, cat_names] = selectManualROIs(image_to_select, 
                                            image_cmap="gray",
                                            pause_t=8)
    
    # have to do different actions depending on the extraction type
    if extraction_params['type'] == 'manual':
        print('\n\nSelect ROIs')
        [roi_masks, roi_names] = selectManualROIs(image_to_select, 
                                                image_cmap="gray",
                                                pause_t=4.5,
                                                ask_name=False)
        all_rois_image = generateROIsImage(roi_masks,
                                                  np.shape(image_to_select))
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, None
            
    elif extraction_params['type'] == 'SIMA-STICA': 
        # Could be copied from process_mov_core -> run_ROI_selection()
        raise NameError("ROI extraction for SIMA-STICA is not yet impletemented.")
    
    elif extraction_params['type'] == 'transfer':
        
        rois = transferROIs(extraction_params['transfer_data_path'],
                                extraction_params['transfer_type'],
                                experiment_info=extraction_params['experiment_conditions'],
                                imaging_info=extraction_params['imaging_information'])
        
        return cat_masks, cat_names, None, None, rois, None
    
    else:
       raise TypeError('ROI selection type not understood.') 

def analyzeTraces(rois, analysis_params,save_fig=True,fig_save_dir=None,summary_save_d=None):
    """ Each different stimulus type has its own analysis way. 
        This function will implement them accordingly.
    """
    plt.style.use('default')
    analysis_type = analysis_params['analysis_type']

    figtitle = 'Summary: %s Gen: %s | Age: %s' % \
           (rois[0].experiment_info['MovieID'].split('-')[0],
            rois[0].experiment_info['Genotype'], rois[0].experiment_info['Age'])

    if analysis_type == 'luminance_gratings':
        
        if ('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype']) and \
            (not('1D' in rois[0].stim_name)):
            map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
            
        rois = PyROI.analyzeSineGratings(rois)
        luminances = rois[0].epoch_luminances.values()
        powers = np.array(map(lambda roi: roi.power_at_sineFreq.values(),rois)).T
        plt.scatter(np.tile(luminances,(powers.shape[1],1)).T,powers,color='k',alpha=0.7)
        plt.plot(np.sort(luminances),powers.mean(axis=1)[np.argsort(luminances)],
            '--k',linewidth=3)

        plt.xlabel('Luminance')
        plt.ylabel('Signal strength')
        fig = plt.gcf()
        f0_n = 'Summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False)

    elif analysis_type == '5sFFF_analyze_save':

        rois = PyROI.fffAnalyze(rois, interpolation = True, int_rate = 10)
        roi_conc_traces = list(map(lambda roi: roi.conc_trace, rois))
        stim_trace  = rois[0].stim_trace
        fig = sf.fffSummary(figtitle,stim_trace, roi_conc_traces,
                    True,rois[0].experiment_info['MovieID'],summary_save_d)
        f1_n = '5sFFF_summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
                       
    elif analysis_type == 'A_B_time_adaptation':
        # Fixed A and B contrast but A step times changing
        #TODO: Fix the plots make them nicer etc.
        rois = PyROI.analyzeAB_steps_time(rois)
        b_steps = []
        a_steps = []
        for roi in rois:
            b_steps.append(roi.sorted_b_steps)
            plt.scatter(roi.sorted_a_durs,roi.sorted_b_steps)
            plt.scatter(32,np.mean(roi.a_step_response.values()))
            a_steps.append(np.mean(roi.a_step_response.values()))
        mean_r = np.mean(b_steps,axis=0)
        plt.plot(roi.sorted_a_durs,mean_r,color='k')
        std_r = np.std(b_steps,axis=0)
        ub = mean_r + std_r
        lb = mean_r - std_r
        plt.fill_between(roi.sorted_a_durs, ub, lb, color='k',alpha=.3)

        plt.xlabel('A step time')
        plt.ylabel('B step response')

        fig = plt.gcf()
        f1_n = 'AB_time_summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)

    elif ((analysis_type == 'luminance_edges_OFF' ) or\
          (analysis_type == 'luminance_edges_ON' )) :
        
        if ('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype']):
            map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
        rois = PyROI.analyzeLuminanceEdges(rois,int_rate = 10)
        roi_image = PyROI.generatePropertyMasks(rois, 'slope')
        fig = sf.summarizeLuminanceEdges(figtitle,rois,roi_image,
                                            rois[0].experiment_info['MovieID'],
                                            summary_save_d)
        
        
        slope_data = PyROI.dataToList(rois, ['slope'])['slope']
        rangecolor= np.max(np.abs([np.min(slope_data),np.max(slope_data)]))
        
        if 'RF_map' in rois[0].__dict__:
            fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='slope',
                                                     cmap='PRGn',
                                                     ylab='Lum sensitivity',
                                                     lims=(-rangecolor,
                                                           rangecolor))
            f2_n = 'Slope_on_screen_%s' % (rois[0].experiment_info['MovieID'])
            os.chdir(fig_save_dir)
            fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
                           transparent=False,dpi=300)
        else:
            print('No RF found for the ROI.')
        
        f1_n = 'Summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)

    return rois

def plotAllMasks(roi_image, underlying_image,n_roi1,exp_ID,
                       save_fig = False, save_dir = None,alpha=0.5):
    """ 

    """

    plt.close('all')
    plt.style.use("dark_background")

    # All masks
    plt.imshow(underlying_image,cmap='gray')
    plt.imshow(roi_image,alpha=alpha,cmap = 'tab20b')
    
    ax1=plt.gca()
    ax1.axis('off')
    ax1.set_title('ROIs n=%d' % n_roi1)
    
    if save_fig:
        # Saving figure
        save_name = 'ROIs_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('ROI images saved')
    return None


