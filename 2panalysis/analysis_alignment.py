'''script to align data in /3_DATA'''
################################################
import os, pickle, copy, scipy#, preprocessing_params
# from Helpers import ROI_mod
import numpy as np
import core_preprocessing as core_pre 
################################################
'''each roi in _ROIS.pkl file[tseries]['final_rois']
    .experiment_info > dict with keys: Genotype (str), Age (int), Sex (str), treatment (str), FlyID (str), expected_polarity )str), MovieID (str), z_depth (str), z_depth_bin (str), analysis_type [str], rotation (int)
    'imaging_info' > dict with keys: FramePeriod (int), micRelTimes (arr), rastersPerFrame (int), frame_rate (int), x_size (int), y_size (int), pixelArea (int), layerPosition [x,y,z]}
    'mask' > ROI mask (2d array of boolean for dimension of Tseries)
    'uniq_id' > int, unique ID for each ROI
    'category' > [str], category defined when selecting rois, if not > ['No_category']
    'source_image' > source image (2d array)
    'extraction_params' > how rois where selected , for me : 'manual'
    'analysis_params' > dict with keys: deltaF_method [str], analysis_type [str], Background (str), independent_var (str)
    'bg_mask' > mask of background roi used for substraction (2d array in dimension of image)
    'stim_info' > dict of information on stimulus used > keys: 'EPOCHS' (int), 'MAXRUNTIME' (int), 'STIMULUSDATA' (str), 'PERSPECTIVE_CORRECTION' (int), 'RANDOMIZATION_MODE' (int), 'stimtype' [float, len epochs], 'bg' [float, len epochs], 'fg' [float, len epochs], 'duration' [float, len epochs], 'velocity' [float, len epochs], 'direction' [float, len epochs], 'bar.height' [float, len epochs], 'tau' [float, len epochs], 'pers.corr' [float, len epochs], 'angle' [float, len epochs], 'subepoch' [float, len epochs], 'epoch_dur' [float, len epochs], 'random' (int), 'output_data' (touple len frames with len epoch entris as nparry), 'frame_timings' [array] in sec, 'input_data' : everything from stim.txt file in dict, 'stim_name' (str), 'trial_coordinates' dict with trial number as kay and 2d array as value, 'baseline_epoch' : None, 'baseline_duration' : None, 'epoch_adjuster' (int), 'output_data_downsampled': dataframe({'tcurr (in sec)', 'boutInd', 'epoch', 'xpos', 'ypos', 'theta', 'data', 'interlude'})
    'stim_name' >> name of stim file use 
    'original_trace' > 1D array of raw traces over frames
    'BG_trace' > 1D array of BG trace over frames
    'raw_trace' > 1D array original trace - Background
    'baseline_method' > str: defines method to calulate baseline method : most of the time : rolling average data<(2*std)+mean window 20secs'
    'df_trace' : DataFrame 1 column, len recording rows. DF/F with baseline
    UNTIL HERE ODOR ROI
    IF VISUAL:
    'base_dur' : []
    'whole_trace_all_epochs' > dict with epoch number as key and cut corresponding part of df_trace
    'resp_trace_all_epochs' > dict with epoch number as key and cut corresponding part of df_trace (same as above)
    'whole_traces_all_epochsTrials' >dict with epoch number as key and value: tuple or longer array with value for each trial within epoch 
    'reliability_all_epochs' > dict with epoch number as key and reliability as value (between 0 and 1)
    'reliability_all_epochs_ON' > dict with epoch number as key and reliability as value (between 0 and 1) for ON part of epoch
    'reliability_all_epochs_OFF' > dict with epoch number as key and reliability as value (between 0 and 1) for OFF part of epoch
    UNTIL HERER ALL VISUAL STIM SAME
        for gratings
            'max_resp_all_epochs' : array 2d, with len epochs and value of max during epoch
            'max_response' ofloat: overall max response
            'max_resp_idx' int index of overall max response
            'tfl_map'  >  temporal frequency dataframe
            'BF' 
            'oneHz_conc_resp' > trace of 1 Hz
            NOT REST
        for FFF 
            'stim_trace_correlation' > float: corr vlaue of roi with stim 
            'max_resp_all_epochs' : > max response for each epoch (here ON flahs, OFF flash)
            'max_response' > overall max response
            'max_resp_idx' >  int index of overall max response
            'conc_trace' > concatenated (avg) trace of one epoch 
            'stim_trace' > trace of one epoch
            'corr_fff'  > some correlation __ 
            'corr_pval' > pvalue of correlation abovce
            'int_con_trace' > 2darray interpolated trace for epoch, 1: time, 2nd trace 
            'int_stim_trace' > 2darray interpolated stimulus for epoch, , 1: time, 2nd stimt race  
            'int_rate' > fps to interpolate to
        for 8D
        'max_resp_all_epochs_ON' > array len of epochs with max of each epoch ON part
        'max_resp_all_epochs_OFF' > array len of epochs with max of each epoch OFF part
        'trace_STD_OFF' > array len of epochs with std of each epoch OFF part
        'trace_STD_ON' > array len of epochs with std of each epoch ON part
        'baseline_STD_OFF' > nan
        'baseline_STD_ON' > nan
        'baseline_mean_ON' > nan
        'mean_of_overall_baseline' > nan
        'baseline_mean_OFF' > nan
        'overall_baseline' > {}
        'shifted_trace_' > {}
        'overall_baseline_loc' > array 
        'max_response_ON' > float: max ON response
        'max_response_OFF' > float: max OFF response
        'max_resp_idx_ON' > int : inedx of max ON response
        'max_resp_idx_OFF' > int : inedx of max OFF response
        'reliability_PD_ON' > float: reliability of Preferred direction fo ON part
        'reliability_PD_OFF' > float: reliability of Preferred direction fo OFF part
        'reliability_PD' > float: overall reliability
        'direction_vector' > array (len of direction in 8D stim) witch angles
        'DSI_ON' > float: direction selectivity of ROI 
        'DSI_OFF' > float: direction selectivity of ROI 
        'PD_ON' >  float: angle of ON  prefrerred direction
        'PD_OFF' >  float: angle of OFF  prefrerred direction
        'dir_max_resp' > float: angle of stim triggering maximum response
        'CSI' : float: contrast sensitivity index > used to correlate to stim
        'CS' : str > whether response to ON or OFF
        'delayed_RF_traces' > dict with epoch number as kay, value: multiD array, > for each trial array len trial
        'non_delayed_RF_traces' > dict with epoch number as kay, value: multiD array, > for each trial array len trial
        'delay_used_in_RF' > float
        'RF_maps'
        'RF_map'
        'RF_map_norm'
        'RF_center_coords'
        'Center_position_filter'
    '''
#align all to same fps
# dataset_folder = 'C:/Master_Project/2P/datasets/241203_whole_OL'
# dataset_folder = r'F:\Master\241203_whole_OL'
dataset_folder = r'C:\phd\02_twophoton\251023_tdc2_cschr_pan'

# dataset_folder = r'F:\Master\250219_moving_bar'
paths = core_pre.dataset(dataset_folder)
# if len(preprocessing_params.experiment)>0:
#     targt = f'{paths.data}/{preprocessing_params.experiment}'
# else:
#     target = f'{paths.data}/'
all_fps, all_fps_LH = [], []
for condition in os.listdir(paths.data):
    if condition.endswith(".pkl"):
        with open(f'{paths.data}/{condition}', 'rb') as fi:
            data = pickle.load(fi) 
        if condition == 'LH_all.pkl' or condition=='OL_all.pkl':
            continue             
        if condition.endswith("_LH.pkl"):
            for tseries in data:
                for roi in data[tseries]['final_rois']:
                    all_fps_LH.append(roi.imaging_info.get('frame_rate'))
                # if tseries == 'alligned_fps':
                #     continue
                # for category in data[tseries]:
                #     if category == 'pulse' or category == 'on':
                #         continue
                #     all_fps_LH.append(data[tseries][category]['frame_rate'])
        else:
            for tseries in data:
                # if tseries == 'alligned_fps':
                #         continue
                for roi in data[tseries]['final_rois']:
                    all_fps.append(roi.imaging_info.get('frame_rate'))
                # for category in data[tseries]['final_rois']:
                #     if category == 'pulse' or category == 'on':
                #         continue
                #     all_fps.append(data[tseries][category]['frame_rate'])
# mean_fps_LH = int(sum(all_fps_LH) / len(all_fps_LH))
mean_fps = int(sum(all_fps) / len(all_fps))
#interpolate to mean fps
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

for condition in os.listdir(paths.data):
    if condition.endswith('.pkl'):
        if condition == 'LH_all.pkl' or condition=='OL_all.pkl':
            continue 
        if condition.endswith("_LH.pkl"):
            with open(f'{paths.data}/{condition}', 'rb') as fi:
                data = pickle.load(fi)
            # data['alligned_fps'] = mean_fps_LH
            # for tseries in data:
            #     # if tseries == 'alligned_fps':
            #     #         continue
            #     for roi in data[tseries]['final_rois']:
            #         roi.alligned_fps = mean_fps_LH
            #         trace = roi.df_trace
            #         fps_curr = roi.imaging_info.get('frame_rate')
            #         stim_duration = len(trace)/fps_curr
            #         time_curr = np.linspace(0, stim_duration, len(trace))
            #         time_ref = np.linspace(0, stim_duration,  int(mean_fps_LH*stim_duration))
            #         new_df = np.interp(time_ref, time_curr, trace)
            #         roi.interpolated_traces_all = new_df
            #         roi.interpolated_time_all = time_ref
            # with open(f'{paths.data}/{condition}', 'wb') as fo:
            #     pickle.dump(data, fo) 
                # if tseries == 'alligned_fps':
                #         continue
                # for category in data[tseries]:
                #     #data[tseries]['final_rois'][0].whole_traces_all_epochsTrials
                #     if category == 'pulse' or category == 'on':
                #         trace = data[tseries][category]
                #         time_curr = np.linspace(rel_time[0],rel_time[-1],len(trace))
                #         time_ref = np.linspace(rel_time[0],rel_time[-1], int(mean_fps_LH*rel_time[-1]))
                #         new_o  = np.interp(time_ref, time_curr,  data[tseries][category])
                #         old_odor = data[tseries][category]
                #         data[tseries][category] = {f'alligned_{category}' : new_o, f'{category}':old_odor}
                #         continue
                    # rel_time = data[tseries][category]['frame_timings']
                    # time_ref = np.linspace(rel_time[0],rel_time[-1], int(mean_fps_LH*rel_time[-1]))
                    # trace = data[tseries][category]['dff_mean']
                    # if 'dff_mean_alligned_fps' not in data[tseries][category].keys():
                    #     if data[tseries][category]['frame_rate']<mean_fps_LH:
                    #         time_curr = np.linspace(rel_time[0],rel_time[-1],len(trace))
                    #         new_df = np.interp(time_ref, time_curr, trace)
                    #     else:
                    #         time_curr = np.linspace(rel_time[0],rel_time[-1],len(trace))
                    #         new_df = np.interp(time_ref, time_curr, trace)
                    #     data[tseries][category]['dff_mean_alligned_fps'] = new_df
                    #     data[tseries][category]['time_alligned_fps'] = time_ref
            # with open(f'{paths.data}/{condition}', 'wb') as fo:
            #     pickle.dump(data, fo)
        else:
            with open(f'{paths.data}/{condition}', 'rb') as fi:
                data = pickle.load(fi)
            # data['alligned_fps'] = mean_fps
            for tseries in data:
                # if tseries == 'alligned_fps':
                #         continue
                for roi in data[tseries]['final_rois']:
                    if 'nan_LH' in condition or 'nan_OL' in condition:
                        roi.alligned_fps = mean_fps
                        trace = roi.df_trace
                        fps_curr = roi.imaging_info.get('frame_rate')
                        stim_duration = len(trace)/fps_curr
                        time_curr = np.linspace(0, stim_duration, len(trace))
                        time_ref = np.linspace(0, stim_duration,  int(mean_fps*stim_duration))
                        new_df = np.interp(time_ref, time_curr, trace)
                        roi.interpolated_traces_all = new_df
                        roi.interpolated_time_all = time_ref
                    else:
                        roi.alligned_fps = mean_fps
                        trace = roi.df_trace
                        fps_curr = roi.imaging_info.get('frame_rate')
                        stim_duration = len(trace)/fps_curr
                        time_curr = np.linspace(0, stim_duration, len(trace))
                        time_ref = np.linspace(0, stim_duration,  int(mean_fps*stim_duration))
                        new_df = np.interp(time_ref, time_curr, trace)
                        roi.interpolated_traces_all = new_df
                        roi.interpolated_time_all = time_ref
                        roi.interpolated_traces_epochs, roi.interpolated_time = {}, {}
                        for epoch in roi.whole_traces_all_epochsTrials:
                            trace = roi.resp_trace_all_epochs[epoch]
                            stim_duration=roi.stim_info['duration'][epoch] 
                            fps_curr = roi.imaging_info.get('frame_rate')
                            time_curr = np.linspace(0, stim_duration, len(trace))
                            time_ref = np.linspace(0, stim_duration,  int(mean_fps*stim_duration))
                            new_df = np.interp(time_ref, time_curr, trace)
                            roi.interpolated_traces_epochs[epoch] = new_df
                            roi.interpolated_time[epoch] = time_ref
                if 'Grating' in condition:
                    align_traces(data[tseries]['final_rois'],grating=True)
                elif 'FFF' in condition:
                    continue
                elif 'nan' not in condition:
                    align_traces(data[tseries]['final_rois'],grating=False)
                elif condition.startswith('layer_nan_'):
                    align_traces(data[tseries]['final_rois'],grating=False)
            with open(f'{paths.data}/{condition}', 'wb') as fo:
                pickle.dump(data, fo) 
                # for category in data[tseries]:
                #     if category == 'pulse' or category == 'on':
                #         trace = data[tseries][category]
                #         time_curr = np.linspace(rel_time[0],rel_time[-1],len(trace))
                #         time_ref = np.linspace(rel_time[0],rel_time[-1], int(mean_fps*rel_time[-1]))
                #         new_o  = np.interp(time_ref, time_curr,  data[tseries][category])
                #         old_odor = data[tseries][category]
                #         data[tseries][category] = {f'alligned_{category}' : new_o, f'{category}':old_odor}
                #         continue
                #     rel_time = data[tseries][category]['frame_timings']
                #     time_ref = np.linspace(rel_time[0],rel_time[-1], int(mean_fps*rel_time[-1]))
                #     trace = data[tseries][category]['dff_mean']
                #     if 'dff_mean_alligned_fps' not in data[tseries][category].keys():
                #         if data[tseries][category]['frame_rate']<mean_fps:
                #             time_curr = np.linspace(rel_time[0],rel_time[-1],len(trace))
                #             new_df = np.interp(time_ref, time_curr, trace)
                #         else:
                #             time_curr = np.linspace(rel_time[0],rel_time[-1],len(trace))
                #             new_df = np.interp(time_ref, time_curr, trace)
                #         data[tseries][category]['dff_mean_alligned_fps'] = new_df
                #         data[tseries][category]['time_alligned_fps'] = time_ref
            # with open(f'{paths.data}/{condition}', 'wb') as fo:
            #     pickle.dump(data, fo)
                
                
                
'''
NOW ADDED
8D: roi.interpolated_time[epoch] > interpolated whole epoch trace timings
    roi.interpolated_traces_epochs[epoch] >  interpolated whole epoch trace 
FFF : skipp, already interpolated
Gratings: 
'''












'''current rois: [dictionary]
for each tseries
    .alligned_fps
    if olfactory stim was used:
    olf_stim : np.array
    
    for each category:
        for each roi:
            .alligned_olf_stim : nparrays
            .raw_trace : raw imaging traces
            .dff_mean : dff calculated with total recording mean as baseline
            .source_image : mean image where roi was selected on
            .category : name of category, none if its ROI mask
            .imaging_info : info on recording (keys: frame_rate, pixel_size, depth, frame_timings)
            .uniq_id : unique id
            .mask : ROI mask 
            .number_id : number of ROI mask, NONE if category
            .baseline_method : str, method of calculating baseline > currently only "mean" 
            .time_alligned_fps : timing aligned to mean fps of whole dataset
            .dff_mean_alligned_fps : dff_mean aligned to mean fps of whole dataset
            .f'alligned_{olf stim (pulse or on)}' : alligned olfactory protocoll
        '''