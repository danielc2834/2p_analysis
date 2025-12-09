'''Script to batch select rois of calcium imaging recordings'''
import sys, pickle, os, preprocessing_params, copy, yaml, math
import core_preprocessing as core_pre
import core_analysis as core_a
import numpy as np
import pandas as pd
from skimage import io
from Helpers import ROI_mod
import Helpers.process_mov_core as pmc
import Helpers.core_functions as core_functions
import matplotlib.pyplot as plt
################################
my_input = sys.argv
dataset_folder, error_log, metadata_path = my_input[1], my_input[2], my_input[3]
paths = core_pre.dataset(dataset_folder)
df_meta = pd.read_excel(metadata_path, sheet_name=paths.name) 
with open(f'{paths.processed}/processing_progress.pkl', 'rb') as fi:
    processing_progress = pickle.load(fi)
name = os.path.basename(__file__)
open(error_log, 'a', encoding="utf8").write(f'\nscript used: {name}\n')
################################
# config_path = f'C:/Master_Project/Code\master_2p/2panalysis\config.yaml'
# config_path = 'F:/Studium/M.Sc/aktuell/Master Projectwork/master_2p/2panalysis/config.yaml'
config_path = '2panalysis/config.yaml'
# config_path = r"C:\Master_Project\Code\master_2p\2panalysis\config.yaml"
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)
filter_next_cycles = config.get('filter_next_cycles')
analize_only_first_cycle=config.get('analize_only_first_cycle')
save_data=config.get('save_data')

extraction_type = preprocessing_params.roi_extraction_type
if extraction_type=='cluster analysis Henning_20' or extraction_type == 'cluster analysis MH_JF':
    ''' if you use the clustering algorithm a clustering mode needs to be defined. 
    options: 'timing', 'contrast_dif', 'timing&contrast'  '''
    
    extraction_params = {'type':extraction_type,
                        'clustering_type':'timing&contrast',  # 'timing&contrast' or 'contrast_dif' or 'timing are the options
                        'roiExtraction_tseries':'1st cycle',
                        'Background':'otsus'} # on cluster size here won't be reflected in the analysis
    first_cycle_ROIs=True
else:
    extraction_params = {'type':extraction_type}
    first_cycle_ROIs=False
#apply_threshold_everyStep=True
#exclude_first_cycle_filter=True # when this is true, the first cycle is only used for

if extraction_type != 'load_manual':
    load_categories=True #False when the categories are already asigned (for example in manually selected rois)
else:
    load_categories=False 

for condition in os.listdir(paths.processed):
    if condition.endswith('_h') or condition.endswith('_v'):
        continue
    if condition.endswith(".pkl"):
            continue
    print(condition)
    for fly in os.listdir(f'{paths.processed}/{condition}'):
        if fly not in processing_progress[condition].keys():
            open(error_log, 'a', encoding="utf8").write(f'{fly}: no data for FLY found >> run motion correction first')
            open(error_log, 'a', encoding="utf8").write('\n')
            continue
        if preprocessing_params.same_rois == True:                       #
            if len(preprocessing_params.experiment)>0:
                skipping_target = f'{paths.processed}/{preprocessing_params.experiment}'
            else:
                skipping_target = f'{paths.processed}/'
            try:
                with open(f'{skipping_target}_skipping_masks.pkl', 'rb') as fi:
                    skipping_masks = pickle.load(fi)
            except:
                skipping_masks={}
            if all(col in list(df_meta.columns) for col in preprocessing_params.same_rois_columns) == False:
                open(error_log, 'a', encoding="utf8").write(f'{fly}: please define metadata-column for saving rois across TSeries')
                open(error_log, 'a', encoding="utf8").write('\n')
                continue
        meta_fly = df_meta[df_meta["fly"]==fly]
        counter=0
        for tseries in os.listdir(f'{paths.processed}/{condition}/{fly}'):
            if tseries.startswith("TSeries"):
                
                current_number=tseries.split('-')[-1]
                current_number = current_number[-1] if current_number [-2] == "0" else current_number[-2:]
                # meta_tseries = meta_fly[meta_fly.TSeries==f'Tseries-{current_number}']
                meta_tseries = meta_fly[meta_fly['TSeries']==int(current_number)]
                motion_target = f'{paths.processed}/{condition}/{fly}/{tseries}'
                if len(preprocessing_params.experiment)>0:
                    target = f'{paths.processed}/{condition}/{fly}/{tseries}/{preprocessing_params.experiment}'
                else:
                    target = f'{paths.processed}/{condition}/{fly}/{tseries}/'
                if tseries not in processing_progress[condition][fly].keys():
                    open(error_log, 'a', encoding="utf8").write(f'{fly}: no data for TSERIES found >> run motion correction first')
                    open(error_log, 'a', encoding="utf8").write('\n')
                    continue
                if fly not in skipping_masks.keys():
                    if os.path.exists(f'{target}_ROIS_skipp.pkl')==True:
                        same_rois_meta=[]
                        for col in preprocessing_params.same_rois_columns:
                            same_rois_meta.append(meta_tseries[col].tolist()[0])
                        with open(f'{target}_ROIS_skipp.pkl', 'rb') as fi:
                            skip_single_pkl = pickle.load(fi)
                        skipping_masks[fly]={}
                        skipping_masks[fly][str(same_rois_meta)] = skip_single_pkl
                        with open(f'{skipping_target}_skipping_masks.pkl', 'wb') as fo:
                            pickle.dump(skipping_masks, fo)
                    else:
                        skipping_masks[fly]={}
                if processing_progress[condition][fly].get(tseries)[2]==True and os.path.exists(f'{target}_ROIS.pkl')==True:
                    print(f'{tseries}: ROIS already selected')
                    continue
                if os.path.exists(f'{target}_ROIS.pkl')==True and processing_progress[condition][fly].get(tseries)[2]==False:
                    print(f'{tseries}: ROIS already selected')
                    processing_progress[condition][fly][tseries] = [True,True,True,False]
                    continue
                if os.path.exists(f'{target}_ROIS.pkl')==False and processing_progress[condition][fly].get(tseries)[2]==True:
                    processing_progress[condition][fly][tseries] = [True,True,False,False]
                visual_stim = meta_tseries['visual_stim_file'].tolist()[0]
                plot_individual_traces=True
                plot_roi_summ = True
                expected_polarity = "ON"
                summary_save_dir = f'{paths.processed}/{condition}/{fly}/{tseries}/'
                os.makedirs(summary_save_dir, exist_ok=True)
                figure_save_dir = f'{paths.processed}/{condition}/{fly}/{tseries}/'
                os.makedirs(figure_save_dir, exist_ok=True)
                cycle = 0
                
                if visual_stim == '155s 1_Dirfting_edges_JF_8dirs_ON_first':
                    analysis_type = ['8D_edges_find_rois_save']
                    dFmethod = ['rolling_mean']
                elif visual_stim == '2.4_Stripe_1sec_1secBG_5deg_ver_random_LumInc':
                    analysis_type = ["stripes_ON_vertRF_transfer"]#['gratings_transfer_rois_save'] #moving_gratings, luminance_gratings
                    dFmethod = ['rolling_mean']
                elif visual_stim == '2.2_Stripe_1sec_1secBG_5deg_hor_random_LumInc':
                    analysis_type = ["stripes_ON_horRF_transfer"]#['gratings_transfer_rois_save'] #moving_gratings, luminance_gratings
                    dFmethod = ['rolling_mean']
                elif visual_stim == '2.3_Stripe_1sec_1secBG_5deg_ver_random_LumDec':
                    analysis_type = ["stripes_OFF_vertRF_transfer"]#['gratings_transfer_rois_save'] #moving_gratings, luminance_gratings
                    dFmethod = ['rolling_mean']
                elif visual_stim == '2.1_Stripe_1sec_1secBG_5deg_hor_random_LumDec':
                    analysis_type = ["stripes_OFF_horRF_transfer"]#['gratings_transfer_rois_save'] #moving_gratings, luminance_gratings
                    dFmethod = ['rolling_mean']
                elif visual_stim == '2.4_Stripe_1sec_1secBG_5deg_ver_random_LumInc' or visual_stim == '2.3_Stripe_1sec_1secBG_5deg_ver_random_LumDec'\
                    or visual_stim == '2.2_Stripe_1sec_1secBG_5deg_hor_random_LumInc' or visual_stim == '2.1_Stripe_1sec_1secBG_5deg_hor_random_LumDec'\
                    or visual_stim == '3.1_StimulusData_Discrete_1_16_100000_El_50ms _H' or visual_stim == '3.2_StimulusData_Discrete_1_16_100000_El_50ms_V'\
                    :
                    continue
                elif 'Chirp' in visual_stim:
                    analysis_type = ['olfaction']#['gratings_transfer_rois_save'] #moving_gratings, luminance_gratings
                    dFmethod = ['rolling_mean']

                elif visual_stim == '155s 3_Gratings_sine_30degwl_1hz_dir90_270_0_180deg':
                    analysis_type = ["luminance_gratings"]#['gratings_transfer_rois_save'] #moving_gratings, luminance_gratings
                    dFmethod = ['rolling_mean']
                elif visual_stim == '155s 2_LocalCircle_5secON_5sec_OFF_120deg_10sec':
                    analysis_type = ['5sFFF_analyze_save']
                    dFmethod = ['rolling_mean']
                elif visual_stim == '15deg_50_Dirfting_edges_JF_8dirs_ON_first' or visual_stim =='15deg_Dirfting_edges_JF_8dirs_ON_first' or visual_stim =='30deg_50_Dirfting_edges_JF_8dirs_ON_first' or visual_stim == '30deg_Dirfting_edges_JF_8dirs_ON_first':
                    analysis_type = ['8D_edges_find_rois_save']
                    dFmethod = ['rolling_mean']
                elif math.isnan(visual_stim):
                    analysis_type = ['olfaction']
                    dFmethod = ['mean_1s_before']
                # baseline_epoch, mean, convolution, rolling_mean
                analysis_params = {'deltaF_method': dFmethod,   #TODO maybe make this dependent on the stimulus type
                                'analysis_type': analysis_type,
                                'Background':'otsus', 
                                "independent_var" :'frequency'}  #angle
                experiment_conditions= \
                        {'Genotype' : meta_tseries['genotype'].tolist()[0], 'Age': meta_tseries['age'].tolist()[0], 'Sex' : meta_tseries['sex'].tolist()[0], 'treatment':condition,
                        'FlyID' : meta_tseries['fly'].tolist()[0],'expected_polarity':expected_polarity, 'MovieID':  tseries,
                        'z_depth':meta_tseries['z'].tolist()[0],'z_depth_bin':meta_tseries['bin'].tolist()[0],'analysis_type': analysis_type,
                            'rotation':meta_tseries['rotation'].tolist()[0]}
                print(f'{tseries}: selecting ROIs')
                while processing_progress[condition][fly].get(tseries)[2]==False:
                    mean_image = io.imread(f'{motion_target}/_motavg.tif')
                    imaging_info = core_pre.processXmlInfo(f'{paths.raw}/{condition}/{fly}/{tseries}')
                    number_of_cycles,lenghtCycles=core_functions.find_cycles(f'{paths.raw}/{condition}/{fly}/{tseries}')
                    skip=False
                    same_rois_meta=[]
                    for col in preprocessing_params.same_rois_columns:
                            same_rois_meta.append(meta_tseries[col].tolist()[0])
                    if preprocessing_params.same_rois == True:
                        # current_meta = meta_fly[meta_fly.TSeries==f'Tseries-{current_number}'][preprocessing_params.same_rois_columns]
                        # previous_meta = meta_fly[meta_fly.TSeries==f'Tseries-{previous_number}'][preprocessing_params.same_rois_columns]
                        if str(same_rois_meta) in skipping_masks[fly].keys():
                        # if (current_meta.reset_index(drop=True) == previous_meta.reset_index(drop=True)).all().all():
                            skip=True
                            if abs(imaging_info.get("layerPosition")[2] - skipping_masks[fly][str(same_rois_meta)].get('depth')) >= preprocessing_params.z_range:
                                open(error_log, 'a', encoding="utf8").write(f'{fly}: z depth was too different for {tseries}')
                                open(error_log, 'a', encoding="utf8").write('\n')
                                skip = False
                            if mean_image.shape != skipping_masks[fly][str(same_rois_meta)].get('img_shape'):
                                open(error_log, 'a', encoding="utf8").write(f'{fly}: image dimensions of {tseries} too different to use same mask')
                                open(error_log, 'a', encoding="utf8").write('\n')
                                skip = False
                            
                    if skip == True:
                        # rois = skipping_masks[fly][str(same_rois_meta)].get('rois')
                        bg_mask = skipping_masks[fly][str(same_rois_meta)].get('bg_mask')
                        cat_masks = skipping_masks[fly][str(same_rois_meta)].get('cat_masks')
                        roi_masks = skipping_masks[fly][str(same_rois_meta)].get('roi_masks')
                        cat_names = skipping_masks[fly][str(same_rois_meta)].get('cat_names')
                        roi_names = skipping_masks[fly][str(same_rois_meta)].get('roi_names')
                    time_series = io.imread(f'{motion_target}/_motCorr.tif')
                    if skip == False:
                        xcorrimg = core_pre.CrossCorrImage(time_series.T,block_size = 5, w = 1)
                        cat_masks, cat_names, roi_masks, roi_names = core_pre.selectROIs(preprocessing_params.roi_extraction_type,error_log,mean_image,xcorrimg)
                        core_pre.save_roi_images(target, mean_image, cat_masks, roi_masks, cat_names, roi_names)
                        for idx, cat_name in enumerate(cat_names):
                            if cat_name.lower() == 'bg':
                                bg_mask = cat_masks[idx]
                                continue
                        if 'bg' not in cat_names:
                            print('please select a background and call it "bg"')
                            continue
                        # cat_bool = np.zeros(shape=np.shape(mean_image))
                        # cat_bool[cat_masks[cat_names.index(cat_name)]] = 1
                        
                        # rois = core_pre.generateROIs(roi_masks, cat_masks, cat_names, mean_image,experiment_info = experiment_conditions, imaging_info =imaging_info)
                    rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                                mean_image, 
                                                experiment_info = experiment_conditions, 
                                                imaging_info =imaging_info,mapping_mode=False)
                    # previous_depth = imaging_info.get('depth')
                    if preprocessing_params.same_rois == True:
                        skipping_mask_single = {'rois':rois, "cat_masks": cat_masks, 'roi_masks': roi_masks, "cat_names": cat_names, 'roi_names': roi_names, 'bg_mask':bg_mask, 'depth':imaging_info.get('layerPosition')[2], 'number': tseries.split('-')[-1], 'img_shape': mean_image.shape}
                        with open(f'{target}_ROIS_skipp.pkl', 'wb') as fo:
                            pickle.dump(skipping_mask_single, fo)
                        skipping_masks[fly][str(same_rois_meta)] = skipping_mask_single
                        core_pre.save_roi_images(target, mean_image, cat_masks, roi_masks, cat_names, roi_names)
                    else:
                        core_pre.save_roi_images(target, mean_image, cat_masks, roi_masks, cat_names, roi_names)
                    with open(f'{skipping_target}_skipping_masks.pkl', 'wb') as fo:
                        pickle.dump(skipping_masks, fo)
                    if analysis_type[0] == 'olfaction':
                        time_series = np.transpose(np.subtract(np.transpose(time_series),
                                                            time_series[:,bg_mask].mean(axis=1)))
                        print('\nBackground subtraction done...')
                        rois = core_pre.getTimeTraces(rois,time_series)
                        final_rois = rois
                    else:
                        (stimulus_information, imaging_information) = \
                                        pmc.get_stim_xml_params(f'{paths.raw}/{condition}/{fly}/{tseries}', paths.stimdata, time_series.shape[0], cplusplus=False)
                        # We can store the parameters inside the objects for further use
                        for roi in rois:
                            roi.extraction_params = extraction_params 
                            
                            # if extraction_type == 'transfer': # Update transferred ROIs
                            roi.experiment_info = experiment_conditions
                            roi.imaging_info = imaging_info
                            roi.analysis_params= analysis_params 
                            # for param in analysis_params.keys():
                            #     roi.analysis_params[param] = analysis_params[param]
                            # else:
                            #     roi.analysis_params= analysis_params 
                        
                        
                        
                        ###########################################
                        ######### divide the Tseries video/stack in trials #############            
                        ###########################################
                        mean_curr_image = time_series.mean(0)
                        # current_movie_ID = current_exp_ID + '-' + current_t_series +'-' + 'cycle_' + str(cycle+1) 
                        
                        # print(current_movie_ID)                       
                        currentTseries = time_series.astype(float) 
                        
                        

                        if 'Chirp' not in visual_stim:
                            (wholeTraces_allTrials_video, respTraces_allTrials, 
                                    baselineTraces_allTrials) = \
                                        pmc.separate_trials_video(currentTseries,stimulus_information,
                                        imaging_information['frame_rate'])
                        
                        ###########################################
                        ######### background subtraction #############            
                        ###########################################

                            
                            pmc.calculate_background_ROIs(bg_mask,rois,in_phase=preprocessing_params.in_phase_bg_subtraction)
                            list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
                        # if "Chirp" in visual_stim:
                        #     rois = pmc.whole_stim_experiment(rois)
                        # else:
                        ###########################################
                        ######### append stimulus information to ROI objects #############            
                        ###########################################
                        
                        
                        
                        
                        
                        
                        
                            
                            (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
                            baselineTraces_allTrials_ROIs) = \
                                pmc.separate_trials_ROI_v4(currentTseries,rois,rois[0].stim_info,
                                                        rois[0].imaging_info['frame_rate'],
                                                        df_method = analysis_params['deltaF_method'][cycle],mov_av=False,analysis_type=analysis_params['analysis_type'][cycle])
                        else:
                            time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,bg_mask].mean(axis=1)))
                            print('\nBackground subtraction done...')
                            list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
                            def getTimeTraces(rois, time_series,df_method = 'mean'):
                                """ Computes the time traces of each ROI given a time series """
                                # dF/F calculation
                                for roi in rois:
                                    roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
                                    roi.calculateDf(stimulus_information, method=df_method[0],moving_avg = False)
                                return rois
                            rois = getTimeTraces(rois,time_series,analysis_params['deltaF_method'])
                            rois = pmc.whole_stim_experiment(rois)

                            def plotAllTraces(rois, fig_save_dir = None):
                                plt.close('')
                                plt.style.use('default')
                                stim_vals = rois[0].stim_info['processed']['epoch_trace_frames']
                                stim_vals = stim_vals/stim_vals.max()
                                stim_vals -= stim_vals.min()
                                plt.plot(stim_vals,'--', lw=1, alpha=.6,color='k')

                                scaler = float(len(rois))
                                for idx, roi in enumerate(rois):
                                    plot_trace = (roi.df_trace+idx)/scaler
                                    plt.plot(plot_trace,lw=1/3.0, alpha=1)

                                plt.xlabel('Frames')
                                plt.title(rois[0].experiment_info['MovieID'])
                                fig = plt.gcf()

                                if fig_save_dir is not None:
                                    f_name = 'Traces_%s' % (rois[0].experiment_info['MovieID'])
                                    os.chdir(fig_save_dir)
                                    fig.savefig('%s.png'% f_name, bbox_inches='tight',transparent=False,dpi=300)
                                    plt.gcf()

                                return fig
                            plotAllTraces(rois,fig_save_dir=summary_save_dir)
                        ###########################################
                        ######### run df/f and analysis #############            
                        ###########################################
                        # analysis_params_local=analysis_params.copy()
                        # analysis_params_local.update({'analysis_type':analysis_type[cycle]})
                        
                        
                        
                                    
                        final_rois = pmc.run_analysis(analysis_params,rois,experiment_conditions,
                                                    imaging_information,summary_save_dir, cycle, expected_polarity, mean_image,
                                                    save_fig=True,fig_save_dir = figure_save_dir,
                                                    exp_ID=('%s_%s' % (target,extraction_params['type'])),
                                                    df_method=analysis_params['deltaF_method'][cycle], keep_prev=False,use_rel_filter=True,reliability_filter=0.4,CSI_filter=0.4,position_filter=True,direction_filter=True,expdir='',  stim_dir = '')

                        
                        
                        ###########################################
                        ######### create visualizations #############            
                        ###########################################
                        # >> put into analysis script
                        images = []
                        (properties, colormaps, vminmax, data_to_extract) = \
                            pmc.select_properties_plot(final_rois , analysis_params['analysis_type'][cycle])
                        if properties is not None:
                            for prop in properties:
                                images.append(ROI_mod.generate_colorMasks_properties(final_rois, prop, cycle=cycle))
                            ##TODO plot only csi for on clusters and off cluster separate
                            pmc.plot_roi_properties(images, properties, colormaps, mean_image,
                                                    vminmax,tseries, imaging_information['depth'],
                                                    save_fig=True, save_dir=target,figsize=(16, 12),
                                                    alpha=0.5)
                        
                        
                        
                        # if plot_individual_traces:
                        #     pmc.plot_roi_traces(final_rois,cycle, analysis_type, figure_save_dir)
                        #     pass
                        #     plt.close('all')
                        
                        
                        # if plot_roi_summ:
                        #     if (analysis_type == 'gratings_transfer_rois_save'): # or ((analysis_type == 'moving_gratings')) 
                        #         import random
                        #         plt.close('all')
                        #         data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id','CSI',
                        #                             'PD', 'exp_ID', 'stim_name']
                                
                                
                        #         roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
                        #         if not os.path.exists(roi_figure_save_dir):
                        #             os.mkdir(roi_figure_save_dir)
                        #         copy_rois = copy.deepcopy(final_rois)
                        #         random.shuffle(copy_rois)
                        #         roi_d = ROI_mod.data_to_list(copy_rois, data_to_extract)
                        #         rois_df = pd.DataFrame.from_dict(roi_d)
                        #         for n,roi in enumerate(copy_rois):
                        #             if n>40:
                        #                 break
                        #             fig = ROI_mod.make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm')
                        #             save_name = '%s_ROI_tunning_summary_%d' % ('a', roi.uniq_id)
                        #             os.chdir(roi_figure_save_dir)
                        #             fig.savefig('%s.png' % save_name,bbox_inches='tight',
                        #                             transparent=False,dpi=300)
                                        
                        #             plt.close('all')
                        #     elif (analysis_type == 'STF_1'):
                        #         data_to_extract = ['reliability', 'uniq_id','CSI']
                                
                        #         roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
                        #         if not os.path.exists(roi_figure_save_dir):
                        #             os.mkdir(roi_figure_save_dir)
                                
                        #         roi_d = ROI_mod.data_to_list(final_rois, data_to_extract)
                        #         rois_df = pd.DataFrame.from_dict(roi_d)
                        #         for n,roi in enumerate(final_rois):
                        #             if n>40:
                        #                 break
                        #             fig = ROI_mod.plot_stf_map(roi,rois_df)
                        #             save_name = '%s_ROI_STF_%d' % (roi.analysis_params['roi_sel_type'][cycle], roi.uniq_id)
                        #             os.chdir(roi_figure_save_dir)
                        #             fig.savefig('%s.png' % save_name,bbox_inches='tight',
                        #                             transparent=False,dpi=300)
                                        
                        #             plt.close('all')
                        
                        
                    
                    
                    '''current rois: [class]
                    for each roi:
                    roi.
                        .raw_trace : raw imaging traces
                        .dff_mean : dff calculated with total recording mean as baseline
                        .source_image : mean image where roi was selected on
                        .category : name of category, none if its ROI mask
                        .imaging_info : info on recording (keys: frame_rate, pixel_size, depth, frame_timings)
                        .uniq_id : unique id
                        .mask : ROI mask 
                        .number_id : number of ROI mask, NONE if category
                        .baseline_method : str, method of calculating baseline > currently only "mean" 
                    '''
                    #saves rois dff in one csv, and all info in dictionary .pkl
                    if '8D' in condition or 'BAR' in condition:
                        ROI_mod.apply_filters(final_rois,None,cycle,CSI_filter=0.4,reliability_filter=None,position_filter=False,direction_filter=False)
                        passed_rois=[]
                        for roi in final_rois:
                            if roi.rejected==False:
                                passed_rois.append(roi)
                        final_rois=passed_rois
                    
                    if 'FFF' in condition:
                        for roi in final_rois:
                            thresh = np.mean(roi.conc_trace) + (2*np.std(roi.conc_trace))
                            if roi.max_response >= thresh:
                                roi.rejected = False
                            else:
                                roi.rejected = True
                        passed_rois=[]
                        for roi in final_rois:
                            if roi.rejected==False:
                                passed_rois.append(roi)
                            
                        final_rois=passed_rois
                    
                    if 'Grating' in condition:
                        for roi in final_rois:
                            thresh = np.mean(roi.oneHz_conc_resp) + (2*np.std(roi.oneHz_conc_resp))
                            if roi.max_response >= thresh:
                                roi.rejected = False
                            else:
                                roi.rejected = True
                        passed_rois=[]
                        for roi in final_rois:
                            if roi.rejected==False:
                                passed_rois.append(roi)
                        final_rois=passed_rois
                        
                        
                    all_rois=pd.DataFrame({})
                    for roi in rois:
                        try: 
                            id =roi.category
                        except:
                            id = roi.number_id
                        if id==['No_category']:
                            continue
                        # id = roi.__dict__.get('number_id')
                        # data_new = pd.DataFrame({id:roi.__dict__.get('dff_mean')})
                        data_new = pd.DataFrame({id:roi.df_trace})
                        all_rois = pd.concat([all_rois, data_new], axis=1)
                    all_rois.to_csv(f'{target}_ROIS.csv', mode='w')
                    if save_data[cycle]:
                        save_dict={'final_rois':final_rois}
                        save_name = f"{target}_ROIS.pkl"
                        # os.makedirs(target, exist_ok=True)
                        with open(save_name, "wb") as fout:
                            pickle.dump(save_dict, fout, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
                        print('\n\n%s saved...\n\n' % save_name)
                        
                        
                        # with open(f'{target}_ROIS.pkl', 'wb') as fo:
                        #     pickle.dump(all_rois_dict, fo)
                        
                        processing_progress[condition][fly][tseries] = [True,True,True,False]
                    
                    if analize_only_first_cycle:
                        break
                    if filter_next_cycles==True and cycle==0:
                        ROI_mod.apply_filters(rois,None,cycle,CSI_filter=0.4,reliability_filter=0.4,position_filter=False,direction_filter=False)
                        passed_rois=[]
                        for roi in rois:
                            if roi.rejected==False:
                                passed_rois.append(roi)
                        rois=passed_rois
                        if len(rois)==0:
                            print('!!!!!!!')
                            print(f'{paths.processed}/{condition}/{fly}/{tseries}/')
                            print('no rois passed')
                            print('!!!!!!!')
                            break
                    # previous_tseries = tseries
                    # previous_number = tseries.split('-')[-1]
                    # 
with open(f'{paths.processed}/processing_progress.pkl', 'wb') as fo:
        pickle.dump(processing_progress, fo)
