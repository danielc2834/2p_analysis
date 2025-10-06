# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:40:15 2020

@author: Jfelipeco. It uses functions and scripts created by Burak Gur. 

"""

import numpy as np
import glob
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import os
from skimage import io
import copy
import pickle
import ast
import warnings
import sys
import yaml
import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the root window

# Ask the user to select the config file
config_path = filedialog.askopenfilename(title="Select Configuration File", filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])

# Ensure the user selected a file
if not config_path:
    raise ValueError("No configuration file selected.")

#load config file 
# config_path = r'C:\Users\ptrimbak\Work_PT\2p_data\241129_L2_zoomed\50ms\config.yaml'
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Validate configuration parameters
required_keys = ['code_path','home_path', 'experiment',  'warnings_filter', 'warnings_category']
for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required configuration parameter: {key}")

# Use configuration parameters
cwd = os.getcwd()
code_path = config.get('code_path', os.path.join(cwd, '2panalysis', 'Helpers'))
sys.path.insert(0, code_path)
# code_path = r'U:\Work_PT_server\Code\2p_analysis_development\2panalysis\Helpers' #Juan desktop

# Import custom functions and workspace
import Helpers.ROI_mod as ROI_mod
import Helpers.core_functions as core_functions
from Helpers.core_functions import saveWorkspace
import Helpers.process_mov_core as pmc


# Apply warnings filter if specified
warnings_filter = config.get('warnings_filter', 'default_filter')
warning_category = config.get('warnings_category', 'default_category')
warning_category_class = getattr(warnings, warning_category, RuntimeWarning)
warnings.simplefilter(warnings_filter, category=warning_category_class)

#%% Initialization variables

"""
This script is designed to deal in particular with T4T5 reocrdings 
that contain one or more cycles

Nevertheless, it can process data containing single Cycle Tseries

for this script ROI selection should be done with SIMA-autoROI selection
or with ROIbuddy

origninal functions by Burak Gür, but includes functions by Jfelipeco
"""

#home_path='F:\\PhD\\experiments\\2p'

# home_path = r'C:\Users\ptrimbak\Work_PT\2p_data\241014_L2_linear_filters\241014_L2_linear_filters' #desktop office Juan
home_path = config.get('home_path') #r'C:\Users\ptrimbak\Work_PT\2p_data\241014_L2_linear_filters\241020_L2_filter_circle' #desktop office Juan

experiment = config.get('experiment')#['50ms'] #'17ms', '50ms']#['0deg', '15deg', '30deg', '45deg', '60deg', '75deg', '90deg']

expected_polarity = config.get('expected_polarity') #'ON' #'ON' # important for fffs for now (only implemented in the fff analysis)
# original_stimDir = home_path + 'Juan_stimuli'
erase_checkpoints = config.get('erase_checkpoints')

# home_path='C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan
#home_path = r'C:\Users\ptrimbak\Work_PT\2p_data\240124_Gcamp_exp'
#experiment= ['Gcamp8m']#['Gcamp8f']#['Gcamp8f_cur5']#['T4T5_L5_gluclRescue']#['second round_L3_f37-467']#['T4T5_r57c10gal4_panneuronal_silencing']#['t4t5_shits_p9silencing']#['t4t5_freqtunning_8dir_Mi1 glucl overexpression']#['T4T5_r57c10gal4_panneuronal_silencing']#['t4t5_microscope_comparison']# ['Mi1_Glucl_acute_rescue']#['Miris_data_piece']
experimenter=config.get('experimenter')#['*_jv_*']

original_stimDir=os.path.join(home_path, 'input_stimuli')



#%%  filtering and preprocessing parameters
#prefilter_rois=False # gets rid of overlaps. should always be true, unless checking for something specific
# if you prefilter choose which quantity to use:
apply_shapeOverlapFilter = config.get('apply_shapeOverlapFilter') #(should be true in general except for manual ROIS)
overlap_filter_property = config.get('overlap_filter_property')#['reliability_PD','reliability_PD'] #!! WARNING: so far only implemented for 1st cycle if there are multicycle recordings
FFT_filtered = config.get('FFT_filtered') # true if fourier algorithm was used in preprocessing
motion_corrected = config.get('motion_corrected') # in principle this should always be true for T4T5 recordings
in_phase_bg_subtraction = config.get('in_phase_bg_subtraction') #in order to compensate for structureed bleedthrough (stripes). 
                                #use to subtract background from the same phase as the individual ROIs
                                #this is not possible with every recording
prefered_direction_filter = config.get('prefered_direction_filter') 
#%% plotting parameters
plot_roi_summ=config.get('plot_roi_summ') #plot roisummary
plot_individual_traces=config.get('plot_individual_traces')
filter_next_cycles = config.get('filter_next_cycles')

#%% extraction parameters
analize_only_first_cycle=config.get('analize_only_first_cycle') #for any analysis where this script is only needed for the first cycle in a tseries with more than one cycle
extraction_type =config.get('extraction_type')#'cluster analysis MH_JF'#'load_manual' #'cluster analysis Henning_20' #'load_manual' # 'load_manual'#'cluster analysis Henning_20' #'transfer' 'manual' 

#stimuli=['8D_edges_find_rois_save','1-dir_edge_1pol','moving_gratings','moving_gratings','1-dir_edge_1pol','1-dir_edge_1pol'] #['4D_edges','12_dir_random_driftingstripe']#['8D_edges_find_rois_save']#['4D_edges','12_dir_random_driftingstripe'],#['5sFFF_analyze_save']#['8D_edges_find_rois_save']#['8D_edges_find_rois_save','moving gratings'] #['5sFFF_analyze_save'],'moving gratings'] #['2-diredges','moving gratings']#['2-diredges','whitenoise_el'] 
#analysis_type =['8D_edges_find_rois_save','1-dir_edge_1pol','moving_gratings','moving_gratings','1-dir_edge_1pol','1-dir_edge_1pol']#['4D_edges','12_dir_random_driftingstripe']# ['8D_edges_find_rois_save']#['4D_edges','12_dir_random_driftingstripe']#['5sFFF_analyze_save','5sFFF_analyze_save']#['8D_edges_find_rois_save'] #['8D_edges_find_rois_save','moving gratings'] #['5sFFF_analyze_save']   # ['2D_edges_find_save','moving_gratings'] #'5sFFF_analyze_save'
#TODO automatize the analysis instruction
use_avg_data_for_roi_extract=config.get('use_avg_data_for_roi_extract')
use_other_series_roiExtraction=config.get('use_other_series_roiExtraction')
 # allcyles means that the whole Tseries (including multiple cycles, are used for ROI autoselection). 1st cycle uses only the first cycle
 #manual otsus
save_data=config.get('save_data') # list of booleans for which traces to keep in the final resulting pickle file


#%% conditional params (not to be changed manually)

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
                                #

if extraction_type != 'load_manual':
    load_categories=True #False when the categories are already asigned (for example in manually selected rois)
else:
    load_categories=False         
#%%

#################################
########### loop through experiment folders #####
#################################

for iidx,exp in enumerate(experiment):
  
    #################################
    ########### get the paths for the Time series folders#####
    #################################
    
    experimenter_code=experimenter[iidx]
    Tseries_paths=core_functions.produce_Tseries_list(home_path, exp,experimenter_code)
    raw_dir=os.path.join(home_path, exp, 'raw', '')
   
    #################################
    ########### check if there is experiment metadata with info about genotype, flyid, etc...#####
    #################################
    
    meta_exists=len(glob.glob(os.path.join(home_path, exp, 'processed', 'metadata_fixed.csv')))>0
    if meta_exists:
        pass
    else:
        raise Exception('%s has no metadata file'%(exp))
    #################################
    ########### loop through TSeries folders#####        
    #################################
    for xx,dataDir in enumerate(Tseries_paths): # in other scripts dataDir is equivalent to Tser
        print('Tser %s / %s' %(xx+1,len(Tseries_paths)))
        print(dataDir)
        
        #################################
        ########### if they exists and boolean is true, eliminate pickle checkpoints#####  
        #################################
        
        if erase_checkpoints==True:
            picklesss=glob.glob(dataDir+'\\*checkpoint*.pickle')
            for pick in picklesss:
                os.remove(pick)
                
        #################################        
        ########### Take the experimental parameters###########
        #################################
        
        metadata_df=core_functions.extract_fly_metadata(dataDir)     
        
        #################################
        ########### if the current Tseries haas no entry in the metadata file, enter mapping mode. this is a mode that can be used to map rois 
        ########### and then continue recording the same fly based on the mapping information. currently implemented only for t4t5 analysis
        #################################
        
        try:
            current_exp_ID = metadata_df['date'][0]
            mapping_mode=False
        except IndexError:
            print('Tser has no metadata entry. mapping mode')         
            current_exp_ID='mapping'
            mapping_mode=True
            #continue
        #TODO extract analysis type directly from metadata
        if mapping_mode==False:
            current_t_series =metadata_df['recording'][0]        
            Genotype = metadata_df['genotype'][0]
            Age = metadata_df['age'][0]
            Sex = metadata_df['sex'][0]
            treatment = metadata_df['treatment'][0]
            z_depth_bin=metadata_df['depth_bin'][0]
            z_depth=metadata_df['depth'][0]
            rotation=metadata_df['rotation'][0]
            analysis_type=ast.literal_eval(metadata_df['analysis_type'][0])
            dFmethod=ast.literal_eval(metadata_df['Df_method'][0])
            analysis_params = {'deltaF_method': dFmethod,   #TODO maybe make this dependent on the stimulus type
                   'analysis_type': analysis_type,
                   'Background':'otsus'}
                #    'stimtype': eval(str(metadata_df['Stimulus'][0]))}  
            
            if prefered_direction_filter:
                orientation_filter = metadata_df['orientation_of_stim'][0]
            else:
                orientation_filter = None

        else:
            current_t_series = 'mapping'        
            Genotype = 'T4T5'    
            Age = None    
            Sex = None    
            treatment = None    
            z_depth_bin = None    
            z_depth = None    
            rotation = None
            analysis_type = ['8D_edges_find_rois_save']
            dFmethod = ['rolling_mean']
            analysis_params = {'deltaF_method': dFmethod,   #TODO maybe make this dependent on the stimulus type
                'analysis_type': analysis_type,
                'Background':'otsus'}  
            orientation_filter = None  

        number_of_cycles,lenghtCycles=core_functions.find_cycles(dataDir)
        experiment_conditions= \
                {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex, 'treatment':treatment,
                'FlyID' : current_exp_ID,'expected_polarity':expected_polarity, 'MovieID':  current_t_series,
                'z_depth':z_depth,'z_depth_bin':z_depth_bin,'analysis_type': analysis_type,
                 'orientation_filter':orientation_filter,'rotation':rotation}

        #################################
        #### get stimulus information paths. when there is more than one cycle in a recording, this collects all relevant stimuli  variable cplusplus is for backwards compatibility with old stimuli#####
        #################################
        
        mult_stim_types, stim_paths,cplusplus=core_functions.find_stimfile_check_stim(exp,dataDir,number_of_cycles)
        
        #################################
        #### extract microscope metadata ####
        #################################
        
        imaging_information=pmc.extract_xml_info(dataDir)
        
        #################################
        ##### load the time series and store it in memory in a cycle by cycle basis####
        #################################
        
        if FFT_filtered==True:
            Tser_string='stack_FFT*.tif'
        else:
             Tser_string='_motCorr.tif'  #pradeep's mot corr
        mean_image, time_series = core_functions.load_Tseries(dataDir,number_of_cycles,lenghtCycles,file_string=Tser_string) 
        
        #################################
        ### prepare paths for output saving ####
        #################################
        
        transfer_data_name = current_exp_ID +'_'+ current_t_series +'.pickle'
        saveOutputDir = os.path.join( home_path, exp, 'processed', current_exp_ID, current_t_series)
        try:
            os.mkdir(os.path.join(home_path, exp, 'processed', current_exp_ID))
            os.mkdir(os.path.join(home_path, exp, 'processed', current_exp_ID, current_t_series))
        except:
            pass
        finally:
            pass

        summary_save_dir= os.path.join(home_path, exp, 'summary', current_exp_ID, current_t_series)
        figure_save_dir = os.path.join(home_path, exp, 'summary', current_exp_ID, current_t_series, 'figures')
        if not os.path.exists(figure_save_dir):
            try:
                os.mkdir(os.path.join(home_path, exp,'summary', current_exp_ID))
            except:
                pass
            finally:
                pass
            try:
                os.mkdir(os.path.join(home_path, exp, 'summary', current_exp_ID, current_t_series))
                os.mkdir(os.path.join(home_path, exp, 'summary', current_exp_ID, current_t_series, 'figures'))
            except:
                pass
            finally:
                pass

        #################################
        """
        ROI auto-extraction or loading. (if more than one cycle in the recording
                              this script assumes that the cycles are 
                              recorded at the same depth/ neurons)
        """
        """make sure there are no prev ROIs around"""
        #################################
        
        try:
            del rois
        except:
            pass
        finally:
            pass

        #################################
        ### load manually selected rois ####
        #################################
        
        if extraction_type == 'load_manual': 
            bg_mask,cat_names,roi_masks,all_clusters_image=pmc.import_clusters(dataDir,manual=True,FFT_filtered=FFT_filtered,video=time_series[0])
            all_rois_image=all_clusters_image


        #################################
        ### perform clustering analysis to do segmentation ####
        #################################
        elif  extraction_type =='cluster analysis MH_JF':               
            dataset = dataDir
            _,cat_names,cat_masks, I_zone=pmc.import_cat_ROIs(dataset,mot_corr=True,ROIs_label='categories', ignore_background = True, time_series = time_series[0])

            cluster_info_dir=os.path.split(figure_save_dir)[0]
            cluster_info_dir=os.path.split(cluster_info_dir)[0]

            extra_info={'stim_dir':original_stimDir,'stim_types':mult_stim_types,
                        'stim_paths':stim_paths,'dataDir':dataDir,'cplusplus':cplusplus,    
                        'cat_names':cat_names,'I_zone':I_zone,
                        'fig_dir':cluster_info_dir,'dataDir':dataDir} #,'I_zone':inclusion_zone
            
            #################################
          # ift ROIs where calculated before, jus upload a checkpoint #commented out by PT: 'cats':categories
          #################################

            if len(glob.glob(dataDir+'rois_checkpoint.pickle'))==1:
                load_path = open(glob.glob(dataDir+'rois_checkpoint.pickle')[0], 'rb')
                roi_dict = pickle.load(load_path)
                roi_dict,additional_params=roi_dict['rois'],roi_dict['params']
            else:
                roi_dict, additional_params = \
                        pmc.run_ROI_selection(extraction_params,time_series[0],image_to_select=mean_image,**extra_info)
                if mapping_mode==False:
                    checkpoint={'rois':roi_dict,'params':additional_params}
                    saveVar = open(dataDir+'rois_checkpoint.pickle', "wb")
                    pickle.dump(checkpoint, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
                                                                # also should be compatible with python 3
                    saveVar.close()

            extraction_params.update(additional_params)            

            roi_masks=roi_dict['roi_masks']
            all_rois_image=roi_dict['all_rois_image']
            bg_mask=roi_dict['background']
            
        elif extraction_type == 'manual':
                (cat_masks, cat_names, roi_masks, all_rois_image, rois,
                threshold_dict) = \
                    pmc.run_ROI_selection(extraction_params,time_series,image_to_select=mean_image)

        
          #################################
          ##### Generate instances of the ROI class ########
          #################################
        
        if extraction_type == 'load_manual':
            rois = ROI_mod.generate_ROI_instances_categorized(roi_masks, cat_names,
                                                    mean_image, 
                                                    experiment_info = experiment_conditions, 
                                                    imaging_info =imaging_information)
        
        else:

            rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                                    mean_image, 
                                                    experiment_info = experiment_conditions, 
                                                    imaging_info =imaging_information,mapping_mode=mapping_mode)
        
        # We can store the parameters inside the objects for further use
        for roi in rois:
            roi.extraction_params = extraction_params 
            
            if extraction_type == 'transfer': # Update transferred ROIs
                roi.experiment_info = experiment_conditions
                roi.imaging_info = imaging_information
                for param in analysis_params.keys():
                    roi.analysis_params[param] = analysis_params[param]
            else:
                roi.analysis_params= analysis_params 

        #################################
        #### filter clusters based on size and overlaps.... currently not in use and not recommended, to be deprecated #########
        #################################
        
        if apply_shapeOverlapFilter:
            lower_limit=int(round(1/imaging_information['x_size'])) # for the functional connectomics paper this info is 1 and 6.25
            higherlimit=int(round(6.25/imaging_information['x_size'])) 
        else:
            rois, sep_masks_image = pmc.clusters_restrict_size_regions(rois,1000,0,no_filter=True)

        #######################
        ###### Processs and analyze data in a cycle to cycle basis #######
        #######################
        
        for cycle,stimOutputFile in enumerate(stim_paths): #perform the relevant analyses per cycle
            #temporary:
            #if cycle == 1:
            #     if rois[0].analysis_params['stimtype'][cycle] != 'moving_binary_persistantlum':
            #    continue   
            
            # temporary
            # if roi.experiment_info['treatment'] != 'Tm3_Homocygous':
            #     continue

            if cycle==0 and first_cycle_ROIs:
                pass
            elif cycle>0:# and first_cycle_ROIs:

                map(lambda roi : roi.clear_prev_atributes(cycle,['CSI','CS','PD_ON','PD_OFF','DSI_OFF','DSI_ON','dir_max_resp','Center_position_filter','extraction_params','imaging_info',\
                            'experiment_info','analysis_params','bg_mask', 'category','mask','uniq_id','cycle1_reliability','rejected']), rois)
                if len(glob.glob(dataDir+'checkpoint2_cycle%s.pickle' %(cycle)))==1 and cycle==0:# and cycle!=2:
                    #temporary:
                    print('skipped cycle: %s'%(cycle))
                    continue
            
            # try to find if cycle is already analyzed and saved, if yes, skip
            # cycle_pickle=glob.glob(saveOutputDir+'\\*cycle%s*.pickle'%(cycle))
            # if len(cycle_pickle)>0 and :
            #     continue


            ###############################
            #%% Get the stimulus and imaging information
            ###############################

                      
            currentTseries=time_series[cycle]
            # currentTseries=time_series
            Tseries_len=currentTseries.shape[0]
            current_movie_ID= current_exp_ID + '_' + current_t_series +'_cycle_'+ str(cycle+1)
            experiment_conditions= \
                {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex, 'treatment':treatment,
                'FlyID' : current_exp_ID, 'MovieID':  current_movie_ID, 'expected_polarity':expected_polarity, 'cycles':[range(len(stim_paths))],
                'z_depth':z_depth,'z_depth_bin':z_depth_bin, 'analysis_type':  analysis_type}
            figure_save_dir_local=figure_save_dir
            figure_save_dir_local=figure_save_dir_local + '\\cycle_0' + str(cycle+1) +'\\'
            #if os.path.exist(figure_save_dir_local)==False:
            try:
                os.mkdir(figure_save_dir_local)
            except:
                pass
            finally:
                pass

            (stimulus_information, imaging_information) = \
                                pmc.get_stim_xml_params(dataDir, stimOutputFile,original_stimDir,Tseries_len,cplusplus=cplusplus)
            
            
            ######## to be deprecated, but perhaps relevant for specific types of stimulus ########]
            # correct the direction info in the stimulus data for chosen stimuli types. so it is consistent accross dif stim types (for example gratings and edges)
            #if mult_stim_types[cycle]=='DriftingStripe_5sec_edges_100deg_degAz_degEl_Sequential_LumDec_right_left_2D_20sec.txt\n': # for 2 edges stim. the direction is inverted with respect to the gratings
                #stimulus_information['direction']=np.array(stimulus_information['direction'])*-1 
            ########
            
            
            ###########################################
            ######### divide the Tseries video/stack in trials #############            
            ###########################################
            mean_curr_image = time_series[cycle].mean(0)
            current_movie_ID = current_exp_ID + '-' + current_t_series +'-' + 'cycle_' + str(cycle+1) 
            
            print(current_movie_ID)                       
            currentTseries = currentTseries.astype(float) 

            (wholeTraces_allTrials_video, respTraces_allTrials, 
                     baselineTraces_allTrials) = \
                         pmc.separate_trials_video(currentTseries,stimulus_information,
                         imaging_information['frame_rate'])
            
            
            ###########################################
            ######### background subtraction #############            
            ###########################################
          
            
            pmc.calculate_background_ROIs(bg_mask,rois,in_phase=in_phase_bg_subtraction)
            


            ###########################################
            ######### append stimulus information to ROI objects #############            
            ###########################################
            
            
            list(map(lambda roi: roi.appendStimInfo(stimulus_information), rois))
            
            
            ###########################################
            ######### analysis checkpoint #############            
            ###########################################
            
            print('curr cycle %s'%(cycle))
            if len(glob.glob(dataDir+'checkpoint2_cycle%s.pickle' %(cycle)))==1:
                load_path = open(glob.glob(dataDir+'checkpoint2_cycle%s.pickle' %(cycle))[0], 'rb')
                roi_dict = pickle.load(load_path)
                rois,analysis_params=roi_dict['rois'],roi_dict['params']
            
            ###########################################
            ######### separate traces in trials #############            
            ###########################################
            else:# ROI trial separated responses
                (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
                baselineTraces_allTrials_ROIs) = \
                    pmc.separate_trials_ROI_v4(currentTseries,rois,rois[0].stim_info,
                                            rois[0].imaging_info['frame_rate'],
                                            df_method = analysis_params['deltaF_method'][cycle],mov_av=False,analysis_type=analysis_params['analysis_type'][cycle])
            
            if mapping_mode==False:
                #create checkpoint
                checkpoint={'rois':rois,'params':analysis_params}
                saveVar = open(dataDir+'checkpoint2_cycle%s.pickle' %(cycle), "wb")
                pickle.dump(checkpoint, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
                saveVar.close()
            
            
            ###########################################
            ######### run df/f and analysis #############            
            ###########################################

            analysis_params_local=analysis_params.copy()
            analysis_params_local.update({'analysis_type':analysis_type[cycle]})
            


            final_rois = pmc.run_analysis(analysis_params_local,rois,experiment_conditions,
                                        imaging_information,summary_save_dir, cycle, expected_polarity, mean_image,
                                        save_fig=True,fig_save_dir = figure_save_dir,
                                        exp_ID=('%s_%s' % (current_movie_ID,extraction_params['type'])),
                                        df_method=analysis_params['deltaF_method'][cycle], keep_prev=False,use_rel_filter=True,reliability_filter=0.4,CSI_filter=0.4,position_filter=True,direction_filter=True,expdir=home_path+exp,  stim_dir = os.path.join(dataDir))


            if mapping_mode:
                sys.exit()

            ###########################################
            ######### create visualizations #############            
            ###########################################

            final_rois=final_rois
            pmc.plot_roi_masks(all_rois_image,mean_image,len(final_rois),
            current_movie_ID,save_fig=True,
            save_dir=figure_save_dir,alpha=0.4)

            
            images = []
            (properties, colormaps, vminmax, data_to_extract) = \
                pmc.select_properties_plot(final_rois , analysis_params_local['analysis_type'])
            if properties is not None:
                for prop in properties:
                    images.append(ROI_mod.generate_colorMasks_properties(final_rois, prop,cycle=cycle))
                ##TODO plot only csi for on clusters and off cluster separate
                pmc.plot_roi_properties(images, properties, colormaps, mean_image,
                                        vminmax,current_movie_ID, imaging_information['depth'],
                                        save_fig=True, save_dir=figure_save_dir_local,figsize=(16, 12),
                                        alpha=0.5)
            

            if plot_individual_traces[cycle]:
                pmc.plot_roi_traces(final_rois,cycle,analysis_type,figure_save_dir_local)
                pass
                plt.close('all')
            
            if plot_roi_summ:
                if (analysis_type == 'gratings_transfer_rois_save'): # or ((analysis_type == 'moving_gratings')) 
                    import random
                    plt.close('all')
                    data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id','CSI',
                                        'PD', 'exp_ID', 'stim_name']
                    
                    
                    roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
                    if not os.path.exists(roi_figure_save_dir):
                        os.mkdir(roi_figure_save_dir)
                    copy_rois = copy.deepcopy(final_rois)
                    random.shuffle(copy_rois)
                    roi_d = ROI_mod.data_to_list(copy_rois, data_to_extract)
                    rois_df = pd.DataFrame.from_dict(roi_d)
                    for n,roi in enumerate(copy_rois):
                        if n>40:
                            break
                        fig = ROI_mod.make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm')
                        save_name = '%s_ROI_tunning_summary_%d' % ('a', roi.uniq_id)
                        os.chdir(roi_figure_save_dir)
                        fig.savefig('%s.png' % save_name,bbox_inches='tight',
                                        transparent=False,dpi=300)
                            
                        plt.close('all')
                elif (analysis_type == 'STF_1'):
                    data_to_extract = ['reliability', 'uniq_id','CSI']
                    
                    roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
                    if not os.path.exists(roi_figure_save_dir):
                        os.mkdir(roi_figure_save_dir)
                    
                    roi_d = ROI_mod.data_to_list(final_rois, data_to_extract)
                    rois_df = pd.DataFrame.from_dict(roi_d)
                    for n,roi in enumerate(final_rois):
                        if n>40:
                            break
                        fig = ROI_mod.plot_stf_map(roi,rois_df)
                        save_name = '%s_ROI_STF_%d' % (roi.analysis_params['roi_sel_type'], roi.uniq_id)
                        os.chdir(roi_figure_save_dir)
                        fig.savefig('%s.png' % save_name,bbox_inches='tight',
                                        transparent=False,dpi=300)
                            
                        plt.close('all')
            
            

            
            
            ###########################################
            ######### save data #############            
            ###########################################

            if save_data[cycle]:

            
                save_dict={'final_rois':final_rois}
                save_name = os.path.join(saveOutputDir,'{ID}_cycle{cycle}_{stm}.pickle'.format(ID=current_movie_ID,cycle=cycle,stm=stimulus_information['stim_name'].split('.')[0]))
                if os.path.exists(saveOutputDir)==False:
                    try:
                        os.mkdir(saveOutputDir)
                    except:
                        pass
                    finally:
                        pass
                saveVar = open(save_name, "wb")
                pickle.dump(save_dict, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
                saveVar.close()
                print('\n\n%s saved...\n\n' % save_name)
            
            # if there is a need, produce a matrix containing the processed traces
            #if export_Datamatrix[cycle]==True:
                #ROI_mod.create_data_matrix(rois,saveOutputDir)
            
            ###########################################
            ######### prepare for next round of analysis if needed #############            
            ###########################################
            
            
            if analize_only_first_cycle:
                break
            if filter_next_cycles==True and cycle==0:
                ROI_mod.apply_filters(rois,orientation_filter,cycle,CSI_filter=0.4,reliability_filter=0.4,position_filter=False,direction_filter=prefered_direction_filter)
                passed_rois=[]
                for roi in rois:
                    if roi.rejected==False:
                        passed_rois.append(roi)
                rois=passed_rois
                if len(rois)==0:
                    print('!!!!!!!')
                    print(dataDir)
                    print('no rois passed')
                    print('!!!!!!!')
                    break
        # except:
        #     print('failed')
        #     print(dataDir, cycle) 
        #     print(len(rois))   
        #     continue
# %%
