#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:58:17 2020

@author: burakgur
modified by sebasto_7
modified by Juan_F
"""
# %% Importing packages
import os
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
code_path = r'D:\progresources\Calcium-imaging-analysis\general_lab_code'
sys.path.insert(0, code_path) 
import ROI_mod
import white_noise_core as wnc
from core_functions import saveWorkspace
import process_mov_core as pmc
import analysis_core as aCore
import pandas as pd
import cPickle
import sima
#%%


def import_cat_ROIs(Tser_path,mot_corr=True,ROIs_label='categories'):
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
    sima_folder=glob.glob(Tser_path+'*MC*.sima')[0] #MC is in all motion corrected SIMA objects
    dataset=sima.ImagingDataset.load(sima_folder)
    Manual_ROIs=dataset.ROIs[ROIs_label]
    ROI_list=[]
    """
    put ROI masks in numpy array
    """
    check=0
    tag_list=[]
    for idx,element in enumerate(Manual_ROIs):
        if 'BG' in element.tags:
            BG=np.squeeze(np.array(element))
            #background=np.reshape(np.array(element),(dataset.frame_shape[1],dataset.frame_shape[2],1))
            #background.astype(int)
            BG=BG.astype(bool)
            bg_index=idx
            check=1
        elif 'BG' not in element.tags:
            region=np.squeeze(np.array(element))
            region.astype(int)
            ROI_list.append(region)
            tags=element.tags
            if len(tags)==0:
                tag_list.append(['No_tag'])
            else:
                tag_list.append([tags])
    if check==0:
        raise Exception('no Background for %s' %(dataDir))    
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
    return  BG,tag_list, ROI_list  #this is equivalent to cat_names, cat_masks in Buraks code

def extract_fly_metadata(Tser):
    metadata_path=os.path.split(os.path.split(os.path.split(Tser)[0])[0])[0]
    metadata_path=os.path.split(metadata_path)[0]+'\\processed\\metadata_fixed.csv'
    fly=os.path.split(os.path.split(os.path.split(Tser)[0])[0])[1]
    Tser=os.path.split(os.path.split(Tser)[0])[1]
    # import metadata as pandas dataframe
    metadata=pd.read_csv(metadata_path,header=0,index_col=0,skipinitialspace=True)
    fly_metadata=metadata.loc[metadata['date']==fly]
    fly_metadata=fly_metadata.loc[fly_metadata['recording']==Tser]
    fly_metadata=fly_metadata.to_dict(orient='list')
    return fly_metadata

# %% Parameters to adjust



 #experiment = 'Mi1_GluCla_OE_Cnt_suff_exp'
# Experimental parameters
""" experiment = 'Mi1_GluCla_OE_Cnt_suff_exp' #  'Mi1_GluCla_Mi1_suff_exp', 'Mi1_GluCla_Neg_Cnt_suff_exp', 'Mi1_GluCla_OE_Cnt_suff_exp'
current_exp_ID = '20201216_seb_fly4'
current_t_series ='TSeries-fly4-002'
Genotype = 'Mi1_GCaMP6f_UAS-GluCla_OECnt_GluCla_suff_exp' # 'Mi1_GCaMP6f_Mi1_GluCla_ExpLine_GluCla_suff_exp' , 'Mi1_GCaMP6f_Mi1_GluCla_PosCnt_GluCla_suff_exp', 'Mi1_GCaMP6f_Mi1_GluCla_NegCnt_GluCla_suff_exp', 'Mi1_GCaMP6f_UAS-GluCla_OECnt_GluCla_suff_exp'
save_folder_geno = 'OECnt'
Age = '2'
Sex = 'm' """

#%%produce Tseries paths and load the experiment information
home_path= 'E:\\PhD\\experiments\\2p\\'
experiment= 'T4T5_RFs_Mi1_glucl_overexpression'
current_exp_ID= '20210315_jv_fly3\\Tseries-fly3-002'####
transfer_data_path= 'processed\\'+ current_exp_ID +'\\'+current_exp_ID.replace('\\','-') + '-cycle_1.pickle' #####relevant if ROIs are transfered from one recording to other or from one cycle to other
fly=current_exp_ID.split('\\')[0]
Tser=current_exp_ID.split('\\')[1]
saveOutputDir=home_path+'\\'+'processed'+'\\'+ fly + '\\'+Tser
transfer_data_path=home_path+ experiment +'\\'+transfer_data_path
save_dataFolder = 'E:\\PhD\\Experiments\\2p'
current_exp_ID=current_exp_ID.replace('\\','-')
#Tseries_paths=produce_Tseries_list(experiment,home_path)

#loop through Tseries files (implementation pending)

# Analysis parameters

analysis_type = 'ternaryWN_elavation_RF'
# 'Dark_screen'
# 'ternaryWN_elavation_RF'

# ROI selection/extraction parameters
extraction_type = 'transfer' # 'SIMA-STICA' 'transfer' 'manual' 'load-STICA'
#transfer_data_name = '20210315_jv_fly3-Tseries-fly3-002-cycle_1.pickle'
load_categories=True
motion_correction=True
roiExtraction_tseries='1st_cylce' # allcyles means that the whole Tseries (including multiple cycles, are used for ROI autoselection)
time_series_stack = 'stack_FFT_filtered_Mcorr.tif' # 'Raw_stack.tif' 'Mot_corr_stack.tif'
plt.close('all')
save_data = True # choose True or False
load_categories=False
use_other_series_roiExtraction=True
use_avg_data_for_roi_extract=False
threshold_dict = {'SNR': 0,'reliability': 0}

if extraction_type=="transfer":
    transfer_traces=True # set to true to include traces from a previous stimuli protocol in the transfered ROIs 
else:
    transfer_traces==False

#TODO include components in the extraction, include maximum displacement (this can be extracted form image size difference?) and alignment method in params 

# %% Setting the directories
dataFolder = 'E:\\PhD\\Experiments\\2p'
initialDirectory = os.path.join(home_path, experiment)
raw_datadir= os.path.join(initialDirectory,'raw')
processed_datadir=os.path.join(initialDirectory,'processed')
summary_datadir=os.path.join(initialDirectory,'summary')
if not os.path.exists(summary_datadir):
    os.mkdir(summary_datadir)
figuredir_base= summary_datadir+ '\\' + fly +'\\' +Tser.replace('_','-') 
if not os.path.exists(summary_datadir+ '\\' + fly):
    os.mkdir(summary_datadir+ '\\' + fly)
if not os.path.exists(figuredir_base):
    os.mkdir(figuredir_base)
figure_save_dir = figuredir_base + '\\figures' #warning.  current_exp_ID contains here flyid + Tseries
                                                             # in other scripts it's different
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)


alignedDataDir = os.path.join(raw_datadir,fly,Tser)+'\\'
original_stimDir = os.path.join(home_path, 'Juan_stimuli')
saveOutputDir = os.path.join(processed_datadir, fly,Tser)
if not os.path.exists(saveOutputDir):
    os.mkdir(saveOutputDir)
summary_save_dir = figure_save_dir

# initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
# alignedDataDir = os.path.join(initialDirectory,
#                               'selected_experiments')
# stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
# saveOutputDir = os.path.join(initialDirectory, 'analyzed_data/200820_Tm2_Tm4')
# summary_save_dir = os.path.join(alignedDataDir,
#                                 '_summaries')



#%% Get the stimulus and imaging information
dataDir = alignedDataDir #, current_t_series

number_of_cycles,lenghtCycles=pmc.find_cycles(dataDir) 
mult_stim_types, stim_paths=pmc.find_stimfile_check_stim(experiment,dataDir,number_of_cycles)
mean_image, time_series = pmc.load_Tseries(dataDir,number_of_cycles,lenghtCycles) 
imaging_information=pmc.extract_xml_info(dataDir)
# temporary piece of code (to just produce ROIs without regard for the roi CSI, DSI, SNR etc):
if number_of_cycles==2:
    #TODO analize first cycle with previous code for now. try Mi9 first and see how that works
    analysis_type_1= '2D_edges_find_save' 
    Tseries_1=time_series[0]
    stimOutputFile_1=stim_paths[0]
    time_series=time_series[1]
    stimOutputFile=stim_paths[1]


(stimulus_information, imaging_information) = \
                    pmc.get_stim_xml_params(dataDir, stimOutputFile,original_stimDir)

if number_of_cycles==2: # if we have more than one cycle, the mictimes need to be 
                        #chopped accordingly, so the reverse corr works
    frame_times=stimulus_information['frame_timings']
    frame_times=frame_times[lenghtCycles[0]:]
    frame_times=frame_times-frame_times[0]
    stimulus_information['frame_timings']=frame_times
mean_image = time_series.mean(0)
current_movie_ID = current_exp_ID
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)

#experiment_conditions = \
#    {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
#    'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}
    

#%% Define analysis/extraction parameters and run region selection
#   generate ROI objects.

# Organizing extraction parameters

        
metadata_df=extract_fly_metadata(dataDir)        
current_exp_ID = metadata_df['date'][0]
current_t_series =metadata_df['recording'][0]         
Genotype = metadata_df['genotype'][0]
Age = metadata_df['age'][0]
Sex = metadata_df['gender'][0]
treatment = metadata_df['treatment'][0]

experiment_conditions= \
        {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex, 'treatment':treatment,
        'FlyID' : current_exp_ID, 'MovieID':  current_t_series}


#TODO continue here
extraction_params = \
pmc.organize_extraction_params(extraction_type,
                        current_t_series=Tser,
                        current_exp_ID=fly,
                        stimInputDir=stim_paths,
                        use_other_series_roiExtraction = use_other_series_roiExtraction,
                        use_avg_data_for_roi_extract = use_avg_data_for_roi_extract,
                        roiExtraction_tseries=roiExtraction_tseries,
                        transfer_data_n = transfer_data_path,
                        transfer_data_store_dir = saveOutputDir,
                        analysis_type = analysis_type,
                        imaging_information=imaging_information,
                        experiment_conditions=experiment_conditions,
                        stimuli=mult_stim_types,
                        thresholds=threshold_dict,
                        )
        
#TODO continue fixing here

analysis_params = {'deltaF_method': 'gray',
                   'analysis_type': analysis_type} 



# Select/extract ROIs
if extraction_type == 'load-STICA': 
    dataset,roi_masks,all_clusters_image=import_clusters(dataDir)
elif extraction_type == 'load_manual': 
    dataset,roi_masks,all_clusters_image=import_clusters(dataDir,manual=True)
elif extraction_type == 'transfer':
    rois = aCore.transferROIs(transfer_data_path,
                            analysis_type,experiment_info = experiment_conditions,transfer_traces=transfer_traces )


else:
    #(cat_masks, cat_names, roi_masks, all_rois_image, rois,
    #threshold_dict) = \
        #pmc.run_ROI_selection(extraction_params,image_to_select=mean_image)
    raise NameError(' direct manual ROI selection not implemented_use ROIbuddy')


### category ROIs loading

if load_categories: 
    #dataset = sima.ImagingDataset.load( dataDir +'TserMC.sima')
    bg_mask,cat_names,cat_masks=import_cat_ROIs(dataset,mot_corr=True,ROIs_label='categories')
elif extraction_type == 'transfer' and load_categories==False:
    bg_mask,_,_=import_cat_ROIs(dataDir)
    print('categories imported from transfer')
else:
    # A mask needed in SIMA STICA to exclude ROIs based on regions
    cat_bool = np.zeros(shape=np.shape(mean_image))
    for idx, cat_name in enumerate(cat_names):
        if cat_name.lower() == 'bg\r': # Seb: added '\r' 
            bg_mask = cat_masks[idx]
            continue
        elif cat_name.lower() == 'bg': 
            bg_mask = cat_masks[idx]
            continue
        elif cat_name.lower() =='otsu\r': # Seb: added '\r' 
            otsu_mask = cat_masks[idx]
            continue
        elif cat_name.lower() =='otsu': 
            otsu_mask = cat_masks[idx]
            continue
        cat_bool[cat_masks[cat_names.index(cat_name)]] = 1

        #### filter ROIs by size, overlaps and circularity:

if extraction_type != 'transfer':
    # Generate ROI_bg instances
    if rois == None:
        del rois
    rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)
    lower_limit=int(round(1/imaging_information['x_size'])) 
    higherlimit=int(round(3/imaging_information['x_size'])) 
    if extraction_type != 'load_manual':
        rois, sep_masks_image = pmc.clusters_restrict_size_regions(rois,higherlimit,lower_limit)
    #else:  #This was used for manually selected t4t5 rois, in order to compare with auto-selected ones
        #rois, sep_masks_image = pmc.clusters_restrict_size_regions(rois,higherlimit*10,lower_limit)


 
# Update transferred ROIs
for roi in rois:
    roi.extraction_params = extraction_params
    if extraction_type != 'transfer':  #transfer takes along with it already some properties
        roi.experiment_info = experiment_conditions
        roi.imaging_info = imaging_information
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
    else:
        roi.analysis_params= analysis_params

 

# %% 
# BG subtraction
time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,bg_mask].mean(axis=1)))
print('\n Background subtraction done...')

#Seb: commented this out
# # Stimulus 
# stimpath = os.path.join(stimInputDir,
#                          'StimulusData_Discrete_1_12_100000_Seed_735723.mat' )
# stim = h5py.File(stimpath)
# #stim file has group names 'stimulus' and 'stimulusMetadata'
# #to reach group keys run list(stim.keys())
# stim = stim['stimulus'][()]  ##stim is np array with frames

#Seb: generating the ternary noise stimulus
choiseArr = [0,0.5,1]
x = 16
y = 1
z= 10000 # z- dimension (here frames presented over time)
np.random.seed(54378) #Fix seed. Do not ever change before calling this from stim_output_file
stim= np.random.choice(choiseArr, size=(z,x,y))

# ROI raw signals
for iROI, roi in enumerate(rois):
    roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
    roi.wn_stim = stim

# Append relevant information and calculate some parameters
map(lambda roi: roi.appendStimInfo(stimulus_information), rois)
map(lambda roi: roi.setSourceImage(mean_image), rois)


#%% White noise analysis

rois = ROI_mod.reverse_correlation_analysis(rois)
rois = ROI_mod.STA_response_prediction(rois,stim)
final_rois = rois
# %% Save data
if save_data:
    os.chdir(dataFolder) # Seb: data_save_vars.txt file needs to be there
    varDict = locals()
    pckl_save_name = ('%s_%s' % (current_movie_ID.replace('\\','-'), extraction_params['analysis_type']))      
    
    save_dict={'final_rois':final_rois}
    save_name = os.path.join(saveOutputDir,'{ID}.pickle'.format(ID=pckl_save_name))
    saveVar = open(save_name, "wb")
    cPickle.dump(save_dict, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
    saveVar.close()
    print('\n\n%s saved...\n\n' % save_name)

print('\n\n%s saved...\n\n' % pckl_save_name)
#%% Plotting the STAs
roi_im = ROI_mod.get_masks_image(rois)
    
# Plotting ROIs and properties
pmc.plot_roi_masks(roi_im,mean_image,len(rois),
                   current_movie_ID.replace('//','-'),save_fig=True,
                   save_dir=figure_save_dir,alpha=0.4)
#fig1= ROI_mod.plot_STRFs(rois, f_w=None,number=None,cmap='coolwarm')
#fig1.suptitle(experiment_conditions['treatment'])
#f1_n = 'STRFs_%s' % (current_exp_ID)
#os.chdir(figure_save_dir)
#fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
#            transparent=False,dpi=300)
# os.chdir(summary_save_dir)
# f1_n = 'Summary_%s' % (current_movie_ID)
# fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
#             transparent=False,dpi=300)

properties_indiv_plots=['PD','CS','SNR','CSI','reliability']    

#ROI_mod.plot_individual_STRFs(rois,current_exp_ID.replace('//','-'),save_path=figure_save_dir)
ROI_mod.plot_individual_STRFs_with_edges(rois,current_exp_ID.replace('//','-'),save_path=figure_save_dir)

#'E:\\PhD\\Experiments\\2p\\T4T5_RFs_Mi1_glucl_overexpression\\processed\\20210315_jv_fly3\\Tseries-fly3-002\\20210315_jv_fly3-Tseries_fly3_002_ternaryWN_elavation_RF.pickle'
#load_path = open('E:\\PhD\\Experiments\\2p\\T4T5_RFs_Mi1_glucl_overexpression\\processed\\20210315_jv_fly3\\Tseries-fly3-002\\20210315_jv_fly3-Tseries_fly3_002_ternaryWN_elavation_RF.pickle', 'rb')
##workspace = cPickle.load(load_path)
#rois = workspace['final_rois']