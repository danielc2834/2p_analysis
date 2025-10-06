

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on thu july  815 2021

@author: burakgur
"""

from __future__ import division
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy
import sys
#code_path = r'D:\progresources\Calcium-imaging-analysis\general_lab_code'
code_path= r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis\helpers' #Juan desktop
sys.path.insert(0, code_path) 
import post_analysis_core as pac
import process_mov_core as pmc
import ROI_mod
import pandas as pd
#%% initialization

#initialDirectory = 'F:\\PhD\\Experiments\\2p\\'
initialDirectory = 'C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan

experiment = 'DATA_PSINA_Miri'
#'t4t5_shits_p9silencing'#'t4t5_freqtunning_8dir_Mi1 glucl overexpression'
#'t4t5_tm3glucl-overexpression_8dir'
#'t4t5_freqtunning_8dir_Mi1 glucl overexpression'
# 't4t5_tm3glucl-overexpression_8dir'
genotype_labels = ['positive','experimental','negative']#['experimental', 'positive','negative']

data_dir = initialDirectory+experiment+'\\'+'processed\\edges\\files'
results_save_dir = initialDirectory+experiment+'\\'+'processed\\results\\edges'

make_vector_plots=True
filter_polarities=True
plot_quantity_comparison=False
statistics=True
comparisons_to_make=[['experimental','positive'],['experimental','negative']]
Alternative_hypothesis=['two-sided','greater']
#auto_rois=False
#set quantities to plot and compare between treatments
quantity_toplot=['DSI_ON','DSI_OFF','max_response_ON','max_response_OFF']#,'evoked_vs_non_evoked_ratio_ON','evoked_vs_non_evoked_ratio_OFF']
                    #,'norm_null_dir_resp_OFF','norm_null_dir_resp_ON'] #TODO consider quantifying the response to opp-dir
#if depth filter is desired, use filter bin. the bins available depend on the experiment:
#filter_z_depth --- #takes in range values in the form of a string. if no filter is desired input is - [None] -
filter_z_depth = [30,45,60,75]#[30-35,50-55,None]#['45-50','30-35','60-65'] #[30-35,50-55]#['45-50','30-35','60-65',None]
filter_rotation_ofFly = [0,60]#None
drop_negatives=False
non_parametric=False
bin_for_count=None# '45-50'
filter_rois=[False,'CSI',0.3] # structure is [boolean to apply or not filter, property to use, treshold value ]
evoked_non_evoked_filter=[False,2] # Boolean to decide if applies, value of the filter follows
reliability_filter = False # filter based on the mean pairwise correlation between trials for the prefered direction epoch
Manual_reliability_filter = [True,0.4]
CS_filter=[True] #USE ONLY with auto ROIs
include_all=False #this overrides all the other filters and just plots ON and OFF responses for every ROI
#%%
 
datasets_to_load = os.listdir(data_dir)

#initialize variables
final_rois_all = []
flyIDs = []
flash_resps = []
ON_steps = []
ON_plateau = []
ON_int = []
flash_corr = []
genotypes = []

# make a pandas dataframe to store the data. this will later be used to make averages either per roi/ per fly/ category, etc

properties = ['DSI_ON','DSI_OFF','PD_ON','PD_OFF','CSI',\
            'CS','max_response_OFF','max_response_ON',\
            'FlyID','treatment','norm_null_dir_resp_ON',\
            'norm_null_dir_resp_OFF','categories','depth',\
            'depth_bin','sex','rotation']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
tfl_maps = []

#['positive', 'negative','experimental']

for dataset in datasets_to_load:

    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    #TODO extract minimum response and make it an attribute of ROIs!!
    
    #### filter ROIs by responsiveness
    #curr_rois=pac.find_responsive_ROIs(curr_rois)
    

    curr_rois=pmc.extract_null_dir_response(curr_rois) ### consider checking this out

    # TODO automatically extract the layer identity of an ROI?

    data_to_extract = ['DSI_ON','DSI_OFF','PD_ON','PD_OFF','CSI',\
         'max_response_OFF','max_response_ON',\
         'norm_null_dir_resp_ON', 'norm_null_dir_resp_OFF',\
         'reliability_PD_ON','reliability_PD_OFF']
    print(curr_rois[0].experiment_info['treatment'])
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    treatments = list(map(lambda roi : roi.experiment_info['treatment'], curr_rois))
    sex = list(map(lambda roi : roi.experiment_info['Sex'], curr_rois))
    depth= list(map(lambda roi : roi.experiment_info['z_depth'], curr_rois))
    depth_bin= list(map(lambda roi : roi.experiment_info['z_depth_bin'], curr_rois))
    FlyId= list(map(lambda roi : roi.experiment_info['FlyID'], curr_rois))
    treatment= list(map(lambda roi : roi.experiment_info['treatment'], curr_rois))
    categories= list(map(lambda roi : roi.category[0], curr_rois)) # in some cases the structure may be different, and this line may require an extra slicing step [0] and make require turning into list. 
    rotations = list(map(lambda roi : roi.experiment_info['rotation'], curr_rois))
    #### this is a temporary back compatibility patch ####
    try:
        Contrast_selectivity=list(map(lambda roi : roi.CS, curr_rois))
    except AttributeError:
        Contrast_selectivity=[]
        Contrast_selectivity_a=np.array(list(map(lambda roi : roi.ON_neuron, curr_rois)))
        Contrast_selectivity_b=np.array(list(map(lambda roi : roi.OFF_neuron, curr_rois)))
        for idx,entry in enumerate(Contrast_selectivity_a):
            if entry==True:
                Contrast_selectivity.append('ON')
            elif entry ==False and Contrast_selectivity_b[idx]==False:
                Contrast_selectivity.append(None)
            else:
                Contrast_selectivity.append('OFF')
        Contrast_selectivity=np.array(Contrast_selectivity)
    ### end of patch ###

    def_cat=[]
    # for cat in categories:
    #     def_cat.append(cat[0])
    # categories=np.array(def_cat)
    df_c = {}
    df_c['categories'] = categories
    df_c['treatment'] = treatments
    df_c['CS'] = Contrast_selectivity
    df_c['PD_ON'] = roi_data['PD_ON']
    df_c['PD_OFF'] = roi_data['PD_OFF']
    #df_c['ON_neuron'] = roi_data['ON_neuron']
    #df_c['OFF_neuron'] = roi_data['OFF_neuron']
    df_c['CSI'] = roi_data['CSI']
    #df_c['BF'] = roi_data['BF']
    df_c['DSI_ON'] = roi_data['DSI_ON']
    df_c['DSI_OFF'] = roi_data['DSI_OFF']
    df_c['max_response_OFF'] = roi_data['max_response_OFF']
    df_c['max_response_ON'] = roi_data['max_response_ON']
    df_c['norm_null_dir_resp_ON'] = roi_data['norm_null_dir_resp_ON']
    df_c['norm_null_dir_resp_OFF'] = roi_data['norm_null_dir_resp_OFF']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    #df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    df_c['sex'] = sex
    df_c['depth'] = depth
    df_c['rotations'] = rotations
    df_c['depth_bin'] = depth_bin
    df_c['FlyId'] = FlyId
    df_c['treatment'] = treatment
    #df_c['evoked_vs_non_evoked_ratio_ON']=df_c['treatment']'peak_to_peak_response_baseline_comparison_ON']
    #df_c['evoked_vs_non_evoked_ratio_OFF']=roi_data['peak_to_peak_response_baseline_comparison_OFF']
    df_c['reliability_OFF']=roi_data['reliability_PD_OFF']
    df_c['reliability_ON']=roi_data['reliability_PD_ON']


    # if we have 2 stimuli. reliability and SNR that matters is the one corresponding to the edges stimulus
    #df_c['reliability'] = list(map(lambda idx : roi_data['reliability'][idx][0], range(len(roi_data['reliability']))))
    #df_c['SNR'] = list(map(lambda idx : roi_data['SNR'][idx][0], range(len(roi_data['SNR']))))

    # for roi in curr_rois:
    #     roi.flynum=flyNum
    # flyNum = flyNum +1
    all_rois.append(curr_rois)
    df = pd.DataFrame.from_dict(df_c) 
    #rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))

## plot a histogram of the reliability values per treatment and one also for the vector lenght per treatment.

##monkey patch: eliminate spaces in treatment names

combined_df['treatment'] = combined_df['treatment'].str.strip()

reliability_treshold=pac.plot_variable_histogram_pertreatment(combined_df,'reliability_ON',results_save_dir,z_depth=filter_z_depth)
rel_tresh_copy=copy.deepcopy(reliability_treshold)
reliability_treshold=pac.plot_variable_histogram_pertreatment(combined_df,'reliability_OFF',results_save_dir,z_depth=filter_z_depth)

if Manual_reliability_filter[0]:
    reliability_treshold=Manual_reliability_filter[1]

pac.plot_variable_histogram_pertreatment(combined_df,'DSI_ON',results_save_dir,z_depth=filter_z_depth)
pac.plot_variable_histogram_pertreatment(combined_df,'DSI_OFF',results_save_dir,z_depth=filter_z_depth)



if filter_rois[0]==True:
    filter_category=filter_rois[1]
    treshold_val=filter_rois[2]
    combined_df=combined_df.loc[combined_df[filter_category]>treshold_val]
else:
    treshold_val=None

if evoked_non_evoked_filter[0]==True:
    ON_column=combined_df['evoked_vs_non_evoked_ratio_ON']>evoked_non_evoked_filter[1]
    OFF_column=combined_df['evoked_vs_non_evoked_ratio_OFF']>evoked_non_evoked_filter[1]
    combined_df['ON_responding']=ON_column
    combined_df['OFF_responding']=OFF_column

if reliability_filter==True or Manual_reliability_filter[0]==True:
    ON_column=combined_df['reliability_ON']>reliability_treshold
    OFF_column=combined_df['reliability_OFF']>reliability_treshold
    combined_df['ON_responding']=ON_column
    combined_df['OFF_responding']=OFF_column
else:
    reliability_treshold=None

if CS_filter[0]==True:
    ON_column=combined_df['CS']=='ON'
    OFF_column=combined_df['CS']=='OFF'
    combined_df['ON_responding']=ON_column
    combined_df['OFF_responding']=OFF_column

if include_all==True:
    column=np.ones(len(combined_df)).astype(bool)
    combined_df['ON_responding']=column
    combined_df['OFF_responding']=column
    reliability_treshold=None

# if there are background ROis, filter them out
combined_df=combined_df.loc[combined_df['categories']!=set(['BG'])]
combined_df=combined_df.loc[combined_df['categories']!=set(['noise_probe'])] # This is an ROI that should be noisy on purpose

#exchange label names if they are sets
# if  isinstance(combined_df['categories'][0],set):
#     for idx in combined_df.index:
#         combined_df.iloc[[idx]]['categories']=list(combined_df.iloc[[idx]]['categories'].values[0])[0]


# extract vector information and make polar plots out of it in an ROI to ROI basis
if make_vector_plots:
    for z_bin in filter_z_depth:    
        for rotation in filter_rotation_ofFly:
            pac.plot_roivectors(combined_df,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities,treshold=reliability_treshold)
            pac.plot_circular_histogram(combined_df,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold)
        'aaa'
    for z_bin in filter_z_depth:#
        for rotation in filter_rotation_ofFly:
            'aaa'
            pac.plot_roivectors_perfly(combined_df,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities)
if plot_quantity_comparison:
    #deprecated#pac.calculate_plot_on_off_ratio(combined_df,comparisons_to_make,rel_tresh_copy,results_save_dir,Alternative_hypothesis,z_bin=bin_for_count)
    'aaa'
    if drop_negatives:
        pac.plotandcompare_quantities_percategory(quantity_toplot,comparisons_to_make,combined_df,results_save_dir,drop_neg=True,statistics=statistics)
        pac.plotandcompare_quantities(quantity_toplot,combined_df,results_save_dir,drop_neg=True,non_parametric=non_parametric, filter=treshold_val,statistics=statistics)
        pac.plotandcompare_quantities(quantity_toplot,Alternative_hypothesis,comparisons_to_make,combined_df,results_save_dir,statistics=statistics,z_bin=bin_for_count,polarity_filter=filter_polarities,treshold=reliability_treshold)

    else:
#        pac.plotandcompare_quantities_percategory(quantity_toplot,combined_df,results_save_dir)
        pac.plotandcompare_quantities(quantity_toplot,Alternative_hypothesis,comparisons_to_make,combined_df,results_save_dir,statistics=statistics,z_bin=bin_for_count,polarity_filter=filter_polarities,treshold=reliability_treshold)
        
    # extra. just take a single depth bin to get a more reliable comparison of the roi counts 
    combined_df_binfiltered=combined_df.loc[combined_df['depth_bin']==bin_for_count]
    results_save_dir2=results_save_dir+'\\single_layer_count'
combined_df_binfiltered=combined_df.loc[combined_df['depth_bin']==bin_for_count]

#pac.calculate_plot_on_off_ratio(combined_df,results_save_dir,non_parametric=non_parametric,statistics=statistics)

#TODO separate plotting and normality variance testing into different functions
# %%
