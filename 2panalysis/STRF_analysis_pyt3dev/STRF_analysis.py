# script with a protocol for STRF analysis 
# use python 3 here

import os
from time import time
import cPickle
import STRF_utils as RF
import sys
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import NMF
code_path= r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis\helpers' #Juan desktop
sys.path.insert(0, code_path) 
import post_analysis_core as pac
import process_mov_core as pmc
import ROI_mod

#initialize 
initialDirectory = 'C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan

experiment = 'T4T5_STRF_glucla_rescues'
data_dir = initialDirectory+experiment+'\\'+'processed\\files'
results_save_dir = initialDirectory+experiment+'\\'+'processed\\results\\STRF'
plot_mapping_hist = True
treatment_names=['L5_rescue', 'control_heterozygous', 'Mi1_rescue', 'control_homocygous']
colors=[(195.0/256.0,170.0/256.0,109.0/256.0),(150.0/256.0,150.0/256.0,150.0/256.0),(89.0/256.0,125.0/256.0,191.0/256.0),(100.0/256.0,100.0/256.0,100.0/256.0)]
regularization = [0.0001,1]
# import rois
datasets_to_load = os.listdir(data_dir)
curr_rois = []
final_rois = []
mapping_rois = []
included_count = 0
excluded_count = 0
for dataset in datasets_to_load:

    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    
    if 'DriftingStripe' in dataset:
        mapping_rois.append(workspace['final_rois'])    
    else:
        curr_rois.append(workspace['final_rois'])

##
curr_rois=np.concatenate(curr_rois)
mapping_rois=np.concatenate(mapping_rois)
# produce some general plots to describe the roi extraction 

map_Df=pac.create_dataframe(mapping_rois,independent_vars=None,mapping=True)

if plot_mapping_hist:
    pac.plot_variable_histogram_pertreatment(map_Df,'reliability_ON',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'reliability_OFF',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'DSI_OFF',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'DSI_ON',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'CSI',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors,filter_CS=True)
    # check if any differences in DSI between treatments are due to the reliability differences (less reliable ROIs)
    filtered_map_df_ON = map_Df.loc[map_Df['CS'] == 'ON']
    filtered_map_df_ON = map_Df.loc[map_Df['reliability_ON'] >= 0.4]
    #pac.plot_variable_histogram_pertreatment(map_Df,'DSI_OFF',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors)
    #pac.plot_variable_histogram_pertreatment(filtered_map_df_ON,'DSI_ON',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors,treshold=0.4)
    #pac.plot_variable_histogram_pertreatment(filtered_map_df_ON,'CSI',results_save_dir,fit_Beta=[False,[]],treatment_names=treatment_names,colors=colors,treshold=0.4)

    filename=results_save_dir + 'ROI_distributions.pdf'
    pac.multipage(filename)
    plt.close('all')

# filter rois for reliability and contrast selectivity 
for roi in curr_rois:
    if roi.STRF_data['status'] == 'included':
        final_rois.append(roi)
        included_count+=1
    else:
        excluded_count+=1



# create distributions of Zscores and correlation prediction for null control and experimnetal data 
    # to do this, use control flies

STRF_Df = RF.create_dataframe_forSTRF(curr_rois,independent_vars=None,mapping=False)


STRF_DF_controls1 = STRF_Df.loc[STRF_Df['treatment'] == 'control_heterozygous']
STRF_DF_controls2 = STRF_Df.loc[STRF_Df['treatment'] == 'control_homocygous']

STRF_DF_controls = pd.concat([STRF_DF_controls1,STRF_DF_controls2],ignore_index=True)

#RF.calculate_STRF_prediction_distributions(STRF_DF_controls)


for ix,roi in enumerate(curr_rois):
    # calculate singular value decomposition for every roi calculated STRF
    # first fallten the strf

    # eliminate features beyond 30 deg distance from the absolute maximum of the RF

    for reg in regularization:

        masked_array = RF.apply_circular_mask(roi,20)
        
        components_xy, components_t = RF.svd_flattenedSTRF(roi,masked_array,components=5)

        # perform NMF on the negative and positive components separately

        H_xy, W_t = RF.segregated_NMf(roi,masked_array,regularization = reg)
        
        #plot the 3 components

        RF.plot_spatial_temporal_components(components_xy,components_t,results_save_dir,type='mixed_svd')

        RF.plot_NMF_segregated_components(H_xy,W_t,results_save_dir)

        #savefigs

        fly_path = os.path.join(results_save_dir,'components',roi.experiment_info['FlyID'])
        try:
            os.mkdir(fly_path)
        except:
            pass
        
        res_path = os.path.join(fly_path,'reg_%s' %(reg))
        try:
            os.mkdir(res_path)
        except:
            pass

        pmc.multipage(res_path + '\\components_%s.pdf' %(ix))


pepe
    # maybe calculate NNMF with 3 components
    # to do that split the negative and positive correlation areas


    # fit gaussians to the 2 or 3 components

    # RF.fit_component_gaussians(components_xy)

    # extract the temporal components (maybe the time aspects of the svd can help?)
    # ---- otherwise extract the filter from the point of maximum absolute value

    # do the stimulus * space * time convolution 

    # fit a softplus nonlinearity. 

    # evaluate predictions

    # Use the fitted gaussians 

    # fit gaussians to the first 2 or 3 components of the single value decomposition or the NNMF 





# create thresholds for Zscores/reliability,etc and trace prediction correlation 

 
# try to fit a gaussian for spatio temporal and for the spatial meanRF

for roi in final_rois:
    RF.fit_double_gaussian2D(roi, mode='spatio_temp')
    RF.fit_double_gaussian2D(roi, mode='spatial')

# compare sizes of RF acrosstreatment/luminance/polarity
# for this, use the extracted sizes of the receptive fieldsaccording to the gaussian fit
