# analysis of glucla tunning experiments


from __future__ import division
from re import sub
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
from matplotlib.gridspec import GridSpec

initialDirectory = 'C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan

experiment = 'T4T5_tunning_glucla_rescues'
data_dir = initialDirectory+experiment+'\\'+'processed\\files'
results_save_dir = initialDirectory+experiment+'\\'+'processed\\results\\'
treatment_names=['L5_rescue', 'control_heterozygous', 'Mi1_rescue', 'control_homocygous']
colors=[(195.0/256.0,170.0/256.0,109.0/256.0),(150.0/256.0,150.0/256.0,150.0/256.0),(89.0/256.0,125.0/256.0,191.0/256.0),(100.0/256.0,100.0/256.0,100.0/256.0)]
#colors_dict={'L5_rescue':(195.0/256.0,170.0/256.0,109.0/256.0),'control_heterozygous':(150.0/256.0,150.0/256.0,150.0/256.0),'Mi1_rescue':(89.0/256.0,125.0/256.0,191.0/256.0),'control_homocygous':(100.0/256.0,100.0/256.0,100.0/256.0)}
#what to do statistics on
comparisons_to_make=[['L5_rescue', 'control_heterozygous'],['Mi1_rescue', 'control_homocygous'],['Mi1_rescue', 'control_heterozygous'],['L5_rescue', 'control_homocygous']]
Alternative_hypothesis=['less','less','less','less']
independent_vars = ['frequency', 'luminance', 'contrast', 'velocity']
scale_traces=[13,10,10,10]
plot_interp=False
plot_hist=False
plot_tunning=True
# decide wheter to analize per layer or combining layers

variables_to_analize= []
# import pickle files. 

datasets_to_load = os.listdir(data_dir)

curr_rois=[]
mapping_rois=[]
for dataset in datasets_to_load:

    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    
    if 'mapping' in dataset:
        mapping_rois.append(workspace['final_rois'])
    else:       
        curr_rois.append(workspace['final_rois'])

# show the reliability, dirselindex and contrastselindex distribution for the mapping stimulus
mapping_rois=np.concatenate(mapping_rois)
map_Df=pac.create_dataframe(mapping_rois,independent_vars=None,mapping=True)
if plot_hist:
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
curr_rois=np.concatenate(curr_rois)

# create a dataframe with the ROI contents

dataF = pac.create_dataframe(curr_rois,independent_vars=independent_vars)
dataF=dataF[dataF['category']!='No_category']
dataF=dataF[dataF['category']!='LPB']
dataF=dataF[dataF['category']!='LPD']
pac.count_rois_flies_per_treatment(dataF,SaveData=True,SaveDir=results_save_dir)
pac.count_rois_flies_per_treatment(dataF,divide_by_category=False,SaveData=True,SaveDir=results_save_dir)

## use the entries containing interpolated traces to make a plot per
# experimental variable 
if plot_interp:
    for variab in np.unique(dataF['independent_var']):

        # initialize the figure
        if 'frequency' in variab:
            units= 'Hz'
        elif 'velocity' in variab: 
            units = 'deg/s'
            lim=[-1,7]
        elif 'contrast' in variab:
            units = 'michel contrast'
            lim=[-1,7]
        elif 'luminance' in variab:
            units = r'$\phi$/Cm$^2$' 
            lim=[-1,7]
            
        fig1,axes1=pac.create_subplots(len(dataF['ind var values'][0]))
        fig2,axes2=pac.create_subplots(len(dataF['ind var values'][0]))
        fig3,axes3=pac.create_subplots(len(dataF['ind var values'][0]))

        # create a frame with interpolated traces for every epoch type
        axes=[axes1,axes2,axes3]
        for ly,layer in enumerate(['LPA','LPC',None]):
            if layer is not None:
                sub_dataF=dataF.loc[dataF['category']==layer]
            else:
                sub_dataF=dataF
            for ix,epoch in enumerate(sub_dataF.loc[sub_dataF['independent_var']==variab]['interpolated_traces'].iloc[0].keys()):
                
                df_interp_traces,interpolated_time,var_values=pac.create_df_interp_traces(dataF, epoch, variab)
                epoch_value=var_values[ix]
                if variab=='luminance':
                    epoch_value=epoch_value*pac.convert_luminance_mWToflux(1.25)
                if ix==0:
                    check=False
                else:
                    check=True
                # calculate means across flies and across treatments in order to plot 
                pac.plot_interp_traces(interpolated_time,df_interp_traces,epoch_value,units,variab, axes[ly][ix],colors,treatment_names,lim,layer,check=check)
        fig1.suptitle('aligned traces %s %s'%(variab,'LPA'))
        fig2.suptitle('aligned traces %s %s'%(variab,'LPC'))
        fig3.suptitle('aligned traces %s all layers'%(variab))
    filename=results_save_dir + 'int_traces_allvariables.pdf'
    pac.multipage(filename)
    plt.close('all')

## use the entries containing maximum responses to make a plot per expr variable 


if plot_tunning:
    for variab in np.unique(dataF['independent_var']):
        # create a subset of the dataset
        subset_df = dataF[dataF['independent_var'] == variab]
        # get indep variable values
        variab_x_values=subset_df.iloc[0]['ind var values']

        # create plot scaffolds
        fig4 = plt.figure(figsize=(15, 10))
        gs = GridSpec(2,3)
        axes4 = []
        for i in range(3):
            ax = fig4.add_subplot(gs[0, i])
            axes4.append(ax)
        fig4.suptitle('%s tunning curve ' %(variab))
        #
        fig5 = plt.figure(figsize=(15, 10))
        axes5 = []
        for i in range(3):
            ax = fig5.add_subplot(gs[0, i])
            axes5.append(ax)
        fig5.suptitle('%s tunning curve ' %(variab))

        # initialize axes units, plotnames
       
        if 'frequency' in variab:
            units= 'Hz'
            axes_plots=[axes4,axes5]
            dependent_vars=['max_resp_all_epochs','freq_power_epochs']
        else:
            axes_plots=[axes4]
            dependent_vars=['max_resp_all_epochs']
        if 'velocity' in variab: 
            units = 'deg/s'        
        elif 'contrast' in variab:
            units = 'michelson contrast'
        elif 'luminance' in variab:
            units =  r'$\phi$/Cm$^2$' 
            variab_x_values=np.array(variab_x_values)*pac.convert_luminance_mWToflux(1.25) # measured screen intensity in the given units

        # plot results per layer
        for ly,layer in enumerate(['LPA','LPC',None]):
            #create df subset
            if layer is not None:
                subset_df_layer = subset_df.loc[subset_df['category'] == layer]
                
            else:
                subset_df_layer = subset_df
            # statistics

            # produce plots
            for plot_ax,depend_var in zip(axes_plots,dependent_vars):   
                
                if layer is None:
                    fly_means = pac.find_fly_means_from_dataframe(subset_df_layer,depend_var)
                    normality,variance_test=pac.test_statsassumptions_DF_columns(fly_means,comparisons_to_make,results_save_dir,variab)
                    pairwise_test=pac.multiple_pairwise_stats(fly_means,normality,variance_test,comparisons_to_make,variab,variab_x_values,Alternative_hypothesis,results_save_dir,depend_var)
                else:
                    fly_means = pac.find_fly_means_from_dataframe(subset_df_layer,depend_var)
                pac.plot_tuning_curve_w_error(fly_means,variab,depend_var,variab_x_values,layer,units,colors,treatment_names,plot_ax[ly])
            
            
            # if 'frequency' in variab:
            #     # if frequency analysis, plot both the highest response and 
            #     # the frequency component analysis
            #     fly_means = pac.find_fly_means_from_dataframe(subset_df_layer,'max_resp_all_epochs')
            #     normality,variance_test=pac.test_statsassumptions_DF_columns(fly_means,comparisons_to_make)
            #     pac.plot_tuning_curve_w_error(fly_means,variab,'max_resp_all_epochs',variab_x_values,layer,units,colors,treatment_names,axes4[ly])
            #     fly_means = pac.find_fly_means_from_dataframe(subset_df_layer,'freq_power_epochs')
            #     normality,variance_test=pac.test_statsassumptions_DF_columns(fly_means,comparisons_to_make)
            #     pac.plot_tuning_curve_w_error(fly_means,variab,'freq_power_epochs',variab_x_values,layer,units,colors,treatment_names,axes5[ly])
            
            # else:
            #     fly_means = pac.find_fly_means_from_dataframe(subset_df_layer,'max_resp_all_epochs')
            #     normality,variance_test=pac.test_statsassumptions_DF_columns(fly_means,comparisons_to_make)
            #     pac.plot_tuning_curve_w_error(fly_means,variab,'max_resp_all_epochs',variab_x_values,layer,units,colors,treatment_names,axes4[ly])
                

    filename=results_save_dir + 'tunning_curves_allvariables.pdf'
    pac.multipage(filename)
    plt.close('all')
    ## repeat the last step for the traces that are specific for frequency tunning stuff
    # perform relevant significance tests


