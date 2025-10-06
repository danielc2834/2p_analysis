#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:07:02 2020

@author: burakgur
"""

import numpy as np
import cPickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage, stats
import sys
code_path = r'D:\progresources\Calcium-imaging-analysis\general_lab_code'
sys.path.insert(0, code_path) 

import ROI_mod
import post_analysis_core as pac

# %% Setting the directories
initialDirectory = '/Volumes/Backup Plus/PhD_Archive/Data_ongoing/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '200210_T4T5_luminance','luminance_gratings')
summary_save_dir = os.path.join(initialDirectory,
                                'results/200302_luminance/T4T5')

# Plotting parameters
colors, _ = pac.run_matplotlib_params()
color = colors[5]
color_pair = colors[2:4]
roi_plots=False
#%%Juan initialization. Setting the directories
initialDirectory = 'E:\\PhD\\Experiments\\2p\\'
#all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
#results_save_dir = os.path.join(initialDirectory,
#                                'results/200310_GluClalpha_NI_cut_experiments')


# Load datasets and desired variables
exp_folder = 't4t5_freqtunning_Mi1_glucl_overexpression'

data_dir = initialDirectory+exp_folder+'\\'+'processed\\TF\\pickles'
results_save_dir = initialDirectory+exp_folder+'\\'+'processed\\FFF_data\\results'


# %% Load datasets and desired variables
exp_t = '200210_T4T5_gratings'
datasets_to_load = os.listdir(data_dir)

                    
properties = ['BF', 'PD', 'SNR','Reliab','depth','DSI']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
tfl_maps = []
# Initialize variables
flyNum = 0
filter_dict={'reliability':0.2}
frequencies=[0.2,0.5,1,2,5]
treatments=['positive', 'negative','experimental']

for dataset in datasets_to_load:

    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    
    #if 'lightBG' in curr_rois[0].stim_name:
    #    curr_stim_type = 'dark background'
    #    continue
    # if 'darkBG' in curr_rois[0].stim_name:
    #     curr_stim_type = 'bright background'
    #     continue
    
    # Reliability thresholding
    #curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.2})

    
    #tfl_maps.append((np.array(map(lambda roi: np.array(roi.tfl_map),curr_rois))))
    data_to_extract = ['SNR','reliability','DSI','CS','PD','CSI','category','BF']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    treatments = list(map(lambda roi : roi.experiment_info['treatment'], curr_rois))

    #TODO include depth in your analysis. now it's not important

    df_c = {}
    df_c['treatment'] = treatments
    df_c['PD'] = roi_data['PD']
    df_c['category'] = roi_data['category']
    df_c['CSI'] = roi_data['CSI']
    df_c['CS'] = roi_data['CS']
    df_c['SNR'] = roi_data['SNR']
    df_c['BF'] = roi_data['BF']
    # we have 2 stimuli. reliability that matters is the one corresponding to the different frequency gratings
    df_c['Reliab'] = list(map(lambda idx : roi_data['reliability'][idx][1], range(len(roi_data['reliability']))))
    df_c['DSI'] = roi_data['DSI']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    for roi in curr_rois:
        roi.flynum=flyNum
    flyNum = flyNum +1
    all_rois.append(curr_rois)
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))


# %%  plot the average frequency tunning. 

# maybe plot it based on prefered frequency also
# TODO include number of flies and rois in plot

amplitudes=filtered_rois[0].fft_amp_norm.copy()
amplitudes.update({'treatment':filtered_rois[0].experiment_info['treatment']})
amplitudes.update({'flynum':filtered_rois[0].flynum})
amplitudes.update({'CS':filtered_rois[0].CS})
amplitudes_dataframe=pd.DataFrame(columns=amplitudes.keys())


for roi in filtered_rois:
    amplitudes=roi.fft_amp_norm.copy()
    amplitudes.update({'treatment':roi.experiment_info['treatment']})
    amplitudes.update({'flynum':roi.flynum})
    amplitudes.update({'CS':roi.CS})
    
    amplitudes_dataframe=amplitudes_dataframe.append(amplitudes,ignore_index=True)

amplitudes_dataframe=amplitudes_dataframe.set_index(['flynum','treatment','CS'])
mean_df=amplitudes_dataframe.groupby(['flynum','treatment','CS']).mean()
sem_df=mean_df=amplitudes_dataframe.groupby(['flynum','treatment','CS']).sem()

all_rois=np.concatenate(all_rois)
#mean_tfls_fly = np.array(map(lambda tfl : tfl.mean(axis=0),tfl_maps))

ROI_num = len(all_rois)
#flyNum
#%%% filter ROIs based on reliability

filtered_df=combined_df[combined_df['Reliab']>0.2]
filtered_ROIlist=all_rois[combined_df['Reliab']>0.2]
sanitycheck=filtered_df['SNR']==roi_data = ROI_mod.data_to_list(curr_rois, ['SNR'])['SNR']
ssssanitycheck
filtered_rois=ROI_mod.analyze_gratings_general(filtered_ROIlist)




#%% plot avg TFL maps
# plt.close("all")
# pac.run_matplotlib_params()


# fig = plt.figure(figsize = (5,5))
# plt.title('TFL map {st} - {f}({roi})'.format(st=curr_stim_type,
#                                              f = len(mean_tfls_fly), 
#                                              roi=ROI_num))
          
# ax=sns.heatmap(mean_tfls_fly.mean(axis=0), cmap='coolwarm',center=0,
#                xticklabels=np.array(all_rois[0].tfl_map.columns.levels[1]).astype(float),
#                yticklabels=np.array(all_rois[0].tfl_map.index),
#                cbar_kws={'label': '$\Delta F/F$'})
# ax.invert_yaxis()
# ax.set_xlabel('Luminance')
# ax.set_ylabel('Hz')
# # Saving figure
# save_name = '{exp_t}_TFL_{st}'.format(exp_t=exp_t,
#                                            st=curr_stim_type)
# os.chdir(summary_save_dir)
# plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%% Plot at 1 Hz
# plt.close("all")
# fig = plt.figure(figsize = (5,5))
# plt.title('Tunings {st} - {f}({roi})'.format(st=curr_stim_type,
#                                                   f = flyNum,
#                                                   roi=ROI_num))

# luminances = np.array(all_rois[0].tfl_map.columns.levels[1]).astype(float)
# freqs = np.array(all_rois[0].tfl_map.index)
# mean = mean_tfls_fly.mean(axis=0)[3,:]
# error = mean_tfls_fly.std(axis=0)[3,:]/np.sqrt(flyNum)
# ub = mean + error
# lb = mean - error
# plt.plot(luminances,mean,'o-',lw=3,alpha=.8,color=[0,0,0],label='1Hz tuning')
# plt.fill_between(luminances, ub, lb,color=[0,0,0], alpha=.2)
# plt.xlabel('Luminance')
# plt.ylabel('dF/F')
# plt.ylim((0, 1))
# plt.legend()

# save_name = '{exp_t}_1Hztuning_{st}'.format(exp_t=exp_t,
#                                            st=curr_stim_type)
# os.chdir(summary_save_dir)
# plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)



# for fly in amplitudes_dataframe.index:
#     subset_df=




# plt.close('all')
# fig, ax =plt.subplots(nrows=2,ncols=1,figsize=(10,15),sharey=True)
# polarities=np.array(['ON','OFF'])
# colors=['slategrey','red','forestgreen']
# treatments=['positive', 'negative','experimental']
# for idx,treatment in enumerate(treatments):
#     if treatment=='experimental':
#         a='a'
#     subset_mean=mean_df.reset_index()
#     subset_mean=subset_mean[subset_mean['treatment']==treatment]
#     color=colors[idx]
#     #subset_sem=sem_df[mean_df['treatment']==treatment]
#     for index in range(2):
#         values_to_plot_mean=subset_mean[subset_mean['CS']==polarities[index]]
#         x_axis=values_to_plot_mean.keys()[3:]
#         values_to_plot_mean=values_to_plot_mean[values_to_plot_mean.keys()[3:]].to_numpy()
#         max_values=np.max(values_to_plot_mean,axis=1)
#         max_values=max_values[:,np.newaxis]
#         values_to_plot=np.divide(values_to_plot_mean,max_values)
#         errors=stats.sem(values_to_plot,axis=0)
#         values_to_plot_mean=np.mean(values_to_plot,axis=0)
#         sort=np.array(x_axis).argsort()

#         ax[index].plot(x_axis[sort],values_to_plot_mean[sort],'--',label='{freq}'.format(freq=treatments),color=color)
#         ax[index].errorbar(x_axis[sort],values_to_plot_mean[sort],yerr=errors[sort],color=color)
#         ax[index].set_xscale('log')
#         ax[index].set_ylim([0,1.2])
#         ax[index].set_xlim([0.1,6])
#         ax[index].set_ylabel('norm_power')
#         ax[index].set_title('%s_frequency_tunning' %(polarities[index]))
# plt.savefig('freqtuning.png')

# #%% Plot all  freqs
# plt.close("all")
# fig = plt.figure(figsize = (5,5))
# plt.title('Tunings {st} - {f}({roi})'.format(st=curr_stim_type,
#                                                   f = flyNum,
#                                                   roi=ROI_num))

# #luminances = np.array(all_rois[0].tfl_map.columns.levels[1]).astype(float)
# freqs = np.array(all_rois[0].tfl_map.index)
# for idx, freq in enumerate(freqs):
#     mean = mean_tfls_fly.mean(axis=0)[idx,:]
#     error = mean_tfls_fly.std(axis=0)[idx,:]/np.sqrt(flyNum)
#     ub = mean + error
#     lb = mean - error
#     plt.plot(luminances,mean,'o-',lw=3,alpha=.8,label='{freq}'.format(freq=freq))
#     plt.fill_between(luminances, ub, lb, alpha=.2)
#     plt.xlabel('Luminance')
#     plt.ylabel('dF/F')
#     # plt.ylim((0, 1))
#     plt.legend()

# save_name = '{exp_t}_AllFreqsTuning_{st}'.format(exp_t=exp_t,
#                                            st=curr_stim_type)
# os.chdir(summary_save_dir)
# plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)


# #%% Plot ON OFF
# plt.close("all")
# colorss,_ = pac.run_matplotlib_params()
# colors = [colorss[-1],colorss[-2]]

# fig = plt.figure(figsize = (5,5))
# plt.title('1 Hz tunings {st}'.format(st=curr_stim_type))
# cs_s = ['OFF', 'ON']
# conc_tfls = np.concatenate(tfl_maps)
# for idx, cs in enumerate(cs_s):
    
    
#     curr_mask = combined_df['CS'] == cs
#     curr_roi_num = (combined_df['CS'] == cs).astype(int).sum()
#     label = '{cs} neurons - {roi}rois'.format(cs=cs,roi=curr_roi_num)
    
#     curr_tfls = conc_tfls[curr_mask]
#     mean = curr_tfls.mean(axis=0)[3,:]
#     error = curr_tfls.std(axis=0)[3,:]/np.sqrt(curr_roi_num)
#     ub = mean + error
#     lb = mean - error
#     plt.plot(luminances,mean,'o-',lw=3,alpha=.8,color=colors[idx],label=label)
#     plt.fill_between(luminances, ub, lb,color=colors[idx], alpha=.2)
#     plt.xlabel('Luminance')
#     plt.ylabel('dF/F')
#     plt.ylim((0, 1))
#     plt.legend()
    
# save_name = '{exp_t}_1Hztuning_{st}_T45_separated'.format(exp_t=exp_t,
#                                            st=curr_stim_type)
# os.chdir(summary_save_dir)
# plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)