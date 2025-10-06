#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:20:14 2020

@author: burakgur
"""

#%% Package
from __future__ import division
from random import sample
#from statistics import mean
from tkinter import OFF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mutual_info_score
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import itertools
from scipy import stats
from scipy.signal import correlate
import cmath
from scipy.stats import beta
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
import copy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import re

#%% Functions
def run_matplotlib_params():
    plt.style.use('default')
    plt.style.use('seaborn-talk')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)    
    plt.rcParams["axes.titlesize"] = 'medium'
    plt.rcParams["axes.labelsize"] = 'small'
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams["legend.fontsize"] = 'small'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["figure.titleweight"] = 'bold'
    plt.rcParams["figure.titlesize"] = 'medium'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['legend.loc'] = 'upper right'
    
    c_dict = {}
    c_dict['dark_gray'] = np.array([77,77,77]).astype(float)/255
    c_dict['light_gray'] = np.array([186,186,186]).astype(float)/255
    c_dict['green1'] = np.array([102,166,30]).astype(float)/255
    c_dict['green2']=np.array([179,226,205]).astype(float)/255
    c_dict['green3'] = np.array([27,158,119]).astype(float)/255
    c_dict['orange']  = np.array([201,102,47]).astype(float)/255
    c_dict['red']  = np.array([228,26,28]).astype(float)/255
    c_dict['magenta']  = np.array([231,41,138]).astype(float)/255
    c_dict['purple']  = np.array([117,112,179]).astype(float)/255
    c_dict['yellow'] = np.array([255,255,51]).astype(float)/255
    c_dict['brown'] = np.array([166,86,40]).astype(float)/255
    
    c_dict['L3_'] = np.array([102,166,30]).astype(float)/255 # Dark2 Green
    c_dict['Tm9'] = np.array([27,158,119]).astype(float)/255 # Dark2 Weird green
    c_dict['L1_'] = np.array([230,171,2]).astype(float)/255 # Dark2 Yellow
    c_dict['L2_'] = np.array([55,126,184]).astype(float)/255 # blue
    c_dict['Mi1'] = np.array([166,118,29]).astype(float)/255 # Dark2 dark yellow
    c_dict['Mi4'] = np.array([217,95,2]).astype(float)/255 # Dark2 orange
    c_dict['Tm3'] = np.array([231,41,138]).astype(float)/255 # Dark2 magenta
    
    

    
    c = []
    c.append(c_dict['dark_gray'])
    c.append(c_dict['light_gray'])
    c.append(c_dict['green1']) # Green
    c.append(c_dict['orange']) # Orange
    c.append(c_dict['red']) # Red
    c.append(c_dict['magenta']) # magenta
    c.append(c_dict['purple'])# purple
    c.append(c_dict['green2']) # Green
    c.append(c_dict['yellow']) # Yellow
    c.append(c_dict['brown']) # Brown
    
    
    return c, c_dict


def compute_over_samples_groups(data = None, group_ids= None, error ='std',
                                   experiment_ids = None):
    """ 
    
    Computes averages and std or SEM of a given dataset over samples and
    groups
    """
    # Input check
    if (data is None):
        raise TypeError('Data missing.')
    elif (group_ids is None):
        raise TypeError('Sample IDs are missing.')
    elif (experiment_ids is None):
        raise TypeError('Experiment IDs are missing.')
    
    # Type check and conversion to numpy arrays
    if (type(data) is list):
        data = np.array(data)
    if (type(group_ids) is list):
        group_ids = np.array(group_ids)
    if (type(experiment_ids) is list):
        experiment_ids = np.array(experiment_ids)
    
    
    data_dict = {}
    data_dict['experiment_ids'] = {}
    
    unique_experiment_ids = np.unique(experiment_ids)
    for exp_id in unique_experiment_ids:
        data_dict['experiment_ids'][exp_id] = {}
        data_dict['experiment_ids'][exp_id]['over_samples_means'] = []
        data_dict['experiment_ids'][exp_id]['over_samples_errors'] = np.array([])
        data_dict['experiment_ids'][exp_id]['uniq_group_ids'] = np.array([])
        
        data_dict['experiment_ids'][exp_id]['all_samples'] = \
            data[experiment_ids == exp_id]
        
        curr_groups = group_ids[np.argwhere(experiment_ids == exp_id)]
        
        for group in np.unique(curr_groups):
            curr_mask = (group == group_ids)
            curr_data = np.nanmean(data[curr_mask],axis=0)
            if error == 'std':
                err = np.nanstd(data[curr_mask],axis=0)
            elif error == 'SEM':
                err = \
                    np.nanstd(data[curr_mask],axis=0) / np.sqrt(np.shape(data[curr_mask])[0])
            data_dict['experiment_ids'][exp_id]['over_samples_means'].append(\
                 curr_data)
            data_dict['experiment_ids'][exp_id]['over_samples_errors'] = \
                np.append(data_dict['experiment_ids'][exp_id]['over_samples_errors'],
                          err)
            data_dict['experiment_ids'][exp_id]['uniq_group_ids'] = \
                np.append(data_dict['experiment_ids'][exp_id]['uniq_group_ids'],
                          group)
        data_dict['experiment_ids'][exp_id]['over_groups_mean'] = \
            np.nanmean(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0)
        if error == 'std':
            err = \
                np.nanstd(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0)
        elif error == 'SEM':
            err = \
                np.nanstd(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0) / \
                    np.sqrt(np.shape(data_dict['experiment_ids'][exp_id]['over_samples_means'])[0])
        data_dict['experiment_ids'][exp_id]['over_groups_error'] =err
            
    return data_dict
        
def bar_bg(all_samples, x, color='k', scat_s =7,ax=None, yerr=None, 
           errtype='std',alpha = .6,width=0.8,label=None):
    
    """ Nice bar plot """
    if yerr is None:
        if errtype == 'std':
            yerr = np.std(all_samples)
        elif errtype == 'SEM':
            yerr = np.std(all_samples) / np.sqrt((len(all_samples)))
            
    if ax is None:
        asd =1
    else:
        ax.bar(x, np.mean(all_samples), color=color,alpha=alpha,
               width=width,label=label)
        x_noise = np.random.normal(size=len(all_samples))
        x_noise = (x_noise-min(x_noise))/(max(x_noise)-min(x_noise))
        x_noise = x_noise/2
        scatt_x = np.zeros(np.shape(all_samples)) + x + x_noise
        ax.scatter(scatt_x,all_samples,color=color,s=scat_s)
        
        
        markers, caps, bars = ax.errorbar(x, np.mean(all_samples),
                                          yerr=yerr,fmt='.', 
                                          ecolor='black',capsize=0)
        markers.set_alpha(0)
        # loop through bars and caps and set the alpha value
        [bar.set_alpha(alpha) for bar in bars]
    return ax


def create_dataframe(curr_rois,independent_vars=None,mapping=False):
    data = []

    # Define the possible values of independent_var
    

    # Loop through the roi objects
    for roi in curr_rois:
        # Check if the independent_var is one of the possible values
        if independent_vars is not None and mapping==False:
            if getattr(roi, 'independent_var', np.nan) in independent_vars:
                # Special case for 'frequency'
                if roi.independent_var == 'frequency':
                    # Check for the two types of frequency: 'square' and other
                    if 'square' in roi.stim_info['stim_name']:
                        # Handle the 'square' case
                        roi.independent_var='frequency_square'
                        pass  # Code for handling 'square' case
                    else:
                        # Handle the other case
                        roi.independent_var='frequency_sin'
                        pass  # Code for handling other case
        elif mapping==False:
            continue
        elif independent_vars is not None and mapping==True:
            continue

        # Create a dictionary with the relevant attributes
        entry = {
            'uniq_id': roi.uniq_id,
            'independent_var': getattr(roi, 'independent_var', np.nan),
            'interpolated_traces': getattr(roi, 'interpolated_traces_epochs', np.nan),
            'interpolated_time': getattr(roi, 'interpolated_time', np.nan),
            'max_resp_all_epochs': getattr(roi, 'max_resp_all_epochs', np.nan),
            'max_resp_all_epoch_on':getattr(roi, 'max_resp_all_epochs_ON', np.nan),
            'freq_power_epochs': getattr(roi, 'freq_power_epochs', np.nan),  # Use np.nan if attribute does not exist
            'FlyId': roi.experiment_info['FlyID'],  # Include FlyId attribute
            'ind var values': getattr(roi, 'independent_var_vals', np.nan),  # Include independent_var_vals attribute
            'treatment': roi.experiment_info['treatment'],
            'category': roi.category[0],
            'reliability_ON': getattr(roi, 'reliability_PD_ON', np.nan),
            'reliability_OFF': getattr(roi, 'reliability_PD_OFF', np.nan),
            'DSI_ON': getattr(roi, 'DSI_ON', np.nan),
            'DSI_OFF': getattr(roi, 'DSI_OFF', np.nan),
            'CS': getattr(roi, 'CS', np.nan),
            'CSI': getattr(roi, 'CSI',np.nan)
        }

        # Append the dictionary to the data list
        data.append(entry)

    # Convert the data list to a pandas DataFrame
    df = pd.DataFrame(data)

    return df            
            
def apply_threshold_df(threshold_dict, df):
    
    if threshold_dict is None:
        print('No threshold used.')
        return df
    
    pass_bool = np.ones((1,len(df)))
    
    for key, value in threshold_dict.items():

        pass_bool = pass_bool * np.array((df[key] > value))
        
    threshold_df = df[pass_bool.astype(bool)[0]]
   
    return threshold_df



def calc_MI(x, y, bins):
    """ Calculating MI """
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    
    # Uses natural log to express information so you can divide by log2 to
    # get bits
    
    return mi/np.log(2)

def calculate_mean_vector(dirs,lengths,alternative_denominator=None):
    # '''Calculate a mean vector, given a list of directions and lengths 
    #     '''
    
    dirs = np.radians(dirs)
    lens=lengths
    if alternative_denominator!=None:
        denom=alternative_denominator
    else:
        denom=len(lengths)
    # exp_values=np.exp(dirs*1j)
    
    # mean_length=np.abs(np.sum(lens*exp_values)/ np.sum(lens))
    # mean_dir=np.angle(np.sum(lens*exp_values)/ np.sum(lens), deg=True)

    # if mean_dir<0:
    #         mean_dir= 360+mean_dir

    x = np.sum(lens*np.cos(dirs))/denom
    y = np.sum(lens*np.sin(dirs))/denom

    mean_length=np.sqrt(x**2+y**2) 
    mean_dir=np.rad2deg(np.arccos(x/mean_length))


    if x<0 and y<0:
        mean_dir=360-mean_dir
    elif x>0 and y<0:
        mean_dir=360-mean_dir

    return mean_length, mean_dir

def plot_roivectors(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=True,treshold=None):
    '''makes quiver polar plots from a dataframe containing categories, treatments, and 
       contrast selectivity information. The output is a polarplot pertreatment and per polarity
       if there are 2 treatments at the end you get 4 plots'''
    
    #filter z_layers if necesary
    if z_bin is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['depth_bin']==z_bin]
        #TODO automatize depth_bin selection
    if rotation is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['rotations']==rotation]


    #get mean vector per category (lobula plate layer)
        # get the mean vector and plot it for every layer of the lobplate 
    
    #check that the categories are not more than 4 or less than 4

    
    cats_to_exclude=[]
    check=0
    #if len(np.unique(roi_dataframe['categories']))!=4:
    #    raise Exception ('labels_are wrong check')
    for cats in np.unique(roi_dataframe['categories'].values):
        if ('_a' in cats) or ('LPA' in cats) or ('lpa'in cats):
            check+=1
        elif ('_b' in cats) or ('LPB' in cats) or ('lpb'in cats):
            check+=1
        elif ('LPC' in cats) or ('lpc'in cats):
            check+=1
        elif ('_d' in cats) or ('LPD' in cats) or ('lpd'in cats):
            check+=1
        else:
            cats_to_exclude.append(cats)
    
    if check!=4 :
        raise Exception ('cat labels_are wrong check')
    
    temp_df=roi_dataframe.copy(deep=True)
    for cats in cats_to_exclude:
        temp_df=temp_df.loc[temp_df['categories']!=cats]

    mean_vector_df={'treatment':[],'layer':[],'direction':[],'length':[],'polarity':[]}
       
    
    for treatment in np.unique(temp_df['treatment']):        
        for pol in ['ON','OFF']:            
            for layer in np.unique(temp_df['categories']):
                local_df=temp_df.loc[temp_df['treatment']==treatment]
                local_df=local_df.loc[local_df['categories']==layer]

                local_df[pol+'_responding'] #this is more relevant for manual ROIs where an roi can respond to both ON and OFF
                local_df=local_df.loc[local_df[pol+'_responding']==True] 


                input_str1='DSI_'+ pol
                input_str2='PD_' + pol                              
                in_angles=np.array(local_df[input_str2])
                in_lengths=np.array(local_df[input_str1])
                # if layer=='LPD':
                #     'aaa'
                mean_length,mean_angle=calculate_mean_vector(in_angles,in_lengths)
                                               
                mean_vector_df['treatment'].append(treatment)
                mean_vector_df['polarity'].append(pol)
                mean_vector_df['layer'].append(layer)
                mean_vector_df['direction'].append(mean_angle)
                mean_vector_df['length'].append(mean_length)
    mean_vector_df=pd.DataFrame.from_dict(mean_vector_df)

    #count number of treatments and categories. also extract their values
    treatment_number=len(np.unique(temp_df['treatment']))
    treatments=np.unique(temp_df['treatment'])
    category_number=len(np.unique(temp_df['categories']))
    categories=np.unique(temp_df['categories'][temp_df['categories']!='No_category'])

    if type(np.array(temp_df['categories'])[0])==set:
        for idx in temp_df['categories'].index:
            if idx%100==0:
                print('%s // %s' %(idx,roi_dataframe['categories'].index[-1]))
            temp_df['categories'][idx]=list(roi_dataframe['categories'][idx])[0]

    temp_df=temp_df.loc[roi_dataframe['categories']!='BG']    

    # turn string entries in dataframe to lowercase
    #temp_df['treatment']=temp_df['treatment'].str.lower()
    #temp_df['categories']=temp_df['categories'].str.upper()

    
    #create plot scaffold
    fig1=plt.figure(figsize=(40,12))
    gs = GridSpec(2, treatment_number)
    treatment_polarity_pairs=itertools.product([0,1],range(treatment_number))
    plt.style.use('seaborn')
    for pair in treatment_polarity_pairs:
        polarity=['ON','OFF'][pair[0]]
        DSI_name=['DSI_ON','DSI_OFF'][pair[0]]
        PD_name=['PD_ON','PD_OFF'][pair[0]]
        treatment= treatments[pair[1]]
        title = polarity[0:2] + ' ' + treatment
        if filter_polarity==True:
            polarity_subset=temp_df.loc[temp_df['CS']==polarity]
        else: # if a clustering algorithm was not used on the data, for each epoch only use clusters that are responsive
              # to the corresponding epoch
            if polarity=='ON':
                polarity_subset=temp_df.loc[temp_df['ON_responding']==True]
            elif polarity=='OFF':   
                polarity_subset=temp_df.loc[temp_df['OFF_responding']==True] 
        treatment_pol_subset=polarity_subset.loc[polarity_subset['treatment']==treatment]
        ax1=fig1.add_subplot(gs[pair[0],pair[1]],polar=True)
        number_of_flies=len(np.unique(treatment_pol_subset['FlyId']))
        number_of_treatment_flies=len(np.unique(temp_df.loc[temp_df['treatment']==treatment]['FlyId']))
        number_of_rois=len(treatment_pol_subset.index)
        for idx in range(len(treatment_pol_subset.index)):
            magnitude= treatment_pol_subset.iloc[idx,:][DSI_name]
            direction= treatment_pol_subset.iloc[idx,:][PD_name]
            category= treatment_pol_subset.iloc[idx,:]['categories']
            if ('_a' in category) or ('LPA' in category) or ('lpa'in category):
                color='tab:green'
            elif ('_b' in category) or ('LPB' in category) or ('lpb'in category):
                color='tab:blue'
            elif ('_c' in category) or ('LPC' in category) or ('lpc'in category):
                color='tab:red'
            elif ('_d' in category) or ('LPD' in category) or ('lpd'in category):
                color='gold'
            else:
                color='grey'
            title= treatment + ' total n_: %s pol n: %s %s rois ' %(number_of_treatment_flies,number_of_flies,number_of_rois)
            ax1.set_ylim(0,0.7)
            ax1.set_yticks([0.2,0.4,0.7])
            ax1.set_title(title,pad=15,fontweight='bold',fontsize=13)
            ax1.set_ylabel(polarity[0:3],labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
            ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
            scale=1,angles="xy",scale_units='xy',alpha=0.7)    
        # plot mean
        mean_df_subset=mean_vector_df.loc[mean_vector_df['polarity']==polarity]
        mean_df_subset=mean_df_subset.loc[mean_df_subset['treatment']==treatment]
        local_lenghts=np.array(mean_df_subset['length'])
        local_dirs=np.array(mean_df_subset['direction'])
        ax1.quiver(0,0, np.radians(local_dirs),local_lenghts,color='black',
            scale=1,angles="xy",scale_units='xy',alpha=1) 
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                        right=0.9, 
                        top=0.93, 
                        wspace=0.015, 
                        hspace=0.36)

    #for layer in np.unique(treatment_pol_subset.loc('_a' in treatment_pol_subset['categories']) or ('LPA' in category) or ('lpa'in category))

    #save the plot 
    if z_bin is not None:
        base_str= savedir +'\\results vector plot Z%s rel_tresh %.3f rotation %s' %(z_bin,treshold,rotation)
    else:
        base_str= savedir + '\\results vector plot allZ rel_tresh %s _rotation %s'% (treshold,rotation) 
    save_str1=base_str +'.png'
    #plt.title('vector plots reliability_treshold: %.2f'% treshold)
    plt.savefig(save_str1)
    save_str2=base_str +'.pdf'
    plt.savefig(save_str2)
    plt.close('all')

def plot_circular_histogram(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=False,treshold=None):
    # calculate the total number of ROIs per treatment!
    
    #filter Z and rotation
    if z_bin is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['depth_bin']==z_bin]
    if rotation is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['rotations']==rotation]
    treatment_layer_len={}
    treatment_layer_passed_on={}    
    treatment_layer_passed_off={}
    for treatment in np.unique(roi_dataframe['treatment']):
        treatment_layer_len[treatment]={}
        treatment_layer_passed_on[treatment]={}
        treatment_layer_passed_off[treatment]={}
        for layer in np.unique(roi_dataframe['categories']):
            treatment_len_temp=roi_dataframe.loc[roi_dataframe['treatment']==treatment]
            treatment_len_temp=treatment_len_temp.loc[treatment_len_temp['categories']==layer].loc[treatment_len_temp['categories']==layer]
            layer_temp={layer:len(treatment_len_temp.loc[treatment_len_temp['categories']==layer])}
            layer_temp_passed_on={layer:len(treatment_len_temp.loc[treatment_len_temp['ON_responding']==True])}
            layer_temp_passed_off={layer:len(treatment_len_temp.loc[treatment_len_temp['OFF_responding']==True])}
            treatment_layer_passed_on[treatment].update(layer_temp_passed_on)
            treatment_layer_passed_off[treatment].update(layer_temp_passed_off)
            treatment_layer_len[treatment].update(layer_temp)

    
    #create fgure scaffold
    fig1=plt.figure(figsize=(40,12))
    gs = GridSpec(2, len(np.unique(roi_dataframe['treatment'])))
    treatment_num=len(np.unique(roi_dataframe['treatment']))
    treatments=np.unique(roi_dataframe['treatment'])
    treatment_polarity_pairs=itertools.product([0,1],range(treatment_num))
    plt.style.use('seaborn')
    max_weight=[]
    for pair in treatment_polarity_pairs:
        polarity=['ON','OFF'][pair[0]]
        treatment=treatments[pair[1]]
        if filter_polarity:
            subset_temp=roi_dataframe.loc[roi_dataframe[polarity+'_responding']==True]
        else:
            subset_temp=roi_dataframe
        subset_temp=subset_temp.loc[subset_temp['treatment']==treatment]
        for layer in np.unique(subset_temp['categories']):
            if layer=='LPA':
                color='tab:green'
            elif layer=='LPB':
                color='tab:blue'
            elif layer=='LPC':
                color='tab:red'
            elif layer=='LPD':
                color='gold'
            else: # if roi is not part of any category, then exclude
                continue
            # create subset with specific layer
            subset_layer=subset_temp.loc[subset_temp['categories']==layer]
            directions= np.array(subset_layer['PD_'+ polarity])
            # create info for plot
            hist_directions=np.histogram(np.radians(directions),np.radians(np.arange(0,366,6)),density=True)
            hist_directions_weights=hist_directions[0]#/treatment_layer_len[treatment][layer]
            #hist_directions_counts=hist_directions[0]/treatment_layer_len[treatment][layer]
            #hist_directions_counts=hist_directions[0]*hist_directions_weights
            ax=fig1.add_subplot(gs[pair[0],pair[1]],polar=True)
            #hist_directions[0]=hist_directions[0]/treatment_len[treatment]
            
            summed_directions = np.sum(hist_directions_weights*(np.exp(np.radians(np.arange(3,363,6))*1j)))
            max_weight.append(np.max(hist_directions_weights))
            mean_angle=np.angle(summed_directions)
            vector_lenght=np.abs(summed_directions)/np.sum(hist_directions_weights)
            
            #lens=np.repeat(len(directions))

            mean_len,mean_dir=calculate_mean_vector(np.arange(3,363,6),hist_directions_weights)
            #scale the mean vector by the proportion of responding ROIS/total ROIs in layer
            
            
            
            ax.bar(np.radians(np.arange(3,363,6)),hist_directions_weights,color=color,alpha=0.7,width=np.radians(6))
            ax.quiver(0,0, mean_angle,np.abs(vector_lenght),zorder = 5,
                        scale=1,angles="xy",scale_units='xy',alpha=1) 
            #histogram_plot[0]=histogram_plot[0]/treatment_len[treatment]
            
            # compute mean direction

            #summed_directions = np.sum((1/treatment_layer_len[treatment][layer])*np.exp(directions*1j))

            
            ax.set_ylim([0,2])
            ax.set_yticks([0.5,1,1.5,2])
            
            ax.set_ylabel(polarity,labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
        if polarity=='ON':
            title= treatment + ' n= A%d:%d B%d:%d C%d:%d D%d:%d rois ' %(treatment_layer_passed_on[treatment]['LPA'],treatment_layer_len[treatment]['LPA'],\
                                    treatment_layer_passed_on[treatment]['LPB'],treatment_layer_len[treatment]['LPB'],
                                    treatment_layer_passed_on[treatment]['LPC'],treatment_layer_len[treatment]['LPC'],
                                    treatment_layer_passed_on[treatment]['LPD'],treatment_layer_len[treatment]['LPD'])
        else:
            title= treatment + ' n= A%d:%d B%d:%d C%d:%d D%d:%d rois ' %(treatment_layer_passed_off[treatment]['LPA'],treatment_layer_len[treatment]['LPA'],\
                                    treatment_layer_passed_off[treatment]['LPB'],treatment_layer_len[treatment]['LPB'],
                                    treatment_layer_passed_off[treatment]['LPC'],treatment_layer_len[treatment]['LPC'],
                                    treatment_layer_passed_off[treatment]['LPD'],treatment_layer_len[treatment]['LPD'])
        ax.set_title(title,pad=15,fontweight='bold',fontsize=13)

            #savefigure
    if z_bin is not None:
        base_str= savedir +'\\results circ_hist Z%s rel_tresh %.3f rotation %s' %(z_bin,treshold,rotation)
    else:
        base_str= savedir + '\\results circ_hist allZ rel_tresh %.3f'% (treshold )
    save_str1=base_str +'.png'
    #plt.title('vector plots reliability_treshold: %.2f'% treshold)
    plt.savefig(save_str1)
    save_str2=base_str +'.pdf'
    plt.savefig(save_str2)
    plt.close('all')
    return None




def plot_average_vectors_across_rois(roi_dataframe,savedir,z_bin=None,filter_polarity=True):
    

    def rayleigh_test(angles,lenghts):
        print('pending')
    #filter z_layers if necesary
    if z_bin is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['depth_bin']==z_bin]
        #TODO automatize depth_bin selection

    #count number of treatments and categories. also extract their values
    treatment_number=len(np.unique(roi_dataframe['treatment']))
    treatments=np.unique(roi_dataframe['treatment'])
    category_number=len(np.unique(roi_dataframe['categories']))
    categories=np.unique(roi_dataframe['categories'][roi_dataframe['categories']!='No_category'])

    if type(np.array(roi_dataframe['categories'])[0])==set:
        for idx in roi_dataframe['categories'].index:
            print(idx)
            roi_dataframe['categories'][idx]=list(roi_dataframe['categories'][idx])[0]

    roi_dataframe=roi_dataframe.loc[roi_dataframe['categories']!='BG']    

    # turn string entries in dataframe to lowercase
    roi_dataframe['treatment']=roi_dataframe['treatment'].str.lower()
    roi_dataframe['categories']=roi_dataframe['categories'].str.lower()

    
    #create plot scaffold
    fig1=plt.figure(figsize=(40,12))
    gs = GridSpec(2, treatment_number)
    treatment_polarity_pairs=itertools.product([0,1],range(treatment_number))
    plt.style.use('seaborn')
    for pair in treatment_polarity_pairs:
        polarity=['ON','OFF'][pair[0]]
        DSI_name=['DSI_ON','DSI_OFF'][pair[0]]
        PD_name=['PD_ON','PD_OFF'][pair[0]]
        treatment= treatments[pair[1]]
        title = polarity[0:2] + ' ' + treatment
        if filter_polarity==True:
            polarity_subset=roi_dataframe.loc[roi_dataframe['CS']==polarity]
        else: # if a clustering algorithm was not used on the data, for each epoch only use clusters that are responsive
              # to the corresponding epoch
            if polarity=='ON':
                polarity_subset=roi_dataframe.loc[roi_dataframe['ON_responding']==True]
            elif polarity=='OFF':   
                polarity_subset=roi_dataframe.loc[roi_dataframe['OFF_responding']==True] 
        treatment_pol_subset=polarity_subset.loc[polarity_subset['treatment']==treatment]
        ax1=fig1.add_subplot(gs[pair[0],pair[1]],polar=True)
        
        #calculate mean vector across ROIs

        
        
        #number_of_flies=len(np.unique(treatment_pol_subset['FlyId']))
        #number_of_treatment_flies=len(np.unique(roi_dataframe.loc[roi_dataframe['treatment']==treatment]['FlyId']))
        #number_of_rois=len(treatment_pol_subset.index)
        for idx in range(len(treatment_pol_subset.index)):
            magnitude= treatment_pol_subset.iloc[idx,:][DSI_name]
            direction= treatment_pol_subset.iloc[idx,:][PD_name]
            category= treatment_pol_subset.iloc[idx,:]['categories']
            if ('_a' in category) or ('LPA' in category) or ('lpa'in category):
                color='tab:green'
            elif ('_b' in category) or ('LPB' in category) or ('lpb'in category):
                color='tab:blue'
            elif ('_c' in category) or ('LPC' in category) or ('lpc'in category):
                color='tab:red'
            elif ('_d' in category) or ('LPD' in category) or ('lpd'in category):
                color='gold'
            else:
                color='grey'
            title= treatment + ' total n_: %s pol n: %s %s rois ' %(number_of_treatment_flies,number_of_flies,number_of_rois)
            ax1.set_ylim(0,0.7)
            ax1.set_yticks([0.4,0.7])
            ax1.set_title(title,pad=15,fontweight='bold',fontsize=13)
            ax1.set_ylabel(polarity[0:3],labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
            ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
            scale=1,angles="xy",scale_units='xy',alpha=0.7)    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.93, 
                    wspace=0.015, 
                    hspace=0.36)
    #save the plot 
    if z_bin is not None:
        base_str= savedir +'\\results vector plot Z%s' %(z_bin)
    else:
        base_str= savedir + '\\results vector plot allZ' 
    save_str1=base_str +'.png'
    plt.savefig(save_str1)
    save_str2=base_str +'.pdf'
    plt.savefig(save_str2)
    plt.close('all')

# def circ_rayleightest(angle,number_of_incidences):
#     '''references: P. Berens, CircStat: A Matlab Toolbox for Circular Statistics, Journal of Statistical Software, Volume 31, Issue 10, 2009

#         funtion adapted from matlab to calculate the rayleigh test for non-uniform circular distribution
    
#         the rayleigh test tests for the uniformity of circular distribution
#     '''



def plot_variable_histogram_pertreatment(combined_df,variable,save_dir,z_depth=None,fit_Beta=[False, ["negative", "positive"]],colors=['tab:blue','tab:red','grey'],treatment_names=None,treshold=None,filter_CS=True):

    treshold=None
    if z_depth!=[None] and z_depth!= None:
        combined_df=combined_df.loc[combined_df['depth_bin']==z_depth[0]]
    else:
        z_depth= 'none'
    unique_treatments=np.unique(combined_df['treatment'])
    distributions={}
    for pol in np.unique(combined_df['CS']):
        datasets=[]
        labels_=[]

        if pol=='ON':
            if 'OFF' in variable:
                continue
            elif filter_CS==True:
                combined_df_loc=combined_df.loc[combined_df['CS']=='ON']
                plt.figure()    
                sns.set_context("paper",font_scale=1.5)
                sns.set_style("ticks")    
            else:
                plt.figure()    
                sns.set_context("paper",font_scale=1.5)
                sns.set_style("ticks") 
            for loop,treat in enumerate(np.sort(unique_treatments)): 
                # if colors is None:
                #     colors=['tab:blue','tab:red','grey']
                #     loc_color=colors[loop]
                # else:
                loc_color=colors[np.where(np.array(unique_treatments)==treat)[0][0]]
                if treat=='control_homocygous':
                    loc_color=(0,0,0)                
                #temp_str=variable+'_ON'
                if 'reliability' in variable or 'reliability_OFF' in variable:
                    #temp_data=np.histogram(combined_df.loc[combined_df['treatment']==treat][temp_str],np.linspace(-0.2,1,21))            
                    plt.hist(combined_df_loc.loc[combined_df_loc['treatment']==treat][variable],bins=np.linspace(-0.2,1,21),histtype=u'step', density=True,label=treat,color=loc_color,lw=2)
                    if (fit_Beta[0]==True) and (treat in fit_Beta[1]):                        
                        data_to_fit=np.array(combined_df_loc.loc[combined_df_loc['treatment']==treat][variable])
                        alpha_param,beta_param,loc,scale=beta.fit(data_to_fit, floc=-1, fscale=2)
                        distributions[treat]=[alpha_param,beta_param,loc,scale]       
                        x_of_fit=np.linspace(-0.2,1,200)
                        plt.plot (x_of_fit,beta.pdf(x_of_fit,alpha_param,beta_param,loc=loc, scale=scale),'--',linewidth=0.8,color=loc_color,label='alph %.3f beta %.3f loc %s sc %s'%(alpha_param,beta_param,loc,scale))
                elif ('DSI_ON' in variable) or ('CSI' in variable) or ('DSI' in variable) or ('CSI_ON' in variable):
                    plt.hist(combined_df_loc.loc[combined_df_loc['treatment']==treat][variable],bins=np.linspace(-0.2,1,21),histtype=u'step',density=True,label=treat,color=loc_color,lw=2)

                    #temp_data=np.histogram(combined_df.loc[combined_df['treatment']==treat][temp_str],np.linspace(0.0,0.6,19))            
                #datasets.append(temp_data[0].astype(float)/np.sum(temp_data[0]))
                labels_.append(treat)
                
            # for idx,ind_data in enumerate(datasets):
            #     if variable=='reliability':
            #         plt.hist(combined_df.loc[combined_df['treatment']==treat][temp_str],bins=np.linspace(-0.2,1,21),histtype=u'step')
            #         #plt.scatter(np.linspace(-0.17,0.97,20),ind_data,label=labels_[idx],s=2)#,alpha=0.5,align='edge',width=0.05)
            #         #plt.plot(np.linspace(-0.17,0.97,20),ind_data,label=labels_[idx])#,alpha=0.5,align='edge',width=0.05)
            #         if (fit_Beta[0]==True) and (treat in fit_Beta[1]):     
            #             x_of_fit=np.linspace(0,1,100)
            #             plt.plot (x_of_fit,beta.pdf(x_of_fit,alpha_param,beta_param,loc=-1, scale=2))
            #         #plt.yticks()
            #     elif variable=='DSI':
            #         plt.hist(combined_df.loc[combined_df['treatment']==treat][temp_str],bins=np.linspace(0.0,0.6,19),histtype=u'step')
            #         #plt.scatter(np.linspace(0.1333,0.6333,18),ind_data,label=labels_[idx],s=2)#)alpha=0.5,align='edge',width=0.02)
            #         #plt.plot(np.linspace(0.1333,0.6333,18),ind_data,label=labels_[idx])#,alpha=0.5,align='edge',width=0.02)

            if (fit_Beta[0]==True) and (variable=='reliability'):
                beta_functions= lambda x: beta.pdf(x,distributions['positive'][0],distributions['positive'][1],distributions['positive'][2], distributions['positive'][3])\
                                     -  beta.pdf(x,distributions['negative'][0],distributions['negative'][1],distributions['negative'][2], distributions['negative'][3])              
                treshold=fsolve(beta_functions,0.4)
                plt.axvline(x=treshold, ymin=0, ymax=3.5, color='k', label='treshold: %.2f' %treshold,ls=':',alpha=0.8, lw=0.8)
                for label in ax.get_xticklabels():
                    label.set_fontname("Arial")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            if 'DSI' in variable:
                plt.ylim([0,4])
               #plt.xlim([0,1])
            if 'reliability' in variable:
                plt.ylim([0,7])
            plt.yticks([0,1,2,3,4,5,6,7])    
            
            # find the intersect between the Beta distributions
            plt.legend(prop={'size': 8})
            if treshold is None:
                plt.title('ON %s distribution depth:%s no filters' %(variable,z_depth[0]) )
            else:
                plt.title('ON %s distribution depth:%s reliability>%s' %(variable,z_depth[0],treshold) )
            ax= plt.gca()
            ax.legend(loc='upper left',prop={'size': 12, 'family': 'Arial'})
            for label in ax.get_xticklabels():
                label.set_fontname("Arial")  
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            save_str=save_dir+'\\%s_histograms_ON_manualROIs.pdf'%(variable)
            plt.savefig(save_str)
            del labels_
            
        if pol=='OFF':
            if 'ON' in variable:
                continue
            elif filter_CS:
                combined_df_loc=combined_df.loc[combined_df['CS']=='OFF']
                plt.figure()    
                sns.set_context("paper",font_scale=1.5)
                sns.set_style("ticks") 
            else:
                plt.figure()    
                sns.set_context("paper",font_scale=1.5)
                sns.set_style("ticks") 
            for loop,treat in enumerate(np.sort(unique_treatments)): 
        
                # if colors is None:
                #     colors=['tab:blue','tab:red','grey']
                #     loc_color=colors[loop]
                # else:
                loc_color=colors[np.where(np.array(unique_treatments)==treat)[0][0]]
                if treat=='control_homocygous':
                    loc_color=(0,0,0)
        
                #temp_str=variable+'_OFF'
                if 'reliability' in variable or 'reliability_OFF' in variable:
                    #temp_data=np.histogram(combined_df.loc[combined_df['treatment']==treat][temp_str],np.linspace(-0.2,1,21))            
                    plt.hist(combined_df_loc.loc[combined_df_loc['treatment']==treat][variable],bins=np.linspace(-0.2,1,21),histtype=u'step', density=True,label=treat,color=loc_color,lw=2)
                elif ('DSI_OFF' in variable) or ('CSI' in variable) or ('DSI' in variable) or ('CSI_OFF' in variable):
                    plt.hist(combined_df_loc.loc[combined_df_loc['treatment']==treat][variable],bins=np.linspace(-0.2,1,21),histtype=u'step',density=True,label=treat,color=loc_color,lw=2)
                    
            # for idx,ind_data in enumerate(datasets):
            #     if variable=='reliability':
            #         plt.scatter(np.linspace(-0.17,0.97,20),ind_data,label=labels_[idx],s=2)#,alpha=0.5,align='edge',width=0.05)
            #         plt.plot(np.linspace(-0.17,0.97,20),ind_data,label=labels_[idx])#,alpha=0.5,align='edge',width=0.05)
            #     elif variable=='DSI':
            #         plt.scatter(np.linspace(0.1333,0.6333,18),ind_data,label=labels_[idx],s=2)#)alpha=0.5,align='edge',width=0.02)
            #         plt.plot(np.linspace(0.1333,0.6333,18),ind_data,label=labels_[idx])#,alpha=0.5,align='edge',width=0.02)                plt.legend()
                  
            #     plt.ylim([0,0.35])
            #     plt.yticks([0,0.1,0.2,0.3])
            if 'DSI' in variable:
                plt.ylim([0,4])
                #plt.xlim([0,1])
            if 'reliability' in variable:
                plt.ylim([0,7])
            plt.yticks([0,1,2,3,4,5])            
            #plt.legend()
            if treshold is None:
                plt.title('OFF %s distribution depth:%s no filters' %(variable,z_depth[0]) )
            else:
                plt.title('OFF %s distribution depth:%s reliability>%s' %(variable,z_depth[0],treshold) )            
            ax= plt.gca()
            ax.legend(loc='upper left',prop={'size': 12, 'family': 'Arial'})
            for label in ax.get_xticklabels():
                label.set_fontname("Arial")  
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            save_str=save_dir+'\\%s_histograms_OFF_manualROIs.pdf'%(variable)
            plt.savefig(save_str)
            del labels_
    
    return treshold

            

def plot_roivectors_perfly(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=True):    
    'plots the vectors of every roi for every fly'

    # filter for layer if required
    
    if z_bin is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['depth_bin']==z_bin]
    if rotation is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['rotations']==rotation]

    uniqueflies=np.unique(roi_dataframe['FlyId'])
    save_folder=savedir +'\\individual fly results'
    
    ###### Temporal lines for testing#####

    roi_dataframe
    #####end#####

    try:
        os.mkdir(save_folder)
    except:
       pass
    finally:
        pass
    
    for fly in uniqueflies:
        fig1=plt.figure(figsize=(40,12))
        gs = GridSpec(1, 3)
        plt.style.use('seaborn')
        fly_subset=roi_dataframe.loc[roi_dataframe['FlyId']==fly]
        
        ax1=fig1.add_subplot(gs[0,0],polar=True)
        ax2=fig1.add_subplot(gs[0,1],polar=True)
        ax3=fig1.add_subplot(gs[0,2])
        treatment=fly_subset.iloc[0,:]['treatment']

        count_OFF=len(fly_subset.loc[fly_subset['CS']=='OFF'])
        
        count_ON=len(fly_subset.loc[fly_subset['CS']=='ON'])
        textstr = '\n'.join((
        r'$\mathrm{treatment}=%s$' % (treatment, ),
        r'$\mathrm{count_OFF}=%s$' % (count_OFF, ),
        r'$\mathrm{count_ON}=%s$' % (count_ON, ),))
        Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=15,
        verticalalignment='top', bbox=Textprops)
        ax3.axis('off')
        for roi in range(len(fly_subset.index)):
            roi_entry=fly_subset.iloc[roi,:]
            category=roi_entry['categories']
            CS=roi_entry['CS']
            if filter_polarity==True:
                if CS=='ON':                
                    magnitude= roi_entry['DSI_ON']
                    direction= roi_entry['PD_ON']
                    category= roi_entry['categories']
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category):
                        color='tab:green'
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category):
                        color='tab:blue'
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category):
                        color='tab:red'
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category):
                        color='gold'
                    else:
                        color='grey'
                    ax1.set_ylim(0,0.7)
                    ax1.set_yticks([0.5,0.7])                
                    ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
                    scale=1,angles="xy",scale_units='xy',alpha=0.7)       
                    ax1.set_title('ON',pad=15,fontweight='bold',fontsize=16)
                if CS=='OFF':
                    magnitude= roi_entry['DSI_OFF']
                    direction= roi_entry['PD_OFF']
                    category= roi_entry['categories']
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category):
                        color='tab:green'
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category):
                        color='tab:blue'
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category):
                        color='tab:red'
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category):
                        color='gold'
                    else:
                        color='grey'
                    ax2.set_ylim(0,1.2)  
                    ax2.set_yticks([0.5,1])  
                    ax2.quiver(0,0, np.radians(direction),magnitude,color=color,
                    scale=1,angles="xy",scale_units='xy',alpha=0.7)  
                    ax2.set_title('OFF',pad=15,fontweight='bold',fontsize=16)
            else: # if you don't want to separate plots based on the prefered polarity (for example when you have manual ROIs)
                  # in this case, I use a response filter to exclude non-responding ROIs for the respective epochs
                if roi_entry['ON_responding']==True:
                    magnitude= roi_entry['DSI_ON']
                    direction= roi_entry['PD_ON']
                    category= roi_entry['categories']
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category):
                        color='tab:green'
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category):
                        color='tab:blue'
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category):
                        color='tab:red'
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category):
                        color='gold'
                    else:
                        color='grey'
                    ax1.set_ylim(0,0.7)
                    ax1.set_yticks([0.5,0.7])                
                    ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
                    scale=1,angles="xy",scale_units='xy',alpha=0.7)       
                    ax1.set_title('ON',pad=15,fontweight='bold',fontsize=16)
                if roi_entry['OFF_responding']==True:
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category):
                        color='tab:green'
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category):
                        color='tab:blue'
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category):
                        color='tab:red'
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category):
                        color='gold'
                    else:
                        color='grey'
                    magnitude= roi_entry['DSI_OFF']
                    direction= roi_entry['PD_OFF']
                    category= roi_entry['categories']
                    ax2.set_ylim(0,0.7)  
                    ax2.set_yticks([0.5,0.7])  
                    ax2.quiver(0,0, np.radians(direction),magnitude,color=color,
                    scale=1,angles="xy",scale_units='xy',alpha=0.7)  
                    ax2.set_title('OFF',pad=15,fontweight='bold',fontsize=16)
        if z_bin is not None:
            base_str= savedir + '\\results vector Z%s rotation %s' %(z_bin,rotation)
        else:
            base_str= savedir + '\\results vector plot ALLZ %s' %(rotation)

        
        save_str=base_str + '_%s .png' %(fly) #],treatment)

        save_str2= base_str + '_%s .pdf' %(fly) #,treatment)
        # pattern = r'[/:*?"<>,|]'
        # save_str = re.sub(pattern, '__', save_str)
        # save_str2 = re.sub(pattern, '__', save_str2)
        plt.savefig(save_str)
        plt.savefig(save_str2)

        
    plt.close('all')

def plotandcompare_quantities_percategory(quantities,roi_dataframe,savedir,drop_neg=False,stats=False,z_bin=None):
    ''' take relevant measurements for the data, group them by fly and category
        and make barplots out of them'''
    
    numberofplots=len(quantities)
    #numberofpolarities=len(polarities)
    fig1=plt.figure(figsize=(40,12))
    gs = GridSpec(2, int(np.ceil(numberofplots/2)),wspace=4,hspace=1)
    plt.style.use('seaborn')
    plot_pairs=itertools.product([0,1],range(int(np.ceil(numberofplots/2))))
    #if numberofpolarities==2:
    idx1=int(0)
    
    if drop_neg:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['treatment']!='negative']

    #homogenize category labels
    roi_dataframe['categories']=roi_dataframe['categories'].str.lower()
    
    for idx2,quantity in enumerate(quantities):
        # filter polarity
        if 'ON' in quantity:
            pol_subset=roi_dataframe.loc[roi_dataframe['CS']=='ON']
        elif 'OFF' in quantity:
            pol_subset=roi_dataframe.loc[roi_dataframe['CS']=='OFF']
        else:
            pol_subset=roi_dataframe
        #define units for ylabel
        if 'DSI' in quantity:
            ylabel='mean vector length'
        elif 'norm' in quantity:
            ylabel= r'null dir response / pref dir response'
        elif 'max' in quantity:
            ylabel= r'df/f'
        else:
            ylabel=' undefined units '
        num_rois={'LPA':[],'LPB':[],'LPC':[],'LPD':[]} 
        #TODO get number of ROIs per category

        #calculate relevant means and sumary statistics
        df_means=pol_subset.groupby(['FlyId','treatment','categories']).mean()
        df_means = df_means.reset_index(level=['categories','treatment'])
        df_means = df_means.loc[df_means['categories'] != 'no_category']
        
        df_sem=pol_subset.groupby(['FlyId','treatment','categories']).sem()
        df_sem = df_sem.reset_index(level=['categories','treatment'])
        df_sem = df_sem.loc[df_sem['categories'] != 'no_category']
        
        df_std=pol_subset.groupby(['FlyId','treatment','categories']).std()
        df_std = df_std.reset_index(level=['categories','treatment'])
        df_std = df_std.loc[df_std['categories'] != 'no_category']
        
         
        if idx2==int(np.ceil(numberofplots/2)):
            idx1=int(1)
        if idx2>=int(np.ceil(numberofplots/2)):
            ax=fig1.add_subplot(gs[idx1,idx2-int(np.ceil(numberofplots/2))],figsize=() )
        else:            
            ax=fig1.add_subplot(gs[idx1,idx2])
        title='%s '%(quantity)
        ax.set_title(title)
        sns.violinplot(x='categories', y=quantity, hue="treatment", data=df_means,ax=ax,palette='muted',split=False,inner="stick")
        ax.set_ylabel(ylabel)
        #sns.swarmplot(x='categories', y=quantity, hue="treatment", data=df_means,ax=ax,palette='muted')
    # else:
    #     raise Exception(NotImplemented)
    if drop_neg:
        save_str=savedir +'\\quantity comparison no neg alltreatments.pdf'
    else:
        save_str=savedir +'\\quantity comparison alltreatments.pdf'
    plt.savefig(save_str)


def FFF_roi_data_extraction_and_clustering(rois,features='steps'):
    ''' Takes in ROi class instances. outputs quantifications 
        (peak response and sustained response) and traces
        
        it additionally clusters the data based on the peak ON and OFF responses
        
        input:  
               rois (list): list of roi_bg class instances
               corr filter (bool): True if you want to eliminate ROis that are 
                                    not correlated (pearsons > or < 0) with the on or off 
                                    epoch ( non responding or inverted ROIs)
                features (str): 'steps' 'steps&plateau' tells the function which features to use 
                                for clustering
        output: 
                data_DF (pandas dataframe): dataframe containing the extracted quantifications
                fff_stim_trace (np array 1d): a copy of the shifted stim trace for later plotting

        '''
    flyIDs = []
    flash_resps = []
    steps_ON = []
    steps_OFF=[]
    plateau_ON = []
    plateau_OFF = []
    integral = []
    flash_corr = []
    genotypes = []
    cluster=[]

    stim_t = rois[0].int_stim_trace
    for roi in rois:
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['treatment'].strip())
        trace = roi.int_con_trace[:]
        flash_corr.append(roi.corr_fff)
        local_trace = np.concatenate((trace[50:],trace[0:50]),axis=0) # after shifting the trace starts with the ON epoch

        steps_ON.append(local_trace[0:10][np.argmax(np.abs(local_trace[0:10]))])-local_trace[90:].mean() #the step is the value with highest magnitude minus the baseline directly before
        plateau_ON.append(local_trace[10:49].mean()-local_trace[90:].mean())
        steps_OFF.append(local_trace[50:60][np.argmax(np.abs(local_trace[50:60]))])-local_trace[40:50].mean() #the step is the value with highest magnitude minus the baseline directly before
        plateau_OFF.append(local_trace[10:49].mean()-local_trace[90:].mean())
        integral.append(local_trace[0:48].sum())

    # make a matrix including the features and perform clustering analysis
    features_df=np.concatenate((steps_ON,steps_OFF,plateau_ON,plateau_OFF),axis=1)
    if features=='steps':
        features_df=features_df[:2]
    
    # find the optimal number of clusters
    for i in range(2,len(rois)):
        kmeans = KMeans(n_clusters = i, random_state = 0)
        clusters = kmeans.fit_predict(df)
        kmeans.inertia_
        inertia.append(kmeans.inertia_)
        score = metrics.silhouette_score(df, kmeans.labels_, metric='euclidean')
        silhouette_score.append(score)
        i +=1

    


def FFF_data_extraction_1_polarity(type_of_neuron,rois,specific_polarity_to_plot):
    ''' Takes in ROi class instances. outputs quantifications 
        (peak response and sustained response) and traces according 
        to the expected polarity of the Neuron analyzed 
        
        it produces an stimulus trace that fits the polarity given as input

        input: type of neuron (str): 'ON', 'OFF' or 'mixed'
               rois (list): list of roi_bg class instances
               specific_polarity_to_plot (str): 'ON' or 'OFF'. specifies what type of response to 
                                                keep
        
        output: data_DF (pandas dataframe): dataframe containing the extracted quantifications
                mean_pol_stims (np array 1d): a copy of the shifted stimulus trace for later plotting
        '''
    flyIDs = []
    flash_resps = []
    steps = []
    plateau = []
    integral = []
    flash_corr = []
    genotypes = []
    polarity= []
    stim_trace=[]

    local_neuron_type= copy.deepcopy(type_of_neuron)
    for roi in rois:
        stim_t = roi.int_stim_trace
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['treatment'].strip())
        trace = roi.int_con_trace[:]
        flash_corr.append(roi.corr_fff)
        
        if local_neuron_type=='mixed':
            
            type_of_neuron=roi.polarity

        if type_of_neuron=='ON':
            stim_trace.append(np.concatenate((stim_t[25:],stim_t[0:25]),axis=0))
            polarity.append('ON')
            # if roi.corr_fff < 0 :#and corr_filter==True
            #     continue
            
            local_trace = np.concatenate((trace[50:],trace[0:50]),axis=0)
            flash_resps.append(np.concatenate((trace[25:],trace[0:25]),axis=0))
        elif type_of_neuron=='OFF':
            polarity.append('OFF')
            stim_trace.append(np.concatenate((stim_t[75:],stim_t[0:75]),axis=0))
            
            # if roi.corr_fff < 0 : #and corr_filter==True
            #     continue
            flash_resps.append(np.concatenate((trace[75:],trace[0:75]),axis=0))
            local_trace=trace
        else:
            raise Exception ('polarity of roi not found')
        steps.append(local_trace[0:10][np.argmax(np.abs(local_trace[0:10]))]-local_trace[90:].mean()) #the step is the value with highest magnitude minus the baseline directly before
        plateau.append(local_trace[10:49].mean()-local_trace[90:].mean())
        integral.append(local_trace[0:48].sum())
    # reorganize data in dictionary

    data_dict={'index':range(len(flash_resps)),'traces':flash_resps,
            'step_response':steps,'plateau_response':plateau,
            'flyIDs':flyIDs,'treatment':genotypes,'integral':integral,
            'polarity':polarity,'stims':stim_trace}
    data_DF=pd.DataFrame.from_dict(data_dict)

    flash_resps=np.array(flash_resps)
    trace_df=pd.DataFrame(flash_resps,index=range(flash_resps.shape[0]),columns=range(flash_resps.shape[1]))
    trace_df['flyIDs']=data_DF['flyIDs']
    trace_df['treatment']=data_DF['treatment']
    trace_df['polarity']=data_DF['polarity']

    stim_df=np.array(stim_trace)
    stim_df=pd.DataFrame(stim_df,index=range(stim_df.shape[0]),columns=range(stim_df.shape[1]))
    stim_df['flyIDs']=data_DF['flyIDs']
    stim_df['treatment']=data_DF['treatment']
    stim_df['polarity']=data_DF['polarity']

    mean_pol_stims=stim_df.groupby(['polarity']).mean()

    # make a mean stim_trace
    # select datasubset based on desired polarity

    trace_df=trace_df.loc[trace_df['polarity']==specific_polarity_to_plot]
    data_DF=data_DF.loc[data_DF['polarity']==specific_polarity_to_plot]
    mean_pol_stims=mean_pol_stims.reset_index()
    mean_pol_stims=mean_pol_stims.loc[mean_pol_stims['polarity']==specific_polarity_to_plot]
    mean_pol_stims=mean_pol_stims.set_index('polarity')
    mean_pol_stims=np.array(mean_pol_stims)
    return data_DF,trace_df,mean_pol_stims[0]

def plotandcompare_quantities(quantities,alternative_hypotesis, comparisons_to_make,roi_dataframe,savedir,drop_neg=False,do_tests=True,non_parametric=True,filter=None,z_bin=None,polarity_filter=False,statistics=True,treshold=None):
    ''' take relevant measurements for the data, group them by fly 
        and make barplots out of them'''

    number_of_comparisons=0
    numberofplots=len(quantities)
    #numberofpolarities=len(polarities)
    fig1=plt.figure(figsize=(40,12))
    gs = GridSpec(1, numberofplots)
    plt.style.use('seaborn')

    pval_dataframe={}
    pval_dataframe['testing_pairs']=[]
    pval_dataframe['pvals']=[]
    pval_dataframe['variable']=[]    
    pval_dataframe['test']=[]
    pval_dataframe['index']=[]
    if drop_neg:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['treatment']!='negative']
    roi_dataframe=roi_dataframe.loc[roi_dataframe['categories']!='no_category']
    idx1=0
    for idx2,quantity in enumerate(quantities):
        # filter polarity
        if polarity_filter==True:
            if 'ON' in quantity:
                pol_subset=roi_dataframe.loc[roi_dataframe['ON_responding']==True]
            elif 'OFF' in quantity:
                pol_subset=roi_dataframe.loc[roi_dataframe['OFF_responding']==True]
        else:
            pol_subset=roi_dataframe
        #define units for ylabel
        if 'DSI' in quantity:
            ylabel='mean vector length'
        elif 'norm' in quantity:
            ylabel= r'null dir response / pref dir response'
        elif 'max' in quantity:
            ylabel= r'df/f'
        else:
            ylabel=' undefined units '        
        #calculate relevant means and sumary statistics
        df_means=pol_subset.groupby(['FlyId','treatment'])[quantities].mean()
        df_means = df_means.reset_index(level=['treatment'])              
        
        df_sem=pol_subset.groupby(['FlyId','treatment'])[quantities].sem()
        df_sem = df_sem.reset_index(level=['treatment'])        

        df_std=pol_subset.groupby(['FlyId','treatment'])[quantities].std()
        df_std = df_std.reset_index(level=['treatment'])
        
        #statistics

        #test normality and equal variance (between pos vs exp  and neg vs exp)
        
        if non_parametric:
            normal_data=False
        else:
            normal_data=True
        equal_variance=True
        # test normality
        #for treat in ['control_32 deg','experimental_18 deg','experimental_32 deg_5h']:
        for treat in np.unique(df_means['treatment']):#['experimental','negative','positive']:
            treatment_subset=df_means.loc[df_means['treatment']==treat]
            #perform shapiro test on counts and ratios
            _,ps=stats.shapiro(treatment_subset[quantity])
            if ps<0.05:
                normal_data=False
        # test equal variance
        
        for idx, pair in comparisons_to_make:
            _,pl=stats.levene(df_means.loc[df_means['treatment']==pair[0]][quantity],df_means.loc[df_means['treatment']==pair[1]][quantity])
            if pl<0.05:
                equal_variance=False
                break



        #pairwise test experimental vs control groups
        if do_tests:
            for idx, pair in enumerate(comparisons_to_make):
                pval_dataframe['testing_pairs'].append(pair) 
                pval_dataframe['variable'].append(quantity)
                alternative_hypotesis_local=alternative_hypotesis[idx]
                number_of_comparisons+=1
                pval_dataframe['index'].append(number_of_comparisons)
                if normal_data and equal_variance:
                   
                    pval_dataframe['test'].append('ttest'+ alternative_hypotesis_local)
                    _,temp_pval=stats.ttest_ind(df_means.loc[df_means['treatment']==pair[0]][quantity],\
                                                df_means.loc[df_means['treatment']==pair[1]][quantity])
                    pval_dataframe['pvals'].append(temp_pval)
                    
                    if alternative_hypotesis=='greater' and (np.mean(df_means.loc[df_means['treatment']==pair[0]][quantity])\
                                                             >np.mean(df_means.loc[df_means['treatment']==pair[1]][quantity])):
                        pval_dataframe['pvals'].append=temp_pval/2
                    elif alternative_hypotesis=='greater' and (np.mean(df_means.loc[df_means['treatment']==pair[0]][quantity])\
                                                             <np.mean(df_means.loc[df_means['treatment']==pair[1]][quantity])):
                        pval_dataframe['pvals'].append=1-(temp_pval/2)
                    elif alternative_hypotesis=='lesser' and (np.mean(df_means.loc[df_means['treatment']==pair[0]][quantity])\
                                                             >np.mean(df_means.loc[df_means['treatment']==pair[1]][quantity])):
                        pval_dataframe['pvals'].append=1-(temp_pval/2)
                    elif alternative_hypotesis=='lesser' and (np.mean(df_means.loc[df_means['treatment']==pair[0]][quantity])\
                                                             <np.mean(df_means.loc[df_means['treatment']==pair[1]][quantity])):
                        pval_dataframe['pvals'].append=temp_pval/2
                else:
                    #print('mannwhitneyu')
                    pval_dataframe['test'].append('mannwhitneyu '+ alternative_hypotesis_local)
                    _,temp_pval=stats.mannwhitneyu(df_means.loc[df_means['treatment']==pair[0]][quantity],\
                                                df_means.loc[df_means['treatment']==pair[1]][quantity],alternative=alternative_hypotesis_local)
                    pval_dataframe['pvals'].append(temp_pval)


        # if idx2==int(np.ceil(numberofplots/2)):
        #     idx1=int(1)
        # if idx2>=int(np.ceil(numberofplots/2)):
        #     ax=fig1.add_subplot(gs[idx1,idx2-int(np.ceil(numberofplots/2))])#,adjustable='box', aspect=2)
        # else:
        #     ax=fig1.add_subplot(gs[idx1,idx2])
        ax=fig1.add_subplot(gs[idx2])
        #title='%s exp_vs_neg:%1.4f exp_vs_pos: %1.4f test:%s CSI filter:%s '%(quantity,pval_dataframe[],pval_dataframe,test,filter)
        #ax.set_title(title)
        ax.set_ylabel (ylabel)
        if 'mean vector length' in ylabel:
            ax.set_ylim([0,0.4])
        if 'max' in quantity:
            ax.set_ylim([0,6.2])

        #sns.violinplot(x="treatment", y=quantity, data=df_means,ax=ax,palette='muted',split=False,inner="stick",order=['experimental','negative','positive'])
        sns.barplot(x="treatment", y=quantity, data=df_means,ax=ax,palette='muted',order=['experimental','negative','positive'])
        #for treatment in ['experimental','negative','positive']:
        #    temp_data=df_means.loc[df_means['treatment']==treatment]
        #sns.scatterplot(x="treatment", y=quantity,data=df_means, palette='muted', ax=ax, zorder=2)       # if do_tests:
        sns.stripplot(x="treatment", y=quantity,data=df_means, palette='muted', ax=ax, zorder=2,order=['experimental','negative','positive'])       # if do_tests:
        #ax.set_aspect(3.2)
        #     textstr = '\n'.join((
        #     r'$\mathrm{negative vs exp p}=%.4f$' % (p_expvsneg, ),
        #     r'$\mathrm{positive vs exp p}=%.4f$' % (p_expvspos, ),
        #     r'$\mathrm{test}=%s $' % (test, ),))
        #     Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        #     ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        #     verticalalignment='top', bbox=Textprops) 

    # else:
    #     raise Exception(NotImplemented)
    pval_dataframe=pd.DataFrame.from_dict(pval_dataframe)
    alpha_corrected=np.ones(len(pval_dataframe))
    pval_dataframe['alpha_corrected']=alpha_corrected
    
    for variabl in np.unique(pval_dataframe['variable']):
        pval_dataframe_subset=pval_dataframe.loc[pval_dataframe['variable']==variabl]
        pval_dataframe_subset= pval_dataframe_subset.sort_values(by=['pvals'])
        for ix,i in enumerate(pval_dataframe_subset['pvals'].index):
            rank=ix+1
            new_alpha= 0.05/(len(pval_dataframe_subset)-(rank)+1)
            pval_dataframe.at[i,'alpha_corrected']= new_alpha

    
    if drop_neg:
        save_str=savedir +'\\quantity comparison no_neg alltreatments all categories treshold_%s.pdf' %(treshold)
    else:
        save_str=savedir +'\\quantity comparison alltreatments all categories treshold_%s.pdf' %(treshold)

    plt.savefig(save_str)

    # save pvalues
    pval_dataframe.to_csv(savedir +'\\pvalues.csv')

def calculate_plot_on_off_ratio(roi_dataframe,pairs_to_compare,treshold,savedir,alternative_h,z_bin=None):
    ''' plots the number of ON and off rois and the ratio of ON to OFF rois'''
    plt.style.use('seaborn')
    #count_df={'fly':[],'roi count on':[],'roi count off':[],'ratio ON_OFF':[]
    #            ,'treatment':[],'total_count':[],'ON to total':[],'OFF to total':[]}
    count_df={'fly':[],'roi count on':[],'roi count off':[],
            'treatment':[],'total_count':[],'ON to total':[],
            'OFF to total':[]}
    fly_count={'positive':[],'negative':[],'experimental':[]}
    
    #make a copy of the roi_dataframe only with a zdepth_bin (use that to count ROIs)
    if z_bin!= None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['depth_bin']==z_bin]
         
    #loop through flies to obtain the number of ON/OFF rois extracted and their ratio
    uniqueflies=np.unique(roi_dataframe['FlyId'])

    for fly in uniqueflies:        
        fly_subset=roi_dataframe.loc[roi_dataframe['FlyId']==fly]
        treatment=fly_subset.iloc[0,:]['treatment']
        count_ON=len(fly_subset.loc[fly_subset['ON_responding']==True])        
        count_OFF=len(fly_subset.loc[fly_subset['OFF_responding']==True])
        count_all=len(fly_subset)
        count_df['fly'].append(fly)
        count_df['treatment'].append(treatment)
        count_df['roi count off'].append(count_OFF)
        count_df['roi count on'].append(count_ON)
        #count_df['ratio ON_OFF'].append(np.divide(float(count_ON),float(count_OFF))) # currently not in use
        count_df['total_count'].append(count_all)
        count_df['ON to total'].append(count_ON/count_all)
        count_df['OFF to total'].append(count_OFF/count_all)
    count_df=pd.DataFrame.from_dict(count_df)
    
    pval_DF=mean_comparison_between_2groups(count_df,pairs_to_compare,alternative_h,test='calculate',save_dir=savedir)
    
    
    #count_df=count_df.loc[~np.isinf(count_df['ratio ON_OFF'])] #not in use for now
    # count_df=count_df.loc[count_df['ratio ON_OFF']!=0]

    # count_sum=pd.DataFrame.from_dict(count_df).groupby('treatment').sum()
    
    # count_sum=count_sum.reset_index()
        
    # iniialize figure

    fig1= plt.figure(figsize=(40,12))
    gs = GridSpec(1,5)
    ax1=fig1.add_subplot(gs[0,0])
    ax2=fig1.add_subplot(gs[0,1])
    ax3=fig1.add_subplot(gs[0,2])
    ax4=fig1.add_subplot(gs[0,3])
  #  ax5=fig1.add_subplot(gs[0,4])
    treats=np.sort(np.unique(np.array(pairs_to_compare).flatten()))
    sns.barplot(x="treatment", y='roi count on', data=count_df,ax=ax1,palette='muted',order=treats)#,split=False,inner="stick")
    sns.stripplot(x="treatment",y='roi count on', data=count_df, palette='muted', ax=ax1, zorder=2,order=treats)       # if do_tests:
    #ax2.set_title('expVsNeg p: %1.3f posVsExp p: %1.3f test:%s' %() )
    sns.barplot(x="treatment", y='roi count off', data=count_df,ax=ax2,palette='muted',order=treats)#,split=False,inner="stick")
    sns.stripplot(x="treatment", y='roi count off', data=count_df, palette='muted', ax=ax2, zorder=2,order=treats)       # if do_tests:
    
    sns.barplot(x="treatment", y='ON to total', data=count_df,ax=ax3,palette='muted',order=treats)#,split=False,inner="stick")
    sns.stripplot(x="treatment", y='ON to total', data=count_df, palette='muted', ax=ax3, zorder=2,order=treats)       # if do_tests:

    sns.barplot(x="treatment", y='OFF to total', data=count_df,ax=ax4,palette='muted',order=treats)#,split=False,inner="stick")
    sns.stripplot(x="treatment", y='OFF to total', data=count_df, palette='muted', ax=ax4, zorder=2,order=treats)       # if do_tests:

    #ax3.set_title('expVsNeg p: %1.3f posVsExp p: %1.3f test:%s' %())
    #sns.barplot(x="treatment", y='ratio ON_OFF', data=count_df,ax=ax1,palette='muted',order=['experimental','negative','positive'])#.set_title('expvsneg %s __ expvspos %s' %(p_expvsneg,p_expvspos))
    #sns.stripplot(x="treatment", y='ratio ON_OFF', data=count_df, palette='muted', ax=ax1, zorder=2,order=['experimental','negative','positive'])       # if do_tests:
    #ax1.set_title('expVsNeg p: %1.3f posVsExp p: %1.3f test:%s' %(p_expvsneg,p_expvspos,test))
    ax1.set_ylim([0,100])  
    ax2.set_ylim([0,100])  
    ax3.set_ylim([0,1.2])  
    ax4.set_ylim([0,1.2]) 
    #ax1.set_yticks([0,1,2,3,4])
    save_str1=savedir + '\\roi count comparison treshold_%s.pdf' %(treshold)
    plt.savefig(save_str1)
    save_str1=savedir + '\\roi count comparison treshold_%s.png' %(treshold)
    plt.savefig(save_str1)


def find_responsive_ROIs(curr_rois):
    
    print('initial_number_of_rois: %s' %(len(curr_rois)))
    included_rois=[]
    responses=[]
    for roi in curr_rois:
        if roi.CS=='ON':
            index=roi.max_resp_idx_ON
            #STD_baseline=roi.baseline_STD_ON[roi.max_resp_idx_ON]
            #mean_baseline=roi.baseline_mean_ON[roi.max_resp_idx_ON]
            std=np.std(roi.whole_trace_all_epochs[roi.max_resp_idx_ON])
            mean=np.mean(roi.whole_trace_all_epochs[roi.max_resp_idx_ON])
            response=roi.max_response_ON
        elif roi.CS=='OFF':
            index=roi.max_resp_idx_OFF
            std=np.std(roi.whole_trace_all_epochs[roi.max_resp_idx_OFF])
            mean=np.mean(roi.whole_trace_all_epochs[roi.max_resp_idx_OFF])
            #mean_baseline=roi.baseline_mean_OFF[roi.max_resp_idx_OFF]
            response=roi.max_response_OFF
        plt.figure()
        plt.plot(roi.whole_trace_all_epochs[index])
        plt.close('all')
        roi.whole_trace_all_epochs[roi.max_resp_idx_OFF]
        if response>=(mean + 2*std):
            included_rois.append(roi)
            responses.append(response)

    print('rois_kept: %s' %(len(included_rois)))
    return included_rois

def mean_comparison_between_2groups(dataset,pairs_to_compare,alternative_h,category_L=None,polarity=None,test='calculate',save_dir=None):
    meanOfMeans=dataset.groupby(['treatment']).mean()
    meanOfMeans=meanOfMeans.reset_index()
    dataset=dataset.reset_index()
    #test for normality
    test_dataframe={}
    parametric={}
    equal_variance={}
    non_parametric=False
    treats=np.unique(np.array(pairs_to_compare).flatten())
    key_list=[]
    if 'index' in dataset.keys():
        dataset=dataset.set_index('index')
    for key in dataset.keys():
        if 'float' in str(dataset[key].dtype) or 'int' in str(dataset[key].dtype):
            key_list.append(key)
    for key in key_list:
       for treatment in np.unique(treats):
            if stats.shapiro(dataset.loc[dataset['treatment']==treatment][key])[0]<0.05:
                parametric[key]=False
                continue
            else: 
                parametric[key]=True
    # test for equal variance
    for key in key_list: 
        equal_variance[key]={}
        #groups=[np.array(dataset.loc[dataset['treatment']==i][key]) for i in np.unique(treats)]
        for pair in pairs_to_compare:
            equal_variance[key]
            _,pl=stats.levene(dataset.loc[dataset['treatment']==pair[0]][key],dataset.loc[dataset['treatment']==pair[1]][key])
            if pl<0.05:
                equal_variance[key][str(pair)]=False
            else:
                equal_variance[key][str(pair)]=True
            

            
    pval_dataframe={}
    pval_dataframe['testing_pairs']=[]
    pval_dataframe['pvals']=[]
    pval_dataframe['variable']=[]
    pval_dataframe['idx']=[]
    pval_dataframe['test']=[]
    pval_dataframe['equal_var']=[]
    count=0
    for index1,category in enumerate(parametric.keys()):
        # if category=='index':
        #     continue
        for index,pair in enumerate(pairs_to_compare):        
            count+=1
            if parametric[category]==False:
                
                if alternative_h[index]!= None:
                    _,pval= stats.mannwhitneyu(dataset.loc[dataset['treatment']==pair[0]][category],dataset.loc[dataset['treatment']==pair[1]][category],alternative=alternative_h[index])
                    pval_dataframe['test'].append('mannwhit_%s'%(alternative_h[index]))
                else:
                    _,pval= stats.mannwhitneyu(dataset.loc[dataset['treatment']==pair[0]][category],dataset.loc[dataset['treatment']==pair[1]][category],alternative='two-sided')
                    pval_dataframe['test'].append('mannwhit_2tail')
                
                test_dataframe[pair].append()
                test_dataframe['idx'].append(category)
            if parametric[category]==True:
                
                if equal_variance[category][str(pair)]==True:
                    _,pval= stats.ttest_ind(dataset.loc[dataset['treatment']==pair[0]][category],dataset.loc[dataset['treatment']==pair[1]][category],equal_var=True)
                    eq_var_str='eqvar'
                else:
                    _,pval= stats.ttest_ind(dataset.loc[dataset['treatment']==pair[0]][category],dataset.loc[dataset['treatment']==pair[1]][category],equal_var=False)
                    eq_var_str='neqvar' 
                if meanOfMeans.loc[meanOfMeans['treatment']==pair[0]][category].values>meanOfMeans.loc[meanOfMeans['treatment']==pair[1]][category].values:
                    if alternative_h[index]=='greater':
                        pval=pval/2
                        pval_dataframe['test'].append('ttest_%s_greater'%(eq_var_str))
                    elif alternative_h[index]=='less':
                        pval=1-(pval/2)
                        pval_dataframe['test'].append('ttest_%s_less'%(eq_var_str))
                    else:
                        pval_dataframe['test'].append('ttest_%s_2tail'%(eq_var_str))

                if meanOfMeans.loc[meanOfMeans['treatment']==pair[0]][category].values<meanOfMeans.loc[meanOfMeans['treatment']==pair[1]][category].values:
                    if alternative_h[index]=='less':
                        pval=pval/2
                        pval_dataframe['test'].append('ttest_%s_less'%(eq_var_str))

                    elif alternative_h[index]=='greater':
                        pval=1-(pval/2)
                        pval_dataframe['test'].append('ttest_%s_greater'%(eq_var_str))
                    else:
                        pval_dataframe['test'].append('ttest_%s_2tail'%(eq_var_str))

            pair_joined='-'.join(pair)
            pval_dataframe['testing_pairs'].append(pair_joined)
            pval_dataframe['pvals'].append(pval)
            pval_dataframe['variable'].append(category)
            pval_dataframe['idx'].append(count)
            pval_dataframe['equal_var'].append(equal_variance[category][str(pair)])
    pval_dataframe=pd.DataFrame(pval_dataframe)
    pval_dataframe=pval_dataframe.set_index(['idx'])
    pval_dataframe['alpha_corrected']=np.ones(len(pval_dataframe))
    #apply the Holm-bonferoni procedure for multiple testing
    for variabl in np.unique(pval_dataframe['variable']):
        pval_dataframe_subset=pval_dataframe.loc[pval_dataframe['variable']==variabl]
        pval_dataframe_subset= pval_dataframe_subset.sort_values(by=['pvals'])
        for ix,i in enumerate(pval_dataframe_subset['pvals'].index):
            rank=ix+1
            new_alpha= 0.05/(len(pval_dataframe_subset)-(rank)+1)
            pval_dataframe.at[i,'alpha_corrected']= new_alpha
    if save_dir!=None:
        string_s='-'.join(np.unique(np.array(pval_dataframe['variable'])))
        pval_dataframe.to_csv(save_dir +'\\pvalues'+string_s +' layer_%s polarity_%s'%(category_L,polarity) +'.csv')
    return pval_dataframe

def create_subplots(num_subplots):
    # Calculate the number of columns based on the number of subplots and the fixed 2-row layout
    num_cols = -(-num_subplots // 2)  # Ceil division
    
    # Initialize the figure using gridspec
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, num_cols)
    
    axes = []
    for i in range(num_subplots):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    return fig, axes

def create_df_interp_traces(df, key, independent_var):
    # Filter the dataframe based on the given independent_var
    subset_df = df[df['independent_var'] == independent_var]
    lenss=[]
    # get the longest time vector for the x axis
    for row in range(len(subset_df)):
        try:
            lenss.append(len(subset_df.iloc[row]['interpolated_time'][key][~np.isnan(subset_df.iloc[row]['interpolated_time'][key])]))
        except:
            continue
    lenss=np.array(lenss)
    interpolated_time = subset_df.iloc[np.where(lenss==np.max(lenss))[0][0]]['interpolated_time'][key]
    
    var_values=subset_df['ind var values'].iloc[0]
    # Extract data from the 'interpolated_traces' column using the provided key
    subset_df['trace_data'] = subset_df['interpolated_traces'].apply(lambda x: x.get(key, np.full(np.max(lenss), np.nan)))
    subset_df=subset_df[['FlyId','independent_var','treatment','trace_data','category']]
    # Create the new dataframe with specified indices and extracted data
    df_interp_traces = subset_df[['FlyId', 'treatment', 'trace_data','category']].set_index(['FlyId', 'treatment','category'])
    
    def explode_arrays_to_columns(df):
        # Convert 'trace_data' arrays into individual columns
        expanded_data = df['trace_data'].apply(pd.Series)
        expanded_data.columns = ['timepoint_{}'.format(i) for i in expanded_data.columns]

        # Concatenate the expanded data with the original dataframe (excluding the 'trace_data' column)
        df_exploded = pd.concat([df.drop('trace_data', axis=1), expanded_data], axis=1)

        return df_exploded
    
    df_interp_traces=explode_arrays_to_columns(df_interp_traces)

    return df_interp_traces, interpolated_time,var_values

def plot_interp_traces(time_vector,df_interp_traces,epoch_value, units,variab, ax,colors,treatment_names,lim,category=None,check=False):
    # sort by category
    if category is not None:
        df_interp_traces_local=df_interp_traces.xs(key=category,level='category')
    else:
        df_interp_traces_local=df_interp_traces
    # Group by 'FlyId' and calculate the mean
    means_of_flies = df_interp_traces_local.groupby(['FlyId','treatment']).mean()
    means_of_flies = align_fly_traces_by_treatment(means_of_flies)
    means_of_flies = align_traces(means_of_flies)
    # Calculate SEM and mean grouped by 'treatment'
    sem = means_of_flies.groupby('treatment').sem()
    means = means_of_flies.groupby('treatment').mean()
    
    treatments = means.index.unique()
    sns.set_context("paper",font_scale=1.5)
    sns.set_style("ticks")
    #TODO realign!!!
    font = FontProperties()
    font.set_family('Arial')
    font.set_size(14)
    ax.axis('off')
    
    for treatment in treatments:
        mean_trace = means.reset_index().loc[means.reset_index()['treatment']==treatment]
        mean_trace=np.squeeze(np.array(mean_trace.set_index('treatment')))
        error = sem.reset_index().loc[sem.reset_index()['treatment']==treatment]
        error = np.squeeze(np.array(error.set_index('treatment')))
        line_color=colors[np.where(np.array(treatment_names)==treatment)[0][0]]
        # Plot the mean trace
        ax.plot([0,time_vector[-1]],[0,0],color='grey')
        sns.lineplot(x=time_vector, y=mean_trace, label=treatment, ax=ax,color=line_color,lw=2)
        
        #make a scale bar
        ax.plot([-0.5, 0.5], [-0.5, -0.5], color='k', lw=2)
        ax.plot([-0.5, -0.5], [-0.5, 1.5], color='k', lw=2)
        ax.plot
        ax.set_ylim(lim)
        ax.text(-1, 0.5, r'2 Df/f', rotation=90, verticalalignment='center',fontname='Arial',fontsize=12)
        ax.text(0, -0.85, '1s', verticalalignment='center',fontname='Arial',fontsize=12)
        # Shade the area around each mean trace using the SEM value
        ax.fill_between(time_vector, mean_trace - error, mean_trace + error,color=line_color, alpha=0.3)

    if check==False:
        ax.legend(prop={'size': 10, 'family': 'Arial'},loc='upper left',bbox_to_anchor=(-0.6, 1.2))
    else:
        ax.legend_.remove()
    

    if variab=='luminance':
        ax.set_title('%.3e %s' %(epoch_value,units))
    else:
        ax.set_title('%.3f %s' %(epoch_value,units))

    #ax.set_ylabel('df/f')
    #ax.set_xlabel('Time (s)')

def align_fly_traces_by_treatment(df_interp_traces):
    # Define a function to align rows within a specific treatment group
    def align_group(group_df):
        # Get the reference row (first row) for this treatment group
        lenss=[]
        def align_row(row):
            if np.sum(~np.isnan(row))==0:
                return row
            else:
                #deal with the nans in here
                row_loc=np.where(np.isnan(row),0,row)
                correlation = correlate(reference, row_loc)
                shift = len(reference) - np.argmax(correlation) - 1
                row = np.roll(np.array(row), -shift)
                return row
        for row in range(len(group_df)):
            lenss.append(len(group_df.iloc[row][~np.isnan(group_df.iloc[row])]))
        lenss=np.array(lenss)
        reference = group_df.iloc[np.where(lenss==np.max(lenss))[0][0]]
        for ix in range(len(group_df)):
            group_df.iloc[ix]=align_row(group_df.iloc[ix])
        # Define a function to align a row with the reference
        

        # Apply the alignment function to each row in the group
        return group_df

    # Group by treatment and apply the alignment function to each group
    df_aligned = df_interp_traces.groupby('treatment').apply(align_group)
    
    return df_aligned

def align_traces(df_interp_traces):
    # Define a function to align rows within a specific treatment group
    def align_group(group_df):
        # Get the reference row (first row) for this treatment group
        lenss=[]
        def align_row(row):
            if np.sum(~np.isnan(row))==0:
                return row
            else:
                #deal with the nans in here
                row_loc=np.where(np.isnan(row),0,row)
                max_pos = np.argmax(row_loc)
                shift = len(row_loc)//2 - max_pos - 1
                row = np.roll(np.array(row), shift)
                return row

        for ix in range(len(group_df)):
            group_df.iloc[ix]=align_row(group_df.iloc[ix])

        return group_df

    # Group by treatment and apply the alignment function to each group
    df_aligned = align_group(df_interp_traces)
    
    return df_aligned

# Example usage (assuming df_interp_traces has been defined):
# df_aligned_by_treatment = align_fly_traces_by_treatment(df_interp_traces)


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def _explode_arrays_to_columns(df,dependent_var):
    # Convert 'trace_data' arrays into individual columns
    expanded_data = df[dependent_var].apply(lambda x: pd.Series(np.squeeze(x)))
    expanded_data.columns = ['y_{}'.format(i) for i in expanded_data.columns]

    # Concatenate the expanded data with the original dataframe (excluding the 'trace_data' column)
    df_exploded = pd.concat([df.drop(dependent_var, axis=1), expanded_data], axis=1)

    return df_exploded

def find_fly_means_from_dataframe(df,dependent_var):
    
    subset_df=df[['FlyId',dependent_var,'treatment']].set_index(['FlyId', 'treatment'])

    subset_df=_explode_arrays_to_columns(subset_df,dependent_var)

    # take the average across flies and treatments
    subset_df.groupby(['FlyId','treatment']).mean()
    means_of_flies = subset_df.groupby(['FlyId','treatment']).mean()
    means_of_flies = means_of_flies.dropna(axis = 1, how = 'all')
    return means_of_flies

def test_statsassumptions_DF_columns(df,comparisons_to_make,save_path,variab):
    """ take in a dataframe where columns are values of an independent variable 
        and the indices correspond to flyids and treatments. 
         use this dataframe to calculate the shapiro test for everycolumn and the levene test for 
         specific pairs of treatments"""
    

    # Unique treatment values
    treatments = df.index.get_level_values('treatment').unique()
    # Create empty dataframes for results
    shapiro_results = pd.DataFrame(index=treatments, columns=df.columns)
    compar_columns=[]
    for comparison in comparisons_to_make:
        compar_columns.append(comparison[0]+'_'+comparison[1])
    levene_results = pd.DataFrame(index=compar_columns, columns=df.columns)

    for treatment in treatments:
        subset = df.xs(treatment,level='treatment', axis=0)        
        for col in df.columns:
            # Shapiro-Wilk test
            try:
                _, p_shapiro = stats.shapiro(subset[col].dropna())
            except ValueError:
                p_shapiro = np.nan
            
            shapiro_results.loc[treatment, col] = p_shapiro 
            
            # Levene test requires at least two samples, so we'll check if we have multiple treatments
    for group in comparisons_to_make:

        for key in df.keys():
            subset1 = df.xs(group[0],level='treatment', axis=0) 
            current_sample1=subset1[key].dropna()            
            subset2 = df.xs(group[1],level='treatment', axis=0) 
            current_sample2=subset2[key].dropna()
            try:
                _, p_levene = stats.levene(current_sample1, current_sample2)
            except:
                p_levene = np.nan
            levene_results[key].loc[group[0]+'_'+group[1]] = p_levene 
    shapiro_results.to_csv(save_path +"\\normality_test.csv")
    levene_results.to_csv(save_path +"\\equal_variance_test.csv")
    return shapiro_results<0.05, levene_results<0.05

def multiple_pairwise_stats(fly_means,normality,variance_test,comparisons_to_make,variab,variab_x_values,alternative,savedir,depend_var):
    """ performs repeated pairwise comparisons among 2 treatments and across all posible values of an 
        independent variable that was examined for each tratment
        
        inputs: fly_means (pd.DataFrame): contains all values to be examined with indices for flyids and treatments
                the columns correspond to the independent variable values to be examined
                
                normality and variance test: boolean dataframes used to examine if parametric or non-parametric tests are used

                comparisons_to_make: lists of lists containing the values of the index treatment to be extracted to perform tests"""
    #eliminate Nans in the xvalue array:
    variab_x_values = np.array(variab_x_values)[~np.isnan(variab_x_values)]
    # create a dataframe
    pval_dataframe={}
    pval_dataframe['testing_pairs']=[]
    pval_dataframe['pvals']=[]
    pval_dataframe['variable']=[]
    pval_dataframe['test']=[]
    pval_dataframe['index']=[]
    pval_dataframe['alternative_H']=[]
    pval_dataframe['ind_var_values']=[]
    pval_dataframe['alpha']=[]
    if np.sum(np.array(normality))>0 or np.sum(np.array(variance_test))>0:
        test_function = stats.mannwhitneyu
        test_used = 'mannwhitneyu'
    else:
        test_function= stats.ttest_ind
        test_used = 'ttest_ind'

    for index,pair in enumerate(comparisons_to_make): 
        alternative_used=alternative[index]
        local_pvals = []
        for ix,key in enumerate(fly_means.keys()):

            group1 = fly_means.xs(key = pair[0], level ='treatment')[key]
            group2 =fly_means.xs(key = pair[1], level ='treatment')[key]
            _,P_val = test_function(group1,group2,alternative=alternative_used)
            
            local_pvals.append(P_val)
            pval_dataframe['test'].append(test_used)
            pval_dataframe['alternative_H'].append(alternative_used)
            pval_dataframe['ind_var_values'].append(variab_x_values[ix])
            pval_dataframe['pvals'].append(P_val)
            pval_dataframe['index'].append(ix)
            pval_dataframe['variable'].append(variab)
            pval_dataframe['testing_pairs'].append(pair)
        # multiple comparison correction
        corrected_alphas = np.ones(len(fly_means.keys()))
        corrected_alphas[:] = 0.05
        # this is the Bonferroni-Holm correction, the highest Pval gets the least stringent test
        pval_order = (np.argsort(np.array(local_pvals)))[::-1]
        
        # GPT: Creating a rank for each index based on its position in the sorted list
        rank = {index: rank for rank, index in enumerate(pval_order, start=1)}
        # GPT: Dividing pvalues by their rank
        corrected_alphas = [value / rank[i] for i, value in enumerate(corrected_alphas)]

        pval_dataframe['alpha'] = pval_dataframe['alpha'] + corrected_alphas
    

    pval_dataframe = pd.DataFrame(pval_dataframe)
    pval_dataframe.to_csv(savedir +'\\' + variab +'_pvalues_' + depend_var +'.csv')

    return pval_dataframe

def plot_tuning_curve_w_error(means_of_flies,variab,dependent_var,indep_var_vals,layer,units,colors,treatment_names,ax):
    
    # eliminate nan values from indep_var
    indep_var_vals = np.array(indep_var_vals)[~np.isnan(indep_var_vals)]
    # calculate statistics
    

    #calculate mean across treatment and sem 
    sem = means_of_flies.groupby('treatment').sem()
    means = means_of_flies.groupby('treatment').mean()

    for treatment in np.unique(means.reset_index()['treatment']):
        

        line_color=colors[np.where(np.array(treatment_names)==treatment)[0][0]]
        mean_trace = means.loc[treatment]
        sem_trace = sem.loc[treatment]
        # Plot the mean trace
        sns.lineplot(x=indep_var_vals, y=mean_trace, label=treatment, markers="o",ax=ax,color=line_color,lw=2) 
        #sns.scatterplot(x=indep_var_vals, y=mean_trace,label='_no_legend_' markers="o",ax=ax,color=line_color, s=55)       
        ax.errorbar(np.array(indep_var_vals), np.array(mean_trace), yerr=np.array(sem_trace), fmt='o',markersize=6, color=line_color,label='_no_legend_', capsize=0)
        # Shade the area around each mean trace using the SEM value
        ax.set_xscale('log')
        sns.set_context("paper",font_scale=1.5)
        sns.set_style("ticks")
        
        if ('frequency' in variab) and ('freq_power_epochs' in dependent_var):
            ax.set_ylabel(r'/Hz')
        else:
            ax.set_ylabel(r'df/f')
            
        ax.set_xlabel(units)
        # Set font name for x-axis tick labels
        for label in ax.get_xticklabels():
            label.set_fontname("Arial")

        # Set font name for y-axis tick labels
        for label in ax.get_yticklabels():
            label.set_fontname("Arial")
        for position in ['left', 'bottom']:
            ax.spines[position].set_linewidth(0.8)
        for position in ['right', 'top']:    
            ax.spines[position].set_linewidth(0)

    #plot results
    ax.set_ylim([0,None])
    if layer is not None:
        ax.set_title('%s_%s tunning'%(layer,variab))
    else:
        ax.set_title('all layers %s tunning'%(variab))
    ax.set_xlabel(units)

def count_rois_flies_per_treatment(roi_dataframe, divide_by_independent_variable=True,divide_by_category=True,SaveData=True,SaveDir=None):
    """ this function takes in a Dataframe containing information 
    of the identity of rois and the treatment they are part of
    
    it counts the number of flies and rois per treatment. any Dataframe can be used that 
    contains info of flyId and treatment

    output: pd.DataFrame({treatment: ['treatment_n',...],independent_var: ['ind_var_name',....],'counts':[[Number of flies,Number of rois],...]})
    """
    sample_counts={'treatment':[],'independent variable':[], 'counts (flies_rois)' :[],'category':[]}

    if divide_by_independent_variable:
        independent_variables=np.unique(roi_dataframe['independent_var'])
    else:             
        independent_variables=[None]
        

    if divide_by_category:
        categories=np.unique(roi_dataframe['category'])
        save_str='sample_counts_perlayer.csv'
    else:             
        categories=[None]
        save_str='sample_counts_alllayers.csv'
    
    treatments = np.unique(roi_dataframe['treatment'])
    
    for ind_var,treatment,category in itertools.product(independent_variables,treatments,categories):
        # extract relevant subsets of the data
        roi_dataframe_subset=subset_Df_by_column_values(['independent_var','treatment','category'],[ind_var,treatment,category],roi_dataframe)
        fly_count=len(np.unique(roi_dataframe_subset['FlyId']))
        roi_count=len(roi_dataframe_subset)
        sample_counts['treatment'].append(treatment)
        sample_counts['counts (flies_rois)'].append([fly_count,roi_count])
        sample_counts['independent variable'].append(ind_var)
        sample_counts['category'].append(category)

    sample_counts=pd.DataFrame(sample_counts)
    if SaveData:
        sample_counts.to_csv(SaveDir + save_str)

    return sample_counts

def subset_Df_by_column_values(keys,values,dataframe):
    
    """extract dataframe subsets based on a specific value across multiple columns"""
    #include cases where the value is not None
    valid_keys = np.array([x is not None for x in values])
    keys = np.array(keys)
    keys = keys[valid_keys]
    values = np.array(values)
    values = values[valid_keys]

    boolean_ix=pd.Series([True]*len(dataframe))
    for key,value in zip(keys,values):
        boolean_ix= boolean_ix*dataframe[key]==value
    
    return dataframe[boolean_ix]
    

def convert_luminance_mWToflux(power_at_screen,distance=5.3,wavelenght=482):

    """converts units of luminance from Mw/cm2 to photonflux (photons/cm2 s) 
    
    inputs: power_at_screen (mW/cm2): measured light intensity at the surface of the screen 
            
            distance: distance from the center of the screen to the fly. defajult is 5.3 cm
            
            wavelenght: peak wavelenght emitted by the screen. if thorlabs luminometer 
                        used, then the readings should be taken at this same wavelenght (you can set wavelenght in the luminometer).
                        default is 482nm
    """

    power_at_fly= power_at_screen / (distance**2)

    photonflux = (power_at_fly*wavelenght)/ 1.989e-10 # units: Ph/cm2*s or phi/cm2 1.989e-10 is the planck constant times speed of light 

    return photonflux

# %%
