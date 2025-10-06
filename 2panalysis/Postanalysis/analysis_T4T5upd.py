import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage, stats
import sys
code_path = r'U:\Work_PT\Code\2p_analysis_development\2panalysis\Helpers'
sys.path.insert(0, code_path) 
from Helpers import ROI_mod
import post_analysis_core as pac
import process_mov_core as pmc
<<<<<<< Updated upstream
=======
from pathlib import Path, PureWindowsPath
>>>>>>> Stashed changes
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use('seaborn-v0_8-talk')
#####______________Parameters________________###########

<<<<<<< Updated upstream
results_save_dir = r'U:\Work_PT\Plots\upd_flies_multi_depth\75deg\d50'
filter_polarities = False
z_bin = 50
reliability_treshold = 0.4
CS_filter = [True]
rotation = None
datapath = r'C:\Users\ptrimbak\Work_PT\2p_data\240416_T4T5_upd_depths\rotation_75deg\d50'
=======

>>>>>>> Stashed changes

"""Saving as PDF funtion"""
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, dpi = dpi, format='pdf')
    pp.close()

""""interpolate the data and find then align the peaks"""
# dataset = '231204_PT_fly1-TSeries-12042023_G8f-069-cycle_1_cycle0_DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_ONEDGEFIRST_80sec.pickle'
def interpolate_data(rois):
    for roi in rois:
        req_epochs = np.where(np.array(curr_rois[0].stim_info['stimtype'])=='driftingstripe')[0]
        roi.interpolated_traces_epochs={}            # this dict is to be filled up with mean traces for the epoch 
        roi.interpolated_time={}
        for idx, epoch in enumerate(req_epochs):
            stim_duration=roi.stim_info['duration'][epoch] 
            roi.interpolated_time[epoch],roi.interpolated_traces_epochs[epoch]=\
                ROI_mod.interpolate_signal(roi.resp_trace_all_epochs[epoch],
                roi.imaging_info['frame_rate'],
                int_rate=11,stim_duration=stim_duration)
    return rois

"""Make a dataframe of the data with all the files to analyse"""
# Convert the data list to a pandas DataFrame    
def make_genotype_df(data, genotype, CS_str = None):
    only_genotype = []
    for xx, pick in enumerate(data):
        for idx, lst in enumerate(pick):
            if genotype in lst['genotype']:
                df = pd.DataFrame(data[xx])
                only_genotype.append(df)
            elif genotype == None:
                df = pd.DataFrame(data[xx])
                only_genotype.append(df)
    df1 = pd.concat(only_genotype)
    if CS_str == None:
        df1 = df1
    else:
<<<<<<< Updated upstream
        df1 = df1[df1['reliability_PD'] >= 0.4]
=======
        df1 = df1[df1['reliability_PD'] >= 0.5]
>>>>>>> Stashed changes
        df1 = df1[df1['categories']!="No_category"]
        df1 = df1[df1['CS']==CS_str]
        df1 = df1.reset_index(drop = True)
    return df1
<<<<<<< Updated upstream


# datapath = r'U:\Work_PT\2p_data\240124_Gcamp_exp\data_processed'


datasets_to_load = []
for root, dirs, files in os.walk(datapath):
    for file in files:
        if 'sec.pickle' in file:
            datasets_to_load.append(os.path.join(root, file))

# datasets_to_load = os.listdir(datapath)
data= [[] for i in range(len(datasets_to_load))]
for idx, dataset in enumerate(datasets_to_load):
    
    # load_path = os.path.join(datapath, dataset)
    load_path = dataset
    load_path = open(load_path, 'rb')
    workspace = pickle.load(load_path)

    curr_rois = workspace['final_rois']
    curr_rois=pmc.extract_null_dir_response(curr_rois)
    interpolate_data(curr_rois)
    req_epochs = np.where(np.array(curr_rois[0].stim_info['stimtype'])=='driftingstripe')[0]
    for icx, epoch in enumerate(req_epochs):
        ROI_mod.align_traces_per_epoch(curr_rois, epoch, grating=False)
    
    
    for roi in curr_rois: 
        entry = {
                # 'uniq_id': roi.uniq_id,
                # 'independent_var': getattr(roi, 'independent_var', np.nan),
                # 'interpolated_traces': getattr(roi, 'interpolated_traces_epochs', np.nan),
                # 'interpolated_time': getattr(roi, 'interpolated_time', np.nan),
                # 'max_resp_all_epochs': getattr(roi, 'max_resp_all_epochs', np.nan),
                'max_resp_all_epoch_on':getattr(roi, 'max_resp_all_epochs_ON', np.nan),
                'max_resp_all_epoch_off':getattr(roi, 'max_resp_all_epochs_OFF', np.nan),
                # 'freq_power_epochs': getattr(roi, 'freq_power_epochs', np.nan),  # Use np.nan if attribute does not exist
                'FlyId': roi.experiment_info['FlyID'],  # Include FlyId attribute
                # 'ind var values': getattr(roi, 'independent_var_vals', np.nan),  # Include independent_var_vals attribute
                # 'treatment': roi.experiment_info['treatment'],
                'raw_trace': getattr(roi, 'raw_trace', np.nan),
                'genotype': roi.experiment_info['Genotype'],
                'categories': roi.category[-1],
                'reliability_PD': getattr(roi, 'reliability_PD', np.nan), 
                'reliability_ON': getattr(roi, 'reliability_PD_ON', np.nan),
                'reliability_OFF': getattr(roi, 'reliability_PD_OFF', np.nan),
                'DSI_ON': getattr(roi, 'DSI_ON', np.nan), #
                'DSI_OFF': getattr(roi, 'DSI_OFF', np.nan),#
                'CS': getattr(roi, 'CS', np.nan),
                'CSI': getattr(roi, 'CSI',np.nan),#
                'whole_traces_all_epochsTrials': getattr(roi, 'whole_traces_all_epochsTrials', np.nan), 
                'df_trace': getattr(roi, 'df_trace', np.nan), 
                'dir_max_resp': getattr(roi, 'dir_max_resp', np.nan),
                'PD_ON': getattr(roi, 'PD_ON', np.nan),#
                'PD_OFF': getattr(roi, 'PD_OFF', np.nan),#
                'direction_vector':getattr(roi, 'direction_vector', np.nan),
                'max_response_ON':getattr(roi, 'max_response_ON', np.nan), #
                'max_response_OFF':getattr(roi, 'max_response_OFF', np.nan), #
                'norm_null_dir_resp_ON':getattr(roi, 'norm_null_dir_resp_ON', np.nan),
                'norm_null_dir_resp_ON':getattr(roi, 'norm_null_dir_resp_ON', np.nan),
                'interpolated_traces_epochs': getattr(roi, 'interpolated_traces_epochs', np.nan), 
                'interpolated_time': getattr(roi, 'interpolated_time', np.nan),
                'treatment': roi.experiment_info['treatment'],
                'sex': roi.experiment_info['Sex'],
                'depth': roi.experiment_info['z_depth'],
                'depth_bin': roi.experiment_info['z_depth_bin'],
                'rotations': roi.experiment_info['rotation'],
            }
        data[idx].append(entry)
    

upd6on = make_genotype_df(data, genotype='Upd_6f', CS_str='ON')
upd6off = make_genotype_df(data, genotype='Upd_6f', CS_str='OFF')
upd6 = make_genotype_df(data, genotype='Upd_6f', CS_str=None)

=======
>>>>>>> Stashed changes
#####response traces normalised####
def filter_interpolated_flyID(df, flyID, layer):
    if flyID == None:
        df = df[df['categories']==layer]
        df = df.reset_index(drop = True)
        arra = df[['interpolated_traces_epochs','interpolated_time']]
    else:    
        df = df[df['FlyId']==flyID]
        df = df[df['categories']==layer]
        df = df.reset_index(drop = True)
        arra = df[['interpolated_traces_epochs','interpolated_time']]
        
    return arra

###############
"8f flies"
###############

def create_sorted_lists(dataframe, epochnum):
    k = [[] for i in range(len(epochnum))]
    for tra, traces in enumerate(dataframe):
        for j in epochnum:
            k[j].append(traces[j])
    return k        

def make_plots_with_CI(arr):
    mean_r = np.mean(arr, axis=0)
    std_r = np.std(arr, axis=0)/ np.sqrt(len(arr[0]))
    # std_r = np.std(arr, axis=0)
    ub = mean_r + std_r
    lb = mean_r - std_r
    return mean_r, std_r, ub, lb
        
def response_traces_aligned_plot(dataframe, layer = str):
    mycolor=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5']
    flyID = np.unique(dataframe['FlyId']) 
    for k, s in enumerate(flyID):
        fly = filter_interpolated_flyID(dataframe, flyID[k], layer=layer)
        fig = plt.figure()
        fig.suptitle(f'Fly {k+1}: For layer "{layer}" ')
        fig.supxlabel('Time(s)')
        fig.supylabel('Response (df/f)')
        gs = fig.add_gridspec(len(req_epochs), hspace=0.2)
        axs = gs.subplots(sharex=True, sharey=True) 
        # k = [[] for i in range(len(req_epochs))]
        klist = create_sorted_lists(fly['interpolated_traces_epochs'], req_epochs)
        # for tra, traces in enumerate(fly['interpolated_traces_epochs']):
        for j in req_epochs:
        #     k[j].append(traces[j])
            # axs[j].plot(fly['interpolated_time'][0][j],[np.mean(z, axis = 0) for z in k][j],lw=3)
            trace = klist[j]
            mean_r, std_r, ub, lb = make_plots_with_CI(trace)
            axs[j].fill_between(fly['interpolated_time'][0][j], ub, lb, alpha=.4, color = mycolor[k])
            axs[j].plot(fly['interpolated_time'][0][j],mean_r,lw=1, color = mycolor[k])
    # plt.show()
    return
# flyupd = response_traces_aligned_plot(upd6, layer='B')

###########################################################
"""resp traces aligned plots averaged across individuals"""
############################################################

def resptraces_aligned_plot_avgind(dataframe, layer = str):
    mycolor=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5']
    flyID = np.unique(dataframe['FlyId']) 
    geno = np.unique(dataframe["genotype"][0])
    flies = filter_interpolated_flyID(dataframe, flyID=None, layer = layer)
    fig = plt.figure()
    fig.suptitle(f'N = {len(flyID)}: For layer "{layer}", Genotype: {geno}')
    fig.supxlabel('Time(s)')
    fig.supylabel('Response (df/f)')
    gs = fig.add_gridspec(len(req_epochs), hspace=0.2)
    axs = gs.subplots(sharex=True, sharey=True)
    klist = create_sorted_lists(flies['interpolated_traces_epochs'], req_epochs)
    for j in req_epochs:
    #     k[j].append(traces[j])
        # axs[j].plot(fly['interpolated_time'][0][j],[np.mean(z, axis = 0) for z in k][j],lw=3)
        trace = klist[j]
        mean_r, std_r, ub, lb = make_plots_with_CI(trace)
        axs[j].fill_between(flies['interpolated_time'][0][j], ub, lb, alpha=.4, color = mycolor[j])
        axs[j].plot(flies['interpolated_time'][0][j],mean_r,lw=1, color = mycolor[j])
    # plt.show() 

    return
# upd6_ind = resptraces_aligned_plot_avgind(upd6, layer="A")
# upd6_ind = resptraces_aligned_plot_avgind(upd6, layer="B")
# upd6_ind = resptraces_aligned_plot_avgind(upd6, layer="C")
# upd6_ind = resptraces_aligned_plot_avgind(upd6, layer="D")

def filter_layers(df, layer):
    df = df[df['categories']==layer]
    df = df.reset_index(drop = True)
    return df

<<<<<<< Updated upstream
upd6A = filter_layers(upd6, layer="A")
upd6B = filter_layers(upd6, layer="B")
upd6C = filter_layers(upd6, layer="C")
upd6D = filter_layers(upd6, layer="D")

# plt.figure()
# ax = plt.subplot(projection='polar')
# # upd6on['max_response_ON']
# # upd6on['dir_max_resp']
# for ipts, pts in enumerate(upd6A['max_response_ON']):

#     theta = np.deg2rad(upd6A['dir_max_resp'][ipts])
#     r = upd6A['max_response_ON'][ipts]/np.max(upd6A['max_response_ON'])
#     ax.annotate("", 
#                 xytext=(0.0,0.0), 
#                 xy=(theta,r),
#                 arrowprops=dict(facecolor='black'))
# plt.show()    

if CS_filter[0]==True:
    ON_column=upd6['CS']=='ON'
    OFF_column=upd6['CS']=='OFF'
    upd6['ON_responding']=ON_column
    upd6['OFF_responding']=OFF_column

pac.plot_roivectors(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities,treshold=reliability_treshold)
pac.plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold)
pac.plot_average_vectors_across_rois(upd6,results_save_dir, z_bin=z_bin, filter_polarity=False)
# pac.plot_variable_histogram_pertreatment(upd6, results_save_dir)
pac.plot_roivectors_perfly(upd6, results_save_dir,z_bin=z_bin,rotation=rotation)
'a'
=======
#############################################################################################################################





 

depths = [30, 40, 50, 60, 70, 80]

for ibins, bins in enumerate(depths):
    results_save_dir = r'U:\Work_PT\Plots\upd_flies_multi_depth\diff_reliability\75deg\d'+str(bins)
    filter_polarities = False
    z_bin = bins
    reliability_treshold = 0.5
    
    rotation = None
    datapath = r'C:\Users\ptrimbak\Work_PT\2p_data\240416_T4T5_upd_depths\rotation_75deg\d'+str(bins)





# datapath = r'U:\Work_PT\2p_data\240124_Gcamp_exp\data_processed'


    datasets_to_load = []
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if 'sec.pickle' in file:
                datasets_to_load.append(os.path.join(root, file))

    # datasets_to_load = os.listdir(datapath)
    data= [[] for i in range(len(datasets_to_load))]
    for idx, dataset in enumerate(datasets_to_load):
        
        # load_path = os.path.join(datapath, dataset)
        load_path = dataset
        load_path = open(load_path, 'rb')
        workspace = pickle.load(load_path)

        curr_rois = workspace['final_rois']
        curr_rois=pmc.extract_null_dir_response(curr_rois)
        interpolate_data(curr_rois)
        req_epochs = np.where(np.array(curr_rois[0].stim_info['stimtype'])=='driftingstripe')[0]
        for icx, epoch in enumerate(req_epochs):
            ROI_mod.align_traces_per_epoch(curr_rois, epoch, grating=False)
        
        
        for roi in curr_rois: 
            entry = {
                    # 'uniq_id': roi.uniq_id,
                    # 'independent_var': getattr(roi, 'independent_var', np.nan),
                    # 'interpolated_traces': getattr(roi, 'interpolated_traces_epochs', np.nan),
                    # 'interpolated_time': getattr(roi, 'interpolated_time', np.nan),
                    # 'max_resp_all_epochs': getattr(roi, 'max_resp_all_epochs', np.nan),
                    'max_resp_all_epoch_on':getattr(roi, 'max_resp_all_epochs_ON', np.nan),
                    'max_resp_all_epoch_off':getattr(roi, 'max_resp_all_epochs_OFF', np.nan),
                    # 'freq_power_epochs': getattr(roi, 'freq_power_epochs', np.nan),  # Use np.nan if attribute does not exist
                    'FlyId': roi.experiment_info['FlyID'],  # Include FlyId attribute
                    # 'ind var values': getattr(roi, 'independent_var_vals', np.nan),  # Include independent_var_vals attribute
                    # 'treatment': roi.experiment_info['treatment'],
                    'raw_trace': getattr(roi, 'raw_trace', np.nan),
                    'genotype': roi.experiment_info['Genotype'],
                    'categories': roi.category[-1],
                    'reliability_PD': getattr(roi, 'reliability_PD', np.nan), 
                    'reliability_ON': getattr(roi, 'reliability_PD_ON', np.nan),
                    'reliability_OFF': getattr(roi, 'reliability_PD_OFF', np.nan),
                    'DSI_ON': getattr(roi, 'DSI_ON', np.nan), #
                    'DSI_OFF': getattr(roi, 'DSI_OFF', np.nan),#
                    'CS': getattr(roi, 'CS', np.nan),
                    'CSI': getattr(roi, 'CSI',np.nan),#
                    'whole_traces_all_epochsTrials': getattr(roi, 'whole_traces_all_epochsTrials', np.nan), 
                    'df_trace': getattr(roi, 'df_trace', np.nan), 
                    'dir_max_resp': getattr(roi, 'dir_max_resp', np.nan),
                    'PD_ON': getattr(roi, 'PD_ON', np.nan),#
                    'PD_OFF': getattr(roi, 'PD_OFF', np.nan),#
                    'direction_vector':getattr(roi, 'direction_vector', np.nan),
                    'max_response_ON':getattr(roi, 'max_response_ON', np.nan), #
                    'max_response_OFF':getattr(roi, 'max_response_OFF', np.nan), #
                    'norm_null_dir_resp_ON':getattr(roi, 'norm_null_dir_resp_ON', np.nan),
                    'norm_null_dir_resp_ON':getattr(roi, 'norm_null_dir_resp_ON', np.nan),
                    'interpolated_traces_epochs': getattr(roi, 'interpolated_traces_epochs', np.nan), 
                    'interpolated_time': getattr(roi, 'interpolated_time', np.nan),
                    'treatment': roi.experiment_info['treatment'],
                    'sex': roi.experiment_info['Sex'],
                    'depth': roi.experiment_info['z_depth'],
                    'depth_bin': roi.experiment_info['z_depth_bin'],
                    'rotations': roi.experiment_info['rotation'],
                }
            data[idx].append(entry)
        
    upd6 = make_genotype_df(data, genotype='Upd_6f', CS_str=None)
    upd6['treatment'] = 'Upd;Gcamp6f/Cyo'
    CS_filter = [True]
    if CS_filter[0]==True:
        ON_column=upd6['CS']=='ON'
        OFF_column=upd6['CS']=='OFF'
        upd6['ON_responding']=ON_column
        upd6['OFF_responding']=OFF_column


    pac.plot_roivectors(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities,treshold=reliability_treshold)
    pac.plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold)


    pac.plot_roivectors_perfly(upd6, results_save_dir,z_bin=z_bin,rotation=rotation)
    pac.plot_average_vectors_across_rois(upd6,results_save_dir, z_bin=z_bin, filter_polarity=False)
    'a'
>>>>>>> Stashed changes
