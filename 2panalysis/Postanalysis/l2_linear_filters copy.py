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
code_path = r'U:\Work_PT_server\Code\2p_analysis_development\2panalysis\Helpers'
sys.path.insert(0, code_path) 
import ROI_mod
import post_analysis_core as pac
import process_mov_core as pmc
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import matplotlib as mpl
plt.style.use('seaborn-v0_8-talk')

#####__________________Parameters________________###########

results_save_dir = r'C:\Users\ptrimbak\Work_PT\2p_data\241014_L2_linear_filters\241014_L2_linear_filters\50ms'
filter_polarities = False
z_bin = None
reliability_treshold = 0.5
CS_filter = [True]
rotation = None
cats_to_exclude=['C', 'A', 'B']
datapath = r'C:\Users\ptrimbak\Work_PT\2p_data\241014_L2_linear_filters\241020_L2_filter_circle\50ms\processed'

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
        # df1 = df1[df1['reliability_PD'] >= 0.5]
    else:
        df1 = df1[df1['reliability_PD'] >= 0.5]
        df1 = df1[df1['categories']!="No_category"]
        df1 = df1[df1['CS']==CS_str]
        df1 = df1.reset_index(drop = True)
    return df1


# datapath = r'U:\Work_PT\2p_data\240124_Gcamp_exp\data_processed'


datasets_to_load = []
for root, dirs, files in os.walk(datapath):
    for file in files:
        if 'PT.pickle' in file:
            datasets_to_load.append(os.path.join(root, file))

# datasets_to_load = os.listdir(datapath)
data= [[] for i in range(len(datasets_to_load))]
for idx, dataset in enumerate(datasets_to_load):
    
    # load_path = os.path.join(datapath, dataset)
    load_path = dataset
    load_path = open(load_path, 'rb')
    workspace = pickle.load(load_path)

    curr_rois = workspace['final_rois']
    # curr_rois=pmc.extract_null_dir_response(curr_rois)
    # interpolate_data(curr_rois)
    # req_epochs = np.where(np.array(curr_rois[0].stim_info['stimtype'])=='driftingstripe')[0]
    # for icx, epoch in enumerate(req_epochs):
        # ROI_mod.align_traces_per_epoch(curr_rois, epoch, grating=False)
    
    
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
                # 'reliability_PD': getattr(roi, 'reliability_PD', np.nan), 
                # 'reliability_ON': getattr(roi, 'reliability_PD_ON', np.nan),
                # 'reliability_OFF': getattr(roi, 'reliability_PD_OFF', np.nan),
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
                'strf': roi.STRF_data,
                # 'mean_strf': roi.STRF_data['mean_strf'],
                'roi_count': len(curr_rois)
            }
        data[idx].append(entry)
    


L2_df = make_genotype_df(data, genotype='L2Gal4_Gcamp6f', CS_str=None)

roi_counts = np.unique(L2_df['roi_count'], return_counts=True)

strf_lst = L2_df['strf']

strf_lst = [k for k in strf_lst if not np.isnan(np.sum(k))]
def make_fig_base(arr, roi_counts):
    fig = plt.figure()
    ax = plt.subplot()
    # ax.set_xlim(0, 50)
    mean_r = np.mean(arr, axis=0)
    std_r = np.std(arr, axis=0)/ np.sqrt(len(arr[0]))
    ub = mean_r + std_r
    lb = mean_r - std_r
    x_ax = range(len(mean_r))
    ax.fill_between(np.array(x_ax)*0.05, ub, lb, alpha=.4)
    
    ax.plot(np.array(x_ax)*0.05,mean_r,lw=3, label = f"N = {len(np.unique(L2_df['FlyId']))}(rois: {np.sum(roi_counts)})", marker = 'o')
    # ax.legend()
    return fig

def filter_ID(df, flyID = None, resp = None):
    df = df[df['FlyId']==flyID]
    df = df.reset_index(drop = True)
    arra = np.max(df['roi_count'])
    # arra= np.array(arra)
    return arra, df


def make_fig_for_all_roi(lst):
    fig = plt.figure()
    ax = plt.subplot()
    for k in lst:
        x_ax = range(len(k))
        ax.plot(np.array(x_ax)*0.05, k)
    # ax.legend(f'N = {len(lst)}')
    return fig    
flyids = np.unique(L2_df['FlyId'])

for idx, fly in enumerate(flyids):
    counts1, fly1 = filter_ID(L2_df, flyID = fly)
    strf_lst = fly1['strf']
    fly1_strf = [k for k in strf_lst if not np.isnan(np.sum(k))]
    # plt.figure()
    fig = make_fig_base(np.flip(fly1_strf), counts1)
    fig2 = make_fig_for_all_roi(np.flip(fly1_strf))
    # fig.savefig(results_save_dir+f'fly_{fly}_strf.svg')
    # fig2.savefig(results_save_dir+f'fly_{fly}_strf_all.svg')
    # multipage(results_save_dir+f'fly_{fly}_strf.pdf', figs = [fig, fig2])




roi_counts = []
for ids in flyids:
    counts = filter_ID(L2_df, ids)
    roi_counts.append(counts)
    

make_fig_base(strf_lst, roi_counts)
make_fig_for_all_roi(strf_lst[15:36])
plt.show()
'a'

