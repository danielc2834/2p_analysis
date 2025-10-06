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
import ROI_mod
import post_analysis_core as pac
import itertools
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use('seaborn-v0_8-talk')




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
    df1 = pd.concat(only_genotype)
    if CS_str == None:
        df1 = df1
    else:
        df1 = df1[df1['reliability_PD'] >= 0.4]
        df1 = df1[df1['category']!="No_category"]
        df1 = df1[df1['CS']==CS_str]
        df1 = df1.reset_index(drop = True)
    return df1


# datapath = r'U:\Work_PT\2p_data\240124_Gcamp_exp\data_processed'
datapath = r'C:\Users\ptrimbak\Work_PT\2p_data\240124_Gcamp_exp'

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
    interpolate_data(curr_rois)
    req_epochs = np.where(np.array(curr_rois[0].stim_info['stimtype'])=='driftingstripe')[0]
    for icx, epoch in enumerate(req_epochs):
        ROI_mod.align_traces_per_epoch(curr_rois, epoch, grating=False)
    
    
    for roi in curr_rois: 
        entry = {
                'uniq_id': roi.uniq_id,
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
                'category': roi.category[-1],
                'reliability_PD': getattr(roi, 'reliability_PD', np.nan), 
                'reliability_ON': getattr(roi, 'reliability_PD_ON', np.nan),
                'reliability_OFF': getattr(roi, 'reliability_PD_OFF', np.nan),
                'DSI_ON': getattr(roi, 'DSI_ON', np.nan),
                'DSI_OFF': getattr(roi, 'DSI_OFF', np.nan),
                'CS': getattr(roi, 'CS', np.nan),
                'CSI': getattr(roi, 'CSI',np.nan),
                'whole_traces_all_epochsTrials': getattr(roi, 'whole_traces_all_epochsTrials', np.nan), 
                'df_trace': getattr(roi, 'df_trace', np.nan), 
                'dir_max_resp': getattr(roi, 'dir_max_resp', np.nan),
                'PD_ON': getattr(roi, 'PD_ON', np.nan),
                'PD_OFF': getattr(roi, 'PD_OFF', np.nan),
                'direction_vector':getattr(roi, 'direction_vector', np.nan),
                'max_response_ON':getattr(roi, 'max_response_ON', np.nan),
                'max_response_OFF':getattr(roi, 'max_response_OFF', np.nan),
                'interpolated_traces_epochs': getattr(roi, 'interpolated_traces_epochs', np.nan), 
                'interpolated_time': getattr(roi, 'interpolated_time', np.nan)
            }
        data[idx].append(entry)
    

gcamp6fon = make_genotype_df(data, genotype='Gcamp6f', CS_str='ON')
gcamp6foff = make_genotype_df(data, genotype='Gcamp6f', CS_str='OFF')

gcamp8fon = make_genotype_df(data,genotype='Gcamp8f', CS_str= 'ON')
gcamp8foff = make_genotype_df(data,genotype='Gcamp8f', CS_str='OFF')

gcamp7fon = make_genotype_df(data,genotype='Gcamp7f', CS_str= 'ON')
gcamp7foff = make_genotype_df(data,genotype='Gcamp7f', CS_str= 'OFF')

gcamp8mon = make_genotype_df(data,genotype='Gcamp8m', CS_str= 'ON')
gcamp8moff = make_genotype_df(data,genotype='Gcamp8m', CS_str= 'OFF')


def filter_ID(df, flyID = None, resp = None):
    df = df[df['FlyId']==flyID]
    df = df.reset_index(drop = True)
    arra = df[resp]
    # arra= np.array(arra)
    return arra


# boxplot

box_param = dict(whis=(5, 95), widths=0.2, patch_artist=True,
                 flierprops=dict(marker='.', markeredgecolor='black',
                 fillstyle=None), medianprops=dict(color='black'))

box6fon = gcamp6fon['max_response_ON']
box8fon = gcamp8fon['max_response_ON']
box7fon = gcamp7fon['max_response_ON']
box8mon = gcamp8mon['max_response_ON']

box6f = gcamp6foff['max_response_OFF']
box8f = gcamp8foff['max_response_OFF']
box7f = gcamp7foff['max_response_OFF']
box8m = gcamp8moff['max_response_OFF']

fig3 = plt.figure()
ax = plt.subplot()
boxes = [box6fon, box6f,
    box8fon,box8f, 
    box7fon,box7f, 
    box8mon,box8m]
ax.boxplot(boxes, **box_param)
# for i, j in enumerate(boxes):
#     y = boxes[i]
#     x = np.random.normal(1+i, 0.04, size=y.shape[0])
#     ax.plot(x, y, 'r.', alpha=0.2)
ax.set_title('Max response for all individuals')
ax.set_xticklabels(['6f ON','6f OFF', 
                    '8f ON','8f OFF', 
                    '7f ON','7f OFF', 
                    '8m ON','8m OFF'])

# plt.show()
# fig3.savefig('boxplot_all_ind.png', bbox_inches='tight')

##################################
"BOX per individual"
##################################
def box_plot_per_idv (data, genotype):
    on_data = make_genotype_df(data= data, genotype=genotype, CS_str='ON')
    off_data = make_genotype_df(data=data, genotype=genotype, CS_str='OFF')
    full_data  = pd.concat([on_data, off_data], axis=0)
    fly_IDs = list(np.unique(on_data['FlyId']))

    boxdfon = []
    boxdfoff = []
    for ifly, fly in enumerate(fly_IDs):
        boxdfon.append(filter_ID(on_data, fly, resp='max_response_ON'))
        boxdfoff.append(filter_ID(on_data, fly, resp='max_response_OFF'))
    # fig4 = plt.figure()
    # ax = plt.subplot()
    
    ax = full_data.boxplot(by='FlyId', column=['max_response_ON', 'max_response_OFF'], rot= 35, grid=False, return_type= None, **box_param)
    # for i, j in enumerate(fly_IDs):
    #     y0 = full_data.max_response_ON[full_data.FlyId==i]
    #     x0 = np.random.normal(1+i, 0.04, size=y0.shape[0])
    #     y1 = full_data.max_response_OFF[full_data.FlyId==i]
    #     x1 = np.random.normal(1+i, 0.04, size=y1.shape[0])
    #     plot(x0, y0, 'r.', alpha=0.2)
    #     ax[1].plot(x1, y1, 'r', alpha = 0.2)
        
    # ax.boxplot(pairing)
    ax[0].set_title(f'{genotype}: max_response_ON')
    ax[1].set_title(f'{genotype}: max_response_OFF')
   
    ax[0].set_ylim(0, 8)
    ax[1].set_ylim(0, 8)
    # ax[1].set_yticks(ticks = ax[0].get_yticks)
    

    # plt.show()
    return 

f6_ind = box_plot_per_idv(data=data, genotype='Gcamp6f')
f7_ind = box_plot_per_idv(data=data, genotype='Gcamp7f')
f8_ind = box_plot_per_idv(data=data, genotype='Gcamp8f')
m8_ind = box_plot_per_idv(data=data, genotype='Gcamp8m')


# plotting raw trace to get the baseline
base6f = make_genotype_df(data, genotype='Gcamp6f', CS_str=None)
base8f = make_genotype_df(data, genotype='Gcamp8f', CS_str= None)
base7f = make_genotype_df(data, genotype='Gcamp7f', CS_str= None)
base8m = make_genotype_df(data, genotype='Gcamp8m', CS_str= None)

base6_arr= base6f['raw_trace'].tolist()
base8_arr= base8f['raw_trace'].tolist()
base7_arr = base7f['raw_trace'].tolist()
base8m_arr = base8m['raw_trace'].tolist()

def make_fig_base(lst):
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_xlim(0, 50)
    for x, arr in lst.items():
        mean_r = np.mean(arr, axis=0)
        std_r = np.std(arr, axis=0)/ np.sqrt(51)
        ub = mean_r + std_r
        lb = mean_r - std_r
        ax.fill_between(range(len(mean_r)), ub, lb, alpha=.4)
        ax.plot(range(len(mean_r)),mean_r,lw=3, label = x)
    ax.legend()
    return fig

lt = {'6f': base6_arr, '8f' : base8_arr, '7f' : base7_arr, '8m': base8m_arr}

make_fig_base(lt)

# multipage('multipage.pdf')
#####response traces normalised####
def filter_interpolated_flyID(df, flyID, layer):
    if flyID == None:
        df = df[df['category']==layer]
        df = df.reset_index(drop = True)
        arra = df[['interpolated_traces_epochs','interpolated_time']]
    else:    
        df = df[df['FlyId']==flyID]
        df = df[df['category']==layer]
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
# fly6f = response_traces_aligned_plot(base6f, layer='B')

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
fly6f_ind = resptraces_aligned_plot_avgind(base8f, layer="A")
fly7f_ind = resptraces_aligned_plot_avgind(base8f, layer="B")
fly8f_ind = resptraces_aligned_plot_avgind(base8f, layer="C")
fly8m_ind = resptraces_aligned_plot_avgind(base8f, layer="D")




#%%
############################################################
"""Extraction of hyperpolarisation and quantifying rise times"""
############################################################

def resptraces_aligned_plot_hyp(dataframe, layer = str):
    mycolor=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5']
    flyID = np.unique(dataframe['FlyId']) 
    flies = filter_interpolated_flyID(dataframe, flyID=None, layer = layer)
    fig = plt.figure()
    fig.suptitle(f'N = {len(flyID)}: For layer "{layer}"')
    fig.supxlabel('Time(s)')
    fig.supylabel('Response (df/f)')
    # gs = fig.add_gridspec(0, hspace=0.2)
    axs = fig.subplots()
    klist = create_sorted_lists(flies['interpolated_traces_epochs'], req_epochs)
    # for j in req_epochs[0]:
    #     k[j].append(traces[j])
        # axs[j].plot(fly['interpolated_time'][0][j],[np.mean(z, axis = 0) for z in k][j],lw=3)
    trace = klist[0]
    mean_r, std_r, ub, lb = make_plots_with_CI(trace)
    # axs[j].fill_between(flies['interpolated_time'][0][j], ub, lb, alpha=.4, color = mycolor[j])
    axs.plot(flies['interpolated_time'][0][0],mean_r,lw=1, color = mycolor[-1])

# hyp = resptraces_aligned_plot_hyp(base8f, layer = 'A')


"""OLD CODE"""
# flyID8f = np.unique(base8m['FlyId']) 
# # fly18f = filter_flyID(base8f, flyID8f[1], layer='D')
# # fly17f = filter_flyID(base7f, flyID8f[1])
# # fly18m = filter_flyID(base8m, flyID8f[2])

# # fig = plt.figure()
# # gs = fig.add_gridspec(len(req_epochs), hspace=0.2)
# # axs = gs.subplots(sharex=True, sharey=True)
# for k, s in enumerate(flyID8f):
#     fly18f = filter_interpolated_flyID(base8m, flyID8f[k], layer='D')
#     fig = plt.figure()
#     gs = fig.add_gridspec(len(req_epochs), hspace=0.2)
#     axs = gs.subplots(sharex=True, sharey=True)
#     for tra, traces in enumerate(fly18f['interpolated_traces_epochs']):
#         for j in req_epochs:
#             axs[j].plot(fly18f['interpolated_time'][tra][j],traces[j],lw=3)
# plt.show()
# 'a'


# %%
