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
import random
plt.style.use('seaborn-v0_8-talk')



results_save_dir = r'U:\Work_PT_server\Plots\Cross_check_with_miriam'
filter_polarities = False
z_bin = None
reliability_treshold = 0.5
CS_filter = [True]
rotation = [15, 30]#[15]#[None, None, None, None, None, None, None]#[0, 15, 30, 45, 60, 75, 90]
cats_to_exclude=[ 'C', 'B', 'D']
datapath_0deg = r'C:\Users\ptrimbak\Work_PT\2p_data\240920_rotation_test\all_rot_contrast_diff_only' #C:\Users\ptrimbak\Work_PT\2p_data\240920_rotation_test\all_rot_data_small_clusters
# datapath_75deg = r'C:\Users\ptrimbak\Work_PT\2p_data\240701_upd_wt_perspective\upd_exp\75deg'
genotype = 'control_6f_T4T5'



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
def make_genotype_df(data, genotype,reliability_treshold,  CS_str = None):
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
        df1 = df1[df1['reliability_PD'] >= reliability_treshold]
    else:
        df1 = df1[df1['reliability_PD'] >= reliability_treshold]
        df1 = df1[df1['categories']!="No_category"]
        df1 = df1[df1['CS']==CS_str]
        df1 = df1.reset_index(drop = True)
    return df1


def iterate_and_give_rot_vectors(dataframe, pd_name, rotation):
    coords_x = np.zeros(len(dataframe))
    coords_y = np.zeros(len(dataframe))
    directions = np.zeros(len(dataframe))

    for idx in range(len(dataframe)):
        direction= dataframe.iloc[idx,:][pd_name]
            
        RF_coord = dataframe.iloc[idx,:]['RF_center_coords']

        if dataframe['rotations'].iloc[idx] == rotation:
            coords_x[idx] = RF_coord[1]
            coords_y[idx] = RF_coord[0]
            directions[idx] = direction
    
    return np.trim_zeros(coords_x), np.trim_zeros(coords_y), np.trim_zeros(directions)

def add_jitter(data, jitter=0.1):
    return data + np.random.uniform(-jitter, jitter, size=len(data))


def plot_RFonScreen(roi_dataframe,savedir,z_bin=None,rotation=[],filter_polarity=True,treshold=None, cats_to_exclude = []):
    '''makes quiver polar plots from a dataframe containing categories, treatments, and 
       contrast selectivity information. The output is a polarplot pertreatment and per polarity
       if there are 2 treatments at the end you get 4 plots'''
    
    #filter z_layers if necesary
    if z_bin is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['depth_bin']==z_bin]
        #TODO automatize depth_bin selection
    


    #get mean vector per category (lobula plate layer)
        # get the mean vector and plot it for every layer of the lobplate 
    
    #check that the categories are not more than 4 or less than 4

    
    cats_to_exclude=cats_to_exclude
    check=0
    #if len(np.unique(roi_dataframe['categories']))!=4:
    #    raise Exception ('labels_are wrong check')
    for cats in np.unique(roi_dataframe['categories'].values):
        if ('_a' in cats) or ('LPA' in cats) or ('lpa'in cats) or ('A' in cats) :
            check+=1
        elif ('_b' in cats) or ('LPB' in cats) or ('lpb'in cats) or ('B' in cats):
            check+=1
        elif ('LPC' in cats) or ('lpc'in cats) or ('C' in cats):
            check+=1
        elif ('_d' in cats) or ('LPD' in cats) or ('lpd'in cats) or ('D' in cats):
            check+=1
        else:
            cats_to_exclude.append(cats)
    
    if check!=4 :
        raise Exception ('cat labels_are wrong check')
    
    temp_df=roi_dataframe.copy(deep=True)
    for cats in cats_to_exclude:
        temp_df=temp_df.loc[temp_df['categories']!=cats]

    mean_vector_df={'treatment':[],'layer':[],'direction':[],'length':[],'polarity':[]}
       
    
    

    #count number of treatments and categories. also extract their values
    treatment_number=len(np.unique(temp_df['treatment']))
    treatments=np.unique(temp_df['treatment'])
    category_number=len(np.unique(temp_df['categories']))
    categories=np.unique(temp_df['categories'][temp_df['categories']!='No_category'])

    # if type(np.array(temp_df['categories'])[0])==set:
    #     for idx in temp_df['categories'].index:
    #         if idx%100==0:
    #             print('%s // %s' %(idx,roi_dataframe['categories'].index[-1]))
    #         temp_df['categories'][idx]=list(roi_dataframe['categories'][idx])[0]

    # temp_df=temp_df.loc[roi_dataframe['categories']!='BG']    

    # turn string entries in dataframe to lowercase
    #temp_df['treatment']=temp_df['treatment'].str.lower()
    #temp_df['categories']=temp_df['categories'].str.upper()

    
    #create plot scaffold
    fig1=plt.figure(figsize=(12,12))
    gs = plt.GridSpec(2, treatment_number)
    treatment_polarity_pairs=itertools.product([0,1],range(treatment_number))
    plt.style.use('seaborn-v0_8-talk')

    #plot the data
    


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
        ax1=fig1.add_subplot(gs[pair[0],pair[1]])
        number_of_flies=len(np.unique(treatment_pol_subset['FlyId']))


        roi_count = np.sum(np.unique(treatment_pol_subset['roi_count']))/len(np.unique(treatment_pol_subset['FlyId']))
        title= treatment + ' total n_: %s %s rois ' %(number_of_flies, int(np.round(roi_count)))
        ax1.set_ylim(-50, 50)
        ax1.set_xlim(-20,120)
        # ax1.set_yticks([0.2,0.4,0.6, 0.8, 1.0])
        ax1.set_title(title,pad=15,fontweight='bold',fontsize=13)
        ax1.set_ylabel(polarity[0:3],labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
        
        # ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
        # scale=1,angles="xy",scale_units='xy',alpha=0.7)    
        
        #"backup"
        widths =0.001 #was 0.001
        # plt.quiver(RF_coord[0], RF_coord[1], u , v, angles="uv", scale=.6, scale_units='xy', alpha=0.7, color = color, width=widths, headwidth=5, headlength=7)
        # widths =0.002
        my_colors = iter([plt.cm.tab10(i) for i in range(8)])
        colors = list(my_colors)
        my_colors2 = iter([plt.cm.viridis_r(i) for i in range(8)])
        colors2 = list(my_colors2)
        rotation_reshaped = np.array(rotation).reshape(1, -1)

    
        for rota in rotation:
            coords_x, coords_y, direction = iterate_and_give_rot_vectors(treatment_pol_subset, PD_name, rota)
            coords_x = coords_x - 80
            coords_y = coords_y - 80    

            u = 1*np.cos(np.radians(direction))
            v = 1*np.sin(np.radians(direction))
            if rota == rotation[0]:
                coords_x = coords_x +rotation[0]
                # coords_x2 = np.where(coords_x>30)
                ax1.quiver(coords_x, coords_y, u , v, scale = 0.3, scale_units='xy', alpha=0.5, color = colors[0], width=widths, cmap = 'tab10')    
                # ax1.quiver(coords_x[coords_x2], coords_y[coords_x2], u[coords_x2] , v[coords_x2], scale = 0.3, scale_units='xy', alpha=1, color = colors[0], width=widths, cmap='viridis_r')
            elif rota == rotation[1]:
                coords_x = coords_x +rotation[1]
                # coords_x2 = np.where(coords_x>35)
                
                ax1.quiver(coords_x, coords_y, u , v, scale = 0.3, scale_units='xy', alpha=0.5, color = colors[1], width=widths)
                # ax1.quiver(coords_x[coords_x2], coords_y[coords_x2], u[coords_x2] , v[coords_x2], scale = 0.3, scale_units='xy', alpha=1, color = colors[1], width=widths)
            elif rota == rotation[2]:
                coords_x = coords_x +rotation[2]
                # coords_x2 = np.where(coords_x>45)
                
                ax1.quiver(coords_x, coords_y, u , v, scale = 0.3, scale_units='xy', alpha=0.5, color = colors[2], width=widths)
                # ax1.quiver(coords_x[coords_x2], coords_y[coords_x2], u[coords_x2] , v[coords_x2], scale = 0.3, scale_units='xy', alpha=1, color = colors[2], width=widths)
            elif rota == rotation[3]:
                coords_x = coords_x +rotation[3]
                # coords_x2 = np.where(coords_x>60)
                
                ax1.quiver(coords_x, coords_y, u , v, scale = 0.3, scale_units='xy', alpha=0.5, color = colors[3], width=widths)
                # ax1.quiver(coords_x[coords_x2], coords_y[coords_x2], u[coords_x2] , v[coords_x2], scale = 0.3, scale_units='xy', alpha=1, color = colors[3], width=widths)
            elif rota == rotation[4]:
                coords_x = coords_x +rotation[4]
                # coords_x2 = np.where(coords_x>80)
                
                ax1.quiver(coords_x, coords_y, u , v, scale = 0.3, scale_units='xy', alpha=0.5, color = colors[4], width=widths)
                # ax1.quiver(coords_x[coords_x2], coords_y[coords_x2], u[coords_x2] , v[coords_x2], scale = 0.3, scale_units='xy', alpha=1, color = colors[4], width=widths)
            elif rota == rotation[5]:
                coords_x = coords_x +rotation[5]
                # coords_x2 = np.where(coords_x>95)
                
                ax1.quiver(coords_x, coords_y, u , v, scale = 0.3, scale_units='xy', alpha=0.5, color = colors[5], width=widths)
                # ax1.quiver(coords_x[coords_x2], coords_y[coords_x2], u[coords_x2] , v[coords_x2], scale = 0.3, scale_units='xy', alpha=1, color = colors[5], width=widths)
            elif rota == rotation[6]:
                coords_x = coords_x +rotation[6]
                # coords_x2 = np.where(coords_x>110)
                
                ax1.quiver(coords_x, coords_y, u , v, scale = 0.3, scale_units='xy', alpha=0.5, color = colors[6], width=widths)
                # ax1.quiver(coords_x[coords_x2], coords_y[coords_x2], u[coords_x2] , v[coords_x2], scale = 0.3, scale_units='xy', alpha=1, color = colors[6], width=widths)
            

    
        # for idx in range(len(treatment_pol_subset.index)):
            
        #     direction= treatment_pol_subset.iloc[idx,:][PD_name]
            
        #     RF_coord = treatment_pol_subset.iloc[idx,:]['RF_center_coords']
            
            
        #     roi_count = np.sum(np.unique(roi_dataframe['roi_count']))/len(np.unique(roi_dataframe['FlyId']))
        #     title= treatment + ' total n_: %s %s rois ' %(number_of_flies, int(np.round(roi_count)))
        #     ax1.set_ylim(-40, 40)
        #     ax1.set_xlim(-20,200)
        #     # ax1.set_yticks([0.2,0.4,0.6, 0.8, 1.0])
        #     ax1.set_title(title,pad=15,fontweight='bold',fontsize=13)
        #     ax1.set_ylabel(polarity[0:3],labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
        #     # ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
        #     # scale=1,angles="xy",scale_units='xy',alpha=0.7)    
        #     u = 1*np.cos(np.radians(direction))
        #     v = 1*np.sin(np.radians(direction))
        #     #"backup"
        #     widths =0.001
        #     # plt.quiver(RF_coord[0], RF_coord[1], u , v, angles="uv", scale=.6, scale_units='xy', alpha=0.7, color = color, width=widths, headwidth=5, headlength=7)
        #     # widths =0.002
        #     my_colors = iter([plt.cm.tab20(i) for i in range(6)])
        #     colors = list(my_colors)

        #     if treatment_pol_subset['rotations'].iloc[idx] == rotation[0]:
        #         plt.quiver(RF_coord[0]-60, RF_coord[1]-80, u , v, scale = 0.5, scale_units='xy', alpha=0.95, color = colors[0], width=widths)
        #     elif treatment_pol_subset['rotations'].iloc[idx] == rotation[1]:
        #         plt.quiver(RF_coord[0]-60+rotation[1], RF_coord[1]-80, u , v, scale = 0.5, scale_units='xy', alpha=0.85, color = colors[1], width=widths)
        #     elif treatment_pol_subset['rotations'].iloc[idx] == rotation[2]:
        #         plt.quiver(RF_coord[0]-60+rotation[2], RF_coord[1]-80, u , v, scale = 0.5, scale_units='xy', alpha=0.75, color = colors[2], width=widths)
        #     elif treatment_pol_subset['rotations'].iloc[idx] == rotation[3]:
        #         plt.quiver(RF_coord[0]-60+rotation[3], RF_coord[1]-80, u , v, scale = 0.5, scale_units='xy', alpha=0.65, color = colors[3], width=widths)
        #     elif treatment_pol_subset['rotations'].iloc[idx] == rotation[4]:
        #         plt.quiver(RF_coord[0]-60+rotation[4], RF_coord[1]-80, u , v, scale = 0.5, scale_units='xy', alpha=0.55, color = colors[4], width=widths)
        #     elif treatment_pol_subset['rotations'].iloc[idx] == rotation[5]:
        #         plt.quiver(RF_coord[0]-60+rotation[5], RF_coord[1]-80, u , v, scale = 0.5, scale_units='xy', alpha=0.45, color = colors[5], width=widths)
        #     elif treatment_pol_subset['rotations'].iloc[idx] == rotation[6]:
        #         plt.quiver(RF_coord[0]-60+rotation[6], RF_coord[1]-80, u , v, scale = 0.5, scale_units='xy', alpha=0.35, color = colors[6], width=widths)            

            # plt.quiver(RF_coord[0], RF_coord[1], u , v,  units='x', 
            #    scale=1 / 0.15, color = color)
        # plot mean
        # mean_df_subset=mean_vector_df.loc[mean_vector_df['polarity']==polarity]
        # mean_df_subset=mean_df_subset.loc[mean_df_subset['treatment']==treatment]
        # local_lenghts=np.array(mean_df_subset['length'])
        # local_dirs=np.array(mean_df_subset['direction'])
        # for divs, divecs in enumerate(local_dirs):
        #     ax1.quiver(0,0, np.radians(divecs),local_lenghts[divs],color='black',
        #         scale=1,angles="xy",scale_units='xy',alpha=1) 
        # plt.subplots_adjust(left=0.1,
        #             bottom=0.1, 
        #                 right=0.9, 
        #                 top=0.93, 
        #                 wspace=0.015, 
        #                 hspace=0.36)

    #for layer in np.unique(treatment_pol_subset.loc('_a' in treatment_pol_subset['categories']) or ('LPA' in category) or ('lpa'in category))

    #save the plot 
        mappable = ax1.collections[0]
        cbar = fig1.colorbar(mappable=mappable, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04) 
        cbar.set_ticks(np.arange(0, 1, 0.1))   
        cbar.set_ticklabels(np.arange(0, 150, 15) )
        cbar.set_label('Rotation (degrees)', rotation=270, labelpad=20)
    if z_bin is not None:
        base_str= savedir +'\\results RFonScreen plot Z%s rel_tresh %.3f _cats  %s_rot %s'% (treshold,cats_to_exclude, rota) 
    else:
        base_str= savedir + '\\results RFonScreen plot allZ rel_tresh %s _cats %s_rot %s'% (treshold,cats_to_exclude, rota) 
    save_str1=base_str +'.png'
    plt.title('vector plots reliability_treshold: %.2f'% treshold)
    plt.savefig(save_str1)
    save_str2=base_str +'.pdf'
    plt.savefig(save_str2)
    plt.close('all')


    return 





datasets_to_load = []
for root, dirs, files in os.walk(datapath_0deg):
    for file in files:
        if 'first.pickle' in file:
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
    # interpolate_data(curr_rois)
    # req_epochs = np.where(np.array(curr_rois[0].stim_info['stimtype'])=='ADS')[0]    ########should be "driftingstripe" for all the stimuli
    # for icx, epoch in enumerate(req_epochs):
        # ROI_mod.align_traces_per_epoch(curr_rois, epoch, grating=False)
    #ROI_mod.map_RF_adjust_edge_time(curr_rois, save_path=results_save_dir, edges=True)
    ROI_mod.map_RF_adjust_edge_time(curr_rois,save_path =results_save_dir, edges=True, delay_use = True)
    for roi in curr_rois: 
        entry = {
                # 'uniq_id': roi.uniq_id,
                'roi_count': len(curr_rois),
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
                'RF_maps_norm': getattr(roi, 'RF_map_norm', np.nan),
                'RF_center_coords': getattr(roi, 'RF_center_coords', np.nan),
            }
        data[idx].append(entry)
    

# upd6on = make_genotype_df(data, genotype='Upd_6f', CS_str='ON')
# upd6off = make_genotype_df(data, genotype='Upd_6f', CS_str='OFF')
control_0deg = make_genotype_df(data, genotype=genotype, reliability_treshold = reliability_treshold, CS_str=None)

if CS_filter[0]==True:
    ON_column=control_0deg['CS']=='ON'
    OFF_column=control_0deg['CS']=='OFF'
    control_0deg['ON_responding']=ON_column
    control_0deg['OFF_responding']=OFF_column


# plot_roivectors(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities,treshold=reliability_treshold, cats_to_exclude = [])
# plot_roivectors_perfly(upd6, results_save_dir,z_bin=z_bin,rotation=rotation, cats_to_exclude = [])
# plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['B','C','D'], save_name = 'A')
# plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['A','C','D'], save_name = 'B')
# plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['B','A','D'], save_name = 'C')
# plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['B','C','A'], save_name = 'D')
# plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = [], save_name = 'ABCD')


# plot_RFonScreen(control_0deg, results_save_dir,z_bin = z_bin,rotation = rotation,filter_polarity = filter_polarities,treshold = reliability_treshold, cats_to_exclude = ['B','C','D']) # plot layer A
# plot_RFonScreen(control_0deg, results_save_dir,z_bin = z_bin,rotation = rotation,filter_polarity = filter_polarities,treshold = reliability_treshold, cats_to_exclude = ['A','C','D']) # plot layer B
# plot_RFonScreen(control_0deg, results_save_dir,z_bin = z_bin,rotation = rotation,filter_polarity = filter_polarities,treshold = reliability_treshold, cats_to_exclude = ['B','A','D']) # plot layer C
plot_RFonScreen(control_0deg, results_save_dir,z_bin = z_bin,rotation = rotation,filter_polarity = filter_polarities,treshold = reliability_treshold, cats_to_exclude = ['B','C','A']) # plot layer D