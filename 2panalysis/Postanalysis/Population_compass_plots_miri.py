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
import process_mov_core as pmc
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import matplotlib as mpl
plt.style.use('seaborn-v0_8-talk')

#####__________________Parameters________________###########

<<<<<<< Updated upstream
results_save_dir = r'U:\Work_PT\Plots\upd_flies_multi_depth\75deg\d50'
filter_polarities = False
z_bin = None
reliability_treshold = 0.4
CS_filter = [True]
rotation = None
datapath = r'C:\Users\ptrimbak\Work_PT\2p_data\240416_T4T5_upd_depths\rotation_0deg'
=======
results_save_dir = r'U:\Work_PT\Plots\perspective_corrected_upd\0deg\d30'
filter_polarities = False
z_bin = None
reliability_treshold = 0.5
CS_filter = [True]
rotation = None
cats_to_exclude=['C', 'A', 'B']
datapath = r'C:\Users\ptrimbak\Work_PT\2p_data\240701_upd_wt_perspective\control_wt\0deg'
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
<<<<<<< Updated upstream
    else:
        df1 = df1[df1['reliability_PD'] >= 0.4]
=======
        df1 = df1[df1['reliability_PD'] >= 0.5]
    else:
        df1 = df1[df1['reliability_PD'] >= 0.5]
>>>>>>> Stashed changes
        df1 = df1[df1['categories']!="No_category"]
        df1 = df1[df1['CS']==CS_str]
        df1 = df1.reset_index(drop = True)
    return df1


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
<<<<<<< Updated upstream
    
=======
    ROI_mod.map_RF_adjust_edge_time(curr_rois, save_path=results_save_dir, edges=True)
>>>>>>> Stashed changes
    
    for roi in curr_rois: 
        entry = {
                # 'uniq_id': roi.uniq_id,
<<<<<<< Updated upstream
=======
                'roi_count': len(curr_rois),
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
                'RF_maps_norm': getattr(roi, 'RF_map_norm', np.nan),
                'RF_center_coords': getattr(roi, 'RF_center_coords', np.nan),
>>>>>>> Stashed changes
            }
        data[idx].append(entry)
    

<<<<<<< Updated upstream
upd6on = make_genotype_df(data, genotype='Upd_6f', CS_str='ON')
upd6off = make_genotype_df(data, genotype='Upd_6f', CS_str='OFF')
upd6 = make_genotype_df(data, genotype='Upd_6f', CS_str=None)

def plot_roivectors(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=True,treshold=None):
=======
# upd6on = make_genotype_df(data, genotype='Upd_6f', CS_str='ON')
# upd6off = make_genotype_df(data, genotype='Upd_6f', CS_str='OFF')
upd6 = make_genotype_df(data, genotype='Upd_6f', CS_str=None)

upd6['treatment'] = 'Upd;Gcamp6f/Cyo'

def plot_roivectors(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=True,treshold=None, cats_to_exclude= None):
>>>>>>> Stashed changes
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

    
<<<<<<< Updated upstream
    cats_to_exclude=['A', 'B', 'C']
=======
    cats_to_exclude=cats_to_exclude
>>>>>>> Stashed changes
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
                mean_length,mean_angle=pac.calculate_mean_vector(in_angles,in_lengths)
                                               
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

    # temp_df=temp_df.loc[roi_dataframe['categories']!='BG']    

    # turn string entries in dataframe to lowercase
    #temp_df['treatment']=temp_df['treatment'].str.lower()
    #temp_df['categories']=temp_df['categories'].str.upper()

    
    #create plot scaffold
    fig1=plt.figure(figsize=(20,12))
    gs = plt.GridSpec(3, treatment_number)
    treatment_polarity_pairs=itertools.product([0,1],range(treatment_number))
    plt.style.use('seaborn-v0_8-talk')
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
        depths= np.unique(treatment_pol_subset['depth'])
        
        for idx in range(len(treatment_pol_subset.index)):
            magnitude= treatment_pol_subset.iloc[idx,:][DSI_name]
            direction= treatment_pol_subset.iloc[idx,:][PD_name]
            category= treatment_pol_subset.iloc[idx,:]['categories']
            dep = treatment_pol_subset.iloc[idx,:]['depth']
            
            if ('_a' in category) or ('LPA' in category) or ('lpa'in category) or ('A'in category):
                color='tab:green'
                cmap = mpl.colormaps['Greens']
                colors = cmap(np.linspace(0.2, 0.8, len(depths)))
            elif ('_b' in category) or ('LPB' in category) or ('lpb'in category) or ('B'in category):
                color='tab:blue'
                cmap = mpl.colormaps['Blues']
                colors = cmap(np.linspace(0.2, 0.8, len(depths)))
            elif ('_c' in category) or ('LPC' in category) or ('lpc'in category) or ('C'in category):
                color='tab:red'
                cmap = mpl.colormaps['Reds']
                colors = cmap(np.linspace(0.2, 0.8, len(depths)))
            elif ('_d' in category) or ('LPD' in category) or ('lpd'in category) or ('D'in category):
                color='gold'
                cmap = mpl.colormaps['Wistia']
                colors = cmap(np.linspace(0.2, 0.8, len(depths)))
            else:
                color='grey'
<<<<<<< Updated upstream
            for ide, deps in enumerate(depths):
                if dep == deps:
                    color = colors[ide]
            title= treatment + ' total n_: %s pol n: %s %s rois ' %(number_of_treatment_flies,number_of_flies,number_of_rois)
            ax1.set_ylim(0,1)
            ax1.set_yticks([0.2, 0.6, 1, 1.4, 2])
=======
            # for ide, deps in enumerate(depths):
            color_id = np.where(dep == depths)[0]
            color = colors[color_id]
            title= treatment + ' total n_: %s pol n: %s %s rois ' %(number_of_treatment_flies,number_of_flies,number_of_rois)
            ax1.set_ylim(0,1)
            ax1.set_yticks([0.2, 0.4, 0.6, 1])
>>>>>>> Stashed changes
            ax1.set_title(title,pad=15,fontweight='bold',fontsize=13)
            ax1.set_ylabel(polarity[0:3],labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
            # for dep in depths:
            ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
<<<<<<< Updated upstream
            scale=1,angles="xy",scale_units='xy',alpha=0.7)    
=======
            scale=1,angles="xy",scale_units='xy')    
>>>>>>> Stashed changes
        # plot mean
        mean_df_subset=mean_vector_df.loc[mean_vector_df['polarity']==polarity]
        mean_df_subset=mean_df_subset.loc[mean_df_subset['treatment']==treatment]
        local_lenghts=np.array(mean_df_subset['length'])
        local_dirs=np.array(mean_df_subset['direction'])
        for divs, divecs in enumerate(local_dirs):
            ax1.quiver(0,0, np.radians(divecs),local_lenghts[divs],color='black',
<<<<<<< Updated upstream
                scale=1,angles="xy",scale_units='xy',alpha=1) 
=======
                scale=1,angles="xy",scale_units='xy') 
>>>>>>> Stashed changes
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

    plt.title('vector plots reliability_treshold: %.2f'% treshold)
<<<<<<< Updated upstream
    # plt.savefig(save_str1)
    # save_str2=base_str +'.pdf'
    # plt.savefig(save_str2)
    # plt.close('all')


=======
    plt.savefig(save_str1)
    save_str2=base_str +'.pdf'
    plt.savefig(save_str2)
    plt.close('all')

def plot_roivectors_perfly(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=True, cats_to_exclude=None):    
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
        gs = plt.GridSpec(1, 3)
        plt.style.use('seaborn-v0_8-talk')
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
        depths= np.unique(fly_subset['depth'])
        for idg, roi in enumerate(range(len(fly_subset.index))):
            roi_entry=fly_subset.iloc[roi,:]
            category=roi_entry['categories']
            CS=roi_entry['CS']
            dep = fly_subset.iloc[idg,:]['depth']
            if filter_polarity==True:
                if CS=='ON':                
                    magnitude= roi_entry['DSI_ON']
                    direction= roi_entry['PD_ON']
                    category= roi_entry['categories']
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category) or ('A'in category) :
                        color='tab:green'
                        cmap = mpl.colormaps['Greens']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category) or ('B'in category):
                        color='tab:blue'
                        cmap = mpl.colormaps['Blues']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category) or ('C'in category):
                        color='tab:red'
                        cmap = mpl.colormaps['Reds']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category) or ('D'in category):
                        color='gold'
                        cmap = mpl.colormaps['Wistia']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    else:
                        color='grey'
                        cmap = mpl.colormaps['binary']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    ax1.set_ylim(0,0.7)
                    ax1.set_yticks([0.5,0.7])              

                    color_id = np.where(dep == depths)[0]
                    color = colors[color_id]        
                    ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
                    scale=1,angles="xy",scale_units='xy', alpha = 0.7)       
                    ax1.set_title('ON',pad=15,fontweight='bold',fontsize=16)
                if CS=='OFF':
                    magnitude= roi_entry['DSI_OFF']
                    direction= roi_entry['PD_OFF']
                    category= roi_entry['categories']
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category) or ('A'in category):
                        color='tab:green'
                        cmap = mpl.colormaps['Greens']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category) or ('B'in category):
                        color='tab:blue'
                        cmap = mpl.colormaps['Blues']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category) or ('C'in category):
                        color='tab:red'
                        cmap = mpl.colormaps['Reds']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category) or ('D'in category):
                        color='gold'
                        cmap = mpl.colormaps['Wistia']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    else:
                        color='grey'
                        cmap = mpl.colormaps['binary']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    ax2.set_ylim(0,1.2)  
                    ax2.set_yticks([0.5,1])  
                    color_id = np.where(dep == depths)[0]
                    color = colors[color_id]
                    ax2.quiver(0,0, np.radians(direction),magnitude,color=color,
                    scale=1,angles="xy",scale_units='xy',alpha=0.7)  
                    ax2.set_title('OFF',pad=15,fontweight='bold',fontsize=16)
            else: # if you don't want to separate plots based on the prefered polarity (for example when you have manual ROIs)
                  # in this case, I use a response filter to exclude non-responding ROIs for the respective epochs
                if roi_entry['ON_responding']==True:
                    magnitude= roi_entry['DSI_ON']
                    direction= roi_entry['PD_ON']
                    category= roi_entry['categories']
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category) or ('A'in category):
                        color='tab:green'
                        cmap = mpl.colormaps['Greens']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category) or ('B'in category):
                        color='tab:blue'
                        cmap = mpl.colormaps['Blues']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category) or ('C'in category):
                        color='tab:red'
                        cmap = mpl.colormaps['Reds']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category) or ('D'in category):
                        color='gold'
                        cmap = mpl.colormaps['Wistia']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    else:
                        color='grey'
                        cmap = mpl.colormaps['Binary']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    color_id = np.where(dep == depths)[0]
                    color = colors[color_id]    
                    ax1.set_ylim(0,0.7)
                    ax1.set_yticks([0.5,0.7])                
                    ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
                    scale=1,angles="xy",scale_units='xy',alpha=0.7)       
                    ax1.set_title('ON',pad=15,fontweight='bold',fontsize=16)
                if roi_entry['OFF_responding']==True:
                    if ('_A' in category) or ('LPA' in category) or ('lpa'in category) or ('A'in category):
                        color='tab:green'
                        cmap = mpl.colormaps['Greens']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_B' in category) or ('LPB' in category) or ('lpb'in category) or ('B'in category):
                        color='tab:blue'
                        cmap = mpl.colormaps['Blues']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_C' in category) or ('LPC' in category) or ('lpc'in category) or ('C'in category):
                        color='tab:red'
                        cmap = mpl.colormaps['Reds']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    elif ('_d' in category) or ('LPD' in category) or ('lpd'in category) or ('D'in category):
                        color='gold'
                        cmap = mpl.colormaps['Wistia']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    else:
                        color='grey'
                        cmap = mpl.colormaps['Binary']
                        colors = cmap(np.linspace(0.2, 0.8, len(depths)))
                    magnitude= roi_entry['DSI_OFF']
                    direction= roi_entry['PD_OFF']
                    category= roi_entry['categories']
                    color_id = np.where(dep == depths)[0]
                    color = colors[color_id]
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

        
    # plt.close('all')    

def plot_circular_histogram(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=False,treshold=None, cats_to_exclude=None):
    # calculate the total number of ROIs per treatment!
    
    #filter Z and rotation
    if z_bin is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['depth_bin']==z_bin]
    if rotation is not None:
        roi_dataframe=roi_dataframe.loc[roi_dataframe['rotations']==rotation]
    if cats_to_exclude is not None:
        cats_to_exclude=cats_to_exclude
        for cats in cats_to_exclude:
            roi_dataframe = roi_dataframe.loc[roi_dataframe['categories']!=cats]
    treatment_layer_len={}
    treatment_layer_passed_on={}    
    treatment_layer_passed_off={}
    for treatment in np.unique(roi_dataframe['treatment']):
        treatment_layer_len[treatment]={}
        treatment_layer_passed_on[treatment]={}
        treatment_layer_passed_off[treatment]={}
        for layer in np.unique(roi_dataframe['categories']):
            treatment_len_temp=roi_dataframe.loc[roi_dataframe['treatment']==treatment]
            # treatment_len_temp=treatment_len_temp.loc[treatment_len_temp['categories']==layer].loc[treatment_len_temp['categories']==layer]
            layer_temp={layer:len(treatment_len_temp.loc[treatment_len_temp['categories']==layer])}
            layer_temp_passed_on={layer:len(treatment_len_temp.loc[treatment_len_temp['ON_responding']==True])}
            layer_temp_passed_off={layer:len(treatment_len_temp.loc[treatment_len_temp['OFF_responding']==True])}
            treatment_layer_passed_on[treatment].update(layer_temp_passed_on)
            treatment_layer_passed_off[treatment].update(layer_temp_passed_off)
            treatment_layer_len[treatment].update(layer_temp)

    
    #create fgure scaffold
    fig1=plt.figure(figsize=(20,16))
    gs = plt.GridSpec(2, len(np.unique(roi_dataframe['treatment'])))
    treatment_num=len(np.unique(roi_dataframe['treatment']))
    treatments=np.unique(roi_dataframe['treatment'])
    treatment_polarity_pairs=itertools.product([0,1],range(treatment_num))
    plt.style.use('seaborn-v0_8-talk')
    max_weight=[]
    for pair in treatment_polarity_pairs:
        polarity=['ON','OFF'][pair[0]]
        treatment=treatments[pair[1]]
        if filter_polarity:
            subset_temp=roi_dataframe.loc[roi_dataframe[polarity+'_responding']==True]
        else:
            subset_temp=roi_dataframe
        subset_temp=subset_temp.loc[subset_temp['treatment']==treatment]
        ax1=fig1.add_subplot(gs[pair[0],pair[1]],polar=True)
        for layer in np.unique(subset_temp['categories']):
            if layer=='LPA' or layer=='A':
                color='tab:green'
            elif layer=='LPB' or layer=='B':
                color='tab:blue'
            elif layer=='LPC' or layer=='C':
                color='tab:red'
            elif layer=='LPD' or layer=='D':
                color='gold'
            else:
                continue
            # create subset with specific layer
            subset_layer=subset_temp.loc[subset_temp['categories']==layer]
            directions= np.array(subset_layer['PD_'+ polarity])
            # create info for plot
            hist_directions=np.histogram(np.radians(directions),np.radians(np.arange(0,366,6)),density=True)
            hist_directions_weights=hist_directions[0]#/treatment_layer_len[treatment][layer]
            #hist_directions_counts=hist_directions[0]/treatment_layer_len[treatment][layer]
            #hist_directions_counts=hist_directions[0]*hist_directions_weights
            
            #hist_directions[0]=hist_directions[0]/treatment_len[treatment]
            
            summed_directions = np.sum(hist_directions_weights*(np.exp(np.radians(np.arange(3,363,6))*1j)))
            max_weight.append(np.max(hist_directions_weights))
            mean_angle=np.angle(summed_directions)
            vector_lenght=np.abs(summed_directions)/np.sum(hist_directions_weights)
            
            #lens=np.repeat(len(directions))

            mean_len,mean_dir=pac.calculate_mean_vector(np.arange(3,363,6),hist_directions_weights)
            #scale the mean vector by the proportion of responding ROIS/total ROIs in layer
            
            
            
            ax1.bar(np.radians(np.arange(3,363,6)),hist_directions_weights,color=color,alpha=0.7,width=np.radians(6))
            # ax1.quiver(0,0, mean_angle,np.abs(vector_lenght),zorder = 5,
                        # scale=1,angles="xy",scale_units='xy',alpha=1) 
            #histogram_plot[0]=histogram_plot[0]/treatment_len[treatment]
            
            # compute mean direction

            #summed_directions = np.sum((1/treatment_layer_len[treatment][layer])*np.exp(directions*1j))

            
            ax1.set_ylim([0,2])
            ax1.set_yticks([0.5,1,1.5,2])
            
            ax1.set_ylabel(polarity,labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
            ####Changed the layer names revert back later
            # if polarity=='ON':
            #     title= treatment + ' n= A%d:%d B%d:%d C%d:%d D%d:%d rois ' %(treatment_layer_passed_on[treatment]['A'],treatment_layer_len[treatment]['A'],\
            #                             treatment_layer_passed_on[treatment]['B'],treatment_layer_len[treatment]['B'],
            #                             treatment_layer_passed_on[treatment]['C'],treatment_layer_len[treatment]['C'],
            #                             treatment_layer_passed_on[treatment]['D'],treatment_layer_len[treatment]['D'])
            #     # # ax.set_title(title,pad=15,fontweight='bold',fontsize=13)
            # else:
            #     title= treatment + ' n= A%d:%d B%d:%d C%d:%d D%d:%d rois ' %(treatment_layer_passed_off[treatment]['A'],treatment_layer_len[treatment]['A'],\
            #                             treatment_layer_passed_off[treatment]['B'],treatment_layer_len[treatment]['B'],
            #                             treatment_layer_passed_off[treatment]['C'],treatment_layer_len[treatment]['C'],
            #                             treatment_layer_passed_off[treatment]['D'],treatment_layer_len[treatment]['D'])
        
            # ax1.set_title(title,pad=15,fontweight='bold',fontsize=13)

            #savefigure
    if z_bin is not None:
        base_str= savedir +'\\results circ_hist Z%s rel_tresh %.3f rotation %s' %(z_bin,treshold,rotation)
    else:
        base_str= savedir + '\\results circ_hist allZ rel_tresh %.3f'% (treshold )
    save_str1=base_str +'.png'
    plt.title('vector plots reliability_treshold: %.2f'% treshold)
    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.savefig(save_str1)
    save_str2=base_str +'.pdf'
    plt.savefig(save_str2)
    plt.close('all')
    return None


##############@###################
#=====================================
def plot_RFonScreen(roi_dataframe,savedir,z_bin=None,rotation=None,filter_polarity=True,treshold=None, cats_to_exclude = []):
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
                mean_length,mean_angle=pac.calculate_mean_vector(in_angles,in_lengths)
                                               
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
    fig1=plt.figure(figsize=(20,12))
    gs = plt.GridSpec(2, treatment_number)
    treatment_polarity_pairs=itertools.product([0,1],range(treatment_number))
    plt.style.use('seaborn-v0_8-talk')
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
        number_of_treatment_flies=len(np.unique(temp_df.loc[temp_df['treatment']==treatment]['FlyId']))
        number_of_rois=len(treatment_pol_subset.index)
        for idx in range(len(treatment_pol_subset.index)):
            magnitude= treatment_pol_subset.iloc[idx,:][DSI_name]
            direction= treatment_pol_subset.iloc[idx,:][PD_name]
            category= treatment_pol_subset.iloc[idx,:]['categories']
            RF_coord = treatment_pol_subset.iloc[idx,:]['RF_center_coords']
            if ('_a' in category) or ('LPA' in category) or ('lpa'in category) or ('A'in category):
                color='tab:green'
            elif ('_b' in category) or ('LPB' in category) or ('lpb'in category) or ('B'in category):
                color='tab:blue'
            elif ('_c' in category) or ('LPC' in category) or ('lpc'in category) or ('C'in category):
                color='tab:red'
            elif ('_d' in category) or ('LPD' in category) or ('lpd'in category) or ('D'in category):
                color='gold'
            else:
                color='grey'
            title= treatment + ' total n_: %s %s rois ' %(number_of_flies,number_of_rois)
            # ax1.set_ylim(-15, 35)
            # ax1.set_xlim(-20,120)
            # ax1.set_yticks([0.2,0.4,0.6, 0.8, 1.0])
            ax1.set_title(title,pad=15,fontweight='bold',fontsize=13)
            ax1.set_ylabel(polarity[0:3],labelpad=50,rotation=0,fontweight='extra bold',fontsize=13)
            # ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
            # scale=1,angles="xy",scale_units='xy',alpha=0.7)    
            u = 1*np.cos(np.radians(direction))
            v = 1*np.sin(np.radians(direction))
            #"backup"
            widths =0.001
            # plt.quiver(RF_coord[0], RF_coord[1], u , v, angles="uv", scale=.6, scale_units='xy', alpha=0.7, color = color, width=widths, headwidth=5, headlength=7)
            # widths =0.002
            plt.quiver(RF_coord[0], RF_coord[1], u , v, scale = 0.5, scale_units='xy', alpha=0.7, color = color, width=widths)
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

    if z_bin is not None:
        base_str= savedir +'\\results RFonScreen plot Z%s rel_tresh %.3f rotation %s' %(z_bin,treshold,rotation)
    else:
        base_str= savedir + '\\results RFonScreen plot allZ rel_tresh %s _rotation %s'% (treshold,rotation) 
    save_str1=base_str +'.png'
    plt.title('vector plots reliability_treshold: %.2f'% treshold)
    plt.savefig(save_str1)
    save_str2=base_str +'.pdf'
    plt.savefig(save_str2)
    plt.close('all')



#=====================================
####################################
>>>>>>> Stashed changes
if CS_filter[0]==True:
    ON_column=upd6['CS']=='ON'
    OFF_column=upd6['CS']=='OFF'
    upd6['ON_responding']=ON_column
    upd6['OFF_responding']=OFF_column
<<<<<<< Updated upstream
plot_roivectors(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities,treshold=reliability_treshold)
=======

    
plot_roivectors(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities,treshold=reliability_treshold, cats_to_exclude = [])
plot_roivectors_perfly(upd6, results_save_dir,z_bin=z_bin,rotation=rotation, cats_to_exclude = [])
plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['B','C','D'])
plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['A','C','D'])
plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['B','A','D'])
plot_circular_histogram(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=False,treshold=reliability_treshold, cats_to_exclude = ['B','C','A'])
# ROI_mod.map_RF_adjust_edge_time(curr_rois, save_path=results_save_dir, edges=True)
# ax1.quiver(0,0, np.radians(direction),magnitude,color=color,
            # scale=1,angles="xy",scale_units='xy',alpha=0.7)    #direction is PD ON or OFF # magnitude is DSI


plot_RFonScreen(upd6,results_save_dir,z_bin=z_bin,rotation=rotation,filter_polarity=filter_polarities,treshold=reliability_treshold,  cats_to_exclude = [])


>>>>>>> Stashed changes
'a'