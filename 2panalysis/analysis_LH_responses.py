'''script to analyse pure odor responses in the LH'''
################################################
import os, pickle, preprocessing_params
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from statistics import mean, stdev
import core_preprocessing as core_pre 
from numpy import trapz
from core_preprocessing import stats_boxplot
################################################
# dataset_folder = 'C:/Master_Project/2P/datasets/241203_whole_OL'
dataset_folder =r'F:\Master\241203_whole_OL'
# dataset_folder = r"C:\Users\Christian\Desktop\241203_whole_OL"
paths = core_pre.dataset(dataset_folder)
os.makedirs(f'{paths.results}/OL', exist_ok=True)
os.makedirs(f'{paths.results}/LH', exist_ok=True)
if len(preprocessing_params.experiment)>0:
    target = f'{paths.results}/LH/{preprocessing_params.experiment}'
    target_ol = f'{paths.results}/OL/{preprocessing_params.experiment}'
else:
    target = f'{paths.results}/LH/'
    target_ol = f'{paths.results}/OL/'
################################################
odor_color = {'ACV' : 'tab:orange', "BA": 'tab:cyan', 'WO' : 'tab:gray'}
neuropile_color = {"ME": "#96ceb4", "LO": "#ff6f69", "LOP": "#ffcc5c"}
# {'pulse' : [5,5,5,20,5], 'on' : 155} #for step: [pre,stim,post,ISI,repetition]
odor_stim = preprocessing_params.olfactory_stimuli.get('pulse')

# if os.path.exists(f"{paths.data}/LH_all.pkl")==False:
#     # os.remove(f"{paths.data}/LH_all.pkl")
#     pkls=[]
#     for pkl in os.listdir(paths.data):
#         if pkl.endswith("_LH.pkl"):
#             pkls.append(pkl)
#     all_LH={}
#     for odor in pkls:
#         odor_str = odor.split('_')[-3]
#         with open(f'{paths.data}/{odor}', 'rb') as fi:
#                 datafile = pickle.load(fi)
        
#         all_series, all_series['before'], all_series['after'] = {}, {}, {}
#         for tseries in datafile:
#             for rois in datafile[tseries]['final_rois']:
#                 if tseries.startswith('TSeries'):
#                     fps = rois.alligned_fps
#                     on_i, off_i=[],[]
#                     conns = '_'.join(odor.split('_')[1:])
#                     olf_stim_array = core_pre.olf_stim_array('pulse', fps, f'{paths.processed}/_{conns.split(".")[0]}/')
#                     # olf_stim_array = core_pre.olf_stim_array('pulse', fps, f'{paths.processed}/{odor.split(".")[0]}/')
#                     all_train={'opto_protocoll':olf_stim_array, 'dff':  rois.interpolated_traces_all, 'time':  rois.interpolated_time_all}
#                     stim_on_stamp = np.where(olf_stim_array[:-1] != olf_stim_array[1:])[0]
#                     on_i = stim_on_stamp[0::2]
#                     off_i = stim_on_stamp[1::2]
#                     ct=0
#                     for train in range(5): 
#                         if (on_i[train] - 3*fps) < 0:
#                             base_start  = 0
#                         else:
#                             base_start = int((on_i[train] - 3*fps))
#                         pulse_end = int(off_i[train] + (odor_stim[2]*fps))
#                         pulse_base = np.mean(rois.interpolated_traces_all[base_start:on_i[train]-2])
#                         pulse = rois.interpolated_traces_all[base_start:pulse_end]
#                         rel_pulse = rois.interpolated_time_all[base_start:pulse_end]
#                         prot_pulse = olf_stim_array[base_start:pulse_end]
#                         df = np.subtract(pulse, pulse_base)
#                         one_pulse = {
#                             'signal_dff': df,
#                             'opto': prot_pulse,
#                             'time': rel_pulse,
#                             'fps' : fps
#                         }
#                         all_train[f'pulse{train}']=one_pulse
#                         ct+=1
#                     if tseries.endswith('008'):
#                         all_series['after'][f'{tseries}']= all_train
#                     elif tseries.endswith('000'):
#                         timing = 'before'
#                         all_series['before'][f'{tseries}']= all_train
#         all_LH[f'{odor_str}']=all_series
#     with open(f'{paths.data}/LH_all.pkl', 'wb') as fp:
        # pickle.dump(all_LH, fp)


# if os.path.exists(f"{paths.data}/OL_all.pkl")==False:
#     # os.remove(f"{paths.data}/OL_all.pkl")
#     pkls=[]
#     for pkl in os.listdir(paths.data):
#         if pkl.endswith("_nan_OL.pkl"):
#             pkls.append(pkl)
#     all_OL={}
#     for odor in pkls:
#         odor_str = odor.split('_')[-3]
#         with open(f'{paths.data}/{odor}', 'rb') as fi:
#                 datafile = pickle.load(fi)
#         # fps = datafile['alligned_fps']
#         all_series, all_series['MEi'], all_series['LOi'], all_series['MEo'], all_series['LOo'], all_series['LOP'], all_series['No_category'] = {}, {}, {}, {}, {}, {}, {}
#         for tseries in datafile:
#             # if tseries == 'alligned_fps':
#             #     continue
#             for rois in datafile[tseries]['final_rois']:
#                 if tseries.startswith('TSeries'):
#                     fps = rois.alligned_fps
#                     # for cat in datafile[tseries]:
#                         # if cat == 'on':
#                         #     continue
#                     cat = rois.category
#                     # olf_stim_array = core_pre.olf_stim_array('on', fps, f'{paths.processed}/{odor.split(".")[0]}/')
#                     conns = '_'.join(odor.split('_')[1:])
#                     olf_stim_array = core_pre.olf_stim_array('on', fps, f'{paths.processed}/_{conns.split(".")[0]}/')
#                     # protocoll = np.round(datafile[cat][tseries]['on']['alligned_on']) rois.interpolated_traces_epochs, 'time':  rois.interpolated_time
#                     # data = datafile[tseries][cat]['dff_mean_alligned_fps']
#                     data =  rois.interpolated_traces_all
#                     avg = sum(data) / len(data)
#                     std = stdev(data)
#                     thresh = avg + (5*std)
#                     i = 0
#                     window_size = 5*fps
#                     signal_data = {}
#                     while i < len(data) - window_size + 1:
#                         window = data[i : i + window_size]
#                         # count = all(x > thresh for x in window)
#                         count = len([x for x in window if x > thresh])
#                         # wherer = np.argwhere(np.array(window)>thresh)
#                         if count > 0:
#                             signal_data[i] = window
#                         i += 5 * fps
#                     if len(signal_data) > 0:
#                         signal_data['fps'] = fps
#                         signal_data['thresh'] = thresh
#                         signal_data['avg'] = avg
#                         all_series[cat][tseries] = signal_data
#         all_OL[f'{odor_str}']=all_series
#     with open(f'{paths.data}/OL_all.pkl', 'wb') as fp:
#         pickle.dump(all_OL, fp)

# plot each time per tseries and avg per neuropile next to each other plot mean and std as line > save in tseries folder
# with open(f'{paths.data}/OL_all.pkl', "rb") as fp:
#     dict_all = pickle.load(fp)

# for odor in dict_all:
#     fig = plt.figure(figsize=(12,6), dpi=400)
#     for n, cat in enumerate(dict_all[odor]):
#         ax = plt.subplot2grid(shape=(1,len(dict_all[odor])), loc=(0,n), fig=fig)
#         mean_cat = pd.DataFrame({})
#         if len(dict_all[odor][cat]) > 0:
#             z=0
#             for tseriies in dict_all[odor][cat]:
#                 for chunk in dict_all[odor][cat][tseriies]:
#                     if chunk == 'fps' or chunk == 'thresh' or chunk == 'avg':
#                         continue
#                     y = dict_all[odor][cat][tseriies][chunk]
#                     x = np.arange(0,dict_all[odor][cat][tseriies]['fps']*5)
#                     ax.plot(x, y, color='tab:gray', zorder=(0-z)*-1)
#                     z+=1
#                     mean_roi = pd.DataFrame({f'{tseriies}': y})
#                     mean_roi.dropna(inplace=True)
#                     mean_roi.reset_index(drop=True, inplace=True)
#                     mean_cat = pd.concat([mean_roi, mean_cat], axis=1)
#         if mean_cat.empty == False:
#             x = np.arange(0,dict_all[odor][cat][tseriies]['fps']*5)
#             ax.plot(x,  mean_cat.median(axis=1), color=neuropile_color.get(cat), zorder=100)
#     fig.suptitle(f'{odor} responses in OL', fontsize=18)
#     fig.supylabel('ΔF/F', fontsize= 16)
#     fig.supxlabel(f'time [s]', fontsize= 16)
#     fig.tight_layout()
#     plt.savefig(f'{target_ol}/OL_{odor}.png', dpi=400, bbox_inches='tight')

    
with open(f'{paths.data}/LH_all.pkl', "rb") as fp:
    dict_all = pickle.load(fp)

# for odor in dict_all:
#     for timing in dict_all[odor]:
#         if os.path.exists(f'{target}/LH_{odor}_{timing}_median.png')==True:
#             continue
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
#         z=0
#         mean_all=pd.DataFrame({})
#         for tseries in dict_all[odor][timing]:
#             if tseries == 'alligned_fps':
#                 continue
#             dff = dict_all[odor][timing][tseries]['dff']
#             time = dict_all[odor][timing][tseries]['time']
#             o = dict_all[odor][timing][tseries]['opto_protocoll']
#             mean_roi = pd.DataFrame({f'{tseries}': dff})
#             mean_roi.dropna(inplace=True)
#             mean_roi.reset_index(drop=True, inplace=True)
#             ax.plot(time, mean_roi[f'{tseries}'][:len(time)], color='tab:gray', zorder=(0-z)*-1)
#             mean_all = pd.concat([mean_roi, mean_all], axis=1)
#             z+=1
#         mean_pd = pd.DataFrame({'0': mean_all.median(axis=1)})
#         ax.plot(time, mean_pd['0'][:len(time)], color=odor_color.get(odor), zorder=100)
#         ax.plot(time[:len(o)], o[:len(time)], color='tab:red', zorder=-1, alpha=0.5)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.set_title(f'{timing}')
#         # ax.set_ylim(-0.4,1.0)
#         if odor=='ACV':
#             ax.set_ylim(-0.35,2)
#         elif odor == 'WO':
#             ax.set_ylim(-1,1)
#         elif odor == 'BA':
#             ax.set_ylim(-0.75, 1.25)
#         fig.suptitle(f'{odor} responses in LH', fontsize=18)
#         fig.supylabel('ΔF/F', fontsize= 16)
#         fig.supxlabel(f'time [s]', fontsize= 16)
#         fig.tight_layout()
#         plt.savefig(f'{target}/LH_{odor}_{timing}_median.png', dpi=400, bbox_inches='tight')

mean_odorants={}
all = pd.DataFrame(columns=['value', 'meassure', 'time', "odorant", 'tseries'])

n=0
for odor in dict_all:
    mean_odorants[f'{odor}']={}
    for timing in dict_all[odor]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        z=0
        for_std = pd.DataFrame({})
        mean_all=pd.DataFrame({})
        for tseries in dict_all[odor][timing]:
            if timing == 'after' and odor == 'ACV':
                color='orange'
            elif timing == 'after' and odor =='BA' :
                color = 'cyan'
            elif timing == 'after' and odor == 'WO':
                color = 'black'
            else:
                color=odor_color.get(odor)
            if tseries == 'alligned_fps':
                continue
            dff = dict_all[odor][timing][tseries]
            time = dff['pulse0']['time']
            mean_pulse=pd.DataFrame({})
            for signal in dff:
                if signal.startswith('puls'):
                    signol=dff[signal]['signal_dff']
                    o=dff[signal]['opto']
                    mean = pd.DataFrame({f'{signal}': signol})
                    mean.dropna(inplace=True)
                    mean.reset_index(drop=True, inplace=True)
                    mean_pulse = pd.concat([mean, mean_pulse], axis =1)
                    for_std = pd.concat([mean, for_std], axis=1)
            mean_pd = pd.DataFrame({'0': mean_pulse.median(axis=1)})
            max = mean_pd['0'].max()
            area = trapz(mean_pd['0'].to_numpy())
            all.loc[n] = [area, 'area', timing, odor, tseries]
            n+=1
            all.loc[n] = [max, 'peak', timing, odor, tseries]
            n+=1
            ax.plot(time, mean_pd['0'][:len(time)], label='ΔF/F', color='gray',alpha=0.2, zorder=1) 
            mean_all = pd.concat([mean_pd, mean_all], axis =1)
        mean_pd = pd.DataFrame({'0': mean_all.median(axis=1)})
        ax.plot(time[:len(o)], o, color='tab:red', zorder=-1, alpha=0.5)
        ax.plot(time, mean_pd['0'][:len(time)], color=color, zorder=3)
        std = for_std.sem(axis='columns')
        mean_odorants[f'{odor}'][timing] = {'x': time, 'y' : mean_pd['0'][:len(time)], 'o': o, 'std': std}
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_ylim(-0.1,1.0)
        # ax.set_ylim(-0.2,1.4)
        if odor=='ACV':
            ax.set_ylim(-0.1, 1.6)
        elif odor == 'WO':
            ax.set_ylim(-0.1, 1.6)
        elif odor == 'BA':
            ax.set_ylim(-0.1, 1.6)
        fig.suptitle(f'LH', fontsize=18)
        fig.supylabel('ΔF/F', fontsize= 14)
        fig.supxlabel(f'time [s]', fontsize= 14)
        fig.tight_layout()
        plt.savefig(f'{target}/LH_{odor}_{timing}_pulse.png', dpi=400, bbox_inches='tight')
        plt.close('all')
for what in ['area', 'peak']:
    if what=='area':
        lim = (-10,85)
    else:
        lim = (0, 2.25)
    pallette_all = ['tab:orange', 'orange', 'tab:cyan', 'cyan', 'tab:grey', 'black']
    palette_all = {"before": "orange", 'after': 'tab:green'}
    df = all[all['meassure']==what]
    # fig = plt.figure(figsize=(5,5), dpi=400, sharey =True)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6,4))
    n=0
    for ax, odor in zip(axes, ["ACV", 'BA', "WO"]):
        df_m = df[df['odorant']==odor]
        # ax = plt.subplot2grid(shape=(1,3), loc=(0,n), fig=fig)
        if odor == "ACV":
            colors = ['tab:orange', 'orange']
        elif odor == 'BA':
            colors = ['tab:cyan', 'cyan']
        elif odor == 'WO':
            colors =['tab:grey', 'black']
        # sns.boxplot(data=df,x='odorant',y='value', hue = "time", order=["ACV", 'BA', "WO"], hue_order=['before', 'after'], palette=pallette_all)
        # sns.stripplot(data=df, x="odorant", y="value", hue = "time", order=["ACV", 'BA', "WO"], hue_order=['before', 'after'], dodge=True, jitter =True, ax=ax, alpha=0.75,  palette=pallette_all, legend=False)
        sns.boxplot(data=df_m,x='odorant',y='value', hue = "time", hue_order=['before', 'after'], width =0.5, palette=colors, ax=ax, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) #
        sns.stripplot(data=df_m, x="odorant", y="value", hue = "time", hue_order=['before', 'after'], dodge = True, jitter = True, ax=ax, alpha=0.5, size=4, palette=colors, legend=False)
        ax.set(xlabel=None, ylabel=None)
        ax.legend_.remove() 
        if odor != "ACV":
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_ylim(-0.1,1.0)  
        ax.set_ylim(lim)
        n+=1
    statlog = f'{target}/{what}_all_stats.txt'
    stats_boxplot(statlog, df, x='odorant',y='value', hue = "time", lh=True)
    fig.suptitle(f'LH responses', fontsize=18)
    fig.supylabel(what, fontsize= 14)
    fig.supxlabel('Odorant', fontsize= 14)
    fig.tight_layout()
    plt.savefig(f'{target}/{what}_all.png', dpi=400, bbox_inches='tight')
    plt.close('all')

# for odor in mean_odorants:
#     all_patches=[]
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#     for timing in mean_odorants[odor]:
#         o = mean_odorants[odor][timing]['o']
#         x = mean_odorants[odor][timing]['x']
#         y = mean_odorants[odor][timing]['y']
#         yerr = mean_odorants[odor][timing]['std']
#         if timing == 'after' and odor == 'ACV':
#             color='orange'
#         elif timing == 'after' and odor =='BA' :
#             color = 'cyan'
#         elif timing == 'after' and odor == 'WO':
#             color = 'black'
#         else:
#             color=odor_color.get(odor)
#         ax.plot(x, y, color=color, zorder=3)
#         patch = mpatches.Patch(color=color, label=timing)
#         all_patches.append(patch)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ub = y + yerr
#         lb = y - yerr
#         ax.fill_between(x, ub, lb,color=color, alpha=.4)
#         # ax.set_ylim(-0.1,1.0)
#         # ax.set_ylim(-0.2,1.4)
#     ax.plot(x, o, color='tab:red', zorder=-1, alpha=0.5)
#     fig.suptitle(f'LH', fontsize=18)
#     fig.supylabel('ΔF/F', fontsize= 14)
#     fig.supxlabel(f'time [s]', fontsize= 14)
#     fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
#     fig.tight_layout()
#     plt.savefig(f'{target}/LH_{odor}_VS.png', dpi=400, bbox_inches='tight')
#     plt.close('all')

# all_delta = pd.DataFrame(columns=['value', 'meassure', "odorant"])
# n=0
# for odor in ["ACV", 'BA', 'WO']:
#     df_odor = all[all['odorant'] == odor]
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#     for tseries in df_odor['tseries'].unique().tolist():
#         if tseries.endswith("000"):
#             delta_ts = ('-').join(tseries.split("-")[:-1])
#             delta_ts = ('-').join([delta_ts, '008'])
#             df_fly_before = df_odor[df_odor['tseries']==tseries]
#             df_fly_after = df_odor[df_odor['tseries']==delta_ts]
#             for what in ['peak', 'area']:
#                 df_fly_after_p = df_fly_after[df_fly_after['meassure']==what]
#                 df_fly_before_p = df_fly_before[df_fly_before['meassure']==what]
#                 try:
#                     # delta = df_fly_before_p['value'].iloc[0] - df_fly_after_p['value'].iloc[0]
#                     delta =  (df_fly_after_p['value'].iloc[0] / df_fly_before_p['value'].iloc[0]) *100
#                     if delta < 0:
#                         delta=100
#                     all_delta.loc[n] = [delta, f'delta_{what}', odor]
#                     n+=1
#                 except:
#                     continue


# for what in ['area', 'peak']:
#     df = all_delta[all_delta['meassure']==f'delta_{what}']
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#     sns.boxplot(data=df,x='odorant',y='value', order=["ACV", 'BA', "WO"], palette=odor_color, width=0.5, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#     sns.stripplot(data=df, x="odorant", y="value", order=["ACV", 'BA', "WO"], dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=odor_color, legend=False)
#     ax.set(xlabel=None, ylabel=None)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     fig.suptitle(f'LH responses', fontsize=18)
#     # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#     fig.supylabel(f'after as % of before', fontsize= 14)
#     fig.supxlabel('Odorant', fontsize= 14)
#     fig.tight_layout()
#     # plt.savefig(f'{target}/{what}_all_delta.png', dpi=400, bbox_inches='tight')
#     plt.savefig(f'{target}/{what}_all_delta_perc.png', dpi=400, bbox_inches='tight')
#     plt.close('all')