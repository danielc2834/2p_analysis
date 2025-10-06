'''script to plot trial averaged traces for each fly and as mean per condition'''
################################################
import pickle, os, preprocessing_params, statistics
import numpy as np
# from Helpers import xmlUtilities
import matplotlib.pyplot as plt
import pandas as pd
import core_preprocessing as core_pre
import matplotlib.patches as mpatches
# import altair as alt
import seaborn as sns
import core_analysis as core_a  
################################################
# dataset_folder = 'C:/Master_Project/2P/datasets/241203_whole_OL'
# dataset_folder =r'C:\Users\Christian\Desktop\241203_whole_OL'
dataset_folder =r'F:\Master\241203_whole_OL'
# dataset_folder = r'F:\250219_moving_bar'
paths = core_pre.dataset(dataset_folder)
if len(preprocessing_params.experiment)>0:
    targt = f'{paths.results}/{preprocessing_params.experiment}'
else:
    target = f'{paths.results}/'
################################################
if preprocessing_params.experiment == 'big':
    neuropile_color = {"ME": "#96ceb4", "LO": "#ff6f69", "LOP": "#ffcc5c"}
elif preprocessing_params.experiment == 'layer':
    neuropile_color = {"MEi": "#96ceb4","MEo": "#96ceb4", "LOi": "#ffcc5c", "LOo": "#ffcc5c", "LOP": "#ff6f69"}
################################################
################################################
#seperate epochs and trials 
#TODO: for column set >> calculate CSI, DSI and filter for it > then plot rest
#TODO: calculate area, peak, slope, time to 5% of peak, 
#TODO: find meassure of variance between trials per fly 
#TODO: plot variance per fly + mean per condition
#TODO: plot mean of variance over conditions + BA vs ACV


################################################FFF################################################
################################################FFF################################################
################################################FFF################################################
################################################FFF################################################
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'FFF' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     if fff == 'LH_all.pkl' or fff=='OL_all.pkl':
#         continue
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
#     for tseries in datafile:
#             for cat in cats:
#                 # if os.path.exists(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_epoch_{cat}.png') == False:
#                     fig = plt.figure(dpi=400) #figsize=(10,20)
#                     for m, key in enumerate(epochs):
#                         mean_fly=pd.DataFrame({})
#                         ax = plt.subplot2grid(shape=(1,len(roi.interpolated_traces_epochs)), loc=(0,m), fig=fig)
#                         for n, roi in enumerate(datafile[tseries]['final_rois']):
#                             if roi.category == cat:
#                                 fps = roi.alligned_fps
#                                 for epoch in roi.interpolated_traces_epochs:
#                                     if epoch == key:
#                                         trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                         time_epoch = roi.interpolated_time.get(epoch)
#                                         ax.plot(time_epoch, trace_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                                         mean_fly[n]=trace_whole_epoch
#                         try:
#                             ax.plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                             plt.xlim(0,5)
#                             plt.title(f'{roi.stim_info.get("fg")[m]}')
#                             plt.ylabel('ΔF/F', fontsize= 14)
#                             plt.xlabel(f'time [s]', fontsize= 14)
#                         except:
#                             continue
#                     fig.suptitle(f'{tseries} : {cat}', fontsize=18)
#                     plt.tight_layout()
#                     plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_epoch_{cat}.png', dpi=400, bbox_inches='tight')
#                     plt.close('all')
# # TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'FFF' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
        
#     for cat in cats:
#         # if os.path.exists(f'{paths.results}/{fff.split(".")[0]}/_epoch_{cat}_mean.png')==False:
#             fig = plt.figure(dpi=400)
#             for m, key in enumerate(epochs):
#                 mean_condition = pd.DataFrame({})
#                 ax = plt.subplot2grid(shape=(1,len(epochs)), loc=(0,m), fig=fig)
#                 for nn, tseries in enumerate(datafile):
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             for epoch in roi.interpolated_traces_epochs:
#                                 if epoch == key:
#                                     trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                     time_epoch = roi.interpolated_time.get(epoch)
#                                     mean_fly[n]=trace_whole_epoch
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     try:
#                         ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                         # ax.plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=(0-n)*-1)
#                     except:
#                         continue
#                 ax.plot(time_epoch, mean_condition.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                 # ax.plot(time_epoch, mean_condition.mean(axis=1), color='tab:orange', zorder=100)
#                 plt.xlim(0,5)
#                 plt.title(f'{roi.stim_info.get("fg")[m]}')
#                 plt.ylabel('ΔF/F', fontsize= 14)
#                 plt.xlabel(f'time [s]', fontsize= 14)
#             fig.suptitle(cat, fontsize=18)
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_epoch_{cat}_mean.png', dpi=400, bbox_inches='tight')
#             plt.close('all')

# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'FFF' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     if fff == 'LH_all.pkl' or fff=='OL_all.pkl':
#         continue
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for tseries in datafile:
#         for cat in cats:
#             # if os.path.exists(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_whole_{cat}.png') ==False:
#                 fig = plt.figure(dpi=400) #figsize=(10,20)
#                 mean_fly=pd.DataFrame({})
#                 # ax = plt.subplot2grid(shape=(1,1), loc=(0,0), fig=fig)
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         # for epoch in roi.interpolated_traces_epochs:
#                         #     if epoch == key:
#                         trace_whole_epoch = roi.int_con_trace[1]
#                         time_epoch = roi.int_con_trace[0]
#                         mean_fly[n]=trace_whole_epoch
#                         try:
#                             plt.plot(time_epoch, trace_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                         except:
#                             continue
#                 try:
#                     plt.plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                     # plt.xlim(0,5)
#                     plt.axvline(x = time_epoch[int(len(time_epoch)/2)], color = 'tab:blue')
#                     plt.ylabel('ΔF/F', fontsize= 14)
#                     plt.xlabel(f'time [s]', fontsize= 14)
#                     plt.title(f'{tseries} : {cat}', fontsize=18)
#                     plt.tight_layout()
#                     plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_whole_{cat}.png', dpi=400, bbox_inches='tight')
#                     plt.close('all')
#                 except:
#                     continue
# # TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'FFF' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for cat in cats:
#         # if os.path.exists(f'{paths.results}/{fff.split(".")[0]}/_whole_{cat}_mean.png') == False:
#             fig = plt.figure(dpi=400)
#             mean_condition = pd.DataFrame({})
#             ax = plt.subplot2grid(shape=(1,1), loc=(0,0), fig=fig)
#             for nn, tseries in enumerate(datafile):
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         trace_whole_epoch = roi.int_con_trace[1]
#                         time_epoch = roi.int_con_trace[0]
#                         mean_fly[n]=trace_whole_epoch
#                 mean_condition[nn] = mean_fly.mean(axis=1)
#                 try:
#                     ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                 except:
#                     continue
#             try:
#                 ax.plot(time_epoch[:len(mean_condition.mean(axis=1))], mean_condition.mean(axis=1)[:len(time_epoch)], color=neuropile_color.get(cat), zorder=100)
#                 # plt.xlim(0,5)
#                 plt.axvline(x = time_epoch[int(len(time_epoch)/2)], color = 'tab:blue')
#                 plt.ylabel('ΔF/F', fontsize= 14)
#                 plt.xlabel(f'time [s]', fontsize= 14)
#                 plt.title(cat, fontsize=18)
#                 plt.tight_layout()
#                 plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_whole_{cat}_mean.png', dpi=400, bbox_inches='tight')
#                 plt.close('all')
#             except:
#                 continue

# odor vs non odor vs no odoer
for odor in ['ACV', 'BA']:
    os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
    to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_FFF_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_FFF_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_FFF_OL.pkl' ]
    if preprocessing_params.experiment == 'big':
        cats = ['ME', 'LO', "LOP"]
    elif preprocessing_params.experiment == 'layer':
        cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
    for cat in cats:
        delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
        peak_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'peak'])
        area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
        color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
        # color = ['tab:orange', '#BCBDC4' , '#5A5D59']
        color_pal = {f'{odor}':neuropile_color.get(cat), 'No Odor':'#BCBDC4', 'without':'#5A5D59'}
        mm=0
        all_patches=[]
        for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
            patch = mpatches.Patch(color=col, label=what)
            all_patches.append(patch)
        fig = plt.figure(dpi=400, figsize=(5,20))
        for m in [0,1]:
            if m == 0:
                ax = plt.subplot2grid(shape=(1,len([0,1])), loc=(0,m), fig=fig)
            else:
                ax = plt.subplot2grid(shape=(1,len([0,1])), loc=(0,m), fig=fig, sharey=ax)
            for color_n, fff in enumerate(to_compare):
                with open(f'{paths.data}/{fff}', 'rb') as fi:
                        datafile = pickle.load(fi)
                epoch_len=[]
                for tseries in datafile:
                    for roi in datafile[tseries]['final_rois']:
                        epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
                epochs = list(set.union(*map(set,epoch_len)))
                if isinstance(epochs[0], list):
                    print('differnet epoch lenth acrross data!! ')
                mean_condition = pd.DataFrame({})
                for_std = pd.DataFrame({})
                for nn, tseries in enumerate(datafile):
                    delta_mean, max_mean, area_mean = [], [], []
                    mean_fly=pd.DataFrame({})
                    for n, roi in enumerate(datafile[tseries]['final_rois']):
                        if roi.category == cat:
                            fps = roi.alligned_fps
                            for epoch in roi.interpolated_traces_epochs:
                                if epoch == m:
                                    trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
                                    trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
                                    delta = np.max(trace_whole_epoch) - np.min(trace_whole_epoch)
                                    delta_mean.append(delta)
                                    max_mean.append(np.max(trace_whole_epoch))
                                    area_mean.append(np.trapz(trace_whole_epoch))
                                    time_epoch = roi.interpolated_time.get(epoch)
                                    mean_fly[n]=trace_whole_epoch
                                    for_std = pd.concat([trace_for_std, for_std], axis=1)
                                    delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("fg")[m]}',delta]
                                    # peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("fg")[m]}',np.max(trace_whole_epoch)]
                                    # area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("fg")[m]}',np.trapz(trace_whole_epoch)]
                                    mm+=1
                    # try:
                    #     delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("fg")[m]}',statistics.mean(delta_mean)]
                    #     peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("fg")[m]}',statistics.mean(max_mean)]
                    #     area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("fg")[m]}',statistics.mean(area_mean)]
                    #     mm+=1
                        
                    # except:
                    #     continue
                    mean_condition[nn] = mean_fly.mean(axis=1)
                    # ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
                ax.plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
                yerr = for_std.sem(axis='columns').to_numpy()
                ub = mean_condition.mean(axis=1) + yerr
                lb = mean_condition.mean(axis=1) - yerr
                ax.fill_between(time_epoch, ub, lb,color=color[color_n], alpha=.4)
                # plt.xlim(0,5)
            # if m == 0:
            #     plt.title(f'{roi.stim_info.get("fg")[m]}')
            #     plt.ylabel('ΔF/F', fontsize= 14)
            #     # plt.xlabel(f'time [s]', fontsize= 14)
            # else:
                plt.title(f'{roi.stim_info.get("fg")[m]}')
        fig.supylabel('ΔF/F', fontsize= 14)
        fig.supxlabel(f'time [s]', fontsize= 14)
        fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.1), frameon=False)
        fig.suptitle(f'{cat}_{odor}', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{paths.results}/{odor}/_epoch_vs_{cat}_FFF_mean_1.png', dpi=400, bbox_inches='tight')
        plt.close('all')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        sns.boxplot(data=delta_all,x='stim_con',y='delta', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
        # sns.stripplot(data=delta_all, x="stim_con", y="delta", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
        statlog = f'{paths.results}/{odor}/_maxmin_{cat}_FFF_stats_1.txt'
        if os.path.exists(statlog) == True:
            os.remove(statlog)
        core_pre.stats_boxplot(statlog, delta_all, x='stim_con',y='delta', hue = "odor_con", lh=False)
        ax.set(xlabel=None, ylabel=None)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.suptitle(f'{cat}', fontsize=18)
        # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
        fig.supylabel(f'range', fontsize= 14)
        fig.supxlabel('Stimulus', fontsize= 14)
        fig.tight_layout()
        plt.savefig(f'{paths.results}/{odor}/_maxmin_{cat}_FFF_1.png', dpi=400, bbox_inches='tight')
        plt.close('all')
        
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        # sns.boxplot(data=peak_all,x='stim_con',y='peak', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
        # # sns.stripplot(data=peak_all, x="stim_con", y="peak", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
        # core_pre.stats_boxplot(statlog, peak_all, x='stim_con',y='peak', hue = "odor_con", lh=False)
        # ax.set(xlabel=None, ylabel=None)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # fig.suptitle(f'{cat}', fontsize=18)
        # # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
        # fig.supylabel(f'max', fontsize= 14)
        # fig.supxlabel('Stimulus', fontsize= 14)
        # fig.tight_layout()
        # plt.savefig(f'{paths.results}/{odor}/_peak_{cat}_FFF.png', dpi=400, bbox_inches='tight')
        # plt.close('all')
        
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        # sns.boxplot(data=area_all,x='stim_con',y='area', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
        # # sns.stripplot(data=area_all, x="stim_con", y="area", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
        # core_pre.stats_boxplot(statlog, area_all, x='stim_con',y='area', hue = "odor_con", lh=False)
        # ax.set(xlabel=None, ylabel=None)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # fig.suptitle(f'{cat}', fontsize=18)
        # # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
        # fig.supylabel(f'area', fontsize= 14)
        # fig.supxlabel('Stimulus', fontsize= 14)
        # fig.tight_layout()
        # plt.savefig(f'{paths.results}/{odor}/_area_{cat}_FFF_1.png', dpi=400, bbox_inches='tight')
        # plt.close('all')
    
# for odor in ['ACV', 'BA']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     # to_compare = [f'big_{odor}_{odor}_FFF_OL.pkl', f'big_{odor}_WO_FFF_OL.pkl', f'big_nan_WO_FFF_OL.pkl' ] 
#     to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_FFF_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_FFF_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_FFF_OL.pkl' ]
#     # to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_FFF_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_FFF_OL.pkl']
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#         all_patches=[]
#         for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
#             patch = mpatches.Patch(color=col, label=what)
#             all_patches.append(patch)
#         fig = plt.figure(dpi=400)
#         for color_n, fff in enumerate(to_compare):
#             with open(f'{paths.data}/{fff}', 'rb') as fi:
#                     datafile = pickle.load(fi)
#             mean_condition = pd.DataFrame({})
#             for_std = pd.DataFrame({})
#             for nn, tseries in enumerate(datafile):
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         trace_whole_epoch = roi.int_con_trace[1]
#                         time_epoch = roi.int_con_trace[0]
#                         mean_fly[n]=trace_whole_epoch
#                         trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                         for_std = pd.concat([trace_for_std, for_std], axis=1)
#                 mean_condition[nn] = mean_fly.mean(axis=1)
#                 # plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#             plt.plot(time_epoch[:len(mean_condition.mean(axis=1))], mean_condition.mean(axis=1)[:len(time_epoch)], color=color[color_n], zorder=100)
#             yerr = for_std.sem(axis='columns').to_numpy()
#             ub = mean_condition.mean(axis=1) + yerr[:len(mean_condition.mean(axis=1))]
#             lb = mean_condition.mean(axis=1) - yerr[:len(mean_condition.mean(axis=1))]
#             plt.fill_between(time_epoch[:len(mean_condition.mean(axis=1))], ub[:len(time_epoch)], lb[:len(time_epoch)],color=color[color_n], alpha=.4)
#             # plt.xlim(0,5)
#         # plt.axvline(x = time_epoch[int(len(time_epoch)/2)], color = 'tab:blue')
#         plt.ylabel('ΔF/F', fontsize= 14)
#         plt.xlabel(f'time [s]', fontsize= 14)
#         plt.title(f'{cat}_{odor}', fontsize=18)
#         plt.tight_layout()
#         # plt.savefig(f'{paths.results}/{odor}/_whole_vs_{cat}_FFF_mean_same flies.png', dpi=400, bbox_inches='tight')
#         plt.savefig(f'{paths.results}/{odor}/_whole_vs_{cat}_FFF_mean_bla.png', dpi=400, bbox_inches='tight')
#         plt.close('all')
# # for odor in ['ACV', 'BA']:
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = neuropile_color.get(cat)
#         # fig = plt.figure(dpi=400)
#         fig, axes = plt.subplots(nrows=10, ncols=1, sharex=True,figsize=(10,20)) #figsize=(10,20)
#         count=0 #
#         for fly in os.listdir(f"{paths.processed}/_{odor}_{odor}_FFF_OL"):
#                 tseries_odor = os.listdir(f"{paths.processed}/_{odor}_{odor}_FFF_OL/{fly}")[0]
#                 tseries_wo = os.listdir(f"{paths.processed}/_{odor}_WO_FFF_OL/{fly}")[0]
#                 with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_{odor}_FFF_OL.pkl', 'rb') as fi:
#                     datafile = pickle.load(fi)
#                 roi_odor = datafile[tseries_odor]['final_rois']
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(roi_odor):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         mean_fly[n]=traces_whole_epoch
#                 trace_odor = mean_fly.mean(axis=1)
#                 with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_WO_FFF_OL.pkl', 'rb') as fi:
#                     datafile = pickle.load(fi) 
#                 roi_wo = datafile[tseries_wo]['final_rois']
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(roi_wo):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                         mean_fly[n]=traces_whole_epoch
#                 trace_wo = mean_fly.mean(axis=1)
#                 diff = trace_odor - trace_wo
#                 axes[count].plot(time_epoch, diff, color=color, zorder=100)
#                 axes[count].axvline(x = time_epoch[int(len(time_epoch)/2)], color = 'tab:blue')
#                 axes[count].set_ylim(-0.1,0.2)
#                 count+=1
#         fig.supxlabel(f'time [s]', fontsize= 14)
#         fig.supylabel('ΔF/F', fontsize= 14)
#         fig.suptitle(f'{cat}_{odor}', fontsize=18)
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_FFF_fly.png', dpi=400, bbox_inches='tight')
#         # plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_FFF_fly.pdf', dpi=400, bbox_inches='tight')
#         plt.close('all')
################################################Gratings################################################
################################################Gratings################################################
################################################Gratings################################################
################################################Gratings################################################
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Grating' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
#     for tseries in datafile:
#             for cat in cats:
#                 fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#                 counter_row =0
#                 counter_column = 0
#                 for m, key in enumerate(epochs):
#                     mean_fly=pd.DataFrame({})
#                     # ax = plt.subplot2grid(shape=(int(len(roi.interpolated_traces_epochs)/2),int(len(roi.interpolated_traces_epochs)/2)), loc=(counter_row,counter_column), fig=fig)
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             for epoch in roi.interpolated_traces_epochs:
#                                 if epoch == key:
#                                     trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                     time_epoch = roi.interpolated_time.get(epoch)
#                                     mean_fly[n]=trace_whole_epoch
#                                     try:
#                                         axes[counter_column].plot(time_epoch, trace_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                                     except:
#                                         continue
#                     try:
#                         axes[counter_column].plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                         title = f'{roi.stim_info.get("orientation")[key]}_{roi.stim_info.get("direction")[key]}'
#                         axes[counter_column].set_title(title)
#                         axes[counter_column].set_xlim(0,4)
#                         # plt.ylabel('ΔF/F', fontsize= 14)
#                         # plt.xlabel(f'time [s]', fontsize= 14)
#                         # if counter_column == 0:
#                         counter_column+=1
#                         # elif counter_column == 1:
#                             # counter_column =0
#                             # counter_row+=1
#                     except:
#                         continue
#                 fig.supxlabel(f'time [s]', fontsize= 14)
#                 fig.supylabel('ΔF/F', fontsize= 14)
#                 fig.suptitle(f'{tseries} : {cat}', fontsize=18)
#                 plt.tight_layout()
#                 plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_epoch_{cat}.png', dpi=400, bbox_inches='tight')
#                 plt.close('all')
# # # # # TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Grating' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
        
#     for cat in cats:
#         fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#         counter_row =0
#         counter_column = 0
        
#         for m, key in enumerate(epochs):
#             mean_condition = pd.DataFrame({})
#             # ax = plt.subplot2grid(shape=(1,len(epochs)), loc=(0,m), fig=fig)
#             for nn, tseries in enumerate(datafile):
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         for epoch in roi.interpolated_traces_epochs:
#                             if epoch == key:
#                                 trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                 time_epoch = roi.interpolated_time.get(epoch)
#                                 mean_fly[n]=trace_whole_epoch
#                 mean_condition[nn] = mean_fly.mean(axis=1)
#                 try:
#                     axes[counter_column].plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                 except:
#                     continue
#             axes[counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#             title = f'{roi.stim_info.get("orientation")[key]}_{roi.stim_info.get("direction")[key]}'
#             axes[counter_column].set_title(title)
#             axes[counter_column].set_xlim(0,4)
#             # if counter_column == 0:
#             counter_column+=1
#             # elif counter_column == 1:
#                 # counter_column =0
#                 # counter_row+=1
#             # plt.xlim(0,5)
#             # plt.ylabel('ΔF/F', fontsize= 14)
#             # plt.xlabel(f'time [s]', fontsize= 14)
#         fig.supxlabel(f'time [s]', fontsize= 14)
#         fig.supylabel('ΔF/F', fontsize= 14)
#         fig.suptitle(f'{cat}', fontsize=18)
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_epoch_{cat}_mean.png', dpi=400, bbox_inches='tight')
#         plt.close('all')

# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Grating' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for tseries in datafile:
#             for cat in cats:
#                 fig = plt.figure(dpi=400) #figsize=(10,20)
#                 # fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True) #figsize=(10,20)
#                 counter_row =0
#                 counter_column = 0
#                 mean_fly=pd.DataFrame({})
#                 # ax = plt.subplot2grid(shape=(int(len(roi.interpolated_traces_epochs)/2),int(len(roi.interpolated_traces_epochs)/2)), loc=(counter_row,counter_column), fig=fig)
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         # trace_whole_epoch = roi.roi.oneHz_conc_resp
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                         mean_fly[n]=traces_whole_epoch
#                         try:
#                             plt.plot(time_epoch, traces_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                         except:
#                             continue
#                 for n in [1,2,3]:
#                     plt.axvline(x = time_epoch[int(len(time_epoch)/4)*n], color = 'tab:blue')
#                 plt.plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                 plt.title(f'{tseries} : {cat}', fontsize=18)
#                 plt.xlim(0,16)
#                 plt.ylabel('ΔF/F', fontsize= 14)
#                 plt.xlabel(f'time [s]', fontsize= 14)
#                 plt.tight_layout()
#                 plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_whole_{cat}.png', dpi=400, bbox_inches='tight')
#                 plt.close('all')
# # # TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Grating' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for cat in cats:
#         fig = plt.figure(dpi=400) #figsize=(10,20)
#         mean_condition = pd.DataFrame({})
#         for nn, tseries in enumerate(datafile):
#             mean_fly=pd.DataFrame({})
#             for n, roi in enumerate(datafile[tseries]['final_rois']):
#                 if roi.category == cat:
#                     fps = roi.alligned_fps
#                     traces_whole_epoch = []
#                     for epoch in roi.interpolated_traces_epochs:
#                         traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                     mean_fly[n]=traces_whole_epoch
#                     time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#             mean_condition[nn] = mean_fly.mean(axis=1)
#             try:
#                 plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#             except:
#                 continue
#         for n in [1,2,3]:
#             plt.axvline(x = time_epoch[int(len(time_epoch)/4)*n], color = 'tab:blue')
#         plt.plot(time_epoch, mean_condition.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#         plt.title(f'{tseries} : {cat}', fontsize=18)
#         plt.xlim(0,16)
#         plt.ylabel('ΔF/F', fontsize= 14)
#         plt.xlabel(f'time [s]', fontsize= 14)
#         plt.tight_layout()
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_whole_{cat}_mean.png', dpi=400, bbox_inches='tight')
#         plt.close('all') 
        
# for odor in ['ACV', 'BA']:
# for odor in ['BA']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     # to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_Grating_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_Grating_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_Grating_OL.pkl' ]
#     to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_Grating_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_Grating_OL.pkl']
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         # cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#         cats = ['LOi']
#     for cat in cats:
#         delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
#         peak_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'peak'])
#         area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
#         mm = 0
#         colorrr = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#         color_pal = {f'{odor}':neuropile_color.get(cat), 'No Odor':'#BCBDC4', 'without':'#5A5D59'}
#         all_patches=[]
#         for col, what in zip(colorrr, [f'{odor}', 'No Odor', 'without']):
#             patch = mpatches.Patch(color=col, label=what)
#             all_patches.append(patch)
#         fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True, figsize =(7,4)) #, sharex=True
#         counter_row , counter_column=0,0
#         for m in [0,1,2,3]:
#             # ax = plt.subplot2grid(shape=(1,len([0,1,2,3])), loc=(0,m), fig=fig)
#             for color_n, fff in enumerate(to_compare):
#                 with open(f'{paths.data}/{fff}', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                 epoch_len=[]
#                 for tseries in datafile:
#                     for roi in datafile[tseries]['final_rois']:
#                         epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#                 epochs = list(set.union(*map(set,epoch_len)))
#                 if isinstance(epochs[0], list):
#                     print('differnet epoch lenth acrross data!! ')
#                 mean_condition = pd.DataFrame({})
#                 for_std = pd.DataFrame({})
#                 for nn, tseries in enumerate(datafile):
#                     delta_mean, max_mean, area_mean = [], [], []
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             for epoch in roi.interpolated_traces_epochs:
#                                 if epoch == m+1:
#                                     trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                     delta = np.max(trace_whole_epoch) - np.min(trace_whole_epoch)
#                                     delta_mean.append(delta)
#                                     max_mean.append(np.max(trace_whole_epoch))
#                                     area_mean.append(np.trapz(trace_whole_epoch))
#                                     time_epoch = roi.interpolated_time.get(epoch)
#                                     mean_fly[n]=trace_whole_epoch
#                                     trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                                     for_std = pd.concat([trace_for_std, for_std], axis=1)
#                                     try:
#                                         delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',delta]
#                                         peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',np.max(trace_whole_epoch)]
#                                         area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',np.trapz(trace_whole_epoch)]
#                                         mm+=1
#                                     except:
#                                         continue
#                     # delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',statistics.mean(delta_mean)]
#                     # peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',statistics.mean(max_mean)]
#                     # area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',statistics.mean(area_mean)]
#                     # mm+=1
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     # ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                 axes[counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=colorrr[color_n], zorder=100)
#                 yerr = for_std.sem(axis='columns').to_numpy()
#                 ub = mean_condition.mean(axis=1) + yerr
#                 lb = mean_condition.mean(axis=1) - yerr
#                 axes[counter_column].fill_between(time_epoch, ub, lb,color=colorrr[color_n], alpha=.4)
#                 title = f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}'
#                 axes[counter_column].set_title(title)
#                 axes[counter_column].set_xlim(0,4)
#             # if counter_column == 0:
#             counter_column+=1
            
#             # elif counter_column == 1:
#                 # counter_column =0
#                 # counter_row+=1
#         fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
#         fig.supxlabel(f'time [s]', fontsize= 14)
#         fig.supylabel('ΔF/F', fontsize= 14)
#         fig.suptitle(f'{cat}', fontsize=18)
#         plt.tight_layout()
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_epoch_vs_{cat}_Grating_mean.png', dpi=400, bbox_inches='tight')
#         plt.close('all')

#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#         sns.boxplot(data=delta_all,x='stim_con',y='delta', hue ='odor_con',palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#         # sns.stripplot(data=delta_all, x="stim_con", y="delta", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
#         statlog = f'{paths.results}/{odor}/_maxmin_{cat}_Grating_stats.txt'
#         if os.path.exists(statlog) ==True:
#             os.remove(statlog)
#         core_pre.stats_boxplot(statlog, delta_all, x='stim_con',y='delta', hue = "odor_con", lh=False)
#         ax.set(xlabel=None, ylabel=None)
#         ax.legend_.remove() 
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         fig.suptitle(f'{cat}', fontsize=18)
#         # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#         fig.supylabel(f'range', fontsize= 14)
#         fig.supxlabel('Stimulus', fontsize= 14)
#         fig.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_maxmin_{cat}_Grating.png', dpi=400, bbox_inches='tight')
#         plt.close('all')
        
#         # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#         # sns.boxplot(data=peak_all,x='stim_con',y='peak', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#         # # sns.stripplot(data=peak_all, x="stim_con", y="peak", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
#         # core_pre.stats_boxplot(statlog, peak_all, x='stim_con',y='peak', hue = "odor_con", lh=False)
#         # ax.set(xlabel=None, ylabel=None)
#         # ax.legend_.remove() 
#         # ax.spines['right'].set_visible(False)
#         # ax.spines['top'].set_visible(False)
#         # fig.suptitle(f'{cat}', fontsize=18)
#         # # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#         # fig.supylabel(f'max', fontsize= 14)
#         # fig.supxlabel('Stimulus', fontsize= 14)
#         # fig.tight_layout()
#         # plt.savefig(f'{paths.results}/{odor}/_peak_{cat}_Grating.png', dpi=400, bbox_inches='tight')
#         # plt.close('all')
        
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#         sns.boxplot(data=area_all,x='stim_con',y='area', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#         # sns.stripplot(data=area_all, x="stim_con", y="area", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
#         core_pre.stats_boxplot(statlog, area_all, x='stim_con',y='area', hue = "odor_con", lh=False)
#         ax.set(xlabel=None, ylabel=None)
#         ax.legend_.remove() 
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         fig.suptitle(f'{cat}', fontsize=18)
#         # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#         fig.supylabel(f'area', fontsize= 14)
#         fig.supxlabel('Stimulus', fontsize= 14)
#         fig.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_area_{cat}_Grating.png', dpi=400, bbox_inches='tight')
#         plt.close('all')
        
# for odor in ['ACV', 'BA']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     # to_compare = [f'big_{odor}_{odor}_Grating_OL.pkl', f'big_{odor}_WO_Grating_OL.pkl', f'big_nan_WO_Grating_OL.pkl' ] 
#     to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_Grating_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_Grating_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_Grating_OL.pkl' ]
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#         all_patches=[]
#         for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
#             patch = mpatches.Patch(color=col, label=what)
#             all_patches.append(patch)
#         fig = plt.figure(dpi=400)
#         for color_n, fff in enumerate(to_compare):
#             with open(f'{paths.data}/{fff}', 'rb') as fi:
#                     datafile = pickle.load(fi)
#             mean_condition = pd.DataFrame({})
#             for_std = pd.DataFrame({})
#             for nn, tseries in enumerate(datafile):
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                         mean_fly[n]=traces_whole_epoch
#                         trace_for_std = pd.DataFrame({f'{n}': traces_whole_epoch})
#                         for_std = pd.concat([trace_for_std, for_std], axis=1)
#                 mean_condition[nn] = mean_fly.mean(axis=1)
#                 # plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#             plt.plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#             yerr = for_std.sem(axis='columns').to_numpy()
#             ub = mean_condition.mean(axis=1)[:len(time_epoch)] + yerr
#             lb = mean_condition.mean(axis=1)[:len(time_epoch)] - yerr
#             plt.fill_between(time_epoch[:len(mean_condition.mean(axis=1))], ub, lb,color=color[color_n], alpha=.4)
#             # plt.xlim(0,5)
#         for n in [1,2,3]:
#             plt.axvline(x = time_epoch[int(len(time_epoch)/4)*n], color = 'tab:blue')
#         plt.ylabel('ΔF/F', fontsize= 14)
#         plt.xlabel(f'time [s]', fontsize= 14)
#         plt.title(f'{cat}_{odor}', fontsize=18)
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_whole_vs_{cat}_Grating_mean.png', dpi=400, bbox_inches='tight')
#         plt.close('all')

# for odor in ['ACV', 'BA']:
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = neuropile_color.get(cat)
#         # fig = plt.figure(dpi=400)
#         fig, axes = plt.subplots(nrows=10, ncols=1, sharex=True,figsize=(10,20)) #figsize=(10,20)
#         count=0 #
#         for fly in os.listdir(f"{paths.processed}/_{odor}_{odor}_Grating_OL"):
#                 tseries_odor = os.listdir(f"{paths.processed}/_{odor}_{odor}_Grating_OL/{fly}")[0]
#                 tseries_wo = os.listdir(f"{paths.processed}/_{odor}_WO_Grating_OL/{fly}")[0]
#                 with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_{odor}_Grating_OL.pkl', 'rb') as fi:
#                     datafile = pickle.load(fi)
#                 roi_odor = datafile[tseries_odor]['final_rois']
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(roi_odor):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         mean_fly[n]=traces_whole_epoch
#                 trace_odor = mean_fly.mean(axis=1)
#                 with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_WO_Grating_OL.pkl', 'rb') as fi:
#                     datafile = pickle.load(fi) 
#                 roi_wo = datafile[tseries_wo]['final_rois']
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(roi_wo):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                         mean_fly[n]=traces_whole_epoch
#                 trace_wo = mean_fly.mean(axis=1)
#                 diff = trace_odor - trace_wo
#                 axes[count].plot(time_epoch, diff, color=color, zorder=100)
#                 for n in [1,2,3]:
#                     axes[count].axvline(x = time_epoch[int(len(time_epoch)/4)*n], color = 'tab:blue')
#                 axes[count].set_ylim(-0.1,0.2)
#                 count+=1
#         fig.supxlabel(f'time [s]', fontsize= 14)
#         fig.supylabel('ΔF/F', fontsize= 14)
#         fig.suptitle(f'{cat}_{odor}', fontsize=18)
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_Grating_fly.png', dpi=400, bbox_inches='tight')
#         # plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_Grating_fly.pdf', dpi=400, bbox_inches='tight')
#         plt.close('all')
# # ##############################################8D################################################
# # ##############################################8D################################################
# # ##############################################8D################################################
# # ##############################################8D################################################
# pkls=[]
# for condition in os.listdir(paths.data):
#     if '8D' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
#     for tseries in datafile:
#             for cat in cats:
#                 # if os.path.exists(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_epoch_{cat}.png')==False:
#                     fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#                     counter_row =0
#                     counter_column = 0
#                     for m, key in enumerate(epochs):
#                         mean_fly=pd.DataFrame({})
#                         # ax = plt.subplot2grid(shape=(int(len(roi.interpolated_traces_epochs)/2),int(len(roi.interpolated_traces_epochs)/2)), loc=(counter_row,counter_column), fig=fig)
#                         for n, roi in enumerate(datafile[tseries]['final_rois']):
#                             if roi.category == cat:
#                                 fps = roi.alligned_fps
#                                 for epoch in roi.interpolated_traces_epochs:
#                                     if epoch == key:
#                                         trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                         time_epoch = roi.interpolated_time.get(epoch)
#                                         axes[counter_row][counter_column].plot(time_epoch, trace_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                                         mean_fly[n]=trace_whole_epoch
#                         try:
#                             axes[counter_row][counter_column].plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                         except:
#                             continue
#                         title = f'{roi.stim_info.get("angle")[key]}'
#                         axes[counter_row][counter_column].set_title(title)
#                         axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#                         # plt.ylabel('ΔF/F', fontsize= 14)
#                         # plt.xlabel(f'time [s]', fontsize= 14)
#                         if counter_column < 3:
#                             counter_column+=1
#                         elif counter_column == 3:
#                             counter_column =0
#                             counter_row+=1
#                     fig.supxlabel(f'time [s]', fontsize= 14)
#                     fig.supylabel('ΔF/F', fontsize= 14)
#                     fig.suptitle(f'{tseries} : {cat}', fontsize=18)
#                     plt.tight_layout()
#                     plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_epoch_{cat}.png', dpi=400, bbox_inches='tight')
#                     plt.close('all')
# # TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if '8D' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
#     for cat in cats:
# #         if os.path.exists(f'{paths.results}/{fff.split(".")[0]}/_epoch_{cat}_mean.png')==False:
#             fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#             counter_row =0
#             counter_column = 0
#             for m, key in enumerate(epochs):
#                 mean_condition = pd.DataFrame({})
#                 # ax = plt.subplot2grid(shape=(1,len(epochs)), loc=(0,m), fig=fig)
#                 for nn, tseries in enumerate(datafile):
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             for epoch in roi.interpolated_traces_epochs:
#                                 if epoch == key:
#                                     trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                     time_epoch = roi.interpolated_time.get(epoch)
#                                     mean_fly[n]=trace_whole_epoch
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     try:
#                         axes[counter_row][counter_column].plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                     except:
#                         continue
#                 axes[counter_row][counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                 title = f'{roi.stim_info.get("angle")[key]}'
#                 axes[counter_row][counter_column].set_title(title)
#                 axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#                 if counter_column < 3:
#                     counter_column+=1
#                 elif counter_column == 3:
#                     counter_column =0
#                     counter_row+=1
#                 # plt.xlim(0,5)
#                 # plt.ylabel('ΔF/F', fontsize= 14)
#                 # plt.xlabel(f'time [s]', fontsize= 14)
#             fig.supxlabel(f'time [s]', fontsize= 14)
#             fig.supylabel('ΔF/F', fontsize= 14)
#             fig.suptitle(f'{cat}', fontsize=18)
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_epoch_{cat}_mean.png', dpi=400, bbox_inches='tight')
#             plt.close('all')

# pkls=[]
# for condition in os.listdir(paths.data):
#     if '8D' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     for tseries in datafile:
#         timepoints = datafile[tseries]['final_rois'][0].stim_info.get('epoch_dur')
#         for idx, n in enumerate(timepoints):
#             if idx > 0  and idx < len(timepoints):
#                 new_n = previous + n 
#                 timepoints[idx] = new_n 
#                 previous=new_n  
#             else:
#                 previous=n  
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for tseries in datafile:
#         for cat in cats:
# #             if os.path.exists(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_whole_{cat}.png') == False:
#                 fig = plt.figure(dpi=400) #figsize=(10,20)
#                 # fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True) #figsize=(10,20)
#                 mean_fly=pd.DataFrame({})
#                 # ax = plt.subplot2grid(shape=(int(len(roi.interpolated_traces_epochs)/2),int(len(roi.interpolated_traces_epochs)/2)), loc=(counter_row,counter_column), fig=fig)
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         # trace_whole_epoch = roi.roi.oneHz_conc_resp
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                         mean_fly[n]=traces_whole_epoch
#                         try:
#                             plt.plot(time_epoch, traces_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                         except:
#                             continue                   
                                
#                 for n in range(7):
#                     plt.axvline(x = timepoints[n], color = 'tab:blue')
#                 try:
#                     plt.plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                     plt.title(f'{tseries} : {cat}', fontsize=18)
#                     plt.xlim(0,90)
#                     plt.ylabel('ΔF/F', fontsize= 14)
#                     plt.xlabel(f'time [s]', fontsize= 14)
#                     plt.tight_layout()
#                     plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_whole_{cat}.png', dpi=400, bbox_inches='tight')
#                     plt.close('all')
#                 except:
#                     continue
# # TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if '8D' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     # timepoints = datafile[0]['final_rois'][0].stim_info.get('epoch_dur')
#     # for idx, n in enumerate(timepoints):
#     #     if idx > 0  and idx < len(timepoints):
#     #         new_n = previous + n 
#     #         timepoints[idx] = new_n 
#     #         previous=new_n  
#     #     else:
#     #         previous=n  
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for cat in cats:
#         fig = plt.figure(dpi=400) #figsize=(10,20)
#         mean_condition = pd.DataFrame({})
#         for nn, tseries in enumerate(datafile):
#             mean_fly=pd.DataFrame({})
#             for n, roi in enumerate(datafile[tseries]['final_rois']):
#                 if roi.category == cat:
#                     fps = roi.alligned_fps
#                     traces_whole_epoch = []
#                     for epoch in roi.interpolated_traces_epochs:
#                         traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                     mean_fly[n]=traces_whole_epoch
#                     time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#             mean_condition[nn] = mean_fly.mean(axis=1)
#             try:
#                 plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#             except:
#                 continue
#         # timepoints = roi.stim_info.get('epoch_dur')
#         # for idx, n in enumerate(timepoints):
#         #     if idx > 0  and idx < len(timepoints):
#         #         new_n = previous + n 
#         #         timepoints[idx] = new_n 
#         #         previous=new_n  
#         #     else:
#         #         previous=n  
#         for n in range(7):
#             plt.axvline(x = timepoints[n], color = 'tab:blue')
#         plt.plot(time_epoch, mean_condition.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#         plt.title(f'{tseries} : {cat}', fontsize=18)
#         plt.xlim(0,90)
#         plt.ylabel('ΔF/F', fontsize= 14)
#         plt.xlabel(f'time [s]', fontsize= 14)
#         plt.tight_layout()
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_whole_{cat}_mean.png', dpi=400, bbox_inches='tight')
#         plt.close('all')

# for odor in ['ACV', 'BA']:
# # for odor in ['ACV']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_8D_OL.pkl' ]
#     # to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl']
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#         # cats = ['LOo', 'LOP']
#     for cat in cats:
#         delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
#         peak_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'peak'])
#         area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
#         color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#         color_pal = {f'{odor}':neuropile_color.get(cat), 'No Odor':'#BCBDC4', 'without':'#5A5D59'}
#         mm=0
#         all_patches=[]
#         for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
#             patch = mpatches.Patch(color=col, label=what)
#             all_patches.append(patch)
#         fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#         counter_row , counter_column=0,0
#         for m in [0,1,2,3,4,5,6,7]:
#             # ax = plt.subplot2grid(shape=(1,len([0,1,2,3])), loc=(0,m), fig=fig)
#             for color_n, fff in enumerate(to_compare):
#                 with open(f'{paths.data}/{fff}', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                 epoch_len=[]
#                 for tseries in datafile:
#                     for roi in datafile[tseries]['final_rois']:
#                         epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#                 epochs = list(set.union(*map(set,epoch_len)))
#                 if isinstance(epochs[0], list):
#                     print('differnet epoch lenth acrross data!! ')
#                 mean_condition = pd.DataFrame({})
#                 for_std = pd.DataFrame({})
#                 for nn, tseries in enumerate(datafile):
#                     delta_mean, max_mean, area_mean = [], [], []
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             for epoch in roi.interpolated_traces_epochs:
#                                 if epoch == m:
#                                     trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                     start = np.argmax(trace_whole_epoch) - (fps*5)
#                                     if start < 0:
#                                         start  = 0
#                                     trace_before = trace_whole_epoch[start:np.argmax(trace_whole_epoch)+1]
#                                     delta = np.max(trace_whole_epoch) - np.min(trace_before)
#                                     # delta_mean.append(delta)
#                                     # max_mean.append(np.max(trace_whole_epoch))
#                                     # area_mean.append(np.trapz(trace_whole_epoch))
#                                     time_epoch = roi.interpolated_time.get(epoch)
#                                     mean_fly[n]=trace_whole_epoch
#                                     trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                                     for_std = pd.concat([trace_for_std, for_std], axis=1)
#                                     area= np.trapz(trace_whole_epoch)
#                                     if area>25:
#                                         area=25
#                                     try:
#                                         delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',delta]
#                                         peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',np.max(trace_whole_epoch)]
#                                         area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',area]
#                                         mm+=1
#                                     except:
#                                         continue
#                     # delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(delta_mean)]
#                     # peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(max_mean)]
#                     # area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(area_mean)]
#                     # mm+=1
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     # ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                 axes[counter_row][counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#                 yerr = for_std.sem(axis='columns').to_numpy()
#                 # yerr = for_std.std(axis='columns').to_numpy()
#                 ub = mean_condition.mean(axis=1) + yerr
#                 lb = mean_condition.mean(axis=1) - yerr
#                 axes[counter_row][counter_column].fill_between(time_epoch, ub, lb,color=color[color_n], alpha=.4)
#                 title = f'{roi.stim_info.get("angle")[m]}'
#                 axes[counter_row][counter_column].set_title(title)
#                 axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#             if counter_column < 3:
#                 counter_column+=1
#             elif counter_column == 3:
#                 counter_column =0
#                 counter_row+=1
#         fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
#         fig.supxlabel(f'time [s]', fontsize= 14)
#         fig.supylabel('ΔF/F', fontsize= 14)
#         fig.suptitle(f'{cat}', fontsize=18)
#         plt.tight_layout()
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_epoch_vs_{cat}_8D_mean_1.png', dpi=400, bbox_inches='tight')
#         plt.close('all')

#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#         sns.boxplot(data=delta_all,x='stim_con',y='delta', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#         # sns.stripplot(data=delta_all, x="stim_con", y="delta", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
#         statlog = f'{paths.results}/{odor}/_maxmin_{cat}_8D_stats_1.txt'
#         if os.path.exists(statlog)==True:
#             os.remove(statlog)
#         core_pre.stats_boxplot(statlog, delta_all, x='stim_con',y='delta', hue = "odor_con", lh=False)
#         ax.set(xlabel=None, ylabel=None)
#         ax.legend_.remove() 
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         fig.suptitle(f'{cat}', fontsize=18)
#         # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#         fig.supylabel(f'range', fontsize= 14)
#         fig.supxlabel('Stimulus', fontsize= 14)
#         fig.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_maxmin_{cat}_8D_1.png', dpi=400, bbox_inches='tight')
#         plt.close('all')
        
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        # sns.boxplot(data=peak_all,x='stim_con',y='peak', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
        # # sns.stripplot(data=peak_all, x="stim_con", y="peak", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
        # core_pre.stats_boxplot(statlog, peak_all, x='stim_con',y='peak', hue = "odor_con", lh=False)
        # ax.set(xlabel=None, ylabel=None)
        # ax.legend_.remove() 
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # fig.suptitle(f'{cat}', fontsize=18)
        # # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
        # fig.supylabel(f'max', fontsize= 14)
        # fig.supxlabel('Stimulus', fontsize= 14)
        # fig.tight_layout()
        # plt.savefig(f'{paths.results}/{odor}/_peak_{cat}_8D.png', dpi=400, bbox_inches='tight')
        # plt.close('all')
        
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        # sns.boxplot(data=area_all,x='stim_con',y='area', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
        # # sns.stripplot(data=area_all, x="stim_con", y="area", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
        # core_pre.stats_boxplot(statlog, area_all, x='stim_con',y='area', hue = "odor_con", lh=False)
        # ax.set(xlabel=None, ylabel=None)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.legend_.remove() 
        # fig.suptitle(f'{cat}', fontsize=18)
        # # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
        # fig.supylabel(f'area', fontsize= 14)
        # fig.supxlabel('Stimulus', fontsize= 14)
        # fig.tight_layout()
        # plt.savefig(f'{paths.results}/{odor}/_area_{cat}_8D_1.png', dpi=400, bbox_inches='tight')
        # plt.close('all')


# for odor in ['ACV', 'BA']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_8D_OL.pkl' ]
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#         all_patches=[]
#         for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
#             patch = mpatches.Patch(color=col, label=what)
#             all_patches.append(patch)
#         fig = plt.figure(dpi=400)
#         if os.path.exists(f'{paths.results}/{odor}/_whole_vs_{cat}_8D_mean.png')==False:
#             for color_n, fff in enumerate(to_compare):
#                 with open(f'{paths.data}/{fff}', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                 mean_condition = pd.DataFrame({})
#                 for_std = pd.DataFrame({})
#                 for nn, tseries in enumerate(datafile):
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             traces_whole_epoch = []
#                             for epoch in roi.interpolated_traces_epochs:
#                                 traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                             time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                             mean_fly[n]=traces_whole_epoch
#                             trace_for_std = pd.DataFrame({f'{n}': traces_whole_epoch})
#                             for_std = pd.concat([trace_for_std, for_std], axis=1)
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     # plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                 plt.plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#                 yerr = for_std.sem(axis='columns').to_numpy()
#                 ub = mean_condition.mean(axis=1)[:len(time_epoch)] + yerr
#                 lb = mean_condition.mean(axis=1)[:len(time_epoch)] - yerr
#                 plt.fill_between(time_epoch[:len(mean_condition.mean(axis=1))], ub, lb,color=color[color_n], alpha=.4)
#                 # plt.xlim(0,5)
#             timepoints = roi.stim_info.get('epoch_dur')
#             for idx, n in enumerate(timepoints):
#                 if idx > 0  and idx < len(timepoints):
#                     new_n = previous + n 
#                     timepoints[idx] = new_n 
#                     previous=new_n  
#                 else:
#                     previous=n  
#             for n in range(7):
#                 plt.axvline(x = timepoints[n], color = 'tab:blue')
#             plt.ylabel('ΔF/F', fontsize= 14)
#             plt.xlim(0,90)
#             plt.xlabel(f'time [s]', fontsize= 14)
#             plt.title(f'{cat}_{odor}', fontsize=18)
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_whole_vs_{cat}_8D_mean.png', dpi=400, bbox_inches='tight')
#             plt.close('all')
        
# for odor in ['ACV', 'BA']:
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = neuropile_color.get(cat)
#         # fig = plt.figure(dpi=400)
#         if os.path.exists(f'{paths.results}/{odor}/_whole_delta_{cat}_8D_fly.png')==False:
#             fig, axes = plt.subplots(nrows=10, ncols=1, sharex=True,figsize=(10,20)) #figsize=(10,20)
#             count=0 #
#             for fly in os.listdir(f"{paths.processed}/_{odor}_{odor}_8D_OL"):
#                 if fly.endswith('.png'):
#                     continue
#                 else:
#                     tseries_odor = os.listdir(f"{paths.processed}/_{odor}_{odor}_8D_OL/{fly}")[0]
#                     tseries_wo = os.listdir(f"{paths.processed}/_{odor}_WO_8D_OL/{fly}")[0]
#                     with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                     roi_odor = datafile[tseries_odor]['final_rois']
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(roi_odor):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             traces_whole_epoch = []
#                             for epoch in roi.interpolated_traces_epochs:
#                                 traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                             mean_fly[n]=traces_whole_epoch
#                     trace_odor = mean_fly.mean(axis=1)
#                     with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                     for tseries in datafile:
#                         timepoints = datafile[tseries]['final_rois'][0].stim_info.get('epoch_dur')
#                         for idx, n in enumerate(timepoints):
#                             if idx > 0  and idx < len(timepoints):
#                                 new_n = previous + n 
#                                 timepoints[idx] = new_n 
#                                 previous=new_n  
#                             else:
#                                 previous=n  
#                     roi_wo = datafile[tseries_wo]['final_rois']
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(roi_wo):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             traces_whole_epoch = []
#                             for epoch in roi.interpolated_traces_epochs:
#                                 traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                             time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                             mean_fly[n]=traces_whole_epoch
#                     trace_wo = mean_fly.mean(axis=1)
#                     # diff = trace_odor - trace_wo
#                     diff = trace_odor - trace_wo
#                     # diff = [a_i - b_i for a_i, b_i in zip(trace_odor, trace_wo)]
#                     # plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                     axes[count].plot(time_epoch, diff, color=color, zorder=100)
#                     for n in range(7):
#                         axes[count].axvline(x = timepoints[n], color = 'tab:blue')
#                     axes[count].set_ylim(-0.1,0.2)
#                     count+=1
                    
#             fig.supxlabel(f'time [s]', fontsize= 14)
#             fig.supylabel('ΔF/F', fontsize= 14)
#             fig.suptitle(f'{cat}_{odor}', fontsize=18)
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_8D_fly.png', dpi=400, bbox_inches='tight')
#             # plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_8D_fly.pdf', dpi=400, bbox_inches='tight')
#             plt.close('all')
# ##############################################BAR################################################
# ##############################################BAR################################################
# ##############################################BAR################################################
# ##############################################BAR################################################
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Bar' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
#     for tseries in datafile:
#             for cat in cats:
#                 # if os.path.exists(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_epoch_{cat}.png')==False:
#                     fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#                     counter_row =0
#                     counter_column = 0
#                     for m, key in enumerate(epochs):
#                         mean_fly=pd.DataFrame({})
#                         # ax = plt.subplot2grid(shape=(int(len(roi.interpolated_traces_epochs)/2),int(len(roi.interpolated_traces_epochs)/2)), loc=(counter_row,counter_column), fig=fig)
#                         for n, roi in enumerate(datafile[tseries]['final_rois']):
#                             if roi.category == cat:
#                                 fps = roi.alligned_fps
#                                 for epoch in roi.interpolated_traces_epochs:
#                                     if epoch == key:
#                                         trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                         time_epoch = roi.interpolated_time.get(epoch)
#                                         axes[counter_row][counter_column].plot(time_epoch, trace_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                                         mean_fly[n]=trace_whole_epoch
#                         try:
#                             axes[counter_row][counter_column].plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                         except:
#                             continue
#                         title = f'{roi.stim_info.get("angle")[key]}'
#                         axes[counter_row][counter_column].set_title(title)
#                         axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#                         # plt.ylabel('ΔF/F', fontsize= 14)
#                         # plt.xlabel(f'time [s]', fontsize= 14)
#                         if counter_column < 3:
#                             counter_column+=1
#                         elif counter_column == 3:
#                             counter_column =0
#                             counter_row+=1
#                     fig.supxlabel(f'time [s]', fontsize= 14)
#                     fig.supylabel('ΔF/F', fontsize= 14)
#                     fig.suptitle(f'{tseries} : {cat}', fontsize=18)
#                     plt.tight_layout()
#                     plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_epoch_{cat}.png', dpi=400, bbox_inches='tight')
#                     plt.close('all')
# TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Bar' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats,epoch_len=[],[]
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     epochs = list(set.union(*map(set,epoch_len)))
#     if isinstance(epochs[0], list):
#         print('differnet epoch lenth acrross data!! ')
#     for cat in cats:
# #         if os.path.exists(f'{paths.results}/{fff.split(".")[0]}/_epoch_{cat}_mean.png')==False:
#             fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#             counter_row =0
#             counter_column = 0
#             for m, key in enumerate(epochs):
#                 mean_condition = pd.DataFrame({})
#                 # ax = plt.subplot2grid(shape=(1,len(epochs)), loc=(0,m), fig=fig)
#                 for nn, tseries in enumerate(datafile):
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             for epoch in roi.interpolated_traces_epochs:
#                                 if epoch == key:
#                                     trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                     time_epoch = roi.interpolated_time.get(epoch)
#                                     mean_fly[n]=trace_whole_epoch
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     try:
#                         axes[counter_row][counter_column].plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                     except:
#                         continue
#                 axes[counter_row][counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                 title = f'{roi.stim_info.get("angle")[key]}'
#                 axes[counter_row][counter_column].set_title(title)
#                 axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#                 if counter_column < 3:
#                     counter_column+=1
#                 elif counter_column == 3:
#                     counter_column =0
#                     counter_row+=1
#                 # plt.xlim(0,5)
#                 # plt.ylabel('ΔF/F', fontsize= 14)
#                 # plt.xlabel(f'time [s]', fontsize= 14)
#             fig.supxlabel(f'time [s]', fontsize= 14)
#             fig.supylabel('ΔF/F', fontsize= 14)
#             fig.suptitle(f'{cat}', fontsize=18)
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_epoch_{cat}_mean.png', dpi=400, bbox_inches='tight')
#             plt.close('all')

# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Bar' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     for tseries in datafile:
#         timepoints = datafile[tseries]['final_rois'][0].stim_info.get('epoch_dur')
#         for idx, n in enumerate(timepoints):
#             if idx > 0  and idx < len(timepoints):
#                 new_n = previous + n 
#                 timepoints[idx] = new_n 
#                 previous=new_n  
#             else:
#                 previous=n  
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for tseries in datafile:
#         for cat in cats:
# #             if os.path.exists(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_whole_{cat}.png') == False:
#                 fig = plt.figure(dpi=400) #figsize=(10,20)
#                 # fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True) #figsize=(10,20)
#                 mean_fly=pd.DataFrame({})
#                 # ax = plt.subplot2grid(shape=(int(len(roi.interpolated_traces_epochs)/2),int(len(roi.interpolated_traces_epochs)/2)), loc=(counter_row,counter_column), fig=fig)
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         # trace_whole_epoch = roi.roi.oneHz_conc_resp
#                         traces_whole_epoch = []
#                         for epoch in roi.interpolated_traces_epochs:
#                             traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                         time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                         mean_fly[n]=traces_whole_epoch
#                         try:
#                             plt.plot(time_epoch, traces_whole_epoch, color='tab:gray', zorder=(0-n)*-1)
#                         except:
#                             continue                   
                                
#                 for n in range(7):
#                     plt.axvline(x = timepoints[n], color = 'tab:blue')
#                 try:
#                     plt.plot(time_epoch, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#                     plt.title(f'{tseries} : {cat}', fontsize=18)
#                     plt.xlim(0,max(time_epoch))
#                     plt.ylabel('ΔF/F', fontsize= 14)
#                     plt.xlabel(f'time [s]', fontsize= 14)
#                     plt.tight_layout()
#                     plt.savefig(f'{dataset_folder}/4_results/{fff.split(".")[0]}/{tseries}/_whole_{cat}.png', dpi=400, bbox_inches='tight')
#                     plt.close('all')
#                 except:
#                     continue
# TODO: plot trial average per fly + mean of condition 
# pkls=[]
# for condition in os.listdir(paths.data):
#     if 'Bar' in condition:
#         pkls.append(condition)
# for fff in pkls:
#     with open(f'{paths.data}/{fff}', 'rb') as fi:
#             datafile = pickle.load(fi)
#     cats=[]
#     # timepoints = datafile[0]['final_rois'][0].stim_info.get('epoch_dur')
#     # for idx, n in enumerate(timepoints):
#     #     if idx > 0  and idx < len(timepoints):
#     #         new_n = previous + n 
#     #         timepoints[idx] = new_n 
#     #         previous=new_n  
#     #     else:
#     #         previous=n  
#     for tseries in datafile:
#         for roi in datafile[tseries]['final_rois']:
#             if roi.category == ['No_category']:
#                 continue
#             cats.append(roi.category)
#     cats = list(set(cats))
#     for cat in cats:
#         fig = plt.figure(dpi=400) #figsize=(10,20)
#         mean_condition = pd.DataFrame({})
#         for nn, tseries in enumerate(datafile):
#             mean_fly=pd.DataFrame({})
#             for n, roi in enumerate(datafile[tseries]['final_rois']):
#                 if roi.category == cat:
#                     fps = roi.alligned_fps
#                     traces_whole_epoch = []
#                     for epoch in roi.interpolated_traces_epochs:
#                         traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                     mean_fly[n]=traces_whole_epoch
#                     time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#             mean_condition[nn] = mean_fly.mean(axis=1)
#             try:
#                 plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#             except:
#                 continue
#         # timepoints = roi.stim_info.get('epoch_dur')
#         # for idx, n in enumerate(timepoints):
#         #     if idx > 0  and idx < len(timepoints):
#         #         new_n = previous + n 
#         #         timepoints[idx] = new_n 
#         #         previous=new_n  
#         #     else:
#         #         previous=n  
#         for n in range(7):
#             plt.axvline(x = timepoints[n], color = 'tab:blue')
#         plt.plot(time_epoch, mean_condition.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
#         plt.title(f'{tseries} : {cat}', fontsize=18)
#         plt.xlim(0,max(time_epoch))
#         plt.ylabel('ΔF/F', fontsize= 14)
#         plt.xlabel(f'time [s]', fontsize= 14)
#         plt.tight_layout()
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{fff.split(".")[0]}/_whole_{cat}_mean.png', dpi=400, bbox_inches='tight')
#         plt.close('all')

# for odor in ['ACV', 'BA']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     for stim in ['BBar50', 'BBar100', 'SBar50', 'SBar100']:
#         to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_{stim}_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_{stim}_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_{stim}_OL.pkl' ]
#         if preprocessing_params.experiment == 'big':
#             cats = ['ME', 'LO', "LOP"]
#         elif preprocessing_params.experiment == 'layer':
#             cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#         for cat in cats:
#             delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
#             peak_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'peak'])
#             area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
#             color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#             color_pal = {f'{odor}':neuropile_color.get(cat), 'No Odor':'#BCBDC4', 'without':'#5A5D59'}
#             mm=0
#             all_patches=[]
#             for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
#                 patch = mpatches.Patch(color=col, label=what)
#                 all_patches.append(patch)
#             fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#             counter_row , counter_column=0,0
#             for m in [0,1,2,3,4,5,6,7]:
#                 # ax = plt.subplot2grid(shape=(1,len([0,1,2,3])), loc=(0,m), fig=fig)
#                 for color_n, fff in enumerate(to_compare):
#                     with open(f'{paths.data}/{fff}', 'rb') as fi:
#                             datafile = pickle.load(fi)
#                     epoch_len=[]
#                     for tseries in datafile:
#                         for roi in datafile[tseries]['final_rois']:
#                             epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#                     epochs = list(set.union(*map(set,epoch_len)))
#                     if isinstance(epochs[0], list):
#                         print('differnet epoch lenth acrross data!! ')
#                     mean_condition = pd.DataFrame({})
#                     for_std = pd.DataFrame({})
#                     for nn, tseries in enumerate(datafile):
#                         delta_mean, max_mean, area_mean = [], [], []
#                         mean_fly=pd.DataFrame({})
#                         for n, roi in enumerate(datafile[tseries]['final_rois']):
#                             if roi.category == cat:
#                                 fps = roi.alligned_fps
#                                 for epoch in roi.interpolated_traces_epochs:
#                                     if epoch == m:
#                                         trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                         delta = np.max(trace_whole_epoch) - np.min(trace_whole_epoch)
#                                         # delta_mean.append(delta)
#                                         # max_mean.append(np.max(trace_whole_epoch))
#                                         # area_mean.append(np.trapz(trace_whole_epoch))
#                                         time_epoch = roi.interpolated_time.get(epoch)
#                                         mean_fly[n]=trace_whole_epoch
#                                         trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                                         for_std = pd.concat([trace_for_std, for_std], axis=1)
#                                         area= np.trapz(trace_whole_epoch)
#                                         if area>25:
#                                             area=25
#                                         try:
#                                             delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',delta]
#                                             peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',np.max(trace_whole_epoch)]
#                                             area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',area]
#                                             mm+=1
#                                         except:
#                                             continue
#                         # delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(delta_mean)]
#                         # peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(max_mean)]
#                         # area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(area_mean)]
#                         # mm+=1
#                         mean_condition[nn] = mean_fly.mean(axis=1)
#                         # ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                     axes[counter_row][counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#                     yerr = for_std.sem(axis='columns').to_numpy()
#                     ub = mean_condition.mean(axis=1) + yerr
#                     lb = mean_condition.mean(axis=1) - yerr
#                     axes[counter_row][counter_column].fill_between(time_epoch, ub, lb,color=color[color_n], alpha=.4)
#                     title = f'{roi.stim_info.get("angle")[m]}'
#                     axes[counter_row][counter_column].set_title(title)
#                     axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#                 if counter_column < 3:
#                     counter_column+=1
#                 elif counter_column == 3:
#                     counter_column =0
#                     counter_row+=1
#             fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
#             fig.supxlabel(f'time [s]', fontsize= 14)
#             fig.supylabel('ΔF/F', fontsize= 14)
#             fig.suptitle(f'{cat}', fontsize=18)
#             plt.tight_layout()
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_epoch_vs_{cat}_{stim}_mean.png', dpi=400, bbox_inches='tight')
#             plt.close('all')

#             fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#             sns.boxplot(data=delta_all,x='stim_con',y='delta', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#             # sns.stripplot(data=delta_all, x="stim_con", y="delta", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
#             statlog = f'{paths.results}/{odor}/_maxmin_{cat}_{stim}_stats.txt'
#             if os.path.exists(statlog)==True:
#                 os.remove(statlog)
#             # core_pre.stats_boxplot(statlog, delta_all, x='stim_con',y='delta', hue = "odor_con", lh=False)
#             ax.set(xlabel=None, ylabel=None)
#             ax.legend_.remove() 
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
#             fig.suptitle(f'{cat}', fontsize=18)
#             # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#             fig.supylabel(f'max - min', fontsize= 14)
#             fig.supxlabel('Stimulus', fontsize= 14)
#             fig.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_maxmin_{cat}_{stim}.png', dpi=400, bbox_inches='tight')
#             plt.close('all')
            
#             fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#             sns.boxplot(data=peak_all,x='stim_con',y='peak', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#             # sns.stripplot(data=peak_all, x="stim_con", y="peak", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
#             # core_pre.stats_boxplot(statlog, peak_all, x='stim_con',y='peak', hue = "odor_con", lh=False)
#             ax.set(xlabel=None, ylabel=None)
#             ax.legend_.remove() 
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
#             fig.suptitle(f'{cat}', fontsize=18)
#             # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#             fig.supylabel(f'max', fontsize= 14)
#             fig.supxlabel('Stimulus', fontsize= 14)
#             fig.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_peak_{cat}_{stim}.png', dpi=400, bbox_inches='tight')
#             plt.close('all')
            
#             fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#             sns.boxplot(data=area_all,x='stim_con',y='area', hue ='odor_con', palette=color_pal, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
#             # sns.stripplot(data=area_all, x="stim_con", y="area", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
#             # core_pre.stats_boxplot(statlog, area_all, x='stim_con',y='area', hue = "odor_con", lh=False)
#             ax.set(xlabel=None, ylabel=None)
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
#             ax.legend_.remove() 
#             fig.suptitle(f'{cat}', fontsize=18)
#             # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
#             fig.supylabel(f'area', fontsize= 14)
#             fig.supxlabel('Stimulus', fontsize= 14)
#             fig.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_area_{cat}_{stim}.png', dpi=400, bbox_inches='tight')
#             plt.close('all')


# for odor in ['ACV', 'BA']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_8D_OL.pkl' ]
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#         all_patches=[]
#         for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
#             patch = mpatches.Patch(color=col, label=what)
#             all_patches.append(patch)
#         fig = plt.figure(dpi=400)
#         if os.path.exists(f'{paths.results}/{odor}/_whole_vs_{cat}_8D_mean.png')==False:
#             for color_n, fff in enumerate(to_compare):
#                 with open(f'{paths.data}/{fff}', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                 mean_condition = pd.DataFrame({})
#                 for_std = pd.DataFrame({})
#                 for nn, tseries in enumerate(datafile):
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             traces_whole_epoch = []
#                             for epoch in roi.interpolated_traces_epochs:
#                                 traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                             time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                             mean_fly[n]=traces_whole_epoch
#                             trace_for_std = pd.DataFrame({f'{n}': traces_whole_epoch})
#                             for_std = pd.concat([trace_for_std, for_std], axis=1)
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     # plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                 plt.plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#                 yerr = for_std.sem(axis='columns').to_numpy()
#                 ub = mean_condition.mean(axis=1)[:len(time_epoch)] + yerr
#                 lb = mean_condition.mean(axis=1)[:len(time_epoch)] - yerr
#                 plt.fill_between(time_epoch[:len(mean_condition.mean(axis=1))], ub, lb,color=color[color_n], alpha=.4)
#                 # plt.xlim(0,5)
#             timepoints = roi.stim_info.get('epoch_dur')
#             for idx, n in enumerate(timepoints):
#                 if idx > 0  and idx < len(timepoints):
#                     new_n = previous + n 
#                     timepoints[idx] = new_n 
#                     previous=new_n  
#                 else:
#                     previous=n  
#             for n in range(7):
#                 plt.axvline(x = timepoints[n], color = 'tab:blue')
#             plt.ylabel('ΔF/F', fontsize= 14)
#             plt.xlim(0,90)
#             plt.xlabel(f'time [s]', fontsize= 14)
#             plt.title(f'{cat}_{odor}', fontsize=18)
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_whole_vs_{cat}_8D_mean.png', dpi=400, bbox_inches='tight')
#             plt.close('all')
        
# for odor in ['ACV', 'BA']:
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#     for cat in cats:
#         color = neuropile_color.get(cat)
#         # fig = plt.figure(dpi=400)
#         if os.path.exists(f'{paths.results}/{odor}/_whole_delta_{cat}_8D_fly.png')==False:
#             fig, axes = plt.subplots(nrows=10, ncols=1, sharex=True,figsize=(10,20)) #figsize=(10,20)
#             count=0 #
#             for fly in os.listdir(f"{paths.processed}/_{odor}_{odor}_8D_OL"):
#                 if fly.endswith('.png'):
#                     continue
#                 else:
#                     tseries_odor = os.listdir(f"{paths.processed}/_{odor}_{odor}_8D_OL/{fly}")[0]
#                     tseries_wo = os.listdir(f"{paths.processed}/_{odor}_WO_8D_OL/{fly}")[0]
#                     with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                     roi_odor = datafile[tseries_odor]['final_rois']
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(roi_odor):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             traces_whole_epoch = []
#                             for epoch in roi.interpolated_traces_epochs:
#                                 traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                             mean_fly[n]=traces_whole_epoch
#                     trace_odor = mean_fly.mean(axis=1)
#                     with open(f'{paths.data}/{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                     for tseries in datafile:
#                         timepoints = datafile[tseries]['final_rois'][0].stim_info.get('epoch_dur')
#                         for idx, n in enumerate(timepoints):
#                             if idx > 0  and idx < len(timepoints):
#                                 new_n = previous + n 
#                                 timepoints[idx] = new_n 
#                                 previous=new_n  
#                             else:
#                                 previous=n  
#                     roi_wo = datafile[tseries_wo]['final_rois']
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(roi_wo):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             traces_whole_epoch = []
#                             for epoch in roi.interpolated_traces_epochs:
#                                 traces_whole_epoch.extend(roi.interpolated_traces_epochs[epoch])
#                             time_epoch = np.linspace(0, len(traces_whole_epoch)/fps, len(traces_whole_epoch))
#                             mean_fly[n]=traces_whole_epoch
#                     trace_wo = mean_fly.mean(axis=1)
#                     # diff = trace_odor - trace_wo
#                     diff = trace_odor - trace_wo
#                     # diff = [a_i - b_i for a_i, b_i in zip(trace_odor, trace_wo)]
#                     # plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                     axes[count].plot(time_epoch, diff, color=color, zorder=100)
#                     for n in range(7):
#                         axes[count].axvline(x = timepoints[n], color = 'tab:blue')
#                     axes[count].set_ylim(-0.1,0.2)
#                     count+=1
                    
#             fig.supxlabel(f'time [s]', fontsize= 14)
#             fig.supylabel('ΔF/F', fontsize= 14)
#             fig.suptitle(f'{cat}_{odor}', fontsize=18)
#             plt.tight_layout()
#             plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_8D_fly.png', dpi=400, bbox_inches='tight')
#             # plt.savefig(f'{paths.results}/{odor}/_whole_delta_{cat}_8D_fly.pdf', dpi=400, bbox_inches='tight')
#             plt.close('all')


# ##############################################ODOR delta################################################
# ##############################################ODOR delta################################################
# ##############################################ODOR delta################################################
# ##############################################ODOR delta################################################
##############################################8D################################################
# os.makedirs(f'{paths.results}/odor_delta', exist_ok=True)
# to_compare = [f'{preprocessing_params.experiment}_ACV_ACV_8D_OL.pkl', f'{preprocessing_params.experiment}_BA_BA_8D_OL.pkl']
# odor_color = {'ACV' : 'tab:orange', "BA": 'tab:cyan', 'WO': 'tab:grey'}
# if preprocessing_params.experiment == 'big':
#     cats = ['ME', 'LO', "LOP"]
# elif preprocessing_params.experiment == 'layer':
#     cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
# for cat in cats:
#     delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
#     area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
#     color = ['tab:orange', 'tab:cyan']
#     mm=0
#     all_patches=[]
#     for col, what in zip(color, ['ACV', 'BA']):
#         patch = mpatches.Patch(color=col, label=what)
#         all_patches.append(patch)
#     fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#     counter_row , counter_column=0,0
#     for m in [0,1,2,3,4,5,6,7]:
#         for color_n, fff in enumerate(to_compare):
#             with open(f'{paths.data}/{fff}', 'rb') as fi:
#                     datafile = pickle.load(fi)
#             epoch_len=[]
#             for tseries in datafile:
#                 for roi in datafile[tseries]['final_rois']:
#                     epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             epochs = list(set.union(*map(set,epoch_len)))
#             if isinstance(epochs[0], list):
#                 print('differnet epoch lenth acrross data!! ')
#             mean_condition = pd.DataFrame({})
#             for_std = pd.DataFrame({})
#             for nn, tseries in enumerate(datafile):
#                 delta_mean, max_mean, area_mean = [], [], []
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         for epoch in roi.interpolated_traces_epochs:
#                             if epoch == m:
#                                 trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                 delta = np.max(trace_whole_epoch) - np.min(trace_whole_epoch)
#                                 time_epoch = roi.interpolated_time.get(epoch)
#                                 mean_fly[n]=trace_whole_epoch
#                                 trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                                 for_std = pd.concat([trace_for_std, for_std], axis=1)
#                                 area= np.trapz(trace_whole_epoch)
#                                 if area>25:
#                                     area=25
#                                 try:
#                                     delta_all.loc[mm] = [['ACV', 'BA'][color_n],f'{roi.stim_info.get("angle")[m]}',delta]
#                                     area_all.loc[mm] = [['ACV', 'BA'][color_n],f'{roi.stim_info.get("angle")[m]}',area]
#                                     mm+=1
#                                 except:
#                                     continue
#                 mean_condition[nn] = mean_fly.mean(axis=1)
#                 # ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#             axes[counter_row][counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#             yerr = for_std.sem(axis='columns').to_numpy()
#             ub = mean_condition.mean(axis=1) + yerr
#             lb = mean_condition.mean(axis=1) - yerr
#             axes[counter_row][counter_column].fill_between(time_epoch, ub, lb,color=color[color_n], alpha=.4)
#             title = f'{roi.stim_info.get("angle")[m]}'
#             axes[counter_row][counter_column].set_title(title)
#             axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#         if counter_column < 3:
#             counter_column+=1
#         elif counter_column == 3:
#             counter_column =0
#             counter_row+=1
#     fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
#     fig.supxlabel(f'time [s]', fontsize= 14)
#     fig.supylabel('ΔF/F', fontsize= 14)
#     fig.suptitle(f'{cat}', fontsize=18)
#     plt.tight_layout()
#     plt.tight_layout()
#     plt.savefig(f'{paths.results}/odor_delta/_epoch_vs_{cat}_8D_mean.png', dpi=400, bbox_inches='tight')
#     plt.close('all')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    # sns.boxplot(data=delta_all,x='stim_con',y='delta', hue ='odor_con', palette=odor_color, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
    # # sns.stripplot(data=delta_all, x="stim_con", y="delta", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
    # statlog = f'{paths.results}/odor_delta/_maxmin_{cat}_8D_stats.txt'
    # if os.path.exists(statlog)==True:
    #     os.remove(statlog)
    # core_pre.stats_boxplot(statlog, delta_all, x='stim_con',y='delta', hue = "odor_con", lh=False)
    # ax.set(xlabel=None, ylabel=None)
    # ax.legend_.remove() 
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.suptitle(f'{cat}', fontsize=18)
    # # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
    # fig.supylabel(f'max - min', fontsize= 14)
    # fig.supxlabel('Stimulus', fontsize= 14)
    # fig.tight_layout()
    # plt.savefig(f'{paths.results}/odor_delta/_maxmin_{cat}_8D.png', dpi=400, bbox_inches='tight')
    # plt.close('all')
    
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    # sns.boxplot(data=area_all,x='stim_con',y='area', hue ='odor_con', palette=odor_color, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
    # # sns.stripplot(data=area_all, x="stim_con", y="area", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
    # core_pre.stats_boxplot(statlog, area_all, x='stim_con',y='area', hue = "odor_con", lh=False)
    # ax.set(xlabel=None, ylabel=None)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.legend_.remove() 
    # fig.suptitle(f'{cat}', fontsize=18)
    # # fig.supylabel(f'Δ{what} (before - after)', fontsize= 14)
    # fig.supylabel(f'area', fontsize= 14)
    # fig.supxlabel('Stimulus', fontsize= 14)
    # fig.tight_layout()
    # plt.savefig(f'{paths.results}/odor_delta/_area_{cat}_8D.png', dpi=400, bbox_inches='tight')
    # plt.close('all')



##############################################Grating################################################
# os.makedirs(f'{paths.results}/odor_delta', exist_ok=True)
# to_compare = [f'{preprocessing_params.experiment}_ACV_ACV_Grating_OL.pkl', f'{preprocessing_params.experiment}_BA_BA_Grating_OL.pkl']
# if preprocessing_params.experiment == 'big':
#     cats = ['ME', 'LO', "LOP"]
# elif preprocessing_params.experiment == 'layer':
#     cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
# odor_color = {'ACV' : 'tab:orange', "BA": 'tab:cyan'}
# for cat in cats:
#     delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
#     area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
#     mm = 0
#     colorrr = ['tab:orange', 'tab:cyan']
#     all_patches=[]
#     for col, what in zip(colorrr, ['ACV', 'BA']):
#         patch = mpatches.Patch(color=col, label=what)
#         all_patches.append(patch)
#     fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True, figsize =(7,4)) #, sharex=True
#     counter_row , counter_column=0,0
#     for m in [0,1,2,3]:
#         # ax = plt.subplot2grid(shape=(1,len([0,1,2,3])), loc=(0,m), fig=fig)
#         for color_n, fff in enumerate(to_compare):
#             with open(f'{paths.data}/{fff}', 'rb') as fi:
#                     datafile = pickle.load(fi)
#             epoch_len=[]
#             for tseries in datafile:
#                 for roi in datafile[tseries]['final_rois']:
#                     epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             epochs = list(set.union(*map(set,epoch_len)))
#             if isinstance(epochs[0], list):
#                 print('differnet epoch lenth acrross data!! ')
#             mean_condition = pd.DataFrame({})
#             for_std = pd.DataFrame({})
#             for nn, tseries in enumerate(datafile):
#                 delta_mean, max_mean, area_mean = [], [], []
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         for epoch in roi.interpolated_traces_epochs:
#                             if epoch == m+1:
#                                 trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                 delta = np.max(trace_whole_epoch) - np.min(trace_whole_epoch)
#                                 delta_mean.append(delta)
#                                 max_mean.append(np.max(trace_whole_epoch))
#                                 area_mean.append(np.trapz(trace_whole_epoch))
#                                 time_epoch = roi.interpolated_time.get(epoch)
#                                 mean_fly[n]=trace_whole_epoch
#                                 trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                                 for_std = pd.concat([trace_for_std, for_std], axis=1)
#                                 try:
#                                     delta_all.loc[mm] = [['ACV', 'BA'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',delta]
#                                     area_all.loc[mm] = [['ACV', 'BA'][color_n],f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}',np.trapz(trace_whole_epoch)]
#                                     mm+=1
#                                 except:
#                                     continue
#                 mean_condition[nn] = mean_fly.mean(axis=1)
#                 # ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#             axes[counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=colorrr[color_n], zorder=100)
#             yerr = for_std.sem(axis='columns').to_numpy()
#             ub = mean_condition.mean(axis=1) + yerr
#             lb = mean_condition.mean(axis=1) - yerr
#             axes[counter_column].fill_between(time_epoch, ub, lb,color=colorrr[color_n], alpha=.4)
#             title = f'{roi.stim_info.get("orientation")[m]}_{roi.stim_info.get("direction")[m]}'
#             axes[counter_column].set_title(title)
#             axes[counter_column].set_xlim(0,4)
#         counter_column+=1
#     fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
#     fig.supxlabel(f'time [s]', fontsize= 14)
#     fig.supylabel('ΔF/F', fontsize= 14)
#     fig.suptitle(f'{cat}', fontsize=18)
#     plt.tight_layout()
#     plt.tight_layout()
#     plt.savefig(f'{paths.results}/odor_delta/_epoch_vs_{cat}_Grating_mean.png', dpi=400, bbox_inches='tight')
#     plt.close('all')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    # sns.boxplot(data=delta_all,x='stim_con',y='delta', hue ='odor_con',palette=odor_color, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
    # # sns.stripplot(data=delta_all, x="stim_con", y="delta", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
    # statlog = f'{paths.results}/odor_delta/_maxmin_{cat}_Grating_stats.txt'
    # if os.path.exists(statlog) ==True:
    #     os.remove(statlog)
    # core_pre.stats_boxplot(statlog, delta_all, x='stim_con',y='delta', hue = "odor_con", lh=False)
    # ax.set(xlabel=None, ylabel=None)
    # ax.legend_.remove() 
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.suptitle(f'{cat}', fontsize=18)
    # fig.supylabel(f'max - min', fontsize= 14)
    # fig.supxlabel('Stimulus', fontsize= 14)
    # fig.tight_layout()
    # plt.savefig(f'{paths.results}/odor_delta/_maxmin_{cat}_Grating.png', dpi=400, bbox_inches='tight')
    # plt.close('all')
    
    
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    # sns.boxplot(data=area_all,x='stim_con',y='area', hue ='odor_con', palette=odor_color, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
    # # sns.stripplot(data=area_all, x="stim_con", y="area", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
    # core_pre.stats_boxplot(statlog, area_all, x='stim_con',y='area', hue = "odor_con", lh=False)
    # ax.set(xlabel=None, ylabel=None)
    # ax.legend_.remove() 
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.suptitle(f'{cat}', fontsize=18)
    # fig.supylabel(f'area', fontsize= 14)
    # fig.supxlabel('Stimulus', fontsize= 14)
    # fig.tight_layout()
    # plt.savefig(f'{paths.results}/odor_delta/_area_{cat}_Grating.png', dpi=400, bbox_inches='tight')
    # plt.close('all')
    
#############################################FFF################################################
# os.makedirs(f'{paths.results}/odor_delta', exist_ok=True)
# to_compare = [f'{preprocessing_params.experiment}_ACV_ACV_FFF_OL.pkl', f'{preprocessing_params.experiment}_BA_BA_FFF_OL.pkl']
# if preprocessing_params.experiment == 'big':
#     cats = ['ME', 'LO', "LOP"]
# elif preprocessing_params.experiment == 'layer':
#     cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
# odor_color = {'ACV' : 'tab:orange', "BA": 'tab:cyan'}
# for cat in cats:
#     delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
#     area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
#     color = ['tab:orange', 'tab:cyan']
#     mm=0
#     all_patches=[]
#     for col, what in zip(color, ['ACV', 'BA']):
#         patch = mpatches.Patch(color=col, label=what)
#         all_patches.append(patch)
#     fig = plt.figure(dpi=400, figsize=(5,20))
#     for m in [0,1]:
#         if m == 0:
#             ax = plt.subplot2grid(shape=(1,len([0,1])), loc=(0,m), fig=fig)
#         else:
#             ax = plt.subplot2grid(shape=(1,len([0,1])), loc=(0,m), fig=fig, sharey=ax)
#         for color_n, fff in enumerate(to_compare):
#             with open(f'{paths.data}/{fff}', 'rb') as fi:
#                     datafile = pickle.load(fi)
#             epoch_len=[]
#             for tseries in datafile:
#                 for roi in datafile[tseries]['final_rois']:
#                     epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#             epochs = list(set.union(*map(set,epoch_len)))
#             if isinstance(epochs[0], list):
#                 print('differnet epoch lenth acrross data!! ')
#             mean_condition = pd.DataFrame({})
#             for_std = pd.DataFrame({})
#             for nn, tseries in enumerate(datafile):
#                 delta_mean, max_mean, area_mean = [], [], []
#                 mean_fly=pd.DataFrame({})
#                 for n, roi in enumerate(datafile[tseries]['final_rois']):
#                     if roi.category == cat:
#                         fps = roi.alligned_fps
#                         for epoch in roi.interpolated_traces_epochs:
#                             if epoch == m:
#                                 trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                 trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                                 delta = np.max(trace_whole_epoch) - np.min(trace_whole_epoch)
#                                 delta_mean.append(delta)
#                                 max_mean.append(np.max(trace_whole_epoch))
#                                 area_mean.append(np.trapz(trace_whole_epoch))
#                                 time_epoch = roi.interpolated_time.get(epoch)
#                                 mean_fly[n]=trace_whole_epoch
#                                 for_std = pd.concat([trace_for_std, for_std], axis=1)
#                                 delta_all.loc[mm] = [['ACV', 'BA'][color_n],f'{roi.stim_info.get("fg")[m]}',delta]
#                                 area_all.loc[mm] = [['ACV', 'BA'][color_n],f'{roi.stim_info.get("fg")[m]}',np.trapz(trace_whole_epoch)]
#                                 mm+=1
#                 # delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(delta_mean)]
#                     # area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(area_mean)]
#                     # mm+=1
#                 mean_condition[nn] = mean_fly.mean(axis=1)
#             ax.plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#             yerr = for_std.sem(axis='columns').to_numpy()
#             ub = mean_condition.mean(axis=1) + yerr
#             lb = mean_condition.mean(axis=1) - yerr
#             ax.fill_between(time_epoch, ub, lb,color=color[color_n], alpha=.4)
#             plt.title(f'{roi.stim_info.get("fg")[m]}')
#     fig.supylabel('ΔF/F', fontsize= 14)
#     fig.supxlabel(f'time [s]', fontsize= 14)
#     fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.1), frameon=False)
#     fig.suptitle(f'{cat}', fontsize=18)
#     plt.tight_layout()
#     plt.savefig(f'{paths.results}/odor_delta/_epoch_vs_{cat}_FFF_mean.png', dpi=400, bbox_inches='tight')
#     plt.close('all')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    # sns.boxplot(data=delta_all,x='stim_con',y='delta', hue ='odor_con', palette=odor_color, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
    # # sns.stripplot(data=delta_all, x="stim_con", y="delta", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
    # statlog = f'{paths.results}/odor_delta/_maxmin_{cat}_FFF_stats.txt'
    # if os.path.exists(statlog) == True:
    #     os.remove(statlog)
    # core_pre.stats_boxplot(statlog, delta_all, x='stim_con',y='delta', hue = "odor_con", lh=False)
    # ax.set(xlabel=None, ylabel=None)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.suptitle(f'{cat}', fontsize=18)
    # fig.supylabel(f'max - min', fontsize= 14)
    # fig.supxlabel('Stimulus', fontsize= 14)
    # fig.tight_layout()
    # plt.savefig(f'{paths.results}/odor_delta/_maxmin_{cat}_FFF.png', dpi=400, bbox_inches='tight')
    # plt.close('all')
    
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    # sns.boxplot(data=area_all,x='stim_con',y='area', hue ='odor_con', palette=odor_color, showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'}) 
    # # sns.stripplot(data=area_all, x="stim_con", y="area", hue ='odor_con', dodge=True, jitter = True , ax=ax, alpha=0.5,  palette=color_pal, legend=False)
    # core_pre.stats_boxplot(statlog, area_all, x='stim_con',y='area', hue = "odor_con", lh=False)
    # ax.set(xlabel=None, ylabel=None)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.suptitle(f'{cat}', fontsize=18)
    # fig.supylabel(f'area', fontsize= 14)
    # fig.supxlabel('Stimulus', fontsize= 14)
    # fig.tight_layout()
    # plt.savefig(f'{paths.results}/odor_delta/_area_{cat}_FFF.png', dpi=400, bbox_inches='tight')
    # plt.close('all')
    

# os.makedirs(f'{paths.results}/odor_delta', exist_ok=True)
# to_compare = [f'{preprocessing_params.experiment}_ACV_ACV_FFF_OL.pkl', f'{preprocessing_params.experiment}_BA_BA_FFF_OL.pkl']
# if preprocessing_params.experiment == 'big':
#     cats = ['ME', 'LO', "LOP"]
# elif preprocessing_params.experiment == 'layer':
#     cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
# odor_color = {'ACV' : 'tab:orange', "BA": 'tab:cyan'}
# for cat in cats:
#     color = ['tab:orange', 'tab:cyan', 'tab:grey']
#     all_patches=[]
#     for col, what in zip(color, ['ACV', 'BA']):
#         patch = mpatches.Patch(color=col, label=what)
#         all_patches.append(patch)
#     fig = plt.figure(dpi=400)
#     for color_n, fff in enumerate(to_compare):
#         with open(f'{paths.data}/{fff}', 'rb') as fi:
#                 datafile = pickle.load(fi)
#         mean_condition = pd.DataFrame({})
#         for_std = pd.DataFrame({})
#         for nn, tseries in enumerate(datafile):
#             mean_fly=pd.DataFrame({})
#             for n, roi in enumerate(datafile[tseries]['final_rois']):
#                 if roi.category == cat:
#                     fps = roi.alligned_fps
#                     trace_whole_epoch = roi.int_con_trace[1]
#                     time_epoch = roi.int_con_trace[0]
#                     mean_fly[n]=trace_whole_epoch
#                     trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                     for_std = pd.concat([trace_for_std, for_std], axis=1)
#             mean_condition[nn] = mean_fly.mean(axis=1)
#             # plt.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#         plt.plot(time_epoch[:len(mean_condition.mean(axis=1))], mean_condition.mean(axis=1)[:len(time_epoch)], color=color[color_n], zorder=100)
#         yerr = for_std.sem(axis='columns').to_numpy()
#         ub = mean_condition.mean(axis=1) + yerr[:len(mean_condition.mean(axis=1))]
#         lb = mean_condition.mean(axis=1) - yerr[:len(mean_condition.mean(axis=1))]
#         plt.fill_between(time_epoch[:len(mean_condition.mean(axis=1))], ub[:len(time_epoch)], lb[:len(time_epoch)],color=color[color_n], alpha=.4)
#         # plt.xlim(0,5)
#     plt.ylabel('ΔF/F', fontsize= 14)
#     plt.xlabel(f'time [s]', fontsize= 14)
#     plt.title(f'{cat}', fontsize=18)
#     plt.tight_layout()
#     plt.savefig(f'{paths.results}/odor_delta/_whole_vs_{cat}_FFF_mean.png', dpi=400, bbox_inches='tight')
#     plt.close('all')

##############################################Summary################################################
##############################################Summary################################################
##############################################Summary################################################
##############################################Summary################################################


##############################################8D################################################
# for odor in ['ACV', 'BA']:
# # for odor in ['ACV']:
#     os.makedirs(f'{paths.results}/{odor}', exist_ok=True)
#     to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_8D_OL.pkl' ]
#     # to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_8D_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_8D_OL.pkl']
#     if preprocessing_params.experiment == 'big':
#         cats = ['ME', 'LO', "LOP"]
#     elif preprocessing_params.experiment == 'layer':
#         cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
#         # cats = ['LOo', 'LOP']
#     for cat in cats:
#         delta_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'delta'])
#         peak_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'peak'])
#         area_all = pd.DataFrame(columns=['odor_con', 'stim_con', 'area'])
#         color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
#         color_pal = {f'{odor}':neuropile_color.get(cat), 'No Odor':'#BCBDC4', 'without':'#5A5D59'}
#         mm=0
#         all_patches=[]
#         for col, what in zip(color, [f'{odor}', 'No Odor', 'without']):
#             patch = mpatches.Patch(color=col, label=what)
#             all_patches.append(patch)
#         fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True) #figsize=(10,20)
#         counter_row , counter_column=0,0
#         for m in [0,1,2,3,4,5,6,7]:
#             # ax = plt.subplot2grid(shape=(1,len([0,1,2,3])), loc=(0,m), fig=fig)
#             for color_n, fff in enumerate(to_compare):
#                 with open(f'{paths.data}/{fff}', 'rb') as fi:
#                         datafile = pickle.load(fi)
#                 epoch_len=[]
#                 for tseries in datafile:
#                     for roi in datafile[tseries]['final_rois']:
#                         epoch_len.append(list(set(roi.interpolated_traces_epochs.keys())))
#                 epochs = list(set.union(*map(set,epoch_len)))
#                 if isinstance(epochs[0], list):
#                     print('differnet epoch lenth acrross data!! ')
#                 mean_condition = pd.DataFrame({})
#                 for_std = pd.DataFrame({})
#                 for nn, tseries in enumerate(datafile):
#                     delta_mean, max_mean, area_mean = [], [], []
#                     mean_fly=pd.DataFrame({})
#                     for n, roi in enumerate(datafile[tseries]['final_rois']):
#                         if roi.category == cat:
#                             fps = roi.alligned_fps
#                             for epoch in roi.interpolated_traces_epochs:
#                                 if epoch == m:
#                                     trace_whole_epoch = roi.interpolated_traces_epochs.get(epoch)
#                                     start = np.argmax(trace_whole_epoch) - (fps*5)
#                                     if start < 0:
#                                         start  = 0
#                                     trace_before = trace_whole_epoch[start:np.argmax(trace_whole_epoch)+1]
#                                     delta = np.max(trace_whole_epoch) - np.min(trace_before)
#                                     # delta_mean.append(delta)
#                                     # max_mean.append(np.max(trace_whole_epoch))
#                                     # area_mean.append(np.trapz(trace_whole_epoch))
#                                     time_epoch = roi.interpolated_time.get(epoch)
#                                     mean_fly[n]=trace_whole_epoch
#                                     trace_for_std = pd.DataFrame({f'{n}': trace_whole_epoch})
#                                     for_std = pd.concat([trace_for_std, for_std], axis=1)
#                                     area= np.trapz(trace_whole_epoch)
#                                     if area>25:
#                                         area=25
#                                     try:
#                                         delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',delta]
#                                         peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',np.max(trace_whole_epoch)]
#                                         area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',area]
#                                         mm+=1
#                                     except:
#                                         continue
#                     # delta_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(delta_mean)]
#                     # peak_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(max_mean)]
#                     # area_all.loc[mm] = [[f'{odor}', 'No Odor', 'without'][color_n],f'{roi.stim_info.get("angle")[m]}',statistics.mean(area_mean)]
#                     # mm+=1
#                     mean_condition[nn] = mean_fly.mean(axis=1)
#                     # ax.plot(time_epoch, mean_fly.mean(axis=1), color='tab:gray', zorder=(0-n)*-1)
#                 axes[counter_row][counter_column].plot(time_epoch, mean_condition.mean(axis=1), color=color[color_n], zorder=100)
#                 yerr = for_std.sem(axis='columns').to_numpy()
#                 # yerr = for_std.std(axis='columns').to_numpy()
#                 ub = mean_condition.mean(axis=1) + yerr
#                 lb = mean_condition.mean(axis=1) - yerr
#                 axes[counter_row][counter_column].fill_between(time_epoch, ub, lb,color=color[color_n], alpha=.4)
#                 title = f'{roi.stim_info.get("angle")[m]}'
#                 axes[counter_row][counter_column].set_title(title)
#                 axes[counter_row][counter_column].set_xlim(0,min(roi.stim_info.get('duration')))
#             if counter_column < 3:
#                 counter_column+=1
#             elif counter_column == 3:
#                 counter_column =0
#                 counter_row+=1
#         fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
#         fig.supxlabel(f'time [s]', fontsize= 14)
#         fig.supylabel('ΔF/F', fontsize= 14)
#         fig.suptitle(f'{cat}', fontsize=18)
#         plt.tight_layout()
#         plt.tight_layout()
#         plt.savefig(f'{paths.results}/{odor}/_epoch_vs_{cat}_8D_mean_1.png', dpi=400, bbox_inches='tight')
#         plt.close('all')