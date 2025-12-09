'''script to plot raw traces for each fly and as mean per condition'''
################################################
import core_preprocessing as core_pre
import pickle, os, preprocessing_params
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
################################################
# dataset_folder = 'C:/Master_Project/2P/datasets/241203_whole_OL'
# dataset_folder =r'F:\241203_whole_OL'
dataset_folder = r'C:\phd\02_twophoton\251023_tdc2_cschr_pan'
# error_log = 'C:/2p/'
paths = core_pre.dataset(dataset_folder)
if preprocessing_params.experiment == 'big':
    neuropile_color = {"ME": "#96ceb4", "LO": "#ff6f69", "LOP": "#ffcc5c"}
elif preprocessing_params.experiment == 'layer':
    neuropile_color = {"MEi": "#96ceb4","MEo": "#96ceb4", "LOi": "#ffcc5c", "LOo": "#ffcc5c", "LOP": "#ff6f69"}
################################################

# if len(preprocessing_params.experiment)>0:
#     target = f'{paths.data}/{preprocessing_params.experiment}'
# else:
#     target = f'{paths.data}/'
    
    
for condition in os.listdir(f'{dataset_folder}/3_DATA'):
    if condition == 'LH_all.pkl' or condition.endswith("_LH.pkl") or condition == 'OL_all.pkl':
        continue
    elif condition.endswith('.pkl'):
        with open(f'{dataset_folder}/3_DATA/{condition}', 'rb') as fi:
            datafile = pickle.load(fi)
        # flilist = list(datafile.keys())
        # while 'alligned_fps' in flilist:
        #     flilist.remove('alligned_fps')
        for fly in datafile:#
            for tseries in datafile:
                for roi in datafile[tseries]['final_rois']:
                    fps = roi.alligned_fps
            os.makedirs(f'{paths.results}/{condition.split(".")[0]}/{fly}', exist_ok=True)
            olf_stim_array = core_pre.olf_stim_array('on', fps, f'{paths.results}/{condition.split(".")[0]}/')
        cats=[]
        for tseries in datafile:
            for roi in datafile[tseries]['final_rois']:
                if roi.category == ['No_category']:
                    continue
                cats.append(roi.category)
        cats = list(set(cats))
        for fly in datafile:
            for cat in cats:
                if os.path.exists(f'{dataset_folder}/4_results/{condition.split(".")[0]}/{fly}/_raw_{cat}.png') == False:
                    # if fly.endswith('.png'):
                    #     continue
                    mean_fly=pd.DataFrame({})
                    fig = plt.figure(figsize=(25,22), dpi=400)
                    for n, roi in enumerate(datafile[fly]['final_rois']):
                        if roi.category == cat:
                            fps = roi.alligned_fps
                            time = roi.interpolated_time_all
                            data = roi.interpolated_traces_all
                            plt.plot(time, data, color='tab:gray', zorder=(0-n)*-1)
                            mean_fly[roi]=data
                    try:
                        plt.plot(time, mean_fly.mean(axis=1), color=neuropile_color.get(cat), zorder=100)
                        plt.xlim(0,max(time))
                        plt.ylabel('ΔF/F', fontsize= 14)
                        plt.xlabel(f'time [s]', fontsize= 14)
                        plt.title(f'{fly} : {cat}', fontsize=18)
                        plt.tight_layout()
                        plt.savefig(f'{dataset_folder}/4_results/{condition.split(".")[0]}/{fly}/_raw_{cat}.png', dpi=400, bbox_inches='tight')
                        plt.close('all')
                    except:
                        continue


for condition in os.listdir(f'{dataset_folder}/3_DATA'):
    fig = plt.figure(figsize=(22,22), dpi=400)
    if condition == 'LH_all.pkl' or condition.endswith("_LH.pkl") or condition == 'OL_all.pkl':
        continue
    elif condition.endswith('.pkl'):
        with open(f'{dataset_folder}/3_DATA/{condition}', 'rb') as fi:
            datafile = pickle.load(fi)
    if preprocessing_params.experiment == 'layer':
        cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
    elif preprocessing_params.experiment == 'big':
        cats = ['ME', 'LO', "LOP"]
    for n, cat in enumerate(cats):##, 
        if os.path.exists(f'{dataset_folder}/4_results/{condition.split(".")[0]}/_raw_{cat}_mean.png') == False:
            fig = plt.figure(figsize=(25,22), dpi=400)
            mean_con = pd.DataFrame({})
            for fly in datafile:
                # if fly == 'alligned_fps':
                #     continue
                mean_fly=pd.DataFrame({})
                for roi in datafile[fly]['final_rois']:
                        if roi.category == cat:
                            data = roi.interpolated_traces_all
                            time = roi.interpolated_time_all
                            mean_fly[roi]=data
                mean_con[fly] = mean_fly.mean(axis=1)
                try:
                    plt.plot(time, mean_fly.mean(axis=1), color='tab:grey', zorder=(0-n)*-1)
                except:
                    continue
            plt.plot(time[:len(mean_con.mean(axis=1))], mean_con.mean(axis=1)[:len(time)], color=neuropile_color.get(cat), zorder=100)
            plt.xlim(0,max(time))
            plt.title(f'{cat}')#, fontsize=18)
            plt.ylabel('ΔF/F')#, fontsize= 14)
            plt.xlabel(f'time [s]')#, fontsize= 14)
            plt.tight_layout()
            os.makedirs(f'{dataset_folder}/4_results/{condition.split(".")[0]}', exist_ok=True)
            plt.savefig(f'{dataset_folder}/4_results/{condition.split(".")[0]}/_raw_{cat}_mean.png', dpi=400, bbox_inches='tight')
            plt.close('all')
        
for odor in ['ACV', 'BA']:
    stims = [x.split('_')[-2] for x in os.listdir(paths.data) if x.startswith(f'{preprocessing_params.experiment}_')]
    stims = list(set(stims))
    stims.remove('nan')
    for stim in stims:
        to_compare = [f'{preprocessing_params.experiment}_{odor}_{odor}_{stim}_OL.pkl', f'{preprocessing_params.experiment}_{odor}_WO_{stim}_OL.pkl', f'{preprocessing_params.experiment}_nan_WO_{stim}_OL.pkl' ]
        # to_compare = [f'layer_{odor}_{odor}_{stim}_OL.pkl', f'layer_{odor}_WO_{stim}_OL.pkl', f'layer_nan_WO_{stim}_OL.pkl' ]
        # for cat in ['ME', 'LO', "LOP"]:
        if preprocessing_params.experiment == 'big':
            cats = ['ME', 'LO', 'LOP']
        elif preprocessing_params.experiment == 'layer':
            cats = ['MEi', 'MEo', 'LOi', 'LOo', 'LOP']
        for cat in cats:
            if os.path.exists(f'{dataset_folder}/4_results/{stim}/_raw_{odor}_{cat}.png') == False:
                color = [neuropile_color.get(cat), '#BCBDC4' , '#5A5D59']
                all_patches=[]
                for col, what in zip(color, ['Odor', 'No Odor', 'without']):
                    patch = mpatches.Patch(color=col, label=what)
                    all_patches.append(patch)
                fig = plt.figure(figsize=(5,10), dpi=400)
                for n, condition in enumerate(to_compare):
                    mean_con = pd.DataFrame({})
                    with open(f'{dataset_folder}/3_DATA/{condition}', 'rb') as fi:
                        datafile = pickle.load(fi)
                    for fly in datafile:
                        # if fly == 'alligned_fps':
                        #     continue
                        mean_fly=pd.DataFrame({})
                        for roi in datafile[fly]['final_rois']:
                            if roi.category == cat:
                                data = roi.interpolated_traces_all
                                time = roi.interpolated_time_all
                                mean_fly[roi]=data
                        mean_con[fly] = mean_fly.mean(axis=1)
                    try:
                        plt.plot(time[:len(mean_con.mean(axis=1))], mean_con.mean(axis=1)[:len(time)], color=color[n], zorder=5-n)
                    except:
                        continue
                plt.xlim(0,max(time))
                plt.title(f'{cat}')#, fontsize=18)
                plt.ylabel('ΔF/F')#, fontsize= 14)
                plt.xlabel(f'time [s]')#, fontsize= 14)
                fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
                fig.tight_layout()
                os.makedirs(f'{dataset_folder}/4_results/{stim}', exist_ok=True)
                fig.savefig(f'{dataset_folder}/4_results/{stim}/_raw_{odor}_{cat}.png', dpi=400, bbox_inches='tight')
                plt.close('all')