'''script to analyse and visualize odor meassurments from PID'''
################################################
import mat73, os, scipy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from statistics import mean
################################################
peaks, deltas, peaks_norm, peaks_norm_mean, all_mean_signal = {}, {}, {}, {}, {}
for exp in os.listdir(f'C:/Master_Project/2P/oodor_meassurments'):
    if exp.endswith('.pdf') or exp.endswith('.png'):
        continue
    peakk_meassure, delta_meassure, peak_norm, peak_norm_mean, meaans_signal = {}, {}, {}, {}, {}
    for odor in ['ACV', 'BA']:
        try:
            mat = mat73.loadmat(f'C:/Master_Project/2P/oodor_meassurments/{exp}/{odor}.mat')
        except:
            mat = scipy.io.loadmat(f'C:/Master_Project/2P/oodor_meassurments/{exp}/{odor}.mat')
        a=1
        fig = plt.figure(figsize=(10,8), dpi=400)
        mean_peak, mean_delta, delta_peak_peak, delta_peak_peak_mean , signal_on_off, mean_signal= [], [], [], [], {}, []
        ax = plt.subplot2grid(shape=(3,1), loc=(2,0), fig=fig)
        if len(mat['data']['STIM']) == 1:
            overall = mat['data']['STIM'][0][0]
        else:
            overall = mat['data']['STIM']
        for n,data in enumerate(overall):
            data = data.round().astype(np.int32)
            if len(overall) < 4:
                stim_time, fps = 5, 100
            else:
                stim_time,fps=3,1000
            if data[0] == 0:
                stim_on_stamp = np.where(data[:-1] != data[1:])[0]
            else: 
                stim_on_stamp = np.where(data[:-1] != data[1:])[0]
                stim_on_stamp = np.hstack((0,stim_on_stamp))
            signal_on = stim_on_stamp[1::2]
            signal_off = signal_on + (stim_time*fps)
            signal_on_off[n] = [signal_on, signal_off]
        for n, data in enumerate(overall):
            if len(overall) < 4:
                color_time = ['tab:blue', 'tab:cyan', "tab:orange"]
                all_patches=[]
                reference_frame = 1000
                for info in zip(color_time, ['0min', '20min', "40min"]):
                    patch = mpatches.Patch(color=info[0], label=info[1])
                    all_patches.append(patch)
            else:
                color_time = ['tab:blue', 'tab:cyan', 'tab:red', "tab:orange", "tab:green", "tab:brown", 'tab:gray']
                all_patches=[]
                for info in zip(color_time, ['10min', '20min', '30min', "40min", "50min", "60min", "70min"]):
                    patch = mpatches.Patch(color=info[0], label=info[1])
                    all_patches.append(patch)
                reference_frame = 5000
            signal_timestamp_first = signal_on_off[n][0][0]
            diff_ref = signal_timestamp_first - reference_frame
            if diff_ref < 0:
                new_data = data[:-abs(diff_ref)]
                new_data = np.hstack((np.zeros(abs(diff_ref)),new_data))
            else:
                new_data = data[diff_ref:]
                new_data = np.hstack((new_data, np.zeros(diff_ref)))
            ax.plot( np.arange(len(new_data), ), new_data, color =color_time[n])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('STIM', fontsize=18)
        plt.ylim(0,5.1)
    
        ax = plt.subplot2grid(shape=(3,1),rowspan = 2, loc=(0,0), fig=fig)
        if len(mat['data']['PID']) == 1:
            overall = mat['data']['PID'][0][0]
        else:
            overall = mat['data']['PID']
        for n, data in enumerate(overall):
            if len(overall) < 4:
                color_time = ['tab:blue', 'tab:cyan', "tab:orange"]
                all_patches=[]
                stim_time, fps, reference_frame = 5, 100, 1000
                for info in zip(color_time, ['0min', '20min', "40min"]):
                    patch = mpatches.Patch(color=info[0], label=info[1])
                    all_patches.append(patch)
            else:
                color_time = ['tab:blue', 'tab:cyan', 'tab:red', "tab:orange", "tab:green", "tab:brown", 'tab:gray']
                all_patches=[]
                stim_time,fps=3,1000
                reference_frame = 5000
                for info in zip(color_time, ['10min', '20min', '30min', "40min", "50min", "60min", "70min"]):
                    patch = mpatches.Patch(color=info[0], label=info[1])
                    all_patches.append(patch)
            signal_timestamp_first = signal_on_off[n][0][0]
            diff_ref = signal_timestamp_first - reference_frame
            if (signal_on_off[n][0][0] - (stim_time*fps)) < 0:
                start =0
            else:
                start = (signal_on_off[n][0][0] - (stim_time*fps))
            data_noise = data[start:(signal_on_off[n][1][0]-(stim_time*fps))]
            if diff_ref < 0:
                new_data = data[:-abs(diff_ref)]
                if len(data_noise) < abs(diff_ref):
                    data_noise = np.interp(np.arange(0,abs(diff_ref),1), np.arange(0,len(data_noise),1),  data_noise)
                new_data = np.hstack((data_noise[:abs(diff_ref)],new_data))
            else:
                new_data = data[diff_ref:]
                new_data = np.hstack((new_data, data_noise[:diff_ref]))
            signal_timestamps = signal_on_off[n]
            mean_signal.append(mean([mean(data[signal_timestamps[0][0]: signal_timestamps[1][0]]), mean(data[signal_timestamps[0][1]: signal_timestamps[1][1]]), mean(data[signal_timestamps[0][2]: signal_timestamps[1][2]])]))
            if len(overall) < 4:
                mean_peak.append(mean([max(data[:3000]), max(data[3001:6000]), max(data[6001:9000])]))
                mean_delta.append(mean([max(data[:3000])-mean(data[-500:]), max(data[3001:6000])-mean(data[-500:]), max(data[6001:9000])-mean(data[-500:])]))
                delta_peak_peak.append([(max(data[:3000])-mean(data[-500:]))/max(data[:3000]), (max(data[3001:6000])-mean(data[-500:]))/max(data[3001:6000]), (max(data[6001:9000])-mean(data[-500:]))/max(data[6001:9000])])
                delta_peak_peak_mean.append(mean([(max(data[:3000])-mean(data[-500:]))/max(data[:3000]), (max(data[3001:6000])-mean(data[-500:]))/max(data[3001:6000]), (max(data[6001:9000])-mean(data[-500:]))/max(data[6001:9000])]))
            else:
                mean_peak.append(mean([max(data[:25000]), max(data[25001:58000]), max(data[55001:80000])]))
                mean_delta.append(mean([max(data[:25000])-mean(data[-7000:]), max(data[25001:58000])-mean(data[-7000:]), max(data[58001:80000])-mean(data[-7000:])]))
                delta_peak_peak.append([(max(data[:25000])-mean(data[-7000:]))/max(data[:25000]), (max(data[25001:58000])-mean(data[-7000:]))/max(data[25001:58000]), (max(data[58001:80000])-mean(data[-7000:]))/max(data[58001:80000])])
                delta_peak_peak_mean.append(mean([(max(data[:25000])-mean(data[-7000:]))/max(data[:25000]), (max(data[25001:58000])-mean(data[-7000:]))/max(data[25001:58000]), (max(data[58001:80000])-mean(data[-7000:]))/max(data[58001:80000])]))
            if odor == 'ACV' and exp=='3' and n==0:
                continue
            ax.plot( np.arange(len(new_data), ), new_data, color =color_time[n])
        peakk_meassure[odor] = mean_peak
        delta_meassure[odor] = mean_delta
        peak_norm[odor] = delta_peak_peak
        peak_norm_mean[odor] = delta_peak_peak_mean
        meaans_signal[odor] = mean_signal
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('PID', fontsize=18)
        
        fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
        fig.suptitle(f'{odor}', fontsize = 18)
        plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/{exp}/{odor}.png', dpi=400, bbox_inches='tight')
        plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/{exp}/{odor}.pdf', dpi=400, bbox_inches='tight')
        plt.close()
    peaks[exp] = peakk_meassure
    deltas[exp] = delta_meassure
    peaks_norm[exp] = peak_norm
    peaks_norm_mean[exp] = peak_norm_mean
    all_mean_signal[exp] = meaans_signal
    
fig = plt.figure(figsize=(14,10), dpi=400)
color=['tab:orange', 'tab:cyan']
all_patches=[]
for idx, odor in enumerate(peaks['1']):
    patch = mpatches.Patch(color=color[idx], label=odor)
    all_patches.append(patch)
column=0
for idx, exp in enumerate(peaks):
    z=1
    for row, odor in enumerate(peaks[exp]):
        labels = ["10", "20", "30", "40", "50", "60", "70"]
        if len(peaks[exp][odor]) > 3:
            labels = ["10", "20", "30", "40", "50", "60", "70"]
            x=[0,10,20,30,40,50,60]
        else:
            x=[0,10,20]
            labels = ["0", "20", "40"]
        ax = plt.subplot2grid(shape=(2,3), loc=(row,column), fig=fig)
        y=peaks[exp][odor]
        ax.plot(x,y,color=color[row], alpha=z)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(x, labels)
    column+=1
    z-=0.1
fig.supylabel('Mean Peak Response', fontsize=18)
fig.supxlabel('Time (min)', fontsize=18)
fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/peaks.png', dpi=400, bbox_inches='tight')
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/peaks.pdf', dpi=400, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(14,10), dpi=400)
all_patches=[]
for idx, odor in enumerate(peaks['1']):
    patch = mpatches.Patch(color=color[idx], label=odor)
    all_patches.append(patch)
column=0
for idx, exp in enumerate(peaks):
    z=1
    for row, odor in enumerate(peaks[exp]):
        labels = ["10", "20", "30", "40", "50", "60", "70"]
        if len(peaks[exp][odor]) > 3:
            labels = ["10", "20", "30", "40", "50", "60", "70"]
            x=[0,10,20,30,40,50,60]
        else:
            x=[0,10,20]
            labels = ["0", "20", "40"]
        ax = plt.subplot2grid(shape=(2,3), loc=(row,column), fig=fig)
        y=peaks[exp][odor]
        diffs=[]
        for idx, value in enumerate(y):
            if idx == 0:
                diff=0
                previous_value = value
            else:
                diff = previous_value-value
                previous_value = value
            diffs.append(diff)
        ax.plot(x,diffs,color=color[row], alpha=z)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(x, labels)
    column+=1
    z-=0.1
fig.supylabel('ΔPeak-Previous Peak', fontsize=18)
fig.supxlabel('Time (min)', fontsize=18)
fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/diff.png', dpi=400, bbox_inches='tight')
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/diff.pdf', dpi=400, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(14,10), dpi=400)
column=0
for idx,exp in enumerate(deltas):
    z=1
    for row, odor in enumerate(deltas[exp]):
        if len(deltas[exp][odor]) > 3:
            labels = ["10", "20", "30", "40", "50", "60", "70"]
            x=[0,10,20,30,40,50,60]
        else:
            x=[0,10,20]
            labels = ["0", "20", "40"]
        ax = plt.subplot2grid(shape=(2,3), loc=(row,column), fig=fig)
        y=deltas[exp][odor]
        ax.plot(x,y,color=color[row], alpha=z)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(x,labels)
    column+=1
    z-=0.1
fig.supylabel('Mean Δ(max - mean last 1/2 min)', fontsize=18)
fig.supxlabel('Time (min)', fontsize=18)
fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/deltas.png', dpi=400, bbox_inches='tight')
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/deltas.pdf', dpi=400, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(14,10), dpi=400)
column=0
for idx,exp in enumerate(peaks_norm_mean):
    z=1
    for row, odor in enumerate(peaks_norm_mean[exp]):
        if len(peaks_norm_mean[exp][odor]) > 3:
            labels = ["10", "20", "30", "40", "50", "60", "70"]
            x=[0,10,20,30,40,50,60]
        else:
            x=[0,10,20]
            labels = ["0", "20", "40"]
        ax = plt.subplot2grid(shape=(2,3), loc=(row,column), fig=fig)
        y=peaks_norm_mean[exp][odor]
        ax.plot(x,y,color=color[row], alpha=z)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(x,labels)
    column+=1
    z-=0.1
fig.supylabel('Mean ΔPeak/Peak', fontsize=18)
fig.supxlabel('Time (min)', fontsize=18)
fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/peak_norm_mean.png', dpi=400, bbox_inches='tight')
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/peak_norm_mean.pdf', dpi=400, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(14,10), dpi=400)
column=0
for idx,exp in enumerate(all_mean_signal):
    z=1
    for row, odor in enumerate(all_mean_signal[exp]):
        if len(all_mean_signal[exp][odor]) > 3:
            labels = ["10", "20", "30", "40", "50", "60", "70"]
            x=[0,10,20,30,40,50,60]
        else:
            x=[0,10,20]
            labels = ["0", "20", "40"]
        ax = plt.subplot2grid(shape=(2,3), loc=(row,column), fig=fig)
        y=all_mean_signal[exp][odor]
        y = [(value/y[0])*100 for value in y]
        ax.plot(x,y,color=color[row], alpha=z)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(x,labels)
        plt.ylim(80,112)
    column+=1
    z-=0.1
# fig.supylabel('Mean Signal', fontsize=18)
fig.supylabel('Mean Signal (%)', fontsize=18)
fig.supxlabel('Time (min)', fontsize=18)
fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/signal_mean.png', dpi=400, bbox_inches='tight')
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/signal_mean.pdf', dpi=400, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(14,10), dpi=400)
column=0
for idx, exp in enumerate(all_mean_signal):
    z=1
    for row, odor in enumerate(all_mean_signal[exp]):
        labels = ["10", "20", "30", "40", "50", "60", "70"]
        if len(all_mean_signal[exp][odor]) > 3:
            labels = ["10", "20", "30", "40", "50", "60", "70"]
            x=[0,10,20,30,40,50,60]
        else:
            x=[0,10,20]
            labels = ["0", "20", "40"]
        ax = plt.subplot2grid(shape=(2,3), loc=(row,column), fig=fig)
        y=all_mean_signal[exp][odor]
        diffs=[]
        for idx, value in enumerate(y):
            if idx == 0:
                diff=(value/y[0])*100
                previous_value = (value/y[0])*100
            else:
                value = (value/y[0])*100
                diff = previous_value-value
                previous_value = value
            diffs.append(diff)
        diffs[0] = 0
        # print(diffs)
        # diffs = [(value/diffs[0])*100 for value in diffs]
        ax.plot(x,diffs,color=color[row], alpha=z)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if odor == 'ACV':
            plt.ylim(-11 , 1)
        elif odor == 'BA':
            plt.ylim(0,8)
        plt.xticks(x, labels)
    column+=1
    z-=0.1
# fig.supylabel('ΔPeak-Previous Peak ', fontsize=18)
fig.supylabel(f'% of previous peak', fontsize=18)
fig.supxlabel('Time (min)', fontsize=18)
fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/signal_mean_diff.png', dpi=400, bbox_inches='tight')
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/signal_mean_diff.pdf', dpi=400, bbox_inches='tight')
plt.close()


delta_norm_all_df=pd.DataFrame({})
for exp in peaks_norm:
    for  odor in peaks_norm[exp]:
        data = [j for sub in peaks_norm[exp][odor] for j in sub]
        delt_norm = pd.DataFrame({'Odor': [odor]*len(data), 'peaks': data})
        delta_norm_all_df=pd.concat([delta_norm_all_df, delt_norm], ignore_index=True)
fig = plt.figure(figsize=(10,10), dpi=400)
ax = plt.subplot2grid(shape=(1,1), loc=(0,0), fig=fig)
sns.boxplot(x='Odor', y='peaks',
                data=delta_norm_all_df, palette=color, ax=ax, width=0.5,
                showmeans=True, meanprops={'marker':'x','markeredgecolor':'black','markersize':'4'})
sns.stripplot(x = 'Odor',y = 'peaks',data = delta_norm_all_df,ax = ax,palette=color,alpha = 0.5, size=4) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('ΔPeak/Peak', fontsize=18)
ax.set(xlabel=None)
plt.xticks(visible=False)
plt.yticks(rotation=0, fontsize=16) 
leg = plt.legend()
ax.legend_ = None
fig.supxlabel('Odorants', fontsize=18)
fig.legend(handles=all_patches, loc="lower center", ncol=len(all_patches), bbox_to_anchor=(0.5,-0.05), frameon=False)
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/peak_norm.png', dpi=400, bbox_inches='tight')
plt.savefig(f'C:/Master_Project/2P/oodor_meassurments/peak_norm.pdf', dpi=400, bbox_inches='tight')
plt.close()

