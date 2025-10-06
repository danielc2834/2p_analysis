#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:33:56 2020

@author: Jfelipeco includes elements by Burak Gür
"""


#from itertools import pairwise
import itertools
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import sys
from scipy import stats
code_path= r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis' #Juan desktop
sys.path.insert(0, code_path) 
from Helpers import post_analysis_core as pac
import ROI_mod
import seaborn as sns
from matplotlib.gridspec import GridSpec
# %% Setting the directories
initialDirectory = 'C:\\Users\\vargasju\\PhD\\experiments\\2p\\'
#all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
#results_save_dir = os.path.join(initialDirectory,
#                                'results/200310_GluClalpha_NI_cut_experiments')


# %% Load datasets and desired variables
exp_folder ='Tm3Nosplitgal4_gluclRescue'# name of folder containing fly raw data folders
exp_t ='Tm3Nosplitgal4_gluclRescue' #
#genotype_labels =['prev','2BG3_F37','2BG3_F37_od1.2','2BG3_F37_od0.9'] #['experimental_18deg']#['experimental','negative','positive']
data_dir = initialDirectory+exp_folder+'\\'+'processed\\FFF_data\\files'
results_save_dir = initialDirectory+exp_folder+'\\'+'processed\\FFF_data\\results'
type_of_neuron='ON' #options: 'ON' 'OFF' 'mixed'. use None when there are inverted neurons
specific_polarity_to_plot='ON'
category_to_plot='M10'
use_corr_filter=False
compute_stats=True
#if compute stats then
comparisons_to_make=[['experimental','positive'],['experimental','negative']]#,['positive','negative']] #choose which pairwise comparisons to do
alternative_h=[None,'greater']#,'less'] #Less, greater (1 tail). or None (2 tails. when None, there is no obvious prediction regarding the magnitude between experimental 
                               #and positive control conditions)

#data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
# Initialize variables
final_rois_all = np.array([])
flyIDs = []
flash_resps = []
steps = []
plateau = []
integral = []
flash_corr = []
genotypes = []
#%%
for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    
    roi_categories=ROI_mod.data_to_list(curr_rois, ['category'])
    for idx,cat in enumerate(roi_categories['category']):
        #clean the ROIs. to get only the string
        cleaned=list(cat[0])[0]
        roi_categories['category'][idx]=cleaned
    #TODO filter ROIs based on category
    roi_categories=np.array(roi_categories['category'])
    category_filter=roi_categories==category_to_plot
    curr_rois=np.array(curr_rois)[category_filter]
    final_rois_all=np.concatenate((final_rois_all,curr_rois),axis=0)
    #stim_t = curr_rois[0].int_stim_trace
    # fff_stim_trace = \
    #     np.around(np.concatenate((stim_t[20:],stim_t[0:30]),axis=0))
    
    # if type_of_neuron==None:
    #     print('not implemented, in process')
    #     # if the recordings include inverted neurons- 
    #     # then cluster the ROIs based on their peak responses
    #     #data_DF, fff_stim_trace = pac.FFF_roi_data_extraction_and_clustering(final_rois_all)
    
    #     #use the correlation with the stimulus to find if neuron is ON or OFF

    # elif type_of_neuron=='ON' or type_of_neuron=='OFF':
data_DF,trace_df,fff_stim_trace=\
    pac.FFF_data_extraction_1_polarity(type_of_neuron,final_rois_all,specific_polarity_to_plot)

#%%


# TODO plot individual traces and extract quantiications from average traces
    
# make summary statistics

mean_samples_traces=trace_df.groupby(['flyIDs','treatment']).mean()
mean_treatments_traces=mean_samples_traces.groupby(['treatment']).mean()
sem_treatments_traces=mean_samples_traces.groupby(['treatment']).sem()


fig = plt.figure(figsize=(15, 3))
fig.suptitle('individual_traces %s' %category_to_plot,fontsize=12)
grid = plt.GridSpec(1,10 ,wspace=1, hspace=1)
x=0
steps=[]
plateaus=[]
integrals=[]
genotypes=[]

genotype_labels = np.unique(data_DF['treatment'])

for idx, genotype in enumerate(np.unique(genotype_labels)):
    color_a=['magenta','dark_gray','green1']
    temp=mean_samples_traces.reset_index(drop=False,inplace=False)
    ax=plt.subplot(grid[0,x:x+3])
    ax.set_title(genotype)
    temp= temp.loc[temp['treatment']==genotype]
    temp=temp.set_index('treatment')
    temp=temp.set_index('flyIDs')
    temp=np.array(temp)
    
    if idx==0:
        steps=(temp[:,25:35][np.array(range(temp.shape[0])),np.argmax(np.abs(temp)[:,25:35],axis=1)])-np.mean(temp[:,15:25],axis=1)
        plateaus=np.mean(temp[:,65:75],axis=1)
        integrals=np.sum(temp[:,25:75],axis=1)
        genotype=(genotype+'_')*temp.shape[0]
        genotypes=np.array(genotype.split('_'))[:-1]
    else:
        steps=np.concatenate((steps,(temp[:,25:35][np.array(range(temp.shape[0])),np.argmax(np.abs(temp)[:,25:35],axis=1)])-np.mean(temp[:,15:25],axis=1)))
        plateaus=np.concatenate((plateaus,np.mean(temp[:,65:75],axis=1)))
        integrals=np.concatenate((integrals,np.sum(temp[:,25:75],axis=1)))
        genotype=(genotype+'_')*temp.shape[0]
        genotypes=np.concatenate((genotypes,np.array(genotype.split('_'))[:-1]))
    x=x+3
    plt.plot(np.arange(0,10,0.1),np.transpose(temp))

    #TODO calculate quantifications per fly. 


per_fly_data_dict={'step_response':steps,'plateau_response':plateaus,'treatment':genotypes,'integral':integrals}
mean_samples_df=pd.DataFrame.from_dict(per_fly_data_dict)
mean_samples_df=mean_samples_df.set_index('treatment')
sem_treatments_df=mean_samples_df.groupby(level='treatment').sem()

save_name = 'all_traces_ind_flies layer_%s %s' %(category_to_plot,specific_polarity_to_plot)
os.chdir(results_save_dir)
plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

###Statistics:
if compute_stats:
    pairw_comparisons=pac.mean_comparison_between_2groups(mean_samples_df,comparisons_to_make,alternative_h,category_L=category_to_plot,polarity=specific_polarity_to_plot,save_dir=results_save_dir)

#%%
#plot quantifications
fig1=plt.figure(figsize=(40,12))
plot_df=pd.DataFrame.from_dict(per_fly_data_dict)
gs = GridSpec(1, len(plot_df.loc[:,plot_df.columns!='treatment'].keys()))
plt.style.use('seaborn')
for idx,variable in enumerate(plot_df.loc[:,plot_df.columns!='treatment'].keys()):
    ax=fig1.add_subplot(gs[idx])

    sns.barplot(x="treatment", y=variable, data=plot_df, ax=ax,palette='muted')
    sns.stripplot(x="treatment", y=variable, data=plot_df, palette='muted', ax=ax, zorder=2)      
    ax.set_ylabel(variable)
    save_str=results_save_dir+'\\quantity comparison layer_%s %s.pdf' %(category_to_plot,specific_polarity_to_plot)
plt.savefig(save_str)



_, colors_d = pac.run_matplotlib_params()
colors = [colors_d['magenta'],colors_d['green1'],colors_d['green3'],
              colors_d['dark_gray'],colors_d['brown']]
colors
plt.close('all')
plt.style.use('seaborn')
fig = plt.figure(figsize=(15, 3))
fig.suptitle('5sFFF properties_%s'%category_to_plot,fontsize=12)

grid = plt.GridSpec(1,7 ,wspace=1, hspace=1)

# FFF responses
ax=plt.subplot(grid[0,:3])
ax2=plt.subplot(grid[0,3])
# ax3=plt.subplot(grid[0,4])
# ax4=plt.subplot(grid[0,5])
# ax5=plt.subplot(grid[0,6])

text_str=[r'stats']
for rowi,row in enumerate(pairw_comparisons.index):
    str_row=[]
    temp_row=pairw_comparisons.iloc[rowi]
    for key in temp_row.keys():        
        text_piece= r'%s:%s' %(key,temp_row[key])
        str_row.append(text_piece)
    text_str.append('_'.join(str_row))
text_str='\n'.join(text_str)
ax2.axis('off')


 # this is just to indicate that the real analisis is done on a version
                                                # of the traces that is shfited 0.45 seconds to the left.
Textprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.00, 1, text_str, transform=ax2.transAxes, fontsize=4,
verticalalignment='top', bbox=Textprops,rotation=90)


labels = ['']
fff_all_max = 0.0
t_trace = np.linspace(0,len(fff_stim_trace),len(fff_stim_trace))/10

#reset index is needed for following steps
mean_samples_traces=mean_samples_traces.reset_index()
mean_treatments_traces=mean_treatments_traces.reset_index()
sem_treatments_df=sem_treatments_df.reset_index()
mean_samples_df=mean_samples_df.reset_index()
sem_treatments_traces=sem_treatments_traces.reset_index()


for idx, genotype in enumerate(np.unique(genotype_labels)): #loop over treatments
    
    
    curr_data_DF=data_DF.loc[data_DF['treatment']==genotype]
    #curr_data = fff_dict['experiment_ids'][genotype]    
    curr_trace= mean_treatments_traces.loc[mean_treatments_traces['treatment']==genotype]
    curr_trace=curr_trace.set_index('treatment')
    curr_trace=np.array(curr_trace)[0,:]            
    curr_trace_sem=sem_treatments_traces.loc[sem_treatments_traces['treatment']==genotype]
    curr_trace_sem=curr_trace_sem.set_index('treatment')
    curr_trace_sem=np.array(curr_trace_sem)[0,:]

#     #extract quantifications  
#     curr_data_sem=sem_treatments_df.loc[sem_treatments_df['treatment']==genotype]
#    #curr_data_sem=curr_data_sem.set_index('treatment')
    
    curr_data_over_samples=mean_samples_df.loc[mean_samples_df['treatment']==genotype]
    # #curr_data_over_samples=curr_data_over_samples.set_index('treatment')


    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype,
                                               nflies =\
                                                curr_data_over_samples.shape[0],
                                               nROIs=\
                                                len(curr_data_DF))

    
    mean = curr_trace
    if np.max(mean) > fff_all_max:
        fff_all_max = np.max(mean)
    error = curr_trace_sem
    ub = mean + error
    lb = mean - error
    ax.plot(t_trace,mean,color=colors[idx],alpha=.8,lw=3,label=gen_str)
    ax.fill_between(t_trace, ub, lb,
                     color=colors[idx], alpha=.4)
    scaler = np.abs(np.max(mean) - np.min(mean))

    plot_stim = (fff_stim_trace) 
    ax.plot(t_trace,
            plot_stim/6+ fff_all_max,'--k',lw=1.5,alpha=.8)

    ax.set_title('Response')  
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$\Delta F/F$')
    ax.legend()
    
    ax.set_yticks(range(-1,6))
    #ax.set_yticks([-0.5,0,0.5,1.0,1.5])
    ax.set_ylim([-1,6])
    labels.append(genotype)    
    # curr_data = np.squeeze(np.array(curr_data_over_samples['step_response']))
    # error = np.array(curr_data_sem['step_response'])
    # pac.bar_bg(curr_data, idx+1, color=colors[idx], 
    #            ax=ax2,yerr=error)
    
    # curr_data = np.squeeze(np.array(curr_data_over_samples['plateau_response']))
    # error = np.array(curr_data_sem['plateau_response'])
    # pac.bar_bg(curr_data, idx+1, color=colors[idx], 
    #            ax=ax3,yerr=error)
    
    # curr_data = np.squeeze(np.array(curr_data_over_samples['integral']))
    # error = np.array(curr_data_sem['integral'])
    # pac.bar_bg(curr_data, idx+1, color=colors[idx], 
    #            ax=ax4,yerr=error)
    
    
# ax2.set_title('ON step')  
# ax2.set_ylabel('$\Delta F/F$')
# ax2.set_xticks(range(idx+2))
# ax2.set_xlim((0,idx+2))
# ax2.set_ylim((-1.7,6.3))
# ax2.set_xticklabels(labels,rotation=45)
# ax2.plot(list(ax2.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

# ax3.set_title('ON plateau')  
# ax3.set_ylabel('$\Delta F/F$')
# ax3.set_xticks(range(idx+2))
# ax3.set_xlim((0,idx+2))
# ax3.set_ylim((-1,8))
# ax3.set_xticklabels(labels,rotation=45)
# ax3.plot(list(ax2.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

# ax4.set_title('ON integral')  
# ax4.set_ylabel('$\Delta F/F$')
# ax4.set_xticks(range(idx+2))
# ax4.set_xlim((0,idx+2))
# ax4.set_ylim((-20,350))
# ax4.set_xticklabels(labels,rotation=45)
# ax4.plot(list(ax2.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

fig.tight_layout()

if 1==1:
    # Saving figure
    save_name = '{exp_t} FFF layer_{layer} polarity{pol}'.format(exp_t=exp_t,layer=category_to_plot,pol=specific_polarity_to_plot)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# plt.close('all')



# %%
