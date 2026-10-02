import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import sima
import copy
import scipy
from warnings import warn
from scipy import signal
import glob, re, io
from roipoly import RoiPoly
from scipy.optimize import curve_fit
from scipy.stats import linregress
from skimage import filters
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import fft
from scipy.signal import blackman
# import post_analysis_core as pac
# import process_mov_core as pmc
from itertools import permutations
def generate_RF_map_stripes(rois, screen_w = 65): #DEBUG: Gaussian fit for ROIs is done for stripes
    screen_coords = np.linspace(0, screen_w, num=screen_w, endpoint=True) # degree of the screen
    for roi in rois:
        findMaxResponse_all_epochs(roi)
        roi_t_v_stripe = np.linspace(0, screen_w, num=len(roi.max_resp_all_epochs), endpoint=True)
        
        roi.i_stripe_resp = np.interp(screen_coords, roi_t_v_stripe, 
                                  np.transpose(list(roi.max_resp_all_epochs.values())))#DEBUG
        try:
            fit_trace, r_squared, coeff = fit_1d_gauss(screen_coords, roi.i_stripe_resp)
        except RuntimeError:
            print('Fit parameters not found... ROI fitting stopped {s}'.format(s=roi))
            roi.discard = True
            roi.stripe_gauss_profile = None
            roi.stripe_gauss_coeff = None
            roi.stripe_r_squared = None
            roi.stripe_gauss_fwhm = None
            continue
        roi.stripe_gauss_profile = fit_trace
        roi.stripe_gauss_coeff = coeff
        roi.stripe_gauss_fwhm = 2.355 * coeff[2]
        roi.stripe_r_squared = r_squared
        
    return rois
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def fit_1d_gauss(data_x, data_y):
    #p0 = [np.max(data_y), np.argmax(data_y), 1] #Original code
    p0 = [np.max(data_y), data_x[np.argmax(data_y)], 1]
    coeff, pcov = curve_fit(gauss, data_x, data_y, p0=p0)
    fit_trace = gauss(data_x, *coeff)
    
    residuals = data_y- fit_trace
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_y-np.mean(data_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return fit_trace, r_squared, coeff

def fit_1d_gauss_neg(data_x, data_y):
    #p0 = [np.max(data_y), np.argmax(data_y), 1] #Original code
    p0 = [np.min(data_y), data_x[np.argmin(data_y)], 1]
    coeff, pcov = curve_fit(gauss, data_x, data_y, p0=p0)
    fit_trace = gauss(data_x, *coeff)
    
    residuals = data_y- fit_trace
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_y-np.mean(data_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return fit_trace, r_squared, coeff

def generate_RF_map_stripes_neg(rois, screen_w = 65): #DEBUG: Gaussian fit for ROIs is done for stripes
    screen_coords = np.linspace(0, screen_w, num=screen_w, endpoint=True) # degree of the screen
    for roi in rois:
        findMinResponse_all_epochs(roi)
        roi_t_v_stripe = np.linspace(0, screen_w, num=len(roi.min_resp_all_epochs), endpoint=True)
        
        roi.i_stripe_resp = np.interp(screen_coords, roi_t_v_stripe, 
                                  np.transpose(list(roi.min_resp_all_epochs.values())))#DEBUG
        try:
            fit_trace, r_squared, coeff = fit_1d_gauss_neg(screen_coords, roi.i_stripe_resp)
        except RuntimeError:
            print('Fit parameters not found... ROI fitting stopped {s}'.format(s=roi))
            roi.discard = True
            roi.stripe_gauss_profile = None
            roi.stripe_gauss_coeff = None
            roi.stripe_r_squared = None
            roi.stripe_gauss_fwhm = None
            continue
        roi.stripe_gauss_profile = fit_trace
        roi.stripe_gauss_coeff = coeff
        roi.stripe_gauss_fwhm = 2.355 * coeff[2]
        roi.stripe_r_squared = r_squared
        
    return rois

def findMaxResponse_all_epochs(roi):
        try:
            roi.resp_traces
        except AttributeError:
            raise AttributeError('ROI_bg: for finding maximum responses \
                            "resp_trace_all_epochs" has to be appended by \
                            appendTrace() method ')
            
        
        roi.max_resp_all_epochs = {}

        median_baseline_epoch=[]
        if 'movingStripes' in roi.stim_info['meta']['stim_name']:
            median_baseline_epoch = np.median(list(roi.whole_traces.values()))
        else:
            for idx, epoch_idx in enumerate(roi.resp_traces):
                coords = np.array(roi.stim_info['processed']['trial_coordinates'][epoch_idx][0])-[roi.stim_info['processed']['trial_coordinates'][epoch_idx][0][0]]
                median_baseline_epoch.append(roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]])
            median_baseline_epoch = np.median(median_baseline_epoch)


        for idx, epoch_idx in enumerate(roi.resp_traces):
            coords = np.array(roi.stim_info['processed']['trial_coordinates'][epoch_idx][0])-[roi.stim_info['processed']['trial_coordinates'][epoch_idx][0][0]]
            #mean F and F0 computed pre-epoch
            #roi.max_resp_all_epochs[epoch_idx] = np.nanmean(roi.resp_traces[epoch_idx]-np.mean(roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]]))
            
            #mean F and F0 computed as median F of baseline epochs
            roi.max_resp_all_epochs[epoch_idx] = np.nanmean(roi.resp_traces[epoch_idx]-median_baseline_epoch)

            ## compute response amplitude after stimulus offset (2s period after offset)
            #roi.max_resp_all_epochs[epoch_idx] = np.nanmean(roi.whole_traces[epoch_idx][40:]-median_baseline_epoch)
            
            #max F (original Code from Burak)
            #roi.max_resp_all_epochs[epoch_idx] = np.nanmax(roi.resp_traces[epoch_idx])
            #roi.max_resp_all_epochs[epoch_idx] = np.nanmean(roi.resp_traces[epoch_idx])
            """plt.plot(range(0,len(roi.whole_traces[epoch_idx])), roi.whole_traces[epoch_idx])
            plt.plot(range(0,len(roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]])),roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]])
            plt.show()"""
        roi.max_response = np.nanmax(roi.max_resp_all_epochs)
        roi.max_resp_idx = np.nanargmax(roi.max_resp_all_epochs)

def findMinResponse_all_epochs(roi):
        try:
            roi.resp_traces
        except AttributeError:
            raise AttributeError('ROI_bg: for finding maximum responses \
                            "resp_trace_all_epochs" has to be appended by \
                            appendTrace() method ')
            
        
        roi.min_resp_all_epochs = {}

        median_baseline_epoch=[]
        for idx, epoch_idx in enumerate(roi.resp_traces):
            coords = np.array(roi.stim_info['processed']['trial_coordinates'][epoch_idx][0])-[roi.stim_info['processed']['trial_coordinates'][epoch_idx][0][0]]
            median_baseline_epoch.append(roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]])
        median_baseline_epoch = np.median(median_baseline_epoch)

        for idx, epoch_idx in enumerate(roi.resp_traces):
            coords = np.array(roi.stim_info['processed']['trial_coordinates'][epoch_idx][0])-[roi.stim_info['processed']['trial_coordinates'][epoch_idx][0][0]]
            #mean F and F0 computed pre-epoch
            #roi.min_resp_all_epochs[epoch_idx] = np.nanmean(roi.resp_traces[epoch_idx]-np.mean(roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]]))#DEBUG: subtract 0.5s F before stim onset
            #mean F and F0 computed as median F of baseline epochs
            roi.min_resp_all_epochs[epoch_idx] = np.nanmean(roi.resp_traces[epoch_idx]-median_baseline_epoch)

            #max F (original Code from Burak)
            #roi.min_resp_all_epochs[epoch_idx] = np.nanmin(roi.resp_traces[epoch_idx])
            #roi.min_resp_all_epochs[epoch_idx] = np.nanmean(roi.resp_traces[epoch_idx])
            """plt.plot(range(0,len(roi.whole_traces[epoch_idx])), roi.whole_traces[epoch_idx])
            plt.plot(range(0,len(roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]])),roi.whole_traces[epoch_idx][(coords[1]-5):coords[1]])
            plt.show()"""
        roi.min_response = np.nanmin(roi.min_resp_all_epochs)
        roi.min_resp_idx = np.nanargmin(roi.min_resp_all_epochs)
        
def calculateReliability(rois):
    """ Calculates the correlation between single trials and assigns as reliability for each ROI """
    total_epoch_numbers = len(rois[0].resp_traces)
    SNR_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))
    Corr_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))

    for roi in rois:
        # Compute the lengths
        stim_info = roi.stim_info
        randomization_condition = stim_info['meta']['randomization_condition']
        stim_coords = stim_info['processed']['trial_coordinates']

        roi.reliabilities = []
        roi.SNR = []
        # Initialize the randomization related variables
        if (randomization_condition == 1) or (randomization_condition == 3):
            baseline_epoch = 'epoch_1'
            resp_start_idx = 1
            resp_end_idx = 2
            base_start_idx = 0
            base_end_idx = 3

            base_dur = stim_info['meta']['epoch_infos'][baseline_epoch]['total_dur_sec']
            base_len = int(np.floor(base_dur*roi.imaging_info['frame_rate']))
        else:
            baseline_epoch = None
            resp_start_idx = 0
            resp_end_idx = 1
            base_start_idx = 0
            base_end_idx = 1
            base_len = 0
        
        for epoch , epoch_coords in stim_coords.items():
            if epoch == baseline_epoch: # Skip the baseline epoch
                continue
            epoch_dur = stim_info['meta']['epoch_infos'][epoch]['total_dur_sec']

            resp_len = int(np.floor(epoch_dur*rois[0].imaging_info['frame_rate']))
            trial_len = base_len + resp_len + base_len
            
            
            trial_num = len(epoch_coords)
            resp_mat = np.zeros([resp_len,trial_num])
            noise_std_mat = np.zeros([1,trial_num])
            resp_std_mat = np.zeros([1,trial_num])
            for iTrial in range(trial_num):
                trial_coords = epoch_coords[iTrial]
                resp_start = trial_coords[resp_start_idx]
                resp_mat[:,iTrial] = roi.df_trace[resp_start:resp_start+resp_len]
                noise_std_mat[:,iTrial] = roi.df_trace[resp_start-5:resp_start].std(axis=0) #use 0.5s before epoch to calculate noise
                resp_std_mat[:,iTrial] = roi.df_trace[resp_start:resp_start+resp_len].std(axis=0)
            perm = permutations(range(trial_num), 2) 
            coeff =[]
            for iPerm, pair in enumerate(perm):
                curr_coeff, pval = pearsonr(resp_mat[:,pair[0]],
                                            resp_mat[:,pair[1]])
                coeff.append(curr_coeff)
                
            roi.reliabilities.append(np.array(coeff).mean())
            noise_std = noise_std_mat.mean(axis=0) 
            resp_std = resp_std_mat.mean(axis=0)
            signal_std = resp_std - noise_std
            # SNR calculation taken from
            curr_SNR_true = ((trial_num+1)/trial_num)*(signal_std/noise_std) - 1/trial_num
        #        curr_SNR = (signal_std/noise_std) 
            roi.SNR.append(np.array(curr_SNR_true))
        
        roi.SNR = np.nanmax(np.concatenate(roi.SNR))
        roi.reliability = np.max(roi.reliabilities)
    return rois

import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import ROI_mod
import post_analysis_core as pac

# %% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '191220_GluClflpTEV_NI_1')
results_save_dir = os.path.join(initialDirectory,
                                'results/191220_GluClflpTEV_NI_1')



# %% Load datasets and desired variables
exp_t = '191220_GluCl-flpTEV-NI-1_combinedROIs'
datasets_to_load = ['191216bg_fly5_combined_data_T_1_4_5_6_7.pickle',
                    '191216bg_fly6_combined_data_T_1_3_4_5_8.pickle',
                    '191216bg_fly7_combined_data_T_1_2_3_5_6.pickle',
                    '191218bg_fly1_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly2_combined_data_T_5_6_7.pickle',
                    '191218bg_fly3_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly4_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly5_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly6_combined_data_T_1_2_3_4_5.pickle',
                    '191219bg_fly3_combined_data_T_1_2_5_6_7.pickle',
                    '191218bg_fly6_combined_data_T_1_2_3_4_5.pickle',
                    '191220bg_fly1_combined_data_T_5_6_9_10_11.pickle']

# Initialize variables
final_rois_all = []
flyIDs = []
flash_corr = []
reliabilities = []
SNRs = []

genotypes = []

vert_RF_ON = []
hor_RF_ON = []
hor_RF_ON_centered = []
hor_RF_ON_gauss = []
hor_RF_ON_gauss_fwhm = []
hor_RF_ON_gauss_rsq = []
vert_RF_OFF = []
hor_RF_OFF = []
SNRs = []
Rsq_vals = []


for idataset, dataset in enumerate(datasets_to_load):
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    fff_stim_trace = \
        np.around(np.concatenate((curr_rois[0].int_stim_trace[20:],
                                  curr_rois[0].int_stim_trace[0:30]),axis=0))
        
    
    for roi in curr_rois:
        if (roi.hor_RF_ON_gauss is None) or \
            (roi.hor_RF_ON_Rsq <0.5):
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])

        flash_corr.append(roi.corr_fff)
        
        # Rel SNR
        reliabilities.append(np.mean(roi.reliabilities))
        SNRs.append(np.mean(roi.SNRs))
        
        # RF props
        vert_RF_ON.append(roi.vert_RF_ON_trace)
        hor_RF_ON.append(roi.hor_RF_ON_trace)
        centered = np.full(np.shape(roi.hor_RF_ON_trace),np.nan)
        center = np.argmax(roi.hor_RF_ON_trace)
        if center > 30:
            centered[:30] = roi.hor_RF_ON_trace[center-30:center]
            centered[30:30+60-center] = roi.hor_RF_ON_trace[center:]
        else:
            centered[30-center:30] = roi.hor_RF_ON_trace[:center]
            centered[30:] = roi.hor_RF_ON_trace[center:center+30]
        
        hor_RF_ON_centered.append(centered)
        
        
        hor_RF_ON_gauss.append(roi.hor_RF_ON_gauss)
        hor_RF_ON_gauss_fwhm.append(roi.hor_RF_ON_fwhm)
        hor_RF_ON_gauss_rsq.append(roi.hor_RF_ON_Rsq)
        # vert_RF_OFF.append(roi.vert_RF_OFF_trace)
        # hor_RF_OFF.append(roi.hor_RF_OFF_trace)
        
        rsq = np.array(roi.stripe_Rsq_vals)
        Rsq_vals.append(np.min(rsq[rsq.argsort()[-2:]]))
        SNRs.append(np.max(roi.SNRs))
        
        
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))



unique_gens = np.unique(genotypes)
#%% Stripes analysis
fwhm = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_fwhm,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)
rsqd = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_rsq,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)

df = pd.DataFrame(list(zip(hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq,genotypes,
                           SNRs)), 
               columns =['fwhm', 'rsq','gen','SNR']) 
df['gen'][df['gen']=='Pos_Mi1Rec__plus_GluClflpSTOPD'] = 'Pos'
df['gen'][df['gen']=='Exp_Mi1Rec__GluClflpTEVNI_GluClflpSTOPD'] = 'Exp'

colorss = pac.run_matplotlib_params()
colors = [colorss[5],colorss[2]]
plt.close('all')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('RF properties',fontsize=12)
grid = plt.GridSpec(len(unique_gens),7 ,wspace=1, hspace=0.5)
ax1 = plt.subplot(grid[0,5:])
ax2 = plt.subplot(grid[1,5:])
labels = []
for idx , genotype in enumerate(unique_gens):
    labels.append(genotype[:3])
    ax = plt.subplot(grid[idx,:5])
    sns.heatmap(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype],
                cmap='coolwarm',center=0,ax=ax,yticklabels=20,
                xticklabels=10,vmax=4.5,vmin=-0.5)
    
    
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('ROI #')
    
    
    
    fwhm_data = fwhm['experiment_ids'][genotype]['all_samples']
    rsqd_data = rsqd['experiment_ids'][genotype]['all_samples']
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype[:3],
                                           nflies =\
                                               len(fwhm['experiment_ids'][genotype]['over_samples_means']),
                                           nROIs=\
                                               len(fwhm['experiment_ids'][genotype]['all_samples']))
    ax.set_title(gen_str)      
    sns.distplot(rsqd_data,ax=ax2,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                 hist=False,bins=10)
    ax2.set_xlim((0.5, 1))

# sns.stripplot(x="gen", y="fwhm", hue="gen",palette=colors,data=df, 
#               dodge=True, jitter=True,order=labels,
#               alpha=.5,ax=ax1)
sns.violinplot(x='gen',y='fwhm',data=df,order=labels ,width=0.5,
                ax=ax1,palette=colors,inner="quartile",linewidth=1.5)
ax1.set_ylim((0, 25))
ax1.set_xlabel('')
ax1.set_title('FWHM') 
ax1.set_ylabel('FWHM ($^\circ$)')
ax1.legend('')
ax2.set_title('R$^2$') 
ax2.set_xlabel('R$^2$')
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)

#%%
import random
plt.close('all')
# Randomize ROIs
a=np.array(range(len(genotypes)))
random.shuffle(a)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams["axes.titlesize"] = 'x-small'
plt.rcParams["axes.labelsize"] = 'x-small'

fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
axs = ax1.flatten()
for idx, ax in enumerate(axs):
    curr_idx = a[idx]
    if genotypes[curr_idx][:3] =='Exp':
        curr_color = colors[0]
    else:
        curr_color = colors[1]
    rsq1=np.around(hor_RF_ON_gauss_rsq[curr_idx],2)
    fwhm=np.around(hor_RF_ON_gauss_fwhm[curr_idx],1)
    ax.plot(hor_RF_ON_gauss[curr_idx], 'k',lw=2, 
            label="FWHM: {fwhm}$^\circ$ R$^2$ {rsq1}".format(fwhm=fwhm,rsq1=rsq1))
    ax.plot(hor_RF_ON[curr_idx],color=curr_color,lw=2)

    ax.legend()
fig1.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_randomexp'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')

sns.lmplot('rsq','fwhm',data=df,hue='gen',hue_order=labels,palette=colors,
           scatter_kws={'alpha':0.7, 's':20})
plt.ylim((0, 25))


# %% Load datasets and desired variables
exp_folder = 'stripesH-OFF_all_5deg2deg'#'FFF-simple_colors_2s_0'#'FFF-Dark2s_ON2s'#'FFF-RAISING_INTENSITIES_3s_UVl'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
genotype_labels = 0

# Initialize variables
final_rois_all = []
flyIDs = []
flash_corr = []
reliabilities = []
SNRs = []

genotypes = []

vert_RF_ON = []
hor_RF_ON = []
hor_RF_ON_centered = []
hor_RF_ON_gauss = []
hor_RF_ON_gauss_fwhm = []
hor_RF_ON_gauss_rsq = []
vert_RF_OFF = []
hor_RF_OFF = []
SNRs = []
Rsq_vals = []
flies_list = []
for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = pickle.load(load_path)
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    if 'stripesH' in exp_folder:
        ROI_mod.generate_RF_map_stripes(curr_rois, screen_w = 48) #vert: 64; horiz: 48
    if 'stripesV' in exp_folder:
        ROI_mod.generate_RF_map_stripes(curr_rois, screen_w = 64) #vert: 64; horiz: 48
    """fff_stim_trace = np.around(curr_rois[0].int_stim_trace[0]) # DEBUG: why cut out stim start and put to stim end in Buraks code??; implement all colors
                
    fff_stim_trace = np.around(np.concatenate((curr_rois[0].int_stim_trace[0][20:],
                                  curr_rois[0].int_stim_trace[0][0:30]),axis=0)) 
    """
    ROI_mod.calculateReliability(curr_rois)
    
    for roi in curr_rois:
        back_projected = np.tile(roi.i_stripe_resp, 
                                     (len(roi.i_stripe_resp),1) )
        roi.hor_RF_ON_trace = roi.i_stripe_resp
        roi.hor_RF_ON = back_projected
        roi.hor_RF_ON_gauss = roi.stripe_gauss_profile
        roi.hor_RF_ON_Rsq = roi.stripe_r_squared
        roi.hor_RF_ON_fwhm = roi.stripe_gauss_fwhm
        if (roi.hor_RF_ON_gauss is None) or \
            (roi.hor_RF_ON_Rsq <0.5): # filter based on gauss fit
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])
        flies_list.append(roi.experiment_info['MovieID'].split('_')[0])
        #flash_corr.append(roi.corr_fff)
        
        # Rel SNR
        reliabilities.append(np.mean(roi.reliabilities))
        SNRs.append(roi.SNR)
        
        # RF props
        #vert_RF_ON.append(roi.vert_RF_ON_trace)
        hor_RF_ON.append(roi.hor_RF_ON_trace)
        centered = np.full(np.shape(roi.hor_RF_ON_trace),np.nan)
        center = np.argmax(roi.hor_RF_ON_trace)
        if 'stripesV' in exp_folder:
            if center > 32:
                centered[:32] = roi.hor_RF_ON_trace[center-32:center]
                centered[32:32+64-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[32-center:32] = roi.hor_RF_ON_trace[:center]
                centered[32:] = roi.hor_RF_ON_trace[center:center+32]
        
        if 'stripesH' in exp_folder:
            if center > 24:
                centered[:24] = roi.hor_RF_ON_trace[center-24:center]
                centered[24:24+48-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[24-center:24] = roi.hor_RF_ON_trace[:center]
                centered[24:] = roi.hor_RF_ON_trace[center:center+24]
        
        hor_RF_ON_centered.append(centered)
        
        
        hor_RF_ON_gauss.append(roi.hor_RF_ON_gauss)
        hor_RF_ON_gauss_fwhm.append(roi.hor_RF_ON_fwhm)
        hor_RF_ON_gauss_rsq.append(roi.hor_RF_ON_Rsq)
        # vert_RF_OFF.append(roi.vert_RF_OFF_trace)
        # hor_RF_OFF.append(roi.hor_RF_OFF_trace)
        
        rsq = np.array(roi.hor_RF_ON_Rsq)
        Rsq_vals.append(rsq)
        #SNRs.append(SNRs))
        
        
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))



unique_gens = np.unique(genotypes)
# Stripes analysis
fwhm = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_fwhm,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)
rsqd = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_rsq,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)

df = pd.DataFrame(list(zip(hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq,genotypes,
                           SNRs)), 
               columns =['fwhm', 'rsq','gen','SNR']) 
#df['gen'][df['gen']=='Pos_Mi1Rec__plus_GluClflpSTOPD'] = 'Pos'
#df['gen'][df['gen']=='Exp_Mi1Rec__GluClflpTEVNI_GluClflpSTOPD'] = 'Exp'

colorss = pac.run_matplotlib_params()
colors = colorss[0]
plt.close('all')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('RF properties',fontsize=12)
grid = plt.GridSpec(len(unique_gens)*2,7 ,wspace=1, hspace=0.5)
ax1 = plt.subplot(grid[0,5:])
ax2 = plt.subplot(grid[1,5:])
labels = []
for idx , genotype in enumerate(unique_gens):
    labels.append(genotype)
    ax = plt.subplot(grid[idx,:5])
    sns.heatmap(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype],
                cmap='coolwarm',center=0,ax=ax,yticklabels=20,
                xticklabels=10,vmin=-0.5)#,vmax=4.5)
    
    
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('ROI #')
    
    # DEUBUG: plot all centered traces
    ax = plt.subplot(grid[idx+1,:5])
    lineplot_data = pd.DataFrame(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype])
    sns.lineplot(data=lineplot_data.transpose(), palette=sns.color_palette(['gray'], len(lineplot_data)), linewidth = 1, linestyle = 'solid')
    sns.lineplot(data=lineplot_data.mean(axis=0), color='red', linewidth = 1)
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('dF/F')
    ax.legend('')
    
    fwhm_data = fwhm['experiment_ids'][genotype]['all_samples']
    rsqd_data = rsqd['experiment_ids'][genotype]['all_samples']
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype,
                                           nflies =\
                                               #len(fwhm['experiment_ids'][genotype]['over_samples_means']),
                                               len(np.unique(flies_list)),
                                           nROIs=\
                                               len(fwhm['experiment_ids'][genotype]['all_samples']))
                                               
    ax.set_title(gen_str)      
    sns.distplot(rsqd_data,ax=ax2,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                 hist=False,bins=10)
    ax2.set_xlim(0.5, 1)

# sns.stripplot(x="gen", y="fwhm", hue="gen",palette=colors,data=df, 
#               dodge=True, jitter=True,order=labels,
#               alpha=.5,ax=ax1)
sns.violinplot(x='gen',y='fwhm',data=df,order=labels ,width=0.5,
                ax=ax1,palette=colors,inner="quartile",linewidth=1.5)
print('Median = '+str(np.median(fwhm_data)))
print('FWHM STDev = '+str(np.std(fwhm_data)))
ax1.set_ylim((0, 20)) # limit of y Axis FWHM Plot
ax1.set_xlabel('')
ax1.set_title('FWHM') 
ax1.set_ylabel('FWHM ($^\circ$)')
ax1.legend('')
ax1.set_xticklabels([''])
ax1.text(1,1,str(np.round(np.median(fwhm_data), decimals = 1))+'° IQR: '+str(np.round(scipy.stats.iqr(fwhm_data), decimals = 1))+'°',style='normal', transform=ax1.transAxes, horizontalalignment='right', verticalalignment='top')
ax2.set_title('R$^2$') 
ax2.set_xlabel('R$^2$')
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
    np.save(os.path.join(results_save_dir, exp_folder + '.npy'),
            [hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq, genotypes, SNRs])

import random
plt.close('all')
# Randomize ROIs
a=np.array(range(len(genotypes)))
random.shuffle(a)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams["axes.titlesize"] = 'x-small'
plt.rcParams["axes.labelsize"] = 'x-small'

fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
axs = ax1.flatten()
for idx, ax in enumerate(axs):
    curr_idx = a[idx]
    if genotypes[curr_idx] =='Exp':
        curr_color = colors[0]
    else:
        curr_color = colors[1]
    rsq1=np.around(hor_RF_ON_gauss_rsq[curr_idx],2)
    fwhm=np.around(hor_RF_ON_gauss_fwhm[curr_idx],1)
    ax.plot(hor_RF_ON_gauss[curr_idx], 'k',lw=2, 
            label="FWHM: {fwhm}$^\circ$ R$^2$ {rsq1}".format(fwhm=fwhm,rsq1=rsq1))
    ax.plot(hor_RF_ON[curr_idx],color=curr_color,lw=2)

    ax.legend()
fig1.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc_randomexp'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
plt.close('all')
sns.lmplot('rsq','fwhm',data=df,hue='gen',hue_order=labels,palette=colors,
           scatter_kws={'alpha':0.7, 's':20})
plt.ylim((0, 25))
plt.show()  


def compute_over_samples_groups(data = None, group_ids= None, error ='std',
                                   experiment_ids = None):
    """ 
    
    Computes averages and std or SEM of a given dataset over samples and
    groups
    """
    # Input check
    if (data is None):
        raise TypeError('Data missing.')
    elif (group_ids is None):
        raise TypeError('Sample IDs are missing.')
    elif (experiment_ids is None):
        raise TypeError('Experiment IDs are missing.')
    
    # Type check and conversion to numpy arrays
    if (type(data) is list):
        data = np.array(data)
    if (type(group_ids) is list):
        group_ids = np.array(group_ids)
    if (type(experiment_ids) is list):
        experiment_ids = np.array(experiment_ids)
    
    
    data_dict = {}
    data_dict['experiment_ids'] = {}
    
    unique_experiment_ids = np.unique(experiment_ids)
    for exp_id in unique_experiment_ids:
        data_dict['experiment_ids'][exp_id] = {}
        data_dict['experiment_ids'][exp_id]['over_samples_means'] = []
        data_dict['experiment_ids'][exp_id]['over_samples_errors'] = []
        data_dict['experiment_ids'][exp_id]['uniq_group_ids'] = np.array([])
        
        data_dict['experiment_ids'][exp_id]['all_samples'] = \
            data[experiment_ids == exp_id]
        
        curr_groups = group_ids[np.argwhere(experiment_ids == exp_id)]
        
        for group in np.unique(curr_groups):
            curr_mask = (group == group_ids)
            
            ### old code ### delete ### DEBUG
            """curr_data = np.nanmean(data[curr_mask],axis=0)
            if error == 'std':
                err = np.nanstd(data[curr_mask],axis=0)
            elif error == 'SEM':
                err = \
                    np.nanstd(data[curr_mask],axis=0) / np.sqrt(np.shape(data[curr_mask])[0])"""
        
            #### new code
            curr_data = np.nanmean(np.vstack(data[curr_mask]),axis=0)
            if error == 'std':
                err = np.nanstd(data[curr_mask],axis=0)
            elif error == 'SEM':
                err = \
                    np.nanstd(np.nanmean(np.vstack(data[curr_mask]),axis=0)) / np.sqrt(np.shape(np.vstack(data[curr_mask]))[0])



            data_dict['experiment_ids'][exp_id]['over_samples_means'].append(\
                 curr_data)
            data_dict['experiment_ids'][exp_id]['over_samples_errors'].append(\
                 err)
            data_dict['experiment_ids'][exp_id]['uniq_group_ids'] = \
                np.append(data_dict['experiment_ids'][exp_id]['uniq_group_ids'],
                          group)
        data_dict['experiment_ids'][exp_id]['over_groups_mean'] = \
            np.nanmean(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0)
        if error == 'std':
            err = \
                np.nanstd(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0)
        elif error == 'SEM':
            err = \
                np.nanstd(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0) / \
                    np.sqrt(np.shape(data_dict['experiment_ids'][exp_id]['over_samples_means'])[0])
        data_dict['experiment_ids'][exp_id]['over_groups_error'] =err
            
    return data_dict

class NumpyCoreFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)
    
## Screen:
## width	    height
## 65.45992168°	51.47236009°


## TO DO
# test if findMaxResponse_all_epochs should compute maxima by subtracting 0.5s mean of F before stim onset for proper dF/F and/ore use mean instead of maxima!? see README.md
# X-axis: put 0° in tenter and +° right and -° left of it


# %% Setting the directories
#initialDirectory = r'Z:\2P\2p_DATA\tm5-3D2'#'R:\\2P\\2p_DATA\\Tm9Lexa_GCamp6f' #ERosco10 Lee713WinterBlue' #'R:\\2P\\2p_DATA\Ronny\\L3_GCaMP6f\\with goggles\Lee713WinterBlue' #ERosco10 Lee713WinterBlue
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results')

## New post analysis structure

# %% Load datasets and desired variables
exp_folder = 'stripesH-OFF_all_5deg2deg'#'FFF-simple_colors_2s_0'#'FFF-Dark2s_ON2s'#'FFF-RAISING_INTENSITIES_3s_UVl'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
genotype_labels = 0

# Initialize variables
final_rois_all = []
flyIDs = []
flash_corr = []
reliabilities = []
SNRs = []

genotypes = []

vert_RF_ON = []
hor_RF_ON = []
hor_RF_ON_centered = []
hor_RF_ON_gauss = []
hor_RF_ON_gauss_fwhm = []
hor_RF_ON_gauss_rsq = []
vert_RF_OFF = []
hor_RF_OFF = []
SNRs = []
Rsq_vals = []
flies_list = []
for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = NumpyCoreFixUnpickler(load_path).load()
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    if 'stripesH' in exp_folder:
        ROI_mod.generate_RF_map_stripes(curr_rois, screen_w = 48) #vert: 64; horiz: 48
    if 'stripesV' in exp_folder:
        ROI_mod.generate_RF_map_stripes(curr_rois, screen_w = 64) #vert: 64; horiz: 48
    """fff_stim_trace = np.around(curr_rois[0].int_stim_trace[0]) # DEBUG: why cut out stim start and put to stim end in Buraks code??; implement all colors
                
    fff_stim_trace = np.around(np.concatenate((curr_rois[0].int_stim_trace[0][20:],
                                  curr_rois[0].int_stim_trace[0][0:30]),axis=0)) 
    """
    ROI_mod.calculateReliability(curr_rois)
    
    for roi in curr_rois:
        back_projected = np.tile(roi.i_stripe_resp, 
                                     (len(roi.i_stripe_resp),1) )
        roi.hor_RF_ON_trace = roi.i_stripe_resp
        roi.hor_RF_ON = back_projected
        roi.hor_RF_ON_gauss = roi.stripe_gauss_profile
        roi.hor_RF_ON_Rsq = roi.stripe_r_squared
        roi.hor_RF_ON_fwhm = roi.stripe_gauss_fwhm
        if (roi.hor_RF_ON_gauss is None) or \
            (roi.hor_RF_ON_Rsq <0.5): # filter based on gauss fit
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])
        flies_list.append(roi.experiment_info['MovieID'].split('_')[0])
        #flash_corr.append(roi.corr_fff)
        
        # Rel SNR
        reliabilities.append(np.mean(roi.reliabilities))
        SNRs.append(roi.SNR)
        
        # RF props
        #vert_RF_ON.append(roi.vert_RF_ON_trace)
        hor_RF_ON.append(roi.hor_RF_ON_trace)
        centered = np.full(np.shape(roi.hor_RF_ON_trace),np.nan)
        center = np.argmax(roi.hor_RF_ON_trace)
        if 'stripesV' in exp_folder:
            if center > 32:
                centered[:32] = roi.hor_RF_ON_trace[center-32:center]
                centered[32:32+64-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[32-center:32] = roi.hor_RF_ON_trace[:center]
                centered[32:] = roi.hor_RF_ON_trace[center:center+32]
        
        if 'stripesH' in exp_folder:
            if center > 24:
                centered[:24] = roi.hor_RF_ON_trace[center-24:center]
                centered[24:24+48-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[24-center:24] = roi.hor_RF_ON_trace[:center]
                centered[24:] = roi.hor_RF_ON_trace[center:center+24]
        
        hor_RF_ON_centered.append(centered)
        
        
        hor_RF_ON_gauss.append(roi.hor_RF_ON_gauss)
        hor_RF_ON_gauss_fwhm.append(roi.hor_RF_ON_fwhm)
        hor_RF_ON_gauss_rsq.append(roi.hor_RF_ON_Rsq)
        # vert_RF_OFF.append(roi.vert_RF_OFF_trace)
        # hor_RF_OFF.append(roi.hor_RF_OFF_trace)
        
        rsq = np.array(roi.hor_RF_ON_Rsq)
        Rsq_vals.append(rsq)
        #SNRs.append(SNRs))
        
        
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))



unique_gens = np.unique(genotypes)
#%% Stripes analysis
fwhm = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_fwhm,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)
rsqd = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_rsq,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)

df = pd.DataFrame(list(zip(hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq,genotypes,
                           SNRs)), 
               columns =['fwhm', 'rsq','gen','SNR']) 
#df['gen'][df['gen']=='Pos_Mi1Rec__plus_GluClflpSTOPD'] = 'Pos'
#df['gen'][df['gen']=='Exp_Mi1Rec__GluClflpTEVNI_GluClflpSTOPD'] = 'Exp'

colorss = pac.run_matplotlib_params()
colors = colorss[0]
plt.close('all')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('RF properties',fontsize=12)
grid = plt.GridSpec(len(unique_gens)*2,7 ,wspace=1, hspace=0.5)
ax1 = plt.subplot(grid[0,5:])
ax2 = plt.subplot(grid[1,5:])
labels = []
for idx , genotype in enumerate(unique_gens):
    labels.append(genotype)
    ax = plt.subplot(grid[idx,:5])
    sns.heatmap(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype],
                cmap='coolwarm',center=0,ax=ax,yticklabels=20,
                xticklabels=10,vmin=-0.5)#,vmax=4.5)
    
    
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('ROI #')
    
    # DEUBUG: plot all centered traces
    ax = plt.subplot(grid[idx+1,:5])
    lineplot_data = pd.DataFrame(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype])
    sns.lineplot(data=lineplot_data.transpose(), palette=sns.color_palette(['gray'], len(lineplot_data)), linewidth = 1, linestyle = 'solid')
    sns.lineplot(data=lineplot_data.mean(axis=0), color='red', linewidth = 1)
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('dF/F')
    ax.legend('')
    
    fwhm_data = fwhm['experiment_ids'][genotype]['all_samples']
    rsqd_data = rsqd['experiment_ids'][genotype]['all_samples']
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype,
                                           nflies =\
                                               #len(fwhm['experiment_ids'][genotype]['over_samples_means']),
                                               len(np.unique(flies_list)),
                                           nROIs=\
                                               len(fwhm['experiment_ids'][genotype]['all_samples']))
                                               
    ax.set_title(gen_str)      
    sns.distplot(rsqd_data,ax=ax2,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                 hist=False,bins=10)
    ax2.set_xlim(0.5, 1)

# sns.stripplot(x="gen", y="fwhm", hue="gen",palette=colors,data=df, 
#               dodge=True, jitter=True,order=labels,
#               alpha=.5,ax=ax1)
sns.violinplot(x='gen',y='fwhm',data=df,order=labels ,width=0.5,
                ax=ax1,palette=colors,inner="quartile",linewidth=1.5)
print('Median = '+str(np.median(fwhm_data)))
print('FWHM STDev = '+str(np.std(fwhm_data)))
ax1.set_ylim((0, 20)) # limit of y Axis FWHM Plot
ax1.set_xlabel('')
ax1.set_title('FWHM') 
ax1.set_ylabel('FWHM ($^\circ$)')
ax1.legend('')
ax1.set_xticklabels([''])
ax1.text(1,1,str(np.round(np.median(fwhm_data), decimals = 1))+'° IQR: '+str(np.round(scipy.stats.iqr(fwhm_data), decimals = 1))+'°',style='normal', transform=ax1.transAxes, horizontalalignment='right', verticalalignment='top')
ax2.set_title('R$^2$') 
ax2.set_xlabel('R$^2$')
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()

#%%
import random
plt.close('all')
# Randomize ROIs
a=np.array(range(len(genotypes)))
random.shuffle(a)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams["axes.titlesize"] = 'x-small'
plt.rcParams["axes.labelsize"] = 'x-small'

fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
axs = ax1.flatten()
for idx, ax in enumerate(axs):
    curr_idx = a[idx]
    if genotypes[curr_idx] =='Exp':
        curr_color = colors[0]
    else:
        curr_color = colors[1]
    rsq1=np.around(hor_RF_ON_gauss_rsq[curr_idx],2)
    fwhm=np.around(hor_RF_ON_gauss_fwhm[curr_idx],1)
    ax.plot(hor_RF_ON_gauss[curr_idx], 'k',lw=2, 
            label="FWHM: {fwhm}$^\circ$ R$^2$ {rsq1}".format(fwhm=fwhm,rsq1=rsq1))
    ax.plot(hor_RF_ON[curr_idx],color=curr_color,lw=2)

    ax.legend()
fig1.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc_randomexp'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
plt.close('all')
#%%
sns.lmplot('rsq','fwhm',data=df,hue='gen',hue_order=labels,palette=colors,
           scatter_kws={'alpha':0.7, 's':20})
plt.ylim((0, 25))
plt.show()  



# %% Load datasets and desired variables
exp_folder = 'stripesH-ON_all_5deg2deg'#'FFF-simple_colors_2s_0'#'FFF-Dark2s_ON2s'#'FFF-RAISING_INTENSITIES_3s_UVl'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
genotype_labels = 0

# Initialize variables
final_rois_all = []
flyIDs = []
flash_corr = []
reliabilities = []
SNRs = []

genotypes = []

vert_RF_ON = []
hor_RF_ON = []
hor_RF_ON_centered = []
hor_RF_ON_gauss = []
hor_RF_ON_gauss_fwhm = []
hor_RF_ON_gauss_rsq = []
vert_RF_OFF = []
hor_RF_OFF = []
SNRs = []
Rsq_vals = []
flies_list = []
for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = NumpyCoreFixUnpickler(load_path).load()
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    if 'stripesH' in exp_folder:
        ROI_mod.generate_RF_map_stripes_neg(curr_rois, screen_w = 48) #vert: 65; horiz: 51
    if 'stripesV' in exp_folder:
        ROI_mod.generate_RF_map_stripes_neg(curr_rois, screen_w = 64) #vert: 65; horiz: 51
    """fff_stim_trace = np.around(curr_rois[0].int_stim_trace[0]) # DEBUG: why cut out stim start and put to stim end in Buraks code??; implement all colors
                
    fff_stim_trace = np.around(np.concatenate((curr_rois[0].int_stim_trace[0][20:],
                                  curr_rois[0].int_stim_trace[0][0:30]),axis=0)) 
    """
    ROI_mod.calculateReliability(curr_rois)
    
    for roi in curr_rois:
        back_projected = np.tile(roi.i_stripe_resp, 
                                     (len(roi.i_stripe_resp),1) )
        roi.hor_RF_ON_trace = roi.i_stripe_resp
        roi.hor_RF_ON = back_projected
        roi.hor_RF_ON_gauss = roi.stripe_gauss_profile
        roi.hor_RF_ON_Rsq = roi.stripe_r_squared
        roi.hor_RF_ON_fwhm = roi.stripe_gauss_fwhm
        if (roi.hor_RF_ON_gauss is None) or \
            (roi.hor_RF_ON_Rsq <0.): # select based on gaussian fit
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])
        flies_list.append(roi.experiment_info['MovieID'].split('_')[0])
        #flash_corr.append(roi.corr_fff)
        
        # Rel SNR
        reliabilities.append(np.mean(roi.reliabilities))
        SNRs.append(roi.SNR)
        
        # RF props
        #vert_RF_ON.append(roi.vert_RF_ON_trace)
        hor_RF_ON.append(roi.hor_RF_ON_trace)
        centered = np.full(np.shape(roi.hor_RF_ON_trace),np.nan)
        center = np.argmin(roi.hor_RF_ON_trace)
        if 'stripesV' in exp_folder:
            if center > 32:
                centered[:32] = roi.hor_RF_ON_trace[center-32:center]
                centered[32:32+64-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[32-center:32] = roi.hor_RF_ON_trace[:center]
                centered[32:] = roi.hor_RF_ON_trace[center:center+32]
        
        if 'stripesH' in exp_folder:
            if center > 24:
                centered[:24] = roi.hor_RF_ON_trace[center-24:center]
                centered[24:24+48-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[24-center:24] = roi.hor_RF_ON_trace[:center]
                centered[24:] = roi.hor_RF_ON_trace[center:center+24]
        
        hor_RF_ON_centered.append(centered)
        
        
        hor_RF_ON_gauss.append(roi.hor_RF_ON_gauss)
        hor_RF_ON_gauss_fwhm.append(roi.hor_RF_ON_fwhm)
        hor_RF_ON_gauss_rsq.append(roi.hor_RF_ON_Rsq)
        # vert_RF_OFF.append(roi.vert_RF_OFF_trace)
        # hor_RF_OFF.append(roi.hor_RF_OFF_trace)
        
        rsq = np.array(roi.hor_RF_ON_Rsq)
        Rsq_vals.append(rsq)
        #SNRs.append(SNRs))
        
        
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))



unique_gens = np.unique(genotypes)
#%% Stripes analysis
fwhm = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_fwhm,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)
rsqd = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_rsq,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)

df = pd.DataFrame(list(zip(hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq,genotypes,
                           SNRs)), 
               columns =['fwhm', 'rsq','gen','SNR']) 
#df['gen'][df['gen']=='Pos_Mi1Rec__plus_GluClflpSTOPD'] = 'Pos'
#df['gen'][df['gen']=='Exp_Mi1Rec__GluClflpTEVNI_GluClflpSTOPD'] = 'Exp'

colorss = pac.run_matplotlib_params()
colors = colorss[0]
plt.close('all')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('RF properties',fontsize=12)
grid = plt.GridSpec(len(unique_gens)*2,7 ,wspace=1, hspace=0.5)
ax1 = plt.subplot(grid[0,5:])
ax2 = plt.subplot(grid[1,5:])
labels = []
for idx , genotype in enumerate(unique_gens):
    labels.append(genotype)
    ax = plt.subplot(grid[idx,:5])
    sns.heatmap(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype],
                cmap='coolwarm',center=0,ax=ax,yticklabels=20,
                xticklabels=10)#,vmin=-0.5)#,vmax=4.5)
    
    
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('ROI #')
    
    # DEUBUG: plot all centered traces
    ax = plt.subplot(grid[idx+1,:5])
    lineplot_data = pd.DataFrame(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype])
    sns.lineplot(data=lineplot_data.transpose(), palette=sns.color_palette(['gray'], len(lineplot_data)), linewidth = 1, linestyle = 'solid')
    sns.lineplot(data=lineplot_data.mean(axis=0), color='red', linewidth = 1)
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('dF/F')
    ax.legend('')
    
    fwhm_data = fwhm['experiment_ids'][genotype]['all_samples']
    rsqd_data = rsqd['experiment_ids'][genotype]['all_samples']
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype,
                                           nflies =\
                                               len(fwhm['experiment_ids'][genotype]['over_samples_means']),
                                           nROIs=\
                                               #len(fwhm['experiment_ids'][genotype]['all_samples']))
                                               len(np.unique(flies_list)))
    ax.set_title(gen_str)      
    sns.distplot(rsqd_data,ax=ax2,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                 hist=False,bins=10)
    ax2.set_xlim((0.5, 1))

# sns.stripplot(x="gen", y="fwhm", hue="gen",palette=colors,data=df, 
#               dodge=True, jitter=True,order=labels,
#               alpha=.5,ax=ax1)
sns.violinplot(x='gen',y='fwhm',data=df,order=labels ,width=0.5,
                ax=ax1,palette=colors,inner="quartile",linewidth=1.5)
print('Median = '+str(np.median(fwhm_data)))
print('FWHM STDev = '+str(np.std(fwhm_data)))
ax1.set_ylim([0, 50])  # limit of y Axis FWHM Plot
#ax1.set_ylim((0, 25))
ax1.set_xlabel('')
ax1.set_title('FWHM') 
ax1.set_ylabel('FWHM ($^\circ$)')
ax1.legend('')
ax1.set_xticklabels([''])
ax1.text(1,1,str(np.round(np.median(fwhm_data), decimals = 1))+'° IQR: '+str(np.round(scipy.stats.iqr(fwhm_data), decimals = 1))+'°',style='normal', transform=ax1.transAxes, horizontalalignment='right', verticalalignment='top')
ax2.set_xlabel('R$^2$')
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_inh'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()

#%%
import random
plt.close('all')
# Randomize ROIs
a=np.array(range(len(genotypes)))
random.shuffle(a)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams["axes.titlesize"] = 'x-small'
plt.rcParams["axes.labelsize"] = 'x-small'

fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
axs = ax1.flatten()
for idx, ax in enumerate(axs):
    curr_idx = a[idx]
    if genotypes[curr_idx] =='Exp':
        curr_color = colors[0]
    else:
        curr_color = colors[1]
    rsq1=np.around(hor_RF_ON_gauss_rsq[curr_idx],2)
    fwhm=np.around(hor_RF_ON_gauss_fwhm[curr_idx],1)
    ax.plot(hor_RF_ON_gauss[curr_idx], 'k',lw=2, 
            label="FWHM: {fwhm}$^\circ$ R$^2$ {rsq1}".format(fwhm=fwhm,rsq1=rsq1))
    ax.plot(hor_RF_ON[curr_idx],color=curr_color,lw=2)

    ax.legend()
fig1.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_inh_randomexp'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
plt.close('all')
#%%
sns.lmplot('rsq','fwhm',data=df,hue='gen',hue_order=labels,palette=colors,
           scatter_kws={'alpha':0.7, 's':20})
plt.ylim((0, 25))
plt.show()  


class NumpyCoreFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)
    
## Screen:
## width	    height
## 65.45992168°	51.47236009°

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### to compute offset responses: change the following function: findMaxResponse_all_epochs(roi) in ROImod.py: ########################################################
# compute response amplitude after stimulus offset (2s period after offset)
# roi.max_resp_all_epochs[epoch_idx] = np.nanmean(roi.whole_traces[epoch_idx][40:]-median_baseline_epoch)
#########################################################################################################################################################



# %% Setting the directories
initialDirectory = r'S:\students\MSc2026\MSc_26_grp_all\Tm1_25C' #ERosco10 Lee713WinterBlue' #'R:\\2P\\2p_DATA\Ronny\\L3_GCaMP6f\\with goggles\Lee713WinterBlue' #ERosco10 Lee713WinterBlue
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results')

## New post analysis structure

# %% Load datasets and desired variables
exp_folder = 'stripesV-ON_all_5deg2deg'#'FFF-simple_colors_2s_0'#'FFF-Dark2s_ON2s'#'FFF-RAISING_INTENSITIES_3s_UVl'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
genotype_labels = 0

# Initialize variables
final_rois_all = []
flyIDs = []
flash_corr = []
reliabilities = []
SNRs = []

genotypes = []

vert_RF_ON = []
hor_RF_ON = []
hor_RF_ON_centered = []
hor_RF_ON_gauss = []
hor_RF_ON_gauss_fwhm = []
hor_RF_ON_gauss_rsq = []
vert_RF_OFF = []
hor_RF_OFF = []
SNRs = []
Rsq_vals = []
flies_list = []
for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = NumpyCoreFixUnpickler(load_path).load()
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    if 'stripesH' in exp_folder:
        ROI_mod.generate_RF_map_stripes(curr_rois, screen_w = 48) #vert: 64; horiz: 48
    if 'stripesV' in exp_folder:
        ROI_mod.generate_RF_map_stripes(curr_rois, screen_w = 64) #vert: 64; horiz: 48
    """fff_stim_trace = np.around(curr_rois[0].int_stim_trace[0]) # DEBUG: why cut out stim start and put to stim end in Buraks code??; implement all colors
                
    fff_stim_trace = np.around(np.concatenate((curr_rois[0].int_stim_trace[0][20:],
                                  curr_rois[0].int_stim_trace[0][0:30]),axis=0)) 
    """
    ROI_mod.calculateReliability(curr_rois)
    
    for roi in curr_rois:
        back_projected = np.tile(roi.i_stripe_resp, 
                                     (len(roi.i_stripe_resp),1) )
        roi.hor_RF_ON_trace = roi.i_stripe_resp
        roi.hor_RF_ON = back_projected
        roi.hor_RF_ON_gauss = roi.stripe_gauss_profile
        roi.hor_RF_ON_Rsq = roi.stripe_r_squared
        roi.hor_RF_ON_fwhm = roi.stripe_gauss_fwhm
        if (roi.hor_RF_ON_gauss is None) or \
            (roi.hor_RF_ON_Rsq <0.5): # filter based on gauss fit
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])
        flies_list.append(roi.experiment_info['MovieID'].split('_')[0])
        #flash_corr.append(roi.corr_fff)
        
        # Rel SNR
        reliabilities.append(np.mean(roi.reliabilities))
        SNRs.append(roi.SNR)
        
        # RF props
        #vert_RF_ON.append(roi.vert_RF_ON_trace)
        hor_RF_ON.append(roi.hor_RF_ON_trace)
        centered = np.full(np.shape(roi.hor_RF_ON_trace),np.nan)
        center = np.argmax(roi.hor_RF_ON_trace)
        if 'stripesV' in exp_folder:
            if center > 32:
                centered[:32] = roi.hor_RF_ON_trace[center-32:center]
                centered[32:32+64-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[32-center:32] = roi.hor_RF_ON_trace[:center]
                centered[32:] = roi.hor_RF_ON_trace[center:center+32]
        
        if 'stripesH' in exp_folder:
            if center > 24:
                centered[:24] = roi.hor_RF_ON_trace[center-24:center]
                centered[24:24+48-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[24-center:24] = roi.hor_RF_ON_trace[:center]
                centered[24:] = roi.hor_RF_ON_trace[center:center+24]
        
        hor_RF_ON_centered.append(centered)
        
        
        hor_RF_ON_gauss.append(roi.hor_RF_ON_gauss)
        hor_RF_ON_gauss_fwhm.append(roi.hor_RF_ON_fwhm)
        hor_RF_ON_gauss_rsq.append(roi.hor_RF_ON_Rsq)
        # vert_RF_OFF.append(roi.vert_RF_OFF_trace)
        # hor_RF_OFF.append(roi.hor_RF_OFF_trace)
        
        rsq = np.array(roi.hor_RF_ON_Rsq)
        Rsq_vals.append(rsq)
        #SNRs.append(SNRs))
        
        
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))



unique_gens = np.unique(genotypes)
#%% Stripes analysis
fwhm = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_fwhm,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)
rsqd = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_rsq,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)

df = pd.DataFrame(list(zip(hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq,genotypes,
                           SNRs)), 
               columns =['fwhm', 'rsq','gen','SNR']) 
#df['gen'][df['gen']=='Pos_Mi1Rec__plus_GluClflpSTOPD'] = 'Pos'
#df['gen'][df['gen']=='Exp_Mi1Rec__GluClflpTEVNI_GluClflpSTOPD'] = 'Exp'

colorss = pac.run_matplotlib_params()
colors = colorss[0]
plt.close('all')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('RF properties',fontsize=12)
grid = plt.GridSpec(len(unique_gens)*2,7 ,wspace=1, hspace=0.5)
ax1 = plt.subplot(grid[0,5:])
ax2 = plt.subplot(grid[1,5:])
labels = []
for idx , genotype in enumerate(unique_gens):
    labels.append(genotype)
    ax = plt.subplot(grid[idx,:5])
    sns.heatmap(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype],
                cmap='coolwarm',center=0,ax=ax,yticklabels=20,
                xticklabels=10,vmin=-0.5)#,vmax=4.5)
    
    
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('ROI #')
    
    # DEUBUG: plot all centered traces
    ax = plt.subplot(grid[idx+1,:5])
    lineplot_data = pd.DataFrame(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype])
    sns.lineplot(data=lineplot_data.transpose(), palette=sns.color_palette(['gray'], len(lineplot_data)), linewidth = 1, linestyle = 'solid')
    sns.lineplot(data=lineplot_data.mean(axis=0), color='red', linewidth = 1)
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('dF/F')
    ax.legend('')
    
    fwhm_data = fwhm['experiment_ids'][genotype]['all_samples']
    rsqd_data = rsqd['experiment_ids'][genotype]['all_samples']
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype,
                                           nflies =\
                                               #len(fwhm['experiment_ids'][genotype]['over_samples_means']),
                                               len(np.unique(flies_list)),
                                           nROIs=\
                                               len(fwhm['experiment_ids'][genotype]['all_samples']))
                                               
    ax.set_title(gen_str)      
    sns.distplot(rsqd_data,ax=ax2,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                 hist=False,bins=10)
    ax2.set_xlim(0.5, 1)

# sns.stripplot(x="gen", y="fwhm", hue="gen",palette=colors,data=df, 
#               dodge=True, jitter=True,order=labels,
#               alpha=.5,ax=ax1)
sns.violinplot(x='gen',y='fwhm',data=df,order=labels ,width=0.5,
                ax=ax1,palette=colors,inner="quartile",linewidth=1.5)
print('Median = '+str(np.median(fwhm_data)))
print('FWHM STDev = '+str(np.std(fwhm_data)))
ax1.set_ylim((0, 20)) # limit of y Axis FWHM Plot
ax1.set_xlabel('')
ax1.set_title('FWHM') 
ax1.set_ylabel('FWHM ($^\circ$)')
ax1.legend('')
ax1.set_xticklabels([''])
ax1.text(1,1,str(np.round(np.median(fwhm_data), decimals = 1))+'° IQR: '+str(np.round(scipy.stats.iqr(fwhm_data), decimals = 1))+'°',style='normal', transform=ax1.transAxes, horizontalalignment='right', verticalalignment='top')
ax2.set_title('R$^2$') 
ax2.set_xlabel('R$^2$')
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()

#%%
import random
plt.close('all')
# Randomize ROIs
a=np.array(range(len(genotypes)))
random.shuffle(a)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams["axes.titlesize"] = 'x-small'
plt.rcParams["axes.labelsize"] = 'x-small'

fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
axs = ax1.flatten()
for idx, ax in enumerate(axs):
    curr_idx = a[idx]
    if genotypes[curr_idx] =='Exp':
        curr_color = colors[0]
    else:
        curr_color = colors[1]
    rsq1=np.around(hor_RF_ON_gauss_rsq[curr_idx],2)
    fwhm=np.around(hor_RF_ON_gauss_fwhm[curr_idx],1)
    ax.plot(hor_RF_ON_gauss[curr_idx], 'k',lw=2, 
            label="FWHM: {fwhm}$^\circ$ R$^2$ {rsq1}".format(fwhm=fwhm,rsq1=rsq1))
    ax.plot(hor_RF_ON[curr_idx],color=curr_color,lw=2)

    ax.legend()
fig1.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc_randomexp'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
plt.close('all')
#%%
sns.lmplot('rsq','fwhm',data=df,hue='gen',hue_order=labels,palette=colors,
           scatter_kws={'alpha':0.7, 's':20})
plt.ylim((0, 25))
plt.show()  

# Local modules and paths.
"""from paths import initialDirectory, stimDir, functionPath
os.chdir(functionPath)
if functionPath not in sys.path:
    sys.path.append(functionPath)"""
import scipy.io

## Screen:
## width	    height
## 65.45992168°	51.47236009°


## TO DO
# test if findMaxResponse_all_epochs should compute maxima by subtracting 0.5s mean of F before stim onset for proper dF/F and/ore use mean instead of maxima!? see README.md
# X-axis: put 0° in tenter and +° right and -° left of it


# %% Setting the directories
initialDirectory = r'Z:\2P\2p_DATA\Tm5a-splitgal4'#'R:\\2P\\2p_DATA\\Tm9Lexa_GCamp6f' #ERosco10 Lee713WinterBlue' #'R:\\2P\\2p_DATA\Ronny\\L3_GCaMP6f\\with goggles\Lee713WinterBlue' #ERosco10 Lee713WinterBlue
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results')

## New post analysis structure

# %% Load datasets and desired variables
exp_folder = 'movingStripes_white15'#'FFF-simple_colors_2s_0'#'FFF-Dark2s_ON2s'#'FFF-RAISING_INTENSITIES_3s_UVl'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
genotype_labels = 0

# Initialize variables
final_rois_all = []
flyIDs = []
flash_corr = []
reliabilities = []
SNRs = []

genotypes = []

vert_RF_ON = []
hor_RF_ON = []
hor_RF_ON_centered = []
hor_RF_ON_gauss = []
hor_RF_ON_gauss_fwhm = []
hor_RF_ON_gauss_rsq = []
vert_RF_OFF = []
hor_RF_OFF = []
SNRs = []
Rsq_vals = []
flies_list = []
for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = pickle.load(load_path)
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    
    ROI_mod.generate_RF_map_moving_stripes(curr_rois, screen_w = 64) #vert: 64; horiz: 48
    """if 'stripesV' in exp_folder:
        ROI_mod.generate_RF_map_stripes(curr_rois, screen_w = 64) #vert: 64; horiz: 48"""

    """fff_stim_trace = np.around(curr_rois[0].int_stim_trace[0]) # DEBUG: why cut out stim start and put to stim end in Buraks code??; implement all colors
                
    fff_stim_trace = np.around(np.concatenate((curr_rois[0].int_stim_trace[0][20:],
                                  curr_rois[0].int_stim_trace[0][0:30]),axis=0)) 
    """
    #ROI_mod.calculateReliability(curr_rois)
    
    for roi in curr_rois:
        back_projected = np.tile(roi.i_stripe_resp, 
                                     (len(roi.i_stripe_resp),1) )
        roi.hor_RF_ON_trace = roi.i_stripe_resp
        roi.hor_RF_ON = back_projected
        roi.hor_RF_ON_gauss = roi.stripe_gauss_profile
        roi.hor_RF_ON_Rsq = roi.stripe_r_squared
        roi.hor_RF_ON_fwhm = roi.stripe_gauss_fwhm
        if (roi.hor_RF_ON_gauss is None) or \
            (roi.hor_RF_ON_Rsq <0.0): # filter based on gauss fit
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])
        flies_list.append(roi.experiment_info['MovieID'].split('_')[0])
        #flash_corr.append(roi.corr_fff)
        
        # Rel SNR
        #reliabilities.append(np.mean(roi.reliabilities))
        #SNRs.append(roi.SNR)
        
        # RF props
        #vert_RF_ON.append(roi.vert_RF_ON_trace)
        hor_RF_ON.append(roi.hor_RF_ON_trace)
        centered = np.full(np.shape(roi.hor_RF_ON_trace),np.nan)
        center = np.argmax(roi.hor_RF_ON_trace)
        center_idx = int((len(centered)-1)/2)
        if True:#  in exp_folder:
            if roi.i_stripe_pos[np.argmax(roi.hor_RF_ON_trace)] > 0:
                centered[:center_idx] = roi.hor_RF_ON_trace[center-center_idx:center]
                centered[center_idx:center_idx+len(centered)-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[center_idx-center:center_idx] = roi.hor_RF_ON_trace[:center]
                centered[center_idx:] = roi.hor_RF_ON_trace[center:center+center_idx+1]
        
        """if 'stripesH' in exp_folder:
            if center > 24:
                centered[:24] = roi.hor_RF_ON_trace[center-24:center]
                centered[24:24+48-center] = roi.hor_RF_ON_trace[center:]
            else:
                centered[24-center:24] = roi.hor_RF_ON_trace[:center]
                centered[24:] = roi.hor_RF_ON_trace[center:center+24]"""
        
        hor_RF_ON_centered.append(centered)
        
        
        hor_RF_ON_gauss.append(roi.hor_RF_ON_gauss)
        hor_RF_ON_gauss_fwhm.append(roi.hor_RF_ON_fwhm)
        hor_RF_ON_gauss_rsq.append(roi.hor_RF_ON_Rsq)
        # vert_RF_OFF.append(roi.vert_RF_OFF_trace)
        # hor_RF_OFF.append(roi.hor_RF_OFF_trace)
        
        rsq = np.array(roi.hor_RF_ON_Rsq)
        Rsq_vals.append(rsq)
        #SNRs.append(SNRs))
        
        
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))



unique_gens = np.unique(genotypes)
#%% Stripes analysis
fwhm = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_fwhm,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)
rsqd = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_rsq,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)

df = pd.DataFrame(list(zip(hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq,genotypes)),# SNRs)), 
               columns =['fwhm', 'rsq','gen'])#,'SNR']) 
#df['gen'][df['gen']=='Pos_Mi1Rec__plus_GluClflpSTOPD'] = 'Pos'
#df['gen'][df['gen']=='Exp_Mi1Rec__GluClflpTEVNI_GluClflpSTOPD'] = 'Exp'

colorss = pac.run_matplotlib_params()
colors = colorss[0]
plt.close('all')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('RF properties',fontsize=12)
grid = plt.GridSpec(len(unique_gens)*2,7 ,wspace=1, hspace=0.5)
ax1 = plt.subplot(grid[0,5:])
ax2 = plt.subplot(grid[1,5:])
labels = []
for idx , genotype in enumerate(unique_gens):
    labels.append(genotype)
    ax = plt.subplot(grid[idx,:5])
    sns.heatmap(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype],
                cmap='coolwarm',center=0,ax=ax,yticklabels=20,
                xticklabels=10,vmin=-0.5)#,vmax=4.5)
    
    
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('ROI #')
    
    # DEUBUG: plot all centered traces
    ax = plt.subplot(grid[idx+1,:5])
    lineplot_data = pd.DataFrame(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype])
    sns.lineplot(data=lineplot_data.transpose(), palette=sns.color_palette(['gray'], len(lineplot_data)), linewidth = 1, linestyle = 'solid')
    sns.lineplot(data=lineplot_data.mean(axis=0), color='red', linewidth = 1)
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('dF/F')
    ax.legend('')
    
    fwhm_data = fwhm['experiment_ids'][genotype]['all_samples']
    rsqd_data = rsqd['experiment_ids'][genotype]['all_samples']
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype,
                                           nflies =\
                                               #len(fwhm['experiment_ids'][genotype]['over_samples_means']),
                                               len(np.unique(flies_list)),
                                           nROIs=\
                                               len(fwhm['experiment_ids'][genotype]['all_samples']))
                                               
    ax.set_title(gen_str)      
    sns.distplot(rsqd_data,ax=ax2,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                 hist=False,bins=10)
    ax2.set_xlim(0.5, 1)

# sns.stripplot(x="gen", y="fwhm", hue="gen",palette=colors,data=df, 
#               dodge=True, jitter=True,order=labels,
#               alpha=.5,ax=ax1)
sns.violinplot(x='gen',y='fwhm',data=df,order=labels ,width=0.5,
                ax=ax1,palette=colors,inner="quartile",linewidth=1.5)
print('Median = '+str(np.median(fwhm_data)))
print('FWHM STDev = '+str(np.std(fwhm_data)))
ax1.set_ylim((0, 20)) # limit of y Axis FWHM Plot
ax1.set_xlabel('')
ax1.set_title('FWHM') 
ax1.set_ylabel('FWHM ($^\circ$)')
ax1.legend('')
ax1.set_xticklabels([''])
ax1.text(1,1,str(np.round(np.median(fwhm_data), decimals = 1))+'° IQR: '+str(np.round(scipy.stats.iqr(fwhm_data), decimals = 1))+'°',style='normal', transform=ax1.transAxes, horizontalalignment='right', verticalalignment='top')
ax2.set_title('R$^2$') 
ax2.set_xlabel('R$^2$')
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
    np.save(os.path.join(results_save_dir, exp_folder + '.npy'),
            [hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq, genotypes, SNRs])

#%%
import random
plt.close('all')
# Randomize ROIs
a=np.array(range(len(genotypes)))
random.shuffle(a)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams["axes.titlesize"] = 'x-small'
plt.rcParams["axes.labelsize"] = 'x-small'

fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
axs = ax1.flatten()
for idx, ax in enumerate(axs):
    curr_idx = a[idx]
    if genotypes[curr_idx] =='Exp':
        curr_color = colors[0]
    else:
        curr_color = colors[1]
    rsq1=np.around(hor_RF_ON_gauss_rsq[curr_idx],2)
    fwhm=np.around(hor_RF_ON_gauss_fwhm[curr_idx],1)
    ax.plot(hor_RF_ON_gauss[curr_idx], 'k',lw=2, 
            label="FWHM: {fwhm}$^\circ$ R$^2$ {rsq1}".format(fwhm=fwhm,rsq1=rsq1))
    ax.plot(hor_RF_ON[curr_idx],color=curr_color,lw=2)

    ax.legend()
fig1.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_exc_randomexp'.format(exp_t=exp_folder)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
plt.close('all')
#%%
sns.lmplot('rsq','fwhm',data=df,hue='gen',hue_order=labels,palette=colors,
           scatter_kws={'alpha':0.7, 's':20})
plt.ylim((0, 25))
plt.show()  

## generating trial average movie
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/200207_T4T5_luminance')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
summary_save_dir = os.path.join(alignedDataDir,
                                '_summaries')

# %% Parameters to adjust
current_exp_ID = '200212bg_fly4'
current_t_series = 'TSeries-02122020-1538-004'


#%%

dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
current_movie_ID = current_exp_ID + '-' + current_t_series

save_fig = True
try:
    movie_path = os.path.join(dataDir, '{t_name}_motCorr.tif'.format(t_name=current_t_series))
    raw_time_series = io.imread(movie_path)
except:
    movie_path = os.path.join(dataDir, 'motCorr.sima',
                              '{t_name}_motCorr.tif'.format(t_name=current_t_series))
    raw_time_series = io.imread(movie_path)
time_series = copy.deepcopy(raw_time_series)

## Get stimulus and xml information
(stimulus_information, trialCoor, frameRate, depth, x_size, y_size, pixelArea,
 stimType, stimOutFile, epochCount, stimName, layerPosition,
 stimInputFile, xmlFile, trialCount, isRandom, stimInputData,
 rawStimData) = \
    get_stim_xml_params(dataDir, stimInputDir)
stimulus_information['input_data'] = stimInputData
stimulus_information['stim_name'] = stimType.split('\\')[-1]
stimulus_information['trial_coordinates'] = trialCoor
if isRandom:
    stimulus_information['epoch_adjuster'] = 1
else:
    stimulus_information['epoch_adjuster'] = 0

imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                         'depth' : depth}
# Generate an average dataset for epoch visualization 
(wholeTraces_allTrials_video, respTraces_allTrials, 
     baselineTraces_allTrials) = \
        separate_trials_video(time_series, trialCoor,
                                               stimulus_information,
                                               imaging_information['frame_rate'])
generate_avg_movie(dataDir, stimulus_information,
                                     wholeTraces_allTrials_video)

def getMicRelativeTime(xmlFile):
    """ Gets the relative microscope times from the XML output.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    micRelTimes : ndarray
    """
    # @TODO check if tehre is a way to load the xml file as columns and rows
    # so that you can get rid of the for loop and just slice a column
    # get relative time points of the microscope xml file
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    micRelTimes = numpy.array([])
    for frame in root.iter('Frame'):
        timept = float(frame.attrib['relativeTime'])
        micRelTimes = numpy.append(micRelTimes, timept)

    return micRelTimes


def getFramePeriod(xmlFile):
    """Gets the frame period from the XML file.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    framePeriod : float
    """
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    framePeriod = None
    for stateVal in root.iter('PVStateValue'):
        if stateVal.get('key') == 'framePeriod':
            framePeriod = float(stateVal.get('value'))

    return framePeriod


def getPixelSize(xmlFile):
    """Gets the frame period from the XML file.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    framePeriod : float
    """
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    for stateVal in root.iter('PVStateValue'):
        if stateVal.get('key') == 'micronsPerPixel':
            for idxValues in stateVal:
                if idxValues.get('index') == "XAxis":
                    x_size = float(idxValues.get('value'))
                elif idxValues.get('index') == "YAxis":
                    y_size = float(idxValues.get('value'))
                    
    pixelArea = x_size*y_size # in micron square
                    
    return x_size, y_size, pixelArea



def getLayerPosition(xmlFile):
    """Gets the layer postions from the xml file.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    layerPosition : list
        Positions as [X, Y, Z]
    """
    # @TODO might want to find a shorter way without going into loops
    # but it is super fast even that way, so not a priority
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    # positions are [x,y,z], respectively
    layerPosition = [0, 0, 0]
    for stateVal in root.iter('PVStateValue'):
        if stateVal.get('key') == 'positionCurrent':
            for idxValues in stateVal:
                if idxValues.get('index') == "XAxis":
                    for subIdx in idxValues:
                        if subIdx.get('subindex') == '0':
                            layerPosition[0] = float(subIdx.get('value'))
                elif idxValues.get('index') == "YAxis":
                    for subIdx in idxValues:
                        if subIdx.get('subindex') == '0':
                            layerPosition[1] = float(subIdx.get('value'))
                elif idxValues.get('index') == "ZAxis":
                    for subIdx in idxValues:
                        if subIdx.get('subindex') == '0':
                            layerPosition[2] = float(subIdx.get('value'))

    return layerPosition

# CODE NOT FINISHED
def processMarkPoints(xmlFile):
    """Processes the mark points xml file.

    Parameters
    ==========
    xmlFile : str
        XML file path.

    Returns
    =======
    
    """
    tree = ET.parse(xmlFile)
    root = tree.getroot()

    for elem in root:  
        is_SLM = elem[0].attrib['AllPointsAtOnce']
        
        elem.attrib['Repetitions']
        
        # Getting the laser power via a proper conversion rate
        round(float(elem.attrib['UncagingLaserPower']) * 526.315)
        
        elem[0].attrib['InitialDelay']
        elem[0].attrib['Duration']
        elem[0].attrib['InterPointDelay']
        elem[0].attrib['SpiralRevolutions']
        elem[0].attrib['Indices'] # Which points are used
        
        elem[0][0].attrib['Index']
        elem[0][0].attrib['IsSpiral']
        elem[0][0].attrib['SpiralHeight']
        elem[0][0].attrib['SpiralWidth']
        elem[0][0].attrib['X']
        elem[0][0].attrib['Y']

    return None

def divideEpochs(rawStimData, epochCount, isRandom, framePeriod,
                 trialDiff=0.20, overlappingFrames=0, firstEpochIdx=0,
                 epochColumn=3, imgFrameColumn=7, incNextEpoch=True,
                 checkLastTrialLen=True):
    """
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    isRandom : int

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    overlappingFrames : int

    firstEpochIdx : int
        Default: 0

        Index of the first epoch.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    incNextEpoch :
    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [Z, D]] where first term is the trial beginning
        (first of first) and end (second of first), and second term is the
        baseline start (first of second) and end (second of second) for that
        trial.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).

    isRandom :
    """
    trialDiff = float(trialDiff)
    firstEpochIdx = int(firstEpochIdx)
    overlappingFrames = int(overlappingFrames)
    trialCoor = {}
    fullEpochSeq = []

    if isRandom == 0:
        fullEpochSeq = range(epochCount)
        # if the key is zero, that means isRandom is 0
        # this is for compatibitibility and
        # to make unified workflow with isRandom == 1
        trialCoor[0] = []

    elif isRandom == 1:
        # in this case fullEpochSeq is just a list of dummy values
        # important thing is its length
        # it's set to 3 since trials will be sth like: 0, X
        # if incNextEpoch is True, then it will be like : 0,X,0
        fullEpochSeq = range(2)
        for epoch in range(1, epochCount):
            # add epoch numbers to the dictionary
            # do not add the first epoch there
            # since it is not the exp epoch
            # instead it is used for baseline and inc coordinates
            trialCoor[epoch] = []

    if incNextEpoch:
        # add the first epoch
        fullEpochSeq.append(firstEpochIdx)
    elif not incNextEpoch:
        pass

    # min and max img frame numbers for each and every trial
    # first terms in frameBaselineCoor are the trial beginning and end
    # second terms are the baseline start and end for that trial
    currentEpochSeq = []
    frameBaselineCoor = [[0, 0], [0, 0]]
    nextMin = 0
    baselineMax = 0

    for line in rawStimData:
        if (len(currentEpochSeq) == 0 and
                len(currentEpochSeq) < len(fullEpochSeq)):
            # it means it is the very beginning of a trial block.
            # in the very first trial,
            # min frame coordinate cannot be set by nextMin.
            # this condition satisfies this purpose.
            currentEpochSeq.append(int(line[epochColumn]))
            if frameBaselineCoor[0][0] == 0:
                frameBaselineCoor[0][0] = int(line[imgFrameColumn])
                frameBaselineCoor[1][0] = int(line[imgFrameColumn])

        elif (len(currentEpochSeq) != 0 and
              len(currentEpochSeq) < len(fullEpochSeq)):
            # only update the current epoch list
            # already got the min coordinate of the trial
            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for the very first trial
                    frameBaselineCoor[1][1] = (int(line[imgFrameColumn])
                                               - overlappingFrames)

        elif len(currentEpochSeq) == len(fullEpochSeq):
            if nextMin == 0:
                nextMin = int(line[imgFrameColumn]) + overlappingFrames

            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for all the trials except the very first trial
                    baselineMax = int(line[imgFrameColumn]) - overlappingFrames

        else:
            frameBaselineCoor[0][1] = (int(line[imgFrameColumn])
                                       - overlappingFrames)

            if frameBaselineCoor[0][1] > 0:
                if isRandom == 0:
                    # if the key is zero, that means isRandom is 0
                    # this is for compatibitibility and
                    # to make unified workflow isRandom == 1
                    trialCoor[0].append(frameBaselineCoor)
                elif isRandom == 1:
                    # get the epoch number
                    # epoch no should be the 2nd term in currentEpochSeq
                    expEpoch = currentEpochSeq[1]
                    trialCoor[expEpoch].append(frameBaselineCoor)
            # this is just a safety check
            # towards the end of the file, the number of epochs might
            # not be enough to form a trial block
            # so if the max img frame coordinate is still 0
            # it means this not-so-complete trial will be discarded
            # only complete trials are appended to trial coordinates
            # if it has a max frame coord, it is safe to say
            # it had nextMin in frameBaselineCoor
            # print(currentEpochSeq)
            currentEpochSeq = []
            currentEpochSeq.append(firstEpochIdx)
            # each time currentEpochSeq resets means that
            # one trial block is complete
            # adding firstEpochIdx is necessary
            # otherwise currentEpochSeq will shift by 1
            # after every trial cycle
            # now that the frame coordinates are stored
            # can reset it
            # and add min coordinate for the next trial
            # then add the max baseline coordinate for the next trial
            frameBaselineCoor = [[0, 0], [0, 0]]
            frameBaselineCoor[0][0] = nextMin
            frameBaselineCoor[1][0] = nextMin
            frameBaselineCoor[1][1] = baselineMax
            nextMin = 0
            baselineMax = 0

    # @TODO: no need to separate isRandoms, make a unified for loop
    if checkLastTrialLen:
        if isRandom == 0:
            lenFirstTrial = trialCoor[0][0][0][1] - trialCoor[0][0][0][0]
            lenLastTrial = trialCoor[0][-1][0][1] - trialCoor[0][-1][0][0]
            if ((lenFirstTrial - lenLastTrial) * framePeriod) >= trialDiff:
                trialCoor[0].pop(-1)
                print("Last trial is discarded since the length was too short")

        elif isRandom == 1:
            for epoch in trialCoor:
                delSwitch = False
                lenFirstTrial = (trialCoor[epoch][0][0][1]
                                 - trialCoor[epoch][0][0][0])
                lenLastTrial = (trialCoor[epoch][-1][0][1]
                                - trialCoor[epoch][-1][0][0])

                if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                    delSwitch = True

                if delSwitch:
                    print("Last trial of epoch " + str(epoch)
                          + " is discarded since the length was too short")
                    trialCoor[epoch].pop(-1)

    trialCount = []
    if isRandom == 0:
        # there is only a single key in the trialCoor dict in this case
        trialCount.append(len(trialCoor[0]))
    elif isRandom == 1:
        epochTrial = []
        for epoch in trialCoor:
            epochTrial.append(len(trialCoor[epoch]))
        # in this case first element in trialCount is min no of trials
        # second element is the max no of trials
        trialCount.append(min(epochTrial))
        trialCount.append(max(epochTrial))

    return trialCoor, trialCount, isRandom

def divide_all_epochs(rawStimData, epochCount, framePeriod, trialDiff=0.20,
                      epochColumn=3, imgFrameColumn=7,checkLastTrialLen=True):
    """
    
    Finds all trial and epoch beginning and end frames
    
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [X, Y]] where X is the trial beginning and Y 
        is the trial end.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).
    """
    trialDiff = float(trialDiff)
    trialCoor = {}
    
    for epoch in range(0, epochCount):
        
        trialCoor[epoch] = []

    previous_epoch = []
    for line in rawStimData:
        
        current_epoch = int(line[epochColumn])
        
        if (not(previous_epoch == current_epoch )): # Beginning of a new epoch trial
            
            
            if (not(previous_epoch==[])): # If this is after stim start (which is normal case)
                epoch_trial_end_frame = previous_frame
                trialCoor[previous_epoch].append([[epoch_trial_start_frame, epoch_trial_end_frame], 
                                            [epoch_trial_start_frame, epoch_trial_end_frame]])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                previous_epoch = int(line[epochColumn])
                
            else:
                previous_epoch = int(line[epochColumn])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                
        previous_frame = int(line[imgFrameColumn])
        
    if checkLastTrialLen:
        for epoch in trialCoor:
            delSwitch = False
            lenFirstTrial = (trialCoor[epoch][0][0][1]
                             - trialCoor[epoch][0][0][0])
            lenLastTrial = (trialCoor[epoch][-1][0][1]
                            - trialCoor[epoch][-1][0][0])
    
            if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                delSwitch = True
    
            if delSwitch:
                print("Last trial of epoch " + str(epoch)
                      + " is discarded since the length was too short")
                trialCoor[epoch].pop(-1)
                
    trialCount = []
    epochTrial = []
    for epoch in trialCoor:
        epochTrial.append(len(trialCoor[epoch]))
    # in this case first element in trialCount is min no of trials
    # second element is the max no of trials
    trialCount.append(min(epochTrial))
    trialCount.append(max(epochTrial))
       

    return trialCoor, trialCount
def readStimOut(stimOutFile, skipHeader=1):
    """Read and get the stimulus output data.

    Parameters
    ==========
    stimOutFile : str
        Stimulus output file path.

    skipHeader : int, optional
        Default: 1

        Number of lines to be skipped from the beginning of the stimulus
        output file.

    Returns
    =======
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file.

    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.
    """
    # skip the first line since it is a file path
    rawStimData = numpy.genfromtxt(stimOutFile, dtype='float',
                                   skip_header=skipHeader)
    # also get the file path
    # do not mix it with numpy
    # only load and read the first line
    stimType = "stimType"
    with open(stimOutFile, 'r') as infile:
        lines_gen = islice(infile, 1)
        for line in lines_gen:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            line = re.sub(' ', '', line)
            stimType = line
            break

    return stimType, rawStimData

def readStimInformation(stimType, stimInputDir):
    """
    Parameters
    ==========
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file. *Required* if `gui` is FALSE and `stimInputDir`
        is given a non-default value.

    stimInputDir : str
        Path of the directory to look for the stimulus input.

    Returns
    =======
    stimInputFile : str
        Path to the stimulus input file (which contains stimulus parameters,
        not the output file).

    stimInputData : dict
        Lines of the stimulus generator file. Keys are the first terms of the
        line (the parameter names), values of the key is the rest of the line
        (a list).
    
    isRandom : int
        A that changes how epochs are randomized 

    Notes
    =====
    This function does not return the values below anymore:

    - epochDur : list
        Duration of the epochs as in the stimulus input file.


    - stimTransAmp : list

    """

    stimType = stimType.split('\\')[-1]
    stimInputFile = glob.glob(os.path.join(stimInputDir, stimType))[0]

    flHandle = open(stimInputFile, 'r')
    stimInputData = {}
    for line in flHandle:
        line = re.sub('\n', '', line)
        line = re.sub('\r', '', line)
        line = line.split('\t')
        stimInputData[line[0]] = line[1:]

#        epochDur = stimInputData['Stimulus.duration']
    
#        stimTransAmp = stimInputData['Stimulus.stimtrans.amp']
#
#        epochDur = [float(sec) for sec in epochDur]
#        stimTransAmp = [float(deg) for deg in stimTransAmp]

    return stimInputFile, stimInputData

def getEpochCount(rawStimData, epochColumn=3):
    """Get the total epoch number.

    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    Returns
    =======
    epochCount : int
        Total number of epochs.
    """
    # get the max epoch count from the rawStimData
    # 4th column is the epoch number
    # add plus 1 since the min epoch no is zero
    
    # BG edit: Changed the previous epoch extraction, which uses the maximum 
    # number + 1 as the epoch number, to a one finding the unique values and 
    # taking the length of it
    epochCount = np.shape(np.unique(rawStimData[:, epochColumn]))[0]
    print("Number of epochs = " + str(epochCount))

    return epochCount


def get_stim_xml_params(t_series_path, stimInputDir):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    t_series_path : str
        Path to the T series folder for retrieving stimulus related information and
        xml file which contains imaging parameters.
    
    stimInputDir : str
        Path to the folder where stimulus input information is located.
        
    Returns
    =======
    stimulus_information : list 
        Stimulus related information is stored here.
    trialCoor : list
        Start, end coordinates for each trial for each epoch
    frameRate : float
        Image acquisiton rate.
    depth :
        Z axis value of the imaging dataset.

    """

    # Finding the xml file and retrieving relevant information
    
    xmlPath = os.path.join(t_series_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    imagetimes = getMicRelativeTime(xmlFile)
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Stimulus output information
    
    stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile, 
                                          skipHeader=1)
    
    # Stimulus information
    (stimInputFile,stimInputData) = readStimInformation(stimType=stimType,
                                                      stimInputDir=stimInputDir)
    isRandom = int(stimInputData['Stimulus.randomize'][0])
    epochDur = stimInputData['Stimulus.duration']
    epochDur = [float(sec) for sec in epochDur]
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    # Finding epoch coordinates and number of trials, if isRandom is 1 then
    # there is a baseline epoch otherwise there is no baseline epoch even 
    # if isRandom = 2 (which randomizes all epochs)                                        
    if epochCount <= 1:
        trialCoor = 0
        trialCount = 0
    elif isRandom == 1:
        (trialCoor, trialCount, _) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=isRandom,
                                                 framePeriod=framePeriod,
                                                 trialDiff=0.20,
                                                 overlappingFrames=0,
                                                 firstEpochIdx=0,
                                                 epochColumn=3,
                                                 imgFrameColumn=7,
                                                 incNextEpoch=True,
                                                 checkLastTrialLen=True)
    else:
        (trialCoor, trialCount) = divide_all_epochs(rawStimData, epochCount, 
                                                    framePeriod, trialDiff=0.20,
                                                    epochColumn=3, imgFrameColumn=7,
                                                    checkLastTrialLen=True)
     
    stimulus_information ={}
    stimulus_data = stimInputData
    stimulus_information['epoch_dur'] = epochDur
    stimulus_information['random'] = isRandom
    stimulus_information['output_data'] = rawStimData
    stimulus_information['frame_timings'] = imagetimes
    if isRandom==0:
        stimulus_information['baseline_epoch'] = 0  
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus non random, baseline epoch selected as 0th epoch\n')
    elif isRandom == 2:
        stimulus_information['baseline_epoch'] = None
        stimulus_information['baseline_duration'] = None
        stimulus_information['epoch_adjuster'] = 0
        print('\n Stimulus all random, no baseline epoch present\n')
    elif isRandom == 1:
        stimulus_information['baseline_epoch'] = 0 
        stimulus_information['baseline_duration'] = \
            stimulus_information['epoch_dur'][stimulus_information['baseline_epoch']]
        stimulus_information['epoch_adjuster'] = 1
        
    stimulus_information['epoch_dir'] = \
            np.asfarray(stimulus_data['Stimulus.stimrot.mean'])
    epoch_speeds = np.asfarray(stimulus_data['Stimulus.stimtrans.mean'])
    stimulus_information['epoch_frequency'] = \
        epoch_speeds/np.asfarray(stimulus_data['Stimulus.spacing'])
    stimulus_information['epochs_duration'] =\
         np.asfarray(stimulus_data['Stimulus.duration'])
    stimulus_information['epoch_number'] =  \
        np.asfarray(stimulus_data['EPOCHS'][0])
    stimulus_information['stim_type'] =  \
        np.asfarray(stimulus_data['Stimulus.stimtype'])
    stimulus_information['input_data'] = stimInputData
    stimulus_information['stim_name'] = stimType.split('\\')[-1]
    stimulus_information['trial_coordinates'] = trialCoor
    
    imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth}
        
    return stimulus_information, imaging_information

def generate_avg_movie(dataDir, stimulus_information, 
                           wholeTraces_allTrials_video):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    dataDir: str
        Path into the directory where the motion corrected dataset with selected
        masks is present.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    wholeTraces_allTrials_video : list containing np arrays
        Epoch list of time traces including all trials in the form of:
        baseline epoch - stimulus epoch - baseline epoch
            
    
        
    Returns
    =======
    
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
    """
    print('Generating averaged movie...\n')
    # Directory for where to save the cluster movie
    selected_movie_dir = os.path.join(dataDir,'processed.sima')
    mov_xDim = np.shape(wholeTraces_allTrials_video[1])[1]
    mov_yDim = np.shape(wholeTraces_allTrials_video[1])[2]    
    
    epochs_to_use = range(len(stimulus_information['epoch_frequency']))
    if stimulus_information['random'] == 1:
        epochs_to_use = np.delete(epochs_to_use,stimulus_information['baseline_epoch'])
    epoch_frames = np.zeros(shape=np.shape(epochs_to_use))
    
    # Generating and filling the movie array movie array
    for index, epoch in enumerate(epochs_to_use):
        epoch_frames[index] = np.shape(wholeTraces_allTrials_video[epoch])[0]
        
    avg_movie = np.zeros(shape=(int(epoch_frames.sum()),1,mov_xDim,mov_yDim,1))
    
    startFrame = 0
    for index, epoch in enumerate(epochs_to_use):
        if index>0:
            startFrame =  endFrame 
        endFrame = startFrame + epoch_frames[index]
        avg_movie[int(startFrame):int(endFrame),0,:,:,0] = \
            wholeTraces_allTrials_video[epoch].mean(axis=3)
            
    
    # Create a sima dataset and export the cluster movie
    b = sima.Sequence.create('ndarray',avg_movie)
    average_dataset = sima.ImagingDataset([b],None)
    average_dataset.export_frames([[[os.path.join(selected_movie_dir,'avg_vid.tif')]]],
                                      fill_gaps=True,scale_values=True)
    print('Averaged movie generated...\n')
    
    return average_dataset

def separate_trials_video(time_series,stimulus_information,frameRate):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : list
        Start, end coordinates for each trial for each epoch
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    dff_baseline_dur_frame: int
        Duration of baseline before the stimulus for using in dF/F calculation.
        
    Returns
    =======
    wholeTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-

    """
    
    trialCoor = stimulus_information['trial_coordinates']
    mov_xDim = time_series.shape[1]
    mov_yDim = time_series.shape[2]
    wholeTraces_allTrials = {}
    respTraces_allTrials = {}
    baselineTraces_allTrials = {}
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            
            base_lens.append(baselineEnd - baselineStart) 
     
            
        trial_len =  min(trial_lens)-4
        resp_len = int(round(frameRate * current_epoch_dur))+1
#        resp_len = int(round(frameRate * 3))
        base_len = min(base_lens)
        
        if stimulus_information['random']==2:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = None
            baselineTraces_allTrials[iEpoch] = None
            
        elif stimulus_information['random'] == 1:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = np.zeros(shape=(resp_len,mov_xDim,mov_yDim,
                               trial_numbers))
            baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                               trial_numbers))
        else:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            base_len  = np.shape(wholeTraces_allTrials\
                                 [stimulus_information['baseline_epoch']])[0]
            baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                               trial_numbers))
                
        
        for trial_num , current_trial_coor in enumerate(currentEpoch):
            
            if stimulus_information['random']==2:
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                raw_signal = time_series[trialStart:trialEnd, : , :]
                
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
               
            elif stimulus_information['random'] == 1:
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                
                baselineStart = current_trial_coor[1][0]
                baselineEnd = current_trial_coor[1][1]
                
                respStart = current_trial_coor[1][1]
                epochEnd = current_trial_coor[0][1]
                
                raw_signal = time_series[trialStart:trialEnd, : , :]
            
                currentResp = time_series[respStart:epochEnd, : , :]
                   #        dffTraces_allTrials[iEpoch].append(dFF[:trial_len,:,:])
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                respTraces_allTrials[iEpoch][:,:,:,trial_num]= currentResp[:resp_len,:,:]
                baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:base_len,:,:]
            else:
                
                # If the sequence is non random  the trials are just separated without any baseline
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                if iEpoch == stimulus_information['baseline_epoch']:
                    baseline_signal = time_series[trialStart:trialEnd, : , :]
                raw_signal = time_series[trialStart:trialEnd, : , :]
                
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                respTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= baseline_signal[:base_len,:,:]
            
           
            
            
 
        
        print('Epoch %d completed \n' % iEpoch)
        
    return (wholeTraces_allTrials, respTraces_allTrials, baselineTraces_allTrials)