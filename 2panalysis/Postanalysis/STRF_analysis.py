# script with a protocol for STRF analysis 
# use python 3 here


import os
from time import time
from tkinter import Grid
import cPickle
import STRF_utils as RF
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import NMF
code_path= r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis\helpers' #Juan desktop
sys.path.insert(0, code_path) 
import post_analysis_core as pac
import process_mov_core as pmc
import ROI_mod
from matplotlib.gridspec import GridSpec
from meansetSTRF import spatio_temporal_set

#initialize 
initialDirectory = 'C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan

experiment = 'T4T5_STRF_glucla_rescues'
data_dir = initialDirectory+experiment+'\\'+'processed\\files'
results_save_dir = initialDirectory+experiment+'\\'+'processed\\results\\STRF'
checkpoint_dir = initialDirectory+experiment+'\\'+'processed\\results\\STRF_checkpoints'
z_scores_dir = initialDirectory+experiment+'\\'+'processed\\results\\Z_scores'
ind_RF_dir = initialDirectory+experiment+'\\'+'processed\\results\\ind_RFs'
summary_dir = os.path.abspath('C:\\Users\\vargasju\\PhD\\experiments\\2p\\T4T5_STRF_glucla_rescues\\RFs')
stimulus_dir = os.path.abspath('C:\\Users\\vargasju\\PhD\\experiments\\2p\\T4T5_STRF_glucla_rescues\\stimuli_arrays')
plot_mapping_hist = False
calculate_av_SxT = True
simple_response_prediction = False
genotypes = ['control_homocygous']#['Tm3_Homocygous'] #['control_heterozygous', 'control_homocygous','L5rescue',  'Mi1rescue']
luminances_treats = ['lum_1'] #['lum_0.1','lum_0.25','lum_1'] 
restrictions = ['spatial_restriction','temporal_restriction','unrestricted']
stim_type = '3max'
colors=[(195.0/256.0,170.0/256.0,109.0/256.0),(150.0/256.0,150.0/256.0,150.0/256.0),(89.0/256.0,125.0/256.0,191.0/256.0),(100.0/256.0,100.0/256.0,100.0/256.0)]
regularization = [1]#[0.0001,0.01,0.1,1]
# import rois
datasets_to_load = os.listdir(data_dir)
curr_rois = []
final_rois = []
mapping_rois = []
included_count = 0
excluded_count = 0


#SxT_dict = {'SxTRF':[],'luminance':[],'genotype':[]}

#%% Load stim

stimulus = RF.load_stimulus(stimulus_dir,stim_type)
stimulus = np.flip(stimulus, axis = 2)

#%%

# create summary information for these ROIs 



if plot_mapping_hist:

    for dataset in datasets_to_load:

        #TODO check treatment
        if not(".pickle" in dataset):
            print('Skipping non pickle file: {d}'.format(d=dataset))
            continue
        load_path = os.path.join(data_dir, dataset)
        load_path = open(load_path, 'rb')
        workspace = cPickle.load(load_path)
        
        if 'DriftingStripe' in dataset:
            mapping_rois.append(workspace['final_rois'])    
        else:
            continue

    mapping_rois=np.concatenate(mapping_rois)

    # produce some general plots to describe the roi extraction 

    map_Df=pac.create_dataframe(mapping_rois,independent_vars=None,mapping=True)

    pac.plot_variable_histogram_pertreatment(map_Df,'reliability_ON',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'reliability_OFF',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'DSI_OFF',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'DSI_ON',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors,filter_CS=True)
    pac.plot_variable_histogram_pertreatment(map_Df,'CSI',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors,filter_CS=True)
    # check if any differences in DSI between treatments are due to the reliability differences (less reliable ROIs)
    filtered_map_df_ON = map_Df.loc[map_Df['CS'] == 'ON']
    filtered_map_df_ON = map_Df.loc[map_Df['reliability_ON'] >= 0.4]
    #pac.plot_variable_histogram_pertreatment(map_Df,'DSI_OFF',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors)
    #pac.plot_variable_histogram_pertreatment(filtered_map_df_ON,'DSI_ON',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors,treshold=0.4)
    #pac.plot_variable_histogram_pertreatment(filtered_map_df_ON,'CSI',results_save_dir,fit_Beta=[False,[]],treatment_names=genotypes,colors=colors,treshold=0.4)

    filename=results_save_dir + 'ROI_distributions.pdf'
    pac.multipage(filename)
    plt.close('all')

#%%

# use an sxt dictionary to produce average SxT receptive fields (mean average spatiotemporal representations).first, initialize




# for unique in treatment and genotype create a class instance and create dictionaries to store tha values of pearson sorrelation results for the response predictions 

if calculate_av_SxT:

    mean_SxT,correlation_prediction_train,correlation_prediction_test = RF.create_corrDF_scaffold(restrictions,genotypes,luminances_treats)


    for dataset in datasets_to_load:
        final_rois = []
        if 'DriftingStripe' in dataset:
            continue    
        #TODO check treatment
        if not(".pickle" in dataset):
            print('Skipping non pickle file: {d}'.format(d=dataset))
            continue
        load_path = os.path.join(data_dir, dataset)
        load_path = open(load_path, 'rb')
        workspace = cPickle.load(load_path)
        print('working on Mean spatiotemporal RFs %s'%(workspace['final_rois'][0].experiment_info['FlyID']))
        print('%s rois'%(len(workspace['final_rois'])))
        for ij,roi in enumerate(workspace['final_rois']):
            

            #curr_rois.append(workspace['final_rois'])

            if roi.STRF_data['status'] == 'excluded':
                excluded_count+=1
                continue

            scaling_factor = 80.0/(roi.STRF_data['strf'].shape[1])
            if roi.CSI < 0.5:
                excluded_count+=1
                continue
            if np.sum(~np.isnan(np.nanmean(roi.STRF_data['SxT_representation_MaxRespDir'],axis=1))) <  30/scaling_factor:
                # if the receptive field is too close to the edge
                # exclude. the RF should be at least 30deg wide         
                # consider instead excluding based on distance to edge   
                excluded_count+=1
                continue
            #if roi.experiment_info['treatment'] != 'L5rescue':
            #    continue
            # if roi.STRF_data['status'] == 'included':
            final_rois.append(roi)
            included_count+=1


        #genotypes = []
        try:
            curr_genot = final_rois[0].experiment_info['treatment']
        except:
            continue
        for ix,roi in enumerate(final_rois):

            ###temporal
            # if ix != 1:
            #     continue
            if ij % 10 == 0:
                print(ix)
            if roi.experiment_info['treatment'] == 'L5rescue' and roi.CS=='ON': #and roi.CSI>0.6:
                 'flag'
            #    print(roi.CSI)
            if 'lum01' in roi.stim_info['stim_name']:
                lum_treatment = 'lum_0.1'
                
            elif 'lum025' in roi.stim_info['stim_name']:
                lum_treatment = 'lum_0.25'
            else:
                lum_treatment = 'lum_1'        

            correlation_values_train = {}
            correlation_values_train[lum_treatment] = {}
            correlation_values_test = {}
            correlation_values_test[lum_treatment] = {}

            # collect the spatio-temporal view of the RF

            mean_SxT[curr_genot][lum_treatment]['ON' if roi.CS == 'ON' else 'OFF'].append_STRF(roi)
        
        # save analysis checkpoint
        
        mean_SxT[curr_genot][lum_treatment]['ON'].update_on_disk('spatiotempset_%s_%s_ON'%(curr_genot,lum_treatment),checkpoint_dir)
        mean_SxT[curr_genot][lum_treatment]['OFF'].update_on_disk('spatiotempset_%s_%s_OFF'%(curr_genot,lum_treatment),checkpoint_dir)

    # plot average spatiotemporal representations and all STRFs for genotype/lum/pol

    for pol in ['ON','OFF']:
        fig = plt.figure(figsize=(20,20))
        fig_ind = plt.figure(figsize=(20,20))
        gs = GridSpec(3,4)
        gs_ind = GridSpec(6,6)
        for i,genotype in enumerate(genotypes):
            for j,lum in enumerate(luminances_treats):
                ax = fig.add_subplot(gs[j,i])
                im_ = np.repeat(mean_SxT[genotype][lum][pol].sumSTRF,4,axis=1)
                im_ = mean_SxT[genotype][lum][pol].sumSTRF
                if pol == 'ON':
                    init_index = np.where(im_ == np.max(im_))[1][0]
                else:
                    init_index = np.where(im_ == np.min(im_))[1][0]
                
                # roll_index = im_.shape[0]//2 - init_index
                # im_ = np.roll(im_,roll_index,axis = 0)
                
                max_= np.max(np.abs(im_))
                ax.imshow(im_, cmap='PRGn',vmax=max_,vmin=-max_)
                ax.title.set_text('%s %s %s'%(genotype,lum,pol))
                ax.axis('off')
                
                if len(mean_SxT[genotype][lum][pol].STRFs)>36:
                    np.random.seed(2)
                    selection = np.random.choice(range(len(mean_SxT[genotype][lum][pol].STRFs)),36)
                    list_strfs = np.array(mean_SxT[genotype][lum][pol].STRFs)[selection]
                else:
                    list_strfs = mean_SxT[genotype][lum][pol].STRFs
                for ix,individual_RF in enumerate(list_strfs):
                    ax2 = fig_ind.add_subplot(gs_ind[ix])
                    ax2.axis('off')
                    max_ = np.max(np.abs(individual_RF))
                    im_2 = np.repeat(individual_RF,3,axis=1)
                    ax2.imshow(im_2, cmap='PRGn',vmax=max_,vmin=-max_)
                    fig_ind.suptitle('ind RFs %s %s %s'%(genotype,lum,pol))
                    fig_ind.savefig(ind_RF_dir +'\\ind_RFs %s %s %s.pdf'%(genotype,lum,pol),)
                plt.close(fig_ind)

    filename=results_save_dir + 'Average_2d_RF.pdf'
    pac.multipage(filename)        



# plot the mean Spatiotemporal views per treatment condition 

# extract temporal and spatial kernels from the strf estimation, having this 2 kernels will allow 
# to predict responses

if simple_response_prediction:

    for dataset in datasets_to_load:
        final_rois = []
        if 'DriftingStripe' in dataset:
            continue    
        #TODO check treatment
        if not(".pickle" in dataset):
            print('Skipping non pickle file: {d}'.format(d=dataset))
            continue
        load_path = os.path.join(data_dir, dataset)
        load_path = open(load_path, 'rb')
        workspace = cPickle.load(load_path)
        print('working on Mean spatiotemporal RFs %s'%(workspace['final_rois'][0].experiment_info['FlyID']))
        print('%s rois'%(len(workspace['final_rois'])))
        roi_test_correlations = []
        roi_train_correlations = []
        for ij,roi in enumerate(workspace['final_rois']):
            
            if ij < 2:
                continue

            #curr_rois.append(workspace['final_rois'])

            if roi.STRF_data['status'] == 'excluded':
                excluded_count+=1
                continue

            scaling_factor = 80.0/(roi.STRF_data['strf'].shape[1])
            if roi.CSI < 0.6:
                excluded_count+=1
                continue
            if np.sum(~np.isnan(np.nanmean(roi.STRF_data['SxT_representation_MaxRespDir'],axis=1))) <  30/scaling_factor:
                # if the receptive field is too close to the edge
                # exclude. the RF should be at least 30deg wide         
                # consider instead excluding based on distance to edge   
                excluded_count+=1
                continue
            #if roi.experiment_info['treatment'] != 'L5rescue':
            #    continue
            # if roi.STRF_data['status'] == 'included':
            final_rois.append(roi)
            included_count+=1

            # extract temporal kernel
            try:
                os.mkdir(z_scores_dir + '\\%s'%(roi.experiment_info['FlyID']))
            except:
                pass    
            
            save_loc = z_scores_dir + '\\%s\\roi_%s_%s.pdf'%(roi.experiment_info['FlyID'],ij,roi.CS)
            filter_valid = RF.extract_temporal_kernel(roi,save_loc)

            included = 0
            excluded = 0
            if filter_valid == 'valid':
                included += 1
            else:
                excluded += 1

            
            RF.predict_response_w_kernels(roi,stimulus,type_='train')
            #RF.predict_response_w_kernels(roi,stimulus,type_='test')
            

            roi_train_correlations.append(roi.STRF_prediction_analysis['kernel_pred']['corr_train'][0])
            roi_test_correlations.append(roi.STRF_prediction_analysis['kernel_pred']['corr_test'][0])
            

            #!!! do a multicomponent kernel estimation

            H_xy, W_t = RF.segregated_NMf(roi,30,regularization = 1 )
            RF.plot_NMF_segregated_components(roi,H_xy,W_t,results_save_dir)
            plt.close('all')
            # take first components and do a prediction with them


            #del roi.STRF_data['strf']
        # plot a distribution of correlation results

        fig_hist = plt.figure()
        GSpec = GridSpec(1,2)
        ax1 = fig_hist.add_subplot(GSpec[0])
        ax2 = fig_hist.add_subplot(GSpec[1])
        ax1.hist(roi_train_correlations)
        ax2.hist(roi_test_correlations)
        # save dataset without RF estimation

        RF_analysis_dir = data_dir #+'\\reduced RFs'

        save_dict={'final_rois':workspace['final_rois']}
        save_name = os.path.join(RF_analysis_dir, dataset)
        if os.path.exists(RF_analysis_dir)==False:
            try:
                os.mkdir(RF_analysis_dir)
            except:
                pass
            finally:
                pass
        saveVar = open(save_name, "wb")
        cPickle.dump(save_dict, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
        saveVar.close()
        print('\n\n%s saved...\n\n' % save_name)

        # plot a distribution of correlation results


#pepe
#%% linear response prediction

# for dataset in os.listdir(RF_analysis_dir):
#     final_rois = []
#     if 'DriftingStripe' in dataset:
#         continue    
#     #TODO check treatment
#     if not(".pickle" in dataset):
#         print('Skipping non pickle file: {d}'.format(d=dataset))
#         continue
#     load_path = os.path.join(data_dir, dataset)
#     load_path = open(load_path, 'rb')
#     workspace = cPickle.load(load_path)
#     print('working on Mean spatiotemporal RFs %s'%(workspace['final_rois'][0].experiment_info['FlyID']))
#     print('%s rois'%(len(workspace['final_rois'])))
#     for ij,roi in enumerate(workspace['final_rois']):
        

#         #curr_rois.append(workspace['final_rois'])

#         if roi.STRF_data['status'] == 'excluded':
#             excluded_count+=1
#             continue

#         scaling_factor = 80.0/(roi.STRF_data['strf'].shape[1])
#         if roi.CSI < 0.6:
#             excluded_count+=1
#             continue
#         if np.sum(~np.isnan(np.nanmean(roi.STRF_data['SxT_representation_MaxRespDir'],axis=1))) <  30/scaling_factor:
#             # if the receptive field is too close to the edge
#             # exclude. the RF should be at least 30deg wide         
#             # consider instead excluding based on distance to edge   
#             excluded_count+=1
#             continue
#         #if roi.experiment_info['treatment'] != 'L5rescue':
#         #    continue
#         # if roi.STRF_data['status'] == 'included':
#         final_rois.append(roi)
#         included_count+=1

#         # use the temporal and spatial kernels to predict the responses of the neurons
    
#         RF.predict_response_w_kernels(roi)

#%% plot distributions of correlations from predictions

#%% extract multiple kernels from components

for dataset in datasets_to_load:
    
    if 'DriftingStripe' in dataset:
        continue    
    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    
    
    for roi in workspace['final_rois']:
        #curr_rois.append(workspace['final_rois'])
        if roi.STRF_data['status'] == 'included':
            final_rois.append(roi)
            included_count+=1
        else:
            excluded_count+=1

    genotypes = []

    for ix,roi in enumerate(final_rois):

        ###temporal
        # if ix != 1:
        #     continue

        if 'lum01' in roi.stim_info['stim_name']:
            lum_treatment = 'lum_0.1'
            
        elif 'lum025' in roi.stim_info['stim_name']:
            lum_treatment = 'lum_0.25'
        else:
            lum_treatment = 'lum_1'        

        correlation_values_train = {}
        correlation_values_train[lum_treatment] = {}
        correlation_values_test = {}
        correlation_values_test[lum_treatment] = {}

        # collect the spatio-temporal view of the RF

        #mean_SxT[roi.experiment_info['treatment']][lum_treatment]['ON' if roi.CS == 'ON' else 'OFF'].append_STRF(roi)

        # calculate train and test correlations of predictions for the train and test data

        for local_restriction in ['unrestricted']:#,'spatial_restriction','temporal_restriction']:

            corr_train,corr_train_relu = RF.predict_signal_correlation(roi,stimulus,type_ = 'test',t_window = 3,restriction=local_restriction)
            corr_test, corr_test_relu = RF.predict_signal_correlation(roi,stimulus,type_ = 'train',t_window = 3,restriction=local_restriction)

            correlation_prediction_train[roi.experiment_info['treatment']][lum_treatment]['ON' if roi.CS == 'ON' else 'OFF'][local_restriction]['None'] = corr_train
            correlation_prediction_train[roi.experiment_info['treatment']][lum_treatment]['ON' if roi.CS == 'ON' else 'OFF'][local_restriction]['Relu'] = corr_train_relu
            
            correlation_prediction_test[roi.experiment_info['treatment']][lum_treatment]['ON' if roi.CS == 'ON' else 'OFF'][local_restriction]['None'] = corr_test
            correlation_prediction_test[roi.experiment_info['treatment']][lum_treatment]['ON' if roi.CS == 'ON' else 'OFF'][local_restriction]['Relu'] = corr_test_relu


            plt.figure()
            plt.plot(roi.white_noise_response)
            plt.plot(roi.STRF_prediction_analysis[local_restriction]['strf_prediction_test_relu'])
            plt.plot(roi.STRF_prediction_analysis[local_restriction]['strf_prediction_train_relu'])
            plt.title('%s strf +relu corr %s'%(local_restriction,corr_train[0]))

            plt.figure()
            plt.plot(roi.white_noise_response)
            plt.plot(roi.STRF_prediction_analysis[local_restriction]['strf_prediction_test'])
            plt.plot(roi.STRF_prediction_analysis[local_restriction]['strf_prediction_train'])        
            plt.title('%s strf corr %s'%(local_restriction,corr_train[0]))



        filename=results_save_dir + 'ROI%s_predictions.pdf' %(ix)
        pac.multipage(filename)
        plt.close('all')

        current_treatment = roi.experiment_info['treatment']
        
# plot the distributions of pearson correlation values according to treatment and lum_treatment 

for lum in ['lum_0.1','lum_0.25','lum_1']:
    fig_train = plt.figure()
    grid_train = GridSpec (len(genotypes),1)
    
    fig_test = plt.figure()
    grid_test = GridSpec(len(genotypes),1)

    for ix,genotype in enumerate(genotypes):
        
        ax_test = grid_test[ix]
        ax_train = grid_train[ix]
        corr_vals_train = correlation_values_train[lum][genotype]
        corr_vals_test = correlation_values_test[lum][genotype]

        ax_test.hist(corr_vals_test,bins=np.linspace(0,20,1),histtype=u'step',density=True,label=genotype,color='black',lw=2)
        ax_train.hist(corr_vals_train,bins=np.linspace(0,20,1),histtype=u'step',density=True,label=genotype,color='grey',lw=2)
        
        fig_test.suptitle('signal-prediction corr. test set\n %s'(lum))
        fig_train.suptitle('signal-prediction corr. train set\n %s'(lum))
                
# produce the average spatiotemporal representations

fig_on = plt.figure()
gs_on = GridSpec((3,3))

fig_off = plt.figure()
gs_off = GridSpec((3,3))

for i,genotype in enumerate(np.unique(mean_SxT['genotype'])):
        for j,lum_treat in enumerate(np.unique(mean_SxT['luminance'])):
            for polarity in ['ON','OFF']:
                local_mean = mean_SxT[genotype][lum_treat][polarity].mean_SxT()
                #plot the results 
                
                if polarity == "ON":
                    ax = plt.subplot(gs_on[i,j])
                    im = plt.imshow(local_mean)
                else:
                    ax = plt.subplot(gs_on[i,j])
                    im = plt.imshow(local_mean)





#%% try to fit gaussians to the subfields of the receptive fields and try to make a LN model for every ROI

for dataset in datasets_to_load:
    
    if 'DriftingStripe' in dataset:
        continue    
    #TODO check treatment
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)

    for ix,roi in enumerate(final_rois):
        if 'lum01' in roi.stim_info['stim_name']:
            lum_treatment = '01'
        elif 'lum025' in roi.stim_info['stim_name']:
            lum_treatment = '025'
        else:
            lum_treatment = '1'
        
        for reg in regularization:

            masked_array = RF.apply_circular_mask(roi,30)
            masked_array = masked_array[masked_array.shape[0]//2:,:,:] # to do component extraction, focus in the 1,5 sec preceding the signal
            components_xy, components_t = RF.svd_flattenedSTRF(roi,masked_array,components=5)

            # perform NMF on the negative and positive components separately

            H_xy, W_t = RF.segregated_NMf(roi,masked_array,regularization = reg)
            
            #plot the 3 components

            RF.plot_spatial_temporal_components(components_xy,components_t,results_save_dir,type='mixed_svd')
            RF.plot_NMF_segregated_components(roi,H_xy,W_t,results_save_dir)

            #savefigs

            fly_path = os.path.join(results_save_dir,'components',roi.experiment_info['FlyID'])
            try:
                os.mkdir(fly_path)
            except:
                pass
            
            res_path = os.path.join(fly_path,'reg_%s' %(reg))
            try:
                os.mkdir(res_path)
            except:
                pass

            lum_path = os.path.join(res_path,'lum_%s' %(lum_treatment))
            try:
                os.mkdir(lum_path)
            except:
                pass
            
            pmc.multipage(lum_path + '\\components_%s.pdf' %(ix))
            plt.close('all')

            # fit gaussians to the NMF components
            # redo the spatitemporal RF with the absolute value

            # RF.produce_spatiotemporal_view(roi,summary_dir,ix)

            #fit gaussians to the NMF components 
            RF.fit_double_gaussian2D(roi, mode='spatio_temp')
            RF.fit_double_gaussian2D(roi, mode='spatial')






# pepe##
# curr_rois=np.concatenate(curr_rois)

# # filter rois for reliability and contrast selectivity 
# for roi in curr_rois:
#     if roi.STRF_data['status'] == 'included':
#         final_rois.append(roi)
#         included_count+=1
#     else:
#         excluded_count+=1



# create distributions of Zscores and correlation prediction for null control and experimnetal data 
    # to do this, use control flies

STRF_Df = RF.create_dataframe_forSTRF(final_rois,independent_vars=None,mapping=False)


STRF_DF_controls1 = STRF_Df.loc[STRF_Df['treatment'] == 'control_heterozygous']
STRF_DF_controls2 = STRF_Df.loc[STRF_Df['treatment'] == 'control_homocygous']

STRF_DF_controls = pd.concat([STRF_DF_controls1,STRF_DF_controls2],ignore_index=True)

#RF.calculate_STRF_prediction_distributions(STRF_DF_controls)


# for ix,roi in enumerate(final_rois):
#     # calculate singular value decomposition for every roi calculated STRF
#     # first fallten the strf

#     # eliminate features beyond 30 deg distance from the absolute maximum of the RF
#     if 'lum01' in roi.stim_info['stim_name']:
#         lum_treatment = '01'
#     elif 'lum025' in roi.stim_info['stim_name']:
#         lum_treatment = '025'
#     else:
#         lum_treatment = '1'
    # for reg in regularization:

    #     masked_array = RF.apply_circular_mask(roi,25)
        
    #     components_xy, components_t = RF.svd_flattenedSTRF(roi,masked_array,components=5)

    #     # perform NMF on the negative and positive components separately

    #     H_xy, W_t = RF.segregated_NMf(roi,masked_array,regularization = reg)
        
    #     #plot the 3 components

    #     RF.plot_spatial_temporal_components(components_xy,components_t,results_save_dir,type='mixed_svd')

    #     RF.plot_NMF_segregated_components(roi,H_xy,W_t,results_save_dir)

    #     #savefigs

    #     fly_path = os.path.join(results_save_dir,'components',roi.experiment_info['FlyID'])
    #     try:
    #         os.mkdir(fly_path)
    #     except:
    #         pass
        
    #     res_path = os.path.join(fly_path,'reg_%s' %(reg))
    #     try:
    #         os.mkdir(res_path)
    #     except:
    #         pass

    #     lum_path = os.path.join(res_path,'lum_%s' %(lum_treatment))
    #     try:
    #         os.mkdir(lum_path)
    #     except:
    #         pass
        
    #     pmc.multipage(lum_path + '\\components_%s.pdf' %(ix))
    #     plt.close('all')


    # redo the spatiotemporal mapping ussing the max value in the last0.5 seconds



    # maybe calculate NNMF with 3 components
    # to do that split the negative and positive correlation areas


    # fit gaussians to the 2 or 3 components

    # RF.fit_component_gaussians(components_xy)

    # extract the temporal components (maybe the time aspects of the svd can help?)
    # ---- otherwise extract the filter from the point of maximum absolute value

    # do the stimulus * space * time convolution 

    # fit a softplus nonlinearity. 

    # evaluate predictions

    # Use the fitted gaussians 

    # fit gaussians to the first 2 or 3 components of the single value decomposition or the NNMF 





# create thresholds for Zscores/reliability,etc and trace prediction correlation 

 
# try to fit a gaussian for spatio temporal and for the spatial meanRF

# for roi in final_rois:
#     RF.fit_double_gaussian2D(roi, mode='spatio_temp')
#     RF.fit_double_gaussian2D(roi, mode='spatial')

# compare sizes of RF acrosstreatment/luminance/polarity
# for this, use the extracted sizes of the receptive fieldsaccording to the gaussian fit
