from __future__ import division
import numpy as np
import glob
import pandas as pd
import sima
import pathlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from skimage import io
import copy
import cPickle
#from preprocessing import produce_Tseries_list # eventually, transfer this function to process_mov_core script
import sys
#code_path = r'D:\progresources\2panalysis\Helpers'
code_path= r'U:\Dokumente\GitHub\IIIcurrentIII2p_analysis_development\2panalysis\Helpers' #Juan desktop
sys.path.insert(0, code_path) 

from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize,getMicRelativeTime,getrastersPerFrame
import ROI_mod
import core_functions
from core_functions import saveWorkspace
import process_mov_core as pmc

#%%
# def calculate_df_f_spont_activity(traces):
#     """
#     calculates df/f using a running window average, which in this case is implemented 
#     with a convolution with a flat kernel
#     """
    
#     #traces_mean=np.mean(traces,axis=1)
#     kernel=np.ones(3000,dtype=int)/3000
#     traces_mean=np.transpose(ndimage.convolve1d(traces,kernel,axis=1,mode='reflect'))
#     traces=np.divide(traces-np.transpose(traces_mean),np.transpose(traces_mean))
#     return traces

# def calculateDf(self,method='mean',moving_avg = False, bins = 3):
#     try:
#         self.raw_trace
#     except NameError:
#         raise NameError('ROI_bg: for deltaF calculations, a raw trace \
#                         needs to be provided: a.raw_trace')
        
#     if method=='mean':
#         df_trace = (self.raw_trace-self.raw_trace.mean(axis=0))/(self.raw_trace.mean(axis=0))
#         self.baseline_method = method
#     if method=='convolution':
#         trace=copy.deepcopy(self.raw_trace)
#         window=len(trace)/3
#         kernel=np.ones(window,dtype=int)/window
#         traces_mean=np.transpose(ndimage.convolve1d(trace,kernel,axis=1,mode='reflect'))
#         self.df_trace=np.divide(traces-np.transpose(traces_mean),np.transpose(traces_mean))
        
#     if method=='postpone': #hold on the df calculation to do it based on stimulus
#         df_trace=self.raw_trace
#         self.baseline_method = method        
#     if moving_avg:
#         self.df_trace = movingaverage(df_trace, bins)
#     else:
#         self.df_trace = df_trace
        
#     return self.df_trace
#%%
save_data=True
home_path='C:\\Users\\vargasju\\PhD\\experiments\\2p\\' #desktop office Juan
experiment=  ['r57c10_Rgeco_panneuronal silencing']#['Miris_data_piece']
experimenter=['_jv_']#['*_jv_*']
extraction_type = 'load_manual' #'load_manual''SIMA-STICA' 'transfer' 'manual'  'load-STICA' 'cluster analysis Henning_20'
analysis_params = {'deltaF_method': 'convolution',   
                   'analysis_type': None,
                   'Background':'manual'} 

for iidx,exp in enumerate(experiment):
    experimenter_code=experimenter[iidx]
    Tseries_paths=core_functions.produce_Tseries_list(home_path,exp,experimenter_code)
    meta_exists=len(glob.glob(home_path+'\\'+exp+'\\'+'processed\\'+'metadata_fixed.csv'))>0
    #Tseries_paths=[Tseries_paths[0]]
    if meta_exists:
        pass
    else:
         core_functions.extract_metadata(exp,home_path)
         raise Exception('%s_Metadata_file_extracted, needs curation'%(exp))
    acumm_roi_list=[]
    for xx,dataDir in enumerate(Tseries_paths): # in other scripts dataDir is equivalent to Tser
        print('Tser %s / %s' %(xx,len(Tseries_paths)))
        print(dataDir)
        metadata_df=core_functions.extract_fly_metadata(dataDir)   
        try:
            current_exp_ID = metadata_df['date'][0]
        except IndexError:
            print('Tser has no metadata entry. skipping')         
            continue
        #TODO generalize the experiment_condition_extraction
        current_t_series =metadata_df['recording'][0]        
        Genotype = metadata_df['genotype'][0]
        age_when_treated = metadata_df['age when treated'][0]
        age_when_recorded =  metadata_df['age when recorded'][0]
        Sex = metadata_df['sex'][0]
        treatment = metadata_df['treatment'][0]
        #z_depth_bin=metadata_df['depth_bin'][0]
        z_depth=metadata_df['depth'][0]
        hatched=metadata_df['hatched'][0]
        continued_developing= metadata_df['continued developing'][0]
        time_after_treatment=metadata_df['time after treatment'][0]
        treatment_duration=metadata_df['treatment duration(h)'][0]
        experiment_conditions= \
        {'Genotype' : Genotype, 'age when treated': age_when_treated,'age when recorded':age_when_recorded, 'Sex' : Sex, 'treatment':treatment,
        'FlyID' : current_exp_ID,'expected_polarity':None, 'MovieID':  current_t_series,
        'z_depth':z_depth,'analysis_type':'spontaneous_activity','hatched':hatched,
        'continued developing':continued_developing,'time after treatment':time_after_treatment,
        'treatment_duration':treatment_duration}
        
        """extract imaging info"""

        imaging_information=pmc.extract_xml_info(dataDir)

        """# load the time series in a cycle by cycle basis"""
        number_of_cycles,lenghtCycles=core_functions.find_cycles(dataDir)
        mean_image, time_series = core_functions.load_Tseries(dataDir,number_of_cycles,lenghtCycles) 


        """create output folders if they don't exist"""
        saveOutputDir=home_path +'\\'  + exp +'\\' + 'processed\\'+ current_exp_ID + '\\' + current_t_series
        try:
            os.mkdir(home_path +'\\'  + exp +'\\' + 'processed\\'+ current_exp_ID)
            os.mkdir(home_path +'\\'  + exp +'\\' + 'processed\\'+ current_exp_ID + '\\' + current_t_series)
        except:
            pass
        finally:
            pass
        summary_save_dir= home_path  + exp +'\\' + 'summary\\'+ current_exp_ID + '\\' + current_t_series
        figure_save_dir = home_path + exp +'\\' + 'summary\\'+ current_exp_ID + '\\' + current_t_series + '\\figures'
        if not os.path.exists(figure_save_dir):
            try:
                os.mkdir(home_path +'\\'  + exp +'\\' + 'summary\\'+ current_exp_ID)
            except:
                pass
            finally:
                pass
            try:
                os.mkdir(home_path +'\\'  + exp +'\\' + 'summary\\'+ current_exp_ID + '\\' + current_t_series)
                os.mkdir(home_path +'\\'  + exp +'\\' + 'summary\\'+ current_exp_ID + '\\' + current_t_series + '\\figures')
            except:
                pass
            finally:
                pass
        """Organizing extraction parameters"""
        extraction_params = \
        pmc.organize_extraction_params(extraction_type,
                               current_t_series=current_t_series,
                               current_exp_ID=current_exp_ID,
                               stimInputDir=None,
                               use_other_series_roiExtraction = False,
                               use_avg_data_for_roi_extract = False,
                               roiExtraction_tseries=None,
                               transfer_data_n = None,
                               transfer_data_store_dir = saveOutputDir,
                               analysis_type = 'spont',
                               imaging_information=imaging_information,
                               experiment_conditions=experiment_conditions,
                               stimuli=None,
                               ) #JUan edited parameters to include more info about multiple cycles and more selection parameters
        """ roi extraction"""

        if extraction_type == 'load_manual':
            bg_mask,cat_names,dataset,roi_masks,all_clusters_image=pmc.import_clusters(dataDir,manual=True,FFT_filtered=False)

        """
        Generate ROI_bg instances
        """

        if extraction_type == 'load_manual':
            rois = ROI_mod.generate_ROI_instances_categorized(roi_masks, cat_names,
                                                mean_image, 
                                                experiment_info = experiment_conditions, 
                                                imaging_info =imaging_information)
       
        # We can store the parameters inside the objects for further use
        for roi in rois:
            roi.extraction_params = extraction_params 
            if extraction_type == 'transfer': # Update transferred ROIs
                roi.experiment_info = experiment_conditions
                roi.imaging_info = imaging_information
                for param in analysis_params.keys():
                    roi.analysis_params[param] = analysis_params[param]
            else:
                roi.analysis_params= analysis_params #TODO check df/f method!!
       
        """procesing"""
       
        currentTseries=time_series[0]
        current_movie_ID= current_exp_ID + '_' + current_t_series  # + '_cycle_'+ str(cycle+1)
        #experiment_conditions= experiment_conditions
        figure_save_dir_local=figure_save_dir
        figure_save_dir_local=figure_save_dir_local #+ '\\cycle_0' + str(cycle+1) +'\\'
        try:
            os.mkdir(figure_save_dir_local)
        except:
            pass
        finally:
            pass
        mean_curr_image = currentTseries.mean(0)
        current_movie_ID = current_exp_ID + '-' + current_t_series #+'-' + 'cycle_' + str(cycle+1) 
        print(current_movie_ID)
        """
        Normalization
        """
        currentTseries = currentTseries.astype(float)/float(np.max(currentTseries))
        """
        Background subtraction
        """
        currentTseries_sub =  currentTseries-np.mean(currentTseries*bg_mask[np.newaxis,:,:],axis=(1,2))[:,np.newaxis,np.newaxis]
        print('\n Background subtraction done...')
        #bg_trace=np.mean(currentTseries*bg_mask[np.newaxis,:,:],axis=(1,2))
        """ subtract background and calculate df/f"""
        #TODO extract the raw trace!!
        if dataDir=='C:\\Users\\vargasju\\PhD\\experiments\\2p\\r57c10_Rgeco_panneuronal silencing\\raw\\20211006_jv_fly4\\TSeries-001\\':
            a='a'
        for roi in rois:
            roi.raw_trace = np.nanmean(currentTseries_sub*roi.mask[np.newaxis,:,:],axis=(1,2))#-bg_trace
        map(lambda roi: roi.calculateDf(method='convolution'), rois)


        """ perform some basic analysis?""" ###TODO adapt to stimless. what to calculate?
        # final_rois = pmc.run_analysis(analysis_params,rois,experiment_conditions,
        #                             imaging_information,summary_save_dir,
        #                             save_fig=True,fig_save_dir = figure_save_dir,
        #                             exp_ID=('%s_%s' % (current_movie_ID,extraction_params['type'])),
        #                             df_method=analysis_params['deltaF_method'], keep_prev=None)
        if save_data:
            save_dict={'final_rois':rois}
            save_name = os.path.join(saveOutputDir,'{ID}.pickle'.format(ID=current_movie_ID))
            try:
                os.mkdir(saveOutputDir)
            except:
                pass
            finally:
                pass
            saveVar = open(save_name, "wb")
            cPickle.dump(save_dict, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
            saveVar.close()
            print('\n\n%s saved...\n\n' % save_name)
        
        for roi in rois:
            acumm_roi_list.append(roi)
    
    ### TODO function for plotting traces ##
        #if Tseries is from same fly, skip 
        #load pickle files
    data_dict={'flyID':[],'trace':[],'TSeries':[],'treatment':[],'fps':[]}
    for roi in acumm_roi_list:
        data_dict['flyID'].append(roi.experiment_info['FlyID'])
        data_dict['trace'].append(roi.df_trace)
        data_dict['TSeries'].append(roi.experiment_info['MovieID'])
        data_dict['treatment'].append(roi.experiment_info['treatment'])
        data_dict['fps'].append(roi.imaging_info['frame_rate'])
    #create empty plot
    data_dict=pd.DataFrame(data_dict)
    #fig=plt.figure()
    # find ceiling of treatments and create gridspec that would contain all treatments
    treatments_unique=np.unique(data_dict['treatment'])
    #grid0=gridspec.GridSpec(1,len(treatments_unique),figure=fig) 

    for ix0,treatment in enumerate(np.unique(data_dict['treatment'])):
        data_subset=data_dict.loc[data_dict['treatment']==treatment]
        flies=len(np.unique(data_subset['flyID']))
        ceil=int(np.ceil(flies/2))
        # create gridspec inside gridspec. find ceiling of flies for this treatment
        # use the ceiling to fill up 
        #grid00=gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=grid0[ix0])
        fig=plt.figure()
        grid0=gridspec.GridSpec(2,2,figure=fig,wspace=0.5,hspace=0.5)
        
        for ix1,fly in enumerate(np.unique(data_subset['flyID'])):
            count=0
            ax=plt.subplot(grid0[ix1])
            plt.title(treatment)
            fly_subset=data_subset.loc[data_subset['flyID']==fly]
            x_axis=np.divide(((1/fly_subset['fps'].values[0])*np.array(range(len(fly_subset['trace'].values[0])))),60)

            for ix2,row in fly_subset.iterrows():            
                plt.plot(x_axis,row['trace']+(count),color='darkslateblue')
                ax.set_xticks([10,20,30])
                ax.set_ylim([-0.50,2.2])
                ax.set_yticks([0,1,2])
                ax.set_xlim([0,40])
                ax.set_ylabel(r'df_f')
                ax.set_xlabel('t (minutes)')

                #TODO set ylims, make xticks
                count+=1

    print('plots done')
            # create subset of flies per treatment
            # make a plot containing all traces per fly, together
    
            #figure out if you need any quantification
    
    #save the plot you just did



    






