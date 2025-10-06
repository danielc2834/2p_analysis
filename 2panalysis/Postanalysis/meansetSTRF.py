import numpy as np
from scipy import stats,signal
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from scipy import ndimage
import cPickle
#code_path = r'D:\progresources\2panalysis\Helpers'
code_path = r'C:\Users\vargasju\PhD\scripts_and_related\github\IIIcurrentIII2p_analysis_development\2panalysis\Helpers' #Juan desktop
sys.path.insert(0, code_path) 
import ROI_mod
import post_analysis_core as pac

#initialize Sppatiotemporalrf class


class spatio_temporal_set:
    """this class assumes that STRFs are already filtered """

    def __init__(self,treatment,polarity,luminance):
        #self.sumSTRF = self._align_firstSTRF(STRF)
        self.treatment = treatment 
        self.luminance = luminance
        self.STRFs = []
        self.FlyId = []
        self.treatment = []
        self.sex = []
        self.counter = 0
        self.polarity = polarity
        #self.RFcenter = np.where(STRF=np.max(STRF))
        

    def normalize_STRF(self):
        "calculate Zscore of the spatiotemporal view of the RF"
        return (self.new_STRF-np.mean(self.new_STRF))/ np.std(self.new_STRF)

    def append_STRF(self,roi):
        """takes in a new STRF instance and sums it to whatever is the averageSTRF"""
        new_STRF = roi.STRF_data['SxT_representation_MaxRespDir']
        new_STRF = np.where(np.isnan(new_STRF),0,new_STRF) # get rid of nans to avoid conflicts later on when summing
        self.STRFs.append(new_STRF)
        self.counter += 1
        self.new_STRF = new_STRF
        self.new_STRF = self.normalize_STRF()
        self.new_STRF = self._align_STRF(roi)
        #self.sumSTRF += new_STRF
        #self.average = self.sumSTRF/self.counter
        self.FlyId.append(roi.experiment_info['FlyID'])
        self.treatment.append(roi.experiment_info['treatment'])
        self.sex.append(roi.experiment_info['Sex'])

    def _align_firstSTRF(self,STRF,roi):
        """find location of absolute max and roll to the middle of the space axis"""
        #if roi.CS == 'ON':
        init_index = np.where(np.abs(STRF) == np.max(np.abs(STRF[:,2*STRF.shape[1]//3:])))[0][0]
        #else:
        #    init_index = np.where(STRF == np.min(STRF[:,2*STRF.shape[1]//3:]))[0][0]
        roll_index = self.new_STRF.shape[0]//2 - init_index
        self.sumSTRF = ndimage.shift(self.new_STRF,[roll_index,0],mode = 'constant', cval = 0)
        
        if roll_index<0:
            self.sumSTRF[roll_index:,:] = 0
        elif roll_index>0:
            self.sumSTRF[:roll_index,:] = 0

    def _align_STRF(self,roi):
        
        """find point of maximum correlation between sumSTRF and new STRF roll to maximize correlation"""
        try: 
            self.sumSTRF
        except: 
            self._align_firstSTRF(self.new_STRF,roi)
            return None
        # compare the sizes of new and sum:
        size_comparison = self.new_STRF.shape[0] - self.sumSTRF.shape[0] # if negative sum is bigger than new
        add_on = np.zeros((np.abs(size_comparison),np.shape(self.new_STRF)[1]))
        
        if size_comparison < 0:
            # increase size of new
            self.new_STRF = np.concatenate([add_on,self.new_STRF],axis=0)
        elif size_comparison > 0:
            # increase size of sum
            self.sumSTRF = np.concatenate([add_on,self.sumSTRF],axis=0)         

        # if roi.CS == 'ON':
        #     rect_sum = np.where(self.sumSTRF>0,self.sumSTRF,0)   
        #     rect_new = np.where(self.new_STRF>0,self.sumSTRF,0)   
        # else:
        #     rect_sum = np.where(self.sumSTRF>0,0,self.sumSTRF)   
        #     rect_new = np.where(self.new_STRF>0,0,self.sumSTRF)  

        # if roi.CS == 'ON':
        #     init_index = np.where(self.new_STRF == np.max(self.new_STRF[:,2*self.new_STRF.shape[1]//3:]))[1][0]
        # else:
        #     init_index = np.where(self.new_STRF == np.min(self.new_STRF[:,2*self.new_STRF.shape[1]//3:]))[1][0]
        
        # roll_index = self.new_STRF.shape[0]//2 - init_index
        # self.sumSTRF += np.roll(self.new_STRF,roll_index,axis = 0)
        
        #crosscor = signal.correlate2d(rect_sum[:,self.sumSTRF.shape[1]//2:],rect_new[:,self.new_STRF.shape[1]//2:],mode='full') 

        crosscor = signal.correlate2d(self.new_STRF[:,2*self.new_STRF.shape[1]//3:],self.sumSTRF[:,2*self.sumSTRF.shape[1]//3:],mode='same') 
        vertical_corr = np.mean(crosscor,axis=1)
        
        lag = np.argmax(vertical_corr)
        shift = lag - (len(vertical_corr)//2)
        #self.sumSTRF += np.roll(self.new_STRF,shift,axis = 0)
        self.sumSTRF += ndimage.shift(self.new_STRF,[-shift,0],mode = 'constant', cval = 0)
        # realign to put max val in the middle
        if roi.CS == 'ON':
            init_index = np.where(self.sumSTRF == np.max(self.sumSTRF[:,2*self.sumSTRF.shape[1]//3:]))[0][0]
        else:
            init_index = np.where(self.sumSTRF == np.min(self.sumSTRF[:,2*self.sumSTRF.shape[1]//3:]))[0][0]
        roll_index = self.sumSTRF.shape[0]//2 - init_index
        #self.sumSTRF = ndimage.shift(self.sumSTRF,[roll_index,0],mode = 'constant', cval = 0)
        self.sumSTRF = np.roll(self.sumSTRF,roll_index,axis=0)


    def mean_SxT(self):
        self.average = self.sumSTRF/self.counter
        return self.average

    def update_on_disk(self,name,savedir):
        # every time this function is called the class instance is saved to disk
        name = name + '.pkl'
        with open(os.path.join(savedir,name), 'wb') as file:
            cPickle.dump(self, file, protocol=2)

def plot_spatiotempRF_dataset(dataset,genotypes,treatments,savedir,polarities = ['ON','OFF']):

    # plots the mean spatiotemporal RFs from datasets

    for polarity in polarities:
        fig = plt.figure()
        gs = GridSpec((3,4))
        for g, genotype in enumerate(genotypes):
            for t,treatment in enumerate(treatments):

                strfset = dataset[genotype][treatment][polarity]
                ax = fig.add_subplot(gs[t,g])
                im = plt.imshow(strfset.mean_SxT())
                cbar = fig.colorbar(im,ax=ax)
    
    pac.multipage(savedir +'\\mean_2dRFrepresentations')
