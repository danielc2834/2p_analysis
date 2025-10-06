""""script to count flies and rois per condition and cat"""
import core_preprocessing as core_pre
import pickle, os
import pandas as pd
################################
dataset_folder = r'C:\Users\Admin\Desktop\Neuer Ordner'
paths = core_pre.dataset(dataset_folder)
###############################
all = pd.DataFrame(columns=['condition', 'neuropile', 'flies', "rois", 'odor'])
m=0
for condition in os.listdir(paths.data):
    
    # counting = pd.DataFrame({})
    if condition.endswith('_all.pkl'):
        continue
    with open(f'{paths.data}/{condition}', 'rb') as fi:
        condition_datafile = pickle.load(fi)
    odor_str = condition.split('_')[-3]
    # fly_n=0
    tseries_list = set(list(condition_datafile.keys()))
    while 'alligned_fps' in tseries_list:
        tseries_list.remove('alligned_fps')
    fly_n = len(tseries_list)
    if condition.endswith('_LH.pkl'):
        fly_n = int(len(tseries_list)/2)
        n=0
        for tseries in tseries_list:
            for roi in condition_datafile[tseries]:
                if roi == 'pulse':
                    continue
                else:
                    n+=1
        all.loc[m] = [condition, 'LH', fly_n, int(n/2), odor_str]
        m+=1
    # for tseries in condition_datafile:
    #     if tseries == 'alligned_fps':
    #         continue
    else:
        
        for pile in ['ME', 'LO', 'LOP']:
            n=0
            for tseries in condition_datafile:
                if tseries == 'alligned_fps':
                    continue
                for cat in condition_datafile[tseries]:
                    if cat == pile:
                        n+=1
            all.loc[m] = [condition, pile, fly_n, n, odor_str]
            m+=1
all.to_csv(f'{paths.results}/n.txt', sep='\t', index=False)
    # for key in condition_datafile:
        
        # key_pd = pd.DataFrame({key:n})