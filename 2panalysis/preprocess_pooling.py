""""script to pool _ROI.pkl data according to conditions to compare"""
import sys, pickle, preprocessing_params, os
import core_preprocessing as core_pre
import pandas as pd
################################
my_input = sys.argv
dataset_folder, error_log, metadata_path = my_input[1], my_input[2], my_input[3]
paths = core_pre.dataset(dataset_folder)
df_meta = pd.read_excel(metadata_path, sheet_name=paths.name) 
tseries_list = df_meta['TSeries'].tolist()
with open(f'{paths.processed}/processing_progress.pkl', 'rb') as fi:
    processing_progress = pickle.load(fi)
name = os.path.basename(__file__)
open(error_log, 'a', encoding="utf8").write(f'\nscript used: {name}\n')
###############################
if len(preprocessing_params.experiment)>0:
    target = f'{paths.data}/{preprocessing_params.experiment}'
else:
    target = f'{paths.data}/'
if len(preprocessing_params.condition_columns)!= len(set(preprocessing_params.condition_columns)) or len(preprocessing_params.condition_columns)==0:
    open(error_log, 'a', encoding="utf8").write(f'Please define multiple, different metadata for data pooling')
    open(error_log, 'a', encoding="utf8").write('\n')
else:
    for tseries_con in os.listdir(paths.processed):
        if tseries_con.endswith('_h') or tseries_con.endswith('_v'):
            continue
        if tseries_con.endswith('.pkl'):
            continue
        for fly in os.listdir(f'{paths.processed}/{tseries_con}'):
            if fly.endswith('.pkl') or fly.endswith('.png'):
                continue
            df_meta_fly = df_meta[df_meta["fly"]==fly]
            for tseries in os.listdir(f'{paths.processed}/{tseries_con}/{fly}'):
                if tseries.startswith("TSeries"):
                    number=tseries.split('-')[-1]
                    number = number[-1] if number [-2] == "0" else number[-2:]
                    # if f'Tseries-{number}' not in tseries_list:
                    if int(number) not in tseries_list:
                        open(error_log, 'a', encoding="utf8").write(f'{fly}/{tseries} : no metadata found')
                        open(error_log, 'a', encoding="utf8").write('\n')
                        continue
                    else:
                        # df_meta_one = df_meta_fly[df_meta_fly['TSeries']==f'Tseries-{number}']
                        df_meta_one = df_meta_fly[df_meta_fly['TSeries']==int(number)]
                        # tseries_con = core_pre.get_condition_tseries(df_meta_one, preprocessing_params.condition_columns)
                        if os.path.exists(f'{target}{tseries_con}.pkl') == False:
                            with open(f"{target}{tseries_con}.pkl", 'wb') as fo:
                                pickle.dump({}, fo)
                        elif os.path.exists(f'{target}{tseries_con}.pkl') == True and processing_progress[tseries_con][fly].get(tseries) == True:
                            with open(f'{target}{tseries_con}.pkl', 'rb') as fi:
                                condition_pkl = pickle.load(fi)
                            if tseries not in condition_pkl.keys():
                                processing_progress[tseries_con][fly][tseries] = [True,True,True,False]
    for tseries_con in os.listdir(paths.processed):
        if tseries_con.endswith('_h') or tseries_con.endswith('_v'):
            continue
        if tseries_con.endswith('.pkl'):
            continue
        for fly in os.listdir(f'{paths.processed}/{tseries_con}'):
            if fly.endswith('.pkl') or fly.endswith('.png'):
                continue
            df_meta_fly = df_meta[df_meta["fly"]==fly]
            for tseries in os.listdir(f'{paths.processed}/{tseries_con}/{fly}'):
                if tseries.startswith('TSeries'):
                    number=tseries.split('-')[-1]
                    number = number[-1] if number [-2] == "0" else number[-2:]
                    # if f'Tseries-{number}' not in tseries_list:
                    if int(number) not in tseries_list:
                        open(error_log, 'a', encoding="utf8").write(f'{fly}/{tseries} : no metadata found')
                        open(error_log, 'a', encoding="utf8").write('\n')
                        continue
                    elif  processing_progress[tseries_con][fly].get(tseries)[3]==False:
                        if len(preprocessing_params.experiment)>0:
                            processed_files = f'{paths.processed}/{tseries_con}/{fly}/{tseries}/{preprocessing_params.experiment}'
                        else:
                            processed_files = f'{paths.processed}/{tseries_con}/{fly}/{tseries}/'
                        # df_meta_one = df_meta_fly[df_meta_fly['TSeries']==f'Tseries-{number}']
                        df_meta_one = df_meta_fly[df_meta_fly['TSeries']==int(number)]
                        tseries_con = core_pre.get_condition_tseries(df_meta_one, preprocessing_params.condition_columns)
                        with open(f'{target}{tseries_con}.pkl', 'rb') as fi:
                            condition_pkl = pickle.load(fi)
                        with open(f'{processed_files}_ROIS.pkl', 'rb') as fii:
                            tseries_pkl = pickle.load(fii)
                        condition_pkl[tseries] = tseries_pkl
                        with open(f'{target}{tseries_con}.pkl', 'wb') as fo:
                            pickle.dump(condition_pkl, fo)
                        processing_progress[tseries_con][fly][tseries] = [True,True,True,True]
                    else:
                        # print(f'{fly}/{tseries} : Data already pooled')
                        continue
with open(f'{paths.processed}/processing_progress.pkl', 'wb') as fo:
    pickle.dump(processing_progress, fo)