""""script to pool _ROI.pkl data according to conditions to compare"""
import sys, pickle, preprocessing_params, os, shutil
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
###############################
if len(preprocessing_params.condition_columns)!= len(set(preprocessing_params.condition_columns)) or len(preprocessing_params.condition_columns)==0:
    open(error_log, 'a', encoding="utf8").write(f'Please define multiple, different metadata for data pooling')
    open(error_log, 'a', encoding="utf8").write('\n')
else:
    for fly in os.listdir(paths.sort):
        if fly.endswith('.pkl') or fly=='stim':
            continue
        df_meta_fly = df_meta[df_meta["fly"]==fly]
        for tseries in os.listdir(f'{paths.sort}/{fly}'):
            source = f'{paths.sort}/{fly}/{tseries}'
            if tseries.startswith("TSeries"):
                number=tseries.split('-')[-1]
                if f'Tseries-{number}' not in tseries_list:
                    open(error_log, 'a', encoding="utf8").write(f'{fly}/{tseries} : no metadata found')
                    open(error_log, 'a', encoding="utf8").write('\n')
                    continue
                else:
                    df_meta_one = df_meta_fly[df_meta_fly['TSeries']==f'Tseries-{number}']
                    tseries_con = core_pre.get_condition_tseries(df_meta_one, preprocessing_params.condition_columns)
                    destination = f'{paths.raw}/{tseries_con}/{fly}/{tseries}'
                    os.makedirs(destination, exist_ok=True)
                    if len(os.listdir(source))>0:
                        for file in os.listdir(source):
                            shutil.move(f'{source}/{file}', f'{destination}/{file}')
                    shutil.rmtree(source)
            elif tseries.startswith('ZSeries'):
                destination = f'{paths.zstacks}/{fly}/{tseries}'
                for file in os.listdir(source):
                    shutil.move(f'{source}/{file}', f'{destination}/{file}')
                shutil.rmtree(source)
            elif tseries.startswith('SingleImage'):
                shutil.rmtree(source)
            elif tseries.startswith('BrightnessOverTime'):
                shutil.rmtree(source)
        if len(os.listdir(f'{paths.sort}/{fly}'))==0:
            shutil.rmtree(f'{paths.sort}/{fly}')