
import sys, os, shutil, math
import core_preprocessing as core_pre
import pandas as pd
################################
my_input = sys.argv
dataset_folder, error_log, metadata_path = my_input[1], my_input[2], my_input[3]
################################
paths = core_pre.dataset(dataset_folder)
df_meta = pd.read_excel(metadata_path, sheet_name=paths.name) 
for fly in os.listdir(paths.sort):
    if fly.endswith(".pkl") or fly == 'stim':
        continue  
    ctime = {}
    for stimfile in os.listdir(f'{paths.stim}/{fly}'):
        time=os.path.getmtime(f'{paths.stim}/{fly}/{stimfile}')
        ctime[stimfile] = time
    ctime = dict(sorted(ctime.items()))
    ctime_tseries = {}
    meta_fly = df_meta[df_meta["fly"]==fly]
    for tseries in os.listdir(f'{paths.sort}/{fly}'):
        if tseries.startswith("TSeries"):
            if os.path.exists(f'{paths.sort}/{fly}/{tseries}/{tseries}_stim_output.txt')==True and  os.path.exists(f'{paths.sort}/{fly}/{tseries}/{tseries}_meta_data.txt')==True:
                continue
            if os.path.exists(f'{paths.stim}/{fly}') == False:
                open(error_log, 'a', encoding="utf8").write(f'{fly}: no stimuli files provided > check /0_to_sort/stim')
                open(error_log, 'a', encoding="utf8").write('\n')
                continue
            number=tseries.split('-')[-1]
            # if 'olf_stim_type' in list(meta_fly[meta_fly.TSeries==f'Tseries-{number}'].columns):
            #     olf_stim = meta_fly[meta_fly.TSeries==f'Tseries-{number}'].olf_stim_type.values[0]
            #     if olf_stim not in preprocessing_params.olfactory_stimuli.keys():
            #         open(error_log, 'a', encoding="utf8").write(f'{fly}: please define {olf_stim} > check preprocessing_params.olfactory_stimuli')
            #         open(error_log, 'a', encoding="utf8").write('\n')
            #         continue
            #     elif os.path.exists(f'{paths.processed}/{fly}/{tseries}/_olf_stim.png') == True:
            #         continue
            #     else:
            #         if len(preprocessing_params.experiment)>0:
            #             target = f'{paths.processed}/{fly}/{tseries}/{preprocessing_params.experiment}'
            #         else:
            #             target = f'{paths.processed}/{fly}/{tseries}/'
            #         with open(f'{target}_ROIS.pkl', 'rb') as fi:
            #             rois_pkl = pickle.load(fi)
            #         fps = rois_pkl[list(rois_pkl.keys())[0]].get('frame_rate')
            #         olf_stim_array = core_pre.olf_stim_array(olf_stim, error_log, fps, target)
            #         rois_pkl[olf_stim] = olf_stim_array
            #         with open(f'{target}_ROIS.pkl', 'wb') as fo:
            #             pickle.dump(rois_pkl, fo)
            visual_stim = meta_fly[meta_fly.TSeries==f'Tseries-{number}'].visual_stim.values[0]
            if isinstance(visual_stim, str) == False:
                if math.isnan(visual_stim) == True:
                    continue
                else:
                    open(error_log, 'a', encoding="utf8").write(f'{fly}/{tseries} : please define visual stimulus, None if none was presented')
                    open(error_log, 'a', encoding="utf8").write('\n')
                    continue
            else:
                time=os.path.getctime(f'{paths.sort}/{fly}/{tseries}/{tseries}.xml')
                ctime_tseries[tseries] = time
    ctime_tseries = dict(sorted(ctime_tseries.items()), reversed=True)
    if len(ctime_tseries) == 1 or len(ctime)==0:
        continue
    elif (len(ctime_tseries)-1)*2 == len(ctime):
        counter=0
        for tseries in ctime_tseries:
            if tseries.startswith('TSeries'):
                shutil.copy(f'{paths.stim}/{fly}/{list(ctime.items())[counter][0]}', f'{paths.sort}/{fly}/{tseries}/{tseries}_meta_data.txt')
                shutil.copy(f'{paths.stim}/{fly}/{list(ctime.items())[counter+1][0]}', f'{paths.sort}/{fly}/{tseries}/{tseries}_stim_output.txt')
                counter+=2
        shutil.rmtree(f'{paths.stim}/{fly}')
    else:
        open(error_log, 'a', encoding="utf8").write(f'{fly} : some stimulus files are missing in {paths.stim}/{fly}')
        open(error_log, 'a', encoding="utf8").write('\n')
            
