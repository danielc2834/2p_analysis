'''Pipeline to Batch Process Calcium Imaging Recordings'''
#TODO:
#bleedtrough corection with caiman
################################################imports
import subprocess, os, time, pickle, GUI_preprocessing
import core_preprocessing as core_pre
from datetime import date
################################################
dataset_folder,motion_alignment,select_rois,metadata_path = GUI_preprocessing.dataset_path,GUI_preprocessing.motion,GUI_preprocessing.roi,GUI_preprocessing.log_path
paths = core_pre.dataset(dataset_folder)
################################################
start = time.time()
# df_meta = pd.read_excel(metadata_path, sheet_name=paths.name) 
today = date.today()
error_log = f"{paths.folder}/preprocessing_error_log_{today}.txt"
error_log_counter=1
while os.path.exists(error_log) == True:
    error_log = f"{paths.folder}/preprocessing_error_log_{today}_{error_log_counter}.txt"
    error_log_counter+=1
lines = [f'Dataset: {paths.name}', f'date: {today}', '','ERROR:','']
with open(error_log, 'w', encoding="utf-8") as f:
    for line in lines:
        f.write(line)
        f.write('\n')
print('organizing datastructures')
core_pre.check_folder_structure(paths.folder)
with open(f'{paths.processed}/processing_progress.pkl', 'rb') as fi:
    processing_progress = pickle.load(fi)
# print('renaming files')
core_pre.rename_Tseries(paths.sort)
#gets stim for each tseries from stim folder before moving to condition folderb
print('moving stimulus files')
#no olf_stim anymore >  put into main
subprocess.run(['python', "2panalysis/preprocess_addingStim.py", paths.folder, error_log, metadata_path])
print('moving to condition folders')
subprocess.run(['python', "2panalysis/preprocess_condition_folders.py", paths.folder, error_log, metadata_path])
if motion_alignment == 0:
    print("aligning motion")
    subprocess.run(['python', "2panalysis/preprocess_motion.py", paths.folder, error_log])
if select_rois == 0:
    print('selecting & processing ROIs')
    subprocess.run(['python', "2panalysis/preprocess_rois.py", paths.folder, error_log, metadata_path])
# pooling to make DATA for analysis_odor_responses, and raw_traces >> TODO:change if time
print('pooling DATA')
#TODO: pooling progress to fals eif not in the file
subprocess.run(['python', "2panalysis/preprocess_pooling.py", paths.folder, error_log, metadata_path])
end = time.time()
runtime = end - start 
h = int((runtime/60)/60)
min = int((runtime/60)-(h*60))
sec = int(runtime)-(min*60)-(h*60*60)
lines = ['',f'total runtime: h:{h} min:{min} sec:{sec}']
with open(error_log, 'a', encoding="utf-8") as f:
    for line in lines:
        f.write(line)
        f.write('\n')     
        

'''current rois: [dictionary]
if olfactory stim was used:
olf_stim : np.array
for each category:
    for each roi:
        .raw_trace : raw imaging traces
        .dff_mean : dff calculated with total recording mean as baseline
        .source_image : mean image where roi was selected on
        .category : name of category, none if its ROI mask
        .imaging_info : info on recording (keys: frame_rate, pixel_size, depth, frame_timings)
        .uniq_id : unique id
        .mask : ROI mask 
        .number_id : number of ROI mask, NONE if category
        .baseline_method : str, method of calculating baseline > currently only "mean" 
    '''