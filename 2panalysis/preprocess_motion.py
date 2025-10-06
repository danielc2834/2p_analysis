'''Script to batch align recordings in x and y '''
import os, sys, pickle, napari, shutil
import core_preprocessing as core_pre
import Helpers.xmlUtilities as xml
import preprocessing_params
from dask_image.imread import imread
from napari.settings import get_settings
################################
my_input = sys.argv
dataset_folder, error_log = my_input[1], my_input[2]
paths = core_pre.dataset(dataset_folder)
# dataset_path = 'C:/Users/Christian/Desktop/2P_test'
# error_log = 'C:/Users/Christian/Desktop/2P_test/preprocessing_error_log_2025-01-17.txt'
with open(f'{paths.processed}/processing_progress.pkl', 'rb') as fi:
    processing_progress = pickle.load(fi)
################################
for condition in os.listdir(paths.raw):
    if condition not in processing_progress.keys():
            processing_progress[condition] = {}
    for fly in os.listdir(f'{paths.raw}/{condition}'):
        if fly not in processing_progress[condition].keys():
            processing_progress[condition][fly] = {}
        # one_fly={}
        for tseries in os.listdir(f'{paths.raw}/{condition}/{fly}'):
            if tseries.startswith("TSeries"):
                target = f'{paths.processed}/{condition}/{fly}/{tseries}'
                core_pre.move_stim_file(f'{paths.raw}/{fly}/{tseries}/{tseries}', target)
                if tseries not in processing_progress[condition][fly].keys():
                    processing_progress[condition][fly][tseries] = None
                if os.path.exists(f'{target}/_motavg.tif')==True and os.path.exists(f'{target}/_motCorr.tif')==True and os.path.exists(f'{target}/_motStack.tif')==True and processing_progress[condition][fly].get(tseries) is None:
                    processing_progress[condition][fly][tseries] = [True,False,False,False]
                if os.path.exists(f'{target}/_motavg.tif')==False and processing_progress[condition][fly].get(tseries) is not None or os.path.exists(f'{target}/_motCorr.tif')==False and processing_progress[condition][fly].get(tseries) is not None or os.path.exists(f'{target}/_motStack.tif')==False and processing_progress[condition][fly].get(tseries) is not None:
                    processing_progress[condition][fly][tseries] = [False,False,False,False]
                if processing_progress[condition][fly].get(tseries) is None or processing_progress[condition][fly].get(tseries)[0]==False: 
                    os.chdir(f'{paths.raw}/{condition}/{fly}/{tseries}')
                    if os.path.exists(f'{paths.processed}/{condition}/{fly}/{tseries}')==False:
                        os.makedirs(f'{paths.processed}/{condition}/{fly}/{tseries}')
                    tseries_path = f'{paths.raw}/{condition}/{fly}/{tseries}'
                    number_cycles = core_pre.find_cycles(tseries_path)
                    cycle_dict = core_pre.filter_cycles(tseries_path,number_cycles)
                    xmlfile_path = f'{tseries_path}/{os.path.basename(tseries_path)}.xml'
                    pixel_x_size,pixel_y_size,pixel_area= xml.getPixelSize(xmlfile_path)
                    # if preprocessing_params.bleedtrough_filter==True:
                    #     # exists=len(glob.glob('stack_filtered_Mcorr*'))>0
                    #     # if exists==False:
                    #     #TODO: implement with caiman
                    #     fft_2d,im_size = core_pre.calculate_spatial_fft(cycle_dict)
                    #     filtered_data_fft = core_pre.filter_bleedthrough_gaussianstripe(cycle_dict,fft_2d,im_size,treshold=preprocessing_params.fft_filter_treshold)
                    #     backtransformed_data = core_pre.backtransform(filtered_data_fft)
                    if preprocessing_params.motion_alignment_algo == 'Caiman':
                        dataset = core_pre.preprocess_dataset(tseries_path,error_log,target,preprocessing_params.caiman_params,usecaiman=True)
                    else:
                        dataset = core_pre.preprocess_dataset(tseries_path,error_log,target,preprocessing_params.caiman_params,usecaiman=False)
                    # if preprocessing_params.auto_roi_selection == True:
                    #     lower_limit=int(round(1.5/pixel_area)) # the limits are hardcoded here 1 to 3 um^2
                    #     higher_limit=int(round(7.5/pixel_area)) 
                    #     find_clusters_STICA_JF_BG(dataset,lower_limit,higher_limit,len(cycle_dict[1]),components=80)
                    processing_progress[condition][fly][tseries] = [True,False,False,False]
        #             one_fly[tseries] = [True, False, False] #motion_alligned, checked motion correction, selected ROIS
        # processing_progress[fly] = one_fly
with open(f'{paths.processed}/processing_progress.pkl', 'wb') as fo:
        pickle.dump(processing_progress, fo)
print('finished motion correction')
#now it will show all motion corrected stacks and you decide if its good enough, if not files will be deleted, also move visual stim into here
settings = get_settings()
settings.application.ipy_interactive = False
for condition in os.listdir(paths.raw):
    for fly in os.listdir(f'{paths.raw}/{condition}'):
        for tseries in os.listdir(f'{paths.raw}/{condition}/{fly}'):
            if tseries.startswith("TSeries"):
                target = f'{paths.processed}/{condition}/{fly}/{tseries}'
                if processing_progress[condition][fly].get(tseries)[1] is None or os.path.exists(f'{target}/_motCorr.tif')==False:
                    open(error_log, 'a', encoding="utf8").write(f'{tseries}: no motion correction output found')
                    open(error_log, 'a', encoding="utf8").write('\n')
                elif processing_progress[condition][fly].get(tseries)[1]==False: 
                    stack = imread(f'{target}/_motCorr.tif')
                    viewer = napari.view_image(stack,multiscale=False)
                    viewer.title = 'Are you happy? [y(es)/ n(o)]'
                    @viewer.bind_key('y')
                    def good_motion_correction(viewer):
                        processing_progress[condition][fly][tseries] = [True, True, False, False]
                        viewer.close_all()
                    @viewer.bind_key('n')
                    def bad_motion_correction(viewer):
                        processing_progress[condition][fly][tseries] = [True, False, False, False]
                        viewer.close_all()
                    napari.run()
with open(f'{paths.processed}/processing_progress.pkl', 'wb') as fo:
        pickle.dump(processing_progress, fo)

#removes motion correection data that were no good
for condition in processing_progress.keys():
    for fly in processing_progress[condition].keys():
        for tseries in processing_progress[condition][fly].keys():
            if processing_progress[condition][fly].get(tseries)[1] == False:
                if os.path.exists(f'{paths.processed}/{condition}/{fly}/{tseries}/_motavg.tif'):
                    os.remove(f'{paths.processed}/{condition}/{fly}/{tseries}/_motavg.tif')
                if os.path.exists(f'{paths.processed}/{condition}/{fly}/{tseries}/_motCorr.tif'):
                    os.remove(f'{paths.processed}/{condition}/{fly}/{tseries}/_motCorr.tif')
                if os.path.exists(f'{paths.processed}/{condition}/{fly}/{tseries}/_motStack.tif'):
                    os.remove(f'{paths.processed}/{condition}/{fly}/{tseries}/_motStack.tif')
                processing_progress[condition][fly][tseries] = [False,False,False,False]
with open(f'{paths.processed}/processing_progress.pkl', 'wb') as fo:
        pickle.dump(processing_progress, fo)
print('faulty corrections removed')