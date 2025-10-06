'''parameters for preprocessing calcium imagaing recordings'''
from caiman.source_extraction.cnmf.params import CNMFParams
################################################ MOTION CORRECTION ################################################
# bleedtrough_filter = True #no filter without sima
motion_alignment_algo = "Chris" 
# auto_roi_selection = True # no auto roi without sima
# fft_filter_treshold = 1
# export_average = True  #>> not implemented with caiman?? or in different place? 
#CIMAN motion correction
opts_dict = {
        'max_shifts' : (6, 6), # default(6, 6) maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
        'strides' : (48, 48), #default(48, 48) create a new patch every x pixels for pw-rigid correction
        'overlaps' : (24, 24), #default(24, 24) overlap between pathes (size of patch strides+overlaps)
        'max_deviation_rigid' : 3, #default(3) maximum deviation allowed for patch with respect to rigid shifts   
        'pw_rigid' : True, # flag for performing rigid or piecewise rigid motion correction
        'shifts_opencv' : True, # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
        'border_nan' : 'copy',  # replicate values along the boundary (if True, fill in with NaN)
        'nonneg_movie': False,}
#create a single options variable to pass into caiman motion correction algorithm
caiman_params = CNMFParams(params_dict=opts_dict)
################################################ ROI SELECTION ################################################
roi_extraction_type = 'manual' #how ROIS will be selected, cuurently just 'manual'
# 'cluster analysis MH_JF'
experiment = 'layer' #name that will be added in front of all outputs, in case you want to analyse data two times, so it doesnt overwrite or delete previous data, FOR ROI selection, not motion correction, if no need > empty str
same_rois = True #true if same rois across tseries
same_rois_columns = ['region'] #uses columns in metadat to determine which rois are the same, will use them to automatically generate traces for next tseries
z_range =  20 #range in zmotion that assigns same roi to other tseries, difference in depth is outside range > select new rois; depth = depth*step_size == micro
in_phase_bg_subtraction = False
################################################ ADDING STIMULI ################################################
olfactory_stimuli = {'pulse' : [5,5,5,20,5], 'on' : 341} #for step: [pre,stim,post,ISI,repetition], for "on" : lenth of stimulus
################################################ DATA POOLING ################################################
condition_columns = ['odorant', 'olf_stim', 'visual_stim', 'region'] #columns used in metadatasheet to make different conditions , currently: pulse and on > if you want to define new ones do so in core_preprocessing.olf_stim_array


################################################ DOCUMENTATION ################################################
#bleedtrough_correction >> whether bleadtrough (from visual stimulus) will be corrected while correcting the recordings for motion, boolean, default=True
#motion_alignment_algo >> which motion correction algorithm will be used, either "Chris" or "Caiman", str, dafault="Chris"
#motion_alignment_displacement >> maximum displacement values for motion correction,list, default=[10,10]
#multiple_modalities >> whether more than one stimulus was used, currently implemented: odorants, boolean, default=True
