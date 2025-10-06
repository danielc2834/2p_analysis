# Preprocessing and analysis of 2photon data

This repository contains all sctipts to process raw 2photon recordings and subsequent analysis that were used in the master thesis of Christian Daniel. 


## Getting started

### Installing the environment

Lastly, before running any code please install our environment which is provided in the **2pIm.yml** file. Anaconda will install everything you need after a few easy steps.

- we need to install mambe in order to create our environment. Open anaconda prompt and type:

   `conda install conda-forge::mamba`

- open the anaconda command prompt and navigate to the folder you saved the repository in via the following command line:

   `cd Path-to-the-repository`

   > after typing cd you can drag and drop the folder into the command window. very convenient, 
   just make sure there is a space in between cd and the path. 

- before installing this environment change anacondas channel priority, otherwise this environment can not be solved by anaconda

    `conda config --set channel_priority flexible`

- next and last we can install the environment with the following command.
 
   `mamba env create -f 2pIm.yml`

- after anaconda is done you can check whether it was succesfully installed with the command:
   
   `conda env list` or `conda info --envs`


### Folder structure

Processing and subsequent analysis utilizes the following automatically generated folder structure:

     ├──Path to the Dataset-Folder
     │    ├──0_to_sort
     │    │   │──one folder per fly     
     │    │   │      └──all Tseries of that fly (raw output from 2P)
     │    │   └──stim (containting all stim output files of 2P)
     │    ├──1_raw_recordings
     │    │   │──one folder for each condition (defined in **preprocessing_params.py**)
     │    │   │      │──for each fly one folder
     │    │   │      │      └──  for each tseries one folder  (containing raw data form 2P)  
     │    ├──2_processed_recordings
     │    │   │──one folder for each condition (defined in **preprocessing_params.py**)
     │    │   │      │──for each fly one folder
     │    │   │      │      └──  for each tseries one folder  (containing processed data)  
     │    │   ├──big_skipping_masks.pkl
     │    │   ├──layer_skiping_masks.pkl
     │    │   └──processing_progress.pkl 
     │    ├──3_DATA
     │    │   └──one .pkl file for all data from one condition (folder in 2_processed_recordings)
     │    ├──4_results
     │    │   └──all vizualizations and quantifications
     │    ├──5_ZStacks
     │    │   │──one folder per fly
     │    │   │      └──all Zseries of that fly
     │    ├──6_stim_files
     │    │   └─ contain all stimuli files used in the 2P


### metadata file

To save experiment specific variables we used a metadata spreadsheet with the following structure: 

Each dataset has its own sheet, which has the same name as the datasetfolder

| **HEADER** | visual_stim | olf_stim | odorant | fps | frames_recorded	| age |	comments	| date	| laser	| z	| bin	| experiment	| fly	| sex	| rotation |	genotype	| TSeries	| region	| DLP_blueled_current	| zoom	| dwell_time	| solution date_id	| visual_stim_file	| olf_stim_type |
| ------ | ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |
| **EXAMPLE** | None	| ACV	| ACV	| 7.287	| 1130	| 14	| flies from navin, bit old	| 241204	| 179	| LH:20.18	| 5um	| 241203_whole_OL	| 241204_fly1	| female	| 130	| GCamp6f-UAS/CyO;R57C10Gal4-GAL4/tdTomato-UAS	| Tseries-000 | LH	| 26	| 4	| 4	| JF 23.11.2024	| None | pulse |
| **EXPLANATION** | abrevation of visual stimulus | Odorant used in the experiment (row) | odorant used for the whole fly | frames per second of recording | # frames recorded | age of fly in days | any additional comment from experimenter | date of recording in format YYMMDD | laser power (setting from 2P software) | z depth of recording plane | name of dataset | id of fly, format YYMMDD_flyX | sex of fly | rotation of flyholder relativ to screen in degree | full genotype | number of TSeries of recording, each fly starts at 0, format: Tseries-XXX | region of recording (LH = lateral horn) | current used for the visual stimulus | optic zoom | dwell time per pixel of scanner | imaging solution date id | name of visual stim file used, None if no visual stim was presented | type of olfactory stim, defined in preprocessing_params.py |

### parameter for preprocessing

We can adjust some processing parameters and important information in **/2panalysis/preprocessing_params.py**

To do so simply change the variables content

The following parameter can be set:
| **PARAMETER** | motion_alignment_algo | opts_dict | experiment | same_rois  | same_rois_columns  | z_range | olfactory_stimuli | condition_columns |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | 
| **EXPLANATION** | which motion alignment algorithm will be used : "Chris" or "Caiman" | parameter for the motion alignment, explanation after each parameter | experiment specific string, all processed data will be saved with this pre-string, this way you can analyse dataset two ways only by changing this |  True or Flase, if ROIs of a fly are used across TSeries |  the column that additionalz defines same roi exeption |  range of z depth after which recordings not similar enough to use same ROIs |  dictionary that defines all used olfactory stimuli with same name as in metadata column: olf_stim_type |  columns metadate file that define to which condition each recording belongs to, list of strings |  

<details><summary>
**EXAMPLE:**</summary>

```
motion_alignment_algo = "Chris" 
opts_dict = {
        'max_shifts' : (6, 6), # default(6, 6) maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
        'strides' : (48, 48), #default(48, 48) create a new patch every x pixels for pw-rigid correction
        'overlaps' : (24, 24), #default(24, 24) overlap between pathes (size of patch strides+overlaps)
        'max_deviation_rigid' : 3, #default(3) maximum deviation allowed for patch with respect to rigid shifts   
        'pw_rigid' : True, # flag for performing rigid or piecewise rigid motion correction
        'shifts_opencv' : True, # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
        'border_nan' : 'copy',  # replicate values along the boundary (if True, fill in with NaN)
        'nonneg_movie': False,}
caiman_params = CNMFParams(params_dict=opts_dict)
roi_extraction_type = 'manual' #how ROIS will be selected, cuurently just 'manual'
experiment = 'layer' #name that will be added in front of all outputs, in case you want to analyse data two times, so it doesnt overwrite or delete previous data, FOR ROI selection, not motion correction, if no need > empty str
same_rois = True #true if same rois across tseries
same_rois_columns = ['region'] #uses columns in metadat to determine which rois are the same, will use them to automatically generate traces for next tseries
z_range =  20 #range in zmotion that assigns same roi to other tseries, difference in depth is outside range > select new rois; depth = depth*step_size == micro
in_phase_bg_subtraction = False
olfactory_stimuli = {'pulse' : [5,5,5,20,5], 'on' : 341} #for step: [pre,stim,post,ISI,repetition], for "on" : lenth of stimulus
condition_columns = ['odorant', 'olf_stim', 'visual_stim', 'region'] 
```
</details>

## Running Preprocessing

In order to start the preprocessing open **2panalysis/pipeline_preprocessing.py** and run the script. The following interface will pop up. 

![SILIESLABLOGO](https://gitlab.rlp.net/cdaniel/master_2p/-/raw/main/wiki/preprocessing_GUI.png?ref_type=heads)

Please select the path to your dataset, the path of your metadatafile and if you want to correct for motiion and/or select ROIs. 

Both steps, correcting for motion as well as selecting ROIs will automaically be skipped if it was already done, however it saves some time if you just skip it.

## Pre-processing steps

### sorting

### renaming

### motion correction

### ROI selection

### data pooling and analysis



