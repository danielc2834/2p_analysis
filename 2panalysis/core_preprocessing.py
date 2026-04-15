'''Functions related to the preprocessing of Calcium Imaging recordings'''
################################################ IMPORTS ################################################
import os, pickle, glob, tifffile, cv2, copy, warnings, preprocessing_params, copy, cmath, math, shutil, time
from matplotlib.gridspec import GridSpec
from scipy.ndimage import label
from skimage.filters import threshold_local
from scipy.stats.stats import pearsonr
from itertools import permutations
from logging import exception
from scipy import ndimage
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.fftpack import fft2,fftshift,ifftshift,ifft2  
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
import caiman as cm 
from caiman.motion_correction import MotionCorrect
import core_analysis as core_a
import seaborn as sns
from roipoly import RoiPoly
from Helpers.xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize, getMicRelativeTime, getrastersPerFrame
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, wilcoxon, kruskal
import scikit_posthocs as sp
from datetime import date
from skimage.color import label2rgb
from matplotlib.backends.backend_agg import FigureCanvasAgg

################################################ Statistics ################################################
def all_possible_pairs(list):
    stat_list = [[a, b] for idx, a in enumerate(list) for b in list[idx + 1:]]
    return stat_list
def sig_level(pvalue):
    if pvalue > 0.05:
        level=None
    elif pvalue < 0.05 and pvalue >0.01:
        level='*'
    elif pvalue < 0.01 and pvalue >0.001:
        level='**'
    elif pvalue < 0.001:
        level='***'
    return level

def CrossCorrImage(tc, block_size=15, w=3, pixel_percentile=10, min_area=10):
    ymax, xmax, numFrames = tc.shape
    ccimage = np.zeros((ymax, xmax))

    start_time = time.time()

    # Select only active pixels based on std
    pixel_stds = np.std(tc, axis=2)
    thresh_val = np.percentile(pixel_stds, pixel_percentile)
    active_mask = pixel_stds > thresh_val
    tc = tc * active_mask[:, :, np.newaxis]

    # Cross-correlation computation
    for y in range(1 + w, ymax - w):
        for x in range(1 + w, xmax - w):
            center = tc[y, x, :] - np.mean(tc[y, x, :])
            center = center.reshape(1, 1, numFrames)
            ad_a = np.sum(center * center, axis=2)

            neighborhood = tc[y - w:y + w + 1, x - w:x + w + 1, :]
            neighborhood_mean = np.mean(neighborhood, axis=2)
            thing2 = neighborhood - neighborhood_mean[:, :, np.newaxis]
            ad_b = np.sum(thing2 * thing2, axis=2)

            ccs = np.sum(center * thing2, axis=2) / np.sqrt(ad_a * ad_b)
            ccs = np.delete(ccs, (ccs.size - 1) // 2)  # Remove center pixel
            ccimage[y, x] = np.mean(ccs)

    # Adaptive thresholding
    adaptive_thresh = threshold_local(ccimage, block_size, method='mean')
    binary_mask = ccimage > adaptive_thresh

    # Label initial ROIs
    labeled_rois, num_features = label(binary_mask)

    ### Filter small ROIs based on min_area
    filtered_rois = np.zeros_like(labeled_rois)
    current_label = 1
    for i in range(1, num_features + 1):
        roi_mask = labeled_rois == i
        if np.sum(roi_mask) >= min_area:
            filtered_rois[roi_mask] = current_label
            current_label += 1

    num_filtered = current_label - 1

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Number of ROIs detected (after filtering): {num_filtered}")

    return ccimage, filtered_rois


def stats_boxplot(statlog, df,x, y, hue , lh = True):
    """calculates wilcoxon, or ttest (depending on shapiro wilk normality result) and saves p value, and sig level in txtfile
    statlog : full path to txt file
    x : column with x axis of boxplot
    y : df column of values
    hues : column defining hue of boxplot
    currently: compares acros hues, but not across x values as its different visual stims >> i dont want to compare FFF to Grating, or 8D"""
    #get unique x values, get unique hues for x, for hue > compare all possible pairs
    for stim_con in df[x].unique():
        print(stim_con)
        lines = [f'Stimulus: {stim_con}', '','Statistics:','']
        with open(statlog, 'a', encoding="utf-8") as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        data=df[df[x] == stim_con]
        norm_p, norm_list =[], []
        for odor_con in data[hue].unique().tolist():
            data_single = data[data[hue]==odor_con][y].to_numpy()
            result_norm = shapiro(data_single)
            if result_norm.pvalue>0.05:
                norm="True"
                pvalue=result_norm.pvalue
            else:
                norm="False"
                pvalue=result_norm.pvalue
            norm_p.append(pvalue)
            norm_list.append(norm)
            
        odor_cons = data[hue].unique().tolist()
        if lh ==True:
            test = 'mann-withney-u'
        else:
            dataone= data[data[hue]==odor_cons[0]][y].to_numpy()
            datatow= data[data[hue]==odor_cons[1]][y].to_numpy()
            datathree= data[data[hue]==odor_cons[2]][y].to_numpy()
            all = [dataone, datatow, datathree]
            results_con = [(odor_cons[0], odor_cons[1]), (odor_cons[0], odor_cons[2]), (odor_cons[1], odor_cons[2])]
            post=False
            if any (x == "False" for x in norm_list):
                sig_wallis = kruskal(dataone, datatow, datathree)
                if sig_wallis.pvalue < 0.05:
                    test = "Kruskal Wallis > Dunn's test with bonferoni correction"
                    post =True
                    result = sp.posthoc_dunn(all, p_adjust = 'bonferroni')
                    resutls = [result[1][2], result[1][3], result[2][3]]
                    results_stars = []
                    for n in resutls:
                        results_stars.append(sig_level(n))
                else:
                    test = "Kruskal Wallis"
                stars = sig_level(sig_wallis.pvalue)
            else:
                test='NONE'
                lines=[f'group: {results_con}',  "\t", f"visual protocol: {stim_con}","\t",  f"Values: {y}", "\t", f"normality: {norm_list}", "\t",f"normp: {norm_p}", "\t", \
                        f'test: {test}',"\n"]
                with open(statlog, 'a', encoding="utf-8") as f:
                    for line in lines:
                        f.write(line)
                with open(statlog, 'a', encoding="utf-8") as f:
                                f.write("\n")
                continue
            if post:
                lines=[f'group: {results_con}',  "\t", f"visual protocol: {stim_con}","\t",  f"Values: {y}", "\t", f"normality: {norm_list}", "\t",f"normp: {norm_p}", "\t", \
                        f'test: {test}', "\t", f'pvalue: {resutls}, {results_stars}' ,"\n"]
            else:
                lines=[f'group: {results_con}',  "\t", f"visual protocol: {stim_con}","\t",  f"Values: {y}", "\t", f"normality: {norm_list}", "\t",f"normp: {norm_p}", "\t", \
                        f'test: {test}', "\t", f'pvalue: {sig_wallis.pvalue}, {stars}' ,"\n"]
        with open(statlog, 'a', encoding="utf-8") as f:
            for line in lines:
                f.write(line)
        with open(statlog, 'a', encoding="utf-8") as f:
                        f.write("\n")

################################################ FILE HANDLING ################################################
def move_stim_file(source, target):
    if os.path.exists(f'{source}_meta_data.txt') == True:
        if os.path.exists(f'{source}_meta_data.txt') ==False:
            shutil.copy(f'{source}_meta_data.txt', f'{target}_meta_data.txt')
    if os.path.exists(f'{source}_stim_output.txt') == True:
        if os.path.exists(f'{source}_stim_output.txt') ==False:
            shutil.copy(f'{source}_stim_output.txt', f'{target}_stim_output.txt') 


def check_folder_structure(dataset_path):
    '''generates empty folder structure and needed files for one dataset 
    
    Parameters
    ----------
    dataset_path : str
        path to dataset folder
    '''
    if os.path.exists(f'{dataset_path}/0_to_sort')==False:
        os.makedirs(f'{dataset_path}/0_to_sort')
    if os.path.exists(f'{dataset_path}/1_raw_recordings')==False:
        os.makedirs(f'{dataset_path}/1_raw_recordings')
    if os.path.exists(f'{dataset_path}/0_to_sort/stim')==False:
        os.makedirs(f'{dataset_path}/0_to_sort/stim')
    if os.path.exists(f'{dataset_path}/2_processed_recordings')==False:
        os.makedirs(f'{dataset_path}/2_processed_recordings')
    if os.path.exists(f'{dataset_path}/3_DATA')==False:
        os.makedirs(f'{dataset_path}/3_DATA')
    if os.path.exists(f'{dataset_path}/4_results')==False:
        os.makedirs(f'{dataset_path}/4_results')
    if os.path.exists(f'{dataset_path}/4_results')==False:
        os.makedirs(f'{dataset_path}/4_results')
    if os.path.exists(f'{dataset_path}/5_ZStacks')==False:
        os.makedirs(f'{dataset_path}/5_ZStacks')
    if os.path.exists(f'{dataset_path}/6_stim_files')==False:
        os.makedirs(f'{dataset_path}/6_stim_files')
    if os.path.exists(f'{dataset_path}/2_processed_recordings/processing_progress.pkl')==False:
        processing_progress={}
        with open(f'{dataset_path}/2_processed_recordings/processing_progress.pkl', 'wb') as file:
            pickle.dump(processing_progress, file)

class dataset:
    """all Paths, relative to dataset-folder, and information relevant for further steps"""
    def __init__(self, folder):
        """Initializes all fixed paths
        
        Parameter
        ---------
        folder : str
            Path to dataset folder, output of UI
        """
        self.today = date.today()
        self.sort = f'{folder}/0_to_sort'
        self.raw = f'{folder}/1_raw_recordings'
        self.processed = f'{folder}/2_processed_recordings'
        self.data = f'{folder}/3_DATA'
        self.results = f'{folder}/4_results'
        self.name = os.path.basename(folder)
        self.folder = folder
        self.stim = f'{folder}/0_to_sort/stim'
        self.zstacks = f'{folder}/5_ZStacks'
        self.stimdata = f'{folder}/6_stim_files'
        self.errorlog = f'{folder}/reprocessing_error_log_{self.today}.txt'

def rename_Tseries(data_path):
    """renames all TSeries to match a common pattern, without dublicates, to make further steps easier
    
    Parameters
    ---------
    data_path : str
        Path to /1_raw_recordings folder
    """
    for fly in data_path:
        for fly in os.listdir(data_path):
            if fly.endswith('.pkl'):
                continue
            for tseries in os.listdir(f'{data_path}/{fly}'):
                if tseries.startswith("TSeries"):
                    new_name = f'{tseries[:7]}-{fly}-{tseries[-3:]}'.replace("_", '-')
                    os.rename(f'{data_path}/{fly}/{tseries}', f'{data_path}/{fly}/{new_name}')
                    os.rename(f'{data_path}/{fly}/{new_name}/{tseries}.xml', f'{data_path}/{fly}/{new_name}/{new_name}.xml')
                    os.rename(f'{data_path}/{fly}/{new_name}/{tseries}.env', f'{data_path}/{fly}/{new_name}/{new_name}.env')

def get_condition_tseries(df_meta, condition_columns):
    '''gets condition string for single tseries that will be compared in analysis
    
    Parameters
    ---------
    df_meta : pandas DataFrame
        single row of metadata that correspond to one TSeries
    condition_columns : [str]
        List of collumn headers that will be used to specify condition, check preprocessing_params.condition_columns
        
    Returns
    ---------
    condition : str
        condition of TSeries made out of all stimuli used 
    '''
    condition = '_'
    for i in condition_columns:
        if i == condition_columns[-1]:
            condition = f'{condition}{df_meta[i].tolist()[0]}'
        else:
            condition = f'{condition}{df_meta[i].tolist()[0]}_'
    return condition
    
################################################ MOTION CORRECTION ################################################
def find_cycles(path):
    '''finds how many cycles where used during tseries
    
    Parameters
    ----------
    path : str
        path of tseries folder with all single .ome files inside
    
    Returns
    ----------
    number_cycles : int
        number cycles beeing used
    '''
    tif_list=glob.glob(f"{path}\\*Cycle*.tif")
    cycles=[]
    for tif in tif_list:
        if tif.endswith('.ome'):
            cycle_num=tif.split('\\')[-1]
            cycle_num=cycle_num.split('_')[-3]
            cycles.append(cycle_num)
    cycles=np.unique(np.array(cycles))
    number_cycles=len(cycles)
    return number_cycles

def filter_cycles(path,number_cycles):
    """gets a dictionary of all cycles containing all correspinding .ome file paths
    
    Parameters
    ----------
    path : str
        path of tseries folder with all single .ome files inside
    number_cycles : int
        number of cycles used during tseries recording, output of core_preprocessing.find_ciycles
    
    Returns
    ----------
    cycle_dict : dict
        {cycle1: [all .ome paths], cycle2 : [all corresponding .ome paths] ... }
    """
    cycle_dict={}
    for cycle in range(1,number_cycles+1):
        #find tifs for each cycle
        cycle_dict[cycle]=glob.glob(path +'\\'+'*_Cycle*'+str(cycle)+'_ch*')
    return cycle_dict  

def calculate_spatial_fft(cycle_dict):
    """calculates spatial FFT
    
    Parameters
    ----------
    cycle_dict : dict
        {cycle1: [all .ome paths], cycle2 : [all corresponding .ome paths] ... }, output of core_preprocessing.filter_cycles

    Returns
    ----------
    fft_2d : dict
        dictionary containing parameter for bleedtrough filtering organized in same way as input cycle_dict
    size : numpy 2D array
        dimensions of the image
        
    """
    fft_2d={}
    tif_array={}
    for cycle in cycle_dict.keys():
        #find image size
        im=Image.open(cycle_dict[cycle][0])
        size = np.array(im.size)
        tif_array[cycle]=np.zeros((size[1],size[0],len(cycle_dict[cycle]))) 
        for n_tif,tif in enumerate(cycle_dict[cycle]): 
            im=Image.open(cycle_dict[cycle][n_tif])
            im=np.array(im)
            tif_array[cycle][:,:,n_tif]=im
        fft_2d[cycle]=fftshift(fft2(tif_array[cycle],axes=(0,1)),axes=(0,1)) #fftshift shifts the dc component to the center of the image
    return fft_2d,size #this should be 2d numpy arrays in the dict 

def filter_bleedthrough_gaussianstripe(data_cycle_dict,fft_2d,im_size,treshold=1):
    """NOT SURE YET 
    
    Parameters
    ----------
    data_cycle_dict : dict
        {cycle1: [all .ome paths], cycle2 : [all corresponding .ome paths] ... }, output of core_preprocessing.filter_cycles
    fft_2d : dict
        dictionary containing parameter for bleedtrough filtering organized in same way as input cycle_dict, output of core_preprocessing.calculate_spatial_fft
    im_size: numpy 2D array
        dimensions of the image
    treshold : int
        threshold for NOT SURE YET , default = 1
        
    Returns
    ----------
    filtered_data_FFT : dict
        NOT SURE YET
        
    """
    height= im_size[1]
    width= im_size[0]
    # height_um=height*pixel_size
    # critical_wavelenght=height_um/2
    middle=[height//2,width//2]
    # wavelenght_vector=np.linspace(height_um,height_um/middle[0],middle[0])
    # diff=np.abs(wavelenght_vector-treshold)
    #idx_of_treshold=np.where(diff==np.min(diff))[0][0]+1
    idx_of_treshold=treshold
    print('filteridx %s'%(idx_of_treshold))
    rectangle_filter=np.ones((im_size[1],im_size[0],1))
    gaussian_filter=gaussian(im_size[0],1)
    gaussian_filter=(gaussian_filter*-1)+1
    rectangle_filter[:,:,0]=np.broadcast_to(gaussian_filter,(im_size[1],im_size[0]))
    # center=gaussian(6,5)
    rectangle_filter[middle[0]-idx_of_treshold:middle[0]+idx_of_treshold,:]=1
    filtered_data_fft={}
    for cycle in data_cycle_dict.keys():
        fft_for_cycle=fft_2d[cycle]
        #final_filter=np.broadcast_to(rectangle_filter,(height,width,data_FFT_2d[cycle].shape[2]))
        fft_for_cycle=fft_for_cycle*rectangle_filter
        filtered_data_fft[cycle]=fft_for_cycle
    return filtered_data_fft

def backtransform(filtered_data_fft):
    """NOT SURE YET 
    
    Parameters
    ----------
    filtered_data_FFT : dict
        NOT SURE YET, output of core_preprocessing.filter_bleedthrough_gaussianstripe
        
    Returns
    ----------
    complete_array : numpy XD array
        NOT SURE YET
        
    """
    #os.chdir(data_path)
    complete_array=[]
    for cycle in filtered_data_fft.keys():
        iFFT_for_cycle=filtered_data_fft[cycle]
        iFFT_for_cycle=np.abs(ifft2(ifftshift(iFFT_for_cycle,axes=(0,1)),axes=(0,1)))
        iFFT_for_cycle=iFFT_for_cycle.astype(np.int16)
        filtered_data_fft[cycle]=iFFT_for_cycle
    for cycle in filtered_data_fft.keys():
        if cycle==1:
            complete_array.append(filtered_data_fft[cycle])
            complete_array=np.array(complete_array[0])
        else:
            complete_array= np.concatenate((complete_array,filtered_data_fft[cycle]),axis=2)
    complete_array=complete_array[:,np.newaxis,:,:,np.newaxis]
    complete_array=np.transpose(complete_array,(3,1,0,2,4))
    return complete_array  

def preprocess_dataset(tser,error_log,target,caiman_params,usecaiman=True):
    """motion stabilizing one tseries 
    
    Parameters
    ----------
    tser : path
        path to tseries folder
    error_log : path
        path to error logbook of corresponding dataset
    target : path
        path to dir for saving processed data, 2_processed_recordings/current fly/current tseries/
    caiman_params : options variable
        caiman internal variable used to pass motion correction parameters, to modify check preprocessing_params file
    usecaiman : boolean
        whether the Caiman or Chris's algorithm will be used to perform motion correction
    """
    os.chdir(tser)
    data_exists=os.path.isdir(tser) != 0
    if data_exists==False:
        open(error_log, 'a', encoding="utf8").write(f'{tser}: no data found')
        open(error_log, 'a', encoding="utf8").write('\n')
        return
    else:
        if os.path.exists(f'{target}/_motCorr.tif')==False:
            motionStabilize_tifsInFolder(tser,error_log,caiman_params,target,usecaiman)

def motionStabilize_tifsInFolder(path,error_log,caiman_params,target,usecaiman=bool):
    """motion stabilizing every single frame of recording inside path 
    
    Parameters
    ----------
    path : path
        path to tseries folder
    error_log : path
        path to error logbook of corresponding dataset
    caiman_params : options variable
        caiman internal variable used to pass motion correction parameters, to modify check preprocessing_params file
    target : path
        path to dir for saving processed data, 2_processed_recordings/current fly/current tseries/
    usecaiman : boolean
        whether the Caiman (true) or Chris's (false) algorithm will be used to perform motion correction, default=True
        
    Output
    ----------
    _motStack.tif : tif file
        tif stack of the recording before motion correction, saved in 2_processed_recordings/current fly/current tseries
    """
    image_list= []
    cm_list=[]
    for filename in glob.glob(path+'/*.ome.tif'):
            image_data = tifffile.imread(filename, key=0)
            cm_images = filename
            cm_list.append(cm_images)
            image_list.append(image_data)
    image_stack = np.stack(image_list)
    tifffile.imsave(f"{target}/_motStack.tif", np.array(image_stack))
    # load all the .ome.tif files as a movie 
    if usecaiman==True:
        #TODO: here actualr ecording of fps?? 
        single_movie = cm.load(cm_list[0], fr=10)
        # according to docs motion percived better when downsampled but we can skip this step
        # downsample_ratio = .2 
        try:
            motionStabilizeCaiman(single_movie,caiman_params,target)
        except:
            open(error_log, 'a', encoding="utf8").write(f'{path}: Caiman-motion correction error')
            open(error_log, 'a', encoding="utf8").write('\n')
            return
    elif usecaiman==False:
        try:
            motionStabilize(image_stack,target)
        except:
            open(error_log, 'a', encoding="utf8").write(f'{path}: Chris-motion correction error')
            open(error_log, 'a', encoding="utf8").write('\n')
            return

def motionStabilizeCaiman(movie,caiman_params,target):
    """finally some actual motion stabilizing with Caiman
    
    Parameters
    ----------
    movie : caiman object
        all frames loaded into one caiman movie object (cm.load)
    caiman_params : options variable
        caiman internal variable used to pass motion correction parameters, to modify check preprocessing_params file
    target : path
        path to dir for saving processed data, 2_processed_recordings/current fly/current tseries/
        
    Output
    ----------
    _motCorr.tif : tif file
        tif stack of the recording after motion correction, saved in 2_processed_recordings/current fly/current tseries
    _motavg.tif : tif file
        average of the recording after motion correction, saved in 2_processed_recordings/current fly/current tseries
    """
    # create a motion correction object
    mc = MotionCorrect(movie, **caiman_params.get_group('motion'))
    # save the motion corrected object as a memory mappable file (not on disk but into memory)
    mc.motion_correct(save_movie=True)
    # load motion corrected movie from the memory (previous line)
    m_rig = cm.load(mc.mmap_file)
    # fix borders
    # bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
    mot_avg = mc.total_template_rig
    # save the motion corrected movie on the disk with the name "_motCorr"
    # m_random = cm.movie(m_rig)
    m_rig.save(f'{target}/_motCorr.tif')
    tifffile.imwrite(f'{target}/_motavg.tif', mot_avg)
    
def motionStabilize(stack, target):
    """finally some actual motion stabilizing with Chris' algo
    
    Parameters
    ----------
    stack : numpay 3D array
        array with all loaded single frame tifs of one TSeries
    target : path
        path to dir for saving processed data, 2_processed_recordings/current fly/current tseries/
        
    Output
    ----------
    _motCorr.tif : tif file
        tif stack of the recording after motion correction, saved in 2_processed_recordings/current fly/current tseries
    _motavg.tif : tif file
        average of the recording after motion correction, saved in 2_processed_recordings/current fly/current tseries
    """
    # use mean image of 40 frames as template
    im1 =  np.mean(stack[0:-1], axis = 0).astype("float32")
    template=cv2.GaussianBlur((im1),(3,3),0)
    # cv2.imshow("Template", 10*cv2.resize(template, None, fx=5, fy=5))
    # cv2.imshow("image", 10*cv2.resize(im1, None, fx=5, fy=5))
    # cv2.waitKey(0)
    # Find size of image
    sz = im1.shape
    # Motion correction
    motCorr=[]
    for image in tqdm(stack):
        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION# cv2.MOTION_EUCLIDEAN #
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Specify the number of iterations.
        number_of_iterations = 10000
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-20
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        try:
            image_t=image.astype(np.float32)
            # _, image_t = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
            # Run the ECC algorithm. The results are stored in warp_matrix.
            (cc, warp_matrix) = cv2.findTransformECC (template,cv2.GaussianBlur(image_t,(3,3),0),warp_matrix, warp_mode, criteria)
            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                # Use warpPerspective for Homography
                image_aligned = cv2.warpPerspective (image_t, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else :
                # Use warpAffine for Translation, Euclidean and Affine
                image_aligned = cv2.warpAffine(image_t, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            # _, image_t = cv2.threshold(image_aligned, 70, 255, cv2.THRESH_TOZERO)   
        except:
            image_aligned = image_t
            print('no significant motion')
        motCorr.append(image_aligned.astype(np.uint16))
        # Show results
        # """cv2.imshow("Image 1", cv2.resize(10*template, None, fx=5, fy=5))
        # cv2.imshow("Image 2", cv2.resize(10*((image/256).astype('uint8')), None, fx=5, fy=5))
        # cv2.imshow("Aligned Image 2", cv2.resize(10*((image_aligned/256).astype('uint8')), None, fx=5, fy=5))
        # cv2.waitKey(0)"""
    mot_avg = np.mean(np.array(motCorr), axis = 0)
    tifffile.imwrite(f'{target}/_motCorr.tif', np.array(motCorr))
    tifffile.imwrite(f'{target}/_motavg.tif', mot_avg)

################################################ ROI SELECTION ################################################
class ROI: 
    '''A region of interest from an image sequence '''
    
    def __init__(self,Mask = None, experiment_info = None, imaging_info = None, uniq_id = None): 
        ''' 
        Initialized with a mask and optionally with experiment and imaging
        information
        '''
        if (Mask is None):
            raise TypeError('ROI: ROI must be initialized with a mask (numpy array)')
        if (imaging_info is not None):
            self.imaging_info = imaging_info
        if (experiment_info is not None):
            self.experiment_info = experiment_info
        if (uniq_id is None):
            self.uniq_id = id(self) # Generate a unique ID everytime it is not given
        else:
            self.uniq_id = uniq_id # Useful during transfer
        self.mask = Mask
        
        
    def __str__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)

    def __repr__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)
        
    def setCategory(self,Category):
        self.category = Category

    def set_z_depth(self,depth):
        self.z_depth = depth
        
    def set_extraction_type(self,extraction_type):
        self.extraction_type = extraction_type
        
        
    def setSourceImage(self, Source_image):
        '''puts nmean image where rois where selected on into .source_image '''
        if np.shape(Source_image) == np.shape(self.mask):
            self.source_image = Source_image
        else:
            raise TypeError('ROI: source image dimensions has to match with ROI mask.')
        
    def showRoiMask(self, cmap = 'Dark2',source_image = None):
        '''shows the roi mask drawn ontop of mean image'''
        if (source_image is None):
            source_image = self.source_image
        curr_mask = np.array(copy.deepcopy(self.mask),dtype=float)
        curr_mask[curr_mask==0] = np.nan
        plt.imshow(source_image,alpha=0.8,cmap = 'gray')
        plt.imshow(curr_mask, alpha=0.4,cmap = cmap)
        plt.axis('off')
        plt.title(self)
        
    def calculateDf(self,stimulus_information=None,method='mean', moving_avg = False, bins = 4):
        '''calculates DFF with mean of total recording and puts into .dff_mean'''
        try:
            self.raw_trace
        except NameError:
            raise NameError('ROI: for deltaF calculations, a raw trace needs to be provided: a.raw_trace')
        if method=='mean':
            trace=copy.deepcopy(self.raw_trace)
            df_trace = (trace-np.mean(trace))/np.mean(trace)
            self.baseline_method = method
        # self.dff_mean = df_trace
        elif method=='begining': #first 5 seconds of recording
            frames=np.array(stimulus_information['output_data_downsampled'].index)
            mean_val=mean_of_initial5secs(self.raw_trace,frames)
            df_trace = (self.raw_trace-mean_val)/(mean_val)
            self.baseline_method = method
        elif method=='rolling_mean':
            if 'T4T5' in self.experiment_info['Genotype']:
                shift=int(np.ceil(0.45*self.imaging_info['frame_rate'])) # trace is shifted to account for a delay in visualization of response delay was calculated by Bgur
                trace=copy.deepcopy(np.roll(self.raw_trace,-shift))
            else:
                trace=copy.deepcopy(self.raw_trace)
            window_size=30
            period=self.imaging_info['FramePeriod']
            window=np.round(window_size/period)
            trace_b=pd.Series(np.where(trace<((np.std(trace)*2)+np.mean(trace)),trace,np.nan)) #np.floor(window/2).astype(int)
            mean_trace=trace_b.rolling(window.astype(int),min_periods=np.floor(window/3).astype(int),center=True).mean()
            if np.all(np.isnan(mean_trace)):
                raise Exception ('all trace is Nan')
            df_trace=(trace-mean_trace)/mean_trace
            self.baseline_method='rolling average data<(2*std)+mean window 20secs'
        elif method=='baseline_epoch':
            stim=np.array(stimulus_information['output_data_downsampled']['epoch'])
            frames=np.array(stimulus_information['output_data_downsampled'].index)
            if 'T4T5' in self.experiment_info['Genotype']:
                shift=int(np.ceil(0.45*self.imaging_info['frame_rate'])) # trace is shifted to account for a delay in visualization of response delay was calculated by Bgur
                trace=copy.deepcopy(np.roll(self.raw_trace,-shift))
            else:
                trace=copy.deepcopy(self.raw_trace)
            # exclude for now the initial frames (with no sitmuli)
            signal_l=trace
            signal_0=signal_l[:frames[0]-1]
            signal_0[:]=np.mean(signal_0)
            signal_1=signal_l[frames[0]-1:]
            mean_trace=compute_local_means(stim, signal_1,mean_=True)
            mean_trace=np.concatenate([signal_0,mean_trace])
            df_trace=(trace-mean_trace)/mean_trace
            self.baseline_method= 'initial_period_with_no_stim'
        elif method=='convolution':
            trace=copy.deepcopy(self.raw_trace)
            period=self.imaging_info['FramePeriod']
            window_size=60 # make a rolling average for every minute of recording 60secs
            window=np.round(window_size/period)
            kernel=np.divide(np.ones(window,dtype=float),float(window))
            traces_mean=ndimage.convolve1d(trace,kernel,axis=0,mode='reflect')
            df_trace=np.divide(trace-traces_mean,traces_mean)
        elif method == 'tau_subepoch' and self.stim_info['EPOCHS'] == 1: # use the initial part of the stimulus (named tau is stimfile; I know, bad name) which shows a pre-epoch that depends on the stimulus
            if self.stim_info['EPOCHS'] == 1: 
                period = self.imaging_info['FramePeriod']  
                len_subepoch = self.stim_info['tau'][0]
                subepoch_frames = int(np.floor(len_subepoch/period))
                baseline = self.raw_trace[self.stim_info['trial_coordinates'][0][0][0] - subepoch_frames: self.stim_info['trial_coordinates'][0][0][0]]
                df_trace = np.divide(self.raw_trace - np.mean(baseline), np.mean(baseline))
                self.imaging_info['baseline_mean']=np.mean(baseline)
            elif self.stim_info['Trials'] > 1:
                period = self.imaging_info['FramePeriod']
                baseline = np.zeros_like(self.raw_trace)
                len_subepoch = self.stim_info['tau'][0]
                subepoch_frames = int(np.floor(len_subepoch/period))
                baseline[0:self.stim_info['trial_coordinates'][0][0][0]-subepoch_frames] = np.mean(self.raw_trace[self.stim_info['trial_coordinates'][0][0][0]-subepoch_frames:self.stim_info['trial_coordinates'][0][0][0]])
                for trial in range(len(self.stim_info['trial_coordinates'][0])):
                    len_subepoch = self.stim_info['tau'][0]
                    subepoch_frames = int(np.floor(len_subepoch/period))
                    baseline[self.stim_info['trial_coordinates'][0][trial][0] - subepoch_frames:] = np.mean(self.raw_trace[self.stim_info['trial_coordinates'][0][trial][0] - subepoch_frames: self.stim_info['trial_coordinates'][0][trial][0]])
                df_trace = np.divide(self.raw_trace - baseline, np.mean(baseline))
        elif method == '0_epoch_1sbefore': # often prefered for FFF stims where one epoch is the baseline of others
            period = self.imaging_info['FramePeriod']  
            trial_counts = []
            #baseline = np.zeros_like(self.raw_trace)
            df_trace = np.zeros_like(self.raw_trace)
            for key in self.stim_info['trial_coordinates']:
                trial_counts.append(len(self.stim_info['trial_coordinates'][key]))
            for i in range(trial_counts[0]):
                for key in self.stim_info['trial_coordinates']:
                    try:
                        boundaries = self.stim_info['trial_coordinates'][key][i][0]
                    except IndexError:
                        continue
                    boundaries_baseline = self.stim_info['trial_coordinates'][0][i][0] 
                    boundaries_baseline = [boundaries_baseline[1]-int(1/period),boundaries_baseline[1]] # get the last second of the baseline epoch
                    base_vals = (np.mean(self.raw_trace[boundaries_baseline[0]:boundaries_baseline[1]]))
                    vals = self.raw_trace[boundaries[0]:boundaries[1]]
                    df_trace[boundaries[0]:boundaries[1]] = (vals - base_vals) / base_vals
            #calculate df for the first 5 seconds (no stim period) 
            df_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]] = (self.raw_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]] - np.mean(self.raw_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]])) / np.mean(self.raw_trace[0:self.stim_info['trial_coordinates'][0][0][0][0]])
            # eliminate any remaining  
            df_trace = np.where(df_trace == 0 ,np.nan, df_trace)
            df_trace = df_trace[~np.isnan(df_trace)]
        else:
            raise Exception('Tau df calculation not yet implemented for more than 1 epoch')
        if moving_avg: # de-trending step
            self.df_trace = movingaverage(df_trace, bins)
        else:
            self.df_trace = df_trace
        return self.df_trace
    def plotDF(self, line_w = 1, adder = 0,color=plt.cm.Dark2(0)):
        plt.plot(self.df_trace+adder, lw=line_w, alpha=.8,color=color)
        try:
            self.stim_info['output_data']
            stim_frames = self.stim_info['output_data'][:,7]  # Frame information
            stim_vals = self.stim_info['output_data'][:,3] # Stimulus value
            uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
            stim_vals = stim_vals[uniq_frame_id]
            # Make normalized values of stimulus values for plotting
            stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
                *np.max(self.df_trace+adder)
            plt.plot(stim_vals,'--', lw=1, alpha=.6,color='k')
        except KeyError:
            print('No raw stimulus information found')
    # def appendPrevTraces(self):

    #     """
    #     in the cases where more than one stimulation was done for the same t series, 
    #     the first stimulation paradigm and the corresponding traces are stored in a seperate
    #     attribute for further use, and to avoid identity mixing among traces from different stimulations

    #     """
    #     try:
    #         self.prevstim_data
    #     except AttributeError:
    #         raise Exception('to append previous stimulation traces a prevstim_data dict is needed')
        
    #     # if self.prevstim_data['stim_info']['stim_name'] == self.stim_info['stim_name']:
    #     #     raise Exception ('previous stim data and current data are the same')
    #     try:
    #         self.resp_trace_all_epochs
    #     except AttributeError:
    #         raise Exception('no previous traces')
    #     # if self.prevstim_data['stim_info']['epochs']==len(resp_trace_all_epochs):
    #     self.prevstim_data['prev_resp_traces']=copy.deepcopy(self.resp_trace_all_epochs)
        
    #     ### TODO fix this next lines! maybe just erase everything except 2 or 3 important ones
    #     del self.resp_trace_all_epochs
    #     del self.raw_trace
    #     del self.stim_name
    #     del self.whole_trace_all_epochs
    #     del self.df_trace
    #     # else:
    #     #     raise Exception('epochs dont coincide between prevstim info and traces')

    # def appendPrevResponseProperties(self):
    #     self.RespProperties={}
    #     try:
    #         self.RespProperties['CS']=self.CS
    #     except:
    #         pass
    #     try:
    #         if self.CS=='OFF':
    #             self.RespProperties['CSI']=self.CSI_OFF
    #         elif self.CS=='ON':  
    #             self.RespProperties['CSI']=self.CSI_ON
    #     except: 
    #         pass
    #     try:
    #         if self.CS=='OFF':
    #             self.RespProperties['DSI']=self.DSI_OFF
    #         elif self.CS=='ON':  
    #             self.RespProperties['DSI']=self.DSI_ON
    #     except:
    #         pass

    # def appendPrevanalysisParams(self):
    #     try:
    #         self.appendPrevanalysisParams
    #     except: 
    #         pass
    def appendTrace(self, trace, epoch_num, trace_type = 'whole'):
        
        if trace_type == 'whole':
            try:
                self.whole_trace_all_epochs
            except AttributeError:
                self.whole_trace_all_epochs = {}
                
            
            self.whole_trace_all_epochs[epoch_num] = trace
        
        elif trace_type == 'response':
            try:
                self.resp_trace_all_epochs
            except AttributeError:
                self.resp_trace_all_epochs = {}
                
            self.resp_trace_all_epochs[epoch_num] = trace
        
        elif trace_type == 'raw':
            try:
                self.whole_traces_all_epochsTrials
            except AttributeError:
                self.whole_traces_all_epochsTrials={}
            
            self.whole_traces_all_epochsTrials[epoch_num] = trace
            
    # def appendprevStimInfo(self, Stim_info ,raw_stim_info = None):

    #     try:
    #         self.prevstim_data
    #     except AttributeError:
    #         self.prevstim_data={}
    #     finally:
    #         pass
    #     try: 
    #         self.stim_info
    #     except:
    #         raise Exception('no previous stimuli found')
    #     if self.stim_info['stim_name']==Stim_info['stim_name']:
    #         raise Exception ('prev stim is the same as current')
    #     else:
    #         self.prevstim_data['stim_info']=copy.deepcopy(self.stim_info)
    #         del self.stim_info
    def appendStimInfo(self, Stim_info ,raw_stim_info = None):
        
        self.stim_info = Stim_info
        self.stim_name = Stim_info['stim_name']
        
        if (raw_stim_info is not None):
            # This part is now stored in the stim_info already but keeping it 
            # for backward compatibility.
            self.raw_stim_info = raw_stim_info
    
    def findMaxResponse_all_epochs(self):
        
        try:
            self.resp_trace_all_epochs
        except AttributeError:
            raise AttributeError('ROI_bg: for finding maximum responses \
                            "resp_trace_all_epochs" has to be appended by \
                            appendTrace() method ')
            
        #obtain analysis type( depending on that, more than one maximum may be needed)
        analysis_type=self.experiment_info['analysis_type']

        if analysis_type=='8D_edges_find_rois_save' or analysis_type=='4D_edges' or analysis_type=='1-dir_ON_OFF': 
            self.max_resp_all_epochs_ON = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) #Seb: epochs_number --> EPOCHS
            self.max_resp_all_epochs_ON[:] = np.nan
            self.max_resp_all_epochs_OFF = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) 
            self.max_resp_all_epochs_OFF[:] = np.nan
            self.max_resp_all_epochs_ON=np.zeros((int(self.stim_info['EPOCHS'])))
            self.max_resp_all_epochs_ON[:]=np.nan
            self.max_resp_all_epochs_OFF=np.zeros((int(self.stim_info['EPOCHS'])))
            self.max_resp_all_epochs_OFF[:]=np.nan
            self.trace_STD_OFF=np.zeros((int(self.stim_info['EPOCHS'])))
            self.trace_STD_OFF[:]=np.nan
            self.trace_STD_ON=np.zeros((int(self.stim_info['EPOCHS'])))
            self.trace_STD_ON[:]=np.nan
            self.shifted_trace_={}
            plt.close('all')
            for epoch_idx in self.resp_trace_all_epochs.keys():
                try:
                    if self.stim_info['stimtype'][epoch_idx]!='driftingstripe' and self.stim_info['stimtype'][epoch_idx]!='ADS' and self.stim_info['stimtype'][epoch_idx]!='G':
                        continue
                except:
                    
                    stim_response_timedelay= 0.45 # this is a hardcoded delay between the stimulus and the peak of a t4t5 axon, Burak gur calculated this number (seconds)
                    # the shift is necessary for the stimulus: 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt'
                    shift=int(stim_response_timedelay*self.imaging_info['frame_rate'])+1 # calculate number of frames that include the delay time required, since this is float, approximate to nearest higher int
                    if (self.stim_info['stim_name'] == 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt'):
                        # for this stimulus type there is a brigth flash and then the first edge is off. hopefully it is not like this for other stimuli
                        shifted_trace=np.roll(self.resp_trace_all_epochs[epoch_idx],-shift)
                        shifted_trace_OFF=shifted_trace[:len(shifted_trace)//2]
                        shifted_trace_ON=shifted_trace[len(shifted_trace)//2:]
                    elif self.stim_info['stim_name'] == 'DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_ONEDGEFIRST_80sec.txt'\
                                    or self.stim_info['stim_name'] == 'Mapping_for_MIexp.txt'\
                                    or self.stim_info['stim_name']=='Drifting_stripesJF_8dir.txt' \
                                    or 'DriftingStripe_4sec' in self.stim_info['stim_name']\
                                    or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_15degoffset_ONEDGEFIRST.txt'\
                                    or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                                    or self.stim_info['stim_name'] =='mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                                    or self.stim_info['stim_name'] == 'Dirfting_edges_JF_8dirs_ON_first.txt'\
                                    or self.stim_info['stim_name'] == '1_Dirfting_edges_JF_8dirs_ON_first.txt':
                        shifted_trace=self.resp_trace_all_epochs[epoch_idx]
                        shifted_trace_OFF=shifted_trace[len(shifted_trace)//2:]
                        shifted_trace_ON=shifted_trace[:len(shifted_trace)//2]
                    peak_interval=int(round(0.5*self.imaging_info['frame_rate'])) #TODO this will be used to exclude the 1s interval containing the maximum response from the mean calculation
                    self.trace_STD_ON[epoch_idx]=np.std(shifted_trace_ON)
                    self.trace_STD_OFF[epoch_idx]=np.std(shifted_trace_OFF)
                    self.max_resp_all_epochs_OFF[epoch_idx] = np.nanmax(shifted_trace_OFF)
                    location_off_peak=np.where(shifted_trace_OFF==self.max_resp_all_epochs_OFF[epoch_idx])[0][0]
                    self.max_resp_all_epochs_ON[epoch_idx] = np.nanmax(shifted_trace_ON)
                    location_on_peak=np.where(shifted_trace_ON==self.max_resp_all_epochs_ON[epoch_idx])[0][0]
                    # the range_x variables produce indices that map to the response traces, excluding the location of the peak response
                    # this is used to calculate a baseline mean and standard deviation
                    if location_on_peak<=shift:
                        range_ON=np.array(range(location_on_peak+peak_interval,len(shifted_trace_ON)-(2*peak_interval)+1))
                    else:
                        range_ON=np.append(np.array(range(0,location_on_peak-peak_interval)),np.array(range(location_on_peak+peak_interval,len(shifted_trace_ON))))
                        #just in case the range ON turns into a float type (this happens if the peak is to close to the end of the trace)
                        range_ON=range_ON.astype(int)
                    if location_off_peak<=shift:
                        range_OFF=np.array(range(location_off_peak+peak_interval,len(shifted_trace_OFF)-(2*peak_interval)+1))
                    else:
                        range_OFF=np.append(np.array(range(0,location_off_peak-peak_interval)),np.array(range(location_off_peak+peak_interval,len(shifted_trace_OFF))))
                        #just in case the range OFF turns into a float type (this happens if the peak is to close to the end of the trace)
                        range_OFF=range_OFF.astype(int)
                    self.overall_baseline_loc=np.append(shifted_trace_ON[range_ON],shifted_trace_OFF[range_OFF])
                    overall_baseline_loc=copy.deepcopy(self.overall_baseline_loc)
            
            self.max_response_ON = np.nanmax(self.max_resp_all_epochs_ON)
            self.max_response_OFF = np.nanmax(self.max_resp_all_epochs_OFF)
            self.max_resp_idx_ON = np.nanargmax(self.max_resp_all_epochs_ON)
            self.max_resp_idx_OFF = np.nanargmax(self.max_resp_all_epochs_OFF)
            #reliability=np.zeros(len(self.reliability_all_epochs))
            # for idx,epoch in enumerate(self.reliability_all_epochs):
                # reliability[epoch]=
            
            #use the preferred directions of the ON and OFF responses to set the reliability for every ROI
            self.reliability_PD_ON= self.reliability_all_epochs_ON[self.max_resp_idx_ON]
            self.reliability_PD_OFF= self.reliability_all_epochs_OFF[self.max_resp_idx_OFF]
            self.reliability_PD = [self.reliability_PD_OFF,self.reliability_PD_ON][np.argmax([self.max_response_OFF,self.max_response_ON])]
            
        elif (self.stim_info['stim_name'] == 'LocalCircle_5secON_5sec_OFF_120deg_10sec.txt' or\
            self.stim_info['stim_name'] == '2_LocalCircle_5secON_5sec_OFF_120deg_10sec.txt' or\
            self.stim_info['stim_name'] =='LocalCircle_5sec_120deg_0degAz_0degEl_Sequential_LumDec_LumInc_10sec.txt' or\
            self.stim_info['stim_name'] =='LocalCircle_5sec_220deg_0degAz_0degEl_Sequential_LumDec_LumInc.txt') or\
            analysis_type== 'Flashes'    :
            
            self.max_resp_all_epochs = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) #Seb: epochs_number --> EPOCHS
            
            self.max_resp_all_epochs[:] = np.nan
            
            for epoch_idx in self.whole_trace_all_epochs_df:
                self.max_resp_all_epochs[epoch_idx] = np.nanmax(self.whole_trace_all_epochs_df[epoch_idx])
            
            self.max_response = np.nanmax(self.max_resp_all_epochs)
            self.max_resp_idx = np.nanargmax(self.max_resp_all_epochs)
        
        elif self.stim_info['stim_name'] == 'Gratings_sine_30sw_TF_0.2_to_4hz_3sec_GRAY_Xsec_moving_right_left_90sec.txt':
            # calculate df/f the baseline here is the gray interlude.
            aaa='aaa'      # it is already done somewhere else, in the calculate df function :)      

        elif self.stim_info['stim_name'] == 'exp_random_ONedges_20dirs_20degPerS.txt':
            #find the maximum for every trial
            self.max_response_alltrialsEpochs= {}  
            self.timing_max_response_alltrialsEpochs={}
            self.max_resp_all_epochs=np.zeros(len(self.whole_traces_all_epochsTrials.keys()))

            for ix,epoch in enumerate(self.whole_traces_all_epochsTrials.keys()):
                #slice out the intermediate epoch. 
                base_duration=self.base_dur[ix]
                alltrials=self.whole_traces_all_epochsTrials[epoch][base_duration:,:]
                self.max_response_alltrialsEpochs[epoch]=np.nanmax(alltrials,axis=0)
                self.timing_max_response_alltrialsEpochs[epoch]=np.argmax(alltrials,axis=0)*self.imaging_info['FramePeriod']
                # find the average max responses across trials 
                self.max_resp_all_epochs[ix]= np.nanmax(self.whole_trace_all_epochs[epoch][self.base_dur[epoch-1]:])
            self.max_response = np.nanmax(self.max_resp_all_epochs)
            self.max_resp_idx = np.nanargmax(self.max_resp_all_epochs)
        else:
            self.max_resp_all_epochs = \
                np.empty(shape=(int(self.stim_info['EPOCHS']),1)) #Seb: epochs_number --> EPOCHS
            
            self.max_resp_all_epochs[:] = np.nan
            
            for epoch_idx in self.resp_trace_all_epochs:
                self.max_resp_all_epochs[epoch_idx] = np.nanmax(self.resp_trace_all_epochs[epoch_idx])
            
            self.max_response = np.nanmax(self.max_resp_all_epochs)
            self.max_resp_idx = np.nanargmax(self.max_resp_all_epochs)
    
    def calculate_DSI_PD(self,method='PDND'):
        '''Calcuates DSI and PD of an ROI '''
        # NOTE:code for methods PDND and vector is not updated. update before using
        # for now it only works with contrast specific responses
        try:
            self.max_resp_all_epochs
            self.max_resp_idx
            self.stim_info
            self.max_response               
        except AttributeError:
            #raise TypeError('ROI_bg: for finding DSI an ROI needs\
                                #max_resp_all_epochs and stim_info')
            self.max_resp_all_epochs_ON
        #finally:
            #pass
        def find_opp_epoch(self, current_dir, current_freq, current_epoch_type): # deprecated
            required_epoch_array = \
                    (np.array(self.stim_info['direction']) == (current_dir*-1)) & \
                    (np.array(self.stim_info['velocity']) == current_freq) & \
                    (np.array(self.stim_info['stimtype']) == current_epoch_type)  
            
            return np.where(required_epoch_array)[0]
        
        
        if method=='Mazurek':
            # this method has been described by Mazurek et al.,  doi: 2014 10.3389/fncir.2014.00092
            #extract responses to the different epochs/subepochs
            ### reorganize orientation vector (this is required to transform the polar axis into one that goes from 0 to 135 degrees counterclockwise (right now is clockwise))
            try: # ideally an stim file should define Stimulus.angle!!
                direction_vector=self.stim_info['angle']
            except:
                direction_vector=np.sort(np.unique(self.stim_info['output_data_downsampled']['theta']))
                direction_vector=direction_vector[~np.isnan(direction_vector)]
            self.direction_vector=direction_vector #store this vector. this will replace self.stim_info['direction'] in a prev version of the code
            dirs = direction_vector
            dirs = np.radians(dirs)
            try: # check if the stimulus has 2 polarities
                self.max_resp_all_epochs_ON
                self.max_resp_all_epochs_OFF
                pols=2
            except:
                pols=1
            if pols==2:
                resps_on  = np.zeros(len(self.max_resp_all_epochs_ON),dtype='c16')
                resps_off = np.zeros(len(self.max_resp_all_epochs_OFF),dtype='c16')
                exp_on=np.zeros(len(self.max_resp_all_epochs_ON),dtype='c16')
                exp_off=np.zeros(len(self.max_resp_all_epochs_ON),dtype='c16')
                for ix,value in enumerate(self.max_resp_all_epochs_ON):
                    exp_on[ix]=cmath.exp(complex(0,dirs[ix]))
                    resps_on[ix]=value
                for ix,value in enumerate(self.max_resp_all_epochs_OFF):
                    exp_off[ix]=cmath.exp(complex(0,dirs[ix]))
                    resps_off[ix]=value
                # calculate the vector 
                print('2 polarities analized')
                self.DSI_ON  = np.abs(np.sum(resps_on*exp_on)/ np.sum(resps_on))
                self.DSI_OFF = np.abs(np.sum(resps_off*exp_off)/ np.sum(resps_off))
                self.PD_ON   = np.angle(np.sum(resps_on*exp_on)/ np.sum(resps_on), deg=True)
                if self.PD_ON <0:
                    self.PD_ON= 360+self.PD_ON
                self.PD_OFF  = np.angle(np.sum(resps_off*exp_off)/ np.sum(resps_off),  deg=True)
                if self.PD_OFF <0:
                    self.PD_OFF= 360+self.PD_OFF
            
                if self.max_response_ON>self.max_response_OFF:
                    self.dir_max_resp=self.direction_vector[self.max_resp_idx_ON]
                elif self.max_response_ON<self.max_response_OFF:
                    self.dir_max_resp=self.direction_vector[self.max_resp_idx_OFF]

            else: 
                print('one polarity analized')
                resps=np.zeros(len(self.max_resp_all_epochs),dtype='c16')
                exp_=np.zeros(len(self.max_resp_all_epochs),dtype='c16')
                for ix,value in enumerate(self.max_resp_all_epochs):
                    exp_[ix]=cmath.exp(complex(0,dirs[ix]))
                    resps[ix]=value
                self.DSI  = np.abs(np.sum(resps*exp_)/ np.sum(resps))
                self.PD   = np.angle(np.sum(resps*exp_)/ np.sum(resps), deg=True)
                if self.PD <0:
                    self.PD= 360+self.PD
    
    def append_SNR_Reliability(self,name_snr,name_rel,SNR_val,corr_val):     #Juan made function
        """
        takes in a  Signal to noise ratio value and a reliability value 
        and appends it to An ROI instance

        names are user input, in case that more than one SNR value wants 
        to be added to the same ROI instance, for example when multiple recording 
        cycles or Tseries use the same ROI
        """
        if self.SNR is not None:
            self.SNR.update({name_snr:SNR_val})
            self.SNR.reliability.update({name_rel:corr_val})
        else:
            self.SNR={name_snr:SNR_val}
            self.reliability={name_rel:corr_val}
            
    def trial_average_noise(self):

        # check length of trials
        lengths = []
        traces = []
        for trial in self.stim_info['trial_coordinates'][0]:
            lengths.append(trial[1]-trial[0])
            traces.append(self.df_trace[trial[0],trial[1]])
        minlen = np.min (np.array(lengths))

        trials_arr = np.zeros(minlen,len(self.stim_info['trial_coordinates'][0]))
        for ix,trace in enumerate(traces):
            trials_arr[ix,:] = trace[:minlen]
        self.whole_traces_all_epochsTrials = {0:trials_arr}
        
    def calculate_reliability(self):
        """ calculates the average pariwise correlation between trials to 
        estimate the reliability of responses.

        Parameters
        ==========
        self: an ROI_bg instance which must include:

            self.whole_trace_all_epochsTrials: dict including all trials and epochs as defined by 
                                            the stimulus epochs.
        Returns
        =======
        
        SNR_max_matrix : np array
            SNR values for all ROIs.
            
        Corr_matrix : np array
            SNR values for all ROIs.
            
        """
        # extract the type of stimulus. for drifting edges we need a reliability for ON and for OFF

        try:
            self.whole_traces_all_epochsTrials
        except AttributeError:
            raise exception ('no raw trial traces in ROI instance, traces needed to compute reliability')

        self.reliability_all_epochs={}
        self.reliability_all_epochs_ON={}
        self.reliability_all_epochs_OFF={}
        copy_traces=copy.deepcopy(self.whole_traces_all_epochsTrials)
            
        for iEpoch, iEpoch_index in enumerate(copy_traces):
            
            trial_numbers = np.shape(copy_traces[iEpoch_index])[1]
            currentRespTrace =  copy_traces[iEpoch_index][:,:]            
            if (np.isnan(currentRespTrace[1]))[-1]:
                currentRespTrace = []
            
                for validx, values in enumerate(copy_traces[iEpoch_index]):
                    #appends = np.array(filter(lambda v: v==v, values))
                    appends = values[~np.isnan(values)]
                    currentRespTrace.append(appends)
                currentRespTrace = np.asarray(currentRespTrace, dtype=np.float32)
                trial_numbers = np.shape(currentRespTrace)[1]
            # Reliability between all possible combinations of trials

            perm = permutations(range(trial_numbers), 2) 
            coeff =[]

            if self.stim_info['stim_name']=='DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_80sec.txt':
                # note: for this stimulus there is a flash and then the off edge goes first in the first trial 
                coeff_all_trace=[]
                coeff_on=[]
                coeff_off=[]
                for _, pair in enumerate(perm):
                    stim_response_timedelay= 0.45 # this is a hardcoded delay between the stimulus 
                                        #and the peak of a t4t5 axon, Burak gur calculated this number (seconds)
                    shift=int(stim_response_timedelay*self.imaging_info['frame_rate'])+1 # calculate number of frames that include the delay time required, since this is float, approximate to nearest higher int

                    shifted_trace1=np.roll(currentRespTrace[:,pair[0]],-shift,axis=0)
                    shifted_trace2=np.roll(currentRespTrace[:,pair[1]],-shift,axis=0)
                    
                    # checknant1 = np.argwhere(np.isnan(zip(shifted_trace1, shifted_trace2)))
                    checknan = ~(np.isnan(shifted_trace1) | np.isnan(shifted_trace2))

                    if False in checknan:
                        shifted_trace1 = shifted_trace1[checknan]
                        shifted_trace2 = shifted_trace2[checknan]

                    if len(shifted_trace1)!=len(shifted_trace2):
                        if np.abs(len(shifted_trace1)-len(shifted_trace2))>=2:
                            continue
                        elif len(shifted_trace1)>len(shifted_trace2):
                            shifted_trace1=shifted_trace1[:len(shifted_trace2)]
                        elif len(shifted_trace2)>len(shifted_trace1):
                            shifted_trace2=shifted_trace2[:len(shifted_trace1)]
                    if 'contrasts' in self.stim_info['stim_name'] or 'luminances' in self.stim_info['stim_name'] or 'speeds' in self.stim_info['stim_name']:
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                    else:
                        curr_coeff_OFF,_ = pearsonr(shifted_trace1[:len(shifted_trace1)//2],
                                                    shifted_trace2[:len(shifted_trace1)//2])
                        curr_coeff_ON,_ = pearsonr(shifted_trace1[len(shifted_trace1)//2:],
                                                    shifted_trace2[len(shifted_trace1)//2:])
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                        
                        coeff_on.append(curr_coeff_ON)
                        coeff_off.append(curr_coeff_OFF)
                    coeff_all_trace.append(curr_coeff_general)

                self.reliability_all_epochs_ON[iEpoch_index] = np.array(coeff_on).mean()
                self.reliability_all_epochs_OFF[iEpoch_index] = np.array(coeff_off).mean()
                self.reliability_all_epochs[iEpoch_index] = np.array(curr_coeff_general).mean()
        
            elif self.stim_info['stim_name']=='DriftingStripe_4sec_6sec_edges_80deg_degAz_degEl_Sequential_LumDec_8D_ONEDGEFIRST_80sec.txt'\
                        or self.stim_info['stim_name']=='Mapping_for_MIexp.txt' \
                        or self.stim_info['stim_name']=='Drifting_stripesJF_8dir.txt' \
                        or 'DriftingStripe_4sec' in self.stim_info['stim_name']\
                        or self.stim_info['stim_name']=='DriftingDriftStripe_Sequential_LumDec_8D_15degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name']== 'mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name']== 'Dirfting_edges_JF_8dirs_ON_first.txt'\
                        or self.stim_info['stim_name']== '1_Dirfting_edges_JF_8dirs_ON_first.txt':
                # in this stimulus there is a 5 sec dark period followed by a bright edge
                coeff_all_trace=[]
                coeff_on=[]
                coeff_off=[]
                for _, pair in enumerate(perm):
                    if self.stim_info['stim_name']=='DriftingDriftStripe_Sequential_LumDec_8D_15degoffset_ONEDGEFIRST'\
                        or self.stim_info['stim_name'] =='DriftingDriftStripe_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt'\
                        or self.stim_info['stim_name'] =='mapping_Grating_Sequential_LumDec_8D_0degoffset_ONEDGEFIRST.txt':
                        stim_response_timedelay=0
                        shift=0
                    else:
                        stim_response_timedelay= 0.45 # this is a hardcoded delay between the stimulus 
                                         #and the peak of a t4t5 axon, Burak gur calculated this number (seconds) this is to shift the traces when there is no waiting time (tau) betweeen epochs
                        shift=int(stim_response_timedelay*self.imaging_info['frame_rate'])+1 # calculate number of frames that include the delay time required, since this is float, approximate to nearest higher int

                    shifted_trace1=np.roll(currentRespTrace[:,pair[0]],-shift,axis=0)
                    shifted_trace2=np.roll(currentRespTrace[:,pair[1]],-shift,axis=0)
                    checknan = ~(np.isnan(shifted_trace1) | np.isnan(shifted_trace2))

                    if False in checknan:
                        shifted_trace1 = shifted_trace1[checknan]
                        shifted_trace2 = shifted_trace2[checknan]
                    
                    if False in checknan:
                        shifted_trace1 = shifted_trace1[checknan]
                        shifted_trace2 = shifted_trace2[checknan]
                    if len(shifted_trace1)!=len(shifted_trace2):
                        if np.abs(len(shifted_trace1)-len(shifted_trace2))>=2:
                            continue
                        elif len(shifted_trace1)>len(shifted_trace2):
                            shifted_trace1=shifted_trace1[:len(shifted_trace2)]
                        elif len(shifted_trace2)>len(shifted_trace1):
                            shifted_trace2=shifted_trace2[:len(shifted_trace1)]
                    if 'contrasts' in self.stim_info['stim_name'] or 'luminances' in self.stim_info['stim_name'] or 'speeds' in self.stim_info['stim_name']:
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                    else:
                        curr_coeff_ON,_ = pearsonr(shifted_trace1[:len(shifted_trace1)//2],
                                                    shifted_trace2[:len(shifted_trace1)//2])
                        curr_coeff_OFF,_ = pearsonr(shifted_trace1[len(shifted_trace1)//2:],
                                                    shifted_trace2[len(shifted_trace1)//2:])
                        curr_coeff_general= pearsonr(shifted_trace1,
                                                    shifted_trace2)
                        coeff_on.append(curr_coeff_ON)
                        coeff_off.append(curr_coeff_OFF)
                    coeff_all_trace.append(curr_coeff_general)

                self.reliability_all_epochs_ON[iEpoch_index] = np.array(coeff_on).mean()
                self.reliability_all_epochs_OFF[iEpoch_index] = np.array(coeff_off).mean()
                self.reliability_all_epochs[iEpoch_index] = np.array(curr_coeff_general).mean()
            else:
                # calculate reliability independent for every epoch
                for _, pair in enumerate(perm):
                    curr_coeff,_ = pearsonr(currentRespTrace[:-2,pair[0]],
                                                currentRespTrace[:-2,pair[1]])
                    coeff.append(curr_coeff)
                    
                self.reliability_all_epochs[iEpoch_index] = np.array(coeff).mean()
    def calculate_stim_signal_correlation(self):
        '''uses self.raw_trace and raw stim trace to calculate correlation
            this is an indication of the polarity of a neuron for fff stimuli in particular'''
        if self.experiment_info['analysis_type']=='5sFFF_analyze_save':
            start_id=np.array(self.stim_info['output_data_downsampled']['data']).astype(int)[0]-1
            self.stim_trace_correlation=pearsonr(np.array(self.stim_info['output_data_downsampled']['epoch']),self.raw_trace[start_id:])[0]
    
    def calculate_CSI(self, frameRate = None):
        
        try:
            self.max_response_ON
            if self.experiment_info['analysis_type']=='8D_edges_find_rois_save'\
                or self.experiment_info['analysis_type']=='4D_edges' \
                or self.experiment_info['analysis_type']=='1-dir_ON_OFF':    
                moving_edges=True

        except AttributeError:
            moving_edges=False 
        finally:
            pass
        
        try:
            self.resp_trace_all_epochs
            self.stim_info
            
        except AttributeError:

            raise TypeError('ROI_bg: for finding CSI an ROI needs\
                                resp_trace_all_epochs and stim_info')
        
        if moving_edges==True:
            if self.max_response_OFF< self.max_response_ON:
                off_pd_response=self.max_resp_all_epochs_OFF[self.max_resp_idx_ON]
                self.CSI = np.abs((self.max_response_ON-off_pd_response)/(self.max_response_ON))
                self.CS = 'ON'
            else:
                on_pd_response=self.max_resp_all_epochs_ON[self.max_resp_idx_OFF]
                self.CSI = np.abs((self.max_response_OFF-on_pd_response)/(self.max_response_OFF))
                self.CS = 'OFF'
        else:
            # Find edge epochs
            edge_epochs = np.where(np.array(self.stim_info['stimtype'])=='driftingstripe')[0]
            if len(edge_epochs)==0:
                edge_epochs = np.where(np.array(self.stim_info['stimtype'])=='ADS')[0]

            epochDur= self.stim_info['duration']
            
            self.edge_response = np.max(self.max_resp_all_epochs[edge_epochs])
            # Find the edge epoch with max response
            idx = np.nanargmax(self.max_resp_all_epochs[edge_epochs])
            max_edge_epoch = edge_epochs[idx]
            
            
            raw_trace = self.resp_trace_all_epochs[max_edge_epoch]
            trace = raw_trace
            # Filtering to decrease noise in max detection
    #        b, a = signal.butter(3, 0.3, 'low')
    #        trace = signal.filtfilt(b, a, raw_trace)
            
            half_dur_frames = int((round(self.imaging_info['frame_rate'] * epochDur[max_edge_epoch]))/2)
            
            OFF_resp = np.nanmax(trace[:half_dur_frames])
            ON_resp = np.nanmax(trace[half_dur_frames:])


            CSI = (ON_resp-OFF_resp)/(ON_resp+OFF_resp)
            
            self.CSI = np.abs(copy.deepcopy(CSI))
            if CSI >0:
                self.CS = 'ON'
            else:
                self.CS = 'OFF'
            
    def calculateTFtuning_BF(self):
        self.stim_info['velocity']=np.array(self.stim_info['velocity'])
        self.stim_info['sWavelength']=np.array(self.stim_info['sWavelength'])
        self.stim_info['epoch_frequency']=  np.divide(self.stim_info['velocity'],self.stim_info['sWavelength'])
        self.stim_info['stimtype'] = np.array(self.stim_info['stimtype'])
        self.stim_info['angle']=np.array(self.stim_info['angle'])
        grating_epochs = np.where((self.stim_info['stimtype'] == 'TFgrating') &\
                                (self.stim_info['epoch_frequency'] > 0))[0]
        # # If there are no grating epochs
        if grating_epochs.size==0:
            raise ValueError('ROI_bg: No grating epoch' )
            
        max_grating_epoch=np.nanargmax(self.max_resp_all_epochs[grating_epochs])      
        max_grating_epoch=grating_epochs[max_grating_epoch]            
        
        current_dir = self.stim_info['direction'][max_grating_epoch]
        current_epoch_type = self.stim_info['stimtype'][max_grating_epoch]
        
        # Finding all same direction moving grating epochs
        required_epoch_array = \
                (self.stim_info['direction'] == current_dir) & \
                (self.stim_info['stimtype'] == current_epoch_type)& \
                (self.stim_info['epoch_frequency'] > 0) 
        opposite_epoch_array = \
                (self.stim_info['direction'] == ((current_dir+180) % 360)) & \
                (self.stim_info['stimtype'] == current_epoch_type)& \
                (self.stim_info['epoch_frequency'] > 0) 
                
        self.TF_curve_stim = self.stim_info['epoch_frequency'][required_epoch_array]
        
        self.ND_TF_curve_stim = self.stim_info['epoch_frequency'][opposite_epoch_array]
        # Get it as integer indices
        req_epochs_PD = np.where(required_epoch_array)[0]
        self.TF_curve_resp = self.max_resp_all_epochs[req_epochs_PD]
        
        req_epochs_ND = np.where(opposite_epoch_array)[0]
        self.ND_TF_curve_resp = self.max_resp_all_epochs[req_epochs_ND]
        
        self.BF = self.stim_info['epoch_frequency'][max_grating_epoch]
    
    def define_independent_variable_gratings(self,cycle=0):

        """ defines how to calculate or what to extract as indepnedent variable for specific analyses later on"""

        self.independent_var = self.analysis_params['independent_var'][cycle]
        
        if self.analysis_params['independent_var'][cycle] == 'frequency':

            self.independent_var_vals = np.where(np.array(self.stim_info['stimtype'])=='G',
                        np.divide(self.stim_info['velocity'],self.stim_info['sWavelength']),
                        np.nan)                  
        elif self.analysis_params['independent_var'][cycle] == 'angle':
            self.independent_var_vals = np.where(np.array(self.stim_info['velocity'])>0,
                        self.stim_info[self.independent_var],np.nan)  
    
    def calculate_freq_powers(self,cycle,per_epoch=True):
        ''' for grating stimuli. Use fourier transform. especifically PSD to find the power of signal at 
            especific frequencies, for different epochs'''

        try:
            self.rejected
        except: 
            self.rejected = False

        if self.rejected==False:

            self.define_independent_variable_gratings(cycle)

            signal_trace=self.df_trace[self.stim_info['output_data_downsampled'].index[0]:] # discard the initial 5 seconds of recording
            sampling_frequency=self.imaging_info['frame_rate']
            stim=self.stim_info['output_data_downsampled']['epoch']
            durations=self.stim_info['duration']
            window=np.sum(durations)*sampling_frequency # the window of analysis for frequencies is the duration of a trial
            #TODO compute the psd per epoch!
            if per_epoch==True:
                self.signal_psd_dict={}
                self.stim_psd_dict={}
                freq_power_epochs_=[]
                freq_power_stims_=[]
                self.stim_traces={}
                self.power_measured_at=np.zeros_like(durations)
                self.power_measured_at[:]=np.nan
                self.deltaf_measurement=np.zeros_like(durations)
                self.deltaf_measurement[:] = np.nan
                for epoch in self.resp_trace_all_epochs.keys():
                        independt_var=self.independent_var_vals[epoch]
                        signal_trace=self.resp_trace_all_epochs[epoch]
                        dur=durations[epoch]
                        window=len(signal_trace)
                        stim=np.repeat(0,len(signal_trace))
                        self.stim_traces[epoch]=generate_sinusoid_fromstim(stim,np.array([self.independent_var_vals[epoch]]),sampling_frequency,multiple=False)
                        self.stim_traces[epoch]=(self.stim_traces[epoch]-np.mean(self.stim_traces[epoch]))/np.mean(self.stim_traces[epoch])
                        self.stim_psd_dict[epoch],meas_power_stim,_,_=compute_and_plot_psd(self.stim_traces[epoch], sampling_frequency,  np.array([independt_var]), np.array([dur]),window)
                        #_,_,_=compute_and_plot_psd((self.stim_traces[epoch]-np.mean(self.stim_traces[epoch]))/np.mean(self.stim_traces[epoch]), sampling_frequency,  np.array([independt_var]), np.array([dur]),window)
                        freq_power_stims_.append(meas_power_stim[0])
                        self.signal_psd_dict[epoch],meas_power,self.power_measured_at[epoch],self.deltaf_measurement[epoch]=compute_and_plot_psd(signal_trace, sampling_frequency, np.array([independt_var]), np.array([dur]),window)
                        freq_power_epochs_.append(meas_power[0])
                        #self.independent_var_vals[epoch]=xmeas_power[0]

                self.freq_power_epochs=np.zeros_like(durations)
                self.freq_power_epochs[:]=np.nan
                self.freq_power_epochs[1:]=freq_power_epochs_
                self.freq_power_stims=np.zeros_like(durations)
                self.freq_power_stims[:]=np.nan
                self.freq_power_stims[1:]=freq_power_stims_
            else:
                stim_trace=generate_sinusoid_fromstim(stim,self.independent_var_vals,sampling_frequency,multiple=True)
                #stim_trace=(stim_trace-np.mean(stim_trace))/np.mean(stim_trace)
                self.stim_psd_dict,self.freq_power_stims,_,_=compute_and_plot_psd(stim_trace, sampling_frequency, self.independent_var_vals, durations,window)
                self.signal_psd_dict,self.freq_power_epochs,self.power_measured_at,self.deltaf_measurement=compute_and_plot_psd(signal_trace, sampling_frequency, self.independent_var_vals, durations,window)
      
    def clear_prev_atributes(self,cycle,attributes_to_retain):
        # Create a set for efficient membership testing
        
        attrs_to_retain_set = set(attributes_to_retain)
        all_attrs = vars(self)
        # Remove attributes not in attrs_to_retain
        try:
            self.cycle1_reliability
            rel_cycle1_exists = True
        except:
            self.cycle1_reliability={}
            rel_cycle1_exists = False
        try:
            self.cycle1_traceinfo
            traceinf_cycle1_exists = True
        except:
            self.cycle1_traceinfo = {}
            traceinf_cycle1_exists = False
        for attr in list(all_attrs):  # We need to make a copy of keys to avoid RuntimeError
            if cycle > 0:
                
                if ('reliability' in attr) and rel_cycle1_exists == False: # reliability is a special variable that should be preserved for the first cycle
                    self.cycle1_reliability.update({attr:getattr(self, attr)})
                # elif (cycle>1 and 'reliability' in attr) and rel_cycle1_exists == False:
                #     self.cycle1_reliability = getattr(self, attr)
                elif (('stim' in attr) or ('df_trace' in attr)) and traceinf_cycle1_exists == False:
                    self.cycle1_traceinfo.update({attr:getattr(self, attr)})
            if attr not in attrs_to_retain_set:
                delattr(self, attr)

def compute_and_plot_psd(signal_trace, sampling_frequency, target_frequencies, duration,window):
    # Compute the frequencies and PSD using Welch's method
    #window=1000#int(window)
    #frequencies, psd = signal.welch(signal_trace, fs=sampling_frequency,window='blackman',noverlap=0, nperseg=duration*sampling_frequency,detrend='linear')
    
    frequencies = np.fft.fftfreq(len(signal_trace), 1/sampling_frequency)
    fft_vals = np.fft.fft(signal_trace)
    psd = np.abs(fft_vals) #just normal fourier values ** 2) #psd normalized by number of samples
    psd = psd/(len(signal_trace)/sampling_frequency)# the aim here is to normalize 
    #signal_trace.tofile(r'C:\\Users\\vargasju\\PhD\\experiments\\2p\\aaa.csv',sep=',')
    # Plot the PSD in the range of 0-5Hz
    fig1=plt.figure()
    gs = GridSpec(1, 2,wspace=0.4,hspace=0.2)
    ax1=fig1.add_subplot(gs[0])
    ax2=fig1.add_subplot(gs[1])
    ax1.plot(frequencies[frequencies>=0], psd[frequencies>=0],color='blue')
    #plt.xlim([0, 5])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (/Hz)') #power spectral density
    ax1.set_title('psd %s Hz'%(target_frequencies))

    ax2.plot(np.array(range(len(signal_trace)))/sampling_frequency,signal_trace,color='blue')
    #plt.xlim([0, 5])
    ax2.set_xlabel('time (S)')
    ax2.set_ylabel('df/f') #power spectral density
    ax2.set_title('signal %s Hz'%(target_frequencies))

    target_psd=np.zeros(len(target_frequencies))
    analized_freq=np.zeros(len(target_frequencies))
    deltaf_epoch=np.zeros(len(target_frequencies))
    for ix,target_frequency in enumerate(target_frequencies):
        if np.isnan(target_frequency):
            analized_freq[ix] = np.nan
            # Extract the PSD at the target frequency
            target_psd[ix] = np.nan
            deltaf_epoch[ix] = np.nan
            continue
        # Find the closest frequency in the computed frequencies to the target frequency
        target_index = np.abs(frequencies - target_frequency).argmin()
        analized_freq[ix]=frequencies[target_index]
        deltaf_epoch[ix]=frequencies[1]
        # Extract the PSD at the target frequency
        target_psd[ix] = psd[target_index]

        #print('The PSD at %sHz is %s '%(target_frequency,target_psd))

    return {'freqs':frequencies,'psd':psd,'window':window,'detrend':'linear'},target_psd,analized_freq,deltaf_epoch

def generate_sinusoid_fromstim(stim, frequencies, fps, multiple=True):
    # Initialize the grating signal with the same length as the stimulus
    stim_grating = np.zeros_like(stim, dtype=float) + 0.5

    # For each unique value in the stimulus (excluding 0)
    for value in np.unique(stim):
        if value != 0:
            # Get the frequency for this value
            frequency = frequencies[int(value)]

            # Generate a sinusoid at this frequency
            sinusoid = 0.5 * np.sin(2 * np.pi * frequency * np.arange(len(stim)) / fps) + 0.5

            coord1=np.where(np.diff((stim == value).astype(int))>0)[0]
            coord2=np.where(np.diff((stim == value).astype(int))<0)[0]
            # Add the sinusoid to the grating signal at the positions where the stimulus equals this value
            
            for ix in range(len(coord1)):
                if ix==len(coord1)-1 and len(coord1)>len(coord2):
                    stim_grating[coord1[ix]:]=sinusoid[:len(stim_grating[coord1[ix]:])]
                else:
                    stim_grating[coord1[ix]:coord2[ix]-1] = sinusoid[:len(stim_grating[coord1[ix]:coord2[ix]-1])]
        if value ==0 and multiple==False:
            frequency = frequencies[int(value)]
            stim_grating = 0.5 * np.sin(2 * np.pi * frequency * np.arange(len(stim)) / fps) + 0.5

    return stim_grating


def compute_local_means(stim, trace, use_second_half=False,mean_=True):
    """
    Compute the local means array based on the stim and trace arrays.

    Parameters:
    - stim (list[int]): The stim array.
    - trace (list[float]): The trace array.
    - use_second_half (bool, optional): If True, the mean is computed using only the second half of the zeros epoch.
                                        If False, the mean is computed using the entire epoch of zeros.
                                        Defaults to False.

    Returns:
    - local_means (list[float]): The computed local means array.
    """
    
    local_means = np.zeros_like(trace)
    unit_start = 0

    for i in range(1, len(stim)):
        if stim[i] != 0 and stim[i-1] == 0:
            zero_indices = [j for j in range(unit_start, i) if stim[j] == 0]
            
            if use_second_half:
                start_idx = zero_indices[len(zero_indices) // 2]
            else:
                start_idx = zero_indices[0]
            if mean_==True:
                current_mean = np.mean(trace[start_idx:i])
            else:
                current_mean = np.min(trace[start_idx:i])
        if stim[i] == 0 and stim[i-1] != 0:
            local_means[unit_start:i] = current_mean
            unit_start = i

    if i == len(stim)-1:
        local_means[unit_start:] = current_mean
    
    return local_means

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def mean_of_initial5secs(signal, frames):
    '''takes raw traces and calculates mean value of first 5 seconds'''
    signal_l=copy.deepcopy(signal)
    signal_0=signal_l[:frames[0]-1]
    signal_0[:]=np.mean(signal_0)
    return np.mean(signal_0)


def calculate_background_ROIs(bg_mask,rois,in_phase=False):
    """
    introduce a background ROI into the roi instances

    parameters
    ===========

    bg_mask: background mask (2d bolean numpy array)

    rois: roi_bg class instance

    In phase: (bool) if True this is intended to compensate  background from the ultima
            that is structured in the y axis. this chooses a background region spanning the same places
            in y as the roi.
    """
    for roi in rois:
        if in_phase:
            y_position=np.where(roi.mask,)[0]
            filter_bg=np.zeros(np.shape(roi.mask))
            filter_bg[y_position,:]=1
            roi.bg_mask = bg_mask*filter_bg
            if np.sum(roi.bg_mask)==0:
                raise Exception ('there is no overlap between background region and roi in y axis')
        else:
            roi.bg_mask = bg_mask
            

def processXmlInfo(data_path):
    '''Extracts the imaging parameters from xml file
    
    Parameter
    ----------
    data_path : str
        Path to TSeries Folder with xml inside
        
    Returns
    ----------
    imaging_info : dictionary
        keys: frame_rate, pixel_size, depth, frame_timings
    '''
    imaging_info={}
    xmlPath = f'{data_path}/{os.path.basename(data_path)}.xml'
    imaging_info['FramePeriod']=getFramePeriod(xmlPath)
    imaging_info['micRelTimes']=getMicRelativeTime(xmlPath)
    imaging_info['rastersPerFrame']=getrastersPerFrame(xmlPath)
    imaging_info['frame_rate']=1/imaging_info['FramePeriod']
    imaging_info['x_size'], imaging_info['y_size'], imaging_info['pixelArea']=getPixelSize(xmlPath)
    imaging_info['layerPosition']=getLayerPosition(xmlPath)
    return imaging_info

def selectManualROIs(image_to_select_from, image_cmap ="gray", ask_name=True, xcorrimg=None, cat_overlay_img=None):
    '''Enables user to select rois from a given image using roipoly module
    
    Parameters
    ----------
    image_to_select_from : numpy.ndarray
        An image to select ROIs from
    image_cmap : str
        matplotlib colormap to desplay image in, default = 'gray'
    ask_name : boolean
        whether to ask for a name for the roi or put a counter for each roi as name, default = True
    
    Returns 
    ----------
    roi_masks : [np.array]
        all rois dimensions
    mask_names : [str] or [int]
        names of ROIS, either have to be defined by user (asl_name=True) or set as number (counter) for each ROI
    '''
    plt.close('all')
    stopsignal, roi_number, iROI, roi_masks, mask_names, signal = 0, 0, 0, [], [], 'notk'
    im_xDim = np.shape(image_to_select_from)[0]
    im_yDim = np.shape(image_to_select_from)[1]
    mask_agg = np.zeros(shape=(im_xDim,im_yDim))
    plt.style.use("dark_background")
    while (stopsignal==0):
        fig = plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(xcorrimg[0].T, cmap='Spectral')
        plt.title('Cross-Correlation Image')
        plt.subplot(1, 2, 2)
        if cat_overlay_img is not None:
            cat_overlay_img[cat_overlay_img==0] = np.nan
            plt.imshow(image_to_select_from, interpolation='nearest', cmap=image_cmap)
            plt.imshow(cat_overlay_img, alpha=0.1,cmap = 'Accent')
        else:
            plt.imshow(image_to_select_from, interpolation='nearest', cmap=image_cmap)
        plt.colorbar()
        curr_agg = mask_agg.copy()
        curr_agg[curr_agg==0] = np.nan
        plt.imshow(curr_agg, alpha=0.3,cmap = 'Accent')
        plt.title(f"Select ROI: ROI{roi_number}")
        plt.show(block=False)
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        if ask_name: 
            mask_name = input("\nEnter the ROI name:\n>> ") 
            
        else:
            mask_name = iROI
        curr_mask = curr_roi.get_mask(image_to_select_from)
        if len(np.where(curr_mask)[0]) ==0 :
            warnings.warn('ROI empty.. discarded.') 
            continue
        mask_names.append(mask_name)
        roi_masks.append(curr_mask)
        mask_agg[curr_mask] += 1
        roi_number += 1
        signal = input("\nPress k for exiting program, otherwise press enter:\n>>")
        if (signal == 'k'):
            stopsignal = 1
    return roi_masks, mask_names, mask_agg

def selectROIs(extraction_type, error_log,image_to_select, xcorrimg=None):
    '''Extraction of ROIS
    
    Parameters
    ----------
    extraction_type : str
        type of ROI selection, check preprocessing.py for more information
    error_log : str
        Path to the processing error logbook
    image_to_select : np.array
        mean image to select ROIS ontop of
    
    Returns
    ----------
    cat_masks : [np.array]
        all categorial masks that where saved in first step
    cat_names : [str]
        names of categorial masks, have to be defined by user
    roi_masks : [np.array]
        all ROI masks that where selected int he second step
    roi_names : [int]
        names of ROIS, set by algo starting from 0
    '''
    # Categories can be used to classify ROIs depending on their location
    # Backgroud mask (named "bg") will be used for background subtraction
    if extraction_type == 'manual':
        plt.close('all')
        plt.style.use("default")
        print('\n\nSelect categories and background')
        cat_masks, cat_names, overlay_img = selectManualROIs(image_to_select, image_cmap="gray", xcorrimg=xcorrimg)
        print('\n\nSelect ROIs')
        roi_masks, roi_names, _ = selectManualROIs(image_to_select, image_cmap="gray",ask_name=False, xcorrimg=xcorrimg, cat_overlay_img=overlay_img)
        return cat_masks, cat_names, roi_masks, roi_names
    else:
        open(error_log, 'a', encoding="utf8").write(f'ROI extraction type not defined >> check documentation')
        open(error_log, 'a', encoding="utf8").write('\n')

def generateROIsImage(roi_masks,im_shape):
    '''cutrs rois from mean image
    
    Parameters
    ----------
    roi_masks : [np.array]
        all rois dimensions
    im_shape : np.shape
        dimension of the mean image to cut rois from (_motavg.tif)
        
    Retrurns
    ----------
    all_rois_image : np.array
        just the rois part of the input image
    '''
    # Generating an image with all clusters
    all_rois_image = np.zeros(shape=im_shape)
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    return all_rois_image

def save_roi_mask(path, image, roi_image, what, masks, mask_names):
    '''makes matplotlib figure to save rois and background as pngs
    
    Parameter
    ----------
    path : str
        path to corresponding TSeries folder
    image : np.arrary
        avarage image of recording (_motavg.tif)
    roi_image : np.array
        image of all rois 
    what : str
        whether ROIS will be saved or Background, 'ROIS' to save ROIS, 'BG' to save the Background
    masks : [np.array]
        multi array with dimensions of masks
    mask_name : [str] or [int]
        names of masks
        
    Output
    ----------
    _ROIS.png or _Categories.png
        saved in corresponding 2_processed_recordings/fly/Tseries/
    '''
    if what=='ROIS':
        cmap='viridis'
    else:
        cmap='Accent'
    plt.close('all')
    plt.style.use("dark_background")
    plt.figure()
    plt.imshow(image, interpolation='nearest', cmap='grey')
    plt.imshow(roi_image, alpha=0.3,cmap = cmap)
    for idx,mask in enumerate(masks):
        name = mask_names[idx]
        all_x=[]
        colsum = np.sum(mask, axis=0)
        for x in mask:
            all_x.append(x.sum())
        x = np.where(colsum>0)[0][0] + len(colsum[colsum>0])/2
        y = np.where(np.array(all_x)>0)[0][0] + len(np.array(all_x)[np.array(all_x)>0])/2
        plt.text(x, y, f'{name}', fontsize=18, color='b')
    plt.title(f"{what}")
    plt.savefig(f'{path}_{what}.png', dpi=400)
    plt.close('all')
    
def save_roi_images(target, mean_image, cat_masks, roi_masks, cat_names, roi_names):
    '''saves roi masks as pngs
    
    Parameters
    ----------
    target : str
        Path to the current TSeries folder
    mean_image : np.array
        image to draw maks onto
    cat_masks : [np.array]
        background masks
    roi_masks : [np.array]
        all ROI masks
    cat_names : [str]
        names of categorial masks, have to be defined by user
    roi_names : [int]
        names of ROIS, set by algo starting from 0
        
    Output
    ----------
    _ROIS.png and _Categories.png
        saved in corresponding 2_processed_recordings/fly/Tseries/
    '''
    all_rois_image = generateROIsImage(roi_masks, np.shape(mean_image))
    bg_image = generateROIsImage(cat_masks, np.shape(mean_image))
    save_roi_mask(target, mean_image, all_rois_image, 'ROIS', roi_masks, roi_names)
    save_roi_mask(target, mean_image, bg_image, 'Categories', cat_masks, cat_names)
    
def generateROIs(roi_masks, category_masks, category_names, source_im,experiment_info =None,imaging_info =None):
    """ Generates ROI instances and adds the category information.

    Parameters
    ----------
    roi_masks : [np.array]
        A list of ROI masks in the form of numpy arrays.
    category_masks: list
        A list of category masks in the form of numpy arrays.
    category_names: [str]
        A list of category names.
    source_im : numpy array
        An array containing a representation of the source image where the ROIs are found
    imaging_info : dictionary
        keys: frame_rate, pixel_size, depth, frame_timings; output of core_pre.processXmlInfo
    
    Returns
    ----------
    rois : list 
        A list containing instances of all ROIS
    """    
    rois = list(map(lambda mask : ROI(mask, experiment_info = experiment_info, imaging_info=imaging_info), roi_masks))
    def assign_region(roi, category_masks, category_names):
        """ Finds which layer the current mask is in"""
        for iLayer, category_mask in enumerate(category_masks):
            if np.sum(roi.mask*category_mask):
                roi.category = category_names[iLayer]
        try:
            roi.category
        except AttributeError:
            roi.category=['No_category']
    # Add information            
    for iROI, roi in enumerate(rois):
        # Regions are assigned if there are category masks and names provided
        assign_region(roi, category_masks, category_names)
        roi.setSourceImage(source_im)
        roi.number_id = iROI # Assign ROIs numbers
    return rois

def getTimeTraces(rois, time_series, df_method = 'mean'):
    '''Computes the time traces of each ROI given a time series with mean of total recording as baseline
    
    Parameters
    ----------
    rois : list 
        A list containing instances of all ROIS, output of core_preprocessing.generateROIs
    time_series : [np.array]
        recording images where background roi was already subtracted
    df_method : str
        method to normalize data, currently only 'mean' implemented
        mean >> data - mean of total recording / mean of total recording
        
    Returns
    ----------
    rois : list 
        same as input, but with raw_trace as well as dff_mean added for each roi
    '''
    # dF/F calculation
    for roi in rois:
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method, stimulus_information = None)
    return rois

################################################ STIMULI HANDLING ################################################
def olf_stim_array(stim_type, fps, target, error_log = None):
    """makes an array of the olfactory stimulus
    
    Parameters
    ----------
    stim_type : str
        type of olfactory stimulus, defined in preprocessing_params.olfactory_stimuli
    error_log : str
        Path to the processing error logbook
    fps : int
        framerate of TSeries
    target : str
        path to save stimulus plot to
        
    Retunrs
    ----------
    protocoll : np.array
        array of the olfactory stimulus over the time of the recording with 0 as off, 1 as maximum and background as in between
    
    Output
    ----------
    _olf_stim.png : plot of used olfactory stimulus, saved in corresponding /2_processed_recordings/fly/tseries/
    """
    stim_info = preprocessing_params.olfactory_stimuli.get(stim_type)
    if stim_type == 'pulse':
        # [pre,stim,post,ISI,repetition]
        width, pre, post, isi = np.ones(int(stim_info[1]*fps),), np.zeros(int(stim_info[0]*fps),),  np.zeros(int(stim_info[2]*fps),), np.zeros(int(stim_info[3]*fps),)
        protocoll = np.hstack((pre, width))
        for rep in range(stim_info[4]-1):
            protocoll = np.hstack((protocoll, post, isi, pre, width))
        protocoll = np.hstack((protocoll, post))
    elif stim_type == 'on':
        if error_log:
            if isinstance(stim_info, int) == False:
                open(error_log, 'a', encoding="utf8").write(f'please define lenth for "on" olfactory stimulus')
                open(error_log, 'a', encoding="utf8").write('\n')
                return
        protocoll = np.hstack((np.zeros(int(1*fps),), np.ones(int((stim_info-2)*fps),), np.zeros(int(1*fps),)))
    else:
        protocoll = None
        return protocoll
    if protocoll is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(np.arange(0,len(protocoll))/fps, protocoll)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('seconds')
        ax.set_ylabel('odorant')
        fig.tight_layout()
        plt.savefig(f'{target}_olf_stim.png', dpi=400)
        plt.ylim(0,1)
        plt.close('all')
        return protocoll   
    
