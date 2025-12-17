import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import napari, os, pickle, core_paths
from pathlib import Path
from skimage.draw import polygon
import pandas as pd
from scipy import signal 
from datetime import date
from itertools import product
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor, as_completed
#################################Setup
#laptop folder
dataset_folder = 'C:/phd/02_twophoton/250611_OA_odor_OL' 
metasheet = "C:/phd/02_twophoton/metadata.xlsx"
today = date.today()
dataset_layout = core_paths.dataset_layout(dataset_folder, today)
dataset_layout.ensure_paths()
metadata = pd.read_excel(metasheet, sheet_name = dataset_layout.name)
#################################Globals and Parameters
CORR_METHODS = ['xcorr', 'corr']
SHUFFLE_METHODS = ["none", 'phase', 'time']
ONLY_PLOTTING = False
OVERWRITE_PLOTS = False
# THRESH = [2, 3, 4, 5] #std above the mean
THRESH = [2]
TRACE = ['whole', 'pulse']
OLF_PROTOCOLL_PARAMS = [5, 5, 5, 5, 20] #[Reps, Stim, Pre, Post, ISI]
dataset_layout.store_globals(CORR_METHODS=CORR_METHODS, SHUFFLE_METHODS=SHUFFLE_METHODS, ONLY_PLOTTING=ONLY_PLOTTING, \
                            OVERWRITE_PLOTS=OVERWRITE_PLOTS, THRESH=THRESH, TRACE=TRACE, \
                            mean_fps=metadata.loc[:, 'fps'].mean(), OLF_PROTOCOLL_PARAMS=OLF_PROTOCOLL_PARAMS)
#################################

class tseries:
    '''
    contains all functions and parameter to apply on one tseries _motCorr.tif stack
    '''
    def __init__(self, tseries_path: str, output_path:str, output_fly:str):
        self.file_path = tseries_path
        self.output = output_path
        self.output_fly = output_fly
        self.name = str(tseries_path).split('\\')[-2]
    
    def load_stack(self, stack:NDArray[np.float32]):
        self.stack = stack
        self.shape = stack.shape
        self.median_stack = np.median(stack, axis=0)
    
    def substract_bg(self, method:str):
        if method == 'frame-wise mean':
            subtracted = np.zeros_like(self.stack, dtype=float)
            for frame_idx in range(self.stack.shape[0]):
                frame_mean = self.stack[frame_idx].mean()
                subtracted[frame_idx] = self.stack[frame_idx] - frame_mean
            status = None
        elif method == 'temporal mean per pixel-frame':
            temporal_mean = self.stack.mean(axis=0)
            subtracted = self.stack - temporal_mean
            status = None
        elif method == 'mean of 20 darkest pixels per pixel-frame':
            darkest_20_indices = np.argpartition(self.median_stack.flatten(), 20)[:20]
            darkest_20_values = self.median_stack.flatten()[darkest_20_indices]
            dark_mean = darkest_20_values.mean()
            subtracted = self.stack - dark_mean
            status=None
        else:
            status = 'Background Substraciton Method not defined'
            subtracted = None
        self.subtracted = subtracted
        self.median_stack_subtracted = np.median(self.stack, axis=0)
        return status

    def randomize_stack(self, seed=None, method :str='phase'):
        """
        Vectorized phase randomization for image stacks.
        """
        try: 
            stack = self.substracted
        except Exception as exc:
            return exc
        if method == 'phase':
            if seed is not None:
                np.random.seed(seed)
            n_frames, height, width = stack.shape
            # Reshape to (n_frames, n_pixels)
            stack_reshaped = stack.reshape(n_frames, -1)
            # FFT along time axis
            fft = np.fft.fft(stack_reshaped, axis=0)
            amplitude = np.abs(fft)
            # Generate random phases for all pixels
            n_pixels = height * width
            random_phase = np.random.uniform(0, 2*np.pi, (n_frames, n_pixels))
            random_phase[0, :] = 0  # DC component
            if n_frames % 2 == 0:
                random_phase[n_frames//2, :] = 0  # Nyquist
            # Reconstruct with random phases
            randomized_fft = amplitude * np.exp(1j * random_phase)
            # Inverse FFT
            shuffled = np.real(np.fft.ifft(randomized_fft, axis=0))
            # Reshape back to original dimensions
            self.stack_shuffled = shuffled.reshape(n_frames, height, width)
            status=None
        elif method == 'time':
            #TODO: implement
            shuffled = stack
            self.stack_shuffled = shuffled
            status=None
        elif method == 'none':
            self.stack_shuffled = None
            status = None
        else:
            status='Randomization Method Not defined'
            self.stack_shuffled = None
        return status
    
    def select_region(self):
        if os.path.exists(self.output_fly /  '_ROIS_skipp.pkl') == False:
            viewer = napari.Viewer()
            T, H, W = self.shape
            image_layer = viewer.add_image(self.median_stack, name='median')
            shapes = viewer.add_shapes(name='ROI', edge_color='cyan', face_color='cyan', opacity=0.2)
            shapes.mode = 'add_polygon'
            napari.run()
            # Convert shapes to mask (union of all drawn shapes)
            roi_mask = np.zeros((H, W), dtype=bool)
            for data, shape_type in zip(shapes.data, shapes.shape_type):
                canvas = np.zeros((H, W), dtype=np.uint8)
                if shape_type in ['polygon', 'path']:
                    rr, cc = polygon(data[:, 0], data[:, 1], shape=(H, W))
                    canvas[rr, cc] = 1
                elif shape_type in ['rectangle', 'ellipse']:
                    y0 = int(np.min(data[:, 0]))
                    y1 = int(np.max(data[:, 0]))
                    x0 = int(np.min(data[:, 1]))
                    x1 = int(np.max(data[:, 1]))
                    canvas[y0:y1+1, x0:x1+1] = 1
                roi_mask = np.logical_or(roi_mask, canvas.astype(bool))
            self.roi_mask_skipp = {"roi_mask": roi_mask}
            with open(self.output_fly /  '_ROIS_skipp.pkl', 'wb') as fo:
                pickle.dump(self.roi_mask_skipp, fo)
            self.substack = self.stack[:, roi_mask]
            self.roi_mask = self.roi_mask
        else:
            with open(self.output_fly /  '_ROIS_skipp.pkl', 'rb') as fi:
                self.roi_mask_skipp = pickle.load(fi)
            self.roi_mask = self.roi_mask_skipp['roi_mask']
            self.substack = self.stack[:, self.roi_mask]
        
    def corr_pixels(self, method:str='xcorr', direction:str='positive', tresh:int=2, trace:str='whole'):
        #cut trace
        #corr
        #filter
        #plot
        if trace == 'whole':
            return None
        elif trace == 'pulse':
            return None
        


        if method == 'xcorr':
            return None
        elif method == 'corr':
            return None
        else:
            return 'correlation method not implemented'


def process_single_tif(args, tif_container, protocol, olf_stim_pulse, dataset_layout):
    #in container: .median_stack_subtracted, .subtracted, .stack, .median_stack, .stack_dimension, name, .output, .file_path
    corr_method, shuffle_method, thresh, trace = args
    tif_container.randomize_stack(method=shuffle_method) #added .stack_shuffled
    T, H, W = tif_container.shape
    #trim olf stim to T
    if protocol.shape[0] < T:
        pad = np.zeros(T - protocol.shape[0], dtype=np.float32)
        olf_stim = np.hstack((protocol, pad))
    elif protocol.shape[0] > T:
        olf_stim = protocol[:T]
    else:
        olf_stim = protocol.copy()
    tif_container.select_region() #added .roi_mask_skipp, .substack, .roi_mask
    for direciton in ['positive', 'negatove']:
        status = tif_container.corr_pixels(method=corr_method, direction=direciton, thresh=thresh, trace=trace)




        if status:
            dataset_layout.log_error('preprocessing', f'{tif_container.name}: {status}')
        continue
    return



def make_olf_protocoll(fps, reps, width, pre, post, isi):
    n_pre = int(pre * fps)
    n_width = int(width * fps)
    n_post = int(post * fps)
    n_isi = int(isi * fps)
    pre = np.zeros(n_pre, dtype=np.float32)
    width = np.ones(n_width, dtype=np.float32)
    post = np.zeros(n_post, dtype=np.float32)
    isi = np.zeros(n_isi, dtype=np.float32)
    protocol = np.hstack((pre, width))
    for rep in range(reps - 1):
        protocol = np.hstack((protocol, post, isi, pre, width))
    protocol = np.hstack((protocol, post))
    pulse_protocol = np.concatenate([np.zeros(n_pre),np.ones(n_width),np.zeros(n_post)])
    return protocol, pulse_protocol

#################################
def main():
    _raw_combos = [list(tup) for tup in product(CORR_METHODS, SHUFFLE_METHODS, THRESH, TRACE)]
    tif_tasks = [[c, s , t, tr]for c, s, t, tr in _raw_combos]
    #[corr_method, shuffle_method, thresh, trace]
    MAX_WORKERS = min(8, len(tif_tasks))
    # MAX_WORKERS = 1
    workers_to_use = min(len(tif_tasks), MAX_WORKERS)
    conditions = dataset_layout.fetch_all_conditions()
    for condition in conditions:
        #TODO: CURRENTLY SKIPPING TOM RECS
        if condition.endswith("_nan"):
            continue
        if os.path.isdir(dataset_layout.processed / condition):
            print(f'Processing  {condition}')
            region = condition.split('_')[1]
            for fly in os.listdir(dataset_layout.processed / condition):
                if os.path.isdir(dataset_layout.processed / condition / fly):
                    print('Calculating  ', fly)
                    meta_fly = metadata[metadata['fly']==fly]
                    meta_fly_region = meta_fly[meta_fly['region']==region]
                    fps = meta_fly_region['fps'].values[0]
                    olf_params = dataset_layout.olf_protocoll_params
                    olf_stim, olf_stim_pulse = make_olf_protocoll(fps, olf_params[0], olf_params[1], olf_params[2], olf_params[3], olf_params[4])
                    tif_files = sorted(Path(dataset_layout.processed / condition / fly).rglob("*_motCorr.tif"))
                    print(f"Found {len(tif_files)} TIF files")
                    for tif_idx, stack_path in enumerate(tif_files):
                        print(str(stack_path))
                        tseries_container = tseries(tseries_path=stack_path, output_path=Path(dataset_layout.results / condition / fly / str(stack_path).split('\\')[-2]),\
                                                    output_fly = Path(dataset_layout.results / condition / fly))
                        tseries_container.load_stack(stack=tiff.imread(str(stack_path)).astype(np.float32))
                        status = tseries_container.substract_bg(method='frame-wise mean')
                        if status:
                            dataset_layout.log_error('preprocessing', f'{tseries_container.name}: {status}')
                        print(f"{tseries_container.name}:  Processing {len(tif_tasks)} tasks with {workers_to_use} workers")
                        with ProcessPoolExecutor(max_workers=workers_to_use) as executor:
                            future_to_neuron = {executor.submit(process_single_tif, args, tseries_container, olf_stim, olf_stim_pulse, dataset_layout): args[0] for args in tif_tasks}
                            for future in as_completed(future_to_neuron):
                                # connectome_type = future_to_neuron[future]
                                try:
                                    sucess = future.result()
                                    # centrality_result[sucess[1][0]][sucess[1][1]] = sucess[0]
                                except Exception as exc:
                                    dataset_layout.log_error('preprocessing', f"error processing correlations for {tseries_container.name}: {exc}\n")       
                        #TODO: first calc and plot all for single tseries
        #TODO: then plot all avg (fly, cond, etc. in seperate loop. So we can skipp the first one)
if __name__ == "__main__":
    main()



a=1