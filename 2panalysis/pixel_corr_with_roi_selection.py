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
from collections import defaultdict
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib as mpl
#################################Setup
#laptop folder
dataset_folder = 'C:/phd/02_twophoton/250611_OA_odor_OL' 
metasheet = "C:/phd/02_twophoton/metadata.xlsx"
today = date.today()
dataset_layout = core_paths.dataset_layout(dataset_folder, today)
dataset_layout.ensure_paths()
metadata = pd.read_excel(metasheet, sheet_name = dataset_layout.name)
#################################Globals and Parameters
# CORR_METHODS = ['xcorr', 'corr', 'rand']
CORR_METHODS = ['xcorr']
SHUFFLE_METHODS = ["none", 'phase'] #time
ONLY_PLOTTING = True
OVERWRITE_PLOTS = False
# THRESH = [2, 3, 4] #std above the mean
THRESH = [2]
# THRESH = [3]
# THRESH = [4]
# THRESH = [5]
# TRACE = ['whole', 'pulse']
TRACE = ['pulse']
OLF_PROTOCOLL_PARAMS = [5, 5, 5, 5, 20] #[Reps, Stim, Pre, Post, ISI]
only_plotting = True
dataset_layout.store_globals(CORR_METHODS=CORR_METHODS, SHUFFLE_METHODS=SHUFFLE_METHODS, ONLY_PLOTTING=ONLY_PLOTTING, \
                            OVERWRITE_PLOTS=OVERWRITE_PLOTS, THRESH=THRESH, TRACE=TRACE, \
                            mean_fps=metadata.loc[:, 'fps'].mean(), OLF_PROTOCOLL_PARAMS=OLF_PROTOCOLL_PARAMS)
#################################
#TODO: distrplot
#           - bin size to 1 frame
#TODO: plots
#           - 
class tseries:
    '''
    contains all functions and parameter to apply on one tseries _motCorr.tif stack
    '''
    def __init__(self, tseries_path: str, output_path:str, output_fly:str, fps:float):
        self.file_path = tseries_path
        self.output = output_path
        self.output_fly = output_fly
        self.name = str(tseries_path).split('\\')[-2]
        self.fps = fps
    
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
            stack = self.subtracted
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
            # Random numbers of the same shape → argsort along the time axis
            rand_order = np.argsort(np.random.rand(*stack.shape), axis=0)
            shuffled = np.take_along_axis(stack, rand_order, axis=0)
            self.stack_shuffled = shuffled
            status=None
        elif method == 'none':
            self.stack_shuffled = stack
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
            # self.substack = self.stack_shuffled[:, roi_mask]
            self.roi_mask = roi_mask
        else:
            with open(self.output_fly /  '_ROIS_skipp.pkl', 'rb') as fi:
                self.roi_mask_skipp = pickle.load(fi)
            self.roi_mask = self.roi_mask_skipp['roi_mask']
            # self.substack = self.stack_shuffled[:, self.roi_mask]

    def corr_pixels(self, dataset_layout, *, method='xcorr', direction='above', thresh=2, trace='whole', max_lag=None):
        """
        Unified function to compute correlation or cross-correlation between pixel traces and stimulus.
        Results are saved as instance attributes.
        
        Parameters
        ----------
        dataset_layout : object
            Contains substack, stack, stim, roi_mask, and protocol parameters
        method : str, default='xcorr'
            'xcorr' for cross-correlation (finds optimal lag) or 'corr' for standard correlation
        direction : str, default='positive'
            'positive' or 'negative' - select positively/negatively correlated pixels
        thresh : float, default=2
            Number of standard deviations above/below mean for thresholding
        trace : str, default='whole'
            'whole' - analyze entire trace, 'pulse' - analyze individual pulse segments
        max_lag : int or None, default=None
            Maximum lag to consider for xcorr (None = use full range, only used when method='xcorr')
        
        Sets Attributes
        ---------------
        dff_results : dict
            Dictionary containing:
                roi_trace : array
                    Mean trace of refined ROI
                roi_mask_refined : array (H, W)
                    Boolean mask of selected pixels
                corr_map : array (H, W)
                    Correlation/cross-correlation map
                lag_map : array (H, W) or None
                    Lag at maximum correlation (only for method='xcorr', else None)
                corr_values_filtered : array
                    Correlation values after thresholding (only valid pixels/segments)
                corr_values_all : array
                    All correlation values (before thresholding)
                lag_corr_dict : dict or None
                    Dictionary with lag as key and list of correlation values as value
                    (only for method='xcorr', else None)
        """
        # Extract data from dataset_layout
        if direction not in ['above', 'below']:
            dataset_layout.log_error('preprocessing', f'{self.name}: direction not implemented')
        if method not in CORR_METHODS:
            dataset_layout.log_error('preprocessing', f'{self.name}: correlation method not implemented')
        if trace not in TRACE:
            dataset_layout.log_error('preprocessing', f'{self.name}: trace method not implemented')
        substack = self.substack
        stack = self.stack_shuffled
        stim = dataset_layout.olf_stim
        roi_mask = self.roi_mask
        T = substack.shape[0]
        n_pixels = roi_mask.sum()
        H, W = roi_mask.shape
        fps = self.fps
        self.corr_results = {}
        # === TRACE SEGMENTATION ===
        if trace == 'pulse':
            # Extract pulse parameters
            olf_params = dataset_layout.olf_protocoll_params
            reps, n_width, n_pre, n_post, n_isi = olf_params[0], int(olf_params[1]*fps), int(olf_params[2]*fps), int(olf_params[3]*fps), int(olf_params[4]*fps)
            pulse_protocol = dataset_layout.olf_stim_pulse
            pulse_length = n_pre + n_width + n_post
            # Calculate pulse onsets
            onsets = []
            idx = 0
            for r in range(reps):
                idx += n_pre
                onsets.append(idx)
                idx += n_width + (n_post + n_isi if r < reps - 1 else n_post)
            onsets = np.array(onsets, dtype=int)
            # Extract pulse segments
            pulse_segments = []
            for onset in onsets:
                start = onset - n_pre
                end = start + pulse_length
                if end <= T:
                    pulse_segments.append(substack[start:end, :])
            if len(pulse_segments) == 0:
                self._set_empty_corr_results(H, W, pulse_length, method)
                return
            pulse_segments = np.array(pulse_segments)  # (reps, pulse_length, n_pixels)
            stim_ref = pulse_protocol[:pulse_length].astype(float)
            n_reps = len(pulse_segments)
        else:
            # Use whole trace
            pulse_segments = substack[None, :, :]  # (1, T, n_pixels) for uniform handling
            stim_ref = stim[:T].astype(float)
            n_reps = 1
            pulse_length = T
        # === NORMALIZE STIMULUS ===
        stim_ref = (stim_ref - stim_ref.mean()) / (stim_ref.std() + 1e-6)
        # === COMPUTE CORRELATIONS ===
        if method == 'xcorr':
            values, lags = self._compute_xcorr(pulse_segments, stim_ref, n_reps, n_pixels, pulse_length, max_lag, direction)
            # Flatten for consistent structure
            self.corr_results['corr_values_all'] = values.flatten()
            lags_flat = lags.flatten()
            # Build lag-correlation dictionary
            self.corr_results['lag_corr_dict'] = {}
            for lag_val, corr_val in zip(lags_flat, self.corr_results['corr_values_all']):
                if lag_val not in self.corr_results['lag_corr_dict']:
                    self.corr_results['lag_corr_dict'][lag_val] = []
                self.corr_results['lag_corr_dict'][lag_val].append(corr_val)
            # Convert lists to arrays for easier processing
            for lag_key in self.corr_results['lag_corr_dict']:
                self.corr_results['lag_corr_dict'][lag_key] = np.array(self.corr_results['lag_corr_dict'][lag_key])
            self.corr_results['lag_map'] = np.zeros((H, W), dtype=float)
        elif method == 'corr':
            values = self._compute_corr(pulse_segments, stim_ref, n_reps, n_pixels)
            # Flatten for consistent structure
            self.corr_results['corr_values_all'] = values.flatten()
            self.corr_results['lag_corr_dict'] = None
            self.corr_results['lag_map'] = None
            self.corr_results['all_optimal_lags'] = None
            lags = None
        elif method == 'rand':
            self.corr_results['corr_values_all'] = None
            self.corr_results['lag_corr_dict'] = None
            self.corr_results['lag_map'] = None
            self.corr_results['all_optimal_lags'] = None
            lags = None
        # === THRESHOLD AND SELECT ===
        if method != 'rand':
            values_flat = values.flatten()
            thr_high = values_flat.mean() + (thresh * values_flat.std())
            thr_low = values_flat.mean() - (thresh * values_flat.std())
        if trace == 'pulse':
            if method == 'rand':
                rng = np.random.default_rng(42)
                valid_mask = rng.random((n_reps, n_pixels)) < 0.05
                rep_idx, pix_idx = np.where(valid_mask) 
                valid_segments = pulse_segments[rep_idx, :, pix_idx]
                # valid_segments = pulse_segments[valid_mask]   
                self.corr_results['roi_trace'] = valid_segments.mean(axis=0)
                self.corr_results['corr_values_filtered'] = None
                self.corr_results['roi_mask_refined']  = np.zeros((H, W), dtype=bool)
                selected_pixels = valid_mask.any(axis=0)
                self.corr_results['roi_mask_refined'][roi_mask] = selected_pixels
                self.corr_results['corr_map'] = None  
                return
            # Per pulse-pixel thresholding
            if direction == 'above':
                valid_mask = values >= thr_high
            else:
                valid_mask = values <= thr_low
            
            # Collect valid segments
            valid_segments = []
            valid_indices = []
            for rep_idx in range(n_reps):
                for pix_idx in range(n_pixels):
                    if valid_mask[rep_idx, pix_idx]:
                        valid_segments.append(pulse_segments[rep_idx, :, pix_idx])
                        valid_indices.append(rep_idx * n_pixels + pix_idx)
            if len(valid_segments) > 0:
                valid_segments = np.array(valid_segments)
                self.corr_results['roi_trace'] = valid_segments.mean(axis=0)
                # Get filtered correlation values
                valid_indices = np.array(valid_indices)
                self.corr_results['corr_values_filtered'] = self.corr_results['corr_values_all'][valid_indices]
                # Pixel-level summaries
                selected_pixels = valid_mask.any(axis=0)
                # mean_val_per_pixel = np.array([
                #     values[valid_mask[:, i], i].mean() if valid_mask[:, i].any() else 0 for i in range(n_pixels)])
                mean_val_per_pixel = np.array([values[:, i].mean() for i in range(n_pixels)])
                if method == 'xcorr':
                    # mean_lag_per_pixel = np.array([
                    #     lags[valid_mask[:, i], i].mean() if valid_mask[:, i].any() else 0 for i in range(n_pixels)])
                    mean_lag_per_pixel = np.array([lags[:, i].mean() for i in range(n_pixels)])
                    self.corr_results['lag_map'][roi_mask] = mean_lag_per_pixel
                    self.corr_results['all_optimal_lags'] = lags[valid_mask].flatten()
            else:
                self._set_empty_corr_results(H, W, pulse_length, method)
                return
        else:
            # Whole trace - per pixel thresholding
            if method == 'rand':
                rng = np.random.default_rng(42)
                valid_mask = rng.random(pulse_segments.shape[2]) < 0.05
                self.corr_results['roi_trace'] = stack[:, roi_mask][:, valid_mask].mean(axis=1)
                self.corr_results['corr_values_filtered'] = None
                self.corr_results['roi_mask_refined']  = np.zeros((H, W), dtype=bool)
                self.corr_results['roi_mask_refined'][roi_mask] = valid_mask
                self.corr_results['corr_map'] = None
                return
            if direction == 'above':
                valid_mask = values_flat >= thr_high
            else:
                valid_mask = values_flat <= thr_low
            
            selected_pixels = valid_mask
            mean_val_per_pixel = values_flat
            # Get filtered values
            self.corr_results['corr_values_filtered'] = values_flat[valid_mask]
            # Calculate mean trace
            if selected_pixels.any():
                self.corr_results['roi_trace'] = stack[:, roi_mask][:, selected_pixels].mean(axis=1)
            else:
                self.corr_results['roi_trace'] = np.zeros(T)
            if method == 'xcorr':
                lags_flat = lags.flatten()
                mean_lag_per_pixel = lags_flat
                self.corr_results['lag_map'][roi_mask] = mean_lag_per_pixel
                self.corr_results['all_optimal_lags'] = lags_flat[valid_mask].flatten()
        # === BUILD OUTPUT MAPS ===
        self.corr_results['roi_mask_refined']  = np.zeros((H, W), dtype=bool)
        self.corr_results['roi_mask_refined'][roi_mask] = selected_pixels
        self.corr_results['corr_map'] = np.zeros((H, W), dtype=float)
        self.corr_results['corr_map'][roi_mask] = mean_val_per_pixel

    def _compute_corr(self, pulse_segments, stim_ref, n_reps, n_pixels):
        """Compute standard correlation."""
        values = np.zeros((n_reps, n_pixels))
        for rep_idx in range(n_reps):
            trace_mat = pulse_segments[rep_idx, :, :]
            trace_mat = (trace_mat - trace_mat.mean(axis=0, keepdims=True)) / \
                        (trace_mat.std(axis=0, keepdims=True) + 1e-6)
            values[rep_idx, :] = (trace_mat * stim_ref[:, None]).mean(axis=0)
        return values

    def _compute_xcorr(self, pulse_segments, stim_ref, n_reps, n_pixels, pulse_length, max_lag, direction):
        """Compute cross-correlation and find optimal lags."""
        values = np.zeros((n_reps, n_pixels))
        lags_array = np.zeros((n_reps, n_pixels), dtype=int)
        for rep_idx in range(n_reps):
            trace_mat = pulse_segments[rep_idx, :, :]
            trace_mat = (trace_mat - trace_mat.mean(axis=0, keepdims=True)) / (trace_mat.std(axis=0, keepdims=True) + 1e-6)
            for pix_idx in range(n_pixels):
                xcorr = signal.correlate(trace_mat[:, pix_idx], stim_ref, mode='same')
                xcorr = xcorr / pulse_length
                # Define lags
                n_lags = len(xcorr)
                lags = np.arange(-(n_lags // 2), (n_lags // 2) + (n_lags % 2))
                # Apply max_lag filter (restrict to positive lags <= max_lag)
                if max_lag is not None:
                    valid = (lags > 0 ) & (lags <= max_lag)
                    # valid = (lags > -max_lag ) & (lags <= max_lag)
                    xcorr = xcorr[valid]
                    lags = lags[valid]
                # Find max/min
                if direction == 'positive':
                    max_idx = np.argmax(xcorr)
                else:
                    max_idx = np.argmin(xcorr)
                values[rep_idx, pix_idx] = xcorr[max_idx]
                lags_array[rep_idx, pix_idx] = lags[max_idx]
        return values, lags_array

    def _set_empty_corr_results(self, H, W, length, method):
        """Set empty results as instance attributes."""
        self.corr_results['roi_trace'] = np.zeros(length)
        self.corr_results['roi_mask_refined'] = np.zeros((H, W), dtype=bool)
        self.corr_results['corr_map'] = np.zeros((H, W), dtype=float)
        self.corr_results['corr_values_filtered'] = np.array([])
        self.corr_results['corr_values_all'] = np.array([])
        if method == 'xcorr':
            self.corr_results['lag_map'] = np.zeros((H, W), dtype=float)
            self.lag_corr_dict = {}
        else:
            self.corr_results['lag_map'] = None
            self.lag_corr_dict = None

    
    def calculate_dff(self, dataset_layout):
        """
        Calculate dF/F for the ROI trace.
        
        Parameters
        ----------
        dataset_layout : object
            Contains stack, roi_mask_refined, roi_trace, and protocol parameters
        fps : float or None
            Frames per second
        
        Sets Attributes
        ---------------
        dff_results : dict
            Dictionary containing:
            - 'dff_trace': full dF/F trace
            - 'baseline': baseline values used for normalization
            - 'pulse_mean': mean dF/F across pulses (or single pulse for trace='pulse')
            - 'pulse_std': std of dF/F across pulses (or None for trace='pulse')
            - 'pulse_sem': SEM of dF/F across pulses (or None for trace='pulse')
            - 'individual_pulses': array of individual pulse dF/F traces (or None for trace='pulse')
            - 'pulse_protocol': pulse protocol array
            - 'n_pulses': number of pulses (1 for trace='pulse')
            - 'n_pixels': number of pixels in refined ROI
            - 'fps': frames per second
        """
        # Extract data
        fps = self.fps
        stack = self.stack_shuffled
        roi_mask_refined = self.corr_results['roi_mask_refined']
        roi_trace = self.corr_results['roi_trace']
        # Count pixels in refined ROI
        roi_pixels = stack[:, roi_mask_refined]
        n_pixels_original = roi_pixels.shape[1]
        olf_params = dataset_layout.olf_protocoll_params
        reps, n_width, n_pre, n_post, n_isi = olf_params[0], int(olf_params[1]*fps), int(olf_params[2]*fps), int(olf_params[3]*fps), int(olf_params[4]*fps)
        pulse_protocol = dataset_layout.olf_stim_pulse
        pulse_length = n_pre + n_width + n_post
        # Check if roi_trace is pulse-length (cut pulse) or full trace
        is_pulse_trace = len(roi_trace) == pulse_length
        if is_pulse_trace:
            # Single pulse - use pre-stimulus as baseline
            baseline = roi_trace[:n_pre].mean()
            baseline_array = np.full_like(roi_trace, baseline)
            dff = (roi_trace - baseline) / (baseline + 1e-6)
            # For pulse trace, there's only one pulse (already averaged)
            self.dff_results = {
                'dff_trace': dff,
                'baseline': baseline_array,
                'pulse_mean': dff,
                'pulse_std': None,
                'pulse_sem': None,
                'individual_pulses': None,
                'pulse_protocol': pulse_protocol[:pulse_length],
                'n_pulses': 1,
                'n_pixels': n_pixels_original,
                'fps': fps}
        else:
            # Calculate pulse onsets
            onsets = []
            idx = 0
            for r in range(reps):
                idx += n_pre
                onsets.append(idx)
                idx += n_width + (n_post + n_isi if r < reps - 1 else n_post)
            onsets = np.array(onsets, dtype=int)
            # Calculate baseline for each pulse period
            baseline = np.zeros_like(roi_trace)
            for onset in onsets:
                start = max(0, onset - n_pre)
                base = roi_trace[start:onset].mean()
                baseline[onset:onset+n_width] = base
            # Fill any zeros in baseline with forward fill
            mask_nan = baseline == 0
            if np.any(mask_nan):
                last = roi_trace[:n_pre].mean()
                for t in range(len(roi_trace)):
                    if baseline[t] == 0:
                        baseline[t] = last
                    else:
                        last = baseline[t]
            dff = (roi_trace - baseline) / (baseline + 1e-6)
            # Extract individual pulses
            pulses = []
            for onset in onsets:
                start = onset - n_pre
                end = start + pulse_length
                if end <= len(dff):
                    pulse = dff[start:end]
                    pulses.append(pulse)
            # Calculate pulse statistics
            if pulses:
                pulses_array = np.array(pulses)
                pulse_avg = pulses_array.mean(axis=0)
                pulse_std = pulses_array.std(axis=0)
                pulse_sem = pulses_array.std(axis=0) / np.sqrt(len(pulses))
            else:
                pulses_array = None
                pulse_avg = np.array([])
                pulse_std = np.array([])
                pulse_sem = np.array([])
            self.dff_results = {
                'dff_trace': dff,
                'baseline': baseline,
                'pulse_mean': pulse_avg,
                'pulse_std': pulse_std,
                'pulse_sem': pulse_sem,
                'individual_pulses': pulses_array,
                'pulse_protocol': pulse_protocol[:pulse_length],
                'n_pulses': len(pulses),
                'n_pixels': n_pixels_original,
                'fps': fps}
            
    def plot_single_tif_results(self, direction, dataset_layout, args):
        corr_method, shuffle_method, thresh, trace = args
        dff_results = self.dff_results
        os.makedirs(self.output / str(thresh) , exist_ok=True)
        fig1, axs = plt.subplots(1, 4, figsize=(14, 4),constrained_layout=True)
        axs[0].imshow(self.median_stack, cmap='gray')
        axs[0].set_title('Median frame (for ROI)')
        axs[1].imshow(self.median_stack_subtracted, cmap='gray')
        overlay = np.zeros((*self.median_stack_subtracted.shape, 4))
        overlay[self.roi_mask, :] = [0, 1, 1, 0.3] 
        axs[1].imshow(overlay)
        axs[1].set_title('Selected napari region on median (subtracted) frame')
        if corr_method != 'rand':
            display1 = self.corr_results['corr_map'].copy()
            display1[~self.roi_mask] = np.nan
            display = self.corr_results['corr_map'].copy()
            display[~self.corr_results['roi_mask_refined']] = np.nan
            all_vals = np.concatenate([display.ravel(), display1.ravel()])
            vmin = np.nanmin(all_vals)
            vmax = np.nanmax(all_vals)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if corr_method == 'rand':
            axs[2].imshow(self.median_stack_subtracted, cmap='gray')
            axs[2].imshow(overlay)
            axs[2].set_title('spaceholder')
            im4 = axs[3].imshow(self.corr_results['roi_mask_refined'], cmap='magma')
        else:
            im3 = axs[2].imshow(display1, cmap='magma', norm=norm)
            axs[2].set_title('Correlation map inside drawn ROI')
            im4 = axs[3].imshow(display, cmap='magma', norm=norm)
        axs[3].set_title('Refined ROI mask (corr-threshold)')
        cbar = fig1.colorbar(im4, ax=[axs[2], axs[3]],fraction=0.046, pad=0.04)
        cbar.set_label('Correlation coefficient')
        fig1.suptitle(f'{direction} - ROI Selection',fontsize=16, fontweight='bold')
        # fig1.subplots_adjust(top=0.88)
        fig1.savefig(self.output / str(thresh)  / f"{corr_method}_{shuffle_method}_{trace}_{direction}_roi_selection.png", dpi=400)
        plt.close(fig1)
        # Save dF/F trace figure
        if trace == 'whole':
            fig2 = plt.figure(figsize=(12, 4))
            stim = dataset_layout.olf_stim
        else:
            fig2 = plt.figure(figsize=(6, 6))
            stim = self.dff_results['pulse_protocol']
        len_in_s = len(dff_results['dff_trace']) / dff_results['fps']
        x_in_s = np.arange(0, len_in_s, 1/dff_results['fps'])
        x_in_s = x_in_s[:len(dff_results['dff_trace'])]
        plt.plot(x_in_s, dff_results['dff_trace'], color='k', linewidth=1.2)
        plt.plot(x_in_s, stim, color='r')
        plt.title(f'ROI mean trace dF/F with stimulus epochs (n={dff_results["n_pixels"]} pixels), \n {direction}({shuffle_method} Randomized)')
        plt.xlabel('Frame')
        plt.ylabel('dF/F')
        if direction == 'above':
            plt.ylim(-5,10)
        else:
            plt.ylim(-10,5)
        plt.xlim(0,len_in_s)
        plt.tight_layout()
        fig2.savefig(self.output / str(thresh) / f"{corr_method}_{shuffle_method}_{trace}_{direction}_dff_trace.png", dpi=400)
        plt.close(fig2)
        if trace == 'whole':
            pulses_array = dff_results['individual_pulses']
            pulse_avg = dff_results['pulse_mean']
            pulse_protocol = dff_results['pulse_protocol']
            pulse_sem = dff_results['pulse_sem']
            pulses = dff_results['n_pulses']
            # Save pulse average figure
            len_in_s = len(pulse_avg) / dff_results['fps']
            x_in_s = np.arange(0, len_in_s, 1/dff_results['fps'])
            x_in_s = x_in_s[:len(pulse_avg)]
            fig3 = plt.figure(figsize=(6, 12))
            plt.subplot(2, 1, 1)
            if pulses_array is not None:
                for i, pulse in enumerate(pulses_array):
                    plt.plot(x_in_s, pulse, 'grey', alpha=0.3, linewidth=0.8)
            plt.plot(x_in_s, pulse_avg, 'k', linewidth=2, label='Average across pulses')
            plt.plot(x_in_s, pulse_protocol, 'r', linewidth=1.5, label='Stimulus')
            plt.title(f'Individual pulses and average (n={pulses} pulses), \n {direction}({shuffle_method} Randomized)')
            plt.ylabel('dF/F')
            if direction == 'above':
                plt.ylim(-5,10)
            else:
                plt.ylim(-10,5)
            plt.xlim(0,len_in_s)
            plt.legend()
            plt.subplot(2, 1, 2)
            # frames_pulse = np.arange(len(pulse_avg))
            plt.plot(x_in_s, pulse_avg, 'k', linewidth=2, label='Mean')
            plt.fill_between(x_in_s, pulse_avg - pulse_sem, pulse_avg + pulse_sem,
                            alpha=0.3, color='gray', label='SEM')
            plt.plot(x_in_s, pulse_protocol, 'r', linewidth=1.5, label='Stimulus')
            plt.title(f'Average pulse with SEM, \n {direction}({shuffle_method} Randomized)')
            plt.xlabel('Frame (relative to pulse onset)')
            plt.ylabel('dF/F')
            if direction == 'above':
                plt.ylim(-5,10)
            else:
                plt.ylim(-10,5)
            plt.legend()
            plt.tight_layout()
            plt.xlim(0,len_in_s)
            fig3.savefig(self.output / str(thresh) / f"{corr_method}_{shuffle_method}_{trace}_{direction}_pulse_average.png", dpi=400)
            plt.close(fig3)

    def plot_single_tif_maps(self, direction, args):
        """
        Plot spatial maps of lag and cross-correlation
        """
        corr_method, shuffle_method, thresh, trace = args
        if corr_method == 'rand':
            return
        fig, axes = plt.subplots(1, 3, figsize=(9,4),constrained_layout=True)
        #all corr values
        xcorr_display = self.corr_results['corr_map'].copy()
        xcorr_display[~self.roi_mask] = np.nan
        lag_display = self.corr_results['corr_map'].copy()
        lag_display[~self.corr_results['roi_mask_refined']] = np.nan
        all_vals = np.concatenate([xcorr_display.ravel(), lag_display.ravel()])
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        ax2 = axes[0]
        # xcorr_display = self.corr_results['corr_map'].copy()
        # xcorr_display[~self.roi_mask] = np.nan
        im2 = ax2.imshow(xcorr_display, cmap='viridis', aspect='auto', norm=norm)
        ax2.set_title(f'{corr_method} Map')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        # cbar2 = plt.colorbar(im2, ax=ax2)
        # cbar2.set_label('Correlation coefficient')
        #ROI mask
        ax3 = axes[1]
        im3 = ax3.imshow(self.corr_results['roi_mask_refined'], cmap='gray', aspect='auto')
        ax3.set_title(f'Selected ROI ({self.corr_results["roi_mask_refined"].sum()} pixels)')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        #filtered corr values
        ax1 = axes[2]
        # lag_display = self.corr_results['corr_map'].copy()
        # lag_display[~self.corr_results['roi_mask_refined']] = np.nan
        im1 = ax1.imshow(lag_display, cmap='viridis', aspect='auto', norm=norm)
        ax1.set_title(f'{corr_method} Map filtered')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        cbar = fig.colorbar(im1, ax=axes[2])
        cbar.set_label('Correlation coefficient')
        # cbar1 = plt.colorbar(im1, ax=ax1)
        # cbar1.set_label('Corr Values')
        # fig.subplots_adjust(top=0.88)
        fig.suptitle(f'{corr_method} Maps', fontsize=14, fontweight='bold')
        # plt.tight_layout()
        plt.savefig(self.output / str(thresh)  / f"{corr_method}_{shuffle_method}_{trace}_{direction}_corr_maps.png", dpi=400)
        plt.close()
        #lag if its xcorr > shows at which lag pixel has highest/ lowest xcorr
        if corr_method == 'xcorr':  
            fig, axes = plt.subplots(1, 3, figsize=(9,4),constrained_layout=True)
            #all lag
            ax2 = axes[0]
            lag_display = self.corr_results['lag_map'].copy()
            lag_display[~self.roi_mask] = np.nan
            im2 = ax2.imshow(lag_display, cmap='viridis', aspect='auto')
            ax2.set_title(f'Optimal Lag Map')
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            # cbar2 = plt.colorbar(im2, ax=ax2)
            # cbar2.set_label('Lag (Frames)')
            #ROI mask
            ax3 = axes[1]
            ax3.imshow(self.corr_results['roi_mask_refined'], cmap='gray', aspect='auto')
            ax3.set_title(f'Selected ROI ({self.corr_results["roi_mask_refined"].sum()} pixels)')
            ax3.set_xlabel('X (pixels)')
            ax3.set_ylabel('Y (pixels)')
            #filtered lag values
            ax1 = axes[2]
            lag_display = self.corr_results['lag_map'].copy()
            lag_display[~self.corr_results['roi_mask_refined']] = np.nan
            im1 = ax1.imshow(lag_display, cmap='viridis', aspect='auto')
            ax1.set_title(f'Optimal Lag Map filtered')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Lag (Frames)')
            plt.suptitle(f'Lag Maps', fontsize=14, fontweight='bold')
            # plt.tight_layout()
            plt.savefig(self.output / str(thresh)  / f"{corr_method}_{shuffle_method}_{trace}_{direction}_lag_maps.png", dpi=400)
        plt.close('all')
    
def process_single_tif_task(args, tif_container, protocol, olf_stim_pulse, dataset_layout):
    #in container: .median_stack_subtracted, .subtracted, .stack, .median_stack, .stack_dimension, name, .output, .file_path
    corr_method, shuffle_method, thresh, trace = args
    result = {}
    if trace =='pulse' and shuffle_method =='time':
        a=1
    tif_container.randomize_stack(method=shuffle_method) #added .stack_shuffled > stack if no randomizaiton selected >> use .stack_shuffled to continue
    tif_container.substack = tif_container.stack_shuffled[:, tif_container.roi_mask]
    T, H, W = tif_container.shape
    #trim olf stim to T
    if protocol.shape[0] < T:
        pad = np.zeros(T - protocol.shape[0], dtype=np.float32)
        dataset_layout.olf_stim = np.hstack((protocol, pad))
    elif protocol.shape[0] > T:
        dataset_layout.olf_stim = protocol[:T]
    else:
        dataset_layout.olf_stim = protocol.copy()
    dataset_layout.olf_stim_pulse = olf_stim_pulse
    # tif_container.select_region() #added .roi_mask_skipp, .substack, .roi_mask
    for direciton in ['above', 'below']:
        tif_container.corr_pixels(dataset_layout, method=corr_method, direction=direciton, thresh=thresh, trace=trace, max_lag=2.5*tif_container.fps)
        #added .corr_results
        #corr results keys: roi_trace, roi_mask_refined, corr_map, lag_map, corr_values_fitlered, corr_values_all, lag_corr_dict
        tif_container.calculate_dff(dataset_layout) #added .dff_results 
        #dff results keys: dff_trace, baseline, pulse_mean, pulse_std, pulse_sem, individual_pulses, pulse_protocol, n_pulses, n_pixels, fps
        tif_container.plot_single_tif_results(direciton, dataset_layout, args)
        #plots in: dataset / 4_results / condition / fly / tseries /
        tif_container.plot_single_tif_maps(direciton, args)
        result[direciton] = {'corr_results': tif_container.corr_results, 
                            'dff_results': tif_container.dff_results}
    return result, args

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

def calc_fly_avgs(fly_results, dataset_layout, mean_fps):
    averaged_data = nested_dict()
    #name, corr method, shuffle method, thresh, trace method
    for corr in dataset_layout.corr_methods:
        for shuffle in dataset_layout.shuffle_methods:
            for thr in dataset_layout.thresh:
                for tr in dataset_layout.trace:
                    for direction in ['above', 'below']:
                        dff_arrays, dff_all_arrays = [], []
                        for tseries_name, tseries_data in fly_results.items():
                            if direction in tseries_data[corr][shuffle][thr][tr]:
                                dff = tseries_data[corr][shuffle][thr][tr][direction]['dff_results']["pulse_mean"]
                                fps_curr = tseries_data[corr][shuffle][thr][tr][direction]['dff_results']['fps']
                                stim_duration = len(dff)/fps_curr
                                time_curr = np.linspace(0, stim_duration, len(dff))
                                time_ref = np.linspace(0, stim_duration,  int(mean_fps*stim_duration))
                                dff_interpolated = np.interp(time_ref, time_curr, dff)
                                dff_arrays.append(dff_interpolated)

                                dff_all = tseries_data[corr][shuffle][thr][tr][direction]['dff_results']["dff_trace"]
                                stim_duration = len(dff_all)/fps_curr
                                time_curr = np.linspace(0, stim_duration, len(dff_all))
                                time_ref = np.linspace(0, stim_duration,  int(mean_fps*stim_duration))
                                dff_all_interpolated = np.interp(time_ref, time_curr, dff_all)
                                dff_all_arrays.append(dff_all_interpolated)

                        if dff_arrays:
                            # Truncate all arrays to minimum length and stack
                            min_len = min(len(arr) for arr in dff_arrays)
                            dff_stacked = np.array([arr[:min_len] for arr in dff_arrays])

                            min_all_len = min(len(arr) for arr in dff_all_arrays)
                            dff_all_stacked = np.array([arr[:min_all_len] for arr in dff_all_arrays])
                            # Calculate mean across all TSeries (axis=0)
                            avg_results = {"dff_mean" : np.mean(dff_stacked, axis=0), 
                                           'dff_std': np.std(dff_stacked, axis=0), 
                                           'dff_sem' : np.std(dff_stacked, axis=0) / np.sqrt(len(dff_arrays)), 
                                           "dff_all_mean" : np.mean(dff_all_stacked, axis=0), 
                                           'dff_all_std': np.std(dff_all_stacked, axis=0), 
                                           'dff_all_sem' : np.std(dff_all_stacked, axis=0) / np.sqrt(len(dff_all_arrays)), 
                                           "n_tseries" : len(dff_arrays),
                                           "fps" : mean_fps}
                            averaged_data[corr][shuffle][thr][tr][direction] = avg_results
    return averaged_data

def calc_condition_avg(data_dict):
    averaged_data = nested_dict()
    #name, corr method, shuffle method, thresh, trace method
    for fly in data_dict:
        for corr in data_dict[fly]:
            for shuffle in data_dict[fly][corr]:
                for thr in data_dict[fly][corr][shuffle]:
                    for tr in data_dict[fly][corr][shuffle][thr]:
                        for direction in data_dict[fly][corr][shuffle][thr][tr]:
                            # Collect all dff arrays
                            dff_arrays,dff_all_arrays = [], []
                            for fly_name, flys_data in data_dict.items():
                                if direction in flys_data[corr][shuffle][thr][tr]:
                                    dff_arrays.append(flys_data[corr][shuffle][thr][tr][direction]['dff_mean'])
                                    fps = flys_data[corr][shuffle][thr][tr][direction]['fps']
                                    dff_all_arrays.append(flys_data[corr][shuffle][thr][tr][direction]['dff_all_mean'])
                            # Find minimum length to handle different array lengths
                            if dff_arrays:
                                # Truncate all arrays to minimum length and stack
                                min_len = min(len(p) for p in dff_arrays)
                                dff_stacked = np.array([arr[:min_len] for arr in dff_arrays])

                                min_all_len = min(len(arr) for arr in dff_all_arrays)
                                dff_all_stacked = np.array([arr[:min_all_len] for arr in dff_all_arrays])
                                # Calculate mean across all TSeries (axis=0)
                                avg_results = {"dff_mean" : np.mean(dff_stacked, axis=0), 
                                            'dff_std': np.std(dff_stacked, axis=0), 
                                            'dff_sem' : np.std(dff_stacked, axis=0) / np.sqrt(len(dff_arrays)),
                                            "dff_all_mean" : np.mean(dff_all_stacked, axis=0), 
                                            'dff_all_std': np.std(dff_all_stacked, axis=0), 
                                            'dff_all_sem' : np.std(dff_all_stacked, axis=0) / np.sqrt(len(dff_all_arrays)), 
                                            "n_flies" : len(dff_arrays),
                                            "fps" : fps}
                                averaged_data[corr][shuffle][thr][tr][direction] = avg_results
    return averaged_data

def plot_avg_results(args, olf_params):
    pkl_path, result_path = args
    os.makedirs(result_path, exist_ok=True)
    with open(pkl_path, 'rb') as f:
        avg_results = pickle.load(f)
    is_condition = 'condition_avg' in str(pkl_path)
    is_ba = 'BA' in str(pkl_path)
    for corr in avg_results:
        for shuffle in avg_results[corr]:
            for thresh in avg_results[corr][shuffle]:
                for trace in avg_results[corr][shuffle][thresh]:
                    for direction in avg_results[corr][shuffle][thresh][trace]:
                        dff = avg_results[corr][shuffle][thresh][trace][direction]['dff_mean']
                        dff_sem = avg_results[corr][shuffle][thresh][trace][direction]['dff_sem']
                        fps = avg_results[corr][shuffle][thresh][trace][direction]['fps']
                        olf_stim, olf_stim_pulse = make_olf_protocoll(fps, olf_params[0], olf_params[1], olf_params[2], olf_params[3], olf_params[4])
                        if is_condition:
                            n_exp = avg_results[corr][shuffle][thresh][trace][direction]["n_flies"]
                            prefix = 'flies'
                        else:
                            n_exp = avg_results[corr][shuffle][thresh][trace][direction]["n_tseries"]
                            prefix= "tseries"
                        if trace == 'whole':
                            stim = olf_stim
                            pulse_avg = avg_results[corr][shuffle][thresh][trace][direction]['dff_all_mean']
                            pulse_sem = avg_results[corr][shuffle][thresh][trace][direction]['dff_all_sem']
                            fps = avg_results[corr][shuffle][thresh][trace][direction]['fps']
                            frames_pulse = np.arange(len(pulse_avg))
                            fig4, (ax_top, ax_bottom) = plt.subplots(nrows=2,ncols=1,figsize=(12, 4), gridspec_kw={'height_ratios': [1, 4]})
                            ax_bottom.plot(pulse_avg, 'k', linewidth=1.2, label='Mean')
                            ax_bottom.fill_between(frames_pulse, pulse_avg - pulse_sem, pulse_avg + pulse_sem,
                                        alpha=0.3,  zorder=4, color='gray', label='SEM')
                            ax_top.plot(stim, 'r', linewidth=1, label='Stimulus')
                            ax_top.set_title(f'mean across {n_exp} {prefix}, \n {corr},{trace},{direction} {thresh}*std, {shuffle}-randomized')
                            ax_bottom.set_xlabel('Seconds (relative to pulse onset)')
                            ax_top.set_ylabel('Olfactory Stim')
                            ax_bottom.set_ylabel('dF/F')
                            # if direction == 'above':
                            #     ax_bottom.set_ylim(-2.5,5)
                            # else:
                            #     ax_bottom.set_ylim(-5,2.5)
                            ax_top.axis('off')
                            ax_bottom.spines['top'].set_visible(False)
                            ax_bottom.spines['right'].set_visible(False)
                            fig4.subplots_adjust(hspace=0.002) 
                            handles_top,   labels_top   = ax_top.get_legend_handles_labels()
                            handles_bottom,labels_bottom = ax_bottom.get_legend_handles_labels()
                            handles = handles_top + handles_bottom
                            labels  = labels_top  + labels_bottom
                            fig4.subplots_adjust(bottom=0.2)
                            fig4.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, fontsize='medium')
                            frame_interval = fps *15
                            ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(frame_interval))
                            ax_bottom.xaxis.set_major_formatter(
                                ticker.FuncFormatter(lambda x, pos: f'{int(x / fps)}')
                            )
                            ax_bottom.tick_params(axis='x', labelsize=9) 
                            ax_bottom.set_xlim(0,)
                            ax_top.set_xlim(0,)
                            fig4.savefig(Path(result_path / f'{prefix}_avg_ALL_{corr}_{shuffle}_{thresh}_{trace}_{direction}.png'), dpi=400)
                            plt.close(fig4)
                        stim = olf_stim_pulse
                        if len(dff)>400:
                            fig2, ax = plt.subplots(figsize=(12, 4))
                        else:
                            fig2, ax = plt.subplots(figsize=(6, 6))
                        len_in_s = len(dff) / fps
                        x_in_s = np.arange(0, len_in_s, 1/fps)
                        x_in_s = x_in_s[:len(dff)]
                        stim = stim[:len(x_in_s)]
                        plt.plot(x_in_s, dff, color='k', linewidth=1.2, label = 'Mean')
                        ax.fill_between(x_in_s, dff - dff_sem, dff + dff_sem, alpha=0.3, zorder=4, color='gray', label='SEM')
                        x_in_s = x_in_s[:len(stim)]
                        plt.plot(x_in_s, stim, color='r',  linewidth=1, label='Stimulus')
                        plt.title(f'mean across {n_exp} {prefix}, \n {corr},{trace},{direction} {thresh}*std, {shuffle}-randomized')
                        plt.xlabel('Seconds (relative to pulse onset)')
                        plt.ylabel('dF/F')
                        if is_ba == True:
                            # if direction == 'above':
                                plt.ylim(-2.5,10.5)
                            # else:
                            #     plt.ylim(-5,2.5)
                        else:
                            # if direction == 'above':
                                plt.ylim(-2.5,6.5)
                            # else:
                            #     plt.ylim(-5,2.5)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.set_xlim(0,)
                        fig2.subplots_adjust(bottom=0.15)
                        fig2.legend(loc= 'lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, fontsize='medium')
                        fig2.savefig(Path(result_path / f'{prefix}_avg_{corr}_{shuffle}_{thresh}_{trace}_{direction}.png'), dpi=400)
                        plt.close(fig2)
    plt.close("all")
    return

def nested_dict():
    return defaultdict(nested_dict)

def is_nested_dict_empty(d):
    """Recursively check if a nested dict (possibly defaultdict) is empty."""
    if isinstance(d, dict):
        if not d:  # empty dict
            return True
        return all(is_nested_dict_empty(v) for v in d.values())
    else:
        # If it's a leaf (e.g., string, int, etc.), it's not empty
        return False
    
def extract_lag_distr(fly_results):
    """collect values of all tseries for the keys: 'all_optimal_lags', 'lag_corr_dict', 'corr_values_all' 
    then plot it and return dict to collect over flies for condition plots
    """
    tmp = defaultdict(
        lambda: {
            "lag_optimal_all": [],                     # list of 1‑D arrays
            "lag_corr_dict": defaultdict(list) # lag → list of 1‑D arrays
            })   
    for tseries, tdata in fly_results.items():
        # We only care about the 'xcorr' block – skip everything else
        xcorr_block = tdata.get("xcorr")
        if xcorr_block is None:
            continue
        for shuffle, shuffle_block in xcorr_block.items():
            thresh = 2
            thresh_block = shuffle_block[thresh]
            for trace, trace_block in thresh_block.items():
                direction = 'above'
                corr_res = trace_block[direction]["corr_results"]
                #Store the whole list/array of lag_all
                lag_all = np.asarray(corr_res["all_optimal_lags"])
                key = (shuffle, trace)
                tmp[key]["lag_optimal_all"].append(lag_all)
                #Merge the per‑lag arrays
                lag_dict = corr_res["lag_corr_dict"]   # dict[int] → np.ndarray
                for lag, arr in lag_dict.items():
                    tmp[key]["lag_corr_dict"][lag].append(np.asarray(arr))
    aggregated = {}
    for key, containers in tmp.items():
        #Concatenate all lag_all arrays into one 1‑D array
        if containers["lag_optimal_all"]:                             
            lag_all_concat = np.concatenate(containers["lag_optimal_all"])
        else:                                               
            lag_all_concat = np.array([])
        #Concatenate the per‑lag arrays
        lag_corr_merged = {}
        for lag, arr_list in containers["lag_corr_dict"].items():
            if arr_list:
                lag_corr_merged[int(lag)] = np.concatenate(arr_list)
            else:
                lag_corr_merged[int(lag)] = np.array([])
        aggregated[key] = {
            "lag_optimal_all":        lag_all_concat,
            "lag_corr_dict":  lag_corr_merged}
    return aggregated

def plot_distr(fly_aggregated_lag, result_path):
    #lag_all histogram
    os.makedirs(result_path, exist_ok=True)
    for key, entry in fly_aggregated_lag.items():
        title = f"shuffle={key[0]}, trace={key[1]}"
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(1, 2, 2)
        sns.histplot(entry["lag_optimal_all"], bins=50, kde=True,
                    ax=ax1, color="#4c72b0")
        ax1.set_title(f"lag_optimal_all distribution {title}")
        ax1.set_xlabel("lag (time steps)")
        ax1.set_ylabel("count")
        # Values from lag_corr_dict
        # Flatten the dict into two parallel lists:
        #   * x = lag (repeated for each value)
        #   * y = the actual correlation value
        xs, ys = [], []
        for lag, vals in entry["lag_corr_dict"].items():
            xs.extend([lag] * len(vals))
            ys.extend(vals.tolist())
        ax2 = plt.subplot(1, 2, 1)
        sns.stripplot(x=xs, y=ys, jitter=0.25, size=3, ax=ax2,
                    palette="viridis")
        ax2.set_title(f"all lags values {title}")
        ax2.set_xlabel("lag (time steps)")
        ax2.set_ylabel("correlation value")
        plt.tight_layout()
        plt.savefig(Path(result_path / f'lag_distr_{key[0]}_{key[1]}.png'), dpi=400)
        plt.close('all')

def combine_aggregations(list_of_aggregated):
    """
    Takes a list of the dictionaries returned by `extract_lag_distr`
    and merges them into a single aggregation that contains the data from
    *all* flies.
    """
    merged = defaultdict(
        lambda: {
            "lag_optimal_all": [],                     # list of 1‑D arrays
            "lag_corr_dict": defaultdict(list)   # lag → list of 1‑D arrays
        })
    for agg in list_of_aggregated:
        for key, entry in agg.items():
            if entry["lag_optimal_all"].size:          # skip empty entries
                merged[key]["lag_optimal_all"].append(entry["lag_optimal_all"])
            for lag, values in entry["lag_corr_dict"].items():
                merged[key]["lag_corr_dict"][lag].append(values)
    final_agg = {}
    for key, containers in merged.items():
        lag_all_concat = np.concatenate(containers["lag_optimal_all"]) \
                if containers["lag_optimal_all"] else np.array([])
        lag_corr_merged = {}
        for lag, arr_list in containers["lag_corr_dict"].items():
            lag_corr_merged[int(lag)] = np.concatenate(arr_list) \
                    if arr_list else np.array([])
        final_agg[key] = {
            "lag_optimal_all": lag_all_concat,
            "lag_corr_dict": lag_corr_merged}
    return final_agg


#################################
def main():
    mean_fps = metadata.loc[:, 'fps'].mean()
    conditions = dataset_layout.fetch_all_conditions()
    olf_params = dataset_layout.olf_protocoll_params
    if only_plotting == False:
        _raw_combos = [list(tup) for tup in product(CORR_METHODS, SHUFFLE_METHODS, THRESH, TRACE)]
        tif_tasks = [[c, s , t, tr]for c, s, t, tr in _raw_combos]
        #[corr_method, shuffle_method, thresh, trace]
        MAX_WORKERS = min(8, len(tif_tasks))
        # MAX_WORKERS = 1 #for dubugging
        workers_to_use = min(len(tif_tasks), MAX_WORKERS)
        for condition in conditions:
            lag_across_flies = []
            #TODO: CURRENTLY SKIPPING TOM RECS
            if condition.endswith("_nan")  or condition.split('_')[2] != 'Wax':#or condition.endswith("Wax_BA")
                continue
            if os.path.isdir(dataset_layout.processed / condition):
                print(f'Processing  {condition}')
                condition_results = {}
                if os.path.exists(dataset_layout.processed / condition / 'condition_results.pkl'):
                    fly_names = [entry.name for entry in Path(dataset_layout.processed / condition).iterdir() if entry.is_dir()]
                    with open(dataset_layout.processed / condition / 'condition_results.pkl', 'rb') as f:
                            condition_results = pickle.load(f)
                    with open(dataset_layout.processed / condition  / 'condition_avg_results.pkl', 'rb') as f:
                            condition_avg_results = pickle.load(f)
                    if is_nested_dict_empty(condition_results) or is_nested_dict_empty(condition_avg_results):
                        print('Empty results, Reprocessing...')
                        condition_results = {}
                    elif not all(key in condition_results for key in fly_names):
                        print("Not all tseries in fly resutls, Reprocessing...")
                    else:
                        print(f"Skipping {condition}")
                        continue
                region = condition.split('_')[1]
                for fly in os.listdir(dataset_layout.processed / condition):
                    if os.path.isdir(dataset_layout.processed / condition / fly):
                        print('Calculating  ', fly)
                        fly_results = nested_dict()
                        tif_files = sorted(Path(dataset_layout.processed / condition / fly).rglob("*_motCorr.tif"))
                        tif_names = [str(Path(p).parent.name) for p in tif_files]
                        if os.path.exists(dataset_layout.processed / condition / fly / 'fly_results.pkl'):
                            with open(dataset_layout.processed / condition / fly / 'fly_results.pkl', 'rb') as f:
                                    fly_results = pickle.load(f)
                            with open(dataset_layout.processed / condition / fly / 'fly_avg_results.pkl', 'rb') as f:
                                    fly_avg_results = pickle.load(f)
                            if is_nested_dict_empty(fly_results) or is_nested_dict_empty(fly_avg_results):
                                print('Empty results, Reprocessing...')
                                fly_results = nested_dict()
                            elif not all(key in fly_results for key in tif_names):
                                print("Not all tseries in fly resutls, Reprocessing...")
                            else:
                                condition_results[fly] = fly_avg_results
                                print(f"Skipping {fly}")
                                continue
                        meta_fly = metadata[metadata['fly']==fly]
                        meta_fly_region = meta_fly[meta_fly['region']==region]
                        fps = meta_fly_region['fps'].values[0]
                        olf_stim, olf_stim_pulse = make_olf_protocoll(fps, olf_params[0], olf_params[1], olf_params[2], olf_params[3], olf_params[4])
                        print(f"Found {len(tif_files)} TIF files")
                        for tif_idx, stack_path in enumerate(tif_files):
                            tseries_results = nested_dict()
                            name = str(stack_path).split('\\')[-2]
                            if os.path.exists(dataset_layout.processed / condition / fly / name / "tseries_results.pkl"):
                                print("Results found")
                                with open(dataset_layout.processed / condition / fly / name / "tseries_results.pkl", 'rb') as f:
                                    tseries_results = pickle.load(f)
                                if is_nested_dict_empty(tseries_results):
                                    print('Empty results, Reprocessing...')
                                else:
                                    fly_results[name] = tseries_results
                                    print(f"Skipping {name}")
                                    continue
                            print(str(stack_path))
                            tseries_container = tseries(tseries_path=stack_path, output_path=Path(dataset_layout.processed / condition / fly / str(stack_path).split('\\')[-2]),\
                                                        output_fly = Path(dataset_layout.processed / condition / fly), fps=fps)
                            tseries_container.load_stack(stack=tiff.imread(str(stack_path)).astype(np.float32))
                            status = tseries_container.substract_bg(method='frame-wise mean')
                            if status:
                                dataset_layout.log_error('preprocessing', f'{tseries_container.name}: {status}')
                            print(f"{tseries_container.name}:  Processing {len(tif_tasks)} tasks with {workers_to_use} workers")
                            tseries_container.select_region() #added .roi_mask_skipp, .substack, .roi_mask
                            with ProcessPoolExecutor(max_workers=workers_to_use) as executor:
                                future_to_neuron = {executor.submit(process_single_tif_task, args, tseries_container, olf_stim, olf_stim_pulse, dataset_layout): args[0] for args in tif_tasks}
                                for future in as_completed(future_to_neuron):
                                    try:
                                        sucess = future.result()
                                        #success[1] : corr_method, shuffle_method, thresh, trace
                                        #sucess[0] : keys: corr_results, dff_results
                                        tseries_results[sucess[1][0]][sucess[1][1]][sucess[1][2]][sucess[1][3]] = sucess[0]
                                    except Exception as exc:
                                        dataset_layout.log_error('preprocessing', f"error processing correlations for {tseries_container.name}: {exc}\n")
                            with open(dataset_layout.processed / condition / fly / tseries_container.name / 'tseries_results.pkl', "wb") as fo:
                                pickle.dump(tseries_results, fo)    
                            fly_results[tseries_container.name]  = tseries_results
                        if len(fly_results)>0:
                            fly_avg_results = calc_fly_avgs(fly_results, dataset_layout, mean_fps)
                            fly_aggregated_lag_distributions = extract_lag_distr(fly_results)
                            lag_across_flies.append(fly_aggregated_lag_distributions)
                            plot_distr(fly_aggregated_lag_distributions, dataset_layout.results / condition / fly)
                            # fly_avg_results = interpolate_to_mean_fps(fly_avg_results, mean_fps)
                            with open(dataset_layout.processed / condition / fly / 'fly_results.pkl', "wb") as fo:
                                pickle.dump(fly_results, fo) 
                            with open(dataset_layout.processed / condition / fly / 'fly_avg_results.pkl', "wb") as fo:
                                pickle.dump(fly_avg_results, fo) 
                            condition_results[fly] = fly_avg_results
                with open(dataset_layout.processed / condition / 'condition_results.pkl', "wb") as fo:
                    pickle.dump(condition_results, fo)      
                condition_avg_results = calc_condition_avg(condition_results) 
                condition_aggregated_lag_distributions = combine_aggregations(lag_across_flies)
                plot_distr(condition_aggregated_lag_distributions, dataset_layout.results / condition)
                # plot_condition_results(condition_avg_results, dataset_layout) 
                with open(dataset_layout.processed / condition / 'condition_avg_results.pkl', "wb") as fo:
                    pickle.dump(condition_avg_results, fo)  
    plotting_tasks = []
    
    for condition in conditions:
        if os.path.exists(dataset_layout.processed / condition / 'condition_avg_results.pkl'):# and os.path.exists(dataset_layout.results / condition / f'flies_avg_{CORR_METHODS[-1]}_{SHUFFLE_METHODS[-1]}_{THRESH[-1]}_{TRACE[-1]}_above.png') == False:
            plotting_tasks.append((dataset_layout.processed / condition / 'condition_avg_results.pkl', dataset_layout.results / condition)) 
        for fly in os.listdir(dataset_layout.processed / condition):
            if os.path.exists(dataset_layout.processed / condition / fly / 'fly_avg_results.pkl') and os.path.exists(dataset_layout.results / condition / fly / f'tseries_avg_{CORR_METHODS[-1]}_{SHUFFLE_METHODS[-1]}_{THRESH[-1]}_{TRACE[-1]}_above.png') == False:
                plotting_tasks.append((dataset_layout.processed / condition / fly / 'fly_avg_results.pkl', dataset_layout.results / condition / fly)) 
    MAX_WORKERS = min(8, len(plotting_tasks))
    # MAX_WORKERS = 1 #for dubugging
    workers_to_use = min(len(plotting_tasks), MAX_WORKERS)
    if workers_to_use > 0:
        print(f'Plotting {len(plotting_tasks)} Conditions and Flies with {workers_to_use} workers')
        with ProcessPoolExecutor(max_workers=workers_to_use) as executor:
            future_to_neuron = {executor.submit(plot_avg_results, args, olf_params): args[0] for args in plotting_tasks}
            for future in as_completed(future_to_neuron):
                try:
                    sucess = future.result()
                except Exception as exc:
                    dataset_layout.log_error('preprocessing', f"error plotting correlations: {exc}\n")

if __name__ == "__main__":
    main()