import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import napari, os, pickle
from pathlib import Path
from skimage.draw import polygon
import pandas as pd
from scipy import signal
"""
Phase Randomization testet die Null-Hypothese:

"Gibt es eine spezifische zeitliche Beziehung zum Stimulus, die nicht durch die intrinsische Frequenzstruktur des Signals erklärt werden kann?"
"""
'''data matrix called trace_mat with shape T × P:

T = number of time points (frames)
P = number of pixels inside the ROI
one pixel’s trace normalized to zero mean and unit variance

Take one pixel inside the ROI. Across time, it has a fluorescence time series (a vector of length T). That’s that pixel’s “trace.”
You subtract that pixel’s own average over time, so its mean becomes 0. This centers the trace.
You divide by that pixel’s own standard deviation over time, so its variability becomes 1. This scales the trace.


Why do this? Because once both the pixel traces and the stimulus vector are standardized, the dot-product averaged over time becomes the Pearson correlation coefficient for each pixel vs the stimulus.

'''
'''Control 1: shuffle_before_dff

Phase-randomisiert den gesamten Stack VOR der dF/F-Berechnung
Testet: Ob die gefundenen Pixel durch zufällige zeitliche Muster entstehen könnten
Erwartung: Sollte deutlich weniger Pixel finden als echte Daten

Control 2: shuffle_after_filter

Phase-randomisiert die gefilterten Pixel-Traces NACH der Response-Selektion
Testet: Ob die Korrelation zum Stimulus echt ist oder nur durch die dF/F-Änderungen entsteht
Erwartung: Sollte ähnlich viele Pixel finden (weil Response-Filter gleich bleibt), aber niedrigere Korrelation'''


'''What correlation is being computed?
for each pixel inside the original ROI, the Pearson correlation between its fluorescence trace and the stimulus time course.

trace_mat is the time-by-pixel matrix of z-scored fluorescence within the ROI: each column is one pixel’s trace normalized to zero mean and unit variance.
s is the stimulus vector, also z-scored to zero mean and unit variance.
corr = (trace_mat * s[:, None]).mean(axis=0) takes the elementwise product between each pixel’s normalized trace and the normalized stimulus across time, then averages across time. Because both sides are standardized, that mean is exactly the Pearson correlation coefficient between the pixel’s trace and the stimulus.
each entry of corr is the correlation coefficient r in [-1, 1] for one ROI pixel versus the stimulus.

How is the threshold defined for picking ROIs?
 mean + 2·std
thr = corr.mean() + 2.0 * corr.std()

roi_mask_refined[roi_mask] = corr >= thr

If most pixels have low or modest correlation, only the most strongly stimulus-locked pixels (the right tail of the correlation distribution) will pass.

'''
#TODO: Control: no phase, normal randomize (2)
#TODO: Control: sleect random pixels (2)
#TODO: Rewrite: so can run withut plotting all (1)
#TODO: Rewrite: into funcitons and classes so not same stuff x times (1)
#TODO: Rewrite: corr, xcorr, shuffled phase, shuffled time >> all with whoel trace and only pulse
#TODO: Write: plot distribution for xcorr/ corr (storngest one) and xcorr vs lag (all) (3)
#TODO: wmoothin all_pixel;_result_avg traces?
#TODO: run for all (4)
#TODO: Write: Final plots  > Illustrator (5)
      

processed_recordings = 'C:/phd/02_twophoton/250611_OA_odor_OL/2_processed_recordings'
metasheet = r"C:\phd\02_twophoton\metadata.xlsx"
metadata = pd.read_excel(metasheet, sheet_name="250611_OA_odor_OL") 



def phase_randomize_stack_vectorized(stack, seed=None):
    """
    Vectorized phase randomization for image stacks.
    Much faster for large datasets.
    """
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
    return shuffled.reshape(n_frames, height, width)

def phase_randomize_traces(traces, seed=None):
    """
    Phase randomization for 2D array of traces (time x pixels).
    
    Parameters:
    -----------
    traces : array (T, n_pixels)
        Time series for multiple pixels
    seed : int, optional
        Random seed
    
    Returns:
    --------
    array : Phase-randomized traces
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_frames, n_pixels = traces.shape
    
    # FFT along time axis
    fft = np.fft.fft(traces, axis=0)
    amplitude = np.abs(fft)
    
    # Generate random phases for all pixels
    random_phase = np.random.uniform(0, 2*np.pi, (n_frames, n_pixels))
    random_phase[0, :] = 0  # DC component
    if n_frames % 2 == 0:
        random_phase[n_frames//2, :] = 0  # Nyquist
    
    # Reconstruct with random phases
    randomized_fft = amplitude * np.exp(1j * random_phase)
    
    # Inverse FFT
    shuffled = np.real(np.fft.ifft(randomized_fft, axis=0))
    
    return shuffled

def interpolate_to_mean_fps(data_dict, mean_fps):
    """interpolates data of dictionary to mean fps of whole dataset
    
    Parameters:
    -----------
    data_dict: dict
        includes mean of 
    mean_fps: float
        mean fps of whole dataset
    
    Returns
    -------
    data_dict: dict
        adjusted and interpolated data
    """
    interpolated_data_dict= {}
    for direction in data_dict:
        interpolated_data_dict[direction] = {}
        for meassure in data_dict[direction]:
            if meassure=='dff_mean' or meassure == 'pulse_mean':
                dff = data_dict[direction][meassure]
                fps_curr = data_dict[direction]['fps']
                stim_duration = len(dff)/fps_curr
                time_curr = np.linspace(0, stim_duration, len(dff))
                time_ref = np.linspace(0, stim_duration,  int(mean_fps*stim_duration))
                dff_interpolated = np.interp(time_ref, time_curr, dff)
                interpolated_data_dict[direction] = {meassure : dff_interpolated, 'fps_curr':fps, 'fps': mean_fps}

    return interpolated_data_dict

def average_across_flies(data_dict):
    """
    Calculate average of dff and dff_mean_pixels across all TSeries.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with structure: {tseries_name: {'positive': {...}, 'negative': {...}}}
    
    Returns:
    --------
    dict : Averaged data for positive and negative directions
    """
    
    averaged_data = {'positive': {}, 'negative': {}}
    
    for direction in ['positive', 'negative']:
        # Collect all dff arrays
        dff_arrays = []
        for fly_name, flys_data in data_dict.items():
            if direction in flys_data:
                try:
                    dff_arrays.append(flys_data[direction]['dff_mean'])
                    what = 'dff'
                except:
                    dff_arrays.append(flys_data[direction]['pulse_mean'])
                    what = 'pulse'
                fps = flys_data[direction]['fps']
        # Find minimum length to handle different array lengths
        if dff_arrays:
            # Truncate all arrays to minimum length and stack
            min_len = min(len(p) for p in dff_arrays)
            dff_stacked = np.array([arr[:min_len] for arr in dff_arrays])

            # Calculate mean across all TSeries (axis=0)
            averaged_data[direction][f'{what}_mean'] = np.mean(dff_stacked, axis=0)
            averaged_data[direction][f'{what}_std'] = np.std(dff_stacked, axis=0)
            averaged_data[direction][f'{what}_sem'] = np.std(dff_stacked, axis=0) / np.sqrt(len(dff_arrays))
            averaged_data[direction]['n_flies'] = len(dff_arrays)
            averaged_data[direction]['fps'] = fps
    return averaged_data


# --- Timing and protocol ---
def average_across_tseries(data_dict):
    """
    Calculate average of dff and dff_mean_pixels across all TSeries.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with structure: {tseries_name: {'positive': {...}, 'negative': {...}}}
    
    Returns:
    --------
    dict : Averaged data for positive and negative directions
    """
    
    averaged_data = {'positive': {}, 'negative': {}}
    
    for direction in ['positive', 'negative']:
        # Collect all dff arrays
        dff_arrays = []
        for tseries_name, tseries_data in data_dict.items():
            if direction in tseries_data:
                for meassure in data_dict[tseries_name][direction]:
                    if meassure=='dff_mean' or meassure == 'pulse_mean' or meassure == 'dff':
                        dff_arrays.append(tseries_data[direction][meassure])
                fps = tseries_data[direction]['fps']
        # Find minimum length to handle different array lengths
        if dff_arrays:
            # Truncate all arrays to minimum length and stack
            min_len = min(len(arr) for arr in dff_arrays)
            dff_stacked = np.array([arr[:min_len] for arr in dff_arrays])
            if ' dff_mean' in data_dict[tseries_name][direction].keys():
                what = "dff"
            elif "pulse_mean" in data_dict[tseries_name][direction].keys():
                what = "pulse"
            else:
                what = 'dff'
            # Calculate mean across all TSeries (axis=0)
            averaged_data[direction][f'{what}_mean'] = np.mean(dff_stacked, axis=0)
            averaged_data[direction][f'{what}_std'] = np.std(dff_stacked, axis=0)
            averaged_data[direction][f'{what}_sem'] = np.std(dff_stacked, axis=0) / np.sqrt(len(dff_arrays))
            averaged_data[direction]['n_tseries'] = len(dff_stacked)
            averaged_data[direction]['fps'] = fps
    return averaged_data

def make_olf_protocoll(fps):
    reps = 5
    n_pre = int(5 * fps)
    n_width = int(5 * fps)
    n_post = int(5 * fps)
    n_isi = int(20 * fps)
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

def get_corr_pixels(substack, stim, roi_mask,stack, *, direction = "positive"):
    T2 = substack.shape[0]
    trace_mat = substack
    trace_mat = trace_mat - trace_mat.mean(axis=0, keepdims=True)
    stdx = trace_mat.std(axis=0, keepdims=True) + 1e-6
    trace_mat = trace_mat / stdx
    s = stim[:T2].astype(float)
    s = s - s.mean()
    s = s / (s.std() + 1e-6)
    corr = (trace_mat * s[:, None]).mean(axis=0)
    corr_map = np.zeros((H, W), dtype=float)
    corr_map[roi_mask] = corr
    thr_high = corr.mean() + 2.0 * corr.std()
    thr_low = corr.mean() - 2.0 * corr.std()
    roi_mask_refined = np.zeros((H, W), dtype=bool)
    if direction == 'positive':
        roi_mask_refined[roi_mask] = (corr >= thr_high)
    else:
        roi_mask_refined[roi_mask] = (corr <= thr_low)
    roi_trace = stack[:, roi_mask_refined].mean(axis=1)
    return roi_trace, roi_mask_refined, corr_map


def get_xcorr_pixels(substack, stack, stim, roi_mask, *, direction="positive", max_lag=None):
    """
    Compute cross-correlation between pixel traces and stimulus (finds optimal lag)
    
    Args:
        substack: 3D array (time, height, width) - ROI time series
        stim: 1D array - stimulus trace
        roi_mask: 2D bool array (height, width) - pixels to analyze
        direction: 'positive' or 'negative' - select positively/negatively correlated pixels
        max_lag: int or None - maximum lag to consider (None = use full range)
    
    Returns:
        roi_trace: 1D array - mean trace of refined ROI
        roi_mask_refined: 2D bool array - pixels passing threshold
        xcorr_map: 2D array (height, width) - maximum cross-correlation map
        lag_map: 2D array (height, width) - lag at maximum correlation (in frames)
    """
    T2 = substack.shape[0]
    H, W = roi_mask.shape
    
    # Get masked pixels
    n_pixels = roi_mask.sum()
    # masked_traces = substack[:, roi_mask]  # (time, n_pixels)
    
    # Normalize traces
    trace_mat = substack - substack.mean(axis=0, keepdims=True)
    stdx = trace_mat.std(axis=0, keepdims=True) + 1e-6
    trace_mat = trace_mat / stdx
    
    # Normalize stimulus
    s = stim[:T2].astype(float)
    s = s - s.mean()
    s = s / (s.std() + 1e-6)
    
    # Compute cross-correlation for each pixel
    max_corr = np.zeros(n_pixels)
    best_lag = np.zeros(n_pixels, dtype=int)
    
    for i in range(n_pixels):
        # Compute cross-correlation
        xcorr = signal.correlate(trace_mat[:, i], s, mode='same')
        xcorr = xcorr / T2  # Normalize by length to match correlation scale
        
        # Define lag range
        # lags = np.arange(-T2 // 2, T2 // 2 + (T2 % 2))
        n_lags = len(xcorr)
        if n_lags % 2 == 0:
            lags = np.arange(-n_lags // 2, n_lags // 2)
        else:
            lags = np.arange(-(n_lags // 2), (n_lags // 2) + 1)
        # Restrict to max_lag if specified
        if max_lag is not None:
            valid_indices = np.abs(lags) <= max_lag
            valid_indices = lags <=max_lag 
            lags = lags[valid_indices]
            xcorr = xcorr[valid_indices]
            valid_indices = lags > 0
            lags = lags[valid_indices]
            xcorr = xcorr[valid_indices]
            
        
        # Find maximum
        max_idx = np.argmax(np.abs(xcorr)) if direction == 'positive' else np.argmin(xcorr)
        max_corr[i] = xcorr[max_idx]
        best_lag[i] = lags[max_idx]
    
    # Create maps
    xcorr_map = np.zeros((H, W), dtype=float)
    xcorr_map[roi_mask] = max_corr
    
    lag_map = np.zeros((H, W), dtype=float)
    lag_map[roi_mask] = best_lag
    
    # Threshold based on maximum correlation values
    thr_high = max_corr.mean() + 2.0 * max_corr.std()
    thr_low = max_corr.mean() - 2.0 * max_corr.std()
    # thr_high = 0.3
    # thr_low = -0.3
    # Refine mask
    roi_mask_refined = np.zeros((H, W), dtype=bool)
    if direction == 'positive':
        roi_mask_refined[roi_mask] = (max_corr >= thr_high)
    else:
        roi_mask_refined[roi_mask] = (max_corr <= thr_low)
    
    # Extract refined ROI trace
    roi_trace = stack[:, roi_mask_refined].mean(axis=1)
    
    return roi_trace, roi_mask_refined, xcorr_map, lag_map, lags, xcorr

# def _next_pow2(n):
#     """Return the next power‑of‑2 ≥ n (used for fast FFT)."""
#     return 1 << (n - 1).bit_length()

# def get_corr_pixels(substack, stim, roi_mask, *, direction = "positive", max_lag = None,eps = 1e-12):
#     """
#     Compute a cross correlation map between a 3D image stack and a stimulus trace.

#     Parameters
#     ----------
#     substack : np.ndarray
#         Shape ``(T, n pixels)``  time series of the whole field of view.
#     stim : np.ndarray
#         1D stimulus trace (length ≥ ``T``). Only the first ``T`` samples are used.
#     roi_mask : np.ndarray
#         Boolean mask of shape ``(H, W)`` that defines the spatial region
#         on which the correlation is evaluated.
#     direction : {"positive", "negative"}, optional
#         Which side of the correlation distribution is used to build the refined mask.
#         ``"positive"`` → keep pixels with correlation ≥ ``mean+2*std``.
#         ``"negative"`` → keep pixels with correlation ≤ ``mean - 2*std``.
#     max_lag : int | None, optional
#         Maximum lag (in frames) that will be examined on both sides.
#         ``None`` means *all* possible lags (``[-T+1, …, T1]``).
#     eps : float, optional
#         Small constant added to denominators to avoid division by zero.

#     Returns
#     -------
#     roi_trace : np.ndarray
#         Mean trace of the refined ROI (shape ``(T,)``).
#     roi_mask_refined : np.ndarray
#         Boolean mask of shape ``(H, W)`` after thresholding the correlation map.
#     corr_map : np.ndarray
#         Full correlation map (shape ``(H, W)``) whose values are the **maximum
#         correlation coefficient** (over all considered lags) for each pixel.
#     """
#     if direction not in {"positive", "negative"}:
#         raise ValueError("direction must be 'positive' or 'negative'")
#     if substack.ndim not in {2, 3}:
#         raise ValueError("substack must have 2 or 3 dimensions")
#     T = substack.shape[0]
#     # T, H, W = substack.shape
#     # assert roi_mask.shape == (H, W), "roi_mask must match substack spatial dimensions"
#     # # flatten the image stack → (T, Npixels)
#     # trace_mat = substack.reshape(T, -1)                 # (T, H*W)
#     # mask_flat = roi_mask.ravel()                       # (H*W,) bool
#     # trace_masked = trace_mat[:, mask_flat]              # (T, N_mask)
#     if substack.ndim == 3:                     # full stack (T, H, W)
#         T, H, W = substack.shape
#         if roi_mask.shape != (H, W):
#             raise ValueError("roi_mask shape does not match substack spatial dimensions")
#         mask_flat = roi_mask.ravel()
#         # flatten spatial dims, then keep only the ROI pixels
#         trace_mat = substack.reshape(T, -1)[:, mask_flat]     # (T, N_roi)
#     else:                                        # already masked (T, N_roi)
#         # ``roi_mask`` is only needed to rebuild the 2‑D output later.
#         if roi_mask.ndim != 2:
#             raise ValueError("roi_mask must be a 2‑D boolean array")
#         H, W = roi_mask.shape
#         mask_flat = roi_mask.ravel()
#         N_expected = mask_flat.sum()
#         if substack.shape[1] != N_expected:
#             raise ValueError(
#                 f"substack has {substack.shape[1]} columns but roi_mask contains "
#                 f"{N_expected} True pixels"
#             )
#         trace_mat = substack                         # already (T, N_roi)
#     N_roi = trace_mat.shape[1]   
#     # normalise stimulus and pixel traces (zero‑mean, unit‑variance)
#     s = stim[:T].astype(float).copy()
#     s -= s.mean()
#     s /= (s.std() + eps)                           # now unit‑variance
#     # normalise pixel traces
#     trace_mat -= trace_mat.mean(axis=0, keepdims=True)
#     trace_mat /= (trace_mat.std(axis=0, keepdims=True) + eps)
#     # full cross‑correlation (FFT version – O(N log N))
#     # length of linear correlation = 2*T‑1
#     full_len = 2 * T - 1
#     nfft = _next_pow2(full_len)
#     # FFT of all pixel traces (axis=0 = time)
#     X = np.fft.rfft(trace_mat, n=nfft, axis=0)          # shape (nfft/2+1, N_roi)
#     # FFT of the stimulus (single vector)
#     Y = np.fft.rfft(s, n=nfft)                         # shape (nfft/2+1,)
#     # Conjugate multiplication → cross‑correlation in frequency domain
#     R = np.fft.irfft(X * np.conj(Y[:, None]), n=nfft, axis=0)   # (nfft, N_roi)
#     # Keep only the linear‑correlation part
#     R = R[:full_len, :]                                 # (2*T‑1, N_roi)
#     # Convert to Pearson‑like correlation coefficient
#     lags = np.arange(-(T - 1), T)                       # (2*T‑1,)
#     overlap = (T - np.abs(lags))[:, None]               # (2*T‑1, 1)

#     R_norm = R / (overlap + eps)                        # still (2*T‑1, N_roi)
#     # Optional lag restriction
#     if max_lag is not None:
#         if not (0 <= max_lag < T):
#             raise ValueError("max_lag must satisfy 0 <= max_lag < T")
#         keep = np.abs(lags) <= max_lag
#         R_norm = R_norm[keep, :]
#         lags = lags[keep]
#     # Pick the *best* correlation value per pixel
#     if direction == "positive":
#         best_idx = np.argmax(R_norm, axis=0)                 # (N_roi,)
#     else:  # negative
#         best_idx = np.argmin(R_norm, axis=0)
#     # advanced indexing to pull the best coefficient
#     best_corr = R_norm[best_idx, np.arange(N_roi)]
#     # Build the full‑frame correlation map and the refined ROI mask
#     corr_map = np.zeros((H, W), dtype=float)
#     corr_map.ravel()[mask_flat] = best_corr
#     mu = best_corr.mean()
#     sigma = best_corr.std()
#     thr_high = mu + 2.0 * sigma
#     thr_low = mu - 2.0 * sigma
#     roi_mask_refined = np.zeros((H, W), dtype=bool)
#     if direction == "positive":
#         roi_mask_refined.ravel()[mask_flat] = best_corr >= thr_high
#     else:
#         roi_mask_refined.ravel()[mask_flat] = best_corr <= thr_low
#     # Mean trace of the refined ROI
#     # indices of the refined ROI in the *flattened* image
#     refined_flat_idx = np.where(roi_mask_refined.ravel())[0]
#     if refined_flat_idx.size == 0:
#         roi_trace = np.zeros(T, dtype=float)
#     else:
#         if substack.ndim == 3:
#             full_flat = substack.reshape(T, -1)          # (T, H*W)
#             roi_trace = full_flat[:, refined_flat_idx].mean(axis=1)
#         else:
#             col_numbers = np.nonzero(mask_flat)[0]          # positions in image → column index
#             # Build a dict: image‑position → column‑index
#             pos2col = dict(zip(col_numbers, np.arange(N_roi)))
#             # Translate refined image positions back to column indices
#             refined_cols = np.array([pos2col[p] for p in refined_flat_idx], dtype=int)
#             roi_trace = substack[:, refined_cols].mean(axis=1)

#     return roi_trace, roi_mask_refined, corr_map


def get_xcorr_pixels_per_pulse(substack, pulse_protocol, roi_mask, reps, n_pre, n_width, n_post, n_isi, direction, max_lag=None):
    """
    Calculate cross-correlation for each pixel by cutting traces into pulse segments (pre+width+post).
    Only pulse segments that correlate with the stimulus are selected and their traces extracted.
    
    Parameters:
    -----------
    substack : array (T, n_pixels)
        Pixel traces within ROI
    pulse_protocol : array
        Single pulse protocol (pre + width + post)
    roi_mask : array (H, W)
        Boolean ROI mask
    reps : int
        Number of pulse repetitions (5)
    n_pre, n_width, n_post, n_isi : int
        Frame counts for protocol segments
    direction : str
        'positive' or 'negative'
    max_lag : int or None
        Maximum lag to consider (None = use full range)
    
    Returns:
    --------
    roi_trace : array
        Mean of all selected pulse segments (length = pulse_length)
    roi_mask_refined : array (H, W)
        Boolean mask of pixels that have at least one valid pulse
    xcorr_map : array (H, W)
        Cross-correlation map (mean of valid pulse max cross-correlations per pixel)
    lag_map : array (H, W)
        Mean lag at maximum correlation per pixel
    pulse_xcorrelations : array (n_valid_pulses,)
        Maximum cross-correlations for each selected pulse-pixel pair
    pulse_lags : array (n_valid_pulses,)
        Lags at maximum correlation for each selected pulse-pixel pair
    valid_segments : array (n_valid_pulses, pulse_length)
        All valid pulse segments
    """
    T, n_pixels = substack.shape
    H, W = roi_mask.shape
    
    # Calculate pulse segment boundaries
    pulse_length = n_pre + n_width + n_post
    onsets = []
    idx = 0
    for r in range(reps):
        idx = idx + n_pre
        onsets.append(idx)
        idx = idx + n_width
        if r < reps - 1:
            idx = idx + n_post + n_isi
        else:
            idx = idx + n_post
    onsets = np.array(onsets, dtype=int)
    
    # Extract pulse segments for each pixel
    pulse_segments = []
    for onset in onsets:
        start = onset - n_pre
        end = start + pulse_length
        if end <= T:
            segment = substack[start:end, :]
            pulse_segments.append(segment)
    
    if len(pulse_segments) == 0:
        roi_mask_refined = np.zeros((H, W), dtype=bool)
        xcorr_map = np.zeros((H, W), dtype=float)
        lag_map = np.zeros((H, W), dtype=float)
        roi_trace = np.zeros(pulse_length)
        return roi_trace, roi_mask_refined, xcorr_map, lag_map, np.array([]), np.array([]), np.array([])
    
    pulse_segments = np.array(pulse_segments)  # Shape: (reps, pulse_length, n_pixels)
    
    # Normalize pulse protocol
    s = pulse_protocol[:pulse_length].astype(float)
    s = s - s.mean()
    s = s / (s.std() + 1e-6)
    
    # Calculate cross-correlation for each pulse-pixel pair
    pulse_xcorrelations = np.zeros((reps, n_pixels))
    pulse_lags_array = np.zeros((reps, n_pixels), dtype=int)
    
    for rep_idx in range(reps):
        trace_mat = pulse_segments[rep_idx, :, :]  # Shape: (pulse_length, n_pixels)
        trace_mat = trace_mat - trace_mat.mean(axis=0, keepdims=True)
        stdx = trace_mat.std(axis=0, keepdims=True) + 1e-6
        trace_mat = trace_mat / stdx
        
        # Calculate cross-correlation for each pixel
        for pix_idx in range(n_pixels):
            xcorr = signal.correlate(trace_mat[:, pix_idx], s, mode='same')
            xcorr = xcorr / pulse_length  # Normalize by length
            
            # Define lag range based on actual xcorr length
            n_lags = len(xcorr)
            if n_lags % 2 == 0:
                lags = np.arange(-n_lags // 2, n_lags // 2)
            else:
                lags = np.arange(-(n_lags // 2), (n_lags // 2) + 1)
            
            # Restrict to max_lag if specified
            if max_lag is not None:
                valid_indices = np.abs(lags) <= max_lag
                xcorr_filtered = xcorr[valid_indices]
                lags_filtered = lags[valid_indices]
                valid_indices = lags <=max_lag 
                lags = lags[valid_indices]
                xcorr = xcorr[valid_indices]
                valid_indices = lags > 0
                lags_filtered = lags[valid_indices]
                xcorr_filtered = xcorr[valid_indices]
            else:
                xcorr_filtered = xcorr
                lags_filtered = lags
            
            # Find maximum
            if direction == 'positive':
                max_idx = np.argmax(xcorr_filtered)
            else:
                max_idx = np.argmin(xcorr_filtered)
            
            pulse_xcorrelations[rep_idx, pix_idx] = xcorr_filtered[max_idx]
            pulse_lags_array[rep_idx, pix_idx] = lags_filtered[max_idx]
    
    # Calculate threshold from ALL pulse-pixel correlations
    all_xcorrs = pulse_xcorrelations.flatten()
    thr_high = all_xcorrs.mean() + 2.0 * all_xcorrs.std()
    thr_low = all_xcorrs.mean() - 2.0 * all_xcorrs.std()
    # thr_high = np.percentile(all_xcorrs, 99.5)
    # thr_low = np.percentile(all_xcorrs, 100-99.5)
    # thr_high = 0.3
    # thr_low = -0.3
    
    # Filter each pulse-pixel pair independently
    valid_pulse_mask = np.zeros((reps, n_pixels), dtype=bool)
    
    if direction == 'positive':
        valid_pulse_mask = pulse_xcorrelations >= thr_high
    else:  # negative
        valid_pulse_mask = pulse_xcorrelations <= thr_low
    
    # Collect valid pulse segments and their correlations
    valid_segments = []
    valid_xcorrs = []
    valid_lags = []
    
    for rep_idx in range(reps):
        for pix_idx in range(n_pixels):
            if valid_pulse_mask[rep_idx, pix_idx]:
                # Extract this specific pulse segment for this pixel
                segment = pulse_segments[rep_idx, :, pix_idx]
                valid_segments.append(segment)
                valid_xcorrs.append(pulse_xcorrelations[rep_idx, pix_idx])
                valid_lags.append(pulse_lags_array[rep_idx, pix_idx])
    
    # Create outputs
    if len(valid_segments) > 0:
        # Stack and average all valid segments to get one pulse-length trace
        valid_segments = np.array(valid_segments)  # Shape: (n_valid_pulses, pulse_length)
        roi_trace = valid_segments.mean(axis=0)  # Shape: (pulse_length,)
        valid_xcorrs = np.array(valid_xcorrs)
        valid_lags = np.array(valid_lags)
        
        # Create pixel-level summary for masks and maps
        # A pixel is selected if it has at least one valid pulse
        selected_pixels = valid_pulse_mask.any(axis=0)
        
        # Correlation map: mean of valid correlations per pixel
        mean_xcorr_per_pixel = np.zeros(n_pixels)
        mean_lag_per_pixel = np.zeros(n_pixels)
        
        for pix_idx in range(n_pixels):
            valid_pulses = valid_pulse_mask[:, pix_idx]
            if valid_pulses.any():
                mean_xcorr_per_pixel[pix_idx] = pulse_xcorrelations[valid_pulses, pix_idx].mean()
                mean_lag_per_pixel[pix_idx] = pulse_lags_array[valid_pulses, pix_idx].mean()
        
        roi_mask_refined = np.zeros((H, W), dtype=bool)
        roi_mask_refined[roi_mask] = selected_pixels
        
        xcorr_map = np.zeros((H, W), dtype=float)
        xcorr_map[roi_mask] = mean_xcorr_per_pixel
        
        lag_map = np.zeros((H, W), dtype=float)
        lag_map[roi_mask] = mean_lag_per_pixel
    else:
        roi_trace = np.zeros(0)
        valid_xcorrs = np.array([])
        valid_lags = np.array([])
        valid_segments = np.array([])
        roi_mask_refined = np.zeros((H, W), dtype=bool)
        xcorr_map = np.zeros((H, W), dtype=float)
        lag_map = np.zeros((H, W), dtype=float)
    
    return roi_trace, roi_mask_refined, xcorr_map, lag_map, valid_xcorrs, valid_lags, valid_segments, pulse_xcorrelations, pulse_lags_array

def get_corr_pixels_per_pulse(substack, pulse_protocol, roi_mask, reps, n_pre, n_width, n_post, n_isi, direction):
    """
    Calculate correlation for each pixel by cutting traces into pulse segments (pre+width+post).
    Only pulse segments that correlate with the stimulus are selected and their traces extracted.
    
    Parameters:
    -----------
    substack : array (T, n_pixels)
        Pixel traces within ROI
    pulse_protocol : array
        Single pulse protocol (pre + width + post)
    roi_mask : array (H, W)
        Boolean ROI mask
    reps : int
        Number of pulse repetitions (5)
    n_pre, n_width, n_post, n_isi : int
        Frame counts for protocol segments
    direction : str
        'positive' or 'negative'
    
    Returns:
    --------
    roi_trace : array
        Mean of all selected pulse segments (length = pulse_length)
    roi_mask_refined : array (H, W)
        Boolean mask of pixels that have at least one valid pulse
    corr_map : array (H, W)
        Correlation map (mean of valid pulse correlations per pixel)
    pulse_correlations : array (n_valid_pulses,)
        Correlations for each selected pulse-pixel pair
    """
    T, n_pixels = substack.shape
    H, W = roi_mask.shape
    
    # Calculate pulse segment boundaries
    pulse_length = n_pre + n_width + n_post
    onsets = []
    idx = 0
    for r in range(reps):
        idx = idx + n_pre
        onsets.append(idx)
        idx = idx + n_width
        if r < reps - 1:
            idx = idx + n_post + n_isi
        else:
            idx = idx + n_post
    onsets = np.array(onsets, dtype=int)
    
    # Extract pulse segments for each pixel
    pulse_segments = []
    for onset in onsets:
        start = onset - n_pre
        end = start + pulse_length
        if end <= T:
            segment = substack[start:end, :]
            pulse_segments.append(segment)
    
    if len(pulse_segments) == 0:
        roi_mask_refined = np.zeros((H, W), dtype=bool)
        corr_map = np.zeros((H, W), dtype=float)
        roi_trace = np.zeros(pulse_length)
        return roi_trace, roi_mask_refined, corr_map, np.array([])
    
    pulse_segments = np.array(pulse_segments)  # Shape: (reps, pulse_length, n_pixels)
    
    # Normalize pulse protocol
    s = pulse_protocol[:pulse_length].astype(float)
    s = s - s.mean()
    s = s / (s.std() + 1e-6)
    
    # Calculate correlation for each pulse-pixel pair
    pulse_correlations = np.zeros((reps, n_pixels))
    
    for rep_idx in range(reps):
        trace_mat = pulse_segments[rep_idx, :, :]  # Shape: (pulse_length, n_pixels)
        trace_mat = trace_mat - trace_mat.mean(axis=0, keepdims=True)
        stdx = trace_mat.std(axis=0, keepdims=True) + 1e-6
        trace_mat = trace_mat / stdx
        
        # Calculate correlation for each pixel
        corr = (trace_mat * s[:, None]).mean(axis=0)
        pulse_correlations[rep_idx, :] = corr
    
    # Calculate threshold from ALL pulse-pixel correlations
    all_corrs = pulse_correlations.flatten()
    thr_high = all_corrs.mean() + 2.0 * all_corrs.std()
    thr_low = all_corrs.mean() - 2.0 * all_corrs.std()
    
    # Filter each pulse-pixel pair independently
    valid_pulse_mask = np.zeros((reps, n_pixels), dtype=bool)
    
    if direction == 'positive':
        valid_pulse_mask = pulse_correlations >= thr_high
    else:  # negative
        valid_pulse_mask = pulse_correlations <= thr_low
    
    # Collect valid pulse segments and their correlations
    valid_segments = []
    valid_corrs = []
    
    for rep_idx in range(reps):
        for pix_idx in range(n_pixels):
            if valid_pulse_mask[rep_idx, pix_idx]:
                # Extract this specific pulse segment for this pixel
                segment = pulse_segments[rep_idx, :, pix_idx]
                valid_segments.append(segment)
                valid_corrs.append(pulse_correlations[rep_idx, pix_idx])
    
    # Create outputs
    if len(valid_segments) > 0:
        # Stack and average all valid segments to get one pulse-length trace
        valid_segments = np.array(valid_segments)  # Shape: (n_valid_pulses, pulse_length)
        roi_trace = valid_segments.mean(axis=0)  # Shape: (pulse_length,)
        valid_corrs = np.array(valid_corrs)
        
        # Create pixel-level summary for masks and corr_map
        # A pixel is selected if it has at least one valid pulse
        selected_pixels = valid_pulse_mask.any(axis=0)
        
        # Correlation map: mean of valid correlations per pixel
        mean_corr_per_pixel = np.zeros(n_pixels)
        for pix_idx in range(n_pixels):
            valid_pulses = valid_pulse_mask[:, pix_idx]
            if valid_pulses.any():
                mean_corr_per_pixel[pix_idx] = pulse_correlations[valid_pulses, pix_idx].mean()
        
        roi_mask_refined = np.zeros((H, W), dtype=bool)
        roi_mask_refined[roi_mask] = selected_pixels
        
        corr_map = np.zeros((H, W), dtype=float)
        corr_map[roi_mask] = mean_corr_per_pixel
    else:
        roi_trace = np.zeros(0)
        valid_corrs = np.array([])
        roi_mask_refined = np.zeros((H, W), dtype=bool)
        corr_map = np.zeros((H, W), dtype=float)
    
    return roi_trace, roi_mask_refined, corr_map, valid_corrs, valid_segments



def get_dff(reps, n_pre, n_width, n_post, n_isi, stack, roi_mask_refined, roi_trace, pulse_protocol, fps):
    # Extract mean trace and compute dF/F
    onsets = []
    idx = 0
    for r in range(reps):
        idx = idx + n_pre
        onsets.append(idx)
        idx = idx + n_width
        if r < reps - 1:
            idx = idx + n_post + n_isi
        else:
            idx = idx + n_post
    onsets = np.array(onsets, dtype=int)
    roi_pixels = stack[:, roi_mask_refined]
    n_pixels_original = roi_pixels.shape[1]
    
    baseline = np.zeros_like(roi_trace)
    for onset in onsets:
        start = max(0, onset - n_pre)
        base = roi_trace[start:onset].mean()
        baseline[onset:onset+n_width] = base
    mask_nan = baseline == 0
    if np.any(mask_nan):
        last = roi_trace[:n_pre].mean()
        for t in range(T):
            if baseline[t] == 0:
                baseline[t] = last
            else:
                last = baseline[t]
    dff = (roi_trace - baseline) / (baseline + 1e-6)
    # df/f averaged across pulses
    pulse_length = n_pre + n_width + n_post
    pulses = []
    for onset in onsets:
        start = onset - n_pre
        end = start + pulse_length
        if end <= len(dff):
            pulse = dff[start:end]
            pulses.append(pulse)
    if pulses:
        pulses_array = np.array(pulses)
        pulse_avg = np.mean(pulses_array, axis=0)
        pulse_std = np.std(pulses_array, axis=0)
        pulse_sem = np.std(pulses_array, axis=0) / np.sqrt(len(pulses))
    else:
        pulse_avg = np.array([])
        pulse_std = np.array([])
        pulse_sem = np.array([])
    dff_pulse = {
        'pulse_mean': pulse_avg,
        'pulse_std': pulse_std,
        'pulse_sem': pulse_sem,
        'individual_pulses': pulses_array if pulses else None,
        'pulse_protocol': pulse_protocol,
        'n_pulses': len(pulses),
        'fps': fps}
    return dff, dff_pulse, n_pixels_original, pulses_array, pulses, pulse_avg, pulse_protocol


def get_dff_cut_pulse(roi_trace, n_pre, n_width, n_post, stack, roi_mask_refined, valid_segments, pulse_protocol, fps):
    roi_pixels = stack[:, roi_mask_refined]
    n_pixels_original = roi_pixels.shape[1]
    basline = roi_trace[0:n_pre].mean()
    dff = (roi_trace - basline) / (basline + 1e-6)
    return dff, n_pixels_original

def get_dff_responsive_pixels(stack, roi_mask, stim, reps, n_pre, n_width, n_post, n_isi, direction='positive', 
                               control_mode=None, seed=None):
    """
    Calculate dF/F for each pixel first, then select pixels showing responses > 2*std,
    then calculate correlation to stimulus.
    
    Parameters:
    -----------
    stack : array (T, H, W)
        Image stack
    roi_mask : array (H, W)
        Boolean ROI mask
    stim : array (T,)
        Stimulus protocol
    reps : int
        Number of stimulus repetitions
    n_pre, n_width, n_post, n_isi : int
        Frame counts for protocol segments
    direction : str
        'positive' or 'negative'
    control_mode : str or None
        None: Normal analysis
        'shuffle_before_dff': Phase randomize stack before calculating dF/F (Control 1)
        'shuffle_after_filter': Phase randomize traces after filtering (Control 2)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    roi_trace : array
        Mean dF/F trace of selected pixels
    roi_mask_refined : array (H, W)
        Boolean mask of selected pixels
    corr_map : array (H, W)
        Correlation map
    dff_pixel_traces : array (T, n_selected_pixels)
        dF/F traces of individual selected pixels
    """
    T, H, W = stack.shape
    
    # CONTROL 1: Shuffle traces BEFORE calculating dF/F
    if control_mode == 'shuffle_before_dff':
        print(f"  [CONTROL 1] Phase randomizing stack before dF/F calculation (seed={seed})")
        stack = phase_randomize_stack_vectorized(stack, seed=seed)
    
    # Calculate stimulus onsets
    onsets = []
    idx = 0
    for r in range(reps):
        idx = idx + n_pre
        onsets.append(idx)
        idx = idx + n_width
        if r < reps - 1:
            idx = idx + n_post + n_isi
        else:
            idx = idx + n_post
    onsets = np.array(onsets, dtype=int)
    
    # Extract pixels in ROI
    roi_pixels = stack[:, roi_mask]  # Shape: (T, n_pixels)
    n_pixels = roi_pixels.shape[1]
    
    # Calculate dF/F for each pixel
    dff_pixels = np.zeros_like(roi_pixels, dtype=float)
    
    for px in range(n_pixels):
        pixel_trace = roi_pixels[:, px]
        baseline_px = np.zeros_like(pixel_trace)
        
        # Calculate baseline for each stimulus window
        for onset in onsets:
            start = max(0, onset - n_pre)
            base = pixel_trace[start:onset].mean()
            baseline_px[onset:onset+n_width] = base
        
        # Forward/backward fill for continuity
        mask_nan = baseline_px == 0
        if np.any(mask_nan):
            last = pixel_trace[:n_pre].mean()
            for t in range(T):
                if baseline_px[t] == 0:
                    baseline_px[t] = last
                else:
                    last = baseline_px[t]
        
        # Calculate dF/F
        dff_pixels[:, px] = (pixel_trace - baseline_px) / (np.abs(baseline_px) + 1e-6)
    
    # Calculate response during stimulus windows
    # Extract dF/F during stimulus presentation (width period)
    stim_responses = []
    for onset in onsets:
        if onset + n_width <= T:
            stim_window = dff_pixels[onset:onset+n_width, :]
            stim_responses.append(stim_window.mean(axis=0))  # Mean response per pixel
    
    stim_responses = np.array(stim_responses)  # Shape: (n_reps, n_pixels)
    mean_stim_response = stim_responses.mean(axis=0)  # Mean across repetitions
    
    # Calculate baseline response (pre-stimulus periods)
    baseline_responses = []
    for onset in onsets:
        start = max(0, onset - n_pre)
        baseline_window = dff_pixels[start:onset, :]
        baseline_responses.append(baseline_window.mean(axis=0))
    
    baseline_responses = np.array(baseline_responses)
    mean_baseline_response = baseline_responses.mean(axis=0)
    
    # Calculate response magnitude (stimulus - baseline)
    response_magnitude = mean_stim_response - mean_baseline_response
    
    # # Threshold: 2 * std above/below mean
    # thr_high = response_magnitude.mean() + 2.0 * response_magnitude.std()
    # thr_low = response_magnitude.mean() - 2.0 * response_magnitude.std()

    thr_high = mean_baseline_response + 2.0 * baseline_responses.std()
    thr_low = mean_baseline_response - 2.0 * baseline_responses.std()
    # Select pixels based on direction
    if direction == 'positive':
        selected_pixels = response_magnitude >= thr_high
    else:  # negative
        selected_pixels = response_magnitude <= thr_low
    
    # Create refined ROI mask
    roi_mask_refined = np.zeros((H, W), dtype=bool)
    roi_mask_refined[roi_mask] = selected_pixels
    
    # Get dF/F traces of selected pixels
    dff_selected = dff_pixels[:, selected_pixels]
    
    # CONTROL 2: Shuffle traces AFTER filtering for responsive pixels
    if control_mode == 'shuffle_after_filter':
        print(f"  [CONTROL 2] Phase randomizing filtered pixel traces (seed={seed})")
        if dff_selected.shape[1] > 0:
            dff_selected = phase_randomize_traces(dff_selected, seed=seed)
    
    # Calculate mean trace
    if dff_selected.shape[1] > 0:
        roi_trace = dff_selected.mean(axis=1)
    else:
        roi_trace = np.zeros(T)
    
    # NOW calculate correlation with stimulus for selected pixels
    if dff_selected.shape[1] > 0:
        # Normalize dF/F traces
        trace_mat = dff_selected.copy()
        trace_mat = trace_mat - trace_mat.mean(axis=0, keepdims=True)
        stdx = trace_mat.std(axis=0, keepdims=True) + 1e-6
        trace_mat = trace_mat / stdx
        
        # Normalize stimulus
        T2 = trace_mat.shape[0]
        s = stim[:T2].astype(float)
        s = s - s.mean()
        s = s / (s.std() + 1e-6)
        
        # Calculate correlation
        corr = (trace_mat * s[:, None]).mean(axis=0)
    else:
        corr = np.array([])
    
    # Create correlation map
    corr_map = np.zeros((H, W), dtype=float)
    if len(corr) > 0:
        corr_map[roi_mask_refined] = corr
    
    return roi_trace, roi_mask_refined, corr_map, dff_selected


def get_dff_and_pulses(roi_trace, dff_pixel_traces, reps, n_pre, n_width, n_post, n_isi, 
                       pulse_protocol, fps):
    """
    Extract pulse-averaged data from already calculated dF/F traces.
    
    Parameters:
    -----------
    roi_trace : array
        Mean dF/F trace across selected pixels
    dff_pixel_traces : array (T, n_pixels)
        Individual pixel dF/F traces
    reps : int
        Number of repetitions
    n_pre, n_width, n_post, n_isi : int
        Frame counts for protocol segments
    pulse_protocol : array
        Single pulse protocol for plotting
    fps : float
        Frame rate
    
    Returns:
    --------
    dff : array
        Mean dF/F trace (same as roi_trace input)
    dff_pulse : dict
        Dictionary with pulse-averaged data
    n_pixels_original : int
        Number of selected pixels
    pulses_array : array
        Individual pulses stacked
    pulses : list
        List of individual pulses
    pulse_avg : array
        Average across pulses
    pulse_protocol : array
        Protocol for plotting
    """
    T = len(roi_trace)
    n_pixels_original = dff_pixel_traces.shape[1] if dff_pixel_traces.ndim > 1 else 0
    
    # Calculate pulse onsets
    onsets = []
    idx = 0
    for r in range(reps):
        idx = idx + n_pre
        onsets.append(idx)
        idx = idx + n_width
        if r < reps - 1:
            idx = idx + n_post + n_isi
        else:
            idx = idx + n_post
    onsets = np.array(onsets, dtype=int)
    
    # Extract pulses from mean trace
    pulse_length = n_pre + n_width + n_post
    pulses = []
    for onset in onsets:
        start = onset - n_pre
        end = start + pulse_length
        if end <= len(roi_trace):
            pulse = roi_trace[start:end]
            pulses.append(pulse)
    
    # Calculate pulse statistics
    if pulses:
        pulses_array = np.array(pulses)
        pulse_avg = np.mean(pulses_array, axis=0)
        pulse_std = np.std(pulses_array, axis=0)
        pulse_sem = np.std(pulses_array, axis=0) / np.sqrt(len(pulses))
    else:
        pulses_array = np.array([])
        pulse_avg = np.array([])
        pulse_std = np.array([])
        pulse_sem = np.array([])
    
    dff_pulse = {
        'pulse_mean': pulse_avg,
        'pulse_std': pulse_std,
        'pulse_sem': pulse_sem,
        'individual_pulses': pulses_array if pulses else None,
        'pulse_protocol': pulse_protocol,
        'n_pulses': len(pulses),
        'fps': fps
    }
    
    dff = roi_trace  # Return the same trace
    
    return dff, dff_pulse, n_pixels_original, pulses_array, pulses, pulse_avg, pulse_protocol

def analyze_roi_dff_first(stack, roi_mask, stim, reps, n_pre, n_width, n_post, n_isi, 
                          pulse_protocol, fps, direction='positive', control_mode=None, seed=None):
    """
    Complete analysis: dF/F first, then response-based selection, then correlation.
    
    This is a drop-in replacement that maintains the same output structure as the
    old get_corr_pixels + get_dff workflow.
    
    Parameters:
    -----------
    stack : array (T, H, W)
        Image stack
    roi_mask : array (H, W)
        Boolean ROI mask
    stim : array (T,)
        Stimulus protocol
    reps : int
        Number of repetitions
    n_pre, n_width, n_post, n_isi : int
        Frame counts for protocol segments
    pulse_protocol : array
        Single pulse protocol for plotting
    fps : float
        Frame rate
    direction : str
        'positive' or 'negative'
    control_mode : str or None
        None: Normal analysis
        'shuffle_before_dff': Phase randomize stack before calculating dF/F (Control 1)
        'shuffle_after_filter': Phase randomize traces after filtering (Control 2)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Contains all analysis outputs with same structure as old functions
    """
    # Step 1: Calculate dF/F and select responsive pixels (with optional control)
    roi_trace, roi_mask_refined, corr_map, dff_pixel_traces = get_dff_responsive_pixels(
        stack, roi_mask, stim, reps, n_pre, n_width, n_post, n_isi, direction,
        control_mode=control_mode, seed=seed
    )
    
    # Step 2: Extract pulse-averaged data
    dff, dff_pulse, n_pixels, pulses_array, pulses, pulse_avg, pulse_protocol_out = get_dff_and_pulses(
        roi_trace, dff_pixel_traces, reps, n_pre, n_width, n_post, n_isi, 
        pulse_protocol, fps
    )
    
    return {
        'roi_trace': roi_trace,
        'roi_mask_refined': roi_mask_refined,
        'corr_map': corr_map,
        'dff': dff,
        'dff_pulse': dff_pulse,
        'n_pixels': n_pixels,
        'pulses_array': pulses_array,
        'pulses': pulses,
        'pulse_avg': pulse_avg,
        'pulse_protocol': pulse_protocol_out,
        'dff_pixel_traces': dff_pixel_traces,
        'control_mode': control_mode
    }


def extract_data(results):
    return results['corr_map'], results['roi_mask_refined'], results['dff'], results['n_pixels'], results['dff_pulse'], results['pulse_avg']


def plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, stim, title_suffix, n_pixels_original, save_suffix, dff_pulse, only_pulse, ylim=None):
    fig1 = plt.figure(figsize=(14, 4))
    plt.suptitle(f'{direction} - ROI Selection', fontsize=16, fontweight='bold')
    plt.subplot(1, 4, 1)
    plt.imshow(rep_frame_o, cmap='gray')
    plt.title('Median frame (for ROI)')
    plt.subplot(1, 4, 2)
    plt.imshow(rep_frame, cmap='gray')
    overlay = np.zeros((*rep_frame.shape, 4))
    overlay[roi_mask, :] = [0, 1, 1, 0.3]
    plt.imshow(overlay)
    plt.title('Selected napari region on median frame')
    plt.subplot(1, 4, 3)
    plt.imshow(corr_map, cmap='magma')
    plt.title('Correlation map inside drawn ROI')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 4, 4)
    plt.imshow(roi_mask_refined, cmap='Greens')
    plt.title('Refined ROI mask (corr-threshold)')
    plt.tight_layout()
    fig1.savefig(Path(output_tif) / f'roi_visualization_{direction}{save_suffix}.png', dpi=400)
    plt.close(fig1)
    
    
    # Save dF/F trace figure
    if len(dff)>400:
        fig2 = plt.figure(figsize=(12, 4))
    else:
        fig2 = plt.figure(figsize=(6, 6))
    plt.plot(dff, color='k', linewidth=1.2)
    plt.plot(stim, color='r')
    # title_suffix = ' (Phase Randomized)' if stack_type == 'shuffled' else ''
    plt.title(f'ROI mean trace dF/F with stimulus epochs (n={n_pixels_original} pixels), {direction}{title_suffix}')
    plt.xlabel('Frame')
    plt.ylabel('dF/F')
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    fig2.savefig(Path(output_tif) / f'all_pixels_{direction}{save_suffix}.png', dpi=400)
    plt.close(fig2)
    if only_pulse == False:
        pulses_array = dff_pulse['individual_pulses']
        pulse_avg = dff_pulse['pulse_mean']
        pulse_protocol = dff_pulse['pulse_protocol']
        pulse_sem = dff_pulse['pulse_sem']
        pulses = dff_pulse['n_pulses']
        # Save pulse average figure
        fig3 = plt.figure(figsize=(6, 12))
        plt.subplot(2, 1, 1)
        if pulses_array is not None:
            for i, pulse in enumerate(pulses_array):
                plt.plot(pulse, 'grey', alpha=0.3, linewidth=0.8)
        plt.plot(pulse_avg, 'k', linewidth=2, label='Average across pulses')
        plt.plot(pulse_protocol, 'r', linewidth=1.5, label='Stimulus')
        plt.title(f'Individual pulses and average (n={pulses} pulses), {direction}{title_suffix}')
        plt.ylabel('dF/F')
        if direction == 'positive':
            plt.ylim(-5,10)
        else:
            plt.ylim(-10,5)
        plt.legend()
        plt.subplot(2, 1, 2)
        frames_pulse = np.arange(len(pulse_avg))
        plt.plot(pulse_avg, 'k', linewidth=2, label='Mean')
        plt.fill_between(frames_pulse, pulse_avg - pulse_sem, pulse_avg + pulse_sem,
                        alpha=0.3, color='gray', label='SEM')
        plt.plot(pulse_protocol, 'r', linewidth=1.5, label='Stimulus')
        plt.title(f'Average pulse with SEM, {direction}{title_suffix}')
        plt.xlabel('Frame (relative to pulse onset)')
        plt.ylabel('dF/F')
        if direction == 'positive':
            plt.ylim(-5,10)
        else:
            plt.ylim(-10,5)
        plt.legend()
        plt.tight_layout()
        fig3.savefig(Path(output_tif) / f'pulse_average_{direction}{save_suffix}.png', dpi=400)
        plt.close(fig3)






def plot_fly_avg(fly_average, stim, tif_files, title_suffix, output_fly, save_str, fly_pulse_average, pulse_protocol, only_pulse):
    # Plot fly-level averages (original and shuffled)
    for direction in ['positive', 'negative']:
        
        dff = fly_average[direction]['dff_mean']
        dff_sem = fly_average[direction]['dff_sem']
        # # Phase randomize the averaged trace for shuffled version
        # if save_str == '_shuffled':
        #     dff = phase_randomize_trace(dff, seed=hash(fly) % 10000)
        frames = np.arange(len(dff))
        if len(dff)>400:
            fig2, ax = plt.subplots(figsize=(12, 4))
        else:
            fig2, ax = plt.subplots(figsize=(6, 6))
        
        plt.plot(dff, color='k', linewidth=1.2)
        plt.plot(stim, color='r')
        ax.fill_between(frames, dff - dff_sem, dff + dff_sem, alpha=0.3, zorder=4, color='gray')
        # title_suffix = ' (Phase Randomized)' if save_str == '_shuffled' else ''
        plt.title(f'mean across {len(tif_files)} recordings, {direction}{title_suffix}')
        plt.xlabel('Frame')
        plt.ylabel('dF/F')
        if direction == 'positive':
            plt.ylim(-2.5,5)
        else:
            plt.ylim(-5,2.5)
        plt.tight_layout()
        fig2.savefig(Path(output_fly) / f'all_pixels_{direction}{save_str}.png', dpi=400)
        plt.close(fig2)
        if only_pulse == False:
            # Plot pulse averages
            pulse_avg = fly_pulse_average[direction]['pulse_mean']
            pulse_sem = fly_pulse_average[direction]['pulse_sem']
            # # Phase randomize pulse average for shuffled version
            # if save_str == '_shuffled':
            #     pulse_avg = phase_randomize_trace(pulse_avg, seed=hash(fly) % 10000 + 1)
            
            frames_pulse = np.arange(len(pulse_avg))
            fig4 = plt.figure(figsize=(4, 4))
            plt.plot(pulse_avg, 'k', linewidth=2, label='Mean')
            plt.fill_between(frames_pulse, pulse_avg - pulse_sem, pulse_avg + pulse_sem,
                        alpha=0.3, color='gray', label='SEM')
            plt.plot(pulse_protocol, 'r', linewidth=1.5, label='Stimulus')
            plt.title(f'Mean pulse across {fly_pulse_average[direction]["n_tseries"]} TSeries, {direction}{title_suffix}')
            plt.xlabel('Frame (relative to pulse onset)')
            plt.ylabel('dF/F')
            if direction == 'positive':
                plt.ylim(-2.5,5)
            else:
                plt.ylim(-5,2.5)
            plt.legend()
            plt.tight_layout()
            fig4.savefig(Path(output_fly) / f'pulse_average_{direction}{save_str}.png', dpi=400)
            plt.close(fig4)


def plot_condition_avg(condition_average, protocoll_new, title_suffix, outpout_condition, save_str, condition_pulse_average, pulse_protocol_new, only_pulse, ylim=None):
    # Plot condition-level averages (original and shuffled)
    for direction in ['positive', 'negative']:
        
        dff = condition_average[direction]['dff_mean']
        dff_sem = condition_average[direction]['dff_sem']
        
        # # Phase randomize for shuffled version
        # if save_str == '_shuffled':
        #     dff = phase_randomize_trace(dff, seed=hash(condition) % 10000)
        
        frames = np.arange(len(dff))
        n_flies = len([d for d in os.listdir(f'{processed_recordings}/{condition}') 
                        if os.path.isdir(f'{processed_recordings}/{condition}/{d}')])
        if len(dff)>400:
            fig2, ax = plt.subplots(figsize=(12, 4))
        else:
            fig2, ax = plt.subplots(figsize=(6, 6))
        # fig2, ax = plt.subplots(figsize=(12, 4))
        plt.plot(dff, color='k', linewidth=1.2)
        plt.plot(protocoll_new, color='r')
        ax.fill_between(frames, dff - dff_sem, dff + dff_sem, alpha=0.3, zorder=4, color='gray')
        # title_suffix = ' (Phase Randomized)' if save_str == '_shuffled' else ''
        plt.title(f'mean across {n_flies} flies, {direction}{title_suffix}')
        plt.xlabel('Frame')
        if direction == 'positive':
            plt.ylim(-2.5,5)
        else:
            plt.ylim(-5,2.5)
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.ylabel('dF/F')
        plt.tight_layout()
        fig2.savefig(Path(outpout_condition) / f'all_pixels_{direction}{save_str}.png', dpi=400)
        plt.close(fig2)
        if only_pulse == False:
            # Plot pulse averages
            pulse_avg = condition_pulse_average[direction]['pulse_mean']
            pulse_sem = condition_pulse_average[direction]['pulse_sem']
            
            # # Phase randomize pulse for shuffled version
            # if save_str == '_shuffled':
            #     pulse_avg = phase_randomize_trace(pulse_avg, seed=hash(condition) % 10000 + 1)
            
            frames_pulse = np.arange(len(pulse_avg))
            fig5 = plt.figure(figsize=(4, 4))
            plt.plot(pulse_avg, 'k', linewidth=2, label='Mean')
            plt.fill_between(frames_pulse, pulse_avg - pulse_sem, pulse_avg + pulse_sem,
                        alpha=0.3, color='gray', label='SEM')
            plt.plot(pulse_protocol_new, 'r', linewidth=1.5, label='Stimulus')
            if ylim:
                plt.ylim(ylim[0], ylim[1])
            plt.title(f'Mean pulse across {condition_pulse_average[direction]["n_flies"]} flies, {direction}{title_suffix}')
            plt.xlabel('Frame (relative to pulse onset)')
            if direction == 'positive':
                plt.ylim(-2.5,5)
            else:
                plt.ylim(-5,2.5)
            plt.ylabel('dF/F')
            plt.legend()
            plt.tight_layout()
            fig5.savefig(Path(outpout_condition) / f'pulse_average_{direction}{save_str}.png', dpi=400)
            plt.close(fig5)

def plot_lag_maps(lag_map, xcorr_map, roi_mask_refined, output_tif, direction, save_str,
                 title='Spatial Maps', frame_rate=None):
    """
    Plot spatial maps of lag and cross-correlation.
    
    Parameters:
    -----------
    lag_map : array (H, W)
        Spatial map of lags
    xcorr_map : array (H, W)
        Spatial map of cross-correlations
    roi_mask_refined : array (H, W)
        Boolean mask of selected pixels
    title : str
        Plot title
    frame_rate : float or None
        If provided, show lag in seconds
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Lag map
    ax1 = axes[0]
    lag_display = lag_map.copy()
    lag_display[~roi_mask_refined] = np.nan
    
    im1 = ax1.imshow(lag_display, cmap='RdBu_r', aspect='auto')
    ax1.set_title('Lag Map')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    if frame_rate is not None:
        cbar1.set_label('Lag (seconds)')
        # Convert colorbar ticks to seconds
        ticks = cbar1.get_ticks()
        cbar1.set_ticklabels([f'{t/frame_rate:.2f}' for t in ticks])
    else:
        cbar1.set_label('Lag (frames)')
    
    # 2. Cross-correlation map
    ax2 = axes[1]
    xcorr_display = xcorr_map.copy()
    xcorr_display[~roi_mask_refined] = np.nan
    
    im2 = ax2.imshow(xcorr_display, cmap='viridis', aspect='auto')
    ax2.set_title('Cross-Correlation Map')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Cross-Correlation')
    
    # 3. ROI mask
    ax3 = axes[2]
    ax3.imshow(roi_mask_refined, cmap='gray', aspect='auto')
    ax3.set_title(f'Selected ROI ({roi_mask_refined.sum()} pixels)')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_tif) / f'pulse_average_{direction}{save_str}.png', dpi=400)

def plot_lag_xcorr_distribution(lag_values, xcorr_values, output_condition, save_str, 
                                normalize=True, bins=50, 
                                title='Cross-Correlation vs Lag',
                                frame_rate=None):
    """
    Plot cross-correlation values as a function of lag.
    
    Parameters:
    -----------
    lag_values : array
        Lag values (in frames)
    xcorr_values : array
        Cross-correlation values
    normalize : bool
        If True, normalize xcorr to [0, 1]
    bins : int or array
        Number of bins or bin edges for lag
    title : str
        Plot title
    frame_rate : float or None
        If provided, show time in seconds on secondary x-axis
    """
    for direciton in ['positive', 'negative']:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Normalize if requested
        if normalize:
            xcorr_plot = (xcorr_values - xcorr_values.min()) / (xcorr_values.max() - xcorr_values.min() + 1e-6)
            ylabel = 'Normalized Cross-Correlation'
        else:
            xcorr_plot = xcorr_values
            ylabel = 'Cross-Correlation'
        
        # 1. Scatter plot: xcorr vs lag
        ax1 = axes[0, 0]
        scatter = ax1.scatter(lag_values, xcorr_plot, alpha=0.5, s=20, c=xcorr_plot, cmap='viridis')
        ax1.set_xlabel('Lag (frames)')
        ax1.set_ylabel(ylabel)
        ax1.set_title('Cross-Correlation vs Lag (Scatter)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero lag')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label=ylabel)
        
        # Add time axis if frame_rate provided
        if frame_rate is not None:
            ax1_time = ax1.twiny()
            ax1_time.set_xlim(ax1.get_xlim()[0] / frame_rate, ax1.get_xlim()[1] / frame_rate)
            ax1_time.set_xlabel('Lag (seconds)')
        
        # 2. Binned average: mean xcorr per lag bin
        ax2 = axes[0, 1]
        
        # Create bins
        if isinstance(bins, int):
            lag_bins = np.linspace(lag_values.min(), lag_values.max(), bins)
        else:
            lag_bins = bins
        
        # Bin the data
        bin_indices = np.digitize(lag_values, lag_bins)
        bin_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
        
        bin_means = []
        bin_stds = []
        bin_counts = []
        
        for i in range(1, len(lag_bins)):
            mask = bin_indices == i
            if mask.any():
                bin_means.append(xcorr_plot[mask].mean())
                bin_stds.append(xcorr_plot[mask].std())
                bin_counts.append(mask.sum())
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
                bin_counts.append(0)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_counts = np.array(bin_counts)
        
        # Plot with error bars
        valid = ~np.isnan(bin_means)
        ax2.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_stds[valid], 
                    fmt='o-', capsize=3, capthick=1, linewidth=2, markersize=6)
        ax2.set_xlabel('Lag (frames)')
        ax2.set_ylabel(f'Mean {ylabel}')
        ax2.set_title('Binned Mean Cross-Correlation vs Lag')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero lag')
        ax2.legend()
        
        if frame_rate is not None:
            ax2_time = ax2.twiny()
            ax2_time.set_xlim(ax2.get_xlim()[0] / frame_rate, ax2.get_xlim()[1] / frame_rate)
            ax2_time.set_xlabel('Lag (seconds)')
        
        # 3. Histogram of lags (weighted by xcorr magnitude)
        ax3 = axes[1, 0]
        
        # Weight by xcorr magnitude
        weights = np.abs(xcorr_plot)
        
        counts, edges, patches = ax3.hist(lag_values, bins=bins, weights=weights, 
                                        alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Lag (frames)')
        ax3.set_ylabel('Sum of Cross-Correlation')
        ax3.set_title('Lag Distribution (weighted by xcorr)')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero lag')
        
        # Add peak lag
        peak_lag_idx = np.argmax(counts)
        peak_lag = (edges[peak_lag_idx] + edges[peak_lag_idx + 1]) / 2
        ax3.axvline(x=peak_lag, color='orange', linestyle='--', alpha=0.7, 
                    label=f'Peak lag: {peak_lag:.1f} frames')
        ax3.legend()
        
        if frame_rate is not None:
            ax3_time = ax3.twiny()
            ax3_time.set_xlim(ax3.get_xlim()[0] / frame_rate, ax3.get_xlim()[1] / frame_rate)
            ax3_time.set_xlabel('Lag (seconds)')
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Cross-Correlation vs Lag Statistics
        {'='*45}
        
        Total pixels/pulses: {len(lag_values)}
        
        Lag (frames):
        Mean: {lag_values.mean():.2f}
        Median: {np.median(lag_values):.2f}
        Std: {lag_values.std():.2f}
        Range: [{lag_values.min():.0f}, {lag_values.max():.0f}]
        Peak (weighted): {peak_lag:.1f}
        """
        
        if frame_rate is not None:
            stats_text += f"""
        Lag (seconds):
        Mean: {lag_values.mean() / frame_rate:.3f}
        Median: {np.median(lag_values) / frame_rate:.3f}
        Peak (weighted): {peak_lag / frame_rate:.3f}
        """
        
        stats_text += f"""
        Cross-Correlation:
        Mean: {xcorr_values.mean():.3f}
        Median: {np.median(xcorr_values):.3f}
        Std: {xcorr_values.std():.3f}
        Range: [{xcorr_values.min():.3f}, {xcorr_values.max():.3f}]
        
        Correlation between lag and xcorr:
        r = {np.corrcoef(lag_values, xcorr_values)[0, 1]:.3f}
        """
        
        ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(output_condition) / f'pulse_average_{direction}{save_str}.png', dpi=400)

mean_fps = metadata.loc[:, 'fps'].mean()
for condition in os.listdir(processed_recordings):
    if condition.endswith("_nan"):#or condition == '_LOP_ACV' or condition == '_LO_ACV':
        continue
    print(f'processing {condition}')
    lag_values_all, xcorr_values_all = [], []
    lag_values_all_shuffled, xcorr_values_all_shuffled = [], []
    lag_values_cut_pulse, xcorr_values_cut_pulse = [], []
    lag_values_cut_pulse_shuffled, xcorr_values_cut_pulse_shuffled = [], []
    condition_results, condition_pulse_results = {}, {}
    condition_pulse_results_cut_pulse, condition_pulse_results_cut_pulse_shuffled = {}, {}
    condition_results_shuffled, condition_pulse_results_shuffled = {}, {}
    condition_results_sortdff, condition_pulse_results_sortdff = {}, {}
    condition_results_shuffle_before_dff, condition_pulse_results_shuffle_before_dff = {}, {}
    condition_results_shuffle_after_filter, condition_pulse_results_shuffle_after_filter = {}, {}
    region = condition.split('_')[1]
    outpout_condition = f'{processed_recordings}/{condition}'
    if os.path.isdir(f'{processed_recordings}/{condition}'):
        if os.path.exists(f'{processed_recordings}/{condition}/pulse_average_positive.png'):
            continue
        for fly in os.listdir(f'{processed_recordings}/{condition}'):
            all_flies ={}
            if os.path.isdir(f'{processed_recordings}/{condition}/{fly}'):
                if os.path.exists(f'{processed_recordings}/{condition}/{fly}/pulse_average_positive.png'):
                    print('skiped', fly)
                    with open(f'{processed_recordings}/{condition}/{fly}/all_results_flies.pkl', 'rb') as file:
                        fly_result = pickle.load(file)
                    condition_pulse_results_cut_pulse[fly] = fly_result["fly_pulse_average_cut_pulse"]
                    condition_pulse_results_cut_pulse_shuffled[fly] = fly_result["fly_pulse_average_cut_pulse_shuffled"]
                    condition_results[fly] = fly_result["fly_average"]
                    condition_results_shuffled[fly] = fly_result["fly_average_shuffled"]
                    condition_pulse_results_shuffled[fly] = fly_result["fly_pulse_average_shuffled"]
                    condition_pulse_results[fly] = fly_result["fly_pulse_average"]
                    print('loaded Results for:  ', fly)
                    continue


                fly_result,fly_pulse_result = {}, {}
                fly_result_sortdff, fly_pulse_result_sortdff = {}, {}
                fly_result_shuffle_before_dff, fly_pulse_result_shuffle_before_dff = {}, {}
                fly_result_shuffle_after_filter, fly_pulse_result_shuffle_after_filter = {}, {}
                fly_result_shuffled, fly_pulse_result_shuffled = {}, {}
                fly_pulse_result_cut_pulse, fly_pulse_result_cut_pulse_shuffled = {}, {}
                meta_fly = metadata[metadata['fly']==fly]
                meta_fly_region = meta_fly[meta_fly['region']==region]
                fps = meta_fly_region['fps'].values[0]
                reps = 5
                n_pre = int(5 * fps)
                n_width = int(5 * fps)
                n_post = int(5 * fps)
                n_isi = int(20 * fps)
                protocol, pulse_protocol = make_olf_protocoll(fps)
                print(fly)
                output_fly = f'{processed_recordings}/{condition}/{fly}'
                # Get all TIF files in folder
                tif_files = sorted(Path(output_fly).rglob("*_motCorr.tif"))
                print(f"Found {len(tif_files)} TIF files")
                # Storage for collecting data across TIFs
                # Loop through each TIF file
                pixel_maks = {}
                for tif_idx, stack_path in enumerate(tif_files):
                    tif_result, tif_pulse_result = {}, {}
                    tif_pulse_result_cut_pulse, tif_pulse_result_cut_pulse_shuffled = {}, {}
                    tif_result_sortdff, tif_pulse_result_sortdff = {}, {}
                    tif_result_shuffle_before_dff, tif_pulse_result_shuffle_before_dff = {}, {}
                    tif_result_shuffle_after_filter, tif_pulse_result_shuffle_after_filter = {}, {}
                    tif_result_shuffled, tif_pulse_result_shuffled = {}, {}
                    print(str(stack_path))
                    tif_name = str(stack_path).split('\\')[-2]
                    
                    output_tif = f'{processed_recordings}/{condition}/{fly}/{tif_name}'
                    print(f"\nProcessing {tif_idx+1}/{len(tif_files)}: {stack_path.name}")
                    # --- Load stack ---
                    stack = tiff.imread(str(stack_path)).astype(np.float32)
                    T, H, W = stack.shape
                    rep_frame_o = np.median(stack, axis=0) 
                    # # Frame-wise mean subtraction
                    subtracted = np.zeros_like(stack, dtype=float)
                    for frame_idx in range(stack.shape[0]):
                        frame_mean = stack[frame_idx].mean()
                        subtracted[frame_idx] = stack[frame_idx] - frame_mean


                    #BG substraction, substract mean of darkest 20 pixels from each pixel and frame
                    # Find mean of darkest 20 pixels in the median image
                    # darkest_20_indices = np.argpartition(rep_frame_o.flatten(), 20)[:20]
                    # darkest_20_values = rep_frame_o.flatten()[darkest_20_indices]
                    # dark_mean = darkest_20_values.mean()
                    # subtracted = stack - dark_mean

                    # substract temporal mean of each pixel per pixel and frame
                    # Calculate mean for each pixel across all frames
                    # temporal_mean = subtracted.mean(axis=0)
                    # # Subtract the temporal mean frame from each frame
                    # subtracted = subtracted - temporal_mean 

                    stack = subtracted
                    stack_shuffled = phase_randomize_stack_vectorized(stack, seed=tif_idx + 42)
                    rep_frame = np.median(stack, axis=0)        
                    # Trim or pad protocol to T
                    if protocol.shape[0] < T:
                        pad = np.zeros(T - protocol.shape[0], dtype=np.float32)
                        stim = np.hstack((protocol, pad))
                    elif protocol.shape[0] > T:
                        stim = protocol[:T]
                    else:
                        stim = protocol.copy()
                    # --- Napari selection (only for first file) ---
                    if os.path.exists(f'{processed_recordings}/{condition}/{fly}/_ROIS_skipp.pkl') == True:
                        with open(f'{processed_recordings}/{condition}/{fly}/_ROIS_skipp.pkl', 'rb') as fo:
                            roi_mask_skipp = pickle.load(fo)
                        roi_mask = roi_mask_skipp['roi_mask']
                    elif tif_idx == 0 and os.path.exists(f'{processed_recordings}/{condition}/{fly}/_ROIS_skipp.pkl') == False:
                        viewer = napari.Viewer()
                        image_layer = viewer.add_image(rep_frame_o, name='median')
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
                        roi_mask_skipp = {"roi_mask": roi_mask}
                        with open(f'{processed_recordings}/{condition}/{fly}/_ROIS_skipp.pkl', 'wb') as fo:
                            pickle.dump(roi_mask_skipp, fo)
                    elif tif_idx > 0 and os.path.exists(f'{processed_recordings}/{condition}/{fly}/_ROIS_skipp.pkl') == False:
                        viewer = napari.Viewer()
                        image_layer = viewer.add_image(rep_frame_o, name='median')
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
                        roi_mask_skipp = {"roi_mask": roi_mask}
                        with open(f'{processed_recordings}/{condition}/{fly}/_ROIS_skipp.pkl', 'wb') as fo:
                            pickle.dump(roi_mask_skipp, fo)
                    
                    # --- Correlation within ROI ---
                    # --- Process both original and shuffled stacks ---
                    # if os.path.exists(f'{processed_recordings}/{condition}/{fly}/{tif_name}/pulse_average_positive.png'):
                    #     print('skiped', tif_name)
                    #     with open(f'{processed_recordings}/{condition}/{fly}/{tif_name}/all_results_tif.pkl', 'rb') as file:
                    #         tif_result = pickle.load(file)
                    #     fly_result[tif_name] = tif_result["tif_result"]
                    #     fly_pulse_result[tif_name] = tif_result["tif_pulse_result"]
                    #     fly_result_shuffled[tif_name] = tif_result["tif_result_shuffled"]
                    #     fly_pulse_result_shuffled[tif_name] = tif_result["tif_pulse_result_shuffled"]

                    #     fly_pulse_result_cut_pulse[tif_name] = tif_result["tif_pulse_result_cut_pulse"]
                    #     fly_pulse_result_cut_pulse_shuffled[tif_name] = tif_result["tif_pulse_result_cut_pulse_shuffled"]
                    #     print('loaded', tif_name)
                    #     continue
                    all_tiff ={}
                    for direction in ['positive', 'negative']:
                        substack = stack[:, roi_mask]

                        #TODO: more controls, have very high df f values> filter non fluorescent pixel traces
                        # results_real = analyze_roi_dff_first(stack, roi_mask, stim, reps, n_pre, n_width, n_post, n_isi,pulse_protocol, fps, direction=direction, control_mode=None)
                        # corr_map, roi_mask_refined, dff, n_pixels_original, dff_pulse, pulse_avg = extract_data(results_real)
                        # plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, stim, '', n_pixels_original, '_sortdff', dff_pulse)
                        # tif_result_sortdff[direction] = {"dff": dff, 'fps': fps}
                        # tif_pulse_result_sortdff[direction] = {'fps': fps, 'pulse_mean': pulse_avg}
                        # # Control 1
                        # results_ctrl1 = analyze_roi_dff_first(stack, roi_mask, stim, reps, n_pre, n_width, n_post, n_isi,pulse_protocol, fps, direction=direction, control_mode='shuffle_before_dff', seed=42)
                        # corr_map, roi_mask_refined, dff, n_pixels_original, dff_pulse, pulse_avg = extract_data(results_ctrl1)
                        # plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, stim, '(Phase Randomized)', n_pixels_original, '_shuffle_before_dff', dff_pulse)
                        # tif_result_shuffle_before_dff[direction] = {"dff": dff, 'fps': fps}
                        # tif_pulse_result_shuffle_before_dff[direction] = {'fps': fps, 'pulse_mean': pulse_avg}
                        # # Control 2
                        # results_ctrl2 = analyze_roi_dff_first(stack, roi_mask, stim, reps, n_pre, n_width, n_post, n_isi,pulse_protocol, fps, direction=direction, control_mode='shuffle_after_filter', seed=42)
                        # corr_map, roi_mask_refined, dff, n_pixels_original, dff_pulse, pulse_avg = extract_data(results_ctrl2)
                        # plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, stim, '(Phase Randomized)', n_pixels_original, '_shuffle_after_filter', dff_pulse)
                        # tif_result_shuffle_after_filter[direction] = {"dff": dff, 'fps': fps}
                        # tif_pulse_result_shuffle_after_filter[direction] = {'fps': fps, 'pulse_mean': pulse_avg}

                        #######
                        
                        # roi_trace, roi_mask_refined, corr_map, valid_corrs, valid_segments = get_corr_pixels_per_pulse(substack, pulse_protocol, roi_mask, reps, n_pre, n_width, n_post, n_isi, direction)
                        roi_trace, roi_mask_refined, corr_map, lag_map, valid_corrs, valid_lags, valid_segments, all_xcorr, all_lags = get_xcorr_pixels_per_pulse(substack, pulse_protocol, roi_mask, reps, n_pre, n_width, n_post, n_isi, direction, max_lag=fps*1)
                        plot_lag_maps(lag_map, corr_map, roi_mask_refined, output_tif, direction, "_cut_pulse_lag_",title='Spatial Maps', frame_rate=1/fps)
                        if len(all_lags) > 0:
                            lag_values_cut_pulse.extend(all_lags.flatten() if hasattr(all_lags, 'flatten') else all_lags)
                            xcorr_values_cut_pulse.extend(all_xcorr.flatten() if hasattr(all_xcorr, 'flatten') else all_xcorr)
                        # dff, dff_pulse, n_pixels_original, pulses_array, pulses, pulse_avg, pulse_protocol = get_dff_cut_pulse(n_pre, n_width, n_post,stack, roi_mask_refined, valid_segments, pulse_protocol, fps)
                        dff, n_pixels_original = get_dff_cut_pulse(roi_trace, n_pre, n_width, n_post,stack, roi_mask_refined, valid_segments, pulse_protocol, fps)
                        plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, pulse_protocol, '', n_pixels_original, '_cut_pulse', [], True, [-5, 15])
                        tif_pulse_result_cut_pulse[direction] = {"dff": dff, 'fps': fps}
                        substack_shuff = stack_shuffled[:, roi_mask]
                        # roi_trace, roi_mask_refined, corr_map, valid_corrs, valid_segments = get_corr_pixels_per_pulse(substack_shuff, pulse_protocol, roi_mask, reps, n_pre, n_width, n_post, n_isi, direction)
                        roi_trace, roi_mask_refined, corr_map, lag_map, valid_corrs, valid_lags, valid_segments, all_xcorr, all_lags = get_xcorr_pixels_per_pulse(substack_shuff, pulse_protocol, roi_mask, reps, n_pre, n_width, n_post, n_isi, direction, max_lag=fps*1)
                        plot_lag_maps(lag_map, corr_map, roi_mask_refined, output_tif, direction, "_cut_pulse_lag_shuffled_",title='Spatial Maps (Phase Randomized)', frame_rate=1/fps)
                        if len(all_lags) > 0:
                            lag_values_cut_pulse_shuffled.extend(all_lags.flatten() if hasattr(all_lags, 'flatten') else all_lags)
                            xcorr_values_cut_pulse_shuffled.extend(all_xcorr.flatten() if hasattr(all_xcorr, 'flatten') else all_xcorr)
                        # dff, dff_pulse, n_pixels_original, pulses_array, pulses, pulse_avg, pulse_protocol = get_dff_cut_pulse(n_pre, n_width, n_post,stack, roi_mask_refined, valid_segments, pulse_protocol, fps)
                        dff, n_pixels_original = get_dff_cut_pulse(roi_trace, n_pre, n_width, n_post,stack, roi_mask_refined, valid_segments, pulse_protocol, fps)
                        plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, pulse_protocol, '(Phase Randomized)', n_pixels_original, '_cut_pulse_shuffled', [], True, [-5, 15])
                        tif_pulse_result_cut_pulse_shuffled[direction] = {"dff": dff, 'fps': fps}

                        # roi_trace, roi_mask_refined, corr_map = get_corr_pixels(substack, stim, roi_mask, direction=direction)
                        roi_trace, roi_mask_refined, corr_map, lag_map, all_lags, all_xcorr = get_xcorr_pixels(substack, stack, stim, roi_mask, direction=direction, max_lag=fps*1)
                        if len(all_lags) > 0:
                            lag_values_all.extend(all_lags.flatten() if hasattr(all_lags, 'flatten') else all_lags)
                            xcorr_values_all.extend(all_xcorr.flatten() if hasattr(all_xcorr, 'flatten') else all_xcorr)
                        plot_lag_maps(lag_map, corr_map, roi_mask_refined, output_tif, direction, "",title='Spatial Maps ', frame_rate=1/fps)
                        dff, dff_pulse, n_pixels_original, pulses_array, pulses, pulse_avg, pulse_protocol = get_dff(reps, n_pre, n_width, n_post, n_isi, stack, roi_mask_refined, roi_trace, pulse_protocol, fps)
                        plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, stim, '', n_pixels_original, '', dff_pulse, False,  [-5, 15])
                        # --- Save figures (only for original stack to avoid duplication in first plot) ---
                        tif_result[direction] = {"dff": dff, 'fps': fps}
                        tif_pulse_result[direction] = {'fps': fps, 'pulse_mean': pulse_avg}
                        substack_shuff = stack_shuffled[:, roi_mask]
                        # roi_trace, roi_mask_refined, corr_map = get_corr_pixels(substack_shuff, stim, roi_mask, direction=direction)
                        roi_trace, roi_mask_refined, corr_map, lag_map, all_lags, all_xcorr = get_xcorr_pixels(substack_shuff, stack_shuffled, stim, roi_mask, direction=direction, max_lag=fps*1)
                        if len(all_lags) > 0:
                            lag_values_all_shuffled.extend(all_lags.flatten() if hasattr(all_lags, 'flatten') else all_lags)
                            xcorr_values_all_shuffled.extend(all_xcorr.flatten() if hasattr(all_xcorr, 'flatten') else all_xcorr)
                        plot_lag_maps(lag_map, corr_map, roi_mask_refined, output_tif, direction, "_lag_shuffeled_",title='Spatial Maps (Phase Randomized)', frame_rate=1/fps)
                        dff, dff_pulse, n_pixels_original, pulses_array, pulses, pulse_avg, pulse_protocol = get_dff(reps, n_pre, n_width, n_post, n_isi, stack, roi_mask_refined, roi_trace, pulse_protocol, fps)
                        plot_tif_results(direction, rep_frame_o, rep_frame, corr_map, roi_mask_refined, output_tif, dff, stim, '(Phase Randomized)', n_pixels_original, '_shuffled', dff_pulse, False,  [-5, 15])
                        
                        tif_result_shuffled[direction] = {"dff": dff, 'fps': fps}
                        tif_pulse_result_shuffled[direction] = {'fps': fps, 'pulse_mean': pulse_avg}
                    all_tiff['tif_result'] = tif_result
                    all_tiff['tif_pulse_result'] = tif_pulse_result
                    all_tiff['tif_result_shuffled'] = tif_result_shuffled
                    all_tiff['tif_pulse_result_shuffled'] = tif_pulse_result_shuffled
                    all_tiff['tif_pulse_result_cut_pulse'] = tif_pulse_result_cut_pulse
                    all_tiff['tif_pulse_result_cut_pulse_shuffled'] = tif_pulse_result_cut_pulse_shuffled
                    with open(f'{processed_recordings}/{condition}/{fly}/{tif_name}/all_results_tif.pkl', "wb") as fp:
                        pickle.dump(all_tiff, fp) 

                    fly_result[tif_name] = tif_result
                    fly_pulse_result[tif_name] = tif_pulse_result
                    fly_result_shuffled[tif_name] = tif_result_shuffled
                    fly_pulse_result_shuffled[tif_name] = tif_pulse_result_shuffled

                    fly_pulse_result_cut_pulse[tif_name] = tif_pulse_result_cut_pulse
                    fly_pulse_result_cut_pulse_shuffled[tif_name] = tif_pulse_result_cut_pulse_shuffled

                    
                    # fly_result_sortdff[tif_name] = tif_result_sortdff
                    # fly_pulse_result_sortdff[tif_name] = tif_pulse_result_sortdff

                    # fly_result_shuffle_before_dff[tif_name] = tif_result_shuffle_before_dff
                    # fly_pulse_result_shuffle_before_dff[tif_name] = tif_pulse_result_shuffle_before_dff

                    # fly_result_shuffle_after_filter[tif_name] = tif_result_shuffle_after_filter
                    # fly_pulse_result_shuffle_after_filter[tif_name] = tif_pulse_result_shuffle_after_filter
                fly_pulse_average_cut_pulse = average_across_tseries(fly_pulse_result_cut_pulse)
                plot_fly_avg(fly_pulse_average_cut_pulse, pulse_protocol, tif_files, '', output_fly, '_cut_pulse', [], pulse_protocol, True)
                fly_pulse_average_cut_pulse = interpolate_to_mean_fps(fly_pulse_average_cut_pulse, mean_fps)
                condition_pulse_results_cut_pulse[fly] = fly_pulse_average_cut_pulse

                fly_pulse_average_cut_pulse_shuffled = average_across_tseries(fly_pulse_result_cut_pulse_shuffled)
                plot_fly_avg(fly_pulse_average_cut_pulse_shuffled, pulse_protocol, tif_files, '(Phase Randomized)', output_fly, '_cut_pulse_shuffled', [], pulse_protocol, True)
                fly_pulse_average_cut_pulse_shuffled = interpolate_to_mean_fps(fly_pulse_average_cut_pulse_shuffled, mean_fps)
                condition_pulse_results_cut_pulse_shuffled[fly] = fly_pulse_average_cut_pulse_shuffled

                # fly_average_sortdff = average_across_tseries(fly_result_sortdff)
                # fly_pulse_average_sortdff = average_across_tseries(fly_pulse_result_sortdff)
                # plot_fly_avg(fly_average_sortdff, stim, tif_files, '', output_fly, '_sortdff', fly_pulse_average_sortdff, pulse_protocol,False)
                # fly_average_sortdff = interpolate_to_mean_fps(fly_average_sortdff, mean_fps)
                # condition_results_sortdff[fly] = fly_average_sortdff
                # fly_pulse_average_sortdff = interpolate_to_mean_fps(fly_pulse_average_sortdff, mean_fps)
                # condition_pulse_results_sortdff[fly] = fly_pulse_average_sortdff

                # fly_average_shuffle_before_dff = average_across_tseries(fly_result_shuffle_before_dff)
                # fly_pulse_average_shuffle_before_dff = average_across_tseries(fly_pulse_result_shuffle_before_dff)
                # plot_fly_avg(fly_average_shuffle_before_dff, stim, tif_files, '(Phase Randomized)', output_fly, '_shuffle_before_dff', fly_pulse_average_shuffle_before_dff, pulse_protocol, False)
                # fly_average_shuffle_before_dff = interpolate_to_mean_fps(fly_average_shuffle_before_dff, mean_fps)
                # condition_results_shuffle_before_dff[fly] = fly_average_shuffle_before_dff
                # fly_pulse_average_shuffle_before_dff = interpolate_to_mean_fps(fly_pulse_average_shuffle_before_dff, mean_fps)
                # condition_pulse_results_shuffle_before_dff[fly] = fly_pulse_average_shuffle_before_dff

                # fly_average_shuffle_after_filter = average_across_tseries(fly_result_shuffle_after_filter)
                # fly_pulse_average_shuffle_after_filter = average_across_tseries(fly_pulse_result_shuffle_after_filter)
                # plot_fly_avg(fly_average_shuffle_after_filter, stim, tif_files, '(Phase Randomized)', output_fly, '_shuffle_after_filter', fly_pulse_average_shuffle_after_filter, pulse_protocol, False)
                # fly_average_shuffle_after_filter = interpolate_to_mean_fps(fly_average_shuffle_after_filter, mean_fps)
                # condition_results_shuffle_after_filter[fly] = fly_average_shuffle_after_filter
                # fly_pulse_average_shuffle_after_filter = interpolate_to_mean_fps(fly_pulse_average_shuffle_after_filter, mean_fps)
                # condition_pulse_results_shuffle_after_filter[fly] = fly_pulse_average_shuffle_after_filter

                # Average across TSeries for original data
                fly_average = average_across_tseries(fly_result)
                fly_pulse_average = average_across_tseries(fly_pulse_result)
                plot_fly_avg(fly_average, stim, tif_files, '', output_fly, '', fly_pulse_average, pulse_protocol, False)
                fly_average = interpolate_to_mean_fps(fly_average, mean_fps)
                condition_results[fly] = fly_average
                fly_pulse_average = interpolate_to_mean_fps(fly_pulse_average, mean_fps)
                condition_pulse_results[fly] = fly_pulse_average

                fly_average_shuffled = average_across_tseries(fly_result_shuffled)
                fly_pulse_average_shuffled = average_across_tseries(fly_pulse_result_shuffled)
                plot_fly_avg(fly_average_shuffled, stim, tif_files, '(Phase Randomized)', output_fly, '_shuffled', fly_pulse_average_shuffled, pulse_protocol, False)
                fly_average_shuffled = interpolate_to_mean_fps(fly_average_shuffled, mean_fps)
                condition_results_shuffled[fly] = fly_average_shuffled
                fly_pulse_average_shuffled = interpolate_to_mean_fps(fly_pulse_average_shuffled, mean_fps)
                condition_pulse_results_shuffled[fly] = fly_pulse_average_shuffled

                all_flies["fly_pulse_average_cut_pulse"] = fly_pulse_average_cut_pulse
                all_flies["fly_pulse_average_cut_pulse_shuffled"] = fly_pulse_average_cut_pulse_shuffled
                all_flies["fly_average"] = fly_average
                all_flies["fly_pulse_average"] = fly_pulse_average
                all_flies["fly_average_shuffled"] = fly_average_shuffled
                all_flies["fly_pulse_average_shuffled"] = fly_pulse_average_shuffled

                with open(f'{processed_recordings}/{condition}/{fly}/all_results_flies.pkl', "wb") as fp:
                        pickle.dump(all_flies, fp) 
                
    else:
        continue
    protocoll_new, pulse_protocol_new = make_olf_protocoll(mean_fps)
    # Condition-level averaging

    # condition_average = average_across_flies(condition_results_sortdff)
    # condition_pulse_average = average_across_flies(condition_pulse_results_sortdff)
    # plot_condition_avg(condition_average, protocoll_new, '', outpout_condition, '_sortdff', condition_pulse_average, pulse_protocol_new, False)

    # condition_average = average_across_flies(condition_results_shuffle_before_dff)
    # condition_pulse_average = average_across_flies(condition_pulse_results_shuffle_before_dff)
    # plot_condition_avg(condition_average, protocoll_new, '(Phase Randomized)', outpout_condition, '_shuffle_before_dff', condition_pulse_average, pulse_protocol_new, False)

    # condition_average = average_across_flies(condition_results_shuffle_after_filter)
    # condition_pulse_average = average_across_flies(condition_pulse_results_shuffle_after_filter)
    # plot_condition_avg(condition_average, protocoll_new, '(Phase Randomized)', outpout_condition, '_shuffle_after_filter', condition_pulse_average, pulse_protocol_new, False)


    condition_pulse_average_cut_pulse = average_across_flies(condition_pulse_results_cut_pulse)
    plot_condition_avg(condition_pulse_average_cut_pulse, pulse_protocol_new, '', outpout_condition, '_cut_pulse', [], pulse_protocol_new, True, [-1,8])

    condition_pulse_average_cut_pulse_shuffled = average_across_flies(condition_pulse_results_cut_pulse_shuffled)
    plot_condition_avg(condition_pulse_average_cut_pulse_shuffled, pulse_protocol_new, '(Phase Randomized)', outpout_condition, '_cut_pulse_shuffled', [], pulse_protocol_new, True, [-1,8])


    lag_values_all = np.array(lag_values_all)
    xcorr_values_all = np.array(xcorr_values_all)
    lag_values_all_shuffled = np.array(lag_values_all_shuffled)
    xcorr_values_all_shuffled = np.array(xcorr_values_all_shuffled)
    lag_values_cut_pulse = np.array(lag_values_cut_pulse)
    xcorr_values_cut_pulse = np.array(xcorr_values_cut_pulse)
    lag_values_cut_pulse_shuffled = np.array(lag_values_cut_pulse_shuffled)
    xcorr_values_cut_pulse_shuffled = np.array(xcorr_values_cut_pulse_shuffled)
    
    # Plot all distributions
    if len(lag_values_all) > 0:
        plot_lag_xcorr_distribution(lag_values_all, xcorr_values_all, outpout_condition, "_lag_xcorr_distribution_", normalize=True, bins=50, title='Cross-Correlation vs Lag', frame_rate=1/fps)
    if len(lag_values_all_shuffled) > 0:
        plot_lag_xcorr_distribution(lag_values_all_shuffled, xcorr_values_all_shuffled, outpout_condition, "_lag_xcorr_distribution_shuffled", normalize=True, bins=50, title='Cross-Correlation vs Lag (Shuffled)', frame_rate=1/fps)
    if len(lag_values_cut_pulse) > 0:
        plot_lag_xcorr_distribution(lag_values_cut_pulse, xcorr_values_cut_pulse, outpout_condition, "_lag_xcorr_cut_pulse_distribution_", normalize=True, bins=50, title='Cross-Correlation vs Lag (Cut Pulse)', frame_rate=1/fps)
    if len(lag_values_cut_pulse_shuffled) > 0:
        plot_lag_xcorr_distribution(lag_values_cut_pulse_shuffled, xcorr_values_cut_pulse_shuffled, outpout_condition, "_lag_xcorr_cut_pulse_distribution_shuffled", normalize=True, bins=50, title='Cross-Correlation vs Lag (Cut Pulse Shuffled)', frame_rate=1/fps)

    condition_average = average_across_flies(condition_results)
    condition_pulse_average = average_across_flies(condition_pulse_results)
    plot_condition_avg(condition_average, protocoll_new, '', outpout_condition, '', condition_pulse_average, pulse_protocol_new, False, [-1,3])

    condition_average_shuffled = average_across_flies(condition_results_shuffled)
    condition_pulse_average_shuffled = average_across_flies(condition_pulse_results_shuffled)
    plot_condition_avg(condition_average_shuffled, protocoll_new, '(Phase Randomized)', outpout_condition, '_shuffled', condition_pulse_average_shuffled, pulse_protocol_new, False, [-1,3])
    