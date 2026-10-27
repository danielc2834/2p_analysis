import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from scipy.signal import butter, filtfilt
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, dendrogram as sp_dendrogram
from collections import defaultdict
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
import umap
import warnings
from scipy import stats
from sklearn.metrics import silhouette_samples
from sklearn.metrics import adjusted_rand_score
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
import joblib
warnings.filterwarnings('ignore')

def create_stimulus_protocol(mean_fps):
    """
    Create stimulus protocol based on predefined values and durations.
    
    Parameters:
    -----------
    mean_fps : float
        Frames per second used for the protocol
    
    Returns:
    --------
    numpy.array
        Stimulus protocol values
    numpy.array
        Time vector for stimulus (in seconds)
    dict
        Segment boundaries {'polarity': (start, end), 'frequency': (start, end), ...}
    """
    value = [0,0,1,0,0.5,0.5,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0.5,0.53125,0.4375,0.59375,0.375,0.65625,0.3125,0.71875,0.25,0.78125,0.1875,0.84375,0.125,0.90625,0.0625,0.96875,0,0.5,0.6,0.7,0.8,0.9,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,0.1,0.2,0.3,0.4,0.5,0]
    duration = [4,2,3,3,2,0.243,0.449,0.406,0.372,0.343,0.317,0.296,0.277,0.26,0.246,0.232,0.221,0.21,0.2,0.192,0.183,0.176,0.169,0.163,0.157,0.151,0.146,0.142,0.137,0.133,0.129,0.125,0.122,0.118,0.115,0.112,0.11,0.107,0.104,0.102,0.099,0.097,0.095,0.093,0.091,0.09,0.087,0.086,0.084,0.082,0.081,0.079,0.078,0.077,0.075,0.074,0.073,0.072,0.07,0.069,0.068,0.068,0.066,0.065,0.064,0.063,0.062,0.062,0.06,0.06,0.059,0.058,0.058,0.056,0.056,0.055,0.055,0.054,0.053,0.052,0.052,0.052,0.05,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,2,2]
    
    protocol = []
    for val, dur in zip(value, duration):
        frames = mean_fps * dur
        segment = [val] * round(frames)
        protocol.extend(segment)
    
    protocol = np.array(protocol)
    time_stim = np.arange(len(protocol)) / mean_fps
    
    # Define segment boundaries (in seconds)
    segments = {
        'polarity': (0, 14),
        'frequency': (14, 14 + 11),
        'contrast': (14 + 11, 14 + 11 + 10),
        'luminance': (14 + 11 + 10, time_stim[-1])
    }
    
    return protocol, time_stim, segments

def segment_trace(trace, time, segments, mean_fps):
    """
    Segment trace into predefined time windows.
    
    Parameters:
    -----------
    trace : array
        Fluorescence trace
    time : array
        Time vector
    segments : dict
        Dictionary with segment names and (start, end) times
    mean_fps : float
        Frames per second
    
    Returns:
    --------
    dict
        Dictionary of segmented traces {segment_name: (trace_segment, time_segment)}
    """
    segmented = {}
    n = len(trace)
    for name, (start, end) in segments.items():
        start_idx = min(int(start * mean_fps), n - 1)
        end_idx   = min(int(end   * mean_fps), n)
        if start_idx >= end_idx:
            # segment falls outside trace — return empty arrays
            segmented[name] = (np.array([]), np.array([]))
            continue
        
        trace_seg = trace[start_idx:end_idx]
        time_seg = time[start_idx:end_idx] - time[start_idx]  # Reset to start at 0
        segmented[name] = (trace_seg, time_seg)
    
    return segmented

def interpolate_trace(df_trace, original_fps, target_fps, multi, degen, trim_seconds=5, detrend=False):
    """
    Interpolate df_trace to target FPS using linear interpolation and trim edges.
    
    Parameters:
    -----------
    df_trace : array-like
        Original fluorescence trace
    original_fps : float
        Original frames per second
    target_fps : float
        Target frames per second for interpolation
    trim_seconds : float
        Number of seconds to trim from start and end
    
    Returns:
    --------
    numpy.array
        Interpolated and trimmed trace
    numpy.array
        Time vector for the trace
    """
    n_frames = len(df_trace)
    original_duration = (n_frames - 1) / original_fps
    original_time = np.arange(n_frames) / original_fps
    
    # Create new time points at target FPS
    n_new_frames = int(original_duration * target_fps) + 1
    new_time = np.linspace(0, original_duration, n_new_frames)
    
    # Interpolate
    f = interp1d(original_time, df_trace, kind='linear', fill_value='extrapolate')
    interpolated_trace = f(new_time)
    
    # Trim first and last trim_seconds
    trim_frames = int(trim_seconds * target_fps)
    
    if len(interpolated_trace) > 2 * trim_frames:
        if multi:
            interpolated_trace = interpolated_trace[trim_frames:-trim_frames]
            new_time = new_time[trim_frames:-trim_frames]
        elif degen:
            interpolated_trace = interpolated_trace[trim_frames:]  
            new_time = new_time[trim_frames:]
        elif temp:
            interpolated_trace = interpolated_trace[trim_frames:]  
            new_time = new_time[trim_frames:]
        ##############################################
        
        # Reset time to start at 0
        new_time = new_time - new_time[0]
    
    return interpolated_trace, new_time

def plot_traces_with_mean(traces, time, stim_protocol, stim_time, title, ylabel='ΔF/F', alpha=0.3, ylim=None):
    """
    Plot individual traces and their mean with stimulus protocol on top.
    
    Parameters:
    -----------
    traces : list of arrays
        List of fluorescence traces
    time : array
        Time vector for traces
    stim_protocol : array
        Stimulus protocol values
    stim_time : array
        Time vector for stimulus
    title : str
        Plot title
    ylabel : str
        Y-axis label
    alpha : float
        Transparency for individual traces
    """
    fig = plt.figure(figsize=(16, 8))
    
    # Create gridspec for subplots with height ratio
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
    
    # Determine axis limits
    max_time = max(time[-1], stim_time[-1])

    # Top subplot: Stimulus protocol (without axes)
    ax_stim = fig.add_subplot(gs[0])
    ax_stim.fill_between(stim_time, 0, stim_protocol, color='lightblue', alpha=0.7, linewidth=0)
    ax_stim.plot(stim_time, stim_protocol, color='blue', linewidth=1.5)
    ax_stim.set_xlim(0, max_time)
    ax_stim.set_ylim(-0.1, 1.1)
    ax_stim.axis('off')
    ax_stim.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Bottom subplot: Fluorescence traces
    ax_trace = fig.add_subplot(gs[1])
    
    # Plot individual traces
    for i, trace in enumerate(traces):
        ax_trace.plot(time, trace, alpha=alpha, color='gray', linewidth=0.5)
    
    # Calculate and plot mean
    mean_trace = np.mean(traces, axis=0)
    sem_trace = np.std(traces, axis=0) / np.sqrt(len(traces))
    
    ax_trace.plot(time, mean_trace, color='red', linewidth=2, label=f'Mean (n={len(traces)})')
    ax_trace.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, 
                          color='red', alpha=0.2, label='SEM')
    
    ax_trace.set_xlabel('Time (s)', fontsize=12)
    ax_trace.set_ylabel(ylabel, fontsize=12)
    ax_trace.legend(loc='upper right')
    ax_trace.grid(True, alpha=0.3)
    ax_trace.set_xlim(0, max_time)
    if ylim is not None:
        ax_trace.set_ylim(ylim[0], ylim[1])
    
    plt.tight_layout()
    return fig, mean_trace

def extract_roi_features(trace, time, segments, mean_fps,
                         polarity_results=None, luminance_results=None,
                         contrast_results=None, frequency_results=None):
    """
    Extract a per-ROI feature vector from a single replicate-averaged trace.
    All features are computed directly from this ROI's own trace — no
    population-level values are used, so every ROI gets a unique feature vector.

    Parameters kept for backward-compatibility but no longer used:
        polarity_results, luminance_results, contrast_results, frequency_results

    Returns
    -------
    numpy.array  (1-D, length = n_features)
    list         feature names (same order)
    """
    features = []
    names = []

    seg_names = ['polarity', 'frequency', 'contrast', 'luminance']

    for seg_name, (t_start, t_end) in segments.items():
        i0 = int(t_start * mean_fps)
        i1 = int(t_end   * mean_fps)
        i1 = min(i1, len(trace))
        seg = trace[i0:i1]
        if len(seg) == 0:
            seg = np.array([0.0])

        # baseline: first 0.5 s of segment
        bl_frames = max(1, int(0.5 * mean_fps))
        baseline  = np.mean(seg[:bl_frames])
        delta     = seg - baseline

        # peak ΔF/F
        peak = float(np.max(delta))
        features.append(peak);  names.append(f'{seg_name}_peak')

        # trough ΔF/F
        trough = float(np.min(delta))
        features.append(trough); names.append(f'{seg_name}_trough')

        # AUC (trapezoid, absolute)
        auc = float(np.trapz(np.abs(delta)))
        features.append(auc);   names.append(f'{seg_name}_auc')

        # Peak latency (frames → seconds)
        peak_lat = float(np.argmax(delta)) / mean_fps
        features.append(peak_lat); names.append(f'{seg_name}_peak_lat')

        # Rise time: frames from 10 % to 90 % of peak (only if peak > 0)
        if peak > 0:
            thresh10 = 0.10 * peak
            thresh90 = 0.90 * peak
            above10  = np.where(delta >= thresh10)[0]
            above90  = np.where(delta >= thresh90)[0]
            if len(above10) > 0 and len(above90) > 0:
                rise = (above90[0] - above10[0]) / mean_fps
            else:
                rise = 0.0
        else:
            rise = 0.0
        features.append(rise); names.append(f'{seg_name}_rise')

    # --- Per-ROI stimulus-locked scalars (all computed from this ROI's trace) ---
    # These replace the previous population-level lookups (overall_on_mean etc.)
    # which gave every ROI in the same condition the same value.
    # The trace passed in is already the replicate-averaged trace for this ROI.

    # ON amplitude: mean ΔF/F in 6-8 s of polarity segment, baseline-subtracted
    # OFF amplitude: mean ΔF/F in 9-11 s of polarity segment, baseline-subtracted
    pol_start, pol_end = segments.get('polarity', (0, 14))
    i0p = int(pol_start * mean_fps)
    i1p = min(int(pol_end * mean_fps), len(trace))
    pol_seg = trace[i0p:i1p]
    on_w_s,  on_w_e  = int(6 * mean_fps), min(int(8  * mean_fps), len(pol_seg))
    off_w_s, off_w_e = int(9 * mean_fps), min(int(11 * mean_fps), len(pol_seg))
    # baseline: 1 s before ON transition (frames 5-6 s)
    bl_s, bl_e = int(5 * mean_fps), int(6 * mean_fps)
    bl_pol = float(np.mean(pol_seg[bl_s:bl_e])) if bl_e <= len(pol_seg) else 0.0

    if on_w_e > on_w_s and on_w_e <= len(pol_seg):
        on_amp = float(np.mean(pol_seg[on_w_s:on_w_e])) - bl_pol
    else:
        on_amp = 0.0
    if off_w_e > off_w_s and off_w_e <= len(pol_seg):
        off_amp = float(np.mean(pol_seg[off_w_s:off_w_e])) - bl_pol
    else:
        off_amp = 0.0
    features.append(on_amp);  names.append('on_amplitude')
    features.append(off_amp); names.append('off_amplitude')

    # Luminance slope: linear fit of mean response at each luminance step.
    # Mirrors analyze_luminance_response but for a single ROI trace.
    lum_values    = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                     0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # lum_durations = [2] + [0.5] * 16 + [2] + [0.5] * 4 + [0.5]  # seconds per step
    lum_durations = [2] + [0.5] * 21 
    lum_start, lum_end = segments.get('luminance', (35, 47))
    i0l = int(lum_start * mean_fps)
    lum_seg = trace[i0l:min(int(lum_end * mean_fps), len(trace))]
    # baseline: first step (2 s of luminance=0)
    bl_frames_l = max(1, int(lum_durations[0] * mean_fps))
    bl_lum      = float(np.mean(lum_seg[:bl_frames_l])) if len(lum_seg) >= bl_frames_l else 0.0
    step_means, step_lum_vals = [], []
    cur = 0
    for lv, dur in zip(lum_values, lum_durations):
        nf = int(dur * mean_fps)
        if cur + nf <= len(lum_seg):
            step_means.append(float(np.mean(lum_seg[cur:cur + nf])) - bl_lum)
            step_lum_vals.append(lv)
        cur += nf
    if len(step_lum_vals) >= 2:
        lum_slope, _ = np.polyfit(step_lum_vals, step_means, 1)
        lum_slope = float(lum_slope)
    else:
        lum_slope = 0.0
    features.append(lum_slope); names.append('lum_slope')

    # --- Stimulus-locked response indices (Baden et al. 2016 style) ----------
    # ON-OFF polarity index: reuse on_amp / off_amp computed above
    denom = abs(on_amp) + abs(off_amp)
    oo_idx = (on_amp - off_amp) / denom if denom > 0 else 0.0
    features.append(oo_idx); names.append('on_off_index')

    # Frequency tuning peak: extract max response per temporal frequency bin,
    # return the frequency (Hz) at which the ROI responds most strongly.
    # Uses the same flash_durations and binning as analyze_frequency_response.
    flash_durations = [
        0.243, 0.449, 0.406, 0.372, 0.343, 0.317, 0.296, 0.277, 0.26, 0.246,
        0.232, 0.221, 0.21, 0.2, 0.192, 0.183, 0.176, 0.169, 0.163, 0.157,
        0.151, 0.146, 0.142, 0.137, 0.133, 0.129, 0.125, 0.122, 0.118, 0.115,
        0.112, 0.11, 0.107, 0.104, 0.102, 0.099, 0.097, 0.095, 0.093, 0.091,
        0.09, 0.087, 0.086, 0.084, 0.082, 0.081, 0.079, 0.078, 0.077, 0.075,
        0.074, 0.073, 0.072, 0.07, 0.069, 0.068, 0.068, 0.066, 0.065, 0.064,
        0.063, 0.062, 0.062, 0.06, 0.06, 0.059, 0.058, 0.058, 0.056, 0.056,
        0.055, 0.055, 0.054, 0.053, 0.052, 0.052, 0.052, 0.05]
    flash_freqs = [1.0 / (2 * d) for d in flash_durations]
    freq_bin_edges = list(range(0, 11))   # 0-1, 1-2, ..., 9-10 Hz

    freq_start, freq_end = segments.get('frequency', (14, 25))
    i0f = int(freq_start * mean_fps)
    # baseline: first 0.5 s of frequency segment
    bl_frames_f = max(1, int(0.5 * mean_fps))
    freq_baseline = float(np.mean(trace[i0f:i0f + bl_frames_f])) \
                    if i0f + bl_frames_f <= len(trace) else 0.0

    bin_responses = defaultdict(list)
    cur_f = i0f + bl_frames_f
    for freq, dur in zip(flash_freqs, flash_durations):
        nf = max(1, int(dur * mean_fps))
        if cur_f + nf > len(trace):
            break
        seg_f = trace[cur_f:cur_f + nf]
        amp_f = float(np.max(seg_f - freq_baseline))
        # assign to 1-Hz bin
        for b in range(len(freq_bin_edges) - 1):
            if freq_bin_edges[b] <= freq < freq_bin_edges[b + 1]:
                bin_responses[freq_bin_edges[b]].append(amp_f)
                break
        cur_f += nf

    if bin_responses:
        bin_means = {b: np.mean(v) for b, v in bin_responses.items()}
        freq_peak = float(max(bin_means, key=bin_means.get))
    else:
        freq_peak = 0.0
    features.append(freq_peak); names.append('freq_tuning_peak_hz')

    # Contrast tuning peak: extract mean response per absolute contrast step,
    # return the Weber contrast (%) at which the ROI responds most strongly.
    # Uses the same contrast_values and durations as analyze_contrast_response.
    contrast_values    = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                          0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    contrast_durations = [2.0] + [0.5] * 15
    background = 0.5

    cont_start, cont_end = segments.get('contrast', (25, 35))
    i0c = int(cont_start * mean_fps)
    bl_frames_c  = max(1, int(contrast_durations[0] * mean_fps))
    cont_baseline = float(np.mean(trace[i0c:i0c + bl_frames_c])) \
                    if i0c + bl_frames_c <= len(trace) else 0.0

    abs_contrast_responses = {}
    cur_c = i0c
    for cv, dur in zip(contrast_values, contrast_durations):
        nf = max(1, int(dur * mean_fps))
        if cur_c + nf <= len(trace):
            seg_c = trace[cur_c:cur_c + nf]
            amp_c = float(np.mean(seg_c)) - cont_baseline
            abs_c = round(abs(cv - background) / background * 100, 1)
            if abs_c not in abs_contrast_responses:
                abs_contrast_responses[abs_c] = []
            abs_contrast_responses[abs_c].append(amp_c)
        cur_c += nf

    if abs_contrast_responses:
        abs_means = {c: np.mean(v) for c, v in abs_contrast_responses.items()}
        cont_peak = float(max(abs_means, key=abs_means.get))
    else:
        cont_peak = 0.0
    features.append(cont_peak); names.append('contrast_tuning_peak_pct')

    return np.array(features, dtype=float), names


def prepare_clustering_data(roi_list, feature_set, segments, mean_fps,
                             polarity_results=None, luminance_results=None,
                             contrast_results=None, frequency_results=None):
    """
    Build data matrix X for clustering.

    Parameters
    ----------
    roi_list : list of (trace, fly_id, tseries_key)
    feature_set : 'full' | 'segment_<name>' | 'features'
        'full'            -> raw full trace vector (aligned to min length)
        'segment_polarity' etc. -> raw segment trace vector
        'features'        -> extracted scalar feature vector

    Returns
    -------
    X          : (n_rois, n_features) float array
    fly_labels : list of fly_id strings (length n_rois)
    traces     : list of raw full traces (for plotting)
    times      : list of time vectors
    valid_idx  : indices into roi_list that survived (non-empty rows)
    """
    traces_out  = []
    fly_labels  = []
    tseries_labels = []
    X_rows      = []
    valid_idx   = []

    seg_names = list(segments.keys())

    if feature_set == 'full':
        # align all traces to minimum length
        all_traces = [r[0] for r in roi_list]
        min_len    = min(len(t) for t in all_traces)
        for i, (trace, fly_id, ts_key) in enumerate(roi_list):
            row = trace[:min_len].astype(float)
            if not np.all(np.isfinite(row)):
                continue
            X_rows.append(row)
            traces_out.append(trace)
            fly_labels.append(fly_id)
            tseries_labels.append(ts_key)
            valid_idx.append(i)

    elif feature_set.startswith('segment_'):
        seg_name = feature_set.split('_', 1)[1]
        t_start, t_end = segments[seg_name]
        i0 = int(t_start * mean_fps)
        i1 = int(t_end   * mean_fps)
        segs = []
        for trace, fly_id, ts_key in roi_list:
            s = trace[i0:min(i1, len(trace))]
            segs.append(s)
        min_len = min(len(s) for s in segs)
        for i, (s, (trace, fly_id, ts_key)) in enumerate(zip(segs, roi_list)):
            row = s[:min_len].astype(float)
            if not np.all(np.isfinite(row)):
                continue
            X_rows.append(row)
            traces_out.append(trace)
            fly_labels.append(fly_id)
            tseries_labels.append(ts_key)
            valid_idx.append(i)

    elif feature_set == 'features':
        for i, (trace, fly_id, ts_key) in enumerate(roi_list):
            row, _ = extract_roi_features(
                trace, None, segments, mean_fps,
                polarity_results, luminance_results, contrast_results, frequency_results)
            if not np.all(np.isfinite(row)):
                continue
            X_rows.append(row)
            traces_out.append(trace)
            fly_labels.append(fly_id)
            tseries_labels.append(ts_key)
            valid_idx.append(i)

    elif feature_set.startswith('features_'):
        seg_name = feature_set.split('_', 1)[1]
        t_start, t_end = segments[seg_name]
        i0 = int(t_start * mean_fps)
        i1 = int(t_end   * mean_fps)
        for i, (trace, fly_id, ts_key) in enumerate(roi_list):
            seg = trace[i0:min(i1, len(trace))]
            if len(seg) == 0:
                continue
            bl_frames = max(1, int(0.5 * mean_fps))
            baseline  = np.mean(seg[:bl_frames])
            delta     = seg - baseline
            peak      = float(np.max(delta))
            trough    = float(np.min(delta))
            auc       = float(np.trapz(np.abs(delta)))
            peak_lat  = float(np.argmax(delta)) / mean_fps
            if peak > 0:
                thresh10 = 0.10 * peak
                thresh90 = 0.90 * peak
                above10  = np.where(delta >= thresh10)[0]
                above90  = np.where(delta >= thresh90)[0]
                rise = (above90[0] - above10[0]) / mean_fps if (len(above10) > 0 and len(above90) > 0) else 0.0
            else:
                rise = 0.0
            row = np.array([peak, trough, auc, peak_lat, rise], dtype=float)
            if not np.all(np.isfinite(row)):
                continue
            X_rows.append(row)
            traces_out.append(trace)
            fly_labels.append(fly_id)
            tseries_labels.append(ts_key)
            valid_idx.append(i)

    elif feature_set == 'spca':
        # ── Sparse PCA following Baden et al. 2016 ──────────────────────────
        # The paper uses SpaSM (Sjöstrand et al.) which produces SPARSE LOADINGS:
        # each component activates at only n_nonzero time bins, making it
        # interpretable as a localised temporal feature (e.g. "sustained OFF
        # response", "high-frequency modulation").
        #
        # sklearn's SparsePCA sparsifies the CODES (per-sample scores), not
        # the loadings — the opposite of what the paper does.  We therefore
        # implement sparse loadings directly:
        #   1. Fit regular PCA (gives globally smooth components).
        #   2. For each component, zero out all but the n_nonzero largest
        #      absolute loadings (hard thresholding).
        #   3. Re-normalise each sparse component to unit length.
        #   4. Project all samples onto the sparse components → feature matrix.
        #   5. Standardise each feature across the population (paper: "we
        #      standardised each feature separately across the population").
        #
        # Parameters from the paper (chirp only):
        #   n_components = 20   (paper: 20 features from chirp)
        #   n_nonzero    = 10   (paper: "10 non-zero time bins")
        #
        # Deviations from paper that cannot be avoided:
        #   • Paper uses 40-dim vector (chirp + colour + moving-bar + RF).
        #     We have chirp only → 20-dim vector.
        #   • Hard thresholding ≠ SpaSM's optimisation, but produces
        #     components with the same structural property (localised in time).
        N_COMP    = 20   # Baden et al.: 20 features from chirp
        N_NONZERO = 10   # Baden et al.: 10 non-zero time bins per component

        all_tr  = [r[0] for r in roi_list]
        min_len = min(len(t) for t in all_tr)
        normed  = []
        for trace, fly_id, ts_key in roi_list:
            n = _norm_trace(trace[:min_len], mean_fps)
            normed.append(n)
            traces_out.append(trace)
            fly_labels.append(fly_id)
            tseries_labels.append(ts_key)
        X_norm = np.vstack(normed)   # shape (n_rois, n_frames)

        # Clamp n_components to what the data can support
        n_comp    = min(N_COMP, X_norm.shape[0] - 1, X_norm.shape[1])
        n_nonzero = min(N_NONZERO, X_norm.shape[1])

        print(f"      Fitting sparse-loading PCA "
              f"(n={X_norm.shape[0]} ROIs × {X_norm.shape[1]} frames, "
              f"{n_comp} components, {n_nonzero} non-zero time bins) ...")

        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_norm)

        # Sparsify: keep only top n_nonzero absolute loadings per component
        sparse_components = pca.components_.copy()   # (n_comp, n_frames)
        for ci in range(n_comp):
            comp   = sparse_components[ci]
            thresh = np.sort(np.abs(comp))[-n_nonzero]   # n_nonzero-th largest
            comp[np.abs(comp) < thresh] = 0.0
            norm = np.linalg.norm(comp)
            if norm > 0:
                sparse_components[ci] = comp / norm

        # Project onto sparse components
        X_features = X_norm @ sparse_components.T   # (n_rois, n_comp)

        # Report average achieved sparsity
        avg_nonzero = np.mean(np.sum(sparse_components != 0, axis=1))
        print(f"      Avg non-zero loadings per component: {avg_nonzero:.1f} "
              f"(target {n_nonzero})")

        for i in range(len(roi_list)):
            row = X_features[i]
            if not np.all(np.isfinite(row)):
                continue
            X_rows.append(row)
            valid_idx.append(i)

        # Re-filter metadata lists if any rows were dropped
        if len(X_rows) != len(traces_out):
            traces_out     = [traces_out[k]     for k in valid_idx]
            fly_labels     = [fly_labels[k]     for k in valid_idx]
            tseries_labels = [tseries_labels[k] for k in valid_idx]

    if len(X_rows) == 0:
        return None, None, None, None, None

    X_raw = np.vstack(X_rows)
    # Standardize features across population (per-feature z-score)
    col_std = X_raw.std(axis=0)
    col_std[col_std == 0] = 1.0
    X = (X_raw - X_raw.mean(axis=0)) / col_std
    return X, fly_labels, tseries_labels, traces_out, valid_idx, X_raw


def _umap_embed(X, n_neighbors=15, min_dist=0.1):
    """UMAP 2-D embedding — visualisation only, no clustering."""
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42)
    return reducer.fit_transform(X)


def _pca_embed(X):
    """PCA 2-D embedding — visualisation only, for small-N per-fly scope."""
    n_comp    = min(2, X.shape[0] - 1, X.shape[1])
    embedding = PCA(n_components=n_comp).fit_transform(X)
    return embedding


def plot_embedding_scatter(embedding, labels, title, save_path,
                           hue_labels=None, hue_title='Fly'):
    """
    2-D scatter of UMAP / PCA embedding.
    If hue_labels supplied, color by those instead of cluster labels.
    """
    fig, axes = plt.subplots(1, 2 if hue_labels is not None else 1,
                             figsize=(14 if hue_labels is not None else 7, 6))
    if hue_labels is None:
        axes = [axes]

    unique_clusters = sorted(set(labels))
    cmap = plt.cm.tab10
    colors_cluster = {c: cmap(i % 10) for i, c in enumerate(unique_clusters)}

    ax = axes[0]
    for c in unique_clusters:
        mask = np.array(labels) == c
        lbl  = f'Cluster {c}' if c != -1 else 'Noise'
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   color=colors_cluster[c], s=20, alpha=0.7, label=lbl)
    ax.set_title(f'{title}\n(colored by cluster)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.legend(markerscale=2, fontsize=8); ax.grid(True, alpha=0.2)

    if hue_labels is not None:
        ax2 = axes[1]
        unique_hues = sorted(set(hue_labels))
        colors_hue  = {h: cmap(i % 10) for i, h in enumerate(unique_hues)}
        for h in unique_hues:
            mask = np.array(hue_labels) == h
            ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                        color=colors_hue[h], s=20, alpha=0.7, label=str(h))
        ax2.set_title(f'{title}\n(colored by {hue_title})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2')
        ax2.legend(markerscale=2, fontsize=8, title=hue_title); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_colored_traces(traces, gmm_labels, fly_labels, mean_fps, output_dir, pfx,
                                 stim_protocol=None, stim_time=None,
                                 spacing_percentile=50, height_per_row=0.20, linewidth=0.7):
    """
    Same stacked layout as PopulationSummaryPlots.plot_response_traces_stacked
    (fixed shared spacing, no axes), but each ROI's raw trace is colored by
    its GMM cluster for this feature_set, and rows are grouped by cluster
    (then by fly within cluster) so each cluster's block is contiguous.
    One of these per feature_set, saved alongside the other clustering_*
    outputs for that fset.
    """
    save_path = output_dir
    if os.path.exists(save_path):
        print(f"      Colored cluster traces already exist for {pfx}, skipping.")
        return save_path

    unique_clusters = sorted(set(gmm_labels))
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap('tab20', max(n_clusters, 1))
    cluster_colors = {c: cmap(i % 20) for i, c in enumerate(unique_clusters)}

    #comment out if want to plot all traces, random subsampling of traces for 50%
    # rng = np.random.default_rng(0)
    # gmm_arr = np.asarray(gmm_labels)
    # total_counts = {c: int(np.sum(gmm_arr == c)) for c in unique_clusters}
    # selected = []
    # for c in unique_clusters:
    #     members = np.where(gmm_arr == c)[0]
    #     n_plot = max(1, int(round(0.5 * len(members))))
    #     chosen = rng.choice(members, size=n_plot, replace=False)
    #     selected.extend(chosen.tolist())
    # order = sorted(selected, key=lambda i: (gmm_labels[i], str(fly_labels[i])))
    #until here
    #plots all
    order = sorted(range(len(traces)), key=lambda i: (gmm_labels[i], str(fly_labels[i])))
    
    min_len = min(len(traces[i]) for i in order)
    matrix = np.array([traces[i][:min_len] for i in order])
    ordered_clusters = [gmm_labels[i] for i in order]
    n_neurons = matrix.shape[0]
    time = np.arange(min_len) / mean_fps

    typical_excursion = matrix.std(axis=1) * 4
    spacing = np.percentile(typical_excursion, spacing_percentile)
    if spacing <= 0:
        spacing = 1.0

    fig_height = max(4, n_neurons * height_per_row)
    if stim_protocol is not None:
        fig = plt.figure(figsize=(12, fig_height + 1))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, fig_height], hspace=0.01)
        ax_stim = fig.add_subplot(gs[0])
        stim_len = min(len(stim_protocol), min_len)
        ax_stim.fill_between(stim_time[:stim_len], 0, stim_protocol[:stim_len],
                             color='lightblue', alpha=0.7, linewidth=0)
        ax_stim.plot(stim_time[:stim_len], stim_protocol[:stim_len], color='blue', linewidth=1)
        ax_stim.set_xlim(time[0], time[-1])
        ax_stim.axis('off')
        ax = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(12, fig_height))

    for i in range(n_neurons):
        offset = (n_neurons - 1 - i) * spacing
        ax.plot(time, matrix[i] + offset, color=cluster_colors[ordered_clusters[i]],
                linewidth=linewidth, alpha=0.85)

    # thin separator where one cluster's block ends and the next begins
    prev_cluster = ordered_clusters[0]
    for i in range(1, n_neurons):
        if ordered_clusters[i] != prev_cluster:
            sep_offset = (n_neurons - i - 0.5) * spacing
            # ax.axhline(sep_offset, color='black', linewidth=0.6, alpha=0.3)
            prev_cluster = ordered_clusters[i]

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-spacing, n_neurons * spacing)
    ax.axis('off')

    legend_elements = [Patch(facecolor=cluster_colors[c], edgecolor='none',
                             label=f'Cluster {c} (n={ordered_clusters.count(c)})')
                       for c in unique_clusters]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
             bbox_to_anchor=(1.12, 1.0), frameon=False)

    fig.suptitle(f'{pfx} | traces colored by GMM cluster (n={n_neurons} neurons, {n_clusters} clusters)',
                 fontsize=13, fontweight='bold', y=0.995 if stim_protocol is not None else 1.02)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_cluster_traces(traces, labels, mean_fps, stim_protocol, stim_time,
                        title, save_path,
                        seg_window=None, feature_subtitle=None, ylim=None):
    """
    For each cluster: plot faint individual traces + mean ± SEM, stimulus on top.

    Parameters
    ----------
    seg_window : tuple (start_s, end_s) or None
        If given, slice traces and stimulus to this time window (in seconds).
        None = plot the full trace.
    feature_subtitle : str or None
        If given, shown as a second smaller title line (used for 'features' set).
    """
    unique_clusters = sorted(set(labels))
    plot_clusters = [c for c in unique_clusters if c != -1]
    if len(plot_clusters) == 0:
        return

    # --- apply segment window to traces and stim ---
    if seg_window is not None:
        start_s, end_s = seg_window
        i0_trace = int(start_s * mean_fps)
        i1_trace = int(end_s   * mean_fps)
        traces = [t[i0_trace:min(i1_trace, len(t))] for t in traces]

        stim_mask  = (stim_time >= start_s) & (stim_time <= end_s)
        stim_protocol = stim_protocol[stim_mask]
        stim_time     = stim_time[stim_mask] - start_s   # reset to 0

    # align traces to min length
    min_len = min(len(t) for t in traces)
    if min_len == 0:
        return
    time = np.arange(min_len) / mean_fps

    n_clusters = len(plot_clusters)
    fig = plt.figure(figsize=(6 * n_clusters, 8))
    outer_gs = fig.add_gridspec(1, n_clusters, hspace=0.3, wspace=0.35)

    cmap   = plt.cm.tab10
    colors = {c: cmap(i % 10) for i, c in enumerate(plot_clusters)}

    max_stim_time = stim_time[-1] if len(stim_time) > 0 else time[-1]

    for col_idx, cluster_id in enumerate(plot_clusters):
        mask     = np.array(labels) == cluster_id
        c_traces = [traces[i][:min_len] for i in range(len(traces)) if mask[i]]
        n_rois   = len(c_traces)

        inner_gs = outer_gs[col_idx].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.05)

        # Stimulus
        ax_stim = fig.add_subplot(inner_gs[0])
        if len(stim_protocol) > 0:
            ax_stim.fill_between(stim_time, 0, stim_protocol,
                                 color='lightblue', alpha=0.7, linewidth=0)
            ax_stim.plot(stim_time, stim_protocol, color='blue', linewidth=1.2)
        ax_stim.set_xlim(0, max(time[-1], max_stim_time))
        ax_stim.set_ylim(-0.1, 1.1); ax_stim.axis('off')
        lbl_str = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
        ax_stim.set_title(f'{lbl_str}  (n={n_rois})', fontsize=12, fontweight='bold')

        # Traces
        ax_trace = fig.add_subplot(inner_gs[1])
        for t in c_traces:
            ax_trace.plot(time, t, color='gray', alpha=0.25, linewidth=0.5)

        if n_rois > 0:
            mat  = np.vstack(c_traces)
            mean = np.mean(mat, axis=0)
            sem  = np.std(mat,  axis=0) / np.sqrt(n_rois)
            ax_trace.plot(time, mean, color=colors[cluster_id], linewidth=2)
            ax_trace.fill_between(time, mean - sem, mean + sem,
                                  color=colors[cluster_id], alpha=0.3)

        ax_trace.set_xlabel('Time (s)', fontsize=10)
        ax_trace.set_ylabel('ΔF/F', fontsize=10)
        ax_trace.set_xlim(0, max(time[-1], max_stim_time))
        ax_trace.grid(True, alpha=0.3)
        ax_trace.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        if ylim is not None:
            ax_trace.set_ylim(ylim[0], ylim[1])

    # Main title + optional feature subtitle
    full_title = title
    if feature_subtitle is not None:
        full_title = f'{title}\n{feature_subtitle}'
    fig.suptitle(full_title, fontsize=12, fontweight='bold', y=1.01)
    for ext in ['pdf', 'svg', 'png']:
        plt.savefig(f'{save_path}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
#  NEW FUNCTIONS: QI, normalisation, GMM, validation, plots
# ============================================================

def _norm_trace(trace, mean_fps, baseline_s=0.5):
    """Baseline-subtract + max-normalise (|max|=1). For clustering only."""
    bl = max(1, int(baseline_s * mean_fps))
    baseline = np.median(trace[:bl])
    delta = trace.astype(float) - baseline
    m = np.max(np.abs(delta))
    return delta / m if m > 0 else delta


def _zscore_trace(trace):
    """Z-score a trace: subtract mean, divide by std. std=0 → return zeros."""
    t = trace.astype(float)
    s = t.std()
    return (t - t.mean()) / s if s > 0 else np.zeros_like(t)


def compute_response_qi(trace_list):
    """
    Signal-to-noise quality index across trial repetitions (Baden et al. 2016).

        QI = Var_t[ mean_r(C) ] / mean_r[ Var_t(C) ]

    C is a (T x R) matrix — time points x trial replicates.

    Numerator  : variance of the TRIAL-AVERAGED trace over time  → signal power
    Denominator: mean across replicates of variance over time
                 per single trial                                → signal + noise power

    Because averaging reduces noise, numerator ≤ denominator, so QI ∈ [0, 1].
      QI → 1 : every trial is identical to the mean  (pure signal)
      QI → 0 : the mean trace is no more informative than a flat line relative
                to within-trial noise  (pure noise)

    Common pitfall: using axis=1 (variance across replicates at each time point)
    gives a pure SNR that is unbounded. The correct axis is 0 (variance over
    time within each replicate).
    """
    if len(trace_list) < 2:
        return 0.0
    min_len = min(len(t) for t in trace_list)
    C   = np.vstack([t[:min_len] for t in trace_list]).T  # shape (T, R)
    num = float(np.var(np.mean(C, axis=1)))               # Var_t[ mean_r(C) ]
    den = float(np.mean(np.var(C, axis=0)))               # mean_r[ Var_t(C) ]  ← axis=0
    return num / den if den > 0 else 0.0


def run_gmm_bic(X, k_range=range(2, 13)):
    """
    Gaussian Mixture Model with BIC-based cluster number selection.
    Follows Baden et al. 2016:
      - Diagonal covariance
      - Covariance regularisation: reg_covar = 1e-5 (adds constant to diagonal)
      - 20 random EM restarts per k, keep solution with highest likelihood
      - BIC as model selection criterion
      - Log Bayes factors 2·ΔBIC(k→k+1) to validate splitting
    """
    bic_vals, models = [], {}
    ks = []
    for k in k_range:
        if k >= len(X):
            break
        gmm = GaussianMixture(n_components=k, covariance_type='diag',
                              reg_covar=1e-5,        # Baden et al. 2016
                              n_init=20, max_iter=300, random_state=42)
        gmm.fit(X)
        bic_vals.append(gmm.bic(X))
        models[k] = gmm
        ks.append(k)

    if not ks:
        return np.zeros(len(X), dtype=int), 1, np.ones((len(X), 1)), {}

    bic_arr = np.array(bic_vals)

    # Adjacent log Bayes factors: 2*(BIC(k_{i-1}) - BIC(k_i))
    # Positive = evidence FOR the larger k.  Stop when value drops below 6.
    log_bf = np.concatenate([[0], 2 * (bic_arr[:-1] - bic_arr[1:])])

    # Best k: scan forward, stop at the FIRST transition where log_bf drops
    # below 6.  This matches Baden et al. 2016: keep splitting only while
    # there is consecutive strong evidence; a spurious positive at a later k
    # does not re-open splitting.  Fall back to argmin(BIC) when no transition
    # is strong (very clean data with only 2 clusters, or too few data points).
    best_idx = 0  # default: smallest k in range
    for i in range(1, len(ks)):
        if log_bf[i] > 6:
            best_idx = i   # accept this split
        else:
            break          # first non-strong transition → stop here

    best_k = ks[best_idx]
    best_gmm   = models[best_k]
    labels     = best_gmm.predict(X)
    posteriors = best_gmm.predict_proba(X)

    # BIC ambiguity score: improvement at best_k as fraction of total BIC range.
    # Near 0 = shallow minimum = ambiguous.  Near 1 = sharp minimum = clear.
    bic_range = bic_arr.max() - bic_arr.min()
    if best_idx > 0 and bic_range > 0:
        improvement = bic_arr[best_idx - 1] - bic_arr[best_idx]
        ambiguity_score = float(improvement / bic_range)
    else:
        ambiguity_score = 0.0

    if ambiguity_score < 0.1:
        print(f"      ⚠ BIC ambiguity: shallow minimum "
              f"(score={ambiguity_score:.3f} < 0.1). "
              f"k={best_k} is a weak recommendation — inspect the BIC plot.")

    diagnostics = {
        'k_range': ks, 'bic': bic_vals, 'log_bf': log_bf.tolist(),
        'best_k': best_k, 'models': models,
        'ambiguity_score': ambiguity_score,
    }
    return labels, best_k, posteriors, diagnostics


def compute_dprime_matrix(X, labels):
    """
    Sensitivity index d′ between every cluster pair (Franke et al. 2017).
    d′ = mean_feature( |μ1-μ2| / sqrt((σ1²+σ2²)/2) )
    """
    ulbls = sorted(set(labels))
    n = len(ulbls)
    D = np.zeros((n, n))
    for i, l1 in enumerate(ulbls):
        for j, l2 in enumerate(ulbls):
            if i >= j:
                continue
            X1 = X[np.array(labels) == l1]
            X2 = X[np.array(labels) == l2]
            mu1, mu2   = X1.mean(0), X2.mean(0)
            s2         = (X1.var(0) + X2.var(0)) / 2 + 1e-10
            d          = float(np.mean(np.abs(mu1 - mu2) / np.sqrt(s2)))
            D[i, j] = D[j, i] = d
    return D, ulbls


def validate_cluster_stability(X, best_k, n_bootstrap=20, frac=0.9):
    """
    Subsamples 90% of ROIs 20×, re-clusters, computes:
      1. Median cluster-mean correlation per bootstrap run (Baden et al. 2016).
      2. N×N co-assignment matrix: entry (i,j) = fraction of bootstrap runs
         in which ROIs i and j were both sampled AND assigned to the same cluster.
         Values near 1 = always co-assigned (stable); near 0 = never (well-separated).

    Returns
    -------
    corrs         : list of floats  (length n_bootstrap)
    co_assign_mat : (N, N) float array
    """
    N = len(X)
    orig_gmm = GaussianMixture(n_components=best_k, covariance_type='diag',
                               reg_covar=1e-5, n_init=20, random_state=42)
    orig_gmm.fit(X)
    orig_means = orig_gmm.means_

    rng        = np.random.default_rng(0)
    corrs      = []
    co_count   = np.zeros((N, N), dtype=float)   # times both sampled + same cluster
    both_count = np.zeros((N, N), dtype=float)   # times both sampled

    n_sub = max(best_k + 1, int(N * frac))

    for i in range(n_bootstrap):
        idx   = rng.choice(N, n_sub, replace=False)
        gmm_s = GaussianMixture(n_components=best_k, covariance_type='diag',
                                reg_covar=1e-5, n_init=10, random_state=i)
        gmm_s.fit(X[idx])

        # Match sub-clusters to original clusters by mean correlation
        sub_means  = gmm_s.means_
        sub_labels = gmm_s.predict(X[idx])

        # Build mapping: sub_cluster_id -> orig_cluster_id
        mapping = {}
        used_orig = set()
        for oc_idx, om in enumerate(orig_means):
            best_r, best_sc = -np.inf, 0
            for sc_idx, sm in enumerate(sub_means):
                r = np.corrcoef(om, sm)[0, 1]
                if np.isfinite(r) and r > best_r and sc_idx not in used_orig:
                    best_r, best_sc = r, sc_idx
            mapping[best_sc] = oc_idx
            used_orig.add(best_sc)

        # Remap sub labels to original cluster space
        remapped = np.array([mapping.get(l, l) for l in sub_labels])

        # Update co-assignment counts
        for a_pos, a in enumerate(idx):
            for b_pos, b in enumerate(idx):
                if a >= b:
                    continue
                both_count[a, b] += 1
                both_count[b, a] += 1
                if remapped[a_pos] == remapped[b_pos]:
                    co_count[a, b] += 1
                    co_count[b, a] += 1

        # Correlation metric (Baden et al.)
        per_cluster = []
        for om in orig_means:
            best_r = max((np.corrcoef(om, sm)[0, 1]
                          for sm in sub_means
                          if np.isfinite(np.corrcoef(om, sm)[0, 1])),
                         default=0.0)
            per_cluster.append(best_r)
        corrs.append(float(np.median(per_cluster)))

    # Compute fraction; diagonal = 1 by definition
    with np.errstate(divide='ignore', invalid='ignore'):
        co_assign_mat = np.where(both_count > 0,
                                 co_count / both_count, 0.0)
    np.fill_diagonal(co_assign_mat, 1.0)

    return corrs, co_assign_mat


def plot_gmm_diagnostics(gmm_diag, dprime, ulbls, bootstrap_corrs, title, save_path):
    """BIC curve (normalised 0-1), log-Bayes-factors per adjacent transition,
    d′ matrix, bootstrap histogram."""
    has_boot = len(bootstrap_corrs) > 0
    ncols = 4 if has_boot else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    ks     = gmm_diag.get('k_range', [])
    bic    = np.array(gmm_diag.get('bic', []))
    log_bf = np.array(gmm_diag.get('log_bf', []))
    best_k = gmm_diag.get('best_k', None)

    # ── BIC curve: normalise to [0,1] so absolute scale doesn't distract ──
    ax = axes[0]
    if len(bic) > 0:
        bic_norm = (bic - bic.min()) / (bic.max() - bic.min() + 1e-10)
        ax.plot(ks, bic_norm, 'o-', color='steelblue', linewidth=2)
        if best_k is not None:
            ax.axvline(best_k, color='red', linestyle='--',
                       label=f'Best k = {best_k}')
        ax.set_ylim(-0.05, 1.05)
        # Ambiguity annotation
        amb = gmm_diag.get('ambiguity_score', None)
        if amb is not None:
            colour = '#2ecc71' if amb >= 0.1 else '#e74c3c'
            label  = ('Sharp minimum — clear k'
                      if amb >= 0.1 else
                      '⚠ Shallow minimum — ambiguous k')
            ax.text(0.97, 0.97, f'Ambiguity score: {amb:.3f}\n{label}',
                    ha='right', va='top', transform=ax.transAxes,
                    fontsize=8, color=colour,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=colour, alpha=0.85))
    ax.set_xlabel('Number of clusters (k)', fontsize=10)
    ax.set_ylabel('Normalised BIC (0 = best)', fontsize=10)
    ax.set_title('BIC — lower is better\n(normalised; absolute values are arbitrary)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Log Bayes factors: one bar per k→k+1 transition ───────────────────
    # log_bf[0] = 0 (no predecessor), log_bf[i] = 2*(BIC(k_{i-1}) - BIC(k_i))
    # Positive = evidence FOR the larger k.  Stop where value drops below 6.
    ax = axes[1]
    if len(log_bf) > 1 and len(ks) > 1:
        # transitions: k[0]→k[1], k[1]→k[2], ...
        trans_ks  = ks[1:]          # destination k values
        trans_lbf = log_bf[1:]      # skip the leading 0
        bar_colors = ['#2ecc71' if v >= 6 else '#e74c3c' for v in trans_lbf]
        ax.bar(range(len(trans_ks)), trans_lbf, color=bar_colors,
               edgecolor='black', alpha=0.85)
        ax.axhline(6, color='black', linestyle='--', linewidth=1.5,
                   label='Threshold = 6\n(strong evidence,\nKass & Raftery 1995)')
        ax.set_xticks(range(len(trans_ks)))
        ax.set_xticklabels([f'{ks[i]}→{ks[i+1]}' for i in range(len(ks)-1)],
                           rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Cluster transition (k → k+1)', fontsize=10)
        ax.set_ylabel('2·(BIC(k) − BIC(k+1))', fontsize=10)
        # Use symlog scale so large values don't squash the threshold line
        ax.set_yscale('symlog', linthresh=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        ax.set_title('Evidence for each additional cluster\nGreen ≥ 6 → keep splitting',
                     fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    # ── d′ matrix ──────────────────────────────────────────────────────────
    ax = axes[2]
    if len(ulbls) > 1 and dprime is not None:
        im = ax.imshow(dprime, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(ulbls)))
        ax.set_xticklabels([f'C{l}' for l in ulbls], fontsize=8)
        ax.set_yticks(range(len(ulbls)))
        ax.set_yticklabels([f'C{l}' for l in ulbls], fontsize=8)
        plt.colorbar(im, ax=ax, label="d′")
        ax.set_title("d′ separation matrix\n(d′ ≥ 2 → ~85% correct discrimination)",
                     fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("d′ separation matrix", fontweight='bold')

    # ── Bootstrap histogram ────────────────────────────────────────────────
    if has_boot:
        ax = axes[3]
        ax.hist(bootstrap_corrs, bins=10, color='coral', edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(bootstrap_corrs), color='red', linestyle='--',
                   label=f'Mean = {np.mean(bootstrap_corrs):.2f} ± {np.std(bootstrap_corrs):.2f}')
        ax.set_xlabel('Median cluster correlation', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Bootstrap stability\n(n=20 × 90% subsampling)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_dendrogram(cluster_means, title, save_path):
    """Hierarchical dendrogram of cluster means using correlation distance."""
    if len(cluster_means) < 2:
        return
    corr  = np.corrcoef(np.vstack(cluster_means))
    np.fill_diagonal(corr, 1.0)
    dist  = squareform(np.clip(1 - corr, 0, 2), checks=False)
    Z     = linkage(dist, method='average')
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(cluster_means)), 5))
    sp_dendrogram(Z, ax=ax, labels=[f'C{i}' for i in range(len(cluster_means))])
    ax.set_ylabel('Correlation distance', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _save_heatmap_data(save_path, traces, labels, mean_fps, title,
                        cluster_means, normalize):
    """Pickle everything needed to regenerate this heatmap later, standalone."""
    data_path = f'{save_path}_data.pkl'
    os.makedirs(os.path.dirname(data_path) or '.', exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump({
            'traces': traces,
            'labels': labels,
            'mean_fps': mean_fps,
            'title': title,
            'save_path': save_path,
            'cluster_means': cluster_means,
            'normalize': normalize,
        }, f)
    return data_path


def plot_cluster_heatmap(traces, labels, mean_fps, title, save_path,
                         cluster_means=None, normalize=False, save_data=True):
    """
    Heat map of ROI responses sorted by cluster, with optional dendrogram.

    Parameters
    ----------
    normalize : False | True | 'zscore'
        False     — raw ΔF/F (amplitude preserved; default for all feature sets)
        True      — peak-normalised to max|ΔF/F|=1 (used for 'spca' feature set)
        'zscore'  — z-scored per ROI (mean=0, std=1; shape-only, same as tSNE papers)
    cluster_means : list of arrays or None
        Pre-computed cluster mean traces in the same representation as the heatmap.
        If provided, a dendrogram is drawn to the right.
    save_data : bool
        If True (default), pickle every argument to f'{save_path}_data.pkl'
        before plotting, so the figure can be regenerated later without
        rerunning the pipeline. See plot_cluster_heatmap_from_file().
    """
    if save_data:
        _save_heatmap_data(save_path, traces, labels, mean_fps, title,
                            cluster_means, normalize)

    unique_clusters = sorted(set(labels))
    plot_clusters   = [c for c in unique_clusters if c != -1]
    if not plot_clusters:
        return

    min_len = min(len(t) for t in traces)
    time    = np.arange(min_len) / mean_fps

    # Sort ROIs by cluster
    sorted_idx, boundaries = [], [0]
    for c in plot_clusters:
        idx = [i for i, l in enumerate(labels) if l == c]
        sorted_idx.extend(idx)
        boundaries.append(len(sorted_idx))

    def _apply_norm(trace):
        if normalize == 'zscore':
            return _zscore_trace(trace)
        elif normalize:
            return _norm_trace(trace, mean_fps)
        else:
            return trace.astype(float)

    mat  = np.vstack([_apply_norm(traces[i][:min_len]) for i in sorted_idx])
    vmax = float(np.percentile(np.abs(mat), 95))

    if normalize == 'zscore':
        cbar_label = 'Z-score'
    elif normalize:
        cbar_label = 'ΔF/F (norm. to peak)'
    else:
        cbar_label = 'ΔF/F'

    # ── layout: heatmap + optional dendrogram ────────────────────────────
    has_dend  = cluster_means is not None and len(cluster_means) > 1
    width_ratios = [4, 1] if has_dend else [1]
    ncols = 2 if has_dend else 1

    fig_w = 16 if has_dend else 13
    fig_h = max(4, len(sorted_idx) * 0.12)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h),
                             gridspec_kw={'width_ratios': width_ratios,
                                          'wspace': 0.02})
    ax_hm = axes[0] if has_dend else axes

    # ── heatmap ───────────────────────────────────────────────────────────
    im = ax_hm.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, #jet
                      extent=[0, time[-1], len(sorted_idx), 0])
    cmap_c = plt.cm.tab10
    for i, (c, b) in enumerate(zip(plot_clusters, boundaries[1:])):
        ax_hm.axhline(b, color='white', linewidth=1.2)
        mid = (boundaries[i] + b) / 2
        ax_hm.text(-0.015 * time[-1], mid, f'C{c}',
                   ha='right', va='center', fontsize=9, fontweight='bold',
                   color=cmap_c(i % 10),
                   transform=ax_hm.get_yaxis_transform())
    cb = plt.colorbar(im, ax=ax_hm, label=cbar_label, shrink=0.6)
    ax_hm.set_xlabel('Time (s)', fontsize=11)
    ax_hm.set_ylabel('ROI (sorted by cluster)', fontsize=11)
    ax_hm.set_title(title, fontsize=13, fontweight='bold')

    # ── dendrogram on right ───────────────────────────────────────────────
    if has_dend:
        ax_dend = axes[1]
        corr = np.corrcoef(np.vstack(cluster_means))
        np.fill_diagonal(corr, 1.0)
        dist = squareform(np.clip(1 - corr, 0, 2), checks=False)
        Z    = linkage(dist, method='average')

        # Draw dendrogram oriented so leaves align with heatmap rows.
        # Each leaf corresponds to one cluster block; position leaves at
        # the block midpoints in the heatmap's y-axis (0 = top).
        n_clusters = len(cluster_means)
        mids = [(boundaries[i] + boundaries[i + 1]) / 2
                for i in range(n_clusters)]

        dend = sp_dendrogram(Z, ax=ax_dend, orientation='right',
                             no_labels=True, color_threshold=0,
                             above_threshold_color='black',
                             link_color_func=lambda k: 'black')

        # Rescale the dendrogram y positions to match heatmap row coordinates.
        # sp_dendrogram with orientation='right' uses leaves 5,15,25...
        # (multiples of 10) on the y-axis. Map them to actual row midpoints.
        leaf_order = dend['leaves']               # order leaves appear top→bottom
        n_leaves   = len(leaf_order)
        dend_ys    = [5 + 10 * i for i in range(n_leaves)]  # default dendrogram positions
        target_ys  = [mids[leaf_order[i]] for i in range(n_leaves)]
        scale_y    = (target_ys[-1] - target_ys[0]) / (dend_ys[-1] - dend_ys[0] + 1e-10)
        offset_y   = target_ys[0] - dend_ys[0] * scale_y

        for coll in ax_dend.collections:
            segs = coll.get_segments()
            new_segs = []
            for seg in segs:
                new_seg = np.array(seg, dtype=float)
                new_seg[:, 1] = new_seg[:, 1] * scale_y + offset_y
                new_segs.append(new_seg)
            coll.set_segments(new_segs)

        ax_dend.set_ylim(ax_hm.get_ylim())
        ax_dend.set_xlim(left=0)
        ax_dend.invert_xaxis()              # root on the right, leaves on the left
        ax_dend.set_xlabel('Corr.\ndist.', fontsize=8)
        ax_dend.set_yticks([])
        ax_dend.spines['top'].set_visible(False)
        ax_dend.spines['right'].set_visible(False)
        ax_dend.spines['left'].set_visible(False)
        ax_dend.tick_params(axis='x', labelsize=7)

    plt.tight_layout()
    for ext in ['pdf', 'svg', 'png']:
        plt.savefig(f'{save_path}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()



def plot_bootstrap_confusion(co_assign_mat, labels, title, save_path):
    """
    Plot the bootstrap co-assignment matrix sorted by GMM cluster labels.

    Each cell (i, j) shows the fraction of bootstrap runs in which ROI i and
    ROI j were assigned to the same cluster.  Bright diagonal blocks = stable
    clusters; dark off-diagonal = well-separated clusters.

    Also overlays the mean within-cluster and between-cluster co-assignment
    values as text annotations on each block.
    """
    if co_assign_mat is None or len(co_assign_mat) == 0:
        return

    unique_clusters = sorted(set(labels))
    plot_clusters   = [c for c in unique_clusters if c != -1]
    if not plot_clusters:
        return

    # Sort ROI indices by cluster
    sorted_idx, boundaries = [], [0]
    for c in plot_clusters:
        idx = [i for i, l in enumerate(labels) if l == c]
        sorted_idx.extend(idx)
        boundaries.append(len(sorted_idx))

    # Reorder matrix
    mat = co_assign_mat[np.ix_(sorted_idx, sorted_idx)]
    N   = len(sorted_idx)

    fig, ax = plt.subplots(figsize=(max(6, N * 0.12 + 2),
                                    max(5, N * 0.12 + 1.5)))

    im = ax.imshow(mat, cmap='hot', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Co-assignment frequency', shrink=0.7)

    # Draw cluster boundary lines and annotate block means
    cmap_c = plt.cm.tab10
    for i, (c, b) in enumerate(zip(plot_clusters, boundaries[1:])):
        # Boundary lines
        ax.axhline(b - 0.5, color='cyan', linewidth=1.5)
        ax.axvline(b - 0.5, color='cyan', linewidth=1.5)

        # Within-cluster mean co-assignment
        blk = mat[boundaries[i]:b, boundaries[i]:b]
        # Exclude diagonal (always 1) for a more informative mean
        mask_off = ~np.eye(blk.shape[0], dtype=bool)
        within_mean = float(np.mean(blk[mask_off])) if mask_off.any() else 1.0
        mid = (boundaries[i] + b) / 2

        # Label on y axis
        ax.text(-0.5, mid, f'C{c}\n(n={b - boundaries[i]})',
                ha='right', va='center', fontsize=8, fontweight='bold',
                color=cmap_c(i % 10))

        # Annotate within-block mean
        ax.text(mid, mid, f'{within_mean:.2f}',
                ha='center', va='center', fontsize=8,
                color='cyan', fontweight='bold')

        # Between-cluster means
        for j, (c2, b2) in enumerate(zip(plot_clusters, boundaries[1:])):
            if j <= i:
                continue
            blk_off = mat[boundaries[i]:b, boundaries[j]:b2]
            between_mean = float(np.mean(blk_off))
            mid2 = (boundaries[j] + b2) / 2
            ax.text(mid2, mid, f'{between_mean:.2f}',
                    ha='center', va='center', fontsize=7, color='lightblue')
            ax.text(mid, mid2, f'{between_mean:.2f}',
                    ha='center', va='center', fontsize=7, color='lightblue')

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xlabel('ROI index (sorted by cluster)', fontsize=10)
    ax.set_ylabel('ROI index (sorted by cluster)', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{title}\n'
                 r'Diagonal blocks = within-cluster stability  |  '
                 r'Off-diagonal = between-cluster separation',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def analyze_response_clustering(condition_rois, segments, mean_fps,
                                 stim_protocol, stim_time, output_dir,
                                 polarity_results=None, luminance_results=None,
                                 contrast_results=None, frequency_results=None,
                                 ylims=None):
    """
    Orchestrate all three clustering scopes × feature sets.
    Primary method: GMM + BIC (Baden et al. 2016).
    Visualisation: UMAP (condition scope) / PCA (per-fly scope).
    Also runs UMAP + k-means for comparison.
    """
    print("\n" + "="*60)
    print("Running response profile clustering...")

    seg_names    = list(segments.keys())
    feature_sets = (
        ['full', 'spca']
        + [f'segment_{s}' for s in seg_names]
        + ['features']
        + [f'features_{s}' for s in seg_names]
    )
    k_range_gmm  = range(2, 13)

    # ── Description of the 26-feature 'features' vector ──────────────────
    # Each ROI contributes one unique feature vector computed from its own
    # replicate-averaged trace.  Features are grouped by segment:
    #
    # Per segment (×4 segments = 20 features):
    #   [seg]_peak      — max ΔF/F above segment baseline
    #   [seg]_trough    — min ΔF/F (captures inhibitory responses)
    #   [seg]_auc       — area under |ΔF/F| curve (total response energy)
    #   [seg]_peak_lat  — time to peak (s) — response onset speed
    #   [seg]_rise      — 10→90% rise time (s) — response kinetics
    #
    # Global scalars (×6 features = 6 features):
    #   on_amplitude    — mean ΔF/F in ON window (6-8 s of polarity segment)
    #   off_amplitude   — mean ΔF/F in OFF window (9-11 s of polarity segment)
    #   lum_slope       — linear slope of response vs luminance level
    #   on_off_index    — (ON - OFF) / (|ON| + |OFF|)  ∈ [-1, +1]
    #                     +1 = pure ON cell, -1 = pure OFF cell
    #   freq_index      — (high-freq half - low-freq half) / sum  ∈ [-1, +1]
    #                     +1 = prefers fast flicker, -1 = prefers slow
    #   contrast_index  — (high-contrast half - low-contrast half) / sum ∈ [-1, +1]
    #                     +1 = response grows with contrast,
    #                     -1 = contrast-suppressed
    #
    # Total: 26 features per ROI.  All computed from the individual ROI
    # trace — no population means used.  All standardised (z-scored) across
    # the population before clustering.
    print("\n  Feature vector description (26 features per ROI):")
    print("    Per segment ×4 (polarity/frequency/contrast/luminance):")
    print("      peak, trough, AUC, peak_latency, rise_time  → 20 features")
    print("    Global scalars:")
    print("      on_amplitude, off_amplitude, lum_slope      → 3 features")
    print("      on_off_index                                → 1 feature")
    print("      freq_tuning_peak_hz  (Hz of peak response)  → 1 feature")
    print("      contrast_tuning_peak_pct  (% Weber contrast at peak) → 1 feature")

    _feat_subtitle     = ('Features per segment: peak, trough, AUC, peak-lat, rise  |  '
                          'Global: ON-amp, OFF-amp, lum-slope, ON-OFF-idx, freq-idx, cont-idx')
    _feat_seg_subtitle = 'Features: peak, trough, AUC, peak-lat, rise'

    def _seg_window(fset):
        if fset.startswith('segment_'):
            return segments[fset.split('_', 1)[1]]
        if fset.startswith('features_') and fset != 'features':
            return segments[fset.split('_', 1)[1]]
        return None

    def _subtitle(fset):
        if fset in ('features', 'spca'):
            return _feat_subtitle if fset == 'features' else '20 sparse PCA components (Baden et al. 2016)'
        if fset.startswith('features_') and fset != 'features':
            return f'{_feat_seg_subtitle}  |  segment: {fset.split("_",1)[1]}'
        return None

    def _ylim(fset):
        """Return the appropriate ylim tuple for this feature set, or None."""
        if ylims is None:
            return None
        if fset.startswith('segment_') or (fset.startswith('features_') and fset != 'features'):
            seg_name = fset.split('_', 1)[1]
            return ylims.get(seg_name)
        return ylims.get('full')

    # ------------------------------------------------------------------ #
    # SCOPE 1 & 3: per-condition  (UMAP visualisation + GMM clustering)  #
    # ------------------------------------------------------------------ #
    print("\n  Scope: per-condition")
    clustering_results = {}
    for fset in feature_sets:
        print(f"    Feature set: {fset}")
        cache_path = os.path.join(output_dir, f'_cache_{fset}.joblib')
        if os.path.exists(cache_path):
            print(f"      Found cached result, loading (skipping GMM/bootstrap/plots)")
            clustering_results[fset] = joblib.load(cache_path)
            continue
        X, fly_lbls, ts_lbls, traces, _, X_raw = prepare_clustering_data(
            condition_rois, fset, segments, mean_fps,
            polarity_results, luminance_results, contrast_results, frequency_results)

        if X is None or len(X) < 5:
            print(f"      Skipping: insufficient data ({0 if X is None else len(X)} ROIs)")
            continue

        pfx = f'condition_{fset}'

        # ---- GMM + BIC (primary clustering) --------------------------------
        gmm_labels, best_k_gmm, posteriors, gmm_diag = run_gmm_bic(X, k_range_gmm)
        clustering_results[fset] = {
            'X': X, 'fly_labels': fly_lbls, 'traces': traces,
            'gmm_labels': gmm_labels, 'best_k': best_k_gmm,
            'posteriors': posteriors, 'gmm_diag': gmm_diag, 'X_raw': X_raw,
        }
        # d′ separation matrix
        dprime, ulbls = compute_dprime_matrix(X, gmm_labels)

        # Bootstrap stability (20 rounds) — returns corrs + co-assignment matrix
        print(f"      Bootstrap validation (k={best_k_gmm}) ...")
        boot_corrs, co_mat = validate_cluster_stability(X, best_k_gmm)
        print(f"        mean stability = {np.mean(boot_corrs):.3f} ± {np.std(boot_corrs):.3f}")

        plot_gmm_diagnostics(gmm_diag, dprime, ulbls, boot_corrs,
            title=f'Condition | {fset} | GMM diagnostics',
            save_path=os.path.join(output_dir, f'clustering_gmm_diag_{pfx}.png'))

        # UMAP for visualisation of GMM clusters
        emb_gmm = _umap_embed(X)
        plot_embedding_scatter(emb_gmm, gmm_labels,
            title=f'Condition | {fset} | GMM (k={best_k_gmm})',
            save_path=os.path.join(output_dir, f'clustering_umap_gmm_{pfx}.png'),
            hue_labels=fly_lbls, hue_title='Fly')
        plot_cluster_traces(traces, gmm_labels, mean_fps, stim_protocol, stim_time,
            title=f'Condition | {fset} | GMM (k={best_k_gmm})',
            save_path=os.path.join(output_dir, f'clustering_traces_gmm_{pfx}'),
            seg_window=_seg_window(fset), feature_subtitle=_subtitle(fset),
            ylim=_ylim(fset))
        for ext in ['pdf', 'png']:
            plot_cluster_colored_traces(traces, gmm_labels, fly_lbls, mean_fps, os.path.join(output_dir, f'clustering_traces_colored_{pfx}.{ext}'), pfx,
                                        stim_protocol=stim_protocol, stim_time=stim_time)
        # Cluster means for dendrogram: use same representation as heatmap
        _do_norm = (fset == 'spca')
        unique_gmm = sorted(set(gmm_labels))
        min_len = min(len(t) for t in traces)
        cmeans  = [np.mean([
                       _norm_trace(traces[i][:min_len], mean_fps) if _do_norm
                       else traces[i][:min_len].astype(float)
                       for i, l in enumerate(gmm_labels) if l == c], axis=0)
                   for c in unique_gmm]

        # Z-scored cluster means for the z-score heatmap dendrogram
        cmeans_z = [np.mean([_zscore_trace(traces[i][:min_len])
                             for i, l in enumerate(gmm_labels) if l == c], axis=0)
                    for c in unique_gmm]

        # Heatmap 1: raw ΔF/F (amplitude preserved) / peak-norm for spca
        plot_cluster_heatmap(traces, gmm_labels, mean_fps,
            title=f'Condition | {fset} | GMM heatmap (k={best_k_gmm})',
            save_path=os.path.join(output_dir, f'clustering_heatmap_gmm_{pfx}'),
            cluster_means=cmeans if len(cmeans) > 1 else None,
            normalize=_do_norm)

        # Heatmap 2: z-scored traces — shape only, amplitude removed
        plot_cluster_heatmap(traces, gmm_labels, mean_fps,
            title=f'Condition | {fset} | GMM heatmap Z-scored (k={best_k_gmm})',
            save_path=os.path.join(output_dir, f'clustering_heatmap_gmm_{pfx}_zscore'),
            cluster_means=cmeans_z if len(cmeans_z) > 1 else None,
            normalize='zscore')

        # Bootstrap co-assignment confusion matrix
        plot_bootstrap_confusion(co_mat, gmm_labels,
            title=f'Condition | {fset} | Bootstrap co-assignment (k={best_k_gmm})',
            save_path=os.path.join(output_dir, f'clustering_confusion_gmm_{pfx}.png'))
        joblib.dump(clustering_results[fset], cache_path, compress=3)

    # ------------------------------------------------------------------ #
    # SCOPE 2: per-fly independent  (PCA visualisation + GMM clustering) #
    # (commented out — re-enable to run per-fly clustering)              #
    # ------------------------------------------------------------------ #
    # print("\n  Scope: per-fly")
    # for fly_id, roi_list in fly_rois.items():
    #     print(f"    Fly: {fly_id}  ({len(roi_list)} ROIs)")
    #     for fset in feature_sets:
    #         X, fly_lbls_f, _, traces_f, _ = prepare_clustering_data(
    #             roi_list, fset, segments, mean_fps,
    #             polarity_results, luminance_results, contrast_results, frequency_results)
    #
    #         if X is None or len(X) < 4:
    #             continue
    #
    #         pfx_f = f'fly{fly_id}_{fset}'
    #
    #         # GMM + BIC
    #         gmm_lbl_f, best_k_f, post_f, gmm_diag_f = run_gmm_bic(X, range(2, min(9, len(X))))
    #         dp_f, ul_f = compute_dprime_matrix(X, gmm_lbl_f)
    #         boot_corrs_f, co_mat_f = validate_cluster_stability(X, best_k_f)
    #         plot_gmm_diagnostics(gmm_diag_f, dp_f, ul_f, boot_corrs_f,
    #             title=f'Fly {fly_id} | {fset} | GMM diagnostics',
    #             save_path=os.path.join(output_dir, f'clustering_gmm_diag_{pfx_f}.png'))
    #
    #         # PCA for visualisation
    #         emb_f = _pca_embed(X)
    #         plot_embedding_scatter(emb_f, gmm_lbl_f,
    #             title=f'Fly {fly_id} | {fset} | GMM (k={best_k_f})',
    #             save_path=os.path.join(output_dir, f'clustering_pca_gmm_{pfx_f}.png'))
    #         plot_cluster_traces(traces_f, gmm_lbl_f, mean_fps, stim_protocol, stim_time,
    #             title=f'Fly {fly_id} | {fset} | GMM (k={best_k_f})',
    #             save_path=os.path.join(output_dir, f'clustering_traces_gmm_{pfx_f}.png'),
    #             seg_window=_seg_window(fset), feature_subtitle=_subtitle(fset),
    #             ylim=_ylim(fset))
    #         _do_norm_f = (fset == 'spca')
    #         unique_gmm_f = sorted(set(gmm_lbl_f))
    #         min_len_f = min(len(t) for t in traces_f)
    #         cmeans_f  = [np.mean([
    #                          _norm_trace(traces_f[i][:min_len_f], mean_fps) if _do_norm_f
    #                          else traces_f[i][:min_len_f].astype(float)
    #                          for i, l in enumerate(gmm_lbl_f) if l == c], axis=0)
    #                      for c in unique_gmm_f]
    #         cmeans_fz = [np.mean([_zscore_trace(traces_f[i][:min_len_f])
    #                               for i, l in enumerate(gmm_lbl_f) if l == c], axis=0)
    #                      for c in unique_gmm_f]
    #         plot_cluster_heatmap(traces_f, gmm_lbl_f, mean_fps,
    #             title=f'Fly {fly_id} | {fset} | GMM heatmap',
    #             save_path=os.path.join(output_dir, f'clustering_heatmap_gmm_{pfx_f}.png'),
    #             cluster_means=cmeans_f if len(cmeans_f) > 1 else None,
    #             normalize=_do_norm_f)
    #         plot_cluster_heatmap(traces_f, gmm_lbl_f, mean_fps,
    #             title=f'Fly {fly_id} | {fset} | GMM heatmap Z-scored',
    #             save_path=os.path.join(output_dir, f'clustering_heatmap_gmm_{pfx_f}_zscore.png'),
    #             cluster_means=cmeans_fz if len(cmeans_fz) > 1 else None,
    #             normalize='zscore')
    #         plot_bootstrap_confusion(co_mat_f, gmm_lbl_f,
    #             title=f'Fly {fly_id} | {fset} | Bootstrap co-assignment (k={best_k_f})',
    #             save_path=os.path.join(output_dir, f'clustering_confusion_gmm_{pfx_f}.png'))

    print("\n  Clustering analysis complete.")
    return clustering_results


def plot_roi_repetitions(roi_accumulator, qi_pass, mean_fps,
                         stim_protocol, stim_time, output_dir, ylim=None):
    """
    For every QC-passing neuron, plot all its TSeries repetitions as faint
    individual traces + mean +/- SEM, with the stimulus protocol on top.
    One file per neuron.

    Saved as: 0_roi_repetitions_<fly_id>_roi<roi_idx>.png
    """
    print("\n" + "="*60)
    print("Plotting per-neuron repetitions ...")
    n_plotted = 0

    for fly_id, roi_dict in roi_accumulator.items():
        passing = [ri for ri in roi_dict if qi_pass.get((fly_id, ri), False)]
        if not passing:
            print(f"  FlyID {fly_id}: no QC-passing neurons, skipping")
            continue

        for roi_idx in sorted(passing):
            save_path = os.path.join(output_dir, f'0_roi_repetitions_{fly_id}_roi{roi_idx}.png')
            if os.path.exists(save_path):
                continue
            trace_list = roi_dict[roi_idx]
            n_reps = len(trace_list)

            stim_length = len(stim_protocol)
            min_len = min(min(len(t) for t in trace_list), stim_length)
            traces = [t[:min_len] for t in trace_list]
            time = np.arange(min_len) / mean_fps
            qi_val = compute_response_qi(trace_list)

            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
            max_time_r = max(time[-1], stim_time[-1])

            ax_stim = fig.add_subplot(gs[0])
            ax_stim.fill_between(stim_time, 0, stim_protocol,
                                 color='lightblue', alpha=0.7, linewidth=0)
            ax_stim.plot(stim_time, stim_protocol, color='blue', linewidth=1.5)
            ax_stim.set_xlim(0, max_time_r)
            ax_stim.set_ylim(-0.1, 1.1)
            ax_stim.axis('off')
            ax_stim.set_title(
                f'FlyID: {fly_id}  |  ROI {roi_idx}  |  {n_reps} TSeries repetitions  |  QI = {qi_val:.3f}',
                fontsize=14, fontweight='bold', pad=10)

            ax_trace = fig.add_subplot(gs[1])
            for rep_idx, tr in enumerate(traces):
                ax_trace.plot(time, tr, alpha=0.4, color='lightgray', linewidth=0.8,
                             label=f'Rep {rep_idx + 1}' if rep_idx == 0 else '_nolegend_')

            mat = np.vstack(traces)
            mean_tr = np.mean(mat, axis=0)
            sem_tr = np.std(mat, axis=0) / np.sqrt(n_reps)
            ax_trace.plot(time, mean_tr, color='black', linewidth=2.5,
                         label=f'Mean (n={n_reps} reps)', zorder=5)
            ax_trace.fill_between(time, mean_tr - sem_tr, mean_tr + sem_tr,
                                  color='black', alpha=0.15, zorder=4)

            ax_trace.set_xlabel('Time (s)', fontsize=12)
            ax_trace.set_ylabel('ΔF/F', fontsize=12)
            ax_trace.legend(loc='upper right', fontsize=9, ncol=2)
            ax_trace.grid(True, alpha=0.3)
            ax_trace.set_xlim(0, max_time_r)
            if ylim is not None:
                ax_trace.set_ylim(ylim[0], ylim[1])
            ax_trace.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)

            plt.tight_layout()
            # plt.savefig(os.path.join(output_dir, f'0_roi_repetitions_{fly_id}_roi{roi_idx}.png'),
            #             dpi=300, bbox_inches='tight')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            n_plotted += 1

        print(f"  FlyID {fly_id}: plotted {len(passing)} neurons")

    print(f"  Per-neuron repetition plots complete ({n_plotted} total).")


def process_pkl_file(pkl_path, metadata, multi, degen, mean_fps, original_fps=None,
                      output_dir='output_plots', ylims=None,
                      qi_threshold=0.45, reliability_threshold=0.4, min_reps=3):
    """
    Process PKL file with fluorescence data. Neuron (fly_id, roi_idx) is the
    unit of analysis throughout: every ROI's repetitions (separate TSeries)
    are interpolated, QC-filtered (QI + reliability + minimum rep count), and
    averaged into ONE representative trace per neuron. fly_id travels as
    metadata for later variance-decomposition — it is never used to average
    across neurons.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_traces_dir = os.path.join(output_dir, 'raw_traces')
    clustering_dir = os.path.join(output_dir, 'clustering')
    dispersion_dir = os.path.join(output_dir, 'dispersion')
    os.makedirs(raw_traces_dir, exist_ok=True)
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(dispersion_dir, exist_ok=True)
    stim_protocol, stim_time, segments = create_stimulus_protocol(mean_fps)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Found {len(data)} TSeries recordings")

    if multi:
        condition_str = pkl_path.split('\\')[-1].split('.')[0]
        visual_stim = condition_str.split('_')[1]
        if len(condition_str.split('_')) > 5:
            led = '_'.join(condition_str.split('_')[2:4])
            region = condition_str.split('_')[4]
            Retinal = condition_str.split('_')[5]
        else:
            led = condition_str.split('_')[2]
            region = condition_str.split('_')[3]
            Retinal = condition_str.split('_')[4]
        metadata_cond = metadata[metadata['Retinal'] == Retinal]
        metadata_cond = metadata_cond[metadata_cond['region'] == region]
        metadata_cond = metadata_cond[metadata_cond['visual_stim'] == visual_stim]
        metadata_cond = metadata_cond[metadata_cond['LED'] == led]
    elif degen:
        condition_str = pkl_path.split('\\')[-1].split('.')[0]
        visual_stim = condition_str.split('_')[2]
        neuron_type = condition_str.split('_')[1]
        metadata_cond = metadata[metadata['visual_stim'] == visual_stim]
        metadata_cond = metadata_cond[metadata_cond['genotype'] == neuron_type]
    elif temp:
        condition_str = pkl_path.split('\\')[-1].split('.')[0]
        visual_stim = condition_str.split('_')[3]
        neuron_type = '_'.join(condition_str.split('_')[1:3])
        metadata_cond = metadata[metadata['visual_stim'] == visual_stim]
        metadata_cond = metadata_cond[metadata_cond['genotype'] == neuron_type]

    segment_names = ['polarity', 'frequency', 'contrast', 'luminance']

    # Per-neuron repetition accumulator: roi_accumulator[fly_id][roi_idx] -> [trace, ...]
    roi_accumulator = defaultdict(lambda: defaultdict(list))
    tseries_raw = {}

    for tseries_idx, (tseries_key, tseries_data) in enumerate(data.items()):
        print(f"\nProcessing TSeries: {tseries_key}")
        tseries_number = tseries_key.split('-')[-1]
        tseries_number = tseries_number[2] if tseries_number[1] == '0' else tseries_number[1:]

        final_rois = tseries_data.get('final_rois', [])
        if not final_rois:
            print(f"  No ROIs found, skipping...")
            continue

        fly_id = final_rois[0].experiment_info.get('FlyID', 'Unknown')
        print(f"  FlyID: {fly_id}, Number of ROIs: {len(final_rois)}")

        meta_tseries = metadata_cond[metadata_cond['fly'] == fly_id]
        meta_tseries = meta_tseries[meta_tseries['TSeries'] == int(tseries_number)]
        tseries_fps = meta_tseries['fps'].values[0]

        interpolated_traces = []
        common_time = None
        for roi_idx, roi in enumerate(final_rois):
            df_trace = roi.df_trace
            roi_fps = tseries_fps if original_fps is None else original_fps
            interp_trace, time = interpolate_trace(df_trace, roi_fps, mean_fps, multi, degen, trim_seconds=5)
            interpolated_traces.append(interp_trace)
            common_time = time

        stim_length = len(stim_protocol)
        min_length = min(min(len(t) for t in interpolated_traces), stim_length)
        interpolated_traces = [t[:min_length] for t in interpolated_traces]
        common_time = common_time[:min_length]
        interpolated_traces = np.array(interpolated_traces)

        for roi_idx, roi_trace in enumerate(interpolated_traces):
            roi_accumulator[fly_id][roi_idx].append(roi_trace)

        tseries_raw[tseries_key] = {
            'traces': interpolated_traces, 'time': common_time,
            'fly_id': fly_id, 'idx': tseries_idx,
        }

    # ---- QC: QI, reliability, *and* minimum repetition count ---------------
    print("\n" + "="*60)
    print("Computing response quality index (QI) and reliability per neuron ...")
    qi_map, qc_pass, qi_all = {}, {}, []
    reliability_map = {}
    for fly_id_k, roi_dict in roi_accumulator.items():
        for roi_idx_k, trace_list in roi_dict.items():
            qi = compute_response_qi(trace_list)
            reliability = compute_reliability(trace_list)
            qi_map[(fly_id_k, roi_idx_k)] = qi
            reliability_map[(fly_id_k, roi_idx_k)] = reliability
            qc_pass[(fly_id_k, roi_idx_k)] = (
                qi >= qi_threshold
                and np.isfinite(reliability) and reliability >= reliability_threshold
                and len(trace_list) >= min_reps
            )
            qi_all.append(qi)
    n_pass = sum(qc_pass.values())
    print(f"  {n_pass}/{len(qi_all)} neurons pass QI>={qi_threshold}, "
          f"reliability>={reliability_threshold}, and reps>={min_reps}")

    if qi_all:
        qi_hist_path = os.path.join(clustering_dir, 'clustering_qi_distribution.png')
        if not os.path.exists(qi_hist_path):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(qi_all, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(qi_threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold = {qi_threshold}')
            ax.set_xlabel('Quality Index (QI)', fontsize=12)
            ax.set_ylabel('Number of neurons', fontsize=12)
            ax.set_title(f'Response Quality Index Distribution\n'
                         f'{n_pass}/{len(qi_all)} neurons pass (QI + reliability + min {min_reps} reps)', fontsize=13)
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(qi_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
    reliability_all = [v for v in reliability_map.values() if np.isfinite(v)]
    if reliability_all:
        rel_hist_path = os.path.join(clustering_dir, 'clustering_reliability_distribution.png')
        if not os.path.exists(rel_hist_path):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(reliability_all, bins=40, color='seagreen', edgecolor='black', alpha=0.7)
            ax.axvline(reliability_threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold = {reliability_threshold}')
            ax.set_xlabel('Response reliability (mean leave-one-out correlation across reps)', fontsize=11)
            ax.set_ylabel('Number of neurons', fontsize=12)
            ax.set_title(f'Response Reliability Distribution\n(n={len(reliability_all)} neurons with >=2 reps)', fontsize=13)
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(rel_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
    # ---- Plot 1: per-TSeries QC (unchanged in spirit, renamed from "ROIs") -
    print("Plotting per-TSeries QC ...")
    for tseries_key, raw in tseries_raw.items():
        fly_id, traces, time, ts_idx = raw['fly_id'], raw['traces'], raw['time'], raw['idx']
        save_path = os.path.join(raw_traces_dir, f'1_tseries_{ts_idx:03d}_{tseries_key}.png')
        if os.path.exists(save_path):
            print(f"  {tseries_key}: already plotted, skipping")
            continue
        keep_idx = [ri for ri in range(len(traces)) if qc_pass.get((fly_id, ri), False)]
        if len(keep_idx) == 0:
            print(f"  {tseries_key}: all ROIs failed QC, skipping")
            continue
        filtered_traces = traces[keep_idx]
        print(f"  {tseries_key}: {len(filtered_traces)}/{len(traces)} ROIs kept")
        fig, _ = plot_traces_with_mean(
            filtered_traces, time, stim_protocol, stim_time,
            f'TSeries: {tseries_key} | FlyID: {fly_id} | QC-passed {len(filtered_traces)}/{len(traces)} ROIs',
            ylabel='ΔF/F', ylim=ylims.get('full') if ylims else None
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ---- Neuron table: one row per QC-passing (fly_id, roi_idx) -----------
    print("\n" + "="*60)
    print("Building neuron-level dataset (QC-filtered, replicate-averaged) ...")
    condition_rois = []
    fly_rois = defaultdict(list)
    for fly_id_k, roi_dict in roi_accumulator.items():
        n_kept = 0
        for roi_idx_k, trace_list in roi_dict.items():
            if not qc_pass.get((fly_id_k, roi_idx_k), False):
                continue
            min_len = min(len(t) for t in trace_list)
            avg_trace = np.mean([t[:min_len] for t in trace_list], axis=0)
            entry = (avg_trace, fly_id_k, f'roi{roi_idx_k}')
            condition_rois.append(entry)
            fly_rois[fly_id_k].append(entry)
            n_kept += 1
        print(f"  FlyID {fly_id_k}: {n_kept} neurons kept")
    print(f"  Total neurons: {len(condition_rois)}")

    # ---- Per-neuron QC plot: every QC-passing neuron, not a per-fly sample -
    plot_roi_repetitions(roi_accumulator, qc_pass, mean_fps,
                          stim_protocol, stim_time, raw_traces_dir,
                          ylim=ylims.get('full') if ylims else None)

    # ---- Segment each neuron's trace; population-level plot per segment ---
    roi_segments = {seg: [] for seg in segment_names}
    for avg_trace, fly_id_r, roi_key in condition_rois:
        seg_result = segment_trace(avg_trace, np.arange(len(avg_trace)) / mean_fps,
                                   segments, mean_fps)
        for seg_name, (seg_tr, seg_t) in seg_result.items():
            if len(seg_tr) > 0:
                roi_segments[seg_name].append((seg_tr, seg_t))

    print("\n" + "="*60)
    print("Plotting per-segment neuron populations...")
    for seg_name in segment_names:
        seg_start, seg_end = segments[seg_name]
        seg_stim_start_idx = int(seg_start * mean_fps)
        seg_stim_end_idx = min(int(seg_end * mean_fps), len(stim_protocol))
        seg_stim = stim_protocol[seg_stim_start_idx:seg_stim_end_idx]
        seg_stim_time = np.arange(len(seg_stim)) / mean_fps
        seg_ylim = ylims.get(seg_name) if ylims else None

        if roi_segments[seg_name]:
            roi_min_len = min(len(tr) for tr, _ in roi_segments[seg_name])
            roi_traces = [tr[:roi_min_len] for tr, _ in roi_segments[seg_name]]
            roi_time = roi_segments[seg_name][0][1][:roi_min_len]
            fig, _ = plot_traces_with_mean(
                roi_traces, roi_time, seg_stim, seg_stim_time,
                f'{seg_name.upper()} | All neurons (QC-filtered, rep.-avg.) | n={len(roi_traces)}',
                ylabel='ΔF/F', ylim=seg_ylim
            )
            for ext in ['pdf', 'png']:
                plt.savefig(os.path.join(output_dir, f'5a_{seg_name}_global_rois.{ext}'),
                            dpi=300, bbox_inches='tight')
            plt.close('all')
        print(f"  {seg_name}: {len(roi_segments[seg_name])} neurons")

    # ---- Population summary plots (neuron-level polarity/luminance/contrast/frequency) ----
    summary_results = PopulationSummaryPlots.run_all(condition_rois, segments, mean_fps, output_dir,
                                                       stim_protocol=stim_protocol, stim_time=stim_time,
                                                       roi_accumulator=roi_accumulator, qi_pass=qc_pass)
    # ---- Response-profile clustering (already neuron-level) ---------------
    clustering_results = analyze_response_clustering(condition_rois, segments, mean_fps,
                                 stim_protocol, stim_time, clustering_dir, ylims=ylims)

    fly_effect_results = FlyEffectStats.run_sweep(clustering_results, clustering_dir)

    variability_results = VariabilityStats.run_sweep(clustering_results, dispersion_dir)

    return (condition_rois, fly_rois, qi_map, roi_segments, summary_results,
            clustering_results, fly_effect_results, variability_results, mean_fps)
    
def plot_cross_type_umap(all_results, condition_names, segments, output_dir):
    """
    Embed all QI-filtered, replicate-averaged ROIs from every cell type into one
    shared UMAP space using scalar features. Colours by cell type.

    Uses the 'features' feature set (26-dim scalar vector) so the embedding
    is interpretable and fast. Each point = one neuron.
    """
    print("\n" + "="*60)
    print("Building cross-type UMAP ...")

    all_X, all_type_labels, all_fly_labels = [], [], []

    for results, cond in zip(all_results, condition_names):
        rois     = results.get('condition_rois', [])
        mean_fps = results.get('mean_fps', 10.0)
        pol_res  = results.get('polarity_results')
        lum_res  = results.get('luminance_results')
        cnt_res  = results.get('contrast_results')
        frq_res  = results.get('frequency_results')

        for (trace, fly_id, roi_key) in rois:
            feat_vec, _ = extract_roi_features(
                trace, None, segments, mean_fps,
                pol_res, lum_res, cnt_res, frq_res)
            if not np.all(np.isfinite(feat_vec)):
                continue
            all_X.append(feat_vec)
            all_type_labels.append(cond)
            all_fly_labels.append(fly_id)

    if len(all_X) < 10:
        print("  Insufficient data for cross-type UMAP, skipping.")
        return

    X = np.vstack(all_X)
    # z-score per feature across all ROIs
    col_std = X.std(axis=0); col_std[col_std == 0] = 1.0
    X = (X - X.mean(axis=0)) / col_std

    print(f"  Running UMAP on {X.shape[0]} ROIs × {X.shape[1]} features ...")
    reducer   = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          random_state=42)
    embedding = reducer.fit_transform(X)

    unique_types = list(dict.fromkeys(all_type_labels))   # preserve order
    cmap   = plt.cm.tab10
    colors = {t: cmap(i % 10) for i, t in enumerate(unique_types)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel — coloured by cell type
    ax = axes[0]
    for t in unique_types:
        mask = np.array(all_type_labels) == t
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   color=colors[t], s=25, alpha=0.7, label=t, edgecolors='none')
    ax.set_title('Cross-type UMAP\n(coloured by cell type)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.legend(markerscale=2, fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    # Right panel — coloured by fly (sanity check for preparation bias)
    ax = axes[1]
    unique_flies = list(dict.fromkeys(all_fly_labels))
    fly_colors   = {f: cmap(i % 10) for i, f in enumerate(unique_flies)}
    for f in unique_flies:
        mask = np.array(all_fly_labels) == f
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   color=fly_colors[f], s=25, alpha=0.5, edgecolors='none')
    # overlay type centroids as large markers
    for t in unique_types:
        mask = np.array(all_type_labels) == t
        cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
        ax.scatter(cx, cy, color=colors[t], s=250, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=5, label=t)
    ax.set_title('Cross-type UMAP\n(coloured by fly, ★ = type centroid)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.legend(markerscale=1, fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    plt.suptitle('All cell types — shared feature space', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'combined_cross_type_umap.{ext}'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Cross-type UMAP saved.")


class CombinedAnalysis:
    """
    Cross-condition comparison, rebuilt around neuron-level summary_results /
    roi_segments / fly_effect_results / variability_results instead of the
    old fly-averaged result dicts. Replaces plot_combined_analysis,
    plot_functional_fingerprint, _extract_freq_peak, _extract_contrast_slope.
    plot_cross_type_umap is untouched — already neuron-level, called directly.
    """
    _FEATURE_NAMES_26 = [
        'polarity_peak', 'polarity_trough', 'polarity_auc', 'polarity_peak_lat', 'polarity_rise',
        'frequency_peak', 'frequency_trough', 'frequency_auc', 'frequency_peak_lat', 'frequency_rise',
        'contrast_peak', 'contrast_trough', 'contrast_auc', 'contrast_peak_lat', 'contrast_rise',
        'luminance_peak', 'luminance_trough', 'luminance_auc', 'luminance_peak_lat', 'luminance_rise',
        'on_amplitude', 'off_amplitude', 'lum_slope', 'on_off_index',
        'freq_tuning_peak_hz', 'contrast_tuning_peak_pct',
    ]
    @staticmethod
    def plot_dispersion_indices_by_genotype(all_results, condition_names, output_dir, feature_set='features'):
        """
        SD / CV / MAD / IQR per genotype, per dimension of one feature_set,
        grouped bars colored by genotype. Sourced from dispersion_table,
        computed separately per genotype inside VariabilityStats.run_sweep --
        comparison happens only on already-computed per-genotype values.
        """
        metrics = ['sd', 'cv', 'mad', 'iqr']
        titles = ['Standard Deviation', 'Coefficient of Variation', 'Median Abs. Deviation', 'IQR']
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        col_names = None
        for results in all_results:
            dt = results.get('variability_results', {}).get('dispersion_table')
            if dt is not None:
                sub = dt[dt['feature_set'] == feature_set]
                if len(sub) > 0:
                    col_names = list(sub['column'])
                    break
        if col_names is None:
            print(f"  No dispersion data for feature_set='{feature_set}' -- skipping.")
            return

        n_dims = len(col_names)
        if feature_set == 'features' and n_dims == len(CombinedAnalysis._FEATURE_NAMES_26):
            display_names = CombinedAnalysis._FEATURE_NAMES_26
        else:
            display_names = [c.replace(f'{feature_set}_', '') for c in col_names]

        x_pos = np.arange(n_dims)
        width = 0.8 / max(len(all_results), 1)

        fig, axes = plt.subplots(2, 2, figsize=(max(16, n_dims * 0.6), 12))
        for metric, title, ax in zip(metrics, titles, axes.flat):
            for gi, (results, cond, color) in enumerate(zip(all_results, condition_names, colors)):
                dt = results.get('variability_results', {}).get('dispersion_table')
                if dt is None:
                    continue
                sub = dt[dt['feature_set'] == feature_set].set_index('column').reindex(col_names)
                vals = sub[metric].values.astype(float)
                vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)
                offset = (gi - len(all_results) / 2 + 0.5) * width
                ax.bar(x_pos + offset, np.nan_to_num(vals, nan=1e-6), width, color=color,
                       edgecolor='black', linewidth=0.6, alpha=0.8, label=cond)
            ax.set_yscale('log')
            ax.set_xticks(x_pos); ax.set_xticklabels(display_names, rotation=75, ha='right', fontsize=7)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', which='both')
            ax.legend(fontsize=8, loc='best')

        fig.suptitle(f'Dispersion Indices by Genotype ({feature_set})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_dispersion_indices_by_genotype_{feature_set}.{ext}'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def plot_full_feature_fingerprint(all_results, condition_names, segments, mean_fps, output_dir):
        """
        Left: population z-score of each genotype's mean (response level,
        kept as context -- computed from the pooled population, see the
        earlier fix for why that's not degenerate at low genotype counts).
        Right: Levene's test per feature, genotype as the grouping variable
        -- replaces the old ICC panel. ICC quantified variance explained by
        between-genotype MEAN differences, which isn't the question; Levene's
        tests variance directly and is provably invariant to mean differences
        (confirmed: same variance + a 10-unit mean gap still gives p=0.73).
        Each genotype's deviations are computed from its own median first;
        genotypes are only compared afterward, at the level of those
        already-separate deviation scores.
        """
        X, conds, feature_names = CombinedAnalysis._collect_condition_feature_matrix(
            all_results, condition_names, segments, mean_fps)
        n_idx, n_cond = len(feature_names), len(condition_names)

        type_means = np.full((n_cond, n_idx), np.nan)
        pop_mean = np.full(n_idx, np.nan)
        pop_std = np.full(n_idx, np.nan)
        levene_p = np.full(n_idx, np.nan)

        for ii in range(n_idx):
            values = X[:, ii]
            pop_mean[ii] = values.mean()
            pop_std[ii] = values.std(ddof=0)
            for ti, cond in enumerate(condition_names):
                cvals = values[conds == cond]
                if len(cvals) > 0:
                    type_means[ti, ii] = cvals.mean()
            groups_data = [values[conds == cond] for cond in condition_names]
            groups_data = [g for g in groups_data if len(g) > 1]
            if len(groups_data) >= 2:
                _, p = stats.levene(*groups_data)
                levene_p[ii] = p

        pop_std_safe = np.where(pop_std > 0, pop_std, 1.0)
        Z = (type_means - pop_mean) / pop_std_safe
        vmax = np.nanmax(np.abs(Z)) if np.any(np.isfinite(Z)) else 1.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, n_idx * 0.7), max(6, n_cond * 1.3 + 2)),
                                        gridspec_kw={'width_ratios': [3, 1]})
        im = ax1.imshow(Z, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        for ti in range(n_cond):
            for ii in range(n_idx):
                if np.isfinite(Z[ti, ii]):
                    fc = 'white' if abs(Z[ti, ii]) > 0.6 * vmax else 'black'
                    ax1.text(ii, ti, f'{Z[ti, ii]:.2f}', ha='center', va='center', fontsize=6, color=fc)
        ax1.set_xticks(range(n_idx)); ax1.set_xticklabels(feature_names, rotation=75, ha='right', fontsize=7)
        ax1.set_yticks(range(n_cond)); ax1.set_yticklabels(condition_names, fontsize=10)
        ax1.set_title('Full Functional Fingerprint (26 features, population z-score)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='population z-score', shrink=0.6)

        neg_log_p = -np.log10(np.clip(levene_p, 1e-300, 1.0))
        colors_bar = ['#2ecc71' if p < 0.01 else '#f39c12' if p < 0.05 else '#e74c3c' for p in levene_p]
        ax2.barh(range(n_idx), neg_log_p, color=colors_bar, edgecolor='black', linewidth=0.6, alpha=0.85)
        ax2.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='p=0.05')
        for i, p in enumerate(levene_p):
            if np.isfinite(p):
                ax2.text(neg_log_p[i] + 0.05, i, f'p={p:.3f}', va='center', fontsize=6)
        ax2.set_yticks(range(n_idx)); ax2.set_yticklabels(feature_names, fontsize=7)
        ax2.set_xlabel('-log10(p), Levene\'s test\n(variance difference across genotypes)', fontsize=9)
        ax2.set_title('Variance difference significance', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x'); ax2.legend(fontsize=8)

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_functional_fingerprint_full.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def _collect_condition_feature_matrix(all_results, condition_names, segments, mean_fps):
        rows, conds, feature_names = [], [], None
        for results, cond in zip(all_results, condition_names):
            for trace, fly_id, roi_key in results['condition_rois']:
                feat, names = extract_roi_features(trace, None, segments, mean_fps)
                if feature_names is None:
                    feature_names = names
                rows.append(feat)
                conds.append(cond)
        return np.vstack(rows), np.array(conds), feature_names

    @staticmethod
    def plot_full_feature_fingerprint(all_results, condition_names, segments, mean_fps, output_dir):
        """
        Color and text both show the population z-score: each genotype's mean
        feature value, minus the pooled population mean (all neurons, all
        genotypes together), divided by the pooled population SD (same pool).
        Not the condition-mean z-score from before -- that measured spread
        across only 2-3 genotype means, which is why it was capped near
        sqrt(N_genotypes-1) regardless of real effect size. This measures
        spread across every neuron, so it isn't degenerate at low genotype
        counts and needs no floor/clip.
        """
        X, conds, feature_names = CombinedAnalysis._collect_condition_feature_matrix(
            all_results, condition_names, segments, mean_fps)
        n_idx, n_cond = len(feature_names), len(condition_names)

        type_means = np.full((n_cond, n_idx), np.nan)
        condition_effect = np.full(n_idx, np.nan)
        pop_mean = np.full(n_idx, np.nan)
        pop_std = np.full(n_idx, np.nan)

        for ii in range(n_idx):
            values = X[:, ii]
            pop_mean[ii] = values.mean()
            pop_std[ii] = values.std(ddof=0)
            for ti, cond in enumerate(condition_names):
                cvals = values[conds == cond]
                if len(cvals) > 0:
                    type_means[ti, ii] = cvals.mean()
            if len(set(conds)) >= 2:
                condition_effect[ii] = FlyEffectStats.compute_icc(values, conds)['icc']

        pop_std_safe = np.where(pop_std > 0, pop_std, 1.0)
        Z = (type_means - pop_mean) / pop_std_safe
        vmax = np.nanmax(np.abs(Z)) if np.any(np.isfinite(Z)) else 1.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, n_idx * 0.7), max(6, n_cond * 1.3 + 2)),
                                        gridspec_kw={'width_ratios': [3, 1]})
        im = ax1.imshow(Z, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        for ti in range(n_cond):
            for ii in range(n_idx):
                if np.isfinite(Z[ti, ii]):
                    fc = 'white' if abs(Z[ti, ii]) > 0.6 * vmax else 'black'
                    ax1.text(ii, ti, f'{Z[ti, ii]:.2f}', ha='center', va='center', fontsize=7, color=fc)
        ax1.set_xticks(range(n_idx)); ax1.set_xticklabels(feature_names, rotation=75, ha='right', fontsize=7)
        ax1.set_yticks(range(n_cond)); ax1.set_yticklabels(condition_names, fontsize=10)
        ax1.set_title('Full Functional Fingerprint (26 features, population z-score)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='population z-score', shrink=0.6)

        colors_bar = ['#2ecc71' if r >= 0.5 else '#e74c3c' if r < 0.25 else '#f39c12' for r in condition_effect]
        ax2.barh(range(n_idx), condition_effect, color=colors_bar, edgecolor='black', linewidth=0.6, alpha=0.85)
        ax2.axvline(0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax2.set_yticks(range(n_idx)); ax2.set_yticklabels(feature_names, fontsize=7)
        ax2.set_xlabel('ICC: variance explained by condition', fontsize=9)
        ax2.set_title('Condition effect size', fontsize=11, fontweight='bold')
        ax2.set_xlim(0, 1.15); ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_functional_fingerprint_full.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def compute_segment_variability_across_rois(roi_segments):
        """
        Per segment: CV and reliability ACROSS neurons (not across each
        neuron's own repetitions) -- the ROI-level analog of the old
        analyze_across_fly_variability. Each QC-passing neuron's (already
        repetition-averaged) segment trace is one replicate. CV = pointwise
        std/|mean| across neurons at each timepoint, averaged across time
        (non-finite treated as 0, matching the original convention).
        Reliability = mean pairwise correlation between different neurons'
        segment traces. Returns one scalar per segment, not a distribution.
        """
        seg_cv = {}
        seg_reliability = {}
        for seg_name, trace_list in roi_segments.items():
            if len(trace_list) < 2:
                continue
            min_len = min(len(tr) for tr, _ in trace_list)
            aligned = np.array([tr[:min_len] for tr, _ in trace_list])

            mean_trace = aligned.mean(axis=0)
            std_trace = aligned.std(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv_trace = std_trace / np.abs(mean_trace)
                cv_trace[~np.isfinite(cv_trace)] = 0
            seg_cv[seg_name] = float(np.mean(cv_trace))

            n_rois = len(aligned)
            correlations = []
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    corr = np.corrcoef(aligned[i], aligned[j])[0, 1]
                    if np.isfinite(corr):
                        correlations.append(corr)
            seg_reliability[seg_name] = float(np.mean(correlations)) if correlations else 0.0
        return {'cv': seg_cv, 'reliability': seg_reliability}
    @staticmethod
    def plot_segment_variability_comparison(all_results, condition_names, output_dir):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
        x_pos = np.arange(len(segment_names))
        width = 0.8 / max(len(all_results), 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
            roi_segments = results.get('roi_segments')
            if not roi_segments:
                continue
            seg_var = CombinedAnalysis.compute_segment_variability_across_rois(roi_segments)
            cv_vals = [seg_var['cv'].get(s, np.nan) for s in segment_names]
            rel_vals = [seg_var['reliability'].get(s, np.nan) for s in segment_names]
            offset = (idx - len(all_results) / 2 + 0.5) * width
            ax1.bar(x_pos + offset, cv_vals, width, color=color,
                    alpha=0.7, edgecolor='black', linewidth=1, label=condition)
            ax2.bar(x_pos + offset, rel_vals, width, color=color,
                    alpha=0.7, edgecolor='black', linewidth=1, label=condition)

        ax1.set_ylabel('Coefficient of Variation', fontsize=12)
        ax1.set_title('Across-ROI Variability (CV)', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos); ax1.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y'); ax1.legend(fontsize=9, loc='best')

        ax2.set_ylabel('Mean Pairwise Correlation', fontsize=12)
        ax2.set_title('Across-ROI Reliability', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos); ax2.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
        ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3, axis='y'); ax2.legend(fontsize=9, loc='best')

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_across_roi_variability_bar.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def plot_luminance_comparison(all_results, condition_names, output_dir):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x_positions = np.arange(len(all_results))
        for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
            lum = results['summary_results']['luminance']
            n_neurons = len(lum['per_neuron_slopes'])
            x_vals = sorted(lum['overall_mean'].keys())
            y_vals = [lum['overall_mean'][x] for x in x_vals]
            y_sem = [lum['overall_sem'][x] for x in x_vals]
            ax1.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
                         markersize=6, capsize=4, linewidth=2,
                         label=f'{condition} (n={n_neurons})', alpha=0.8)
            ax2.bar(x_positions[idx], lum['mean_slope'], yerr=lum['sem_slope'], color=color,
                    edgecolor='black', linewidth=1.5, capsize=5, width=0.7, alpha=0.8, label=condition)
            jitter = np.random.default_rng(0).normal(0, 0.08, size=n_neurons)
            ax2.scatter(x_positions[idx] + jitter, lum['per_neuron_slopes'], color='black', s=12, alpha=0.4, zorder=3)
        ax1.set_xlabel('Luminance Value', fontsize=12); ax1.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
        ax1.set_title('Luminance Response Curves (neuron-level)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)
        ax2.set_ylabel('Slope (ΔF/F per luminance unit)', fontsize=12)
        ax2.set_title('Luminance Slopes — every neuron shown', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_positions); ax2.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.legend(fontsize=8)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_luminance_analysis.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_polarity_comparison(all_results, condition_names, output_dir):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        fig, ax = plt.subplots(figsize=(10, 6))
        x_positions = np.arange(len(all_results)); width = 0.35
        for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
            pol = results['summary_results']['polarity']
            ax.bar(x_positions[idx] - width/2, pol['mean_on'], width, yerr=pol['sem_on'],
                   color=color, alpha=0.7, edgecolor='black', linewidth=1.5, capsize=4,
                   label=f'{condition} - ON' if idx == 0 else '')
            ax.bar(x_positions[idx] + width/2, pol['mean_off'], width, yerr=pol['sem_off'],
                   color=color, alpha=0.4, edgecolor='black', linewidth=1.5, capsize=4,
                   label=f'{condition} - OFF' if idx == 0 else '')
            jitter_on = np.random.default_rng(0).normal(0, 0.06, size=len(pol['on_amplitudes']))
            jitter_off = np.random.default_rng(1).normal(0, 0.06, size=len(pol['off_amplitudes']))
            ax.scatter(x_positions[idx] - width/2 + jitter_on, pol['on_amplitudes'], color='black', s=8, alpha=0.3, zorder=3)
            ax.scatter(x_positions[idx] + width/2 + jitter_off, pol['off_amplitudes'], color='black', s=8, alpha=0.3, zorder=3)
        legend_elements = [Patch(facecolor=c, edgecolor='black', label=cond) for cond, c in zip(condition_names, colors)]
        legend_elements += [Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='ON'),
                             Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='OFF')]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
        ax.set_title('Polarity Responses (ON vs OFF) — every neuron shown', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions); ax.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_polarity_analysis.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def _sd_cv_from_curves(curve_list):
        """
        curve_list: list of per-neuron {x_value: y_value} dicts (one dict
        per neuron, all in one genotype). Returns (sorted x values, SD-per-
        value dict, CV-per-value dict), computed across neurons at each x.
        """
        all_x = sorted(set(x for c in curve_list for x in c))
        sd_out, cv_out = {}, {}
        for x in all_x:
            vals = np.array([c[x] for c in curve_list if x in c])
            if len(vals) < 2:
                continue
            sd = vals.std()
            mean = vals.mean()
            sd_out[x] = sd
            cv_out[x] = sd / abs(mean) if abs(mean) > 1e-8 else np.nan
        return all_x, sd_out, cv_out
    @staticmethod
    def _values_by_x_across_genotypes(all_results, condition_names, summary_key, curve_key):
        """
        Reshapes summary_results[summary_key][curve_key] (list of per-neuron
        {x: y} dicts, per genotype) into: sorted x values, and for each x,
        a {genotype: np.array of per-neuron values at that x}. One entry
        per x becomes one subplot in the ridgeline grid.
        """
        all_x = set()
        per_geno_curves = {}
        for results, cond in zip(all_results, condition_names):
            curves = results.get('summary_results', {}).get(summary_key, {}).get(curve_key, [])
            per_geno_curves[cond] = curves
            for c in curves:
                all_x.update(c.keys())
        all_x = sorted(all_x)

        data_by_x = {}
        for x in all_x:
            data_by_x[x] = {}
            for cond in condition_names:
                vals = [c[x] for c in per_geno_curves.get(cond, []) if x in c]
                data_by_x[x][cond] = np.array(vals)
        return data_by_x

    @staticmethod
    def _plot_ridgeline_grid(data_by_x, condition_names, output_dir, title, file_name, n_cols=5):
        """
        Generic ridgeline grid: one subplot per key in data_by_x, each
        showing every genotype's population-z-scored density (reference
        computed from all genotypes' values at THAT key only, not other
        keys), stacked vertically -- same convention as
        plot_feature_distributions_by_genotype, generalized to work off
        any {key: {genotype: values}} mapping instead of per-feature data.
        """
        keys = list(data_by_x.keys())
        n_idx = len(keys)
        n_geno = len(condition_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_geno))
        n_rows = int(np.ceil(n_idx / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), squeeze=False)

        for ki, key in enumerate(keys):
            ax = axes[ki // n_cols][ki % n_cols]
            geno_vals = data_by_x[key]
            pooled = np.concatenate([v for v in geno_vals.values() if len(v) > 0]) if geno_vals else np.array([])
            if len(pooled) < 2:
                ax.axis('off')
                continue
            pop_mean = pooled.mean()
            pop_std = pooled.std(ddof=0)
            pop_std = pop_std if pop_std > 0 else 1.0

            z_range = np.linspace((pooled.min() - pop_mean) / pop_std - 0.5,
                                  (pooled.max() - pop_mean) / pop_std + 0.5, 300)
            kdes = []
            for cond in condition_names:
                vals = geno_vals.get(cond, np.array([]))
                if len(vals) < 2 or np.std(vals) == 0:
                    kdes.append(np.zeros_like(z_range))
                    continue
                z_vals = (vals - pop_mean) / pop_std
                kde = stats.gaussian_kde(z_vals)
                kdes.append(kde(z_range))
            max_height = max((k.max() for k in kdes if k.max() > 0), default=1.0)
            offset_step = max_height * 1.1

            for gi, (color, dens) in enumerate(zip(colors, kdes)):
                baseline = gi * offset_step
                ax.fill_between(z_range, baseline, baseline + dens, color=color, alpha=0.6, linewidth=0.8, edgecolor=color)
                ax.plot(z_range, baseline + dens, color=color, linewidth=0.8)

            title_str = f'{key:g}' if isinstance(key, (int, float)) else str(key)
            ax.set_title(title_str, fontsize=9, fontweight='bold')
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=6)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        for ki in range(n_idx, n_rows * n_cols):
            axes[ki // n_cols][ki % n_cols].axis('off')

        legend_elements = [Patch(facecolor=colors[gi], edgecolor='black', label=cond)
                           for gi, cond in enumerate(condition_names)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                   ncol=min(n_geno, 6), fontsize=10, frameon=False)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_polarity_distributions_by_genotype(all_results, condition_names, output_dir):
        data_by_x = {'ON': {}, 'OFF': {}}
        for results, cond in zip(all_results, condition_names):
            pol = results.get('summary_results', {}).get('polarity', {})
            data_by_x['ON'][cond] = np.asarray(pol.get('on_amplitudes', []))
            data_by_x['OFF'][cond] = np.asarray(pol.get('off_amplitudes', []))
        for ext in ['pdf', 'png']:
            CombinedAnalysis._plot_ridgeline_grid(
                data_by_x, condition_names, output_dir,
                title='Polarity Response Distributions by Genotype (population z-score)',
                file_name=f'combined_polarity_distributions_by_genotype.{ext}', n_cols=2)

    @staticmethod
    def plot_luminance_distributions_by_genotype(all_results, condition_names, output_dir):
        data_by_x = CombinedAnalysis._values_by_x_across_genotypes(
            all_results, condition_names, 'luminance', 'per_neuron_curves')
        for ext in ['pdf', 'png']:
            CombinedAnalysis._plot_ridgeline_grid(
                data_by_x, condition_names, output_dir,
                title='Luminance Response Distributions by Genotype (population z-score)',
                file_name=f'combined_luminance_distributions_by_genotype.{ext}')

    @staticmethod
    def plot_frequency_distributions_by_genotype(all_results, condition_names, output_dir):
        data_by_x = CombinedAnalysis._values_by_x_across_genotypes(
            all_results, condition_names, 'frequency', 'curves')
        for ext in ['pdf', 'png']:
            CombinedAnalysis._plot_ridgeline_grid(
                data_by_x, condition_names, output_dir,
                title='Frequency Response Distributions by Genotype (population z-score)',
                file_name=f'combined_frequency_distributions_by_genotype.{ext}')

    @staticmethod
    def plot_contrast_distributions_by_genotype(all_results, condition_names, output_dir):
        data_weber = CombinedAnalysis._values_by_x_across_genotypes(
            all_results, condition_names, 'contrast', 'weber_curves')
        for ext in ['pdf', 'png']:
            CombinedAnalysis._plot_ridgeline_grid(
                data_weber, condition_names, output_dir,
                title='Weber Contrast Response Distributions by Genotype (population z-score)',
                file_name=f'combined_contrast_weber_distributions_by_genotype.{ext}')
        data_abs = CombinedAnalysis._values_by_x_across_genotypes(
            all_results, condition_names, 'contrast', 'abs_curves')
        for ext in ['pdf', 'png']:
            CombinedAnalysis._plot_ridgeline_grid(
                data_abs, condition_names, output_dir,
                title='Absolute Contrast Response Distributions by Genotype (population z-score)',
                file_name=f'combined_contrast_absolute_distributions_by_genotype.{ext}')
    @staticmethod
    def plot_polarity_variability_comparison(all_results, condition_names, output_dir):
        """SD/CV of ON and OFF amplitude, across neurons, per genotype."""
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        labels = ['ON', 'OFF']
        x_pos = np.arange(len(labels))
        width = 0.8 / max(len(all_results), 1)

        per_geno = []
        for results in all_results:
            pol = results.get('summary_results', {}).get('polarity')
            if not pol:
                per_geno.append((np.array([]), np.array([])))
                continue
            per_geno.append((np.asarray(pol['on_amplitudes']), np.asarray(pol['off_amplitudes'])))

        fig, (ax_sd, ax_cv) = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (cond, color, (on_vals, off_vals)) in enumerate(zip(condition_names, colors, per_geno)):
            if len(on_vals) < 2:
                continue
            sd_vals = [on_vals.std(), off_vals.std()]
            cv_vals = [sd_vals[0] / abs(on_vals.mean()) if abs(on_vals.mean()) > 1e-8 else np.nan,
                       sd_vals[1] / abs(off_vals.mean()) if abs(off_vals.mean()) > 1e-8 else np.nan]
            offset = (idx - len(all_results) / 2 + 0.5) * width
            ax_sd.bar(x_pos + offset, sd_vals, width, color=color, edgecolor='black', linewidth=0.8, alpha=0.8, label=cond)
            ax_cv.bar(x_pos + offset, cv_vals, width, color=color, edgecolor='black', linewidth=0.8, alpha=0.8, label=cond)
        ax_sd.set_xticks(x_pos); ax_sd.set_xticklabels(labels); ax_sd.set_ylabel('SD (ΔF/F)')
        ax_sd.set_title('Standard Deviation'); ax_sd.grid(True, alpha=0.3, axis='y'); ax_sd.legend(fontsize=8, loc='best')
        ax_cv.set_xticks(x_pos); ax_cv.set_xticklabels(labels); ax_cv.set_ylabel('CV')
        ax_cv.set_title('Coefficient of Variation'); ax_cv.grid(True, alpha=0.3, axis='y'); ax_cv.legend(fontsize=8, loc='best')
        fig.suptitle('Polarity Response Variability by Genotype', fontsize=14, fontweight='bold')
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_polarity_variability_by_value.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()

        fig, (ax_sd, ax_cv) = plt.subplots(1, 2, figsize=(10, 5))
        for idx, (cond, color, (on_vals, off_vals)) in enumerate(zip(condition_names, colors, per_geno)):
            if len(on_vals) < 2:
                continue
            cv_on = on_vals.std() / abs(on_vals.mean()) if abs(on_vals.mean()) > 1e-8 else np.nan
            cv_off = off_vals.std() / abs(off_vals.mean()) if abs(off_vals.mean()) > 1e-8 else np.nan
            ax_sd.bar(idx, np.mean([on_vals.std(), off_vals.std()]), color=color, edgecolor='black', linewidth=1, alpha=0.8)
            ax_cv.bar(idx, np.nanmean([cv_on, cv_off]), color=color, edgecolor='black', linewidth=1, alpha=0.8)
        ax_sd.set_xticks(range(len(condition_names))); ax_sd.set_xticklabels(condition_names, rotation=45, ha='right')
        ax_sd.set_ylabel('Mean SD (ON, OFF)'); ax_sd.set_title('Standard Deviation (averaged)'); ax_sd.grid(True, alpha=0.3, axis='y')
        ax_cv.set_xticks(range(len(condition_names))); ax_cv.set_xticklabels(condition_names, rotation=45, ha='right')
        ax_cv.set_ylabel('Mean CV (ON, OFF)'); ax_cv.set_title('Coefficient of Variation (averaged)'); ax_cv.grid(True, alpha=0.3, axis='y')
        fig.suptitle('Polarity Response Variability by Genotype — Averaged', fontsize=14, fontweight='bold')
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_polarity_variability_averaged.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_curve_variability(all_results, condition_names, output_dir, summary_key, curve_key,
                                 xlabel, title_prefix, file_prefix):
        """Generic SD/CV-by-value and SD/CV-averaged for a curve-dict-based analysis
        (summary_results[summary_key][curve_key] = list of per-neuron {x: y} dicts)."""
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        per_geno = []
        all_x = set()
        for results in all_results:
            curves = results.get('summary_results', {}).get(summary_key, {}).get(curve_key, [])
            xs, sd_d, cv_d = CombinedAnalysis._sd_cv_from_curves(curves)
            per_geno.append((xs, sd_d, cv_d))
            all_x.update(xs)
        all_x = sorted(all_x)
        if not all_x:
            print(f"  No data for {summary_key}/{curve_key} -- skipping variability comparison.")
            return
        x_pos = np.arange(len(all_x))
        width = 0.8 / max(len(all_results), 1)

        fig, (ax_sd, ax_cv) = plt.subplots(1, 2, figsize=(max(10, len(all_x) * 0.8), 5))
        for idx, (cond, color, (xs, sd_d, cv_d)) in enumerate(zip(condition_names, colors, per_geno)):
            sd_vals = [sd_d.get(x, np.nan) for x in all_x]
            cv_vals = [cv_d.get(x, np.nan) for x in all_x]
            offset = (idx - len(all_results) / 2 + 0.5) * width
            ax_sd.bar(x_pos + offset, sd_vals, width, color=color, edgecolor='black', linewidth=0.6, alpha=0.8, label=cond)
            ax_cv.bar(x_pos + offset, cv_vals, width, color=color, edgecolor='black', linewidth=0.6, alpha=0.8, label=cond)
        ax_sd.set_xticks(x_pos); ax_sd.set_xticklabels([f'{x:g}' for x in all_x], rotation=45, ha='right', fontsize=8)
        ax_sd.set_xlabel(xlabel); ax_sd.set_ylabel('SD (ΔF/F)'); ax_sd.set_title('Standard Deviation')
        ax_sd.grid(True, alpha=0.3, axis='y'); ax_sd.legend(fontsize=8, loc='best')
        ax_cv.set_xticks(x_pos); ax_cv.set_xticklabels([f'{x:g}' for x in all_x], rotation=45, ha='right', fontsize=8)
        ax_cv.set_xlabel(xlabel); ax_cv.set_ylabel('CV'); ax_cv.set_title('Coefficient of Variation')
        ax_cv.grid(True, alpha=0.3, axis='y'); ax_cv.legend(fontsize=8, loc='best')
        fig.suptitle(f'{title_prefix} Variability by Genotype', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_variability_by_value.png'), dpi=300, bbox_inches='tight')
        plt.close()

        fig, (ax_sd, ax_cv) = plt.subplots(1, 2, figsize=(10, 5))
        for idx, (cond, color, (xs, sd_d, cv_d)) in enumerate(zip(condition_names, colors, per_geno)):
            sd_avg = np.nanmean(list(sd_d.values())) if sd_d else np.nan
            cv_avg = np.nanmean(list(cv_d.values())) if cv_d else np.nan
            ax_sd.bar(idx, sd_avg, color=color, edgecolor='black', linewidth=1, alpha=0.8)
            ax_cv.bar(idx, cv_avg, color=color, edgecolor='black', linewidth=1, alpha=0.8)
        ax_sd.set_xticks(range(len(condition_names))); ax_sd.set_xticklabels(condition_names, rotation=45, ha='right')
        ax_sd.set_ylabel(f'Mean SD (over all {xlabel.lower()} values)')
        ax_sd.set_title('Standard Deviation (averaged)'); ax_sd.grid(True, alpha=0.3, axis='y')
        ax_cv.set_xticks(range(len(condition_names))); ax_cv.set_xticklabels(condition_names, rotation=45, ha='right')
        ax_cv.set_ylabel(f'Mean CV (over all {xlabel.lower()} values)')
        ax_cv.set_title('Coefficient of Variation (averaged)'); ax_cv.grid(True, alpha=0.3, axis='y')
        fig.suptitle(f'{title_prefix} Variability by Genotype — Averaged', fontsize=14, fontweight='bold')
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'{file_prefix}_variability_averaged.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_luminance_variability_comparison(all_results, condition_names, output_dir):
        CombinedAnalysis._plot_curve_variability(
            all_results, condition_names, output_dir,
            summary_key='luminance', curve_key='per_neuron_curves',
            xlabel='Luminance Value', title_prefix='Luminance Response', file_prefix='combined_luminance')

    @staticmethod
    def plot_frequency_variability_comparison(all_results, condition_names, output_dir):
        CombinedAnalysis._plot_curve_variability(
            all_results, condition_names, output_dir,
            summary_key='frequency', curve_key='curves',
            xlabel='Temporal Frequency (Hz)', title_prefix='Frequency Response', file_prefix='combined_frequency')

    @staticmethod
    def plot_contrast_variability_comparison(all_results, condition_names, output_dir):
        CombinedAnalysis._plot_curve_variability(
            all_results, condition_names, output_dir,
            summary_key='contrast', curve_key='weber_curves',
            xlabel='Weber Contrast (%)', title_prefix='Weber Contrast Response', file_prefix='combined_contrast_weber')
        CombinedAnalysis._plot_curve_variability(
            all_results, condition_names, output_dir,
            summary_key='contrast', curve_key='abs_curves',
            xlabel='Absolute Contrast (%)', title_prefix='Absolute Contrast Response', file_prefix='combined_contrast_absolute')
    @staticmethod
    def plot_contrast_comparison(all_results, condition_names, output_dir):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for results, condition, color in zip(all_results, condition_names, colors):
            cont = results['summary_results']['contrast']
            x1 = sorted(cont['weber_mean'].keys())
            ax1.errorbar(x1, [cont['weber_mean'][x] for x in x1], yerr=[cont['weber_sem'][x] for x in x1],
                        fmt='o-', color=color, markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
            x2 = sorted(cont['abs_mean'].keys())
            ax2.errorbar(x2, [cont['abs_mean'][x] for x in x2], yerr=[cont['abs_sem'][x] for x in x2],
                        fmt='o-', color=color, markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
        ax1.set_xlabel('Weber Contrast (%)', fontsize=12); ax1.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
        ax1.set_title('Weber Contrast Response — Signed (neuron-level)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.legend(fontsize=9, loc='best')
        ax2.set_xlabel('Absolute Contrast (%)', fontsize=12); ax2.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
        ax2.set_title('Absolute Contrast Response (neuron-level)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.legend(fontsize=9, loc='best')
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_contrast_analysis.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_frequency_comparison(all_results, condition_names, output_dir):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        fig, ax = plt.subplots(figsize=(12, 6))
        for results, condition, color in zip(all_results, condition_names, colors):
            freq = results['summary_results']['frequency']
            x_vals = sorted(freq['overall_mean'].keys())
            ax.errorbar(x_vals, [freq['overall_mean'][x] for x in x_vals], yerr=[freq['overall_sem'][x] for x in x_vals],
                       fmt='o-', color=color, markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
        ax.set_xlabel('Temporal Frequency (Hz)', fontsize=12); ax.set_ylabel('Maximum Response (ΔF/F)', fontsize=12)
        ax.set_title('Frequency Response Curves (neuron-level)', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, 10.5)
        ax.grid(True, alpha=0.3); ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(fontsize=9, loc='best')
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_frequency_analysis.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_segment_trace_comparison(all_results, condition_names, output_dir):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
        for seg_name in segment_names:
            fig, ax = plt.subplots(figsize=(14, 7))
            for results, condition, color in zip(all_results, condition_names, colors):
                seg_list = results['roi_segments'].get(seg_name, [])
                if not seg_list:
                    continue
                min_len = min(len(tr) for tr, _ in seg_list)
                traces = np.array([tr[:min_len] for tr, _ in seg_list])
                time = seg_list[0][1][:min_len]
                mean_trace = traces.mean(axis=0)
                sem_trace = traces.std(axis=0) / np.sqrt(len(traces))
                ax.plot(time, mean_trace, color=color, linewidth=2, label=f'{condition} (n={len(traces)})', alpha=0.85)
                ax.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.15)
            ax.set_xlabel('Time (s)', fontsize=12); ax.set_ylabel('ΔF/F', fontsize=12)
            ax.set_title(f'{seg_name.upper()} — combined across conditions (neuron-level mean ± SEM)',
                         fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            plt.tight_layout()
            for ext in ['pdf', 'png']:
                plt.savefig(os.path.join(output_dir, f'combined_segment_{seg_name}.{ext}'), dpi=300, bbox_inches='tight')
            plt.close()

    @staticmethod
    def _collect_condition_indices(all_results, condition_names):
        """Per-neuron (value, condition) pairs for 5 key indices, pulled from each
        condition's already-computed summary_results — no raw-trace recomputation."""
        index_data = {'ON amplitude': [], 'OFF amplitude': [], 'Luminance slope': [],
                      'Freq. tuning peak (Hz)': [], 'Contrast slope (abs.)': []}
        for results, cond in zip(all_results, condition_names):
            sr = results['summary_results']
            for v in sr['polarity']['on_amplitudes']:
                index_data['ON amplitude'].append((v, cond))
            for v in sr['polarity']['off_amplitudes']:
                index_data['OFF amplitude'].append((v, cond))
            for v in sr['luminance']['per_neuron_slopes']:
                index_data['Luminance slope'].append((v, cond))
            for curve in sr['frequency']['curves']:
                if curve:
                    index_data['Freq. tuning peak (Hz)'].append((max(curve, key=curve.get), cond))
            for curve in sr['contrast']['abs_curves']:
                if len(curve) >= 2:
                    xs = np.array(sorted(curve.keys())); ys = np.array([curve[x] for x in xs])
                    slope, _ = np.polyfit(xs, ys, 1)
                    index_data['Contrast slope (abs.)'].append((float(slope), cond))
        return index_data

    @staticmethod
    def plot_condition_effect_summary(all_results, condition_names, output_dir):
        """
        Successor to plot_functional_fingerprint: same heatmap-plus-effect-size
        layout, sourced from per-neuron indices. Color and text both show the
        population z-score: each condition's mean value, minus the pooled
        population mean (all neurons, all conditions together), divided by
        the pooled population SD (same pool) -- not a z-score across just the
        2-3 condition means, which is degenerate at low condition counts.
        Right panel still uses FlyEffectStats.compute_icc with `condition` as
        the grouping variable in place of `fly` -- unchanged by any of this.
        """
        index_data = CombinedAnalysis._collect_condition_indices(all_results, condition_names)
        index_names = list(index_data.keys())
        n_idx, n_cond = len(index_names), len(condition_names)
        type_means = np.full((n_cond, n_idx), np.nan)
        condition_effect = np.full(n_idx, np.nan)
        pop_mean = np.full(n_idx, np.nan)
        pop_std = np.full(n_idx, np.nan)

        for ii, iname in enumerate(index_names):
            pairs = index_data[iname]
            if len(pairs) < 2:
                continue
            values = np.array([v for v, c in pairs])
            conds = np.array([c for v, c in pairs])
            pop_mean[ii] = values.mean()
            pop_std[ii] = values.std(ddof=0)
            for ti, cond in enumerate(condition_names):
                cvals = values[conds == cond]
                if len(cvals) > 0:
                    type_means[ti, ii] = cvals.mean()
            if len(set(conds)) >= 2:
                icc_res = FlyEffectStats.compute_icc(values, conds)
                condition_effect[ii] = icc_res['icc']

        pop_std_safe = np.where(pop_std > 0, pop_std, 1.0)
        Z = (type_means - pop_mean) / pop_std_safe
        vmax = np.nanmax(np.abs(Z)) if np.any(np.isfinite(Z)) else 1.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5 + n_idx * 1.2, max(6, n_cond * 0.9 + 2)),
                                        gridspec_kw={'width_ratios': [3, 1]})
        im = ax1.imshow(Z, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        for ti in range(n_cond):
            for ii in range(n_idx):
                if np.isfinite(Z[ti, ii]):
                    fc = 'white' if abs(Z[ti, ii]) > 0.6 * vmax else 'black'
                    ax1.text(ii, ti, f'{Z[ti, ii]:.2f}', ha='center', va='center',
                             fontsize=9, color=fc, fontweight='bold')
        ax1.set_xticks(range(n_idx)); ax1.set_xticklabels(index_names, rotation=40, ha='right', fontsize=9)
        ax1.set_yticks(range(n_cond)); ax1.set_yticklabels(condition_names, fontsize=10)
        ax1.set_title('Functional Fingerprint (neuron-level, population z-score)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='population z-score', shrink=0.6)

        colors_bar = ['#2ecc71' if r >= 0.5 else '#e74c3c' if r < 0.25 else '#f39c12' for r in condition_effect]
        ax2.barh(range(n_idx), condition_effect, color=colors_bar, edgecolor='black', linewidth=0.8, alpha=0.85)
        ax2.axvline(0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='50% threshold')
        for i, v in enumerate(condition_effect):
            if np.isfinite(v):
                ax2.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
        ax2.set_yticks(range(n_idx)); ax2.set_yticklabels(index_names, fontsize=9)
        ax2.set_xlabel('ICC: variance explained\nby condition (neuron-level)', fontsize=9)
        ax2.set_title('Condition effect size', fontsize=11, fontweight='bold')
        ax2.set_xlim(0, 1.15); ax2.grid(True, alpha=0.3, axis='x'); ax2.legend(fontsize=8)

        plt.suptitle('Cross-condition functional comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_functional_fingerprint.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def plot_multivariate_dispersion_by_genotype(all_results, condition_names, output_dir):
        """
        Mean pairwise neuron-to-neuron distance per genotype, one panel per
        feature_set, fly-nested bootstrap mean + 95% CI. Companion to
        plot_participation_ratio_by_genotype -- PR asks "how many effective
        independent directions of variation", this asks "how far apart,
        overall, are this genotype's neurons from each other". Both are
        computed entirely within each genotype by VariabilityStats.run_sweep;
        only placed side by side here.
        """
        feature_sets = []
        for results in all_results:
            boot_table = results.get('variability_results', {}).get('nested_bootstrap_table')
            if boot_table is not None:
                for fset in boot_table['feature_set']:
                    if fset not in feature_sets:
                        feature_sets.append(fset)

        if not feature_sets:
            print("  No dispersion bootstrap data found across conditions -- skipping.")
            return

        n_fsets = len(feature_sets)
        n_cols = min(4, n_fsets)
        n_rows = int(np.ceil(n_fsets / n_cols))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

        for fi, fset in enumerate(feature_sets):
            ax = axes[fi // n_cols][fi % n_cols]
            for gi, (results, cond, color) in enumerate(zip(all_results, condition_names, colors)):
                boot_table = results.get('variability_results', {}).get('nested_bootstrap_table')
                row = boot_table[boot_table['feature_set'] == fset] if boot_table is not None else None
                if row is None or len(row) == 0:
                    continue
                d_mean = row['dispersion_mean'].values[0]
                d_lo = row['dispersion_ci_lower'].values[0]
                d_hi = row['dispersion_ci_upper'].values[0]
                lower_err = max(0.0, d_mean - d_lo)
                upper_err = max(0.0, d_hi - d_mean)
                ax.bar(gi, d_mean, yerr=[[lower_err], [upper_err]], color=color, edgecolor='black',
                       linewidth=1, capsize=4, alpha=0.8)
            ax.set_xticks(range(len(condition_names)))
            ax.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=8)
            ax.set_title(fset, fontsize=10, fontweight='bold')
            ax.set_ylabel('Mean pairwise distance', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        for fi in range(n_fsets, n_rows * n_cols):
            axes[fi // n_cols][fi % n_cols].axis('off')

        legend_elements = [Patch(facecolor=colors[gi], edgecolor='black', label=cond)
                           for gi, cond in enumerate(condition_names)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.06),
                   ncol=min(len(condition_names), 6), fontsize=10, frameon=False)
        fig.suptitle('Multivariate Dispersion by Genotype (fly-nested bootstrap mean ± 95% CI)',
                     fontsize=13, fontweight='bold', y=1.12)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
         plt.savefig(os.path.join(output_dir, f'combined_multivariate_dispersion_by_genotype.{ext}'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def plot_participation_ratio_by_genotype(all_results, condition_names, output_dir):
        """
        Participation ratio per genotype/condition, one panel per
        feature_set, using the fly-nested bootstrap mean + 95% CI (not
        the raw point estimate) for proper uncertainty on the comparison.
        """
        feature_sets = []
        for results in all_results:
            boot_table = results.get('variability_results', {}).get('nested_bootstrap_table')
            if boot_table is not None:
                for fset in boot_table['feature_set']:
                    if fset not in feature_sets:
                        feature_sets.append(fset)

        if not feature_sets:
            print("  No participation-ratio bootstrap data found across conditions -- skipping.")
            return

        n_fsets = len(feature_sets)
        n_cols = min(4, n_fsets)
        n_rows = int(np.ceil(n_fsets / n_cols))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

        for fi, fset in enumerate(feature_sets):
            ax = axes[fi // n_cols][fi % n_cols]
            for gi, (results, cond, color) in enumerate(zip(all_results, condition_names, colors)):
                boot_table = results.get('variability_results', {}).get('nested_bootstrap_table')
                row = boot_table[boot_table['feature_set'] == fset] if boot_table is not None else None
                if row is None or len(row) == 0:
                    continue
                pr_mean = row['pr_mean'].values[0]
                pr_lo = row['pr_ci_lower'].values[0]
                pr_hi = row['pr_ci_upper'].values[0]
                lower_err = max(0.0, pr_mean - pr_lo)
                upper_err = max(0.0, pr_hi - pr_mean)
                ax.bar(gi, pr_mean, yerr=[[lower_err], [upper_err]], color=color, edgecolor='black',
                       linewidth=1, capsize=4, alpha=0.8)
            ax.set_xticks(range(len(condition_names)))
            ax.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=8)
            ax.set_title(fset, fontsize=10, fontweight='bold')
            ax.set_ylabel('Participation ratio', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        for fi in range(n_fsets, n_rows * n_cols):
            axes[fi // n_cols][fi % n_cols].axis('off')

        legend_elements = [Patch(facecolor=colors[gi], edgecolor='black', label=cond)
                           for gi, cond in enumerate(condition_names)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.06),
                   ncol=min(len(condition_names), 6), fontsize=10, frameon=False)
        fig.suptitle('Participation Ratio by Genotype (fly-nested bootstrap mean ± 95% CI)',
                     fontsize=13, fontweight='bold', y=1.12)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_participation_ratio_by_genotype.{ext}'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def combine_stat_tables(all_results, condition_names, output_dir):
        """
        Concatenates each condition's already-saved ICC/PERMANOVA/cluster-fly/
        dispersion/PR/Levene-Bartlett/bootstrap tables into one combined CSV
        per type, tagged with a `condition` column. Supersedes the old fixed
        within/across-fly-variability combined bar charts with something you
        can filter and re-plot however the actual comparison needs.
        """
        table_specs = [
            ('fly_effect_results', 'icc_table', 'combined_icc_table.csv'),
            ('fly_effect_results', 'permanova_table', 'combined_permanova_table.csv'),
            ('fly_effect_results', 'cluster_fly_table', 'combined_cluster_fly_table.csv'),
            ('variability_results', 'dispersion_table', 'combined_dispersion_table.csv'),
            ('variability_results', 'participation_ratio_table', 'combined_participation_ratio_table.csv'),
            ('variability_results', 'levene_bartlett_by_cluster', 'combined_levene_bartlett_by_cluster.csv'),
            ('variability_results', 'levene_bartlett_by_fly', 'combined_levene_bartlett_by_fly.csv'),
            ('variability_results', 'nested_bootstrap_table', 'combined_nested_bootstrap_table.csv'),
        ]
        for result_key, table_key, filename in table_specs:
            pieces = []
            for results, cond in zip(all_results, condition_names):
                df = results.get(result_key, {}).get(table_key)
                if df is None or len(df) == 0:
                    continue
                df = df.copy()
                df.insert(0, 'condition', cond)
                pieces.append(df)
            if pieces:
                pd.concat(pieces, ignore_index=True).to_csv(os.path.join(output_dir, filename), index=False)
        print('  Combined statistics tables saved.')
    @staticmethod
    def plot_feature_distributions_by_genotype(all_results, condition_names, segments, mean_fps, output_dir):
        """
        One subplot per feature (26 total). Within each, every genotype's
        distribution for that feature as a ridgeline-style filled density
        curve, z-normalized against a population reference shared across
        all genotypes (computed once per feature, all neurons from all
        genotypes pooled -- used only as a common ruler, never reported as
        a statistic itself). Densities are NOT independently rescaled to a
        common peak height, so a genotype's actual width/peakedness stays
        visually meaningful and comparable to the others in the same panel.
        """
        X, conds, feature_names = CombinedAnalysis._collect_condition_feature_matrix(
            all_results, condition_names, segments, mean_fps)
        n_idx = len(feature_names)
        n_geno = len(condition_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_geno))

        n_cols = 5
        n_rows = int(np.ceil(n_idx / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), squeeze=False)

        for fi in range(n_idx):
            ax = axes[fi // n_cols][fi % n_cols]
            values = X[:, fi]
            pop_mean = values.mean()
            pop_std = values.std(ddof=0)
            pop_std = pop_std if pop_std > 0 else 1.0
            z_all = (values - pop_mean) / pop_std

            kdes = []
            z_range = np.linspace(z_all.min() - 0.5, z_all.max() + 0.5, 300)
            for cond in condition_names:
                gvals = z_all[conds == cond]
                gvals = gvals[np.isfinite(gvals)]

                if len(gvals) < 2:
                    kdes.append(np.zeros_like(z_range))
                    print(f"  {feature_names[fi]!r}, {cond!r}: skipping KDE (n={len(gvals)} < 2)")
                    continue

                gvals_std = np.std(gvals)
                if gvals_std < 1e-8:
                    kdes.append(np.zeros_like(z_range))
                    print(f"  {feature_names[fi]!r}, {cond!r}: skipping KDE (std={gvals_std:.2g} ~ 0, n={len(gvals)})")
                    continue

                try:
                    kde = stats.gaussian_kde(gvals)
                    kdes.append(kde(z_range))
                except (np.linalg.LinAlgError, ValueError):
                    kdes.append(np.zeros_like(z_range))
                    print(f"  {feature_names[fi]!r}, {cond!r}: gaussian_kde failed unexpectedly (n={len(gvals)}), skipping")

            max_height = max((k.max() for k in kdes if k.max() > 0), default=1.0)
            offset_step = max_height * 1.1

            for gi, (color, dens) in enumerate(zip(colors, kdes)):
                baseline = gi * offset_step
                ax.fill_between(z_range, baseline, baseline + dens, color=color, alpha=0.6, linewidth=0.8, edgecolor=color)
                ax.plot(z_range, baseline + dens, color=color, linewidth=0.8)

            ax.set_title(feature_names[fi], fontsize=8, fontweight='bold')
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=6)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        for fi in range(n_idx, n_rows * n_cols):
            axes[fi // n_cols][fi % n_cols].axis('off')

        legend_elements = [Patch(facecolor=colors[gi], edgecolor='black', label=cond)
                           for gi, cond in enumerate(condition_names)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                   ncol=min(n_geno, 6), fontsize=10, frameon=False)
        fig.suptitle('Feature Distributions by Genotype (population z-score, ridgeline)',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'combined_feature_distributions_by_genotype.{ext}'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    @staticmethod
    def run_all(all_results, pkl_files, output_dir):
        print("\n" + "="*60)
        print("Creating combined analysis (neuron-level, across conditions)...")
        os.makedirs(output_dir, exist_ok=True)
        condition_names = [r.get('condition_name') for r in all_results]

        CombinedAnalysis.plot_luminance_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_polarity_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_contrast_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_frequency_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_segment_trace_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_condition_effect_summary(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_segment_variability_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.combine_stat_tables(all_results, condition_names, output_dir)
        segments_ref = create_stimulus_protocol(all_results[0]['mean_fps'])[2]
        CombinedAnalysis.plot_full_feature_fingerprint(all_results, condition_names, segments_ref,all_results[0]['mean_fps'], output_dir)
        CombinedAnalysis.plot_feature_distributions_by_genotype(all_results, condition_names, segments_ref,all_results[0]['mean_fps'], output_dir)
        CombinedAnalysis.plot_participation_ratio_by_genotype(all_results, condition_names, output_dir)
        plot_cross_type_umap(all_results, condition_names, segments_ref, output_dir)
        CombinedAnalysis.plot_polarity_variability_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_luminance_variability_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_frequency_variability_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_contrast_variability_comparison(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_multivariate_dispersion_by_genotype(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_dispersion_indices_by_genotype(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_polarity_distributions_by_genotype(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_luminance_distributions_by_genotype(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_frequency_distributions_by_genotype(all_results, condition_names, output_dir)
        CombinedAnalysis.plot_contrast_distributions_by_genotype(all_results, condition_names, output_dir)
        print("  Combined analysis complete.")

# def plot_functional_fingerprint(all_results, condition_names, output_dir):
#     """
#     Two-panel figure:
#       Left  — Functional fingerprint heatmap: rows = cell types,
#                columns = response indices (z-scored across types).
#                Raw values shown as text in each cell.
#       Right — Within-type vs between-type variance decomposition:
#                bar chart of variance explained by type identity for
#                each response index.
#     """
#     print("\n" + "="*60)
#     print("Building functional fingerprint ...")

#     # ── collect response indices per type ────────────────────────────────
#     # For each index we want fly-level values so we can compute within-type SD.
#     # Structure: index_name -> {type: [fly_values]}
#     index_definitions = [
#         ('ON amplitude',     lambda r: list(r['polarity_results']['fly_on_mean'].values())),
#         ('OFF amplitude',    lambda r: list(r['polarity_results']['fly_off_mean'].values())),
#         ('Luminance slope',  lambda r: list(r['luminance_results']['fly_slopes'].values())),
#         ('Freq. peak (Hz)',  _extract_freq_peak),
#         ('Contrast slope',   _extract_contrast_slope),
#         ('Within-fly CV',    lambda r: [np.mean(list(r['variability_results']['fly_cv'][s].values()))
#                                         for s in ['polarity','frequency','contrast','luminance']
#                                         if r['variability_results']['fly_cv'][s]]),
#         ('Reliability',      lambda r: [np.mean(list(r['variability_results']['fly_reliability'][s].values()))
#                                         for s in ['polarity','frequency','contrast','luminance']
#                                         if r['variability_results']['fly_reliability'][s]]),
#     ]

#     index_names  = [d[0] for d in index_definitions]
#     n_idx        = len(index_names)
#     n_types      = len(condition_names)

#     # type_means[type_idx, idx] and type_sems, plus fly-level for variance decomp
#     type_means = np.full((n_types, n_idx), np.nan)
#     type_sems  = np.full((n_types, n_idx), np.nan)
#     fly_values = {name: {cond: [] for cond in condition_names}
#                   for name in index_names}

#     for ti, (results, cond) in enumerate(zip(all_results, condition_names)):
#         for ii, (iname, ifn) in enumerate(index_definitions):
#             try:
#                 vals = ifn(results)
#                 vals = [v for v in vals if np.isfinite(v)]
#                 if len(vals) == 0:
#                     continue
#                 type_means[ti, ii] = np.mean(vals)
#                 type_sems[ti, ii]  = np.std(vals) / np.sqrt(len(vals))
#                 fly_values[iname][cond] = vals
#             except Exception:
#                 pass

#     # z-score each column for the heatmap colour scale
#     col_mean = np.nanmean(type_means, axis=0)
#     col_std  = np.nanstd(type_means,  axis=0)
#     col_std[col_std == 0] = 1.0
#     Z = (type_means - col_mean) / col_std

#     # ── within vs between variance ────────────────────────────────────────
#     variance_ratio = np.full(n_idx, np.nan)
#     for ii, iname in enumerate(index_names):
#         all_fly_vals = [v for cond in condition_names
#                         for v in fly_values[iname][cond]]
#         if len(all_fly_vals) < 2:
#             continue
#         between_var = np.nanvar(type_means[:, ii])
#         within_var  = np.nanmean([np.var(fly_values[iname][c])
#                                   for c in condition_names
#                                   if len(fly_values[iname][c]) > 1])
#         total = between_var + within_var
#         variance_ratio[ii] = between_var / total if total > 0 else 0.0

#     # ── figure ────────────────────────────────────────────────────────────
#     fig, (ax1, ax2) = plt.subplots(1, 2,
#                                     figsize=(5 + n_idx * 1.0, max(6, n_types * 0.9 + 2)),
#                                     gridspec_kw={'width_ratios': [3, 1]})

#     # Left: heatmap
#     vmax = np.nanmax(np.abs(Z))
#     im = ax1.imshow(Z, cmap='RdBu_r', aspect='auto',
#                     vmin=-vmax, vmax=vmax)

#     # Annotate cells with raw mean ± SEM
#     for ti in range(n_types):
#         for ii in range(n_idx):
#             m = type_means[ti, ii]
#             s = type_sems[ti, ii]
#             if np.isfinite(m):
#                 txt = f'{m:.2f}\n±{s:.2f}'
#                 bg  = Z[ti, ii]
#                 fc  = 'white' if abs(bg) > 0.8 * vmax else 'black'
#                 ax1.text(ii, ti, txt, ha='center', va='center',
#                          fontsize=7, color=fc, fontweight='bold')

#     ax1.set_xticks(range(n_idx))
#     ax1.set_xticklabels(index_names, rotation=40, ha='right', fontsize=9)
#     ax1.set_yticks(range(n_types))
#     ax1.set_yticklabels(condition_names, fontsize=10)
#     ax1.set_title('Functional Fingerprint\n(z-scored across types)',
#                   fontsize=12, fontweight='bold')
#     plt.colorbar(im, ax=ax1, label='z-score', shrink=0.6)

#     # Right: variance explained by type identity
#     colors_bar = ['#2ecc71' if r >= 0.5 else '#e74c3c' if r < 0.25 else '#f39c12'
#                   for r in variance_ratio]
#     bars = ax2.barh(range(n_idx), variance_ratio, color=colors_bar,
#                     edgecolor='black', linewidth=0.8, alpha=0.85)
#     ax2.axvline(0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
#                 label='50% threshold')
#     for i, v in enumerate(variance_ratio):
#         if np.isfinite(v):
#             ax2.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
#     ax2.set_yticks(range(n_idx))
#     ax2.set_yticklabels(index_names, fontsize=9)
#     ax2.set_xlabel('Variance explained\nby cell type identity', fontsize=9)
#     ax2.set_title('Between-type\nvs total variance', fontsize=11, fontweight='bold')
#     ax2.set_xlim(0, 1.15)
#     ax2.grid(True, alpha=0.3, axis='x')
#     ax2.legend(fontsize=8)

#     plt.suptitle('Cross-type functional comparison', fontsize=14,
#                  fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_functional_fingerprint.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
#     print("  Functional fingerprint saved.")


# def _extract_freq_peak(results):
#     """Return list of per-fly frequency-tuning peak (Hz)."""
#     freq_res  = results['frequency_results']
#     fly_resps = freq_res.get('fly_responses', {})
#     peaks = []
#     for fly_id, freq_dict in fly_resps.items():
#         if not freq_dict:
#             continue
#         best_freq = max(freq_dict, key=lambda f: freq_dict[f])
#         peaks.append(float(best_freq))
#     return peaks


# def _extract_contrast_slope(results):
#     """Return list of per-fly linear slope of the absolute contrast response."""
#     cont_res  = results['contrast_results']
#     fly_resps = cont_res.get('fly_responses_absolute', {})
#     slopes = []
#     for fly_id, cont_dict in fly_resps.items():
#         if len(cont_dict) < 2:
#             continue
#         xs = np.array(sorted(cont_dict.keys()), dtype=float)
#         ys = np.array([cont_dict[x] for x in xs], dtype=float)
#         if len(xs) >= 2 and np.all(np.isfinite(ys)):
#             slope, _ = np.polyfit(xs, ys, 1)
#             slopes.append(float(slope))
#     return slopes


# def plot_combined_analysis(all_results, pkl_files, output_dir):
#     """
#     Create combined plots across multiple PKL files.
    
#     Parameters:
#     -----------
#     all_results : list of dicts
#         List of result dictionaries from each PKL file
#     pkl_files : list of str
#         List of PKL file paths
#     output_dir : str
#         Directory to save plots
#     """
#     print("\n" + "="*60)
#     print("Creating combined analysis plots...")
    
#     # Extract condition names from file paths
#     condition_names = [path.split('\\')[-1].split('.')[0] for path in pkl_files]
    
#     # Define colors for each condition
#     colors = plt.cm.tab10(np.linspace(0, 1, len(pkl_files)))
    
#     # ============== LUMINANCE COMBINED PLOT ==============
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Left: Luminance response curves
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         lum_results = results['luminance_results']
#         x_vals = sorted(lum_results['overall_mean'].keys())
#         y_vals = [lum_results['overall_mean'][x] for x in x_vals]
#         y_sem = [lum_results['overall_sem'][x] for x in x_vals]
        
#         ax1.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
#                      markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
#     ax1.set_xlabel('Luminance Value', fontsize=12)
#     ax1.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
#     ax1.set_title('Luminance Response Curves', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend(fontsize=9)
    
#     # Right: Slope comparison
#     x_positions = np.arange(len(all_results))
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         lum_results = results['luminance_results']
#         mean_slope = lum_results['mean_slope']
#         sem_slope = lum_results['sem_slope']
        
#         ax2.bar(x_positions[idx], mean_slope, yerr=sem_slope, color=color,
#                 edgecolor='black', linewidth=1.5, capsize=5, width=0.7, alpha=0.8)
        
#         # Overlay individual fly data
#         fly_slopes = list(lum_results['fly_slopes'].values())
#         x_scatter = np.full(len(fly_slopes), x_positions[idx])
#         ax2.scatter(x_scatter, fly_slopes, color='black', s=40, alpha=0.5, zorder=3)
    
#     ax2.set_ylabel('Slope (ΔF/F per luminance unit)', fontsize=12)
#     ax2.set_title('Luminance Response Slopes', fontsize=14, fontweight='bold')
#     ax2.set_xticks(x_positions)
#     ax2.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=9)
#     ax2.grid(True, alpha=0.3, axis='y')
#     ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_luminance_analysis.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # ============== POLARITY COMBINED PLOT ==============
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     x_positions = np.arange(len(all_results))
#     width = 0.35
    
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         pol_results = results['polarity_results']
        
#         # ON responses
#         on_mean = pol_results['overall_on_mean']
#         on_sem = pol_results['overall_on_sem']
#         ax.bar(x_positions[idx] - width/2, on_mean, width, yerr=on_sem,
#                color=color, alpha=0.7, edgecolor='black', linewidth=1.5,
#                capsize=4, label=f'{condition} - ON' if idx == 0 else '')
        
#         # OFF responses
#         off_mean = pol_results['overall_off_mean']
#         off_sem = pol_results['overall_off_sem']
#         ax.bar(x_positions[idx] + width/2, off_mean, width, yerr=off_sem,
#                color=color, alpha=0.4, edgecolor='black', linewidth=1.5,
#                capsize=4, label=f'{condition} - OFF' if idx == 0 else '')
        
#         # Overlay individual fly data
#         fly_on = list(pol_results['fly_on_mean'].values())
#         fly_off = list(pol_results['fly_off_mean'].values())
#         ax.scatter(np.full(len(fly_on), x_positions[idx] - width/2), fly_on,
#                    color='black', s=30, alpha=0.5, zorder=3)
#         ax.scatter(np.full(len(fly_off), x_positions[idx] + width/2), fly_off,
#                    color='black', s=30, alpha=0.5, zorder=3)
    
#     # Create custom legend
#     from matplotlib.patches import Patch
#     legend_elements = []
#     for condition, color in zip(condition_names, colors):
#         legend_elements.append(Patch(facecolor=color, edgecolor='black', label=condition))
#     legend_elements.extend([
#         Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='ON'),
#         Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='OFF')
#     ])
#     ax.legend(handles=legend_elements, fontsize=9)
    
#     ax.set_ylabel('Maximum Amplitude (ΔF/F)', fontsize=12)
#     ax.set_title('Polarity Responses (ON vs OFF)', fontsize=14, fontweight='bold')
#     ax.set_xticks(x_positions)
#     ax.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=9)
#     ax.grid(True, alpha=0.3, axis='y')
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_polarity_analysis.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#      # ============== COMBINED SEGMENT TRACES ==============
#     segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
#     for seg_name in segment_names:
#         fig = plt.figure(figsize=(16, 8))
#         gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
        
#         # Get stimulus protocol from first result (should be same for all)
#         # We need to reconstruct it or pass it through - for now use placeholder
#         # Top subplot: Stimulus protocol (without axes)
#         ax_stim = fig.add_subplot(gs[0])
#         ax_stim.axis('off')
#         ax_stim.set_title(f'{seg_name.upper()} - Combined', fontsize=14, fontweight='bold', pad=10)
        
#         # Bottom subplot: Fluorescence traces
#         ax_trace = fig.add_subplot(gs[1])
        
#         # Find minimum length across all conditions for this segment
#         min_length = float('inf')
#         for results in all_results:
#             seg_data = results['segment_results'][seg_name]['global_average']
#             if seg_data is not None:
#                 min_length = min(min_length, len(seg_data[0]))
        
#         # Plot mean trace for each condition
#         for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#             seg_data = results['segment_results'][seg_name]['global_average']
#             if seg_data is not None:
#                 mean_trace = seg_data[0][:min_length]
#                 time = seg_data[1][:min_length]
                
#                 # Calculate SEM from fly averages
#                 fly_avgs = results['segment_results'][seg_name]['fly_averages']
#                 fly_traces = []
#                 for fly_id, (fly_trace, fly_time) in fly_avgs.items():
#                     if len(fly_trace) >= min_length:
#                         fly_traces.append(fly_trace[:min_length])
                
#                 if len(fly_traces) > 0:
#                     fly_traces = np.array(fly_traces)
#                     sem_trace = np.std(fly_traces, axis=0) / np.sqrt(len(fly_traces))
                    
#                     # Plot mean with SEM
#                     ax_trace.plot(time, mean_trace, color=color, linewidth=2, 
#                                  label=condition, alpha=0.8)
#                     ax_trace.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace,
#                                          color=color, alpha=0.2)
        
#         ax_trace.set_xlabel('Time (s)', fontsize=12)
#         ax_trace.set_ylabel('ΔF/F', fontsize=12)
#         ax_trace.legend(loc='upper right', fontsize=10)
#         ax_trace.grid(True, alpha=0.3)
#         ax_trace.set_xlim(time[0], time[-1])
#         ax_trace.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'combined_segment_{seg_name}.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.close()
    
#     print("  Combined segment traces saved!")

#     # ============== FREQUENCY COMBINED PLOT ==============
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Frequency response curves
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         freq_results = results['frequency_results']
#         x_vals = sorted(freq_results['overall_mean'].keys())
#         y_vals = [freq_results['overall_mean'][x] for x in x_vals]
#         y_sem = [freq_results['overall_sem'][x] for x in x_vals]
        
#         ax.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
#                      markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
#     ax.set_xlabel('Temporal Frequency (Hz)', fontsize=12)
#     ax.set_ylabel('Maximum Response (ΔF/F)', fontsize=12)
#     ax.set_title('Frequency Response Curves', fontsize=14, fontweight='bold')
#     ax.set_xscale('log')
#     ax.grid(True, alpha=0.3, which='both')
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
#     ax.legend(fontsize=9)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_frequency_analysis.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()

#     # ============== WITHIN-FLY VARIABILITY COMBINED PLOT ==============
#     segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
#     x_pos = np.arange(len(segment_names))
#     width = 0.8 / len(all_results)
    
#     # Top left: CV comparison
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         var_results = results['variability_results']
#         cv_means = [var_results['overall_cv'].get(seg, 0) for seg in segment_names]
#         cv_sems = [var_results['overall_cv_sem'].get(seg, 0) for seg in segment_names]
        
#         offset = (idx - len(all_results)/2 + 0.5) * width
#         ax1.bar(x_pos + offset, cv_means, width, yerr=cv_sems, 
#                 color=color, alpha=0.7, edgecolor='black', linewidth=1,
#                 capsize=3, label=condition)
    
#     ax1.set_ylabel('Coefficient of Variation', fontsize=12)
#     ax1.set_title('Within-Fly Variability (CV)', fontsize=14, fontweight='bold')
#     ax1.set_xticks(x_pos)
#     ax1.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
#     ax1.grid(True, alpha=0.3, axis='y')
#     ax1.legend(fontsize=9)
    
#     # Top right: Reliability comparison
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         var_results = results['variability_results']
#         rel_means = [var_results['overall_reliability'].get(seg, 0) for seg in segment_names]
#         rel_sems = [var_results['overall_reliability_sem'].get(seg, 0) for seg in segment_names]
        
#         offset = (idx - len(all_results)/2 + 0.5) * width
#         ax2.bar(x_pos + offset, rel_means, width, yerr=rel_sems,
#                 color=color, alpha=0.7, edgecolor='black', linewidth=1,
#                 capsize=3, label=condition)
    
#     ax2.set_ylabel('Mean Pairwise Correlation', fontsize=12)
#     ax2.set_title('Response Reliability', fontsize=14, fontweight='bold')
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
#     ax2.set_ylim(0, 1)
#     ax2.grid(True, alpha=0.3, axis='y')
#     ax2.legend(fontsize=9)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_within_fly_variability_bar.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
#     x_pos = np.arange(len(segment_names))
#     width = 0.8 / len(all_results)
#     # Bottom: Individual fly scatter plots
#     for seg_idx, seg_name in enumerate(segment_names):
#         # ax = ax3 if seg_idx < 2 else ax4
#         if seg_idx == 0:
#             ax=ax1
#         elif seg_idx == 1:
#             ax=ax2
#         elif seg_idx == 2:
#             ax=ax3
#         else:
#             ax=ax4
        
#         for cond_idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#             var_results = results['variability_results']
#             fly_cv_values = list(var_results['fly_cv'][seg_name].values())
#             fly_rel_values = list(var_results['fly_reliability'][seg_name].values())
            
#             if len(fly_cv_values) > 0 and len(fly_rel_values) > 0:
#                 ax.scatter(fly_cv_values, fly_rel_values, color=color, s=60, 
#                           alpha=0.6, label=condition if seg_idx == 0 else '')
        
#         ax.set_xlabel('CV', fontsize=10)
#         ax.set_ylabel('Reliability', fontsize=10)
#         ax.set_title(f'{seg_name.capitalize()}', fontsize=12, fontweight='bold')
#         ax.set_ylim(0,1)
#         ax.set_xlim(0,15)
#         ax.grid(True, alpha=0.3)
#         if seg_idx == 0:
#             ax.legend(fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_within_fly_variability_scatter.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print("  Combined variability analysis saved!")

#     # ============== ACROSS-FLY VARIABILITY COMBINED PLOT ==============
#     segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
#     x_pos = np.arange(len(segment_names))
#     width = 0.8 / len(all_results)
    
#     # Top left: CV comparison
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         across_var = results['across_fly_variability']
#         cv_values = [across_var['segment_cv'].get(seg, 0) for seg in segment_names]
        
#         offset = (idx - len(all_results)/2 + 0.5) * width
#         ax1.bar(x_pos + offset, cv_values, width,
#                 color=color, alpha=0.7, edgecolor='black', linewidth=1,
#                 label=condition)
    
#     ax1.set_ylabel('Coefficient of Variation', fontsize=12)
#     ax1.set_title('Across-Fly Variability (CV)', fontsize=14, fontweight='bold')
#     ax1.set_xticks(x_pos)
#     ax1.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
#     ax1.grid(True, alpha=0.3, axis='y')
#     ax1.legend(fontsize=9)
    
#     # Top right: Reliability comparison
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         across_var = results['across_fly_variability']
#         rel_values = [across_var['segment_reliability'].get(seg, 0) for seg in segment_names]
        
#         offset = (idx - len(all_results)/2 + 0.5) * width
#         ax2.bar(x_pos + offset, rel_values, width,
#                 color=color, alpha=0.7, edgecolor='black', linewidth=1,
#                 label=condition)
    
#     ax2.set_ylabel('Mean Pairwise Correlation', fontsize=12)
#     ax2.set_title('Across-Fly Reliability', fontsize=14, fontweight='bold')
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
#     ax2.set_ylim(0, 1)
#     ax2.grid(True, alpha=0.3, axis='y')
#     ax2.legend(fontsize=9)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_across_fly_variability_bar.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print("  Combined across-fly variability analysis saved!")

#     # ============== CONTRAST COMBINED PLOTS ==============
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
#     # Top left: Weber contrast curves
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         cont_results = results['contrast_results']
#         x_vals = sorted(cont_results['overall_mean_weber'].keys())
#         y_vals = [cont_results['overall_mean_weber'][x] for x in x_vals]
#         y_sem = [cont_results['overall_sem_weber'][x] for x in x_vals]
        
#         ax1.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
#                      markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
#     ax1.set_xlabel('Weber Contrast (%)', fontsize=12)
#     ax1.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
#     ax1.set_title('Weber Contrast Response (Signed)', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
#     ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
#     ax1.legend(fontsize=9)
    
#     # Top right: Absolute contrast curves
#     for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
#         cont_results = results['contrast_results']
#         x_vals = sorted(cont_results['overall_mean_absolute'].keys())
#         y_vals = [cont_results['overall_mean_absolute'][x] for x in x_vals]
#         y_sem = [cont_results['overall_sem_absolute'][x] for x in x_vals]
        
#         ax2.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
#                      markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
#     ax2.set_xlabel('Absolute Contrast (%)', fontsize=12)
#     ax2.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
#     ax2.set_title('Absolute Contrast Response', fontsize=14, fontweight='bold')
#     ax2.grid(True, alpha=0.3)
#     ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
#     ax2.legend(fontsize=9)
    
#     # Bottom: Individual comparison plots (can be expanded based on needs)
#     ax3.axis('off')
#     ax4.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'combined_contrast_analysis.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print("  Combined plots saved!")

#     # ============== CROSS-TYPE UMAP ==============
#     # Reconstruct segments from first result's mean_fps (same chirp for all)
#     first_fps = all_results[0].get('mean_fps', 10.0)
#     _, _, cross_segments = create_stimulus_protocol(first_fps)
#     plot_cross_type_umap(all_results, condition_names, cross_segments, output_dir)

#     # ============== FUNCTIONAL FINGERPRINT ==============
#     plot_functional_fingerprint(all_results, condition_names, output_dir)
def compute_reliability(trace_list):
    """Mean leave-one-out correlation across a neuron's repetitions. 0=noise, 1=identical."""
    n_reps = len(trace_list)
    if n_reps < 2:
        return np.nan
    min_len = min(len(t) for t in trace_list)
    reps = np.array([t[:min_len] for t in trace_list])
    correlations = []
    for r in range(n_reps):
        others_mean = np.mean(np.delete(reps, r, axis=0), axis=0)
        corr = np.corrcoef(reps[r], others_mean)[0, 1]
        if np.isfinite(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else np.nan
class PopulationSummaryPlots:
    """
    Neuron-level population summary plots — successors to the old
    fly-averaged analyze_polarity/luminance/contrast/frequency_response.
    Every point in every plot below is one neuron; fly_id is used only
    for point color, never to pre-average.
    """

    @staticmethod
    def _fly_color_map(fly_ids):
        unique_flies = sorted(set(fly_ids))
        cmap = plt.get_cmap('tab20', max(len(unique_flies), 1))
        return {fid: cmap(i % 20) for i, fid in enumerate(unique_flies)}

    # ---- per-neuron extractors -------------------------------------------

    @staticmethod
    def extract_polarity_amplitudes(trace, segments, mean_fps):
        pol_start, pol_end = segments.get('polarity', (0, 14))
        i0p = int(pol_start * mean_fps)
        i1p = min(int(pol_end * mean_fps), len(trace))
        pol_seg = trace[i0p:i1p]
        on_w_s, on_w_e = int(6 * mean_fps), min(int(8 * mean_fps), len(pol_seg))
        off_w_s, off_w_e = int(9 * mean_fps), min(int(11 * mean_fps), len(pol_seg))
        bl_s, bl_e = int(5 * mean_fps), int(6 * mean_fps)
        bl_pol = float(np.mean(pol_seg[bl_s:bl_e])) if bl_e <= len(pol_seg) else 0.0
        on_amp = (float(np.mean(pol_seg[on_w_s:on_w_e])) - bl_pol
                  if on_w_e > on_w_s and on_w_e <= len(pol_seg) else 0.0)
        off_amp = (float(np.mean(pol_seg[off_w_s:off_w_e])) - bl_pol
                   if off_w_e > off_w_s and off_w_e <= len(pol_seg) else 0.0)
        return on_amp, off_amp

    @staticmethod
    def extract_luminance_curve(trace, segments, mean_fps):
        lum_values = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                      0.4, 0.3, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
        lum_durations = [2] + [0.5] * 21   # fixed, see note above
        lum_start, lum_end = segments.get('luminance', (35, 47))
        i0l = int(lum_start * mean_fps)
        lum_seg = trace[i0l:min(int(lum_end * mean_fps), len(trace))]
        bl_frames_l = max(1, int(lum_durations[0] * mean_fps))
        bl_lum = float(np.mean(lum_seg[:bl_frames_l])) if len(lum_seg) >= bl_frames_l else 0.0
        step_means = defaultdict(list)
        cur = 0
        for lv, dur in zip(lum_values, lum_durations):
            nf = int(dur * mean_fps)
            if cur + nf <= len(lum_seg):
                step_means[lv].append(float(np.mean(lum_seg[cur:cur + nf])) - bl_lum)
            cur += nf
        return {lv: float(np.mean(v)) for lv, v in step_means.items()}

    @staticmethod
    def extract_contrast_curve(trace, segments, mean_fps):
        contrast_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                           0.4, 0.3, 0.2, 0.1, 0.0]
        contrast_durations = [2.0] + [0.5] * 15
        background = 0.5
        cont_start, cont_end = segments.get('contrast', (25, 35))
        i0c = int(cont_start * mean_fps)
        cont_seg = trace[i0c:min(int(cont_end * mean_fps), len(trace))]
        bl_frames_c = max(1, int(contrast_durations[0] * mean_fps))
        bl_cont = float(np.mean(cont_seg[:bl_frames_c])) if len(cont_seg) >= bl_frames_c else 0.0
        weber_means, abs_means = defaultdict(list), defaultdict(list)
        cur = 0
        for cv, dur in zip(contrast_values, contrast_durations):
            nf = int(dur * mean_fps)
            if cur + nf <= len(cont_seg):
                amp = float(np.mean(cont_seg[cur:cur + nf])) - bl_cont
                weber = round(((cv - background) / background) * 100, 1)
                absc = round((abs(cv - background) / background) * 100, 1)
                weber_means[weber].append(amp)
                abs_means[absc].append(amp)
            cur += nf
        return ({c: float(np.mean(v)) for c, v in weber_means.items()},
                {c: float(np.mean(v)) for c, v in abs_means.items()})

    @staticmethod
    def extract_frequency_curve(trace, segments, mean_fps):
        flash_durations = [
            0.243, 0.449, 0.406, 0.372, 0.343, 0.317, 0.296, 0.277, 0.26, 0.246,
            0.232, 0.221, 0.21, 0.2, 0.192, 0.183, 0.176, 0.169, 0.163, 0.157,
            0.151, 0.146, 0.142, 0.137, 0.133, 0.129, 0.125, 0.122, 0.118, 0.115,
            0.112, 0.11, 0.107, 0.104, 0.102, 0.099, 0.097, 0.095, 0.093, 0.091,
            0.09, 0.087, 0.086, 0.084, 0.082, 0.081, 0.079, 0.078, 0.077, 0.075,
            0.074, 0.073, 0.072, 0.07, 0.069, 0.068, 0.068, 0.066, 0.065, 0.064,
            0.063, 0.062, 0.062, 0.06, 0.06, 0.059, 0.058, 0.058, 0.056, 0.056,
            0.055, 0.055, 0.054, 0.053, 0.052, 0.052, 0.052, 0.05]
        flash_freqs = [1.0 / (2 * d) for d in flash_durations]
        freq_bin_edges = list(range(0, 11))
        freq_start, freq_end = segments.get('frequency', (14, 25))
        i0f = int(freq_start * mean_fps)
        bl_frames_f = max(1, int(0.5 * mean_fps))
        freq_baseline = (float(np.mean(trace[i0f:i0f + bl_frames_f]))
                          if i0f + bl_frames_f <= len(trace) else 0.0)
        bin_responses = defaultdict(list)
        cur_f = i0f + bl_frames_f
        for freq, dur in zip(flash_freqs, flash_durations):
            nf = max(1, int(dur * mean_fps))
            if cur_f + nf > len(trace):
                break
            seg_f = trace[cur_f:cur_f + nf]
            amp_f = float(np.max(seg_f - freq_baseline))
            for b in range(len(freq_bin_edges) - 1):
                if freq_bin_edges[b] <= freq < freq_bin_edges[b + 1]:
                    bin_responses[freq_bin_edges[b]].append(amp_f)
                    break
            cur_f += nf
        return {b: float(np.mean(v)) for b, v in bin_responses.items()}

    # ---- population plots ---------------------------------------------

    @staticmethod
    def plot_polarity_summary(condition_rois, segments, mean_fps, output_dir):
        fly_colors = PopulationSummaryPlots._fly_color_map([f for _, f, _ in condition_rois])
        on_amps, off_amps, flies = [], [], []
        for trace, fly_id, roi_key in condition_rois:
            on_amp, off_amp = PopulationSummaryPlots.extract_polarity_amplitudes(trace, segments, mean_fps)
            on_amps.append(on_amp); off_amps.append(off_amp); flies.append(fly_id)
        on_amps, off_amps = np.array(on_amps), np.array(off_amps)
        mean_on, sem_on = np.mean(on_amps), np.std(on_amps) / np.sqrt(len(on_amps))
        mean_off, sem_off = np.mean(off_amps), np.std(off_amps) / np.sqrt(len(off_amps))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar([0, 1], [mean_on, mean_off], yerr=[sem_on, sem_off],
               color=['darkgreen', 'darkred'], alpha=0.5, edgecolor='black',
               linewidth=2, capsize=8, width=0.6, zorder=1)
        for on_a, off_a, fly_id in zip(on_amps, off_amps, flies):
            ax.plot([0, 1], [on_a, off_a], '-', color=fly_colors[fly_id], alpha=0.35, linewidth=0.8, zorder=2)
        ax.scatter(np.zeros(len(on_amps)), on_amps, c=[fly_colors[f] for f in flies], s=25, alpha=0.7, zorder=3, edgecolor='none')
        ax.scatter(np.ones(len(off_amps)), off_amps, c=[fly_colors[f] for f in flies], s=25, alpha=0.7, zorder=3, edgecolor='none')
        ax.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
        ax.set_title(f'Polarity Response (ON vs OFF) | n={len(on_amps)} neurons, {len(fly_colors)} flies',
                     fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['ON', 'OFF'], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'7_polarity_summary.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
        return {'on_amplitudes': on_amps, 'off_amplitudes': off_amps, 'fly_ids': flies,
                'mean_on': mean_on, 'sem_on': sem_on, 'mean_off': mean_off, 'sem_off': sem_off}

    @staticmethod
    def plot_luminance_summary(condition_rois, segments, mean_fps, output_dir):
        fly_colors = PopulationSummaryPlots._fly_color_map([f for _, f, _ in condition_rois])
        curves, slopes, flies = [], [], []
        for trace, fly_id, roi_key in condition_rois:
            curve = PopulationSummaryPlots.extract_luminance_curve(trace, segments, mean_fps)
            if len(curve) < 2:
                continue
            curves.append(curve)
            x = np.array(list(curve.keys())); y = np.array(list(curve.values()))
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(float(slope)); flies.append(fly_id)

        unique_lum = sorted(set(lv for c in curves for lv in c))
        overall_mean = {lv: np.mean([c[lv] for c in curves if lv in c]) for lv in unique_lum}
        overall_sem = {lv: np.std([c[lv] for c in curves if lv in c]) / np.sqrt(
                        len([c[lv] for c in curves if lv in c])) for lv in unique_lum}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for c in curves:
            xs = sorted(c.keys())
            ax1.plot(xs, [c[x] for x in xs], color='gray', alpha=0.12, linewidth=0.7)
        x_vals = sorted(overall_mean.keys())
        ax1.errorbar(x_vals, [overall_mean[x] for x in x_vals], yerr=[overall_sem[x] for x in x_vals],
                     fmt='o-', color='black', markersize=7, capsize=4, linewidth=2,
                     label=f'Mean ± SEM (n={len(curves)} neurons)')
        ax1.set_xlabel('Luminance Value', fontsize=12); ax1.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
        ax1.set_title('Luminance Response Curve (neuron-level)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.legend()

        mean_slope, sem_slope = np.mean(slopes), np.std(slopes) / np.sqrt(len(slopes))
        ax2.bar(0, mean_slope, yerr=sem_slope, color='lightblue', edgecolor='black',
                linewidth=2, capsize=5, width=0.5, zorder=1)
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(slopes))
        ax2.scatter(jitter, slopes, c=[fly_colors[f] for f in flies], s=20, alpha=0.7, zorder=2, edgecolor='none')
        ax2.set_ylabel('Slope (ΔF/F per luminance unit)', fontsize=12)
        ax2.set_title(f'Luminance Slope | n={len(slopes)} neurons, {len(fly_colors)} flies', fontsize=14, fontweight='bold')
        ax2.set_xticks([0]); ax2.set_xticklabels(['Slope'])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'6_luminance_summary.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
        return {'per_neuron_curves': curves, 'per_neuron_slopes': slopes, 'fly_ids': flies,
                'overall_mean': overall_mean, 'overall_sem': overall_sem,
                'mean_slope': mean_slope, 'sem_slope': sem_slope}
    @staticmethod
    def compute_segment_variability_per_roi(roi_accumulator, qi_pass, segments, mean_fps):
        """
        Per QC-passing neuron, per segment: CV and reliability across that
        neuron's OWN repetitions (not across neurons) -- the ROI-level
        analog of the old analyze_within_fly_variability. Same formulas:
        CV = pointwise std/|mean| across reps at each timepoint, averaged
        across time (non-finite treated as 0, matching the original);
        reliability = mean pairwise correlation among segment-restricted
        reps (not leave-one-out -- kept this way specifically to match
        the original figure).
        """
        seg_cv = {s: [] for s in segments}
        seg_reliability = {s: [] for s in segments}
        for fly_id, roi_dict in roi_accumulator.items():
            for roi_idx, trace_list in roi_dict.items():
                if not qi_pass.get((fly_id, roi_idx), False) or len(trace_list) < 2:
                    continue
                min_len = min(len(t) for t in trace_list)
                reps = np.array([t[:min_len] for t in trace_list])
                for seg_name, (s_start, s_end) in segments.items():
                    i0 = int(s_start * mean_fps); i1 = min(int(s_end * mean_fps), min_len)
                    if i1 <= i0:
                        continue
                    seg_reps = reps[:, i0:i1]
                    mean_trace = seg_reps.mean(axis=0); std_trace = seg_reps.std(axis=0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cv_trace = std_trace / np.abs(mean_trace)
                        cv_trace[~np.isfinite(cv_trace)] = 0
                    seg_cv[seg_name].append(float(np.mean(cv_trace)))
                    n_reps = seg_reps.shape[0]; correlations = []
                    for i in range(n_reps):
                        for j in range(i + 1, n_reps):
                            corr = np.corrcoef(seg_reps[i], seg_reps[j])[0, 1]
                            if np.isfinite(corr):
                                correlations.append(corr)
                    seg_reliability[seg_name].append(float(np.mean(correlations)) if correlations else 0.0)
        return {'cv': seg_cv, 'reliability': seg_reliability}
    @staticmethod
    def plot_contrast_summary(condition_rois, segments, mean_fps, output_dir):
        weber_curves, abs_curves, flies = [], [], []
        for trace, fly_id, roi_key in condition_rois:
            wc, ac = PopulationSummaryPlots.extract_contrast_curve(trace, segments, mean_fps)
            if len(wc) < 2:
                continue
            weber_curves.append(wc); abs_curves.append(ac); flies.append(fly_id)

        def _agg(curves):
            keys = sorted(set(k for c in curves for k in c))
            m = {k: np.mean([c[k] for c in curves if k in c]) for k in keys}
            s = {k: np.std([c[k] for c in curves if k in c]) / np.sqrt(
                 len([c[k] for c in curves if k in c])) for k in keys}
            return m, s

        weber_mean, weber_sem = _agg(weber_curves)
        abs_mean, abs_sem = _agg(abs_curves)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for c in weber_curves:
            xs = sorted(c.keys()); ax1.plot(xs, [c[x] for x in xs], color='gray', alpha=0.12, linewidth=0.7)
        x1 = sorted(weber_mean.keys())
        ax1.errorbar(x1, [weber_mean[x] for x in x1], yerr=[weber_sem[x] for x in x1],
                    fmt='o-', color='black', markersize=7, capsize=4, linewidth=2, label=f'n={len(weber_curves)} neurons')
        ax1.set_xlabel('Weber Contrast (%)', fontsize=12); ax1.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
        ax1.set_title('Weber Contrast Response — Signed (neuron-level)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.legend()

        for c in abs_curves:
            xs = sorted(c.keys()); ax2.plot(xs, [c[x] for x in xs], color='lightblue', alpha=0.15, linewidth=0.7)
        x2 = sorted(abs_mean.keys())
        ax2.errorbar(x2, [abs_mean[x] for x in x2], yerr=[abs_sem[x] for x in x2],
                    fmt='o-', color='darkblue', markersize=7, capsize=4, linewidth=2, label=f'n={len(abs_curves)} neurons')
        ax2.set_xlabel('Absolute Contrast (%)', fontsize=12); ax2.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
        ax2.set_title('Absolute Contrast Response (neuron-level)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.legend()
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'8_contrast_summary.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
        return {'weber_curves': weber_curves, 'abs_curves': abs_curves, 'fly_ids': flies,
                'weber_mean': weber_mean, 'weber_sem': weber_sem, 'abs_mean': abs_mean, 'abs_sem': abs_sem}

    @staticmethod
    def plot_frequency_summary(condition_rois, segments, mean_fps, output_dir):
        curves, flies = [], []
        for trace, fly_id, roi_key in condition_rois:
            c = PopulationSummaryPlots.extract_frequency_curve(trace, segments, mean_fps)
            if len(c) >= 2:
                curves.append(c); flies.append(fly_id)
        keys = sorted(set(k for c in curves for k in c))
        mean_ = {k: np.mean([c[k] for c in curves if k in c]) for k in keys}
        sem_ = {k: np.std([c[k] for c in curves if k in c]) / np.sqrt(
                len([c[k] for c in curves if k in c])) for k in keys}

        fig, ax = plt.subplots(figsize=(12, 6))
        for c in curves:
            xs = sorted(c.keys()); ax.plot(xs, [c[x] for x in xs], color='gray', alpha=0.12, linewidth=0.7)
        ax.errorbar(keys, [mean_[k] for k in keys], yerr=[sem_[k] for k in keys],
                   fmt='o-', color='black', markersize=7, capsize=4, linewidth=2, label=f'n={len(curves)} neurons')
        ax.set_xlabel('Temporal Frequency (Hz)', fontsize=12); ax.set_ylabel('Maximum Response (ΔF/F)', fontsize=12)
        ax.set_title('Frequency Response Curve (neuron-level)', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, 10.5); ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(output_dir, f'9_frequency_summary.{ext}'), dpi=300, bbox_inches='tight')
        plt.close()
        return {'curves': curves, 'fly_ids': flies, 'overall_mean': mean_, 'overall_sem': sem_}

    @staticmethod
    def plot_feature_overview(condition_rois, segments, mean_fps, output_dir):
        """
        SD and CV bar panels (left), computed on raw, un-normalized, per-feature
        values -- each feature's own mean/SD, independent of every other
        feature. Main panel (right): each neuron's per-feature value, scaled
        independently per feature to [0, 1] using that feature's own observed
        min/max across neurons in this condition -- not z-scored, and not
        referenced against any other feature. With dozens-to-hundreds of
        neurons this is well-posed (unlike min-max across only 2-3 genotype
        means, which degenerates).
        """
        fly_colors = PopulationSummaryPlots._fly_color_map([f for _, f, _ in condition_rois])
        rows, flies, feature_names = [], [], None
        for trace, fly_id, roi_key in condition_rois:
            feat, names = extract_roi_features(trace, None, segments, mean_fps)
            if feature_names is None:
                feature_names = names
            rows.append(feat); flies.append(fly_id)

        X = np.vstack(rows)
        n_feat = X.shape[1]

        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        X_scaled = (X - col_min) / col_range   # per-feature, independent min-max

        sd_vals = X.std(axis=0)
        mean_vals = X.mean(axis=0)
        cv_vals = np.array([sd / abs(m) if abs(m) > 1e-8 else np.nan
                            for sd, m in zip(sd_vals, mean_vals)])

        fig, (ax_sd, ax_cv, ax_main) = plt.subplots(
            1, 3, figsize=(16, max(6, 0.4 * n_feat)),
            gridspec_kw={'width_ratios': [1, 1, 3]}, sharey=True)
        y_pos = np.arange(n_feat)

        sd_display = np.maximum(sd_vals, 1e-6)
        ax_sd.barh(y_pos, sd_display, color='steelblue', edgecolor='black', alpha=0.7)
        ax_sd.set_xscale('log')
        ax_sd.set_xlabel('SD (log scale)', fontsize=10)
        ax_sd.set_title('Std. Dev.', fontsize=11, fontweight='bold')
        ax_sd.grid(True, alpha=0.3, axis='x', which='both')

        cv_display = np.where(np.isfinite(cv_vals) & (cv_vals > 0), cv_vals, 1e-6)
        ax_cv.barh(y_pos, cv_display, color='indianred', edgecolor='black', alpha=0.7)
        ax_cv.set_xscale('log')
        for i, cv in enumerate(cv_vals):
            if not np.isfinite(cv):
                ax_cv.text(1e-6, i, ' undef.', va='center', ha='left', fontsize=7, color='gray', style='italic')
        ax_cv.set_xlabel('CV (log scale)', fontsize=10)
        ax_cv.set_title('Coefficient of Variation', fontsize=11, fontweight='bold')
        ax_cv.grid(True, alpha=0.3, axis='x', which='both')

        rng = np.random.default_rng(0)
        for i, name in enumerate(feature_names):
            vals = X_scaled[:, i]
            jitter = rng.normal(0, 0.12, size=len(vals))
            ax_main.scatter(vals, np.full(len(vals), i) + jitter,
                            c=[fly_colors[f] for f in flies], s=8, alpha=0.5, edgecolor='none')
            ax_main.boxplot(vals, positions=[i], vert=False, widths=0.6, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.2),
                            medianprops=dict(color='black', linewidth=1.5))
        ax_main.set_xlim(-0.05, 1.05)
        ax_main.set_xlabel("Min-max scaled feature value (0=feature's min, 1=feature's max)", fontsize=10)
        ax_main.set_title('Per-neuron distribution', fontsize=11, fontweight='bold')
        ax_main.grid(True, alpha=0.3, axis='x')

        ax_sd.set_yticks(y_pos); ax_sd.set_yticklabels(feature_names, fontsize=9)
        ax_sd.invert_yaxis()

        fig.suptitle(f'Per-neuron feature overview | n={len(rows)} neurons, {len(fly_colors)} flies',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '12_feature_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return {'feature_matrix': X, 'feature_names': feature_names, 'fly_ids': flies,
                'sd': sd_vals, 'cv': cv_vals}
    @staticmethod
    def run_all(condition_rois, segments, mean_fps, output_dir, stim_protocol=None, stim_time=None,
                roi_accumulator=None, qi_pass=None):
        print("\n" + "="*60)
        print("Population summary plots (neuron-level) ...")
        results = {
            'polarity': PopulationSummaryPlots.plot_polarity_summary(condition_rois, segments, mean_fps, output_dir),
            'luminance': PopulationSummaryPlots.plot_luminance_summary(condition_rois, segments, mean_fps, output_dir),
            'contrast': PopulationSummaryPlots.plot_contrast_summary(condition_rois, segments, mean_fps, output_dir),
            'frequency': PopulationSummaryPlots.plot_frequency_summary(condition_rois, segments, mean_fps, output_dir),
            'feature_overview': PopulationSummaryPlots.plot_feature_overview(condition_rois, segments, mean_fps, output_dir),
        }
        PopulationSummaryPlots.plot_response_traces_stacked(condition_rois, mean_fps, output_dir,
                                                              stim_protocol=stim_protocol, stim_time=stim_time)
        if roi_accumulator is not None and qi_pass is not None:
            results['segment_variability'] = PopulationSummaryPlots.compute_segment_variability_per_roi(
                roi_accumulator, qi_pass, segments, mean_fps)
        print("  Population summary plots complete.")
        return results
    
    # @staticmethod
    # def plot_response_heatmap(condition_rois, mean_fps, output_dir,
    #                           stim_protocol=None, stim_time=None,
    #                           sort_by='fly', cmap='RdBu_r', vmax_percentile=99):
    #     """
    #     One figure per condition: every QC-passing neuron's mean trace as
    #     one heatmap row (color = ΔF/F), all rows sharing the same color
    #     scale so amplitude is comparable across neurons -- not per-row
    #     normalized. Style after Baden et al. 2016 Fig. 4B: stimulus strip
    #     on top, no axes on the trace panel itself.

    #     sort_by: 'fly' (default) groups each fly's neurons together, so
    #     fly-driven blocks are visible at a glance; None keeps condition_rois
    #     order as given.
    #     """
    #     save_path = os.path.join(output_dir, '13_response_heatmap.png')
    #     if os.path.exists(save_path):
    #         print("  Response heatmap already exists, skipping.")
    #         return save_path

    #     rois = list(condition_rois)
    #     if sort_by == 'fly':
    #         def _sort_key(r):
    #             _, fly_id, roi_key = r
    #             roi_num = int(''.join(filter(str.isdigit, str(roi_key))) or 0)
    #             return (str(fly_id), roi_num)
    #         rois = sorted(rois, key=_sort_key)

    #     min_len = min(len(tr) for tr, _, _ in rois)
    #     matrix = np.array([tr[:min_len] for tr, _, _ in rois])
    #     n_neurons = matrix.shape[0]

    #     # Shared, symmetric, outlier-robust color scale across ALL rows --
    #     # this is what "same axes lim" actually requires: without it,
    #     # imshow would auto-scale per call and rows wouldn't be comparable.
    #     vmax = np.percentile(np.abs(matrix), vmax_percentile)
    #     vmax = vmax if vmax > 0 else 1.0

    #     fig_height = min(24, max(4, n_neurons * 0.045))
    #     if stim_protocol is not None:
    #         fig = plt.figure(figsize=(12, fig_height + 1))
    #         gs = fig.add_gridspec(2, 1, height_ratios=[1, fig_height], hspace=0.02)
    #         ax_stim = fig.add_subplot(gs[0])
    #         stim_len = min(len(stim_protocol), min_len)
    #         ax_stim.fill_between(stim_time[:stim_len], 0, stim_protocol[:stim_len],
    #                              color='lightblue', alpha=0.7, linewidth=0)
    #         ax_stim.plot(stim_time[:stim_len], stim_protocol[:stim_len], color='blue', linewidth=1)
    #         ax_stim.set_xlim(0, stim_time[:stim_len][-1])
    #         ax_stim.axis('off')
    #         ax_heat = fig.add_subplot(gs[1])
    #     else:
    #         fig, ax_heat = plt.subplots(figsize=(12, fig_height))

    #     im = ax_heat.imshow(matrix, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
    #                         interpolation='nearest')
    #     ax_heat.axis('off')

    #     n_flies = len(set(r[1] for r in rois))
    #     fig.suptitle(f'All ROI mean traces (n={n_neurons} neurons, {n_flies} flies)',
    #                  fontsize=13, fontweight='bold', y=0.995 if stim_protocol is not None else 1.02)
    #     cbar = fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)
    #     cbar.set_label('ΔF/F', fontsize=10)
    #     cbar.ax.tick_params(labelsize=8)

    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     return save_path
    @staticmethod
    def plot_response_traces_stacked(condition_rois, mean_fps, output_dir,
                                      stim_protocol=None, stim_time=None,
                                      sort_by='fly', spacing_percentile=50,
                                      height_per_row=0.20, color='black', linewidth=0.6):
        """
        One figure per condition: every QC-passing neuron's mean trace drawn
        as an actual line (ΔF/F on y), stacked vertically with ONE FIXED
        offset per row shared across every ROI -- not per-row normalized,
        so a strong responder visibly overflows into its neighbours' space
        rather than being scaled down to fit. No axes.

        spacing_percentile: the row offset is fixed at this percentile of
        each trace's (4 x SD) typical excursion across the population --
        raise it for less overlap (more blank space, smaller/typical
        neurons harder to see), lower it for a denser, more overlapping
        stack.
        """
        save_path = os.path.join(output_dir, '13_response_traces_stacked.png')
        if os.path.exists(save_path):
            print("  Stacked response traces already exist, skipping.")
            return save_path

        rois = list(condition_rois)
        if sort_by == 'fly':
            def _sort_key(r):
                _, fly_id, roi_key = r
                roi_num = int(''.join(filter(str.isdigit, str(roi_key))) or 0)
                return (str(fly_id), roi_num)
            rois = sorted(rois, key=_sort_key)

        min_len = min(len(tr) for tr, _, _ in rois)
        matrix = np.array([tr[:min_len] for tr, _, _ in rois])
        n_neurons = matrix.shape[0]
        time = np.arange(min_len) / mean_fps

        # Fixed, shared row spacing -- same basis for every ROI. 4xSD is a
        # robust stand-in for "typical excursion" (less outlier-sensitive
        # than raw max-min range).
        typical_excursion = matrix.std(axis=1) * 4
        spacing = np.percentile(typical_excursion, spacing_percentile)
        if spacing <= 0:
            spacing = 1.0
        print(f"  Stacked traces: row spacing = {spacing:.3f} (dF/F units)")

        fig_height = max(4, n_neurons * height_per_row)
        if stim_protocol is not None:
            fig = plt.figure(figsize=(12, fig_height + 1))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, fig_height], hspace=0.01)
            ax_stim = fig.add_subplot(gs[0])
            stim_len = min(len(stim_protocol), min_len)
            ax_stim.fill_between(stim_time[:stim_len], 0, stim_protocol[:stim_len],
                                 color='lightblue', alpha=0.7, linewidth=0)
            ax_stim.plot(stim_time[:stim_len], stim_protocol[:stim_len], color='blue', linewidth=1)
            ax_stim.set_xlim(time[0], time[-1])
            ax_stim.axis('off')
            ax = fig.add_subplot(gs[1])
        else:
            fig, ax = plt.subplots(figsize=(12, fig_height))

        for i in range(n_neurons):
            offset = (n_neurons - 1 - i) * spacing   # first ROI at the top
            ax.plot(time, matrix[i] + offset, color=color, linewidth=linewidth, alpha=0.85)

        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-spacing, n_neurons * spacing)
        ax.axis('off')

        n_flies = len(set(r[1] for r in rois))
        fig.suptitle(f'All ROI mean traces, stacked (n={n_neurons} neurons, {n_flies} flies)',
                     fontsize=13, fontweight='bold', y=0.995 if stim_protocol is not None else 1.02)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

class FlyEffectStats:
    """
    Quantifies how much of neuron-level variability is explained by fly
    identity, swept across every feature_set analyze_response_clustering
    already fit. Reuses that function's GMM output rather than re-fitting.
    """

    @staticmethod
    def compute_icc(values, fly_labels):
        """
        ICC(1) via an unconditional random-intercept mixed model:
            value ~ 1 + (1 | fly_id)
        ICC = var_fly / (var_fly + var_residual).
        """
        df = pd.DataFrame({'value': values, 'fly_id': fly_labels})
        fly_counts = df['fly_id'].value_counts()
        if df['fly_id'].nunique() < 2 or (fly_counts < 2).all():
            return {'icc': np.nan, 'var_fly': np.nan, 'var_resid': np.nan, 'converged': False}
        try:
            model = smf.mixedlm('value ~ 1', df, groups=df['fly_id'])
            result = model.fit(reml=True)
            var_fly = max(float(result.cov_re.iloc[0, 0]), 0.0)
            var_resid = float(result.scale)
            denom = var_fly + var_resid
            icc = var_fly / denom if denom > 0 else np.nan
            return {'icc': icc, 'var_fly': var_fly, 'var_resid': var_resid,
                    'converged': bool(result.converged)}
        except Exception as e:
            return {'icc': np.nan, 'var_fly': np.nan, 'var_resid': np.nan,
                    'converged': False, 'error': str(e)}

    @staticmethod
    def run_permanova(X, fly_labels, n_permutations=999, seed=0):
        """
        PERMANOVA (McArdle & Anderson 2001): partitions total squared
        Euclidean distance into between-fly / within-fly, tests the
        pseudo-F against a null built by permuting fly labels.
        """
        fly_labels = np.asarray(fly_labels)
        n = len(fly_labels)
        D = squareform(pdist(X, metric='euclidean')) ** 2

        def pseudo_F(labels):
            groups = pd.unique(labels)
            ss_total = D.sum() / (2 * n)
            ss_within = 0.0
            for g in groups:
                idx = np.where(labels == g)[0]
                ng = len(idx)
                if ng < 2:
                    continue
                ss_within += D[np.ix_(idx, idx)].sum() / (2 * ng)
            ss_between = ss_total - ss_within
            a = len(groups)
            df_between, df_within = a - 1, n - a
            if df_between <= 0 or df_within <= 0 or ss_within <= 0:
                return np.nan
            return (ss_between / df_between) / (ss_within / df_within)

        F_obs = pseudo_F(fly_labels)
        rng = np.random.default_rng(seed)
        F_perm = np.array([pseudo_F(rng.permutation(fly_labels)) for _ in range(n_permutations)])
        valid = np.isfinite(F_perm)
        if not np.isfinite(F_obs) or valid.sum() == 0:
            p_value = np.nan
        else:
            p_value = (np.sum(F_perm[valid] >= F_obs) + 1) / (valid.sum() + 1)
        return {'pseudo_F': F_obs, 'p_value': p_value, 'n_permutations': int(valid.sum())}

    @staticmethod
    def cluster_fly_association(gmm_labels, fly_labels, n_permutations=999, seed=0):
        """Cramer's V (+ permuted-null p-value) and ARI between cluster label and fly identity."""
        gmm_labels = np.asarray(gmm_labels)
        fly_labels = np.asarray(fly_labels)
        contingency = pd.crosstab(pd.Series(fly_labels, name='fly'),
                                   pd.Series(gmm_labels, name='cluster'))

        def cramers_v(labels_a, labels_b):
            ct = pd.crosstab(labels_a, labels_b)
            chi2 = stats.chi2_contingency(ct, correction=False)[0]
            n = ct.values.sum()
            r, k = ct.shape
            denom = min(r - 1, k - 1)
            return np.sqrt(chi2 / (n * denom)) if denom > 0 and n > 0 else np.nan

        v_obs = cramers_v(fly_labels, gmm_labels)
        rng = np.random.default_rng(seed)
        v_perm = np.array([cramers_v(rng.permutation(fly_labels), gmm_labels)
                            for _ in range(n_permutations)])
        valid = np.isfinite(v_perm)
        p_value = ((np.sum(v_perm[valid] >= v_obs) + 1) / (valid.sum() + 1)
                   if np.isfinite(v_obs) and valid.sum() > 0 else np.nan)
        ari = adjusted_rand_score(fly_labels, gmm_labels)
        return {'contingency': contingency, 'cramers_v': v_obs, 'p_value': p_value,
                'adjusted_rand_index': ari, 'n_permutations': int(valid.sum())}

    @staticmethod
    def plot_cluster_fly_heatmap(contingency, title, save_path):
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * contingency.shape[1] + 2),
                                         max(5, 0.3 * contingency.shape[0] + 2)))
        row_sums = contingency.sum(axis=1).values[:, None]
        row_sums[row_sums == 0] = 1
        normed = contingency.values / row_sums
        im = ax.imshow(normed, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(contingency.shape[1])); ax.set_xticklabels(contingency.columns, fontsize=9)
        ax.set_yticks(range(contingency.shape[0])); ax.set_yticklabels(contingency.index, fontsize=8)
        ax.set_xlabel('Cluster', fontsize=11); ax.set_ylabel('Fly ID', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                n_val = contingency.values[i, j]
                if n_val > 0:
                    ax.text(j, i, str(int(n_val)), ha='center', va='center',
                            color='white' if normed[i, j] > 0.5 else 'black', fontsize=7)
        fig.colorbar(im, ax=ax, label="Fraction of fly's neurons")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def run_sweep(clustering_results, output_dir, n_permutations=999, pc_threshold=30, n_pc=8):
        """
        Runs ICC + PERMANOVA + cluster-fly association for every feature_set
        already present in clustering_results (i.e. every fset that survived
        analyze_response_clustering's >=10-neuron minimum).

        ICC is computed per column of X when X has <= pc_threshold columns
        (true for 'features'/'features_X' [5-26 named features] and 'spca'
        [20 components] — genuinely interpretable). For 'full'/'segment_X'
        (hundreds of raw timepoints), ICC runs on the top n_pc PCs instead,
        so every feature_set gets a comparably-sized, readable ICC table.
        """
        icc_path = os.path.join(output_dir, 'icc_table.csv')
        if os.path.exists(icc_path):
            print("Fly-effect statistics sweep: found existing tables, loading instead of recomputing.")
            return {
                'icc_table': pd.read_csv(icc_path),
                'permanova_table': pd.read_csv(os.path.join(output_dir, 'permanova_table.csv')),
                'cluster_fly_table': pd.read_csv(os.path.join(output_dir, 'cluster_fly_table.csv')),
            }
        print("\n" + "="*60)
        print("Fly-effect statistics sweep (ICC / PERMANOVA / cluster-fly) ...")
        icc_rows, permanova_rows, cluster_fly_rows = [], [], []

        for fset, res in clustering_results.items():
            X, fly_labels, gmm_labels = res['X'], res['fly_labels'], res['gmm_labels']
            n_flies = len(set(fly_labels))
            print(f"  Feature set: {fset}  (n={len(X)} neurons, {n_flies} flies)")
            if n_flies < 2:
                print(f"    Skipping: only {n_flies} fly, can't assess a fly effect")
                continue

            col_source = X
            col_names = [f'{fset}_dim{i}' for i in range(X.shape[1])]
            if X.shape[1] > pc_threshold:
                n_pc_use = min(n_pc, X.shape[1], len(X) - 1)
                pca = PCA(n_components=n_pc_use, random_state=42)
                col_source = pca.fit_transform(X)
                col_names = [f'{fset}_PC{i+1}' for i in range(n_pc_use)]

            for j, name in enumerate(col_names):
                icc_res = FlyEffectStats.compute_icc(col_source[:, j], fly_labels)
                icc_rows.append({'feature_set': fset, 'column': name, **icc_res})

            perm_res = FlyEffectStats.run_permanova(X, fly_labels, n_permutations=n_permutations)
            permanova_rows.append({'feature_set': fset, **perm_res})
            print(f"    PERMANOVA: pseudo-F={perm_res['pseudo_F']:.3f}, p={perm_res['p_value']:.4f}")

            cf_res = FlyEffectStats.cluster_fly_association(gmm_labels, fly_labels, n_permutations=n_permutations)
            cluster_fly_rows.append({'feature_set': fset, 'cramers_v': cf_res['cramers_v'],
                                      'p_value': cf_res['p_value'], 'ari': cf_res['adjusted_rand_index'],
                                      'n_clusters': len(set(gmm_labels))})
            print(f"    Cluster x fly: V={cf_res['cramers_v']:.3f}, p={cf_res['p_value']:.4f}, "
                  f"ARI={cf_res['adjusted_rand_index']:.3f}")

            v_str = cf_res['cramers_v']
            for ext in ['pdf', 'png']:
                FlyEffectStats.plot_cluster_fly_heatmap(
                    cf_res['contingency'],
                    f"{fset} | Cluster vs Fly | V={v_str:.2f}",
                    os.path.join(output_dir, f'clusterfly_heatmap_{fset}.{ext}'))

        icc_table = pd.DataFrame(icc_rows)
        permanova_table = pd.DataFrame(permanova_rows)
        cluster_fly_table = pd.DataFrame(cluster_fly_rows)
        icc_table.to_csv(os.path.join(output_dir, 'icc_table.csv'), index=False)
        permanova_table.to_csv(os.path.join(output_dir, 'permanova_table.csv'), index=False)
        cluster_fly_table.to_csv(os.path.join(output_dir, 'cluster_fly_table.csv'), index=False)

        print("\n  Fly-effect statistics sweep complete.")
        return {'icc_table': icc_table, 'permanova_table': permanova_table,
                'cluster_fly_table': cluster_fly_table}

class VariabilityStats:
    """
    Population-level variability statistics, swept across every
    feature_set analyze_response_clustering already fit: dispersion
    indices (SD/CV/MAD/IQR + multivariate + PERMDISP), participation
    ratio, per-neuron silhouette + GMM posterior entropy, Levene's/
    Bartlett's tests, and a generic fly-nested bootstrap engine.
    """

    @staticmethod
    def dispersion_indices(X, column_names):
        """SD, CV, MAD, IQR across neurons, per column of X."""
        rows = []
        for j, name in enumerate(column_names):
            col = X[:, j]
            mean_ = col.mean(); sd = col.std()
            cv = sd / abs(mean_) if abs(mean_) > 1e-8 else np.nan
            mad = np.median(np.abs(col - np.median(col)))
            iqr = np.percentile(col, 75) - np.percentile(col, 25)
            rows.append({'column': name, 'mean': mean_, 'sd': sd, 'cv': cv, 'mad': mad, 'iqr': iqr})
        return pd.DataFrame(rows)

    @staticmethod
    def compute_multivariate_dispersion(X, fly_labels, n_permutations=999, seed=0):
        """
        Mean/median pairwise distance (overall spread), plus PERMDISP:
        each neuron's distance to its OWN fly's centroid, tested for a
        fly effect via a permutation one-way ANOVA on those distances
        (Anderson 2006) — is one fly's population more heterogeneous
        than another's, not just differently located.
        """
        fly_labels = np.asarray(fly_labels)
        unique_flies = pd.unique(fly_labels)
        disp = np.zeros(len(X))
        for g in unique_flies:
            idx = np.where(fly_labels == g)[0]
            centroid = X[idx].mean(axis=0)
            disp[idx] = np.linalg.norm(X[idx] - centroid, axis=1)

        def anova_F(labels, values):
            groups = pd.unique(labels)
            grand_mean = values.mean()
            ss_between = sum(len(values[labels == g]) * (values[labels == g].mean() - grand_mean) ** 2
                              for g in groups)
            ss_within = sum(((values[labels == g] - values[labels == g].mean()) ** 2).sum() for g in groups)
            df_between, df_within = len(groups) - 1, len(values) - len(groups)
            if df_between <= 0 or df_within <= 0 or ss_within <= 0:
                return np.nan
            return (ss_between / df_between) / (ss_within / df_within)

        F_obs = anova_F(fly_labels, disp)
        rng = np.random.default_rng(seed)
        F_perm = np.array([anova_F(rng.permutation(fly_labels), disp) for _ in range(n_permutations)])
        valid = np.isfinite(F_perm)
        p_value = ((np.sum(F_perm[valid] >= F_obs) + 1) / (valid.sum() + 1)
                   if np.isfinite(F_obs) and valid.sum() > 0 else np.nan)
        pdists = pdist(X, metric='euclidean')
        return {'mean_pairwise_distance': float(pdists.mean()),
                'median_pairwise_distance': float(np.median(pdists)),
                'mean_dispersion_to_fly_centroid': float(disp.mean()),
                'permdisp_F': F_obs, 'permdisp_p_value': p_value}

    @staticmethod
    def participation_ratio(X):
        """Effective dimensionality: PR = (sum eigenvalues)^2 / sum(eigenvalues^2)."""
        Xc = X - X.mean(axis=0)
        cov = np.cov(Xc, rowvar=False)
        eigvals = np.clip(np.linalg.eigvalsh(np.atleast_2d(cov)), 0, None)
        s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
        pr = (s1 ** 2) / s2 if s2 > 0 else np.nan
        return {'participation_ratio': float(pr), 'n_dims': X.shape[1], 'n_samples': len(X)}

    @staticmethod
    def neuron_ambiguity_scores(X, gmm_labels, posteriors):
        """Silhouette (hard-assignment ambiguity) + GMM posterior entropy (soft-assignment ambiguity)."""
        n_clusters = len(set(gmm_labels))
        if n_clusters < 2 or len(X) <= n_clusters:
            sil = np.full(len(X), np.nan)
        else:
            sil = silhouette_samples(X, gmm_labels, metric='euclidean')
        eps = 1e-12
        entropy = -np.sum(posteriors * np.log(posteriors + eps), axis=1)
        max_entropy = np.log(posteriors.shape[1]) if posteriors.shape[1] > 0 else np.nan
        norm_entropy = entropy / max_entropy if max_entropy and max_entropy > 0 else entropy
        return {'silhouette': sil, 'posterior_entropy': entropy,
                'posterior_entropy_normalized': norm_entropy, 'max_posterior': posteriors.max(axis=1)}

    @staticmethod
    def levene_bartlett_by_group(X, group_labels, column_names, min_group_size=3):
        """Per column: does VARIANCE differ across groups? Levene's (robust) + Bartlett's (classic)."""
        group_labels = np.asarray(group_labels)
        unique_groups = [g for g in pd.unique(group_labels) if np.sum(group_labels == g) >= min_group_size]
        rows = []
        for j, name in enumerate(column_names):
            groups_data = [X[group_labels == g, j] for g in unique_groups]
            if len(groups_data) < 2:
                rows.append({'column': name, 'levene_stat': np.nan, 'levene_p': np.nan,
                            'bartlett_stat': np.nan, 'bartlett_p': np.nan, 'n_groups': len(groups_data)})
                continue
            lev_stat, lev_p = stats.levene(*groups_data)
            try:
                bart_stat, bart_p = stats.bartlett(*groups_data)
            except Exception:
                bart_stat, bart_p = np.nan, np.nan
            rows.append({'column': name, 'levene_stat': lev_stat, 'levene_p': lev_p,
                        'bartlett_stat': bart_stat, 'bartlett_p': bart_p, 'n_groups': len(groups_data)})
        return pd.DataFrame(rows)

    @staticmethod
    def nested_bootstrap(fly_labels, statistic_fn, n_iterations=200, seed=0):
        """
        Generic fly-nested bootstrap: resample flies with replacement,
        then resample each resampled fly's neurons with replacement.
        statistic_fn(idx) is called with an array of indices into
        whatever data it closes over, once per iteration.
        """
        fly_labels = np.asarray(fly_labels)
        unique_flies = np.array(sorted(set(fly_labels)))
        fly_to_indices = {f: np.where(fly_labels == f)[0] for f in unique_flies}
        rng = np.random.default_rng(seed)
        boot_stats = []
        for _ in range(n_iterations):
            resampled_flies = rng.choice(unique_flies, size=len(unique_flies), replace=True)
            idx = []
            for f in resampled_flies:
                fly_idx = fly_to_indices[f]
                idx.extend(rng.choice(fly_idx, size=len(fly_idx), replace=True))
            idx = np.array(idx)
            try:
                boot_stats.append(statistic_fn(idx))
            except Exception:
                continue
        boot_stats = np.array(boot_stats, dtype=float)
        if len(boot_stats) == 0:
            return {'bootstrap_values': boot_stats, 'mean': np.nan, 'ci_lower': np.nan,
                    'ci_upper': np.nan, 'n_valid': 0}
        ci_lo, ci_hi = np.nanpercentile(boot_stats, [2.5, 97.5])
        return {'bootstrap_values': boot_stats, 'mean': float(np.nanmean(boot_stats)),
                'ci_lower': float(ci_lo), 'ci_upper': float(ci_hi), 'n_valid': len(boot_stats)}

    @staticmethod
    def plot_dispersion_figure(disp_table, mv_disp, fset, output_dir, pc_threshold=30):
        """Bar chart per named column (<=pc_threshold cols) or a dispersion-vs-index
        curve (trace-shaped representations), for each of SD/CV/MAD/IQR."""
        n_cols = len(disp_table)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['sd', 'cv', 'mad', 'iqr']
        titles = ['Standard Deviation', 'Coefficient of Variation', 'Median Abs. Deviation', 'IQR']
        x_pos = np.arange(n_cols)
        for ax, metric, title in zip(axes.flat, metrics, titles):
            if n_cols <= pc_threshold:
                ax.bar(x_pos, disp_table[metric].values, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_xticks(x_pos); ax.set_xticklabels(disp_table['column'], rotation=75, ha='right', fontsize=7)
            else:
                ax.plot(x_pos, disp_table[metric].values, color='steelblue', linewidth=1.2)
                ax.set_xlabel('Column index (time-ordered)', fontsize=9)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        mean_dist = mv_disp['mean_pairwise_distance']
        fig.suptitle(f'{fset} | dispersion across neurons (n={n_cols} cols) | '
                     f'multivariate mean-dist={mean_dist:.2f}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dispersion_{fset}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def run_sweep(clustering_results, output_dir, n_permutations=999, n_bootstrap=200, pc_threshold=30):
        disp_path = os.path.join(output_dir, 'dispersion_table.csv')
        if os.path.exists(disp_path):
            print("Variability statistics sweep: found existing tables, loading instead of recomputing.")
            return {
                'dispersion_table': pd.read_csv(disp_path),
                'participation_ratio_table': pd.read_csv(os.path.join(output_dir, 'participation_ratio_table.csv')),
                'neuron_ambiguity_table': pd.read_csv(os.path.join(output_dir, 'neuron_ambiguity_table.csv')),
                'levene_bartlett_by_cluster': pd.read_csv(os.path.join(output_dir, 'levene_bartlett_by_cluster.csv')),
                'levene_bartlett_by_fly': pd.read_csv(os.path.join(output_dir, 'levene_bartlett_by_fly.csv')),
                'nested_bootstrap_table': pd.read_csv(os.path.join(output_dir, 'nested_bootstrap_table.csv')),
            }
        print("\n" + "="*60)
        print("Variability statistics sweep (dispersion / PR / silhouette / Levene-Bartlett / bootstrap) ...")
        dispersion_rows, pr_rows, ambiguity_rows = [], [], []
        levene_cluster_rows, levene_fly_rows, bootstrap_rows = [], [], []

        for fset, res in clustering_results.items():
            X, fly_labels = res['X'], res['fly_labels']
            X_disp = res.get('X_raw', X)
            gmm_labels, posteriors = res['gmm_labels'], res['posteriors']
            print(f"  Feature set: {fset}  (n={len(X)} neurons)")
            col_names = [f'{fset}_dim{i}' for i in range(X.shape[1])]

            disp_table = VariabilityStats.dispersion_indices(X_disp, col_names)
            mv_disp = VariabilityStats.compute_multivariate_dispersion(X, fly_labels, n_permutations=n_permutations)
            VariabilityStats.plot_dispersion_figure(disp_table, mv_disp, fset, output_dir, pc_threshold=pc_threshold)
            disp_table.insert(0, 'feature_set', fset)
            dispersion_rows.append(disp_table)

            pr_res = VariabilityStats.participation_ratio(X)
            pr_rows.append({'feature_set': fset, **pr_res, **mv_disp})
            print(f"    PR={pr_res['participation_ratio']:.2f}/{X.shape[1]} dims  |  "
                  f"PERMDISP p={mv_disp['permdisp_p_value']:.3f}")

            amb = VariabilityStats.neuron_ambiguity_scores(X, gmm_labels, posteriors)
            for i in range(len(X)):
                ambiguity_rows.append({'feature_set': fset, 'neuron_idx': i, 'fly_id': fly_labels[i],
                                       'cluster': gmm_labels[i], 'silhouette': amb['silhouette'][i],
                                       'posterior_entropy': amb['posterior_entropy'][i],
                                       'max_posterior': amb['max_posterior'][i]})

            lev_cluster = VariabilityStats.levene_bartlett_by_group(X, gmm_labels, col_names)
            lev_cluster.insert(0, 'feature_set', fset)
            levene_cluster_rows.append(lev_cluster)
            lev_fly = VariabilityStats.levene_bartlett_by_group(X, fly_labels, col_names)
            lev_fly.insert(0, 'feature_set', fset)
            levene_fly_rows.append(lev_fly)

            def _pr_stat(idx, X=X):
                return VariabilityStats.participation_ratio(X[idx])['participation_ratio']
            def _disp_stat(idx, X=X):
                return float(pdist(X[idx], metric='euclidean').mean())
            pr_boot = VariabilityStats.nested_bootstrap(fly_labels, _pr_stat, n_iterations=n_bootstrap)
            disp_boot = VariabilityStats.nested_bootstrap(fly_labels, _disp_stat, n_iterations=n_bootstrap)
            bootstrap_rows.append({'feature_set': fset, 'pr_mean': pr_boot['mean'],
                                   'pr_ci_lower': pr_boot['ci_lower'], 'pr_ci_upper': pr_boot['ci_upper'],
                                   'dispersion_mean': disp_boot['mean'], 'dispersion_ci_lower': disp_boot['ci_lower'],
                                   'dispersion_ci_upper': disp_boot['ci_upper'], 'n_bootstrap': n_bootstrap})
            print(f"    fly-nested bootstrap: PR=[{pr_boot['ci_lower']:.2f}, {pr_boot['ci_upper']:.2f}]")

        dispersion_table = pd.concat(dispersion_rows, ignore_index=True) if dispersion_rows else pd.DataFrame()
        pr_table = pd.DataFrame(pr_rows)
        ambiguity_table = pd.DataFrame(ambiguity_rows)
        levene_cluster_table = pd.concat(levene_cluster_rows, ignore_index=True) if levene_cluster_rows else pd.DataFrame()
        levene_fly_table = pd.concat(levene_fly_rows, ignore_index=True) if levene_fly_rows else pd.DataFrame()
        bootstrap_table = pd.DataFrame(bootstrap_rows)

        for name, table in [('dispersion_table', dispersion_table), ('participation_ratio_table', pr_table),
                            ('neuron_ambiguity_table', ambiguity_table),
                            ('levene_bartlett_by_cluster', levene_cluster_table),
                            ('levene_bartlett_by_fly', levene_fly_table),
                            ('nested_bootstrap_table', bootstrap_table)]:
            table.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)

        print("\n  Variability statistics sweep complete.")
        return {'dispersion_table': dispersion_table, 'participation_ratio_table': pr_table,
                'neuron_ambiguity_table': ambiguity_table, 'levene_bartlett_by_cluster': levene_cluster_table,
                'levene_bartlett_by_fly': levene_fly_table, 'nested_bootstrap_table': bootstrap_table}
def save_condition_result(result_dict, output_dir):
    """
    Cache everything process_pkl_file returns for this condition, so a
    later run can skip straight to CombinedAnalysis without recomputing
    anything for it. gmm_diag['models'] (fitted GaussianMixture objects,
    up to ~12 per feature_set) is stripped first -- bulky, and nothing
    downstream needs the model objects themselves, only bic/best_k/labels.
    """
    path = os.path.join(output_dir, 'condition_result.pkl')
    cacheable = dict(result_dict)
    if 'clustering_results' in cacheable:
        slim = {}
        for fset, res in cacheable['clustering_results'].items():
            res = dict(res)
            if 'gmm_diag' in res:
                gmm_diag = dict(res['gmm_diag'])
                gmm_diag.pop('models', None)
                res['gmm_diag'] = gmm_diag
            slim[fset] = res
        cacheable['clustering_results'] = slim
    joblib.dump(cacheable, path, compress=3)
    print(f"  Condition result cached -> {path}")

def load_condition_result(output_dir, force_recompute=False):
    """Loads a cached condition result if present (and not overridden), else returns None."""
    path = os.path.join(output_dir, 'condition_result.pkl')
    if not force_recompute and os.path.exists(path):
        print(f"  Found cached condition result -> {path}, loading instead of recomputing")
        return joblib.load(path)
    return None


# Example usage
if __name__ == "__main__":
    # Configuration
    multi = False
    degen = True
    temp = False
    FORCE_RECOMPUTE = True 
    if multi: 
        pkl_files = [r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_ME_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_ME_Y.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LO_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LO_Y.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LOP_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LOP_Y.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_ME_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_ME_Y.pkl",
        #         r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LO_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LO_Y.pkl",
        #         r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LOP_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LOP_Y.pkl"]
    elif degen:
        # pkl_files = [r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Tm1_chirp.pkl", r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Tm9_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_chirp.pkl",
        #             r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_chirp.pkl"]
        pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_chirp.pkl",]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_chirp.pkl"]
    elif temp:
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_12_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_25_chirp.pkl",\
                        # r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_25_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_12_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_25_chirp.pkl"]
        pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_25_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_25_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_25_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_18_chirp.pkl"]
    # ---------------------------------------------------------------
    # X-axis limits per pkl file.
    # One dict per entry in pkl_files (same order).
    # Keys: 'full', 'polarity', 'frequency', 'contrast', 'luminance'
    # Value: (xmin, xmax) in seconds, or None for auto.
    # ---------------------------------------------------------------
    ylims_list = [
        # _Tm1
        {'full': [-0.75, 3.5], 'polarity': [-0.75, 3.5], 'frequency': [-0.75, 1.75], 'contrast': [-0.75, 3.5], 'luminance': [-0.75, 3]},
        # _Tm4
        {'full': [-4.5, 6], 'polarity': [-4.5, 6.5], 'frequency': [-4, 3.5], 'contrast': [-4, 3.5], 'luminance': [-4.5, 3.5]},
        # _Tm9
        {'full': [-1.5, 2.5], 'polarity': [-1, 2.75], 'frequency': [-1.2, 1.5], 'contrast': [-1.2, 1.25], 'luminance': [-1, 2]},
        # _Mi1
        {'full': [-2, 3], 'polarity': [-2, 3], 'frequency': [-0.75, 1.75], 'contrast': [-0.7, 1.5], 'luminance': [-2.5, 2.5]},
        # _Mi4
        {'full': [-1.5, 2.25], 'polarity': [-1.5, 2.25], 'frequency': [-0.75, 1.5], 'contrast': [-0.5, 1.75], 'luminance': [-1.25, 1.5]},
        # _Mi9
        {'full': [-1.5, 3.5], 'polarity': [-3, 3], 'frequency': [-1, 1.25], 'contrast': [-1, 0.75], 'luminance': [-1, 1.6]},
    ]
    # Pad with None-dicts if xlims_list is shorter than pkl_files
    _empty_ylims = {'full': None, 'polarity': None, 'frequency': None, 'contrast': None, 'luminance': None}
    while len(ylims_list) < len(pkl_files):
        ylims_list.append(_empty_ylims.copy())

    all_results = []
    for PKL_FILE, ylims in zip(pkl_files, ylims_list):
        folder = PKL_FILE.split('\\')[-1].split('.')[0]
        condition_label = folder.lstrip('_')
        if multi:
            # metadata = pd.read_excel("D:/Christian/02_twophoton/metadata.xlsx", sheet_name='251023_tdc2_cschr_pan')
            output_dir=f"C:/phd/02_twophoton/251023_tdc2_cschr_pan/4_results/{folder}"
        elif degen:
            # metadata = pd.read_excel("D:/Christian/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            # output_dir=f"D:/Christian/02_twophoton/250730_degen_var/4_results/{folder}"
            # metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            output_dir=f"C:/phd/02_twophoton/250730_degen_var/4_results/{folder}"
        elif temp:
            # metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            output_dir=f"C:/phd/02_twophoton/20260617_temp/4_results/{folder}"
        os.makedirs(output_dir, exist_ok=True)
        cached = load_condition_result(output_dir, force_recompute=FORCE_RECOMPUTE)
        if cached is not None:
            all_results.append(cached)
            continue

        if multi:
            metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='251023_tdc2_cschr_pan')
        elif degen:
            metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
        elif temp:
            metadata = pd.read_excel("C:/phd/02_twophoton/20260617_temp/metadata_2photon_tempdependency.xlsx", sheet_name='20260617_temp')
        mean_fps = pd.to_numeric(metadata['fps'], errors='coerce').dropna()
        mean_fps = mean_fps[mean_fps.between(1, 200)].mean()
        print(f"  mean_fps = {mean_fps:.4f} Hz")
        TARGET_FPS = mean_fps  # Target frames per second for interpolation
        ORIGINAL_FPS = None  # Set to None to extract from data, or specify (e.g., 30)
        
        # Run analysis
        (condition_rois, fly_rois, qi_map, roi_segments, summary_results,
        clustering_results, fly_effect_results, variability_results, run_fps) = process_pkl_file(
            PKL_FILE, metadata, multi, degen, TARGET_FPS,
            original_fps=ORIGINAL_FPS, output_dir=output_dir, ylims=ylims,
            qi_threshold=0.0, reliability_threshold=0.0, min_reps=3, #TODO: change back to 0.45, and 0.4
        )
        result_dict = {
            'condition_rois': condition_rois, 'fly_rois': fly_rois, 'qi_map': qi_map,
            'roi_segments': roi_segments, 'summary_results': summary_results,
            'clustering_results': clustering_results, 'fly_effect_results': fly_effect_results,
            'variability_results': variability_results,
            'mean_fps': run_fps, 'condition_name': condition_label,
        }
        save_condition_result(result_dict, output_dir)
        all_results.append(result_dict)
    
    if multi:
        combined_output = "C:/phd/02_twophoton/251023_tdc2_cschr_pan/4_results/combined"
    elif degen:
        # combined_output = "D:/Christian/02_twophoton/250730_degen_var/4_results/combined"
        
        combined_output = "C:/phd/02_twophoton/250730_degen_var/4_results/combined"
    elif temp:
            # combined_output = "D:/Christian/02_twophoton/250730_degen_var/4_results/combined"
            combined_output = "C:/phd/02_twophoton/20260617_temp/4_results/combined"
    os.makedirs(combined_output, exist_ok=True)
    CombinedAnalysis.run_all(all_results, pkl_files, combined_output)
    print("\nDone!")
