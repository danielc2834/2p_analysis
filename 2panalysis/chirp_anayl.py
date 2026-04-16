import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram as sp_dendrogram
from collections import defaultdict
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import umap
import warnings
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

def interpolate_trace(df_trace, original_fps, target_fps, trim_seconds=5, detrend=False):
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

def analyze_polarity_response(fly_segments, mean_fps, output_dir):
    """
    Analyze polarity (ON/OFF) responses by extracting maximum amplitude.
    
    Parameters:
    -----------
    fly_segments : dict
        Segmented traces organized by fly
    mean_fps : float
        Frames per second
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict
        Polarity analysis results
    """
    print("\n" + "="*60)
    print("Analyzing polarity (ON/OFF) responses...")
    
    # Polarity stimulus timing (in seconds from start of polarity segment)
    # Based on: value = [0,0,1,0,0.5,0.5,1,0,1,0,1,0,...]
    # duration = [4,2,3,3,2,...]
    # ON transitions (0->1): at 4s, 6+3+3+2=14s (but this is outside first 14s)
    # Let's use: 0->1 at 6s (after [4,2])
    # OFF transition (1->0): at 9s (after [4,2,3])
    
    on_transition_time = 6.0  # seconds
    off_transition_time = 9.0  # seconds
    
    # Storage for results
    fly_on_amplitudes = defaultdict(list)  # fly_id -> [ON amplitudes]
    fly_off_amplitudes = defaultdict(list)  # fly_id -> [OFF amplitudes]
    
    # Process each fly's polarity traces
    for fly_id, traces_and_times in fly_segments['polarity'].items():
        print(f"\n  Processing FlyID: {fly_id}")
        
        for trace, time in traces_and_times:
            # ON response (0->1 transition)
            on_baseline_start = int((on_transition_time - 1.0) * mean_fps)
            on_baseline_end = int(on_transition_time * mean_fps)
            on_response_start = int((on_transition_time - 1.0) * mean_fps)
            on_response_end = int((on_transition_time + 1.0) * mean_fps)
            
            if on_response_end <= len(trace):
                on_baseline = np.mean(trace[on_baseline_start:on_baseline_end])
                # on_response = trace[on_response_start:on_response_end] - on_baseline
                # on_amplitude = np.max(on_response)
                on_response = trace[on_response_start:on_response_end] - on_baseline
                on_amplitude = np.max(np.abs(on_response))
                fly_on_amplitudes[fly_id].append(on_amplitude)
            
            # OFF response (1->0 transition)
            off_baseline_start = int((off_transition_time - 1.0) * mean_fps)
            off_baseline_end = int(off_transition_time * mean_fps)
            off_response_start = int((off_transition_time) * mean_fps)
            off_response_end = int((off_transition_time + 1.5) * mean_fps)
            
            if off_response_end <= len(trace):
                off_baseline = np.mean(trace[off_baseline_start:off_baseline_end])
                # off_response = trace[off_response_start:off_response_end] - off_baseline
                # off_amplitude = np.max(off_response)
                off_response = trace[off_response_start:off_response_end] - off_baseline
                off_amplitude = np.max(np.abs(off_response))
                fly_off_amplitudes[fly_id].append(off_amplitude)
    
    # Calculate fly averages
    fly_on_mean = {fly_id: np.mean(amps) for fly_id, amps in fly_on_amplitudes.items()}
    fly_off_mean = {fly_id: np.mean(amps) for fly_id, amps in fly_off_amplitudes.items()}
    
    # Calculate overall statistics
    on_values = list(fly_on_mean.values())
    off_values = list(fly_off_mean.values())
    
    overall_on_mean = np.mean(on_values)
    overall_on_sem = np.std(on_values) / np.sqrt(len(on_values))
    overall_off_mean = np.mean(off_values)
    overall_off_sem = np.std(off_values) / np.sqrt(len(off_values))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = [0, 1]
    means = [overall_on_mean, overall_off_mean]
    sems = [overall_on_sem, overall_off_sem]
    colors = ['darkgreen', 'darkred']
    labels = ['ON', 'OFF']
    
    # Bar plot with error bars
    bars = ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2, capsize=8, width=0.6)
    
    # Overlay individual fly data
    for i, (fly_id, on_amp) in enumerate(fly_on_mean.items()):
        off_amp = fly_off_mean[fly_id]
        ax.plot([0, 1], [on_amp, off_amp], 'o-', color='gray', 
                alpha=0.5, markersize=6, linewidth=1)
    
    # Scatter individual fly points
    ax.scatter([0]*len(on_values), on_values, color='black', s=60, alpha=0.6, zorder=3)
    ax.scatter([1]*len(off_values), off_values, color='black', s=60, alpha=0.6, zorder=3)
    
    ax.set_ylabel('Maximum Amplitude (ΔF/F)', fontsize=12)
    ax.set_title('Polarity Response (ON vs OFF)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_polarity_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ON response: {overall_on_mean:.4f} ± {overall_on_sem:.4f}")
    print(f"  OFF response: {overall_off_mean:.4f} ± {overall_off_sem:.4f}")
    print(f"  Number of flies: {len(fly_on_mean)}")
    
    results = {
        'fly_on_mean': fly_on_mean,
        'fly_off_mean': fly_off_mean,
        'overall_on_mean': overall_on_mean,
        'overall_on_sem': overall_on_sem,
        'overall_off_mean': overall_off_mean,
        'overall_off_sem': overall_off_sem
    }
    
    return results


def analyze_luminance_response(segment_results, fly_segments, mean_fps, output_dir):
    """
    Analyze luminance response by extracting amplitude for each luminance value.
    
    Parameters:
    -----------
    segment_results : dict
        Results from segment analysis
    fly_segments : dict
        Segmented traces organized by fly
    mean_fps : float
        Frames per second
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict
        Luminance analysis results
    """
    print("\n" + "="*60)
    print("Analyzing luminance responses...")
    
    # Luminance values and their durations (in seconds)
    luminance_values = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 
                        0, 0.1, 0.2, 0.3, 0.4, 0.5]
    luminance_durations = [2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                          0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # Get unique luminance values
    unique_lum = sorted(list(set(luminance_values)))
    
    # Storage for results
    fly_amplitudes = defaultdict(lambda: defaultdict(list))  # fly_id -> luminance -> [amplitudes]
    
    # Process each fly's luminance traces
    for fly_id, traces_and_times in fly_segments['luminance'].items():
        print(f"\n  Processing FlyID: {fly_id}")
        
        for trace, time in traces_and_times:
            # Calculate baseline from first 0.5s (first luminance value duration)
            baseline_frames = int(0.5 * mean_fps)
            baseline = np.mean(trace[:baseline_frames])
            
            # Extract amplitude for each luminance value
            current_idx = 0
            for lum_val, duration in zip(luminance_values, luminance_durations):
                n_frames = int(duration * mean_fps)
                segment = trace[current_idx:current_idx + n_frames]
                
                # Calculate mean amplitude (delta from baseline)
                amplitude = np.mean(segment) - baseline
                fly_amplitudes[fly_id][lum_val].append(amplitude)
                
                current_idx += n_frames
    
    # Average amplitudes across presentations for each fly and luminance value
    fly_avg_amplitudes = {}  # fly_id -> {luminance: mean_amplitude}
    fly_slopes = {}  # fly_id -> slope
    
    for fly_id in fly_amplitudes.keys():
        fly_avg_amplitudes[fly_id] = {}
        for lum_val in unique_lum:
            if lum_val in fly_amplitudes[fly_id]:
                fly_avg_amplitudes[fly_id][lum_val] = np.mean(fly_amplitudes[fly_id][lum_val])
        
        # Calculate slope for this fly
        x = np.array(list(fly_avg_amplitudes[fly_id].keys()))
        y = np.array(list(fly_avg_amplitudes[fly_id].values()))
        slope, _ = np.polyfit(x, y, 1)
        fly_slopes[fly_id] = slope
    
    # Calculate overall average across flies
    overall_amplitudes = {}  # luminance -> [mean_amplitudes_per_fly]
    for lum_val in unique_lum:
        overall_amplitudes[lum_val] = [fly_avg_amplitudes[fly][lum_val] 
                                        for fly in fly_avg_amplitudes.keys() 
                                        if lum_val in fly_avg_amplitudes[fly]]
    
    # Calculate mean and SEM for each luminance value
    overall_mean = {lum: np.mean(amps) for lum, amps in overall_amplitudes.items()}
    overall_sem = {lum: np.std(amps) / np.sqrt(len(amps)) for lum, amps in overall_amplitudes.items()}
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Luminance vs Amplitude
    x_vals = sorted(overall_mean.keys())
    y_vals = [overall_mean[x] for x in x_vals]
    y_sem = [overall_sem[x] for x in x_vals]
    
    ax1.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o', color='black', 
                 markersize=8, capsize=5, linewidth=2, label='Mean ± SEM')
    ax1.plot(x_vals, y_vals, '-', color='gray', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Luminance Value', fontsize=12)
    ax1.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
    ax1.set_title('Luminance Response Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right plot: Slope barplot with individual flies
    slopes = list(fly_slopes.values())
    mean_slope = np.mean(slopes)
    sem_slope = np.std(slopes) / np.sqrt(len(slopes))
    
    ax2.bar(0, mean_slope, yerr=sem_slope, color='lightblue', 
            edgecolor='black', linewidth=2, capsize=5, width=0.5)
    
    # Overlay individual fly data points
    x_scatter = np.zeros(len(slopes))
    ax2.scatter(x_scatter, slopes, color='red', s=80, alpha=0.7, zorder=3)
    
    ax2.set_ylabel('Slope (ΔF/F per luminance unit)', fontsize=12)
    ax2.set_title('Luminance Response Slope', fontsize=14, fontweight='bold')
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Slope'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_luminance_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Mean slope: {mean_slope:.4f} ± {sem_slope:.4f}")
    print(f"  Number of flies: {len(fly_slopes)}")
    
    results = {
        'fly_amplitudes': fly_avg_amplitudes,
        'fly_slopes': fly_slopes,
        'overall_mean': overall_mean,
        'overall_sem': overall_sem,
        'mean_slope': mean_slope,
        'sem_slope': sem_slope
    }
    
    return results

def analyze_contrast_response(fly_segments, mean_fps, output_dir):
    """
    Analyze contrast responses using Weber contrast calculation.
    
    Parameters:
    -----------
    fly_segments : dict
        Segmented traces organized by fly
    mean_fps : float
        Frames per second
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict
        Contrast analysis results
    """
    print("\n" + "="*60)
    print("Analyzing contrast responses...")
    
    # Contrast stimulus values and durations (in seconds)
    # From the luminance section after frequency: starts with contrast changes
    # Pattern: 0.5 baseline, then steps from 0.5
    contrast_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    contrast_durations = [2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # Calculate Weber contrast for each step (contrast = (L - L_background) / L_background * 100)
    # Background is the initial 0.5 value
    background = 0.5
    weber_contrasts = []
    absolute_contrasts = []
    for val in contrast_values:
        if background > 0:
            contrast_percent = ((val - background) / background) * 100
            abs_contrast_percent = (abs(val - background) / background) * 100
        else:
            contrast_percent = 0
            abs_contrast_percent = 0
        weber_contrasts.append(contrast_percent)
        absolute_contrasts.append(abs_contrast_percent)
    
    # Storage for results
    fly_contrast_responses = defaultdict(lambda: defaultdict(list))  # fly_id -> contrast% -> [responses]
    fly_abs_contrast_responses = defaultdict(lambda: defaultdict(list))  # fly_id -> abs_contrast% -> [responses]
    
    # Process each fly's contrast traces
    for fly_id, traces_and_times in fly_segments['contrast'].items():
        print(f"\n  Processing FlyID: {fly_id}")
        
        for trace, time in traces_and_times:
            # Use first segment as baseline (2 seconds of 0.5 value)
            baseline_frames = int(contrast_durations[0] * mean_fps)
            baseline = np.mean(trace[:baseline_frames])
            
            # Extract response magnitude for each contrast step
            current_idx = 0
            for contrast_val, duration, weber_contrast, abs_contrast in zip(contrast_values, contrast_durations, 
                                                                            weber_contrasts, absolute_contrasts):
                n_frames = int(duration * mean_fps)
                segment = trace[current_idx:current_idx + n_frames]
                
                # Calculate mean magnitude (delta from baseline)
                magnitude = np.mean(segment) - baseline
                fly_contrast_responses[fly_id][weber_contrast].append(magnitude)
                fly_abs_contrast_responses[fly_id][abs_contrast].append(magnitude)
                
                current_idx += n_frames
    
    # Average across TSeries for each fly - Weber contrast
    fly_avg_responses = {}  # fly_id -> {contrast%: mean_magnitude}
    for fly_id in fly_contrast_responses.keys():
        fly_avg_responses[fly_id] = {}
        for contrast_pct in fly_contrast_responses[fly_id].keys():
            fly_avg_responses[fly_id][contrast_pct] = np.mean(fly_contrast_responses[fly_id][contrast_pct])
    
    # Average across TSeries for each fly - Absolute contrast
    fly_avg_abs_responses = {}  # fly_id -> {abs_contrast%: mean_magnitude}
    for fly_id in fly_abs_contrast_responses.keys():
        fly_avg_abs_responses[fly_id] = {}
        for contrast_pct in fly_abs_contrast_responses[fly_id].keys():
            fly_avg_abs_responses[fly_id][contrast_pct] = np.mean(fly_abs_contrast_responses[fly_id][contrast_pct])
    
    # Get unique contrast values
    unique_contrasts = sorted(list(set(weber_contrasts)))
    unique_abs_contrasts = sorted(list(set(absolute_contrasts)))
    
    # Calculate overall average across flies - Weber
    overall_responses = {}  # contrast% -> [mean_responses_per_fly]
    for contrast_pct in unique_contrasts:
        overall_responses[contrast_pct] = [fly_avg_responses[fly][contrast_pct]
                                           for fly in fly_avg_responses.keys()
                                           if contrast_pct in fly_avg_responses[fly]]
    
    overall_mean = {contrast: np.mean(resps) for contrast, resps in overall_responses.items()}
    overall_sem = {contrast: np.std(resps) / np.sqrt(len(resps)) 
                   for contrast, resps in overall_responses.items()}
    
    # Calculate overall average across flies - Absolute
    overall_abs_responses = {}  # abs_contrast% -> [mean_responses_per_fly]
    for contrast_pct in unique_abs_contrasts:
        overall_abs_responses[contrast_pct] = [fly_avg_abs_responses[fly][contrast_pct]
                                               for fly in fly_avg_abs_responses.keys()
                                               if contrast_pct in fly_avg_abs_responses[fly]]
    
    overall_abs_mean = {contrast: np.mean(resps) for contrast, resps in overall_abs_responses.items()}
    overall_abs_sem = {contrast: np.std(resps) / np.sqrt(len(resps)) 
                       for contrast, resps in overall_abs_responses.items()}
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Weber contrast (signed)
    x_vals = sorted(overall_mean.keys())
    y_vals = [overall_mean[x] for x in x_vals]
    y_sem = [overall_sem[x] for x in x_vals]
    
    ax1.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o', color='black',
                markersize=8, capsize=5, linewidth=2, label='Mean ± SEM')
    ax1.plot(x_vals, y_vals, '-', color='gray', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Weber Contrast (%)', fontsize=12)
    ax1.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
    ax1.set_title('Weber Contrast Response (Signed)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend()
    
    # Right plot: Absolute contrast
    x_vals_abs = sorted(overall_abs_mean.keys())
    y_vals_abs = [overall_abs_mean[x] for x in x_vals_abs]
    y_sem_abs = [overall_abs_sem[x] for x in x_vals_abs]
    
    ax2.errorbar(x_vals_abs, y_vals_abs, yerr=y_sem_abs, fmt='o', color='darkblue',
                markersize=8, capsize=5, linewidth=2, label='Mean ± SEM')
    ax2.plot(x_vals_abs, y_vals_abs, '-', color='lightblue', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Absolute Contrast (%)', fontsize=12)
    ax2.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
    ax2.set_title('Absolute Contrast Response', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_contrast_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Weber contrast range: {min(x_vals):.1f}% to {max(x_vals):.1f}%")
    print(f"  Absolute contrast range: {min(x_vals_abs):.1f}% to {max(x_vals_abs):.1f}%")
    print(f"  Number of flies: {len(fly_avg_responses)}")
    
    results = {
        'fly_responses_weber': fly_avg_responses,
        'fly_responses_absolute': fly_avg_abs_responses,
        'overall_mean_weber': overall_mean,
        'overall_sem_weber': overall_sem,
        'overall_mean_absolute': overall_abs_mean,
        'overall_sem_absolute': overall_abs_sem,
        'weber_contrasts': unique_contrasts,
        'absolute_contrasts': unique_abs_contrasts
    }
    
    return results

def analyze_frequency_response(fly_segments, mean_fps, output_dir):
    """
    Analyze frequency responses by extracting maximum amplitude for each temporal frequency.
    
    Parameters:
    -----------
    fly_segments : dict
        Segmented traces organized by fly
    mean_fps : float
        Frames per second
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict
        Frequency analysis results
    """
    print("\n" + "="*60)
    print("Analyzing frequency responses...")
    
    # Frequency stimulus: series of flashes at different temporal frequencies
    # The flashes alternate between values (e.g., 0.5 and 1) at different rates
    # Based on the stimulus protocol, the frequency section contains flashes
    # Duration pattern suggests frequencies increase (durations decrease)
    
    # Define frequency bins based on flash durations
    # Each pair of durations represents one cycle (ON + OFF)
    flash_durations = [0.243, 0.449, 0.406, 0.372, 0.343, 0.317, 0.296, 0.277, 0.26, 0.246, 
                      0.232, 0.221, 0.21, 0.2, 0.192, 0.183, 0.176, 0.169, 0.163, 0.157, 
                      0.151, 0.146, 0.142, 0.137, 0.133, 0.129, 0.125, 0.122, 0.118, 0.115, 
                      0.112, 0.11, 0.107, 0.104, 0.102, 0.099, 0.097, 0.095, 0.093, 0.091, 
                      0.09, 0.087, 0.086, 0.084, 0.082, 0.081, 0.079, 0.078, 0.077, 0.075, 
                      0.074, 0.073, 0.072, 0.07, 0.069, 0.068, 0.068, 0.066, 0.065, 0.064, 
                      0.063, 0.062, 0.062, 0.06, 0.06, 0.059, 0.058, 0.058, 0.056, 0.056, 
                      0.055, 0.055, 0.054, 0.053, 0.052, 0.052, 0.052, 0.05]
    
    # Calculate frequencies (Hz) - each duration is half a cycle
    frequencies = [1.0 / (2 * dur) for dur in flash_durations]
    
    # Define 10 frequency bins: 1-2 Hz, 2-3 Hz, ..., 9-10 Hz
    freq_bin_edges = np.arange(0, 11, 1)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # freq_bin_centers = np.arange(1.5, 10, 1)  # [1.5, 2.5, 3.5, ..., 9.5] for plotting
    freq_bin_centers = freq_bin_edges
    
    def bin_frequency(freq):
        """Assign frequency to one of 10 bins (1-2, 2-3, ..., 9-10 Hz)"""
        if freq < 0.0 or freq > 10.0:
            return None  # Exclude frequencies outside range
        # Find which bin it belongs to
        for i in range(len(freq_bin_edges) - 1):
            if freq_bin_edges[i] <= freq < freq_bin_edges[i+1]:
                return freq_bin_centers[i]
        return None
    
    # Storage for ROI-level responses
    roi_freq_responses = defaultdict(lambda: defaultdict(list))  # (fly_id, roi_idx) -> bin_center -> [responses]
    
    # Process each fly's frequency traces at ROI level
    for fly_id, traces_and_times in fly_segments['frequency'].items():
        print(f"\n  Processing FlyID: {fly_id}")
        
        for roi_idx, (trace, time) in enumerate(traces_and_times):
            # Use first 0.5s as baseline
            baseline_frames = int(0.5 * mean_fps)
            if baseline_frames >= len(trace):
                continue
            baseline = np.mean(trace[:baseline_frames])
            
            # Extract maximum response for each frequency flash
            current_idx = baseline_frames
            for freq, duration in zip(frequencies, flash_durations):
                n_frames = int(duration * mean_fps)
                if n_frames == 0:
                    n_frames = 1  # Ensure at least 1 frame
                    
                if current_idx + n_frames <= len(trace):
                    segment = trace[current_idx:current_idx + n_frames]
                    
                    # Check if segment is not empty
                    if len(segment) > 0:
                        # Calculate maximum amplitude (delta from baseline)
                        max_amplitude = np.max(segment - baseline)
                        
                        # Bin the frequency
                        bin_center = bin_frequency(freq)
                        if bin_center is not None:  # Only include if within 1-10 Hz range
                            roi_freq_responses[(fly_id, roi_idx)][bin_center].append(max_amplitude)
                    
                    current_idx += n_frames
                else:
                    break  # Stop if we run out of trace
    
    # Average responses within each bin for each ROI
    roi_binned_avg = {}  # (fly_id, roi_idx) -> {bin_center: mean_response}
    for roi_key, freq_data in roi_freq_responses.items():
        roi_binned_avg[roi_key] = {}
        for bin_center, responses in freq_data.items():
            if len(responses) > 0:
                roi_binned_avg[roi_key][bin_center] = np.mean(responses)
    
    # Now aggregate by fly: average across ROIs for each fly
    fly_freq_responses = defaultdict(lambda: defaultdict(list))  # fly_id -> bin_center -> [roi_averages]
    for (fly_id, roi_idx), freq_data in roi_binned_avg.items():
        for bin_center, avg_response in freq_data.items():
            fly_freq_responses[fly_id][bin_center].append(avg_response)
    
    # Average across ROIs for each fly
    fly_avg_responses = {}  # fly_id -> {bin_center: mean_amplitude}
    for fly_id, freq_data in fly_freq_responses.items():
        fly_avg_responses[fly_id] = {}
        for bin_center, roi_responses in freq_data.items():
            if len(roi_responses) > 0:
                fly_avg_responses[fly_id][bin_center] = np.mean(roi_responses)
    
    # Use all 10 bin centers for consistency
    unique_freqs = list(freq_bin_centers)
    
    # Calculate overall average across flies
    overall_responses = {}  # bin_center -> [mean_responses_per_fly]
    for freq in unique_freqs:
        overall_responses[freq] = [fly_avg_responses[fly][freq]
                                   for fly in fly_avg_responses.keys()
                                   if freq in fly_avg_responses[fly]]
    
    # Calculate mean and SEM for each frequency
    overall_mean = {}
    overall_sem = {}
    for freq, resps in overall_responses.items():
        if len(resps) > 0:
            overall_mean[freq] = np.mean(resps)
            overall_sem[freq] = np.std(resps) / np.sqrt(len(resps))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_vals = sorted(overall_mean.keys())
    y_vals = [overall_mean[x] for x in x_vals]
    y_sem = [overall_sem[x] for x in x_vals]
    
    ax.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o', color='black',
                markersize=8, capsize=5, linewidth=2, label='Mean ± SEM')
    ax.plot(x_vals, y_vals, '-', color='gray', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Temporal Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Maximum Response (ΔF/F)', fontsize=12)
    ax.set_title('Frequency Response Curve', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(freq_bin_centers)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '9_frequency_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  10 frequency bins from 1-10 Hz")
    print(f"  Number of flies: {len(fly_avg_responses)}")
    print(f"  Bins with data: {len(overall_mean)}")
    
    results = {
        'fly_responses': fly_avg_responses,
        'overall_mean': overall_mean,
        'overall_sem': overall_sem,
        'frequencies': unique_freqs
    }
    
    return results

def analyze_within_fly_variability(fly_segments, mean_fps, output_dir):
    """
    Analyze within-fly variability using CV and response reliability.
    
    Parameters:
    -----------
    fly_segments : dict
        Segmented traces organized by fly
    mean_fps : float
        Frames per second
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict
        Variability analysis results
    """
    print("\n" + "="*60)
    print("Analyzing within-fly variability...")
    
    segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
    # Storage for results
    fly_cv = {seg: {} for seg in segment_names}  # segment -> fly_id -> CV
    fly_reliability = {seg: {} for seg in segment_names}  # segment -> fly_id -> mean_correlation
    
    for seg_name in segment_names:
        print(f"\n  Processing segment: {seg_name}")
        
        for fly_id, traces_and_times in fly_segments[seg_name].items():
            if len(traces_and_times) < 2:
                print(f"    FlyID {fly_id}: Only 1 TSeries, skipping")
                continue
            
            print(f"    FlyID {fly_id}: {len(traces_and_times)} TSeries")
            
            # Align traces to minimum length
            min_length = min(len(trace) for trace, _ in traces_and_times)
            aligned_traces = np.array([trace[:min_length] for trace, _ in traces_and_times])
            
            # Calculate CV (coefficient of variation) across TSeries
            mean_trace = np.mean(aligned_traces, axis=0)
            std_trace = np.std(aligned_traces, axis=0)
            
            # Calculate mean CV across time points (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv_trace = std_trace / np.abs(mean_trace)
                cv_trace[~np.isfinite(cv_trace)] = 0  # Handle inf/nan
            
            fly_cv[seg_name][fly_id] = np.mean(cv_trace)
            
            # Calculate response reliability (pairwise correlations between TSeries)
            n_traces = len(aligned_traces)
            correlations = []
            for i in range(n_traces):
                for j in range(i + 1, n_traces):
                    corr = np.corrcoef(aligned_traces[i], aligned_traces[j])[0, 1]
                    if np.isfinite(corr):
                        correlations.append(corr)
            
            if len(correlations) > 0:
                fly_reliability[seg_name][fly_id] = np.mean(correlations)
            else:
                fly_reliability[seg_name][fly_id] = 0
    
    # Calculate overall statistics per segment
    overall_cv = {}
    overall_cv_sem = {}
    overall_reliability = {}
    overall_reliability_sem = {}
    
    for seg_name in segment_names:
        cv_values = list(fly_cv[seg_name].values())
        rel_values = list(fly_reliability[seg_name].values())
        
        if len(cv_values) > 0:
            overall_cv[seg_name] = np.mean(cv_values)
            overall_cv_sem[seg_name] = np.std(cv_values) / np.sqrt(len(cv_values))
        
        if len(rel_values) > 0:
            overall_reliability[seg_name] = np.mean(rel_values)
            overall_reliability_sem[seg_name] = np.std(rel_values) / np.sqrt(len(rel_values))
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Coefficient of Variation
    x_pos = np.arange(len(segment_names))
    cv_means = [overall_cv.get(seg, 0) for seg in segment_names]
    cv_sems = [overall_cv_sem.get(seg, 0) for seg in segment_names]
    
    bars1 = ax1.bar(x_pos, cv_means, yerr=cv_sems, color='steelblue', 
                    alpha=0.7, edgecolor='black', linewidth=1.5, capsize=5)
    
    # Overlay individual fly data
    for idx, seg_name in enumerate(segment_names):
        fly_values = list(fly_cv[seg_name].values())
        if len(fly_values) > 0:
            x_scatter = np.full(len(fly_values), idx)
            ax1.scatter(x_scatter, fly_values, color='black', s=40, alpha=0.6, zorder=3)
    
    ax1.set_ylabel('Coefficient of Variation', fontsize=12)
    ax1.set_title('Within-Fly Variability (CV)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Response Reliability (correlation)
    rel_means = [overall_reliability.get(seg, 0) for seg in segment_names]
    rel_sems = [overall_reliability_sem.get(seg, 0) for seg in segment_names]
    
    bars2 = ax2.bar(x_pos, rel_means, yerr=rel_sems, color='coral', 
                    alpha=0.7, edgecolor='black', linewidth=1.5, capsize=5)
    
    # Overlay individual fly data
    for idx, seg_name in enumerate(segment_names):
        fly_values = list(fly_reliability[seg_name].values())
        if len(fly_values) > 0:
            x_scatter = np.full(len(fly_values), idx)
            ax2.scatter(x_scatter, fly_values, color='black', s=40, alpha=0.6, zorder=3)
    
    ax2.set_ylabel('Mean Pairwise Correlation', fontsize=12)
    ax2.set_title('Response Reliability', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_within_fly_variability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Variability analysis complete")
    for seg_name in segment_names:
        print(f"    {seg_name}: CV={overall_cv.get(seg_name, 0):.3f}, Reliability={overall_reliability.get(seg_name, 0):.3f}")
    
    results = {
        'fly_cv': fly_cv,
        'fly_reliability': fly_reliability,
        'overall_cv': overall_cv,
        'overall_cv_sem': overall_cv_sem,
        'overall_reliability': overall_reliability,
        'overall_reliability_sem': overall_reliability_sem
    }
    
    return results

def analyze_across_fly_variability(fly_segments, mean_fps, output_dir):
    """
    Analyze across-fly variability using CV and response reliability.
    
    Parameters:
    -----------
    fly_segments : dict
        Segmented traces organized by fly
    mean_fps : float
        Frames per second
    output_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict
        Across-fly variability analysis results
    """
    print("\n" + "="*60)
    print("Analyzing across-fly variability...")
    
    segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
    # Storage for results
    segment_cv = {}  # segment -> CV across flies
    segment_reliability = {}  # segment -> mean correlation across fly pairs
    
    for seg_name in segment_names:
        print(f"\n  Processing segment: {seg_name}")
        
        # First, calculate mean trace for each fly
        fly_mean_traces = {}
        for fly_id, traces_and_times in fly_segments[seg_name].items():
            # Align traces to minimum length
            min_length = min(len(trace) for trace, _ in traces_and_times)
            aligned_traces = np.array([trace[:min_length] for trace, _ in traces_and_times])
            
            # Calculate mean across TSeries for this fly
            fly_mean_traces[fly_id] = np.mean(aligned_traces, axis=0)
        
        if len(fly_mean_traces) < 2:
            print(f"    Only {len(fly_mean_traces)} fly, skipping")
            continue
        
        print(f"    {len(fly_mean_traces)} flies")
        
        # Align fly mean traces to minimum length
        min_length = min(len(trace) for trace in fly_mean_traces.values())
        aligned_fly_traces = np.array([trace[:min_length] for trace in fly_mean_traces.values()])
        
        # Calculate CV (coefficient of variation) across flies
        mean_across_flies = np.mean(aligned_fly_traces, axis=0)
        std_across_flies = np.std(aligned_fly_traces, axis=0)
        
        # Calculate mean CV across time points (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_trace = std_across_flies / np.abs(mean_across_flies)
            cv_trace[~np.isfinite(cv_trace)] = 0  # Handle inf/nan
        
        segment_cv[seg_name] = np.mean(cv_trace)
        
        # Calculate reliability (pairwise correlations between flies)
        n_flies = len(aligned_fly_traces)
        correlations = []
        fly_ids = list(fly_mean_traces.keys())
        
        for i in range(n_flies):
            for j in range(i + 1, n_flies):
                corr = np.corrcoef(aligned_fly_traces[i], aligned_fly_traces[j])[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)
        
        if len(correlations) > 0:
            segment_reliability[seg_name] = np.mean(correlations)
        else:
            segment_reliability[seg_name] = 0
        
        print(f"    CV: {segment_cv[seg_name]:.3f}, Reliability: {segment_reliability[seg_name]:.3f}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Coefficient of Variation
    x_pos = np.arange(len(segment_names))
    cv_values = [segment_cv.get(seg, 0) for seg in segment_names]
    
    bars1 = ax1.bar(x_pos, cv_values, color='darkblue', 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Coefficient of Variation', fontsize=12)
    ax1.set_title('Across-Fly Variability (CV)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Response Reliability (correlation)
    rel_values = [segment_reliability.get(seg, 0) for seg in segment_names]
    
    bars2 = ax2.bar(x_pos, rel_values, color='darkred', 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Mean Pairwise Correlation', fontsize=12)
    ax2.set_title('Across-Fly Reliability', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_across_fly_variability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Across-fly variability analysis complete")
    
    results = {
        'segment_cv': segment_cv,
        'segment_reliability': segment_reliability
    }
    
    return results

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
    lum_durations = [2] + [0.5] * 16 + [2] + [0.5] * 4 + [0.5]  # seconds per step
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

    X = np.vstack(X_rows)
    # Standardize features across population (per-feature z-score)
    col_std = X.std(axis=0)
    col_std[col_std == 0] = 1.0
    X = (X - X.mean(axis=0)) / col_std
    return X, fly_labels, tseries_labels, traces_out, valid_idx


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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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


def compute_response_qi(trace_list):
    """
    Signal-to-noise quality index across trial repetitions (Baden et al. 2016).

        QI = Var_t[ mean_r(C) ] / mean_r[ Var_t(C) ]

    C is a (T × R) matrix — time points × trial replicates.

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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_heatmap(traces, labels, mean_fps, title, save_path,
                         cluster_means=None, normalize=False):
    """
    Heat map of ROI responses sorted by cluster, with optional dendrogram.

    Parameters
    ----------
    normalize : bool
        If False (default): show raw ΔF/F — amplitude variation is preserved,
        consistent with feature clustering where amplitude features (peak, trough,
        AUC) were used.
        If True: normalise each ROI to its peak (max|ΔF/F|=1) — appropriate
        only for the 'spca' feature set where sPCA was fit on normalised traces.
    """
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

    mat  = np.vstack([
        _norm_trace(traces[i][:min_len], mean_fps) if normalize
        else traces[i][:min_len].astype(float)
        for i in sorted_idx
    ])
    vmax = float(np.percentile(np.abs(mat), 95))
    cbar_label = 'ΔF/F (norm. to peak)' if normalize else 'ΔF/F'

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
    im = ax_hm.imshow(mat, aspect='auto', cmap='jet', vmin=-vmax, vmax=vmax,
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def analyze_response_clustering(condition_rois, fly_rois, segments, mean_fps,
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
    for fset in feature_sets:
        print(f"    Feature set: {fset}")
        X, fly_lbls, ts_lbls, traces, _ = prepare_clustering_data(
            condition_rois, fset, segments, mean_fps,
            polarity_results, luminance_results, contrast_results, frequency_results)

        if X is None or len(X) < 10:
            print(f"      Skipping: insufficient data ({0 if X is None else len(X)} ROIs)")
            continue

        pfx = f'condition_{fset}'

        # ---- GMM + BIC (primary clustering) --------------------------------
        gmm_labels, best_k_gmm, posteriors, gmm_diag = run_gmm_bic(X, k_range_gmm)

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
            save_path=os.path.join(output_dir, f'clustering_traces_gmm_{pfx}.png'),
            seg_window=_seg_window(fset), feature_subtitle=_subtitle(fset),
            ylim=_ylim(fset))

        # Cluster means for dendrogram: use same representation as heatmap
        _do_norm = (fset == 'spca')
        unique_gmm = sorted(set(gmm_labels))
        min_len = min(len(t) for t in traces)
        cmeans  = [np.mean([
                       _norm_trace(traces[i][:min_len], mean_fps) if _do_norm
                       else traces[i][:min_len].astype(float)
                       for i, l in enumerate(gmm_labels) if l == c], axis=0)
                   for c in unique_gmm]

        # Heatmap sorted by GMM cluster — dendrogram embedded on the right
        plot_cluster_heatmap(traces, gmm_labels, mean_fps,
            title=f'Condition | {fset} | GMM heatmap (k={best_k_gmm})',
            save_path=os.path.join(output_dir, f'clustering_heatmap_gmm_{pfx}.png'),
            cluster_means=cmeans if len(cmeans) > 1 else None,
            normalize=_do_norm)

        # Bootstrap co-assignment confusion matrix
        plot_bootstrap_confusion(co_mat, gmm_labels,
            title=f'Condition | {fset} | Bootstrap co-assignment (k={best_k_gmm})',
            save_path=os.path.join(output_dir, f'clustering_confusion_gmm_{pfx}.png'))


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
    #         plot_cluster_heatmap(traces_f, gmm_lbl_f, mean_fps,
    #             title=f'Fly {fly_id} | {fset} | GMM heatmap',
    #             save_path=os.path.join(output_dir, f'clustering_heatmap_gmm_{pfx_f}.png'),
    #             cluster_means=cmeans_f if len(cmeans_f) > 1 else None,
    #             normalize=_do_norm_f)
    #         plot_bootstrap_confusion(co_mat_f, gmm_lbl_f,
    #             title=f'Fly {fly_id} | {fset} | Bootstrap co-assignment (k={best_k_f})',
    #             save_path=os.path.join(output_dir, f'clustering_confusion_gmm_{pfx_f}.png'))

    print("\n  Clustering analysis complete.")


def plot_roi_repetitions(roi_accumulator, qi_pass, mean_fps,
                         stim_protocol, stim_time, output_dir, ylim=None,
                         seed=42):
    """
    For each fly, pick one random QI-passing ROI and plot all its TSeries
    repetitions as faint individual traces + mean ± SEM, with the stimulus
    protocol on top.  Layout mirrors the 1_tseries plots.

    Saved as: 0_roi_repetitions_<fly_id>.png
    (prefix 0_ so it sorts before the TSeries plots in the output folder)
    """
    print("\n" + "="*60)
    print("Plotting ROI repetitions per fly ...")
    rng = np.random.default_rng(seed)

    for fly_id, roi_dict in roi_accumulator.items():
        # Collect QI-passing ROI indices for this fly
        passing = [ri for ri in roi_dict
                   if qi_pass.get((fly_id, ri), False)]

        if not passing:
            print(f"  FlyID {fly_id}: no QI-passing ROIs, skipping")
            continue

        # Pick one at random
        chosen_roi_idx = int(rng.choice(passing))
        trace_list     = roi_dict[chosen_roi_idx]
        n_reps         = len(trace_list)

        # Align all repetitions to minimum length and stimulus length
        stim_length = len(stim_protocol)
        min_len = min(min(len(t) for t in trace_list), stim_length)
        traces  = [t[:min_len] for t in trace_list]
        time    = np.arange(min_len) / mean_fps

        qi_val = compute_response_qi(trace_list)

        fig = plt.figure(figsize=(16, 8))
        gs  = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)

        max_time_r = max(time[-1], stim_time[-1])

        # Stimulus panel
        ax_stim = fig.add_subplot(gs[0])
        ax_stim.fill_between(stim_time, 0, stim_protocol,
                             color='lightblue', alpha=0.7, linewidth=0)
        ax_stim.plot(stim_time, stim_protocol, color='blue', linewidth=1.5)
        ax_stim.set_xlim(0, max_time_r)
        ax_stim.set_ylim(-0.1, 1.1)
        ax_stim.axis('off')
        ax_stim.set_title(
            f'FlyID: {fly_id}  |  ROI {chosen_roi_idx}  '
            f'|  {n_reps} TSeries repetitions  |  QI = {qi_val:.3f}',
            fontsize=14, fontweight='bold', pad=10)

        # Trace panel
        ax_trace = fig.add_subplot(gs[1])

        # Faint individual repetitions — all light grey
        for rep_idx, tr in enumerate(traces):
            ax_trace.plot(time, tr, alpha=0.4, color='lightgray',
                          linewidth=0.8, label=f'Rep {rep_idx + 1}' if rep_idx == 0 else '_nolegend_')

        # Mean ± SEM across repetitions
        mat       = np.vstack(traces)
        mean_tr   = np.mean(mat, axis=0)
        sem_tr    = np.std(mat,  axis=0) / np.sqrt(n_reps)
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
        ax_trace.axhline(y=0, color='black', linestyle='--',
                         linewidth=0.8, alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f'0_roi_repetitions_{fly_id}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  FlyID {fly_id}: plotted ROI {chosen_roi_idx} "
              f"({n_reps} reps, QI={qi_val:.3f})")

    print("  ROI repetition plots complete.")


def process_pkl_file(pkl_path, metadata, multi, degen, mean_fps, original_fps=None, output_dir='output_plots', ylims=None, qi_threshold=0.45):
    """
    Process PKL file with fluorescence data and create hierarchical plots.
    
    Parameters:
    -----------
    pkl_path : str
        Path to the PKL file
    metadata : pandas.DataFrame
        Metadata dataframe containing fps information for each TSeries
    mean_fps : float
        Target FPS for interpolation
    original_fps : float, optional
        Original FPS (if None, will extract from metadata)
    output_dir : str
        Directory to save output plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create stimulus protocol
    stim_protocol, stim_time, segments = create_stimulus_protocol(mean_fps)
    
    # Load PKL file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Found {len(data)} TSeries recordings")
    
    # Extract condition information from filename
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
    
        # Filter metadata for this condition
        metadata_cond = metadata[metadata['Retinal'] == Retinal]
        metadata_cond = metadata_cond[metadata_cond['region'] == region]
        metadata_cond = metadata_cond[metadata_cond['visual_stim'] == visual_stim]
        metadata_cond = metadata_cond[metadata_cond['LED'] == led]
    ##############################################
    elif degen:
        condition_str = pkl_path.split('\\')[-1].split('.')[0]
        visual_stim = condition_str.split('_')[2]
        neuron_type = condition_str.split('_')[1]
        metadata_cond = metadata[metadata['visual_stim'] == visual_stim]
        metadata_cond = metadata_cond[metadata_cond['genotype'] == neuron_type]


    ##############################################
    # Storage for hierarchical data
    tseries_means = {}
    fly_means = defaultdict(list)
    all_means = []
    
    # Storage for segmented data
    segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    tseries_segments = {seg: {} for seg in segment_names}
    fly_segments = {seg: defaultdict(list) for seg in segment_names}
    all_segments = {seg: [] for seg in segment_names}

    # Storage for individual ROI traces per TSeries (for QI computation + clustering)
    # roi_accumulator[fly_id][roi_idx] -> list of traces (one per TSeries replicate)
    roi_accumulator = defaultdict(lambda: defaultdict(list))
    # tseries_raw[tseries_key] = {'traces': array, 'time': array, 'fly_id': str, 'idx': int}
    tseries_raw = {}

    # Process each TSeries — interpolate and store raw per-ROI traces only.
    # Mean computation is deferred until after QI filtering below.
    for tseries_idx, (tseries_key, tseries_data) in enumerate(data.items()):
        print(f"\nProcessing TSeries: {tseries_key}")
        tseries_number = tseries_key.split('-')[-1]
        if tseries_number[1] == '0':
            tseries_number = tseries_number[2]
        else:
            tseries_number = tseries_number[1:]
        
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
            interp_trace, time = interpolate_trace(df_trace, roi_fps, mean_fps, trim_seconds=5)
            interpolated_traces.append(interp_trace)
            common_time = time
        
        stim_length = len(stim_protocol)
        min_length = min(min(len(trace) for trace in interpolated_traces), stim_length)
        interpolated_traces = [trace[:min_length] for trace in interpolated_traces]
        common_time = common_time[:min_length]
        interpolated_traces = np.array(interpolated_traces)

        # Accumulate per-ROI traces for QI computation
        for roi_idx, roi_trace in enumerate(interpolated_traces):
            roi_accumulator[fly_id][roi_idx].append(roi_trace)

        # Store raw TSeries data for deferred mean computation after QI filtering
        tseries_raw[tseries_key] = {
            'traces': interpolated_traces,
            'time':   common_time,
            'fly_id': fly_id,
            'idx':    tseries_idx,
        }

    # ---- Compute QI for every ROI across its replicates -------------------
    print("\n" + "="*60)
    print("Computing response quality index (QI) per ROI ...")
    qi_map   = {}   # (fly_id, roi_idx) -> QI value
    qi_pass  = {}   # (fly_id, roi_idx) -> bool
    qi_all   = []
    for fly_id_k, roi_dict in roi_accumulator.items():
        for roi_idx_k, trace_list in roi_dict.items():
            qi = compute_response_qi(trace_list)
            qi_map[(fly_id_k, roi_idx_k)]  = qi
            qi_pass[(fly_id_k, roi_idx_k)] = qi >= qi_threshold
            qi_all.append(qi)
    n_pass = sum(qi_pass.values())
    print(f"  {n_pass}/{len(qi_all)} ROIs pass QI ≥ {qi_threshold}")

    # QI distribution plot
    if qi_all:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(qi_all, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(qi_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold = {qi_threshold}')
        ax.set_xlabel('Quality Index (QI)', fontsize=12)
        ax.set_ylabel('Number of ROIs', fontsize=12)
        ax.set_title(f'Response Quality Index Distribution\n'
                     f'{n_pass}/{len(qi_all)} ROIs pass threshold', fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering_qi_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ---- Build QI-filtered means and feed all downstream analyses ----------
    # For each TSeries, keep only ROIs whose (fly_id, roi_idx) passes QI,
    # then compute mean across the surviving ROIs.  This ensures mean traces,
    # segment analyses, and clustering all use the same filtered population.
    print("Building QI-filtered mean traces ...")
    for tseries_key, raw in tseries_raw.items():
        fly_id   = raw['fly_id']
        traces   = raw['traces']      # shape (n_rois, n_frames)
        time     = raw['time']
        ts_idx   = raw['idx']

        # Select rows whose ROI index passes QI
        keep_idx = [ri for ri in range(len(traces))
                    if qi_pass.get((fly_id, ri), False)]

        if len(keep_idx) == 0:
            print(f"  {tseries_key}: all ROIs failed QI, skipping")
            continue

        filtered_traces = traces[keep_idx]
        n_filtered = len(filtered_traces)
        n_total    = len(traces)
        print(f"  {tseries_key}: {n_filtered}/{n_total} ROIs kept (QI≥{qi_threshold})")

        # Plot 1: filtered ROIs + mean
        fig, mean_trace = plot_traces_with_mean(
            filtered_traces, time, stim_protocol, stim_time,
            f'TSeries: {tseries_key} | FlyID: {fly_id} | QI-filtered {n_filtered}/{n_total} ROIs',
            ylabel='ΔF/F',
            ylim=ylims.get('full') if ylims else None
        )
        plt.savefig(os.path.join(output_dir,
                    f'1_tseries_{ts_idx:03d}_{tseries_key}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        tseries_means[tseries_key] = (mean_trace, time, fly_id)
        fly_means[fly_id].append((mean_trace, time))
        all_means.append((mean_trace, time))

        segmented = segment_trace(mean_trace, time, segments, mean_fps)
        for seg_name, (seg_trace, seg_time) in segmented.items():
            if len(seg_trace) == 0:
                continue
            tseries_segments[seg_name][tseries_key] = (seg_trace, seg_time, fly_id)
            fly_segments[seg_name][fly_id].append((seg_trace, seg_time))
            all_segments[seg_name].append((seg_trace, seg_time))

    # ---- Build clustering registries: QI-filtered + TSeries-averaged -------
    print("\n" + "="*60)
    print("Building clustering registries (QI-filtered, TSeries-averaged) ...")
    condition_rois = []
    fly_rois       = defaultdict(list)
    qi_passed      = 0
    for fly_id_k, roi_dict in roi_accumulator.items():
        n_averaged = 0
        for roi_idx_k, trace_list in roi_dict.items():
            if not qi_pass.get((fly_id_k, roi_idx_k), False):
                continue
            min_len   = min(len(t) for t in trace_list)
            avg_trace = np.mean([t[:min_len] for t in trace_list], axis=0)
            entry = (avg_trace, fly_id_k, f'roi{roi_idx_k}')
            condition_rois.append(entry)
            fly_rois[fly_id_k].append(entry)
            n_averaged += 1
            qi_passed  += 1
        print(f"  FlyID {fly_id_k}: {n_averaged} ROIs averaged across "
              f"{len(next(iter(roi_dict.values())))} TSeries replicates")
    print(f"  Total ROIs for clustering: {len(condition_rois)}")

    # ---- Plot one random QI-passing ROI per fly over all repetitions -------
    plot_roi_repetitions(
        roi_accumulator, qi_pass, mean_fps,
        stim_protocol, stim_time, output_dir,
        ylim=ylims.get('full') if ylims else None)

    # ---- Segment the replicate-averaged ROI traces for plot 5a -------------
    # roi_segments[seg_name] = list of (seg_trace, seg_time)
    # one entry per QI-passing ROI — used for the individual-ROI global plot
    roi_segments = {seg: [] for seg in segment_names}
    for avg_trace, fly_id_r, roi_key in condition_rois:
        seg_result = segment_trace(avg_trace,
                                   np.arange(len(avg_trace)) / mean_fps,
                                   segments, mean_fps)
        for seg_name, (seg_tr, seg_t) in seg_result.items():
            if len(seg_tr) > 0:
                roi_segments[seg_name].append((seg_tr, seg_t))

    # Plot 2: Averages per FlyID
    print("\n" + "="*60)
    print("Creating FlyID-level plots...")
    fly_averaged_traces = {}
    
    for fly_id, traces_and_times in fly_means.items():
        print(f"\nProcessing FlyID: {fly_id} ({len(traces_and_times)} TSeries)")
        
        # Align traces to minimum length
        min_length = min(len(trace) for trace, _ in traces_and_times)
        aligned_traces = [trace[:min_length] for trace, _ in traces_and_times]
        common_time = traces_and_times[0][1][:min_length]
        
        # Plot
        fig, mean_trace = plot_traces_with_mean(
            aligned_traces,
            common_time,
            stim_protocol,
            stim_time,
            f'FlyID: {fly_id} | Average across {len(aligned_traces)} TSeries',
            ylabel='ΔF/F',
            ylim=ylims.get('full') if ylims else None
        )
        plt.savefig(os.path.join(output_dir, f'2_fly_{fly_id}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        fly_averaged_traces[fly_id] = (mean_trace, common_time)
    
    # Plot 3: Global average across all TSeries
    print("\n" + "="*60)
    print("Creating global average plot...")
    
    min_length = min(len(trace) for trace, _ in all_means)
    all_traces = [trace[:min_length] for trace, _ in all_means]
    common_time = all_means[0][1][:min_length]
    
    fig, global_mean = plot_traces_with_mean(
        all_traces,
        common_time,
        stim_protocol,
        stim_time,
        f'Global Average | All {len(all_traces)} TSeries from {len(fly_means)} Flies',
        ylabel='ΔF/F',
        ylim=ylims.get('full') if ylims else None
    )
    plt.savefig(os.path.join(output_dir, f'3_global_average.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print(f"Analysis complete! Plots saved to '{output_dir}'")
    print(f"  - {len(tseries_means)} TSeries plots")
    print(f"  - {len(fly_means)} FlyID plots")
    print(f"  - 1 Global average plot")
    
    # Analyze segments for each fly
    print("\n" + "="*60)
    print("Analyzing stimulus segments...")
    
    segment_results = {}
    for seg_name in segment_names:
        print(f"\nSegment: {seg_name.upper()}")
        segment_results[seg_name] = {
            'fly_averages': {},
            'global_average': None
        }
        
        # Get segment boundaries for plotting
        seg_start, seg_end = segments[seg_name]
        seg_stim_start_idx = int(seg_start * mean_fps)
        seg_stim_end_idx = int(seg_end * mean_fps)
        if seg_stim_end_idx > len(stim_protocol):
            seg_stim_end_idx = len(stim_protocol)
        seg_stim = stim_protocol[seg_stim_start_idx:seg_stim_end_idx]
        seg_stim_time = np.arange(len(seg_stim)) / mean_fps
        seg_ylim = ylims.get(seg_name) if ylims else None

        # Average per fly
        for fly_id, traces_and_times in fly_segments[seg_name].items():
            min_length = min(len(trace) for trace, _ in traces_and_times)
            aligned_traces = [trace[:min_length] for trace, _ in traces_and_times]
            common_time = traces_and_times[0][1][:min_length]
            
            mean_trace = np.mean(aligned_traces, axis=0)
            segment_results[seg_name]['fly_averages'][fly_id] = (mean_trace, common_time)
            
            # Plot fly average for this segment
            fig, _ = plot_traces_with_mean(
                aligned_traces,
                common_time,
                seg_stim,
                seg_stim_time,
                f'{seg_name.upper()} | FlyID: {fly_id} | {len(aligned_traces)} TSeries',
                ylabel='ΔF/F',
                ylim=seg_ylim
            )
            plt.savefig(os.path.join(output_dir, f'4_{seg_name}_fly_{fly_id}.png'), 
                        dpi=150, bbox_inches='tight')
            plt.close()
        
        # Global average for this segment
        min_length = min(len(trace) for trace, _ in all_segments[seg_name])
        all_traces = [trace[:min_length] for trace, _ in all_segments[seg_name]]
        common_time = all_segments[seg_name][0][1][:min_length]
        
        global_mean = np.mean(all_traces, axis=0)
        segment_results[seg_name]['global_average'] = (global_mean, common_time)

        # ── Plot 5a: individual QI-filtered replicate-averaged ROIs + global mean ──
        if roi_segments[seg_name]:
            roi_min_len = min(len(tr) for tr, _ in roi_segments[seg_name])
            roi_traces  = [tr[:roi_min_len] for tr, _ in roi_segments[seg_name]]
            roi_time    = roi_segments[seg_name][0][1][:roi_min_len]
            fig, _ = plot_traces_with_mean(
                roi_traces, roi_time,
                seg_stim, seg_stim_time,
                f'{seg_name.upper()} | All ROIs (QI-filtered, rep.-avg.) '
                f'| n={len(roi_traces)} ROIs',
                ylabel='ΔF/F', ylim=seg_ylim
            )
            plt.savefig(os.path.join(output_dir,
                        f'5a_{seg_name}_global_rois.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        # ── Plot 5b: per-fly averages + global mean from fly averages ─────────
        fly_avg_traces = []
        fly_avg_time   = None
        for fly_id_p, (fly_mean_tr, fly_mean_t) in \
                segment_results[seg_name]['fly_averages'].items():
            fly_avg_traces.append(fly_mean_tr)
            fly_avg_time = fly_mean_t

        if fly_avg_traces and fly_avg_time is not None:
            fly_min_len   = min(len(t) for t in fly_avg_traces)
            fly_avg_traces_aligned = [t[:fly_min_len] for t in fly_avg_traces]
            fly_avg_time_aligned   = fly_avg_time[:fly_min_len]
            fig, _ = plot_traces_with_mean(
                fly_avg_traces_aligned, fly_avg_time_aligned,
                seg_stim, seg_stim_time,
                f'{seg_name.upper()} | Per-fly averages '
                f'| n={len(fly_avg_traces_aligned)} flies',
                ylabel='ΔF/F', ylim=seg_ylim
            )
            plt.savefig(os.path.join(output_dir,
                        f'5b_{seg_name}_global_flies.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  - {len(fly_segments[seg_name])} fly plots")
        print(f"  - 2 global plots (5a: ROI-level, 5b: fly-level)")
    
    # Analyze luminance responses
    luminance_results = analyze_luminance_response(segment_results, fly_segments, mean_fps, output_dir)
    
    # Analyze polarity responses
    polarity_results = analyze_polarity_response(fly_segments, mean_fps, output_dir)
    
    # Analyze contrast responses
    contrast_results = analyze_contrast_response(fly_segments, mean_fps, output_dir)
    
    frequency_results = analyze_frequency_response(fly_segments, mean_fps, output_dir)
    
    variability_results = analyze_within_fly_variability(fly_segments, mean_fps, output_dir)
    
    across_fly_variability = analyze_across_fly_variability(fly_segments, mean_fps, output_dir)
    
    # Run response profile clustering
    analyze_response_clustering(
        condition_rois, fly_rois, segments, mean_fps,
        stim_protocol, stim_time, output_dir,
        polarity_results=polarity_results,
        luminance_results=luminance_results,
        contrast_results=contrast_results,
        frequency_results=frequency_results,
        ylims=ylims)

    return tseries_means, fly_averaged_traces, global_mean, segment_results, luminance_results, polarity_results, contrast_results, frequency_results, variability_results, across_fly_variability, condition_rois, mean_fps

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
    plt.savefig(os.path.join(output_dir, 'combined_cross_type_umap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Cross-type UMAP saved.")


def plot_functional_fingerprint(all_results, condition_names, output_dir):
    """
    Two-panel figure:
      Left  — Functional fingerprint heatmap: rows = cell types,
               columns = response indices (z-scored across types).
               Raw values shown as text in each cell.
      Right — Within-type vs between-type variance decomposition:
               bar chart of variance explained by type identity for
               each response index.
    """
    print("\n" + "="*60)
    print("Building functional fingerprint ...")

    # ── collect response indices per type ────────────────────────────────
    # For each index we want fly-level values so we can compute within-type SD.
    # Structure: index_name -> {type: [fly_values]}
    index_definitions = [
        ('ON amplitude',     lambda r: list(r['polarity_results']['fly_on_mean'].values())),
        ('OFF amplitude',    lambda r: list(r['polarity_results']['fly_off_mean'].values())),
        ('Luminance slope',  lambda r: list(r['luminance_results']['fly_slopes'].values())),
        ('Freq. peak (Hz)',  _extract_freq_peak),
        ('Contrast slope',   _extract_contrast_slope),
        ('Within-fly CV',    lambda r: [np.mean(list(r['variability_results']['fly_cv'][s].values()))
                                        for s in ['polarity','frequency','contrast','luminance']
                                        if r['variability_results']['fly_cv'][s]]),
        ('Reliability',      lambda r: [np.mean(list(r['variability_results']['fly_reliability'][s].values()))
                                        for s in ['polarity','frequency','contrast','luminance']
                                        if r['variability_results']['fly_reliability'][s]]),
    ]

    index_names  = [d[0] for d in index_definitions]
    n_idx        = len(index_names)
    n_types      = len(condition_names)

    # type_means[type_idx, idx] and type_sems, plus fly-level for variance decomp
    type_means = np.full((n_types, n_idx), np.nan)
    type_sems  = np.full((n_types, n_idx), np.nan)
    fly_values = {name: {cond: [] for cond in condition_names}
                  for name in index_names}

    for ti, (results, cond) in enumerate(zip(all_results, condition_names)):
        for ii, (iname, ifn) in enumerate(index_definitions):
            try:
                vals = ifn(results)
                vals = [v for v in vals if np.isfinite(v)]
                if len(vals) == 0:
                    continue
                type_means[ti, ii] = np.mean(vals)
                type_sems[ti, ii]  = np.std(vals) / np.sqrt(len(vals))
                fly_values[iname][cond] = vals
            except Exception:
                pass

    # z-score each column for the heatmap colour scale
    col_mean = np.nanmean(type_means, axis=0)
    col_std  = np.nanstd(type_means,  axis=0)
    col_std[col_std == 0] = 1.0
    Z = (type_means - col_mean) / col_std

    # ── within vs between variance ────────────────────────────────────────
    variance_ratio = np.full(n_idx, np.nan)
    for ii, iname in enumerate(index_names):
        all_fly_vals = [v for cond in condition_names
                        for v in fly_values[iname][cond]]
        if len(all_fly_vals) < 2:
            continue
        between_var = np.nanvar(type_means[:, ii])
        within_var  = np.nanmean([np.var(fly_values[iname][c])
                                  for c in condition_names
                                  if len(fly_values[iname][c]) > 1])
        total = between_var + within_var
        variance_ratio[ii] = between_var / total if total > 0 else 0.0

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                    figsize=(5 + n_idx * 1.0, max(6, n_types * 0.9 + 2)),
                                    gridspec_kw={'width_ratios': [3, 1]})

    # Left: heatmap
    vmax = np.nanmax(np.abs(Z))
    im = ax1.imshow(Z, cmap='RdBu_r', aspect='auto',
                    vmin=-vmax, vmax=vmax)

    # Annotate cells with raw mean ± SEM
    for ti in range(n_types):
        for ii in range(n_idx):
            m = type_means[ti, ii]
            s = type_sems[ti, ii]
            if np.isfinite(m):
                txt = f'{m:.2f}\n±{s:.2f}'
                bg  = Z[ti, ii]
                fc  = 'white' if abs(bg) > 0.8 * vmax else 'black'
                ax1.text(ii, ti, txt, ha='center', va='center',
                         fontsize=7, color=fc, fontweight='bold')

    ax1.set_xticks(range(n_idx))
    ax1.set_xticklabels(index_names, rotation=40, ha='right', fontsize=9)
    ax1.set_yticks(range(n_types))
    ax1.set_yticklabels(condition_names, fontsize=10)
    ax1.set_title('Functional Fingerprint\n(z-scored across types)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='z-score', shrink=0.6)

    # Right: variance explained by type identity
    colors_bar = ['#2ecc71' if r >= 0.5 else '#e74c3c' if r < 0.25 else '#f39c12'
                  for r in variance_ratio]
    bars = ax2.barh(range(n_idx), variance_ratio, color=colors_bar,
                    edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
                label='50% threshold')
    for i, v in enumerate(variance_ratio):
        if np.isfinite(v):
            ax2.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
    ax2.set_yticks(range(n_idx))
    ax2.set_yticklabels(index_names, fontsize=9)
    ax2.set_xlabel('Variance explained\nby cell type identity', fontsize=9)
    ax2.set_title('Between-type\nvs total variance', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(fontsize=8)

    plt.suptitle('Cross-type functional comparison', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_functional_fingerprint.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Functional fingerprint saved.")


def _extract_freq_peak(results):
    """Return list of per-fly frequency-tuning peak (Hz)."""
    freq_res  = results['frequency_results']
    fly_resps = freq_res.get('fly_responses', {})
    peaks = []
    for fly_id, freq_dict in fly_resps.items():
        if not freq_dict:
            continue
        best_freq = max(freq_dict, key=lambda f: freq_dict[f])
        peaks.append(float(best_freq))
    return peaks


def _extract_contrast_slope(results):
    """Return list of per-fly linear slope of the absolute contrast response."""
    cont_res  = results['contrast_results']
    fly_resps = cont_res.get('fly_responses_absolute', {})
    slopes = []
    for fly_id, cont_dict in fly_resps.items():
        if len(cont_dict) < 2:
            continue
        xs = np.array(sorted(cont_dict.keys()), dtype=float)
        ys = np.array([cont_dict[x] for x in xs], dtype=float)
        if len(xs) >= 2 and np.all(np.isfinite(ys)):
            slope, _ = np.polyfit(xs, ys, 1)
            slopes.append(float(slope))
    return slopes


def plot_combined_analysis(all_results, pkl_files, output_dir):
    """
    Create combined plots across multiple PKL files.
    
    Parameters:
    -----------
    all_results : list of dicts
        List of result dictionaries from each PKL file
    pkl_files : list of str
        List of PKL file paths
    output_dir : str
        Directory to save plots
    """
    print("\n" + "="*60)
    print("Creating combined analysis plots...")
    
    # Extract condition names from file paths
    condition_names = [path.split('\\')[-1].split('.')[0] for path in pkl_files]
    
    # Define colors for each condition
    colors = plt.cm.tab10(np.linspace(0, 1, len(pkl_files)))
    
    # ============== LUMINANCE COMBINED PLOT ==============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Luminance response curves
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        lum_results = results['luminance_results']
        x_vals = sorted(lum_results['overall_mean'].keys())
        y_vals = [lum_results['overall_mean'][x] for x in x_vals]
        y_sem = [lum_results['overall_sem'][x] for x in x_vals]
        
        ax1.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
                     markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
    ax1.set_xlabel('Luminance Value', fontsize=12)
    ax1.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
    ax1.set_title('Luminance Response Curves', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Right: Slope comparison
    x_positions = np.arange(len(all_results))
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        lum_results = results['luminance_results']
        mean_slope = lum_results['mean_slope']
        sem_slope = lum_results['sem_slope']
        
        ax2.bar(x_positions[idx], mean_slope, yerr=sem_slope, color=color,
                edgecolor='black', linewidth=1.5, capsize=5, width=0.7, alpha=0.8)
        
        # Overlay individual fly data
        fly_slopes = list(lum_results['fly_slopes'].values())
        x_scatter = np.full(len(fly_slopes), x_positions[idx])
        ax2.scatter(x_scatter, fly_slopes, color='black', s=40, alpha=0.5, zorder=3)
    
    ax2.set_ylabel('Slope (ΔF/F per luminance unit)', fontsize=12)
    ax2.set_title('Luminance Response Slopes', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_luminance_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============== POLARITY COMBINED PLOT ==============
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(all_results))
    width = 0.35
    
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        pol_results = results['polarity_results']
        
        # ON responses
        on_mean = pol_results['overall_on_mean']
        on_sem = pol_results['overall_on_sem']
        ax.bar(x_positions[idx] - width/2, on_mean, width, yerr=on_sem,
               color=color, alpha=0.7, edgecolor='black', linewidth=1.5,
               capsize=4, label=f'{condition} - ON' if idx == 0 else '')
        
        # OFF responses
        off_mean = pol_results['overall_off_mean']
        off_sem = pol_results['overall_off_sem']
        ax.bar(x_positions[idx] + width/2, off_mean, width, yerr=off_sem,
               color=color, alpha=0.4, edgecolor='black', linewidth=1.5,
               capsize=4, label=f'{condition} - OFF' if idx == 0 else '')
        
        # Overlay individual fly data
        fly_on = list(pol_results['fly_on_mean'].values())
        fly_off = list(pol_results['fly_off_mean'].values())
        ax.scatter(np.full(len(fly_on), x_positions[idx] - width/2), fly_on,
                   color='black', s=30, alpha=0.5, zorder=3)
        ax.scatter(np.full(len(fly_off), x_positions[idx] + width/2), fly_off,
                   color='black', s=30, alpha=0.5, zorder=3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    for condition, color in zip(condition_names, colors):
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=condition))
    legend_elements.extend([
        Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='ON'),
        Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='OFF')
    ])
    ax.legend(handles=legend_elements, fontsize=9)
    
    ax.set_ylabel('Maximum Amplitude (ΔF/F)', fontsize=12)
    ax.set_title('Polarity Responses (ON vs OFF)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_polarity_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
     # ============== COMBINED SEGMENT TRACES ==============
    segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
    for seg_name in segment_names:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
        
        # Get stimulus protocol from first result (should be same for all)
        # We need to reconstruct it or pass it through - for now use placeholder
        # Top subplot: Stimulus protocol (without axes)
        ax_stim = fig.add_subplot(gs[0])
        ax_stim.axis('off')
        ax_stim.set_title(f'{seg_name.upper()} - Combined', fontsize=14, fontweight='bold', pad=10)
        
        # Bottom subplot: Fluorescence traces
        ax_trace = fig.add_subplot(gs[1])
        
        # Find minimum length across all conditions for this segment
        min_length = float('inf')
        for results in all_results:
            seg_data = results['segment_results'][seg_name]['global_average']
            if seg_data is not None:
                min_length = min(min_length, len(seg_data[0]))
        
        # Plot mean trace for each condition
        for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
            seg_data = results['segment_results'][seg_name]['global_average']
            if seg_data is not None:
                mean_trace = seg_data[0][:min_length]
                time = seg_data[1][:min_length]
                
                # Calculate SEM from fly averages
                fly_avgs = results['segment_results'][seg_name]['fly_averages']
                fly_traces = []
                for fly_id, (fly_trace, fly_time) in fly_avgs.items():
                    if len(fly_trace) >= min_length:
                        fly_traces.append(fly_trace[:min_length])
                
                if len(fly_traces) > 0:
                    fly_traces = np.array(fly_traces)
                    sem_trace = np.std(fly_traces, axis=0) / np.sqrt(len(fly_traces))
                    
                    # Plot mean with SEM
                    ax_trace.plot(time, mean_trace, color=color, linewidth=2, 
                                 label=condition, alpha=0.8)
                    ax_trace.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace,
                                         color=color, alpha=0.2)
        
        ax_trace.set_xlabel('Time (s)', fontsize=12)
        ax_trace.set_ylabel('ΔF/F', fontsize=12)
        ax_trace.legend(loc='upper right', fontsize=10)
        ax_trace.grid(True, alpha=0.3)
        ax_trace.set_xlim(time[0], time[-1])
        ax_trace.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'combined_segment_{seg_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print("  Combined segment traces saved!")

    # ============== FREQUENCY COMBINED PLOT ==============
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Frequency response curves
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        freq_results = results['frequency_results']
        x_vals = sorted(freq_results['overall_mean'].keys())
        y_vals = [freq_results['overall_mean'][x] for x in x_vals]
        y_sem = [freq_results['overall_sem'][x] for x in x_vals]
        
        ax.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
                     markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
    ax.set_xlabel('Temporal Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Maximum Response (ΔF/F)', fontsize=12)
    ax.set_title('Frequency Response Curves', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_frequency_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ============== WITHIN-FLY VARIABILITY COMBINED PLOT ==============
    segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(segment_names))
    width = 0.8 / len(all_results)
    
    # Top left: CV comparison
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        var_results = results['variability_results']
        cv_means = [var_results['overall_cv'].get(seg, 0) for seg in segment_names]
        cv_sems = [var_results['overall_cv_sem'].get(seg, 0) for seg in segment_names]
        
        offset = (idx - len(all_results)/2 + 0.5) * width
        ax1.bar(x_pos + offset, cv_means, width, yerr=cv_sems, 
                color=color, alpha=0.7, edgecolor='black', linewidth=1,
                capsize=3, label=condition)
    
    ax1.set_ylabel('Coefficient of Variation', fontsize=12)
    ax1.set_title('Within-Fly Variability (CV)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9)
    
    # Top right: Reliability comparison
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        var_results = results['variability_results']
        rel_means = [var_results['overall_reliability'].get(seg, 0) for seg in segment_names]
        rel_sems = [var_results['overall_reliability_sem'].get(seg, 0) for seg in segment_names]
        
        offset = (idx - len(all_results)/2 + 0.5) * width
        ax2.bar(x_pos + offset, rel_means, width, yerr=rel_sems,
                color=color, alpha=0.7, edgecolor='black', linewidth=1,
                capsize=3, label=condition)
    
    ax2.set_ylabel('Mean Pairwise Correlation', fontsize=12)
    ax2.set_title('Response Reliability', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_within_fly_variability_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(segment_names))
    width = 0.8 / len(all_results)
    # Bottom: Individual fly scatter plots
    for seg_idx, seg_name in enumerate(segment_names):
        # ax = ax3 if seg_idx < 2 else ax4
        if seg_idx == 0:
            ax=ax1
        elif seg_idx == 1:
            ax=ax2
        elif seg_idx == 2:
            ax=ax3
        else:
            ax=ax4
        
        for cond_idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
            var_results = results['variability_results']
            fly_cv_values = list(var_results['fly_cv'][seg_name].values())
            fly_rel_values = list(var_results['fly_reliability'][seg_name].values())
            
            if len(fly_cv_values) > 0 and len(fly_rel_values) > 0:
                ax.scatter(fly_cv_values, fly_rel_values, color=color, s=60, 
                          alpha=0.6, label=condition if seg_idx == 0 else '')
        
        ax.set_xlabel('CV', fontsize=10)
        ax.set_ylabel('Reliability', fontsize=10)
        ax.set_title(f'{seg_name.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_ylim(0,1)
        ax.set_xlim(0,15)
        ax.grid(True, alpha=0.3)
        if seg_idx == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_within_fly_variability_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Combined variability analysis saved!")

    # ============== ACROSS-FLY VARIABILITY COMBINED PLOT ==============
    segment_names = ['polarity', 'frequency', 'contrast', 'luminance']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(segment_names))
    width = 0.8 / len(all_results)
    
    # Top left: CV comparison
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        across_var = results['across_fly_variability']
        cv_values = [across_var['segment_cv'].get(seg, 0) for seg in segment_names]
        
        offset = (idx - len(all_results)/2 + 0.5) * width
        ax1.bar(x_pos + offset, cv_values, width,
                color=color, alpha=0.7, edgecolor='black', linewidth=1,
                label=condition)
    
    ax1.set_ylabel('Coefficient of Variation', fontsize=12)
    ax1.set_title('Across-Fly Variability (CV)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9)
    
    # Top right: Reliability comparison
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        across_var = results['across_fly_variability']
        rel_values = [across_var['segment_reliability'].get(seg, 0) for seg in segment_names]
        
        offset = (idx - len(all_results)/2 + 0.5) * width
        ax2.bar(x_pos + offset, rel_values, width,
                color=color, alpha=0.7, edgecolor='black', linewidth=1,
                label=condition)
    
    ax2.set_ylabel('Mean Pairwise Correlation', fontsize=12)
    ax2.set_title('Across-Fly Reliability', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.capitalize() for s in segment_names], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_across_fly_variability_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Combined across-fly variability analysis saved!")

    # ============== CONTRAST COMBINED PLOTS ==============
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: Weber contrast curves
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        cont_results = results['contrast_results']
        x_vals = sorted(cont_results['overall_mean_weber'].keys())
        y_vals = [cont_results['overall_mean_weber'][x] for x in x_vals]
        y_sem = [cont_results['overall_sem_weber'][x] for x in x_vals]
        
        ax1.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
                     markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
    ax1.set_xlabel('Weber Contrast (%)', fontsize=12)
    ax1.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
    ax1.set_title('Weber Contrast Response (Signed)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(fontsize=9)
    
    # Top right: Absolute contrast curves
    for idx, (results, condition, color) in enumerate(zip(all_results, condition_names, colors)):
        cont_results = results['contrast_results']
        x_vals = sorted(cont_results['overall_mean_absolute'].keys())
        y_vals = [cont_results['overall_mean_absolute'][x] for x in x_vals]
        y_sem = [cont_results['overall_sem_absolute'][x] for x in x_vals]
        
        ax2.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o-', color=color,
                     markersize=6, capsize=4, linewidth=2, label=condition, alpha=0.8)
    
    ax2.set_xlabel('Absolute Contrast (%)', fontsize=12)
    ax2.set_ylabel('Response Magnitude (ΔF/F)', fontsize=12)
    ax2.set_title('Absolute Contrast Response', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(fontsize=9)
    
    # Bottom: Individual comparison plots (can be expanded based on needs)
    ax3.axis('off')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_contrast_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Combined plots saved!")

    # ============== CROSS-TYPE UMAP ==============
    # Reconstruct segments from first result's mean_fps (same chirp for all)
    first_fps = all_results[0].get('mean_fps', 10.0)
    _, _, cross_segments = create_stimulus_protocol(first_fps)
    plot_cross_type_umap(all_results, condition_names, cross_segments, output_dir)

    # ============== FUNCTIONAL FINGERPRINT ==============
    plot_functional_fingerprint(all_results, condition_names, output_dir)


# Example usage
if __name__ == "__main__":
    # Configuration
    multi = False
    degen = True
    if multi: 
        # pkl_files = [r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_ME_Y.pkl", r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_ME_Y.pkl"]
        # pkl_files = [r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LO_Y.pkl", r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LO_Y.pkl"]
        pkl_files = [r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LOP_Y.pkl", r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LOP_Y.pkl"]
    elif degen:
        # pkl_files = [r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Mi1_chirp.pkl", r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Mi4_chirp.pkl", r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Mi9_chirp.pkl"]
        pkl_files = [r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Tm1_chirp.pkl", r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Tm4_chirp.pkl", r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Tm9_chirp.pkl",\
                     r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Mi1_chirp.pkl", r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Mi4_chirp.pkl", r"D:\Christian\02_twophoton\250730_degen_var\3_DATA\_Mi9_chirp.pkl"]

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
        if multi:
            metadata = pd.read_excel("D:/Christian/02_twophoton/metadata.xlsx", sheet_name='251023_tdc2_cschr_pan')
            output_dir=f"D:/Christian/02_twophoton/251023_tdc2_cschr_pan/4_results/{folder}"
        elif degen:
            metadata = pd.read_excel("D:/Christian/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            output_dir=f"D:/Christian/02_twophoton/250730_degen_var/4_results/{folder}"
        os.makedirs(output_dir, exist_ok=True)
        mean_fps = pd.to_numeric(metadata['fps'], errors='coerce').dropna()
        mean_fps = mean_fps[mean_fps.between(1, 200)].mean()
        print(f"  mean_fps = {mean_fps:.4f} Hz")
        TARGET_FPS = mean_fps  # Target frames per second for interpolation
        ORIGINAL_FPS = None  # Set to None to extract from data, or specify (e.g., 30)
        
        # Run analysis
        tseries_means, fly_averaged_traces, global_mean, segment_results, luminance_results, polarity_results, contrast_results, frequency_results, variability_results, across_fly_variability, condition_rois, run_fps = process_pkl_file(
            PKL_FILE, 
            metadata,
            multi,
            degen,
            TARGET_FPS,
            original_fps=ORIGINAL_FPS,
            output_dir=output_dir,
            ylims=ylims,
            qi_threshold=0.45,
        )
        # Store results
        all_results.append({
            'luminance_results': luminance_results,
            'polarity_results': polarity_results,
            'contrast_results': contrast_results,
            'frequency_results': frequency_results,
            'segment_results': segment_results,
            'variability_results': variability_results,
            'across_fly_variability': across_fly_variability,
            'condition_rois': condition_rois,
            'mean_fps': run_fps,
            'condition_name': folder,
        })
    
    if multi:
        combined_output = "D:/Christian/02_twophoton/251023_tdc2_cschr_pan/4_results/combined"
    elif degen:
        combined_output = "D:/Christian/02_twophoton/250730_degen_var/4_results/combined"
    
    os.makedirs(combined_output, exist_ok=True)
    plot_combined_analysis(all_results, pkl_files, combined_output)
    print("\nDone!")
