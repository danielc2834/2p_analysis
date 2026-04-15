import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import defaultdict
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
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
    for name, (start, end) in segments.items():
        start_idx = int(start * mean_fps)
        end_idx = int(end * mean_fps)
        if end_idx > len(trace):
            end_idx = len(trace)
        
        trace_seg = trace[start_idx:end_idx]
        time_seg = time[start_idx:end_idx] - time[start_idx]  # Reset to start at 0
        segmented[name] = (trace_seg, time_seg)
    
    return segmented

def interpolate_trace(df_trace, original_fps, target_fps, trim_seconds=5):
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

def plot_traces_with_mean(traces, time, stim_protocol, stim_time, title, ylabel='ΔF/F', alpha=0.3):
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
    
    # Determine x-axis limits based on the maximum of trace and stimulus duration
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
    Extract a feature vector from a single ROI trace.

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

    # --- scalar metrics from already-computed analyses (optional) ---
    # ON amplitude
    on_amp  = np.nan
    off_amp = np.nan
    if polarity_results is not None:
        on_amp  = polarity_results.get('overall_on_mean',  np.nan)
        off_amp = polarity_results.get('overall_off_mean', np.nan)
    features.append(float(on_amp)  if np.isfinite(on_amp)  else 0.0); names.append('on_amplitude')
    features.append(float(off_amp) if np.isfinite(off_amp) else 0.0); names.append('off_amplitude')

    # Luminance slope
    lum_slope = np.nan
    if luminance_results is not None:
        lum_slope = luminance_results.get('mean_slope', np.nan)
    features.append(float(lum_slope) if np.isfinite(lum_slope) else 0.0); names.append('lum_slope')

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

    if len(X_rows) == 0:
        return None, None, None, None, None

    X = np.vstack(X_rows)
    return X, fly_labels, tseries_labels, traces_out, valid_idx


def run_umap_hdbscan(X, min_cluster_size=10, n_neighbors=15, min_dist=0.1):
    """UMAP embedding → HDBSCAN clustering."""
    reducer   = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                          min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 prediction_data=True)
    labels    = clusterer.fit_predict(embedding)
    return embedding, labels


def run_umap_kmeans(X, k_range=range(2, 9), n_neighbors=15, min_dist=0.1):
    """UMAP embedding → k-means sweep, return embedding + best labels + diagnostics."""
    reducer   = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                          min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X)

    inertias    = []
    sil_scores  = []
    all_labels  = {}

    for k in k_range:
        if k >= len(X):
            break
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls   = km.fit_predict(X)          # cluster in original space
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X, lbls)
        except Exception:
            sil = 0.0
        sil_scores.append(sil)
        all_labels[k] = lbls

    diagnostics = {'k_range': list(k_range)[:len(inertias)],
                   'inertias': inertias,
                   'silhouette': sil_scores,
                   'all_labels': all_labels}

    best_k    = diagnostics['k_range'][int(np.argmax(sil_scores))] if sil_scores else 2
    best_lbls = all_labels.get(best_k, np.zeros(len(X), dtype=int))
    return embedding, best_lbls, best_k, diagnostics


def run_pca_kmeans(X, k_range=range(2, 9)):
    """PCA embedding → k-means sweep (for per-fly scope)."""
    n_comp  = min(2, X.shape[0] - 1, X.shape[1])
    pca     = PCA(n_components=n_comp)
    embedding = pca.fit_transform(X)

    inertias   = []
    sil_scores = []
    all_labels = {}

    for k in k_range:
        if k >= len(X):
            break
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X, lbls)
        except Exception:
            sil = 0.0
        sil_scores.append(sil)
        all_labels[k] = lbls

    diagnostics = {'k_range': list(k_range)[:len(inertias)],
                   'inertias': inertias,
                   'silhouette': sil_scores,
                   'all_labels': all_labels}

    best_k    = diagnostics['k_range'][int(np.argmax(sil_scores))] if sil_scores else 2
    best_lbls = all_labels.get(best_k, np.zeros(len(X), dtype=int))
    return embedding, best_lbls, best_k, diagnostics


def plot_cluster_diagnostics(diagnostics, title, save_path):
    """Silhouette score and elbow (inertia) plots."""
    k_range  = diagnostics['k_range']
    if len(k_range) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_range, diagnostics['silhouette'], 'o-', color='steelblue', linewidth=2)
    best_idx = int(np.argmax(diagnostics['silhouette']))
    ax1.axvline(k_range[best_idx], color='red', linestyle='--',
                label=f'Best k={k_range[best_idx]}')
    ax1.set_xlabel('Number of clusters (k)', fontsize=12)
    ax1.set_ylabel('Silhouette score', fontsize=12)
    ax1.set_title('Silhouette Diagnostics', fontsize=13, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, diagnostics['inertias'], 'o-', color='coral', linewidth=2)
    ax2.set_xlabel('Number of clusters (k)', fontsize=12)
    ax2.set_ylabel('Inertia', fontsize=12)
    ax2.set_title('Elbow Curve', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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
                        title, save_path):
    """
    For each cluster: plot faint individual traces + mean ± SEM, stimulus on top.
    """
    unique_clusters = sorted(set(labels))
    # exclude noise label -1 for hdbscan
    plot_clusters = [c for c in unique_clusters if c != -1]
    if len(plot_clusters) == 0:
        return

    # align traces to min length
    min_len = min(len(t) for t in traces)
    time    = np.arange(min_len) / mean_fps

    n_clusters = len(plot_clusters)
    fig = plt.figure(figsize=(6 * n_clusters, 8))
    outer_gs = fig.add_gridspec(1, n_clusters, hspace=0.3, wspace=0.35)

    cmap = plt.cm.tab10
    colors = {c: cmap(i % 10) for i, c in enumerate(plot_clusters)}

    max_stim_time = stim_time[-1]

    for col_idx, cluster_id in enumerate(plot_clusters):
        mask   = np.array(labels) == cluster_id
        c_traces = [traces[i][:min_len] for i in range(len(traces)) if mask[i]]
        n_rois   = len(c_traces)

        inner_gs = outer_gs[col_idx].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.05)

        # Stimulus
        ax_stim = fig.add_subplot(inner_gs[0])
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

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_response_clustering(condition_rois, fly_rois, segments, mean_fps,
                                 stim_protocol, stim_time, output_dir,
                                 polarity_results=None, luminance_results=None,
                                 contrast_results=None, frequency_results=None):
    """
    Orchestrate all three clustering scopes × three feature sets.

    Parameters
    ----------
    condition_rois : list of (trace, fly_id, tseries_key)
        All ROIs for this condition.
    fly_rois : dict  fly_id -> list of (trace, fly_id, tseries_key)
        ROIs grouped by fly.
    """
    print("\n" + "="*60)
    print("Running response profile clustering...")

    seg_names    = list(segments.keys())
    feature_sets = ['full'] + [f'segment_{s}' for s in seg_names] + ['features']
    k_range      = range(2, 9)

    # ------------------------------------------------------------------ #
    # SCOPE 1 & 3: per-condition (UMAP)  +  shared space colored by fly   #
    # ------------------------------------------------------------------ #
    print("\n  Scope: per-condition (UMAP + HDBSCAN + k-means)")
    for fset in feature_sets:
        print(f"    Feature set: {fset}")
        X, fly_lbls, ts_lbls, traces, _ = prepare_clustering_data(
            condition_rois, fset, segments, mean_fps,
            polarity_results, luminance_results, contrast_results, frequency_results)

        if X is None or len(X) < 10:
            print(f"      Skipping {fset}: insufficient data ({0 if X is None else len(X)} ROIs)")
            continue

        # --- k-means diagnostics & best labels ---
        emb_km, best_labels_km, best_k, diag = run_umap_kmeans(X, k_range)

        plot_cluster_diagnostics(
            diag,
            title=f'Condition | {fset} | k-means diagnostics',
            save_path=os.path.join(output_dir, f'clustering_diag_condition_{fset}.png'))

        plot_embedding_scatter(
            emb_km, best_labels_km,
            title=f'Condition | {fset} | k-means (k={best_k})',
            save_path=os.path.join(output_dir, f'clustering_umap_condition_{fset}_kmeans.png'),
            hue_labels=fly_lbls, hue_title='Fly')

        plot_cluster_traces(
            traces, best_labels_km, mean_fps, stim_protocol, stim_time,
            title=f'Condition | {fset} | k-means (k={best_k})',
            save_path=os.path.join(output_dir, f'clustering_traces_condition_{fset}_kmeans.png'))

        # --- HDBSCAN ---
        emb_hdb, labels_hdb = run_umap_hdbscan(X)

        plot_embedding_scatter(
            emb_hdb, labels_hdb,
            title=f'Condition | {fset} | HDBSCAN',
            save_path=os.path.join(output_dir, f'clustering_umap_condition_{fset}_hdbscan.png'),
            hue_labels=fly_lbls, hue_title='Fly')

        plot_cluster_traces(
            traces, labels_hdb, mean_fps, stim_protocol, stim_time,
            title=f'Condition | {fset} | HDBSCAN',
            save_path=os.path.join(output_dir, f'clustering_traces_condition_{fset}_hdbscan.png'))

    # ------------------------------------------------------------------ #
    # SCOPE 2: per-fly independent (PCA + k-means)                        #
    # ------------------------------------------------------------------ #
    print("\n  Scope: per-fly (PCA + k-means)")
    for fly_id, roi_list in fly_rois.items():
        print(f"    Fly: {fly_id}  ({len(roi_list)} ROIs)")
        for fset in feature_sets:
            X, fly_lbls_f, ts_lbls_f, traces_f, _ = prepare_clustering_data(
                roi_list, fset, segments, mean_fps,
                polarity_results, luminance_results, contrast_results, frequency_results)

            if X is None or len(X) < 4:
                continue

            emb, best_labels, best_k, diag = run_pca_kmeans(X, k_range)

            plot_cluster_diagnostics(
                diag,
                title=f'Fly {fly_id} | {fset} | k-means diagnostics',
                save_path=os.path.join(output_dir,
                    f'clustering_diag_fly{fly_id}_{fset}.png'))

            plot_embedding_scatter(
                emb, best_labels,
                title=f'Fly {fly_id} | {fset} | k-means (k={best_k})',
                save_path=os.path.join(output_dir,
                    f'clustering_pca_fly{fly_id}_{fset}_kmeans.png'))

            plot_cluster_traces(
                traces_f, best_labels, mean_fps, stim_protocol, stim_time,
                title=f'Fly {fly_id} | {fset} | k-means (k={best_k})',
                save_path=os.path.join(output_dir,
                    f'clustering_traces_fly{fly_id}_{fset}_kmeans.png'))

    print("\n  Clustering analysis complete.")


def process_pkl_file(pkl_path, metadata, multi, degen, mean_fps, original_fps=None, output_dir='output_plots'):
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

    # Storage for individual ROI traces (for clustering)
    condition_rois = []              # list of (trace, fly_id, tseries_key)
    fly_rois       = defaultdict(list)  # fly_id -> list of (trace, fly_id, tseries_key)
    
    # Process each TSeries
    for tseries_idx, (tseries_key, tseries_data) in enumerate(data.items()):
        print(f"\nProcessing TSeries: {tseries_key}")
        # Extract TSeries number from key
        tseries_number = tseries_key.split('-')[-1]
        if tseries_number[1] == '0':
            tseries_number = tseries_number[2]
        else:
            tseries_number = tseries_number[1:]
        
        final_rois = tseries_data.get('final_rois', [])
        if not final_rois:
            print(f"  No ROIs found, skipping...")
            continue
        
        # Get FlyID from first ROI
        fly_id = final_rois[0].experiment_info.get('FlyID', 'Unknown')
        print(f"  FlyID: {fly_id}, Number of ROIs: {len(final_rois)}")
        
        # Get FPS from metadata
        meta_tseries = metadata_cond[metadata_cond['fly'] == fly_id]
        meta_tseries = meta_tseries[meta_tseries['TSeries'] == int(tseries_number)]
        tseries_fps = meta_tseries['fps'].values[0]
        ##############################################
        # Collect and interpolate traces for this TSeries
        interpolated_traces = []
        common_time = None
        
        for roi_idx, roi in enumerate(final_rois):
            df_trace = roi.df_trace
            
            # Determine original FPS
            if original_fps is None:
                roi_fps = tseries_fps
            else:
                roi_fps = original_fps
            
            # Interpolate and trim trace
            interp_trace, time = interpolate_trace(df_trace, roi_fps, mean_fps, trim_seconds=5)
            interpolated_traces.append(interp_trace)
            common_time = time
        
        # Convert to array with common length (match to stimulus length)
        # Use the minimum of trace length and stimulus length
        stim_length = len(stim_protocol)
        min_length = min(min(len(trace) for trace in interpolated_traces), stim_length)
        
        interpolated_traces = [trace[:min_length] for trace in interpolated_traces]
        common_time = common_time[:min_length]
        interpolated_traces = np.array(interpolated_traces)

        # ---- collect individual ROIs for clustering ----
        for roi_trace in interpolated_traces:
            entry = (roi_trace, fly_id, tseries_key)
            condition_rois.append(entry)
            fly_rois[fly_id].append(entry)
        
        # Plot 1: Individual ROIs and mean for this TSeries
        fig, mean_trace = plot_traces_with_mean(
            interpolated_traces,
            common_time,
            stim_protocol,
            stim_time,
            f'TSeries: {tseries_key} | FlyID: {fly_id}',
            ylabel='ΔF/F'
        )
        plt.savefig(os.path.join(output_dir, f'1_tseries_{tseries_idx:03d}_{tseries_key}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store TSeries mean
        tseries_means[tseries_key] = (mean_trace, common_time, fly_id)
        fly_means[fly_id].append((mean_trace, common_time))
        all_means.append((mean_trace, common_time))
        
        # Segment the mean trace
        segmented = segment_trace(mean_trace, common_time, segments, mean_fps)
        for seg_name, (seg_trace, seg_time) in segmented.items():
            tseries_segments[seg_name][tseries_key] = (seg_trace, seg_time, fly_id)
            fly_segments[seg_name][fly_id].append((seg_trace, seg_time))
            all_segments[seg_name].append((seg_trace, seg_time))
    
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
            ylabel='ΔF/F'
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
        ylabel='ΔF/F'
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
                ylabel='ΔF/F'
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
        
        # Plot global average for this segment
        fig, _ = plot_traces_with_mean(
            all_traces,
            common_time,
            seg_stim,
            seg_stim_time,
            f'{seg_name.upper()} | Global Average | {len(all_traces)} TSeries',
            ylabel='ΔF/F'
        )
        plt.savefig(os.path.join(output_dir, f'5_{seg_name}_global.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - {len(fly_segments[seg_name])} fly plots")
        print(f"  - 1 global plot")
    
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
        frequency_results=frequency_results)

    return tseries_means, fly_averaged_traces, global_mean, segment_results, luminance_results, polarity_results, contrast_results, frequency_results, variability_results, across_fly_variability

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


# Example usage
if __name__ == "__main__":
    # Configuration
    multi = False
    degen = True
    if multi: 
        # pkl_files = [r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_ME_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_ME_Y.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LO_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LO_Y.pkl"]
        pkl_files = [r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LOP_Y.pkl", r"C:\phd\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LOP_Y.pkl"]
    elif degen:
        # pkl_files =[r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_chirp.pkl"]
        pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_chirp.pkl"\
                     r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_chirp.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_chirp.pkl"]    
    all_results = []
    for PKL_FILE in pkl_files:
        folder = PKL_FILE.split('\\')[-1].split('.')[0]
        if multi:
            metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='251023_tdc2_cschr_pan')
            output_dir=f"C:/phd/02_twophoton/251023_tdc2_cschr_pan/4_results/{folder}"
        elif degen:
            metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            output_dir=f"C:/phd/02_twophoton/250730_degen_var/4_results/{folder}"
        os.makedirs(output_dir, exist_ok=True)
        mean_fps = metadata['fps'].mean()
        TARGET_FPS = mean_fps  # Target frames per second for interpolation
        ORIGINAL_FPS = None  # Set to None to extract from data, or specify (e.g., 30)
        
        # Run analysis
        tseries_means, fly_averaged_traces, global_mean, segment_results, luminance_results, polarity_results, contrast_results, frequency_results, variability_results, across_fly_variability = process_pkl_file(
            PKL_FILE, 
            metadata,
            multi,
            degen,
            TARGET_FPS, 
            original_fps=ORIGINAL_FPS,
            output_dir=output_dir
            
        )
        # Store results
        all_results.append({
            'luminance_results': luminance_results,
            'polarity_results': polarity_results,
            'contrast_results': contrast_results,
            'frequency_results': frequency_results,
            'segment_results': segment_results,
            'variability_results': variability_results,
            'across_fly_variability': across_fly_variability
        })
    
    if multi:
        combined_output = "C:/phd/02_twophoton/251023_tdc2_cschr_pan/4_results/combined"
    elif degen:
        combined_output = "C:/phd/02_twophoton/250730_degen_var/4_results/combined"
    
    os.makedirs(combined_output, exist_ok=True)
    plot_combined_analysis(all_results, pkl_files, combined_output)
    print("\nDone!")
