
"""only need to change in main() function: 
    path to pkl files (pkl_file_path list)
    target path (save_path)
    colormap variable: cmap"""
import numpy as np 
import matplotlib.pyplot as plt
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, dendrogram as sp_dendrogram
import pickle
from pathlib import Path
############################
def _norm_trace(trace):
    """normalizes to max (|max|=1)"""
    m = np.max(np.abs(trace))
    return trace / m if m > 0 else trace

def _zscore_trace(trace):
    '''zscores trace
    substract mean, divide by std'''
    t = trace.astype(float)
    s=t.std()
    return (t-t.mean()) / s if s > 0 else np.zeros_like(t)

def plot_cluster_heatmap(traces, clusterlabels, mean_fps, title, save_path, cmap, cluster_means=None, normalize=False):
    '''heatmap of all traces, sorted by cluster no dendogram
    '''
    unique_clusters = sorted(set(clusterlabels))
    plot_clusters = [c for c in unique_clusters if c !=-1]
    if not plot_clusters:
        return
    min_len = min(len(t) for t in traces)
    time = np.arange(min_len) / mean_fps
    sorted_idx, boundaries = [], [0]
    for c in plot_clusters:
        idx = [i for i, l in enumerate (clusterlabels) if l==c]
        sorted_idx.extend(idx)
        boundaries.append(len(sorted_idx))
    def _apply_norm(trace):
        if normalize == 'zscore':
            return _zscore_trace(trace)
        elif normalize:
            return _norm_trace(trace)
        else:
            return trace.astype(float)
    mat = np.vstack([_apply_norm(traces[i][:min_len]) for i in sorted_idx])
    vmax = float(np.percentile(np.abs(mat), 95))
    if normalize == 'zscore':
        cbar_label = 'Z-score'
    elif normalize:
        cbar_label = 'ΔF/F (norm. to peak)'
    else:
        cbar_label = 'ΔF/F'
    fig_h = max(4, len(sorted_idx)*0.12)
    fig, axes = plt.subplots(1, 1, figsize=(13, fig_h), gridspec_kw={'width_ratios': [1], 'wspace':0.02})
    im = axes.imshow(mat, aspect='auto', cmap=cmap, vmax=vmax, vmin=-vmax, extent=[0, time[-1], len(sorted_idx), 0])
    cmap_clabels = plt.cm.tab10
    for i, (c,b) in enumerate(zip(plot_clusters, boundaries[1:])):
        axes.axhline(b, color='white', linewidth=1.2)
        mid = (boundaries[i]+b) / 2
        axes.text(-0.015 * time[-1], mid, f'C{c}', ha='right', va='center', fontsize=9, fontweight='bold', color=cmap_clabels(i%10))
    cb = plt.colorbar(im, ax=axes, label=cbar_label, shrink=0.6)
    axes.set_xlabel('Time (s)', fontsize=11)
    axes.set_ylabel('ROI (sorted by cluster)', fontsize=11)
    axes.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    for ext in ['pdf', 'svg', 'png']:
        plt.savefig(f'{save_path}.{ext}', dpi=300, bbox_inches='tight')
    plt.close('all')
############################
def main():
    for neuron_type, pkl_file_path in zip(['Tm1', 'Tm9'], [Path(r"C:\phd\02_twophoton\250730_degen_var\4_results\_Tm1_chirp\clustering\clustering_heatmap_gmm_condition_spca_zscore_data.pkl"), Path(r"C:\phd\02_twophoton\250730_degen_var\4_results\_Tm9_chirp\clustering\clustering_heatmap_gmm_condition_spca_zscore_data.pkl")]):
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        save_path = Path(r'C:\phd\02_twophoton')
        save_path = save_path / f"heatmap_{neuron_type}"
        cmap = 'RdBu_r'
        plot_cluster_heatmap(data['traces'], data['labels'], data['mean_fps'], data['title'], save_path, cmap, data['cluster_means'], 'z-score')
    
    
if __name__ == "__main__":
    main()