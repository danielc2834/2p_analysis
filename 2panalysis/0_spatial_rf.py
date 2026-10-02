'''analyses spatial receptive fields
discrete bar stimulus
bar ternary wn stimulus
analysis include spatial field, heterogeneity & WN + bar combined'''
import pickle, os
import numpy as np
import pandas as pd
from itertools import permutations
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, levene
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import re
from collections import defaultdict 
# ---------------------------------------------------------------- config --
RESPONSE_SKIP = 0.15   # s after bar onset to skip (indicator rise time)
RESPONSE_DUR  = 1.3    # s response window length
BASELINE_DUR  = 0.5    # s pre-onset baseline window length
N_SHUFFLES    = 2000


############################################


def _detect_position_column(stim):
    """Vertical-bar files carry the real bar position in xpos; horizontal
    files are the mirror image. NaN presence alone doesn't distinguish them
    -- both columns go NaN together during the background sub-period and
    the blank/catch condition -- so instead pick whichever column actually
    takes many distinct real values (the tested positions) versus the other,
    which stays at one constant value (e.g. 0.0) whenever it isn't NaN."""
    xpos_num = pd.to_numeric(stim['xpos'].astype(str).str.strip(), errors='coerce')
    ypos_num = pd.to_numeric(stim['ypos'].astype(str).str.strip(), errors='coerce')
    x_variety, y_variety = xpos_num.dropna().nunique(), ypos_num.dropna().nunique()
    if x_variety > y_variety:
        return xpos_num
    elif y_variety > x_variety:
        return ypos_num
    raise ValueError("Could not tell whether xpos or ypos carries the real bar "
                      "position in this file (both take the same number of "
                      "distinct real values) -- inspect it directly rather than "
                      "trusting either column.")
 
 
def load_trials(stim_log_path):
    """Parse a stim_output.txt into one row per trial (a contiguous boutInd
    run). Works for either orientation -- see _detect_position_column.
    Drops the always-blank catch condition and any truncated trial."""
    stim = pd.read_csv(stim_log_path, skiprows=2)
    stim.columns = stim.columns.str.strip()
    stim['position_num'] = _detect_position_column(stim)
    stim['run_id'] = (stim['boutInd'] != stim['boutInd'].shift()).cumsum()
 
    def summarize(g):
        real = g.dropna(subset=['position_num'])
        nan_rows = g[g['position_num'].isna()]
        bar_off = nan_rows['tcurr'].iloc[0] if len(nan_rows) else g['tcurr'].iloc[-1]
        return pd.Series({'boutInd': g['boutInd'].iloc[0],
                           'position': real['position_num'].iloc[0] if len(real) else np.nan,
                           't_onset': g['tcurr'].iloc[0], 'n_frames': len(g)})
 
    trials = stim.groupby('run_id').apply(summarize).reset_index(drop=True)
    # drop catch-condition trials (no real bar position) and any truncated trial
    return trials[(trials['n_frames'] >= 100) & (trials['position'].notna())].reset_index(drop=True)
 
 
def extract_roi_responses(roi, trials):
    """Per-trial response amplitude for one ROI: mean dF/F in the response
    window minus mean dF/F in the pre-onset baseline window. tcurr and
    micRelTimes share a clock, so no offset is needed between them. A
    trial whose window contains a non-finite dF/F sample (NaN/Inf -- an
    upstream data artifact, not a real measurement) is skipped rather than
    silently propagating into the mean."""
    t, df = roi.imaging_info['micRelTimes'], roi.df_trace
    rows = []
    for _, tr in trials.iterrows():
        onset = tr['t_onset']
        resp = (t >= onset + RESPONSE_SKIP) & (t < onset + RESPONSE_SKIP + RESPONSE_DUR)
        base = (t >= onset - BASELINE_DUR) & (t < onset)
        if resp.sum() == 0 or base.sum() == 0:
            continue
        resp_trace, base_trace = df[resp], df[base]
        if not (np.all(np.isfinite(resp_trace)) and np.all(np.isfinite(base_trace))):
            continue
        rows.append({'position': tr['position'], 'response': resp_trace.mean() - base_trace.mean()})
    return pd.DataFrame(rows)
 
 
# ------------------------------------------------------- fit + significance --
def gaussian(x, amp, center, sigma):
    return amp * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
 
 
def fit_tuning_curve(positions, responses):
    """Fit a Gaussian -- no offset term, no bounds, matching the model from
    the prior lab pipeline -- to one ROI's tuning curve, for FWHM/R^2
    quantification (alignment uses the raw peak instead -- see
    _peak_position). Tries both a positive- and negative-seeded fit and
    keeps whichever the data supports better (higher R^2), since OFF-
    condition responses in an ON-polarity cell dip rather than bump. A
    center outside the tested range is still treated as a failed fit --
    unconstrained fitting can wander further than bounded fitting did, so
    this check matters more here, not less."""
    positions = np.asarray(positions, dtype=float)
    responses = np.asarray(responses, dtype=float)
    rng = positions.max() - positions.min()
    best = None
    for seed_idx in (np.argmax(responses), np.argmin(responses)):
        try:
            popt, _ = curve_fit(gaussian, positions, responses,
                                 p0=[responses[seed_idx], positions[seed_idx], rng / 4],
                                 maxfev=8000)
        except Exception:
            continue
        amp, center, sigma = popt
        sigma = abs(sigma)
        pred = gaussian(positions, *popt)
        ss_res = np.sum((responses - pred) ** 2)
        ss_tot = np.sum((responses - responses.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if best is None or r2 > best['r2']:
            best = {'amp': amp, 'center': center, 'sigma': sigma,
                    'fwhm': 2 * np.sqrt(2 * np.log(2)) * sigma, 'r2': r2}
    if best is not None and not (positions.min() <= best['center'] <= positions.max()):
        return None  # center extrapolated outside the tested range -- not a trustworthy estimate
    return best
 
 
def shuffle_test(positions, responses, n_shuffles=N_SHUFFLES, seed=0):
    """Null: response doesn't depend on position. Statistic: variance of the
    per-position mean response curve. Returns (p_value, observed_stat, null_stats)."""
    positions = np.asarray(positions)
    responses = np.asarray(responses)
    obs_stat = pd.Series(responses).groupby(positions).mean().var()
    rng = np.random.default_rng(seed)
    null_stats = np.array([pd.Series(rng.permutation(responses)).groupby(positions).mean().var()
                            for _ in range(n_shuffles)])
    p = (np.sum(null_stats >= obs_stat) + 1) / (n_shuffles + 1)
    return p, obs_stat, null_stats
 
 
def compute_reliability_snr(roi, trials):
    """Reliability and SNR, adapted from the prior lab pipeline's
    calculateReliability to work on trial-extracted windows rather than
    per-epoch trial matrices. Per tested position: reliability is the mean
    pairwise Pearson correlation between repeats' response-window time
    courses; SNR is a bias-corrected estimate from response vs. pre-onset
    noise std ((n+1)/n * signal_std/noise_std - 1/n). Both are then
    maximized across positions, matching roi.reliability/roi.SNR in the old
    code. Positions with fewer than 2 repeats contribute to neither. A
    trial whose window contains a non-finite dF/F sample (NaN/Inf -- an
    upstream data artifact) is skipped, with a one-line warning naming the
    fly if any were, rather than crashing pearsonr on it. Returns (nan,
    nan) if no position has enough (finite) repeats."""
    t, df = roi.imaging_info['micRelTimes'], roi.df_trace
    by_position = {}
    n_skipped = 0
    for _, tr in trials.iterrows():
        onset = tr['t_onset']
        resp_idx = (t >= onset + RESPONSE_SKIP) & (t < onset + RESPONSE_SKIP + RESPONSE_DUR)
        base_idx = (t >= onset - BASELINE_DUR) & (t < onset)
        if resp_idx.sum() == 0 or base_idx.sum() == 0:
            continue
        resp_trace, base_trace = df[resp_idx], df[base_idx]
        if not (np.all(np.isfinite(resp_trace)) and np.all(np.isfinite(base_trace))):
            n_skipped += 1
            continue
        by_position.setdefault(tr['position'], []).append((resp_trace, base_trace.std()))
    if n_skipped:
        flyid = (getattr(roi, 'experiment_info', {}) or {}).get('FlyID', '?')
        print(f"compute_reliability_snr: skipped {n_skipped} trial(s) with non-finite "
              f"dF/F (fly {flyid}) -- likely a data artifact worth checking upstream")
 
    reliabilities, snrs = [], []
    for traces_and_noise in by_position.values():
        n = len(traces_and_noise)
        if n < 2:
            continue
        traces = [tr for tr, _ in traces_and_noise]
        noise_stds = np.array([ns for _, ns in traces_and_noise])
        min_len = min(len(tr) for tr in traces)
        mat = np.array([tr[:min_len] for tr in traces])
        coeffs = [pearsonr(mat[i], mat[j])[0] for i, j in permutations(range(n), 2)]
        if coeffs:
            reliabilities.append(np.nanmean(coeffs))
        noise_std = noise_stds.mean()
        if noise_std > 0:
            signal_std = mat.std(axis=1).mean() - noise_std
            snrs.append(((n + 1) / n) * (signal_std / noise_std) - 1 / n)
 
    reliability = np.max(reliabilities) if reliabilities else np.nan
    snr = np.max(snrs) if snrs else np.nan
    return reliability, snr
 
 
def analyze_roi(roi, trials):
    """Extract, fit and significance-test one ROI. Returns a dict with
    everything downstream code needs, or None if there isn't enough data
    or the fit failed."""
    resp = extract_roi_responses(roi, trials)
    if resp.empty:
        return None
    tuning = resp.groupby('position')['response'].mean().sort_index()
    fit = fit_tuning_curve(tuning.index.values, tuning.values)
    if fit is None:
        return None
    p, obs_stat, null_stats = shuffle_test(resp['position'].values, resp['response'].values)
    reliability, snr = compute_reliability_snr(roi, trials)
    experiment_info = getattr(roi, 'experiment_info', {}) or {}
    return {'positions': tuning.index.values, 'responses': tuning.values,
            'fit': fit, 'p': p, 'obs_stat': obs_stat, 'null_stats': null_stats,
            'reliability': reliability, 'snr': snr,
            'genotype': experiment_info.get('Genotype'), 'fly_id': experiment_info.get('FlyID'),
            'category': getattr(roi, 'category', None)}
 
 
def analyze_condition(pooled, ts_to_stimlog):
    """Loop over every TSeries in a pooled condition dict and every ROI
    within it. `pooled` is the already-loaded pickle content (load it
    however works in your environment); `ts_to_stimlog` maps TSeries name
    -> stim_output.txt path. Returns a flat list of per-ROI result dicts
    (see analyze_roi), each tagged with ts_name and roi_idx. A TSeries with
    no mapped or unreadable stim log is skipped (with a warning) rather
    than crashing the whole batch -- useful when looping over many flies."""
    results = []
    for ts_name, entry in pooled.items():
        if ts_name not in ts_to_stimlog:
            print(f"no stim log mapped for {ts_name}, skipping")
            continue
        try:
            trials = load_trials(ts_to_stimlog[ts_name])
        except Exception as e:
            print(f"couldn't load stim log for {ts_name} ({ts_to_stimlog[ts_name]}): {e}, skipping")
            continue
        for roi_idx, roi in enumerate(entry['final_rois']):
            r = analyze_roi(roi, trials)
            if r is not None:
                r['ts_name'], r['roi_idx'] = ts_name, roi_idx
                results.append(r)
    return results
 
 
def load_or_analyze_condition(pooled, ts_to_stimlog, cache_path, force=False):
    """Load analyze_condition's output from cache_path if it already
    exists (and force is False); otherwise run analyze_condition and save
    the result to cache_path for next time. analyze_condition is the slow
    step -- parsing every stim log and fitting + shuffle-testing every ROI
    -- so caching it lets you re-run the rest of your script (filtering,
    plotting, the combined/2D summary) without redoing it, as long as
    nothing upstream (stim logs, response window, fit code) changed. Pass
    force=True to ignore an existing cache and recompute. Returns the same
    list of result dicts as analyze_condition."""
    if not force and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    results = analyze_condition(pooled, ts_to_stimlog)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    return results
 
 
# --------------------------------------------------------------- figures --
def _peak_position(positions, responses):
    """Raw-data peak/trough position (largest |response|) -- used for
    alignment, kept deliberately separate from the Gaussian fit's `center`,
    which is reserved for FWHM/R^2 quantification. Unlike a fitted center,
    this is always one of the actual tested positions, so it can't land
    outside the tested range the way an unconstrained fit occasionally can."""
    return positions[np.argmax(np.abs(responses))]
 
 
def build_aligned_heatmap(results, grid=None):
    """Peak-align every ROI's tuning curve and interpolate onto a common
    1-degree grid. Returns (grid, heat) where heat has shape
    (len(results), len(grid)); NaN where a row's shifted data doesn't reach
    that grid point. Shared by plot_aligned_summary and
    filter_by_responsiveness so both work from the same aligned data.
 
    grid defaults to 1-degree bins spanning the full shifted range of every
    ROI (computed from the data, not a fixed guess) -- a row whose peak
    sits near the edge of the tested position range needs a wider window
    than one near the middle to show its full data without clipping."""
    if grid is None:
        lo = min(d['positions'].min() - _peak_position(d['positions'], d['responses']) for d in results)
        hi = max(d['positions'].max() - _peak_position(d['positions'], d['responses']) for d in results)
        grid = np.arange(np.floor(lo), np.ceil(hi) + 1, 1.0)
    heat = np.full((len(results), len(grid)), np.nan)
    for row, d in enumerate(results):
        rel = d['positions'] - _peak_position(d['positions'], d['responses'])
        order = np.argsort(rel)
        heat[row] = np.interp(grid, rel[order], d['responses'][order], left=np.nan, right=np.nan)
    return grid, heat
 
 
def filter_by_responsiveness(results, grid=None, window=1.0):
    """Exclude ROIs whose peak-window response -- mean of the aligned,
    interpolated heatmap within +/-window degrees of 0 -- has the "wrong"
    sign relative to the population: if the population mean is positive,
    remove ROIs with a negative peak-window value, and vice versa. A
    genuinely responsive ROI should point the same direction as the
    population's real signal; one that doesn't is more likely picking up
    noise at its "peak" than a real response."""
    grid, heat = build_aligned_heatmap(results, grid)
    win = (grid >= -window) & (grid <= window)
    peak_vals = np.nanmean(heat[:, win], axis=1)
    pop_mean = np.nanmean(peak_vals)
    keep = (peak_vals >= 0) if pop_mean > 0 else (peak_vals <= 0)
    return [d for d, k in zip(results, keep) if k]
 
 
def plot_aligned_summary(results, grid=None, title='', xlim=(-20, 20), vlim=2.0):
    """3-panel figure: peak-aligned heatmap, raw aligned tuning curves +
    mean, and an FWHM violin (one violin per call -- for a multi-condition
    comparison, call once per condition and combine the FWHM arrays into a
    multi-violin plot yourself). Alignment is by each ROI's raw peak
    (_peak_position), not the Gaussian fit's center -- FWHM still comes
    from the fit.
 
    grid still defaults to 1-degree bins spanning the full shifted range of
    every ROI (so the mean trace and interpolation aren't truncated); xlim
    only zooms the heatmap's displayed x-range (None to show the full
    grid). vlim sets a fixed +/-dF/F colorbar range (None to scale to this
    data's own max |value| instead)."""
    grid, heat = build_aligned_heatmap(results, grid)
    # a numpy "mean of empty slice" warning here just means no ROI's shifted
    # data reaches that grid edge -- harmless, and should vanish with more ROIs
    mean_trace = np.nanmean(heat, axis=0)
    fwhm_vals = np.array([d['fit']['fwhm'] for d in results])
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
 
    ax = axes[0]
    im = ax.imshow(heat, aspect='auto', cmap='RdBu_r',
                    norm=TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim) if vlim is not None
                    else TwoSlopeNorm(vcenter=0, vmin=-np.nanmax(np.abs(heat)), vmax=np.nanmax(np.abs(heat))),
                    extent=[grid[0], grid[-1], len(results) - 0.5, -0.5])
    im.cmap.set_bad('lightgray')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel('position rel. to ROI peak (deg)')
    ax.set_ylabel('ROI (row order = results list order)')
    plt.colorbar(im, ax=ax, label='dF/F', fraction=0.046)
    ax.set_title(f'{title} -- aligned RF heatmap (n={len(results)})')
 
    ax = axes[1]
    for d in results:
        ax.plot(d['positions'] - _peak_position(d['positions'], d['responses']),
                 d['responses'], color='gray', alpha=0.35, lw=1)
    ax.plot(grid, mean_trace, color='crimson', lw=2.5, label='mean')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('position rel. to ROI peak (deg)')
    ax.set_ylabel('dF/F')
    ax.set_title('raw peak-aligned tuning curves')
    ax.legend()
 
    ax = axes[2]
    ax.violinplot([fwhm_vals], showmeans=True, showmedians=True)
    med, mean = np.median(fwhm_vals), np.mean(fwhm_vals)
    q1, q3 = np.percentile(fwhm_vals, [25, 75])
    ax.set_xticks([1])
    ax.set_xticklabels([title or 'condition'])
    ax.set_ylabel('FWHM (deg)')
    ax.set_title(f'FWHM (n={len(fwhm_vals)})')
    ax.text(1.12, ax.get_ylim()[1] * 0.95,
            f"mean={mean:.1f}\nmedian={med:.1f}\nIQR=[{q1:.1f}, {q3:.1f}]",
            va='top', fontsize=9)
 
    plt.tight_layout()
    return fig
 
 
def plot_fwhm_by_genotype(fwhm_by_genotype, title=''):
    """Per-condition genotype heterogeneity comparison, from a
    {genotype: [fwhm values]} dict. Two panels: FWHM violin (as before),
    and CV = SD/mean per genotype as a bar chart -- CV normalizes spread by
    scale, so a genotype can be *relatively* more variable than another
    even with a smaller absolute SD. The figure title reports a
    Brown-Forsythe test (Levene's test, median-centered -- robust to
    non-normal FWHM distributions) for equal FWHM variance across the
    genotypes shown, i.e. whether the heterogeneity itself differs.
 
    Raises ValueError if fwhm_by_genotype is empty, or every genotype's
    list is empty -- there is nothing to plot. Check for that case in your
    calling loop (e.g. `if not by_genotype: continue`) rather than relying
    on this to fail gracefully, since a genuinely empty condition (e.g. no
    ROI passed filtering for this genotype) is worth its own diagnostic
    message naming which condition/polarity it was, not a generic one
    raised from inside the plotting code."""
    genotypes = [g for g in fwhm_by_genotype if len(fwhm_by_genotype[g])]
    if not genotypes:
        raise ValueError("fwhm_by_genotype has no genotype with any data -- "
                          "nothing to plot (check for this before calling)")
    data = [np.asarray(fwhm_by_genotype[g], dtype=float) for g in genotypes]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.2 * len(genotypes) + 4, 5.5))
 
    ax1.violinplot(data, showmeans=True, showmedians=True)
    ax1.set_xticks(range(1, len(genotypes) + 1))
    ax1.set_xticklabels(genotypes)
    ax1.set_ylabel('FWHM (deg)')
    ax1.set_title('FWHM by genotype')
    ymax = max((v.max() for v in data if len(v)), default=1)
    ax1.set_ylim(top=ymax * 1.35)
    for i, vals in enumerate(data, start=1):
        if not len(vals):
            continue
        med, mean = np.median(vals), np.mean(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        ax1.text(i, ymax * 1.05, f"n={len(vals)}\nmean={mean:.1f}\nmed={med:.1f}\nIQR=[{q1:.1f},{q3:.1f}]",
                  ha='center', va='bottom', fontsize=7)
 
    cvs = [vals.std() / vals.mean() if len(vals) and vals.mean() != 0 else np.nan for vals in data]
    ax2.bar(genotypes, cvs, color='steelblue')
    ax2.set_ylabel('CV (SD / mean FWHM)')
    ax2.set_title('relative variability')
    finite_cvs = [c for c in cvs if np.isfinite(c)]
    if finite_cvs:
        ax2.set_ylim(top=max(finite_cvs) * 1.25)
    for i, cv in enumerate(cvs):
        if np.isfinite(cv):
            ax2.text(i, cv, f'{cv:.2f}', ha='center', va='bottom', fontsize=9)
 
    testable = [(g, v) for g, v in zip(genotypes, data) if len(v) >= 2]
    if len(testable) >= 2:
        stat, p = levene(*(v for _, v in testable), center='median')
        test_result = {'stat': stat, 'p': p, 'genotypes': [g for g, _ in testable]}
        test_str = f"Brown-Forsythe (equal FWHM variance across genotypes): p={p:.4f}"
    else:
        test_result = {'stat': np.nan, 'p': np.nan, 'genotypes': [g for g, _ in testable]}
        test_str = "Brown-Forsythe: not enough genotypes/data to test"
    fig.suptitle(f'{title} -- {test_str}'.strip(' -'))
    plt.tight_layout()
    return fig, test_result
 
 
# R^2 thresholds from Ramos-Traslosheros & Silies 2021 (Fig. 2 legend,
# Methods "Tuning curves"): they used a looser cutoff for ON responses,
# which were harder to fit cleanly even when real.
R2_THRESH = {'ON': 0.2, 'OFF': 0.5}
 
 
def r2_threshold(label):
    """Look up the paper-matching R^2 threshold from any underscore/space/
    dash-separated label containing an ON or OFF token -- works with your
    condition_label ('Mi1_bar_OFF_v'), a normalized 'OFF_V'-style key, a
    bare filename, etc. -- so there's no separate variable to keep in
    sync. Matches a whole token (split on [_\\s-]), not a substring, so a
    genotype name that happens to contain "on"/"off" won't false-match."""
    tokens = re.split(r'[_\s-]+', label.upper())
    found = {R2_THRESH[t] for t in tokens if t in R2_THRESH}
    if not found:
        raise ValueError(f"no ON/OFF token found in {label!r}")
    if len(found) > 1:
        raise ValueError(f"both ON and OFF tokens found in {label!r} -- ambiguous")
    return found.pop()
 
 
def filter_by_r2(results, min_r2=0.5):
    """R^2 >= min_r2 quality gate. Use this before plot_aligned_summary too
    if you want the alignment/FWHM figure itself restricted to well-fit
    ROIs. min_r2 defaults to 0.5 here as a generic fallback; pass
    r2_threshold(condition_key) explicitly for the paper-matching,
    polarity-specific value (0.2 for ON, 0.5 for OFF)."""
    return [d for d in results if d['fit']['r2'] >= min_r2]
 
 
def plot_quality_distributions(results, title='', min_r2=0.5):
    """3-row x 2-column figure -- R^2, reliability, SNR (rows) each shown
    for every ROI (left column) and for only the ROIs passing the R^2 >=
    min_r2 filter (right column). Same bins/axis per row across both
    columns so before/after is directly comparable."""
    passing = filter_by_r2(results, min_r2)
    rows = [('R$^2$', [d['fit']['r2'] for d in results], [d['fit']['r2'] for d in passing]),
            ('reliability', [d['reliability'] for d in results], [d['reliability'] for d in passing]),
            ('SNR', [d['snr'] for d in results], [d['snr'] for d in passing])]
 
    fig, axes = plt.subplots(3, 2, figsize=(9, 10))
    for row, (label, all_vals, pass_vals) in enumerate(rows):
        all_vals = np.array(all_vals, dtype=float)
        pass_vals = np.array(pass_vals, dtype=float)
        finite = all_vals[np.isfinite(all_vals)]
        bins = np.linspace(finite.min(), finite.max(), 21) if len(finite) else 20
        cols = [(all_vals, f'all (n={len(all_vals)})'),
                (pass_vals, f'R$^2\\geq${min_r2} (n={len(pass_vals)})')]
        for col, (vals, sub) in enumerate(cols):
            ax = axes[row, col]
            finite_vals = vals[np.isfinite(vals)]
            ax.hist(finite_vals, bins=bins, color='steelblue', edgecolor='white')
            if len(finite_vals):
                ax.axvline(np.median(finite_vals), color='crimson', lw=2,
                           label=f'median={np.median(finite_vals):.2f}')
                ax.legend(fontsize=8)
            ax.set_title(f'{label} -- {sub}', fontsize=10)
    fig.suptitle(f'{title} -- quality metrics'.strip(' -'))
    plt.tight_layout()
    return fig
 
 
def plot_shuffle_test(result, ax=None):
    """Visualize one ROI's shuffle test: the null distribution of the test
    statistic (variance of the position-mean response curve under shuffled
    position labels), with the observed statistic and resulting p-value
    marked. Call with any single entry from analyze_condition()'s output,
    e.g. plot_shuffle_test(results[0])."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure
    ax.hist(result['null_stats'], bins=40, color='lightgray', edgecolor='white', label='shuffled null')
    ax.axvline(result['obs_stat'], color='crimson', lw=2, label='observed')
    ax.set_xlabel('test statistic (variance of position-mean response)')
    ax.set_ylabel('count (shuffles)')
    ax.set_title(f"shuffle test  (p={result['p']:.4f}, n={len(result['null_stats'])} shuffles)")
    ax.legend()
    plt.tight_layout()
    return fig
 
 
# ------------------------------------------------------- save to disk --
def save_fwhm_csv(fwhm_by_genotype, path):
    """Write {genotype: [fwhm values]} to a tidy CSV (genotype, fwhm
    columns, one row per ROI) for one condition."""
    rows = [{'genotype': g, 'fwhm': v} for g, vals in fwhm_by_genotype.items() for v in vals]
    pd.DataFrame(rows).to_csv(path, index=False)
 
 
def save_fwhm_by_genotype_csv(fwhm_by_condition, genotype, path):
    """Write ALL of one genotype's FWHM values (degrees), pooled across
    every condition in fwhm_by_condition -- all 4 (ON_V/ON_H/OFF_V/
    OFF_H) if all 4 have been accumulated -- to a single CSV. One file
    per genotype rather than one per condition: a `condition` column is
    kept so you can still facet/filter by condition downstream, but the
    file itself contains every ROI of this genotype regardless of which
    condition it came from. fwhm_by_condition is expected already
    quality-filtered (e.g. via filter_by_r2/filter_by_responsiveness) --
    this only pools and writes, it doesn't filter."""
    rows = [{'condition': condition_key, 'fwhm': v}
             for condition_key, by_genotype in fwhm_by_condition.items()
             for v in by_genotype.get(genotype, [])]
    pd.DataFrame(rows).to_csv(path, index=False)
 
 
def save_variance_test_txt(test_result, path, title=''):
    """Write a Brown-Forsythe test result (as returned by
    plot_fwhm_by_genotype) to a plain-text file."""
    with open(path, 'w') as f:
        if title:
            f.write(f"{title}\n")
        f.write("Brown-Forsythe test (Levene, median-centered) for equal FWHM variance across genotypes\n")
        f.write(f"genotypes compared: {', '.join(test_result['genotypes'])}\n")
        f.write(f"statistic = {test_result['stat']:.6f}\n")
        f.write(f"p-value   = {test_result['p']:.6f}\n")
 
 
# ---------------------------------------------- 2D receptive fields --
# Combines a fly's vertical-bar (x) and horizontal-bar (y) 1D receptive
# fields into a single 2D map, following Ramos-Traslosheros & Silies 2021
# (Nat Commun, https://pubmed.ncbi.nlm.nih.gov/34404776/), Fig. 2d: a
# "sign-consistent Cartesian product" of the two orthogonal 1D profiles,
# color-coded green (both positive), yellow (both negative), or gray
# (opposing signs), with luminance proportional to |product|.
 
def build_2d_receptive_field(v_result, h_result, grid=None):
    """Combine one ROI's vertical-bar (x-profile) and horizontal-bar
    (y-profile) Gaussian fits into a 2D color-coded map. Each profile is
    evaluated from its own fit, peak-centered (x=0 is the V profile's own
    peak, y=0 is the H profile's own peak) -- matching the paper's
    "receptive field centers are aligned for illustrative purposes" (Fig.
    2b legend) -- rather than the raw discrete tuning curve, so the map is
    smooth.
 
    v_result, h_result: single analyze_roi() result dicts for the SAME ROI
    under vertical-bar and horizontal-bar conditions of the SAME polarity
    (both ON, or both OFF) -- use combine_2d_receptive_fields to pair them
    correctly rather than calling this directly on unmatched ROIs.
 
    Returns (grid, rgb): grid is the shared 1-degree axis (same range for
    x and y), rgb has shape (len(grid), len(grid), 3) in [0, 1]."""
    if grid is None:
        grid = np.arange(-30, 30.01, 1.0)
    fx = gaussian(grid, v_result['fit']['amp'], 0.0, v_result['fit']['sigma'])
    gy = gaussian(grid, h_result['fit']['amp'], 0.0, h_result['fit']['sigma'])
 
    F = fx[np.newaxis, :]   # varies along columns (x)
    G = gy[:, np.newaxis]   # varies along rows (y)
    product = F * G
    mag = np.abs(product)
    peak = mag.max()
    mag = mag / peak if peak > 0 else mag
 
    both_pos = (F > 0) & (G > 0)
    both_neg = (F < 0) & (G < 0)
    opposing = ~(both_pos | both_neg)
 
    rgb = np.zeros((len(grid), len(grid), 3))
    rgb[..., 1] = np.where(both_pos, mag, rgb[..., 1])                       # green
    rgb[..., 0] = np.where(both_neg, mag, rgb[..., 0])                       # yellow = R+G
    rgb[..., 1] = np.where(both_neg, mag, rgb[..., 1])
    for ch in range(3):                                                      # gray = R+G+B
        rgb[..., ch] = np.where(opposing, mag, rgb[..., ch])
    return grid, rgb
 
 
def combine_2d_receptive_fields(v_results, h_results, grid=None, min_r2=None):
    """Pair the same physical ROI between v_results and h_results -- matched
    by (ts_name, roi_idx), i.e. the same fly and the same position in that
    fly's ROI list, relying on the stable per-fly ROI index across a fly's
    condition files -- and build each pair's 2D receptive field. v_results
    and h_results should be analyze_condition() output for the two
    orientation files of the SAME polarity (both ON, or both OFF).
    Unmatched ROIs (present in one list but not the other, e.g. because one
    side was already filtered) are skipped, so this is safe to call before
    or after filtering either side independently -- pairing never relies on
    list position. min_r2, if given, additionally requires BOTH the V and H
    fit to individually pass R^2 >= min_r2.
 
    Returns a list of dicts: {'grid', 'rgb', 'fwhm_2d', 'genotype',
    'fly_id', 'ts_name', 'roi_idx'} -- fwhm_2d is the geometric mean of the
    two 1D FWHMs, a single characteristic width for the combined map, fed
    straight into plot_fwhm_by_genotype like any other condition's FWHM
    list."""
    h_by_key = {(d['ts_name'], d['roi_idx']): d for d in h_results}
    out = []
    for v in v_results:
        h = h_by_key.get((v['ts_name'], v['roi_idx']))
        if h is None:
            continue
        if min_r2 is not None and (v['fit']['r2'] < min_r2 or h['fit']['r2'] < min_r2):
            continue
        g, rgb = build_2d_receptive_field(v, h, grid)
        out.append({'grid': g, 'rgb': rgb,
                     'fwhm_2d': float(np.sqrt(v['fit']['fwhm'] * h['fit']['fwhm'])),
                     'genotype': v['genotype'], 'fly_id': v['fly_id'],
                     'ts_name': v['ts_name'], 'roi_idx': v['roi_idx']})
    return out
 
 
def plot_2d_receptive_field(combined, ax=None, title=''):
    """Show one ROI's color-coded 2D receptive field (one entry from
    combine_2d_receptive_fields): green = both profiles positive, yellow =
    both negative, gray = opposing signs, luminance = |Cartesian product|
    magnitude (Ramos-Traslosheros & Silies 2021, Fig. 2d)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    grid = combined['grid']
    ax.imshow(combined['rgb'], extent=[grid[0], grid[-1], grid[0], grid[-1]], origin='lower')
    ax.set_xlabel('x, deg (vertical-bar axis)')
    ax.set_ylabel('y, deg (horizontal-bar axis)')
    ax.set_title(title or f"FWHM$_{{2D}}$={combined['fwhm_2d']:.1f} deg")
    plt.tight_layout()
    return fig
 
 
def plot_2d_receptive_field_montage(combined, title=''):
    """One example 2D receptive field per genotype -- the first matched
    ROI found for each -- arranged in a grid, for visually comparing RF
    shape/size across neuron types within one polarity. `combined` is
    combine_2d_receptive_fields() output, typically spanning several
    genotypes when results_by_condition has been accumulated (not
    overwritten) across all of a dataset's neuron-type pkl files."""
    first_per_genotype = {}
    for c in combined:
        first_per_genotype.setdefault(c['genotype'], c)
    genotypes = list(first_per_genotype.keys())
    n = len(genotypes)
    if n == 0:
        raise ValueError("combined has no entries -- nothing to show")
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.3 * nrows), squeeze=False)
    for i, g in enumerate(genotypes):
        ax = axes[i // ncols, i % ncols]
        c = first_per_genotype[g]
        grid = c['grid']
        ax.imshow(c['rgb'], extent=[grid[0], grid[-1], grid[0], grid[-1]], origin='lower')
        ax.set_title(f"{g}\nFWHM$_{{2D}}$={c['fwhm_2d']:.1f} deg", fontsize=10)
        ax.set_xlabel('x, deg')
        ax.set_ylabel('y, deg')
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    return fig
# Example usage
if __name__ == "__main__":
    # Configuration
    multi = False
    degen = True
    temp = False
    if multi: 
        # pkl_files = [r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_ME_Y.pkl", r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_ME_Y.pkl"]
        # pkl_files = [r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LO_Y.pkl", r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LO_Y.pkl"]
        pkl_files = ["D:/Christian/02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_ON_500_LOP_Y.pkl", r"D:\Christian\02_twophoton\251023_tdc2_cschr_pan\3_DATA\_Chirp_OFF_LOP_Y.pkl"]
    elif degen:
        processed_path = "C:/phd/02_twophoton/250730_degen_var/2_processed_data"
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_ON_h.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_ON_h.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_ON_h.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_ON_h.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_ON_h.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_ON_h.pkl"]
        #TMs
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_ON_h.pkl", \
        #     r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_ON_h.pkl", \
        #     r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_ON_h.pkl"]
        #MIs
        # pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_ON_h.pkl", \
        #             r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_ON_h.pkl", \
        #             r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_ON_h.pkl"]
        # ALL
        pkl_files = [r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi1_bar_ON_h.pkl", \
                    r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi4_bar_ON_h.pkl", \
                    r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Mi9_bar_ON_h.pkl",\
                    r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm1_bar_ON_h.pkl", \
                    r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm4_bar_ON_h.pkl", \
                    r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_OFF_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_OFF_h.pkl",r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_ON_v.pkl", r"C:\phd\02_twophoton\250730_degen_var\3_DATA\_Tm9_bar_ON_h.pkl"]
    elif temp:
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_12_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_25_chirp.pkl",\
                        # r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_25_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_12_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_25_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_25_chirp.pkl"]
        pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_25_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_25_chirp.pkl"]
        # pkl_files = [r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm1_18_chirp.pkl", r"C:\phd\02_twophoton\20260617_temp\3_DATA\_Tm9_18_chirp.pkl"]

    all_results = []
    fwhm_by_condition = defaultdict(lambda: defaultdict(list))
    results_by_condition = defaultdict(list) 
    for PKL_FILE in pkl_files:
        folder = PKL_FILE.split('\\')[-1].split('.')[0]
        condition_label = folder.lstrip('_')
        m = re.search(r'bar_(ON|OFF)_(v|h)', condition_label, re.IGNORECASE)
        if m is None:
            raise ValueError(f"condition_label {condition_label!r} doesn't match the expected bar_ON/OFF_v/h pattern")
        condition_key = f"{m.group(1).upper()}_{m.group(2).upper()}"
        if multi:
            # metadata = pd.read_excel("D:/Christian/02_twophoton/metadata.xlsx", sheet_name='251023_tdc2_cschr_pan')
            output_dir=f"D:/Christian/02_twophoton/251023_tdc2_cschr_pan/4_results/{folder}"
        elif degen:
            # metadata = pd.read_excel("D:/Christian/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            # output_dir=f"D:/Christian/02_twophoton/250730_degen_var/4_results/{folder}"
            # metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            output_dir=f"C:/phd/02_twophoton/250730_degen_var/4_results/{folder}"
        elif temp:
            # metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
            output_dir=f"C:/phd/02_twophoton/20260617_temp/4_results/{folder}"
        os.makedirs(output_dir, exist_ok=True)
        
        # if multi:
        #     metadata = pd.read_excel("D:/Christian/02_twophoton/metadata.xlsx", sheet_name='251023_tdc2_cschr_pan')
        # elif degen:
        #     metadata = pd.read_excel("C:/phd/02_twophoton/metadata.xlsx", sheet_name='250730_degen_var')
        # elif temp:
        #     metadata = pd.read_excel("C:/phd/02_twophoton/20260617_temp/metadata_2photon_tempdependency.xlsx", sheet_name='20260617_temp')
            
        with open(PKL_FILE, 'rb') as f:
            data = pickle.load(f)
        ts_to_stimlog = {}
        for tseries in data:
            fly = ('_').join(tseries.split('-')[1:3])
            simlogpath = f'{processed_path}/{folder}/{fly}/{tseries}/{tseries}_stim_output.txt'
            ts_to_stimlog[tseries] = simlogpath
        cache_path = f"{output_dir}/{condition_label}_results.pkl"
        results = load_or_analyze_condition(data, ts_to_stimlog, cache_path)
        if not results:
            print(f"no usable ROIs for {condition_label}, skipping plots")
            continue
            
        plot_aligned_summary(results, title=condition_label).savefig(
            f"{output_dir}/{condition_label}_aligned_summary.png", dpi=150, bbox_inches='tight')
        # passing = filter_by_r2(filter_by_responsiveness(results))
        passing = filter_by_r2(filter_by_responsiveness(results), min_r2=r2_threshold(condition_key))
        results_by_condition[condition_key].extend(passing)
        plot_aligned_summary(passing, title=f"{condition_label} (R2>={r2_threshold(condition_key)})").savefig(
            f"{output_dir}/{condition_label}_aligned_summary_filtered.png", dpi=150, bbox_inches='tight')
        # plot_aligned_summary(results, title=condition_label).savefig(
        #     f"{output_dir}/{condition_label}_aligned_summary.png", dpi=150, bbox_inches='tight')
        # plot_quality_distributions(results, title=condition_label).savefig(
        #     f"{output_dir}/{condition_label}_r2_distribution.png", dpi=150, bbox_inches='tight')
        plot_quality_distributions(results, title=condition_key, min_r2=r2_threshold(condition_key)).savefig(
            f"{output_dir}/{condition_label}_r2_distribution.png", dpi=150, bbox_inches='tight')
        plot_shuffle_test(results[0]).savefig(
            f"{output_dir}/{condition_label}_shuffle_test_example.png", dpi=150, bbox_inches='tight')
        plt.close('all')
        # m = re.search(r'bar_(ON|OFF)_(v|h)', condition_label, re.IGNORECASE)
        # condition_key = f"{m.group(1).upper()}_{m.group(2).upper()}"
        filtered = filter_by_r2(filter_by_responsiveness(results), min_r2=r2_threshold(condition_key))
        for d in filtered:
            fwhm_by_condition[condition_key][d['genotype']].append(d['fit']['fwhm'])
    
    summary_dir = "C:/phd/02_twophoton/250730_degen_var/4_results/srf_summary"  # adjust to taste
    all_genotypes = set()
    for by_genotype in fwhm_by_condition.values():
        all_genotypes.update(by_genotype.keys())
    for genotype in all_genotypes:
        save_fwhm_by_genotype_csv(fwhm_by_condition, genotype,
                                f"{summary_dir}/{genotype}_fwhm_all_conditions.csv")
    os.makedirs(summary_dir, exist_ok=True)
    for condition_key, by_genotype in fwhm_by_condition.items():
        fig, test_result = plot_fwhm_by_genotype(by_genotype, title=condition_key)
        fig.savefig(f"{summary_dir}/{condition_key}_fwhm_by_genotype.png", dpi=150, bbox_inches='tight')
        save_fwhm_csv(by_genotype, f"{summary_dir}/{condition_key}_fwhm.csv")
        save_variance_test_txt(test_result, f"{summary_dir}/{condition_key}_variance_test.txt", title=condition_key)
        plt.close(fig)
    for polarity in ['ON', 'OFF']:
        v_key, h_key = f'{polarity}_V', f'{polarity}_H'
        if v_key not in results_by_condition or h_key not in results_by_condition:
            continue
        combined = combine_2d_receptive_fields(results_by_condition[v_key],
                                                results_by_condition[h_key], min_r2=R2_THRESH[polarity])
        print(f"{polarity}: {len(results_by_condition[v_key])} V, {len(results_by_condition[h_key])} H, "f"{len(combined)} matched pairs across {len(set(c['genotype'] for c in combined))} genotypes")
        by_genotype_2d = defaultdict(list)
        for c in combined:
            by_genotype_2d[c['genotype']].append(c['fwhm_2d'])
        if not by_genotype_2d:
            print(f"no matched pairs for {polarity} -- skipping")
            continue
        fig, test_result = plot_fwhm_by_genotype(dict(by_genotype_2d), title=f'{polarity}_2D')
        fig.savefig(f"{summary_dir}/{polarity}_2D_fwhm_by_genotype.png", dpi=150, bbox_inches='tight')
        save_fwhm_csv(dict(by_genotype_2d), f"{summary_dir}/{polarity}_2D_fwhm.csv")
        save_variance_test_txt(test_result, f"{summary_dir}/{polarity}_2D_variance_test.txt", title=f'{polarity}_2D')
        plt.close(fig)
        montage = plot_2d_receptive_field_montage(combined, title=f'{polarity}_2D by genotype')
        montage.savefig(f"{summary_dir}/{polarity}_2D_montage_by_genotype.png", dpi=150, bbox_inches='tight')
        plt.close(montage)
