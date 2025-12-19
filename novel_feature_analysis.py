"""
Novel EMG Feature Analysis Script
- State-of-the-art features from 2024 literature
- TKEO, Spectral Moments, Hjorth Parameters, Sample Entropy, etc.
- Feature-Joint Angle Correlation Analysis
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Novel EMG Feature Analysis (2024 State-of-the-Art)")
print("="*70)

# =====================
# Data Loading
# =====================
print("\nLoading data...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']

TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

train_segments = [seg for seg in segments if seg['subject_id'] in TRAIN_SUBJECTS]
print(f"Train: {len(train_segments)} segments")

# =====================
# Novel Feature Extraction Functions
# =====================

# --- TKEO (Teager-Kaiser Energy Operator) ---
def compute_tkeo(x):
    """
    Teager-Kaiser Energy Operator
    Captures instantaneous energy - excellent for detecting muscle activation onset
    TKEO[n] = x[n]^2 - x[n-1]*x[n+1]
    """
    if len(x) < 3:
        return np.zeros(x.shape[1])
    tkeo = x[1:-1]**2 - x[:-2] * x[2:]
    return np.mean(np.abs(tkeo), axis=0)

def compute_tkeo_mean(x):
    """Mean TKEO value"""
    return compute_tkeo(x)

def compute_tkeo_std(x):
    """Standard deviation of TKEO"""
    if len(x) < 3:
        return np.zeros(x.shape[1])
    tkeo = x[1:-1]**2 - x[:-2] * x[2:]
    return np.std(tkeo, axis=0)

# --- Spectral Moments ---
def compute_spectral_moments(x, fs=200):
    """
    Spectral Moments (SM1, SM2, SM3)
    Novel frequency-domain descriptors capturing spectral shape
    """
    sm1, sm2, sm3 = [], [], []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        total_power = np.sum(psd) + 1e-10

        # First spectral moment (centroid)
        m1 = np.sum(f * psd) / total_power
        sm1.append(m1)

        # Second spectral moment (spread)
        m2 = np.sqrt(np.sum((f - m1)**2 * psd) / total_power)
        sm2.append(m2)

        # Third spectral moment (skewness)
        m3 = np.sum((f - m1)**3 * psd) / (total_power * (m2**3 + 1e-10))
        sm3.append(m3)

    return np.array(sm1), np.array(sm2), np.array(sm3)

# --- Sparsity ---
def compute_sparsity(x):
    """
    Sparsity measure using Gini index
    Indicates how concentrated the signal energy is
    """
    sparsity = []
    for ch in range(x.shape[1]):
        abs_x = np.abs(x[:, ch])
        sorted_x = np.sort(abs_x)
        n = len(sorted_x)
        if n == 0 or np.sum(sorted_x) < 1e-10:
            sparsity.append(0)
        else:
            # Gini coefficient
            cumsum = np.cumsum(sorted_x)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            sparsity.append(gini)
    return np.array(sparsity)

# --- Irregularity Factor (IRF) ---
def compute_irf(x, fs=200):
    """
    Irregularity Factor
    Ratio of spectral variability - indicates signal regularity
    """
    irf = []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        total_power = np.sum(psd) + 1e-10

        # Calculate spectral moments for IRF
        m0 = total_power
        m2 = np.sum(f**2 * psd)
        m4 = np.sum(f**4 * psd)

        # IRF = m2^2 / (m0 * m4)
        if m0 * m4 > 1e-10:
            irf.append(m2**2 / (m0 * m4))
        else:
            irf.append(0)
    return np.array(irf)

# --- Waveform Length Ratio (WLR) ---
def compute_wlr(x):
    """
    Waveform Length Ratio
    Ratio of waveform complexity in different parts of the window
    """
    n = len(x)
    if n < 4:
        return np.zeros(x.shape[1])

    mid = n // 2
    wl_first = np.sum(np.abs(np.diff(x[:mid], axis=0)), axis=0)
    wl_second = np.sum(np.abs(np.diff(x[mid:], axis=0)), axis=0)

    return (wl_second + 1e-10) / (wl_first + 1e-10)

# --- Coefficient of Variation (COV) ---
def compute_cov(x):
    """
    Coefficient of Variation
    Normalized measure of dispersion
    """
    mean = np.mean(np.abs(x), axis=0)
    std = np.std(x, axis=0)
    return std / (mean + 1e-10)

# --- Hjorth Parameters ---
def compute_hjorth_activity(x):
    """
    Hjorth Activity (Variance)
    Represents signal power
    """
    return np.var(x, axis=0)

def compute_hjorth_mobility(x):
    """
    Hjorth Mobility
    Square root of ratio of variance of first derivative to variance of signal
    Represents mean frequency
    """
    dx = np.diff(x, axis=0)
    var_x = np.var(x, axis=0) + 1e-10
    var_dx = np.var(dx, axis=0)
    return np.sqrt(var_dx / var_x)

def compute_hjorth_complexity(x):
    """
    Hjorth Complexity
    Represents bandwidth
    """
    dx = np.diff(x, axis=0)
    if len(dx) < 2:
        return np.zeros(x.shape[1])
    ddx = np.diff(dx, axis=0)

    var_dx = np.var(dx, axis=0) + 1e-10
    var_ddx = np.var(ddx, axis=0)

    mobility_x = compute_hjorth_mobility(x)
    mobility_dx = np.sqrt(var_ddx / var_dx)

    return mobility_dx / (mobility_x + 1e-10)

# --- Sample Entropy (Approximate) ---
def compute_sample_entropy(x, m=2, r=0.2):
    """
    Sample Entropy (Approximate)
    Measures signal complexity/regularity - lower = more regular
    """
    se = []
    for ch in range(x.shape[1]):
        data = x[:, ch]
        n = len(data)
        if n < m + 2:
            se.append(0)
            continue

        # Use standard deviation-based threshold
        threshold = r * np.std(data)
        if threshold < 1e-10:
            se.append(0)
            continue

        # Count template matches for length m and m+1
        def count_matches(template_len):
            count = 0
            templates = []
            for i in range(n - template_len):
                templates.append(data[i:i + template_len])

            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < threshold:
                        count += 1
            return count

        A = count_matches(m + 1) + 1e-10
        B = count_matches(m) + 1e-10

        se.append(-np.log(A / B) if A > 0 and B > 0 else 0)

    return np.array(se)

# --- Kurtosis and Skewness ---
def compute_kurtosis(x):
    """
    Kurtosis - measures tail heaviness of distribution
    """
    return kurtosis(x, axis=0, fisher=True)

def compute_skewness(x):
    """
    Skewness - measures asymmetry of distribution
    """
    return skew(x, axis=0)

# --- Normalized Logarithmic Energy ---
def compute_nle(x):
    """
    Normalized Logarithmic Energy
    Log-domain energy measure
    """
    energy = np.sum(x**2, axis=0)
    return np.log(energy / (len(x) + 1e-10) + 1e-10)

# --- Maximum Fractal Length (MFL) ---
def compute_mfl(x):
    """
    Maximum Fractal Length
    Approximation using waveform characteristics
    """
    n = len(x)
    if n < 2:
        return np.zeros(x.shape[1])

    # Calculate fractal-like measure
    diff = np.abs(np.diff(x, axis=0))
    mfl = np.log(np.sum(diff, axis=0) / n + 1e-10)
    return mfl

# --- Wilson Amplitude Enhanced (WAMP with adaptive threshold) ---
def compute_wamp_adaptive(x):
    """
    Wilson Amplitude with adaptive threshold based on signal characteristics
    """
    threshold = 0.5 * np.std(x, axis=0)  # Adaptive threshold
    diff = np.abs(np.diff(x, axis=0))
    return np.mean(diff > threshold, axis=0)

# --- Average Amplitude Change (AAC) ---
def compute_aac(x):
    """
    Average Amplitude Change
    Mean absolute change between consecutive samples
    """
    return np.mean(np.abs(np.diff(x, axis=0)), axis=0)

# --- Difference Absolute Mean Value (DAMV) ---
def compute_damv(x):
    """
    Difference Absolute Mean Value
    """
    diff = np.diff(x, axis=0)
    return np.mean(np.abs(diff), axis=0)

# --- Max-to-Min Ratio ---
def compute_mmr(x):
    """
    Max-to-Min Ratio
    Dynamic range indicator
    """
    max_val = np.max(x, axis=0)
    min_val = np.min(x, axis=0)
    return (max_val - min_val) / (np.abs(min_val) + 1e-10)

# --- Peak Count ---
def compute_peak_count(x, prominence=0.1):
    """
    Number of prominent peaks - indicates muscle firing patterns
    """
    peaks = []
    for ch in range(x.shape[1]):
        try:
            p, _ = signal.find_peaks(x[:, ch], prominence=prominence * np.std(x[:, ch]))
            peaks.append(len(p))
        except:
            peaks.append(0)
    return np.array(peaks)

# --- Spectral Edge Frequency ---
def compute_sef(x, fs=200, edge=0.95):
    """
    Spectral Edge Frequency
    Frequency below which edge% of total power is contained
    """
    sef = []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        cumsum = np.cumsum(psd)
        total = cumsum[-1]
        idx = np.searchsorted(cumsum, edge * total)
        sef.append(f[min(idx, len(f)-1)])
    return np.array(sef)

# --- Mean Absolute Deviation (MAD) ---
def compute_mad(x):
    """
    Mean Absolute Deviation
    Robust measure of variability
    """
    median = np.median(x, axis=0)
    return np.mean(np.abs(x - median), axis=0)

# --- Traditional features for comparison ---
def compute_mav(x):
    return np.mean(np.abs(x), axis=0)

def compute_rms(x):
    return np.sqrt(np.mean(x**2, axis=0))

def compute_var(x):
    return np.var(x, axis=0)

def compute_wl(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

# =====================
# Extract All Novel Features
# =====================
def extract_novel_features(window):
    """Extract all novel features from a window"""
    features = {}

    # TKEO-based features
    features['TKEO'] = compute_tkeo(window)
    features['TKEO_STD'] = compute_tkeo_std(window)

    # Spectral Moments
    sm1, sm2, sm3 = compute_spectral_moments(window)
    features['SM1'] = sm1  # Spectral centroid
    features['SM2'] = sm2  # Spectral spread
    features['SM3'] = sm3  # Spectral skewness

    # Novel descriptors
    features['SPARSITY'] = compute_sparsity(window)
    features['IRF'] = compute_irf(window)  # Irregularity Factor
    features['WLR'] = compute_wlr(window)  # Waveform Length Ratio
    features['COV'] = compute_cov(window)  # Coefficient of Variation

    # Hjorth Parameters
    features['HJORTH_ACT'] = compute_hjorth_activity(window)
    features['HJORTH_MOB'] = compute_hjorth_mobility(window)
    features['HJORTH_COM'] = compute_hjorth_complexity(window)

    # Statistical descriptors
    features['KURTOSIS'] = compute_kurtosis(window)
    features['SKEWNESS'] = compute_skewness(window)

    # Energy features
    features['NLE'] = compute_nle(window)
    features['MFL'] = compute_mfl(window)

    # Amplitude features
    features['WAMP_ADPT'] = compute_wamp_adaptive(window)
    features['AAC'] = compute_aac(window)
    features['DAMV'] = compute_damv(window)
    features['MMR'] = compute_mmr(window)

    # Peak and frequency features
    features['PEAKS'] = compute_peak_count(window)
    features['SEF95'] = compute_sef(window, edge=0.95)
    features['MAD'] = compute_mad(window)

    # Traditional (for comparison)
    features['MAV'] = compute_mav(window)
    features['RMS'] = compute_rms(window)
    features['VAR'] = compute_var(window)
    features['WL'] = compute_wl(window)

    return features

# =====================
# Sample Data Creation
# =====================
print("\n" + "="*70)
print("Creating Sample Windows for Analysis")
print("="*70)

WINDOW_SIZE = 20
SAMPLE_SIZE = 50000

X_windows = []
y_targets = []
count = 0

for seg in tqdm(train_segments[:500], desc="Processing segments"):
    emg = seg['emg']
    glove = seg['glove']
    for i in range(WINDOW_SIZE, len(emg)):
        X_windows.append(emg[i-WINDOW_SIZE:i])
        y_targets.append(glove[i])
        count += 1
        if count >= SAMPLE_SIZE:
            break
    if count >= SAMPLE_SIZE:
        break

X_windows = np.array(X_windows, dtype=np.float32)
y_targets = np.array(y_targets, dtype=np.float32)

print(f"Sample size: {len(X_windows)}")

# =====================
# Feature Extraction
# =====================
print("\n" + "="*70)
print("Extracting Novel Features")
print("="*70)

# Get all feature names from first extraction
sample_features = extract_novel_features(X_windows[0])
feature_names = list(sample_features.keys())
print(f"Total features: {len(feature_names)}")

feature_matrices = {name: [] for name in feature_names}

for i in tqdm(range(len(X_windows)), desc="Extracting features"):
    features = extract_novel_features(X_windows[i])
    for name in feature_names:
        feature_matrices[name].append(features[name])

# Convert to numpy arrays
for name in feature_names:
    feature_matrices[name] = np.array(feature_matrices[name])
    # Handle NaN and Inf
    feature_matrices[name] = np.nan_to_num(feature_matrices[name], nan=0.0, posinf=0.0, neginf=0.0)

# =====================
# Correlation Analysis
# =====================
print("\n" + "="*70)
print("Feature-Joint Correlation Analysis")
print("="*70)

# Define feature categories
novel_features = ['TKEO', 'TKEO_STD', 'SM1', 'SM2', 'SM3', 'SPARSITY', 'IRF',
                  'WLR', 'COV', 'HJORTH_ACT', 'HJORTH_MOB', 'HJORTH_COM',
                  'KURTOSIS', 'SKEWNESS', 'NLE', 'MFL', 'WAMP_ADPT', 'AAC',
                  'DAMV', 'MMR', 'PEAKS', 'SEF95', 'MAD']
traditional_features = ['MAV', 'RMS', 'VAR', 'WL']

correlation_results = {}

for feat_name in feature_names:
    feat_data = feature_matrices[feat_name]
    if feat_data.ndim == 1:
        feat_data = feat_data.reshape(-1, 1)

    corrs_per_joint = []
    for joint_idx in range(22):
        joint_corrs = []
        for ch in range(feat_data.shape[1]):
            if np.std(feat_data[:, ch]) > 1e-10:
                corr = np.corrcoef(feat_data[:, ch], y_targets[:, joint_idx])[0, 1]
                if not np.isnan(corr):
                    joint_corrs.append(abs(corr))
        if joint_corrs:
            corrs_per_joint.append(np.max(joint_corrs))
        else:
            corrs_per_joint.append(0)

    mean_corr = np.mean(corrs_per_joint)
    max_corr = np.max(corrs_per_joint)

    correlation_results[feat_name] = {
        'mean_correlation': mean_corr,
        'max_correlation': max_corr,
        'per_joint': corrs_per_joint,
        'is_novel': feat_name in novel_features
    }

# Sort by correlation
sorted_features = sorted(correlation_results.items(),
                        key=lambda x: x[1]['mean_correlation'],
                        reverse=True)

print("\nFeature Ranking by Mean Correlation:")
print("-"*75)
print(f"{'Rank':<6}{'Feature':<15}{'Mean Corr':>12}{'Max Corr':>12}{'Type':>15}{'Novel':>10}")
print("-"*75)

for rank, (name, data) in enumerate(sorted_features, 1):
    is_novel = "Novel" if data['is_novel'] else "Traditional"
    feat_type = "Novel" if name in novel_features else "Traditional"
    print(f"{rank:<6}{name:<15}{data['mean_correlation']:>12.4f}{data['max_correlation']:>12.4f}{feat_type:>15}{is_novel:>10}")

# =====================
# Random Forest Feature Importance
# =====================
print("\n" + "="*70)
print("Feature Importance Analysis (Random Forest)")
print("="*70)

# Combine all features
all_features = []
for i in range(len(X_windows)):
    feat_vec = []
    for name in feature_names:
        feat = feature_matrices[name][i]
        if isinstance(feat, np.ndarray):
            feat_vec.extend(feat.flatten())
        else:
            feat_vec.append(feat)
    all_features.append(feat_vec)

X_features = np.array(all_features)
X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Combined feature dimension: {X_features.shape[1]}")

# Normalize
scaler_X = StandardScaler()
X_features_scaled = scaler_X.fit_transform(X_features)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_targets)

# Random Forest
sample_idx = np.random.choice(len(X_features_scaled), min(10000, len(X_features_scaled)), replace=False)
X_sample = X_features_scaled[sample_idx]
y_sample = y_scaled[sample_idx]

print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_sample, y_sample)

# Calculate importance by feature type
feature_importance = rf.feature_importances_

feat_indices = {}
idx = 0
for name in feature_names:
    feat_size = feature_matrices[name].shape[1] if feature_matrices[name].ndim > 1 else 1
    feat_indices[name] = (idx, idx + feat_size)
    idx += feat_size

importance_by_feature = {}
for name in feature_names:
    start, end = feat_indices[name]
    importance_by_feature[name] = np.sum(feature_importance[start:end])

sorted_importance = sorted(importance_by_feature.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance Ranking:")
print("-"*60)
print(f"{'Rank':<6}{'Feature':<15}{'Importance':>15}{'Cumulative':>15}{'Novel':>10}")
print("-"*60)

cumulative = 0
for rank, (name, importance) in enumerate(sorted_importance, 1):
    cumulative += importance
    is_novel = "Novel" if name in novel_features else "Traditional"
    print(f"{rank:<6}{name:<15}{importance:>15.4f}{cumulative:>15.4f}{is_novel:>10}")

# =====================
# Analysis Summary
# =====================
print("\n" + "="*70)
print("NOVEL vs TRADITIONAL COMPARISON")
print("="*70)

# Novel features stats
novel_corrs = [correlation_results[n]['mean_correlation'] for n in novel_features if n in correlation_results]
trad_corrs = [correlation_results[n]['mean_correlation'] for n in traditional_features if n in correlation_results]

novel_imps = [importance_by_feature[n] for n in novel_features if n in importance_by_feature]
trad_imps = [importance_by_feature[n] for n in traditional_features if n in importance_by_feature]

print(f"\nNovel Features ({len(novel_features)} features):")
print(f"  Mean Correlation: {np.mean(novel_corrs):.4f} (max: {np.max(novel_corrs):.4f})")
print(f"  Total Importance: {np.sum(novel_imps):.4f}")

print(f"\nTraditional Features ({len(traditional_features)} features):")
print(f"  Mean Correlation: {np.mean(trad_corrs):.4f} (max: {np.max(trad_corrs):.4f})")
print(f"  Total Importance: {np.sum(trad_imps):.4f}")

# Best novel features
print("\n" + "="*70)
print("TOP NOVEL FEATURES")
print("="*70)

top_novel = [(n, d) for n, d in sorted_features if d['is_novel']][:10]
print("\nTop 10 Novel Features by Correlation:")
for rank, (name, data) in enumerate(top_novel, 1):
    print(f"  {rank}. {name}: Mean={data['mean_correlation']:.4f}, Max={data['max_correlation']:.4f}")

# =====================
# Save Results
# =====================
results = {
    'analysis_info': {
        'sample_size': len(X_windows),
        'window_size': WINDOW_SIZE,
        'num_joints': 22,
        'num_emg_channels': 16,
        'total_features': len(feature_names),
        'novel_features_count': len(novel_features),
        'traditional_features_count': len(traditional_features)
    },
    'correlation_ranking': [
        {
            'rank': i+1,
            'feature': name,
            'mean_correlation': float(data['mean_correlation']),
            'max_correlation': float(data['max_correlation']),
            'is_novel': data['is_novel']
        }
        for i, (name, data) in enumerate(sorted_features)
    ],
    'importance_ranking': [
        {
            'rank': i+1,
            'feature': name,
            'importance': float(importance),
            'is_novel': name in novel_features
        }
        for i, (name, importance) in enumerate(sorted_importance)
    ],
    'novel_vs_traditional': {
        'novel_mean_correlation': float(np.mean(novel_corrs)),
        'novel_max_correlation': float(np.max(novel_corrs)),
        'novel_total_importance': float(np.sum(novel_imps)),
        'traditional_mean_correlation': float(np.mean(trad_corrs)),
        'traditional_max_correlation': float(np.max(trad_corrs)),
        'traditional_total_importance': float(np.sum(trad_imps))
    },
    'top_novel_features': [name for name, _ in top_novel],
    'recommendations': {
        'best_overall': [name for name, _ in sorted_features[:8]],
        'best_novel': [name for name, _ in top_novel[:6]],
        'best_combined': list(set([n for n, _ in sorted_features[:5]] + [n for n, _ in top_novel[:5]]))
    }
}

with open('novel_feature_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: novel_feature_analysis_results.json")

# =====================
# Visualization
# =====================
plt.figure(figsize=(16, 12))

# 1. Correlation ranking
plt.subplot(2, 2, 1)
names = [x[0] for x in sorted_features[:15]]
corrs = [x[1]['mean_correlation'] for x in sorted_features[:15]]
colors = ['#e74c3c' if correlation_results[n]['is_novel'] else '#3498db' for n in names]
plt.barh(range(len(names)), corrs, color=colors)
plt.yticks(range(len(names)), names)
plt.xlabel('Mean Correlation')
plt.title('Top 15 Features - Correlation Ranking')
plt.gca().invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Novel'), Patch(facecolor='#3498db', label='Traditional')]
plt.legend(handles=legend_elements, loc='lower right')

# 2. Importance ranking
plt.subplot(2, 2, 2)
names_imp = [x[0] for x in sorted_importance[:15]]
imps = [x[1] for x in sorted_importance[:15]]
colors_imp = ['#e74c3c' if n in novel_features else '#3498db' for n in names_imp]
plt.barh(range(len(names_imp)), imps, color=colors_imp)
plt.yticks(range(len(names_imp)), names_imp)
plt.xlabel('Feature Importance')
plt.title('Top 15 Features - RF Importance')
plt.gca().invert_yaxis()

# 3. Novel vs Traditional comparison
plt.subplot(2, 2, 3)
categories = ['Novel', 'Traditional']
corr_means = [np.mean(novel_corrs), np.mean(trad_corrs)]
corr_maxs = [np.max(novel_corrs), np.max(trad_corrs)]

x = np.arange(len(categories))
width = 0.35
plt.bar(x - width/2, corr_means, width, label='Mean Corr', color='#3498db')
plt.bar(x + width/2, corr_maxs, width, label='Max Corr', color='#e74c3c')
plt.xticks(x, categories)
plt.ylabel('Correlation')
plt.title('Novel vs Traditional: Correlation')
plt.legend()

# 4. Feature-joint heatmap for top novel features
plt.subplot(2, 2, 4)
top_novel_names = [name for name, _ in top_novel[:8]]
heatmap_data = []
for name in top_novel_names:
    heatmap_data.append(correlation_results[name]['per_joint'])

heatmap_data = np.array(heatmap_data)
plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
plt.colorbar(label='|Correlation|')
plt.xticks(range(22), [f'J{i+1}' for i in range(22)], rotation=45, fontsize=7)
plt.yticks(range(len(top_novel_names)), top_novel_names)
plt.xlabel('Joint')
plt.ylabel('Feature')
plt.title('Top 8 Novel Features - Joint Correlation')

plt.tight_layout()
plt.savefig('novel_feature_analysis.png', dpi=150)
plt.close()
print("Saved: novel_feature_analysis.png")

print("\n" + "="*70)
print("Novel Feature Analysis Complete!")
print("="*70)
