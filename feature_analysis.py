"""
EMG特徴量分析スクリプト
- 時間領域特徴量 (Time-Domain Features)
- 周波数領域特徴量 (Frequency-Domain Features)
- 特徴量と関節角度の相関分析
- 特徴量の重要度評価
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EMG Feature Analysis for Joint Angle Prediction")
print("="*70)

# =====================
# データ読み込み
# =====================
print("\nLoading data...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']

TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

train_segments = [seg for seg in segments if seg['subject_id'] in TRAIN_SUBJECTS]
test_segments = [seg for seg in segments if seg['subject_id'] in TEST_SUBJECTS]

print(f"Train: {len(train_segments)} segments")
print(f"Test: {len(test_segments)} segments")

# =====================
# 特徴量抽出関数 (Time-Domain)
# =====================
def compute_mav(x):
    """Mean Absolute Value - 筋収縮レベルの指標"""
    return np.mean(np.abs(x), axis=0)

def compute_rms(x):
    """Root Mean Square - 平均パワーの指標"""
    return np.sqrt(np.mean(x**2, axis=0))

def compute_wl(x):
    """Waveform Length - 波形の複雑さの指標"""
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def compute_zc(x, threshold=0.01):
    """Zero Crossings - ゼロ交差回数（周波数情報）"""
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    signs = np.sign(x_centered)
    sign_changes = np.abs(np.diff(signs, axis=0))
    # しきい値を超える変化のみカウント
    amplitude_check = np.abs(np.diff(x, axis=0)) > threshold
    return np.sum((sign_changes > 0) & amplitude_check, axis=0)

def compute_ssc(x, threshold=0.01):
    """Slope Sign Change - 傾きの符号変化回数"""
    diff = np.diff(x, axis=0)
    signs = np.sign(diff)
    sign_changes = np.abs(np.diff(signs, axis=0))
    # しきい値を超える変化のみカウント
    amplitude_check = np.abs(diff[:-1]) + np.abs(diff[1:]) > threshold
    return np.sum((sign_changes > 0) & amplitude_check, axis=0)

def compute_wamp(x, threshold=0.1):
    """Willison Amplitude - しきい値を超える振幅変化"""
    diff = np.abs(np.diff(x, axis=0))
    return np.sum(diff > threshold, axis=0)

def compute_var(x):
    """Variance - 分散"""
    return np.var(x, axis=0)

def compute_iemg(x):
    """Integrated EMG - 絶対値の積分"""
    return np.sum(np.abs(x), axis=0)

def compute_dasdv(x):
    """Difference Absolute Standard Deviation Value"""
    diff = np.diff(x, axis=0)
    return np.sqrt(np.mean(diff**2, axis=0))

def compute_log_detector(x):
    """Log Detector - 対数検出器"""
    return np.exp(np.mean(np.log(np.abs(x) + 1e-8), axis=0))

def compute_myop(x, threshold=0.1):
    """Myopulse Percentage Rate"""
    return np.mean(np.abs(x) > threshold, axis=0)

def compute_ar_coeffs(x, order=4):
    """Autoregressive Coefficients (近似)"""
    # 簡易的な自己相関ベースの計算
    coeffs = []
    for ch in range(x.shape[1]):
        signal_ch = x[:, ch]
        ac = np.correlate(signal_ch, signal_ch, mode='full')
        ac = ac[len(ac)//2:]
        ac = ac[:order+1] / (ac[0] + 1e-8)
        coeffs.append(ac[1:])  # AR係数（ラグ1からorder）
    return np.array(coeffs).T.flatten()

# =====================
# 特徴量抽出関数 (Frequency-Domain)
# =====================
def compute_mnf(x, fs=200):
    """Mean Frequency - 平均周波数"""
    mnf = []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        mnf.append(np.sum(f * psd) / (np.sum(psd) + 1e-8))
    return np.array(mnf)

def compute_mdf(x, fs=200):
    """Median Frequency - 中央周波数"""
    mdf = []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        cumsum = np.cumsum(psd)
        mdf.append(f[np.searchsorted(cumsum, cumsum[-1]/2)])
    return np.array(mdf)

def compute_psr(x, fs=200):
    """Power Spectrum Ratio (低周波/高周波)"""
    psr = []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        low_band = psd[f < 50].sum()
        high_band = psd[f >= 50].sum() + 1e-8
        psr.append(low_band / high_band)
    return np.array(psr)

def compute_pkf(x, fs=200):
    """Peak Frequency - ピーク周波数"""
    pkf = []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        pkf.append(f[np.argmax(psd)])
    return np.array(pkf)

def compute_total_power(x, fs=200):
    """Total Power - 全パワー"""
    tp = []
    for ch in range(x.shape[1]):
        f, psd = signal.welch(x[:, ch], fs=fs, nperseg=min(len(x[:, ch]), 20))
        tp.append(np.sum(psd))
    return np.array(tp)

# =====================
# 勾配特徴量（時間変化を捉える）
# =====================
def compute_gradient_features(x):
    """勾配ベースの特徴量"""
    # 最後のタイムステップの勾配
    if len(x) < 2:
        return np.zeros(x.shape[1])
    gradient = x[-1] - x[-2]
    return gradient

def compute_trend(x):
    """線形トレンド（傾き）"""
    trends = []
    for ch in range(x.shape[1]):
        t = np.arange(len(x))
        if len(t) > 1:
            slope = np.polyfit(t, x[:, ch], 1)[0]
        else:
            slope = 0
        trends.append(slope)
    return np.array(trends)

# =====================
# 全特徴量の抽出
# =====================
def extract_all_features(window):
    """ウィンドウから全特徴量を抽出"""
    features = {}

    # Time-Domain Features
    features['MAV'] = compute_mav(window)
    features['RMS'] = compute_rms(window)
    features['WL'] = compute_wl(window)
    features['ZC'] = compute_zc(window)
    features['SSC'] = compute_ssc(window)
    features['WAMP'] = compute_wamp(window)
    features['VAR'] = compute_var(window)
    features['IEMG'] = compute_iemg(window)
    features['DASDV'] = compute_dasdv(window)
    features['LOG'] = compute_log_detector(window)
    features['MYOP'] = compute_myop(window)

    # Gradient Features
    features['GRAD'] = compute_gradient_features(window)
    features['TREND'] = compute_trend(window)

    # Frequency-Domain Features
    features['MNF'] = compute_mnf(window)
    features['MDF'] = compute_mdf(window)
    features['PSR'] = compute_psr(window)
    features['PKF'] = compute_pkf(window)
    features['TP'] = compute_total_power(window)

    return features

# =====================
# サンプルデータで特徴量分析
# =====================
print("\n" + "="*70)
print("Analyzing Feature-Joint Correlations")
print("="*70)

WINDOW_SIZE = 20
SAMPLE_SIZE = 50000  # 分析用のサンプル数

# サンプルデータの作成
print("\nCreating sample windows...")
X_windows = []
y_targets = []
count = 0

for seg in tqdm(train_segments[:500], desc="Processing segments"):  # 最初の500セグメント
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
# 各特徴量を計算して相関分析
# =====================
print("\nExtracting features...")

feature_names = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'WAMP', 'VAR', 'IEMG',
                 'DASDV', 'LOG', 'MYOP', 'GRAD', 'TREND', 'MNF', 'MDF',
                 'PSR', 'PKF', 'TP']

feature_matrices = {name: [] for name in feature_names}

for i in tqdm(range(len(X_windows)), desc="Extracting features"):
    features = extract_all_features(X_windows[i])
    for name in feature_names:
        feature_matrices[name].append(features[name])

# 各特徴量をnumpy配列に変換
for name in feature_names:
    feature_matrices[name] = np.array(feature_matrices[name])

# =====================
# 特徴量と関節角度の相関分析
# =====================
print("\n" + "="*70)
print("Feature-Joint Correlation Analysis")
print("="*70)

correlation_results = {}

for feat_name in feature_names:
    feat_data = feature_matrices[feat_name]
    if feat_data.ndim == 1:
        feat_data = feat_data.reshape(-1, 1)

    # 各チャンネルの特徴量と各関節の相関
    corrs_per_joint = []
    for joint_idx in range(22):
        joint_corrs = []
        for ch in range(feat_data.shape[1]):
            if not np.isnan(feat_data[:, ch]).any() and np.std(feat_data[:, ch]) > 1e-10:
                corr = np.corrcoef(feat_data[:, ch], y_targets[:, joint_idx])[0, 1]
                if not np.isnan(corr):
                    joint_corrs.append(abs(corr))
        if joint_corrs:
            corrs_per_joint.append(np.max(joint_corrs))  # 最大相関を使用
        else:
            corrs_per_joint.append(0)

    mean_corr = np.mean(corrs_per_joint)
    max_corr = np.max(corrs_per_joint)

    correlation_results[feat_name] = {
        'mean_correlation': mean_corr,
        'max_correlation': max_corr,
        'per_joint': corrs_per_joint
    }

# 結果を相関の高い順にソート
sorted_features = sorted(correlation_results.items(),
                         key=lambda x: x[1]['mean_correlation'],
                         reverse=True)

print("\nFeature Ranking by Mean Correlation with Joint Angles:")
print("-"*60)
print(f"{'Rank':<6}{'Feature':<12}{'Mean Corr':>12}{'Max Corr':>12}{'Type':>15}")
print("-"*60)

time_domain = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'WAMP', 'VAR', 'IEMG', 'DASDV', 'LOG', 'MYOP']
freq_domain = ['MNF', 'MDF', 'PSR', 'PKF', 'TP']
gradient_features = ['GRAD', 'TREND']

for rank, (name, data) in enumerate(sorted_features, 1):
    if name in time_domain:
        feat_type = "Time-Domain"
    elif name in freq_domain:
        feat_type = "Freq-Domain"
    else:
        feat_type = "Gradient"
    print(f"{rank:<6}{name:<12}{data['mean_correlation']:>12.4f}{data['max_correlation']:>12.4f}{feat_type:>15}")

# =====================
# 特徴量の組み合わせ評価（Random Forestで重要度評価）
# =====================
print("\n" + "="*70)
print("Feature Importance Analysis (Random Forest)")
print("="*70)

# 全特徴量を結合
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
print(f"Combined feature dimension: {X_features.shape[1]}")

# 正規化
scaler_X = StandardScaler()
X_features_scaled = scaler_X.fit_transform(X_features)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_targets)

# Random Forestで特徴量重要度を評価（サンプルを減らして高速化）
sample_idx = np.random.choice(len(X_features_scaled), min(10000, len(X_features_scaled)), replace=False)
X_sample = X_features_scaled[sample_idx]
y_sample = y_scaled[sample_idx]

print("\nTraining Random Forest for feature importance...")
rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_sample, y_sample)

# 特徴量重要度の集計（特徴量タイプ別）
feature_importance = rf.feature_importances_

# 各特徴量タイプの開始インデックスを計算
feat_indices = {}
idx = 0
for name in feature_names:
    feat_size = feature_matrices[name].shape[1] if feature_matrices[name].ndim > 1 else 1
    feat_indices[name] = (idx, idx + feat_size)
    idx += feat_size

# 特徴量タイプごとの重要度を集計
importance_by_feature = {}
for name in feature_names:
    start, end = feat_indices[name]
    importance_by_feature[name] = np.sum(feature_importance[start:end])

# 重要度でソート
sorted_importance = sorted(importance_by_feature.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance Ranking (Random Forest):")
print("-"*50)
print(f"{'Rank':<6}{'Feature':<12}{'Importance':>15}{'Cumulative':>15}")
print("-"*50)

cumulative = 0
for rank, (name, importance) in enumerate(sorted_importance, 1):
    cumulative += importance
    print(f"{rank:<6}{name:<12}{importance:>15.4f}{cumulative:>15.4f}")

# =====================
# 推奨特徴量セットの決定
# =====================
print("\n" + "="*70)
print("Recommended Feature Sets")
print("="*70)

# 相関ベースのTop特徴量
top_corr_features = [name for name, _ in sorted_features[:8]]

# 重要度ベースのTop特徴量
top_importance_features = [name for name, _ in sorted_importance[:8]]

# 共通する特徴量
common_features = set(top_corr_features) & set(top_importance_features)

print("\nTop 8 by Correlation:", top_corr_features)
print("Top 8 by RF Importance:", top_importance_features)
print("Common (High-Priority):", list(common_features))

# リアルタイム向け推奨セット（計算コスト考慮）
realtime_recommended = []
low_cost_features = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR', 'GRAD']  # 計算コストが低い

for name in sorted_features:
    if name[0] in low_cost_features and name[0] not in realtime_recommended:
        realtime_recommended.append(name[0])
        if len(realtime_recommended) >= 5:
            break

print("\nReal-time Recommended (Low Computational Cost):")
for name in realtime_recommended:
    data = correlation_results[name]
    print(f"  {name}: Mean Corr = {data['mean_correlation']:.4f}")

# =====================
# 結果の保存
# =====================
results = {
    'analysis_info': {
        'sample_size': len(X_windows),
        'window_size': WINDOW_SIZE,
        'num_joints': 22,
        'num_emg_channels': 16
    },
    'correlation_ranking': [
        {
            'rank': i+1,
            'feature': name,
            'mean_correlation': data['mean_correlation'],
            'max_correlation': data['max_correlation'],
            'type': 'Time-Domain' if name in time_domain else ('Freq-Domain' if name in freq_domain else 'Gradient')
        }
        for i, (name, data) in enumerate(sorted_features)
    ],
    'importance_ranking': [
        {
            'rank': i+1,
            'feature': name,
            'importance': float(importance)
        }
        for i, (name, importance) in enumerate(sorted_importance)
    ],
    'recommendations': {
        'top_correlation': top_corr_features,
        'top_importance': top_importance_features,
        'common_high_priority': list(common_features),
        'realtime_recommended': realtime_recommended
    }
}

with open('feature_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: feature_analysis_results.json")

# =====================
# 可視化
# =====================
# 相関ランキングプロット
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
names = [x[0] for x in sorted_features]
corrs = [x[1]['mean_correlation'] for x in sorted_features]
colors = ['#2ecc71' if n in time_domain else '#3498db' if n in freq_domain else '#e74c3c' for n in names]
bars = plt.barh(range(len(names)), corrs, color=colors)
plt.yticks(range(len(names)), names)
plt.xlabel('Mean Correlation with Joint Angles')
plt.title('Feature Correlation Ranking')
plt.gca().invert_yaxis()

# 凡例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Time-Domain'),
    Patch(facecolor='#3498db', label='Freq-Domain'),
    Patch(facecolor='#e74c3c', label='Gradient')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.subplot(1, 2, 2)
names_imp = [x[0] for x in sorted_importance]
imps = [x[1] for x in sorted_importance]
colors_imp = ['#2ecc71' if n in time_domain else '#3498db' if n in freq_domain else '#e74c3c' for n in names_imp]
plt.barh(range(len(names_imp)), imps, color=colors_imp)
plt.yticks(range(len(names_imp)), names_imp)
plt.xlabel('Feature Importance (Random Forest)')
plt.title('Feature Importance Ranking')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('feature_analysis_ranking.png', dpi=150)
plt.close()
print("Saved: feature_analysis_ranking.png")

# 特徴量-関節ヒートマップ
plt.figure(figsize=(16, 10))

# 上位8特徴量の関節別相関をヒートマップで表示
top_8_names = [x[0] for x in sorted_features[:8]]
heatmap_data = []
for name in top_8_names:
    heatmap_data.append(correlation_results[name]['per_joint'])

heatmap_data = np.array(heatmap_data)

plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
plt.colorbar(label='Absolute Correlation')
plt.xticks(range(22), [f'J{i+1}' for i in range(22)], rotation=45)
plt.yticks(range(len(top_8_names)), top_8_names)
plt.xlabel('Joint')
plt.ylabel('Feature')
plt.title('Top 8 Features - Per-Joint Correlation Heatmap')

# 値を表示
for i in range(len(top_8_names)):
    for j in range(22):
        plt.text(j, i, f'{heatmap_data[i, j]:.2f}', ha='center', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('feature_joint_heatmap.png', dpi=150)
plt.close()
print("Saved: feature_joint_heatmap.png")

print("\n" + "="*70)
print("Feature Analysis Complete!")
print("="*70)
