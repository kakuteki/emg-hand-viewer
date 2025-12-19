"""
Ninapro DB5 の Diffusion Maps による可視化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams.update({
    'font.family': ['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif'],
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.unicode_minus': False,
})

print("="*60)
print("Ninapro DB5 - Diffusion Maps 可視化")
print("="*60)

# =============================================================================
# データ読み込み
# =============================================================================
print("\nデータ読み込み中...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']
print(f"総セグメント数: {len(segments)}")

# =============================================================================
# 特徴量抽出（各セグメントを1つのベクトルに）
# =============================================================================
print("\n特徴量抽出中...")

def extract_segment_features(seg):
    """セグメントから特徴量ベクトルを抽出"""
    emg = seg['emg']

    # 時間領域特徴量（チャンネルごと）
    mav = np.mean(np.abs(emg), axis=0)  # 16
    rms = np.sqrt(np.mean(emg**2, axis=0))  # 16
    var = np.var(emg, axis=0)  # 16
    wl = np.sum(np.abs(np.diff(emg, axis=0)), axis=0)  # 16

    # 統計量
    mean_val = np.mean(emg, axis=0)  # 16
    std_val = np.std(emg, axis=0)  # 16

    # 全特徴量を結合
    features = np.concatenate([mav, rms, var, wl, mean_val, std_val])
    return features

# 全セグメントから特徴量抽出
features_list = []
labels_subject = []
labels_movement = []
labels_exercise = []

for seg in tqdm(segments, desc="特徴量抽出"):
    feat = extract_segment_features(seg)
    features_list.append(feat)
    labels_subject.append(seg['subject_id'])
    labels_movement.append(seg['movement'])
    labels_exercise.append(seg['exercise_id'])

X = np.array(features_list)
labels_subject = np.array(labels_subject)
labels_movement = np.array(labels_movement)
labels_exercise = np.array(labels_exercise)

print(f"特徴量行列: {X.shape}")

# =============================================================================
# サンプリング（計算効率のため）
# =============================================================================
MAX_SAMPLES = 2000
if len(X) > MAX_SAMPLES:
    print(f"\nサンプリング: {len(X)} -> {MAX_SAMPLES}")
    np.random.seed(42)
    idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
    X_sampled = X[idx]
    labels_subject_sampled = labels_subject[idx]
    labels_movement_sampled = labels_movement[idx]
    labels_exercise_sampled = labels_exercise[idx]
else:
    X_sampled = X
    labels_subject_sampled = labels_subject
    labels_movement_sampled = labels_movement
    labels_exercise_sampled = labels_exercise

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)

print(f"解析サンプル数: {len(X_scaled)}")

# =============================================================================
# Diffusion Maps 実装
# =============================================================================
print("\nDiffusion Maps 計算中...")

def diffusion_maps(X, n_components=3, alpha=0.5, epsilon='auto'):
    """
    Diffusion Maps による次元削減

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        入力データ
    n_components : int
        出力次元数
    alpha : float
        正規化パラメータ (0: graph Laplacian, 0.5: Fokker-Planck, 1: Laplace-Beltrami)
    epsilon : float or 'auto'
        カーネルバンド幅

    Returns:
    --------
    embedding : array, shape (n_samples, n_components)
        低次元埋め込み
    eigenvalues : array
        固有値
    """
    n_samples = X.shape[0]

    # 距離行列の計算
    print("  距離行列計算中...")
    distances = squareform(pdist(X, metric='euclidean'))

    # εの自動決定（中央値ヒューリスティック）
    if epsilon == 'auto':
        epsilon = np.median(distances[distances > 0]) ** 2
        print(f"  epsilon (自動): {epsilon:.4f}")

    # ガウスカーネル
    print("  カーネル行列計算中...")
    K = np.exp(-distances**2 / epsilon)

    # 密度推定（正規化のため）
    q = np.sum(K, axis=1)

    # α正規化
    K_alpha = K / np.outer(q**alpha, q**alpha)

    # 行正規化（マルコフ行列）
    d = np.sum(K_alpha, axis=1)
    P = K_alpha / d[:, np.newaxis]

    # 対称化（数値安定性のため）
    d_sqrt = np.sqrt(d)
    P_sym = (P * d_sqrt[:, np.newaxis]) / d_sqrt[np.newaxis, :]

    # 固有値分解
    print("  固有値分解中...")
    n_eig = n_components + 1
    eigenvalues, eigenvectors = eigsh(P_sym, k=n_eig, which='LM')

    # 降順にソート
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 右固有ベクトルに変換
    eigenvectors = eigenvectors / d_sqrt[:, np.newaxis]

    # 最初の固有ベクトル（定数）を除外
    embedding = eigenvectors[:, 1:n_components+1]
    eigenvalues = eigenvalues[1:n_components+1]

    # 固有値でスケーリング
    embedding = embedding * eigenvalues[np.newaxis, :]

    return embedding, eigenvalues

# Diffusion Maps 実行
embedding, eigenvalues = diffusion_maps(X_scaled, n_components=3, alpha=0.5)

print(f"\n固有値: {eigenvalues}")
print(f"埋め込み形状: {embedding.shape}")

# =============================================================================
# 可視化
# =============================================================================
print("\n可視化中...")

# 色の設定
colors_subject = plt.cm.tab10(labels_subject_sampled % 10)
colors_exercise = plt.cm.Set1(labels_exercise_sampled % 9)

# ユニークな動作の数を確認
unique_movements = np.unique(labels_movement_sampled)
print(f"動作種類数: {len(unique_movements)}")

# 動作ラベルを0からの連番に変換
movement_to_idx = {m: i for i, m in enumerate(unique_movements)}
movement_idx = np.array([movement_to_idx[m] for m in labels_movement_sampled])
colors_movement = plt.cm.tab20(movement_idx % 20)

# --- Figure 1: 被験者別 ---
fig = plt.figure(figsize=(15, 5))

# 2D (DC1 vs DC2)
ax1 = fig.add_subplot(131)
scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1],
                       c=labels_subject_sampled, cmap='tab10',
                       s=20, alpha=0.6, edgecolors='none')
ax1.set_xlabel('拡散座標 1', fontweight='bold')
ax1.set_ylabel('拡散座標 2', fontweight='bold')
ax1.set_title('被験者別', fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('被験者ID')

# 2D (DC1 vs DC3)
ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 2],
                       c=labels_subject_sampled, cmap='tab10',
                       s=20, alpha=0.6, edgecolors='none')
ax2.set_xlabel('拡散座標 1', fontweight='bold')
ax2.set_ylabel('拡散座標 3', fontweight='bold')
ax2.set_title('被験者別', fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('被験者ID')

# 3D
ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                       c=labels_subject_sampled, cmap='tab10',
                       s=15, alpha=0.6, edgecolors='none')
ax3.set_xlabel('DC1')
ax3.set_ylabel('DC2')
ax3.set_zlabel('DC3')
ax3.set_title('3次元表示', fontweight='bold')

plt.suptitle('Diffusion Maps - Ninapro DB5（被験者別）', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig_DiffusionMaps_被験者別.png', dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("保存: Fig_DiffusionMaps_被験者別.png")

# --- Figure 2: 運動別 ---
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131)
scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1],
                       c=movement_idx, cmap='tab20',
                       s=20, alpha=0.6, edgecolors='none')
ax1.set_xlabel('拡散座標 1', fontweight='bold')
ax1.set_ylabel('拡散座標 2', fontweight='bold')
ax1.set_title('動作別', fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('動作ID')

ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 2],
                       c=movement_idx, cmap='tab20',
                       s=20, alpha=0.6, edgecolors='none')
ax2.set_xlabel('拡散座標 1', fontweight='bold')
ax2.set_ylabel('拡散座標 3', fontweight='bold')
ax2.set_title('動作別', fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('動作ID')

ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                       c=movement_idx, cmap='tab20',
                       s=15, alpha=0.6, edgecolors='none')
ax3.set_xlabel('DC1')
ax3.set_ylabel('DC2')
ax3.set_zlabel('DC3')
ax3.set_title('3次元表示', fontweight='bold')

plt.suptitle('Diffusion Maps - Ninapro DB5（動作別）', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig_DiffusionMaps_動作別.png', dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("保存: Fig_DiffusionMaps_動作別.png")

# --- Figure 3: 課題別 ---
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131)
scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1],
                       c=labels_exercise_sampled, cmap='Set1',
                       s=20, alpha=0.6, edgecolors='none')
ax1.set_xlabel('拡散座標 1', fontweight='bold')
ax1.set_ylabel('拡散座標 2', fontweight='bold')
ax1.set_title('課題別', fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('課題ID')

ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 2],
                       c=labels_exercise_sampled, cmap='Set1',
                       s=20, alpha=0.6, edgecolors='none')
ax2.set_xlabel('拡散座標 1', fontweight='bold')
ax2.set_ylabel('拡散座標 3', fontweight='bold')
ax2.set_title('課題別', fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('課題ID')

ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                       c=labels_exercise_sampled, cmap='Set1',
                       s=15, alpha=0.6, edgecolors='none')
ax3.set_xlabel('DC1')
ax3.set_ylabel('DC2')
ax3.set_zlabel('DC3')
ax3.set_title('3次元表示', fontweight='bold')

plt.suptitle('Diffusion Maps - Ninapro DB5（課題別）', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('Fig_DiffusionMaps_課題別.png', dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("保存: Fig_DiffusionMaps_課題別.png")

# --- Figure 4: 固有値の減衰 ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, len(eigenvalues)+1), eigenvalues, color='steelblue', edgecolor='black')
ax.set_xlabel('拡散座標', fontweight='bold')
ax.set_ylabel('固有値', fontweight='bold')
ax.set_title('Diffusion Maps 固有値', fontweight='bold')
ax.set_xticks(range(1, len(eigenvalues)+1))
plt.tight_layout()
plt.savefig('Fig_DiffusionMaps_固有値.png', dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("保存: Fig_DiffusionMaps_固有値.png")

print("\n" + "="*60)
print("完了")
print("="*60)
