"""
Integrated Gradients Analysis for CNN-LSTM EMG Model
EMG信号のどの部分がモデル予測に影響を与えているかを可視化
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Captumがインストールされていない場合のためにIntegrated Gradientsを自前実装
try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    USE_CAPTUM = True
    print("Using Captum library for Integrated Gradients")
except ImportError:
    USE_CAPTUM = False
    print("Captum not found. Using custom Integrated Gradients implementation")

# =====================
# 設定
# =====================
WINDOW_SIZE = 20
BATCH_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 2
FC_SIZE = 64
DROPOUT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJECTS = [9, 10]

# EMGチャンネル名（NinaPro DB5）
EMG_CHANNEL_NAMES = [
    'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8',
    'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16'
]

# 関節名（22関節）
JOINT_NAMES = [
    'Thumb_CMC_F', 'Thumb_CMC_A', 'Thumb_MCP', 'Thumb_IP',
    'Index_MCP_F', 'Index_MCP_A', 'Index_PIP', 'Index_DIP',
    'Middle_MCP_F', 'Middle_MCP_A', 'Middle_PIP', 'Middle_DIP',
    'Ring_MCP_F', 'Ring_MCP_A', 'Ring_PIP', 'Ring_DIP',
    'Pinky_MCP_F', 'Pinky_MCP_A', 'Pinky_PIP', 'Pinky_DIP',
    'Wrist_F', 'Wrist_R'
]

print("="*60)
print("Integrated Gradients Analysis for CNN-LSTM")
print("="*60)
print(f"Device: {DEVICE}")

# =====================
# CNN-LSTMモデル定義（学習スクリプトと同じ）
# =====================
class EMGtoJointCNNLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=256, num_layers=2, fc_size=64, dropout=0.2, output_size=22):
        super(EMGtoJointCNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size // 2, output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 特定の関節出力のみを返すラッパー
class ModelWrapper(nn.Module):
    def __init__(self, model, joint_idx):
        super().__init__()
        self.model = model
        self.joint_idx = joint_idx

    def forward(self, x):
        out = self.model(x)
        return out[:, self.joint_idx:self.joint_idx+1]

# =====================
# 自前実装のIntegrated Gradients
# =====================
def integrated_gradients_custom(model, input_tensor, baseline, target_idx, steps=50):
    """
    Integrated Gradientsの自前実装
    """
    # LSTM backward requires training mode for cuDNN
    was_training = model.training
    model.train()

    # 勾配を累積
    accumulated_grads = torch.zeros_like(input_tensor)

    for i in range(steps + 1):
        # 補間入力を作成
        alpha = float(i) / steps
        scaled_input = baseline + alpha * (input_tensor - baseline)
        scaled_input = scaled_input.clone().detach().requires_grad_(True)

        # 順伝播
        output = model(scaled_input)
        if output.dim() > 1:
            output = output[:, target_idx]

        # 逆伝播
        model.zero_grad()
        output.sum().backward()

        # 勾配を累積
        accumulated_grads += scaled_input.grad

    # 勾配の平均
    avg_grads = accumulated_grads / (steps + 1)

    # Integrated Gradients = (input - baseline) * avg_grads
    integrated_grads = (input_tensor - baseline) * avg_grads

    # Restore original mode
    if not was_training:
        model.eval()

    return integrated_grads.detach()

# =====================
# データ読み込み
# =====================
print("\nLoading data...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']

test_segments = [seg for seg in segments if seg['subject_id'] in TEST_SUBJECTS]
print(f"Test segments: {len(test_segments)}")

# ウィンドウ作成（テストデータのみ）
def create_windows(segments, window_size):
    X_list = []
    y_list = []
    for seg in tqdm(segments, desc="Creating windows", ncols=100):
        emg = seg['emg']
        glove = seg['glove']
        for i in range(window_size, len(emg)):
            X_list.append(emg[i-window_size:i])
            y_list.append(glove[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

X_test, y_test = create_windows(test_segments, WINDOW_SIZE)
print(f"X_test shape: {X_test.shape}")

# 正規化（学習時と同じスケーラーを再現するためにtrainデータも読み込み）
print("Loading training data for scaler fitting...")
train_segments = [seg for seg in segments if seg['subject_id'] not in TEST_SUBJECTS]
X_train, y_train = create_windows(train_segments, WINDOW_SIZE)

emg_scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
emg_scaler.fit(X_train_reshaped)

joint_scaler = StandardScaler()
joint_scaler.fit(y_train)

# テストデータを正規化
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_reshaped = emg_scaler.transform(X_test_reshaped)
X_test_normalized = X_test_reshaped.reshape(X_test.shape)

del X_train, y_train, train_segments  # メモリ解放

# =====================
# モデル読み込み
# =====================
print("\nLoading model...")
model = EMGtoJointCNNLSTM(
    input_size=16, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
    fc_size=FC_SIZE, dropout=DROPOUT, output_size=22
).to(DEVICE)

model.load_state_dict(torch.load('best_model_cnn_lstm.pth', map_location=DEVICE))
model.eval()
print("Model loaded successfully")

# =====================
# Integrated Gradients計算
# =====================
print("\n" + "="*60)
print("Computing Integrated Gradients...")
print("="*60)

# サンプル選択（複数のサンプルで平均を取る）
num_samples = 50  # 解析するサンプル数
sample_indices = np.random.choice(len(X_test_normalized), num_samples, replace=False)

# 結果を格納
all_attributions = []  # shape: (num_samples, seq_len, channels, num_joints)

# ベースライン（ゼロ入力）
baseline = torch.zeros(1, WINDOW_SIZE, 16).to(DEVICE)

for idx in tqdm(sample_indices, desc="Computing IG for samples"):
    input_tensor = torch.FloatTensor(X_test_normalized[idx:idx+1]).to(DEVICE)

    sample_attributions = []  # 各関節への帰属

    for joint_idx in range(22):
        if USE_CAPTUM:
            wrapped_model = ModelWrapper(model, joint_idx)
            ig = IntegratedGradients(wrapped_model)
            attr = ig.attribute(input_tensor, baselines=baseline, n_steps=50)
            attr = attr.detach().cpu().numpy()[0]
        else:
            attr = integrated_gradients_custom(
                model,
                input_tensor,
                baseline,
                target_idx=joint_idx,
                steps=50
            ).cpu().numpy()[0]

        sample_attributions.append(attr)

    # shape: (22, seq_len, channels) -> transpose to (seq_len, channels, 22)
    sample_attributions = np.array(sample_attributions).transpose(1, 2, 0)
    all_attributions.append(sample_attributions)

all_attributions = np.array(all_attributions)  # (num_samples, seq_len, channels, num_joints)
print(f"Attributions shape: {all_attributions.shape}")

# =====================
# 可視化1: 平均影響度ヒートマップ（チャンネル×時間）
# =====================
print("\nGenerating visualizations...")

# 全サンプル・全関節の平均帰属
mean_attr_all = np.mean(np.abs(all_attributions), axis=(0, 3))  # (seq_len, channels)

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(mean_attr_all.T, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('EMG Channel', fontsize=12)
ax.set_title('Integrated Gradients: Average Attribution (All Joints)', fontsize=14)
ax.set_xticks(range(WINDOW_SIZE))
ax.set_yticks(range(16))
ax.set_yticklabels(EMG_CHANNEL_NAMES)
plt.colorbar(im, ax=ax, label='Attribution Magnitude')
plt.tight_layout()
plt.savefig('ig_heatmap_average.png', dpi=150)
plt.close()
print("Saved: ig_heatmap_average.png")

# =====================
# 可視化2: 関節ごとの影響度ヒートマップ
# =====================
fig, axes = plt.subplots(4, 6, figsize=(20, 14))
axes = axes.flatten()

for joint_idx in range(22):
    ax = axes[joint_idx]
    joint_attr = np.mean(np.abs(all_attributions[:, :, :, joint_idx]), axis=0)  # (seq_len, channels)
    im = ax.imshow(joint_attr.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_title(f'{JOINT_NAMES[joint_idx]}', fontsize=9)
    ax.set_xticks([0, 9, 19])
    ax.set_yticks([0, 7, 15])
    if joint_idx >= 18:
        ax.set_xlabel('Time', fontsize=8)
    if joint_idx % 6 == 0:
        ax.set_ylabel('Channel', fontsize=8)

# 余りの軸を非表示
for idx in range(22, 24):
    axes[idx].axis('off')

plt.suptitle('Integrated Gradients Attribution per Joint', fontsize=14)
plt.tight_layout()
plt.savefig('ig_heatmap_per_joint.png', dpi=150)
plt.close()
print("Saved: ig_heatmap_per_joint.png")

# =====================
# 可視化3: チャンネル重要度ランキング
# =====================
channel_importance = np.mean(np.abs(all_attributions), axis=(0, 1, 3))  # (channels,)

fig, ax = plt.subplots(figsize=(12, 6))
sorted_idx = np.argsort(channel_importance)[::-1]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 16))
bars = ax.barh(range(16), channel_importance[sorted_idx], color=colors)
ax.set_yticks(range(16))
ax.set_yticklabels([EMG_CHANNEL_NAMES[i] for i in sorted_idx])
ax.invert_yaxis()
ax.set_xlabel('Average Attribution Magnitude', fontsize=12)
ax.set_title('EMG Channel Importance Ranking (Integrated Gradients)', fontsize=14)
ax.grid(True, axis='x', alpha=0.3)

# 値をバーの横に表示
for i, (idx, val) in enumerate(zip(sorted_idx, channel_importance[sorted_idx])):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('ig_channel_importance.png', dpi=150)
plt.close()
print("Saved: ig_channel_importance.png")

# =====================
# 可視化4: 時間方向の重要度
# =====================
time_importance = np.mean(np.abs(all_attributions), axis=(0, 2, 3))  # (seq_len,)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(WINDOW_SIZE), time_importance, color='steelblue', edgecolor='navy')
ax.set_xlabel('Time Step (past → present)', fontsize=12)
ax.set_ylabel('Average Attribution Magnitude', fontsize=12)
ax.set_title('Temporal Importance (Integrated Gradients)', fontsize=14)
ax.set_xticks(range(WINDOW_SIZE))
ax.set_xticklabels([f't-{WINDOW_SIZE-1-i}' if i < WINDOW_SIZE-1 else 't' for i in range(WINDOW_SIZE)])
ax.grid(True, axis='y', alpha=0.3)

# 最新のタイムステップを強調
ax.axvline(x=WINDOW_SIZE-1, color='red', linestyle='--', alpha=0.7, label='Current time')
ax.legend()

plt.tight_layout()
plt.savefig('ig_temporal_importance.png', dpi=150)
plt.close()
print("Saved: ig_temporal_importance.png")

# =====================
# 可視化5: チャンネル×関節の影響度マトリクス
# =====================
channel_joint_attr = np.mean(np.abs(all_attributions), axis=(0, 1))  # (channels, joints)

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(channel_joint_attr, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Joint', fontsize=12)
ax.set_ylabel('EMG Channel', fontsize=12)
ax.set_title('EMG Channel to Joint Attribution Matrix', fontsize=14)
ax.set_xticks(range(22))
ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(16))
ax.set_yticklabels(EMG_CHANNEL_NAMES)
plt.colorbar(im, ax=ax, label='Attribution Magnitude')
plt.tight_layout()
plt.savefig('ig_channel_joint_matrix.png', dpi=150)
plt.close()
print("Saved: ig_channel_joint_matrix.png")

# =====================
# アニメーション1: 時系列での影響度変化
# =====================
print("\nCreating animations...")

# 連続したサンプルを取得（アニメーション用）
anim_start = 0
anim_length = 100  # フレーム数
anim_samples = X_test_normalized[anim_start:anim_start+anim_length]

print("Computing attributions for animation...")
anim_attributions = []
for i in tqdm(range(anim_length), desc="Animation frames"):
    input_tensor = torch.FloatTensor(anim_samples[i:i+1]).to(DEVICE)

    # 全関節の平均帰属を計算
    joint_attrs = []
    for joint_idx in range(22):
        if USE_CAPTUM:
            wrapped_model = ModelWrapper(model, joint_idx)
            ig = IntegratedGradients(wrapped_model)
            attr = ig.attribute(input_tensor, baselines=baseline, n_steps=30)
            attr = attr.detach().cpu().numpy()[0]
        else:
            attr = integrated_gradients_custom(
                model,
                input_tensor,
                baseline,
                target_idx=joint_idx,
                steps=30
            ).cpu().numpy()[0]
        joint_attrs.append(attr)

    # 全関節の絶対値平均
    mean_attr = np.mean(np.abs(np.array(joint_attrs)), axis=0)  # (seq_len, channels)
    anim_attributions.append(mean_attr)

anim_attributions = np.array(anim_attributions)  # (frames, seq_len, channels)

# アニメーション作成
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 上段: EMG信号
ax1 = axes[0]
# 下段: Attribution heatmap
ax2 = axes[1]

# 初期化
line_data = anim_samples[0]  # (seq_len, channels)
emg_lines = []
for ch in range(16):
    line, = ax1.plot(range(WINDOW_SIZE), line_data[:, ch] + ch * 3, lw=1)
    emg_lines.append(line)

ax1.set_xlim(0, WINDOW_SIZE-1)
ax1.set_ylim(-2, 16 * 3 + 2)
ax1.set_xlabel('Time Step')
ax1.set_ylabel('EMG Channels (offset for visibility)')
ax1.set_title('EMG Signal Input')
ax1.set_yticks([i * 3 for i in range(16)])
ax1.set_yticklabels(EMG_CHANNEL_NAMES)

im = ax2.imshow(anim_attributions[0].T, aspect='auto', cmap='hot',
                interpolation='nearest', vmin=0, vmax=np.percentile(anim_attributions, 95))
ax2.set_xlabel('Time Step')
ax2.set_ylabel('EMG Channel')
ax2.set_title('Integrated Gradients Attribution')
ax2.set_yticks(range(16))
ax2.set_yticklabels(EMG_CHANNEL_NAMES)
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Attribution')

frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

def update(frame):
    # EMG信号を更新
    line_data = anim_samples[frame]
    for ch, line in enumerate(emg_lines):
        line.set_ydata(line_data[:, ch] + ch * 3)

    # Attributionヒートマップを更新
    im.set_array(anim_attributions[frame].T)

    frame_text.set_text(f'Frame: {frame+1}/{anim_length}')

    return emg_lines + [im, frame_text]

print("Rendering animation (this may take a while)...")
anim = animation.FuncAnimation(fig, update, frames=anim_length, interval=100, blit=True)
anim.save('ig_animation.gif', writer='pillow', fps=10)
plt.close()
print("Saved: ig_animation.gif")

# =====================
# アニメーション2: チャンネル重要度の時間変化
# =====================
fig, ax = plt.subplots(figsize=(12, 8))

# チャンネルごとの重要度（時間方向に集約）
channel_over_time = np.mean(anim_attributions, axis=1)  # (frames, channels)

bars = ax.barh(range(16), channel_over_time[0], color='steelblue')
ax.set_xlim(0, np.max(channel_over_time) * 1.1)
ax.set_yticks(range(16))
ax.set_yticklabels(EMG_CHANNEL_NAMES)
ax.set_xlabel('Attribution Magnitude')
ax.set_title('EMG Channel Importance Over Time')
ax.invert_yaxis()

frame_text2 = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right', fontsize=12)

def update_bars(frame):
    for bar, val in zip(bars, channel_over_time[frame]):
        bar.set_width(val)
    frame_text2.set_text(f'Frame: {frame+1}/{anim_length}')
    return list(bars) + [frame_text2]

anim2 = animation.FuncAnimation(fig, update_bars, frames=anim_length, interval=100, blit=True)
anim2.save('ig_channel_animation.gif', writer='pillow', fps=10)
plt.close()
print("Saved: ig_channel_animation.gif")

# =====================
# 結果をJSONに保存
# =====================
results = {
    'channel_importance': {name: float(val) for name, val in zip(EMG_CHANNEL_NAMES, channel_importance)},
    'channel_importance_ranking': [EMG_CHANNEL_NAMES[i] for i in sorted_idx],
    'temporal_importance': time_importance.tolist(),
    'channel_joint_attribution': channel_joint_attr.tolist(),
    'num_samples_analyzed': num_samples,
    'window_size': WINDOW_SIZE,
    'num_joints': 22,
    'num_channels': 16
}

with open('ig_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: ig_analysis_results.json")

# =====================
# サマリー表示
# =====================
print("\n" + "="*60)
print("INTEGRATED GRADIENTS ANALYSIS SUMMARY")
print("="*60)
print("\n【EMG Channel Importance Ranking】")
for i, idx in enumerate(sorted_idx[:5]):
    print(f"  {i+1}. {EMG_CHANNEL_NAMES[idx]}: {channel_importance[idx]:.4f}")
print("  ...")

print("\n【Temporal Importance】")
print(f"  Most important time step: t-{WINDOW_SIZE-1-np.argmax(time_importance)} (value: {np.max(time_importance):.4f})")
print(f"  Current time (t): {time_importance[-1]:.4f}")

print("\n【Generated Files】")
print("  Static Images:")
print("    - ig_heatmap_average.png       : Average attribution heatmap")
print("    - ig_heatmap_per_joint.png     : Per-joint attribution heatmaps")
print("    - ig_channel_importance.png    : Channel importance ranking")
print("    - ig_temporal_importance.png   : Temporal importance")
print("    - ig_channel_joint_matrix.png  : Channel-Joint attribution matrix")
print("  Animations:")
print("    - ig_animation.gif             : EMG signal + Attribution over time")
print("    - ig_channel_animation.gif     : Channel importance bar chart animation")
print("  Data:")
print("    - ig_analysis_results.json     : Numerical results")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
