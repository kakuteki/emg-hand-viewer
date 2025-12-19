"""
Log-Frequency EMG Spectrogram Attention-LSTM モデル
EMG周波数帯域（5-500Hz）に対数スケールを適用したスペクトログラムで学習
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import math

# =====================
# ハイパーパラメータ
# =====================
WINDOW_SIZE = 64  # 他モデルと同条件
BATCH_SIZE = 1024
HIDDEN_SIZE = 256
NUM_LAYERS = 2
FC_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0002
EPOCHS = 50

# Attention設定
NUM_HEADS = 4
ATTENTION_DIM = 64

# EMGスペクトログラム設定
SAMPLE_RATE = 200  # Hz (NinaPro DB5のサンプリングレート)
N_FFT = 64  # FFTサイズ
HOP_LENGTH = 4  # ホップ長
N_FREQ_BINS = 32  # 対数周波数ビン数
EMG_FREQ_MIN = 5  # EMG最低周波数 (Hz)
EMG_FREQ_MAX = 100  # EMG最高周波数 (Hz) - ナイキスト周波数=100Hz

# 固定設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

print("="*60)
print("Log-Frequency EMG Spectrogram Attention-LSTM Training")
print("="*60)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()
print("EMG Spectrogram Parameters:")
print(f"  Sample Rate: {SAMPLE_RATE} Hz")
print(f"  N_FFT: {N_FFT}")
print(f"  Hop Length: {HOP_LENGTH}")
print(f"  Frequency Bins: {N_FREQ_BINS}")
print(f"  Frequency Range: {EMG_FREQ_MIN}-{EMG_FREQ_MAX} Hz (Log scale)")
print()
print("Model Hyperparameters:")
print(f"  window_size: {WINDOW_SIZE}")
print(f"  batch_size: {BATCH_SIZE}")
print(f"  hidden_size: {HIDDEN_SIZE}")
print(f"  num_layers: {NUM_LAYERS}")
print(f"  epochs: {EPOCHS}")
print()

# =====================
# EMG用対数周波数スペクトログラム
# =====================
class EMGLogFreqSpectrogram(nn.Module):
    """
    EMG信号用の対数周波数スケールスペクトログラム
    Mel scaleの代わりにEMG帯域（5-500Hz）に最適化した対数スケールを使用
    """
    def __init__(self, sample_rate=200, n_fft=64, hop_length=4,
                 n_freq_bins=32, freq_min=5, freq_max=100):
        super(EMGLogFreqSpectrogram, self).__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_bins = n_freq_bins
        self.freq_min = freq_min
        self.freq_max = freq_max

        # FFT周波数ビン
        n_fft_bins = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sample_rate / 2, n_fft_bins)

        # 対数スケールの周波数ビン境界を作成
        log_freq_min = np.log10(max(freq_min, 1))
        log_freq_max = np.log10(freq_max)
        log_freqs = torch.logspace(log_freq_min, log_freq_max, n_freq_bins + 1, base=10)

        # フィルタバンクを作成（三角フィルタ）
        filterbank = torch.zeros(n_freq_bins, n_fft_bins)
        for i in range(n_freq_bins):
            f_low = log_freqs[i]
            f_center = (log_freqs[i] + log_freqs[i+1]) / 2
            f_high = log_freqs[i+1]

            for j, f in enumerate(fft_freqs):
                if f_low <= f <= f_center:
                    filterbank[i, j] = (f - f_low) / (f_center - f_low + 1e-6)
                elif f_center < f <= f_high:
                    filterbank[i, j] = (f_high - f) / (f_high - f_center + 1e-6)

        # 正規化
        filterbank = filterbank / (filterbank.sum(dim=1, keepdim=True) + 1e-6)

        self.register_buffer('filterbank', filterbank)

        # ハニング窓
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)

        print(f"  Log-Freq Filterbank shape: {filterbank.shape}")
        print(f"  Frequency bins: {log_freqs.numpy()[:5]}... to {log_freqs.numpy()[-3:]} Hz")

    def forward(self, x):
        """
        入力: x (batch, seq_len, n_channels) - 生EMG信号
        出力: (batch, n_time_frames, n_channels * n_freq_bins)
        """
        batch_size, seq_len, n_channels = x.shape

        specs = []
        for ch in range(n_channels):
            # チャンネルごとにSTFT
            signal = x[:, :, ch]  # (batch, seq_len)

            # パディング
            pad_len = self.n_fft // 2
            signal_padded = F.pad(signal, (pad_len, pad_len), mode='reflect')

            # フレーム分割
            frames = signal_padded.unfold(1, self.n_fft, self.hop_length)  # (batch, n_frames, n_fft)

            # 窓関数適用
            frames = frames * self.window

            # FFT
            fft_result = torch.fft.rfft(frames, dim=-1)
            power_spec = torch.abs(fft_result) ** 2  # (batch, n_frames, n_fft_bins)

            # 対数周波数スケールに変換
            log_freq_spec = torch.matmul(power_spec, self.filterbank.T)  # (batch, n_frames, n_freq_bins)

            # 対数スケール（パワー → dB的な変換）
            log_freq_spec = torch.log1p(log_freq_spec)

            specs.append(log_freq_spec)

        # 全チャンネルを結合
        output = torch.cat(specs, dim=-1)  # (batch, n_frames, n_channels * n_freq_bins)

        return output


# =====================
# データ読み込み
# =====================
print("Loading data...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']

train_segments = [seg for seg in segments if seg['subject_id'] in TRAIN_SUBJECTS]
test_segments = [seg for seg in segments if seg['subject_id'] in TEST_SUBJECTS]

print(f"Train: {len(train_segments)} segments (subjects {TRAIN_SUBJECTS})")
print(f"Test: {len(test_segments)} segments (subjects {TEST_SUBJECTS})")

# =====================
# データセット
# =====================
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================
# ウィンドウ作成
# =====================
def create_windows(segments, window_size, desc="Creating windows"):
    X_list = []
    y_list = []
    for seg in tqdm(segments, desc=desc, ncols=100):
        emg = seg['emg']
        glove = seg['glove']
        for i in range(window_size, len(emg)):
            X_list.append(emg[i-window_size:i])
            y_list.append(glove[i])
    print("Converting to numpy arrays...")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

print(f"\nCreating windows (window_size={WINDOW_SIZE})...")
X_train, y_train = create_windows(train_segments, WINDOW_SIZE, "Train")
X_test, y_test = create_windows(test_segments, WINDOW_SIZE, "Test")

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# =====================
# 正規化（生EMGを正規化）
# =====================
print("\nNormalizing EMG data...")
emg_scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_reshaped = emg_scaler.fit_transform(X_train_reshaped)
X_train = X_train_reshaped.reshape(X_train.shape)

X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_reshaped = emg_scaler.transform(X_test_reshaped)
X_test = X_test_reshaped.reshape(X_test.shape)

joint_scaler = StandardScaler()
y_train = joint_scaler.fit_transform(y_train)
y_test = joint_scaler.transform(y_test)

# DataLoader
train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(EMGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# =====================
# Positional Encoding
# =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# =====================
# Log-Freq EMG Attention-LSTMモデル定義
# =====================
class LogFreqEMGAttentionLSTM(nn.Module):
    def __init__(self, n_channels=16, hidden_size=256, num_layers=2, fc_size=64,
                 dropout=0.2, output_size=22, num_heads=4, attention_dim=64,
                 sample_rate=200, n_fft=64, hop_length=4, n_freq_bins=32,
                 freq_min=5, freq_max=100):
        super(LogFreqEMGAttentionLSTM, self).__init__()

        # EMG対数周波数スペクトログラム変換
        self.spectrogram = EMGLogFreqSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_freq_bins=n_freq_bins,
            freq_min=freq_min,
            freq_max=freq_max
        )

        # スペクトログラムの特徴次元
        spec_feature_dim = n_channels * n_freq_bins  # 16 * 32 = 512

        # 入力を attention_dim に射影
        self.input_projection = nn.Linear(spec_feature_dim, attention_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(attention_dim)

        # Multi-Head Self-Attention層
        self.self_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.attention_dropout = nn.Dropout(dropout)

        # Feed-forward層
        self.ff = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim * 2, attention_dim)
        )
        self.ff_norm = nn.LayerNorm(attention_dim)

        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=attention_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全結合層
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
        # x: (batch, seq_len, n_channels) - 生EMG

        # 対数周波数スペクトログラムに変換
        x = self.spectrogram(x)  # -> (batch, n_frames, n_channels * n_freq_bins)

        # 入力射影
        x = self.input_projection(x)  # -> (batch, n_frames, attention_dim)

        # Positional Encoding
        x = self.pos_encoder(x)

        # Self-Attention (residual connection)
        attn_out, _ = self.self_attention(x, x, x)
        x = self.attention_norm(x + self.attention_dropout(attn_out))

        # Feed-forward (residual connection)
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 最後のタイムステップを使用
        return self.fc(lstm_out[:, -1, :])


model = LogFreqEMGAttentionLSTM(
    n_channels=16, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
    fc_size=FC_SIZE, dropout=DROPOUT, output_size=22,
    num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_freq_bins=N_FREQ_BINS, freq_min=EMG_FREQ_MIN, freq_max=EMG_FREQ_MAX
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nLog-Freq EMG Attention-LSTM Model parameters: {total_params:,}")

# テスト: スペクトログラム出力形状確認
with torch.no_grad():
    test_input = torch.randn(2, WINDOW_SIZE, 16).to(DEVICE)
    test_output = model.spectrogram(test_input)
    print(f"Spectrogram output shape: {test_output.shape}")
    print(f"  (batch=2, time_frames={test_output.shape[1]}, features={test_output.shape[2]})")

# =====================
# 学習設定
# =====================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# =====================
# 学習ループ
# =====================
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False, ncols=100)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc="Evaluating", leave=False, ncols=100)
    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            loss = criterion(model(X_batch), y_batch)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

# 学習実行
print("\n" + "="*60)
print("Training Log-Freq EMG Attention-LSTM started...")
print("="*60)

train_losses = []
test_losses = []
best_test_loss = float('inf')

epoch_pbar = tqdm(range(EPOCHS), desc="Overall Progress", ncols=100)
for epoch in epoch_pbar:
    train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
    test_loss = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'best_model_logfreq_emg_attention_lstm.pth')
        marker = " *"
    else:
        marker = ""

    epoch_pbar.set_postfix({'train': f'{train_loss:.4f}', 'test': f'{test_loss:.4f}', 'best': f'{best_test_loss:.4f}'})
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] Train: {train_loss:.6f} | Test: {test_loss:.6f}{marker}")

print(f"\nBest Test Loss: {best_test_loss:.6f}")

# =====================
# 結果の可視化
# =====================
# 学習曲線
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Log-Freq EMG Attention-LSTM Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('logfreq_emg_attention_lstm_training_loss.png', dpi=150)
plt.close()
print("Saved: logfreq_emg_attention_lstm_training_loss.png")

# 予測評価
model.load_state_dict(torch.load('best_model_logfreq_emg_attention_lstm.pth'))
model.eval()

# 全テストデータで予測
all_preds = []
all_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y_batch.numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_true)

# 逆正規化
y_pred_original = joint_scaler.inverse_transform(y_pred)
y_true_original = joint_scaler.inverse_transform(y_true)

# 相関係数計算
correlations = []
for i in range(22):
    corr = np.corrcoef(y_true_original[:, i], y_pred_original[:, i])[0, 1]
    correlations.append(corr)

print("\n" + "="*60)
print("Log-Freq EMG Attention-LSTM Correlation Coefficients per Joint:")
print("="*60)
for i, corr in enumerate(correlations):
    print(f"Joint {i+1:2d}: {corr:.4f}")
print(f"\nMean Correlation: {np.mean(correlations):.4f}")
print(f"Max Correlation: {np.max(correlations):.4f} (Joint {np.argmax(correlations)+1})")
print(f"Min Correlation: {np.min(correlations):.4f} (Joint {np.argmin(correlations)+1})")

# 予測比較プロット
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sample_range = range(500)
for i, ax in enumerate(axes.flat):
    ax.plot([y_true_original[j, i] for j in sample_range], label='True', alpha=0.7)
    ax.plot([y_pred_original[j, i] for j in sample_range], label='Predicted', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Joint Angle')
    ax.set_title(f'Joint {i+1} (r={correlations[i]:.3f})')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig('logfreq_emg_attention_lstm_prediction_comparison.png', dpi=150)
plt.close()
print("Saved: logfreq_emg_attention_lstm_prediction_comparison.png")

# 相関係数バープロット
plt.figure(figsize=(12, 4))
colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' for c in correlations]
plt.bar(range(1, 23), correlations, color=colors)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(y=np.mean(correlations), color='blue', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
plt.xlabel('Joint')
plt.ylabel('Correlation')
plt.title('Log-Freq EMG Attention-LSTM Prediction Correlation per Joint')
plt.xticks(range(1, 23))
plt.legend()
plt.grid(True, axis='y')
plt.savefig('logfreq_emg_attention_lstm_correlation_per_joint.png', dpi=150)
plt.close()
print("Saved: logfreq_emg_attention_lstm_correlation_per_joint.png")

# 結果をJSONに保存
results = {
    'model': 'Log-Freq-EMG-Attention-LSTM',
    'best_test_loss': best_test_loss,
    'mean_correlation': float(np.mean(correlations)),
    'max_correlation': float(np.max(correlations)),
    'min_correlation': float(np.min(correlations)),
    'correlations_per_joint': [float(c) for c in correlations],
    'total_parameters': total_params,
    'hyperparameters': {
        'window_size': WINDOW_SIZE,
        'batch_size': BATCH_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'fc_size': FC_SIZE,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'num_heads': NUM_HEADS,
        'attention_dim': ATTENTION_DIM
    },
    'spectrogram_params': {
        'sample_rate': SAMPLE_RATE,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_freq_bins': N_FREQ_BINS,
        'freq_min': EMG_FREQ_MIN,
        'freq_max': EMG_FREQ_MAX,
        'scale': 'logarithmic'
    }
}
with open('logfreq_emg_attention_lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: logfreq_emg_attention_lstm_results.json")

# =====================
# 全モデル比較
# =====================
print("\n" + "="*60)
print("COMPARISON: All Attention-LSTM Models (window=64)")
print("="*60)

comparison_data = []
comparison_data.append(('Log-Freq EMG', best_test_loss, np.mean(correlations), np.max(correlations), np.min(correlations), total_params))

try:
    with open('raw_emg_attention_lstm_results.json', 'r') as f:
        raw_results = json.load(f)
    comparison_data.append(('Raw EMG', raw_results['best_test_loss'], raw_results['mean_correlation'],
                           raw_results['max_correlation'], raw_results['min_correlation'], raw_results['total_parameters']))
except:
    pass

try:
    with open('melspec_attention_lstm_results.json', 'r') as f:
        mel_results = json.load(f)
    comparison_data.append(('MelSpectrogram', mel_results['best_test_loss'], mel_results['mean_correlation'],
                           mel_results['max_correlation'], mel_results['min_correlation'], mel_results['total_parameters']))
except:
    pass

print(f"{'Model':<20} {'Test Loss':>12} {'Mean Corr':>12} {'Max Corr':>12} {'Min Corr':>12} {'Params':>12}")
print("-"*80)
for name, loss, mean_c, max_c, min_c, params in comparison_data:
    print(f"{name:<20} {loss:>12.4f} {mean_c:>12.4f} {max_c:>12.4f} {min_c:>12.4f} {params:>12,}")

print("\n" + "="*60)
print("Log-Freq EMG Attention-LSTM RESULTS SUMMARY")
print("="*60)
print(f"Best Test Loss: {best_test_loss:.6f}")
print(f"Mean Correlation: {np.mean(correlations):.4f}")
print(f"Model Parameters: {total_params:,}")
print("="*60)
print("Done!")
