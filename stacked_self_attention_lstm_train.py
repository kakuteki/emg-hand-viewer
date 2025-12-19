"""
6-Layer Stacked Self-Attention + LSTM Model
Self-Attentionを6段スタックしてからLSTMに入力
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# =====================
# 同じハイパーパラメータ
# =====================
WINDOW_SIZE = 20
BATCH_SIZE = 2048
HIDDEN_SIZE = 256
NUM_LAYERS = 2
FC_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE = 0.00021
EPOCHS = 50
ATTENTION_DIM = 64
NUM_ATTENTION_LAYERS = 6  # Self-Attentionのスタック数

# 固定設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

print("="*60)
print("6-Layer Stacked Self-Attention LSTM Model Training")
print("="*60)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()
print("Hyperparameters (same as other models):")
print(f"  window_size: {WINDOW_SIZE}")
print(f"  batch_size: {BATCH_SIZE}")
print(f"  hidden_size: {HIDDEN_SIZE}")
print(f"  num_layers: {NUM_LAYERS}")
print(f"  fc_size: {FC_SIZE}")
print(f"  dropout: {DROPOUT}")
print(f"  learning_rate: {LEARNING_RATE}")
print(f"  epochs: {EPOCHS}")
print(f"  attention_dim: {ATTENTION_DIM}")
print(f"  num_attention_layers: {NUM_ATTENTION_LAYERS}")
print()

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
# 正規化
# =====================
print("\nNormalizing data...")
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
# Self-Attention Layer (単一層)
# =====================
class SelfAttentionLayer(nn.Module):
    """Single Self-Attention Layer with residual connection and layer norm"""
    def __init__(self, input_dim, attention_dim, dropout=0.2):
        super(SelfAttentionLayer, self).__init__()

        self.attention_dim = attention_dim

        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

        # Output projection back to input dimension
        self.output_proj = nn.Linear(attention_dim, input_dim)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = attention_dim ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        residual = x

        # Self-attention
        Q = self.query(x)  # (batch, seq_len, attention_dim)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch, seq_len, attention_dim)

        # Project back to input dimension
        output = self.output_proj(attended)  # (batch, seq_len, input_dim)
        output = self.dropout(output)

        # Residual connection and layer normalization
        output = self.layer_norm(residual + output)

        return output

# =====================
# Stacked Self-Attention + LSTM モデル
# =====================
class StackedSelfAttentionLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=256, num_layers=2, fc_size=64,
                 dropout=0.2, output_size=22, attention_dim=64, num_attention_layers=6):
        super(StackedSelfAttentionLSTM, self).__init__()

        self.num_attention_layers = num_attention_layers

        # 6層のSelf-Attention
        self.attention_layers = nn.ModuleList([
            SelfAttentionLayer(input_size, attention_dim, dropout)
            for _ in range(num_attention_layers)
        ])

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,  # Self-Attentionは入力次元を保持
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
        # x: (batch, seq_len, input_size)

        # 6層のSelf-Attentionを順番に適用
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 最後のタイムステップを使用
        return self.fc(lstm_out[:, -1, :])

model = StackedSelfAttentionLSTM(
    input_size=16, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
    fc_size=FC_SIZE, dropout=DROPOUT, output_size=22,
    attention_dim=ATTENTION_DIM, num_attention_layers=NUM_ATTENTION_LAYERS
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n6-Layer Stacked Self-Attention LSTM Model parameters: {total_params:,}")

# モデル構造の確認
print("\nModel Architecture:")
print(f"  Input: (batch, {WINDOW_SIZE}, 16)")
print(f"  -> 6x Self-Attention Layers (dim={ATTENTION_DIM})")
print(f"  -> LSTM (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
print(f"  -> FC layers -> Output: 22 joints")

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
print("Training 6-Layer Stacked Self-Attention LSTM started...")
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
        torch.save(model.state_dict(), 'best_model_stacked_self_attention_lstm.pth')
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
plt.title('6-Layer Stacked Self-Attention LSTM Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('stacked_self_attention_lstm_training_loss.png', dpi=150)
plt.close()
print("Saved: stacked_self_attention_lstm_training_loss.png")

# 予測評価
model.load_state_dict(torch.load('best_model_stacked_self_attention_lstm.pth'))
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
print("6-Layer Stacked Self-Attention LSTM Correlation Coefficients per Joint:")
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
plt.savefig('stacked_self_attention_lstm_prediction_comparison.png', dpi=150)
plt.close()
print("Saved: stacked_self_attention_lstm_prediction_comparison.png")

# 相関係数バープロット
plt.figure(figsize=(12, 4))
colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' for c in correlations]
plt.bar(range(1, 23), correlations, color=colors)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(y=np.mean(correlations), color='blue', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
plt.xlabel('Joint')
plt.ylabel('Correlation')
plt.title('6-Layer Stacked Self-Attention LSTM Prediction Correlation per Joint')
plt.xticks(range(1, 23))
plt.legend()
plt.grid(True, axis='y')
plt.savefig('stacked_self_attention_lstm_correlation_per_joint.png', dpi=150)
plt.close()
print("Saved: stacked_self_attention_lstm_correlation_per_joint.png")

# 結果をJSONに保存
results = {
    'model': '6-Layer-Stacked-Self-Attention-LSTM',
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
        'attention_dim': ATTENTION_DIM,
        'num_attention_layers': NUM_ATTENTION_LAYERS,
        'architecture': '6x Self-Attention -> LSTM -> FC'
    }
}
with open('stacked_self_attention_lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: stacked_self_attention_lstm_results.json")

# 全モデル比較
print("\n" + "="*60)
print("ALL MODELS COMPARISON")
print("="*60)
print(f"{'Model':<35} {'Test Loss':>12} {'Mean Corr':>12} {'Parameters':>12}")
print("-"*72)

# 各モデルの結果を読み込み
model_files = [
    ('final_results.json', 'LSTM'),
    ('cnn_lstm_results.json', 'CNN-LSTM'),
    ('attention_lstm_results.json', 'MultiHead-Attn-LSTM'),
    ('self_attention_lstm_results.json', 'Self-Attn-LSTM'),
    ('causal_self_attention_lstm_results.json', 'Causal-Self-Attn-LSTM'),
]

all_results = []
for filename, name in model_files:
    try:
        with open(filename, 'r') as f:
            r = json.load(f)
            all_results.append((name, r['best_test_loss'], r['mean_correlation'], r['total_parameters']))
    except:
        pass

# 現在のモデルを追加
all_results.append(('6-Layer-Stacked-Self-Attn-LSTM', best_test_loss, float(np.mean(correlations)), total_params))

# 表示
for name, loss, corr, params in all_results:
    print(f"{name:<35} {loss:>12.4f} {corr:>12.4f} {params:>12,}")

print("\n" + "="*60)
print("6-Layer Stacked Self-Attention LSTM RESULTS SUMMARY")
print("="*60)
print(f"Best Test Loss: {best_test_loss:.6f}")
print(f"Mean Correlation: {np.mean(correlations):.4f}")
print(f"Model Parameters: {total_params:,}")
print("="*60)
print("Done!")
