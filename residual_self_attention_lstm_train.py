"""
Residual Self-Attention LSTM モデルでの学習・評価
Self-Attention + LSTM に包括的な残差接続を追加
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import math

# =====================
# 同じハイパーパラメータ（他モデルと同じ）
# =====================
WINDOW_SIZE = 20
BATCH_SIZE = 2048
HIDDEN_SIZE = 256
NUM_LAYERS = 2
FC_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE = 0.00021
EPOCHS = 50

# Attention設定
ATTENTION_DIM = 64

# 固定設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

print("="*60)
print("Residual Self-Attention LSTM Model Training")
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
print(f"  architecture: Self-Attention + LSTM with Comprehensive Residual Connections")
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
# Positional Encoding
# =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
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
# Single-Head Self-Attention
# =====================
class SelfAttention(nn.Module):
    """シンプルなSingle-Head Self-Attention"""
    def __init__(self, embed_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)

        # Q, K, V の線形変換
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # 出力の線形変換
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()

        # Q, K, V を計算
        Q = self.W_q(x)  # (batch, seq_len, embed_dim)
        K = self.W_k(x)  # (batch, seq_len, embed_dim)
        V = self.W_v(x)  # (batch, seq_len, embed_dim)

        # Scaled Dot-Product Attention
        # scores: (batch, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Attention出力
        # (batch, seq_len, seq_len) x (batch, seq_len, embed_dim) -> (batch, seq_len, embed_dim)
        attention_output = torch.bmm(attention_weights, V)

        # 出力変換
        output = self.W_o(attention_output)

        return output, attention_weights

# =====================
# Residual Self-Attention LSTMモデル定義
# 包括的な残差接続を追加
# =====================
class EMGtoJointResidualSelfAttentionLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=256, num_layers=2, fc_size=64,
                 dropout=0.2, output_size=22, attention_dim=64):
        super(EMGtoJointResidualSelfAttentionLSTM, self).__init__()

        self.attention_dim = attention_dim
        self.hidden_size = hidden_size

        # 入力を attention_dim に射影
        self.input_projection = nn.Linear(input_size, attention_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(attention_dim)

        # Single-Head Self-Attention層
        self.self_attention = SelfAttention(
            embed_dim=attention_dim,
            dropout=dropout
        )
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.attention_dropout = nn.Dropout(dropout)

        # Feed-forward層（Transformer風）
        self.ff = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim * 2, attention_dim)
        )
        self.ff_norm = nn.LayerNorm(attention_dim)
        self.ff_dropout = nn.Dropout(dropout)

        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=attention_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # LSTM周りの残差接続用
        # attention_dim -> hidden_size への射影（残差接続用）
        self.residual_projection = nn.Linear(attention_dim, hidden_size)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # 全結合層（残差接続付き）
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc1_norm = nn.LayerNorm(fc_size)
        self.fc1_dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(fc_size, fc_size // 2)
        self.fc2_norm = nn.LayerNorm(fc_size // 2)
        self.fc2_dropout = nn.Dropout(dropout)

        # fc_size -> fc_size // 2 の残差接続用射影
        self.fc_residual = nn.Linear(fc_size, fc_size // 2)

        self.fc_out = nn.Linear(fc_size // 2, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # 入力射影
        x = self.input_projection(x)  # -> (batch, seq_len, attention_dim)

        # Positional Encoding
        x = self.pos_encoder(x)

        # 残差接続用に保存
        identity_for_lstm = x

        # Self-Attention (Pre-LN Residual connection)
        attn_out, _ = self.self_attention(x)
        x = self.attention_norm(x + self.attention_dropout(attn_out))

        # Feed-forward (Pre-LN Residual connection)
        ff_out = self.ff(x)
        x = self.ff_norm(x + self.ff_dropout(ff_out))

        # LSTM
        lstm_out, _ = self.lstm(x)

        # LSTM周りの残差接続
        # 入力を hidden_size に射影して加算
        residual = self.residual_projection(identity_for_lstm[:, -1, :])
        lstm_final = self.lstm_norm(lstm_out[:, -1, :] + residual)

        # FC層 with 残差接続
        # FC1
        fc1_out = self.fc1_dropout(torch.relu(self.fc1_norm(self.fc1(lstm_final))))

        # FC2 with residual
        fc2_identity = self.fc_residual(fc1_out)  # 次元合わせの射影
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.fc2_dropout(torch.relu(self.fc2_norm(fc2_out + fc2_identity)))

        # Output
        return self.fc_out(fc2_out)

model = EMGtoJointResidualSelfAttentionLSTM(
    input_size=16, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
    fc_size=FC_SIZE, dropout=DROPOUT, output_size=22,
    attention_dim=ATTENTION_DIM
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nResidual Self-Attention LSTM Model parameters: {total_params:,}")

print("\nModel Architecture:")
print("  Input: (batch, 20, 16)")
print("  -> Input Projection (16 -> 64)")
print("  -> Positional Encoding")
print("  -> Self-Attention + Residual + LayerNorm")
print("  -> Feed-Forward + Residual + LayerNorm")
print("  -> LSTM + Residual (from attention input) + LayerNorm")
print("  -> FC1 + LayerNorm")
print("  -> FC2 + Residual + LayerNorm")
print("  -> Output: 22 joints")

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
print("Training Residual Self-Attention LSTM started...")
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
        torch.save(model.state_dict(), 'best_model_residual_self_attention_lstm.pth')
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
plt.title('Residual Self-Attention LSTM Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('residual_self_attention_lstm_training_loss.png', dpi=150)
plt.close()
print("Saved: residual_self_attention_lstm_training_loss.png")

# 予測評価
model.load_state_dict(torch.load('best_model_residual_self_attention_lstm.pth'))
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
print("Residual Self-Attention LSTM Correlation Coefficients per Joint:")
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
plt.savefig('residual_self_attention_lstm_prediction_comparison.png', dpi=150)
plt.close()
print("Saved: residual_self_attention_lstm_prediction_comparison.png")

# 相関係数バープロット
plt.figure(figsize=(12, 4))
colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' for c in correlations]
plt.bar(range(1, 23), correlations, color=colors)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(y=np.mean(correlations), color='blue', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
plt.xlabel('Joint')
plt.ylabel('Correlation')
plt.title('Residual Self-Attention LSTM Prediction Correlation per Joint')
plt.xticks(range(1, 23))
plt.legend()
plt.grid(True, axis='y')
plt.savefig('residual_self_attention_lstm_correlation_per_joint.png', dpi=150)
plt.close()
print("Saved: residual_self_attention_lstm_correlation_per_joint.png")

# 結果をJSONに保存
results = {
    'model': 'Residual-Self-Attention-LSTM',
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
        'architecture': 'Self-Attention + LSTM with Comprehensive Residual Connections'
    }
}
with open('residual_self_attention_lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: residual_self_attention_lstm_results.json")

# 全モデル比較
print("\n" + "="*60)
print("ALL MODELS COMPARISON")
print("="*60)

models_data = []
try:
    with open('cnn_lstm_results.json', 'r') as f:
        cnn_lstm_results = json.load(f)
    models_data.append(('CNN-LSTM', cnn_lstm_results['best_test_loss'], cnn_lstm_results['mean_correlation'], cnn_lstm_results['total_parameters']))
except:
    pass

try:
    with open('attention_lstm_results.json', 'r') as f:
        mh_attn_results = json.load(f)
    models_data.append(('MultiHead-Attn-LSTM', mh_attn_results['best_test_loss'], mh_attn_results['mean_correlation'], mh_attn_results['total_parameters']))
except:
    pass

try:
    with open('self_attention_lstm_results.json', 'r') as f:
        sa_results = json.load(f)
    models_data.append(('Self-Attn-LSTM', sa_results['best_test_loss'], sa_results['mean_correlation'], sa_results['total_parameters']))
except:
    pass

try:
    with open('causal_self_attention_lstm_results.json', 'r') as f:
        causal_results = json.load(f)
    models_data.append(('Causal-Self-Attn-LSTM', causal_results['best_test_loss'], causal_results['mean_correlation'], causal_results['total_parameters']))
except:
    pass

try:
    with open('stacked_self_attention_lstm_results.json', 'r') as f:
        stacked_results = json.load(f)
    models_data.append(('6-Layer-Stacked-Self-Attn', stacked_results['best_test_loss'], stacked_results['mean_correlation'], stacked_results['total_parameters']))
except:
    pass

models_data.append(('Residual-Self-Attn-LSTM', best_test_loss, float(np.mean(correlations)), total_params))

print(f"{'Model':<28} {'Test Loss':>12} {'Mean Corr':>12} {'Parameters':>12}")
print("-"*70)
for name, loss, corr, params in sorted(models_data, key=lambda x: x[1]):
    print(f"{name:<28} {loss:>12.4f} {corr:>12.4f} {params:>12,}")

print("\n" + "="*60)
print("Residual Self-Attention LSTM RESULTS SUMMARY")
print("="*60)
print(f"Best Test Loss: {best_test_loss:.6f}")
print(f"Mean Correlation: {np.mean(correlations):.4f}")
print(f"Model Parameters: {total_params:,}")
print("="*60)
print("Done!")
