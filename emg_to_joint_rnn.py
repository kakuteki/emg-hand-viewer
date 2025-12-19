"""
EMGから関節角度を予測するRNN（LSTM）モデル
NinaPro DB5データセットを使用
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================
# 設定
# =====================
WINDOW_SIZE = 50  # 過去50サンプル（250ms @ 200Hz）を使用
BATCH_SIZE = 4096  # GPUフル活用
HIDDEN_SIZE = 512  # モデル容量大幅増加
NUM_LAYERS = 4  # より深いネットワーク
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_WORKERS = 0  # Windowsでは0推奨
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CUDNNの最適化を有効化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =====================
# データ読み込み
# =====================
print("Loading data...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']

# =====================
# 被験者単位でのデータ分割（データリーク防止）
# =====================
TEST_SUBJECTS = [9, 10]  # テスト用被験者（2人）
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]  # 学習用被験者（8人）

print(f"Train subjects: {TRAIN_SUBJECTS}")
print(f"Test subjects: {TEST_SUBJECTS}")

# 被験者ごとにセグメントを分割
train_segments = [seg for seg in segments if seg['subject_id'] in TRAIN_SUBJECTS]
test_segments = [seg for seg in segments if seg['subject_id'] in TEST_SUBJECTS]

print(f"Train segments: {len(train_segments)}")
print(f"Test segments: {len(test_segments)}")

# =====================
# データ前処理（ウィンドウ化）
# =====================
def create_windows(segments, window_size, desc="Creating windows"):
    """
    セグメントからウィンドウを作成
    入力: 過去window_sizeサンプルのEMG
    出力: 現在の関節角度
    """
    X_list = []
    y_list = []

    for seg in tqdm(segments, desc=desc, ncols=100):
        emg = seg['emg']  # (samples, 16)
        glove = seg['glove']  # (samples, 22)

        # ウィンドウを作成
        for i in range(window_size, len(emg)):
            X_list.append(emg[i-window_size:i])  # (window_size, 16)
            y_list.append(glove[i])  # (22,)

    print("Converting to numpy arrays...")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y

# 学習データとテストデータを別々に作成
print(f"\nCreating training windows (window_size={WINDOW_SIZE})...")
X_train, y_train = create_windows(train_segments, WINDOW_SIZE, "Train windows")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print(f"\nCreating test windows (window_size={WINDOW_SIZE})...")
X_test, y_test = create_windows(test_segments, WINDOW_SIZE, "Test windows")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# =====================
# データ正規化（学習データのみでfitする）
# =====================
print("\nNormalizing data (fit on training data only)...")

# EMGの正規化
emg_scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_reshaped = emg_scaler.fit_transform(X_train_reshaped)  # 学習データでfit
X_train = X_train_reshaped.reshape(X_train.shape)

X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_reshaped = emg_scaler.transform(X_test_reshaped)  # 学習データのスケーラーでtransform
X_test = X_test_reshaped.reshape(X_test.shape)

# 関節角度の正規化
joint_scaler = StandardScaler()
y_train = joint_scaler.fit_transform(y_train)  # 学習データでfit
y_test = joint_scaler.transform(y_test)  # 学習データのスケーラーでtransform

print(f"\nTrain: {X_train.shape[0]} samples (subjects {TRAIN_SUBJECTS})")
print(f"Test: {X_test.shape[0]} samples (subjects {TEST_SUBJECTS})")

# =====================
# Dataset & DataLoader
# =====================
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EMGDataset(X_train, y_train)
test_dataset = EMGDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# =====================
# RNNモデル定義
# =====================
class EMGtoJointLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, output_size=22):
        super(EMGtoJointLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 全結合層（より大きなネットワーク）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 最後の時刻の出力を使用
        out = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 全結合層
        out = self.fc(out)  # (batch, output_size)

        return out

model = EMGtoJointLSTM(
    input_size=16,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=22
).to(DEVICE)

print(f"\nModel architecture:")
print(model)

# パラメータ数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# =====================
# 学習設定
# =====================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# =====================
# 学習ループ
# =====================
def train_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                leave=False, ncols=100)

    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
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
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

# 学習実行
print("\n" + "="*50)
print("Training started...")
print("="*50)

train_losses = []
test_losses = []
best_test_loss = float('inf')

# エポック全体のプログレスバー
epoch_pbar = tqdm(range(EPOCHS), desc="Overall Progress", ncols=100)

for epoch in epoch_pbar:
    train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, EPOCHS)
    test_loss = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    scheduler.step(test_loss)

    # ベストモデルの保存
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'best_model.pth')
        saved_marker = " *saved*"
    else:
        saved_marker = ""

    # プログレスバーの説明を更新
    epoch_pbar.set_postfix({
        'train': f'{train_loss:.4f}',
        'test': f'{test_loss:.4f}',
        'best': f'{best_test_loss:.4f}'
    })

    # 毎エポック結果を表示
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] Train: {train_loss:.6f} | Test: {test_loss:.6f}{saved_marker}")

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
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png', dpi=150)
plt.close()
print("Saved: training_loss.png")

# 予測と実測の比較
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# テストデータの一部で予測
X_sample = torch.FloatTensor(X_test[:500]).to(DEVICE)
with torch.no_grad():
    y_pred = model(X_sample).cpu().numpy()
y_true = y_test[:500]

# 逆正規化
y_pred_original = joint_scaler.inverse_transform(y_pred)
y_true_original = joint_scaler.inverse_transform(y_true)

# 関節角度の比較プロット（最初の4関節）
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.plot(y_true_original[:200, i], label='True', alpha=0.7)
    ax.plot(y_pred_original[:200, i], label='Predicted', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Joint Angle')
    ax.set_title(f'Joint {i+1}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=150)
plt.close()
print("Saved: prediction_comparison.png")

# 相関係数の計算
correlations = []
for i in range(22):
    corr = np.corrcoef(y_true_original[:, i], y_pred_original[:, i])[0, 1]
    correlations.append(corr)

print("\n" + "="*50)
print("Correlation coefficients per joint:")
print("="*50)
for i, corr in enumerate(correlations):
    print(f"Joint {i+1:2d}: {corr:.4f}")

print(f"\nMean correlation: {np.mean(correlations):.4f}")

# 相関係数のバープロット
plt.figure(figsize=(12, 4))
plt.bar(range(1, 23), correlations)
plt.xlabel('Joint')
plt.ylabel('Correlation')
plt.title('Prediction Correlation per Joint')
plt.xticks(range(1, 23))
plt.grid(True, axis='y')
plt.savefig('correlation_per_joint.png', dpi=150)
plt.close()
print("Saved: correlation_per_joint.png")

print("\nDone!")
