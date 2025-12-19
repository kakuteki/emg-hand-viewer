"""
OptunaによるEMG→関節角度予測モデルのハイパーパラメータ最適化
被験者単位分割（データリーク防止）を維持
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.trial import TrialState
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================
# 固定設定
# =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
N_TRIALS = 50  # 最適化試行回数
EPOCHS_PER_TRIAL = 20  # 各試行のエポック数（高速化のため少なめ）

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

train_segments = [seg for seg in segments if seg['subject_id'] in TRAIN_SUBJECTS]
test_segments = [seg for seg in segments if seg['subject_id'] in TEST_SUBJECTS]

print(f"Train subjects: {TRAIN_SUBJECTS} ({len(train_segments)} segments)")
print(f"Test subjects: {TEST_SUBJECTS} ({len(test_segments)} segments)")

# =====================
# データセットクラス
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
# ウィンドウ作成関数
# =====================
def create_windows(segments, window_size):
    X_list = []
    y_list = []
    for seg in segments:
        emg = seg['emg']
        glove = seg['glove']
        for i in range(window_size, len(emg)):
            X_list.append(emg[i-window_size:i])
            y_list.append(glove[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# =====================
# モデル定義
# =====================
class EMGtoJointLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2,
                 fc_size=128, dropout=0.2, output_size=22):
        super(EMGtoJointLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
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
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)

# =====================
# 目的関数（Optunaが最小化する）
# =====================
def objective(trial):
    # ハイパーパラメータのサンプリング
    window_size = trial.suggest_int('window_size', 20, 100, step=10)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    fc_size = trial.suggest_categorical('fc_size', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    try:
        # データ準備
        X_train, y_train = create_windows(train_segments, window_size)
        X_test, y_test = create_windows(test_segments, window_size)

        # 正規化
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
        train_loader = DataLoader(
            EMGDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        test_loader = DataLoader(
            EMGDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        # モデル作成
        model = EMGtoJointLSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            fc_size=fc_size,
            dropout=dropout,
            output_size=22
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 学習
        best_test_loss = float('inf')

        for epoch in range(EPOCHS_PER_TRIAL):
            # Train
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item()

            test_loss = total_loss / len(test_loader)

            if test_loss < best_test_loss:
                best_test_loss = test_loss

            # Pruning（早期停止）
            trial.report(test_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_test_loss

    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

# =====================
# 最適化実行
# =====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*60)
    print(f"Number of trials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS_PER_TRIAL}")
    print()

    # Studyの作成
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    # 最適化実行
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True
    )

    # 結果表示
    print("\n" + "="*60)
    print("Optimization Results")
    print("="*60)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Test Loss): {trial.value:.6f}")
    print("\n  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 結果を保存
    import json
    results = {
        'best_value': trial.value,
        'best_params': trial.params,
        'n_trials': len(study.trials),
        'n_pruned': len(pruned_trials),
        'n_complete': len(complete_trials)
    }

    with open('optuna_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: optuna_results.json")

    # 最適化履歴のプロット
    try:
        import matplotlib.pyplot as plt

        # 最適化履歴
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 試行ごとの値
        trial_values = [t.value for t in study.trials if t.value is not None and t.value != float('inf')]
        axes[0].plot(trial_values, 'b.-', alpha=0.7)
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Test Loss')
        axes[0].set_title('Optimization History')
        axes[0].grid(True)

        # パラメータ重要度
        try:
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())
            values = list(importances.values())
            axes[1].barh(params, values)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Hyperparameter Importance')
            axes[1].grid(True, axis='x')
        except:
            axes[1].text(0.5, 0.5, 'Importance calculation failed', ha='center', va='center')

        plt.tight_layout()
        plt.savefig('optuna_history.png', dpi=150)
        plt.close()
        print("Saved: optuna_history.png")
    except Exception as e:
        print(f"Plot failed: {e}")

    print("\nDone!")
