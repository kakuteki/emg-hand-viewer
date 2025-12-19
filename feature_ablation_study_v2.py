"""
Feature Ablation Study v2
- Proper normalization
- Test each novel feature's contribution to model performance
- Compare: Baseline (raw EMG) vs Baseline + Feature
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =====================
# Feature Extraction Functions (applied to normalized EMG)
# =====================
def compute_tkeo(windows):
    """Teager-Kaiser Energy Operator - batch processing"""
    # windows: (N, window_size, channels)
    if windows.shape[1] < 3:
        return np.zeros((windows.shape[0], windows.shape[2]))
    tkeo = windows[:, 1:-1, :]**2 - windows[:, :-2, :] * windows[:, 2:, :]
    return np.mean(np.abs(tkeo), axis=1)

def compute_tkeo_std(windows):
    """Standard deviation of TKEO"""
    if windows.shape[1] < 3:
        return np.zeros((windows.shape[0], windows.shape[2]))
    tkeo = windows[:, 1:-1, :]**2 - windows[:, :-2, :] * windows[:, 2:, :]
    return np.std(tkeo, axis=1)

def compute_mad(windows):
    """Mean Absolute Deviation"""
    median = np.median(windows, axis=1, keepdims=True)
    return np.mean(np.abs(windows - median), axis=1)

def compute_nle(windows):
    """Normalized Logarithmic Energy"""
    energy = np.sum(windows**2, axis=1)
    return np.log(energy / (windows.shape[1] + 1e-10) + 1e-10)

def compute_aac(windows):
    """Average Amplitude Change"""
    return np.mean(np.abs(np.diff(windows, axis=1)), axis=1)

def compute_var(windows):
    """Variance"""
    return np.var(windows, axis=1)

def compute_mav(windows):
    """Mean Absolute Value"""
    return np.mean(np.abs(windows), axis=1)

def compute_rms(windows):
    """Root Mean Square"""
    return np.sqrt(np.mean(windows**2, axis=1))

# =====================
# Self-Attention-LSTM Model with optional feature input
# =====================
class SelfAttentionLSTMWithFeatures(nn.Module):
    def __init__(self, input_dim=16, hidden_size=256, num_layers=2,
                 output_dim=22, dropout=0.2, extra_feature_dim=0):
        super().__init__()

        self.extra_feature_dim = extra_feature_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, 64)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 20, 64) * 0.1)

        # Self-Attention
        self.attention = nn.MultiheadAttention(64, num_heads=1, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(64)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        self.norm2 = nn.LayerNorm(64)

        # LSTM
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Output layers - include extra features
        fc_input_dim = hidden_size + extra_feature_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, extra_features=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_final = lstm_out[:, -1, :]

        # Concatenate extra features if provided
        if extra_features is not None and self.extra_feature_dim > 0:
            lstm_final = torch.cat([lstm_final, extra_features], dim=1)

        return self.fc(lstm_final)

# =====================
# Training Function
# =====================
def train_and_evaluate(X_train, y_train, X_test, y_test,
                       train_features=None, test_features=None,
                       epochs=50, batch_size=2048, lr=0.00021):

    extra_dim = train_features.shape[1] if train_features is not None else 0

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(train_features, dtype=torch.float32) if train_features is not None else torch.zeros(len(X_train), 0),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(test_features, dtype=torch.float32) if test_features is not None else torch.zeros(len(X_test), 0),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SelfAttentionLSTMWithFeatures(
        input_dim=16, hidden_size=256, num_layers=2,
        output_dim=22, dropout=0.2, extra_feature_dim=extra_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_test_loss = float('inf')
    best_corr = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for windows, features, targets in train_loader:
            windows, features, targets = windows.to(device), features.to(device), targets.to(device)
            if extra_dim > 0:
                outputs = model(windows, features)
            else:
                outputs = model(windows)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Testing
        model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for windows, features, targets in test_loader:
                windows, features, targets = windows.to(device), features.to(device), targets.to(device)
                if extra_dim > 0:
                    outputs = model(windows, features)
                else:
                    outputs = model(windows)

                test_loss += criterion(outputs, targets).item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_loss /= len(test_loader)
        scheduler.step(test_loss)

        # Calculate correlation
        preds = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        correlations = []
        for j in range(22):
            corr = np.corrcoef(preds[:, j], targets_np[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        mean_corr = np.mean(correlations)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_corr = mean_corr

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}, Corr={mean_corr:.4f}")

    return best_test_loss, best_corr

# =====================
# Main Experiment
# =====================
print("="*70)
print("Feature Ablation Study v2 (with proper normalization)")
print("="*70)

# Load data
print("\nLoading data...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']

TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

train_segments = [seg for seg in segments if seg['subject_id'] in TRAIN_SUBJECTS]
test_segments = [seg for seg in segments if seg['subject_id'] in TEST_SUBJECTS]

print(f"Train segments: {len(train_segments)}")
print(f"Test segments: {len(test_segments)}")

# Create windows
WINDOW_SIZE = 20

print("\nCreating windows...")
X_train_list = []
y_train_list = []
for seg in tqdm(train_segments, desc="Train"):
    emg = seg['emg']
    glove = seg['glove']
    for i in range(WINDOW_SIZE, len(emg)):
        X_train_list.append(emg[i-WINDOW_SIZE:i])
        y_train_list.append(glove[i])

X_test_list = []
y_test_list = []
for seg in tqdm(test_segments, desc="Test"):
    emg = seg['emg']
    glove = seg['glove']
    for i in range(WINDOW_SIZE, len(emg)):
        X_test_list.append(emg[i-WINDOW_SIZE:i])
        y_test_list.append(glove[i])

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_test = np.array(X_test_list, dtype=np.float32)
y_test = np.array(y_test_list, dtype=np.float32)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# =====================
# Normalize EMG and Joint Angles
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

# =====================
# Extract features from normalized windows
# =====================
print("\nExtracting features from normalized windows...")

feature_funcs = {
    'TKEO': compute_tkeo,
    'MAD': compute_mad,
    'TKEO_STD': compute_tkeo_std,
    'NLE': compute_nle,
    'AAC': compute_aac,
    'VAR': compute_var,
    'MAV': compute_mav,
    'RMS': compute_rms,
}

train_features = {}
test_features = {}

for name, func in feature_funcs.items():
    print(f"  Computing {name}...")
    train_features[name] = func(X_train)
    test_features[name] = func(X_test)

# Normalize features
print("\nNormalizing features...")
feature_scalers = {}
for name in feature_funcs.keys():
    scaler = StandardScaler()
    train_features[name] = scaler.fit_transform(train_features[name])
    test_features[name] = scaler.transform(test_features[name])
    feature_scalers[name] = scaler

# =====================
# Define experiments
# =====================
experiments = {
    'Baseline (Raw EMG)': [],
    'Baseline + TKEO': ['TKEO'],
    'Baseline + MAD': ['MAD'],
    'Baseline + TKEO_STD': ['TKEO_STD'],
    'Baseline + NLE': ['NLE'],
    'Baseline + AAC': ['AAC'],
    'Baseline + VAR': ['VAR'],
    'Baseline + MAV': ['MAV'],
    'Baseline + RMS': ['RMS'],
    'Baseline + TKEO + MAD': ['TKEO', 'MAD'],
    'Baseline + TKEO + MAD + NLE': ['TKEO', 'MAD', 'NLE'],
    'Baseline + All Novel (TKEO, MAD, NLE, AAC, TKEO_STD)': ['TKEO', 'MAD', 'NLE', 'AAC', 'TKEO_STD'],
    'Baseline + Traditional (VAR, MAV, RMS)': ['VAR', 'MAV', 'RMS'],
    'Baseline + Best Combo (TKEO, MAD, VAR, MAV)': ['TKEO', 'MAD', 'VAR', 'MAV'],
}

results = {}

for exp_name, feature_names in experiments.items():
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")

    if feature_names:
        train_feat = np.concatenate([train_features[n] for n in feature_names], axis=1)
        test_feat = np.concatenate([test_features[n] for n in feature_names], axis=1)
        extra_dim = train_feat.shape[1]
    else:
        train_feat = None
        test_feat = None
        extra_dim = 0

    print(f"Extra feature dimension: {extra_dim}")

    # Train and evaluate
    print("Training...")
    best_loss, best_corr = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        train_features=train_feat, test_features=test_feat,
        epochs=50
    )

    results[exp_name] = {
        'test_loss': float(best_loss),
        'mean_correlation': float(best_corr),
        'extra_feature_dim': extra_dim,
        'features': feature_names
    }

    print(f"\nResult: Test Loss = {best_loss:.4f}, Mean Corr = {best_corr:.4f}")

# =====================
# Summary
# =====================
print("\n" + "="*70)
print("ABLATION STUDY RESULTS SUMMARY")
print("="*70)

baseline_loss = results['Baseline (Raw EMG)']['test_loss']
baseline_corr = results['Baseline (Raw EMG)']['mean_correlation']

print(f"\n{'Experiment':<55} {'Test Loss':>10} {'Mean Corr':>10} {'Loss Δ':>10} {'Corr Δ':>10}")
print("-"*95)

sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_correlation'], reverse=True)

for exp_name, data in sorted_results:
    loss_delta = (baseline_loss - data['test_loss']) / baseline_loss * 100
    corr_delta = (data['mean_correlation'] - baseline_corr) / baseline_corr * 100
    print(f"{exp_name:<55} {data['test_loss']:>10.4f} {data['mean_correlation']:>10.4f} {loss_delta:>+9.2f}% {corr_delta:>+9.2f}%")

# Save results
with open('feature_ablation_results.json', 'w') as f:
    json.dump({
        'baseline': {'test_loss': baseline_loss, 'mean_correlation': baseline_corr},
        'experiments': results,
        'sorted_by_correlation': [
            {
                'name': name,
                'test_loss': data['test_loss'],
                'mean_correlation': data['mean_correlation'],
                'loss_improvement_pct': (baseline_loss - data['test_loss']) / baseline_loss * 100,
                'corr_improvement_pct': (data['mean_correlation'] - baseline_corr) / baseline_corr * 100,
                'features': data['features']
            }
            for name, data in sorted_results
        ]
    }, f, indent=2)

print("\nSaved: feature_ablation_results.json")
print("\n" + "="*70)
print("Ablation Study Complete!")
print("="*70)
