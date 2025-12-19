"""
Feature Ablation Study
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
# Feature Extraction Functions
# =====================
def compute_tkeo(x):
    """Teager-Kaiser Energy Operator"""
    if x.shape[0] < 3:
        return np.zeros(x.shape[1])
    tkeo = x[1:-1]**2 - x[:-2] * x[2:]
    return np.mean(np.abs(tkeo), axis=0)

def compute_tkeo_std(x):
    """Standard deviation of TKEO"""
    if x.shape[0] < 3:
        return np.zeros(x.shape[1])
    tkeo = x[1:-1]**2 - x[:-2] * x[2:]
    return np.std(tkeo, axis=0)

def compute_mad(x):
    """Mean Absolute Deviation"""
    median = np.median(x, axis=0)
    return np.mean(np.abs(x - median), axis=0)

def compute_nle(x):
    """Normalized Logarithmic Energy"""
    energy = np.sum(x**2, axis=0)
    return np.log(energy / (len(x) + 1e-10) + 1e-10)

def compute_aac(x):
    """Average Amplitude Change"""
    return np.mean(np.abs(np.diff(x, axis=0)), axis=0)

def compute_var(x):
    """Variance"""
    return np.var(x, axis=0)

def compute_mav(x):
    """Mean Absolute Value"""
    return np.mean(np.abs(x), axis=0)

def compute_rms(x):
    """Root Mean Square"""
    return np.sqrt(np.mean(x**2, axis=0))

# =====================
# Dataset
# =====================
class EMGDataset(Dataset):
    def __init__(self, segments, window_size=20, feature_funcs=None):
        self.windows = []
        self.targets = []
        self.feature_funcs = feature_funcs or []

        for seg in segments:
            emg = seg['emg']
            glove = seg['glove']
            for i in range(window_size, len(emg)):
                window = emg[i-window_size:i]
                self.windows.append(window)
                self.targets.append(glove[i])

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        # Extract features if specified
        if self.feature_funcs:
            print(f"Extracting {len(self.feature_funcs)} feature types...")
            features_list = []
            for func in self.feature_funcs:
                feat = np.array([func(w) for w in tqdm(self.windows, desc=func.__name__)])
                features_list.append(feat)
            self.extra_features = np.concatenate(features_list, axis=1)
            print(f"Extra features shape: {self.extra_features.shape}")
        else:
            self.extra_features = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.tensor(self.windows[idx])
        target = torch.tensor(self.targets[idx])

        if self.extra_features is not None:
            extra = torch.tensor(self.extra_features[idx])
            return window, extra, target
        return window, target

# =====================
# Model with Optional Feature Input
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
def train_and_evaluate(train_dataset, test_dataset, extra_feature_dim=0,
                       epochs=30, batch_size=2048, lr=0.00021):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SelfAttentionLSTMWithFeatures(
        input_dim=16, hidden_size=256, num_layers=2,
        output_dim=22, dropout=0.2, extra_feature_dim=extra_feature_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_test_loss = float('inf')
    best_corr = 0

    has_extra = extra_feature_dim > 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            if has_extra:
                windows, extra, targets = batch
                windows, extra, targets = windows.to(device), extra.to(device), targets.to(device)
                outputs = model(windows, extra)
            else:
                windows, targets = batch
                windows, targets = windows.to(device), targets.to(device)
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
            for batch in test_loader:
                if has_extra:
                    windows, extra, targets = batch
                    windows, extra, targets = windows.to(device), extra.to(device), targets.to(device)
                    outputs = model(windows, extra)
                else:
                    windows, targets = batch
                    windows, targets = windows.to(device), targets.to(device)
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
print("Feature Ablation Study")
print("="*70)

# Load data
print("\nLoading data...")
data = np.load('ninapro_db5_segmented.npz', allow_pickle=True)
segments = data['segments']

TEST_SUBJECTS = [9, 10]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]

train_segments = [seg for seg in segments if seg['subject_id'] in TRAIN_SUBJECTS]
test_segments = [seg for seg in segments if seg['subject_id'] in TEST_SUBJECTS]

# Limit segments for faster experimentation
train_segments = train_segments[:500]
test_segments = test_segments[:125]

print(f"Train segments: {len(train_segments)}")
print(f"Test segments: {len(test_segments)}")

# Define experiments
experiments = {
    'Baseline (Raw EMG)': [],
    'Baseline + TKEO': [compute_tkeo],
    'Baseline + MAD': [compute_mad],
    'Baseline + TKEO_STD': [compute_tkeo_std],
    'Baseline + NLE': [compute_nle],
    'Baseline + AAC': [compute_aac],
    'Baseline + VAR': [compute_var],
    'Baseline + MAV': [compute_mav],
    'Baseline + TKEO + MAD': [compute_tkeo, compute_mad],
    'Baseline + TKEO + MAD + NLE': [compute_tkeo, compute_mad, compute_nle],
    'Baseline + All Novel (TKEO, MAD, NLE, AAC, TKEO_STD)': [compute_tkeo, compute_mad, compute_nle, compute_aac, compute_tkeo_std],
    'Baseline + Traditional (VAR, MAV, RMS)': [compute_var, compute_mav, compute_rms],
    'Baseline + Best Combo (TKEO, MAD, VAR, MAV)': [compute_tkeo, compute_mad, compute_var, compute_mav],
}

results = {}

for exp_name, feature_funcs in experiments.items():
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")

    extra_dim = len(feature_funcs) * 16 if feature_funcs else 0
    print(f"Extra feature dimension: {extra_dim}")

    # Create datasets
    print("Creating train dataset...")
    train_dataset = EMGDataset(train_segments, window_size=20, feature_funcs=feature_funcs)
    print("Creating test dataset...")
    test_dataset = EMGDataset(test_segments, window_size=20, feature_funcs=feature_funcs)

    # Train and evaluate
    print("Training...")
    best_loss, best_corr = train_and_evaluate(
        train_dataset, test_dataset,
        extra_feature_dim=extra_dim,
        epochs=30
    )

    results[exp_name] = {
        'test_loss': float(best_loss),
        'mean_correlation': float(best_corr),
        'extra_feature_dim': extra_dim
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

print(f"\n{'Experiment':<50} {'Test Loss':>12} {'Mean Corr':>12} {'Loss Δ':>10} {'Corr Δ':>10}")
print("-"*94)

sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_correlation'], reverse=True)

for exp_name, data in sorted_results:
    loss_delta = (baseline_loss - data['test_loss']) / baseline_loss * 100
    corr_delta = (data['mean_correlation'] - baseline_corr) / baseline_corr * 100
    print(f"{exp_name:<50} {data['test_loss']:>12.4f} {data['mean_correlation']:>12.4f} {loss_delta:>+9.2f}% {corr_delta:>+9.2f}%")

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
                'corr_improvement_pct': (data['mean_correlation'] - baseline_corr) / baseline_corr * 100
            }
            for name, data in sorted_results
        ]
    }, f, indent=2)

print("\nSaved: feature_ablation_results.json")
print("\n" + "="*70)
print("Ablation Study Complete!")
print("="*70)
