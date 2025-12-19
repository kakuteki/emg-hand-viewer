"""
論文引用可能な表画像を作成
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 日本語フォント設定（英語のみ使用するため不要だが念のため）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================
# Table 1: Model Comparison
# =====================

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

# データ（モデル名を少し短縮）
columns = ['Model', 'Test MSE', 'Mean r', 'Max r', 'Min r', 'Parameters']
data = [
    ['LSTM (Baseline)', '0.9585', '0.336', '0.527', '−0.048', '826,166'],
    ['CNN-LSTM', '0.9433', '0.361', '0.557', '0.085', '895,766'],
    ['Multi-Head Attn. LSTM', '0.9405', '0.355', '0.522', '0.060', '909,878'],
    ['Causal Self-Attn. LSTM', '0.9361', '0.360', '0.543', '0.071', '909,878'],
    ['Self-Attention LSTM', '0.9282', '0.364', '0.552', '0.082', '909,878'],
    ['Residual Self-Attn. LSTM', '0.9497', '0.351', '0.527', '0.093', '929,302'],
    ['Stacked Self-Attn. (6-layer)', '0.9507', '0.345', '0.536', '0.058', '852,182'],
]

# 表を作成
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc='center',
    cellLoc='center',
    colColours=['#E6E6E6'] * len(columns),
    colWidths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.14],
)

# スタイル設定
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 1.8)

# ヘッダーのスタイル
for i in range(len(columns)):
    table[(0, i)].set_text_props(weight='bold')
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', weight='bold')

# ベストモデル（Self-Attention LSTM, 行インデックス5）をハイライト
best_row = 5  # 0-indexed, ヘッダー含む
for i in range(len(columns)):
    table[(best_row, i)].set_facecolor('#D6EAF8')
    table[(best_row, i)].set_text_props(weight='bold')

# 交互に背景色を変更
for i in range(1, len(data) + 1):
    if i != best_row:
        color = '#FFFFFF' if i % 2 == 1 else '#F5F5F5'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)

# セルの枠線
for key, cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)

# タイトル
plt.title('Table 1. Comparison of Deep Learning Models for EMG-to-Joint Angle Prediction\n',
          fontsize=12, fontweight='bold', y=0.95)

# 注釈
note_text = (
    "Note: Test MSE = Mean Squared Error on held-out test set (subjects 9–10); "
    "r = Pearson correlation coefficient across 22 finger joint angles.\n"
    "All models trained with identical hyperparameters: window size = 20 (100 ms), "
    "batch size = 2,048, hidden size = 256, LSTM layers = 2,\n"
    "dropout = 0.2, learning rate = 2.1×10⁻⁴, epochs = 50. "
    "Bold row indicates best overall performance."
)
plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=8, style='italic',
            wrap=True, bbox=dict(boxstyle='round', facecolor='#FAFAFA', edgecolor='#CCCCCC'))

plt.tight_layout()
plt.savefig('table1_model_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('table1_model_comparison.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Saved: table1_model_comparison.png")
print("Saved: table1_model_comparison.pdf")


# =====================
# Table 2: Experimental Configuration
# =====================

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

columns2 = ['Parameter', 'Value']
data2 = [
    ['Dataset', 'NinaPro DB5'],
    ['EMG channels', '16 (dual Myo Armband)'],
    ['Sampling rate', '200 Hz'],
    ['Output', '22 joint angles'],
    ['Training subjects', '1–8 (n = 8)'],
    ['Test subjects', '9–10 (n = 2)'],
    ['Window size', '20 samples (100 ms)'],
    ['Batch size', '2,048'],
    ['Optimizer', 'Adam'],
    ['Learning rate', '2.1×10⁻⁴'],
    ['LR scheduler', 'ReduceLROnPlateau'],
    ['Epochs', '50'],
    ['Loss function', 'MSE'],
]

table2 = ax.table(
    cellText=data2,
    colLabels=columns2,
    loc='center',
    cellLoc='left',
    colColours=['#E6E6E6'] * len(columns2),
)

table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1.5, 1.6)

# ヘッダーのスタイル
for i in range(len(columns2)):
    table2[(0, i)].set_text_props(weight='bold')
    table2[(0, i)].set_facecolor('#4472C4')
    table2[(0, i)].set_text_props(color='white', weight='bold')
    table2[(0, i)].set_text_props(ha='center')

# 交互に背景色
for i in range(1, len(data2) + 1):
    color = '#FFFFFF' if i % 2 == 1 else '#F5F5F5'
    for j in range(len(columns2)):
        table2[(i, j)].set_facecolor(color)

# セルの枠線
for key, cell in table2.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)

plt.title('Table 2. Experimental Configuration\n', fontsize=12, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig('table2_experimental_config.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('table2_experimental_config.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Saved: table2_experimental_config.png")
print("Saved: table2_experimental_config.pdf")


# =====================
# Combined Figure (両方の表を1つの画像に)
# =====================

fig, axes = plt.subplots(2, 1, figsize=(14, 11), height_ratios=[1.2, 1])

# Table 1
ax1 = axes[0]
ax1.axis('off')

table1_combined = ax1.table(
    cellText=data,
    colLabels=columns,
    loc='center',
    cellLoc='center',
    colColours=['#E6E6E6'] * len(columns),
    colWidths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.14],
)

table1_combined.auto_set_font_size(False)
table1_combined.set_fontsize(9)
table1_combined.scale(1.2, 1.6)

for i in range(len(columns)):
    table1_combined[(0, i)].set_text_props(weight='bold')
    table1_combined[(0, i)].set_facecolor('#4472C4')
    table1_combined[(0, i)].set_text_props(color='white', weight='bold')

for i in range(len(columns)):
    table1_combined[(best_row, i)].set_facecolor('#D6EAF8')
    table1_combined[(best_row, i)].set_text_props(weight='bold')

for i in range(1, len(data) + 1):
    if i != best_row:
        color = '#FFFFFF' if i % 2 == 1 else '#F5F5F5'
        for j in range(len(columns)):
            table1_combined[(i, j)].set_facecolor(color)

for key, cell in table1_combined.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)

ax1.set_title('Table 1. Comparison of Deep Learning Models for EMG-to-Joint Angle Prediction',
              fontsize=11, fontweight='bold', pad=10)

# Table 2
ax2 = axes[1]
ax2.axis('off')

table2_combined = ax2.table(
    cellText=data2,
    colLabels=columns2,
    loc='center',
    cellLoc='left',
    colColours=['#E6E6E6'] * len(columns2),
)

table2_combined.auto_set_font_size(False)
table2_combined.set_fontsize(9)
table2_combined.scale(1.3, 1.4)

for i in range(len(columns2)):
    table2_combined[(0, i)].set_text_props(weight='bold')
    table2_combined[(0, i)].set_facecolor('#4472C4')
    table2_combined[(0, i)].set_text_props(color='white', weight='bold')

for i in range(1, len(data2) + 1):
    color = '#FFFFFF' if i % 2 == 1 else '#F5F5F5'
    for j in range(len(columns2)):
        table2_combined[(i, j)].set_facecolor(color)

for key, cell in table2_combined.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)

ax2.set_title('Table 2. Experimental Configuration', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('tables_combined.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('tables_combined.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Saved: tables_combined.png")
print("Saved: tables_combined.pdf")

print("\nAll table images created successfully!")
