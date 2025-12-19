# EMG Hand Pose Estimation

筋電図（EMG）信号から手のポーズをリアルタイムで推定・可視化するライブラリ

## Features

- **リアルタイム3D可視化**: EMG信号と手のポーズを3Dで表示
- **推論モデル統合**: 学習済みモデルによる手のポーズ推定
- **Ninapro DB5対応**: Ninapro Database 5のデータ形式をサポート
- **Myo Armband対応**: Myo Armbandからのリアルタイムデータ取得（予定）

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### EMG推論ビューワー

Subject/Movementをプルダウンで選択し、EMGから推論した手のポーズを表示：

```bash
python run_inference_app.py
```

### 手モデルビューワー

Ground Truth（実測値）と推論結果を並べて表示：

```bash
python run_hand_viz.py --demo-inference
```

### リアルタイム可視化

```bash
python run_realtime_viz.py
```

## Library Usage

### HandViewer API

他のアプリケーションから手モデルを制御：

```python
from emg_realtime_viz import HandViewer

# ビューワー起動
viewer = HandViewer()
viewer.start()

# 手のポーズを更新（0-1の範囲、20次元）
viewer.set_prediction(angles)
viewer.set_ground_truth(gt_angles)

# 終了
viewer.stop()
```

### コンテキストマネージャー

```python
from emg_realtime_viz import HandViewerContext

with HandViewerContext() as viewer:
    for emg_data in stream:
        pred = model.predict(emg_data)
        viewer.set_prediction(pred)
```

## Project Structure

```
.
├── emg_realtime_viz/          # メインライブラリ
│   ├── core/                  # コア機能
│   │   ├── data_loader.py     # データローダー
│   │   ├── feature_extractor.py  # 特徴量抽出
│   │   └── stream.py          # ストリーミング
│   ├── devices/               # デバイス対応
│   │   ├── base.py            # 基底クラス
│   │   ├── file_source.py     # ファイル再生
│   │   └── myo_source.py      # Myo Armband
│   └── viz/                   # 可視化
│       ├── realtime_3d.py     # 3D可視化
│       ├── hand_model.py      # 手モデル
│       ├── hand_visualizer.py # 統合ビューワー
│       └── hand_viewer.py     # 外部API
├── run_inference_app.py       # 推論アプリ（GUI）
├── run_hand_viz.py            # 手モデルビューワー
├── run_realtime_viz.py        # リアルタイム可視化
└── *_train.py                 # 学習スクリプト
```

## Hand Model

20次元の関節角度（各指4関節 × 5本）：

| Index | Joint |
|-------|-------|
| 0-3   | Thumb (CMC, MCP, IP, TIP) |
| 4-7   | Index (MCP, PIP, DIP, TIP) |
| 8-11  | Middle (MCP, PIP, DIP, TIP) |
| 12-15 | Ring (MCP, PIP, DIP, TIP) |
| 16-19 | Little (MCP, PIP, DIP, TIP) |

## Requirements

- Python 3.8+
- PyQt5
- pyqtgraph
- PyOpenGL
- NumPy
- PyTorch (推論モデル使用時)

## Data

Ninapro DB5のセグメント化データ（`ninapro_db5_segmented.npz`）が必要です。
データは別途ダウンロードしてください。

## License

MIT License
