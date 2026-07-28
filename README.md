# EMG Hand Pose Estimation

[![CI](https://github.com/kakuteki/emg-hand-viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/kakuteki/emg-hand-viewer/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

筋電図（EMG）信号から手のポーズをリアルタイムで推定・可視化するためのPythonライブラリ

## 概要

本ライブラリは、表面筋電図（sEMG）信号を入力として、手の関節角度をリアルタイムで推定し、3Dモデルとして可視化する機能を提供します。Ninapro Database 5のデータ形式に対応しており、学習済みモデルを用いた推論結果の可視化が可能です。

### 主な機能

- リアルタイム3D手モデル可視化
- EMG信号からの手のポーズ推論
- Ground Truth（実測値）と予測値の比較表示
- 外部アプリケーションから利用可能なWrapper API
- Ninapro DB5データセット対応
- 学習済みモデルを用いた推論

## インストール

```bash
git clone https://github.com/kakuteki/emg-hand-viewer.git
cd emg-hand-viewer
pip install -r requirements.txt
```

### 依存パッケージ

必須は表示に使うものだけです。

- Python 3.9以上
- NumPy
- PyQt5 / pyqtgraph / PyOpenGL

用途に応じて追加します。

- PyTorch: 学習済みモデルを使うとき（`pip install torch`）
- pyomyo: Myo Armbandを実機で使うとき（`pip install pyomyo`）
- pytest / ruff: 開発時（`pip install -r requirements-dev.txt`）

## 使い方

### 1. EMG推論ビューワー（GUI）

Subject/Movementをプルダウンメニューで選択し、EMGデータの再生と推論結果の表示を行います。
モデルは`models/`、データは`data/`に置くと自動で一覧に出ます（画面のImportボタンでも取り込めます）。

```bash
python run_inference_app.py
```

オプション:
- `--model`, `-m`: 起動時に選ぶ学習済みモデル（.pth）
- `--data`, `-d`: 起動時に選ぶデータファイル（.npz）

```bash
# 学習済みモデルを指定して起動
python run_inference_app.py --model models/best_model.pth
```

### 2. 手モデルビューワー

Ground Truth（青）と予測値（緑）を並べて表示します。

```bash
# デモ推論モデル付きで起動
python run_hand_viz.py --demo-inference

# 学習済みモデルを使用
python run_hand_viz.py --model best_model.pth

# 実測値のみ表示
python run_hand_viz.py --no-prediction
```

### 3. リアルタイム可視化

EMG信号と特徴量をリアルタイムで可視化します。

```bash
python run_realtime_viz.py
```

## Wrapper API

他のアプリケーションから手モデルを制御するためのAPIを提供しています。

### 基本的な使い方

```python
from emg_realtime_viz import HandViewer
import numpy as np

# ビューワーを作成
viewer = HandViewer(
    title="My Hand Viewer",
    show_ground_truth=True,
    show_prediction=True,
    angle_scale=1.0
)

# 別スレッドでビューワーを起動
viewer.start()

# メインループで手のポーズを更新
while viewer.is_running():
    # EMGデータを取得（例）
    emg_data = get_emg_from_sensor()

    # モデルで推論
    predicted_angles = model.predict(emg_data)

    # 予測値（緑の手）を更新
    viewer.set_prediction(predicted_angles)

    # 実測値がある場合（青の手）
    if ground_truth is not None:
        viewer.set_ground_truth(ground_truth)

# 終了
viewer.stop()
```

### コンテキストマネージャーを使用

```python
from emg_realtime_viz import HandViewerContext

with HandViewerContext() as viewer:
    for emg_frame in data_stream:
        prediction = model.predict(emg_frame)
        viewer.set_prediction(prediction)

# withブロックを抜けると自動的にビューワーが終了
```

### HandViewer API リファレンス

```python
class HandViewer:
    def __init__(
        self,
        title: str = "Hand Viewer",
        size: tuple = (800, 600),
        show_ground_truth: bool = True,
        show_prediction: bool = True,
        angle_scale: float = 1.0
    ):
        """
        Parameters
        ----------
        title : str
            ウィンドウタイトル
        size : tuple
            ウィンドウサイズ (width, height)
        show_ground_truth : bool
            実測値（青い手）を表示するか
        show_prediction : bool
            予測値（緑の手）を表示するか
        angle_scale : float
            関節角度の表示スケール
        """

    def start(self, blocking: bool = False):
        """
        ビューワーを開始

        Parameters
        ----------
        blocking : bool
            Trueの場合、ウィンドウが閉じるまでブロック
            Falseの場合、別スレッドで実行（デフォルト）
        """

    def stop(self):
        """ビューワーを停止"""

    def set_prediction(self, angles: np.ndarray):
        """
        予測値（緑の手）を更新

        Parameters
        ----------
        angles : np.ndarray
            関節角度 shape=(20,)、各値は0-1の範囲
        """

    def set_ground_truth(self, angles: np.ndarray):
        """
        実測値（青の手）を更新

        Parameters
        ----------
        angles : np.ndarray
            関節角度 shape=(20,)、各値は0-1の範囲
        """

    def set_both(self, ground_truth: np.ndarray, prediction: np.ndarray):
        """両手を同時に更新"""

    def is_running(self) -> bool:
        """ビューワーが実行中かどうかを返す"""
```

## 手モデルの関節角度

20次元のベクトルで手のポーズを表現します（各指4関節 x 5本）。
各値は0（伸展）から1（屈曲）の範囲です。

| インデックス | 関節 |
|-------------|------|
| 0-3 | 親指 (CMC, MCP, IP, TIP) |
| 4-7 | 人差し指 (MCP, PIP, DIP, TIP) |
| 8-11 | 中指 (MCP, PIP, DIP, TIP) |
| 12-15 | 薬指 (MCP, PIP, DIP, TIP) |
| 16-19 | 小指 (MCP, PIP, DIP, TIP) |

骨格の動かし方は次のとおりです。

- 親指は手首からの4本の骨がすべて回ります
- 他の4本指は、手首からMCP（付け根）までが手のひらの骨なので回しません。
  曲げても付け根の位置は動きません。残る3本の骨にMCP・PIP・DIPを割り当てるため、
  TIPの値は姿勢計算には使われません
- 描画なしで姿勢だけ計算したい場合は `forward_kinematics(angles)` を使います
  （21点の関節位置を返す。PyQtは不要）

## 推論モデルの入出力

推論関数はすべて「EMGの1フレーム（生値）を受け取り、20次元の関節角度（0-1）を返す」形にそろえてあります。
特徴量ではなく生値を渡すのは、学習時の入力に合わせるためです。

`load_torch_model()` は、モデルが受け付ける入力の形を最初の呼び出しで次の順に試し、通った形を覚えます。

1. `(1, 20, 16)` 時系列モデル（LSTM等）
2. `(1, 16)` 1フレーム入力のモデル
3. `(1, 320)` 窓を平らに並べたモデル

出力が22次元（グローブと同じ並び）のときは先頭20要素を使います。

## プロジェクト構成

```
.
├── emg_realtime_viz/           # メインライブラリ
│   ├── __init__.py
│   ├── core/                   # コア機能
│   │   ├── data_loader.py      # Ninaproデータローダー
│   │   ├── feature_extractor.py # EMG特徴量抽出
│   │   ├── glove.py            # グローブ値の正規化（唯一の定義元）
│   │   ├── inference.py        # モデル読み込みと推論関数
│   │   └── stream.py           # データストリーミング
│   ├── devices/                # デバイス対応
│   │   ├── base.py             # 基底クラス
│   │   ├── file_source.py      # ファイル再生
│   │   └── myo_source.py       # Myo Armband対応
│   └── viz/                    # 可視化
│       ├── realtime_3d.py      # リアルタイム3D表示
│       ├── hand_model.py       # 手の3Dモデルと順運動学
│       ├── hand_visualizer.py  # 統合ビューワー
│       └── hand_viewer.py      # 外部API（Wrapper）
│
├── run_inference_app.py        # 推論アプリ（メインGUI）
├── run_hand_viz.py             # 手モデルビューワー
├── run_realtime_viz.py         # リアルタイム可視化
├── hand_model_playground.py    # 手モデルを手で動かす確認用GUI
├── tests/                      # 自動試験（pytest・GUI不要）
│
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml              # パッケージ定義とpytest設定
├── ruff.toml                   # Linter設定
├── README.md
└── LICENSE
```

## 開発

```bash
pip install -r requirements-dev.txt

pytest              # 試験（GUIもPyQtも不要）
ruff check .        # 文法・書式の検査
ruff format .       # 書式の自動整形
```

手モデルの動きだけ手早く確かめたいときは、指ごとのつまみと
「開く / 握る / 指さす / 波」のボタンが付いた確認用GUIを使います。

```bash
python hand_model_playground.py
```

## データ

Ninapro Database 5のセグメント化データが必要です。

データ形式（`ninapro_db5_segmented.npz`）:
```python
{
    'segments': [
        {
            'subject_id': int,      # 被験者ID (1-10)
            'exercise_id': int,     # エクササイズID
            'movement': int,        # 動作ID (1-23)
            'repetition': int,      # 繰り返し回数
            'emg': ndarray,         # EMG信号 shape=(T, 16)
            'glove': ndarray,       # グローブデータ shape=(T, 22)
            'n_samples': int        # サンプル数
        },
        ...
    ]
}
```

## ライセンス

Apache License 2.0

Copyright 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 参考文献

- Ninapro Database: http://ninapro.hevs.ch/
- Ninapro DB5: Pizzolato et al., "Comparison of six electromyography acquisition setups on hand movement classification tasks," PLoS ONE, 2017.
