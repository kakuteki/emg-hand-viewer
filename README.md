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

QtのGUIは主スレッドでしか作れません。そのため使い方は次の2通りです。
（以前の版にあった「別スレッドでGUIを回す」使い方はQtの制約で動かないため、
呼ぶと理由を添えた例外になります）

### 1. 自分のループの中で回す

```python
from emg_realtime_viz import HandViewer

viewer = HandViewer(
    title="My Hand Viewer",
    show_ground_truth=True,
    show_prediction=True,
    angle_scale=1.0
)
viewer.open()

while viewer.process():
    emg_frame = get_emg_from_sensor()
    viewer.set_prediction(model.predict(emg_frame))   # ここで描画も進む

    if ground_truth is not None:
        viewer.set_ground_truth(ground_truth)

viewer.close()
```

`with` でも同じことができます。

```python
from emg_realtime_viz import HandViewerContext

with HandViewerContext() as viewer:
    for emg_frame in data_stream:
        viewer.set_prediction(model.predict(emg_frame))
        if not viewer.is_running():
            break
```

### 2. データ供給を別スレッドにする

ウィンドウを閉じるまで `run()` から戻りません。

```python
from emg_realtime_viz import HandViewer

def producer(viewer):
    while viewer.is_running():
        viewer.set_prediction(model.predict(get_emg_from_sensor()))

HandViewer().run(producer)
```

### HandViewer API リファレンス

| メソッド | 役割 |
|---------|------|
| `open()` | ウィンドウを作る（主スレッドから呼ぶこと。イベントループは回さない） |
| `process()` | 溜まった描画と入力を処理する。開いていれば `True` を返す |
| `run(producer=None)` | イベントループを回す。`producer` は別スレッドで実行される |
| `close()` / `stop()` | 閉じる（どのスレッドからでも呼べる） |
| `is_running()` | 動いているか |
| `set_prediction(angles)` | 予測値（緑の手）を更新。20次元・0-1 |
| `set_ground_truth(angles)` | 実測値（青の手）を更新。20次元・0-1 |
| `set_both(gt, pred)` | 両手を同時に更新 |
| `set_angle_scale(scale)` | 角度のスケールを変える |

姿勢の更新はどのスレッドからでも呼べます。主スレッドから呼んだ場合はその場で描画まで進み、
別スレッドから呼んだ場合はキューに積まれてGUI側が取り出します。

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

例外が出ないことだけでは判別できない点に注意しています。全結合層は`(1, 20, 16)`を渡しても
内部で時間方向に放送されて`(1, 20, 22)`を返してしまうため、出力の要素数が20か22であることまで
確かめたうえで入力の形を決めています。

出力が22次元（グローブと同じ並び）のときは先頭20要素を使います。
**出力は0-1の範囲であることを前提**にしており、範囲外は0と1に丸めます。
標準化した値（平均0）を出すモデルは、手が伸びたまま動かないように見えるので、
学習側の出口にsigmoidなどを入れてください。

## 環境の注意点

- **PyTorchはQtより先に読み込む必要があります。** 逆にするとWindowsでtorchのDLL初期化が
  失敗し（WinError 1114）、学習済みモデルが一切読めません。起動スクリプトの側で対処済みですが、
  自分でスクリプトを書くときは`import torch`を先に置いてください
- **PySide6が入っている環境**では、pyqtgraphがそちらを掴んでQtが二重に載り、
  `QWidget: Must construct a QApplication before a QWidget`で即クラッシュします。
  `emg_realtime_viz.qt_compat`が読み込み前にPyQt5を指定して防いでいます。
  このライブラリはPyQt5を直接使うので、環境変数`PYQTGRAPH_QT_LIB`に別のQtを
  指定すると同じ問題が再発します（その場合は警告を出します）
- 同じプロセスで3D表示のウィンドウを閉じてから開き直すと、pyqtgraph側の都合で
  描画に失敗することがあります。開き直す場合はプロセスを分けてください

## プロジェクト構成

```
.
├── emg_realtime_viz/           # メインライブラリ
│   ├── __init__.py
│   ├── qt_compat.py            # pyqtgraphが使うQtの指定
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

## データグローブの扱い

DB5が使うCyberGlove IIの22センサは、**指1本あたり屈曲3個（計15個）、指の間の外転4個、
手のひらのそり1個、手首2個**という構成です（3×5 + 4 + 1 + 2 = 22）。
「指1本あたり4個 × 5本 + 手首2個」ではありません。

そのため、先頭20列をそのまま指の関節と見なすと、外転センサや手のひらのそりを
指の曲げとして扱ってしまいます。使う列は`FLEXION_COLUMNS`で明示しています。

```python
FLEXION_COLUMNS = {
    "thumb":  (0, 1, 2),    # ひねり(CMC), MP, IP
    "index":  (4, 5, 6),    # MP, PIP, DIP
    "middle": (8, 9, 10),
    "ring":   (12, 13, 14),
    "pinky":  (16, 17, 18),
}
```

⚠ 列の並びを断定できる一次資料は確認できていません。Ninapro公式の説明ページ
（ninapro.hevs.ch/node/123）は現在404です。上の並びはCyberGloveの取扱説明書に沿ったもので、
屈曲センサを先にまとめて並べる実装向けに`FLEXION_COLUMNS_GROUPED`も用意しています。
実データで確かめる場合は、指を1本ずつ曲げた区間で各列との相関を見てください。

値は未校正（角度に比例する生の値）なので、正規化の範囲は読み込んだデータの分位点から
求めます（`GloveNormalizer`）。度数を仮定した固定の範囲は、データから求められないときの
最後の手段です。

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
