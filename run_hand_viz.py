#!/usr/bin/env python3
"""
EMG Hand Visualizer with Inference Display
==========================================

EMG信号から推論した手のポーズをリアルタイムで3D表示

使用方法:
    # ファイルから再生（実測値のみ）
    python run_hand_viz.py

    # ダミー推論モデル付き
    python run_hand_viz.py --demo-inference

    # 学習済みモデルを使用
    python run_hand_viz.py --model path/to/model.pth

    # Myo Armbandを使用
    python run_hand_viz.py --myo --model path/to/model.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ライブラリパスを追加
sys.path.insert(0, str(Path(__file__).parent))

from emg_realtime_viz import (
    HandVisualizer,
)


def create_demo_model():
    """
    デモ用の疑似推論モデル

    実際の推論ではなく、入力特徴量に基づいた
    疑似的な関節角度を生成

    角度インデックス:
    - 0-3: 親指 (CMC, MCP, IP, TIP)
    - 4-7: 人差し指 (MCP, PIP, DIP, TIP)
    - 8-11: 中指
    - 12-15: 薬指
    - 16-19: 小指
    """
    import time

    start_time = time.time()

    def demo_inference(features: np.ndarray) -> np.ndarray:
        # 20次元の関節角度（各指4関節 x 5本）
        n_angles = 20

        # 特徴量の平均エネルギーを計算
        energy = np.mean(np.abs(features))
        energy = np.clip(energy, 0, 2)  # クリップ

        # 時間ベースのアニメーション
        t = time.time() - start_time

        # 各関節の角度（0-1の範囲）
        angles = np.zeros(n_angles)

        # 各指を個別にアニメーション
        for finger_idx in range(5):
            base = finger_idx * 4
            # 指ごとに位相をずらす
            phase = finger_idx * 0.3

            for joint_idx in range(4):
                # 根元から先端へ徐々に遅延
                delay = joint_idx * 0.15

                # 波のようなモーション
                wave = np.sin(t * 2 + phase + delay)

                # エネルギーに応じて屈曲
                flex = 0.3 + 0.3 * (wave * 0.5 + 0.5) + 0.2 * energy

                angles[base + joint_idx] = flex

        # ノイズを追加
        angles += np.random.randn(n_angles) * 0.02

        return np.clip(angles, 0, 1)

    return demo_inference


def load_pytorch_model(model_path: str):
    """
    PyTorchモデルをロード

    Parameters
    ----------
    model_path : str
        モデルファイルのパス (.pth)

    Returns
    -------
    callable
        推論関数
    """
    try:
        import torch
    except ImportError:
        print("PyTorchがインストールされていません: pip install torch")
        return None

    print(f"モデルをロード中: {model_path}")

    # モデルをロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        print("state_dictとしてロードを試みます...")

        # state_dictの場合は構造が必要
        # ここでは一般的なMLPを仮定
        from torch import nn

        class SimpleRegressor(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, output_dim),
                )

            def forward(self, x):
                return self.net(x)

        # 入力次元を推測
        state_dict = torch.load(model_path, map_location=device)
        first_layer = list(state_dict.keys())[0]
        input_dim = state_dict[first_layer].shape[1]
        output_dim = 22  # Ninapro glove

        model = SimpleRegressor(input_dim, output_dim)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

    print(f"モデルロード完了 (device: {device})")

    def inference(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            y = model(x)
            return y.cpu().numpy().flatten()

    return inference


def main():
    parser = argparse.ArgumentParser(
        description="EMG Hand Visualizer with Inference Display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # データソース
    parser.add_argument(
        "--file", "-f", type=str, default="ninapro_db5_segmented.npz", help="データファイルパス"
    )

    # 推論モデル
    parser.add_argument("--model", type=str, help="学習済みモデルのパス (.pth)")
    parser.add_argument("--demo-inference", action="store_true", help="デモ用疑似推論モデルを使用")

    # 表示設定
    parser.add_argument("--no-ground-truth", action="store_true", help="実測値を非表示")
    parser.add_argument("--no-prediction", action="store_true", help="予測値を非表示")
    parser.add_argument("--speed", type=float, default=1.0, help="再生速度")

    args = parser.parse_args()

    print("=" * 60)
    print("EMG Hand Visualizer with Inference Display")
    print("=" * 60)

    # データファイルパス
    data_path = Path(args.file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / args.file

    if not data_path.exists():
        print(f"エラー: データファイルが見つかりません: {data_path}")
        sys.exit(1)

    print(f"データファイル: {data_path}")

    # 推論モデル
    inference_model = None
    if args.model:
        inference_model = load_pytorch_model(args.model)
        if inference_model is None:
            print("モデルロードに失敗しました。デモモードで起動します。")
            inference_model = create_demo_model()
    elif args.demo_inference:
        print("デモ推論モデルを使用")
        inference_model = create_demo_model()
    else:
        print("推論モデルなし（実測値のみ表示）")

    print(f"実測値表示: {'OFF' if args.no_ground_truth else 'ON'}")
    print(f"予測値表示: {'OFF' if args.no_prediction else 'ON'}")
    print("=" * 60)

    # 可視化アプリ作成（GUIでデータ選択可能）
    viz = HandVisualizer(
        filepath=str(data_path),
        inference_model=inference_model,
        show_ground_truth=not args.no_ground_truth,
        show_prediction=not args.no_prediction and inference_model is not None,
        playback_speed=args.speed,
    )

    print("\nアプリケーションを起動中...")
    print("操作方法:")
    print("  - Subject/Movement: プルダウンでデータ選択")
    print("  - Play: 再生開始")
    print("  - Pause: 一時停止")
    print("  - Ground Truth: 実測値（青）の表示切替")
    print("  - Prediction: 予測値（緑）の表示切替")
    print("  - Angle Scale: 関節角度のスケール調整")
    print("  - マウスドラッグ: 視点回転")
    print()

    return viz.run()


if __name__ == "__main__":
    sys.exit(main())
