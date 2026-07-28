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
"""

import argparse
import sys
from pathlib import Path

# ライブラリパスを追加
sys.path.insert(0, str(Path(__file__).parent))

from emg_realtime_viz import (
    HandVisualizer,
)
from emg_realtime_viz.core.inference import load_torch_model, wave_demo_model


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
        print(f"モデルをロード中: {args.model}")
        try:
            inference_model = load_torch_model(args.model)
            print(f"モデルロード完了 (device: {inference_model.device})")
        except Exception as e:
            print(f"モデルロードエラー: {e}")
            print("デモモードで起動します。")
            inference_model = wave_demo_model()
    elif args.demo_inference:
        print("デモ推論モデルを使用")
        inference_model = wave_demo_model()
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
