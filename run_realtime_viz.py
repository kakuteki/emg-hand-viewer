#!/usr/bin/env python3
"""
EMG Realtime 3D Visualization
=============================

Ninapro DB5 または Myo Armband からのデータをリアルタイムで3D可視化

使用方法:
    # ファイルから再生
    python run_realtime_viz.py [options]

    # Myo Armbandを使用
    python run_realtime_viz.py --myo

    # Myoシミュレーションモード
    python run_realtime_viz.py --myo --simulated

オプション:
    --file PATH     データファイルパス (デフォルト: ninapro_db5_segmented.npz)
    --subject ID    被験者ID (複数指定可: --subject 1 --subject 2)
    --movement ID   動作ID (複数指定可: --movement 1 --movement 2)
    --speed FLOAT   再生速度 (デフォルト: 1.0)
    --window INT    特徴量ウィンドウサイズ (デフォルト: 20)
    --trail INT     軌跡の長さ (デフォルト: 500)
    --myo           Myo Armbandを使用
    --simulated     Myoシミュレーションモード
"""

import argparse
import sys
from pathlib import Path

# ライブラリパスを追加
sys.path.insert(0, str(Path(__file__).parent))

from emg_realtime_viz import NinaproDataSource, RealtimeVisualizer, get_myo_source


def main():
    parser = argparse.ArgumentParser(
        description="EMG Realtime 3D Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # データソース選択
    source_group = parser.add_argument_group("Data Source")
    source_group.add_argument("--myo", action="store_true", help="Myo Armbandを使用")
    source_group.add_argument(
        "--simulated", action="store_true", help="Myoシミュレーションモード（実機不要）"
    )

    # ファイルモード用オプション
    file_group = parser.add_argument_group("File Mode Options")
    file_group.add_argument(
        "--file", "-f", type=str, default="ninapro_db5_segmented.npz", help="データファイルパス"
    )
    file_group.add_argument(
        "--subject", "-s", type=int, action="append", help="被験者ID (複数指定可)"
    )
    file_group.add_argument(
        "--movement", "-m", type=int, action="append", help="動作ID (複数指定可)"
    )
    file_group.add_argument("--no-loop", action="store_true", help="データ終了時にループしない")

    # 可視化オプション
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument("--speed", type=float, default=1.0, help="再生速度 (デフォルト: 1.0)")
    viz_group.add_argument(
        "--window", type=int, default=20, help="特徴量ウィンドウサイズ (デフォルト: 20)"
    )
    viz_group.add_argument("--trail", type=int, default=500, help="軌跡の長さ (デフォルト: 500)")
    viz_group.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["mav", "rms", "var"],
        help="使用する特徴量 (デフォルト: mav rms var)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EMG Realtime 3D Visualization")
    print("=" * 60)

    # データソース作成
    if args.myo or args.simulated:
        # Myo Armbandモード
        print(f"モード: Myo Armband {'(シミュレーション)' if args.simulated else '(実機)'}")
        source = get_myo_source(simulated=args.simulated, n_channels=8, sample_rate=200.0)
    else:
        # ファイルモード
        data_path = Path(args.file)
        if not data_path.is_absolute():
            data_path = Path(__file__).parent / args.file

        if not data_path.exists():
            print(f"エラー: データファイルが見つかりません: {data_path}")
            sys.exit(1)

        print("モード: ファイル再生")
        print(f"データファイル: {data_path}")
        print(f"被験者フィルタ: {args.subject if args.subject else 'すべて'}")
        print(f"動作フィルタ: {args.movement if args.movement else 'すべて'}")

        source = NinaproDataSource(
            filepath=str(data_path),
            subject_ids=args.subject,
            movements=args.movement,
            loop=not args.no_loop,
            n_channels=16,
            sample_rate=200.0,
        )

    print(f"再生速度: {args.speed}x")
    print(f"ウィンドウサイズ: {args.window}")
    print(f"軌跡の長さ: {args.trail}")
    print(f"特徴量: {args.features}")
    print("=" * 60)

    # 可視化アプリ作成
    viz = RealtimeVisualizer(
        source=source,
        window_size=args.window,
        trail_length=args.trail,
        playback_speed=args.speed,
        features=args.features,
    )

    print("\nアプリケーションを起動中...")
    print("操作方法:")
    print("  - Play: 再生開始")
    print("  - Pause: 一時停止")
    print("  - Reset: 最初から再生")
    print("  - Speed: 再生速度調整")
    print("  - Trail: 軌跡の長さ調整")
    print("  - Color Mode: 色分けモード切替")
    print("  - マウスドラッグ: 視点回転")
    print("  - マウスホイール: ズーム")
    print()

    # アプリケーション実行
    return viz.run()


if __name__ == "__main__":
    sys.exit(main())
