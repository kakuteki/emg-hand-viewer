#!/usr/bin/env python3
"""
手モデルのテストスクリプト
"""

import sys
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl

sys.path.insert(0, '.')
from emg_realtime_viz.viz.hand_model import HandModel3D, HandSkeleton


class HandModelTest:
    def __init__(self):
        self.app = QtWidgets.QApplication([])

        # ウィンドウ
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle('Hand Model Test')
        self.window.resize(1000, 800)

        # 中央ウィジェット
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 3Dビュー
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=5, elevation=30, azimuth=45)
        layout.addWidget(self.gl_widget, stretch=3)

        # グリッド
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.gl_widget.addItem(grid)

        # 軸
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self.gl_widget.addItem(axis)

        # 手モデル
        self.hand = HandModel3D(
            self.gl_widget,
            scale=1.0,
            position=(0, 0, 0),
            color=(0.8, 0.6, 0.4, 1.0)
        )

        # コントロールパネル
        control = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control)
        layout.addWidget(control, stretch=1)

        # スライダーで各指を制御
        self.sliders = {}
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

        for i, name in enumerate(finger_names):
            group = QtWidgets.QGroupBox(name)
            group_layout = QtWidgets.QVBoxLayout(group)

            slider = QtWidgets.QSlider(1)  # Horizontal
            slider.setRange(0, 100)
            slider.setValue(0)
            slider.valueChanged.connect(self.update_hand)
            group_layout.addWidget(slider)

            self.sliders[name] = slider
            control_layout.addWidget(group)

        # 全体スケール
        scale_group = QtWidgets.QGroupBox('Angle Scale')
        scale_layout = QtWidgets.QVBoxLayout(scale_group)
        self.scale_slider = QtWidgets.QSlider(1)
        self.scale_slider.setRange(1, 30)
        self.scale_slider.setValue(10)
        self.scale_slider.valueChanged.connect(self.update_hand)
        scale_layout.addWidget(self.scale_slider)
        control_layout.addWidget(scale_group)

        # プリセットボタン
        preset_group = QtWidgets.QGroupBox('Presets')
        preset_layout = QtWidgets.QVBoxLayout(preset_group)

        btn_open = QtWidgets.QPushButton('Open Hand')
        btn_open.clicked.connect(lambda: self.set_preset([0, 0, 0, 0, 0]))
        preset_layout.addWidget(btn_open)

        btn_fist = QtWidgets.QPushButton('Fist')
        btn_fist.clicked.connect(lambda: self.set_preset([80, 80, 80, 80, 80]))
        preset_layout.addWidget(btn_fist)

        btn_point = QtWidgets.QPushButton('Point')
        btn_point.clicked.connect(lambda: self.set_preset([50, 0, 80, 80, 80]))
        preset_layout.addWidget(btn_point)

        btn_wave = QtWidgets.QPushButton('Wave Animation')
        btn_wave.clicked.connect(self.start_wave)
        preset_layout.addWidget(btn_wave)

        control_layout.addWidget(preset_group)
        control_layout.addStretch()

        # アニメーション用タイマー
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_wave)
        self.wave_time = 0

    def set_preset(self, values):
        """プリセット値を設定"""
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        for i, name in enumerate(finger_names):
            self.sliders[name].setValue(values[i])

    def start_wave(self):
        """波アニメーション開始"""
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.wave_time = 0
            self.timer.start(50)

    def animate_wave(self):
        """波アニメーション更新"""
        self.wave_time += 0.1
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        for i, name in enumerate(finger_names):
            value = int(50 + 40 * np.sin(self.wave_time + i * 0.5))
            self.sliders[name].setValue(value)

    def update_hand(self):
        """手モデルを更新"""
        # 各指の屈曲角度を取得
        angles = np.zeros(20)

        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        for i, name in enumerate(finger_names):
            value = self.sliders[name].value() / 100.0  # 0-1に正規化
            # この指の4関節すべてに同じ値を設定
            base = i * 4
            angles[base:base+4] = value

        # スケール
        scale = self.scale_slider.value() / 10.0

        # 手モデル更新
        self.hand.update_from_angles(angles, scale)

    def run(self):
        self.window.show()
        return self.app.exec_()


if __name__ == '__main__':
    test = HandModelTest()
    sys.exit(test.run())
