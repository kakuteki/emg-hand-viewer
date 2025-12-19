"""
3D Hand Model Visualization
===========================

推論結果を表示するための3D手モデル
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# PyQt5 / PyQtGraph imports
HAS_PYQT = False
try:
    import pyqtgraph.opengl as gl
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt

    HAS_PYQT = True
except ImportError:
    pass


class HandSkeleton:
    """
    手の骨格構造定義

    座標系:
    - X軸: 手の横方向（親指側が負、小指側が正）
    - Y軸: 指の長さ方向（手首から指先へ正）
    - Z軸: 手のひらに垂直（手のひら側が負、手の甲側が正）
    """

    # 関節インデックス定義
    WRIST = 0

    # 親指 (Thumb) - 4関節
    THUMB_CMC = 1  # 手根中手関節
    THUMB_MCP = 2  # 中手指節関節
    THUMB_IP = 3  # 指節間関節
    THUMB_TIP = 4  # 指先

    # 人差し指 (Index) - 4関節
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    # 中指 (Middle) - 4関節
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    # 薬指 (Ring) - 4関節
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    # 小指 (Pinky) - 4関節
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    N_JOINTS = 21

    # 指ごとの関節リスト（根元から先端へ）
    FINGER_CHAINS = {
        "thumb": [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
        "index": [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
        "middle": [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
        "ring": [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
        "pinky": [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
    }

    @classmethod
    def get_rest_pose(cls) -> np.ndarray:
        """
        休止姿勢の関節位置を返す (21, 3)
        手のひらを上に向けた状態
        """
        positions = np.zeros((cls.N_JOINTS, 3))

        # 手首（原点）
        positions[cls.WRIST] = [0, 0, 0]

        # 親指（横に開いた位置）
        positions[cls.THUMB_CMC] = [-0.4, 0.3, 0.1]
        positions[cls.THUMB_MCP] = [-0.6, 0.5, 0.15]
        positions[cls.THUMB_IP] = [-0.75, 0.7, 0.15]
        positions[cls.THUMB_TIP] = [-0.85, 0.85, 0.15]

        # 人差し指
        positions[cls.INDEX_MCP] = [-0.22, 0.9, 0]
        positions[cls.INDEX_PIP] = [-0.22, 1.25, 0]
        positions[cls.INDEX_DIP] = [-0.22, 1.5, 0]
        positions[cls.INDEX_TIP] = [-0.22, 1.7, 0]

        # 中指（最長）
        positions[cls.MIDDLE_MCP] = [0, 0.95, 0]
        positions[cls.MIDDLE_PIP] = [0, 1.35, 0]
        positions[cls.MIDDLE_DIP] = [0, 1.65, 0]
        positions[cls.MIDDLE_TIP] = [0, 1.85, 0]

        # 薬指
        positions[cls.RING_MCP] = [0.22, 0.9, 0]
        positions[cls.RING_PIP] = [0.22, 1.25, 0]
        positions[cls.RING_DIP] = [0.22, 1.5, 0]
        positions[cls.RING_TIP] = [0.22, 1.7, 0]

        # 小指
        positions[cls.PINKY_MCP] = [0.44, 0.8, 0]
        positions[cls.PINKY_PIP] = [0.44, 1.05, 0]
        positions[cls.PINKY_DIP] = [0.44, 1.25, 0]
        positions[cls.PINKY_TIP] = [0.44, 1.4, 0]

        return positions

    @classmethod
    def get_bones(cls) -> List[Tuple[int, int]]:
        """骨（関節間の接続）のリストを返す"""
        bones = []

        # 手のひらの骨格
        bones.append((cls.WRIST, cls.THUMB_CMC))
        bones.append((cls.WRIST, cls.INDEX_MCP))
        bones.append((cls.WRIST, cls.MIDDLE_MCP))
        bones.append((cls.WRIST, cls.RING_MCP))
        bones.append((cls.WRIST, cls.PINKY_MCP))

        # 各指のMCP同士を接続（手のひら）
        bones.append((cls.INDEX_MCP, cls.MIDDLE_MCP))
        bones.append((cls.MIDDLE_MCP, cls.RING_MCP))
        bones.append((cls.RING_MCP, cls.PINKY_MCP))

        # 親指
        bones.append((cls.THUMB_CMC, cls.THUMB_MCP))
        bones.append((cls.THUMB_MCP, cls.THUMB_IP))
        bones.append((cls.THUMB_IP, cls.THUMB_TIP))

        # 人差し指
        bones.append((cls.INDEX_MCP, cls.INDEX_PIP))
        bones.append((cls.INDEX_PIP, cls.INDEX_DIP))
        bones.append((cls.INDEX_DIP, cls.INDEX_TIP))

        # 中指
        bones.append((cls.MIDDLE_MCP, cls.MIDDLE_PIP))
        bones.append((cls.MIDDLE_PIP, cls.MIDDLE_DIP))
        bones.append((cls.MIDDLE_DIP, cls.MIDDLE_TIP))

        # 薬指
        bones.append((cls.RING_MCP, cls.RING_PIP))
        bones.append((cls.RING_PIP, cls.RING_DIP))
        bones.append((cls.RING_DIP, cls.RING_TIP))

        # 小指
        bones.append((cls.PINKY_MCP, cls.PINKY_PIP))
        bones.append((cls.PINKY_PIP, cls.PINKY_DIP))
        bones.append((cls.PINKY_DIP, cls.PINKY_TIP))

        return bones

    @classmethod
    def get_parent(cls, joint_idx: int) -> Optional[int]:
        """指定関節の親関節を返す"""
        parent_map = {
            cls.THUMB_CMC: cls.WRIST,
            cls.THUMB_MCP: cls.THUMB_CMC,
            cls.THUMB_IP: cls.THUMB_MCP,
            cls.THUMB_TIP: cls.THUMB_IP,
            cls.INDEX_MCP: cls.WRIST,
            cls.INDEX_PIP: cls.INDEX_MCP,
            cls.INDEX_DIP: cls.INDEX_PIP,
            cls.INDEX_TIP: cls.INDEX_DIP,
            cls.MIDDLE_MCP: cls.WRIST,
            cls.MIDDLE_PIP: cls.MIDDLE_MCP,
            cls.MIDDLE_DIP: cls.MIDDLE_PIP,
            cls.MIDDLE_TIP: cls.MIDDLE_DIP,
            cls.RING_MCP: cls.WRIST,
            cls.RING_PIP: cls.RING_MCP,
            cls.RING_DIP: cls.RING_PIP,
            cls.RING_TIP: cls.RING_DIP,
            cls.PINKY_MCP: cls.WRIST,
            cls.PINKY_PIP: cls.PINKY_MCP,
            cls.PINKY_DIP: cls.PINKY_PIP,
            cls.PINKY_TIP: cls.PINKY_DIP,
        }
        return parent_map.get(joint_idx)


class HandModel3D:
    """
    3D手モデル可視化

    Parameters
    ----------
    gl_widget : GLViewWidget
        PyQtGraph OpenGLウィジェット
    scale : float
        モデルのスケール
    position : tuple
        モデルの位置 (x, y, z)
    color : tuple
        骨の色 (r, g, b, a)
    """

    def __init__(
        self,
        gl_widget,
        scale: float = 1.0,
        position: Tuple[float, float, float] = (0, 0, 0),
        color: Tuple[float, float, float, float] = (0.9, 0.7, 0.5, 1.0),
    ):
        if not HAS_PYQT:
            raise ImportError("PyQt5 and pyqtgraph are required")

        self.gl_widget = gl_widget
        self.scale = scale
        self.position = np.array(position)
        self.color = color

        # 休止姿勢
        self._rest_pose = HandSkeleton.get_rest_pose()

        # 現在の関節位置
        self._joint_positions = self._rest_pose.copy() * scale + self.position

        # 各関節の局所回転角度（ラジアン）
        self._joint_angles = np.zeros(HandSkeleton.N_JOINTS)

        # グラフィックスアイテム
        self._joint_scatter: Optional[gl.GLScatterPlotItem] = None
        self._bone_lines: List[gl.GLLinePlotItem] = []

        self._setup_graphics()

    def _setup_graphics(self):
        """グラフィックス要素の初期化"""
        # 関節を球として描画
        joint_colors = np.ones((HandSkeleton.N_JOINTS, 4))
        joint_colors[:, :3] = [1.0, 0.85, 0.7]  # 肌色

        self._joint_scatter = gl.GLScatterPlotItem(
            pos=self._joint_positions, size=12, color=joint_colors, pxMode=True
        )
        self.gl_widget.addItem(self._joint_scatter)

        # 骨を線として描画
        bones = HandSkeleton.get_bones()
        for start_idx, end_idx in bones:
            start = self._joint_positions[start_idx]
            end = self._joint_positions[end_idx]

            line = gl.GLLinePlotItem(
                pos=np.array([start, end]), color=self.color, width=4, antialias=True
            )
            self._bone_lines.append(line)
            self.gl_widget.addItem(line)

    def _rotation_matrix_x(self, angle: float) -> np.ndarray:
        """X軸周りの回転行列"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Y軸周りの回転行列"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Z軸周りの回転行列"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _compute_finger_positions(
        self, finger_chain: List[int], angles: List[float], is_thumb: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        指のForward Kinematicsを計算

        Parameters
        ----------
        finger_chain : list
            関節インデックスのリスト（根元から先端へ）
        angles : list
            各関節の屈曲角度（ラジアン）
        is_thumb : bool
            親指かどうか

        Returns
        -------
        dict
            関節インデックス -> 位置のマップ
        """
        positions = {}
        rest = self._rest_pose

        # 根元（手首）
        positions[finger_chain[0]] = rest[finger_chain[0]].copy()

        # 累積回転
        accumulated_rotation = np.eye(3)

        for i in range(1, len(finger_chain)):
            parent_idx = finger_chain[i - 1]
            current_idx = finger_chain[i]

            # 休止姿勢での骨ベクトル
            bone_vector = rest[current_idx] - rest[parent_idx]

            # この関節の屈曲角度
            if i - 1 < len(angles):
                angle = angles[i - 1]
            else:
                angle = 0

            # 回転軸を決定
            if is_thumb:
                # 親指は複雑な動きをする
                if i == 1:  # CMC関節
                    rotation = self._rotation_matrix_z(angle * 0.5)
                else:
                    rotation = self._rotation_matrix_x(-angle)
            else:
                # 他の指はX軸周りに屈曲（手のひら側へ曲がる）
                rotation = self._rotation_matrix_x(-angle)

            accumulated_rotation = accumulated_rotation @ rotation

            # 回転した骨ベクトル
            rotated_bone = accumulated_rotation @ bone_vector

            # 新しい位置
            positions[current_idx] = positions[parent_idx] + rotated_bone

        return positions

    def update_from_angles(self, angles: np.ndarray, angle_scale: float = 1.0):
        """
        関節角度から手のポーズを更新

        Parameters
        ----------
        angles : np.ndarray
            関節角度。値の範囲は0-1を想定（0=伸展、1=最大屈曲）
        angle_scale : float
            角度のスケーリング係数
        """
        # 新しい関節位置を計算
        new_positions = self._rest_pose.copy()

        # 最大屈曲角度（ラジアン）
        max_flex = np.pi / 2 * angle_scale  # 90度

        # 各指のFK計算
        finger_data = [
            ("thumb", [0, 1, 2, 3], True),  # indices 0-3
            ("index", [4, 5, 6, 7], False),  # indices 4-7
            ("middle", [8, 9, 10, 11], False),  # indices 8-11
            ("ring", [12, 13, 14, 15], False),  # indices 12-15
            ("pinky", [16, 17, 18, 19], False),  # indices 16-19
        ]

        for finger_name, angle_indices, is_thumb in finger_data:
            chain = HandSkeleton.FINGER_CHAINS[finger_name]

            # この指の角度を取得
            finger_angles = []
            for idx in angle_indices:
                if idx < len(angles):
                    # 0-1を0-max_flexにマッピング
                    finger_angles.append(angles[idx] * max_flex)
                else:
                    finger_angles.append(0)

            # FK計算
            finger_positions = self._compute_finger_positions(chain, finger_angles, is_thumb)

            # 位置を更新
            for joint_idx, pos in finger_positions.items():
                if joint_idx != HandSkeleton.WRIST:  # 手首は動かさない
                    new_positions[joint_idx] = pos

        # スケールと位置を適用
        self._joint_positions = new_positions * self.scale + self.position

        # グラフィックス更新
        self._update_graphics()

    def update_positions(self, positions: np.ndarray):
        """関節位置を直接更新"""
        n_joints = min(positions.shape[0], HandSkeleton.N_JOINTS)
        new_positions = self._rest_pose.copy()
        new_positions[:n_joints] = positions[:n_joints]

        self._joint_positions = new_positions * self.scale + self.position
        self._update_graphics()

    def _update_graphics(self):
        """グラフィックス要素を更新"""
        # 散布図更新
        self._joint_scatter.setData(pos=self._joint_positions)

        # 骨の更新
        bones = HandSkeleton.get_bones()
        for i, (start_idx, end_idx) in enumerate(bones):
            start = self._joint_positions[start_idx]
            end = self._joint_positions[end_idx]
            self._bone_lines[i].setData(pos=np.array([start, end]))

    def set_color(self, color: Tuple[float, float, float, float]):
        """骨の色を設定"""
        self.color = color
        for line in self._bone_lines:
            line.setData(color=color)

    def set_visible(self, visible: bool):
        """表示/非表示"""
        self._joint_scatter.setVisible(visible)
        for line in self._bone_lines:
            line.setVisible(visible)

    def remove(self):
        """グラフィックスアイテムを削除"""
        self.gl_widget.removeItem(self._joint_scatter)
        for line in self._bone_lines:
            self.gl_widget.removeItem(line)


class DualHandModel3D:
    """
    両手モデル（実測値と予測値の比較表示）
    """

    def __init__(self, gl_widget):
        self.gl_widget = gl_widget

        # 実測値（青系）- 左側
        self.ground_truth = HandModel3D(
            gl_widget, scale=1.0, position=(-2.5, 0, 0), color=(0.3, 0.5, 0.9, 1.0)
        )

        # 予測値（緑系）- 右側
        self.prediction = HandModel3D(
            gl_widget, scale=1.0, position=(2.5, 0, 0), color=(0.3, 0.9, 0.5, 1.0)
        )

    def update(
        self,
        ground_truth_angles: Optional[np.ndarray] = None,
        prediction_angles: Optional[np.ndarray] = None,
        angle_scale: float = 1.0,
    ):
        """両手を更新"""
        if ground_truth_angles is not None:
            self.ground_truth.update_from_angles(ground_truth_angles, angle_scale)

        if prediction_angles is not None:
            self.prediction.update_from_angles(prediction_angles, angle_scale)

    def set_ground_truth_visible(self, visible: bool):
        """実測値の表示/非表示"""
        self.ground_truth.set_visible(visible)

    def set_prediction_visible(self, visible: bool):
        """予測値の表示/非表示"""
        self.prediction.set_visible(visible)
