"""手の骨格と順運動学の試験（描画なし）"""

import numpy as np
import pytest

from emg_realtime_viz.viz.hand_model import (
    FINGER_ANGLE_SLOTS,
    HandSkeleton,
    bone_angles,
    forward_kinematics,
)

FINGERS = {name: slots for name, slots, _ in FINGER_ANGLE_SLOTS}


def test_休止姿勢は21点の3次元座標():
    rest = HandSkeleton.get_rest_pose()
    assert rest.shape == (HandSkeleton.N_JOINTS, 3)
    assert rest[HandSkeleton.WRIST] == pytest.approx(np.zeros(3))


def test_中指が一番長く小指が一番短い():
    rest = HandSkeleton.get_rest_pose()
    lengths = {
        name: np.linalg.norm(rest[HandSkeleton.FINGER_CHAINS[name][-1]])
        for name in ("index", "middle", "ring", "pinky")
    }
    assert lengths["middle"] == max(lengths.values())
    assert lengths["pinky"] == min(lengths.values())


def test_骨は存在する関節だけをつなぐ():
    bones = HandSkeleton.get_bones()
    # 手首から5本 + 手のひら3本 + 各指3本ずつ
    assert len(bones) == 5 + 3 + 3 * 5
    for start, end in bones:
        assert 0 <= start < HandSkeleton.N_JOINTS
        assert 0 <= end < HandSkeleton.N_JOINTS
        assert start != end


def test_親子関係と指の連なりが一致する():
    for name, chain in HandSkeleton.FINGER_CHAINS.items():
        assert len(chain) == 5, name
        for parent, child in zip(chain, chain[1:]):
            assert HandSkeleton.get_parent(child) == parent


def test_角度が0なら休止姿勢のまま():
    positions = forward_kinematics(np.zeros(20))
    assert positions == pytest.approx(HandSkeleton.get_rest_pose())


def test_曲げても指の付け根は動かない():
    """4本指の付け根（MCP）は手のひらの一部なので固定される"""
    rest = HandSkeleton.get_rest_pose()
    positions = forward_kinematics(np.ones(20))

    for name in ("index", "middle", "ring", "pinky"):
        mcp = HandSkeleton.FINGER_CHAINS[name][1]
        assert positions[mcp] == pytest.approx(rest[mcp]), name


def test_手首は常に原点():
    positions = forward_kinematics(np.ones(20))
    assert positions[HandSkeleton.WRIST] == pytest.approx(np.zeros(3))


def test_握ると指先が手のひら側へ来る():
    rest = HandSkeleton.get_rest_pose()
    fist = forward_kinematics(np.ones(20))

    for name in ("index", "middle", "ring", "pinky"):
        tip = HandSkeleton.FINGER_CHAINS[name][-1]
        # 指先が手首に近づく
        assert np.linalg.norm(fist[tip]) < np.linalg.norm(rest[tip]), name
        # 手のひら側（Zの負方向）へ回り込む
        assert fist[tip][2] < rest[tip][2], name


def test_指は独立して動く():
    angles = np.zeros(20)
    angles[FINGERS["index"]] = 1.0
    positions = forward_kinematics(angles)
    rest = HandSkeleton.get_rest_pose()

    index_tip = HandSkeleton.FINGER_CHAINS["index"][-1]
    middle_tip = HandSkeleton.FINGER_CHAINS["middle"][-1]

    assert not np.allclose(positions[index_tip], rest[index_tip])
    assert positions[middle_tip] == pytest.approx(rest[middle_tip])


def test_骨の長さは曲げても変わらない():
    rest = HandSkeleton.get_rest_pose()
    fist = forward_kinematics(np.ones(20))

    for start, end in HandSkeleton.get_bones():
        rest_length = np.linalg.norm(rest[end] - rest[start])
        fist_length = np.linalg.norm(fist[end] - fist[start])
        # 手のひら側の骨（MCP同士のつなぎ）は固定なので全て一致するはず
        assert fist_length == pytest.approx(rest_length, abs=1e-9), (start, end)


def test_角度スケールが効く():
    half = forward_kinematics(np.ones(20), angle_scale=0.5)
    full = forward_kinematics(np.ones(20), angle_scale=1.0)
    rest = HandSkeleton.get_rest_pose()

    tip = HandSkeleton.FINGER_CHAINS["index"][-1]
    assert np.linalg.norm(rest[tip]) > np.linalg.norm(half[tip]) > np.linalg.norm(full[tip])


def test_NaNや無限大が座標に漏れない():
    """外部APIから直接おかしな値を渡されても、描画が黙って壊れないこと"""
    for bad in (np.nan, np.inf, -np.inf):
        positions = forward_kinematics(np.full(20, bad))
        assert np.all(np.isfinite(positions)), bad


def test_角度が20個未満でも落ちない():
    positions = forward_kinematics(np.ones(8))
    assert positions.shape == (HandSkeleton.N_JOINTS, 3)


def test_骨への角度の割り当て():
    # 親指は手首からの4本すべてが回る
    assert bone_angles([1.0, 2.0, 3.0, 4.0], is_thumb=True) == [1.0, 2.0, 3.0, 4.0]
    # 他の指は手のひらの骨を回さず、残り3本にMCP/PIP/DIPを割り当てる
    assert bone_angles([1.0, 2.0, 3.0, 4.0], is_thumb=False) == [0.0, 1.0, 2.0, 3.0]
