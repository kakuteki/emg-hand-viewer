"""
Qt Compatibility
================

pyqtgraphが掴むQtを、このライブラリが使うQtにそろえる

pyqtgraphは環境に入っているQtを自分で選ぶ。PySide6が入っていると
そちらを選ぶが、このライブラリはPyQt5を直接importするため、
1つのプロセスに2種類のQtが載って次のように即クラッシュする。

    QWidget: Must construct a QApplication before a QWidget

pyqtgraphを最初にimportする前に環境変数を立てておけば防げるので、
このモジュールをパッケージの入口で最初に読み込む。
すでに利用者が指定している場合はそちらを尊重する。
"""

import os
import warnings

QT_LIB = "PyQt5"
ENV_NAME = "PYQTGRAPH_QT_LIB"


def ensure() -> str:
    """
    pyqtgraphが使うQtを指定する（何度呼んでもよい）

    pyqtgraphをimportする前に呼ぶこと。importの並べ替えで順序が
    崩れないよう、importではなく関数呼び出しの形にしてある。

    このライブラリはPyQt5を直接importするので、環境変数で別のQtを
    指定されるとやはり2種類のQtが載ってしまう。黙って落ちないよう
    警告を出す。

    Returns
    -------
    str
        実際に指定されているQtの名前
    """
    current = os.environ.get(ENV_NAME)

    if current is None:
        os.environ[ENV_NAME] = QT_LIB
        return QT_LIB

    if current != QT_LIB:
        warnings.warn(
            f"{ENV_NAME}={current} が指定されていますが、"
            f"このライブラリは{QT_LIB}を使います。"
            f"Qtが2種類読み込まれて異常終了することがあります。",
            RuntimeWarning,
            stacklevel=2,
        )

    return current


ensure()
