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

QT_LIB = "PyQt5"


def ensure() -> str:
    """
    pyqtgraphが使うQtを指定する（何度呼んでもよい）

    pyqtgraphをimportする前に呼ぶこと。importの並べ替えで順序が
    崩れないよう、importではなく関数呼び出しの形にしてある。

    Returns
    -------
    str
        実際に指定されているQtの名前
    """
    os.environ.setdefault("PYQTGRAPH_QT_LIB", QT_LIB)
    return os.environ["PYQTGRAPH_QT_LIB"]


ensure()
