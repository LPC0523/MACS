"""Overview / Dashboard page."""

# ─────────────────────────────────────────────────────────────────────────
# LEGACY / NOT USED BY CURRENT QT APP
#
# 未加入 main_window 的页面栈，程序启动后不会被实例化。
#
# 替代模块：widgets/shim_widget.py（一键匀场）、widgets/selfcheck_widget.py（系统自检）
#
# 保留原因：路径与 import 兼容，且其中的实现细节仍有参考价值。修改这里不会影响
# 运行中的程序——先确认调用方是谁再动手。
# ─────────────────────────────────────────────────────────────────────────

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QGridLayout,
)

from core.theme import palette

_STEPS = [
    ("1", "B0 图处理", "DICOM → NIfTI\n相位解缠 + BET"),
    ("2", "一键匀场", "加载 B0 与通道场图\nMSE 优化求解"),
    ("3", "Hybrid Shim", "二阶 SH + PFC 线圈\n联合优化，颞叶保护"),
    ("4", "结果分析", "Before / After\n直方图与导出"),
    ("5", "Teensy 上传", "串口连接\n64 通道电流控制"),
]


class OverviewWidget(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.p = palette(False)
        self._step_labels = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 34, 40, 30)
        layout.setSpacing(6)

        self.hero = QLabel("MACS")
        self.hero.setStyleSheet("font-size: 40px; font-weight: 600; letter-spacing: -1px;")
        layout.addWidget(self.hero)

        desc = QLabel("多通道主动匀场平台 · Multi-channel Active Coil Shimming")
        desc.setObjectName("pageSubtitle")
        layout.addWidget(desc)
        layout.addSpacing(26)

        flow = QLabel("工作流程")
        flow.setObjectName("sectionTitle")
        layout.addWidget(flow)
        layout.addSpacing(8)

        grid = QGridLayout()
        grid.setSpacing(12)
        for i, (num, title, detail) in enumerate(_STEPS):
            grid.addWidget(self._make_step(num, title, detail), 0, i)
        layout.addLayout(grid)

        layout.addSpacing(24)
        tip = QLabel("提示 · 页面之间通过共享状态传递数据：完成「一键匀场」或「Hybrid Shim」"
                     "后，电流会自动出现在 Teensy 上传页。")
        tip.setWordWrap(True)
        tip.setObjectName("hint")
        layout.addWidget(tip)

        layout.addStretch()

    def _make_step(self, num, title, detail):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setMinimumHeight(112)
        v = QVBoxLayout(frame)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(6)

        n = QLabel(num)
        v.addWidget(n)
        t = QLabel(title)
        v.addWidget(t)
        d = QLabel(detail)
        d.setObjectName("hint")
        d.setWordWrap(True)
        v.addWidget(d)
        v.addStretch()

        self._step_labels.append((n, t))
        self._style_step(n, t, self.p)
        return frame

    @staticmethod
    def _style_step(n, t, p):
        n.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {p['accent']};")
        t.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {p['text']};")

    def apply_theme(self, dark: bool, p: dict):
        self.p = p
        for n, t in self._step_labels:
            self._style_step(n, t, p)
