"""Main Window with left navigation, theme toggle, language switch."""
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QStackedWidget, QLabel, QListWidgetItem,
    QPushButton, QScrollArea,
)
from PySide6.QtCore import Qt, QSize, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPixmap, QImage

from widgets.b0_widget import B0Widget
from widgets.shim_widget import ShimWidget
from widgets.system_shim_widget import SystemShimWidget
from widgets.results_widget import ResultsWidget
from widgets.teensy_widget import TeensyWidget
from widgets.selfcheck_widget import SelfCheckWidget
from widgets.linear_encoding_widget import LinearEncodingWidget
from core.theme import stylesheet, palette
from core.version import version_string


# ── Nav page definitions (zh, en) ──
# Two pages carry the workflow; the rest are kept for the advanced paths that
# do not fit "drop a folder, press once".
# Order matches the workflow, and the two shim pages sit together because the
# choice between them is the first one a user makes:
#   一键匀场  drop a folder, press once -- the app decides everything
#   系统匀场  tell it the scanner's current shim, it says what to change
_NAV = [
    ("一键匀场",     "One-Click Shim"),
    ("系统匀场",     "System Shim"),
    ("Teensy 上传",  "Teensy Upload"),
    ("结果分析",     "Results"),
    ("B0 图处理",    "B0 Map"),
    ("线性编码仿真", "Linear Encoding Simulation"),
    ("系统自检",     "Self-check"),
]



#: How long to give a worker thread to finish after being asked to stop, in ms.
#: Long enough for a solver iteration or a serial read to return, short enough
#: that closing the window still feels immediate.
_THREAD_STOP_MS = 3000

#: Threads that would not stop in time, kept alive and unparented so neither
#: Qt nor Python destroys one while it is still running.
_ORPHANED_THREADS = []

class MainWindow(QMainWindow):
    # Signal emitted when theme or language changes — pages can connect to refresh
    theme_changed = Signal(bool)   # True = dark
    lang_changed = Signal(str)     # "zh" or "en"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f"MACS {version_string(short=True)} — 多通道主动匀场平台")
        self.setMinimumSize(1200, 750)
        self.resize(1440, 900)

        self.dark_mode = False   # Apple-style light appearance by default
        self.lang = "zh"

        # Shared application state
        self.app_state = {
            "b0_data": None,
            "b0_vol3d": None,
            "b0_affine": None,
            "b0_label": "",
            "optimal_currents": None,
            "ch_names_active": [],
            "teensy_connected": False,
        }

        # Change this one path when replacing the static application logo.
        self._logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "resources", "logo.jpg")

        self._build_ui()
        self._apply_theme()

    # ══════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left nav panel ──
        nav_panel = QWidget()
        nav_panel.setObjectName("navPanel")
        nav_panel.setFixedWidth(230)
        nav = QVBoxLayout(nav_panel)
        nav.setContentsMargins(14, 14, 14, 14)
        nav.setSpacing(4)

        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.logo_label)
        self._refresh_logo()

        # Title
        self.nav_title = QLabel(f"MACS {version_string(short=True)}")
        self.nav_title.setObjectName("navTitle")
        self.nav_title.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.nav_title)
        nav.addSpacing(4)

        # Theme + Language buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_theme = QPushButton("☀ Light")
        self.btn_theme.setObjectName("ghostBtn")
        self.btn_theme.setFixedHeight(30)
        self.btn_theme.clicked.connect(self._toggle_theme)
        self.btn_lang = QPushButton("EN")
        self.btn_lang.setObjectName("ghostBtn")
        self.btn_lang.setFixedHeight(30)
        self.btn_lang.clicked.connect(self._toggle_lang)
        btn_row.addWidget(self.btn_theme)
        btn_row.addWidget(self.btn_lang)
        nav.addLayout(btn_row)
        nav.addSpacing(10)

        # Navigation list
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        self.nav_list.setMaximumHeight(len(_NAV) * 40)
        self._fill_nav_list()
        self.nav_list.setCurrentRow(0)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        nav.addWidget(self.nav_list)

        nav.addStretch()

        # Status
        self.status_label = QLabel()
        self.status_label.setObjectName("navStatus")
        self.status_label.setWordWrap(True)
        nav.addWidget(self.status_label)
        self._update_status()

        # Footer
        footer = QLabel("MRI System Lab")
        footer.setObjectName("navFooter")
        footer.setAlignment(Qt.AlignCenter)
        nav.addWidget(footer)

        root.addWidget(nav_panel)

        # ── Page stack (inside a scroll area so content never clips) ──
        self.stack = QStackedWidget()
        self.stack.setObjectName("pageStack")

        self.page_shim = ShimWidget(self.app_state)
        self.page_teensy = TeensyWidget(self.app_state)
        self.page_system = SystemShimWidget(self.app_state)
        self.page_b0 = B0Widget(self.app_state)
        self.page_results = ResultsWidget(self.app_state)
        self.page_selfcheck = SelfCheckWidget(self.app_state)
        # Research page. Offline only -- it owns no thread and no port,
        # and nothing on it can reach the production shim path.
        self.page_encoding = LinearEncodingWidget(self.app_state)

        # Pages ask to navigate rather than reaching into the stack, so a page
        # never needs a reference to the window that contains it.
        for page in (self.page_shim, self.page_system):
            if hasattr(page, "request_page"):
                page.request_page.connect(self.goto_page)

        # Must match _NAV order -- the nav row index is the stack index.
        for p in [self.page_shim, self.page_system, self.page_teensy,
                  self.page_results, self.page_b0, self.page_encoding,
                  self.page_selfcheck]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(p)
            scroll.setFrameShape(QScrollArea.NoFrame)
            self.stack.addWidget(scroll)

        root.addWidget(self.stack)
        self.statusBar().showMessage("就绪")

        # Cheap poll rather than threading a callback through every page: the
        # pages write to a plain dict, and there is no signal to subscribe to.
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1500)

    # ══════════════════════════════════════════════════════════════════
    #  Navigation
    # ══════════════════════════════════════════════════════════════════
    def _fill_nav_list(self):
        self.nav_list.clear()
        idx = 0 if self.lang == "zh" else 1
        for pair in _NAV:
            item = QListWidgetItem(pair[idx])
            item.setSizeHint(QSize(0, 38))
            self.nav_list.addItem(item)

    def _on_nav_changed(self, index):
        self.stack.setCurrentIndex(index)
        # The panel reports shared state that other pages change while this one
        # is hidden. Refreshing on every switch keeps it from showing what was
        # true when the window opened.
        self._update_status()

    @Slot(int)
    def goto_page(self, index: int):
        """Switch pages, keeping the nav highlight in step."""
        if 0 <= index < self.stack.count():
            self.nav_list.setCurrentRow(index)      # this drives the stack
            self.stack.setCurrentIndex(index)

    def _update_status(self):
        s = self.app_state
        if self.lang == "zh":
            lines = [
                f"● B0 图: {'已加载' if s.get('b0_data') is not None else '未加载'}",
                f"● 优化电流: {'已生成' if s.get('optimal_currents') is not None else '未生成'}",
                f"● Teensy: {'已连接' if s.get('teensy_connected') else '未连接'}",
            ]
        else:
            lines = [
                f"● B0: {'loaded' if s.get('b0_data') is not None else 'not loaded'}",
                f"● Currents: {'ready' if s.get('optimal_currents') is not None else 'pending'}",
                f"● Teensy: {'connected' if s.get('teensy_connected') else 'disconnected'}",
            ]
        self.status_label.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    #  Theme / Language toggles
    # ══════════════════════════════════════════════════════════════════
    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()
        self._refresh_logo()
        self.theme_changed.emit(self.dark_mode)

    def _toggle_lang(self):
        self.lang = "en" if self.lang == "zh" else "zh"
        cur_row = self.nav_list.currentRow()
        self._fill_nav_list()
        self.nav_list.setCurrentRow(cur_row)
        self._update_status()
        self.btn_lang.setText("中文" if self.lang == "en" else "EN")
        self.lang_changed.emit(self.lang)

    def _refresh_logo(self):
        if not os.path.exists(self._logo_path):
            return
        img = QImage(self._logo_path)
        pixmap = QPixmap.fromImage(img).scaledToWidth(195, Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)

    # ══════════════════════════════════════════════════════════════════
    #  Theme stylesheet
    # ══════════════════════════════════════════════════════════════════
    def _apply_theme(self):
        """Apply the Apple-style stylesheet and let every page restyle itself."""
        self.btn_theme.setText("浅色" if self.dark_mode else "深色")
        self.setStyleSheet(stylesheet(self.dark_mode))

        p = palette(self.dark_mode)
        for page in [self.page_shim, self.page_teensy, self.page_system,
                     self.page_b0, self.page_results]:
            fn = getattr(page, "apply_theme", None)
            if fn:
                fn(self.dark_mode, p)

    def closeEvent(self, event):
        """Stop every background thread before Qt starts deleting things.

        Without this the process aborted on exit with 0xC0000409 and Qt printed
        "QThread: Destroyed while thread '' is still running". Qt delivers
        closeEvent only to top-level windows, so `TeensyWidget.closeEvent` --
        which stops the serial reader and the port-refresh timer -- never ran:
        the page is a child of a QStackedWidget, not a window. Every worker
        (self-check, B0 pipeline, shim, system shim, serial reader) was left
        running while its QThread was destroyed underneath it.

        Pages that own something stoppable get asked first, via `shutdown()`.
        The sweep afterwards is the backstop, and catches a worker started from
        a page that has no `shutdown()` of its own -- closing mid-run used to
        crash the same way.
        """
        for page in (self.page_teensy, self.page_selfcheck, self.page_b0,
                     self.page_shim, self.page_system, self.page_results,
                     self.page_encoding):
            stop = getattr(page, "shutdown", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    # A page failing to shut down cleanly must not stop the
                    # window from closing, and must not skip the others.
                    pass

        stubborn = []
        for th in self.findChildren(QThread):
            if not th.isRunning():
                continue
            th.requestInterruption()
            th.quit()
            if not th.wait(_THREAD_STOP_MS):
                stubborn.append(th)

        # `terminate()` is deliberately not used. It kills a thread wherever it
        # happens to be, and these threads run Python -- terminating one that
        # holds the GIL hangs the process outright, which is what happened when
        # the window was closed while the startup self-check was still going.
        # Detaching instead lets Qt tear down without deleting a live QThread;
        # the reference keeps Python from collecting it. The process is exiting
        # anyway, so a worker that will not stop simply goes with it.
        for th in stubborn:
            th.setParent(None)
            _ORPHANED_THREADS.append(th)

        super().closeEvent(event)
