"""Teensy Serial Control Widget — 64-channel current upload & monitoring.

Full Qt reimplementation of _page_teensy.py with proper QThread serial I/O.
"""
import re
import struct
import time
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit,
    QGroupBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QHeaderView, QSplitter,
    QCheckBox, QApplication,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QColor

import serial
from serial.tools import list_ports

from workers.serial_worker import SerialReaderWorker, clamp_current, CURRENT_LIMIT


from core.theme import palette


def _status(label, kind):
    """Tag a status label so the global stylesheet colours it."""
    label.setObjectName({"ok": "statusOk", "err": "statusErr",
                         "warn": "statusWarn"}.get(kind, "hint"))
    label.style().unpolish(label)
    label.style().polish(label)


class TeensyWidget(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.p = palette(False)
        self.ser: Optional[serial.Serial] = None
        self.reader: Optional[SerialReaderWorker] = None
        #: Set while a current table is in flight, cleared on reply or timeout.
        self._table_expect = None

        # Channel data: {index: {current, gain, ok}}
        self.ch_data: dict = {}
        self.set_matrix = [[0.0] * 8 for _ in range(8)]
        self.active_boards = list(range(8))
        self.ch_mapping: dict = {}  # {ch_name: (B, C)}

        # Current plan
        self.plan_values: list = []
        self.plan_index: int = 0

        # Log buffer
        self.log_lines: list = []

        self._build_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._auto_refresh_channels)

    # ══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # Title
        title = QLabel("Teensy 串口控制")
        title.setObjectName("pageTitle")
        main_layout.addWidget(title)

        # ── 1. Connection Group ──
        conn_group = QGroupBox("串口连接")
        conn_layout = QVBoxLayout(conn_group)

        row1 = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["115200", "230400", "57600", "9600"])
        row1.addWidget(QLabel("端口:"))
        row1.addWidget(self.port_combo)
        row1.addWidget(QLabel("波特率:"))
        row1.addWidget(self.baud_combo)
        row1.addStretch()
        conn_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_connect = QPushButton("连接")
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_disconnect.setEnabled(False)
        self.btn_refresh_ports = QPushButton("刷新端口")
        self.btn_refresh_ports.clicked.connect(self._refresh_ports)
        self.conn_status = QLabel("● 未连接")
        _status(self.conn_status, "err")
        row2.addWidget(self.btn_connect)
        row2.addWidget(self.btn_disconnect)
        row2.addWidget(self.btn_refresh_ports)
        row2.addStretch()
        row2.addWidget(self.conn_status)
        conn_layout.addLayout(row2)

        main_layout.addWidget(conn_group)

        # ── 2. Quick Commands ──
        cmd_group = QGroupBox("快捷指令")
        cmd_layout = QHBoxLayout(cmd_group)
        for label, cmd in [("初始化 E", "E"), ("校准 D", "D"),
                           ("电流归零 Z", "Z"), ("读全部电流", "Icurrentall"),
                           ("显示电流 I", "I")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, c=cmd: self._send(c))
            cmd_layout.addWidget(btn)
        main_layout.addWidget(cmd_group)

        # ── 3. Upload Optimized Currents ──
        upload_group = QGroupBox("上传匀场电流")
        upload_layout = QVBoxLayout(upload_group)

        # Announces a handoff from the shim page. Hidden until one arrives, so
        # the page looks no different when it is used on its own.
        self.handoff_label = QLabel("")
        self.handoff_label.setObjectName("statusOk")
        self.handoff_label.setWordWrap(True)
        self.handoff_label.setVisible(False)
        upload_layout.addWidget(self.handoff_label)

        self.upload_table = QTableWidget(0, 3)
        self.upload_table.setHorizontalHeaderLabels(["通道", "B·C", "电流 (A)"])
        self.upload_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.upload_table.setMaximumHeight(200)
        upload_layout.addWidget(self.upload_table)

        btn_row = QHBoxLayout()
        self.btn_table = QPushButton("上传电流表")
        self.btn_table.setObjectName("primaryBtn")
        self.btn_table.setVisible(False)
        self.btn_table.clicked.connect(self._upload_table)
        btn_row.addWidget(self.btn_table)

        self.btn_step = QPushButton("手动步进")
        self.btn_step.setVisible(False)
        self.btn_step.setToolTip(
            "模拟一次扫描仪 trigger，用于 trigger 线还没接通时离线验证逐层切换")
        self.btn_step.clicked.connect(lambda: self._send("STEP"))
        btn_row.addWidget(self.btn_step)

        self.btn_upload = QPushButton("上传电流")
        self.btn_upload.setObjectName("primaryBtn")
        self.btn_upload.clicked.connect(self._upload_currents)
        self.btn_zero_upload = QPushButton("归零后上传")
        self.btn_zero_upload.clicked.connect(self._zero_then_upload)
        btn_row.addWidget(self.btn_upload)
        btn_row.addWidget(self.btn_zero_upload)
        btn_row.addStretch()
        upload_layout.addLayout(btn_row)

        self.verify_label = QLabel("")
        upload_layout.addWidget(self.verify_label)

        main_layout.addWidget(upload_group)

        # ── 4. Manual Command ──
        manual_group = QGroupBox("手动发送")
        manual_layout = QHBoxLayout(manual_group)
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("例：E / Z / Track,0,0,0.5 / Icurrentall")
        self.manual_input.returnPressed.connect(self._send_manual)
        btn_send = QPushButton("发送")
        btn_send.clicked.connect(self._send_manual)
        manual_layout.addWidget(self.manual_input)
        manual_layout.addWidget(btn_send)
        main_layout.addWidget(manual_group)

        # ── 5. Current Plan ──
        plan_group = QGroupBox("电流计划（每64值一组）")
        plan_layout = QVBoxLayout(plan_group)
        plan_row1 = QHBoxLayout()
        self.btn_load_plan = QPushButton("加载 TXT 文件")
        self.btn_load_plan.clicked.connect(self._load_plan_file)
        self.plan_status = QLabel("未加载")
        plan_row1.addWidget(self.btn_load_plan)
        plan_row1.addWidget(self.plan_status)
        plan_row1.addStretch()
        plan_layout.addLayout(plan_row1)

        plan_row2 = QHBoxLayout()
        self.btn_prev_group = QPushButton("上一组")
        self.btn_prev_group.clicked.connect(self._prev_group)
        self.btn_next_group = QPushButton("下一组")
        self.btn_next_group.clicked.connect(self._next_group)
        self.btn_reset_plan = QPushButton("重置归零")
        self.btn_reset_plan.clicked.connect(self._reset_plan)
        self.plan_idx_label = QLabel("0 / 0")
        plan_row2.addWidget(self.btn_prev_group)
        plan_row2.addWidget(self.btn_next_group)
        plan_row2.addWidget(self.btn_reset_plan)
        plan_row2.addWidget(self.plan_idx_label)
        plan_row2.addStretch()
        plan_layout.addLayout(plan_row2)
        main_layout.addWidget(plan_group)

        # ── 6. 64-Channel Monitor ──
        monitor_group = QGroupBox("64 通道监控")
        monitor_layout = QVBoxLayout(monitor_group)

        mon_btn_row = QHBoxLayout()
        btn_mon_refresh = QPushButton("手动刷新")
        btn_mon_refresh.clicked.connect(self._manual_refresh_channels)
        self.btn_auto_refresh = QCheckBox("自动刷新 (2s)")
        self.btn_auto_refresh.toggled.connect(self._toggle_auto_refresh)
        mon_btn_row.addWidget(btn_mon_refresh)
        mon_btn_row.addWidget(self.btn_auto_refresh)
        mon_btn_row.addStretch()
        monitor_layout.addLayout(mon_btn_row)

        self.monitor_table = QTableWidget(8, 8)
        self.monitor_table.setHorizontalHeaderLabels([f"CH{c}" for c in range(8)])
        self.monitor_table.setVerticalHeaderLabels([f"B{r}" for r in range(8)])
        self.monitor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.monitor_table.setMaximumHeight(300)
        monitor_layout.addWidget(self.monitor_table)

        # Single channel set
        single_row = QHBoxLayout()
        single_row.addWidget(QLabel("Board:"))
        self.spin_board = QSpinBox()
        self.spin_board.setRange(0, 7)
        single_row.addWidget(self.spin_board)
        single_row.addWidget(QLabel("Ch:"))
        self.spin_ch = QSpinBox()
        self.spin_ch.setRange(0, 7)
        single_row.addWidget(self.spin_ch)
        single_row.addWidget(QLabel("电流(A):"))
        self.spin_current = QDoubleSpinBox()
        self.spin_current.setRange(-CURRENT_LIMIT, CURRENT_LIMIT)
        self.spin_current.setDecimals(4)
        self.spin_current.setSingleStep(0.001)
        single_row.addWidget(self.spin_current)
        btn_single = QPushButton("发送")
        btn_single.clicked.connect(self._send_single_channel)
        single_row.addWidget(btn_single)
        single_row.addStretch()
        monitor_layout.addLayout(single_row)

        main_layout.addWidget(monitor_group)

        # ── 7. Serial Log ──
        log_group = QGroupBox("串口日志")
        log_layout = QVBoxLayout(log_group)
        log_btn_row = QHBoxLayout()
        btn_clear_log = QPushButton("清空")
        btn_clear_log.clicked.connect(self._clear_log)
        log_btn_row.addWidget(btn_clear_log)
        log_btn_row.addStretch()
        log_layout.addLayout(log_btn_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

        # Initial port scan
        self._refresh_ports()
        self._update_upload_table()

    # ══════════════════════════════════════════════════════════════════════════
    # Serial Connection
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_ports(self):
        self.port_combo.clear()
        ports = list_ports.comports()
        teensy_port = None
        for p in ports:
            self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
            if "Teensy" in (p.description or "") or (p.vid == 0x16C0 and p.pid == 0x0483):
                teensy_port = p.device

        if teensy_port:
            idx = next((i for i in range(self.port_combo.count())
                        if self.port_combo.itemData(i) == teensy_port), 0)
            self.port_combo.setCurrentIndex(idx)
            self._log(f"[自动检测] Teensy: {teensy_port}")

    def _connect(self):
        port = self.port_combo.currentData()
        baud = int(self.baud_combo.currentText())
        if not port:
            QMessageBox.warning(self, "错误", "请选择一个串口")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self._start_reader()
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.conn_status.setText(f"● 已连接 {port} @ {baud}")
            _status(self.conn_status, "ok")
            self.app_state["teensy_connected"] = True
            self._log(f"[OK] 已连接 {port} @ {baud}")
        except Exception as e:
            QMessageBox.critical(self, "连接失败", str(e))

    def _disconnect(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.conn_status.setText("● 未连接")
        _status(self.conn_status, "err")
        self.app_state["teensy_connected"] = False
        self.refresh_timer.stop()
        self.btn_auto_refresh.setChecked(False)
        self._log("已断开")

    def _start_reader(self):
        if self.reader and self.reader.isRunning():
            self.reader.stop()
        self.reader = SerialReaderWorker(self.ser, self)
        self.reader.log_received.connect(self._on_log)
        self.reader.channel_updated.connect(self._on_channel_update)
        self.reader.batch_complete.connect(self._on_batch_complete)
        self.reader.error_occurred.connect(self._on_reader_error)
        self.reader.start()

    # ══════════════════════════════════════════════════════════════════════════
    # Signal Handlers (from SerialReaderWorker)
    # ══════════════════════════════════════════════════════════════════════════
    def _status(self, label, kind):
        """Tag a label so the global stylesheet colours it."""
        label.setObjectName({"ok": "statusOk", "err": "statusErr",
                             "warn": "statusWarn"}.get(kind, "hint"))
        label.style().unpolish(label)
        label.style().polish(label)

    @Slot(str)
    def _on_log(self, msg: str):
        self._log(msg)
        self._note_table_reply(msg)

    @Slot(int, int, float, float, bool)
    def _on_channel_update(self, board: int, ch: int, current: float, gain: float, ok: bool):
        idx = board * 8 + ch
        self.ch_data[idx] = {"current": current, "gain": gain, "ok": ok}
        # Update monitor table cell
        ok_mark = "✓" if ok else "!"
        text = f"{ok_mark} {current:.3f}A\nG={gain:.2f}"
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        if not ok:
            item.setForeground(QColor(self.p["danger"]))
        else:
            item.setForeground(QColor(self.p["success"]))
        self.monitor_table.setItem(board, ch, item)

    @Slot()
    def _on_batch_complete(self):
        self._detect_active_boards()

    @Slot(str)
    def _on_reader_error(self, msg: str):
        self._log(f"[读取错误] {msg}")
        self._disconnect()

    # ══════════════════════════════════════════════════════════════════════════
    # Send Commands
    # ══════════════════════════════════════════════════════════════════════════
    def _send(self, cmd: str) -> bool:
        if not self.ser or not self.ser.is_open:
            QMessageBox.warning(self, "未连接", "请先连接串口")
            return False
        full = cmd if cmd.endswith("\n") else cmd + "\n"
        self.ser.write(full.encode())
        self._log(f"→ {cmd.strip()}")
        return True

    def _send_manual(self):
        cmd = self.manual_input.text().strip()
        if cmd:
            self._send(cmd)
            self.manual_input.clear()

    def _send_single_channel(self):
        b = self.spin_board.value()
        c = self.spin_ch.value()
        val = clamp_current(self.spin_current.value())
        self.set_matrix[b][c] = val
        self._send(f"Track,{b},{c},{val:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Upload Optimized Currents
    # ══════════════════════════════════════════════════════════════════════════
    def showEvent(self, event):
        """Pick up anything the shim page handed over while we were hidden."""
        super().showEvent(event)
        try:
            from core import amp_bridge

            payload = amp_bridge.take(self.app_state)
        except Exception:
            payload = None
        if payload:
            n = len(payload.get("channel_values") or [])
            table = payload.get("slice_currents")
            if table:
                self.handoff_label.setText(
                    f"✓ 收到匀场结果：动态匀场 · {len(table)} 层 × {len(table[0])} 通道"
                    f"　连接串口后点「上传电流表」。")
                self._log(f"[匀场] 收到电流表 {len(table)} 层 × {len(table[0])} 通道")
            else:
                self.handoff_label.setText(
                    f"✓ 收到匀场结果，就绪：{payload.get('coil') or '线圈'} · "
                    f"{n} 通道　连接串口后点「上传电流」。")
                self._log(f"[匀场] 收到 {n} 个通道的电流（{payload.get('source')}）")
            self.handoff_label.setVisible(True)
        self._refresh_mode()
        self._update_upload_table()

    def _pending(self):
        from core import amp_bridge

        return amp_bridge.peek(self.app_state) or {}

    def _refresh_mode(self):
        """Show the table controls only when a table is actually waiting."""
        table = self._pending().get("slice_currents")
        dynamic = bool(table)
        self.btn_table.setVisible(dynamic)
        self.btn_step.setVisible(dynamic)
        self.btn_upload.setVisible(not dynamic)
        self.btn_zero_upload.setVisible(not dynamic)

    # ══════════════════════════════════════════════════════════════════
    # Dynamic current table
    # ══════════════════════════════════════════════════════════════════
    #: Frame layout, see docs/协议_电流表.md.
    CMD_LOAD_TABLE = 0x03
    TABLE_MAGIC = b"MACS"
    TABLE_PROTO_VERSION = 1

    #: How long to wait for the firmware to answer before calling the upload
    #: unconfirmed. The table is a few kB over 115200 baud, so a second is
    #: generous for a device that is going to reply at all.
    ACK_TIMEOUT_MS = 3000

    def _build_table_frame(self, table):
        """Header, payload and CRC as one bytes object.

        The original frame was a bare command byte plus raw floats. On a serial
        line with no framing, a device that resets mid-transfer, or a stale byte
        in the buffer, would leave the firmware reading current values as a
        length -- and it would have no way to notice. Magic, version, an
        explicit length and a CRC make a corrupted table detectable instead of
        silently loaded.
        """
        import struct
        import zlib

        n_layers, n_ch = len(table), len(table[0])
        body = bytearray()
        for row in table:
            for v in row:
                body += struct.pack("<f", float(v))

        header = (self.TABLE_MAGIC
                  + bytes([self.CMD_LOAD_TABLE, self.TABLE_PROTO_VERSION])
                  + struct.pack("<HH", n_layers, n_ch)
                  + struct.pack("<I", len(body)))
        crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF
        return bytes(header + body + struct.pack("<I", crc)), n_layers, n_ch

    def _upload_table(self):
        """Send the whole slice/channel table in one framed transfer.

        Sent as one frame rather than a row at a time because the table has to
        be complete before the first trigger arrives -- a partially loaded table
        would step into rows that were never written.

        Success here means "the bytes left this machine", which is not the same
        as "the amplifier is applying them". The recorded status says so.
        """
        payload = self._pending()
        table = payload.get("slice_currents")
        if not table:
            QMessageBox.information(self, "提示", "没有待上传的电流表")
            return
        if not self.ser or not self.ser.is_open:
            QMessageBox.warning(self, "未连接", "请先连接串口")
            return

        n_layers, n_ch = len(table), len(table[0])
        if any(len(row) != n_ch for row in table):
            QMessageBox.warning(self, "数据异常", "电流表各层通道数不一致，已取消。")
            return

        # Two limits, both enforced. The coil's configured limit is a property
        # of the array; CURRENT_LIMIT is what the hardware itself tolerates, and
        # the static path has always applied it. The table path checked only the
        # first, so an over-specified coil config could have sent the amplifier
        # more than it is rated for.
        coil_limit = payload.get("current_limit_a") or 0.0
        over_coil = (sum(1 for row in table for v in row
                         if abs(v) > coil_limit + 1e-6) if coil_limit else 0)
        over_hw = sum(1 for row in table for v in row
                      if abs(v) > CURRENT_LIMIT + 1e-6)
        if over_coil or over_hw:
            parts = []
            if over_coil:
                parts.append(f"{over_coil} 个值超过线圈上限 ±{coil_limit:g} A")
            if over_hw:
                parts.append(f"{over_hw} 个值超过硬件上限 ±{CURRENT_LIMIT:g} A")
            QMessageBox.warning(self, "超出电流上限",
                                "，".join(parts) + "。已取消上传。")
            self._log("[!] 电流表超出上限，未发送")
            return

        frame, n_layers, n_ch = self._build_table_frame(table)
        self._log(f"→ CMD_LOAD_TABLE  {n_layers} 层 × {n_ch} 通道  "
                  f"({len(frame)} 字节，含 CRC32)")

        # Arm the acknowledgement watch before writing, so a fast reply cannot
        # arrive before anything is listening for it.
        self._table_expect = {"layers": n_layers, "channels": n_ch,
                              "acked": False, "done": False}
        self.btn_table.setEnabled(False)
        try:
            step = 1024
            for i in range(0, len(frame), step):
                self.ser.write(frame[i:i + step])
                self.handoff_label.setText(
                    f"写入 {min(i + step, len(frame))}/{len(frame)} 字节 …")
                QApplication.processEvents()
            self.ser.flush()
        except Exception as e:
            self._log(f"[!] 上传失败：{type(e).__name__}: {e}")
            QMessageBox.critical(self, "上传失败", f"{type(e).__name__}: {e}")
            self._record_upload("failed", 0, error=f"{type(e).__name__}: {e}")
            self.btn_table.setEnabled(True)
            self._table_expect = None
            return
        finally:
            pass

        self.handoff_label.setText(
            f"已发送 {n_layers} 层 × {n_ch} 通道，等待固件确认 …")
        self._status(self.handoff_label, "warn")
        self._log("[发送完成] 等待 LAYER / TABLE DONE 回传")
        self._record_upload("sent", n_layers * n_ch)
        QTimer.singleShot(self.ACK_TIMEOUT_MS, self._table_ack_timeout)

    def _note_table_reply(self, line: str):
        """Pick the firmware's LAYER / TABLE DONE replies out of the log stream."""
        exp = getattr(self, "_table_expect", None)
        if not exp:
            return
        text = line.strip().upper()
        if text.startswith("LAYER"):
            exp["acked"] = True
            self.handoff_label.setText(f"固件确认：{line.strip()}")
            self._status(self.handoff_label, "ok")
        elif "TABLE DONE" in text:
            exp["acked"] = True
            exp["done"] = True
            self.btn_table.setEnabled(True)
            self.handoff_label.setText(
                f"固件已走完 {exp['layers']} 层（TABLE DONE）")
            self._status(self.handoff_label, "ok")
            self._record_upload("verified", exp["layers"] * exp["channels"])
            self._log("[OK] 固件回传 TABLE DONE")
            self._table_expect = None
        elif "ERR" in text or "CRC" in text:
            self.btn_table.setEnabled(True)
            self.handoff_label.setText(f"固件报错：{line.strip()}")
            self._status(self.handoff_label, "err")
            self._record_upload("failed", 0, error=line.strip())
            self._table_expect = None

    def _table_ack_timeout(self):
        """No reply in time -- say so rather than leaving 'sent' looking final."""
        exp = getattr(self, "_table_expect", None)
        if not exp or exp.get("done"):
            return
        self.btn_table.setEnabled(True)
        if exp.get("acked"):
            self._record_upload("acknowledged", exp["layers"] * exp["channels"])
            self._log("[!] 已收到 LAYER 回传，但未等到 TABLE DONE")
            return
        self.handoff_label.setText(
            "已发送，但固件没有回传确认——无法确定放大器是否已加载电流表。"
            "请确认固件已实现 CMD_LOAD_TABLE。")
        self._status(self.handoff_label, "err")
        self._record_upload(
            "unconfirmed", exp["layers"] * exp["channels"],
            error=f"no LAYER/TABLE DONE within {self.ACK_TIMEOUT_MS} ms")
        self._log("[!] 未收到固件确认，状态记为 unconfirmed")
        self._table_expect = None

    def _record_upload(self, status, n, error=""):
        try:
            from core import amp_bridge

            path = amp_bridge.record_upload(self.app_state, status=status,
                                            channels_written=n, error=error)
            if path:
                self._log(f"[记录] 上传状态 {status} → {path}")
        except Exception as e:
            self._log(f"[!] 上传结果未能记录：{type(e).__name__}: {e}")

    def _update_upload_table(self):
        """Refresh the upload preview table from app_state."""
        w_opt = self.app_state.get("optimal_currents")
        names = self.app_state.get("ch_names_active", [])
        if w_opt is None:
            self.upload_table.setRowCount(1)
            self.upload_table.setItem(0, 0, QTableWidgetItem("请先在匀场页面完成优化"))
            return

        n = len(w_opt)
        labels = names if names else [f"ch{i+1}" for i in range(n)]
        self.upload_table.setRowCount(n)
        for i, label in enumerate(labels):
            b, c = self._ch_to_bc(label)
            bc_str = f"B{b}·C{c}" if b is not None else "?"
            self.upload_table.setItem(i, 0, QTableWidgetItem(label))
            self.upload_table.setItem(i, 1, QTableWidgetItem(bc_str))
            self.upload_table.setItem(i, 2, QTableWidgetItem(f"{w_opt[i]:.5f}"))

    def _upload_currents(self):
        w_opt = self.app_state.get("optimal_currents")
        names = self.app_state.get("ch_names_active", [])
        if w_opt is None:
            QMessageBox.information(self, "提示", "无优化电流数据")
            return
        labels = names if names else [f"ch{i+1}" for i in range(len(w_opt))]
        self._send_optimal_currents(list(w_opt), labels)

    def _zero_then_upload(self):
        self._send("Z")
        QTimer.singleShot(500, self._upload_currents)

    def _send_optimal_currents(self, values: list, labels: list):
        if not self.ser or not self.ser.is_open:
            QMessageBox.warning(self, "未连接", "请先连接串口")
            return

        sent, skipped = 0, 0
        for i, label in enumerate(labels):
            b, c = self._ch_to_bc(label)
            if b is None:
                self._log(f"[!] 跳过无法解析的通道名: {label}")
                skipped += 1
                continue
            if b not in self.active_boards:
                self._log(f"[!] 跳过 {label} → B{b}C{c}（板子 B{b} 未激活）")
                skipped += 1
                continue
            val = clamp_current(float(values[i]))
            self.ser.write(f"Track,{b},{c},{val:.4f}\n".encode())
            self.set_matrix[b][c] = val
            self._log(f"→ Track,{b},{c},{val:.4f}  ({label})")
            time.sleep(0.002)
            sent += 1

        self._log(f"[OK] 发送完成：{sent} 个通道，跳过 {skipped} 个")

        # Record the outcome against the run that produced these currents, so
        # "computed" and "actually applied" stay distinguishable afterwards.
        try:
            from core import amp_bridge

            path = amp_bridge.record_upload(
                self.app_state,
                status="ok" if skipped == 0 else "partial",
                channels_written=sent,
                failed_channels=None if skipped == 0 else skipped)
            if path:
                self._log(f"[记录] 上传结果已写入 {path}")
        except Exception as e:
            self._log(f"[!] 上传结果未能记录：{type(e).__name__}: {e}")

        # Auto verify after delay
        QTimer.singleShot(2000, self._request_verify)

    def _request_verify(self):
        self._send("Icurrentall")
        QTimer.singleShot(2500, self._do_verify)

    def _do_verify(self):
        w_opt = self.app_state.get("optimal_currents")
        names = self.app_state.get("ch_names_active", [])
        if w_opt is None:
            return
        labels = names if names else [f"ch{i+1}" for i in range(len(w_opt))]

        errors = []
        for i, label in enumerate(labels):
            b, c = self._ch_to_bc(label)
            if b is None:
                continue
            d = self.ch_data.get(b * 8 + c)
            if d is None:
                errors.append(f"{label}: 无回报")
                continue
            diff = abs(d["current"] - w_opt[i])
            if diff > 0.1:
                errors.append(f"{label}: 设定{w_opt[i]:.3f} 实测{d['current']:.3f} 误差{diff:.3f}A")

        if not errors:
            self.verify_label.setText("✓ 验证通过：所有通道误差 < 0.1A")
            _status(self.verify_label, "ok")
        else:
            self.verify_label.setText(f"✗ {len(errors)} 个通道异常：\n" + "\n".join(errors[:5]))
            _status(self.verify_label, "err")

    # ══════════════════════════════════════════════════════════════════════════
    # Current Plan
    # ══════════════════════════════════════════════════════════════════════════
    def _load_plan_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择电流计划文件", "", "Text (*.txt)")
        if not path:
            return
        with open(path, "rb") as f:
            raw = f.read()
        vals = self._parse_txt(raw)
        if vals:
            self.plan_values = vals
            self.plan_index = 0
            n_grp = len(vals) // 64
            self.plan_status.setText(f"已加载 {len(vals)} 值，{n_grp} 组")
            self._update_plan_label()
        else:
            QMessageBox.warning(self, "解析失败", "文件格式错误（需要空格/逗号分隔的数字）")

    def _parse_txt(self, raw: bytes) -> list:
        vals = []
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            for v in re.split(r"[,\s]+", line.strip()):
                if v:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
        return vals

    def _prev_group(self):
        if self.plan_index > 0:
            self.plan_index -= 1
            self._apply_group(self.plan_index)
            self._update_plan_label()

    def _next_group(self):
        n_grp = len(self.plan_values) // 64
        if self.plan_index < n_grp:
            self._apply_group(self.plan_index)
            self.plan_index += 1
            self._update_plan_label()

    def _reset_plan(self):
        self.plan_index = 0
        self._send("Z")
        self._update_plan_label()

    def _apply_group(self, idx: int):
        if not self.ser or not self.ser.is_open:
            return
        start = idx * 64
        group = (self.plan_values[start:start + 64] + [0.0] * 64)[:64]
        for i, val in enumerate(group):
            b, c = i // 8, i % 8
            val = clamp_current(val)
            self.ser.write(f"Track,{b},{c},{val:.4f}\n".encode())
            self.set_matrix[b][c] = val
            time.sleep(0.002)
        self._log(f"→ Applied group {idx + 1} ({len(group)} channels)")
        QTimer.singleShot(1500, lambda: self._send("Icurrentall"))

    def _update_plan_label(self):
        n_grp = len(self.plan_values) // 64 if self.plan_values else 0
        self.plan_idx_label.setText(f"{self.plan_index} / {n_grp}")

    # ══════════════════════════════════════════════════════════════════════════
    # Channel Monitor
    # ══════════════════════════════════════════════════════════════════════════
    def _manual_refresh_channels(self):
        self._send("Icurrentall")

    def _toggle_auto_refresh(self, checked: bool):
        if checked:
            self._send("Icurrentall")
            self.refresh_timer.start(2000)
        else:
            self.refresh_timer.stop()

    def _auto_refresh_channels(self):
        if self.ser and self.ser.is_open:
            self.ser.write(b"Icurrentall\n")

    def _detect_active_boards(self):
        if not self.ch_data:
            return
        active = []
        for b in range(8):
            gains = []
            for c in range(8):
                d = self.ch_data.get(b * 8 + c)
                if d:
                    gains.append(d["gain"])
            if not gains:
                continue
            all_zero = all(abs(g) < 0.05 for g in gains)
            all_default = all(abs(g + 1.6) < 0.05 for g in gains)
            if not all_zero and not all_default:
                active.append(b)
        if active:
            self.active_boards = active

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _ch_to_bc(self, name: str):
        if name in self.ch_mapping:
            return self.ch_mapping[name]
        try:
            idx = int(str(name).replace("ch", "").replace("CH", "")) - 1
            return idx // 8, idx % 8
        except (ValueError, IndexError):
            return None, None

    def _log(self, msg: str):
        self.log_lines.append(msg)
        if len(self.log_lines) > 400:
            self.log_lines = self.log_lines[-300:]
        self.log_text.append(msg)
        # Auto-scroll to bottom
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_log(self):
        self.log_lines.clear()
        self.log_text.clear()

    # ══════════════════════════════════════════════════════════════════════════
    # Style
    # ══════════════════════════════════════════════════════════════════════════

    def shutdown(self):
        """Stop the reader thread, the refresh timer and the port.

        Called by `MainWindow.closeEvent`. It has to be a normal method rather
        than living in `closeEvent`, because Qt only sends close events to
        top-level windows and this page is a child of a QStackedWidget -- the
        close handler below never fires in the running app.
        """
        try:
            self.refresh_timer.stop()
        except Exception:
            pass
        if self.reader:
            self.reader.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def apply_theme(self, dark: bool, p: dict):
        self.p = p
