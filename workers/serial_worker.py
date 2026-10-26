"""Background serial reader thread using QThread + Signals.

Replaces the Streamlit module-level threading hack with proper Qt signal/slot.
"""
import time
import struct
from PySide6.QtCore import QThread, Signal


CURRENT_LIMIT = 3.5  # Hardware safety limit (A)


class SerialReaderWorker(QThread):
    """Continuously reads from Teensy serial port, emits parsed data via signals."""

    # Signals
    log_received = Signal(str)           # Raw log line
    channel_updated = Signal(int, int, float, float, bool)  # board, ch, current, gain, ok
    batch_complete = Signal()             # Full Icurrentall response parsed
    error_occurred = Signal(str)          # Fatal error

    def __init__(self, serial_port, parent=None):
        super().__init__(parent)
        self.ser = serial_port
        self._running = True

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        buf = []
        in_batch = False

        while self._running:
            try:
                if not self.ser or not self.ser.is_open:
                    break
                if self.ser.in_waiting == 0:
                    time.sleep(0.005)
                    continue

                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Batch start
                if line == "Icurrentall":
                    in_batch = True
                    buf = []
                    self.log_received.emit(f"← {line}")
                    continue

                # Batch end
                if line == "IcurrentallEnd":
                    self.log_received.emit(f"← {line}")
                    if in_batch:
                        for l in buf:
                            self._parse_batch_line(l)
                    in_batch = False
                    self.batch_complete.emit()
                    continue

                # Batch content line (format: B C current gain T/F)
                if in_batch:
                    buf.append(line)
                    continue

                # Single channel (format: Icurrent B C current gain T/F)
                if line.startswith("Icurrent "):
                    parts = line.split()
                    if len(parts) == 6:
                        try:
                            self.channel_updated.emit(
                                int(parts[1]), int(parts[2]),
                                float(parts[3]), float(parts[4]),
                                parts[5].upper() == "T"
                            )
                        except (ValueError, IndexError):
                            pass

                # All lines go to log
                self.log_received.emit(f"← {line}")

            except Exception as e:
                self.error_occurred.emit(str(e))
                break

    def _parse_batch_line(self, line: str):
        parts = line.split()
        if len(parts) == 5:
            try:
                self.channel_updated.emit(
                    int(parts[0]), int(parts[1]),
                    float(parts[2]), float(parts[3]),
                    parts[4].upper() == "T"
                )
            except (ValueError, IndexError):
                pass


def clamp_current(val: float) -> float:
    """Enforce hardware safety limit."""
    if abs(val) > CURRENT_LIMIT:
        return max(-CURRENT_LIMIT, min(CURRENT_LIMIT, val))
    return val
