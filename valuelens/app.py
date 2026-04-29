import os
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from valuelens.config.settings import SettingsManager
from valuelens.ui.overlay_window import OverlayWindow


def _enable_per_monitor_dpi_awareness() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
            return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
    except Exception:
        pass


def run() -> None:
    _enable_per_monitor_dpi_awareness()

    app = QApplication(sys.argv)
    app.setApplicationName("ValueLens")

    settings = SettingsManager().load()
    settings.custom_palette = []
    window = OverlayWindow(settings=settings)
    window.show()

    def _graceful_or_force_exit(*_) -> None:
        window.force_quit()
        QTimer.singleShot(1200, lambda: os._exit(0))

    signal.signal(signal.SIGINT, _graceful_or_force_exit)
    signal.signal(signal.SIGTERM, _graceful_or_force_exit)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _graceful_or_force_exit)
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    sys.exit(app.exec())


