from __future__ import annotations

from PySide6.QtCore import QObject, Signal, QTimer
from valuelens.config.settings import AppSettings, SettingsManager

class AppStore(QObject):
    """
    全域狀態總線 (State Store)
    負責集中管理 AppSettings 的存取與持久化，解耦 UI 和渲染邏輯。
    使用延遲寫入 (debounce) 減少高頻拖動時的磁碟 I/O。
    """
    state_changed = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self._manager = SettingsManager()
        self._settings = self._manager.load()
        # 延遲寫入計時器：500ms 後才真正寫入磁碟
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._flush_save)

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def update(self, **kwargs) -> None:
        """批次更新設定，並發送更新信號"""
        changed_keys = []
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                old_val = getattr(self._settings, key)
                if old_val != value:
                    setattr(self._settings, key, value)
                    changed_keys.append(key)
        
        if changed_keys:
            # 重啟延遲寫入計時器（debounce）
            self._save_timer.start()
            self.state_changed.emit(changed_keys)

    def _flush_save(self) -> None:
        """實際執行磁碟寫入。"""
        self._manager.save(self._settings)

    def force_save(self) -> None:
        """強制立即寫入（用於關閉程式時）。"""
        self._save_timer.stop()
        self._manager.save(self._settings)

