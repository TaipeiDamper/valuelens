from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from valuelens.config.settings import AppSettings, SettingsManager

class AppStore(QObject):
    """
    全域狀態總線 (State Store)
    負責集中管理 AppSettings 的存取與持久化，解耦 UI 和渲染邏輯。
    """
    state_changed = Signal(AppSettings)

    def __init__(self) -> None:
        super().__init__()
        self._manager = SettingsManager()
        self._settings = self._manager.load()

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def update(self, **kwargs) -> None:
        """批次更新設定，並發送更新信號"""
        changed = False
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                old_val = getattr(self._settings, key)
                if old_val != value:
                    setattr(self._settings, key, value)
                    changed = True
        
        if changed:
            self._manager.save(self._settings)
            self.state_changed.emit(self._settings)
