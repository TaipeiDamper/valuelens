from __future__ import annotations

from typing import Callable

import keyboard


class HotkeyService:
    def __init__(self) -> None:
        self._handlers: dict[str, int] = {}

    def register(self, slot: str, hotkey: str, callback: Callable[[], None]) -> None:
        self.unregister(slot)
        self._handlers[slot] = keyboard.add_hotkey(hotkey, callback, suppress=False)

    def unregister(self, slot: str) -> None:
        handler_id = self._handlers.pop(slot, None)
        if handler_id is not None:
            keyboard.remove_hotkey(handler_id)

    def shutdown(self) -> None:
        for slot in list(self._handlers.keys()):
            self.unregister(slot)

