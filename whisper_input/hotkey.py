"""Global hotkey listener using pynput."""

import logging

from pynput import keyboard

logger = logging.getLogger(__name__)


class HotkeyListener:
    def __init__(
        self,
        hotkey_str: str,
        mode: str,
        on_start: callable,
        on_stop: callable,
    ):
        self.mode = mode
        self._on_start = on_start
        self._on_stop = on_stop
        self._target_keys = set(keyboard.HotKey.parse(hotkey_str))
        self._pressed_keys: set = set()
        self._active = False
        self._listener: keyboard.Listener | None = None

    def _canonical(self, key):
        if self._listener is not None:
            return self._listener.canonical(key)
        return key

    def _on_press(self, key):
        canon = self._canonical(key)
        self._pressed_keys.add(canon)

        if self._target_keys.issubset(self._pressed_keys):
            if self.mode == "hold":
                if not self._active:
                    self._active = True
                    logger.info("Hold hotkey pressed — starting")
                    self._on_start()
            elif self.mode == "toggle":
                if not self._active:
                    self._active = True
                    logger.info("Toggle hotkey pressed — starting")
                    self._on_start()
                else:
                    self._active = False
                    logger.info("Toggle hotkey pressed — stopping")
                    self._on_stop()

    def _on_release(self, key):
        canon = self._canonical(key)

        if self.mode == "hold" and self._active:
            if canon in self._target_keys:
                self._active = False
                logger.info("Hold hotkey released — stopping")
                self._on_stop()

        self._pressed_keys.discard(canon)

    def start(self) -> None:
        """Start listening for the hotkey."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info("Hotkey listener started (mode=%s, keys=%s)", self.mode, self._target_keys)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
