"""System tray icon showing daemon state."""

import logging
import threading

from PIL import Image, ImageDraw
import pystray

logger = logging.getLogger(__name__)

COLORS = {
    "idle": "#808080",
    "recording": "#FF0000",
    "transcribing": "#FFD700",
}


def _create_icon_image(color: str) -> Image.Image:
    """Create a simple colored circle icon."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    return img


class TrayIcon:
    def __init__(self, on_quit: callable, on_mode_toggle: callable | None = None):
        self._on_quit = on_quit
        self._on_mode_toggle = on_mode_toggle
        self._current_mode = "hold"
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None

    def _build_menu(self):
        return pystray.Menu(
            pystray.MenuItem(
                lambda _: f"Mode: {self._current_mode}",
                self._toggle_mode,
            ),
            pystray.MenuItem("Quit", self._quit),
        )

    def _toggle_mode(self):
        if self._on_mode_toggle:
            self._current_mode = "toggle" if self._current_mode == "hold" else "hold"
            self._on_mode_toggle(self._current_mode)
            self._icon.update_menu()
            logger.info("Mode toggled to %s", self._current_mode)

    def _quit(self):
        logger.info("Quit requested from tray")
        self._on_quit()
        if self._icon:
            self._icon.stop()

    def start(self, mode: str = "hold") -> None:
        """Start the tray icon in a background thread."""
        self._current_mode = mode
        self._icon = pystray.Icon(
            "whisper-input",
            icon=_create_icon_image(COLORS["idle"]),
            title="whisper-input (idle)",
            menu=self._build_menu(),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()
        logger.info("Tray icon started")

    def set_state(self, state: str) -> None:
        """Update the tray icon to reflect current state: idle, recording, transcribing."""
        if self._icon is None:
            return
        self._icon.icon = _create_icon_image(COLORS.get(state, COLORS["idle"]))
        self._icon.title = f"whisper-input ({state})"

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()
            self._icon = None
