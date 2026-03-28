"""Text output via xdotool (keyboard simulation) and xclip (clipboard)."""

import logging
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

REQUIRED_COMMANDS = ["xdotool", "xclip"]


def check_dependencies() -> list[str]:
    """Return list of missing system commands."""
    return [cmd for cmd in REQUIRED_COMMANDS if shutil.which(cmd) is None]


def type_text(text: str) -> None:
    """Type text into the focused window using xdotool."""
    subprocess.run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "20", text],
        check=True,
    )
    logger.debug("Typed %d characters via xdotool", len(text))


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard using xclip."""
    subprocess.run(
        ["xclip", "-selection", "clipboard"],
        input=text,
        text=True,
        check=True,
    )
    logger.debug("Copied to clipboard")


def paste_text() -> None:
    """Paste clipboard content into focused window using xdotool."""
    subprocess.run(
        ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
        check=True,
    )
    logger.debug("Pasted via Ctrl+V")


def output_text(text: str) -> None:
    """Copy text to clipboard and paste into focused window."""
    if not text:
        return
    time.sleep(0.3)  # let hotkey release propagate
    # Delete the stray backtick that leaks from the hotkey release
    subprocess.run(["xdotool", "key", "--clearmodifiers", "BackSpace"], check=True)
    time.sleep(0.05)
    copy_to_clipboard(text)
    time.sleep(0.05)
    # Use Shift+Insert — more reliable paste than Ctrl+V on KDE/X11
    subprocess.run(["xdotool", "key", "--clearmodifiers", "shift+Insert"], check=True)
    logger.debug("Pasted via Shift+Insert")
