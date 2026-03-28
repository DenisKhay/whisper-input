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
        ["xdotool", "type", "--clearmodifiers", "--delay", "12", text],
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


def output_text(text: str) -> None:
    """Type text into focused window and copy to clipboard."""
    if not text:
        return
    time.sleep(0.15)  # let hotkey release propagate
    copy_to_clipboard(text)
    type_text(text)
