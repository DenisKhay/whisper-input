"""Main entry point — wires all components together."""

import argparse
import logging
import signal
import sys
import threading

from whisper_input.config import load_config
from whisper_input.output import check_dependencies, output_text
from whisper_input.recorder import Recorder
from whisper_input.transcriber import Transcriber
from whisper_input.hotkey import HotkeyListener
from whisper_input.tray import TrayIcon

logger = logging.getLogger("whisper_input")


class App:
    def __init__(self, config: dict):
        self.config = config
        self._shutdown_event = threading.Event()
        self._recorder = Recorder(device=config.get("audio_device"))
        self._transcriber: Transcriber | None = None
        self._hotkey: HotkeyListener | None = None
        self._tray: TrayIcon | None = None

    def _on_recording_start(self) -> None:
        logger.info("Recording started")
        if self._tray:
            self._tray.set_state("recording")
        self._recorder.start()

    def _on_recording_stop(self) -> None:
        logger.info("Recording stopped, transcribing...")
        audio = self._recorder.stop()
        if len(audio) == 0:
            logger.warning("No audio captured")
            if self._tray:
                self._tray.set_state("idle")
            return

        import numpy as np
        max_amp = float(np.max(np.abs(audio)))
        logger.info("Audio max amplitude: %.4f", max_amp)
        if max_amp < 0.01:
            logger.warning("Audio too quiet (%.4f), skipping transcription", max_amp)
            if self._tray:
                self._tray.set_state("idle")
            return

        if self._tray:
            self._tray.set_state("transcribing")

        text = self._transcriber.transcribe(audio)
        output_text(text)

        if self._tray:
            self._tray.set_state("idle")

    def _on_mode_toggle(self, new_mode: str) -> None:
        if self._hotkey:
            self._hotkey.mode = new_mode
            logger.info("Switched to %s mode", new_mode)

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        self._shutdown_event.set()
        if self._hotkey:
            self._hotkey.stop()
        if self._tray:
            self._tray.stop()

    def run(self) -> None:
        # Check system deps
        missing = check_dependencies()
        if missing:
            print(f"Missing system dependencies: {', '.join(missing)}")
            print("Install with: sudo apt install " + " ".join(missing))
            sys.exit(1)

        # Load transcriber
        self._transcriber = Transcriber(
            model=self.config["model"],
            device=self.config["device"],
            compute_type=self.config["compute_type"],
            language=self.config["language"],
        )

        # Start tray
        self._tray = TrayIcon(
            on_quit=self._shutdown,
            on_mode_toggle=self._on_mode_toggle,
        )
        self._tray.start(mode=self.config["mode"])

        # Start hotkey listener
        self._hotkey = HotkeyListener(
            hotkey_str=self.config["hotkey"],
            mode=self.config["mode"],
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
        )
        self._hotkey.start()

        # Handle signals
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        print(f"whisper-input running (mode={self.config['mode']}, hotkey={self.config['hotkey']})")
        print("Press Ctrl+C or use tray icon to quit.")

        self._shutdown_event.wait()


def main():
    parser = argparse.ArgumentParser(description="Voice-to-text input daemon")
    parser.add_argument("-c", "--config", help="Path to config YAML file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
