from unittest.mock import MagicMock, patch
from pynput import keyboard

from whisper_input.hotkey import HotkeyListener


def test_hold_mode_press_triggers_start():
    on_start = MagicMock()
    on_stop = MagicMock()

    with patch("whisper_input.hotkey.keyboard.Listener") as MockListener:
        mock_listener_instance = MagicMock()
        mock_listener_instance.canonical = lambda k: k
        MockListener.return_value = mock_listener_instance

        listener = HotkeyListener(
            hotkey_str="<cmd>+v",
            mode="hold",
            on_start=on_start,
            on_stop=on_stop,
        )
        listener.start()

        call_kwargs = MockListener.call_args[1]
        on_press = call_kwargs["on_press"]
        on_release = call_kwargs["on_release"]

    # Simulate pressing cmd then v
    on_press(keyboard.Key.cmd)
    on_press(keyboard.KeyCode.from_char("v"))
    on_start.assert_called_once()

    # Simulate releasing v
    on_release(keyboard.KeyCode.from_char("v"))
    on_stop.assert_called_once()


def test_toggle_mode_two_presses():
    on_start = MagicMock()
    on_stop = MagicMock()

    with patch("whisper_input.hotkey.keyboard.Listener") as MockListener:
        mock_listener_instance = MagicMock()
        mock_listener_instance.canonical = lambda k: k
        MockListener.return_value = mock_listener_instance

        listener = HotkeyListener(
            hotkey_str="<cmd>+v",
            mode="toggle",
            on_start=on_start,
            on_stop=on_stop,
        )
        listener.start()

        call_kwargs = MockListener.call_args[1]
        on_press = call_kwargs["on_press"]
        on_release = call_kwargs["on_release"]

    # First combo press — starts recording
    on_press(keyboard.Key.cmd)
    on_press(keyboard.KeyCode.from_char("v"))
    on_start.assert_called_once()

    # Release keys
    on_release(keyboard.KeyCode.from_char("v"))
    on_release(keyboard.Key.cmd)

    # Second combo press — stops recording
    on_press(keyboard.Key.cmd)
    on_press(keyboard.KeyCode.from_char("v"))
    on_stop.assert_called_once()
