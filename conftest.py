"""Root conftest: stub out native libraries that may not be present in CI."""
import sys
from unittest.mock import MagicMock

# sounddevice requires the PortAudio shared library at import time.
# Pre-mock it so tests can run on machines without audio hardware/drivers.
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()
