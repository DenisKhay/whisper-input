"""Interactive mic test — press Enter to start, Enter to stop."""
import sounddevice as sd
import numpy as np
import sys

chunks = []

def callback(indata, frames, time, status):
    chunks.append(indata.copy())

print("Available input devices:")
for i, d in enumerate(sd.query_devices()):
    if d['max_input_channels'] > 0:
        marker = " *" if i == sd.default.device[0] else ""
        print(f"  [{i}] {d['name']} (rate={d['default_samplerate']}){marker}")

dev = input("\nDevice number (Enter for default): ").strip()
dev = int(dev) if dev else None

info = sd.query_devices(dev or sd.default.device[0])
rate = int(info['default_samplerate'])
print(f"\nUsing: {info['name']} at {rate}Hz")

input("\nPress Enter to START recording...")
print(">>> RECORDING — speak now! Press Enter to STOP <<<")

stream = sd.InputStream(samplerate=rate, channels=1, dtype='float32', callback=callback, device=dev)
stream.start()

input()  # wait for Enter

stream.stop()
stream.close()

if not chunks:
    print("No audio captured!")
    sys.exit(1)

audio = np.concatenate(chunks).flatten()
duration = len(audio) / rate
max_amp = np.max(np.abs(audio))

print(f"\nDuration: {duration:.1f}s")
print(f"Max amplitude: {max_amp:.4f}")

# Show amplitude over time
window = int(rate * 0.5)
for i in range(min(20, len(audio) // window)):
    chunk = audio[i * window:(i + 1) * window]
    amp = np.max(np.abs(chunk))
    bar = '#' * int(amp * 200)
    print(f"  {i*0.5:.1f}s: {amp:.4f} {bar}")

if max_amp < 0.01:
    print("\n⚠ Very quiet! This mic may not be the right one.")
else:
    print(f"\n✓ Looks good! Amplitude {max_amp:.4f} should work for Whisper.")
