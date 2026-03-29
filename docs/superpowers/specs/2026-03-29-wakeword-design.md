# Wake Word Feature Design

## Overview

Add voice-triggered recording alongside the existing hotkey. Say "Claude" to start recording, speak naturally, pause, say "over" to transcribe and paste. Say "Claude, cancel" to discard.

## User Flow

1. **Idle** — daemon listens for speech via silero-vad (CPU only)
2. **Speech detected** — short utterance transcribed with Whisper `tiny` (~100ms)
3. **"Claude" found** — beep plays, tray turns red, main recording starts
4. **User speaks** — audio buffered (same as current flow)
5. **~1s pause detected** — pause segment checked with Whisper `tiny`
6. **"over" found** — stop recording, transcribe full buffer with `medium`, paste via Shift+Insert
7. **"Claude cancel" found** — discard buffer, low beep, back to idle
8. **Neither found** — resume recording (user was just pausing to think)
9. **Timeout (5s of silence after wake)** — cancel silently, back to idle

## Coexistence with Hotkey

- Hotkey (Ctrl+`) continues to work in both hold and toggle modes
- Wake word listener runs in parallel, same recorder/transcriber/output pipeline
- If hotkey is used while wake word is active, hotkey takes priority
- If wake word triggers while hotkey recording is active, ignored

## New Components

### `wakeword.py`

State machine managing the voice activation flow:

```
IDLE -> LISTENING -> RECORDING -> TRANSCRIBING -> IDLE
                  \-> IDLE (timeout/cancel)
```

- **IDLE**: silero-vad monitors audio stream for speech activity
- Speech detected → buffer the speech segment, when silence returns, send to `tiny` model
- If transcription contains "claude" → transition to RECORDING, play beep
- **RECORDING**: audio buffered (uses existing Recorder)
- On ~1s silence → check pause segment with `tiny` model
  - "over" → transition to TRANSCRIBING
  - "claude cancel" → discard, beep (low), back to IDLE
  - anything else → continue recording
- On 5s total silence after wake → back to IDLE silently
- **TRANSCRIBING**: full buffer sent to `medium` model, result pasted

### `beep.py`

Simple audio feedback using sounddevice:
- Wake beep: 800Hz sine wave, 150ms, moderate volume
- Cancel beep: 400Hz sine wave, 150ms, moderate volume
- Generated programmatically, no audio files needed

## Modified Components

### `transcriber.py`

Add Whisper `tiny` model for quick keyword checks:
- `tiny` loaded at startup alongside `medium` (~75MB VRAM extra)
- New method `quick_transcribe(audio) -> str` — uses `tiny`, no VAD, fast
- Existing `transcribe(audio) -> str` unchanged (uses `medium`)

### `recorder.py`

No changes to the Recorder class itself. The wakeword module creates its own continuous audio stream for VAD monitoring, separate from the on-demand Recorder used for main recording.

### `main.py`

- Initialize wakeword listener if config enables it
- Wire wakeword callbacks to same `_on_recording_start` / `_on_recording_stop` pipeline
- Handle priority: hotkey overrides wakeword if both active

### `config.py`

New config options with defaults:

```yaml
wakeword:
  enabled: true
  word: "claude"
  stop_word: "over"
  cancel_phrase: "claude cancel"
  timeout: 5
  beep: true
```

## Resource Usage

| State | CPU | GPU (VRAM) |
|-------|-----|------------|
| Idle (VAD listening) | <1% | 0 (models loaded but idle, ~3.1GB resident) |
| Speech detected (tiny check) | brief spike | ~100ms burst |
| Recording | <1% (audio buffering) | 0 |
| Transcribing (medium) | brief spike | ~1-2s burst |

Silero-vad is a small PyTorch model (~2MB) that runs entirely on CPU. It processes 30ms audio chunks and returns a speech probability score. No GPU needed.

## Dependencies

New Python package:
- `silero-vad` via `torch` (silero-vad is distributed as a torch hub model, needs PyTorch)

Note: PyTorch is already installed as a dependency of faster-whisper (CTranslate2). We need `torchaudio` for silero-vad. Check if the existing torch installation is sufficient or if torchaudio needs to be added.

## Audio Stream Architecture

Two separate audio streams:
1. **VAD stream** (always-on): continuous 16kHz mono from DMIC, feeds silero-vad. Managed by wakeword module.
2. **Recording stream** (on-demand): started by hotkey or wake word trigger. Managed by existing Recorder class.

When wake word triggers recording, the VAD stream pauses (or its output is ignored) and the Recorder takes over. When recording ends, VAD stream resumes.

## Keyword Matching

The `tiny` model output is checked with simple substring matching:
- `"claude" in text.lower()` → wake word detected
- `"over" in text.lower()` → stop word detected (only checked after pause during recording)
- `"claude cancel" in text.lower()` or `"claude, cancel" in text.lower()` → cancel detected

The `tiny` model may produce slight variations ("cloud", "claud"), so matching should be fuzzy:
- Check for: "claude", "claud", "cloud" (common misheard variants)
- For "over": exact match only (it's distinctive enough)
- For "cancel": check after any "claude"/"claud"/"cloud" prefix
