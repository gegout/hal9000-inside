"""Audio capture, playback, resampling, half-duplex, and VAD support."""

import asyncio
import collections
import logging
import math
import struct
import time
from typing import AsyncGenerator, Callable, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000  # Parakeet STT input rate
PIPER_SAMPLE_RATE = 22050   # Native Piper HAL 9000 output rate

class BiquadHighPass:
    """2nd-order Butterworth IIR high-pass filter with streaming state."""
    def __init__(self, cutoff_hz: float = 100.0, sample_rate: int = 16000):
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        self.enabled = cutoff_hz > 0.0
        
        if self.enabled:
            # Bilinear transform of analog 2nd-order Butterworth HPF
            w0 = math.tan(math.pi * cutoff_hz / sample_rate)
            k = math.sqrt(2.0) * w0
            denom = 1.0 + k + w0 * w0
            self.b0 = 1.0 / denom
            self.b1 = -2.0 / denom
            self.b2 = 1.0 / denom
            self.a1 = 2.0 * (w0 * w0 - 1.0) / denom
            self.a2 = (1.0 - k + w0 * w0) / denom
        else:
            self.b0 = 1.0
            self.b1 = self.b2 = self.a1 = self.a2 = 0.0

        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, frame_bytes: bytes) -> bytes:
        """Filter 16-bit mono PCM bytes while updating filter state continuously."""
        if not self.enabled or not frame_bytes:
            return frame_bytes

        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return frame_bytes

        out = np.empty_like(samples)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2

        for i in range(len(samples)):
            x0 = samples[i]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, x0
            y2, y1 = y1, y0
            out[i] = y0

        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        out_clipped = np.clip(out, -32768, 32767).astype(np.int16)
        return out_clipped.tobytes()

    def reset(self):
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0


class SimpleVAD:
    """Voice activity detector with DC-offset removal, speech-band filtering, and adaptive noise-floor tracking."""
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        speech_pad_ms: int = 300,
        silence_threshold_factor: float = 2.0,
        min_silence_duration_ms: int = 900,
        min_speech_duration_ms: int = 150,
        hpf_cutoff_hz: float = 100.0,
        legacy_mode: bool = False,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.threshold_factor = silence_threshold_factor
        self.legacy_mode = legacy_mode

        # High-pass filter for VAD feature extraction
        self.hpf = BiquadHighPass(cutoff_hz=0.0 if legacy_mode else hpf_cutoff_hz, sample_rate=sample_rate)

        self.noise_floor = 40.0  # baseline AC standard deviation
        self._initial_calibration_frames = 0 if legacy_mode else 10  # first ~300ms calibrate noise floor
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
        self.last_energy = 0.0

    def compute_energy(self, frame_bytes: bytes) -> float:
        """Compute AC energy (standard deviation, removing DC bias and infrasound)."""
        if len(frame_bytes) < self.frame_samples * 2:
            return 0.0

        if not self.legacy_mode:
            filtered_bytes = self.hpf.process(frame_bytes[:self.frame_samples * 2])
            samples = np.frombuffer(filtered_bytes, dtype=np.int16)
        else:
            samples = np.frombuffer(frame_bytes[:self.frame_samples * 2], dtype=np.int16)

        if len(samples) == 0:
            return 0.0
        return float(np.std(samples.astype(np.float64)))

    def is_speech_frame(self, frame_bytes: bytes) -> bool:
        ac_energy = self.compute_energy(frame_bytes)
        self.last_energy = ac_energy

        # Initial calibration phase: quickly converge noise floor during silence
        if self._initial_calibration_frames > 0:
            self._initial_calibration_frames -= 1
            if self._initial_calibration_frames == 9:
                self.noise_floor = ac_energy
            else:
                self.noise_floor = self.noise_floor * 0.7 + ac_energy * 0.3
            return False

        # Dynamic threshold based on AC energy floor
        min_thresh = 90.0 if self.legacy_mode else 70.0
        threshold = max(min_thresh, self.noise_floor * self.threshold_factor)
        
        if ac_energy > threshold:
            return True
        else:
            # Adapt noise floor slowly during non-speech
            self.noise_floor = self.noise_floor * 0.95 + ac_energy * 0.05
            return False

    def process_frame(self, frame_bytes: bytes) -> Tuple[bool, bool, float]:
        """Returns (is_speaking, utterance_finished, energy)."""
        speech_detected = self.is_speech_frame(frame_bytes)
        
        utterance_finished = False
        if speech_detected:
            self.speech_frame_count += 1
            self.silence_frame_count = 0
            if self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
        else:
            if self.is_speaking:
                self.silence_frame_count += 1
                if self.silence_frame_count >= self.min_silence_frames:
                    self.is_speaking = False
                    utterance_finished = True
                    self.speech_frame_count = 0
                    self.silence_frame_count = 0
            else:
                self.speech_frame_count = 0
                
        return self.is_speaking, utterance_finished, self.last_energy

    def reset(self):
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.last_energy = 0.0
        self.hpf.reset()


def resample_pcm(pcm_bytes: bytes, in_rate: int, out_rate: int) -> bytes:
    """Resample 16-bit mono PCM using linear interpolation."""
    if in_rate == out_rate or not pcm_bytes:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(samples) == 0:
        return b""
    num_output_samples = int(len(samples) * out_rate / in_rate)
    if num_output_samples == 0:
        return b""
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, num_output_samples)
    resampled = np.interp(x_new, x_old, samples).astype(np.int16)
    return resampled.tobytes()


class HalfDuplexAudioController:
    """Coordinates microphone capture and speaker output to prevent acoustic feedback."""
    def __init__(self, cooldown_seconds: float = 0.5):
        self.cooldown_seconds = cooldown_seconds
        self._is_playing = False
        self._playback_end_time = 0.0
        self._muted = False
        self._push_to_talk_active = False
        self._auto_mode = True

    def set_auto_mode(self, enabled: bool):
        self._auto_mode = enabled

    def set_muted(self, muted: bool):
        self._muted = muted

    def set_push_to_talk(self, active: bool):
        self._push_to_talk_active = active

    def notify_playback_started(self):
        self._is_playing = True

    def notify_playback_stopped(self):
        self._is_playing = False
        self._playback_end_time = time.monotonic()

    def can_capture(self) -> bool:
        if self._muted:
            return False
        if self._is_playing:
            return False
        if time.monotonic() - self._playback_end_time < self.cooldown_seconds:
            return False
        if not self._auto_mode and not self._push_to_talk_active:
            return False
        return True

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def push_to_talk_active(self) -> bool:
        return self._push_to_talk_active

    @property
    def is_playing(self) -> bool:
        return self._is_playing


class AudioSink:
    """Abstract interface for audio playback."""
    def play_pcm(self, pcm_bytes: bytes):
        raise NotImplementedError

    def stop(self):
        pass


class AudioSource:
    """Abstract interface for audio capture."""
    async def stream_frames(self, frame_samples: int = 480) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError

    def purge_buffered(self):
        """Purge any stale audio buffered during a spoken turn."""
        pass

    def stop(self):
        pass


class AplaySink(AudioSink):
    """ALSA playback via aplay subprocess."""
    def __init__(self, sample_rate: int = 22050, device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._proc = None

    def play_pcm(self, pcm_bytes: bytes):
        import subprocess
        try:
            cmd = ["aplay", "-q", "-r", str(self.sample_rate), "-f", "S16_LE", "-c", "1", "-t", "raw"]
            if self.device:
                cmd.extend(["-D", self.device])
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._proc.communicate(input=pcm_bytes)
        except Exception as e:
            logger.debug(f"AplaySink error: {e}")
        finally:
            self._proc = None

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


# Backward compatibility alias
AlsaPlayback = AplaySink


class WavFileSink(AudioSink):
    """Saves synthesized audio to a WAV file."""
    def __init__(self, file_path: str, sample_rate: int = 22050):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self._all_pcm = bytearray()

    def play_pcm(self, pcm_bytes: bytes):
        import wave
        self._all_pcm.extend(pcm_bytes)
        with wave.open(self.file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(self._all_pcm))

    def stop(self):
        pass


class NullSink(AudioSink):
    """Discards synthesized audio (useful for silent benchmarks & tests)."""
    def play_pcm(self, pcm_bytes: bytes):
        pass

    def stop(self):
        pass


class ArecordSource(AudioSource):
    """Streams 16-bit mono 16kHz audio from microphone using arecord."""
    def __init__(self, sample_rate: int = 16000, device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._proc = None

    async def stream_frames(self, frame_samples: int = 480) -> AsyncGenerator[bytes, None]:
        """Asynchronously yield PCM frames of size frame_samples * 2 bytes."""
        cmd = ["arecord", "-q", "-r", str(self.sample_rate), "-f", "S16_LE", "-c", "1", "-t", "raw"]
        if self.device:
            cmd.extend(["-D", self.device])

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        chunk_size = frame_samples * 2  # 16-bit = 2 bytes per sample
        try:
            while True:
                data = await self._proc.stdout.readexactly(chunk_size)
                yield data
        except (asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        finally:
            self.stop()

    def purge_buffered(self):
        """Purge buffered audio from arecord pipe so stale speech is discarded."""
        if self._proc and self._proc.stdout:
            try:
                # Read whatever is currently in OS buffer without blocking
                while True:
                    # In asyncio StreamReader, _buffer holds unconsumed bytes
                    if hasattr(self._proc.stdout, "_buffer") and self._proc.stdout._buffer:
                        self._proc.stdout._buffer.clear()
                    break
            except Exception:
                pass

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


# Backward compatibility alias
AlsaCapture = ArecordSource


class WavFileSource(AudioSource):
    """Reads 16-bit mono 16kHz PCM audio from a WAV file and streams frames."""
    def __init__(self, file_path: str, sample_rate: int = 16000, real_time: bool = False):
        import wave
        with wave.open(file_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        self.pcm = resample_pcm(raw, sr, sample_rate)
        self.sample_rate = sample_rate
        self.real_time = real_time
        self._current_offset = 0

    def purge_buffered(self):
        """Purge buffered audio."""
        pass

    def skip_bytes(self, num_bytes: int):
        """Skip bytes in the audio stream (e.g. while sleeping)."""
        self._current_offset = min(len(self.pcm), self._current_offset + num_bytes)

    async def stream_frames(self, frame_samples: int = 480) -> AsyncGenerator[bytes, None]:
        chunk_size = frame_samples * 2
        self._current_offset = 0
        while self._current_offset < len(self.pcm):
            i = self._current_offset
            chunk = self.pcm[i:i + chunk_size]
            self._current_offset += chunk_size
            if len(chunk) < chunk_size:
                chunk = chunk + b"\x00" * (chunk_size - len(chunk))
            if self.real_time:
                await asyncio.sleep(frame_samples / self.sample_rate)
            yield chunk

        # Append trailing silence so VAD marks utterance_finished
        silence_frame = b"\x00" * chunk_size
        for _ in range(35):
            if self.real_time:
                await asyncio.sleep(frame_samples / self.sample_rate)
            yield silence_frame


def list_audio_devices():
    """List available capture and playback devices via arecord / aplay."""
    import subprocess
    captures = []
    playbacks = []
    try:
        out = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("card "):
                captures.append(line.strip())
    except Exception:
        pass
    try:
        out = subprocess.check_output(["aplay", "-l"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("card "):
                playbacks.append(line.strip())
    except Exception:
        pass
    return captures, playbacks
