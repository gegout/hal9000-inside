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

class SimpleVAD:
    """Voice activity detector with DC-offset removal and adaptive noise-floor tracking."""
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        speech_pad_ms: int = 300,
        silence_threshold_factor: float = 2.0,
        min_silence_duration_ms: int = 650,
        min_speech_duration_ms: int = 150,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.threshold_factor = silence_threshold_factor
        
        self.noise_floor = 40.0  # baseline AC standard deviation
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
        self.last_energy = 0.0

    def compute_energy(self, frame_bytes: bytes) -> float:
        """Compute AC energy (standard deviation, removing any DC bias)."""
        if len(frame_bytes) < self.frame_samples * 2:
            return 0.0
        samples = np.frombuffer(frame_bytes[:self.frame_samples * 2], dtype=np.int16)
        if len(samples) == 0:
            return 0.0
        return float(np.std(samples.astype(np.float64)))

    def is_speech_frame(self, frame_bytes: bytes) -> bool:
        ac_energy = self.compute_energy(frame_bytes)
        self.last_energy = ac_energy

        # Dynamic threshold based on AC energy floor
        threshold = max(90.0, self.noise_floor * self.threshold_factor)
        
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


class AlsaPlayback:
    """Reliable ALSA playback fallback via aplay subprocess when PortAudio is unavailable."""
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._proc = None

    def play_pcm(self, pcm_bytes: bytes):
        import subprocess
        try:
            self._proc = subprocess.Popen(
                ["aplay", "-q", "-r", str(self.sample_rate), "-f", "S16_LE", "-c", "1", "-t", "raw"],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._proc.communicate(input=pcm_bytes)
        except Exception as e:
            logger.debug(f"AlsaPlayback error: {e}")
        finally:
            self._proc = None

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


class AlsaCapture:
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

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


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
