"""Fake TTS engine for test isolation without loading Piper models."""

import io
import wave
from typing import List, Optional


class FakeTTS:
    """Produces silent PCM/WAV with length proportional to text length."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.is_loaded = True
        self.synthesized_texts: List[str] = []
        self._canceled = False

    def load(self):
        self.is_loaded = True

    def cancel(self):
        self._canceled = True

    def synthesize_raw_pcm(self, text: str) -> bytes:
        self.synthesized_texts.append(text)
        # 0.05 seconds of silent 16-bit mono PCM per character, capped between 0.1s and 2.0s
        duration_s = max(0.1, min(2.0, len(text) * 0.03))
        num_samples = int(self.sample_rate * duration_s)
        return b"\x00\x00" * num_samples

    def synthesize_wav_bytes(self, text: str) -> bytes:
        pcm = self.synthesize_raw_pcm(text)
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return bio.getvalue()
