"""HAL 9000 Piper TTS wrapper with ordered playback, sentence chunking, and immediate cancellation."""

import asyncio
import io
import json
import logging
import os
import re
import wave
from typing import AsyncGenerator, Callable, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/hal9000"))
MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "hal.onnx")
CONFIG_PATH = os.path.join(DEFAULT_MODEL_DIR, "hal.onnx.json")

EXPECTED_ONNX_SHA256 = "0e08c82dc027bc72b8b839324801709e205e8c201ccae171b92e37a664d94361"
EXPECTED_JSON_SHA256 = "cb6f82c03fc9e6db1f3b6978b9601bab9854f92c40d3a0681501944b8c26745e"

SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences for pipelined synthesis."""
    clean = re.sub(r'[*_#`\[\]]', '', text).strip()
    if not clean:
        return []
    parts = SENTENCE_SPLIT_PATTERN.split(clean)
    results = [p.strip() for p in parts if p.strip()]
    return results


def resolve_model_path(path: Optional[str] = None) -> str:
    """Resolve model path, falling back to bundled MODEL_PATH if relative path is missing."""
    if path:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded) and os.path.isfile(expanded):
            return expanded
        if os.path.isfile(expanded):
            return os.path.abspath(expanded)
    return MODEL_PATH


def resolve_config_path(path: Optional[str] = None) -> str:
    """Resolve config path, falling back to bundled CONFIG_PATH if relative path is missing."""
    if path:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded) and os.path.isfile(expanded):
            return expanded
        if os.path.isfile(expanded):
            return os.path.abspath(expanded)
    return CONFIG_PATH


class HalPiperTTS:
    """Resident Piper voice synthesizer for HAL 9000."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_cuda: bool = False,
    ):
        self.model_path = resolve_model_path(model_path)
        self.config_path = resolve_config_path(config_path)
        self.use_cuda = use_cuda
        self._voice = None
        self._cancelled = False
        self._is_synthesizing = False

    def load(self):
        """Loads and verifies the resident Piper voice model."""
        import hashlib
        from piper.voice import PiperVoice

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"HAL 9000 model not found at {self.model_path}")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"HAL 9000 config not found at {self.config_path}")

        # Check checksums
        with open(self.model_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            if h != EXPECTED_ONNX_SHA256:
                raise ValueError(f"Checksum mismatch for {self.model_path}: expected {EXPECTED_ONNX_SHA256}, got {h}")

        with open(self.config_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            if h != EXPECTED_JSON_SHA256:
                raise ValueError(f"Checksum mismatch for {self.config_path}: expected {EXPECTED_JSON_SHA256}, got {h}")

        logger.info("Verified HAL 9000 model checksums successfully.")
        try:
            self._voice = PiperVoice.load(
                self.model_path,
                config_path=self.config_path,
                use_cuda=self.use_cuda,
            )
        except Exception as e:
            if self.use_cuda:
                logger.warning(f"Failed to load Piper on CUDA ({e}). Falling back to CPU.")
                self._voice = PiperVoice.load(
                    self.model_path,
                    config_path=self.config_path,
                    use_cuda=False,
                )
            else:
                raise
        logger.info(f"Loaded Piper HAL 9000 voice: native sample rate {self._voice.config.sample_rate} Hz")

    @property
    def sample_rate(self) -> int:
        if self._voice:
            return self._voice.config.sample_rate
        return 22050

    def cancel(self):
        """Immediately signals cancellation to discard buffered playback and synthesis."""
        self._cancelled = True

    def reset_cancellation(self):
        self._cancelled = False

    def synthesize_wav_bytes(self, text: str) -> bytes:
        """Synthesize text into complete WAV bytes."""
        if not self._voice:
            self.load()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return buf.getvalue()

    def synthesize_raw_pcm(self, text: str) -> bytes:
        """Synthesize text into raw 16-bit mono 22,050 Hz PCM bytes."""
        wav_bytes = self.synthesize_wav_bytes(text)
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                return wf.readframes(wf.getnframes())

    async def speak_sentences(
        self,
        text: str,
        on_audio_chunk: Callable[[bytes], None],
        on_sentence_start: Optional[Callable[[str], None]] = None,
    ):
        """Asynchronously synthesizes and yields sentences in order, respecting cancellation."""
        self.reset_cancellation()
        sentences = split_sentences(text)
        if not sentences:
            return

        for sentence in sentences:
            if self._cancelled:
                logger.info("Speech playback cancelled by user.")
                break
            if on_sentence_start:
                on_sentence_start(sentence)
            
            # Run synthesis in executor thread to keep asyncio event loop responsive
            loop = asyncio.get_running_loop()
            raw_pcm = await loop.run_in_executor(None, self.synthesize_raw_pcm, sentence)
            
            if self._cancelled:
                break
            
            # Send raw PCM chunk to playback callback
            on_audio_chunk(raw_pcm)
