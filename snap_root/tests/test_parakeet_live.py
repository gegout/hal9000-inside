import unittest
import asyncio
import io
import wave
import numpy as np

from myna.core import AudioFormat, PcmChunk, SessionConfig, TranscriptionDone
from myna.testbed.parakeet import ParakeetAdapter
from hal9000_inside.tts import HalPiperTTS
from hal9000_inside.audio import resample_pcm

FORMAT = AudioFormat(sample_rate_hz=16_000, channels=1, sample_width_bytes=2)

class TestParakeetLive(unittest.TestCase):
    def test_parakeet_transcription_live(self):
        # 1. Synthesize known audio using Piper HAL 9000 TTS
        tts = HalPiperTTS()
        tts.load()
        wav_bytes = tts.synthesize_wav_bytes("Affirmative, Dave. I read you.")

        # Extract 16kHz mono PCM for Parakeet
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
        
        pcm_16k = resample_pcm(raw, sr, 16000)

        # 2. Run through Myna ParakeetAdapter via run_session
        adapter = ParakeetAdapter()

        async def audio_gen():
            yield PcmChunk(data=pcm_16k, format=FORMAT)
            yield PcmChunk(data=b"", format=FORMAT)

        done_text = ""
        finals = []
        async def mock_emit(ev):
            nonlocal done_text
            if isinstance(ev, TranscriptionDone):
                done_text = ev.text
            elif hasattr(ev, "text") and ev.text:
                finals.append(ev.text)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cfg = SessionConfig(audio_format=FORMAT)
            loop.run_until_complete(adapter.run_session(cfg, audio_gen(), mock_emit))
            transcript = done_text.strip() if done_text else " ".join(finals).strip()
            print(f"\nRecognized Live Parakeet Transcript: '{transcript}'\n")
            self.assertTrue(len(transcript) > 0)
            t_lower = transcript.lower()
            self.assertTrue(any(word in t_lower for word in ["affirmative", "dave", "read", "you"]))
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
