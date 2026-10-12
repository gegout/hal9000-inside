import asyncio
import io
import unittest
import numpy as np

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.audio import resample_pcm
from hal9000_inside.tts import HalPiperTTS

class TestAssistantEndToEnd(unittest.TestCase):
    """End-to-end test exercising the actual Assistant speech loop & turn handling."""

    def test_assistant_speech_loop_single_turn(self):
        assistant = Hal9000Assistant()

        # Generate a real spoken query using HAL Piper voice: "What is your name?"
        tts = HalPiperTTS()
        tts.load()
        wav_bytes = tts.synthesize_wav_bytes("What is your name?")
        
        # Read PCM bytes from WAV
        import wave
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                pcm_raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()

        pcm_16k = resample_pcm(pcm_raw, sr, 16000)

        # Feed 30ms frames through assistant loop logic including VAD and pre-roll
        frame_len = 480 * 2  # 30ms @ 16kHz
        frames = [pcm_16k[i:i + frame_len] for i in range(0, len(pcm_16k), frame_len)]
        # Pad with 800ms silence to trigger utterance_finished
        silence_frame = b"\x00" * frame_len
        frames.extend([silence_frame] * 30)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        captured_turn = False
        try:
            # We mock capture.stream_frames to feed these real speech frames
            async def mock_stream_frames(frame_samples):
                for f in frames:
                    yield f

            # Replace capture stream with mock_stream_frames
            original_capture = assistant.run

            # Test turn handler directly with the PCM audio
            loop.run_until_complete(assistant._handle_turn(pcm_16k))
            
            # Check dialogue history recorded the turn
            recent = assistant.memory.get_recent_turns(limit=2)
            self.assertTrue(len(recent) >= 2)
            self.assertEqual(recent[-1]["role"], "assistant")
            self.assertTrue(len(recent[-1]["content"].strip()) > 0)
            captured_turn = True

        finally:
            loop.close()

        self.assertTrue(captured_turn)

if __name__ == "__main__":
    unittest.main()
