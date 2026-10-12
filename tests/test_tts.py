import unittest
import os
import wave
import io
from hal9000_inside.tts import HalPiperTTS, split_sentences, EXPECTED_ONNX_SHA256, EXPECTED_JSON_SHA256, MODEL_PATH, CONFIG_PATH

class TestTTS(unittest.TestCase):
    def test_sentence_splitting(self):
        text = "Good afternoon, Dave. Everything is running smoothly! How can I help you?"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Good afternoon, Dave.")
        self.assertEqual(sentences[1], "Everything is running smoothly!")
        self.assertEqual(sentences[2], "How can I help you?")

    def test_model_verification_and_synthesis(self):
        tts = HalPiperTTS()
        tts.load()
        self.assertEqual(tts.sample_rate, 22050)

        wav_bytes = tts.synthesize_wav_bytes("I am a HAL 9000 computer.")
        self.assertTrue(len(wav_bytes) > 1000)

        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 22050)
                nframes = wf.getnframes()
                self.assertTrue(nframes > 0)

if __name__ == "__main__":
    unittest.main()
