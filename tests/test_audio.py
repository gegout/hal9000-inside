import unittest
import numpy as np
from hal9000_inside.audio import resample_pcm, SimpleVAD, HalfDuplexAudioController

class TestAudio(unittest.TestCase):
    def test_resample_pcm(self):
        # 1 second of 44100Hz audio
        num_samples_in = 44100
        in_samples = np.sin(np.linspace(0, 100, num_samples_in) * 2 * np.pi).astype(np.int16)
        in_bytes = in_samples.tobytes()

        out_bytes = resample_pcm(in_bytes, 44100, 16000)
        out_samples = np.frombuffer(out_bytes, dtype=np.int16)
        self.assertEqual(len(out_samples), 16000)

    def test_vad_silence_vs_speech(self):
        vad = SimpleVAD(sample_rate=16000, frame_duration_ms=30)
        frame_len = int(16000 * 0.03) * 2

        # Silence frame
        silence_frame = b"\x00" * frame_len
        is_speech = vad.is_speech_frame(silence_frame)
        self.assertFalse(is_speech)

        # Loud tone frame
        t = np.linspace(0, 0.03, int(16000 * 0.03))
        tone = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)
        tone_frame = tone.tobytes()
        is_speech = vad.is_speech_frame(tone_frame)
        self.assertTrue(is_speech)

    def test_half_duplex_controller(self):
        ctrl = HalfDuplexAudioController(cooldown_seconds=0.1)
        self.assertTrue(ctrl.can_capture())

        # Start playback
        ctrl.notify_playback_started()
        self.assertFalse(ctrl.can_capture())

        # Stop playback (cooldown active)
        ctrl.notify_playback_stopped()
        self.assertFalse(ctrl.can_capture())

        # Mute test
        ctrl.set_muted(True)
        self.assertFalse(ctrl.can_capture())

if __name__ == "__main__":
    unittest.main()
