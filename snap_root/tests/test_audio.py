import unittest
import numpy as np
from hal9000_inside.audio import resample_pcm, SimpleVAD, HalfDuplexAudioController, BiquadHighPass

class TestAudio(unittest.TestCase):
    def test_resample_pcm(self):
        # 1 second of 44100Hz audio
        num_samples_in = 44100
        in_samples = np.sin(np.linspace(0, 100, num_samples_in) * 2 * np.pi).astype(np.int16)
        in_bytes = in_samples.tobytes()

        out_bytes = resample_pcm(in_bytes, 44100, 16000)
        out_samples = np.frombuffer(out_bytes, dtype=np.int16)
        self.assertEqual(len(out_samples), 16000)

    def test_biquad_highpass_attenuation(self):
        # Test 100Hz HPF at 16000Hz sampling rate
        hpf = BiquadHighPass(cutoff_hz=100.0, sample_rate=16000)
        duration_s = 0.5
        t = np.linspace(0, duration_s, int(16000 * duration_s), endpoint=False)

        # 30 Hz sub-rumble signal
        rumble = (np.sin(2 * np.pi * 30 * t) * 10000).astype(np.int16)
        rumble_filtered = np.frombuffer(hpf.process(rumble.tobytes()), dtype=np.int16)
        
        # Attenuation of 30 Hz relative to 100 Hz cutoff should be > 10 dB (> 3x amplitude reduction)
        rumble_in_amp = np.max(np.abs(rumble[1000:]))
        rumble_out_amp = np.max(np.abs(rumble_filtered[1000:]))
        self.assertLess(rumble_out_amp, rumble_in_amp * 0.35)

        # 400 Hz voice fundamental signal
        hpf.reset()
        voice = (np.sin(2 * np.pi * 400 * t) * 10000).astype(np.int16)
        voice_filtered = np.frombuffer(hpf.process(voice.tobytes()), dtype=np.int16)
        voice_in_amp = np.max(np.abs(voice[1000:]))
        voice_out_amp = np.max(np.abs(voice_filtered[1000:]))
        # 400 Hz passband should pass with minimal attenuation (> 90% amplitude preserved)
        self.assertGreater(voice_out_amp, voice_in_amp * 0.90)

    def test_vad_silence_vs_speech(self):
        vad = SimpleVAD(sample_rate=16000, frame_duration_ms=30)
        frame_len = int(16000 * 0.03) * 2

        # Silence frames to pass calibration
        silence_frame = b"\x00" * frame_len
        for _ in range(12):
            is_speech = vad.is_speech_frame(silence_frame)
            self.assertFalse(is_speech)

        # Loud tone frame in speech band (440 Hz)
        t = np.linspace(0, 0.03, int(16000 * 0.03), endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)
        tone_frame = tone.tobytes()
        is_speech = vad.is_speech_frame(tone_frame)
        self.assertTrue(is_speech)

    def test_vad_rumble_rejection(self):
        # With HPF enabled (default)
        vad_hpf = SimpleVAD(sample_rate=16000, frame_duration_ms=30, hpf_cutoff_hz=100.0, legacy_mode=False)
        # With legacy mode (no HPF, legacy threshold)
        vad_legacy = SimpleVAD(sample_rate=16000, frame_duration_ms=30, legacy_mode=True)

        t = np.linspace(0, 0.03, int(16000 * 0.03), endpoint=False)
        # Moderate 30 Hz chassis rumble
        rumble = (np.sin(2 * np.pi * 30 * t) * 1200).astype(np.int16).tobytes()

        # Legacy VAD calculates raw energy
        legacy_energy = vad_legacy.compute_energy(rumble)
        self.assertGreater(legacy_energy, 600.0)

        # Filtered VAD suppresses 30Hz energy significantly
        hpf_energy = vad_hpf.compute_energy(rumble)
        self.assertLess(hpf_energy, legacy_energy * 0.3)

    def test_vad_legacy_mode_rollback(self):
        vad = SimpleVAD(sample_rate=16000, legacy_mode=True)
        self.assertTrue(vad.legacy_mode)
        self.assertFalse(vad.hpf.enabled)
        self.assertEqual(vad._initial_calibration_frames, 0)

    def test_half_duplex_controller(self):
        ctrl = HalfDuplexAudioController(cooldown_seconds=0.1)
        self.assertTrue(ctrl.can_capture())

        # Start playback
        ctrl.notify_playback_started()
        self.assertFalse(ctrl.can_capture())

        # Stop playback (cooldown active)
        ctrl.notify_playback_stopped()
        self.assertFalse(ctrl.can_capture())

        import time
        time.sleep(0.12)  # wait for cooldown (0.1s) to expire

        # Mute test
        ctrl.set_muted(True)
        self.assertTrue(ctrl.muted)
        self.assertFalse(ctrl.can_capture())
        ctrl.set_muted(False)
        self.assertFalse(ctrl.muted)

        # Push to talk in non-auto mode
        ctrl.set_auto_mode(False)
        self.assertFalse(ctrl.can_capture())
        ctrl.set_push_to_talk(True)
        self.assertTrue(ctrl.push_to_talk_active)
        self.assertTrue(ctrl.can_capture())
        ctrl.set_push_to_talk(False)
        self.assertFalse(ctrl.push_to_talk_active)
        self.assertFalse(ctrl.can_capture())

        # Mute overrides PTT
        ctrl.set_push_to_talk(True)
        ctrl.set_muted(True)
        self.assertFalse(ctrl.can_capture())

if __name__ == "__main__":
    unittest.main()
