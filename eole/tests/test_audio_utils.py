import unittest
import wave
from tempfile import NamedTemporaryFile

import torch

from eole.inputters.audio_utils import MelSpectrogram, load_audio, log_mel_spectrogram


class TestAudioUtils(unittest.TestCase):
    def _write_stereo_wav(self, path, sample_rate=22050, duration_seconds=0.25):
        samples = int(sample_rate * duration_seconds)
        frames = bytearray()
        for i in range(samples):
            left = int(12000 * torch.sin(torch.tensor(2 * torch.pi * 440 * i / sample_rate)).item())
            right = int(8000 * torch.sin(torch.tensor(2 * torch.pi * 880 * i / sample_rate)).item())
            frames.extend(left.to_bytes(2, byteorder="little", signed=True))
            frames.extend(right.to_bytes(2, byteorder="little", signed=True))

        with wave.open(path, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(frames))

    def test_load_audio_decodes_resamples_and_downmixes(self):
        try:
            from torchcodec.decoders import AudioDecoder  # noqa: F401
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"torchcodec is not available in this environment: {exc}")

        with NamedTemporaryFile(suffix=".wav") as tmp:
            self._write_stereo_wav(tmp.name)

            audio = load_audio(tmp.name, sample_rate=16000)

        self.assertEqual(audio.dim(), 1)
        self.assertEqual(audio.dtype, torch.float32)
        self.assertTrue(torch.isfinite(audio).all())
        self.assertGreater(audio.abs().max().item(), 0.0)
        self.assertAlmostEqual(audio.shape[0], 4000, delta=2)

    def test_mel_spectrogram_default_shape(self):
        audio = torch.zeros(16000)
        mel_transform = MelSpectrogram()

        mel = mel_transform(audio)

        self.assertEqual(tuple(mel.shape), (128, 81))
        self.assertEqual(mel.dtype, torch.float32)

    def test_mel_spectrogram_whisper_shape(self):
        audio = torch.sin(2 * torch.pi * 440 * torch.arange(30 * 16000) / 16000)
        mel_transform = MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            f_max=8000.0,
        )

        mel = mel_transform(audio)
        log_mel = log_mel_spectrogram(audio, mel_transform, n_frames=3000)

        self.assertEqual(tuple(mel.shape), (80, 3001))
        self.assertEqual(tuple(log_mel.shape), (80, 3000))
        self.assertEqual(log_mel.dtype, torch.float32)
        self.assertTrue(torch.isfinite(log_mel).all())
        self.assertGreater(log_mel.std().item(), 0.0)

    def test_mel_spectrogram_random_audio_is_finite(self):
        generator = torch.Generator().manual_seed(0)
        audio = torch.randn(30 * 16000, generator=generator) * 0.05
        mel_transform = MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            f_max=8000.0,
        )

        log_mel = log_mel_spectrogram(audio, mel_transform, n_frames=3000)

        self.assertEqual(tuple(log_mel.shape), (80, 3000))
        self.assertTrue(torch.isfinite(log_mel).all())
        self.assertGreater(log_mel.std().item(), 0.0)

    def test_mel_spectrogram_supported_options(self):
        audio = torch.zeros(16000)
        mel_transform = MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=128,
            f_min=20.0,
            f_max=7600.0,
            pad=2,
            n_mels=40,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )

        mel = mel_transform(audio)

        self.assertEqual(mel.shape[0], 40)
        self.assertEqual(mel.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
