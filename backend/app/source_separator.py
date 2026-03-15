import os
import numpy as np
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


class SourceSeparator:
    """Separate vocals and instrumentals using Demucs"""

    def __init__(self):
        print("🎵 Loading Demucs model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = get_model('htdemucs')
        self.model.to(self.device)
        self.model.eval()
        print(f"   ✓ Demucs loaded on {self.device}")

    def separate(self, audio_source, output_dir: str, sr_hint: int = 44100):
        """
        Separate audio into vocals and instrumental.

        Args:
            audio_source: Either a file path (str) or a pre-loaded stereo/mono
                          NumPy array with shape [samples] or [2, samples].
                          Passing an array avoids a redundant disk read.
            output_dir:   Directory for any auxiliary outputs (not used but
                          kept for API compatibility).
            sr_hint:      Sample rate of the provided array. Used only when
                          audio_source is a NumPy array.

        Returns:
            {'vocals': np.array, 'instrumental': np.array, 'sr': int}
        """
        if isinstance(audio_source, np.ndarray):
            waveform_np = audio_source
            sr = sr_hint
            label = "in-memory array"
        else:
            import librosa
            label = os.path.basename(audio_source)
            # Load as stereo for Demucs (more accurate than upsampling mono)
            waveform_np, sr = librosa.load(audio_source, sr=44100, mono=False)

        print(f"   Separating: {label}")

        # Convert to torch tensor, ensuring shape [2, samples]
        if waveform_np.ndim == 1:
            waveform = torch.from_numpy(np.stack([waveform_np, waveform_np]))
        else:
            waveform = torch.from_numpy(waveform_np)

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Resample if needed
        if sr != self.model.samplerate:
            print(f"   Resampling {sr} → {self.model.samplerate}")
            resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
            waveform = resampler(waveform)
            sr = self.model.samplerate

        # Apply Demucs
        print(f"   Processing with Demucs (this takes ~30-60s)...")
        with torch.no_grad():
            waveform = waveform.to(self.device)
            sources = apply_model(self.model, waveform[None], device=self.device)[0]

        # Demucs outputs: [drums, bass, other, vocals]
        sources = sources.cpu().numpy()

        drums = np.mean(sources[0], axis=0)
        bass = np.mean(sources[1], axis=0)
        other = np.mean(sources[2], axis=0)
        vocals = np.mean(sources[3], axis=0)

        instrumental = drums + bass + other

        print(f"   ✓ Separated successfully!")

        return {
            'vocals': vocals,
            'instrumental': instrumental,
            'sr': sr
        }
