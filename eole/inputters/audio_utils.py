import numpy as np
import torch


class MelSpectrogram(torch.nn.Module):
    """Pure-Torch mel spectrogram transform for EOLE audio preprocessing.

    This implements the mel transform parameter subset EOLE needs. Defaults
    track common mel-spectrogram behavior, while Whisper call sites pass their
    explicit Slaney settings.
    """

    HTK_MEL_SCALE = 2595.0
    HTK_MEL_BREAK_FREQUENCY = 700.0
    SLANEY_HZ_PER_MEL = 200.0 / 3
    SLANEY_MIN_LOG_HZ = 1000.0
    SLANEY_LOG_RANGE = 6.4
    SLANEY_LOG_STEPS = 27.0
    SLANEY_LOG_STEP = np.log(SLANEY_LOG_RANGE) / SLANEY_LOG_STEPS

    @classmethod
    def _hz_to_mel(cls, freq, mel_scale="slaney"):
        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale must be one of "slaney" or "htk"')
        if mel_scale == "htk":
            return cls.HTK_MEL_SCALE * np.log10(1.0 + freq / cls.HTK_MEL_BREAK_FREQUENCY)

        # Slaney's scale is linear below 1 kHz and logarithmic above it.
        mels = freq / cls.SLANEY_HZ_PER_MEL
        min_log_mel = cls.SLANEY_MIN_LOG_HZ / cls.SLANEY_HZ_PER_MEL
        if freq >= cls.SLANEY_MIN_LOG_HZ:
            mels = min_log_mel + np.log(freq / cls.SLANEY_MIN_LOG_HZ) / cls.SLANEY_LOG_STEP
        return mels

    @classmethod
    def _mel_to_hz(cls, mels, mel_scale="slaney"):
        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale must be one of "slaney" or "htk"')
        if mel_scale == "htk":
            return cls.HTK_MEL_BREAK_FREQUENCY * (10.0 ** (mels / cls.HTK_MEL_SCALE) - 1.0)

        # Inverse of Slaney's piecewise linear/log frequency mapping.
        freqs = cls.SLANEY_HZ_PER_MEL * mels
        min_log_mel = cls.SLANEY_MIN_LOG_HZ / cls.SLANEY_HZ_PER_MEL
        log_t = mels >= min_log_mel
        freqs[log_t] = cls.SLANEY_MIN_LOG_HZ * torch.exp(cls.SLANEY_LOG_STEP * (mels[log_t] - min_log_mel))
        return freqs

    @classmethod
    def _create_triangular_filterbank(cls, all_freqs, f_pts):
        # Return shape is (freq_bins, n_mels), matching TorchAudio's orientation
        # so a spectrogram shaped (..., frames, freq_bins) can be multiplied by it.
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
        zero = torch.zeros(1, dtype=all_freqs.dtype, device=all_freqs.device)
        down_slopes = -slopes[:, :-2] / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        return torch.max(zero, torch.min(down_slopes, up_slopes))

    @classmethod
    def _melscale_fbanks(cls, n_freqs, f_min, f_max, n_mels, sample_rate, norm="slaney", mel_scale="slaney"):
        # The mel helpers intentionally mirror the TorchAudio/librosa formulas
        # for the subset EOLE uses.
        all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
        m_min = cls._hz_to_mel(f_min, mel_scale=mel_scale)
        m_max = cls._hz_to_mel(f_max, mel_scale=mel_scale)
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)
        f_pts = cls._mel_to_hz(m_pts, mel_scale=mel_scale)
        fb = cls._create_triangular_filterbank(all_freqs, f_pts)

        if norm == "slaney":
            # Area-normalize each triangular filter by mel-band width. This is
            # required for parity with norm="slaney" feature extraction.
            enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
            fb *= enorm.unsqueeze(0)
        elif norm is not None:
            raise ValueError('norm must be one of None or "slaney"')
        return fb

    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        win_length=None,
        hop_length=None,
        f_min=0.0,
        f_max=None,
        pad=0,
        n_mels=128,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        onesided=True,
        norm=None,
        mel_scale="htk",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = True if onesided is None else onesided
        window_kwargs = {} if wkwargs is None else wkwargs
        self.register_buffer("window", window_fn(self.win_length, **window_kwargs), persistent=False)
        self.register_buffer(
            "fb",
            self._melscale_fbanks(
                n_fft // 2 + 1 if self.onesided else n_fft,
                f_min,
                sample_rate / 2.0 if f_max is None else f_max,
                n_mels,
                sample_rate,
                norm=norm,
                mel_scale=mel_scale,
            ),
            persistent=False,
        )

    def forward(self, waveform):
        # Buffers are created on CPU/float32; match the incoming waveform so the
        # transform works with alternate devices or dtypes.
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        fb = self.fb.to(device=waveform.device, dtype=waveform.dtype)
        if self.pad > 0:
            waveform = torch.nn.functional.pad(waveform, (self.pad, self.pad), mode="constant")
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        spec = spec.abs().pow(self.power)
        return torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)


def dynamic_time_warping(cost_matrix):
    """Classic DTW on a cost matrix.

    Uses partial numpy vectorization for row/diagonal minima, with a
    scalar inner loop due to the left-neighbor dependency.

    Args:
        cost_matrix: numpy array of shape (N, M) — cost of aligning
            text index i to time index j.

    Returns:
        (text_indices, time_indices): numpy arrays representing the
            optimal alignment path from (0,0) to (N-1, M-1).
    """
    N, M = cost_matrix.shape
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, N + 1):
        prev_row = dtw[i - 1, 1:]
        prev_diag = dtw[i - 1, :-1]
        row_cost = cost_matrix[i - 1]
        min_prev = np.minimum(prev_row, prev_diag)
        for j in range(1, M + 1):
            dtw[i, j] = row_cost[j - 1] + min(min_prev[j - 1], dtw[i, j - 1])
    i, j = N, M
    text_indices = []
    time_indices = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1]]
            argmin = np.argmin(candidates)
            if argmin == 0:
                i, j = i - 1, j - 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
    text_indices = np.array(text_indices[::-1])
    time_indices = np.array(time_indices[::-1])
    return text_indices, time_indices


def median_filter(inputs, filter_width=7):
    """Apply 1D median filter along the last dimension.

    Args:
        inputs: tensor of shape (..., T)
        filter_width: width of the median filter (odd integer)

    Returns:
        Filtered tensor of same shape.
    """
    if filter_width <= 1:
        return inputs
    pad = filter_width // 2
    padded = torch.nn.functional.pad(inputs, (pad, pad), mode="reflect")
    windows = padded.unfold(-1, filter_width, 1)
    return windows.median(dim=-1).values


def load_audio(audio_path, sample_rate=16000):
    """Load audio file and resample to target sample rate.

    Returns:
        1D float tensor of mono audio samples at the target sample rate.
    """
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(audio_path, sample_rate=sample_rate).get_all_samples()
    waveform = samples.data
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def log_mel_spectrogram(audio, mel_transform, n_frames):
    """Compute Whisper-style log-mel spectrogram for audio preprocessing.

    Args:
        audio: 1D tensor of audio samples (already padded/trimmed to chunk size)
        mel_transform: callable mel spectrogram transform
        n_frames: expected number of mel frames to output

    Returns:
        Log-mel spectrogram tensor of shape (n_mels, n_frames).
    """
    mel = mel_transform(audio)
    # With centered STFT, a 30s Whisper chunk produces 3001 frames. Whisper
    # models expect 3000 frames, so this truncation is intentional.
    mel = mel[:, :n_frames]
    # Whisper log-mel normalization (from OpenAI reference implementation):
    # clamp, log10, cap at 8 dB below peak, then shift/scale to ~[-1, 1]
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def tensorify_audio(minibatch, device):
    """Transform a batch of audio waveform examples into tensors.

    Args:
        minibatch: List of (example, index) tuples
        device: Target device for tensors

    Returns:
        Dictionary of batch tensors with audio-specific fields.
    """
    tensor_batch = {}
    examples = [ex for ex, _ in minibatch]
    indices = [idx for _, idx in minibatch]

    tensor_batch["src"] = examples[0]["src"]
    tensor_batch["src_type"] = "waveform"
    tensor_batch["srclen"] = torch.tensor([examples[0]["src"].shape[0]], dtype=torch.long)
    tensor_batch["prefix_len"] = None
    tensor_batch["images"] = None
    tensor_batch["left_pad"] = False
    tensor_batch["audio_file"] = [ex["audio_file"] for ex in examples]
    tensor_batch["ind_in_bucket"] = indices
    # cid/cid_line_number are optional metadata from corpus config
    tensor_batch["cid"] = [ex.get("cid") for ex in examples]
    tensor_batch["cid_line_number"] = [ex.get("cid_line_number") for ex in examples]
    return tensor_batch


def tensorify_audio_training(vocabs, minibatch, device):
    """Transform a batch of mel+text training examples into tensors.

    Args:
        vocabs: Vocabulary dictionaries
        minibatch: List of (example, index) tuples where example has
            src={src: mel_tensor, src_type: "mel"} and tgt={tgt_ids: [...]}
        device: Target device for tensors

    Returns:
        Dictionary of batch tensors for training.
    """
    from torch.nn.utils.rnn import pad_sequence
    from eole.constants import DefaultTokens

    tensor_batch = {}
    examples = [ex for ex, _ in minibatch]
    indices = [idx for _, idx in minibatch]

    mels = [ex["src"]["src"] for ex in examples]
    tensor_batch["src"] = torch.stack(mels, dim=0).to(device)
    tensor_batch["srclen"] = torch.tensor([mel.shape[-1] for mel in mels], dtype=torch.long, device=device)

    pad_token = vocabs["specials"].get("pad_token", DefaultTokens.PAD)
    tgt_pad_idx = vocabs["tgt"][pad_token]
    tgt_sequences = [ex["tgt"]["tgt_ids"] for ex in examples]
    tgt_tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in tgt_sequences]
    tensor_batch["tgt"] = pad_sequence(tgt_tensors, batch_first=True, padding_value=tgt_pad_idx)
    tensor_batch["tgtlen"] = torch.tensor([len(seq) for seq in tgt_sequences], dtype=torch.long, device=device)

    tensor_batch["prefix_len"] = None
    tensor_batch["images"] = None
    tensor_batch["left_pad"] = False
    tensor_batch["ind_in_bucket"] = indices
    tensor_batch["cid"] = [ex.get("cid") for ex in examples]
    tensor_batch["cid_line_number"] = [ex.get("cid_line_number") for ex in examples]
    tensor_batch["sco"] = torch.tensor([ex.get("sco", 1.0) for ex in examples], device=device)

    return tensor_batch
