import numpy as np
import torch


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
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def log_mel_spectrogram(audio, mel_transform, n_frames):
    """Compute log-mel spectrogram for audio preprocessing.

    Args:
        audio: 1D tensor of audio samples (already padded/trimmed to chunk size)
        mel_transform: torchaudio.transforms.MelSpectrogram instance
        n_frames: expected number of mel frames to output

    Returns:
        Log-mel spectrogram tensor of shape (n_mels, n_frames).
    """
    mel = mel_transform(audio)
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
