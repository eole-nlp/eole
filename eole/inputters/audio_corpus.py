"""Audio corpus classes for inference and training data loading."""


class AudioCorpus(object):
    """Corpus that reads audio file paths (one per line from src)."""

    def __init__(self, name, src, is_train=False):
        self.id = name
        self.src = src
        self.is_train = is_train

    def load(self, offset=0, stride=1):
        if isinstance(self.src, list):
            for i, path in enumerate(self.src):
                if (i // stride) % stride == offset:
                    yield {"audio_path": path.strip()}
        else:
            with open(self.src, mode="r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if (i // stride) % stride == offset:
                        yield {"audio_path": line.strip()}

    def __str__(self):
        cls_name = type(self).__name__
        return f"{cls_name}({self.id}, {self.src})"


class AudioTextCorpus(object):
    """Corpus that reads parallel audio file paths and text transcriptions.

    path_src contains one audio file path per line,
    path_tgt contains the corresponding transcription text per line.
    """

    def __init__(self, name, path_src, path_tgt, path_sco=None):
        self.id = name
        self.src = path_src
        self.tgt = path_tgt
        self.sco = path_sco

    def load(self, offset=0, stride=1):
        from eole.inputters.text_corpus import exfile_open

        with open(self.src, mode="r", encoding="utf-8") as fs, open(
            self.tgt, mode="r", encoding="utf-8"
        ) as ft, exfile_open(self.sco, mode="rb") as fsco:
            for i, (sline, tline, scoline) in enumerate(zip(fs, ft, fsco)):
                if (i // stride) % stride == offset:
                    if scoline is not None:
                        scoline = float(scoline.strip().decode("utf-8"))
                    else:
                        scoline = 1.0
                    yield {
                        "audio_path": sline.strip(),
                        "tgt": tline.strip(),
                        "sco": scoline,
                    }

    def __str__(self):
        cls_name = type(self).__name__
        return f"{cls_name}({self.id}, {self.src}, {self.tgt})"


class AudioCorpusIterator(object):
    """Iterator that loads audio and yields raw waveforms for prediction.

    Yields one raw waveform per audio file. The seeking loop in
    AudioPredictor handles windowing and mel computation.
    """

    def __init__(
        self,
        corpus,
        transform,
        skip_empty_level="warning",
        stride=1,
        offset=0,
        is_train=False,
        sample_rate=16000,
    ):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset
        self.is_train = is_train
        self.sample_rate = sample_rate

    def _process(self, stream):
        from eole.inputters.audio_utils import load_audio

        for i, example in enumerate(stream):
            audio_path = example["audio_path"]
            line_number = i * self.stride + self.offset

            audio = load_audio(audio_path, sample_rate=self.sample_rate)

            yield {
                "src": audio,
                "src_type": "waveform",
                "tgt": None,
                "audio_file": audio_path,
                "cid": self.cid,
                "cid_line_number": line_number,
            }, self.transform, self.cid

    def __iter__(self):
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        yield from self._process(corpus_stream)


class AudioTextCorpusIterator(object):
    """Iterator that loads audio, computes mel spectrograms, and pairs with text.

    For training: loads audio, pads/truncates to chunk_length, computes
    log-mel spectrogram, and yields with raw text transcription for
    downstream tokenization via transforms.
    """

    def __init__(
        self,
        corpus,
        transform,
        skip_empty_level="warning",
        stride=1,
        offset=0,
        is_train=False,
        sample_rate=16000,
        num_mel_bins=80,
        n_fft=400,
        hop_length=160,
        chunk_length=30,
    ):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.chunk_samples = chunk_length * sample_rate
        self.n_frames = self.chunk_samples // hop_length
        self._mel_transform = None

    def _get_mel_transform(self):
        if self._mel_transform is None:
            import torchaudio.transforms as T

            self._mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.num_mel_bins,
                power=2.0,
                norm="slaney",
                mel_scale="slaney",
                f_max=self.sample_rate / 2.0,
            )
        return self._mel_transform

    def _process(self, stream):
        import torch
        from eole.inputters.audio_utils import load_audio, log_mel_spectrogram

        for i, example in enumerate(stream):
            audio_path = example["audio_path"]
            line_number = i * self.stride + self.offset

            audio = load_audio(audio_path, sample_rate=self.sample_rate)

            if audio.shape[0] < self.chunk_samples:
                audio = torch.nn.functional.pad(audio, (0, self.chunk_samples - audio.shape[0]))
            else:
                audio = audio[: self.chunk_samples]

            mel = log_mel_spectrogram(audio, self._get_mel_transform(), self.n_frames)

            yield {
                "mel": mel,
                "src": "",
                "tgt": example["tgt"],
                "sco": example.get("sco", 1.0),
                "cid": self.cid,
                "cid_line_number": line_number,
            }, self.transform, self.cid

    def __iter__(self):
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        yield from self._process(corpus_stream)
