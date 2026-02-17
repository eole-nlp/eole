"""AudioTranslator: Audio prediction with forced decoder IDs,
token suppression, and sequential timestamp-seeking for long audio."""

import itertools
import json
import os
import torch
import torchaudio.transforms as T
from time import time

from eole.inputters.audio_utils import dynamic_time_warping, log_mel_spectrogram, median_filter
from eole.predict.translator import Translator


class AudioTranslator(Translator):
    """Translator subclass for audio encoder-decoder models.

    Adds:
    - Token suppression (suppress_tokens from eole config)
    - Forced decoder prefix (SOT, language, task tokens)
    - Sequential timestamp-seeking: decodes audio windows using timestamp
      tokens to determine seek advancement
    - Configurable timestamp output: none (plain text), segment (JSON), word
    """

    def __init__(
        self,
        model,
        vocabs,
        config,
        model_config,
        device_id=0,
        global_scorer=None,
        report_score=True,
        logger=None,
        return_gold_log_probs=False,
    ):
        super().__init__(
            model,
            vocabs,
            config,
            model_config,
            device_id=device_id,
            global_scorer=global_scorer,
            report_score=report_score,
            logger=logger,
            return_gold_log_probs=return_gold_log_probs,
        )

        encoder_cfg = model_config.encoder
        self.sample_rate = encoder_cfg.sample_rate
        self.chunk_length = encoder_cfg.chunk_length
        self.n_fft = encoder_cfg.n_fft
        self.hop_length = encoder_cfg.hop_length
        self.num_mel_bins = encoder_cfg.num_mel_bins
        self.timestamp_resolution = encoder_cfg.timestamp_resolution
        self.chunk_samples = self.chunk_length * self.sample_rate
        self.n_frames = self.chunk_samples // self.hop_length
        self.timestamps_output = getattr(config, "timestamps", "none")

        # Mel transform lives on CPU; output is moved to device in the seeking loop
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

        if config.data_type != "audio":
            raise ValueError(
                f"AudioTranslator requires data_type='audio', got '{config.data_type}'. Check your model config."
            )

        if config.batch_size > 1:
            self._log(
                "INFO: batch_size > 1 is ignored for audio; "
                "files are processed sequentially in timestamp-seeking mode."
            )

        self.suppress_tokens = getattr(model_config, "suppress_tokens", None) or []
        self.begin_suppress_tokens = getattr(model_config, "begin_suppress_tokens", None) or []
        self.no_timestamps_token_id = getattr(model_config, "no_timestamps_token_id", None)
        self.word_timestamp_heads = getattr(model_config, "word_timestamp_heads", None)
        self.median_filter_width = getattr(model_config, "median_filter_width", 7)

        self._tokenizer = None
        try:
            from tokenizers import Tokenizer

            model_path = config.get_model_path()
            tokenizer_path = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tokenizer_path):
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
        except ImportError:
            pass

        # Decoder prefix: [startofprev, prompt..., SOT, lang?, task?]
        self._decoder_prefix_ids = []

        initial_prompt = getattr(config, "initial_prompt", None)
        if initial_prompt:
            if self._tokenizer is None:
                raise ValueError(
                    "initial_prompt requires a tokenizer.json in the model "
                    "directory. Ensure the model was converted with the "
                    "tokenizer file."
                )
            startofprev_id = self._tgt_vocab.lookup_token("<|startofprev|>")
            self._decoder_prefix_ids.append(startofprev_id)
            prompt_ids = self._tokenizer.encode(initial_prompt).ids
            self._decoder_prefix_ids.extend(prompt_ids)

        sot_start_idx = len(self._decoder_prefix_ids)
        self._decoder_prefix_ids.append(self._tgt_start_with)

        language = getattr(config, "language", None)
        if language:
            lang_token = f"<|{language}|>"
            lang_id = self._tgt_vocab.lookup_token(lang_token)
            unk_id = self._tgt_vocab.lookup_token("<unk>")
            if lang_id == unk_id:
                raise ValueError(
                    f"Language token {lang_token} not found in vocabulary. "
                    "Check the language code or use a multilingual model."
                )
            self._decoder_prefix_ids.append(lang_id)

        task = getattr(config, "task", None)
        if task:
            task_token = f"<|{task}|>"
            task_id = self._tgt_vocab.lookup_token(task_token)
            unk_id = self._tgt_vocab.lookup_token("<unk>")
            if task_id == unk_id:
                raise ValueError(f"Task token {task_token} not found in vocabulary.")
            self._decoder_prefix_ids.append(task_id)

        # Store the SOT sequence for dynamic prefix construction
        self._sot_sequence = self._decoder_prefix_ids[sot_start_idx:]

        self.condition_on_previous_text = getattr(config, "condition_on_previous_text", False)
        self._startofprev_id = self._tgt_vocab.lookup_token("<|startofprev|>")
        self._max_prompt_length = self.max_length // 2 - 1
        self._initial_prompt_tokens = []
        if initial_prompt and self._tokenizer:
            self._initial_prompt_tokens = list(self._tokenizer.encode(initial_prompt).ids)

        # Keep a copy of the static prefix for restoring between chunks
        self._static_prefix_ids = list(self._decoder_prefix_ids)

        # Override parent defaults for audio:
        # - start token is the first token of the decoder prefix
        # - src dim 1 is mel bins not sequence length, so ratio is meaningless
        self._tgt_start_with = self._decoder_prefix_ids[0]
        self.max_length_ratio = 0

    def predict_batch(self, batch, attn_debug):
        """Override to inject decoder prefix tensor into batch."""
        if "tgt" not in batch and len(self._decoder_prefix_ids) > 1:
            device = batch["src"].device
            batch_size = len(batch["srclen"])
            batch["tgt"] = torch.tensor(
                [self._decoder_prefix_ids] * batch_size,
                dtype=torch.long,
                device=device,
            )
        self.tgt_file_prefix = "tgt" in batch
        return Translator.predict_batch(self, batch, attn_debug)

    def _decode_and_generate(
        self,
        decoder_in,
        enc_out,
        src_len,
        step=None,
        return_attn=False,
        images=None,
    ):
        """Override to apply token suppression after getting log_probs."""
        log_probs, attn = super()._decode_and_generate(
            decoder_in,
            enc_out,
            src_len,
            step=step,
            return_attn=return_attn,
            images=images,
        )

        # Suppression is skipped in two cases:
        # 1. During prefix-forcing (step < prefix_len): beam search forces
        #    tokens via -10000 penalty; -inf here would override it.
        # 2. During gold scoring (step is None): log_probs is 3D
        #    (batch, tgt_len, vocab) so token-ID indexing hits the wrong dim.
        # The step is not None guard is needed because prefix_active is also
        # False after the prefix, so it alone can't distinguish case 2.
        prefix_len = len(self._decoder_prefix_ids) - 1
        prefix_active = step is not None and step < prefix_len

        if self.suppress_tokens and not prefix_active and step is not None:
            log_probs[:, self.suppress_tokens] = float("-inf")

        if step is not None and step == prefix_len and self.begin_suppress_tokens:
            log_probs[:, self.begin_suppress_tokens] = float("-inf")

        return log_probs, attn

    def _predict(
        self,
        infer_iter,
        transform=None,
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        """Override _predict to dispatch between timestamp-seeking and
        fallback batch modes for audio data."""
        if self.data_type != "audio":
            return super()._predict(
                infer_iter,
                transform=transform,
                attn_debug=attn_debug,
                align_debug=align_debug,
                phrase_table=phrase_table,
            )

        all_scores = []
        all_estim = []
        all_predictions = []

        start_time = time()
        self.step0_time = []
        self.warmup_time = []
        device = next(self.model.parameters()).device

        for batch, bucket_idx in infer_iter:
            if batch.get("src_type") == "waveform":
                waveform = batch["src"]
                segments, word_segments = self._predict_with_timestamps(waveform, device)
                if segments:
                    avg_score = sum(seg["score"] for seg in segments) / len(segments)
                else:
                    avg_score = 0.0

                if self.timestamps_output == "segment":
                    all_predictions.append([json.dumps(segments)])
                elif self.timestamps_output == "word":
                    if self.word_timestamp_heads is None:
                        raise ValueError(
                            "Word-level timestamps require word_timestamp_heads "
                            "in the model config. This model may not "
                            "support word-level timestamps."
                        )
                    all_predictions.append([json.dumps(word_segments)])
                else:
                    text = " ".join(seg["text"] for seg in segments)
                    all_predictions.append([text])
                all_scores.append([avg_score])
                all_estim.append([1.0])

                if self.verbose:
                    self._log(f"Transcribed {len(segments)} segments from {batch.get('audio_file', ['?'])[0]}")
            else:
                # Prepend the consumed batch back onto the iterator
                restored_iter = itertools.chain([(batch, bucket_idx)], infer_iter)
                return super()._predict(
                    restored_iter,
                    transform=transform,
                    attn_debug=attn_debug,
                    align_debug=align_debug,
                    phrase_table=phrase_table,
                )

        end_time = time()

        if self.report_score and all_scores:
            pred_score_total = sum(s[0] for s in all_scores)
            msg = self._report_score("PRED", pred_score_total, len(all_scores))
            self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            if len(all_predictions) > 0:
                self._log(f"Prediction time (s): {total_time:.2f}")

        return all_scores, all_estim, all_predictions

    def _predict_with_timestamps(self, waveform, device):
        """Sequential timestamp-seeking transcription.

        Decodes audio windows, using timestamp tokens to determine
        how far to advance the seek position.

        Args:
            waveform: 1D float tensor of audio samples
            device: torch device for model inference

        Returns:
            (all_segments, word_segments) where:
            - all_segments: [{"start": float, "end": float,
                              "text": str, "score": float}, ...]
            - word_segments: [{"text": str, "start": float,
                               "end": float}, ...] (empty if word
                              timestamps not requested)
        """
        if self.no_timestamps_token_id is None:
            raise ValueError("Timestamp-seeking mode requires no_timestamps_token_id in the model config.")
        token_beg = self.no_timestamps_token_id + 1
        do_word_timestamps = self.timestamps_output == "word" and self.word_timestamp_heads is not None

        total_samples = waveform.shape[0]
        seek = 0
        all_segments = []
        word_segments = []
        all_tokens = list(self._initial_prompt_tokens) if self.condition_on_previous_text else []

        while seek < total_samples:
            chunk = waveform[seek : seek + self.chunk_samples]
            if chunk.shape[0] < self.chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, self.chunk_samples - chunk.shape[0]))

            mel = log_mel_spectrogram(
                chunk,
                self._mel_transform,
                n_frames=self.n_frames,
            )

            batch = {
                "src": mel.unsqueeze(0).to(device=device, dtype=next(self.model.parameters()).dtype),
                "srclen": torch.tensor([mel.shape[-1]], device=device),
            }

            # Build dynamic prefix if conditioning on previous text
            if self.condition_on_previous_text:
                if all_tokens:
                    prefix = self._build_conditioned_prefix(all_tokens)
                    batch["tgt"] = torch.tensor([prefix], dtype=torch.long, device=device)
                    self._decoder_prefix_ids = prefix
                    self._tgt_start_with = self._startofprev_id
                else:
                    self._decoder_prefix_ids = list(self._static_prefix_ids)
                    self._tgt_start_with = self._static_prefix_ids[0]

            with torch.no_grad():
                results = self.predict_batch(batch, attn_debug=False)

            token_ids = results["predictions"][0][0].tolist()
            # BeamSearch strips only position 0; remaining prefix tokens
            # are still at the front (_decoder_prefix_ids is synced to the
            # active prefix when condition_on_previous_text is enabled)
            prefix_strip = len(self._decoder_prefix_ids) - 1
            token_ids = token_ids[prefix_strip:]
            score = results["scores"][0][0]

            segments, seek_delta = self._parse_timestamp_tokens(token_ids, seek, token_beg)
            for seg in segments:
                seg["score"] = score
            all_segments.extend(segments)

            if do_word_timestamps:
                chunk_end_time = seek / self.sample_rate + float(self.chunk_length)
                word_segs = self._extract_word_timestamps(token_ids, mel, seek, device, chunk_end_time)
                word_segments.extend(word_segs)

            # Accumulate tokens for next chunk's conditioning
            # Filter out EOS — target_prefixing skips forcing EOS tokens,
            # so an EOS in the middle of the prefix would let the model
            # generate freely (likely ending the beam immediately).
            if self.condition_on_previous_text:
                all_tokens.extend(tid for tid in token_ids if tid not in self._tgt_eos_idx)

            if seek_delta <= 0:
                seek_delta = self.chunk_samples
            seek += seek_delta

        # Restore static prefix state so the next file starts clean
        if self.condition_on_previous_text:
            self._decoder_prefix_ids = list(self._static_prefix_ids)
            self._tgt_start_with = self._static_prefix_ids[0]

        # Clamp the last segment's end time to actual audio duration
        audio_duration = round(total_samples / self.sample_rate, 2)
        if all_segments and all_segments[-1]["end"] > audio_duration:
            all_segments[-1]["end"] = audio_duration
        if word_segments and word_segments[-1]["end"] > audio_duration:
            word_segments[-1]["end"] = audio_duration

        return all_segments, word_segments

    def _build_conditioned_prefix(self, prev_tokens):
        """Build decoder prefix with previous text conditioning.

        Returns: [startofprev, prev_tokens[-max:], SOT, lang?, task?, ...]
        """
        truncated = prev_tokens[-self._max_prompt_length :]
        return [self._startofprev_id] + truncated + self._sot_sequence

    def _parse_timestamp_tokens(self, token_ids, seek_samples, token_beg):
        """Parse timestamp tokens from decoder output into segments.

        Audio models generate: <|0.00|> text text <|5.00|> text <|10.00|>
        Each timestamp pair delimits a segment. The last timestamp
        determines the seek advancement for the next window.

        Args:
            token_ids: List of decoded token IDs
            seek_samples: Current seek position in samples
            token_beg: First timestamp token ID

        Returns:
            segments: list of {"start": float, "end": float, "text": str}
            seek_delta_samples: how many audio samples to advance
        """
        seek_offset = seek_samples / self.sample_rate
        segments = []
        current_text_tokens = []
        segment_start_time = seek_offset
        last_timestamp_id = None

        for tid in token_ids:
            if tid >= token_beg:
                timestamp = (tid - token_beg) * self.timestamp_resolution + seek_offset
                if current_text_tokens:
                    text = self._decode_token_ids(current_text_tokens)
                    if text.strip():
                        segments.append(
                            {
                                "start": round(segment_start_time, 2),
                                "end": round(timestamp, 2),
                                "text": text.strip(),
                            }
                        )
                    current_text_tokens = []
                segment_start_time = timestamp
                last_timestamp_id = tid
            elif tid in self._tgt_eos_idx:
                break
            else:
                current_text_tokens.append(tid)

        if current_text_tokens:
            text = self._decode_token_ids(current_text_tokens)
            if text.strip():
                segments.append(
                    {
                        "start": round(segment_start_time, 2),
                        "end": round(seek_offset + float(self.chunk_length), 2),
                        "text": text.strip(),
                    }
                )

        if last_timestamp_id is not None:
            token_offset = last_timestamp_id - token_beg
            seek_delta_samples = int(token_offset * self.timestamp_resolution * self.sample_rate)
        else:
            seek_delta_samples = self.chunk_samples

        return segments, seek_delta_samples

    def _decode_token_ids(self, token_ids):
        """Decode token IDs to text using the HuggingFace tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return " ".join(
            self._tgt_vocab.lookup_index(t) for t in token_ids if not self._tgt_vocab.lookup_index(t).startswith("<|")
        )

    def _collect_cross_attention(self, token_ids, mel, device):
        """Run a teacher-forcing decoder pass to collect cross-attention.

        Args:
            token_ids: list of decoded token IDs (text + timestamp tokens)
            mel: mel spectrogram tensor (n_mels, T)
            device: torch device

        Returns:
            List of per-layer cross-attention tensors, each
            (1, heads, tgt_len, src_len).
        """
        enc_out, _ = self.model.encoder(mel.unsqueeze(0).to(device=device, dtype=next(self.model.parameters()).dtype))

        tgt_ids = list(self._static_prefix_ids) + token_ids
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)

        emb = self.model.tgt_emb(tgt)

        # Disable KV cache for full-sequence teacher-forcing pass,
        # then restore clean state so predict_batch can re-init cache
        self.model.decoder._disable_cache()
        try:
            src_pad_mask = torch.zeros((1, 1, enc_out.size(1)), dtype=torch.bool, device=device)
            tgt_pad_mask = torch.zeros((1, 1, tgt.size(1)), dtype=torch.bool, device=device)
            _, attns = self.model.decoder(
                emb,
                enc_out=enc_out,
                src_pad_mask=src_pad_mask,
                tgt_pad_mask=tgt_pad_mask,
                return_attn=True,
                collect_cross_attns=True,
            )
        finally:
            self.model.decoder._disable_cache()

        return attns.get("cross_attns", [])

    def _extract_word_timestamps(self, token_ids, mel, seek_samples, device, chunk_end_time):
        """Extract word-level timestamps using cross-attention DTW.

        Args:
            token_ids: list of decoded token IDs from beam search
            mel: mel spectrogram tensor (n_mels, T)
            seek_samples: current seek position in samples
            device: torch device
            chunk_end_time: end time of current chunk in seconds

        Returns:
            List of word dicts: [{"text": str, "start": float, "end": float}]
        """
        token_beg = self.no_timestamps_token_id + 1
        seek_offset = seek_samples / self.sample_rate

        text_token_ids = []
        text_token_indices = []
        for i, tid in enumerate(token_ids):
            if tid in self._tgt_eos_idx:
                break
            if tid >= token_beg:
                continue
            tok = self._tgt_vocab.lookup_index(tid)
            if tok.startswith("<|") and tok.endswith("|>"):
                continue
            text_token_ids.append(tid)
            text_token_indices.append(i)

        if not text_token_ids:
            return []

        with torch.no_grad():
            cross_attns = self._collect_cross_attention(token_ids, mel, device)

        if not cross_attns:
            return []

        prefix_len = len(self._static_prefix_ids)
        weights_list = []
        for layer_idx, head_idx in self.word_timestamp_heads:
            if layer_idx < len(cross_attns):
                w = cross_attns[layer_idx][0, head_idx]
                weights_list.append(w)

        if not weights_list:
            return []

        weights = torch.stack(weights_list)
        weights = weights[:, prefix_len:, :]
        std, mean = torch.std_mean(weights, dim=-2, keepdim=True)
        weights = (weights - mean) / (std + 1e-6)

        weights = median_filter(weights, self.median_filter_width)
        weights = weights.mean(dim=0)

        text_rows = torch.tensor(text_token_indices, device=device)
        weights = weights[text_rows]
        cost_matrix = -weights.cpu().float().numpy()
        text_indices, time_indices = dynamic_time_warping(cost_matrix)

        token_timestamps = []
        prev_text_idx = -1
        for path_pos in range(len(text_indices)):
            if text_indices[path_pos] != prev_text_idx:
                t = seek_offset + time_indices[path_pos] * self.timestamp_resolution
                token_timestamps.append(t)
                prev_text_idx = text_indices[path_pos]

        while len(token_timestamps) < len(text_token_ids):
            token_timestamps.append(token_timestamps[-1] if token_timestamps else seek_offset)

        return self._group_tokens_to_words(text_token_ids, token_timestamps, chunk_end_time)

    def _group_tokens_to_words(self, token_ids, token_timestamps, chunk_end_time):
        """Group sub-word tokens into words using space boundaries.

        Args:
            token_ids: list of text token IDs
            token_timestamps: list of start times for each token
            chunk_end_time: end time of current chunk

        Returns:
            List of word dicts: [{"text": str, "start": float, "end": float}]
        """
        if not token_ids:
            return []

        token_texts = []
        for tid in token_ids:
            if self._tokenizer is not None:
                text = self._tokenizer.decode([tid], skip_special_tokens=False)
            else:
                text = self._tgt_vocab.lookup_index(tid)
            token_texts.append(text)

        words = []
        current_word = ""
        current_start = token_timestamps[0]

        for text, ts in zip(token_texts, token_timestamps):
            if text.startswith(" ") and current_word:
                next_start = ts
                words.append(
                    {
                        "text": current_word.strip(),
                        "start": round(current_start, 2),
                        "end": round(next_start, 2),
                    }
                )
                current_word = text
                current_start = ts
            else:
                current_word += text

        if current_word.strip():
            words.append(
                {
                    "text": current_word.strip(),
                    "start": round(current_start, 2),
                    "end": round(chunk_end_time, 2),
                }
            )

        return words
