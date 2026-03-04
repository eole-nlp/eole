#!/usr/bin/env python
"""Translator Class and builder"""

import torch
from torch.nn.utils.rnn import pad_sequence
from eole.predict.beam_search import BeamSearch
from eole.predict.greedy_search import GreedySearch
from eole.utils.misc import tile, sequence_mask
from eole.utils.alignment import extract_alignment
from eole.predict.inference import Inference
from eole import EOLE_TORCH_COMPILE, EOLE_COMPILE_MODE
from time import time


class Translator(Inference):
    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """

        # (0) add BOS and padding to tgt prediction
        if "tgt" in batch.keys() and self.gold_align:
            self._log("Computing alignments with gold target")
            batch_tgt_idxs = batch["tgt"].transpose(1, 2)
        else:
            batch_tgt_idxs = self._align_pad_prediction(predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx)

        tgt_mask = (
            batch_tgt_idxs.eq(self._tgt_pad_idx)
            | batch_tgt_idxs.eq(self._tgt_bos_idx)
            | torch.any(
                torch.stack([batch_tgt_idxs.eq(eos_idx) for eos_idx in self._tgt_eos_idx]),
                dim=0,
            )
        )

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, enc_out, src_len = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(batch * n_best, src_len)``
        src = tile(src, n_best)
        if enc_states is not None:
            # Quick fix. Transformers return None as enc_states.
            # enc_states are only used later on to init decoder's state
            # but are never used in Transformer decoder, so we can skip
            enc_states = tile(enc_states, n_best)
        if isinstance(enc_out, tuple):
            enc_out = tuple(tile(x, n_best) for x in enc_out)
        else:
            enc_out = tile(enc_out, n_best)
        src_len = tile(src_len, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(enc_out=enc_out, enc_states=enc_states)

        # (4) reshape and apply pad masking in the target sequence
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1))
        src_pad_mask = src.eq(self._src_pad_idx).unsqueeze(1)
        tgt_pad_mask = tgt[:, :-1].eq(self._tgt_pad_idx).unsqueeze(1)
        dec_in = tgt[:, :-1]
        _, attns = self.model.decoder(
            self.model.tgt_emb(dec_in),
            enc_out=enc_out,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
            with_align=True,
        )

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignment = extract_alignment(alignment_attn, prediction_mask, src_len, n_best)
        return alignment

    def predict_batch(self, batch, attn_debug):
        """Translate a batch of sentences."""
        if self.max_length_ratio > 0:
            max_length = int(min(self.max_length, batch["src"].size(1) * self.max_length_ratio + 5))
        else:
            max_length = self.max_length
        with torch.no_grad():
            if self.top_k != 0 or self.top_p != 0:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    batch_size=len(batch["srclen"]),
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                    add_estimator=self.add_estimator,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=len(batch["srclen"]),
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                    add_estimator=self.add_estimator,
                )
            return self._predict_batch_with_strategy(batch, decode_strategy)

    def _run_encoder(self, batch):
        src = batch["src"]
        src_len = batch["srclen"]
        batch_size = len(batch["srclen"])

        if self.model.src_emb is not None:
            emb = self.model.src_emb(src)
            pad_mask = src.eq(self._src_pad_idx).unsqueeze(1)  # [B, 1, T_src]
            enc_out, enc_final_hs = self.model.encoder(emb, pad_mask=pad_mask)
        else:
            enc_out, enc_final_hs = self.model.encoder(src)
            # Encoder output length may differ from input (e.g. conv downsampling)
            src_len = torch.full((batch_size,), enc_out.size(1), dtype=torch.long, device=enc_out.device)

        if src_len is None:
            assert not isinstance(enc_out, tuple), "Ensemble decoding only supported for text data"
            src_len = torch.Tensor(batch_size).type_as(enc_out).long().fill_(enc_out.size(1))
        return src, enc_final_hs, enc_out, src_len

    def _predict_batch_with_strategy(self, batch, decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        if self.dynamic_shapes is not None:
            decode_strategy.static_batch_size = not self.dynamic_shapes
        else:
            decode_strategy.static_batch_size = EOLE_TORCH_COMPILE
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        batch_size = len(batch["srclen"])

        # (1) Run the encoder on the src.
        if self.report_time:
            torch.cuda.synchronize()
            beg_time = time()
        src, enc_final_hs, enc_out, src_len = self._run_encoder(batch)
        self.model.decoder.init_state(enc_out=enc_out, enc_final_hs=enc_final_hs)

        gold_score, gold_log_probs = self._gold_score(
            batch,
            enc_out,
            src_len,
            enc_final_hs,
            batch_size,
            src,
        )

        # (2) prep decode_strategy. Possibly repeat src objects.
        target_prefix = batch["tgt"] if self.tgt_file_prefix else None
        fn_tile, enc_out = decode_strategy.initialize(enc_out, src_len, target_prefix=target_prefix)

        if fn_tile is not None:
            self.model.decoder.map_state(fn_tile)

        # (3) Begin decoding step by step:
        # save the initial encoder out for later estimator
        if torch.is_tensor(enc_out):
            enc_out2 = enc_out.clone()
            src_len2 = decode_strategy.src_len.clone()
        elif isinstance(enc_out, tuple):
            enc_out2 = tuple(tensor.clone() for tensor in enc_out)
            src_len2 = tuple(tensor.clone() for tensor in decode_strategy.src_len)
        else:
            raise TypeError("enc_out must be either a tensor or a tuple of tensors")

        if self.report_time:
            torch.cuda.synchronize()
            self.step0_time.append(time() - beg_time)

        # (4) warmup for Torch compile
        # use the current batch to generate the decode graph (B, 1)
        # we need proper set up to run the forward pass of the decoder or decoder layer
        if EOLE_TORCH_COMPILE:
            assert not isinstance(enc_out, tuple), "torch.compile does not work with Ensembles"
            start_wu = time()
            self._log("Warmup started")
            decoder_input = decode_strategy.current_predictions.view(-1, 1)
            emb = self.model.tgt_emb(decoder_input)
            tgt_pad_mask = decoder_input.eq(self._tgt_pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
            src_max_len = enc_out.size(1)
            src_pad_mask = sequence_mask(decode_strategy.src_len, src_max_len).unsqueeze(1)  # [B, 1, T_src]
            self.model.decoder._init_cache(emb, tgt_pad_mask, enc_out=enc_out)
            self.model.decoder._forward_eager(
                emb, enc_out=enc_out, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
            )
            if EOLE_COMPILE_MODE in ["0", "1"]:
                self.model.decoder._compile_decoder(emb, enc_out, src_pad_mask, tgt_pad_mask, fn_tile)
            elif EOLE_COMPILE_MODE in ["2", "3"]:
                current_step = self.model.decoder.cache_seqlens[0]
                pos_ids_1d = current_step + torch.arange(1, device=emb.device)
                if self.model.decoder.rope.cos_sin is not None:
                    position_embeddings = self.model.decoder.rope.cos_sin[pos_ids_1d]
                else:
                    position_embeddings = None
                self.model.decoder.transformer_layers[0]._compile_decoder(
                    emb,
                    enc_out,
                    src_pad_mask.unsqueeze(1),
                    position_embeddings=position_embeddings,
                    cache_seqlens=self.model.decoder.cache_seqlens,
                )
            self.warmup_time.append(time() - start_wu)
            self._log(f"Warmup lasted: {time() - start_wu:.1f} sec")

        # (5) We start the Decoding loop
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(-1, 1)
            log_probs, attn = self._decode_and_generate(
                decoder_input,
                enc_out,
                src_len=decode_strategy.src_len,
                step=step,
                return_attn=decode_strategy.return_attention,
            )
            decode_strategy.advance(log_probs, attn)
            any_finished = any([any(sublist) for sublist in decode_strategy.is_finished_list])
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished and not decode_strategy.static_batch_size:
                # Reorder states.
                if isinstance(enc_out, tuple):
                    enc_out = tuple(x[select_indices] for x in enc_out)
                else:
                    enc_out = enc_out[select_indices]

            if parallel_paths > 1 or (any_finished and not decode_strategy.static_batch_size):
                self.model.decoder.map_state(lambda state: state[select_indices])

            self.model.decoder._extend_cache()  # noop when dynamic_shapes is False

        self.model.decoder._disable_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.add_estimator:
            dec_in = [item for sublist in decode_strategy.predictions for item in sublist]

            # Prepare estimator input = decoder out of each pred with initial enc_out
            dec_in = pad_sequence(dec_in, batch_first=True, padding_value=self._tgt_pad_idx)
            prepend_value = torch.full(
                (dec_in.size(0), 1),
                self._tgt_bos_idx,
                dtype=dec_in.dtype,
                device=dec_in.device,
            )
            dec_in = torch.cat((prepend_value, dec_in), dim=1)
            src_pad_mask = src.eq(self._src_pad_idx).unsqueeze(1)  # [B, 1, T_src]
            src_pad_mask = tile(src_pad_mask, parallel_paths)
            tgt_pad_mask = dec_in[:, :-1].eq(self._tgt_pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
            emb = self.model.tgt_emb(dec_in[:, :-1])
            self.model.decoder._disable_cache()
            dec_out, _ = self.model.decoder(
                emb,
                enc_out=enc_out2,
                src_len=src_len2,
                step=None,
                return_attn=False,
                src_pad_mask=src_pad_mask,
                tgt_pad_mask=tgt_pad_mask,
            )
            if self.estimator_type == "average":
                pad_mask2 = ~dec_in[:, :-1].eq(self._tgt_pad_idx)
                in_estim2 = (dec_out * pad_mask2.unsqueeze(-1).float()).sum(dim=1) / pad_mask2.sum(
                    dim=1, keepdim=True
                ).float()
            elif self.estimator_type == "last_token":
                in_estim2 = dec_out[:, -1, :]
            else:
                raise ValueError("EncoderDecoder model should use average or last token estimator")
            estim = self.model.estimator(in_estim2.to(dec_out.dtype)).squeeze(-1)
        else:
            if torch.is_tensor(enc_out2):
                estim = torch.ones([enc_out2.size(0)])
            else:
                estim = torch.ones([enc_out2[0].size(0)])
        estim = [[estim[i].item() for i in range(j, j + self.beam_size)] for j in range(0, len(estim), self.beam_size)]

        return self.report_results(
            gold_score,
            gold_log_probs,
            batch,
            batch_size,
            decode_strategy,
            estim,
        )

    def _score_target(self, batch, enc_out, src_len):
        tgt = batch["tgt"]
        tgt_in = tgt[:, :-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in,
            enc_out,
            src_len=src_len,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[:, 1:].unsqueeze(-1)
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=1).view(-1)
        return gold_scores, None
