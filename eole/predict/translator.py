#!/usr/bin/env python
""" Translator Class and builder """
import torch
from torch.nn.utils.rnn import pad_sequence
from eole.predict.beam_search import BeamSearch
from eole.predict.greedy_search import GreedySearch
from eole.utils.misc import tile, sequence_mask
from eole.utils.alignment import extract_alignment
from eole.predict.inference import Inference


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
            batch_tgt_idxs = self._align_pad_prediction(
                predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx
            )
        tgt_mask = (
            batch_tgt_idxs.eq(self._tgt_pad_idx)
            | batch_tgt_idxs.eq(self._tgt_eos_idx)
            | batch_tgt_idxs.eq(self._tgt_bos_idx)
        )

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, enc_out, src_len = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(batch * n_best, src_len, nfeat)``
        src = tile(src, n_best, dim=0)
        if enc_states is not None:
            # Quick fix. Transformers return None as enc_states.
            # enc_states are only used later on to init decoder's state
            # but are never used in Transformer decoder, so we can skip
            enc_states = tile(enc_states, n_best, dim=0)
        if isinstance(enc_out, tuple):
            enc_out = tuple(tile(x, n_best, dim=0) for x in enc_out)
        else:
            enc_out = tile(enc_out, n_best, dim=0)
        src_len = tile(src_len, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src=src, enc_out=enc_out, enc_states=enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        # it should be done in a better way
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1].transpose(0, 1)  # exclude last target from inputs
        # here dec_in is batch first
        _, attns = self.model.decoder(dec_in, enc_out, src_len=src_len, with_align=True)

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(alignment_attn, prediction_mask, src_len, n_best)
        return alignement

    def predict_batch(self, batch, attn_debug):
        """Translate a batch of sentences."""
        if self.max_length_ratio > 0:
            max_length = int(
                min(self.max_length, batch["src"].size(1) * self.max_length_ratio + 5)
            )
        else:
            max_length = self.max_length
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
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
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
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
            return self._translate_batch_with_strategy(batch, decode_strategy)

    def _run_encoder(self, batch):
        src = batch["src"]
        src_len = batch["srclen"]
        batch_size = len(batch["srclen"])
        mask = sequence_mask(src_len)
        emb = self.model.src_emb(src)
        enc_out, enc_final_hs = self.model.encoder(emb, mask)

        if src_len is None:
            assert not isinstance(
                enc_out, tuple
            ), "Ensemble decoding only supported for text data"
            src_len = (
                torch.Tensor(batch_size).type_as(enc_out).long().fill_(enc_out.size(1))
            )
        return src, enc_final_hs, enc_out, src_len

    def _translate_batch_with_strategy(self, batch, decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        batch_size = len(batch["srclen"])

        # (1) Run the encoder on the src.
        src, enc_final_hs, enc_out, src_len = self._run_encoder(batch)

        self.model.decoder.init_state(
            src=src, enc_out=enc_out, enc_final_hs=enc_final_hs
        )

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
        (fn_map_state, enc_out) = decode_strategy.initialize(
            enc_out, src_len, target_prefix=target_prefix
        )

        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        if torch.is_tensor(enc_out):
            enc_out2 = enc_out.clone()
            src_len2 = decode_strategy.src_len.clone()
        elif isinstance(enc_out, tuple):
            enc_out2 = tuple(tensor.clone() for tensor in enc_out)
            src_len2 = tuple(tensor.clone() for tensor in decode_strategy.src_len)
        else:
            raise TypeError("enc_out must be either a tensor or a tuple of tensors")

        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(-1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                enc_out,
                batch,
                src_len=decode_strategy.src_len,
                step=step,
                batch_offset=decode_strategy.batch_offset,
                return_attn=decode_strategy.return_attention,
            )

            decode_strategy.advance(log_probs, attn)
            any_finished = any(
                [any(sublist) for sublist in decode_strategy.is_finished_list]
            )
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(enc_out, tuple):
                    enc_out = tuple(x[select_indices] for x in enc_out)
                else:
                    enc_out = enc_out[select_indices]

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(lambda state, dim: state[select_indices])

        if self.add_estimator:
            dec_in = [
                item for sublist in decode_strategy.predictions for item in sublist
            ]

            # make it clearer the pad index and BOS prepend
            dec_in = pad_sequence(dec_in, batch_first=True, padding_value=1)
            prepend_value = torch.full(
                (dec_in.size(0), 1), 2, dtype=dec_in.dtype, device=dec_in.device
            )
            dec_in = torch.cat((prepend_value, dec_in), dim=1)
            src_max_len = src_len2.max()
            src_pad_mask = sequence_mask(src_len2, src_max_len).unsqueeze(
                1
            )  # [B, 1, T_src]
            tgt_pad_mask = dec_in[:, :-1].eq(1).unsqueeze(1)  # [B, 1, T_tgt]
            emb = self.model.tgt_emb(dec_in[:, :-1])
            dec_out, _ = self.model.decoder(
                emb,
                enc_out=enc_out2,
                src_len=src_len2,
                step=None,
                return_attn=False,
                src_pad_mask=src_pad_mask,
                tgt_pad_mask=tgt_pad_mask,
            )
            pad_mask2 = ~dec_in[:, :-1].eq(1)
            in_estim2 = (dec_out * pad_mask2.unsqueeze(-1).float()).sum(
                dim=1
            ) / pad_mask2.sum(dim=1, keepdim=True).float()
            estim = self.model.estimator(in_estim2.to(dec_out.dtype)).squeeze(-1)
        else:
            if torch.is_tensor(enc_out2):
                estim = torch.ones([enc_out2.size(0)])
            else:
                estim = torch.ones([enc_out2[0].size(0)])
        estim = [
            [estim[i].item() for i in range(j, j + self.beam_size)]
            for j in range(0, len(estim), self.beam_size)
        ]

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
            batch,
            src_len=src_len,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[:, 1:].unsqueeze(-1)
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=1).view(-1)
        return gold_scores, None
