import torch
from eole.predict.inference import Inference
from eole.constants import ModelType
from eole.predict.greedy_search import GreedySearchLM
from eole.predict.beam_search import BeamSearchLM


class GeneratorLM(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelType.DECODER:
            raise ValueError(
                f"GeneratorLM does not support task {task}."
                f" Tasks supported: {ModelType.DECODER}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        raise NotImplementedError

    def predict_batch(self, batch, attn_debug, scoring=False):
        """Predict a batch of sentences."""
        max_length = 0 if scoring else self.max_length
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearchLM(
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
                decode_strategy = BeamSearchLM(
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
                )
            return self._predict_batch_with_strategy(batch, decode_strategy)

    @classmethod
    def split_src_to_prevent_padding(cls, src, src_len):
        min_len_batch = torch.min(src_len).item()
        target_prefix = None
        if min_len_batch > 0 and min_len_batch < src.size(1):
            target_prefix = src[:, min_len_batch:, :]
            src = src[:, :min_len_batch, :]
            src_len[:] = min_len_batch
        return src, src_len, target_prefix

    def tile_to_beam_size_after_initial_step(self, fn_map_state, log_probs):
        if fn_map_state is not None:
            log_probs = fn_map_state(log_probs, dim=0)
            self.model.decoder.map_state(fn_map_state)
            log_probs = log_probs[:, -1, :]
        return log_probs

    def _predict_batch_with_strategy(self, batch, decode_strategy, left_pad=True):
        """Predict a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate prediction step by step.

        Returns:
            results (dict): The prediction results.
        """
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = len(batch["srclen"])

        # (1) split src into src and target_prefix to avoid padding.
        src = batch["src"]
        src_len = batch["srclen"]

        if left_pad:
            target_prefix = None
        else:
            src, src_len, target_prefix = self.split_src_to_prevent_padding(
                src, src_len
            )

        # (2) init decoder
        self.model.decoder.init_state(src=src)
        gold_score, gold_log_probs = self._gold_score(
            batch, None, src_len, None, batch_size, src
        )

        # (3) prep decode_strategy. Possibly repeat src objects.
        (fn_map_state, src) = decode_strategy.initialize(
            src,
            src_len,
            target_prefix=target_prefix,
        )

        # (4) Begin decoding step by step:
        # beg_time = time()
        for step in range(decode_strategy.max_length):
            decoder_input = (
                src if step == 0 else decode_strategy.current_predictions.view(-1, 1)
            )
            log_probs, attn = self._decode_and_generate(
                decoder_input,
                None,
                batch,
                src_len=decode_strategy.src_len,
                step=step if step == 0 else step + max(src_len.tolist()),
                batch_offset=decode_strategy.batch_offset,
            )

            if step == 0:
                log_probs = self.tile_to_beam_size_after_initial_step(
                    fn_map_state, log_probs
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

            if parallel_paths > 1 or any_finished:
                # select indexes in model state/cache
                self.model.decoder.map_state(lambda state, dim: state[select_indices])
            # if step == 0:
            #    print("step0 time: ", time() - beg_time)

        estim = [[1.0 for _ in range(self.beam_size)] for _ in range(batch_size)]

        return self.report_results(
            gold_score,
            gold_log_probs,
            batch,
            batch_size,
            decode_strategy,
            estim,
        )

    def _score_target(self, batch, enc_out, src_len):
        src = batch["src"]
        src_len = batch["srclen"]
        tgt = batch["tgt"]

        log_probs, attn = self._decode_and_generate(
            src,
            None,
            batch,
            src_len=src_len,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        tgt = tgt.unsqueeze(2)
        gold_log_probs = log_probs.gather(2, tgt).squeeze(-1)
        gold_scores = gold_log_probs.sum(dim=1).view(-1)

        if self.return_gold_log_probs:
            return gold_scores, gold_log_probs

        return gold_scores, None
