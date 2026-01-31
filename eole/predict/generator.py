import torch
from torch.nn.utils.rnn import pad_sequence
from eole.predict.inference import Inference
from eole.constants import ModelType
from eole.predict.greedy_search import GreedySearchLM
from eole.predict.beam_search import BeamSearchLM
from eole.utils.misc import tile
from eole import EOLE_TORCH_COMPILE, EOLE_COMPILE_MODE
from time import time


class GeneratorLM(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelType.DECODER:
            raise ValueError(f"GeneratorLM does not support task {task}." f" Tasks supported: {ModelType.DECODER}")

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        raise NotImplementedError

    def predict_batch(self, batch, attn_debug, scoring=False):
        """Predict a batch of sentences."""
        batch_size = batch["srclen"].size(0)
        max_length = 0 if scoring else self.max_length - batch["srclen"].max()
        with torch.no_grad():
            if self.top_k != 0 or self.top_p != 0:
                decode_strategy = GreedySearchLM(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    batch_size=batch_size,
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
                decode_strategy = BeamSearchLM(
                    self.beam_size,
                    batch_size=batch_size,
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

    @classmethod
    def split_src_to_prevent_padding(cls, src, src_len):
        min_len_batch = torch.min(src_len).item()
        target_prefix = None
        if min_len_batch > 0 and min_len_batch < src.size(1):
            target_prefix = src[:, min_len_batch:]
            src = src[:, :min_len_batch]
            src_len[:] = min_len_batch
        return src, src_len, target_prefix

    def tile_to_beam_size_after_initial_step(self, fn_tile, log_probs):
        if fn_tile is not None:
            log_probs = fn_tile(log_probs)
            self.model.decoder.map_state(fn_tile)
            log_probs = log_probs[:, -1, :]
        return log_probs

    def _predict_batch_with_strategy(self, batch, decode_strategy):
        """Predict a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate prediction step by step.

        Returns:
            results (dict): The prediction results.
        """
        decode_strategy.static_batch_size = EOLE_TORCH_COMPILE
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = len(batch["srclen"])

        # (1) split src into src and target_prefix to avoid padding.
        src = batch["src"]
        src_len = batch["srclen"]

        if batch["left_pad"]:
            target_prefix = None
        else:
            src, src_len, target_prefix = self.split_src_to_prevent_padding(src, src_len)

        # (2) init decoder
        self.model.decoder.init_state()
        gold_score, gold_log_probs = self._gold_score(batch, None, src_len, None, batch_size, src)

        # (3) prep decode_strategy. Possibly repeat src objects.
        (fn_tile, src) = decode_strategy.initialize(
            src,
            src_len,
            target_prefix=target_prefix,
        )
        prefill_length = max(src_len.tolist())

        # (4) warmup for Torch compile
        # use the current batch to generate the decode graph (B, 1)
        # we need proper set up to run the forward pass of the decoder or decoder layer
        if EOLE_TORCH_COMPILE:
            start_wu = time()
            self._log("Warmup started")
            images = batch.get("images", None)
            if images is not None:
                emb, _ = self.model.embed_vision_language_features(src, images=images)
            else:
                emb = self.model.tgt_emb(src, step=0)
            tgt_pad_mask = src.eq(self._tgt_pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
            if EOLE_COMPILE_MODE in ["0", "1"]:
                self.model.decoder._compile_decoder(emb, tgt_pad_mask, fn_tile)
            elif EOLE_COMPILE_MODE in ["2", "3"]:
                self.model.decoder._init_cache(emb, tgt_pad_mask)
                self.model.decoder.map_state(fn_tile)
                current_step = self.model.decoder.cache_seqlens[0]
                pos_ids_1d = current_step + torch.arange(1, device=emb.device)
                position_embeddings = self.model.decoder.rope.cos_sin[pos_ids_1d]
                self.model.decoder.transformer_layers[0]._compile_decoder(
                    emb, position_embeddings=position_embeddings, cache_seqlens=self.model.decoder.cache_seqlens
                )
            self.warmup_time.append(time() - start_wu)
            self._log(f"Warmup lasted: {time() - start_wu:.1f} sec")

        if not self.estim_only:
            # (5) Begin decoding step by step:
            if self.report_time:
                torch.cuda.synchronize()
                beg_time = time()

            for step in range(decode_strategy.max_length):
                decoder_input = src if step == 0 else decode_strategy.current_predictions.view(-1, 1)

                log_probs, attn = self._decode_and_generate(
                    decoder_input,
                    None,
                    src_len=decode_strategy.src_len,
                    step=step if step == 0 else step + prefill_length - 1,
                    images=batch.get("images", None) if step == 0 else None,
                )

                if step == 0:
                    log_probs = self.tile_to_beam_size_after_initial_step(fn_tile, log_probs)

                decode_strategy.advance(log_probs, attn)
                any_finished = any([any(sublist) for sublist in decode_strategy.is_finished_list])
                if any_finished:
                    decode_strategy.update_finished()
                    if decode_strategy.done:
                        break
                select_indices = decode_strategy.select_indices

                if parallel_paths > 1 or any_finished:
                    # select indexes in model state/cache
                    self.model.decoder.map_state(lambda state: state[select_indices])

                if self.report_time and step == 0:
                    torch.cuda.synchronize()
                    self.step0_time.append(time() - beg_time)

            self.model.decoder._disable_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.add_estimator:
            # Prepare estimator input = decoder out of each pred with initial enc_out
            if not self.estim_only:
                dec_in = [item for sublist in decode_strategy.predictions for item in sublist]
                src = tile(src, parallel_paths)
                concat_seq = [torch.cat((s, d), dim=0) for s, d in zip(src, dec_in)]
                # make padding left sided
                concat_seq = [x.flip(dims=[0]) for x in concat_seq]
                concat_seq = pad_sequence(concat_seq, batch_first=True, padding_value=self._tgt_pad_idx)
                concat_seq = concat_seq.flip(dims=[1])
                # remove eos
                dec_in = concat_seq[:, :-1]
            else:
                dec_in = src
                parallel_paths = 1
            tgt_pad_mask = dec_in.eq(self._tgt_pad_idx).unsqueeze(1)  # [B, T_tgt]
            emb = self.model.tgt_emb(dec_in)
            self.model.decoder._disable_cache()
            dec_out, _ = self.model.decoder(
                emb,
                enc_out=None,
                return_attn=False,
                tgt_pad_mask=tgt_pad_mask,
            )
            if self.estimator_type == "average":
                pad_mask = ~dec_in.eq(self._tgt_pad_idx)
                in_estim = (dec_out * pad_mask.unsqueeze(-1).float()).sum(dim=1) / pad_mask.sum(
                    dim=1, keepdim=True
                ).float()
            elif self.estimator_type == "last_token":
                in_estim = dec_out[:, -1, :]
            else:
                raise ValueError("Decoder only model should use average or last token estimator")
            estim = self.model.estimator(in_estim.to(dec_out.dtype)).squeeze(-1)
            estim = [
                [estim[i].item() for i in range(j, j + parallel_paths)] for j in range(0, len(estim), parallel_paths)
            ]
        else:
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
            src_len=src_len,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        tgt = tgt.unsqueeze(2)
        gold_log_probs = log_probs.gather(2, tgt).squeeze(-1)
        gold_scores = gold_log_probs.sum(dim=1).view(-1)

        if self.return_gold_log_probs:
            return gold_scores, gold_log_probs

        return gold_scores, None
