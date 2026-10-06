import torch
from torch.nn.utils.rnn import pad_sequence
from eole.predict.inference import Inference
from eole.constants import ModelType
from eole.predict.greedy_search import GreedySearchLM
from eole.predict.beam_search import BeamSearchLM
from eole.utils.misc import tile
from eole import EOLE_TORCH_COMPILE, EOLE_COMPILE_MODE
from eole.modules.gated_delta_net import GatedDeltaNet
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

    def predict_batch(self, batch, attn_debug, scoring=False, streamer=None):
        """Predict a batch of sentences.

        Args:
            batch: Batch of source data.
            attn_debug (bool): Whether to return attention weights.
            scoring (bool): Whether to run in scoring mode.
            streamer (GenerationStreamer, optional): If provided, tokens are
                pushed to the streamer at each decoding step to enable
                token-by-token output streaming.
        """
        batch_size = batch["srclen"].size(0)
        max_length = 0 if scoring else self.max_length
        with torch.no_grad():
            if self.top_k != 0 or self.top_p != 0 or self.self_speculative_decoding:
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
            return self._predict_batch_with_strategy(batch, decode_strategy, streamer=streamer)

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
        return log_probs

    # Not used for now but might be better to run warmup at model loading
    # at least for batch size 1 - would avoid to do it at first request
    def warmup_compile(self):
        """Pre-initialize CUDA graph infrastructure in the calling thread.

        Must be called from the thread that will later run inference so that
        PyTorch's CUDA-graph C++ thread-local storage (TLS) is initialized
        there.  ``infer_list_stream`` uses a ``ThreadPoolExecutor`` whose
        ``initializer`` calls this method, ensuring every worker thread has
        its TLS set up before it accepts any inference work.
        """
        if not EOLE_TORCH_COMPILE:
            return
        if not hasattr(self.model, "decoder") or self.model.decoder is None:
            return
        decoder = self.model.decoder
        device = self._dev
        try:
            dtype = next(self.model.parameters()).dtype
        except StopIteration:
            return

        H = decoder.hidden_size
        # Single sequence, single token — the shape used by the decode loop.
        dummy_emb = torch.zeros(1, 1, H, device=device, dtype=dtype)
        dummy_pad_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
        decoder.kvcache_maxsize = self.max_length
        with torch.no_grad():
            decoder._init_cache(dummy_emb, dummy_pad_mask)
            decoder._compile_decoder(emb=dummy_emb, tgt_pad_mask=dummy_pad_mask)
            decoder._disable_cache()

    def _predict_batch_with_strategy(self, batch, decode_strategy, streamer=None):
        try:
            return self._predict_batch_with_strategy_impl(batch, decode_strategy, streamer=streamer)
        except Exception:
            # Restore request-local MTP attention state even if prefill,
            # drafting, compilation, or streaming failed before normal cleanup.
            if hasattr(self.model, "clear_mtp_cache"):
                self.model.clear_mtp_cache()
            decoder = self.model.decoder
            decoder._speculative_forward = False
            for module in decoder.modules():
                if isinstance(module, GatedDeltaNet):
                    module.discard_speculation()
            decoder._disable_cache()
            raise

    def _predict_batch_with_strategy_impl(self, batch, decode_strategy, streamer=None):
        """Predict a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate prediction step by step.
            streamer (GenerationStreamer, optional): If provided, each newly
                generated token (for the first sequence) is pushed to the
                streamer at every decoding step so that callers can consume
                partial results before generation is complete.

        Returns:
            results (dict): The prediction results.
        """
        if self.dynamic_shapes is not None:
            decode_strategy.static_batch_size = not self.dynamic_shapes
        else:
            decode_strategy.static_batch_size = EOLE_TORCH_COMPILE

        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = len(batch["srclen"])

        # (1) check if we use left padding or not
        src = batch["src"]
        src_len = batch["srclen"]
        if batch["left_pad"]:
            target_prefix = None
        else:
            # split src into src and target_prefix to avoid padding.
            src, src_len, target_prefix = self.split_src_to_prevent_padding(src, src_len)

        # (2) init decoder
        self.model.decoder.init_state()  # noop for Transformer
        gold_score, gold_log_probs = self._gold_score(batch, None, src_len, None, batch_size, src)

        # (3) prep decode_strategy. Possibly repeat src objects.
        (fn_tile, src) = decode_strategy.initialize(
            src,
            src_len,
            target_prefix=target_prefix,
        )
        prefill_length = max(src_len.tolist())

        use_spec_decoding = (
            self.self_speculative_decoding
            and isinstance(decode_strategy, GreedySearchLM)
            and self.beam_size == 1
            and self.n_best == 1
            and batch_size == 1
            and (self.top_k == 1 or self.temperature == 0.0)
            and self.min_length == 0
            and not self.ban_unk_token
            and self.block_ngram_repeat == 0
            and not decode_strategy.return_attention
            and decode_strategy.target_prefix is None
            and batch.get("images") is None
            and len(getattr(self.model, "mtp_heads", [])) > 0
        )
        if self.self_speculative_decoding and not use_spec_decoding:
            self._log(
                "self_speculative_decoding requires one sequence, one hypothesis, deterministic greedy selection, "
                "no decode constraints, text-only inputs, no attention output, "
                "and a model with loaded MTP heads; using normal decoding"
            )

        # (4) warmup for Torch compile
        # use the current batch to generate the decode graph (B, 1)
        # we need proper set up to run the forward pass of the decoder or decoder layer
        if EOLE_TORCH_COMPILE:
            start_wu = time()
            images = batch.get("images", None)
            if images is not None:
                emb, _ = self.model.embed_vision_language_features(src, images=images)
            else:
                emb = self.model.tgt_emb(src, step=0)
            tgt_pad_mask = src.eq(self._tgt_pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
            self.model.decoder._init_cache(emb, tgt_pad_mask)
            self.model.decoder.map_state(fn_tile)
            if EOLE_COMPILE_MODE in ["0", "1"]:
                self.model.decoder._compile_decoder(emb=emb, tgt_pad_mask=tgt_pad_mask)
                if use_spec_decoding and self.max_length > 1:
                    # Compile the short verifier chunk too. Without this
                    # shape-specific warmup, compile modes 0/1 compile only
                    # S=1 decode and leave every speculative verifier pass in
                    # eager mode.
                    qwen_recurrent_mtp = getattr(self.model.mtp_heads[0], "emb_norm", None) is not None
                    draft_count = self.self_speculative_num_tokens if qwen_recurrent_mtp else len(self.model.mtp_heads)
                    verify_len = min(draft_count + 1, self.max_length)
                    dummy_verify = torch.zeros(
                        emb.size(0), verify_len, self.model.decoder.hidden_size, device=emb.device, dtype=emb.dtype
                    )
                    dummy_verify_mask = torch.zeros(emb.size(0), 1, verify_len, dtype=torch.bool, device=emb.device)
                    linear_layers = [
                        module
                        for module in self.model.decoder.modules()
                        if isinstance(module, GatedDeltaNet)
                        and module.conv_state is not None
                        and module.recurrent_state is not None
                    ]
                    try:
                        for layer in linear_layers:
                            layer.begin_speculation(verify_len)
                        self.model.decoder._speculative_forward = True
                        self.model.decoder(
                            dummy_verify,
                            enc_out=None,
                            step=1,
                            return_attn=False,
                            tgt_pad_mask=dummy_verify_mask,
                        )
                    finally:
                        self.model.decoder._speculative_forward = False
                        for layer in linear_layers:
                            layer.end_speculation()
            elif EOLE_COMPILE_MODE in ["2", "3"]:
                current_step = self.model.decoder.cache_seqlens[0]
                pos_ids_1d = current_step + torch.arange(1, device=emb.device)
                if self.model.decoder.rope.cos_sin is not None:
                    position_embeddings = self.model.decoder.rope.cos_sin[pos_ids_1d]
                else:
                    position_embeddings = None
                self.model.decoder.transformer_layers[0]._compile_decoder(
                    emb, position_embeddings=position_embeddings, cache_seqlens=self.model.decoder.cache_seqlens
                )
            self.warmup_time.append(time() - start_wu)
            self._log(f"Warmup lasted: {time() - start_wu:.1f} sec")

        self._log_inference_backends(speculative=use_spec_decoding)

        # (5) Start the decoding loop with timers
        if not self.estim_only:
            # (5) Begin decoding step by step:
            if self.report_time:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                beg_time = time()

            step = 0
            pending_spec = None
            self._speculative_drafted_tokens = 0
            self._speculative_accepted_tokens = 0
            self._speculative_drafted_by_position = [0] * self.self_speculative_num_tokens
            self._speculative_accepted_by_position = [0] * self.self_speculative_num_tokens
            self._mtp_profile_enabled = use_spec_decoding and self.report_time and torch.cuda.is_available()
            self._mtp_profile_events = {
                name: []
                for name in (
                    "draft",
                    "mtp_head",
                    "draft_vocab",
                    "verify",
                    "verify_decoder",
                    "verify_vocab",
                    "state_commit",
                )
            }
            if use_spec_decoding:
                # Keep the single batch row stable while a verification chunk
                # is temporarily longer than a normal decode step.
                decode_strategy.static_batch_size = True
            while step < decode_strategy.max_length:
                if use_spec_decoding and pending_spec is not None:
                    hidden, hidden_position = pending_spec
                    step, next_hidden, next_position = self._speculative_draft_verify(
                        decode_strategy, hidden, hidden_position, step, streamer
                    )
                    pending_spec = (next_hidden, next_position)
                    any_finished = any([any(sublist) for sublist in decode_strategy.is_finished_list])
                    if any_finished:
                        decode_strategy.update_finished()
                        if decode_strategy.done:
                            break
                    # The next verifier reserves its full input chunk. Skip
                    # the redundant dynamic-cache growth check and host sync.
                    continue

                decoder_input = src if step == 0 else decode_strategy.current_predictions.view(-1, 1)
                cur_position = step if step == 0 else step + prefill_length - 1
                decoded = self._decode_and_generate(
                    decoder_input,
                    None,
                    src_len=decode_strategy.src_len,
                    step=cur_position,
                    images=batch.get("images", None) if step == 0 else None,
                    return_hidden=use_spec_decoding,
                    return_all_hidden=use_spec_decoding and step == 0,
                )
                if use_spec_decoding:
                    if step == 0:
                        log_probs, attn, hidden, all_hidden = decoded
                        if getattr(self.model.mtp_heads[0], "emb_norm", None) is not None:
                            self.model.init_mtp_cache(all_hidden, decoder_input, self.max_length)
                    else:
                        log_probs, attn, hidden = decoded
                    hidden_position = prefill_length - 1 if step == 0 else cur_position
                else:
                    log_probs, attn = decoded

                if step == 0:
                    log_probs = self.tile_to_beam_size_after_initial_step(fn_tile, log_probs)

                decode_strategy.advance(log_probs, attn)
                step += 1
                if self.report_time and step == 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    self.step0_time.append(time() - beg_time)
                any_finished = any([any(sublist) for sublist in decode_strategy.is_finished_list])

                if streamer is not None:
                    streamer.put(decode_strategy.current_predictions[:1])

                if any_finished:
                    decode_strategy.update_finished()
                    if decode_strategy.done:
                        break

                if use_spec_decoding and step < decode_strategy.max_length:
                    step, next_hidden, next_position = self._speculative_draft_verify(
                        decode_strategy, hidden, hidden_position, step, streamer
                    )
                    pending_spec = (next_hidden, next_position)
                    any_finished = any([any(sublist) for sublist in decode_strategy.is_finished_list])
                    if any_finished:
                        decode_strategy.update_finished()
                        if decode_strategy.done:
                            break

                if parallel_paths > 1 or (any_finished and not decode_strategy.static_batch_size):
                    self.model.decoder.map_state(lambda state: state[decode_strategy.select_indices])

                if not (use_spec_decoding and pending_spec is not None):
                    # The speculative verifier has already reserved capacity
                    # for its entire chunk; no normal one-token growth check
                    # is needed before the next speculative step.
                    self.model.decoder._extend_cache()

            if use_spec_decoding and self._speculative_drafted_tokens:
                accepted_rate = self._speculative_accepted_tokens / self._speculative_drafted_tokens
                self._log(
                    "MTP draft acceptance: "
                    f"{self._speculative_accepted_tokens}/{self._speculative_drafted_tokens} tokens "
                    f"({accepted_rate:.1%})"
                )
                position_rates = [
                    f"p{index + 1}={accepted / drafted:.1%}"
                    for index, (accepted, drafted) in enumerate(
                        zip(self._speculative_accepted_by_position, self._speculative_drafted_by_position)
                    )
                    if drafted
                ]
                self._log("MTP acceptance by draft position: " + ", ".join(position_rates))
                if self._mtp_profile_enabled:
                    torch.cuda.synchronize()
                    phase_ms = {
                        name: sum(start.elapsed_time(end) for start, end in events)
                        for name, events in self._mtp_profile_events.items()
                    }
                    self._log(
                        "MTP phase time (GPU ms): "
                        f"draft={phase_ms['draft']:.1f}, verify={phase_ms['verify']:.1f}, "
                        f"mtp_head={phase_ms['mtp_head']:.1f}, draft_vocab={phase_ms['draft_vocab']:.1f}, "
                        f"verify_decoder={phase_ms['verify_decoder']:.1f}, "
                        f"verify_vocab={phase_ms['verify_vocab']:.1f}, "
                        f"state_commit={phase_ms['state_commit']:.1f}, "
                        f"cycles={len(self._mtp_profile_events['draft'])}"
                    )

            if use_spec_decoding and hasattr(self.model, "clear_mtp_cache"):
                self.model.clear_mtp_cache()

            self.model.decoder._disable_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if streamer is not None:
                streamer.end()

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

    def _speculative_draft_verify(self, decode_strategy, hidden, hidden_position, step, streamer=None):
        """Draft with MTP heads and verify the chunk in one main-model pass."""
        seed = decode_strategy.current_predictions.view(-1, 1)
        remaining = decode_strategy.max_length - step
        if remaining <= 1:
            # Speculative iterations skip the normal end-of-step cache growth;
            # reserve this last token explicitly when no further verifier will
            # run to reserve capacity for us.
            if hasattr(self.model.decoder, "_extend_cache"):
                self.model.decoder._extend_cache(threshold=1, addzeros=32)
            log_probs, _, next_hidden = self._decode_and_generate(
                seed,
                None,
                src_len=decode_strategy.src_len,
                step=hidden_position + 1,
                return_hidden=True,
                profile_events=self._mtp_profile_events if self._mtp_profile_enabled else None,
            )
            decode_strategy.advance(log_probs, None)
            if streamer is not None:
                streamer.put(decode_strategy.current_predictions[:1])
            return step + 1, next_hidden[:, -1:, :], hidden_position + 1

        draft_events = None
        if self._mtp_profile_enabled:
            draft_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            draft_events[0].record()
        drafts = self.model.draft_mtp_tokens(
            hidden,
            seed,
            hidden_position,
            max_tokens=min(self.self_speculative_num_tokens, remaining - 1),
            profile_events=self._mtp_profile_events if self._mtp_profile_enabled else None,
        )
        if draft_events is not None:
            draft_events[1].record()
            self._mtp_profile_events["draft"].append(draft_events)
        num_draft = min(len(drafts), remaining - 1)
        if num_draft == 0:
            return step, hidden, hidden_position
        drafts = drafts[:num_draft]
        draft_tensor = torch.cat(drafts, dim=1)
        verify_input = torch.cat([seed, draft_tensor], dim=1)

        # Eager decoding grows its KV cache on demand. The ordinary decode
        # loop extends it after each forward, but speculative verification
        # consumes several positions in one forward and runs before that
        # extension point. Reserve the entire chunk first; otherwise
        # flash_attn_with_kvcache can write past the dynamic cache allocation
        # when compilation is disabled.
        if hasattr(self.model.decoder, "_extend_cache"):
            self.model.decoder._extend_cache(threshold=verify_input.size(1), addzeros=max(32, verify_input.size(1)))

        linear_layers = [
            module
            for module in self.model.decoder.modules()
            if isinstance(module, GatedDeltaNet)
            and module.conv_state is not None
            and module.recurrent_state is not None
        ]
        for layer in linear_layers:
            layer.begin_speculation(verify_input.size(1))
        self.model.decoder._speculative_forward = True
        verify_events = None
        if self._mtp_profile_enabled:
            verify_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            verify_events[0].record()
        try:
            verify_log_probs, _, verify_hidden = self._decode_and_generate(
                verify_input,
                None,
                src_len=decode_strategy.src_len,
                step=hidden_position + 1,
                return_hidden=True,
                profile_events=self._mtp_profile_events if self._mtp_profile_enabled else None,
            )
            if verify_events is not None:
                verify_events[1].record()
                self._mtp_profile_events["verify"].append(verify_events)
        except Exception:
            for layer in linear_layers:
                layer.discard_speculation()
            self.model.commit_mtp_draft(0)
            raise
        finally:
            self.model.decoder._speculative_forward = False
            for layer in linear_layers:
                layer.end_speculation()

        predicted = verify_log_probs.argmax(dim=-1)
        fast_greedy_advance = decode_strategy.target_prefix is None
        if fast_greedy_advance:
            # Determine the accepted draft prefix and any EOS truncation on
            # device. A per-position torch.equal() loop synchronizes once for
            # every proposed token; transfer the three scalar decisions only
            # once for this batch-one greedy path.
            matches = predicted[:, :num_draft].eq(draft_tensor)
            mismatches = ~matches
            first_mismatch = mismatches.to(torch.int32).argmax(dim=1)
            accepted_tensor = torch.where(mismatches.any(dim=1), first_mismatch, num_draft)
            candidate_count = accepted_tensor + 1  # accepted drafts plus target correction/bonus
            candidate_positions = torch.arange(predicted.size(1), device=predicted.device).unsqueeze(0)
            eos_hits = torch.isin(predicted, decode_strategy.eos_t)
            eos_hits = eos_hits & candidate_positions.lt(candidate_count.unsqueeze(1))
            has_eos = eos_hits.any(dim=1)
            first_eos = eos_hits.to(torch.int32).argmax(dim=1)
            count_tensor = torch.where(has_eos, first_eos + 1, candidate_count)
            accepted, advance_count, finished = map(
                int,
                torch.stack((accepted_tensor[0], count_tensor[0], has_eos[0].to(torch.int64))).tolist(),
            )
        else:
            accepted = 0
            while accepted < num_draft and torch.equal(predicted[:, accepted], drafts[accepted].squeeze(1)):
                accepted += 1
        self._speculative_drafted_tokens += num_draft
        self._speculative_accepted_tokens += accepted
        for index in range(num_draft):
            self._speculative_drafted_by_position[index] += 1
            if index < accepted:
                self._speculative_accepted_by_position[index] += 1
        state_commit_events = None
        if self._mtp_profile_enabled:
            state_commit_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            state_commit_events[0].record()
        self.model.commit_mtp_draft(accepted + 1)

        # The verifier consumed seed plus all draft inputs. Commit only those
        # through the last accepted draft; attention cache writes after that
        # point are discarded by rewinding its position counter.
        GatedDeltaNet.commit_speculation_group(linear_layers, accepted + 1)
        rejected = num_draft - accepted
        if rejected and self.model.decoder.cache_seqlens is not None:
            self.model.decoder.cache_seqlens.sub_(rejected)
        if state_commit_events is not None:
            state_commit_events[1].record()
            self._mtp_profile_events["state_commit"].append(state_commit_events)

        if fast_greedy_advance:
            emitted = decode_strategy.advance_speculative(predicted, verify_log_probs, advance_count, finished)
            step += advance_count
            last_advanced = advance_count - 1
            if streamer is not None:
                for index in range(advance_count):
                    streamer.put(emitted[:, index])
        else:
            last_advanced = 0
            for index in range(accepted + 1):
                decode_strategy.advance(verify_log_probs[:, index, :], None)
                step += 1
                last_advanced = index
                if streamer is not None:
                    streamer.put(decode_strategy.current_predictions[:1])
                if any(any(done) for done in decode_strategy.is_finished_list):
                    break

        self.model.set_mtp_context(
            verify_hidden[:, : last_advanced + 1, :],
            predicted[:, : last_advanced + 1],
            hidden_position + 1,
        )

        # The verifier's hidden at `last_advanced` produced the current token,
        # so it is the correct context for drafting the next chunk.
        return (
            step,
            verify_hidden[:, last_advanced : last_advanced + 1, :],
            hidden_position + 1 + last_advanced,
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
