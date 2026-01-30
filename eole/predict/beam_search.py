import torch
from eole.predict import penalties
from eole.predict.decode_strategy import DecodeStrategy
import warnings


class BeamSearchBase(DecodeStrategy):
    """Generation beam search with constant batch size for torch.compile.

    Modified to maintain constant batch_size * beam_size throughout decoding
    by using masking instead of removing finished batches.
    """

    def __init__(
        self,
        beam_size,
        batch_size,
        pad,
        bos,
        eos,
        unk,
        start,
        n_best,
        global_scorer,
        min_length,
        max_length,
        return_attention,
        block_ngram_repeat,
        exclusion_tokens,
        stepwise_penalty,
        ratio,
        ban_unk_token,
        add_estimator=False,
    ):
        super(BeamSearchBase, self).__init__(
            pad,
            bos,
            eos,
            unk,
            start,
            batch_size,
            beam_size,
            global_scorer,
            min_length,
            block_ngram_repeat,
            exclusion_tokens,
            return_attention,
            max_length,
            ban_unk_token,
            add_estimator,
        )
        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.ratio = ratio
        if add_estimator:
            self.num_hyp = self.beam_size
        else:
            self.num_hyp = self.n_best

        # beam state
        self._initial_batch_size = batch_size  # Store original batch size

        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = stepwise_penalty and self.global_scorer.has_cov_pen
        self._vanilla_cov_pen = not stepwise_penalty and self.global_scorer.has_cov_pen
        self._cov_pen = self.global_scorer.has_cov_pen

        self.src_len = None

        # NEW: Track which batches are completely finished
        self.batch_finished = None  # Shape: (batch_size,) boolean tensor

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_(self, enc_out, device, target_prefix):
        super(BeamSearchBase, self).initialize(device, target_prefix)
        self.best_scores = [-1e10 for _ in range(self._initial_batch_size)]
        self._beam_offset = torch.arange(
            0,
            self._initial_batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
            device=device,
        )
        self.topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1), device=device)
            .repeat(self._initial_batch_size)
            .reshape(self._initial_batch_size, self.beam_size)
        )

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((self._initial_batch_size, self.beam_size), dtype=torch.float, device=device)

        # NEW: Initialize finished tracking
        self.batch_finished = torch.zeros(self._initial_batch_size, dtype=torch.bool, device=device)

        # MPS doesn't support torch.isin() in Torch 2.3
        self._is_finished_list = (
            self._is_finished_list_mps if (device is not None and device.type == "mps") else self._is_finished_list_isin
        )

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    def _pick(self, log_probs):
        """Take a token pick decision for a step.

        Args:
            log_probs (FloatTensor): (B * beam_size, vocab_size)

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        vocab_size = log_probs.size(-1)
        # maybe fix some prediction at this step by modifying log_probs
        log_probs = self.target_prefixing(log_probs)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)

        # NEW: Mask out finished batches - set their scores to -inf
        # so they don't affect topk selection
        batch_finished_expanded = self.batch_finished.unsqueeze(1).expand_as(curr_scores)
        curr_scores = curr_scores.masked_fill(batch_finished_expanded, float("-inf"))

        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)

        return topk_scores, topk_ids

    def update_finished(self):
        """Update finished state without removing batches."""
        step = self.alive_seq.shape[-1]

        # Mask finished beams to prevent reactivation
        self.topk_log_probs.masked_fill_(
            torch.tensor(self.is_finished_list, device=self.topk_log_probs.device),
            float("-inf"),
        )

        predictions = self.alive_seq.view(self._initial_batch_size, self.beam_size, step)
        attention = (
            self.alive_attn.view(self._initial_batch_size, self.beam_size, step - 1, self.alive_attn.size(-1))
            if self.alive_attn is not None
            else None
        )

        topk_scores_list = self.topk_scores.tolist()

        # Process each batch - same structure as original beams_non_finished
        for i in range(self._initial_batch_size):
            if self.batch_finished[i]:
                continue

            if any(self.is_finished_list[i]):
                # Store finished hypotheses for this batch
                for j in [k for k, fin in enumerate(self.is_finished_list[i]) if fin]:
                    if self.ratio > 0:
                        s = topk_scores_list[i][j] / (step + 1)
                        self.best_scores[i] = max(s, self.best_scores[i])
                    self.hypotheses[i].append(
                        (
                            topk_scores_list[i][j],
                            predictions[i, j, 1:],
                            attention[i, j, :, self.src_len[i * self.beam_size]] if attention is not None else None,
                        )
                    )

                # Check if this batch should be marked as finished
                if self.ratio > 0:
                    pred_len = self.src_len[i * self.beam_size] * self.ratio
                    finish_flag = ((topk_scores_list[i][0] / pred_len) <= self.best_scores[i]) or all(
                        self.is_finished_list[i]
                    )
                else:
                    finish_flag = self.is_finished_list[i][0]

                if finish_flag and len(self.hypotheses[i]) >= self.num_hyp:
                    self.batch_finished[i] = True
                    self.hypotheses[i] = sorted(self.hypotheses[i], key=lambda x: x[0], reverse=True)
                    for score, pred, attn in self.hypotheses[i][: self.num_hyp]:
                        self.scores[i].append(score)
                        self.predictions[i].append(pred)
                        self.attention[i].append(attn if attn is not None else [])

        # Check if all batches are done
        if self.batch_finished.all():
            self.done = True
            return

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)
        _B = self._initial_batch_size  # Use constant batch size

        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(self._coverage + attn, self.global_scorer.beta).view(
                _B, self.beam_size
            )

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)
        self.ensure_unk_removed(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(step + 1, alpha=self.global_scorer.alpha)
        if length_penalty != 1:
            curr_scores = log_probs / length_penalty
        else:
            curr_scores = log_probs

        # Avoid any direction that would repeat unwanted ngrams
        self.block_ngram_repeats(curr_scores)

        # Pick up candidate token by curr_scores
        self.topk_scores, self.topk_ids = self._pick(curr_scores)

        # Recover log probs.
        self.topk_log_probs = self.topk_scores * length_penalty

        self.topk_ids = self.topk_ids % vocab_size

        # Append last prediction to reordered alive sequence
        self.alive_seq = torch.cat(
            [
                self.alive_seq,
                self.topk_ids.view(_B * self.beam_size, 1),
            ],
            -1,
        )

        self.maybe_update_forbidden_tokens()

        if self.return_attention or self._cov_pen:
            if step == 1:
                self.alive_attn = attn
                # update global state (step == 1)
                if self._cov_pen:  # coverage penalty
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 1)
                # update global state (step > 1)
                if self._cov_pen:
                    self._coverage += attn
                    self._prev_penalty = self.global_scorer.cov_penalty(
                        self._coverage, beta=self.global_scorer.beta
                    ).view(_B, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            cov_penalty = self.global_scorer.cov_penalty(self._coverage, beta=self.global_scorer.beta)
            self.topk_scores -= cov_penalty.view(_B, self.beam_size).float()

        self.is_finished_list = self._is_finished_list()
        self.ensure_max_length()

    def _is_finished_list_isin(self):
        return torch.isin(self.topk_ids, self.eos_t).tolist()

    def _is_finished_list_mps(self):
        return (self.topk_ids.unsqueeze(1) == self.eos_t).sum(dim=1).bool().tolist()


class BeamSearch(BeamSearchBase):
    """
    Beam search for seq2seq/encoder-decoder models
    """

    def initialize(self, enc_out, src_len, device=None, target_prefix=None):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """

        (fn_tile, enc_out, target_prefix) = self.initialize_tile(enc_out, src_len, target_prefix)
        if device is None:
            device = self.get_device_from_enc_out(enc_out)

        self.eos_t = torch.tensor(self.eos).to(device)
        super(BeamSearch, self).initialize_(enc_out, device, target_prefix)

        return fn_tile, enc_out


class BeamSearchLM(BeamSearchBase):
    """
    Beam search for language/decoder only models
    """

    def initialize(self, src, src_len, device=None, target_prefix=None):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """
        (fn_tile, _, target_prefix) = self.initialize_tile(None, src_len, target_prefix)
        if device is None:
            device = src.device

        self.eos_t = torch.tensor(self.eos).to(device)
        super(BeamSearchLM, self).initialize_(
            None,
            device=device,
            target_prefix=target_prefix,
        )

        return fn_tile, src

    def advance(self, log_probs, attn):
        super(BeamSearchLM, self).advance(log_probs, attn)

        # in LM task src_len is associated with currently generated src
        # and therefore needs to follow the generation
        self.src_len += 1


class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    @classmethod
    def from_config(cls, config):
        return cls(config.alpha, config.beta, config.length_penalty, config.coverage_penalty)

    def __init__(self, alpha, beta, length_penalty, coverage_penalty):
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(coverage_penalty, length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is not None and alpha == 0.0:
            warnings.warn("Using length penalty with alpha==0 " "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn("Non-default `beta` with no coverage penalty. " "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.0:
                warnings.warn(
                    "Non-default coverage penalty with beta==0 " "is equivalent to using coverage penalty none."
                )
