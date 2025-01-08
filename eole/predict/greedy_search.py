import torch
from torch.nn.functional import softmax
from eole.predict.decode_strategy import DecodeStrategy


def sample_topp(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)

    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_keep = cumulative_probs.lt(top_p)

    # keep indices until overflowing p
    cumsum_mask = sorted_indices_to_keep.cumsum(dim=1)
    last_included = cumsum_mask[:, -1:]
    last_included.clamp_(0, sorted_indices_to_keep.size()[1] - 1)
    sorted_indices_to_keep = sorted_indices_to_keep.scatter_(1, last_included, 1)

    # Set all logits that are not in the top-p to -10000.
    # This puts the probabilities close to 0.
    keep_indices = sorted_indices_to_keep.scatter(
        1,
        sorted_indices,
        sorted_indices_to_keep,
    )
    return logits.masked_fill(~keep_indices, -10000)


def sample_topk(logits, top_k):
    top_values, _ = torch.topk(logits, top_k, dim=1)
    kth_best = top_values[:, -1].view([-1, 1])
    kth_best = kth_best.repeat([1, logits.shape[1]]).float()

    # Set all logits that are not in the top-k to -10000.
    # This puts the probabilities close to 0.
    ignore = torch.lt(logits, kth_best)
    return logits.masked_fill(ignore, -10000)


def sample_with_temperature(logits, temperature, top_k, top_p):
    """Select next tokens randomly from the top k possible next tokens.

    Samples from a categorical distribution over the ``top_k`` words using
    the category probabilities ``logits / temperature``.

    Args:
        logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
            These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
            (The distribution actually uses the log-probabilities
            ``logits - logits.logsumexp(-1)``, which equals the logits if
            they are log-probabilities summing to 1.)
        temperature (float): Used to scale down logits. The higher the
            value, the more likely it is that a non-max word will be
            sampled.
        top_k (int): This many words could potentially be chosen. The
            other logits are set to have probability 0.
        top_p (float): Keep most likely words until the cumulated
            probability is greater than p. If used with top_k: both
            conditions will be applied

    Returns:
        (LongTensor, FloatTensor):

        * topk_ids: Shaped ``(batch_size, 1)``. These are
          the sampled word indices in the output vocab.
        * topk_scores: Shaped ``(batch_size, 1)``. These
          are essentially ``(logits / temperature)[topk_ids]``.
    """

    if temperature == 0.0 or top_k == 1:
        # For temp=0.0, take the argmax to avoid divide-by-zero errors.
        # top_k=1 is also equivalent to argmax.
        topk_scores, topk_ids = logits.topk(1, dim=-1)
        if temperature > 0:
            topk_scores /= temperature
    else:
        logits = torch.div(logits, temperature)
        if top_p > 0:
            logits = sample_topp(logits, top_p)
        if top_k > 0:
            logits = sample_topk(logits, top_k)
        dist = torch.distributions.Categorical(logits=logits)
        topk_ids = dist.sample().view(-1, 1)
        topk_scores = logits.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores


class GreedySearch(DecodeStrategy):
    """Select next tokens randomly from the top k possible next tokens.

    The ``scores`` attribute's lists are the score, after applying temperature,
    of the final prediction (either EOS or the final token in the event
    that ``max_length`` is reached)

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        start (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        batch_size (int): See base.
        global_scorer (eole.predict.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        ban_unk_token (Boolean): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        return_attention (bool): See base.
        max_length (int): See base.
        temperature (float): See
            :func:`~eole.predict.greedy_search.sample_with_temperature()`.
        top_k (int): See
            :func:`~eole.predict.greedy_search.sample_with_temperature()`.
        top_p (float): See
            :func:`~eole.predict.greedy_search.sample_with_temperature()`.
        beam_size (int): Number of beams to use.
    """

    def __init__(
        self,
        pad,
        bos,
        eos,
        unk,
        start,
        n_best,
        batch_size,
        global_scorer,
        min_length,
        block_ngram_repeat,
        exclusion_tokens,
        return_attention,
        max_length,
        temperature,
        top_k,
        top_p,
        beam_size,
        ban_unk_token,
        add_estimator=False,
    ):
        super(GreedySearch, self).__init__(
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
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.topk_scores = None
        self.beam_size = beam_size
        self.n_best = n_best
        if add_estimator:
            self.num_hyp = self.beam_size
        else:
            self.num_hyp = self.n_best

    def initialize(self, enc_out, src_len, device=None, target_prefix=None):
        """Initialize for decoding."""
        (fn_map_state, enc_out, target_prefix) = self.initialize_tile(enc_out, src_len, target_prefix)
        if device is None:
            device = self.get_device_from_enc_out(enc_out)

        self.eos_t = torch.tensor(self.eos).to(device)
        super(GreedySearch, self).initialize(device, target_prefix)
        self.select_indices = torch.arange(self.batch_size * self.beam_size, dtype=torch.long, device=device)
        self.original_batch_idx = fn_map_state(torch.arange(self.batch_size, dtype=torch.long, device=device), dim=0)
        self.beams_scores = torch.zeros((self.batch_size * self.beam_size, 1), dtype=torch.float, device=device)
        # MPS doesn't support torch.isin() in Torch 2.3
        # Avoiding need to CPU fallback by adding alternative implementation
        # Can be removed when Torch 2.4 is supported
        self._is_finished_list = (
            self._is_finished_list_mps if (device is not None and device.type == "mps") else self._is_finished_list_isin
        )
        return fn_map_state, enc_out

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def batch_offset(self):
        return self.select_indices

    def _pick(self, log_probs):
        """Function used to pick next tokens.

        Args:
            log_probs (FloatTensor): ``(batch_size, vocab_size)``.
        """
        # maybe fix some prediction at this step by modifying log_probs
        log_probs = self.target_prefixing(log_probs)
        topk_ids, topk_scores = sample_with_temperature(log_probs, self.temperature, self.top_k, self.top_p)

        return topk_ids, topk_scores

    def align_select_indices(self):
        nb_finished_beams = len(self.is_finished_list) - self.select_indices.size(0)
        if nb_finished_beams:
            self.select_indices = torch.arange(
                self.select_indices.size(0),
                dtype=torch.long,
                device=self.select_indices.device,
            )

    def advance(self, log_probs, attn):
        """Select next tokens randomly from the top k possible next tokens.

        Args:
            log_probs (FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs
                (``(-inf, 0]``). (The distribution actually uses the
                log-probabilities ``logits - logits.logsumexp(-1)``,
                which equals the logits if they are log-probabilities summing
                to 1.)
            attn (FloatTensor): Shaped ``(1, B, inp_seq_len)``.
        """
        self.align_select_indices()

        self.ensure_min_length(log_probs)
        self.ensure_unk_removed(log_probs)
        self.block_ngram_repeats(log_probs)

        topk_ids, self.topk_scores = self._pick(log_probs)
        self.beams_scores += self.topk_scores

        self.is_finished_list = torch.isin(topk_ids, self.eos_t).tolist()

        self.alive_seq = torch.cat([self.alive_seq, topk_ids], -1)
        if self.return_attention:
            if self.alive_attn is None:
                self.alive_attn = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 1)
        self.ensure_max_length()

    def update_finished(self):
        """Finalize scores and predictions."""
        # shape: (sum(~ self.is_finished), 1)
        step = len(self)
        non_finished_batch = [b for b, fin in enumerate(self.is_finished_list) if not fin[0]]
        length_penalty = self.global_scorer.length_penalty(step, alpha=self.global_scorer.alpha)
        for b in [i for i, fin in enumerate(self.is_finished_list) if fin[0]]:
            b_orig = self.original_batch_idx[b]
            score = self.beams_scores[b, 0] / length_penalty
            pred = self.alive_seq[b, 1:]
            attention = (
                self.alive_attn[
                    b,
                    :,
                    : self.src_len[b],
                ]
                if self.alive_attn is not None
                else []
            )
            self.hypotheses[b_orig].append((score, pred, attention))
        self.done = len(non_finished_batch) == 0
        if self.done:
            for b in range(self.batch_size):
                best_hyp = sorted(self.hypotheses[b], key=lambda x: x[0], reverse=True)[: self.num_hyp]
                for score, pred, attn in best_hyp:
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(attn)
            return
        self.select_indices = torch.tensor(non_finished_batch, device=self.alive_seq.device)
        self.alive_seq = self.alive_seq[self.select_indices]
        self.beams_scores = self.beams_scores[self.select_indices]
        self.src_len = self.src_len[self.select_indices]
        if self.alive_attn is not None:
            self.alive_attn = self.alive_attn[self.select_indices]
        self.original_batch_idx = self.original_batch_idx[self.select_indices]
        self.maybe_update_target_prefix(self.select_indices)

    def _is_finished_list_isin(self):
        return torch.isin(self.topk_ids, self.eos_t).tolist()

    def _is_finished_list_mps(self):
        return (self.topk_ids.unsqueeze(1) == self.eos_t).sum(dim=1).bool().tolist()


class GreedySearchLM(GreedySearch):
    def update_finished(self):
        super(GreedySearchLM, self).update_finished()

    def initialize(self, src, src_len, device=None, target_prefix=None):
        """Initialize for decoding."""

        if device is None:
            device = src.device
        self.eos_t = torch.tensor(self.eos).to(device)
        (fn_map_state, _) = super(GreedySearchLM, self).initialize(None, src_len, device, target_prefix)

        return fn_map_state, src

    def advance(self, log_probs, attn):
        super(GreedySearchLM, self).advance(log_probs, attn)

        # in LM task src_len is associated with currently generated src
        # and therefore needs to follow the generation
        self.src_len += 1
