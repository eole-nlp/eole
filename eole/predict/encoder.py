import torch
from eole.predict.inference import Inference
from eole.constants import ModelType
from eole.predict.greedy_search import GreedySearch
from eole.predict.beam_search import BeamSearch
from eole.utils.misc import sequence_mask


class Encoder(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelType.ENCODER:
            raise ValueError(
                f"Encoder does not support task {task}."
                f" Tasks supported: {ModelType.ENCODER}"
            )

    def predict_batch(self, batch, attn_debug):
        """Predict a batch of sentences."""
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
                )
            return self._predict_batch_with_strategy(batch, decode_strategy)

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

    def _predict_batch_with_strategy(self, batch, decode_strategy):
        """Predict a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate prediction step by step.

        Returns:
            results (dict): The prediction results.
        """
        # (0) Prep the components of the search.
        batch_size = len(batch["srclen"])

        # (1) Run the encoder on the src.
        src, enc_final_hs, enc_out, src_len = self._run_encoder(batch)

        gold_score, gold_log_probs = self._gold_score(
            batch,
            enc_out,
            src_len,
            enc_final_hs,
            batch_size,
            src,
        )

        if self.add_estimator:
            """
            # Version with encoder out average
            pad_mask1 = ~src.eq(1)
            in_estim1 = (enc_out * pad_mask1.unsqueeze(-1).float()).sum(
                dim=1
            ) / pad_mask1.sum(dim=1, keepdim=True).float()
            estim = self.model.estimator(in_estim1.half()).squeeze(-1)
            """
            # Version with first token embedding (same as COMET)
            estim = self.model.estimator(enc_out[:, 0, :]).squeeze(-1)
        else:
            estim = torch.ones([enc_out.size(0)])
        estim = [[item] for item in estim.tolist()]

        results = {
            "predictions": [[torch.tensor([0])] for _ in range(batch_size)],
            "scores": [[0]] * batch_size,
            "attention": [[0]] * batch_size,
            "estim": estim,
            "batch": batch,
            "gold_score": [0] * batch_size,
            "gold_log_probs": None,
            "alignment": [[] for _ in range(batch_size)],
        }

        return results

    def _score_target(self, batch, enc_out, src_len, src_map):
        tgt = batch["tgt"]
        tgt_in = tgt[:, :-1, :]

        log_probs, attn = self._decode_and_generate(
            tgt_in,
            enc_out,
            batch,
            src_len=src_len,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[:, 1:, :]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=1).view(-1)
        return gold_scores, None
