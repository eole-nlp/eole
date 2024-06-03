import torch
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pad_sequence
from itertools import count, zip_longest
from copy import deepcopy
from time import time
from math import exp
import codecs

from eole.transforms import TransformPipe
from eole.constants import DefaultTokens
from eole.predict.prediction import PredictionBuilder
from eole.utils.misc import set_random_seed, report_matrix, sequence_mask
from eole.utils.alignment import build_align_pharaoh


class Inference(object):
    """Predict a batch of sentences with a saved model.

    Args:
        model (eole.modules.BaseModel): Model to use for prediction
        vocabs (dict[str, Vocab]): A dict
            mapping each side's Vocab.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`eole.predict.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`eole.predict.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`eole.predict.greedy_search.GreedySearch`.
        random_sampling_temp (float): See
            :class:`eole.predict.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`eole.predict.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`eole.predict.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        tgt_file_prefix (bool): Force the predictions begin with provided -tgt.
        data_type (str): Source data type.
        verbose (bool): Print/log every prediction.
        report_time (bool): Print/log total time/frequency.
        global_scorer (eole.predict.GNMTGlobalScorer): Prediction
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        vocabs,
        gpu=-1,
        n_best=1,
        min_length=0,
        max_length=100,
        max_length_ratio=1.5,
        ratio=0.0,
        beam_size=30,
        random_sampling_topk=0,
        random_sampling_topp=0.0,
        random_sampling_temp=1.0,
        stepwise_penalty=None,
        dump_beam=False,
        block_ngram_repeat=0,
        ignore_when_blocking=frozenset(),
        replace_unk=False,
        ban_unk_token=False,
        tgt_file_prefix=False,
        phrase_table="",
        data_type="text",
        verbose=False,
        report_time=False,
        global_scorer=None,
        out_file=None,
        report_align=False,
        gold_align=False,
        report_score=True,
        logger=None,
        seed=-1,
        with_score=False,
        return_gold_log_probs=False,
        add_estimator=False,
    ):
        self.model = model
        self.vocabs = vocabs
        self._tgt_vocab = vocabs["tgt"]
        self._tgt_eos_idx = vocabs["tgt"].lookup_token(DefaultTokens.EOS)
        self._tgt_pad_idx = vocabs["tgt"].lookup_token(DefaultTokens.PAD)
        self._tgt_bos_idx = vocabs["tgt"].lookup_token(DefaultTokens.BOS)
        self._tgt_unk_idx = vocabs["tgt"].lookup_token(DefaultTokens.UNK)
        self._tgt_sep_idx = vocabs["tgt"].lookup_token(DefaultTokens.SEP)
        self._tgt_start_with = vocabs["tgt"].lookup_token(vocabs["decoder_start_token"])
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = (
            torch.device("cuda", self._gpu) if self._use_cuda else torch.device("cpu")
        )

        self.n_best = n_best
        self.max_length = max_length
        self.max_length_ratio = max_length_ratio

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp

        self.min_length = min_length
        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {self._tgt_vocab[t] for t in self.ignore_when_blocking}
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError("replace_unk requires an attentional decoder.")
        self.tgt_file_prefix = tgt_file_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and not self.model.decoder.attentional:
            raise ValueError("Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_align = report_align
        self.gold_align = gold_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

        set_random_seed(seed, self._use_cuda)
        self.with_score = with_score

        self.return_gold_log_probs = return_gold_log_probs
        self.add_estimator = add_estimator

    @classmethod
    def from_config(
        cls,
        model,
        vocabs,
        config,  # running/predict config
        model_config,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
    ):
        """Alternate constructor.

        Args:
            model (eole.modules.BaseModel): See :func:`__init__()`.
            vocabs (dict[str, Vocab]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (eole.predict.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """
        # TODO: maybe add dynamic part

        return cls(
            model,
            vocabs,
            gpu=config.gpu,
            n_best=config.n_best,
            min_length=config.min_length,
            max_length=config.max_length,
            max_length_ratio=config.max_length_ratio,
            ratio=config.ratio,
            beam_size=config.beam_size,
            random_sampling_topk=config.random_sampling_topk,
            random_sampling_topp=config.random_sampling_topp,
            random_sampling_temp=config.random_sampling_temp,
            stepwise_penalty=config.stepwise_penalty,
            dump_beam=config.dump_beam,
            block_ngram_repeat=config.block_ngram_repeat,
            ignore_when_blocking=set(config.ignore_when_blocking),
            replace_unk=config.replace_unk,
            ban_unk_token=config.ban_unk_token,
            tgt_file_prefix=config.tgt_file_prefix,
            phrase_table=config.phrase_table,
            data_type=config.data_type,
            verbose=config.verbose,
            report_time=config.report_time,
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            gold_align=config.gold_align,
            report_score=report_score,
            logger=logger,
            seed=config.seed,
            with_score=config.with_score,
            add_estimator=model_config.add_estimator,
        )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, enc_out, src_len, enc_final_hs, batch_size, src):
        if "tgt" in batch.keys() and not self.tgt_file_prefix:
            gs, glp = self._score_target(batch, enc_out, src_len)
            self.model.decoder.init_state(
                src=src, enc_out=enc_out, enc_final_hs=enc_final_hs
            )
        else:
            gs = [0] * batch_size
            glp = None
        return gs, glp

    def _predict(
        self,
        infer_iter,
        transform=None,
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        """Predict content of ``src`` and get gold scores from ``tgt``.

        Args:
            infer_iter: tensored batch iterator from DynamicDatasetIter
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        transform_pipe = (
            TransformPipe.build_from([transform[name] for name in transform])
            if transform
            else None
        )
        prediction_builder = PredictionBuilder(
            self.vocabs,
            self.n_best,
            self.replace_unk,
            self.phrase_table,
        )

        # Statistics
        counter = count(1)
        pred_score_total, estim_total, pred_words_total = 0, 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_estim = []
        all_predictions = []

        start_time = time()

        def _maybe_retranslate(translations, batch):
            """Here we handle the cases of mismatch in number of segments
            between source and target. We re-translate seg by seg."""
            inds, perm = torch.sort(batch["ind_in_bucket"])
            trans_copy = deepcopy(translations)
            inserted_so_far = 0
            for j, trans in enumerate(translations):
                if (trans.src == self._tgt_sep_idx).sum().item() != trans.pred_sents[
                    0
                ].count(DefaultTokens.SEP):
                    self._log("Mismatch in number of ((newline))")
                    # those two should be the same except feat dim
                    # batch['src'][perm[j], :, :])
                    # trans.src

                    # we rebuild a small batch made of the sub-segments
                    # in the long segment.
                    idx = (trans.src == self._tgt_sep_idx).nonzero()
                    sub_src = []
                    start_idx = 0
                    for i in range(len(idx)):
                        end_idx = idx[i]
                        sub_src.append(batch["src"][perm[j], start_idx:end_idx, :])
                        start_idx = end_idx + 1
                    end_idx = (
                        batch["src"][perm[j], :, 0].ne(self._tgt_pad_idx).sum() - 1
                    )
                    sub_src.append(batch["src"][perm[j], start_idx:end_idx, :])
                    t_sub_src = pad_sequence(
                        sub_src, batch_first=True, padding_value=self._tgt_pad_idx
                    )
                    t_sub_src_len = t_sub_src[:, :, 0].ne(self._tgt_pad_idx).sum(1)
                    t_sub_src_ind = torch.tensor(
                        [i for i in range(len(sub_src))], dtype=torch.int16
                    )
                    device = batch["src"].device
                    t_sub_batch = {
                        "src": t_sub_src.to(device),
                        "srclen": t_sub_src_len.to(device),
                        "ind_in_bucket": t_sub_src_ind.to(device),
                    }
                    # new sub-batch ready to be predicted
                    sub_data = self.predict_batch(t_sub_batch, attn_debug)
                    sub_trans = prediction_builder.from_batch(sub_data)

                    # we re-insert the sub-batch in the initial predictions
                    trans_copy[j + inserted_so_far] = sub_trans[0]
                    for i in range(1, len(sub_src)):
                        trans_copy.insert(j + i + inserted_so_far, sub_trans[i])
                    inserted_so_far += len(sub_src) - 1
            return trans_copy

        def _process_bucket(bucket_predictions):
            bucket_scores = []
            bucket_estims = []
            bucket_preds = []
            bucket_score = 0
            bucket_estim = 0
            bucket_words = 0
            bucket_gold_score = 0
            bucket_gold_words = 0
            voc_src = self.vocabs["src"].ids_to_tokens
            bucket_predictions = sorted(
                bucket_predictions, key=lambda x: x.ind_in_bucket
            )
            for trans in bucket_predictions:
                bucket_scores += [trans.pred_scores[: self.n_best]]
                bucket_score += trans.pred_scores[0]
                bucket_estims += [trans.estim[: self.n_best]]
                bucket_estim += trans.estim[0]
                bucket_words += len(trans.pred_sents[0])
                if "tgt" in batch.keys():
                    bucket_gold_score += trans.gold_score
                    bucket_gold_words += len(trans.gold_sent) + 1

                n_best_preds = [
                    " ".join(pred) for pred in trans.pred_sents[: self.n_best]
                ]

                if self.report_align:
                    align_pharaohs = [
                        build_align_pharaoh(align)
                        for align in trans.word_aligns[: self.n_best]
                    ]
                    n_best_preds_align = [
                        " ".join(align[0]) for align in align_pharaohs
                    ]
                    n_best_preds = [
                        pred + DefaultTokens.ALIGNMENT_SEPARATOR + align
                        for pred, align in zip(n_best_preds, n_best_preds_align)
                    ]

                if transform_pipe is not None:
                    n_best_preds = transform_pipe.batch_apply_reverse(n_best_preds)

                bucket_preds += [n_best_preds]

                if self.with_score:
                    n_best_scores = [
                        score for score in trans.pred_scores[: self.n_best]
                    ]
                    n_best_estims = [estim for estim in trans.estim[: self.n_best]]
                    out_all = [
                        pred + "\t" + str(score) + "\t" + str(estim)
                        for (pred, score, estim) in zip(
                            n_best_preds, n_best_scores, n_best_estims
                        )
                    ]
                    self.out_file.write("\n".join(out_all) + "\n")
                else:
                    self.out_file.write("\n".join(n_best_preds) + "\n")
                self.out_file.flush()

                if self.verbose:
                    srcs = [voc_src[tok] for tok in trans.src[: trans.srclen]]
                    sent_number = next(counter)
                    output = trans.log(sent_number, src_raw=srcs)
                    self._log(output)

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append(DefaultTokens.EOS)
                    attns = trans.attns[0].tolist()
                    if self.data_type == "text":
                        srcs = [
                            voc_src[tok] for tok in trans.src[: trans.srclen].tolist()
                        ]
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    output = report_matrix(srcs, preds, attns)
                    self._log(output)

                if align_debug:
                    if self.gold_align:
                        tgts = trans.gold_sent
                    else:
                        tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    if self.data_type == "text":
                        srcs = [
                            voc_src[tok] for tok in trans.src[: trans.srclen].tolist()
                        ]
                    else:
                        srcs = [str(item) for item in range(len(align[0]))]
                    output = report_matrix(srcs, tgts, align)
                    self._log(output)

            return (
                bucket_scores,
                bucket_estims,
                bucket_preds,
                bucket_score,
                bucket_estim,
                bucket_words,
                bucket_gold_score,
                bucket_gold_words,
            )

        bucket_predictions = []
        prev_idx = 0

        for batch, bucket_idx in infer_iter:

            batch_data = self.predict_batch(batch, attn_debug)

            predictions = prediction_builder.from_batch(batch_data)
            is_seq2seq = hasattr(self.model, "encoder") and hasattr(
                self.model, "decoder"
            )
            if (
                is_seq2seq
                and self._tgt_sep_idx != self._tgt_unk_idx
                and (batch["src"] == self._tgt_sep_idx).any().item()
            ):
                # For seq2seq when we need to force doc to spit the same number of sents
                # TODO: this should be moved to Translator class instead of testing is_seq2seq
                predictions = _maybe_retranslate(predictions, batch)

            bucket_predictions += predictions

            if (
                not isinstance(infer_iter, list)
                and len(bucket_predictions) >= infer_iter.bucket_size
            ):
                bucket_idx += 1

            if bucket_idx != prev_idx:
                prev_idx = bucket_idx
                (
                    bucket_scores,
                    bucket_estims,
                    bucket_preds,
                    bucket_score,
                    bucket_estim,
                    bucket_words,
                    bucket_gold_score,
                    bucket_gold_words,
                ) = _process_bucket(bucket_predictions)
                all_scores += bucket_scores
                all_estim += bucket_estims
                all_predictions += bucket_preds
                pred_score_total += bucket_score
                estim_total += bucket_estim
                pred_words_total += bucket_words
                gold_score_total += bucket_gold_score
                gold_words_total += bucket_gold_words
                bucket_predictions = []

        if len(bucket_predictions) > 0:
            (
                bucket_scores,
                bucket_estims,
                bucket_preds,
                bucket_score,
                bucket_estim,
                bucket_words,
                bucket_gold_score,
                bucket_gold_words,
            ) = _process_bucket(bucket_predictions)
            all_scores += bucket_scores
            all_estim += bucket_estims
            all_predictions += bucket_preds
            pred_score_total += bucket_score
            estim_total += bucket_estim
            pred_words_total += bucket_words
            gold_score_total += bucket_gold_score
            gold_words_total += bucket_gold_words

        end_time = time()

        if self.report_score:
            msg = self._report_score("PRED", pred_score_total, len(all_scores))
            self._log(msg)
            msg = self._report_score("ESTIM", estim_total, len(all_estim))
            self._log(msg)
            if "tgt" in batch.keys() and not self.tgt_file_prefix:
                msg = self._report_score("GOLD", gold_score_total, len(all_scores))
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total prediction time (s): %.1f" % total_time)
            self._log(
                "Average prediction time (ms): %.1f"
                % (total_time / len(all_predictions) * 1000)
            )
            self._log("Tokens per second: %.1f" % (pred_words_total / total_time))
            self._log("pred_words_total: %.1f" % (pred_words_total))

        if self.dump_beam:
            import json

            json.dump(
                self.predictor.beam_accum,
                codecs.open(self.dump_beam, "w", "utf-8"),
            )

        return all_scores, all_estim, all_predictions

    def _score(self, infer_iter):
        self.with_scores = True
        score_res = []
        processed_bucket = {}
        prev_bucket_idx = 0
        for batch, bucket_idx in infer_iter:
            if bucket_idx != prev_bucket_idx:
                prev_bucket_idx += 1
                score_res += [item for _, item in sorted(processed_bucket.items())]
                processed_bucket = {}
            batch_data = self.predict_batch(batch, attn_debug=False, scoring=True)
            batch_gold_scores = batch_data["gold_score"].cpu().numpy().tolist()
            batch_tgt_lengths = batch["tgtlen"].cpu().numpy().tolist()
            batch_inds_in_bucket = batch["ind_in_bucket"]
            if self.return_gold_log_probs:
                batch_gold_log_probs = (
                    batch_data["gold_log_probs"].cpu().numpy().tolist()
                )
            else:
                batch_gold_log_probs = [
                    None for i, _ in enumerate(batch_inds_in_bucket)
                ]
            for i, ind in enumerate(batch_inds_in_bucket):
                processed_bucket[ind] = [
                    batch_gold_scores[i],
                    batch_gold_log_probs[i],
                    batch_tgt_lengths[i],
                ]
        if processed_bucket:
            score_res += [item for _, item in sorted(processed_bucket.items())]
        return score_res

    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.

        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.

        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [best.tolist() for bests in predictions for best in bests]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype,
            device=device,
        ).T
        bos_tensor = torch.full([paded_tgt.size(0), 1], bos, dtype=dtype, device=device)
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(
            len(predictions), -1, full_tgt.size(-1)
        )  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _report_score(self, name, score_total, nb_sentences):
        # In the case of length_penalty = none we report the total logprobs
        # divided by the number of sentence to get an approximation of the
        # per sentence logprob. We also return the corresponding ppl
        # When a length_penalty is used eg: "avg" or "wu" since logprobs
        # are normalized per token we report the per line per token logprob
        # and the corresponding "per word perplexity"
        if nb_sentences == 0:
            msg = "%s No predictions" % (name,)
        else:
            score = score_total / nb_sentences
            try:
                ppl = exp(-score_total / nb_sentences)
            except OverflowError:
                ppl = float("inf")
            msg = "%s SCORE: %.4f, %s PPL: %.2f NB SENTENCES: %d" % (
                name,
                score,
                name,
                ppl,
                nb_sentences,
            )
        return msg

    def _decode_and_generate(
        self,
        decoder_in,
        enc_out,
        batch,
        src_len,
        step=None,
        batch_offset=None,
        return_attn=False,
    ):

        # Decoder forward, takes [batch, tgt_len, nfeats] as input
        # and [batch, src_len, hidden] as enc_out
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        if isinstance(enc_out, tuple):
            src_max_len = enc_out[0].size(1)
            src_pad_mask = sequence_mask(src_len, src_max_len).unsqueeze(
                1
            )  # [B, 1, T_src]
        elif enc_out is not None:
            src_max_len = enc_out.size(1)  # src_len.max() ce bug était sournois
            src_pad_mask = sequence_mask(src_len, src_max_len).unsqueeze(
                1
            )  # [B, 1, T_src]
        else:
            src_pad_mask = None
        tgt_pad_mask = decoder_in.eq(self._tgt_pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        emb = self.model.tgt_emb(decoder_in, step=step)
        dec_out, dec_attn = self.model.decoder(
            emb,
            enc_out=enc_out,
            src_len=src_len,
            step=step,
            return_attn=self.global_scorer.has_cov_pen or return_attn,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
        )
        # Generator forward.
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        scores = self.model.generator(dec_out.squeeze(1))
        log_probs = log_softmax(scores, dim=-1)  # we keep float16 if FP16
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [batch_size, tgt_len, vocab ] when full sentence
        return log_probs, attn

    def predict_batch(self, batch, attn_debug):
        """Predict a batch of sentences."""
        raise NotImplementedError

    def _score_target(self, batch, enc_out, src_len):
        raise NotImplementedError

    def report_results(
        self,
        gold_score,
        gold_log_probs,
        batch,
        batch_size,
        decode_strategy,
        estim,
    ):
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "estim": None,
            "batch": batch,
            "gold_score": gold_score,
            "gold_log_probs": gold_log_probs,
        }

        if self.report_align:
            results["alignment"] = self._align_forward(
                batch, decode_strategy.predictions
            )
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        preds = []
        scores = []
        attns = []
        sorted_estim = []

        for sublist_estim, sublist_scores, sublist_preds, sublist_attns in zip(
            estim,
            decode_strategy.scores,
            decode_strategy.predictions,
            decode_strategy.attention,
        ):
            sorted_tuples = sorted(
                zip(sublist_scores, sublist_preds, sublist_attns, sublist_estim),
                key=lambda x: (x[3]),
                reverse=True,
            )
            scores.append([item[0] for item in sorted_tuples])
            preds.append([item[1] for item in sorted_tuples])
            attns.append([item[2] for item in sorted_tuples])
            sorted_estim.append([item[3] for item in sorted_tuples])

        results["scores"] = scores
        results["predictions"] = preds
        results["attention"] = attns
        results["estim"] = sorted_estim
        return results
