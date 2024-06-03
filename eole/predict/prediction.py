""" Prediction main class """
import os
from eole.constants import DefaultTokens
from eole.utils.alignment import build_align_pharaoh


class PredictionBuilder(object):
    """
    Build a word-based prediction from the batch output
    of predictor and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data ():
       vocabs ():
       n_best (int): number of predictions produced
       replace_unk (bool): replace unknown words using attention
    """

    def __init__(self, vocabs, n_best=1, replace_unk=False, phrase_table=""):
        self.vocabs = vocabs
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table_dict = {}
        if phrase_table != "" and os.path.exists(phrase_table):
            with open(phrase_table) as phrase_table_fd:
                for line in phrase_table_fd:
                    phrase_src, phrase_trg = line.rstrip("\n").split(
                        DefaultTokens.PHRASE_TABLE_SEPARATOR
                    )
                    self.phrase_table_dict[phrase_src] = phrase_trg

    def _build_target_tokens(self, src, srclen, pred, attn, voc, dyn_voc):
        if dyn_voc is None:
            tokens = [voc[tok] for tok in pred.tolist()]
        else:
            tokens = [
                voc[tok]
                if tok < len(voc)
                else dyn_voc.ids_to_tokens[tok - len(self.vocabs["src"].ids_to_tokens)]
                for tok in pred.tolist()
            ]
        if tokens[-1] == DefaultTokens.EOS:
            tokens = tokens[:-1]

        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == DefaultTokens.UNK:
                    _, max_index = attn[i][:srclen].max(0)
                    src_tok = self.vocabs["src"].ids_to_tokens[src[max_index.item()]]
                    tokens[i] = src_tok
                    if self.phrase_table_dict:
                        if src_tok in self.phrase_table_dict:
                            tokens[i] = self.phrase_table_dict[src_tok]
        return tokens

    def from_batch(self, prediction_batch):
        batch = prediction_batch["batch"]
        if "src_ex_vocab" in batch.keys():
            dyn_voc_batch = batch["src_ex_vocab"]
        else:
            dyn_voc_batch = None
        assert len(prediction_batch["gold_score"]) == len(
            prediction_batch["predictions"]
        )
        batch_size = len(batch["srclen"])

        preds, pred_score, estim, attn, align, gold_score, ind = (
            prediction_batch["predictions"],
            prediction_batch["scores"],
            prediction_batch["estim"],
            prediction_batch["attention"],
            prediction_batch["alignment"],
            prediction_batch["gold_score"],
            batch["ind_in_bucket"],
        )

        if not any(align):  # when align is a empty nested list
            align = [None] * batch_size

        src = batch["src"]
        srclen = batch["srclen"][:]
        if "tgt" in batch.keys():
            tgt = batch["tgt"]
        else:
            tgt = None

        predictions = []
        voc_tgt = self.vocabs["tgt"].ids_to_tokens

        # These comp lists are costy but less than for loops
        for b in range(batch_size):
            if dyn_voc_batch is not None:
                dyn_voc = dyn_voc_batch[b]
            else:
                dyn_voc = None
            pred_sents = [
                self._build_target_tokens(
                    src[b, :] if src is not None else None,
                    srclen[b],
                    preds[b][n] if len(preds[b]) > 0 else None,
                    align[b][n] if align[b] is not None else attn[b][n],
                    voc_tgt,
                    dyn_voc,
                )
                for n in range(self.n_best)
            ]

            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[b, :] if src is not None else None,
                    srclen[b],
                    tgt[b, 1:] if tgt is not None else None,
                    None,
                    voc_tgt,
                    dyn_voc,
                )

            prediction = Prediction(
                src[b, :] if src is not None else None,
                srclen[b],
                pred_sents,
                attn[b],
                pred_score[b],
                estim[b],
                gold_sent,
                gold_score[b],
                align[b],
                ind[b],
            )
            predictions.append(prediction)

        return predictions


class Prediction(object):
    """Container for a predicted sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        srclen (List[int]): Source lengths.
        pred_sents (List[List[str]]): Words from the n-best predictions.
        pred_scores (List[List[float]]): Log-probs of n-best predictions.
        attns (List[FloatTensor]) : Attention distribution for each
            prediction.
        gold_sent (List[str]): Words from gold prediction.
        gold_score (List[float]): Log-prob of gold prediction.
        word_aligns (List[FloatTensor]): Words Alignment distribution for
            each prediction.
    """

    __slots__ = [
        "src",
        "srclen",
        "pred_sents",
        "attns",
        "pred_scores",
        "estim",
        "gold_sent",
        "gold_score",
        "word_aligns",
        "ind_in_bucket",
    ]

    def __init__(
        self,
        src,
        srclen,
        pred_sents,
        attn,
        pred_scores,
        estim,
        tgt_sent,
        gold_score,
        word_aligns,
        ind_in_bucket,
    ):
        self.src = src
        self.srclen = srclen
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.estim = estim
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        self.word_aligns = word_aligns
        self.ind_in_bucket = ind_in_bucket

    def log(self, sent_number, src_raw=""):
        """
        Log prediction.
        """

        msg = ["\nSENT {}: {}\n".format(sent_number, src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        best_estim = self.estim[0]
        pred_sent = " ".join(best_pred)
        msg.append("PRED {}: {}\n".format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))
        msg.append("ESTIM SCORE: {:.4f}\n".format(best_estim))

        if self.word_aligns is not None:
            pred_align = self.word_aligns[0]
            pred_align_pharaoh, _ = build_align_pharaoh(pred_align)
            pred_align_sent = " ".join(pred_align_pharaoh)
            msg.append("ALIGN: {}\n".format(pred_align_sent))

        if self.gold_sent is not None:
            tgt_sent = " ".join(self.gold_sent)
            msg.append("GOLD {}: {}\n".format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append("\nBEST HYP:\n")
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)
