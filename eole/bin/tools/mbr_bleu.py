#!/usr/bin/env python3
# coding: utf-8

import sacrebleu
import codecs
from eole.bin import BaseBin, register_bin


# Let's say you have a source file with N sentences in SL - eg: source.sl
# Translate your file in TL with the -n_best nbest options nbest being
# then number of hypotheses and output the target to -output target.nbest.tl
# This script can be run (for instance with nbest = 5) as follows:
# python mbr_bleu.py --nbest-hyp target.5.tl \
#       --nbest-order 5 --output target.mbr.tl
# It will compare all hyp with eachother and output the max bleu


def chunks(lgth, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(lgth), n):
        yield lgth[i : i + n]


@register_bin(name="mbr_bleu")
class MBRBleu(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--nbest-hyp", type=str, help="file with nbest to rerank", required=True
        )
        parser.add_argument(
            "--nbest-order", type=int, help="nbest order", required=True
        )
        parser.add_argument("--output", type=str, help="output file", required=True)

    @classmethod
    def run(cls, args):
        with codecs.open(args.nbest_hyp, encoding="utf-8") as file:
            nbests = file.readlines()
            nbests = [item.strip() for item in nbests]
            nbests = chunks(nbests, args.nbest_order)

        with codecs.open(args.output, "w", encoding="utf-8") as output_file:
            best_indices = []
            for nbest in nbests:
                texts = []
                scores = []
                for hyp in nbest:
                    bleu = 0
                    for ref in nbest:
                        bleu = (
                            bleu
                            + sacrebleu.sentence_bleu(hyp, [ref]).score
                            / args.nbest_order
                        )
                        texts.append(hyp)
                        scores.append(bleu)
                max_index = scores.index(max(scores))
                output_file.write(texts[max_index] + "\n")
                best_indices.append(max_index)
