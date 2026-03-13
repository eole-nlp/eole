"""
Language model perplexity scorer.

Scores all sentences of a file and computes per-sentence and corpus
perplexity using the current inference infrastructure.

Example inference.yaml config::

    model_path: /path/to/model
    src: /path/to/input.txt
    output: /path/to/output.ppl
    world_size: 1
    gpu_ranks: [0]
    batch_size: 16

The output file will contain one perplexity value per line.
Corpus PPL is reported in the log.
"""

import codecs
import math

from eole.bin import register_bin
from eole.bin.run import RunBin
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY
from eole.utils.logging import init_logger, logger
from eole.utils.misc import set_random_seed, use_gpu


@register_bin(name="LM_scoring")
class LMScoring(RunBin):
    """Compute perplexity of a language model on a dataset.

    Uses the standard inference engine to score each sentence and reports
    per-sentence perplexity to the output file as well as corpus-level
    perplexity in the log.
    """

    config_class = PredictConfig
    require_config = False

    @classmethod
    def run(cls, args):
        config = cls.build_config(args)
        init_logger(config.log_file)
        set_random_seed(config.seed, use_gpu(config))

        engine = InferenceEnginePY(config)
        # score_file uses config.src as both source and target reference,
        # which is the correct setup for LM perplexity scoring.
        score_results = engine.score_file()
        engine.terminate()

        # score_results is a list of [gold_score, gold_log_probs, tgt_length]
        # gold_score = sum of token log-probabilities (negative value)
        # tgt_length = number of scored tokens
        cumul_loss = 0.0
        cumul_length = 0

        with codecs.open(config.output, "w+", "utf-8") as ppl_file:
            for gold_score, _gold_log_probs, tgt_length in score_results:
                if tgt_length > 0:
                    sent_ppl = math.exp(-gold_score / tgt_length)
                else:
                    sent_ppl = float("inf")
                ppl_file.write(f"{sent_ppl:.4f}\n")
                cumul_loss += -gold_score
                cumul_length += tgt_length

        if cumul_length > 0:
            corpus_ppl = math.exp(cumul_loss / cumul_length)
        else:
            corpus_ppl = float("inf")

        logger.info(
            "Loss: %.2f Tokens: %d Corpus PPL: %.2f" % (cumul_loss, cumul_length, corpus_ppl)
        )

