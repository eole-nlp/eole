import os
import yaml
import torch
import numpy as np
import codecs
from eole.utils.misc import set_random_seed
from eole.utils.logging import init_logger, logger
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
from eole.utils.loss import LossCompute
from eole.constants import DefaultTokens, CorpusTask
from eole.transforms import get_transforms_cls, make_transforms
from eole.models.model import BaseModel
from eole.decoders.ensemble import load_test_model as ensemble_load_test_model

from argparse import ArgumentParser
from eole.bin import BaseBin, register_bin
from eole.config.cli import add_model
from eole.config.run import PredictConfig

"""
This script scores all sentences of a file using dynamic data.
For this purpose we use the same pipeline as the validation of a file
Below is an example of settings of a config.yaml file

verbose: false
n_best: 3
top_p: 0.9
beam_size: 10
world_size: 1
gpu_ranks: [0]
# use symlinks to last saved step
model_path: data/wikitext/wikitext-103-raw-v1/run/model-lm
src: data/wikitext/wikitext-103-raw-v1/lm_input.txt
output: data/wikitext/wikitext-103-raw-v1/lm_pred.txt

Output is the data and tab separated score
use the -output setting for preds + scores
Corpus PPL is in the logger.info
"""


@register_bin(name="LM_scoring")
class LMScoring(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "-config",
            "--config",
            "-c",
            required=False,
            help="Path of main YAML config file.")

    @classmethod
    def run(cls, args):
        # logic copied from eole.bin.translate -> to refactor at some point
        if args.config is not None:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        _parser = ArgumentParser()
        add_model(_parser, PredictConfig)
        config = PredictConfig(**config)
        init_logger(config.log_file)
        set_random_seed(config.seed, False)
        ppl_file = codecs.open(config.output + ".ppl", "w+", "utf-8")

        # no tensor_parallel support
        device = torch.device("cuda", config.gpu_ranks[0]) if len(config.gpu_ranks) > 0 else torch.device("cpu")
        if len(config.gpu_ranks) > 1:
            logger.warning(f"gpu_ranks is {str(config.gpu_ranks)} but only the first one will be used.")

        load_test_model = ensemble_load_test_model if len(config.model_path) > 1 else BaseModel.load_test_model
        vocabs, model, model_opt = load_test_model(config, 0)
        pad_token = vocabs["specials"].get("pad_token", DefaultTokens.PAD)
        padding_idx = vocabs["tgt"].tokens_to_ids[pad_token]
        criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx, reduction="none")
        valid_loss = LossCompute(
            criterion,
            model.generator,
            tgt_shift_index=0,
            lambda_coverage=model_opt.decoder.lambda_coverage,
            lambda_align=model_opt.decoder.lambda_align,
            vocabs=vocabs,
        )
        valid_loss.to(device)
        transforms_cls = get_transforms_cls(config._all_transform)
        transforms_cls = make_transforms(config, transforms_cls, vocabs)

        # if tgt is not precised in the inference config file, used from src
        if config.tgt is None:
            config.tgt = config.src
        infer_iter = build_dynamic_dataset_iter(
            config, transforms_cls, vocabs, task=CorpusTask.INFER, device_id=device.index
        )

        model.to(device)
        model.eval()

        cumul_loss = 0.0
        cumul_length = 0
        # Now we can pipe the full file through the model using the Iterator

        for i, (batch, bucket_idx) in enumerate(infer_iter):
            # reminder a batch includes .src .tgt .indices and it is sorted
            batch_size = len(batch["srclen"])
            src = batch["src"]
            src_len = batch["srclen"]
            # print(batch)
            outputs, attns, _ = model(src, None, src_len, with_align=False)
            # Compute and retrieve the loss for EACH sentence
            loss, _, _ = valid_loss(batch, outputs, attns)
            loss = loss.view(batch_size, -1)  # (B, T)
            losspertoken = loss.sum(1) / batch["tgt"][:, 1:].ne(padding_idx).sum(1)
            ppl = torch.exp(losspertoken)
            cumul_loss += loss.sum().item()
            cumul_length += batch["tgt"][:, 1:].ne(padding_idx).sum().cpu()
            # Now we need to rearrange the batch of ppl
            # in the original order with indices
            sent_ppl_orig = ppl.gather(
                0,
                torch.tensor(
                    sorted(range(len(batch["cid_line_number"])), key=lambda k: batch["cid_line_number"][k]),
                    device=ppl.device,
                ),
            )
            for j in range(batch_size):
                ppl_file.write(str(sent_ppl_orig[j].item()) + "\n")
        logger.info(
            "Loss: %.2f Tokens: %d Corpus PPL: %.2f"
            % (cumul_loss / cumul_length.item(), cumul_length, np.exp(cumul_loss / cumul_length))
        )
        ppl_file.close()

        os.system('paste "' + config.src + '" "' + config.output + '".ppl > "' + config.output + '"')


if __name__ == "__main__":
    parser = ArgumentParser()
    LMScoring.add_args(parser)
    args = parser.parse_args()
    LMScoring.run(args)
