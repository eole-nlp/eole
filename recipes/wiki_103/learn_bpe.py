# learn bpe merge operations

import os
import pyonmttok
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("data_dir")
args = parser.parse_args()

tok_args = {
    "mode": "aggressive",
    "joiner_annotate": True,
    "preserve_placeholders": True,
    "case_markup": True,
    "soft_case_regions": True,
    "preserve_segmented_tokens": True,
}
n_symbols = 40000

tokenizer_default = pyonmttok.Tokenizer(**tok_args)
learner = pyonmttok.BPELearner(tokenizer=tokenizer_default, symbols=n_symbols)
# load training corpus
learner.ingest_file(os.path.join(args.data_dir, "train.txt"))

# learn and store bpe model
tokenizer = learner.learn(os.path.join(args.data_dir, "subwords.bpe"))
