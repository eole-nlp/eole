# simple parser to replace original raw data

import os
import pandas as pd
from glob import glob
from argparse import ArgumentParser
import tiktoken
import tqdm

parser = ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--valid_size", default=10**8, help="Valid set size in tokens.")
args = parser.parse_args()


file_pattern = os.path.join(args.data_dir, "*")
files = sorted(glob(file_pattern))

enc = tiktoken.get_encoding("gpt2")

valid_path = os.path.join(args.data_dir, "fineweb10B_valid.txt")
train_path = os.path.join(args.data_dir, "fineweb10B_train.txt")

# Open validation file initially
current_path = valid_path
current_file = open(current_path, "w")

n_tokens = 0
for file in files:
    df = pd.read_parquet(file)
    for line in tqdm.tqdm(df["text"], desc=file):
        if n_tokens > args.valid_size and current_path != train_path:
            # Switch to train file once valid size is exceeded
            current_file.close()
            current_path = train_path
            current_file = open(current_path, "w")
        if n_tokens < args.valid_size:
            n_tokens += len(enc.encode_ordinary(line))
        if line.strip() != "":
            current_file.write(line + "\n")
