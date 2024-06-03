# simple parser to replace original raw data

import os
import pandas as pd
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("data_dir")
args = parser.parse_args()


for dataset in ["test", "train", "validation"]:
    file_pattern = os.path.join(args.data_dir, f"{dataset}-*")
    files = sorted(glob(file_pattern))
    dfs = [pd.read_parquet(file) for file in files]
    df = pd.concat((dfs))
    out_path = os.path.join(args.data_dir, f"{dataset}.txt")
    with open(out_path, "w") as f:
        for line in df["text"]:
            if line.strip() != "":
                f.write(line)
