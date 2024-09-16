#!/bin/bash

##################################################################################
# This script will download wikitext-103-raw and will do basic data preparation
# for BPE and training
##################################################################################

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# set relevant paths
SP_PATH=/usr/local/bin
DATA_PATH=./data
TEST_PATH=$DATA_PATH/test

CUR_DIR=$(pwd)

# Download the default datasets into the $DATA_PATH; mkdir if it doesn't exist
mkdir -p $DATA_PATH
# cd $DATA_PATH

echo "Downloading and extracting WikiText-103 (183 MB) for training and inference..."
# These files are not publicly available anymore
# wget --trust-server-names https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
# unzip wikitext-103-raw-v1.zip
# rm wikitext-103-raw-v1.zip
# cd wikitext-103-raw
huggingface-cli download wikitext --repo-type dataset --local-dir $DATA_PATH --revision b08601e04326c79dfdd32d625aee71d232d685c3
python3 parse_wikitext_103_parquet.py $DATA_PATH/wikitext-103-raw-v1
cd $DATA_PATH/wikitext-103-raw-v1


echo "Removing empty lines and shuffling training data"
sed -r '/^\s*$/d' -i train.txt # wiki.train.raw
sed -r '/^\s*$/d' -i validation.txt # wiki.valid.raw
sed -r '/^\s*$/d' -i test.txt # wiki.test.raw
sort --random-source=<(get_seeded_random 42) -R -o train.txt train.txt # wiki.train.raw wiki.train.raw

cd $CUR_DIR
python3 learn_bpe.py $DATA_PATH/wikitext-103-raw-v1