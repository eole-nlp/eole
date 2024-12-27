#!/usr/bin/env bash

# Set default values for method and encoding
METHOD="bpe"  # Default method
ENCODE=true   # Default is to encode the data

suffix=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --method)
      if [[ "$2" == "bpe" || "$2" == "sentencepiece" ]]; then
        METHOD="$2"
        shift
      else
        echo "Invalid method specified. Use '--method bpe' or '--method sentencepiece'."
        exit 1
      fi
      ;;
    --encode)
      if [[ "$2" == "true" || "$2" == "false" ]]; then
        ENCODE="$2"
        shift
      else
        echo "Invalid value for --encode. Use 'true' or 'false'."
        exit 1
      fi
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
  shift
done

# Check if wget is installed
if ! command -v wget &>/dev/null; then
  echo "Please install wget: apt-get install wget or brew install wget"
  exit 2
fi

# Check if subword-nmt is installed for BPE
if [[ "$METHOD" == "bpe" ]] && ! command -v subword-nmt &>/dev/null; then
  echo "Please install Subword NMT: pip3 install subword-nmt"
  exit 2
fi

# Check if sentencepiece is installed for SentencePiece
if [[ "$METHOD" == "sentencepiece" ]] && (! command -v spm_train &>/dev/null || ! command -v spm_encode &>/dev/null); then
  echo "Please install SentencePiece: https://github.com/google/sentencepiece?tab=readme-ov-file#build-and-install-sentencepiece-command-line-tools-from-c-source"
  exit 2
fi

mkdir -p wmt17_en_de
cd wmt17_en_de

# Download and prepare data
if true; then
  wget -nc 'http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz'
  wget -nc 'http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz'
  wget -nc 'http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz'
  tar xf dev.tgz

  [ ! -f train.src.gz ] && ln -s corpus.tc.en.gz train.src.gz
  [ ! -f train.trg.gz ] && ln -s corpus.tc.de.gz train.trg.gz
  cat newstest2014.tc.en newstest2015.tc.en >dev.src
  cat newstest2014.tc.de newstest2015.tc.de >dev.trg
  [ ! -f test.src ] && ln -s newstest2016.tc.en test.src
  [ ! -f test.trg ] && ln -s newstest2016.tc.de test.trg

  zcat train.src.gz > train.src
  zcat train.trg.gz > train.trg

  # Learn subword model based on selected method
  if [[ "$METHOD" == "sentencepiece" ]]; then
    # Using SentencePiece
    zcat train.src.gz train.trg.gz > train.all
    spm_train --input=train.all --model_prefix=spm --vocab_size=32000 --character_coverage=1.0 --model_type=unigram
  else
    # Using BPE with subword-nmt
    zcat train.src.gz train.trg.gz | subword-nmt learn-bpe -s 32000 > codes
  fi

  # Encode data if --encode=true
  if [[ "$ENCODE" == "true" ]]; then
    suffix=".encoded.filter"
    if [[ "$METHOD" == "sentencepiece" ]]; then
      # Apply SentencePiece encoding
      for LANG in src trg; do
        zcat train.$LANG.gz | spm_encode --model=spm.model > train.$LANG.encoded
        for SET in dev test; do
          spm_encode --model=spm.model < $SET.$LANG > $SET.$LANG.encoded
        done
      done
    else
      # Apply BPE encoding with subword-nmt
      for LANG in src trg; do
        zcat train.$LANG.gz | subword-nmt apply-bpe -c codes > train.$LANG.encoded
        for SET in dev test; do
          subword-nmt apply-bpe -c codes < $SET.$LANG > $SET.$LANG.encoded
        done
      done
    fi

    # Filter the training data
    python3 ../filter_train.py
  else
    echo "Encoding skipped as per --encode=false"
  fi
fi

# Shuffle the training data
paste -d '\t' train.src$suffix train.trg$suffix | shuf | awk -v FS="\t" '{ print $1 > "train.src.shuf" ; print $2 > "train.trg.shuf" }'
