#!/usr/bin/env bash
set -euo pipefail

: "${WMT17_RUN:?Set WMT17_RUN to the output directory for this run}"
command -v subword-nmt >/dev/null || { echo "Install subword-nmt with python -m pip install subword-nmt" >&2; exit 1; }

data_url=https://data.statmt.org/wmt17/translation-task/preprocessed/de-en
mkdir -p "$WMT17_RUN"
cd "$WMT17_RUN"

download() {
    local file=$1
    if [[ ! -s "$file" ]]; then
        curl --fail --location --retry 2 --output "$file.part" "$data_url/$file"
        mv "$file.part" "$file"
    fi
}

download corpus.tc.en.gz
download corpus.tc.de.gz
download dev.tgz

for lang in en de; do
    gzip -dc "corpus.tc.$lang.gz" > "train.$lang"
    tar -xOf dev.tgz "newstest2014.tc.$lang" "newstest2015.tc.$lang" > "valid.$lang"
    tar -xOf dev.tgz "newstest2016.tc.$lang" > "test.$lang"
done

for split in train valid test; do
    en_lines=$(wc -l < "$split.en")
    de_lines=$(wc -l < "$split.de")
    if (( en_lines == 0 || en_lines != de_lines )); then
        echo "Expected nonempty and equally sized $split.en and $split.de" >&2
        exit 1
    fi
    echo "$split: $en_lines sentence pairs"
done

cat train.en train.de | subword-nmt learn-bpe -s 32000 > codes.part
mv codes.part codes
