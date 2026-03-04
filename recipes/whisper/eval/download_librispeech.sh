#!/usr/bin/env bash
# Download and extract LibriSpeech test-clean dataset (~350MB)
# Extracts to ./LibriSpeech/test-clean/ with structure:
#   speaker/chapter/*.flac + *.trans.txt

set -euo pipefail

cd "$(dirname "$0")"

URL="https://www.openslr.org/resources/12/test-clean.tar.gz"
ARCHIVE="test-clean.tar.gz"


echo "Downloading LibriSpeech test-clean..."
wget -c "$URL" -O "$ARCHIVE"

echo "Extracting..."
tar -xf "$ARCHIVE"

echo "Done. Dataset at: $(pwd)/LibriSpeech/test-clean/"
echo "Total files: $(find LibriSpeech/test-clean -name '*.flac' | wc -l) FLAC files"
