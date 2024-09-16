#!/bin/bash

# Check if the correct number of arguments are provided
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 source.txt target.txt output.txt"
  exit 1
fi

# Assign arguments to variables
source_file=$1
target_file=$2
output_file=$3

# Ensure both source.txt and target.txt exist
if [[ ! -f "$source_file" || ! -f "$target_file" ]]; then
  echo "source.txt or target.txt does not exist."
  exit 1
fi

# Use awk to read both files and combine lines
awk 'NR==FNR{a[NR]=$0; next} {print $0 " </s></s> " a[FNR]}' "$source_file" "$target_file" > "$output_file"

