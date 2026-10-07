---
sidebar_position: 5
description: Understanding vocabulary building and format in EOLE.
---

# Vocabulary Building

EOLE accepts UTF-8 vocabulary files with one entry per line, with or without frequency counts.

## Tokens with counts

The `eole build_vocab` command writes tab-separated entries:

```text
hello\t100
world\t50
```

Here `\t` denotes a literal tab. Counts must be nonnegative integers. The `src_words_min_frequency` and `tgt_words_min_frequency` settings retain entries whose counts meet the threshold, preserving their file order.

A tab separates the count from the token. Parsing uses the last tab on the line so token text, including spaces, Unicode whitespace, and embedded tabs, is preserved. Whitespace characters are not automatically removed from tokens.

Legacy files with counts separated by spaces or other whitespace remain supported:

```text
hello 100
world 50
```

In this legacy format, tokens cannot contain whitespace. Each counted row uses tab parsing when a tab is present, and otherwise uses legacy whitespace parsing. Consequently a counted file may mix these separator styles for tokens that do not contain whitespace.

## Tokens without counts

```text
hello
world
```

Each complete line is a token, preserving its text. Frequency filtering does not apply because the file contains no counts.

## Format detection and errors

The first line determines the format for the entire file:

- A tab on the first line selects the counted format, and its count is validated.
- Otherwise, a first line with a token followed by a nonnegative integer separated by whitespace selects the legacy counted format.
- Otherwise, the file is treated as token-only.

This automatic detection is ambiguous for a token-only first entry such as `hello 100`, or one containing a tab: those entries are interpreted as counted rows. Avoid these ambiguous first entries in token-only files. A malformed first space-separated count such as `hello invalid` also cannot be distinguished from a token-only entry.

Once counted format is selected, every row must contain a nonempty token and a valid count. Errors include the file path, the one-based line number, and escaped content so invisible characters can be identified. Blank rows in counted files are rejected. Empty files are rejected. LF and CRLF line endings and a final entry without a newline are supported.

Tokenization and text cleanup belong in the corpus preprocessing or transforms. The vocabulary reader preserves Unicode token content; it does not classify invisible characters as unwanted text.
