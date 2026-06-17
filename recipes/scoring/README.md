# Scoring Recipes

This directory contains validated examples for EOLE scoring models and metrics.

## Native COMET

See [`comet_native/`](comet_native/) for native scoring with converted
[Unbabel COMET](https://github.com/Unbabel/COMET) / COMET-KIWI models:

- `EOLE-COMET` (reference-based)
- `EOLE-COMET-KIWI` (reference-free)

Converted COMET models use the generic encoder-only `transformer_encoder_scorer`
architecture with the COMET-specific `scoring_type: comet` specialization.
