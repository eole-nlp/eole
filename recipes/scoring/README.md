# Scoring Recipes

This directory contains validated examples for EOLE scoring models and metrics.

## Native COMET

See [`comet_native/`](comet_native/) for native scoring with converted
[Unbabel COMET](https://github.com/Unbabel/COMET) / COMET-KIWI / XCOMET models:

- `EOLE-COMET` (reference-based)
- `EOLE-COMET-KIWI` (reference-free)
- `EOLE-XCOMET` (reference-based xCOMET scalar score; span parity validation supported by the recipe harness)

Converted COMET models use the generic encoder-only `transformer_encoder_scorer`
architecture with the COMET-specific `scoring_type: comet` specialization.
