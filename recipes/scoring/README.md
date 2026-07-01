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

## Native MetricX

See [`metricx_native/`](metricx_native/) for native scoring with converted
[Google MetricX](https://github.com/google-research/metricx) models:

- `EOLE-METRICX` (reference-based)
- `EOLE-METRICX-QE` (reference-free)

Converted MetricX models use the generic encoder-decoder
`transformer_encoder_decoder_scorer` architecture with template-driven scorer
inputs. MetricX raw scores are lower-is-better and are preserved as raw scores
by the native scorer.

## Custom Scorer Modules

Custom validation scorers can be registered from external Python modules without
modifying EOLE source code:

```yaml
scorer_modules:
  - my_private_eole_scorers.custom_metric

scorer_configs:
  PRIVATE-METRIC:
    model: /path/to/private/model

valid_metrics:
  - PRIVATE-METRIC

early_stopping_criteria: PRIVATE-METRIC
```

The module is imported before validation scorers are resolved. Use the existing
scorer registry decorator and declare the early-stopping direction if the metric
should be usable as an early-stopping criterion:

In distributed training, scorer modules are imported in each training process,
so module import side effects should be idempotent.

```python
from eole.scorers import register_scorer
from eole.scorers.scorer import Scorer


@register_scorer(metric="PRIVATE-METRIC")
class PrivateMetricScorer(Scorer):
    higher_is_better = True

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        ...
```

Custom scorers can read their private settings from `config.scorer_configs`
without adding scorer-specific fields to EOLE's public config.
