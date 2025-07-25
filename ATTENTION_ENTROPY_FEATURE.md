# Attention Entropy Feature

This document describes the new attention entropy functionality added to EOLE, which allows you to monitor attention entropy during training and validation.

## Overview

Attention entropy is a measure of how "spread out" or "focused" the attention weights are. High entropy indicates that attention is distributed across many positions (less focused), while low entropy indicates that attention is concentrated on few positions (more focused).

The entropy is computed as: `H(A) = -Σ(A_ij * log(A_ij))` where `A_ij` are the attention weights.

## Features

- **Real-time monitoring**: Attention entropy is computed and logged during training
- **Configurable**: Choose which attention types and layers to monitor
- **Multiple aggregation methods**: Mean, max, or min aggregation across attention heads/layers
- **TensorBoard integration**: Entropy values are automatically logged to TensorBoard
- **Training logs**: Entropy appears in console training logs alongside other metrics

## Configuration

Add the following options to your training configuration:

```yaml
# Enable/disable attention entropy logging (default: true)
log_attention_entropy: true

# Which attention types to monitor (default: all available)
# Common types: ['std', 'self', 'context']
attention_entropy_types: null  # null means all types

# Which attention layers to monitor (default: all layers)
# Example: [0, 1, 2] to monitor only first 3 layers
attention_entropy_layers: null  # null means all layers

# How to aggregate entropy across attention types/layers
# Options: "mean", "max", "min" (default: "mean")
attention_entropy_aggregation: "mean"
```

## Usage Examples

### Basic Usage (Default Settings)
```yaml
# In your training config YAML file
log_attention_entropy: true
```

### Monitor Specific Attention Types
```yaml
log_attention_entropy: true
attention_entropy_types: ["std", "self"]  # Only monitor standard and self-attention
```

### Monitor Specific Layers
```yaml
log_attention_entropy: true
attention_entropy_layers: [0, 1, 2, 3]  # Only monitor first 4 layers
```

### Use Maximum Entropy Aggregation
```yaml
log_attention_entropy: true
attention_entropy_aggregation: "max"  # Use max instead of mean
```

## Training Output

When enabled, attention entropy will appear in your training logs:

```
Step 100; acc: 45.2; ppl: 12.34; xent: 2.51; aux: 0.123; attn_ent: 3.456; lr: 1.00e-03; ...
```

The `attn_ent: 3.456` shows the current attention entropy value.

## TensorBoard Integration

Attention entropy is automatically logged to TensorBoard under:
- `progress/attention_entropy` (for training)
- `valid/attention_entropy` (for validation)

## Implementation Details

### Files Modified/Added

1. **`eole/utils/attention_entropy.py`** (NEW)
   - Core entropy computation functions
   - Handles different attention tensor formats
   - Supports batch processing and aggregation

2. **`eole/utils/statistics.py`** (MODIFIED)
   - Added `attention_entropy` and `n_attention_samples` fields
   - Updated `update()`, `output()`, and `log_tensorboard()` methods
   - Added `avg_attention_entropy()` method

3. **`eole/utils/loss.py`** (MODIFIED)
   - Integrated entropy computation into loss calculation
   - Added configuration parameter handling
   - Updated `_stats()` method to include entropy

4. **`eole/config/training.py`** (MODIFIED)
   - Added configuration options for attention entropy
   - Includes validation and documentation

5. **`eole/utils/__init__.py`** (MODIFIED)
   - Added imports for attention entropy functions

### Key Functions

- `compute_attention_entropy()`: Computes entropy for a single attention tensor
- `compute_attention_entropy_from_dict()`: Computes entropy from attention dictionary
- `compute_batch_attention_entropy()`: Convenience function for batch-level entropy
- `aggregate_attention_entropy()`: Aggregates entropy across multiple sources

## Interpreting Attention Entropy

### Typical Values
- **High entropy (>4.0)**: Very distributed attention, model is "looking everywhere"
- **Medium entropy (2.0-4.0)**: Moderately focused attention, typical for well-trained models
- **Low entropy (<2.0)**: Very focused attention, model is concentrating on few positions

### What Changes in Entropy Mean
- **Decreasing entropy over training**: Model is learning to focus attention more precisely
- **Increasing entropy**: Model attention is becoming more distributed
- **Stable entropy**: Model has reached a stable attention pattern

### Use Cases
1. **Training monitoring**: Track how attention patterns evolve during training
2. **Model comparison**: Compare attention behaviors between different architectures
3. **Debugging**: Identify if attention is too focused or too distributed
4. **Research**: Analyze attention patterns for interpretability studies

## Performance Impact

The attention entropy computation has minimal performance impact:
- Computation is done only when attention weights are already available
- Uses efficient PyTorch operations
- Can be disabled by setting `log_attention_entropy: false`

## Troubleshooting

### Entropy Values Are Always Zero
- Check that `log_attention_entropy: true` in your config
- Verify that your model actually computes attention weights
- Ensure attention weights are being returned by the model

### Entropy Values Seem Too High/Low
- Check the `attention_entropy_aggregation` setting
- Verify which attention types/layers are being monitored
- Consider the model architecture and expected attention patterns

### Performance Issues
- Disable entropy logging with `log_attention_entropy: false`
- Monitor specific layers only with `attention_entropy_layers`
- Use fewer attention types with `attention_entropy_types`

## Example Training Command

```bash
eole train \
    --config your_config.yaml \
    --log_attention_entropy true \
    --attention_entropy_aggregation mean \
    --tensorboard true
```

## Future Enhancements

Potential future improvements:
- Per-head entropy analysis
- Entropy-based regularization
- Attention entropy visualization tools
- Integration with attention analysis tools

## References

- Shannon, C. E. (1948). A mathematical theory of communication.
- Attention entropy has been used in various NLP papers for analyzing attention patterns.
