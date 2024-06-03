# Performance tips

* use `fp16`
* use `batch_size_multiple` 8
* use `vocab_size_multiple` 8
* Depending on the number of GPU use num_workers 4 (for 1 GPU) or 2 (for multiple GPU)
* To avoid averaging checkpoints you can use the "during training" average decay system.
* If you train a transformer we support `max_relative_positions` (use 20) instead of position_encoding. 
* for very fast inference convert your model to CTranslate2 format.