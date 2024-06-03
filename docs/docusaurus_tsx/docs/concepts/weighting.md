---
sidebar_position: 3
description: How to weight datasets to under/oversample on the fly.
---

# Dataset Weighting

This is naturally embedded in the data configuration format introduced in OpenNMT-py 2.0. Each entry of the `data` configuration will have its own *weight*. When building batches, we'll sequentially take *weight* example from each corpus.

**Note**: don't worry about batch homogeneity/heterogeneity, the bucketing mechanism is here for that reason. Instead of building batches one at a time, we will load `bucket_size` examples, sort them by length, build batches and then yield them in a random order.

### Example

In the following example, we will sequentially sample 7 examples from *corpus_1*, and 3 examples from *corpus_2*, and so on:

```yaml
# <your_config>.yaml

...

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        weight: 7
    corpus_2:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        weight: 3
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
...

```