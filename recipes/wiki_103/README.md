# Language Model Wiki-103


## Step 1: Download and clean the data, prepare subword model

Preliminary steps are defined in the `prepare_wikitext-103_data.sh` script.

The following command will download the [WikiText103 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/), remove empty lines and shuffle the training corpus. It will also call the `learn_bpe.py` script to train a bpe of 40000 symbols on the training dataset using pyonmttok. The bpe model will be stored in `data/wikitext/wikitext-103-raw-v1/subwords.bpe`.

```bash
chmod u+x prepare_wikitext-103_data.sh
./prepare_wikitext-103_data.sh
```

## Step 2: Build the vocabulary
An example of yaml configuration for language modeling task is available in `wiki_103.yaml`. This configuration will be used for building the vocabulary and training the model.
BPE and language modeling specificities are explained in the following sections.

### Language Model specificities

In LM tasks we expect a single source, therefore path_tgt is not required for LM tasks.

```yaml
data:
    corpus_1:
        path_src: data/wikitext-103-raw/wiki.train.raw
```

### BPE specificities

To use BPE tokenization on the fly, the following parameters must be in the config file.
Slight differences between on the fly tokenization and outputed tokenized files from step 1 can be observed.

```yaml
transforms: [onmt_tokenize, filtertoolong]
transforms_configs:
  onmt_tokenize:
    src_subword_type: bpe
    src_subword_model: data/wikitext-103-raw/subwords.bpe
    src_onmttok_kwargs: {"mode": "aggressive", "joiner_annotate": True, "preserve_placeholders":
    True, "case_markup": True, "soft_case_regions": True, "preserve_segmented_tokens":
    True}
```

### Build vocabulary command
The vocabulary is built using:
```bash
eole build_vocab -config wiki_103.yaml -n_sample -1 # -num_threads 4
```

## Step 3: Train the model

The training is launched using:
```bash
eole train -config wiki_103.yaml
```
Tensorboard can be used to monitor the training.

**Expected results:** perplexity of 20-22 on the validation set.

## Step 4: Generate output
Options contained in the loaded model will trigger language modeling specific inference.

`input.txt` must contain already tokenized examples, with the same method as the training data. Here, part of validation data will be used:
```bash
head data/wikitext-103-raw/wiki.valid.bpe | cut -d" " -f-15 > data/wikitext/wikitext-103-raw-v1/test_input.txt
```

To proceed with LM inference, sampling methods such as top-k sampling or nucleus sampling are usually applied. Details and options about inference methods can be found in [`eole/config/inference.py`](https://github.com/eole-nlp/eole/tree/master/eole/config/inference.py).

The following command will provide inference with nucleus sampling of p=0.9 and return the 3 sequences with the lowest perplexity out of the 10 generated sequences (see `inference.yaml`):
```bash
eole predict -config inference.yaml -model_path data/wikitext/wikitext-103-raw-v1/run/model-lm/step_1000000 -src data/wikitext/wikitext-103-raw-v1/test_input.txt -output data/wikitext/wikitext-103-raw-v1/test_pred.txt
```

**Note**: main transform-related settings are now stored within the model and its configuration, which makes the (rather complex) `inference.yaml` config not strictly necessary anymore. The above command can be converted to a simple command line with the desired settings:
```bash
eole predict -model_path data/wikitext/wikitext-103-raw-v1/run/model-lm/step_1000000 -src data/wikitext/wikitext-103-raw-v1/test_input.txt -output data/wikitext/wikitext-103-raw-v1/test_pred.txt -world_size 1 -gpu_ranks 0 -n_best 3 -top_p 0.9 -beam_size 10
```