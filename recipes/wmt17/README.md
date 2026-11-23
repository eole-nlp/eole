# Train a Transformer for English–German translation

Start with a 600-step CPU run that builds a vocabulary, trains a model, translates
two held-out sentences, and scores the output. Then use the same workflow on
the WMT corpus with a GPU. Both runs train an encoder-decoder Transformer from
scratch.

English–German translation was one of the tasks used to introduce the Transformer
in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The paper used
WMT 2014. The larger run below uses the WMT17-hosted preprocessed English–German
corpus and evaluates on Newstest2016.

## Run the small example on CPU

Run from the repository root with [Eole installed](../../README.md#installation).
Install the BPE and evaluation tools in the same Python environment:

```bash
python -m pip install subword-nmt sacrebleu
```

Use a fresh output directory for each run. The bundled [English](smoke/train.en) and
[German](smoke/train.de) data consists of short example sentences. Each line in an English file corresponds to the same
line in its German partner. Training learns from the `train` pair, validation
checks progress using the `valid` pair, and final prediction reads `test.en`.

### 1. Learn subwords and build the vocabulary

```bash
export WMT17_RUN="$PWD/runs/wmt17-smoke"
export EOLE_TORCH_COMPILE=0
export OMP_NUM_THREADS=4
mkdir -p "$WMT17_RUN"
cat recipes/wmt17/smoke/train.en recipes/wmt17/smoke/train.de \
  | subword-nmt learn-bpe -s 64 > "$WMT17_RUN/codes"
eole build_vocab -c recipes/wmt17/smoke.yaml --n_sample -1 --num_threads 1
```

Byte-pair encoding (BPE) learns frequent character merges from both training
languages. The `codes` file defines how words split into subwords. Eole applies
those codes to the training text and counts the resulting pieces to create the
shared `vocab.txt`. Vocabulary and BPE learning use training data only.

### 2. Train and save a checkpoint

```bash
eole train -c recipes/wmt17/smoke.yaml
```

[smoke.yaml](smoke.yaml) configures one encoder layer and one decoder layer.
The encoder uses self-attention to mix information across the English sentence.
The decoder attends to earlier German tokens and uses cross-attention to read
the encoder's output. Sinusoidal position encodings tell both stacks about word
order. During training, the decoder learns to predict the next reference token
given earlier reference tokens.

Look for step 600 in the log and `$WMT17_RUN/model/step_600` on disk. The model
directory also links to the latest saved checkpoint.

### 3. Translate the held-out sentences and score them

```bash
eole predict --model_path "$WMT17_RUN/model/step_600" \
  --src recipes/wmt17/smoke/test.en --output "$WMT17_RUN/test.hyp.de" \
  --beam_size 2 --max_length 20 --compute_dtype fp32
paste recipes/wmt17/smoke/test.en "$WMT17_RUN/test.hyp.de" recipes/wmt17/smoke/test.de
sacrebleu recipes/wmt17/smoke/test.de --input "$WMT17_RUN/test.hyp.de" \
  --tokenize none --metrics bleu chrf
```

The saved transforms apply BPE to the input and join the output subwords.
Beam search keeps the best partial German sequences as the decoder generates.
Expect one generated line per input and a score report. BLEU compares word
sequences, while chrF compares character sequences. The three columns show
English input, generated German, and reference German. Check the wording
as well as the score. For example, the model may learn the form of a sentence
while substituting the wrong color. Two hand-written examples exercise the
pipeline and are too few to measure translation quality.

## Train on the WMT corpus

The full run needs a BF16-capable NVIDIA GPU and several GB for data plus
checkpoint storage. It uses the [WMT17 preprocessed parallel data](https://www.statmt.org/wmt17/translation-task.html).
The preparation script downloads English and German training text and the
development archive over HTTPS. It sets aside Newstest2014 and Newstest2015 for
validation and Newstest2016 for testing. It learns 32,000 shared BPE merges from
the training text.

```bash
export WMT17_RUN="$PWD/runs/wmt17-full"
export EOLE_TORCH_COMPILE=0
bash recipes/wmt17/prepare_wmt_ende_data.sh
eole build_vocab -c recipes/wmt17/wmt17_ende_bpe.yaml --n_sample -1 --num_threads 4
eole train -c recipes/wmt17/wmt17_ende_bpe.yaml
```

The [full training config](wmt17_ende_bpe.yaml) uses six encoder and six decoder
layers, 16 attention heads, a 1,024-wide model, and sinusoidal positions.
It uses the Noam learning-rate schedule with warmup, label smoothing, and
token-based batching. The config trains for 50,000 optimizer steps on one GPU
and saves a checkpoint every 5,000 steps. Adjust `training.keep_checkpoint`
if you need to limit disk usage. If VRAM is limited, reduce
`training.batch_size` and adjust `training.accum_count` to change how many
microbatches contribute to each optimizer update.

Evaluate a saved checkpoint on the test set:

```bash
eole predict --model_path "$WMT17_RUN/model/step_50000" \
  --src "$WMT17_RUN/test.en" --output "$WMT17_RUN/test.hyp.de" \
  --beam_size 5 --batch_size 4096 --batch_type tokens --gpu_ranks 0
sacrebleu "$WMT17_RUN/test.de" --input "$WMT17_RUN/test.hyp.de" \
  --tokenize none --metrics bleu chrf
```

The release supplies tokenized, truecased text. The scoring commands use
`--tokenize none` to match that preprocessing. Save SacreBLEU's signature with
the score and use validation data when tuning settings. The paper's published
scores use a different test set.

The older [WMT17 settings and results](https://github.com/eole-nlp/eole/tree/282528e25acbb2dacdd0ee1a8e319d88f06a0798/recipes/wmt17)
and WMT22 prompting experiments remain available in Git history.
[WMT22 Tower/Mistral](https://github.com/eole-nlp/eole/tree/282528e25acbb2dacdd0ee1a8e319d88f06a0798/recipes/wmt22_with_TowerInstruct-Mistral) ·
[WMT22 Tower/Llama 2](https://github.com/eole-nlp/eole/tree/282528e25acbb2dacdd0ee1a8e319d88f06a0798/recipes/wmt22_with_TowerInstruct-llama2) ·
[WMT22 Llama 3.1](https://github.com/eole-nlp/eole/tree/282528e25acbb2dacdd0ee1a8e319d88f06a0798/recipes/wmt22_with_llama3.1)
