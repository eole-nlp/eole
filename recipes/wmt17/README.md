# Translation WMT17 en-de

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/wmt17`).
---

### Tokenization methods

The following configurations as provided as example:
- `wmt17_ende_yaml`: "legacy" configuration, using already tokenized data;
- `wmt17_ende_bpe.yaml`: on-the-fly bpe tokenization, using the "official" `subword-nmt` based `bpe` transform;
- `wmt17_ende_bpe_onmt_tokenize.yaml`: on-the-fly bpe tokenization, using the `pyonmttok` based `onmt_tokenize` transform;
- `wmt17_ende_spm.yaml`: on-the-fly sentencepiece tokenization, using the official `sentencepiece` based `sentencepiece` transform;
- `wmt17_ende_spm_onmt_tokenize.yaml`: on-the-fly sentencepiece tokenization, using the `pyonmttok` based `onmt_tokenize` transform;

### Get Data and prepare

WMT17 English-German data set:

```bash
cd recipes/wmt17
bash prepare_wmt_ende_data.sh
```

Options:
- `--method`: `bpe`/`sentencepiece` (subwords method to use)
- `--encode`: `true`/`false` (tokenize all datasets, not necessary if using on the fly transforms)

If you want to use one of the aforementioned configurations with on-the-fly transforms, set `--encode false`, and either of `--method bpe`/`--method sentecepiece`.

### Train

Choose the config you want to run:

```bash
export CONFIG="wmt17_ende_bpe.yaml"
```

Training the following big transformer for 50K steps takes less than 10 hours on a single RTX 4090

```bash
eole build_vocab --config $CONFIG --n_sample -1 # --num_threads 4
eole train --config $CONFIG
```

Note: if you need to perform some visual checks on the "transformed" data, you can enable the `dump_samples` flag at the `build_vocab` stage (and specify a smaller `-n_sample` for efficiency).

Translate test sets with various settings on local GPU and CPUs.

Notes:
- the exact model path depends on the config you chose. You can check your logs for the exact path.
- the "root" model links to the last saved step, but you can choose any step subfolder if needed (e.g. `--model_path wmt17_en_de/transformer_big_bpe/step_10000`)

```bash
eole predict --src wmt17_en_de/test.src.bpe --model_path wmt17_en_de/transformer_big_bpe --beam_size 5 --batch_size 4096 --batch_type tokens --output wmt17_en_de/pred.trg.bpe --gpu 0
sed -re 's/@@( |$)//g' < wmt17_en_de/pred.trg.bpe > wmt17_en_de/pred.trg.tok
sacrebleu -tok none wmt17_en_de/test.trg < wmt17_en_de/pred.trg.tok
```

BLEU scored at 40K, 45K, 50K steps on the test set (Newstest2016)

```
{
 "name": "BLEU",
 "score": 35.4,
 "signature": "nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.0.0",
 "verbose_score": "66.2/41.3/28.5/20.3 (BP = 0.998 ratio = 0.998 hyp_len = 64244 ref_len = 64379)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "none",
 "smooth": "exp",
 "version": "2.0.0"
}
{
 "name": "BLEU",
 "score": 35.2,
 "signature": "nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.0.0",
 "verbose_score": "65.9/41.0/28.3/20.2 (BP = 1.000 ratio = 1.000 hyp_len = 64357 ref_len = 64379)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "none",
 "smooth": "exp",
 "version": "2.0.0"
}
{
 "name": "BLEU",
 "score": 35.1,
 "signature": "nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.0.0",
 "verbose_score": "66.2/41.2/28.4/20.3 (BP = 0.992 ratio = 0.992 hyp_len = 63885 ref_len = 64379)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "none",
 "smooth": "exp",
 "version": "2.0.0"
}

```
