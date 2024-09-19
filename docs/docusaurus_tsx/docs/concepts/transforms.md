---
sidebar_position: 4
description: Recap of available on-the-fly data transforms.
---

# Data Transforms

It's your lucky day! We already embedded several transforms that can be used easily.

Note: all the details about every flag and options for each transform can be found in the [Transforms Config](../reference/Config/transforms.md) section.

### Transform Types

The concept of `TransformType` was introduced to facilitate transparent configuration management. The underlying issue at stake is that all transforms are not meant to be used in the same concept. For instance, the `filtertoolong` transform is meant as a "safeguard" to limit the size of training batches. Enabling this transform when predicting can introduce some unwanted behaviours and poor results.
For now, the possible transform types are:
- `Default` // `"any"`: usable in any context (default unless specified otherwise in the transform class definition);
- `Train` // `"train"`: usable only in training context;
- `Predict` // `"predict"`: usable only in prediction context.

This concept might be extended later for various needs, such as different data types, etc.

### General purpose

#### Filter examples by length

Transform name: `filtertoolong`

Class: `eole.transforms.misc.FilterTooLongTransform`

The following options can be added to the configuration :
- `src_seq_length`: maximum source sequence length;
- `tgt_seq_length`: maximum target sequence length.

#### Add custom prefix to examples

Transform name: `prefix`

Class: `eole.transforms.misc.PrefixTransform`

For each dataset that the `prefix` transform is applied to, you can set the additional `src_prefix` and `tgt_prefix` parameters in its data configuration:

```yaml
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        transforms: [prefix]
        weight: 1
        src_prefix: __some_src_prefix__
        tgt_prefix: __some_tgt_prefix__
```

At inference if you want to use the target prefix feature to prefix your target segment with a unique prefix (as opposed to a target prefix coming from a line-by-line file)
you need to set your yaml file as follow (example given with a target language as in the NLLB-200 case):
``` yaml
tgt_prefix: "spa_Latn" 
tgt_file_prefix: true
```

#### Add custom suffix to examples

Transform name: `suffix`

Class: `eole.transforms.misc.SuffixTransform`

For each dataset that the `suffix` transform is applied to, you can set the additional `src_suffix` and `tgt_suffix` parameters in its data configuration:

```yaml
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        transforms: [suffix]
        weight: 1
        src_suffix: __some_src_suffix__
        tgt_suffix: __some_tgt_suffix__
```

#### Convert examples to uppercase

Transform name: `uppercase`

Class: `eole.transforms.uppercase.UpperCaseTransform`

Converts source and target (if present) examples to uppercase so the model can learn better to translate
sentences in all caps. This transform normalizes the examples so the uppercased strings are stripped from
any diacritics and accents. Usually this is desirable for most languages, although there are few exceptions.

The following option can be added to the main configuration (same ratio for all dataset with this transform):
- `upper_corpus_ratio`: ratio of the corpus that will be transformed to uppercase (default: 0.01);

#### Normalize punctuation

Transform name: `normalize`

Class: `eole.transforms.normalize.NormalizeTransform`

Normalizes source and target (if present) examples using the same rules as Moses punctuation normalizer.

The following options can be added to the configuration of each dataset:
- `src_lang`: en, de, cz/cs, fr (default='')
- `tgt_lang`: en, de, cz/cs, fr (default='')
- `penn`: Penn substitution (default=True)
- `norm_quote_commas`: Normalize quotations and commas (default=True)
- `norm_numbers`: Normalize numbers (default=True)
- `pre_replace_unicode_punct`: Replace unicode punct (default=False)
- `post_remove_control_chars`: Remove control chars (default=False)

#### Clean dataset

Transform name: `clean`

Class: `eole.transforms.clean.CleanTransform`

Cleans source and target (if present) examples using a set of rules.

The following options can be added to the configuration of each dataset:
- `src_eq_tgt`: Remove example when source=target (default=True)
- `same_char`: Remove example if the same char is repeated 4 times (default=True)
- `same_word`: Remove example if the same word is repeated 3 times (default=True)
- `script_ok`: Remove example which contains chars that do not belong to these scripts (default=['Latin', 'Common'])
- `script_nok`: Remove example which contains chars that belong to these scripts  (default=[])
- `src_tgt_ratio`: Remove example for which src/tgt ration is <1/ratio or >ratio (default=2)
- `avg_tok_min`: Remove example for which the average token length is < X (default=3)
- `avg_tok_max`: Remove example for which the average token length is > X (default=20)
- `lang_id`: Remove example for which detected language is not in [] (default=['en', 'fr'])

#### Context / Doc aware transform

Transform name: `docify`

Class: `eole.transforms.docify.DocifyTransform`

Concatenates several segments into one, separated with a delimiter.

Pre-requisite:

Dataset must be "Docs" separated by an empty line which will make clear a story ends at this empty line.

The following options can be added to the main configuration (same options for all dataset with this transform):
- `doc_length`: max token to be concatenated (default=200)
- `max_context`: number of delimiter (default=1 , ie 2 segments concatenated)

When working with several workers, this require some precaution in order to make sure "doc" are read linearly.

`max_context + 1` needs to be a multiple of `stride` = `Number of gpu x num_workers`

Example: `max_context=1` and 1 GPU, then num_workers must be 2 or 4.


#### Augment source segments with fuzzy matches for Neural Fuzzy Repair

Transform name: `fuzzymatch`

Class: `eole.transforms.fuzzymatch.FuzzyMatchTransform`

Augments source segments with fuzzy matches for Neural Fuzzy Repair, as described in [Neural Fuzzy Repair: Integrating Fuzzy Matches into Neural Machine Translation](https://aclanthology.org/P19-1175). Currently, the transform augments source segments with only a single fuzzy match.
The Translation Memory (TM) format should be a flat text file, with each line containing the source and the target segment separated by a delimiter. As fuzzy matching during training is computational intensive, we offer some advice to achieve good performance and minimize overhead:

- Depending on your system's specs, you may have to experiment with the options `bucket_size`, `bucket_size_init`, and `bucket_size_increment`;
- You should increase the `num_workers` and `prefetch_factor` so your GPU does not have to wait for the batches to be augmented with fuzzy matches;
- Try to use a sensible Translation Memory size. 200k-250k translation units should be enough for yielding a sufficient number of matches;
- Although the transform performs some basic filtering both in the TM and in the corpus for very short or very long segments, some examples may still be long enough, so you should increase a bit the `src_seq_length`;
- Currently, when using `n_sample`, examples are always processed one by one and not in batches.

The following options can be added to the main configuration (valid for all datasets using this transform):
- `tm_path`: The path to the Translation Memory text file;
- `fuzzy_corpus_ratio`: Ratio of corpus to augment with fuzzy matches (default: 0.1);
- `fuzzy_threshold`: The fuzzy matching threshold (default: 70);
- `tm_delimiter`: The delimiter used in the flat text TM (default: "\t");
- `fuzzy_token`: The fuzzy token to be added with the matches (default: "｟fuzzy｠");
- `fuzzymatch_min_length`: Min length for TM entries and examples to match (default: 4);
- `fuzzymatch_max_length`: Max length for TM entries and examples to match (default: 70).

#### Augment source and target segments with inline tags

Transform name: `inlinetags`

Class: `eole.transforms.inlinetags.InlineTagsTransform`

Augments source and target segments with inline tags (placeholders). The transform adds 2 kind of tags, paired tags (an opening and a closing tag) and isolated (standalone) tags, and requires a tab-delimited dictionary text file with source and target terms and phrases. A dictionary with 20-30k entries is recommended. User-defined tags must include the number placeholder #, e.g. "｟user_start_tag_#｠".

The following options can be added to the main configuration (valid for all datasets using this transform):
- `tags_dictionary_path`: The path to the dictionary text file;
- `tags_corpus_ratio`: Ratio of corpus to augment with inline tags (default: 0.1);
- `max_tags`: Maximum number of tags that can be added to a single sentence. (default: 12);
- `paired_stag`: The format of an opening paired inline tag. Must include the character # (default: "｟ph_#_beg｠");
- `paired_etag`: The format of a closing paired inline tag. Must include the character # (default: "｟ph_#_end｠");
- `isolated_tag`: The format of an isolated inline tag. Must include the character # (default: "｟ph_#_std｠");
- `src_delimiter`: Any special token used for augmented src sentences (default: "｟fuzzy｠");

#### Make the model learn to use terminology

Transform name: `terminology`

Class: `eole.transforms.terminology.TerminologyTransform`

Augments source segments with terms so the model can learn to use user-provided terms at inference. It requires a dictionary with source and target terms, delimited with a tab. The transform uses Spacy's lemmatization facilities in order to a) solve the word inflection problem when searching for terms in any form, and b) make the model inflect correctly most target terms at inference. The lemmatization is applied at the dictionary entries and also at the source and target examples, and the term searches during training are performed on the lemmatized examples.
 The format of a processed segment augmented with terms is as follows:
`This is an ｟src_term_start｠ augmented ｟tgt_term_start｠ target_lemma_for_augmented ｟tgt_term_end｠ example.`
The following options can be added to the main configuration (valid for all datasets using this transform):
- `termbase_path`: The path to the dictionary text file;
- `src_spacy_language_model`: Name of the spacy language model for the source corpus;
- `tgt_spacy_language_model`: Name of the spacy language model for the target corpus;
- `term_corpus_ratio`: Ratio of corpus to augment with terms # (default: 0.3);
- `term_example_ratio`: Max terms allowed in an example # (default: 0.2);
- `src_term_stoken`: The source term start token # (default: "｟src_term_start｠");
- `tgt_term_stoken`: The target term start token # (default: "｟tgt_term_start｠");
- `tgt_term_etoken`: The target term end token # (default: "｟tgt_term_end｠");
- `term_source_delimiter`: Any special token used for augmented src sentences. The default is the fuzzy token used in the FuzzyMatch transform # (default: "｟fuzzy｠");

### Tokenization

Common options for the tokenization transforms are the following:

- `src_subword_model`: path of source side (or both if shared) subword model;
- `tgt_subword_model`: path of target side subword model;
- `src_subword_nbest`: number of candidates for subword regularization (sentencepiece), source side;
- `tgt_subword_nbest`: number of candidates for subword regularization (sentencepiece), target_side;
- `src_subword_alpha`: smoothing parameter for sentencepiece regularization / dropout probability for BPE, source side;
- `tgt_subword_alpha`: smoothing parameter for sentencepiece regularization / dropout probability for BPE, target side.

#### [OpenNMT Tokenizer](https://github.com/opennmt/Tokenizer)

Transform name: `onmt_tokenize`

Class: `eole.transforms.tokenize.OnmtTokenizerTransform`

Additional options are available:
- `src_subword_type`: type of subword model for source side (from `["none", "sentencepiece", "bpe"]`);
- `tgt_subword_type`: type of subword model for target side (from `["none", "sentencepiece", "bpe"]`);
- `src_onmttok_kwargs`: additional kwargs for pyonmttok Tokenizer class, source side;
- `tgt_onmttok_kwargs`: additional kwargs for pyonmttok Tokenizer class, target side.

#### [SentencePiece](https://github.com/google/sentencepiece)

Transform name: `sentencepiece`

Class: `eole.transforms.tokenize.SentencePieceTransform`

The `src_subword_model` and `tgt_subword_model` should be valid sentencepiece models.

#### [BPE subword-nmt](https://github.com/rsennrich/subword-nmt)

Transform name: `bpe`

Class: `eole.transforms.tokenize.BPETransform`

The `src_subword_model` and `tgt_subword_model` should be valid BPE models.

### BART-style noise

BART-style noise is composed of several parts, as described in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461).

These different types of noise can be controlled with the following options:

- `permute_sent_ratio`: proportion of sentences to permute (default boundaries are ".", "?" and "!");
- `rotate_ratio`: proportion of inputs to permute;
- `insert_ratio`: proportion of additional random tokens to insert;
- `random_ratio`: proportion of tokens to replace with random;
- `mask_ratio`: proportion of words/subwords to mask;
- `mask_length`: length of masking window (from `["subword", "word", "span-poisson"]`);
- `poisson_lambda`: $\lambda$ value for Poisson distribution to sample span length (in the case of `mask_length` set to `span-poisson`);
- `replace_length`: when masking N tokens, replace with 0, 1, " "or N tokens. (set to -1 for N).

### SwitchOut and sampling

#### [SwitchOut](https://arxiv.org/abs/1808.07512)

Transform name: `switchout`

Class: `eole.transforms.sampling.SwitchOutTransform`

Options:

- `switchout_temperature`: sampling temperature for SwitchOut.

#### Drop some tokens

Transform name: `tokendrop`

Class: `eole.transforms.sampling.TokenDropTransform`

Options:

- `tokendrop_temperature`: sampling temperature for token deletion.

#### Mask some tokens

Transform name: `tokenmask`

Class: `eole.transforms.sampling.TokenMaskTransform`

Options:

- `tokenmask_temperature`: sampling temperature for token masking.
