# Changelog

This is just a centralised version of the Github automatically generated Release changelogs.
## 0.2.0

* Fix docs build/deploy by @francoishernandez in https://github.com/eole-nlp/eole/pull/216
* Enable HF nllb conversion by @francoishernandez in https://github.com/eole-nlp/eole/pull/204
* Introduce pure BF16 training with Kaha summation - (torch-optimi package) by @vince62s in https://github.com/eole-nlp/eole/pull/213
* Ensure unicode support, strip carriage returns from vocab by @ArtanisTheOne in https://github.com/eole-nlp/eole/pull/215
* Recipe to train estimator for Eurollm by @vince62s in https://github.com/eole-nlp/eole/pull/219
* Support Mistral-3.1-24B by @vince62s in https://github.com/eole-nlp/eole/pull/220
* Fix typo in wmt17 readme configuration names by @francoishernandez in https://github.com/eole-nlp/eole/pull/224
* better lora merging + fixes by @vince62s in https://github.com/eole-nlp/eole/pull/226

## 0.1.2

* quick fixes by @vince62s in https://github.com/eole-nlp/eole/pull/207
* push rope back to encoder/decoder by @vince62s in https://github.com/eole-nlp/eole/pull/208
* Keep track of datasets stats - log at validation by @vince62s in https://github.com/eole-nlp/eole/pull/209
* bug in estim translator by @vince62s in https://github.com/eole-nlp/eole/pull/210

## 0.1.1

* fix rope when very long sequence precision is key by @vince62s in https://github.com/eole-nlp/eole/pull/200
* Better fix for long rope (training was not optimized) by @vince62s in https://github.com/eole-nlp/eole/pull/201
* add filtertooshort transform by @vince62s in https://github.com/eole-nlp/eole/pull/202
* Basic pixtral support, paving the way for vision models 🖼️ by @francoishernandez in https://github.com/eole-nlp/eole/pull/153
* Clean / rename / simplify by @vince62s in https://github.com/eole-nlp/eole/pull/203

## 0.1.0

* reinstate cuda rmsnorm (much faster in fp16/awq) + ct2 enc/dec config by @vince62s in https://github.com/eole-nlp/eole/pull/167
* [patch] remove dummy_load, move gpu_ranks warning out of TrainingConfig by @francoishernandez in https://github.com/eole-nlp/eole/pull/168
* fix batch inference by @vince62s in https://github.com/eole-nlp/eole/pull/169
* Code clean-ups by @vince62s in https://github.com/eole-nlp/eole/pull/171
* 120 columns makes more sense on modern screens by @vince62s in https://github.com/eole-nlp/eole/pull/176
* refactor transformer decoder and revamp the left padding attention mask by @vince62s in https://github.com/eole-nlp/eole/pull/178
* Major refactoring of convert HF by @francoishernandez in https://github.com/eole-nlp/eole/pull/156
* [patch] handle self_attn_backend edge case by @francoishernandez in https://github.com/eole-nlp/eole/pull/180
* hotfix post #178 by @vince62s in https://github.com/eole-nlp/eole/pull/181
* fix update vocab param loading by @vince62s in https://github.com/eole-nlp/eole/pull/184
* remove verbosity at validation/scoring by @vince62s in https://github.com/eole-nlp/eole/pull/185
* [patch] Add missing `is_train` kwarg in `tokenize_id` by @francoishernandez in https://github.com/eole-nlp/eole/pull/187
* Hugging face dataset streaming support by @vince62s in https://github.com/eole-nlp/eole/pull/177
* misc fixes by @vince62s in https://github.com/eole-nlp/eole/pull/192
* Gemma2 support by @francoishernandez in https://github.com/eole-nlp/eole/pull/160
* [convert_HF] handle special tokens defined in tokenizer_config.json by @francoishernandez in https://github.com/eole-nlp/eole/pull/196
* patch max_length handling in tokenize_id by @francoishernandez in https://github.com/eole-nlp/eole/pull/197

## 0.0.3

* [patch] minor fixes for 0.0.2 by @francoishernandez in https://github.com/eole-nlp/eole/pull/109
* **Rework handling of special tokens** by @francoishernandez in https://github.com/eole-nlp/eole/pull/45
* [patch] get_transforms_cls after update_config_with_checkpoint by @francoishernandez in https://github.com/eole-nlp/eole/pull/110
* [patch] get_transforms_cls after update_config_with_checkpoint BIS by @francoishernandez in https://github.com/eole-nlp/eole/pull/111
* Updated translator.py to handle updated special token logic when computing alignments by @dameikle in https://github.com/eole-nlp/eole/pull/113
* clearer log by @vince62s in https://github.com/eole-nlp/eole/pull/112
* fix training tensor parallel by @vince62s in https://github.com/eole-nlp/eole/pull/115
* restore all_reduce directly but with detach.clone first - fix #115 by @vince62s in https://github.com/eole-nlp/eole/pull/116
* **Initial support for Metal Performance Shaders (MPS)** by @dameikle in https://github.com/eole-nlp/eole/pull/98
* Manage `share_decoder_embeddings` in `convert_HF`, misc fixes and improvements by @francoishernandez in https://github.com/eole-nlp/eole/pull/121
* Deduce share_decoder_embeddings from HF tie_word_embeddings flag by @francoishernandez in https://github.com/eole-nlp/eole/pull/123
* [docs] Upgrading docusaurus packages, should fix dependabot warnings by @francoishernandez in https://github.com/eole-nlp/eole/pull/124
* **add estimator in decoder-only** + clean code by @vince62s in https://github.com/eole-nlp/eole/pull/120
* fineweb10B/gpt2 recipe, and supporting changes by @francoishernandez in https://github.com/eole-nlp/eole/pull/32
* enable pure bf16 training by @vince62s in https://github.com/eole-nlp/eole/pull/133
* Update WMT17 recipe with working tokenization transforms examples by @francoishernandez in https://github.com/eole-nlp/eole/pull/129
* fixes #131, module 'eole.utils' has no attribute 'distributed' error when training multi-gpu by @isanvicente in https://github.com/eole-nlp/eole/pull/132
* add estimator in greedy inference by @vince62s in https://github.com/eole-nlp/eole/pull/135
* Some QOL config/saving improvements by @francoishernandez in https://github.com/eole-nlp/eole/pull/134
* fix #136. Updated eole/bin/model/average_models.py to work with safetensors model format. by @isanvicente in https://github.com/eole-nlp/eole/pull/137
* fix head dim in rope by @vince62s in https://github.com/eole-nlp/eole/pull/140
* fix autocast at scoring when doing AMP by @vince62s in https://github.com/eole-nlp/eole/pull/141
* Some minor fixes by @francoishernandez in https://github.com/eole-nlp/eole/pull/143
* fix lora lm head by @vince62s in https://github.com/eole-nlp/eole/pull/142
* fix missing pad change by @vince62s in https://github.com/eole-nlp/eole/pull/148
* flash_attn_func does not support padding mask maybe we need to drop a… by @vince62s in https://github.com/eole-nlp/eole/pull/149
* fix maybe_retranslate when number of newline does not match by @vince62s in https://github.com/eole-nlp/eole/pull/150
* **Supporting HF tokenizers** by @francoishernandez in https://github.com/eole-nlp/eole/pull/122
* **Model Validator Recipe** by @francoishernandez in https://github.com/eole-nlp/eole/pull/146
* apply bytefallback at detok (onmt_tokenize with sentencepiece) by @vince62s in https://github.com/eole-nlp/eole/pull/155
* patch eos_token_id list handling by @francoishernandez in https://github.com/eole-nlp/eole/pull/158
* **Compile and Ctranslate2 support** by @vince62s in https://github.com/eole-nlp/eole/pull/161
* Move predict config update from model loading to config validation by @francoishernandez in https://github.com/eole-nlp/eole/pull/163
* EuroLLM Gradio (web based) translator 35 languages to 35 languages by @vince62s in https://github.com/eole-nlp/eole/pull/164

## 0.0.2

* Refactor position encoding configuration by @vince62s in https://github.com/eole-nlp/eole/pull/60
* fix update vocab by @vince62s in https://github.com/eole-nlp/eole/pull/63
* bfloat16 support, and an attempt at homogenizing model_dtype & precision by @francoishernandez in https://github.com/eole-nlp/eole/pull/54
* Fix prefix and suffix transforms - avoid adding empty suffix or prefix by @sersh88 in https://github.com/eole-nlp/eole/pull/57
* fix the incorrect dockerimages in the ReadMe by @aaaallleen in https://github.com/eole-nlp/eole/pull/68
* Remove unnecessary optim in convert_HF by @francoishernandez in https://github.com/eole-nlp/eole/pull/71
* Add onmt_config converter to facilitate switch by @francoishernandez in https://github.com/eole-nlp/eole/pull/69
* Update some FAQ sections by @francoishernandez in https://github.com/eole-nlp/eole/pull/74
* Added TER and BLEU for early stopping  by @aaaallleen in https://github.com/eole-nlp/eole/pull/73
* [fix] fix normalize and clean transforms config management by @francoishernandez in https://github.com/eole-nlp/eole/pull/87
* [docs] Fix quickstart config and command by @francoishernandez in https://github.com/eole-nlp/eole/pull/90
* add head_dim setting when diff from hidden // heads by @vince62s in https://github.com/eole-nlp/eole/pull/78
* Some MHA and RoPE refactoring, llama-3.1 rope_scaling by @francoishernandez in https://github.com/eole-nlp/eole/pull/91
* Fixed variable referenced before assignment when position_embeddings is None error by @dameikle in https://github.com/eole-nlp/eole/pull/95
* Send src_pad_mask and tgt_pad_mask to decoder in _align_forward by @dameikle in https://github.com/eole-nlp/eole/pull/96
* Fixdistrib by @vince62s in https://github.com/eole-nlp/eole/pull/100
* fix added tokens by @vince62s in https://github.com/eole-nlp/eole/pull/101
* Support mapped tokens eg: <im_start> ==> ｟im_start｠in inference.yaml … by @vince62s in https://github.com/eole-nlp/eole/pull/102
* add wmt22 recipes with TowerInstruct and Llama3.1 LLMs by @vince62s in https://github.com/eole-nlp/eole/pull/103
* Remove duplicate sentencepiece requirement by @francoishernandez in https://github.com/eole-nlp/eole/pull/104
* [patch] Adapt some warning behaviours for reduced verbosity by @francoishernandez in https://github.com/eole-nlp/eole/pull/105
* [patch] Update precision to compute_dtype in forgotten places by @francoishernandez in https://github.com/eole-nlp/eole/pull/106
* Inference server, lots of related changes by @francoishernandez in https://github.com/eole-nlp/eole/pull/42

**Full Changelog**: https://github.com/eole-nlp/eole/compare/0.0.1...0.0.2


## 0.0.1
* mlp refact by @vince62s in https://github.com/eole-nlp/eole/pull/1
* fix llama3 and parallel_residual by @vince62s in https://github.com/eole-nlp/eole/pull/4
* fixed mismatch between mask and batch dimensions by @l-k-11235 in https://github.com/eole-nlp/eole/pull/6
* simplify LayerNorm access as a constant by @vince62s in https://github.com/eole-nlp/eole/pull/7
* Fix the checkpoint directory cleaning by @l-k-11235 in https://github.com/eole-nlp/eole/pull/10
* Modify default model config behaviour by @francoishernandez in https://github.com/eole-nlp/eole/pull/8
* rename num_kv remove multiquery by @vince62s in https://github.com/eole-nlp/eole/pull/12
* fix mmlu config by @vince62s in https://github.com/eole-nlp/eole/pull/13
* Fix the tokenizer saving in the HF converter by @l-k-11235 in https://github.com/eole-nlp/eole/pull/14
* remove unsused average attn by @vince62s in https://github.com/eole-nlp/eole/pull/15
* MHA refac: rope without complex operations + query only as input of the forward by @vince62s in https://github.com/eole-nlp/eole/pull/20
* Revert "MHA refac: rope without complex operations + query only as input of the forward" by @vince62s in https://github.com/eole-nlp/eole/pull/22
* missing removal of average attn by @vince62s in https://github.com/eole-nlp/eole/pull/23
* `config.models.BaseModelConfig._override_values` updates everything once by @francoishernandez in https://github.com/eole-nlp/eole/pull/24
* [fix] Patch lora bin to dump json config by @francoishernandez in https://github.com/eole-nlp/eole/pull/28
* review flash/sdpa arg by @vince62s in https://github.com/eole-nlp/eole/pull/25
* fix missing layers names by @vince62s in https://github.com/eole-nlp/eole/pull/30
* Split MHA by @vince62s in https://github.com/eole-nlp/eole/pull/29
* Resize the key_pad_mask by @l-k-11235 in https://github.com/eole-nlp/eole/pull/36
* [patch] upgrade docusaurus deps, fix build script by @francoishernandez in https://github.com/eole-nlp/eole/pull/37
* Add gpt2 converter, hellaswag eval tool, misc fixes by @francoishernandez in https://github.com/eole-nlp/eole/pull/38
* Forgot hellaswag.py tool in #38 by @francoishernandez in https://github.com/eole-nlp/eole/pull/39
* estim lambda scheduler by @vince62s in https://github.com/eole-nlp/eole/pull/40
* Add support for XLM-Roberta-XL (and XXL) conversion by @vince62s in https://github.com/eole-nlp/eole/pull/41
* Some fixes, get rid of data_task, homogenize model_task to model_type by @francoishernandez in https://github.com/eole-nlp/eole/pull/43
* Some improvements to config.json readability by @francoishernandez in https://github.com/eole-nlp/eole/pull/44
* [docs] Github Actions workflow to facilitate docs deployment by @francoishernandez in https://github.com/eole-nlp/eole/pull/47
* [fix] Allow to build_vocab with full train config, patch vocab validation by @francoishernandez in https://github.com/eole-nlp/eole/pull/49
* Enable PyPI release workflow by @francoishernandez in https://github.com/eole-nlp/eole/pull/50
* [fix] Fix paths in wiki_103 recipe, add pyarrow opt requirement by @francoishernandez in https://github.com/eole-nlp/eole/pull/51
* Estim first token instead of average by @vince62s in https://github.com/eole-nlp/eole/pull/46
* Add Recipe to train a cometkiwi-like encoder model (which can be used to score sentence pairs) by @vince62s in https://github.com/eole-nlp/eole/pull/53
* Simplify __init__ files, remove some unused code by @francoishernandez in https://github.com/eole-nlp/eole/pull/52

**Full Changelog**: https://github.com/eole-nlp/eole/commits/0.0.1rc1
