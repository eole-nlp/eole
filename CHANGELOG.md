# Changelog

This is just a centralised version of the Github automatically generated Release changelogs.

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