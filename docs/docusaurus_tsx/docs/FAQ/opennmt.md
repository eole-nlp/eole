# How to switch from OpenNMT-py to EOLE?

## Configuration conversion

One of the main pillars of EOLE is the full revamping of the configuration structure and validation logic. That means OpenNMT-py configuration files are not supported by default.
That being said, a conversion tool has been created to facilitate the transition: [`eole convert onmt_config`](https://github.com/eole-nlp/eole/blob/master/eole/bin/convert/convert_onmt_config.py)

There are a few key things to know:
- what was previous fully "flat" in OpenNMT-py configurations is now mostly nested in nested sections with specific scope such as `training`, `model`, `transforms_configs`;
- some parameters were renamed, removed, or replaced by other logics, which makes the conversion script not 100% exhaustive;
- the conversion script will log the remaining "unmapped settings", to facilitate fixing the last issues manually.

## Model conversion

Models trained with OpenNMT-py can technically be converted to be used with EOLE, but there is no automated tool for now. Feel free to get in touch via [Issues](https://github.com/eole-nlp/eole/issues) or [Discussions](https://github.com/eole-nlp/eole/discussions) if that is a blocker.