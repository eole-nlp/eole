# Whisper

> **NOTE:** To make your life easier, run these commands from the recipe directory (here `recipes/whisper`).

## Supported models

Any HuggingFace Whisper checkpoint can be converted:

| Model | HuggingFace ID | Languages |
|-------|----------------|-----------|
| tiny | `openai/whisper-tiny` | Multilingual |
| tiny.en | `openai/whisper-tiny.en` | English only |
| base | `openai/whisper-base` | Multilingual |
| base.en | `openai/whisper-base.en` | English only |
| small | `openai/whisper-small` | Multilingual |
| small.en | `openai/whisper-small.en` | English only |
| medium | `openai/whisper-medium` | Multilingual |
| medium.en | `openai/whisper-medium.en` | English only |
| large-v3 | `openai/whisper-large-v3` | Multilingual |
| large-v3-turbo | `openai/whisper-large-v3-turbo` | Multilingual |

## Quick start

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
```

### Download and convert model

```
eole convert HF --model_dir openai/whisper-base.en --output ${EOLE_MODEL_DIR}/whisper-base-en-eole
```

### Download sample audio files (optional)

```
bash download_samples.sh
```

This downloads a few public-domain audio clips and converts them to 16kHz mono WAV.
Requires `wget` and `ffmpeg`.

### Run inference

The included `audio_files.txt` lists the English sample files. After downloading, run:

```
eole predict -config whisper_predict.yaml -model_path ${EOLE_MODEL_DIR}/whisper-base-en-eole -src ./audio_files.txt -output ./transcription.txt
```

Each line in the output corresponds to the audio file on the same line in the input.

## Output modes

### Plain text (default)

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -output ./text.txt
```

### Segment timestamps

Outputs JSON with start/end times for each segment:

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -output ./segments.json -timestamps segment
```

### Word timestamps

Outputs JSON with per-word timing via cross-attention DTW alignment:

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -output ./words.json -timestamps word
```

Note: word-level timestamps require a model with `alignment_heads` in its `generation_config.json` (e.g. `whisper-base.en`, `whisper-small`, `whisper-large-v3`).

## Language and task

Multilingual Whisper models support specifying the source language and task.

### Language hint

Force a specific source language (useful when auto-detection is unreliable):

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -language fr
```

### Translation (to English)

Translation requires a multilingual model (not `.en` variants). First convert one:

```
eole convert HF --model_dir openai/whisper-small --output ${EOLE_MODEL_DIR}/whisper-small-eole
```

Then run with `task: translate` on a French audio sample (included in `download_samples.sh`):

```
echo "samples/fr0.wav" > ./french_audio.txt
eole predict -config whisper_predict.yaml -model_path ${EOLE_MODEL_DIR}/whisper-small-eole -src ./french_audio.txt -output ./translation.txt -language fr -task translate
```

Or with segment timestamps:

```
eole predict -config whisper_predict_translate.yaml -model_path ${EOLE_MODEL_DIR}/whisper-small-eole -src ./french_audio.txt -output ./translation.json
```

## Prompt conditioning

Use `initial_prompt` to condition the decoder output style and vocabulary.
The prompt text is prepended as previous context using the `<|startofprev|>` mechanism:

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -initial_prompt "This is a presidential radio address."
```

This can help with:
- Spelling of proper nouns and domain-specific terms
- Output formatting and punctuation style
- Steering the model toward a particular topic or register

## Config reference

| Field | Type | Default  | Description                                                     |
|-------|------|----------|-----------------------------------------------------------------|
| `model_path` | str  | required | Path to converted eole model                                    |
| `src` | str  | required | File listing audio paths (one per line)                         |
| `output` | str  | required | Output file path                                                |
| `beam_size` | int  | 5        | Beam search width                                               |
| `max_length` | int  | 250      | Maximum output tokens                                           |
| `batch_size` | int  | 1        | Batch size (use 1 for audio)                                    |
| `gpu_ranks` | list | []       | GPU device IDs                                                  |
| `timestamps` | str  | "none"   | Output mode: "none", "segment", "word"                          |
| `language` | str  | null     | Source language code (e.g. "en", "fr", "zh")                    |
| `task` | str  | null     | "transcribe" or "translate"                                     |
| `initial_prompt` | str  | null     | Text prompt for decoder conditioning                            |
| `condition_on_previous_text` | bool | false    | Condition next segment transcription on previous segment's text |
