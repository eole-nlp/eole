# CometKiwi

---
**NOTE**

This is NOT the exact replication of the Unbabel Comet Kiwi.

What is common:
We use the same base model xlm-roberta-xl or xlm-roberta-xxl encoders
We use the same dataset (1720-da.mlqe)

You can get the dataset to train the model below here: https://huggingface.co/eole-nlp/cometkiwi-xxl-eole/

What is different:
wmt23-cometkiwi-da-xl(or xxl) use Layerwise attention which brings complexity without significant better accuracy
we use Gelu instead of Tanh in the Estimator

Our scores for XL and XXL are in the same range when they are very different for Unbabel/wmt23-cometkiwi-XL or XXL


To make your life easier, run these commands from the recipe directory (here `recipes/cometkiwi`).
---

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
```

### Download and convert the base model

```
eole convert HF --model_dir facebook/xlm-roberta-xxl --output $EOLE_MODEL_DIR/xlm-roberta-xxl-eole
```
**NOTE**
The facebook original model is stored in FP32 but we convert it to FP16 at conversion.

XXL is a 10.7G params model hence will save a 21.4GB file on disk (safetensors format)
XL is a 3.5GB params model hence will save a 7.0GB file on disk

After conversion in `$EOLE_MODEL_DIR` you will get the following files:
config.json
model.00.safetensors
sentencepiece.bpe.model
vocab.json
vocab.txt

The vocab.txt file contains 250000 entries (from sentencepiece) but the model was trained with extra tokens
You need to concat the `added_vocab.txt` file to the `vocab.txt` file resulting in 250880 tokens


## Training cometkiwi

Training will happen in two steps (see cometkiwi-xxl-eole.yaml file)
FYI the trained model can be downloaded here: https://huggingface.co/eole-nlp/cometkiwi-xxl-eole/

STEP 1:
We train from the converted xlm-roberta-xxl model but we keep everything frozen.
We just add an Estimator layer that will be trained during 4000 steps
To make sure we do not lose anything we will rename the pre-trained subfolder into `step_4000_pretrain`
In this step we do NOT use LoRA but we use 4bit quant to make things easier and fit in a smaller cards

STEP 2:
We release the encoder to make weights trainable hence we need to use LoRA since the model is big and gradients would not fit in consumer grade cards.
We train during 40000 steps

For the two steps above, the training command is the same but make sure the yaml file is modified according to the instructions
`eole train --config cometkiwi-xxl-eole.yaml`

After this step we need to merge the LoRA weights into the original model with the following command:

`eole model lora --base_model "./cometkiwi-xxl-eole/step_4000_pretrain/" --lora_weights "./cometkiwi-xxl-eole/" --output "./cometkiwi-xxl-eole/merged"`


## Cometkiwi Inference

Format the source / target file you want to score in the Comet format:

`./combine.sh newstest2014.en newstest2014.de newstest2014.comb`

Score the .comb file:

`eole predict --config cometkiwi-xxl-inference.yaml --src newstest2014.comb --output newstest2014.scores --with_score`

For now the scores are in the third column, so you can `cut -f3` the output file

