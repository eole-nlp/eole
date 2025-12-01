# HunyuanOCR

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

## Convert the model

```
eole convert HF --model_dir tencent/HunyuanOCR --output $EOLE_MODEL_DIR/HunyuanOCR --token $HF_TOKEN
```

## Run the test script

```
python3 test_inference.py
```

This script shows the difference between an English and a Chinese prompt for the same task.
Chinese gives better results.

```
Figure 4 | To test model performance under different compression ratios (requiring different numbers of vision tokens) and enhance the practicality of DeepSeek-OCR, we configure it with 
multiple resolution modes.

<table><caption>Table 1 | Multi resolution support of DeepEncoder. For both research and application purposes, we design DeepEncoder with diverse native resolution and dynamic resolution 
modes.</caption><tr><td></td><td colspan="4">Native Resolution</td><td colspan="2">Dynamic 
Resolution</td></tr><tr><td>Mode</td><td>Tiny</td><td>Small</td><td>Base</td><td>Large</td><td>Gundam</td><td>Gundam-M</td></tr><tr><td>Resolution</td><td>512</td><td>640</td><td>1024</td><t
d>1280</td><td>640+1024</td><td>1024+1280</td></tr><tr><td>Tokens</td><td>64</td><td>100</td><td>256</td><td>400</td><td>n×100+256</td><td>n×256+400</td></tr><tr><td>Process</td><td>resize</
td><td>resize</td><td>padding</td><td>padding</td><td>resize + padding</td><td>resize + padding</td></tr></table>

## 3.2.2. Multiple resolution support

Suppose we have an image with 1000 optical characters and we want to test how many vision tokens are needed for decoding. This requires the model to support a variable number of vision 
tokens. That is to say the DeepEncoder needs to support multiple resolutions.

We meet the requirement aforementioned through dynamic interpolation of positional encodings, and design several resolution modes for simultaneous model training to achieve the capability of
a single DeepSeek-OCR model supporting multiple resolutions. As shown in Figure 4, DeepEncoder mainly supports two major input modes: native resolution and dynamic resolution. Each of them 
contains multiple sub-modes.

Native resolution supports four sub-modes: Tiny, Small, Base, and Large, with corresponding resolutions and token counts of 512×512 (64), 640×640 (100), 1024×1024 (256), and 1280×1280 (400) 
respectively. Since Tiny and Small modes have relatively small resolutions, to avoid wasting vision tokens, images are processed by directly resizing the original shape. For Base and Large 
modes, in order to preserve the original image aspect ratio, images are padded to the corresponding size. After padding, the number of valid vision tokens is less than the actual number of 
vision tokens, with the calculation formula being:

$$ N_{valid}=\lceil N_{actual}\times[1-((max(w,h)-min(w,h))/(max(w,h)))] \rceil $$

where w and h represent the width and height of the original input image.
```

```
<pFig>Figure 4 | To test model performance under different compression ratios (requiring different numbers of vision tokens) and enhance the practicality of DeepSeek-OCR, we configure it 
with multiple resolution modes.</pFig><quad>(108,56),(875,240)</quad>

the 4096 tokens go through the compression module and the token becomes 4096/16=256, thus making the overall activation memory controllable.

<table><caption>Table 1 | Multi resolution support of DeepEncoder. For both research and application purposes, we design DeepEncoder with diverse native resolution and dynamic resolution 
modes.</caption><tr><td></td><td colspan="4">Native Resolution</td><td colspan="2">Dynamic 
Resolution</td></tr><tr><td>Mode</td><td>Tiny</td><td>Small</td><td>Base</td><td>Large</td><td>Gundam</td><td>Gundam-M</td></tr><tr><td>Resolution</td><td>512</td><td>640</td><td>1024</td><t
d>1280</td><td>640+1024</td><td>1024+1280</td></tr><tr><td>Tokens</td><td>64</td><td>100</td><td>256</td><td>400</td><td>n×100+256</td><td>n×256+400</td></tr><tr><td>Process</td><td>resize</
td><td>resize</td><td>padding</td><td>padding</td><td>resize + padding</td><td>resize + padding</td></tr></table>

## 3.2.2. Multiple resolution support

Suppose we have an image with 1000 optical characters and we want to test how many vision tokens are needed for decoding. This requires the model to support a variable number of vision 
tokens. That is to say the DeepEncoder needs to support multiple resolutions.

We meet the requirement aforementioned through dynamic interpolation of positional encodings, and design several resolution modes for simultaneous model training to achieve the capability of
a single DeepSeek-OCR model supporting multiple resolutions. As shown in Figure 4, DeepEncoder mainly supports two major input modes: native resolution and dynamic resolution. Each of them 
contains multiple sub-modes.

Native resolution supports four sub-modes: Tiny, Small, Base, and Large, with corresponding resolutions and token counts of 512×512 (64), 640×640 (100), 1024×1024 (256), and 1280×1280 (400) 
respectively. Since Tiny and Small modes have relatively small resolutions, to avoid wasting vision tokens, images are processed by directly resizing the original shape. For Base and Large 
modes, in order to preserve the original image aspect ratio, images are padded to the corresponding size. After padding, the number of valid vision tokens is less than the actual number of 
vision tokens, with the calculation formula being:

$$ N_{valid}=\lceil N_{actual}\times[1-((max(w,h)-min(w,h))/(max(w,h)))] \rceil $$

where w and h represent the width and height of the original input image.
```
