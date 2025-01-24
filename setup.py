#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eole",
    description="Open language modeling toolkit based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=find_packages(),
    project_urls={
        "Source": "https://github.com/eole-nlp/eole/",
    },
    python_requires=">=3.10",
    install_requires=[
        "configargparse",
        "ctranslate2>=4,<5",
        "fastapi",
        "fasttext-wheel",
        "huggingface_hub",
        "datasets",
        "numpy<2.0",
        "pandas",
        "protobuf==3.20.1",
        "pyahocorasick",
        "pyonmttok>=1.37,<2",
        "pyyaml",
        "rapidfuzz",
        "rich",
        "sacrebleu",
        "safetensors",
        "sentencepiece>=0.1.94,<0.1.98",
        "six",
        "spacy",
        "subword-nmt>=0.3.7",
        "tensorboard>=2.3",
        "torch>=2.5,<2.6",
        "uvicorn",
        "waitress",
    ],
    entry_points={
        "console_scripts": [
            "eole=eole.bin.main:main",
        ],
    },
)
