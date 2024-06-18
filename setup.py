#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="EOLE",
    description="Experiment with various kinds of language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    packages=find_packages(),
    project_urls={
        "Source": "https://github.com/eole-nlp/eole/",
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3,<2.4",
        "configargparse",
        "ctranslate2>=4,<5",
        "tensorboard>=2.3",
        "flask",
        "waitress",
        "pyonmttok>=1.37,<2",
        "pyyaml",
        "sacrebleu",
        "rapidfuzz",
        "pyahocorasick",
        "fasttext-wheel",
        "spacy",
        "six",
        "safetensors",
        "pandas",
        "sentencepiece",
        "huggingface_hub",
        "sentencepiece>=0.1.94,<0.1.98",
        "subword-nmt>=0.3.7",
        "numpy<2.0"
    ],
    entry_points={
        "console_scripts": [
            "eole=eole.bin.main:main",
        ],
    },
)
