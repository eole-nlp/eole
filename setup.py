#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_cuda_arch_flags():
    flags = []

    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = major * 10 + minor
            flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
            flags.append(f"-gencode=arch=compute_{arch},code=compute_{arch}")
            return flags
    except (RuntimeError, AssertionError):
        pass

    # Conservative fallback
    return [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_80,code=compute_80",
    ]


cuda_arch_flags = get_cuda_arch_flags()

setup(
    name="eole",
    description="Open language modeling toolkit based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.4.4",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="eole.ops",  # This will be accessible as eole.ops
            sources=[
                "csrc/rms_norm.cu",
                "csrc/rotary_embedding.cu",
                "csrc/activation_kernels.cu",
                "csrc/bindings.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + cuda_arch_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
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
        "numpy>=2.0",
        "pandas",
        "protobuf==3.20.1",
        "pyahocorasick",
        "pyonmttok>=1.38.1,<2",
        "pyyaml",
        "rapidfuzz",
        "rich",
        "sacrebleu",
        "safetensors",
        "sentencepiece>=0.1.94,<0.1.98",
        "six",
        "spacy",
        "subword-nmt>=0.3.7",
        "tensorboard>=2.18.0",
        "torch>=2.8,<2.10",
        "torch-optimi",
        "uvicorn",
        "waitress",
        "pydantic",
    ],
    entry_points={
        "console_scripts": [
            "eole=eole.bin.main:main",
        ],
    },
)
