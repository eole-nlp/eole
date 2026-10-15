#!/usr/bin/env python
"""
setup.py for eole.

csrc/ is fully self-contained.

"""
from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

EOLE_CSRC = path.join(this_directory, "csrc")


def get_ext_modules_and_cmdclass():
    try:
        import torch
    except ImportError:
        return [], {}

    if not torch.cuda.is_available():
        return [], {}

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # ── CUDA arch flags ───────────────────────────────────────────────────────
    # Device indices respect CUDA_VISIBLE_DEVICES. Deduplicate architectures
    # so identical GPUs do not produce redundant compilation targets.
    capabilities = set()
    for device in range(torch.cuda.device_count()):
        try:
            capabilities.add(torch.cuda.get_device_capability(device))
        except Exception as error:
            raise RuntimeError(f"Cannot determine CUDA architecture for visible GPU {device}") from error
    flags = []
    for major, minor in sorted(capabilities):
        arch = major * 10 + minor
        flags.extend(
            [
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
                f"-gencode=arch=compute_{arch},code=compute_{arch}",
            ]
        )

    # ── Core sources (always compiled) ───────────────────────────────────────
    core_sources = [
        "csrc/rms_norm.cu",
        "csrc/rotary_embedding.cu",
        "csrc/activation_kernels.cu",
        "csrc/quantization/marlin/moe_align.cu",
        "csrc/quantization/marlin/marlin_repack.cu",
        "csrc/quantization/marlin/marlin_dense.cu",
        "csrc/quantization/marlin/marlin_moe_wna16.cu",
        "csrc/bindings.cpp",
    ]

    include_dirs = [EOLE_CSRC]

    # BuildExtension selects the C++ standard required by the installed PyTorch.
    cxx_args = ["-O3"]
    nvcc_args = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-maxrregcount=64",
    ] + flags

    ext_modules = [
        CUDAExtension(
            name="eole._ops",
            sources=core_sources,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = get_ext_modules_and_cmdclass()

setup(
    name="eole",
    description="Open language modeling toolkit based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.6.2",
    packages=find_packages(),
    package_data={"eole.server": ["web/*"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    project_urls={"Source": "https://github.com/eole-nlp/eole/"},
    python_requires=">=3.11",
    install_requires=[
        "configargparse",
        "ctranslate2>=4,<5",
        "fastapi",
        "fasttext-wheel",
        "huggingface_hub",
        "datasets",
        "numpy>=2.0",
        "pandas",
        "protobuf==7.36.2",
        "pyahocorasick",
        "pyonmttok>=1.38.1,<2",
        "pyyaml",
        "rapidfuzz",
        "rich",
        "sacrebleu",
        "safetensors",
        "sentencepiece>=0.1.94,<=0.2.2",
        "six",
        "spacy",
        "subword-nmt>=0.3.7",
        "tensorboard>=2.18.0",
        "tokenizers",
        "torch>=2.10,<2.15",
        "torchcodec",
        "torch-optimi",
        "uvicorn",
        "waitress",
        "pydantic",
    ],
    extras_require={
        "ui": ["nvidia-ml-py>=12.0.0", "psutil>=5.9.0"],
        "wer": ["jiwer>=3.0", "whisper-normalizer>=0.1"],
        "trackio": [
            "trackio>=0.38.1,<0.41.0",
            "nvidia-ml-py>=12.0.0; sys_platform != 'darwin'",
            "psutil>=5.9.0",
        ],
    },
    entry_points={"console_scripts": ["eole=eole.bin.main:main"]},
)
