ARG TORCH_VERSION=2.10.0
ARG CUDA_VERSION=12.8.0

FROM ghcr.io/eole-nlp/eole-base:torch$TORCH_VERSION-ubuntu24.04-cuda$CUDA_VERSION

COPY . /eole
WORKDIR /eole
RUN pip install -r requirements.opt.txt
RUN pip install -e .

WORKDIR /

ENTRYPOINT /bin/bash
