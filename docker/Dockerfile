ARG TORCH_VERSION=2.14.1
ARG CUDA_VERSION=13.2.1
FROM ghcr.io/eole-nlp/eole-base:torch${TORCH_VERSION}-ubuntu24.04-cuda${CUDA_VERSION}

ARG EOLE_VERSION=local
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/eole-nlp/eole" \
      org.opencontainers.image.version="${EOLE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"
COPY . /eole
WORKDIR /eole
RUN python -m pip install -r requirements.opt.txt && \
    python -m pip install 'fla-core==0.5.2' 'transformers>=5.20,<6' && \
    python -m pip install -e '.[ui,translation,ocr,rag]' && \
    python -m pip check && eole --help
COPY docker/build-kernels.sh /usr/local/bin/eole-build-kernels
RUN chmod +x /usr/local/bin/eole-build-kernels
ENTRYPOINT ["/bin/bash"]
