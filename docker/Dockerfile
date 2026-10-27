ARG TORCH_VERSION=2.14.1
ARG CUDA_VERSION=13.2.1
FROM ghcr.io/eole-nlp/eole-base:torch${TORCH_VERSION}-ubuntu24.04-cuda${CUDA_VERSION}

ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 10.3 12.0"
ARG MAX_JOBS=2
ARG EOLE_VERSION=local
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/eole-nlp/eole" \
      org.opencontainers.image.version="${EOLE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"
COPY . /eole
WORKDIR /eole
COPY docker/install-vision.sh /usr/local/bin/eole-install-vision
RUN bash /usr/local/bin/eole-install-vision && \
    python -m pip install -r requirements.opt.txt && \
    python -m pip install 'triton==3.8.0' 'fla-core==0.6.0' 'transformers>=5.20,<6' && \
    EOLE_SKIP_CUDA_BUILD=1 python -m pip install -e '.[ui,translation,ocr,rag]' --no-build-isolation && \
    python -m pip check && eole --help
COPY docker/build-kernels.sh /usr/local/bin/eole-build-kernels
RUN chmod +x /usr/local/bin/eole-build-kernels && \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" eole-build-kernels && \
    python -c 'import torch; from eole import _ops; print(_ops.__file__)' && \
    rm -rf build
LABEL io.eole.cuda.architectures="${TORCH_CUDA_ARCH_LIST}"
ENTRYPOINT ["/bin/bash"]
