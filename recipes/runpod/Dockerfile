ARG EOLE_IMAGE=ghcr.io/eole-nlp/eole:0.6.3-torch2.14.1-ubuntu24.04-cuda13.2.1
FROM ${EOLE_IMAGE}
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/eole-nlp/eole" \
      org.opencontainers.image.revision="${VCS_REF}"
COPY recipes/runpod /opt/eole-runpod
COPY docker/build-kernels.sh /usr/local/bin/eole-build-kernels
ENV PYTHONUNBUFFERED=1 EOLE_DEPLOY_DIR=/workspace/eole
EXPOSE 5000
ENTRYPOINT ["python", "/opt/eole-runpod/launch.py"]
