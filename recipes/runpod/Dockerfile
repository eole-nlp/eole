ARG EOLE_IMAGE
FROM ${EOLE_IMAGE}
# Refuse older parent images that defer extension compilation to deployment.
# Import on a GPU-less builder checks the Python/PyTorch ABI, not GPU execution.
RUN python -c 'import torch; from eole import _ops; print(_ops.__file__)'
# Readiness must use the matching server implementation, not just new launcher YAML.
RUN python -c 'from eole.utils.verifier_cache import CachedVerifier; from eole.server.model import Model; assert hasattr(Model, "warmup"), "Build the full Eole image from this source before building RunPod"'
# Older compatible parents may lack the processor dependency used by RAG.
COPY docker/install-vision.sh /usr/local/bin/eole-install-vision
RUN bash /usr/local/bin/eole-install-vision
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/eole-nlp/eole" \
      org.opencontainers.image.revision="${VCS_REF}"
COPY recipes/runpod /opt/eole-runpod
COPY recipes/ui/serve-showcase.yaml /eole/recipes/ui/serve-showcase.yaml
COPY docker/build-kernels.sh /usr/local/bin/eole-build-kernels
ENV PYTHONUNBUFFERED=1 EOLE_DEPLOY_DIR=/workspace/eole
EXPOSE 5000
ENTRYPOINT ["python", "/opt/eole-runpod/launch.py"]
