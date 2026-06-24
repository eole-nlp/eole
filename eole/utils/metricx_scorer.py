METRICX_INPUT_TEMPLATES = {
    "23": {
        "reference": "candidate: {tgt} reference: {ref}",
        "qe": "candidate: {tgt} source: {src}",
    },
    "24": {
        "reference": "source: {src} candidate: {tgt} reference: {ref}",
        "qe": "source: {src} candidate: {tgt}",
    },
}


def metricx_input_templates(version):
    if version not in METRICX_INPUT_TEMPLATES:
        raise ValueError(f"Unsupported MetricX version: {version}")
    return METRICX_INPUT_TEMPLATES[version]
