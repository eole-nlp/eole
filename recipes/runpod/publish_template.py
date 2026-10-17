"""Register a reusable template; never deploy or rent a Pod."""

import argparse
import json
import os
from pathlib import Path
import re

import requests


def template_spec(image, public=False):
    if (
        not re.fullmatch(r"[A-Za-z0-9./:_@-]+", image)
        or image.endswith(":latest")
        or (":" not in image.rsplit("/", 1)[-1])
    ):
        raise ValueError("Use a published image with an explicit version tag or digest")
    spec = json.loads((Path(__file__).parent / "template.json").read_text())
    spec.update(imageName=image, isPublic=public)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--public", action="store_true", help="Make the template public after validating a private deployment"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without accessing RunPod")
    args = parser.parse_args()
    spec = template_spec(args.image, args.public)
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        parser.error("Set RUNPOD_API_KEY to register the template")
    response = requests.post(
        "https://rest.runpod.io/v1/templates", json=spec, headers={"Authorization": f"Bearer {key}"}, timeout=30
    )
    response.raise_for_status()
    template_id = response.json()["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", template_id):
        raise ValueError("RunPod returned an unexpected template ID")
    url = f"https://console.runpod.io/hub/template/{template_id}"
    print(url)
    if args.public:
        print(f"[Deploy Eole on RunPod]({url})")


if __name__ == "__main__":
    main()
