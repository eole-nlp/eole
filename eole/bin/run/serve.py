#!/usr/bin/env python
"""Command entry point for the Eole inference server."""

import uvicorn

from eole.bin import register_bin, BaseBin
from eole.server.app import create_app


@register_bin(name="serve")
class Serve(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--config",
            "-config",
            "-c",
            default="./server_conf.yaml",
            help="Path of server YAML config file.",
        )
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default="5000")

    @classmethod
    def run(cls, args):
        app = create_app(args.config)
        uvicorn.run(app=app, host=args.host, port=args.port, log_level="info")
