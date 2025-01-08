#!/usr/bin/env python
import torch
from eole.bin import BaseBin, register_bin


@register_bin(name="release")
class ReleaseModel(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--model", "-m", help="The model path", required=True)
        parser.add_argument("--output", "-o", help="The output path", required=True)
        parser.add_argument(
            "--format",
            choices=["pytorch", "ctranslate2"],
            default="pytorch",
            help="The format of the released model",
        )
        parser.add_argument(
            "--quantization",
            "-q",
            choices=["int8", "int16", "float16", "int8_float16"],
            default=None,
            help="Quantization type for CT2 model.",
        )

    @classmethod
    def run(cls, args):
        raise NotImplementedError("Model release and CTranslate2 conversion are not yet implemented for Eole models.")
        model = torch.load(args.model, map_location=torch.device("cpu"))
        if args.format == "pytorch":
            model["optim"] = None
            torch.save(model, args.output)
        elif args.format == "ctranslate2":
            import ctranslate2

            converter = ctranslate2.converters.OpenNMTPyConverter(args.model)
            converter.convert(args.output, force=True, quantization=args.quantization)
