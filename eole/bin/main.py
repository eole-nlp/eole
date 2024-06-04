# Main Entrypoint

from argparse import ArgumentParser
from eole.bin import AVAILABLE_BINS


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="Executable to run.", dest="bin")
    for bin_type, bins in AVAILABLE_BINS.items():
        # specific case for "run" which should be done at root level
        if bin_type == "run":
            for bin_name, bin_cls in bins.items():
                subparser = subparsers.add_parser(bin_name)
                bin_cls.add_args(subparser)

        else:
            subparser = subparsers.add_parser(bin_type)
            sub_subparsers = subparser.add_subparsers(dest="sub_bin")
            for bin_name, bin_cls in bins.items():
                sub_subparser = sub_subparsers.add_parser(
                    bin_name
                )  # define some helpstring in the tool class
                bin_cls.add_args(sub_subparser)

    args = parser.parse_args()

    if (
        args.bin in AVAILABLE_BINS.keys()
        and args.sub_bin in AVAILABLE_BINS[args.bin].keys()
    ):
        bin_cls = AVAILABLE_BINS[args.bin][args.sub_bin]
    elif args.bin in AVAILABLE_BINS["run"].keys():
        bin_cls = AVAILABLE_BINS["run"][args.bin]
    else:
        print(parser.format_help())
        exit(1)

    bin_cls.run(args)


if __name__ == "__main__":
    main()
