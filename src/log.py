from argparse import ArgumentParser
from logging import basicConfig


def arg_to_log_level() -> None:
    """Get the log level from the command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="loglevel",
        const="DEBUG",
        default="WARNING",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="loglevel",
        const="INFO",
        default="WARNING",
    )
    args = parser.parse_args()
    basicConfig(
        level=args.loglevel,
        format="[{levelname:.4}] {name}: {message}",
        style="{",
    )
