import argparse

parser = argparse.ArgumentParser(description="select which model to run")

parser.add_argument(
    "-m", "--model", type=str, default="model", help="select which model to run"
)

parser.add_argument(
    "-t", "--test", type=bool, default=False, help="select whether to run in test mode"
)

args = vars(parser.parse_args())