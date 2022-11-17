import argparse

parser = argparse.ArgumentParser(description="select which model to run")

parser.add_argument(
    "-m", "--model", type=str, default="baseline", help="The model to run"
)

parser.add_argument(
    "-t", "--test", default="true", help="Run the model on a small dataset"
)

args = parser.parse_args()

if args.test == "true":
    args.test = True
else:
    args.test = False
