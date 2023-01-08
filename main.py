import os

import wandb

from src.data_module import ClimateNetDataModule
from src.models import get_model
from src.processor import Processor
from src.utils import Config, Generic, Parser


def main():
    config = Config.init_config()
    args = Parser.parse_args()

    files = Generic.list_files(config.data_dir)
    if args.small_dataset:
        files = files[:10]

    if args.stage == "fit":
        Processor.fit(files, config, args, max_epochs=50)
    elif args.stage == "cv":
        Processor.cv(files, config, args, max_epochs=50)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
