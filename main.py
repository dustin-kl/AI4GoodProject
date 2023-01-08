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

    datamodule = ClimateNetDataModule(
        files,
        config.features,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    model = get_model(args.model)

    Processor.fit(model, datamodule, max_epochs=1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    wandb.init(project="ClimateNet")
    main()
