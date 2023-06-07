import argparse
import os
from typing import Any

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from configs.config import Config
from src.constants import PROJECT_PATH
from src.datamodule import AmazonForestDM
from src.training_module import AmazonForestClassifier


def arg_parse() -> Any:  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config) -> None:  # noqa: D103
    datamodule = AmazonForestDM(config.data_config)
    model = AmazonForestClassifier(config)

    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )

    task.connect(config.dict())

    os.makedirs(config.experiment_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.experiment_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_weights_only=True,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        callbacks=[
            checkpoint_callback,
            EarlyStopping(
                monitor=config.monitor_metric,
                patience=4,
                mode=config.monitor_mode,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(
        ckpt_path=checkpoint_callback.best_model_path,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    args = arg_parse()
    seed_everything(1337, workers=True)  # noqa: WPS432
    config_path = os.path.join(PROJECT_PATH, args.config_file)
    config = Config.from_yaml(config_path)
    config.setup_project_path(PROJECT_PATH)
    train(config)
