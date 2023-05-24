import os
from clearml import Task

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything

from configs.config import Config
from src.constants import PROJECT_PATH
from src.training_module import AmazonForestClassifier
from src.datamodule import AmazonForestDM


def train(config: Config) -> None:
    datamodule = AmazonForestDM(config.data_config)
    model = AmazonForestClassifier(config)

    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True
    )

    task.connect(config.dict())
    
    os.makedirs(experiment_path, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.experiment_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}} - {{{config.monitor_metric}:.3f}}'
    )
    
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch')
        ]
    )
    
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)
    
    
if __name__ == '__main__':
    seed_everything(1337, workers=True)
    config_path = os.path.join(PROJECT_PATH, 'configs/config.yaml')
    config = Config.from_yaml(config_path)
    
    datapath = os.path.join(PROJECT_PATH, 'data/')
    experiment_path = os.path.join(PROJECT_PATH, config.experiment_path)
    config.data_config.data_path = datapath
    config.experiment_path = experiment_path
    train(config)
     