from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import List


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    n_workers: int
    data_path: str
    batch_size: int
    train_size: float
    img_width: int
    img_height: int


class FakeConfig(BaseModel):
    data_config: DataConfig
    num_classes: int
    lr: float
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]
    n_epochs: int
    accelerator: str
    device: int

    @classmethod
    def from_yaml(cls, path: str) -> 'FakeConfig':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
