from dataclasses import dataclass
from typing import List

from torch import nn

from configs.config import LossConfig
from src.helpers import load_object


@dataclass
class Loss(object):
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: List[LossConfig]) -> List[Loss]:
    """Get a filled `Loss` class.

    Fill `Loss` data class with parameters
    from `losses_cfg`.

    Args:
        losses_cfg (List[LossConfig]): config for losses.

    Returns:
        List[Loss]: `Loss` for every loss in config.
    """
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
        )
        for loss_cfg in losses_cfg
    ]
