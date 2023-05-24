import torch
from pytorch_lightning import LightningModule
from timm import create_model

from configs.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.helpers import load_object


class AmazonForestClassifier(LightningModule):  # noqa: D101
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self._model = create_model(
            num_classes=self._config.num_classes,
            **self._config.model_kwargs,
        )
        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(
            num_classes=self._config.num_classes,
            num_labels=self._config.num_classes,
            task='multilabel',
            average='macro',
            threshold=0.5,
        )
        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters(self._config.dict())

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self._model(input_x)

    def configure_optimizers(self):  # noqa: D102
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            lr=self._config.lr,
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(
            optimizer, **self._config.scheduler_kwargs,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):  # noqa: D102
        images, gt_labels = batch
        pr_logits = self(images)
        return self._calculate_loss(pr_logits, gt_labels, 'train_')

    def validation_step(self, batch, batch_idx):  # noqa: D102
        images, gt_labels = batch
        pr_logits = self(images)
        self._calculate_loss(pr_logits, gt_labels, 'val_')
        pr_labels = torch.sigmoid(pr_logits)
        self._valid_metrics(pr_labels, gt_labels)

    def test_step(self, batch, batch_idx):  # noqa: D102
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, gt_labels)

    def on_validation_epoch_start(self) -> None:  # noqa: D102
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:  # noqa: D102
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            self.log(
                '{prefix}{loss_name}_loss'.format(
                    prefix=prefix, loss_name=cur_loss.name,
                ),
                loss.item(),
            )
        self.log(
            '{prefix}total_loss'.format(prefix=prefix),
            total_loss.item(),
        )
        return total_loss
