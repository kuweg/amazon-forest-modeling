import pytorch_lightning as pl


def test_training_pipeline(test_model, conf, datamodule):

    _ = pl.Trainer(
        max_epochs=conf.n_epochs,
        accelerator=conf.accelerator,
        devices=[conf.device],
        fast_dev_run=True
    )
