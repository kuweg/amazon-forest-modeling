import os
import shutil
import pytorch_lightning as pl
from src.datamodule import AmazonForestDM, _split_and_save_datasets
from tests.helpers import cleanup_after, DATAFILES, TESTS_DIR


class TestDM(AmazonForestDM):

    def prepare_data(self):
        _split_and_save_datasets(self._data_path, self._train_size)


@cleanup_after(
    files_list=DATAFILES,
    files_path=os.path.join(TESTS_DIR, 'small_data')
)
def test_training_pipeline(test_model, conf):
    # Все предыдущие тесты подтирают за собой файлы
    # Нужен большой df_encoded.csv, так как там все метки классов

    conf.data_config.n_workers = 1
    conf.data_config.data_path = os.path.join(
        TESTS_DIR, 'small_data'
    )

    shutil.copy(
        os.path.join(TESTS_DIR, 'df_encoded.csv'),
        os.path.join(conf.data_config.data_path, 'df_encoded.csv'),
    )
    pl.seed_everything(1337)

    dm = TestDM(config=conf.data_config)

    trainer = pl.Trainer(
        max_epochs=conf.n_epochs,
        accelerator=conf.accelerator,
        devices=[conf.device],
        fast_dev_run=True
    )

    trainer.fit(
        test_model,
        dm,
    )
