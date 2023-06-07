import os
from torch.utils.data import DataLoader
from src.datamodule import AmazonForestDM
from tests.helpers import cleanup_after, DATAFILES, TESTS_DIR


@cleanup_after(
    files_list=DATAFILES,
    files_path=os.path.join(TESTS_DIR, 'small_data')
)
def test_files_creating_while_processing(datamodule):
    datamodule.prepare_data()
    files_list = os.listdir(
        os.path.join(TESTS_DIR, 'small_data')
    )
    diff = set(DATAFILES) ^ set(files_list)
    assert diff == {'train_classes.csv'}


@cleanup_after(
    files_list=DATAFILES,
    files_path=os.path.join(TESTS_DIR, 'small_data')
)
def test_creating_train_dataloader(datamodule):
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    train_dataloader = datamodule.train_dataloader()
    assert isinstance(train_dataloader, DataLoader)


@cleanup_after(
    files_list=DATAFILES,
    files_path=os.path.join(TESTS_DIR, 'small_data')
)
def test_creating_test_dataloader(datamodule):
    datamodule.prepare_data()
    datamodule.setup(stage='test')
    test_dataloader = datamodule.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)


@cleanup_after(
    files_list=DATAFILES,
    files_path=os.path.join(TESTS_DIR, 'small_data')
)
def test_creating_val_dataloader(datamodule):
    datamodule.prepare_data()
    datamodule.setup(stage='val')
    val_dataloader = datamodule.val_dataloader()
    assert isinstance(val_dataloader, DataLoader)


@cleanup_after(
    files_list=DATAFILES,
    files_path=os.path.join(TESTS_DIR, 'small_data')
)
def test_batch_size_8(conf):
    expected_batch_size = 8
    conf.data_config.batch_size = expected_batch_size
    datamodule = AmazonForestDM(config=conf.data_config)
    datamodule.prepare_data()
    datamodule.setup(stage='test')
    test_data = datamodule.test_dataloader()

    assert expected_batch_size == test_data.batch_size


@cleanup_after(
    files_list=DATAFILES,
    files_path=os.path.join(TESTS_DIR, 'small_data')
)
def test_batch_size_32(conf):
    expected_batch_size = 32
    conf.data_config.batch_size = expected_batch_size
    datamodule = AmazonForestDM(config=conf.data_config)
    datamodule.prepare_data()
    datamodule.setup(stage='test')
    test_data = datamodule.test_dataloader()

    assert expected_batch_size == test_data.batch_size
