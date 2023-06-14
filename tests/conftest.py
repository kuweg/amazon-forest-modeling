import pytest
import os
from torch import zeros
from helpers import TESTS_DIR
from configs.config import Config
from src.datamodule import AmazonForestDM
from src.training_module import AmazonForestClassifier


@pytest.fixture(scope='session')
def conf():
    return Config.from_yaml(os.path.join(TESTS_DIR, 'test_conf.yaml'))


@pytest.fixture
def test_model(conf):
    model = AmazonForestClassifier(config=conf)
    return model


@pytest.fixture
def fake_image(conf):
    # [B, C, H, W]
    image = zeros(
        (
            1,
            3,
            conf.data_config.img_height,
            conf.data_config.img_width,
        )
    )
    return image


@pytest.fixture
def datamodule(conf):
    conf.data_config.data_path = os.path.join(
        TESTS_DIR, 'small_data'
    )
    datamodule = AmazonForestDM(config=conf.data_config)
    return datamodule
