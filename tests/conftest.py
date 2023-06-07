
import numpy as np
import pytest
import os
from fake_conf import FakeConfig
from helpers import TESTS_DIR
from src.augmentations import get_valid_transforms
from src.datamodule import AmazonForestDM
from src.training_module import AmazonForestClassifier


@pytest.fixture(scope='session')
def conf():
    return FakeConfig.from_yaml(os.path.join(TESTS_DIR, 'test_conf.yaml'))


@pytest.fixture
def test_model(conf):
    model = AmazonForestClassifier(config=conf)
    return model


@pytest.fixture
def fake_image(conf):
    image = np.random.rand(
        conf.data_config.img_height,
        conf.data_config.img_width,
        3,
    )
    preprocess = get_valid_transforms(
        img_width=conf.data_config.img_width,
        img_height=conf.data_config.img_height,
    )
    return preprocess(image=image)['image'][None].to('cpu')


@pytest.fixture
def datamodule(conf):
    conf.data_config.data_path = os.path.join(
        TESTS_DIR, 'small_data'
    )
    datamodule = AmazonForestDM(config=conf.data_config)
    return datamodule
