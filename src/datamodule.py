import os
from typing import List, Optional, Tuple

import pandas as pd
from clearml.logger import Logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from configs.config import DataConfig
from src.augmentations import get_train_transforms, get_valid_transforms
from src.data_splitter import stratify_shuffle_split_subsets
from src.dataset import AmazonForestDataset


class AmazonForestDM(LightningDataModule):

    def __init__(self, config: DataConfig, logger: Logger = None):
        super().__init__()
        self._batch_size = config.batch_size
        self._n_workers = config.n_workers
        self._train_size = config.train_size
        self._data_path = config.data_path
        self._train_transforms = get_train_transforms(
            config.img_width, config.img_height,
        )
        self._valid_transforms = get_valid_transforms(
            config.img_width, config.img_height,
        )
        self._image_folder = os.path.join(config.data_path, 'train-jpg')

        self.logger = logger
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        _encode_labels(self._data_path)
        _split_and_save_datasets(self._data_path, self._train_size)

    def setup(self, stage: Optional[str] = None):
        train_df, valid_df, test_df = _read_datasets(self._data_path)

        if stage == 'fit':
            self.train_dataset = AmazonForestDataset(
                train_df,
                image_folder=self._image_folder,
                transforms=self._train_transforms,
            )
            self.valid_dataset = AmazonForestDataset(
                valid_df,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

        elif stage == 'test':
            self.test_dataset = AmazonForestDataset(
                test_df,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def _encode_labels(data_path: str, logger: Logger = None) -> None:
    """Encode labels in `AmazonForest` dataset.

    One-hot encoding for labels. Transforms column `tags` into
    n columns with `1.` or `0.` for corresponding tag,
    where n is number of unique tags

    Args:
        data_path (str): path to AmazonForest .csv file
        logger (Logger): ClearML Logger for logging
    """
    df = pd.read_csv(os.path.join(data_path, 'train_classes.csv'))
    df_encoded = df.copy()

    if logger:
        logger.report_text(
            'Initial dataset: {len_df} samples, {len_col} columns'.
            format(
                len_df=len(df_encoded),
                len_col=len(df_encoded.columns),
            ),
        )

    tags = df_encoded['tags'].apply(lambda tags: tags.split())
    tags_flatten = [tag for tag_sublist in tags for tag in tag_sublist]
    labels = list(set(tags_flatten))
    df_encoded['tags'] = tags

    for tag in labels:
        df_encoded[tag] = _encode_tag(df_encoded['tags'], tag)

    df_encoded = df_encoded.drop(['tags'], axis=1)
    if logger:
        logger.report_text(
            'Encoded dataset: {len_df} samples, {len_col} columns'.
            format(
                len_df=len(df_encoded),
                len_col=len(df_encoded.columns),
            ),
        )

    df_encoded.to_csv(os.path.join(data_path, 'df_encoded.csv'), index=False)


def _encode_tag(column_vals: pd.Series, tag: str) -> List[float]:
    """Encode dataframe's column with tag existance.

    Check that tag exist in every line of column for
    having provided tag.

    Args:
        column_vals (pd.Series):
            DataFrame columns with list of tags.
        tag (str):
            Tag for encoding.

    Returns:
        List[float]:
        Zero-One encoded list for tag columns.
    """
    tag_encoding = []
    for tag_list in column_vals:
        if tag in tag_list:
            tag_encoding.append(1.)
        else:
            tag_encoding.append(0.)
    return tag_encoding


def _split_and_save_datasets(
    data_path: str,
    train_fraction: float = 0.8,
    logger: Logger = None,
) -> None:
    """Split `AmazonForest` datasets.

    Splits unprocessed `AmazonForest` dataset
    into 3 train/val/test subsets.

    Args:
        data_path (str):
            path to dataset file.
        train_fraction (float):
            Ratio for train size. Defaults to 0.8.
        logger (Logger):
            ClearMl Logger object for logging messages.
    """
    print(os.path.join(data_path, 'df_encoded.csv'))
    df = pd.read_csv(os.path.join(data_path, 'df_encoded.csv'))
    if logger:
        logger.report_text('Initial dataset: {len_df}'.format(len_df=len(df)))

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(
        df,
        img_path_column='image_name',
        train_fraction=train_fraction,
    )
    train_df.to_csv(os.path.join(data_path, 'train_df.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'valid_df.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'test_df.csv'), index=False)

    if logger:
        logger.report_text(
            'Train dataset: {len_df}'.format(len_df=len(train_df)),
        )
        logger.report_text(
            'Valid dataset: {len_df}'.format(len_df=len(valid_df)),
        )
        logger.report_text(
            'Test dataset: {len_df}'.format(len_df=len(test_df)),
        )
        logger.report_text('Datasets successfully saved!')


def _read_datasets(
    data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(data_path, 'train_df.csv'))
    valid_df = pd.read_csv(os.path.join(data_path, 'valid_df.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_df.csv'))
    return train_df, valid_df, test_df
