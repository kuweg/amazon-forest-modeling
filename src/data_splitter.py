import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)


def stratify_shuffle_split_subsets(
    full_dataset: pd.DataFrame,
    img_path_column: str = 'image_name',
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split dataset.

    Stratify, shuffle, split a multi-class multi-label dataset
    into train/validation/test sets.
    Source code got from here:
    https://www.richarddecal.com/2020-11-22-Howto-stratified-splitting-Multiclass-Multilabeled-image-classification-dataset/

    Args:
        full_dataset (pd.DataFrame):
            Full supervised dataset.
            One column is the img urls, and the rest are binary labels.
        img_path_column (str):
            Name of the img path column. Defaults to image_name.
        train_fraction (float):
            The fraction of data to reserve for the training dataset.
            The remaining data will be evenly
            split into the dev and validation subsets. Defaults to 0.8.

    Raises:
        ValueError: if datasets contains a duplicated fields

    Returns:
        tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train/validation/test subsets.
    """
    # pandas documentation says to use .to_numpy() for consistency
    img_urls = full_dataset[img_path_column].to_numpy()

    # sanity check: no duplicate labels
    if len(img_urls) != len(set(img_urls)):
        raise ValueError('Duplicate image keys detected.')

    labels = full_dataset.drop(
        columns=[img_path_column],
    ).to_numpy().astype(float)
    # NOTE generators are replicated across workers.
    # do stratified shuffle split beforehand.
    logging.info('Stratifying dataset iteratively. this may take a while.')
    # NOTE: splits >2 broken;
    # https://github.com/scikit-multilearn/scikit-multilearn/issues/209
    # so, do 2 rounds of iterative splitting.
    train_indexes, everything_else_indexes = _split(
        img_urls,
        labels,
        [1.0 - train_fraction, train_fraction],
    )
    x_train = img_urls[train_indexes]
    x_else = img_urls[everything_else_indexes]
    y_train = labels[train_indexes, :]
    y_else = labels[everything_else_indexes, :]

    dev_indexes, val_indexes = _split(x_else, y_else)

    x_dev = x_else[dev_indexes]
    x_val = x_else[val_indexes]
    y_dev = y_else[dev_indexes, :]
    y_val = y_else[val_indexes, :]

    # combine (x,y) data into dataframes
    train_subset = combine_subset(
        img_path_column,
        pd.Series(x_train),
        y_train,
        full_dataset.columns,
    )

    dev_subset = combine_subset(
        img_path_column,
        pd.Series(x_dev),
        y_dev,
        full_dataset.columns,
    )

    val_subset = combine_subset(
        img_path_column,
        pd.Series(x_val),
        y_val,
        full_dataset.columns,
    )

    logging.info('Stratifying dataset is completed.')

    return train_subset, val_subset, dev_subset


def combine_subset(
    x_col_name: str,
    x_col_vals: pd.Series,
    y_vals: np.array,
    columns: list = None,
) -> pd.DataFrame:
    """Combine X and Y data into single DataFrame.

    Args:
        x_col_name (str): columns name for X vals.
        x_col_vals (pd.Series): X vals.
        y_vals (np.array): Y vals.
        columns (list): Optional. New columns

    Returns:
        pd.DataFrame: merged DataFrame with X and Y.
    """
    subset = pd.DataFrame(y_vals)
    subset.insert(0, x_col_name, x_col_vals)
    if columns is not None:
        subset.columns = columns
    return subset


def _split(
    img_urls: np.array,
    labels: np.array,
    sample_distribution_per_fold: Union[None, List[float]] = None,
) -> Tuple[np.array, np.array]:
    """Create split for dataset by given sample distribution.

    Args:
        img_urls (np.array): columns with images names.
        labels (np.array): labels for images.
        sample_distribution_per_fold (None, Tuple[float]): ratio for splitting.

    Raises:
        ValueError: if splitting creates overlaps.

    Returns:
        Tuple[np.array, np.array]:
        A 2 subsets of indexes for image urls and labels.
    """
    stratifier = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=sample_distribution_per_fold,
    )

    # this class is a generator that produces k-folds.
    # we just want to iterate it once to make a single static split.
    train_indexes, everything_else_indexes = next(
        stratifier.split(X=img_urls, y=labels),
    )

    num_overlapping_samples = len(
        set(train_indexes).intersection(set(everything_else_indexes)),
    )
    if num_overlapping_samples != 0:
        raise ValueError(
            'First splitting failed, {num} overlapping samples detected'.
            format(num=num_overlapping_samples),
        )

    return train_indexes, everything_else_indexes
