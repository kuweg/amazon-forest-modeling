import os
from typing import Optional

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class AmazonForestDataset(Dataset):  # noqa: D101
    def __init__(
        self,
        df: pd.DataFrame,
        image_folder: str,
        transforms: Optional[albu.Compose] = None,
    ):
        self.df = df
        self.image_folder = image_folder
        self.transforms = transforms

    def __getitem__(self, idx: int):  # noqa: D105
        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.image_folder,
            '{im_name}.jpg'.format(im_name=row.image_name),
        )
        labels = np.array(row.drop(['image_name']), dtype='float32')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        dataset = {'image': image, 'labels': labels}

        if self.transforms:
            dataset = self.transforms(**dataset)

        return dataset['image'], dataset['labels']

    def __len__(self) -> int:  # noqa: D105
        return len(self.df)
