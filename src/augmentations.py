import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_width: int, img_height: int) -> albu.Compose:
    """
    Augmentation transformations for train dataset.

    Args:
        img_width (int): img width
        img_height (int): img height

    Returns:
        albu.Compose: transformations.
    """
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(),
        albu.GaussianBlur(),
        albu.Resize(height=img_height, width=img_width),
        albu.Normalize(),
        ToTensorV2(),
    ],
    )


def get_valid_transforms(img_width: int, img_height: int) -> albu.Compose:
    """
    Augmentation transformations for validation dataset.

    Args:
        img_width (int): img width
        img_height (int): img height

    Returns:
        albu.Compose: transformations
    """
    return albu.Compose([
        albu.Resize(height=img_height, width=img_width),
        albu.Normalize(),
        ToTensorV2(),
    ],
    )
