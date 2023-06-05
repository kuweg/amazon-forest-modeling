import argparse
import os
import shutil
from typing import Any
from zipfile import ZipFile

import wget

from configs.config import Config
from src.constants import DATA_ARCHIVE_URL, LOCAL_ROOT


def arg_parse() -> Any:  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    parser.add_argument(
        '--rewrite',
        type=bool,
        default=False,
        required=False,
        help='rewriting data folder',
    )
    return parser.parse_args()


def main(
    config: Config,
    rewrite: bool = False,
) -> None:
    """Dowloading AmazonForest dataset.

    Args:
        config (Config):
            Yaml file for data manimulation.
        rewrite (bool):
            Rewriting directory if it exists.
            Defaults to False.

    Raises:
        FileExistsError: If data directory exists,
            contains files and flag rewrite is False
    """
    data_path = config.data_config.data_path
    if os.path.exists(data_path):
        if rewrite:
            shutil.rmtree(data_path)
        else:
            if os.listdir(data_path):
                raise FileExistsError(
                    (
                        'Specified directory {dir} already'.
                        format(dir=data_path),
                        'exists and contains files.',
                        ' You may use flag `rewrite` or give a new path.',
                    ),
                )
            else:
                pass  # noqa: WPS420
    else:
        os.mkdir(data_path)

    data_archive = wget.download(DATA_ARCHIVE_URL)

    zip_archive = ZipFile(data_archive)
    zip_archive.extractall(path=data_path)

    local_root = os.path.join(data_path, LOCAL_ROOT)
    all_files = os.listdir(local_root)

    for s_file in all_files:
        shutil.move(os.path.join(local_root, s_file), data_path)

    shutil.rmtree(local_root)
    shutil.rmtree(os.path.join(data_path, 'test-jpg'))
    os.remove(os.path.join(data_path, 'sample_submission.csv'))


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    main(config, args.rewrite)
