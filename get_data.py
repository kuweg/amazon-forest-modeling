import os
from io import BytesIO
from zipfile import ZipFile
import shutil
import wget
import argparse
from typing import Any


from configs.config import Config
from src.constants import DATA_ARCHIVE_URL, LOCAL_ROOT


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    parser.add_argument("--rewrite",
                        type=bool,
                        default=False,
                        required=False,
                        help="rewriting data folder",)
    return parser.parse_args()


def main(config: Config,
         rewrite: bool=False,
    ) -> None:

    data_path = config.data_config.data_path
    if os.path.exists(data_path):
        if rewrite:
            shutil.rmtree(data_path)
        else:
            if os.listdir(data_path):
                raise FileExistsError(
                    'Specified directory {dir} already exists and contains files.'.
                    format(dir=data_path) +
                    ' You may use flag `rewrite` or give a new path.')
            else:
                pass
    else:
        os.mkdir(data_path)
    
    
    data_archive = wget.download(DATA_ARCHIVE_URL)
    
    zip_archive = ZipFile(data_archive)
    zip_archive.extractall(path=data_path)
    
    local_root = os.path.join(data_path, LOCAL_ROOT)
    all_files = os.listdir(local_root)
        
    for f in all_files:
        shutil.move(os.path.join(local_root, f), data_path)
        
    shutil.rmtree(local_root)
    shutil.rmtree(os.path.join(data_path, 'test-jpg'))
    os.remove(os.path.join(data_path, 'sample_submission.csv'))


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    main(config, args.rewrite)
    