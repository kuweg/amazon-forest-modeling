import os
from functools import wraps


TESTS_DIR = os.path.dirname(__file__)
DATAFILES = [
    'df_encoded.csv',
    'test_df.csv',
    'train_df.csv',
    'valid_df.csv',
]


def cleanup_after(files_list: list, files_path: str):

    files_list = [os.path.join(files_path, fname) for fname in files_list]

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)

            for file in files_list:
                os.remove(file)
        return wrapper
    return inner
