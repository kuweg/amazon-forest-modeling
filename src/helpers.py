import importlib
from typing import Any


def load_object(obj_path: str, default_obj_path: str = '') -> Any:
    """Load object dynamically.

    Import and load object while execution.

    Args:
        obj_path (str): Object name to import
        default_obj_path (str): Defaults to ''.

    Raises:
        AttributeError: If object cannot be found.

    Returns:
        Any: Imported object.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    if len(obj_path_list) > 1:
        obj_path = obj_path_list.pop(0)
    else:
        obj_path = default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):  # noqa: WPS421
        raise AttributeError(
            'Object {obj_name} cannot be loaded from {obj_path}'.
            format(obj_name=obj_name, obj_path=obj_path),
        )
    return getattr(module_obj, obj_name)
