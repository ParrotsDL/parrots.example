# Copyright (c) Open-MMLab. All rights reserved.
from pathlib import Path

from .handlers import (BaseFileHandler, JsonHandler, PickleHandler,
                       PythonHandler, YamlHandler)

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler(),
    'py': PythonHandler()
}


def load(file_obj, file_format=None, from_obj=False, **kwargs):
    """Load data from json/yaml/pickle/py files or objs.

    This method provides a unified api for loading data from serialized files
        or objs.

    Args:
        file_obj (str or :obj:`Path` or file-like object or file_format
            object): Filename or a file-like object or a file_format like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        from_obj (bool): If specified, it will load from file_format object.

    Returns:
        The content from the file or object.
    """
    if isinstance(file_obj, Path):
        file_obj = str(file_obj)
    if file_format is None and isinstance(file_obj, str):
        file_format = file_obj.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    if from_obj:
        obj = handler.load_from_str(file_obj, **kwargs)
    elif isinstance(file_obj, str):
        obj = handler.load_from_path(file_obj, **kwargs)
    elif hasattr(file_obj, 'read'):
        obj = handler.load_from_fileobj(file_obj, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, **kwargs):
    """Dump data to json/yaml/pickle/py strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dump to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError('Unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        handler.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def _register_handler(handler, file_formats):
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(
            'handler must be a child of BaseFileHandler, not {}'.format(
                type(handler)))
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    for ext in file_formats:
        if not isinstance(ext, str):
            raise TypeError('file_formats must be a str or a list of str')
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats, **kwargs):

    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
