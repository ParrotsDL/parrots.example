from collections import OrderedDict

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .base import BaseFileHandler  # isort:skip


class OrderedLoader(Loader):
    """ Ordered loader """
    pass


class OrderedDumper(Dumper):
    """ Ordered dumper """
    pass


def dict_representer(dumper, data):
    return dumper.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())


def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                              construct_mapping)
OrderedDumper.add_representer(dict, dict_representer)


class YamlHandler(BaseFileHandler):

    def load_from_str(self, obj, **kwargs):
        if 'ordered' in kwargs.keys() and kwargs.pop('ordered'):
            kwargs.setdefault('Loader', OrderedLoader)
        else:
            kwargs.setdefault('Loader', Loader)
        return yaml.load(obj, **kwargs)

    def load_from_fileobj(self, file, **kwargs):
        if 'ordered' in kwargs.keys() and kwargs.pop('ordered'):
            kwargs.setdefault('Loader', OrderedLoader)
        else:
            kwargs.setdefault('Loader', Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        if 'ordered' in kwargs.keys() and kwargs.pop('ordered'):
            kwargs.setdefault('Dumper', OrderedDumper)
        else:
            kwargs.setdefault('Dumper', Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        if 'ordered' in kwargs.keys() and kwargs.pop('ordered'):
            kwargs.setdefault('Dumper', OrderedDumper)
        else:
            kwargs.setdefault('Dumper', Dumper)
        return yaml.dump(obj, **kwargs)
