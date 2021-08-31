import os
import importlib
from collections import namedtuple

from .. import environ

if environ.pytorch_version:
    def load_ext(name, funcs):
        ext = importlib.import_module('pape.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), '{} miss in module {}'.format(fun, name)
        return ext
else:
    from parrots import extension

    def load_ext(name, funcs):
        ExtModule = namedtuple('ExtModule', funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            ext_list.append(extension.load(fun, name, lib_dir=lib_root).op_)
        return ExtModule(*ext_list)
