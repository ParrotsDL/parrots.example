from parrots.utils.build_extension import Extension, BuildExtension
from setuptools import setup, distutils
import distutils.command.build
import distutils.command.clean

import os
import glob
import shutil

path = os.path.dirname(os.path.realpath(__file__))
pybind_src_path = os.path.join(path, 'dumpExample.cpp')

extensions = []

ext_pybind = Extension(
    name='dump_example',
    sources=[pybind_src_path]
)
extensions.append(ext_pybind)


class build(distutils.command.build.build):
    def run(self):
        self.run_command('build_ext')
        for filename in glob.glob('build/lib*/*.so'):
            shutil.copy(filename, '.')
        distutils.command.build.build.run(self)


class clean(distutils.command.clean.clean):
    def run(self):
        dirs = ['build', './*.so', 'dump_extension.egg-info', 'lib', 'dist']
        for wildcard in dirs:
            for filename in glob.glob(wildcard):
                try:
                    os.remove(filename)
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)
        distutils.command.clean.clean.run(self)


if __name__ == '__main__':
    setup(
        name='dump_extension',
        ext_modules=extensions,
        cmdclass={
            'build_ext': BuildExtension,
            'build': build,
            'clean': clean,
        },
    )
