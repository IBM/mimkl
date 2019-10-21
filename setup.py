"""Install package."""
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import traceback
import unittest
from distutils.command.build import build as _build
from multiprocessing import cpu_count

from setuptools import Command, find_packages, setup
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.develop import develop as _develop

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSES = cpu_count()
PROCESSES = str(PROCESSES - 1) if (PROCESSES > 1) else 1


class setup_pymimkl(Command):
    """Build MIMKL and install pymimkl."""

    description = 'Run script to setup MIMKL'

    def initialize_options(self):
        """Set initialize options."""
        pass

    def finalize_options(self):
        """Set finalize options."""
        pass

    def _install_pymimkl(self):
        """Build and install _pymimkl."""
        print('Install _pymimkl')
        print('Updating submodules')
        # probably redundant but needed if someone just cloned the repo
        if os.path.exists(os.path.join(SETUP_DIR, '.git')):
            subprocess.check_call(['git', 'submodule', 'init'])
            subprocess.check_call(['git', 'submodule', 'update'])

        build_directory = os.path.join(SETUP_DIR, 'build_')
        os.makedirs(build_directory, exist_ok=True)
        print('Building _pymimkl in {}'.format(build_directory))
        subprocess.check_call(
            [
                os.path.join(SETUP_DIR, 'setup_mimkl.sh'),
                SETUP_DIR, build_directory, sys.executable, PROCESSES
            ]
        )

        package_directory = os.path.join(SETUP_DIR, 'python', 'pymimkl')
        print('Adding _pymimkl to local site {}'.format(package_directory))
        # then it is shipped py setup(package_data)
        pymimkl_build_directory = os.path.join(
            build_directory, 'python', 'pymimkl'
        )
        pymimkl_built_files = [
            os.path.join(pymimkl_build_directory, entry)
            for entry in os.listdir(pymimkl_build_directory)
            if (
                entry.startswith('_pymimkl') and
                entry.endswith('.so')
            )
        ]

        for module_file in pymimkl_built_files:
            shutil.copy(
                module_file,
                package_directory
            )
        try:
            if self.develop:
                pass
            else:
                raise AttributeError
        except AttributeError:
            print('Cleaning up')
            shutil.rmtree(build_directory, ignore_errors=True)

    def run(self):
        """Run installation of _pymimkl."""
        self._install_pymimkl()


class build(_build):
    """Build command."""

    sub_commands = [
        ('setup_pymimkl', None)
    ] + _build.sub_commands


class bdist_egg(_bdist_egg):
    """Build bdist_egg."""

    def run(self):
        """Run build bdist_egg."""
        self.run_command('setup_pymimkl')
        _bdist_egg.run(self)


class develop(_develop):
    """Build develop."""

    def run(self):
        """Run build develop."""
        setup_pymimkl = self.distribution.get_command_obj(
            'setup_pymimkl'
        )
        setup_pymimkl.develop = True
        self.run_command('setup_pymimkl')
        _develop.run(self)


def test_suite():
    """Enable `python setup.py test`."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('python/pymimkl/tests')
    return test_suite


if __name__ == '__main__':
    setup(
        name='pymimkl',
        version='0.2.0',
        description=('Supervised and Unsupervised Multiple Kernel Learning '
                     'including matrix induction'),
        url='https://github.com/IBM/mimkl',
        author='Joris Cadow, Matteo Manica',
        author_email='joriscadow@gmail.com, drugilsberg@gmail.com',
        packages=find_packages('python'),
        package_dir={'': 'python'},
        package_data={'pymimkl': ['_pymimkl*.so']},
        zip_safe=False,
        cmdclass={
            'bdist_egg': bdist_egg,
            'build': build,
            'setup_pymimkl': setup_pymimkl,
            'develop': develop
        },
        tests_require=["numpy", "scipy"],
        install_requires=["numpy"],
        test_suite="setup.test_suite",
        license="MIT",
        keywords=[
            'multiple kernel learning', 'MKL',
            'unsupervised', 'kernel', 'matrix induced',
            'mimkl', 'pimkl', 'pathway induced',
            'EasyMKL'
        ],
        classifiers=[
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Mathematics',
        ],
    )
