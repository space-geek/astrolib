
import os
from setuptools import find_packages
from setuptools import setup

_NAME = 'integrationutils'
_VERSION = os.environ.get('PY_VERSION') or '0.0.1'
_DESCRIPTION = 'Utility library providing implementations of various numerical integration schemes.'

_INSTALL_REQUIRES = [
    '',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=_NAME,
    version=_VERSION,
    author='David Black',
    author_email='dblack1021@gmail.com',
    description=_DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    install_requires=_INSTALL_REQUIRES,
    packages=find_packages('src'),
    package_dir={'': 'src'}
)
