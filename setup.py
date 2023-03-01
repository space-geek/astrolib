""" Module contains packaging specification for the wheel
    for the parent codebase.
"""
import os
from setuptools import find_packages
from setuptools import setup

_NAME = "astrolib"
_AUTHOR = "David Black"
_AUTHOR_EMAIL = "dblack1021@gmail.com"
_VERSION = os.environ.get("PY_VERSION") or "0.0.1"
_DESCRIPTION = "Utility library providing implementations of various astrodynamics-related utilities and models."

_INSTALL_REQUIRES = []

_DEV_REQUIRES = [
    "pytest==7.1.2",
    "pylint==2.16.2",
    "wheel==0.37.1",
    "black==22.12.0",
]

with open("README.md", "r", encoding="UTF-8") as f:
    _LONG_DESCRIPTION = f.read()


setup(
    name=_NAME,
    version=_VERSION,
    author=_AUTHOR,
    author_email=_AUTHOR_EMAIL,
    description=_DESCRIPTION,
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="",
    install_requires=_INSTALL_REQUIRES,
    extras_require={
        "dev": _DEV_REQUIRES,
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
