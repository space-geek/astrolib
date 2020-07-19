
import setuptools

_NAME='integrationutils'
_VERSION='0.0.1'
_DESCRIPTION = 'Utility library providing implementations of various numerical integration schemes.'

_INSTALL_REQUIRES = [
    '',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name=_NAME,
        version=_VERSION,
        author='David Black',
        author_email='dblack1021@gmail.com',
        description=_DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='',
        instal_requires=_INSTALL_REQUIRES,
        packages=[NAME],
        package_dir={'': 'src'}
        )
