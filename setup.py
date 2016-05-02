from setuptools import setup

import os
import sys
import re
import io

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_wheel upload")
    sys.exit()


setup(
    name="hmf",
    version=find_version("hmf", "__init__.py"),
    packages=['hmf', 'hmf.fitting'],
    install_requires=["numpy>=1.6.2",
                      "scipy>=0.12.0",
                      "astropy>=1.0"],
    scripts=["scripts/hmf", "scripts/hmf-fit"],
    author="Steven Murray",
    author_email="steven.murray@uwa.edu.au",
    description="A halo mass function calculator",
    long_description=read('README.rst'),
    license="MIT",
    keywords="halo mass function",
    url="https://github.com/steven-murray/hmf",
    # could also include long_description, download_url, classifiers, etc.
)
