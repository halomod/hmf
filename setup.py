from setuptools import setup, find_packages

import os
import sys

class Mock(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        elif name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        else:
            return Mock()

MOCK_MODULES = ['cosmolopy']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

from hmf.hmf import version

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_egg upload")
    sys.exit()

setup(
    name="hmf",
    version=version,
    packages=['hmf'],
    install_requires=["numpy>=1.6.2",
                      "scipy>=0.12.0",
                      "cosmolopy",
                      "emcee>=2.0"],
    scripts=["scripts/hmf"],
    author="Steven Murray",
    author_email="steven.murray@uwa.edu.au",
    description="A halo mass function calculator",
    long_description=read('README.rst'),
    license="MIT",
    keywords="halo mass function",
    url="https://github.com/steven-murray/hmf",
    # could also include long_description, download_url, classifiers, etc.
)
