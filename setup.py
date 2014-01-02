from setuptools import setup, find_packages
from hmf.hmf import version
import os
import sys

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
    install_requires=["numpy >= 1.6.2",
                      "scitools",
                      "scipy >= 0.12.0",
                      "cosmolopy",
                      "emcee"],
    author="Steven Murray",
    author_email="steven.murray@uwa.edu.au",
    description="A halo mass function calculator",
    long_description=read('README.rst'),
    license='BSD',
    keywords="halo mass function",
    url="https://github.com/steven-murray/hmf",
    # could also include long_description, download_url, classifiers, etc.
)
