from setuptools import setup, find_packages
from hmf.hmf import version
import os
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="hmf",
    version=version,
    packages=['hmf'],
    install_requires=[],
    requires=['numpy',
                'scitools',
                'scipy',
                'pycamb'],
    author="Steven Murray",
    author_email="steven.murray@uwa.edu.au",
    description="A halo mass function calculator",
    long_description=read('README'),
    license='BSD',
    keywords="halo mass function",
    url="https://github.com/steven-murray/hmf",
    # could also include long_description, download_url, classifiers, etc.
)
