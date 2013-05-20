from setuptools import setup, find_packages
from hmf.Perturbations import version

setup(
    name="hmf",
    version=version,
    packages=find_packages(),
    install_requires=[],
    requires=['numpy',
                'scitools',
                'scipy',
                'pycamb'],
    author="Steven Murray",
    author_email="steven.jeanette.m@gmail.com",
    description="A halo mass function calculator",
    keywords="halo mass function",
    url="https://github.com/steven-murray/hmf",
    # could also include long_description, download_url, classifiers, etc.
)
