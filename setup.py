from setuptools import setup, find_packages

import os
import sys
import io


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


if sys.argv[-1] == "publish":
    os.system("rm dist/*")
    os.system("python setup.py sdist")
    os.system("python setup.py bdist_wheel")
    os.system("twine upload dist/*")
    sys.exit()

test_req = [
    "coverage>=4.5.1",
    "pytest>=3.5.1",
    "pytest-cov>=2.5.1",
    "pre-commit",
    "mpmath>=1.0.0",
    "colossus>=1.2.1",
]

docs_req = [
    "Sphinx==1.7.5",
    "numpydoc>=0.8.0",
    "nbsphinx",
]
setup(
    name="hmf",
    packages=find_packages(),
    use_scm_version={"local_scheme": "no-local-version"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "numpy>=1.6.2",
        "scipy>=0.12.0",
        "astropy>=1.1",
        "camb>=1.0.0<2.0",
    ],
    extras_require={"test": test_req, "doc": docs_req, "dev": test_req + docs_req,},
    scripts=["scripts/hmf", "scripts/hmf-fit"],
    author="Steven Murray",
    author_email="steven.g.murray@asu.edu",
    description="A halo mass function calculator",
    long_description=read("README.rst"),
    license="MIT",
    keywords="halo mass function; cosmology",
    url="https://github.com/steven-murray/hmf",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
