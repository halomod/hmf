from setuptools import setup, find_packages
from hmf.Perturbations import version

def generate_version_py():
    fid = open("__version.py",'w')
    try:
        fid.write("version = %s\n" % version)
    finally:
        fid.close()
        
#generate_version_py()

setup(
    name = "hmf",
    version = version,
    packages = find_packages(),
    
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['numpy',
                        'scitools',
                        'scipy',
                        'pycamb'],

    # metadata for upload to PyPI
    author = "Steven Murray",
    author_email = "steven.jeanette.m@gmail.com",
    description = "A halo mass function calculator",
    keywords = "halo mass function",
    url = "https://github.com/steven-murray/hmf",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)