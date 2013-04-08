from setuptools import setup, find_packages
setup(
    name = "hmf",
    version = "0.9.9",
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