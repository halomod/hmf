import inspect
import os

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf.cosmo import Cosmology, WMAP7
import numpy as np
