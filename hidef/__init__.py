__version__ = '0.1.0'
__release__ = __version__

__all__ = []

from .logger import PackageLogger
LOGGER = PackageLogger('hidef')

from . import weaver
from .weaver import *
__all__.extend(weaver.__all__)

