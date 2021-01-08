__version__ = '1.1.1'
__release__ = __version__

__all__ = ['hidef_finder', 'weaver', 'utils', 'logger']

from .logger import PackageLogger
LOGGER = PackageLogger('hidef')

# from . import hidef_finder, weaver, utils
# from .weaver import *
# __all__.extend(weaver.__all__)

