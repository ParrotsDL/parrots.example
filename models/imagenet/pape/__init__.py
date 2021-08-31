# flake8: noqa
import sys
import logging
from .version import __version__
from . import environ
# from . import data
from . import distributed
from . import half
from . import op
from . import parallel
from . import utils
# from . import quantization

is_match, err = environ.check_current_env()
if not is_match:
    print(err, flush=True)
    sys.exit()

logger = logging.getLogger('pape')
logger_all = logging.getLogger('pape_all')

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
logger_all.addHandler(ch)

logger_all.setLevel(logging.INFO)
