import logging
import os
from collections import defaultdict

fileHandler = defaultdict(lambda: False)


def getLogger(name):
    if not os.path.exists("log"):
        os.mkdir("log")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(levelname)5s] %(filename)s    %(message)s')
    _logger = logging.getLogger(name)
    return _logger


logger = getLogger("log")
