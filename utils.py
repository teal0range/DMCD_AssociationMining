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
    if not fileHandler[name]:
        fh = logging.FileHandler(os.path.join('log', f'{name}.log'), mode='a', encoding='utf-8', delay=False)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s][%(levelname)5s] %(filename)s    %(message)s")
        fh.setFormatter(fmt)
        _logger.handlers.append(fh)
        fileHandler[name] = True
    return _logger


logger = getLogger("log")
