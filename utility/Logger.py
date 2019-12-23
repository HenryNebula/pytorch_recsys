import logging
import os


def create_logger(log_fname):

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "(%(asctime)s)\t%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)

    log_fname = os.path.join("./log/", log_fname)
    dir_path = os.path.split(log_fname)[0]

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fhlr = logging.FileHandler(log_fname)
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger