import logging
import os


def get_logger():
    return logging.getLogger('skeltorch')


def get_config_path():
    return os.path.join(os.path.dirname(__file__), 'config.tests.json')
