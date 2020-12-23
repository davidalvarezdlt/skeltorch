from . import get_config_path
import os
import json


def _create_config_file(config_path):
    config_content = {
        'category_1': {
            'item_1': 1234,
            'item_2': 5678
        },
        'category_2': {
            'item_1': 'item_1_value',
            'item_2': 'item_2_value'
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config_content, f)


def pytest_sessionstart(session):
    _create_config_file(get_config_path())


def pytest_sessionfinish(session, exitstatus):
    os.remove(get_config_path())
