import json
import logging
import os
import skeltorch


class TestUtils:
    _paths = {
        'config': 'config.tests.json',
        'config_bin': 'config.tests.pkl',
        'data': 'data.tests.pkl'
    }

    @staticmethod
    def get_path(path_name):
        return os.path.join(
            os.path.dirname(__file__), TestUtils._paths[path_name]
        )

    @staticmethod
    def get_logger():
        return logging.getLogger('skeltorch')

    @staticmethod
    def create_test_files():
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
        with open(TestUtils.get_path('config'), 'w') as f:
            json.dump(config_content, f)


class TestData(skeltorch.data.Data):

    def create(self, data_path):
        pass

    def load_datasets(self, data_path):
        pass

    def load_loaders(self, data_path, num_workers):
        pass
