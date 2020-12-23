from . import get_logger, get_config_path
import skeltorch
import pytest


@pytest.mark.parametrize('category_name', ['category_1', 'category_2'])
def test_categories_load(category_name):
    config = skeltorch.Configuration(get_logger())
    config.create(get_config_path())
    assert category_name in dir(config)


@pytest.mark.parametrize('category_name', ['category_1', 'category_2'])
@pytest.mark.parametrize('item_name', ['item_1', 'item_2'])
def test_items_load(category_name, item_name):
    config = skeltorch.Configuration(get_logger())
    config.create(get_config_path())
    assert item_name in getattr(config, category_name).keys()


@pytest.mark.parametrize('category_name,item_name,value', [
    ('category_1', 'item_1', 1234),
    ('category_1', 'item_2', 5678),
    ('category_2', 'item_1', 'item_1_value'),
    ('category_2', 'item_2', 'item_2_value')
])
def test_get(category_name, item_name, value):
    config = skeltorch.Configuration(get_logger())
    config.create(get_config_path())
    assert config.get(category_name, item_name) == value


@pytest.mark.parametrize('category_name,item_name,value', [
    ('category_1', 'item_1', 1234),
    ('category_1', 'item_2', 5678),
    ('category_2', 'item_1', 'item_1_value'),
    ('category_2', 'item_2', 'item_2_value')
])
def test_set(category_name, item_name, value):
    config = skeltorch.Configuration(get_logger())
    config.set(category_name, item_name, value)
    assert config.get(category_name, item_name) == value
