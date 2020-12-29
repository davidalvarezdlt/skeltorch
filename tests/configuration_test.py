from . import TestUtils
import pytest
import skeltorch


@pytest.mark.parametrize('category_name', ['category_1', 'category_2'])
def test_configuration_create_categories(category_name):
    config = skeltorch.Configuration(TestUtils.get_logger())
    config.create(TestUtils.get_path('config'))
    assert category_name in dir(config)


@pytest.mark.parametrize('category_name', ['category_1', 'category_2'])
@pytest.mark.parametrize('item_name', ['item_1', 'item_2'])
def test_configuration_create_items(category_name, item_name):
    config = skeltorch.Configuration(TestUtils.get_logger())
    config.create(TestUtils.get_path('config'))
    assert item_name in getattr(config, category_name).keys()


@pytest.mark.parametrize('category_name,item_name,value', [
    ('category_1', 'item_1', 1234),
    ('category_1', 'item_2', 5678),
    ('category_2', 'item_1', 'item_1_value'),
    ('category_2', 'item_2', 'item_2_value')
])
def test_configuration_get(category_name, item_name, value):
    config = skeltorch.Configuration(TestUtils.get_logger())
    config.create(TestUtils.get_path('config'))
    assert config.get(category_name, item_name) == value


@pytest.mark.parametrize('category_name,item_name,value', [
    ('category_1', 'item_1', 1234),
    ('category_1', 'item_2', 5678),
    ('category_2', 'item_1', 'item_1_value'),
    ('category_2', 'item_2', 'item_2_value')
])
def test_configuration_set(category_name, item_name, value):
    config = skeltorch.Configuration(TestUtils.get_logger())
    config.set(category_name, item_name, value)
    assert config.get(category_name, item_name) == value


@pytest.mark.parametrize('category_name', ['category_1', 'category_2'])
@pytest.mark.parametrize('item_name', ['item_1', 'item_2'])
def test_configuration_save(category_name, item_name):
    config = skeltorch.Configuration(TestUtils.get_logger())
    config.create(TestUtils.get_path('config'))
    config.save(TestUtils.get_path('config_bin'))
    config_2 = skeltorch.Configuration(TestUtils.get_logger())
    config_2.load(TestUtils.get_path('config_bin'))
    assert config.get(category_name, item_name) == \
           config_2.get(category_name, item_name)
