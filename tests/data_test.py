from . import TestData, TestUtils


def test_data_save_load():
    data, data_2 = TestData(), TestData()
    data.attr1, data._attr2 = 1234, 5678
    data.save(TestUtils.get_path('data'))
    data_2.load('', TestUtils.get_path('data'), 1)
    assert data_2.attr1 == 1234 and data_2._attr2 == 5678
