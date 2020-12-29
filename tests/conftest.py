import os
import shutil
from . import TestUtils


def pytest_sessionstart():
    os.makedirs('tests_tmp')
    os.chdir('tests_tmp')
    TestUtils.create_test_files()


def pytest_sessionfinish():
    shutil.rmtree('tests_tmp')
