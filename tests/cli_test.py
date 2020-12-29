import os
import pytest
import subprocess


def test_create_empty_name():
    res = subprocess.run(['skeltorch', 'create'], capture_output=True)
    assert res.returncode == 1


def test_create():
    res = subprocess.run(
        ['skeltorch', 'create', '--name', 'cli_test'], capture_output=True
    )
    assert res.returncode == 0


@pytest.mark.parametrize('file_path', [
    'data/.gitkeep', 'experiments/.gitkeep', 'cli_test/__init__.py',
    'cli_test/__main__.py', 'cli_test/data.py', 'cli_test/model.py',
    'cli_test/runner.py', 'config.default.json', 'config.schema.json',
    'README.md'
])
def test_create_files_exist(file_path):
    assert os.path.exists(file_path)
