import skeltorch
import examples.mnist_classifier.mnist_classifier as mnist_example
import os
import sys

experiment_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'examples',
    'mnist_classifier'
)


def test_init():
    sys.argv = [
        sys.argv[0],
        '--experiments-path', os.path.join(experiment_path, 'experiments'),
        '--data-path', os.path.join(experiment_path, 'data'),
        '--experiment-name', 'unit_testing',
        '--verbose',
        'init',
        '--config-path', os.path.join(experiment_path, 'config.default.json')
    ]
    skeltorch.Skeltorch(
        mnist_example.MNISTClassifierData(),
        mnist_example.MNISTClassifierRunner()
    ).run()


def test_train():
    sys.argv = [
        sys.argv[0],
        '--experiments-path', os.path.join(experiment_path, 'experiments'),
        '--data-path', os.path.join(experiment_path, 'data'),
        '--experiment-name', 'unit_testing',
        '--verbose',
        'train'
    ]
    skeltorch.Skeltorch(
        mnist_example.MNISTClassifierData(),
        mnist_example.MNISTClassifierRunner()
    ).run()


test_train()
