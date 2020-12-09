import argparse
import logging
import os.path
from skeltorch.experiment import Experiment
from skeltorch.execution import Execution
from skeltorch.configuration import Configuration
from skeltorch.data import Data
from skeltorch.runner import Runner
import sys

__all__ = ['Skeltorch', 'Data', 'Runner']
__cli_commands__ = {
    'args': [
        {
            'id': '--experiment-name',
            'type': str,
            'nargs': None,
            'default': None,
            'required': True,
            'help': 'Name of the experiment.'
        },
        {
            'id': '--base-path',
            'type': str,
            'nargs': None,
            'required': False,
            'default': os.path.dirname(os.path.dirname(sys.argv[0])),
            'help': 'Base path from which other default paths are referenced.'
        },
        {
            'id': '--experiments-path',
            'type': str,
            'nargs': None,
            'default': None,
            'required': False,
            'help': 'Folder where the experiments are stored.'
        },
        {
            'id': '--data-path',
            'type': str,
            'nargs': None,
            'default': None,
            'required': False,
            'help': 'Folder where the data is stored.'
        },
        {
            'id': '--verbose',
            'type': bool,
            'nargs': None,
            'default': None,
            'required': False,
            'help': 'Whether to log using standard output.'
        }
    ],
    'commands': [
        {
            'name': 'init',
            'args': [
                {
                    'id': '--config-path',
                    'type': str,
                    'nargs': None,
                    'default': None,
                    'required': True,
                    'help': 'Path of the configuration file used to create '
                            'the experiment.'
                },
                {
                    'id': '--config-schema-path',
                    'type': str,
                    'nargs': None,
                    'default': None,
                    'required': False,
                    'help': 'Path of the schema file used to validate the '
                            'configuration.'
                },
                {
                    'id': '--seed',
                    'type': int,
                    'nargs': None,
                    'default': 0,
                    'required': False,
                    'help': 'Seed used to initialize random value generators.'
                }
            ]
        },
        {
            'name': 'info',
            'args': []
        },
        {
            'name': 'train',
            'args': [
                {
                    'id': '--epoch',
                    'type': int,
                    'nargs': None,
                    'default': None,
                    'required': False,
                    'help': 'Epoch from which to continue the training.'
                },
                {
                    'id': '--max-epochs',
                    'type': int,
                    'nargs': None,
                    'default': 999,
                    'required': False,
                    'help': 'Maximum number of epochs to complete.'
                },
                {
                    'id': '--log-period',
                    'type': int,
                    'nargs': None,
                    'default': 100,
                    'required': False,
                    'help': 'Number of mini-batch iterations between status '
                            'logs.'
                },
                {
                    'id': '--num-workers',
                    'type': int,
                    'nargs': None,
                    'default': 1,
                    'required': False,
                    'help': 'Number of workers used when loading the data.'
                },
                {
                    'id': '--device',
                    'type': int,
                    'nargs': '+',
                    'default': None,
                    'required': False,
                    'help': 'PyTorch-friendly name of the device where the '
                            'model should be stored and trained/tested.'
                }
            ]
        },
        {
            'name': 'test',
            'args': [
                {
                    'id': '--epoch',
                    'type': int,
                    'nargs': None,
                    'default': None,
                    'required': True,
                    'help': 'Epoch used to run the testing.'
                },
                {
                    'id': '--num-workers',
                    'type': int,
                    'nargs': None,
                    'default': 1,
                    'required': False,
                    'help': 'Number of workers used when loading the data.'
                },
                {
                    'id': '--device',
                    'type': int,
                    'nargs': '+',
                    'default': None,
                    'required': False,
                    'help': 'PyTorch-friendly name of the device where the '
                            'model should be stored and trained/tested.'
                }
            ]
        },
        {
            'name': 'tensorboard',
            'args': [
                {
                    'id': '--port',
                    'type': int,
                    'nargs': None,
                    'default': 6006,
                    'required': False,
                    'help': 'Port where to run the TensorBoard instance.'
                },
                {
                    'id': '--dev',
                    'type': bool,
                    'nargs': None,
                    'default': None,
                    'required': False,
                    'help': 'Whether to use a TensorBoard.dev instance.'
                },
                {
                    'id': '--compare',
                    'type': bool,
                    'nargs': None,
                    'default': None,
                    'required': False,
                    'help': 'Whether to run TensorBoard on the experiments '
                            'root folder.'
                }
            ]
        }
    ]
}


class Skeltorch:
    """Skeltorch project initializer class.

    ``skeltorch.Skeltorch`` class is responsible for creating the objects
    required to run a project. It also contains the links between command-line
    interfaces and Python commands, which can be used to run both default and
    custom pipelines.

    Attributes:
        execution (skeltorch.Execution): Execution object.
        experiment (skeltorch.Experiment): Experiment object.
        configuration (skeltorch.Configuration): Configuration object.
        data (skeltorch.Data): Data object.
        runner (skeltorch.Runner): Runner object.
        logger (logging.Logger): Logger object.
    """
    execution = None
    experiment = None
    configuration = None
    data = None
    runner = None
    logger = None
    _parser = argparse.ArgumentParser()
    _subparsers = dict()
    _command_handlers = dict()

    def __init__(self, data, runner):
        """``skeltorch.Skeltorch`` constructor.

        Args:
            data (skeltorch.Data): custom data object.
            runner (skeltorch.Runner): custom runner object.
        """
        self._init_logger()
        self.execution = Execution()
        self.configuration = Configuration(self.logger)
        self.data = data
        self.experiment = Experiment(
            self.configuration, self.data, self.logger
        )
        self.runner = runner
        self._init_default_parsers()
        self._init_default_commands()

    def _init_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('skeltorch')

    def _init_default_parsers(self):
        def _add_args(parser, args):
            for arg in args:
                if arg['type'] == bool:
                    parser.add_argument(
                        arg['id'],
                        action='store_true',
                        required=arg['required'],
                        help=arg['help']
                    )
                else:
                    parser.add_argument(
                        arg['id'],
                        type=arg['type'],
                        nargs=arg['nargs'],
                        default=arg['default'],
                        required=arg['required'],
                        help=arg['help']
                    )

        # Create main parser
        self._subparsers['_creator'] = self._parser.add_subparsers(
            dest='command', required=True
        )
        _add_args(self._parser, __cli_commands__['args'])

        # Create secondary parsers
        for command in __cli_commands__['commands']:
            secondary_parser = self.create_parser(command['name'])
            _add_args(secondary_parser, command['args'])

    def _init_default_commands(self):
        __cli_handlers__ = {
            'init': {
                'handler': self.experiment.create,
                'params': [
                    'data_path', 'config_path', 'config_schema_path', 'seed',
                    'verbose'
                ]
            },
            'info': {
                'handler': self.experiment.info,
                'params': []
            },
            'train': {
                'handler': self.runner.train,
                'params': ['epoch', 'max_epochs', 'log_period', 'device']
            },
            'test': {
                'handler': self.runner.test,
                'params': ['epoch', 'device']
            },
            'tensorboard': {
                'handler': self.experiment.run_tensorboard,
                'params': ['port', 'dev', 'compare', 'experiments_path']
            }
        }
        for command in __cli_commands__['commands']:
            self.create_command(
                self._subparsers[command['name']],
                __cli_handlers__[command['name']]['handler'],
                __cli_handlers__[command['name']]['params']
            )

    def get_parser(self, command_name):
        """Gets an already-created parser.

        Returns the parser associated with ``command_name``. You may use this
        parser to add custom arguments or even modify the existing ones.

        Args:
            command_name (str): Name of the command associated to the returned
            parser.

        Returns:
            argparse.ArgumentParser: Argument parser associated to
            `command_name`.
        """
        return self._subparsers[command_name]

    def create_parser(self, command_name):
        """Creates and returns a new parser.

        Creates a new parser associated with ``command_name``. You can this
        parser to add custom arguments and customize the behavior of your
        pipeline.

        Args:
            command_name (str): Name of the command associated with the
            returned parser.

        Returns:
            argparse.ArgumentParser: Argument parser associated to
            `command_name`.
        """
        self._subparsers[command_name] = \
            self._subparsers['_creator'].add_parser(name=command_name)
        return self._subparsers[command_name]

    def create_command(
            self, command_parser, command_handler, command_args_keys
    ):
        """Creates a new command-method association.

        Associates ``command_parser`` with ``command_handler``. Every time that
        the command of ``command_parser`` is executed, ``command_handler`` will
        be run with the parameters given in ``command_args_keys``.

        Args:
            command_parser (argparse.ArgumentParser): Argument parser which
            executes ``command_handler``.
            command_handler (Callable): Method to run when calling
            ``command_name``.
            command_args_keys (list): List containing the names of the command
            arguments passed to ``command_handler``.
        """
        command_name = list(self._subparsers.keys())[
            list(self._subparsers.values()).index(command_parser)
        ]
        self._command_handlers[command_name] = (
            command_handler, command_args_keys
        )

    def run(self):
        """Runs a Skeltorch project."""

        # Loading of the skeltorch.Execution instance
        self.execution.load(self._parser.parse_args())

        # Initialization of the skeltorch.Experiment instance
        self.experiment.init(
            self.execution.args['experiment_name'],
            self.execution.args['experiments_path']
        )

        # Initialization of the skeltorch.Data instance
        self.data.init(self.experiment, self.logger)

        # Conditional loading of the skeltorch.Experiment instance
        if self.execution.command not in ['init']:
            self.experiment.load(
                data_path=self.execution.args['data_path'],
                num_workers=self.execution.args['num_workers'] if
                'num_workers' in self.execution.args else 1,
                verbose=self.execution.args['verbose']
            )

        # Conditional initialization of the skeltorch.Runner instance
        if self.execution.command not in ['init', 'tensorboard', 'info']:
            self.runner.init(
                self.experiment, self.logger, self.execution.args['device']
            )

        # Retrieve command handler and its parameters
        command_handler, command_params = self._command_handlers[
            self.execution.command
        ]
        command_params = {
            param: self.execution.args[param] for param in command_params
        }
        if command_handler == self.runner.test:
            # Compatibility purposes until release 2.0
            command_params['device'] = command_params['device'][0]
        command_handler(**command_params)
