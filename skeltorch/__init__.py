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
    _commandHandlers = dict()

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
        self._subparsers['_creator'] = self._parser.add_subparsers(
            dest='command', required=True
        )
        self._parser.add_argument(
            '--experiment-name', required=True, help='Name of the experiment'
        )
        self._parser.add_argument(
            '--base-path',
            default=os.path.dirname(os.path.dirname(sys.argv[0])),
            help='Base path from which other default paths are referenced'
        )
        self._parser.add_argument(
            '--experiments-path',
            help='Experiments path'
        )
        self._parser.add_argument('--data-path', help='Data path')
        self._parser.add_argument(
            '--verbose',
            action='store_true',
            help='Log to standard output'
        )

        # init parser
        init_subparser = self.create_parser('init')
        init_subparser.add_argument(
            '--config-path', required=True, help='Configuration file path'
        )
        init_subparser.add_argument(
            '--config-schema-path',
            default=None,
            help='Configuration schema file path'
        )
        init_subparser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Seed for random value generators'
        )

        # info subparser
        self.create_parser('info')

        # train parser
        train_subparser = self.create_parser('train')
        train_subparser.add_argument(
            '--epoch',
            type=int,
            default=None,
            help='Starting epoch from which continue the training'
        )
        train_subparser.add_argument(
            '--max-epochs',
            type=int,
            default=999,
            help='Maximum number of epochs to perform'
        )
        train_subparser.add_argument(
            '--log-period',
            type=int,
            default=100,
            help='Number of iterations between logs'
        )
        train_subparser.add_argument(
            '--num-workers',
            type=int,
            default=1,
            help='Number of DataLoader workers'
        )
        train_subparser.add_argument(
            '--device',
            nargs='+',
            default=None,
            help='PyTorch-friendly device names'
        )

        test_subparser = self.create_parser('test')
        test_subparser.add_argument(
            '--epoch',
            type=int,
            required=True,
            help='Epoch from which run the test'
        )
        test_subparser.add_argument(
            '--num-workers',
            type=int,
            default=1,
            help='Number of DataLoader workers'
        )
        test_subparser.add_argument(
            '--device',
            nargs='+',
            default=None,
            help='PyTorch-friendly device names'
        )

        tensorboard_subparser = self.create_parser('tensorboard')
        tensorboard_subparser.add_argument(
            '--port', default=6006, help='Port where to run TensorBoard'
        )
        tensorboard_subparser.add_argument(
            '--dev', action='store_true', help='Use TensorBoard.dev'
        )
        tensorboard_subparser.add_argument(
            '--compare',
            action='store_true',
            help='Run TensorBoard on all experiments'
        )

    def _init_default_commands(self):
        self.create_command(
            self._subparsers['init'],
            self.experiment.create,
            ['data_path', 'config_path', 'config_schema_path', 'seed',
             'verbose']
        )
        self.create_command(
            self._subparsers['info'],
            self.experiment.info,
            []
        )
        self.create_command(
            self._subparsers['train'],
            self.runner.train,
            ['epoch', 'max_epochs', 'log_period', 'device']
        )
        self.create_command(
            self._subparsers['test'],
            self.runner.test,
            ['epoch', 'device']
        )
        self.create_command(
            self._subparsers['tensorboard'],
            self.experiment.run_tensorboard,
            ['port', 'dev', 'compare', 'experiments_path']
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
        self._subparsers[command_name] = self._subparsers[
            '_creator'].add_parser(name=command_name)
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
        self._commandHandlers[command_name] = (
            command_handler, command_args_keys
        )

    def run(self):
        """Runs a Skeltorch project."""
        print(sys.argv)
        self.execution.load(self._parser.parse_args())
        self.experiment.init(
            self.execution.args['experiment_name'],
            self.execution.args['experiments_path']
        )
        self.data.init(self.experiment, self.logger)
        if self.execution.command not in ['init']:
            self.experiment.load(
                data_path=self.execution.args['data_path'],
                num_workers=self.execution.args['num_workers'] if
                'num_workers' in self.execution.args else 1,
                verbose=self.execution.args['verbose']
            )
        if self.execution.command not in ['init', 'tensorboard', 'info']:
            self.runner.init(
                self.experiment, self.logger, self.execution.args['device']
            )
        command_handler, command_args_keys = self._commandHandlers[
            self.execution.command
        ]
        commands_args = {
            command_arg_key: self.execution.args[command_arg_key]
            for command_arg_key in command_args_keys
        }
        # Compatibility purposes until next release
        if command_handler == self.runner.test:
            commands_args['device'] = commands_args['device'][0]
        command_handler(**commands_args)
