import logging
from skeltorch.experiment import Experiment
from skeltorch.execution import Execution
from skeltorch.configuration import Configuration
from skeltorch.data import Data
from skeltorch.runner import Runner

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
        self._command_handlers = dict()
        self._init_default_commands()

    def _init_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('skeltorch')

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
            'test_sample': {
                'handler': self.runner.test_sample,
                'params': ['sample', 'epoch', 'device']
            },
            'create_release': {
                'handler': self.experiment.create_release,
                'params': ['epoch']
            },
            'tensorboard': {
                'handler': self.experiment.run_tensorboard,
                'params': ['port', 'dev', 'compare', 'experiments_path']
            }
        }
        for command, command_items in __cli_handlers__.items():
            self.create_command(
                command, command_items['handler'], command_items['params']
            )

    def get_parser(self, command):
        """Shortcut to ``self.execution.get_parser()``.

        Args:
            command (str): Name of the command associated with the parser.

        Returns:
            argparse.ArgumentParser: Argument parser associated to `command`.
        """
        return self.execution.get_parser(command)

    def create_parser(self, command):
        """Shortcut to ``self.execution.create_parser()``.

        Args:
            command (str): Name of the command associated with the parser.

        Returns:
            argparse.ArgumentParser: Argument parser associated to `command`.
        """
        return self.execution.create_parser(command)

    def create_command(
            self, command, command_handler, command_args_keys
    ):
        """Creates a new command-method association.

        Associates ``command`` with ``command_handler``. Every time that
        ``command`` is executed, ``command_handler`` will be run with the
        parameters given in ``command_args_keys``.

        Args:
            command (str): Name of the command associated with the handler.
            command_handler (Callable): Method to run when calling ``command``.
            command_args_keys (list): List containing the names of the command
                arguments passed to ``command_handler``.
        """
        self._command_handlers[command] = (command_handler, command_args_keys)

    def run(self):
        """Runs a Skeltorch project."""

        # Loading of the skeltorch.Execution instance
        self.execution.load()

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
        if self.execution.command not in [
            'init', 'info', 'create_release', 'tensorboard'
        ]:
            self.runner.init(
                self.experiment, self.logger, self.execution.args['device']
            )

        # Retrieve command handler and its parameters
        command_handler, command_params = \
            self._command_handlers[self.execution.command]
        command_params = {
            param: self.execution.args[param] for param in command_params
        }
        command_handler(**command_params)

        # Flush TensorBoard data
        if self.execution.command not in ['init']:
            self.experiment.tbx.flush()
