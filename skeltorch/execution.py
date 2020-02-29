import os
import torch
import logging


class Execution:
    """Skeltorch execution class.

    An execution object stores information related to the execution of a command. It includes not only which command has
    been called, but also its arguments and other auxiliary information.

    Arguments are automatically passed as function parameters if specified inside ``command_args_keys`` of
    ``create_command()``. Read our tutorial *"Creating custom pipelines"* to get details about the exact procedure.

    Attributes:
        logger (logging.Logger): Logger object.
        command (str): Name of the executed command.
        args (dict): Dictionary containing the arguments of the execution.
    """
    logger = None
    command = None
    args = None

    def __init__(self, logger):
        """``skeltorch.Execution`` constructor."""
        self.logger = logger

    def load(self, args):
        """Loads the executed command and its arguments inside the execution.

        It also sets the default values of ``--experiments-path`` and ``--data-path`` if the user has not provided them
        manually.

        Args:
            args (argparse.Namespace): Arguments of the execution in raw format.

        Raises:
            ValueError: Raised when one of the argument is not valid.
        """
        self.command = args.command
        self.args = vars(args)
        self.args.pop('command')
        self._load_default_paths()
        self._validate()

    def _load_default_paths(self):
        if self.args['experiments_path'] is None:
            self.args['experiments_path'] = os.path.join(self.args['base_path'], 'experiments')
        if self.args['data_path'] is None:
            self.args['data_path'] = os.path.join(self.args['base_path'], 'data')

    def _validate(self):
        self._validate_main_args()
        if self.command == 'init':
            self._validate_init_args()
        else:
            self._validate_pipelines_args()

    def _validate_main_args(self):
        if not os.path.exists(self.args['experiments_path']):
            raise ValueError('Experiments path does not exist.')
        if not os.path.exists(self.args['data_path']):
            raise ValueError('Data path does not exist.')
        if 'device' in self.args and self.args['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but not available.')

    def _validate_init_args(self):
        if os.path.exists(os.path.join(self.args['experiments_path'], self.args['experiment_name'])):
            self.logger.error('An experiment with name "{}" already exists.'.format(self.args['experiment_name']))
            exit()
        if not os.path.exists(self.args['config_path']):
            self.logger.error('Configuration file path is not correct')
            exit()
        if not self.args['config_schema_path'] or not os.path.exists(self.args['config_schema_path']):
            self.logger.warning('Configuration schema file path is not correct. Configuration will not be validated.')

    def _validate_pipelines_args(self):
        if not os.path.exists(os.path.join(self.args['experiments_path'], self.args['experiment_name'])):
            self.logger.error('Experiment with name "{}" does not exist.'.format(self.args['experiment_name']))
            exit()
