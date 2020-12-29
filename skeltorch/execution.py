import os
import argparse
import torch
import re
import sys

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
                    'type': str,
                    'nargs': '+',
                    'default': ['cpu'],
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
                    'type': str,
                    'nargs': '+',
                    'default': ['cpu'],
                    'required': False,
                    'help': 'PyTorch-friendly name of the device where the '
                            'model should be stored and trained/tested.'
                }
            ]
        },
        {
            'name': 'test_sample',
            'args': [
                {
                    'id': '--sample',
                    'type': str,
                    'nargs': None,
                    'default': None,
                    'required': True,
                    'help': 'Unique identifier of the sample to test.'
                },
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
                    'type': str,
                    'nargs': '+',
                    'default': ['cpu'],
                    'required': False,
                    'help': 'PyTorch-friendly name of the device where the '
                            'model should be stored and trained/tested.'
                }
            ]
        },
        {
            'name': 'create_release',
            'args': [
                {
                    'id': '--epoch',
                    'type': int,
                    'nargs': None,
                    'default': None,
                    'required': True,
                    'help': 'Epoch used to run the testing.'
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


class Execution:
    """Skeltorch execution class.

    An execution object stores information related to the execution of a
    command. It includes not only which command has been called, but also its
    arguments and other auxiliary information.

    Arguments are automatically passed as function parameters if specified
    inside ``command_args_keys`` of ``create_command()``. Read our tutorial
    *"Creating custom pipelines"* to get details about the exact procedure.

    Attributes:
        command (str): Name of the executed command.
        args (dict): Dictionary containing the arguments of the execution.
    """

    def __init__(self):
        """``skeltorch.Execution`` constructor."""
        self.command = None
        self.args = None
        self.parser = argparse.ArgumentParser()
        self.subparsers = dict()
        self._init_default_parsers()

    def _init_default_parsers(self):
        # Create main parser
        self.subparsers['_creator'] = self.parser.add_subparsers(
            dest='command', required=True
        )

        # Create a function to add arguments to the parsers
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

        # Add the arguments to the different parsers
        _add_args(self.parser, __cli_commands__['args'])
        for command in __cli_commands__['commands']:
            secondary_parser = self.create_parser(command['name'])
            _add_args(secondary_parser, command['args'])

    def load(self):
        """Loads and validates the executed command and its arguments inside
        the execution.

        It also sets the default values of ``--experiments-path`` and
        ``--data-path`` if the user has not provided them manually.
        """
        self.args = vars(self.parser.parse_args())
        self.command = self.args['command']
        self.args.pop('command')
        self._load_default_args()
        self._validate()

    def _load_default_args(self):
        if self.args['experiments_path'] is None:
            self.args['experiments_path'] = os.path.join(
                self.args['base_path'], 'experiments'
            )
        if self.args['data_path'] is None:
            self.args['data_path'] = os.path.join(
                self.args['base_path'], 'data'
            )

    def _validate(self):
        # Validate --experiments-path
        if not os.path.exists(self.args['experiments_path']):
            exit('Experiments path does not exist.')

        # Validate --data-path
        if not os.path.exists(self.args['data_path']):
            exit('Data path does not exist.')

        # Validate --experiment-name
        experiment_path = os.path.join(
            self.args['experiments_path'], self.args['experiment_name']
        )
        if self.command == 'init' and os.path.exists(experiment_path):
            exit('An experiment with name "{}" already exists.'.
                 format(self.args['experiment_name']))

        if self.command != 'init' and not os.path.exists(experiment_path):
            exit('Experiment with name "{}" does not exist.'
                 .format(self.args['experiment_name']))

        # Validate --config-path
        if 'config_path' in self.args and not os.path.exists(
                self.args['config_path']):
            exit('Configuration file path is not correct')

        # Validate --device
        if 'device' not in self.args:
            return

        # Sort devices and that the name of the devices is correct
        self.args['device'] = sorted(self.args['device'])
        for device in self.args['device']:
            if not re.match(r'(^cpu$|^cuda$|^cuda:\d+$)', device):
                exit('Device {} is not valid.'.format(device))
            if re.match(r'^cuda:\d+$', device) and \
                    torch.device(device).index > \
                    torch.cuda.device_count() - 1:
                exit('Device {} is not available.'.format(device))

        # Detect duplicated devices
        if len(self.args['device']) != len(set(self.args['device'])):
            exit('Device argument not valid. Duplicated device found.')

        # Force user to select GPU number if multiple GPUs available
        if 'cuda' in self.args['device'] and len(self.args['device']) > 1:
            exit('Invalid choice of devices. You must specify device'
                 'indexes if multiple GPUs are required.')

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
        return self.subparsers[command_name]

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
        self.subparsers[command_name] = \
            self.subparsers['_creator'].add_parser(name=command_name)
        return self.subparsers[command_name]
