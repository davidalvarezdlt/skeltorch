import numpy as np
import os
import torch
import random
import re


class Execution:
    """Skeltorch execution class.

    An execution object stores information related to the execution of a command. It includes not only which command has
    been called, but also its arguments and other auxiliary information.

    Arguments are automatically passed as function parameters if specified inside ``command_args_keys`` of
    ``create_command()``. Read our tutorial *"Creating custom pipelines"* to get details about the exact procedure.

    Attributes:
        command (str): Name of the executed command.
        args (dict): Dictionary containing the arguments of the execution.
    """
    command = None
    args = None

    def __init__(self):
        """``skeltorch.Execution`` constructor."""
        pass

    def load(self, args):
        """Loads the executed command and its arguments inside the execution.

        It also sets the default values of ``--experiments-path`` and ``--data-path`` if the user has not provided them
        manually.

        Args:
            args (argparse.Namespace): Arguments of the execution in raw format.
        """
        self.command = args.command
        self.args = vars(args)
        self.args.pop('command')
        self._load_default_args()
        self._load_seed()
        self._validate()

    def _load_default_args(self):
        if self.args['experiments_path'] is None:
            self.args['experiments_path'] = os.path.join(self.args['base_path'], 'experiments')
        if self.args['data_path'] is None:
            self.args['data_path'] = os.path.join(self.args['base_path'], 'data')
        if 'device' in self.args and self.args['device'] is None:
            self.args['device'] = ['cuda'] if torch.cuda.is_available() else ['cpu']
        elif 'device' in self.args:
            self.args['device'] = sorted(self.args['device'])

    def _load_seed(self):
        if 'seed' in self.args:
            random.seed(self.args['seed'])
            np.random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])
            torch.cuda.manual_seed_all(self.args['seed'])

    def _validate(self):
        self._validate_main_args()
        if self.command == 'init':
            self._validate_init_args()
        else:
            self._validate_pipelines_args()

    def _validate_main_args(self):
        if not os.path.exists(self.args['experiments_path']):
            exit('Experiments path does not exist.')
        if not os.path.exists(self.args['data_path']):
            exit('Data path does not exist.')
        if 'device' in self.args:
            for device in self.args['device']:
                if not re.match(r'(^cpu$|^cuda$|^cuda:\d+$)', device):
                    exit('Device {} is not valid.'.format(device))
                if re.match(r'^cuda:\d+$', device) and torch.device(device).index > torch.cuda.device_count() - 1:
                    exit('Device {} is not available.'.format(device))
            if len(self.args['device']) != len(set(self.args['device'])):
                exit('Device argument not valid. Duplicated device found.')
            if 'cpu' in self.args['device'] and len(self.args['device']) > 1:
                exit('Invalid choice of devices. You can not mix CPU and GPU devices.')
            if 'cuda' in self.args['device'] and len(self.args['device']) > 1:
                exit('Invalid choice of devices. You must specify device indexes if multiple GPUs are required.')
            if 'cpu' not in self.args['device'] and len(self.args['device']) > torch.cuda.device_count():
                exit('Invalid choice of devices. You requested {} GPUs but only {} is/are available.'
                     .format(len(self.args['device']), torch.cuda.device_count()))

    def _validate_init_args(self):
        if os.path.exists(os.path.join(self.args['experiments_path'], self.args['experiment_name'])):
            exit('An experiment with name "{}" already exists.'.format(self.args['experiment_name']))
        if not os.path.exists(self.args['config_path']):
            exit('Configuration file path is not correct')

    def _validate_pipelines_args(self):
        if not os.path.exists(os.path.join(self.args['experiments_path'], self.args['experiment_name'])):
            exit('Experiment with name "{}" does not exist.'.format(self.args['experiment_name']))
