import logging
import numpy as np
import os
import random
import re
import torch
import tensorboardX


class Experiment:
    """Skeltorch experiment class.

    An experiment object stores the information related to an experiment and is
    in charge of handling the files belonging to it. It provides an easy
    interface to create and load experiments, log project information and
    restore object states.

    Read our tutorial *"File structure of an experiment"* to get details about
    the files and folders stored inside an experiment.

    Attributes:
        configuration (skeltorch.Configuration): Configuration object.
        data (skeltorch.Data): Data object of the experiment.
        logger (logging.Logger): Logger object.
        tbx (tensorboardX.SummaryWriter): TensorBoard logging helper.
        experiment_name (str): Name of the experiment.
        paths (dict): Dictionary containing absolute paths to the files and
            folders of the experiment.
    """

    def __init__(self, configuration, data, logger):
        """``skeltorch.Experiment`` constructor.

        Args:
            configuration (skeltorch.Configuration): Configuration object.
            data (skeltorch.Data): Data object.
            logger (logging.Logger): Logger object.
        """
        self.configuration = configuration
        self.data = data
        self.logger = logger
        self.tbx = None
        self.experiment_name = None
        self.paths = dict()

    def init(self, experiment_name, experiments_path):
        """Lazy-loading of ``skeltorch.Experiment`` attributes.

        Args:
            experiment_name (str): ``--experiment-name`` command argument.
            experiments_path (str): ``--experiments-path`` command argument.
        """
        self.experiment_name = experiment_name
        self.paths['experiment'] = os.path.join(
            experiments_path, experiment_name
        )
        self.paths['checkpoints'] = os.path.join(
            self.paths['experiment'], 'checkpoints'
        )
        self.paths['results'] = os.path.join(
            self.paths['experiment'], 'results'
        )
        self.paths['tensorboard'] = os.path.join(
            self.paths['experiment'], 'tensorboard'
        )
        self.paths['configuration'] = os.path.join(
            self.paths['experiment'], 'config.pkl'
        )
        self.paths['data'] = os.path.join(
            self.paths['experiment'], 'data.pkl'
        )
        self.paths['log'] = os.path.join(
            self.paths['experiment'], 'verbose.log'
        )

    def _init_loggers(self, verbose):
        logger_handler = logging.FileHandler(self.paths['log'])
        logger_handler.setFormatter(self.logger.parent.handlers[0].formatter)
        self.logger.addHandler(logger_handler)
        self.logger.propagate = verbose
        self.tbx = tensorboardX.SummaryWriter(
            self.paths['tensorboard'], flush_secs=10
        )

    def _init_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load(self, data_path, num_workers, verbose):
        """Loads the experiment named ``experiment_name``.

        Loads the experiment and its dependencies, that is, its
        ``skeltorch.Configuration`` and ``skeltorch.Data`` objects. It also
        initializes the loggers.

        Args:
            data_path (str): ``--data-path`` command argument.
            num_workers (int): ``--num-workers`` command argument.
        """
        self._init_loggers(verbose)
        self.configuration.load(self.paths['configuration'])
        self._init_seed(self.configuration.seed)
        self.data.load(data_path, self.paths['data'], num_workers)
        self.logger.info('Experiment "{}" loaded successfully.'.format(
            self.experiment_name
        ))

    def create(self, data_path, config_path, config_schema_path, seed,
               verbose):
        """Creates an experiment named ``experiment_name``.

        Creates the experiment and its associated files and folders. It also
        creates and saves its dependencies, that is, its
        ``skeltorch.Configuration`` and ``skeltorch.Data`` objects.

        Args:
            data_path (str): ``--data-path`` command argument.
            config_path (str): ``--config-path`` command argument.
            config_schema_path (str): ``--config-schema-path`` command
                argument.
            seed (int): ``--seed`` command argument.
            verbose (bool): ``--verbose`` command argument.
        """
        os.makedirs(self.paths['experiment'])
        os.makedirs(self.paths['checkpoints'])
        os.makedirs(self.paths['tensorboard'])
        os.makedirs(self.paths['results'])
        open(self.paths['log'], 'a').close()
        self._init_loggers(verbose)
        self._init_seed(seed)
        self.configuration.create(config_path, config_schema_path)
        self.configuration.seed = seed
        self.configuration.save(self.paths['configuration'])
        self.data.create(data_path)
        self.data.save(self.paths['data'])
        self.logger.info('Experiment "{}" created successfully.'.format(
            self.experiment_name
        ))

    def checkpoints_get(self, get_releases=False):
        """Returns a list with available checkpoints.

        Args:
            get_releases (bool): Whether to get the list of release
                checkpoints.

        Returns:
            list: List containing the epochs with available checkpoint.
        """
        checkpoints = []
        reg_pattern = r'(\d+).checkpoint.release.pkl' if get_releases \
            else r'(\d+).checkpoint.pkl'
        for checkpoint_file in os.listdir(self.paths['checkpoints']):
            checkpoint_item = re.search(reg_pattern, checkpoint_file)
            if checkpoint_item:
                checkpoints.append(int(checkpoint_item.group(1)))
        return sorted(checkpoints)

    def checkpoint_load(self, epoch, device, is_release=False):
        """Loads the checkpoint associated to ``epoch``.

        Args:
            epoch (int): Epoch number used to identify the checkpoint.
            device (str): Device where the checkpoint is loaded.
            is_release (bool): Whether or not the epoch is a release.

        Returns:
            dict: Dictionary containing the states stored inside the
                checkpoint.
        """
        checkpoint_path = os.path.join(
            self.paths['checkpoints'], '{}.checkpoint{}.pkl'.format(
                epoch, '.release' if is_release else ''
            )
        )
        with open(checkpoint_path, 'rb') as checkpoint_file:
            self.logger.info('Checkpoint ({}) of epoch {} loaded.'.format(
                'release' if is_release else 'standard', epoch
            ))
            return torch.load(checkpoint_file, map_location=device)

    def checkpoint_save(self, checkpoint_data, epoch, is_release=False):
        """Saves a checkpoint associated to ``epoch``.

        Args:
            checkpoint_data (dict): Dictionary containing the states to store
                inside the checkpoint.
            epoch (int): Epoch number used to identify the checkpoint.
            is_release (bool): Whether or not the epoch is a release.
        """
        checkpoint_path = os.path.join(
            self.paths['checkpoints'], '{}.checkpoint{}.pkl'.format(
                epoch, '.release' if is_release else ''
            )
        )
        with open(checkpoint_path, 'wb') as checkpoint_file:
            torch.save(checkpoint_data, checkpoint_file)
            self.logger.info('Checkpoint ({}) of epoch {} saved.'.format(
                'release' if is_release else 'standard', epoch
            ))

    def info(self):
        print(
            'Information about experiment "{}"'.format(self.experiment_name)
        )
        print('=' * len('Information about experiment "{}"'.format(
            self.experiment_name
        )))

        # Logging of configuration fields
        print('Configuration:')
        for config_cat, config_cat_data in \
                sorted(self.configuration.__dict__.items()):
            if type(config_cat_data) != dict:
                continue
            print('\t{}'.format(config_cat))
            for config_param, config_val in config_cat_data.items():
                print('\t\t{}: {}'.format(config_param, config_val))

        # Logging of experiment checkpoints
        for c_type in ['standard', 'release']:
            if len(self.checkpoints_get()) > 0:
                print('Available {} checkpoints:'.format(c_type))
                print('\t' + ', '.join([str(c) for c in self.checkpoints_get(
                    get_releases=c_type == 'release'
                )]))
            else:
                print('No {} checkpoints available'.format(c_type))

    def create_release(self, epoch):
        """Creates a release checkpoint of a certain epoch.

        A release checkpoint only contains the data associated with the model.

        Args:
            epoch (int): Epoch from which to create a release checkpoint.
        """
        if epoch not in self.checkpoints_get():
            self.logger.error('Epoch {} does not exist.'.format(epoch))
            exit()
        checkpoint_data = self.checkpoint_load(epoch, 'cpu')
        self.checkpoint_save(
            {'model': checkpoint_data['model']}, epoch, is_release=True
        )

    def run_tensorboard(self, port, dev, compare, experiments_path):
        """Runs TensorBoard using experiment files.

        Args:
            port (int): ``--port`` command argument.
            dev (bool): ``--dev`` command argument.
            compare (bool): ``--compare`` command argument.
            experiments_path (str): ``--experiments-path`` command argument.
        """
        tensorboard_dir = self.paths['tensorboard'] if not compare \
            else experiments_path
        if dev:
            os.system('tensorboard dev upload --logdir {} --name "{}"'.format(
                tensorboard_dir, self.experiment_name
            ))
        else:
            os.system('tensorboard --port {} --logdir {}'.format(
                port, tensorboard_dir
            ))
