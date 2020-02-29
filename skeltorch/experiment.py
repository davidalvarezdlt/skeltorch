import logging
import os
import re
import torch
import tensorboardX
import random
import numpy as np


class Experiment:
    """Skeltorch execution class.

    An experiment object stores the information related to an experiment and is in charge of handling the files
    belonging to it. It provides an easy interface to create and load experiments, log project information and restore
    object states.

    Read our tutorial *"File structure of an experiment"* to get details about the files and folders stored inside an
    experiment.

    Attributes:
        configuration (skeltorch.Configuration): Configuration object.
        data (skeltorhc.Data): Data object of the experiment.
        logger (logging.Logger): Logger object.
        tbx (tensorboardX.SummaryWriter): TensorBoard logging helper.
        experiment_name (str): Name of the experiment.
        paths (dict): Dictionary containing absolute paths to the files and folders of the experiment.
    """
    configuration = None
    data = None
    logger = None
    tbx = None
    experiment_name = None
    paths = dict()

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

    def init(self, experiment_name, experiments_path):
        """Lazy-loading of ``skeltorch.Experiment`` attributes.

        Args:
            experiment_name (str): ``--experiment-name`` command argument.
            experiments_path (str): ``--experiments-path`` command argument.
        """
        self.experiment_name = experiment_name
        self.paths['experiment'] = os.path.join(experiments_path, experiment_name)
        self.paths['checkpoints'] = os.path.join(self.paths['experiment'], 'checkpoints')
        self.paths['results'] = os.path.join(self.paths['experiment'], 'results')
        self.paths['tensorboard'] = os.path.join(self.paths['experiment'], 'tensorboard')
        self.paths['configuration'] = os.path.join(self.paths['experiment'], 'config.pkl')
        self.paths['data'] = os.path.join(self.paths['experiment'], 'data.pkl')
        self.paths['log'] = os.path.join(self.paths['experiment'], 'verbose.log')

    def load(self, data_path, num_workers, verbose):
        """Loads the experiment named ``experiment_name``.

        Loads the experiment and its dependencies, that is, its ``skeltorch.Configuration`` and ``skeltorch.Data``
        objects. It also initializes the loggers and restores the seed established during the creation of the
        experiment.

        Args:
            data_path (str): ``--data-path`` command argument.
            num_workers (int): ``--num-workers`` command argument.
            verbose (bool): ``--verbose`` command argument.
        """
        self.configuration.load(self.paths['configuration'])
        self.data.init(self, self.logger)
        self.data.load(data_path, self.paths['data'], num_workers)
        self.logger.addHandler(logging.FileHandler(self.paths['log']))
        self.logger.propagate = verbose
        self.tbx = tensorboardX.SummaryWriter(self.paths['tensorboard'], flush_secs=10)
        random.seed(self.configuration.get('_others', 'seed'))
        np.random.seed(self.configuration.get('_others', 'seed'))
        torch.manual_seed(self.configuration.get('_others', 'seed'))
        torch.cuda.manual_seed_all(self.configuration.get('_others', 'seed'))
        self.logger.info('Experiment {} loaded successfully.'.format(self.experiment_name))

    def create(self, data_path, config_path, config_schema_path, seed):
        """Creates an experiment named ``experiment_name``.

        Creates the experiment and its associated files and folders. It also creates and saves its dependencies, that
        is, its ``skeltorch.Configuration`` and ``skeltorch.Data`` objects.

        Args:
            data_path (str): ``--data-path`` command argument.
            config_path (str): ``--config-path`` command argument.
            config_schema_path (str): ``--config-schema-path`` command argument.
            seed (int): ``--seed`` command argument.
        """
        self.configuration.create(config_path, config_schema_path)
        self.configuration.set('_others', 'seed', seed)
        self.data.init(self, self.logger)
        self.data.create(data_path)
        os.makedirs(self.paths['experiment'])
        os.makedirs(self.paths['checkpoints'])
        os.makedirs(self.paths['tensorboard'])
        os.makedirs(self.paths['results'])
        open(self.paths['log'], 'a').close()
        self.configuration.save(self.paths['configuration'])
        self.data.save(self.paths['data'])
        self.logger.info('Experiment {} created successfully.'.format(self.experiment_name))

    def checkpoints_get(self):
        """Returns a list with available checkpoints.

        Returns:
            list: List containing the epochs with available checkpoint.
        """
        available_checkpoints = []
        for checkpoint_file in os.listdir(os.path.join(self.paths['checkpoints'])):
            modeling_regex = re.search(r'(\d+).checkpoint.pkl', checkpoint_file)
            if modeling_regex:
                available_checkpoints.append(int(modeling_regex.group(1)))
        return sorted(available_checkpoints)

    def checkpoint_load(self, epoch, device):
        """Loads the checkpoint associated to ``epoch``.

        Args:
            epoch (int): Epoch number used to identify the checkpoint.
            device (str): Device where the checkpoint is loaded.

        Returns:
            dict: Dictionary containing the states stored inside the checkpoint.
        """
        with open(os.path.join(self.paths['checkpoints'], '{}.checkpoint.pkl'.format(epoch)), 'rb') as \
                checkpoint_file:
            self.logger.info('Checkpoint of epoch {} restored.'.format(epoch))
            return torch.load(checkpoint_file, map_location=device)

    def checkpoint_save(self, checkpoint_data, epoch):
        """Saves a checkpoint associated to ``epoch``.

        Args:
            checkpoint_data (dict): Dictionary containing the states to store inside the checkpoint.
            epoch (int): Epoch number used to identify the checkpoint.
        """
        with open(os.path.join(self.paths['checkpoints'], '{}.checkpoint.pkl'.format(epoch)), 'wb') as \
                checkpoint_file:
            torch.save(checkpoint_data, checkpoint_file)
            self.logger.info('Checkpoint of epoch {} saved.'.format(epoch))
