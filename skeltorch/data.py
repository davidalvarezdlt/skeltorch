import pickle


class Data:
    """Skeltorch data class.

    Class used to store data-related information such as path references, data
    features or even raw data. Used to provide a transparent bridge between the
    file system and the pipelines.

    **You are required to extend this class and implement its abstract
    methods**. Check out examples to find real implementations of
    ``skeltorch.Data`` classes.

    Attributes:
        experiment (skeltorch.Experiment): Experiment object.
        logger (logging.Logger): Logger object.
        datasets (dict): Dictionary containing the datasets of the train,
            validation and test splits. To be loaded using ``load_datasets()``.
        loaders (dict): Dictionary containing the loaders of the train,
            validation and test splits. To be loaded using ``load_loaders()``.
    """
    _dont_save_atts = {
        '_dont_save_atts',
        '__dict__',
        '__module__',
        '__doc__',
        '__weakref__',
        'datasets',
        'experiment',
        'loaders',
        'logger',
    }

    def __init__(self):
        """``skeltorch.Data`` constructor."""
        self.experiment = None
        self.logger = None
        self.datasets = {'train': None, 'validation': None, 'test': None}
        self.loaders = {'train': None, 'validation': None, 'test': None}

    def init(self, experiment, logger):
        """Lazy-loading of ``skeltorch.Data`` attributes.

        Args:
            experiment (skeltorch.Experiment): Experiment object.
            logger (logging.Logger): Logger object.
        """
        self.experiment = experiment
        self.logger = logger

    def get_conf(self, config_cat, config_param):
        """Shortcut to ``self.experiment.configuration.get()``.

        Args:
            config_cat (str): Category of the configuration parameter.
            config_param (str): Identifier of the configuration parameter.

        Return:
            any: Retrieved configuration value.
        """
        return self.experiment.configuration.get(config_cat, config_param)

    def create(self, data_path):
        """Initializes data-related attributes required in the experiment.

        The purpose of this method is to create all data-related parameters
        which may take some time or that should be unique inside an experiment.
        Called during the creation of a new experiment.

        Some examples of these type of tasks are:

        - Given a set of data samples, create appropriate splits.
        - Compute the mean and standard deviation of a set of data to normalize
          it.
        - Compute features of the data whose computation time would be too
          expensive if done on every iteration.

        To preserve data, you must store it as a class attribute. It will be
        automatically saved using the ``save()`` method during the execution of
        the ``init`` pipeline.

        Args:
            data_path (str): ``--data-path`` command argument.
        """
        raise NotImplementedError

    def save(self, data_file_path):
        """Saves class attributes inside a binary file stored in
        ``data_file_path``.

        Args:
            data_file_path (str): Path where the binary file will be stored.
        """
        with open(data_file_path, 'wb') as data_file:
            data = dict()
            attrs_list = [
                att for att in dir(self)
                if not callable(self.__getattribute__(att))
                and att not in self._dont_save_atts
            ]
            for att in attrs_list:
                data[att] = self.__getattribute__(att)
            pickle.dump(data, data_file)

    def load(self, data_path, data_file_path, num_workers):
        """Loads class attributes from the binary file stored in
        ``data_file_path``.

        Args:
            data_path (str): ``--data-path`` command argument.
            data_file_path (str): Path where the binary file is stored.
            num_workers (int): Number of workers to use in the loaders.
        """
        with open(data_file_path, 'rb') as data_file:
            data = pickle.load(data_file)
            for attr, value in data.items():
                setattr(self, attr, value)
        self.load_datasets(data_path)
        self.load_loaders(data_path, num_workers)

    def load_datasets(self, data_path):
        """Loads the attribute ``self.datasets``.

        Creates and stores inside ``self.datasets`` the
        ``torch.utils.data.Dataset`` objects of the project.

        Args:
            data_path (str): ``--data-path`` parameter.
        """
        raise NotImplementedError

    def load_loaders(self, data_path, num_workers):
        """Loads the attribute ``self.loaders``.

        Creates and stores inside ``self.datasets`` the
        ``torch.utils.data.DataLoader`` objects of the project.

        Args:
            data_path (str): ``--data-path`` command argument.
            num_workers (int): Number of workers to use in the loaders.
        """
        raise NotImplementedError
