import json
import jsonschema
import pickle


class Configuration:
    """Skeltorch configuration class.

    Configuration parameters given in the configuration file are stored as
    class attributes dynamically. The class provides an easy-to-use interface
    to get them without having to handle file parsing.

    Read our tutorial *"Dealing with configuration files"* to get details about
    how to manage configuration files inside your project.

    Attributes:
        logger (logging.Logger): Logger object.
    """
    seed = None
    logger = None

    def __init__(self, logger):
        """``skeltorch.Configuration`` constructor.

        Args:
            logger (logging.Logger): Logger object.
        """
        self.logger = logger

    def create(self, config_path, config_schema_path=None):
        """Loads (and validates) a raw configuration file.

        Uses the arguments ``--config-path`` and ``--config-schema-path`` of
        the ``init`` pipeline to load a raw configuration file and, if
        provided, validate it with the schema. Only called during the creation
        of a new experiment.

        Args:
            config_path (str): ``--config-path`` command argument.
            config_schema_path (str or None): ``--config-schema-path`` command
            argument.

        Raises:
            json.decoder.JSONDecodeError: Raised when the format of one of the
            `.json` files is not valid.
            jsonschema.exceptions.ValidationError: Raised when the
            configuration file does not match the schema.
        """
        if not config_schema_path:
            self.logger.warning(
                'Configuration schema not provided. The configuration file '
                'will not be validated.'
            )
        with open(config_path, 'r') as config_file:
            config_content = json.load(config_file)
        if config_schema_path:
            with open(config_schema_path, 'r') as schema_file:
                schema_content = json.load(schema_file)
                jsonschema.validate(config_content, schema_content)
        for config_cat, config_cat_items in config_content.items():
            setattr(self, config_cat, dict())
            for config_param, config_value in config_cat_items.items():
                self.set(config_cat, config_param, config_value)

    def save(self, config_file_path):
        """Saves class attributes inside a binary file stored in
        ``config_file_path``.

        Args:
            config_file_path (str): Path where the binary file will be stored.
        """
        with open(config_file_path, 'wb') as config_file:
            config = dict()
            for attr, value in self.__dict__.items():
                config[attr] = value
            pickle.dump(config, config_file)

    def load(self, config_file_path):
        """Loads class attributes from the binary file stored in
        ``config_file_path``.

        Args:
            config_file_path (str): Path where the binary file is stored.
        """
        with open(config_file_path, 'rb') as config_file:
            config = pickle.load(config_file)
            for attr, value in config.items():
                setattr(self, attr, value)

    def get(self, config_cat, config_param):
        """Gets the parameter named ``config_param`` of the category
        ``config_cat``.

        Args:
            config_cat (str): Category of the configuration parameter.
            config_param (str): Identifier of the configuration parameter.

        Return:
            any: Retrieved configuration value.
        """
        try:
            return getattr(self, config_cat)[config_param]
        except KeyError:
            return None

    def set(self, config_cat, config_param, config_value):
        """Sets the parameter named ``config_param`` inside the category
        ``config_cat`` with the value ``config_value``.

        Args:
            config_cat (str): Category of the configuration parameter.
            config_param (str): Identifier of the configuration parameter.
            config_value (any): Configuration value to set.
        """
        getattr(self, config_cat)[config_param] = config_value
