import numpy as np
import random
import torch
import torch.nn
import torch.optim
import torch.utils.data


class Runner:
    """Skeltorch runner class.

    The runner object stores the logic associated with both default and
    user-implemented pipelines. It is in charge of andling the flow of the data
    since it leaves the loader until the final result is obtained, where the
    final result depends on which data pipeline is executed.

    **You are required to extend this class and implement its abstract
    methods**. Check out examples to find real implementations of
    ``skeltorch.Runner`` classes.

    Attributes:
        experiment (skeltorch.Experiment): Experiment object.
        logger (logging.Logger): Logger object.
        model (torch.nn.Module): Model object.
        optimizer (torch.optim.Optimizer): Optimizer object.
        counters (dict): Counters of the training iterations, validation
        iterations and epochs.
        losses_it (dict): Iteration losses of both training and validation
        splits.
        losses_epoch (dict): Epoch losses of both training and validation
        splits.
    """
    experiment = None
    logger = None
    model = None
    optimizer = None
    counters = {'epoch': 0, 'train_it': 0, 'validation_it': 0}
    losses_it = {'train': {}, 'validation': {}}
    losses_epoch = {'train': {}, 'validation': {}}

    def __init__(self):
        """``skeltorch.Runner`` constructor."""
        pass

    def init(self, experiment, logger, device):
        """Lazy-loading of ``skeltorch.Runner`` attributes.

        Args:
            experiment (skeltorch.Experiment): Experiment object.
            logger (logging.Logger): Logger object.
            device (list): ``--device`` command argument.
        """
        self.experiment = experiment
        self.logger = logger
        self.logger.info('Device(s) used in the execution: {}'.format(device))
        self.init_model(device[0])
        self.init_optimizer(device[0])
        self.init_others(device[0])
        if len(device) > 1:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=[torch.device(d).index for d in device]
            )

    def init_model(self, device):
        """Initializes the model used in the project.

        Creates and stores inside ``self.model`` the model to be used in the
        project. Use ``device`` to move the model to the proper device.

        Args:
            device (str): first ordered element of ``--device`` command
            argument.
        """
        raise NotImplementedError

    def init_optimizer(self, device):
        """Initializes the optimizer used in the project.

        Creates and stores inside ``self.optimizer`` the optimizer to be used
        in the project. Use ``device`` to move the optimizer to the proper
        device, if required.

        Args:
            device (str): first ordered element of ``--device`` command
            argument.
        """
        raise NotImplementedError

    def init_others(self, device):
        """Initializes other objects used in the project.

        Creates and stores other objects inside class attributes that may be
        required in the project. use ``device`` to move the objects to the
        proper device, if required.

        Args:
            device (str): first ordered element of ``--device`` command
            argument.
        """

    def load_states(self, epoch, device):
        """Loads the states from the checkpoint associated with ``epoch``.

        Args:
            epoch (int): ``--epoch`` command argument.
            device (str): ``--device`` command argument.
        """
        checkpoint_data = self.experiment.checkpoint_load(epoch, device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint_data['model'])
        else:
            self.model.load_state_dict(checkpoint_data['model'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer'])
        random.setstate(checkpoint_data['random_states'][0])
        np.random.set_state(checkpoint_data['random_states'][1])
        torch.set_rng_state(checkpoint_data['random_states'][2].cpu())
        if torch.cuda.is_available() and checkpoint_data['random_states'][3]:
            torch.cuda.set_rng_state(checkpoint_data['random_states'][3].cpu())
        self.counters = checkpoint_data['counters']
        # Compatibility purposes until next release
        if 'losses' in checkpoint_data:
            self.losses_epoch = checkpoint_data['losses']
        else:
            self.losses_epoch = checkpoint_data['losses_epoch']
            self.losses_it = checkpoint_data['losses_it']
        self.load_states_others(checkpoint_data)

    def load_states_others(self, checkpoint_data):
        """Loads the states of other objects from the checkpoint associated
        with ``epoch``.

        Args:
            checkpoint_data (dict): Dictionary with the states of both default
            and other objects.
        """
        pass

    def save_states(self):
        """Saves the states inside a checkpoint associated with ``epoch``."""
        checkpoint_data = dict()
        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint_data['model'] = self.model.module.state_dict()
        else:
            checkpoint_data['model'] = self.model.state_dict()
        checkpoint_data['optimizer'] = self.optimizer.state_dict()
        checkpoint_data['random_states'] = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
        checkpoint_data['counters'] = self.counters
        checkpoint_data['losses_epoch'] = self.losses_epoch
        checkpoint_data['losses_it'] = self.losses_it
        checkpoint_data.update(self.save_states_others())
        self.experiment.checkpoint_save(
            checkpoint_data, self.counters['epoch']
        )

    def save_states_others(self):
        """Saves the states of other objects inside a checkpoint associated
        with ``epoch``."""
        return {}

    def train(self, epoch, max_epochs, log_period, device):
        """Runs the ``train`` pipeline.

        Implements a highly-customizable training/validation pipeline. In
        detail, the pipeline:

        1. Loads a checkpoint, if given. If not, tries to restore the last
        checkpoint or departs from scratch.
        2. Iterates for a maximum of ``max_epochs``. In each epoch, the model
        extracts data from the loaders to train and validate the model.
        3. Propagates the data of each iteration using auxiliary method
        ``self.train_step()``.
        4. Saves a checkpoint at the end of the epoch.

        In order to extend or modify the default behavior of the pipeline,
        several hooks are also provided:

        - ``self.train_before_epoch_tasks()``
        - ``self.train_iteration_log()``
        - ``self.train_epoch_log()``
        - ``self.validation_iteration_log()``
        - ``self.validation_epoch_log()``
        - ``self.train_after_epoch_tasks()``
        - ``self.train_early_stopping()``

        Args:
            epoch (int or None): ``--epoch`` command argument.
            max_epochs (int): ``--max-epochs`` command argument.
            log_period (int): ``--log-period`` command argument.
            device (list): ``--device`` command argument.
        """
        # Restore checkpoint if exists or is forced
        epochs_list = self.experiment.checkpoints_get()
        if epoch is None and len(epochs_list) > 0:
            epoch = epochs_list[-1]
        if epoch and epoch not in epochs_list:
            raise ValueError('Epoch {} not found.'.format(epoch))
        elif epoch:
            self.load_states(epoch, device[0])

        # Start from the checkpoint epoch if exists. Otherwise it will start at
        # 1. Add +1 so max_epochs is respected.
        for self.counters['epoch'] in range(
                self.counters['epoch'] + 1, max_epochs + 1
        ):
            # Call self-implemented tasks which run before an epoch has
            # finished
            self.train_before_epoch_tasks(device[0])

            # Run Train
            self.model.train()
            e_train_losses = []
            for self.counters['train_it'], it_data in enumerate(
                    self.experiment.data.loaders['train'],
                    start=self.counters['train_it'] + 1
            ):
                self.optimizer.zero_grad()
                it_loss = self.train_step(it_data, device[0])
                it_loss.backward()
                self.optimizer.step()
                e_train_losses.append(it_loss.item())
                if self.counters['train_it'] % log_period == 0:
                    self.train_iteration_log(
                        e_train_losses, log_period, device[0]
                    )

            # Run Validation
            self.model.eval()
            e_validation_losses = []
            for self.counters['validation_it'], it_data in enumerate(
                    self.experiment.data.loaders['validation'],
                    start=self.counters['validation_it'] + 1
            ):
                with torch.no_grad():
                    it_loss = self.train_step(it_data, device[0])
                e_validation_losses.append(it_loss.item())
                if self.counters['validation_it'] % log_period == 0:
                    self.validation_iteration_log(
                        e_validation_losses, log_period, device[0]
                    )

            # Log Train
            self.train_epoch_log(e_train_losses, device[0])
            self.validation_epoch_log(e_validation_losses, device[0])

            # Call self-implemented tasks which run after an epoch has finished
            self.train_after_epoch_tasks(device[0])

            # Save the checkpoint
            self.save_states()

            # Apply early stopping using a self-implemented function. Must
            # return a boolean.
            if self.train_early_stopping():
                self.logger.info('Early stopping condition fulfilled')
                break

        # Flush TensorBoard and log end of training
        self.experiment.tbx.flush()
        self.logger.info('Training completed')

    def train_step(self, it_data, device):
        """Performs training steps associated with one data iteration.

        Args:
            it_data (any): output of the loader for the current iteration.
            device (str): first ordered element of ``--device`` command
            argument.

        Returns:
            loss (float): measured value the loss.
        """
        raise NotImplementedError

    def train_before_epoch_tasks(self, device):
        """Run at the beginning of an epoch.

        By default, it logs an initializing message.

        Args:
            device (str): first ordered element of ``--device`` command
            argument.
        """
        self.logger.info('Initializing Epoch {}'.format(
            self.counters['epoch'])
        )

    def train_iteration_log(self, e_train_losses, log_period, device):
        """Run every ``log_period`` train iterations.

        By default, it logs a small report of the last ``log_period`` train
        iterations both using the logger and TensorBoard.

        Args:
            e_train_losses (list): List containing all train losses of the
            epoch.
            log_period (int): ``--log-period`` command argument.
            device (str): first ordered element of ``--device`` command
            argument.
        """
        self.losses_it['train'][self.counters['train_it']] = np.mean(
            e_train_losses[-log_period:]
        )
        self.logger.info('Train Iteration {} - Loss {:.3f}'.format(
            self.counters['train_it'],
            self.losses_it['train'][self.counters['train_it']]
        ))
        self.experiment.tbx.add_scalar(
            'loss/iteration/train',
            self.losses_it['train'][self.counters['train_it']],
            self.counters['train_it']
        )

    def train_epoch_log(self, e_train_losses, device):
        """Run at the end of an epoch.

        By default, it logs a small report of the epoch both using the logger
        and TensorBoard.

        Args:
            e_train_losses (list): List containing all train losses of the
            epoch.
            device (str): first ordered element of ``--device`` command
            argument.
        """
        self.losses_epoch['train'][self.counters['epoch']] = np.mean(
            e_train_losses
        )
        self.experiment.tbx.add_scalar(
            'loss/epoch/train',
            self.losses_epoch['train'][self.counters['epoch']],
            self.counters['epoch']
        )

    def validation_iteration_log(
            self, e_validation_losses, log_period, device
    ):
        """Run every ``log_period`` validation iterations.

        By default, it logs a small report of the last ``log_period``
        validations iterations both using the logger and TensorBoard.

        Args:
            e_validation_losses (list): List containing all validation losses
            of the epoch.
            log_period (int): ``--log-period`` command argument.
            device (str): first ordered element of ``--device`` command
            argument.
        """
        self.losses_it['validation'][self.counters['validation_it']] = np.mean(
            e_validation_losses[-log_period:]
        )
        self.logger.info('Validation Iteration {} - Loss {:.3f}'.format(
            self.counters['validation_it'],
            self.losses_it['validation'][self.counters['validation_it']]
        ))
        self.experiment.tbx.add_scalar(
            'loss/iteration/validation',
            self.losses_it['validation'][self.counters['validation_it']],
            self.counters['validation_it']
        )

    def validation_epoch_log(self, e_validation_losses, device):
        """Run at the end of validation epoch.

        By default, it logs a small report of the epoch both using the logger
        and TensorBoard.

        Args:
            e_validation_losses (list): List containing all validation losses
            of the epoch.
            device (str): first ordered element of ``--device`` command
            argument.
        """
        self.losses_epoch['validation'][self.counters['epoch']] = np.mean(
            e_validation_losses
        )
        self.experiment.tbx.add_scalar(
            'loss/epoch/validation',
            self.losses_epoch['validation'][self.counters['epoch']],
            self.counters['epoch']
        )

    def train_after_epoch_tasks(self, device):
        """Run at the end of an epoch.

        By default, it logs a summary of the epoch using the logger.

        Args:
            device (str): first ordered element of ``--device`` command
            argument.
        """
        self.logger.info(
            'Epoch: {} | Average Training Loss: {:.3f} | Average Validation '
            'Loss: {:.3f}'.format(
                self.counters['epoch'],
                self.losses_epoch['train'][self.counters['epoch']],
                self.losses_epoch['validation'][self.counters['epoch']]
            )
        )

    def train_early_stopping(self):
        """Run before starting a new epoch. Would ``True`` in case that the
        training should stop at the current epoch.

        By default, it always returns ``False``.

        Returns:
            bool: whether or not the training loop should stop at the current
            epoch.
        """
        self.logger.info('Early stopping not implemented.')
        return False

    def test(self, epoch, device):
        """Runs the ``test`` pipeline.

        Args:
            epoch (int or None): ``--epoch`` command argument.
            device (list): first ordered element of ``--device`` command
            argument.
        """
        raise NotImplementedError
