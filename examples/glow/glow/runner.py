import numpy as np
import torch
import torch.optim
import skeltorch
from .model import GlowModel


class GlowRunner(skeltorch.Runner):
    scheduler = None

    def init_model(self, device):
        self.model = GlowModel(
            num_channels=3,
            num_blocks=self.experiment.configuration.get('model', 'num_blocks'),
            num_flows=self.experiment.configuration.get('model', 'num_flows'),
            squeezing_factor=self.experiment.configuration.get('model', 'squeezing_factor'),
            permutation_type=self.experiment.configuration.get('model', 'permutation_type'),
            coupling_type=self.experiment.configuration.get('model', 'coupling_type'),
            num_filters=self.experiment.configuration.get('model', 'num_filters'),
            kernel_size=self.experiment.configuration.get('model', 'kernel_size')
        ).to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=self.experiment.configuration.get('training', 'lr_scheduler_patience')
        )

    def train_step(self, it_data, device):
        x = it_data[0].to(device)
        x = GlowRunner.add_noise(x, self.experiment.configuration.get('data', 'pixel_depth'))
        z, log_det = self.model(x)
        return GlowRunner.compute_loss(z, log_det, self.experiment.configuration.get('data', 'pixel_depth'))

    def train_before_epoch_tasks(self, device):
        super().train_before_epoch_tasks(device)
        self.experiment.tbx.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.counters['epoch'])

    def train_after_epoch_tasks(self, device):
        super().train_after_epoch_tasks(device)
        self.scheduler.step(self.losses_epoch['validation'][self.counters['epoch']], self.counters['epoch'])
        self.test(None, device)

    def train_early_stopping(self):
        best_epoch = min(self.losses_epoch['validation'], key=self.losses_epoch['validation'].get)
        early_stopping_patience = self.experiment.configuration.get('training', 'early_stopping_patience')
        return True if self.counters['epoch'] - best_epoch > early_stopping_patience else False

    def test(self, epoch, device):
        # Check if test has a forced epoch to load objects and restore checkpoint
        if epoch is not None and epoch not in self.experiment.checkpoints_get():
            self.logger.error('Epoch {} not found.'.format(epoch))
            exit()
        elif epoch is not None:
            self.load_states(epoch, device)

        # Log the start of the test
        self.logger.info('Starting the test of epoch {}...'.format(self.counters['epoch']))

        # Generate random Gaussian z's and reverse the model
        z = torch.randn((5, self.experiment.configuration.get('data', 'image_size') ** 2 * 3)).to(device)
        x = self.model.reverse(z)

        # Save generated images in TensorBoard
        self.experiment.tbx.add_images('samples', x, self.counters['epoch'])

        # Log the end of the test
        self.logger.info('Random samples stored in TensorBoard')

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict()}

    @staticmethod
    def add_noise(it_data, pixel_depth: int):
        it_data *= 2 ** pixel_depth - 1
        it_data += torch.rand_like(it_data)
        it_data /= 2 ** pixel_depth - 1
        return it_data

    @staticmethod
    def compute_loss(z, log_det, pixel_depth: int):
        n_dim = z.size(1)
        log_p = -n_dim * 0.5 * np.log(2 * np.pi).item() - 0.5 * (z ** 2).sum(1)
        c = -n_dim * np.log(1 / (2 ** pixel_depth))
        nll = -log_p - log_det + c
        nll /= np.log(2)
        nll /= z.size(1)
        return nll.mean()
