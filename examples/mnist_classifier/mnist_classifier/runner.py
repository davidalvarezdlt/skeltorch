import skeltorch
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler
from .model import MNISTClassifierModel


class MNISTClassifierRunner(skeltorch.Runner):
    scheduler = None

    def init_model(self, device):
        self.model = MNISTClassifierModel().to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adadelta(
            params=self.model.parameters(),
            lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=1,
            gamma=self.experiment.configuration.get('training', 'lr_gamma')
        )

    def train_step(self, it_data, device):
        it_data_input = it_data[0].to(device)
        it_data_target = it_data[1].to(device)
        it_data_prediction = self.model(it_data_input)
        return F.nll_loss(it_data_prediction, it_data_target)

    def train_after_epoch_tasks(self, device):
        super().train_after_epoch_tasks(device)
        self.scheduler.step()
        self.test(None, device)

    def test(self, epoch, device):
        if epoch and epoch not in self.experiment.checkpoints_get():
            raise ValueError('Epoch {} not found.'.format(epoch))
        elif epoch is not None:
            self.init_model(device)
            self.init_optimizer(device)
            self.load_states(epoch, device)

        # Log the start of the test
        self.logger.info('Starting the test of epoch {}...'.format(
            self.counters['epoch'])
        )

        # Create a variable to store correct predictions
        n_correct = 0

        # Iterate over the entire test split
        for it_data in self.experiment.data.loaders['test']:
            it_data_input = it_data[0].to(device)
            it_data_target = it_data[1].to(device)

            # Propagate the input through the model
            with torch.no_grad():
                it_data_prediction = self.model(it_data_input)

            # Increase the number of correct predictions
            it_data_prediction_labels = it_data_prediction.argmax(
                dim=1, keepdim=True
            )
            n_correct += it_data_prediction_labels.eq(
                it_data_target.view_as(it_data_prediction_labels)
            ).sum().item()

        # Compute accuracy dividing by the entire dataset
        test_acc = n_correct / len(self.experiment.data.loaders['test'])

        # Log accuracy using textual logger and TensorBoard
        self.logger.info('Test of epoch {} | Accuracy: {:.2f}%'.format(
            self.counters['epoch'], test_acc)
        )
        self.experiment.tbx.add_scalar(
            'accuracy/epoch/test', test_acc, self.counters['epoch']
        )
        self.experiment.tbx.flush()

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict()}
