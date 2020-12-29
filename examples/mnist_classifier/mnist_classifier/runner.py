import skeltorch
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler
from .model import MNISTClassifierModel


class MNISTClassifierRunner(skeltorch.Runner):
    scheduler = None

    def init_model(self, device):
        self.model = MNISTClassifierModel().to(device[0])

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adadelta(
            params=self.model.parameters(),
            lr=self.get_conf('training', 'lr')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=1,
            gamma=self.get_conf('training', 'lr_gamma')
        )

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict()}

    def train_step(self, it_data, device):
        it_data_input = it_data[0].to(device[0])
        it_data_target = it_data[1].to(device[0])
        it_data_prediction = self.model(it_data_input)
        return F.nll_loss(it_data_prediction, it_data_target)

    def train_after_epoch_tasks(self, device):
        super().train_after_epoch_tasks(device)
        self.scheduler.step()
        self.test(None, device)

    def test(self, epoch, device):
        if epoch is not None:
            self.restore_states_if_possible(epoch, device)

        # Log the start of the test
        self.logger.info(
            'Starting the test of epoch {}...'.format(self.counters['epoch'])
        )

        # Iterate over the entire test split
        n_correct = 0
        for it_data in self.experiment.data.loaders['test']:
            it_data_input = it_data[0].to(device[0])
            it_data_target = it_data[1].to(device[0])

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
            self.counters['epoch'], test_acc
        ))
        self.experiment.tbx.add_scalar(
            'accuracy/epoch/test', test_acc, self.counters['epoch']
        )

    def test_sample(self, sample, epoch, device):
        if epoch is not None:
            self.restore_states_if_possible(epoch, device)

        # Verify that the sample ID is valid
        sample = int(sample)
        if sample < 0 or sample > len(self.experiment.data.datasets['test']):
            self.logger.error('Invalid sample ID.')
            exit()

        # Predict category for the sample data item
        it_data = self.experiment.data.datasets['test'][sample]
        with torch.no_grad():
            it_data_prediction = self.model(
                it_data[0].to(device[0]).unsqueeze(0)
            ).squeeze(0)

        # Compute the class with maximum probability and print it
        it_data_class = it_data_prediction.argmax()
        self.logger.info(
            'Predicted class {} with probability {} for sample {}. Real class '
            'is {}.'.format(
                it_data_class, it_data_prediction[it_data_class].exp(),
                sample, it_data[1]
            )
        )
