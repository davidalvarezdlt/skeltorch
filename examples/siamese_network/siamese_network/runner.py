from .model import SiameseNetworkModel
import numpy as np
import skeltorch
import torch
import torch.nn.functional as F


class SiameseNetworkRunner(skeltorch.Runner):
    loss_margin = None
    pr_max_threshold = None
    pr_n_threshold = None
    scheduler = None

    def init_model(self, device):
        self.model = SiameseNetworkModel(
            n_mfcc=self.get_conf('data', 'n_mfcc'),
            sf=self.get_conf('data', 'sf_target'),
            cut_length=self.get_conf('data', 'cut_length'),
            hop_length=self.get_conf('data', 'hop_length'),
            n_components=self.get_conf('model', 'n_components')
        ).to(device[0])

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.get_conf('training', 'lr'),
            weight_decay=self.get_conf('training', 'weight_decay')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.get_conf('training', 'lr_scheduler_step_size'),
            gamma=self.get_conf('training', 'lr_scheduler_gamma')
        )

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict()}

    def train_step(self, it_data, device):
        y1, y2 = self.model(it_data[0].to(device[0]), it_data[1].to(device[0]))
        return SiameseNetworkRunner.compute_loss(
            y1, y2, it_data[2].to(device[0]),
            self.get_conf('model', 'loss_margin')
        )

    def train_before_epoch_tasks(self, device):
        self.experiment.tbx.add_scalar(
            'lr',
            self.optimizer.param_groups[0]['lr'],
            self.counters['epoch']
        )

    def train_after_epoch_tasks(self, device):
        self.scheduler.step(self.counters['epoch'])
        self.test(None, device)

    def test(self, epoch, device):
        if epoch is not None:
            self.restore_states_if_possible(epoch, device)

        # Log start of test
        self.logger.info(
            'Starting test of epoch {}...'.format(self.counters['epoch'])
        )

        # Create list to store GT and predictions
        gt, pred, loss = [], [], []

        # Iterate over the test data loader
        for it_data in self.experiment.data.loaders['test']:
            with torch.no_grad():
                y1, y2 = self.model(
                    it_data[0].to(device[0]), it_data[1].to(device[0])
                )
                loss.append(
                    self.compute_loss(
                        y1, y2, it_data[2].to(device[0]),
                        self.get_conf('model', 'loss_margin')
                    ).item()
                )
            gt += it_data[2].tolist()
            pred += (F.pairwise_distance(y1, y2)).tolist()

        # Compute loss, metrics and distance measures
        loss_mean = np.mean(loss)
        tp, fp, tn, fn, precision, recall, f_score = self.compute_metrics(
            np.array(gt),
            np.array(pred),
            self.get_conf('testing', 'pr_max_threshold'),
            self.get_conf('testing', 'pr_n_threshold')
        )
        mean_dist_same_speaker = np.mean(
            [pred[i] for i in range(len(gt)) if gt[i] == 0]
        )
        mean_dist_diff_speaker = np.mean(
            [pred[i] for i in range(len(gt)) if gt[i] == 1]
        )

        # Add plots to Tensorboard
        self.experiment.tbx.add_scalar(
            'loss/epoch/test', loss_mean, self.counters['epoch']
        )
        self.experiment.tbx.add_scalar(
            'mean_distance/same_speaker',
            mean_dist_same_speaker,
            self.counters['epoch']
        )
        self.experiment.tbx.add_scalar(
            'mean_distance/diff_speaker',
            mean_dist_diff_speaker,
            self.counters['epoch']
        )
        self.experiment.tbx.add_pr_curve_raw(
            'pr', tp, fp, tn, fn, precision, recall, self.counters['epoch']
        )

        # Log end of test
        self.logger.info(
            'Test of epoch {} finished. Results logged in TensorBoard.'.format(
                self.counters['epoch']
            )
        )

    def test_sample(self, sample, epoch, device):
        raise NotImplementedError

    @staticmethod
    def compute_loss(y1, y2, is_different_speaker, loss_margin):
        euclidean_distance = F.pairwise_distance(y1, y2)
        loss = (1 - is_different_speaker) * euclidean_distance.pow(2)
        loss += is_different_speaker * F.relu(
            loss_margin - euclidean_distance
        ).pow(2)
        return (0.5 * loss).mean()

    @staticmethod
    def compute_metrics(gt, pred, pr_max_threshold, pr_n_threshold, eps=1e-6):
        threshold = np.linspace(0, pr_max_threshold, num=pr_n_threshold)
        tp, fp, tn, fn, precision, recall, f_score = [], [], [], [], [], [], []
        for th in threshold:
            tp.append(np.sum(np.logical_and(pred >= th, gt)))
            fp.append(np.sum(np.logical_and(pred >= th, np.logical_not(gt))))
            tn.append(np.sum(np.logical_and(pred < th, np.logical_not(gt))))
            fn.append(np.sum(np.logical_and(pred < th, gt)))
            precision.append(tp[-1] / (tp[-1] + fp[-1] + eps))
            recall.append(tp[-1] / (tp[-1] + fn[-1] + eps))
            f_score.append(
                (2 * precision[-1] * recall[-1]) /
                (precision[-1] + recall[-1] + eps)
            )
        return tp, fp, tn, fn, precision, recall, f_score
