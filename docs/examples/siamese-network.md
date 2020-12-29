# Siamese Network

In this example we will implement a simple siamese network that estimates if
two speech segments are from the same speaker. We will use VCTK, a speech
dataset that comes with dozens of different speakers in several accents.

You can check and download the files exposed in this tutorial from our
[official GitHub repository](https://github.com/davidalvarezdlt/skeltorch/tree/master/examples/siamese_network).

## 1. File structure

The first step is to create the files required to create a Skeltorch project.
You can use the CLI of Skeltorch to do so:

```
skeltorch create --name siamese_network
```

## 2. Configuration file
We will use the configuration file to store the different parameters used to
handle the data, the model and the training process. Inside the
``config.default.json`` file, copy:

```
{
  "data": {
    "dataset": "vctk",
    "sf_original": 48000,
    "sf_target": 16000,
    "cut_length": 1,
    "n_mfcc": 40,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "val_split": 0.2,
    "n_test_speakers": 20
  },
  "model": {
    "n_components": 16,
    "loss_margin": 2.0
  },
  "training": {
    "batch_size": 32,
    "lr": 0.0001,
    "lr_scheduler_step_size": 10,
    "lr_scheduler_gamma": 0.5,
    "weight_decay": 0.001,
    "train_max_samples": 10000,
    "validation_max_samples": 1000,
    "test_max_samples": 1000
  },
  "testing": {
    "pr_max_threshold": 4,
    "pr_n_threshold": 100
  }
}
```

In this example, we will not fill the ``config.schema.json`` document. We can
remove it to avoid misunderstandings.

## 3. Data class
We will start by implementing the ``skeltorch.Data`` class of the project.
We will create three different splits using VCTK v0.92, where the training
and validation sets will belong to the same speakers while the test split will
contain speakers that have not been seen during training.

```
from .dataset import SiameseDataset
import os.path
import random
import skeltorch
import torch.utils.data
import torchaudio


class SiameseNetworkData(skeltorch.Data):
    data_meta_train = {}
    data_meta_validation = {}
    data_meta_test = {}
```

The first method to implement is ``create()``. This method is called when
creating a new experiment with the ``init`` pipeline and will contain the
code related with the creating of the splits. The amount of data used for each
split will be stored as a configuration parameter:

```
def create(self, data_path):
    torchaudio.datasets.VCTK_092(root=data_path, download=True)

    # Get list of files
    vctk_folder = os.path.join(data_path, 'VCTK-Corpus-0.92')
    vctk_files_list = list(
        torchaudio.datasets.utils.walk_files(vctk_folder, suffix='.flac')
    )

    # Fill items with partial path
    for i, vctk_item in enumerate(vctk_files_list):
        speaker_id, utterance_id, _ = vctk_item.split('_')
        vctk_files_list[i] = os.path.join(
            'VCTK-Corpus-0.92', 'wav48_silence_trimmed', speaker_id,
            vctk_item
        )

    # Create a dictionary containing samples for each speaker
    vctk_files_dict = {}
    for vctk_file_item in vctk_files_list:
        speaker_id = os.path.basename(vctk_file_item).split('_')[0]
        if speaker_id not in vctk_files_dict:
            vctk_files_dict[speaker_id] = [vctk_file_item]
        else:
            vctk_files_dict[speaker_id].append(vctk_file_item)

    # Store test samples and remove the speakers from the dict
    n_test_speakers = self.get_conf('data', 'n_test_speakers')
    test_speakers = random.sample(vctk_files_dict.keys(), n_test_speakers)
    for speaker_id in test_speakers:
        self.data_meta_test[speaker_id] = vctk_files_dict[speaker_id]
        vctk_files_dict.pop(speaker_id)

    # Create train/validation split for the rest of speakers
    val_split = self.get_conf('data', 'val_split')
    for speaker_id in vctk_files_dict.keys():
        n_validation_samples = round(
            len(vctk_files_dict[speaker_id]) * val_split
        )
        validation_samples = random.sample(
            vctk_files_dict[speaker_id], n_validation_samples
        )
        self.data_meta_validation[speaker_id] = validation_samples
        self.data_meta_train[speaker_id] = list(
            set(vctk_files_dict[speaker_id]) - set(validation_samples)
        )
```

Notice that the first line of the method uses ``torchaudio`` to download
the dataset. The process might take some hours to complete.

We will now implement the other two mandatory methods, that is,
``load_datasets()`` and ``load_loaders``. We will implement our own dataset
class inside ``siamese_network/dataset.py``. Check the content of the file
in our GitHub repository to know about the details of this implementation:

```
def load_datasets(self, data_path):
    self.datasets['train'] = SiameseDataset(
        data_path=data_path,
        data_meta=self.data_meta_train,
        sf_original=self.get_conf('data', 'sf_original'),
        sf_target=self.get_conf('data', 'sf_target'),
        cut_length=self.get_conf('data', 'cut_length'),
        n_mfcc=self.get_conf('data', 'n_mfcc'),
        n_fft=self.get_conf('data', 'n_fft'),
        win_length=self.get_conf('data', 'win_length'),
        hop_length=self.get_conf('data', 'hop_length'),
        max_samples=self.get_conf('training', 'train_max_samples')
    )
    self.datasets['validation'] = SiameseDataset(
        data_path=data_path,
        data_meta=self.data_meta_validation,
        sf_original=self.get_conf('data', 'sf_original'),
        sf_target=self.get_conf('data', 'sf_target'),
        cut_length=self.get_conf('data', 'cut_length'),
        n_mfcc=self.get_conf('data', 'n_mfcc'),
        n_fft=self.get_conf('data', 'n_fft'),
        win_length=self.get_conf('data', 'win_length'),
        hop_length=self.get_conf('data', 'hop_length'),
        max_samples=self.get_conf('training', 'validation_max_samples')
    )
    self.datasets['test'] = SiameseDataset(
        data_path=data_path,
        data_meta=self.data_meta_test,
        sf_original=self.get_conf('data', 'sf_original'),
        sf_target=self.get_conf('data', 'sf_target'),
        cut_length=self.get_conf('data', 'cut_length'),
        n_mfcc=self.get_conf('data', 'n_mfcc'),
        n_fft=self.get_conf('data', 'n_fft'),
        win_length=self.get_conf('data', 'win_length'),
        hop_length=self.get_conf('data', 'hop_length'),
        max_samples=self.get_conf('training', 'test_max_samples')
    )

def load_loaders(self, data_path, num_workers):
    self.loaders['train'] = torch.utils.data.DataLoader(
        dataset=self.datasets['train'],
        shuffle=True,
        batch_size=self.get_conf('training', 'batch_size'),
        num_workers=num_workers
    )
    self.loaders['validation'] = torch.utils.data.DataLoader(
        dataset=self.datasets['validation'],
        shuffle=True,
        batch_size=self.get_conf('training', 'batch_size'),
        num_workers=num_workers
    )
    self.loaders['test'] = torch.utils.data.DataLoader(
        dataset=self.datasets['test'],
        shuffle=True,
        batch_size=self.get_conf('training', 'batch_size'),
        num_workers=num_workers
    )
```

## 4. Runner class: train pipeline

We will now implement the logic associated with our project. To do so, we will
extend ``skeltorch.Runner`` and implement its methods according to our
requirements.

```
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
```

We will initialize both our model and the optimizer used to train it. As
always, you can check the exact implementation of the model inside
``siamese_network/model.py``. To make things better, we will also create a
training scheduler:

```
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
```

In order to keep track of the scheduler state, we will implement our custom
``load_states_others()`` and ``save_states_others()`` methods.

```
def load_states_others(self, checkpoint_data):
    self.scheduler.load_state_dict(checkpoint_data['scheduler'])

def save_states_others(self):
    return {'scheduler': self.scheduler.state_dict()}
```

We are now ready to implement the main method of the training pipeline. This
method will receive the set of data associated with one iteration and will
return the value of the loss. Skeltorch will handle back-propagation
automatically.

Our custom dataset object returns three different items: two random chunks of
speech data of certain length, and a parameter containing 1 if the chunks are
from different speakers or 0 otherwise. We will implement our siamese loss
function in a different function to keep things simple:

```
def train_step(self, it_data, device):
    y1, y2 = self.model(it_data[0].to(device[0]), it_data[1].to(device[0]))
    return SiameseNetworkRunner.compute_loss(
        y1, y2, it_data[2].to(device[0]),
        self.get_conf('model', 'loss_margin')
    )

@staticmethod
def compute_loss(y1, y2, is_different_speaker, loss_margin):
    euclidean_distance = F.pairwise_distance(y1, y2)
    loss = (1 - is_different_speaker) * euclidean_distance.pow(2)
    loss += is_different_speaker * F.relu(
        loss_margin - euclidean_distance
    ).pow(2)
    return (0.5 * loss).mean()
```

We will log the learning rate before every epoch and run the test pipeline once
it's finished. As we have implemented our custom scheduler, we will also update
its state after every epoch:

```
def train_before_epoch_tasks(self, device):
    self.experiment.tbx.add_scalar(
        'lr',
        self.optimizer.param_groups[0]['lr'],
        self.counters['epoch']
    )

def train_after_epoch_tasks(self, device):
    self.scheduler.step(self.counters['epoch'])
    self.test(None, device)
```

## 5. Runner class: test pipeline

We will test our model by measuring four different things: the loss, the
average distance when the speaker is the same, the average distance when
the speakers are different, and the Precision & Recall curve:

```
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
```

## 6. Initializing Skeltorch

The last step is to use our custom ``SiameseNetworkData`` and
``SiameseNetworkRunner`` classes to create a Skeltorch project. Inside our
``siamese_network/__main__.py`` file:

```
import skeltorch
from .data import SiameseNetworkData
from .runner import SiameseNetworkRunner

skeltorch.Skeltorch(SiameseNetworkData(), SiameseNetworkRunner()).run()
```

## 7. Running the pipelines

We are ready to run our example. We have implemented the three pipelines, now
it is time to execute them. First, we will start by creating a new experiment
with the ``init`` pipeline:

```
python -m siamese_network --experiment-name siamese_network_example --verbose init --config-path config.default.json
```

The next step is to train the model. Do not forget to include ``--device cuda``
in case you want to run it in a GPU:

```
python -m siamese_network --experiment-name siamese_network_example --verbose train
```

We already have our model trained. We have already performed the test of every
epoch by calling it inside ``train_after_epoch_tasks()``. In any case, we could
run it again by calling:

```
python -m siamese_network --experiment-name --verbose siamese_network_example test --epoch 10
```

Where ``--epoch`` may receive any epoch from which the checkpoint exists.

## 8. Visualizing the results

Skeltorch comes with two ways of visualizing results. The simplest one is the
``verbose.log`` file stored inside every experiment, containing a complete log
of everything that has happened since its creation. However, the best way to
visualize results is by using TensorBoard. You can initialize it by calling:

```
python -m siamese_network --experiment-name siamese_network_example --verbose tensorboard
```

In case you want to check the configuration parameters of your experiment and
the available checkpoints, the ``info`` pipeline is the best way to do so. You
can run it by calling:

```
python -m siamese_network --experiment-name siamese_network_example --verbose info
```

## 9. Creating a release checkpoint

While standard checkpoints keep data related to the training, a release
checkpoint is a checkpoint that only preserves model information. You can
create a release checkpoint if you want to share the model you have trained
without having to share training states.

Creating a release checkpoint is straightforward. Just call the pipeline with
the epoch number from which you want to create the release, let's say epoch 10:

```
python -m siamese_network --experiment-name siamese_network_example --verbose create_release --epoch 10
```

The new release checkpoint will be stored inside
``experiments/siamese_network_example/checkpoints/10.checkpoint.release.pkl``.
