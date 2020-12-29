# Glow

In this example, we will implement an unconditional version
of [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)
. Glow is a normalizing flow, capable of learning a generative model without
any supervision. We will use CIFAR10 to train the network, a data set
containing 60.000 images from 10 different classes. At the end of this example,
you will be able to generate artificial images by randomly sampling from a
Gaussian.

A (slightly) modified version of this implementation has been used in the paper
[Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models](https://openreview.net/forum?id=SyxIWpVYvr)
. Please, cite the paper if its content is relevant for your research:

```
@inproceedings{Serra2020Input,
    title={Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models},
    author={Joan Serrà and David Álvarez and Vicenç Gómez and Olga Slizovskaia and José F. Núñez and Jordi Luque},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=SyxIWpVYvr}
}
```

You can check and download the files exposed in this tutorial from
our [official GitHub repository](https://github.com/davidalvarezdlt/skeltorch/tree/master/examples/glow)
.

## 1. File structure

The first step is to create the files required to create a Skeltorch project.
You can use the CLI of Skeltorch to do so:

```
skeltorch create --name glow
```

## 2. Configuration file

Glow has quite a few hyper-parameters that can be tuned during training. To
provide a flexible environment, we will set them using the configuration file
inside the ``model`` configuration group. Inside the ``config.default.json``
file, copy:

```
{
  "data": {
    "dataset": "cifar10",
    "image_size": 32,
    "pixel_depth": 8
  },
  "model": {
    "num_blocks": 3,
    "num_flows": 32,
    "squeezing_factor": 2,
    "permutation_type": "conv",
    "coupling_type": "affine",
    "num_filters": 512,
    "kernel_size": 3
  },
  "training": {
    "batch_size": 32,
    "lr": 1e-4,
    "lr_scheduler_patience": 2,
    "early_stopping_patience": 5
  }
}
```

Extra: to validate it, we will also fill an auxiliary file named
``config.schema.json``. You can check it in our GitHub repository if you are
interested in creating them for your projects.

## 3. Data class

We will start implementing our own ``skeltorch.Data`` class, which is used to
handle all data-related tasks. In this example, this class will be extremely
simple and will only consist of loading the data sets required to train the
model.

```
import skeltorch
import torch.utils.data
import torchvision.transforms


class GlowData(skeltorch.Data):
    transforms = None
```

We will only create two splits: one for training and another for validation. As
CIFAR10 comes with a default division of training and testing data, we will use
this last one as the validation split. Consequently, the only action required
when creating a new experiment is downloading the data, if not done already:

```
def create(self, data_path):
    torchvision.datasets.CIFAR10(
        data_path, transform=self.transforms, download=True
    )
```

We will use the default ``torch.utils.data.Dataset`` implementation of
``torchvision``. By default, this implementation returns ``PIL.Image`` objects
instead of ``torch.Tensor``. We will load a composition of required
transformations (only one in this example, but could be more) inside
``self.transforms``:

```
def _load_transforms(self):
    self.transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
```

We will load them using the method ``load_datasets()``, which is mandatory to
implement in any Skeltorch project:

```
def load_datasets(self, data_path):
    self._load_transforms()
    self.datasets['train'] = torchvision.datasets.CIFAR10(
        data_path, train=True, transform=self.transforms, download=True
    )
    self.datasets['validation'] = torchvision.datasets.CIFAR10(
        data_path, train=False, transform=self.transforms, download=True
    )
```

We will finish our implementation of the data class by extending the
``load_loaders()`` method:

```
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
```

## 4. Runner class: train pipeline

It is time to implement a custom ``skeltorch.Runner`` class. This class will
handle the implementation of the different pipelines using the data provided by
``GlowData`` according to the configuration parameters established by the user:

```
import numpy as np
import torch
import torch.optim
import skeltorch
from .model import GlowModel


class GlowRunner(skeltorch.Runner):
    scheduler = None
```

Notice that the ``torch.nn.Module`` object associated with the model is stored
inside ``glow/model.py``. Check the example files to get a detailed
implementation of it. We will start creating a new instance of both the model
and optimizer using Skeltorch default methods:

```
def init_model(self, device):
    self.model = GlowModel(
        num_channels=3,
        num_blocks=self.get_conf('model', 'num_blocks'),
        num_flows=self.get_conf('model', 'num_flows'),
        squeezing_factor=self.get_conf('model', 'squeezing_factor'),
        permutation_type=self.get_conf('model', 'permutation_type'),
        coupling_type=self.get_conf('model', 'coupling_type'),
        num_filters=self.get_conf('model', 'num_filters'),
        kernel_size=self.get_conf('model', 'kernel_size')
    ).to(device[0])

def init_optimizer(self, device):
    self.optimizer = torch.optim.Adam(
        params=self.model.parameters(),
        lr=self.get_conf('training', 'lr')
    )
```

In both cases, the hyper-parameters are being extracted from the
``skeltorch.Configuration`` object of the experiment (``self.experiment``). We
will also initialize a learning rate scheduler using ``init_others()``:

```
def init_others(self, device):
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=self.optimizer,
        patience=self.get_conf('training', 'lr_scheduler_patience')
    )
```

As the scheduler has a state, we have to save and load it. Skeltorch comes with
two default methods for this task: ``load_states_others()`` and
``save_states_others()``:

```
def load_states_others(self, checkpoint_data):
    self.scheduler.load_state_dict(checkpoint_data['scheduler'])

def save_states_others(self):
    return {'scheduler': self.scheduler.state_dict()}
```

Finally, as the learning rate will be reduced dynamically, we would like to
have a graphical representation of it. We can create a new Tensorboard plot for
this purpose. We will extend the method ``train_before_epoch_tasks()``, which
is a hook that runs at the beginning of every epoch. We will keep default
behavior by calling ``super().train_before_epoch_tasks()`` before appending our
own functionalities:

```
def train_before_epoch_tasks(self, device):
    super().train_before_epoch_tasks(device)
    self.experiment.tbx.add_scalar(
        'lr',
        self.optimizer.param_groups[0]['lr'],
        self.counters['epoch']
    )
```

We are ready to implement the main method of the training pipeline: the
``step_train()`` method. This method receives the data associated with one
iteration of the loaders (both training and validation splits) and returns the
loss. Skeltorch uses that loss to back-propagate the model and updates its
parameters:

```
def train_step(self, it_data, device):
    x = it_data[0].to(device[0])
    x = GlowRunner.add_noise(x, self.get_conf('data', 'pixel_depth'))
    z, log_det = self.model(x)
    return GlowRunner.compute_loss(
        z, log_det, self.get_conf('data', 'pixel_depth')
    )
```

The first step is to move the data to the correct device, stored in ``device``.
This command-line argument is a list of devices with at least one element, so
we will select the first one by default.

Notice that while ``it_data`` is a tuple containing both the images and the
labels, we are only using the images for our project. After adding noise to the
images (to simulate a continuous input), we compute the loss using the static
method ``GlowRunner.compute_loss()``. Check the runner file
(``glow/runner.py``) for a detailed implementation of it.

Finally, we will update the scheduler and call the test pipeline at the end of
every epoch by extending the ``train_after_epoch_tasks()`` hook. Check the next
section for detailed explanation of the behavior of this pipeline:

```
def train_after_epoch_tasks(self, device):
    super().train_after_epoch_tasks(device)
    self.scheduler.step(
        self.losses_epoch['validation'][self.counters['epoch']],
        self.counters['epoch']
    )
    self.test(None, device)
```

To avoid unnecessary computation, we will stop the training if the validation
loss has not improved for a certain amount of epochs (parameter given in the
configuration file). We can easily do that with the ``train_early_stopping()``,
which must return a ``bool`` representing whether the training should be
stopped or not:

```
def train_early_stopping(self):
    best_epoch = min(
        self.losses_epoch['validation'],
        key=self.losses_epoch['validation'].get
    )
    return self.counters['epoch'] - best_epoch > self.get_conf(
        'training', 'early_stopping_patience'
    )
```

## 5. Runner class: test pipeline

Glow is a generative model, and the best possible test is to make it generate
images. To generate images using a trained version of Glow, we have to generate
vectors where each position corresponds to a value sampled from a standard
Gaussian distribution. We can then reverse the flows to estimate its associated
image. If this process does not sound familiar to you, read the original paper
for more details:

```
def test(self, epoch, device):
    if epoch is not None:
        self.restore_states_if_possible(epoch, device)

    # Log the start of the test
    self.logger.info('Starting the test of epoch {}...'.format(
        self.counters['epoch'])
    )

    # Generate random Gaussian z's and reverse the model
    z = torch.randn(
        (5, self.get_conf('data', 'image_size') ** 2 * 3)
    ).to(device[0])
    with torch.no_grad():
        x = self.model.reverse(z)

    # Save generated images in TensorBoard
    self.experiment.tbx.add_images('samples', x, self.counters['epoch'])

    # Log the end of the test
    self.logger.info('Random samples stored in TensorBoard')
```

Notice that the first step of every pipeline is to load object states. In this
example, we will assume that they are already loaded if ``epoch=None``.

We generate a batch of 5 random Gaussian vectors of size
``img_height * img_width * channels`` = ``img_size^2 * 3``, where we assume
squared color images. Finally, the ``reverse()`` method of the model is called
to obtain their associated images.

We will save the generated images inside TensorBoard using the
``SummaryWriter`` object stored inside ``self.experiment.tbx``.

## 6. Initializing Skeltorch

The last step is to use our custom ``GlowData`` and ``GlowRunner`` classes to
create a Skeltorch project. Inside our ``glow/__main__.py`` file:

```
import skeltorch
from .data import GlowData
from .runner import GlowRunner

skeltorch.Skeltorch(GlowData(), GlowRunner()).run()
```

## 7. Running the pipelines

We are ready to run our example. We have implemented the three pipelines, now
it is time to execute them. First, we will start by creating a new experiment
with the ``init`` pipeline:

```
python -m glow --experiment-name glow_example --verbose init --config-path config.default.json --config-schema-file config.schema.json
```

The next step is to train the model. Do not forget to include ``--device cuda``
in case you want to run it in a GPU:

```
python -m glow --experiment-name glow_example --verbose train
```

We already have our model trained. We have already performed the test of every
epoch by calling it inside ``train_after_epoch_tasks()``. In any case, we could
run it again by calling:

```
python -m glow --experiment-name glow_example --verbose test --epoch 10
```

Where ``--epoch`` may receive any epoch from which the checkpoint exists.

## 8. Visualizing the results

Skeltorch comes with two ways of visualizing results. The simplest one is the
``verbose.log`` file stored inside every experiment, containing a complete log
of everything that has happened since its creation. However, the best way to
visualize results is by using TensorBoard. You can initialize it by calling:

```
python -m glow --experiment-name glow_example --verbose tensorboard
```

In case you want to check the configuration parameters of your experiment and
the available checkpoints, the ``info`` pipeline is the best way to do so. You
can run it by calling:

```
python -m glow --experiment-name glow_example --verbose info
```

## 9. Creating a release checkpoint

While standard checkpoints keep data related to the training, a release
checkpoint is a checkpoint that only preserves model information. You can
create a release checkpoint if you want to share the model you have trained
without having to share training states.

Creating a release checkpoint is straightforward. Just call the pipeline with
the epoch number from which you want to create the release, let's say epoch 10:

```
python -m glow --experiment-name glow_example --verbose create_release --epoch 10
```

The new release checkpoint will be stored inside
``experiments/glow_example/checkpoints/10.checkpoint.release.pkl``.
