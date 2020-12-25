# MNIST Classifier
In this example we will implement a model capable of classifying images of digits. To do so, we will use the MNIST
dataset, which contains a total of 60.000 images of numbers ranging from 0 to 9.

The model used in this example is the same used in the
[example provided by PyTorch](https://github.com/pytorch/examples/tree/master/mnist). You can check and download the
files exposed in this tutorial from our
[official GitHub repository](https://github.com/davidalvarezdlt/skeltorch/tree/master/examples/mnist_classifier).

## 1. File structure
The first step is to create the files required to create a Skeltorch project. In this example, we will create not only
mandatory files, but also a ``config.schema.json`` file to validate the configuration file:

```
data/
experiments/
mnist_classifier/
    __init__.py
    __main__.py
    data.py
    model.py
    runner.py
config.default.json
config.schema.json
README.md
requirements.txt
```

## 2. Configuration file
The project is quite simple, and lucky us we will need only a few configuration parameters. Inside the
``config.default.json`` file:

```
{
  "data": {
    "val_split": 0.2
  },
  "training": {
    "batch_size": 64,
    "lr": 1,
    "lr_gamma": 0.7
  }
}
```

We could decide to also include configuration parameters related to the design of the model and intermediate layers. To
keep things simple, we will assume that the model is already defined an immutable.

Extra: in order to validate it, we create an auxiliary file named ``config.schema.json``. You can check it in our GitHub
repository if you are interested in creating them for your own projects.

## 3. Data class
The first class that we have to extend is ``skeltorch.Data``. This class serves as an interface between the file system
and the pipelines. We will start by creating our data class inside ``mnist_classifier/data.py``:

```
import random
import skeltorch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms


class MNISTClassifierData(skeltorch.Data):
    train_indexes = None
    validation_indexes = None
    transforms = None
```

We will start by implementing the method ``load_datasets()``. The function of this methods is to load
``torch.utils.data.Dataset`` objects in ``self.datasets``. This attribute is in fact a dictionary with keys identifying
each one of the three natural splits: ``train``, ``validation`` and ``test``.

In our experiment we are going to use MNIST, which already has a ``torch.utils.data.Dataset`` implementation in
``torchvision``. By default, data items provided by this implementation are ``PIL.Image`` objects, which must be
transformed to ``torch.Tensor`` objects. The parameter ``transform`` receives a set of transformations and applies them
to every item.

We create an auxiliary method ``_load_transforms()`` to load this set of transformations as a class attribute. To help
the training, we will also normalize the data as done in the original implementation:

```
def _load_transforms(self):
    self.transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
```

```
def load_datasets(self, data_path):
    self._load_transforms()
    self.datasets['train'] = torchvision.datasets.MNIST(data_path, train=True, transform=self.transforms, download=True)
    self.datasets['validation'] = self.datasets['train']
    self.datasets['test'] = torchvision.datasets.MNIST(data_path, train=False, transform=self.transforms, download=True)
```

Notice that, by default, only training and testing splits are provided. We will use a portion (given in the
configuration parameter ``val_split``) of the training data to create a validation split. We will perform this task in
the method ``create()``. This method is only called during the creation of a new experiment and its purpose is to run
tasks that should only be done once:

```
def create(self, data_path):
    self.load_datasets(data_path)

    # Create a list containing the indexes of the default MNIST Train split
    train_set_len = len(self.datasets['train'])
    train_set_indexes = list(range(train_set_len))
    random.shuffle(train_set_indexes)

    # Create a validation split using the percentage of data specified in the configuration file
    val_split = self.configuration.get('data', 'val_split')
    self.train_indexes = train_set_indexes[:round((1 - val_split) * len(train_set_indexes))]
    self.validation_indexes = train_set_indexes[round((1 - val_split) * len(train_set_indexes)):]
```

We are finally ready to implement the final method: ``load_loaders()``. This method uses the previous ones to do exactly
the same as ``load_datasets()`` but with  ``torch.utils.data.DataLoader`` objects. These loaders are then used in the
pipelines to obtain the data of the iterations.

As both the training and validation splits share the same ``torch.data.utils.Dataset`` object, we have to somehow force
the loader to use the indexes created inside the ``create()`` method. The mechanism that PyTorch provides for this
purpose are samplers, which define a procedure to extract data from a dataset. Specifically, we will use the
``torch.utils.data.SubsetRandomSampler()`` class, which both limits the range of returned data and shuffles it on every
epoch.

For the case of the test split, as there already exists a unique ``torch.utils.data.Dataset`` object, we can simply set
``shuffle=True`` to shuffle the data on every epoch. We will extract the batch size from the configuration file:

```
def load_loaders(self, data_path, num_workers):
    self.loaders['train'] = torch.utils.data.DataLoader(
        dataset=self.datasets['train'],
        sampler=torch.utils.data.SubsetRandomSampler(self.train_indexes),
        batch_size=self.configuration.get('training', 'batch_size'),
        num_workers=num_workers
    )
    self.loaders['validation'] = torch.utils.data.DataLoader(
        dataset=self.datasets['validation'],
        sampler=torch.utils.data.SubsetRandomSampler(self.validation_indexes),
        batch_size=self.configuration.get('training', 'batch_size'),
        num_workers=num_workers
    )
    self.loaders['test'] = torch.utils.data.DataLoader(
        dataset=self.datasets['test'],
        shuffle=True,
        batch_size=self.configuration.get('training', 'batch_size'),
        num_workers=num_workers
    )
```

## 4. Runner class: train pipeline
With our data class prepared, its time to define our pipelines. In this example, we will use default ``train`` and
``test`` pipelines with some small extensions. We will use the model used in the PyTorch example and store it
inside ``mnist_example/model.py`` file. For the pipelines, inside our ``mnist_example/runner.py`` file:

```
import skeltorch
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler
from .model import MNISTClassifierModel


class MNISTClassifierRunner(skeltorch.Runner):
    scheduler = None
```

We will start implementing two straightforward but necessary methods: ``init_model()`` and ``init_optimizer()``. These
methods initialize both ``self.model`` and ``self.optimizer`` with the ``torch.nn.Module`` and ``torch.optim.Optimizer``
objects to be used in the project. Do not forget to send the objects to the device with the ``to()`` method:

```
def init_model(self, device):
    self.model = MNISTClassifierModel().to(device)

def init_optimizer(self, device):
    self.optimizer = torch.optim.Adadelta(
        params=self.model.parameters(),
        lr=self.experiment.configuration.get('training', 'lr')
    )
```

We also want to include a scheduler to control the learning rate. We will extend the ``init_others()`` method to
initialize it:

```
def init_others(self, device):
    self.scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=self.optimizer,
        step_size=1,
        gamma=self.experiment.configuration.get('training', 'lr_gamma')
    )
```

The ``train`` pipeline is already defined. From our part, we only have to implement ``step_train()``, which receives
the data associated to one iteration and returns the loss after propagating through the model. The data returned by the
loaders is always stored in the CPU, we should always use the ``to()`` method to move it to other devices that may have
been required:

```
def train_step(self, it_data, device):
    it_data_input = it_data[0].to(device)
    it_data_target = it_data[1].to(device)
    it_data_prediction = self.model(it_data_input)
    return F.nll_loss(it_data_prediction, it_data_target)
```

We will finally use the ``train_after_epoch_tasks()`` hook to perform a step on the scheduler at the end of every epoch.
We will also call the ``test`` pipeline to compute the accuracy of the current epoch (implemented below). Do not forget
to call ``super().train_after_epoch_tasks()`` to avoid losing default behavior:

```
def train_after_epoch_tasks(self, device):
    super().train_after_epoch_tasks(device)
    self.scheduler.step()
    self.test(None, device)
```

## 5. Runner class: test pipeline
Negative log-likelihood and any other loss are usually good measures of how a model is performing, but at the end what
we want is to improve the accuracy of our classifier. We will implement the ``test`` pipeline to measure it and log it
using both the default logger and TensorBoard. Some details about the implementation:

- We will implement the method in order to handle a possible argument ``epoch=None``. In that precise case, the method
assumes that no checkpoint should be restored. In any other case, it checks that it exists and loads it.
- We use the loader stored inside ``self.experiment.data.loaders['test']``, which is the one we initialized in the data
class.
- Do not forget to move the data coming our of the loader to the correct device with ``.to(device)``.
- As we do not want to back-propagate the model, we can use ``torch.no_grad()`` to increase computation speed.
- Do not forget to call ``self.experiment.tbx.flush()`` to commit TensorBoard logs.

```
def test(self, epoch, device):
    # Check if test has a forced epoch to load objects and restore checkpoint
    if epoch is not None and epoch not in self.experiment.checkpoints_get():
        raise ValueError('Epoch {} not found.'.format(epoch))
    elif epoch is not None:
        self.load_states(epoch, device)

    # Log the start of the test
    self.logger.info('Starting the test of epoch {}...'.format(self.counters['epoch']))

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
        it_data_prediction_labels = it_data_prediction.argmax(dim=1, keepdim=True)
        n_correct += it_data_prediction_labels.eq(it_data_target.view_as(it_data_prediction_labels)).sum().item()

    # Compute accuracy dividing by the entire dataset
    test_acc = n_correct / len(self.experiment.data.loaders['test'])

    # Log accuracy using textual logger and TensorBoard
    self.logger.info('Test of epoch {} | Accuracy: {:.2f}%'.format(self.counters['epoch'], test_acc))
    self.experiment.tbx.add_scalar('accuracy/epoch/test', test_acc, self.counters['epoch'])
    self.experiment.tbx.flush()
```

## 5. Initializing Skeltorch
The last step is to use our custom ``MNISTClassifierData`` and ``MNISTClassifierRunner`` classes to create a Skeltorch
project. Inside our ``mnist_example/__main__.py`` file:

```
import skeltorch
from .data import MNISTClassifierData
from .runner import MNISTClassifierRunner

skeltorch.Skeltorch(MNISTClassifierData(), MNISTClassifierRunner()).run()
```

## 6. Running the pipelines
We are ready to run our example. We have implemented the three pipelines, now it is time to execute them. First, we will
start by creating a new experiment with the ``init`` pipeline:

```
python -m mnist_classifier --experiment-name mnist_example init --config-path config.default.json --config-schema-file config.schema.json
```

The next step is to train the model. We will limit the number of epoch to ``--max-epochs 10``. Do not forget to include
``--device cuda`` in case you want to run it in a GPU:

```
python -m mnist_classifier --experiment-name mnist_example train --max-epochs 10
```

We already have our model trained. We have already performed the test of every epoch by calling it inside
``train_after_epoch_tasks()``. In any case, we could run it again by calling:

```
python -m mnist_classifier --experiment-name mnist_example test --epoch 10
```

Where ``--epoch`` may receive any epoch from which the checkpoint exists.

## 7. Visualizing the results
Skeltorch comes with two ways of visualizing results. The simplest one is the ``verbose.log`` file stored inside every
experiment, containing a complete log of everything that has happened since its creation:

```
...
Train Iteration 2200 - Loss 0.058
Validation Iteration 400 - Loss 0.031
Validation Iteration 500 - Loss 0.048
Epoch: 3 | Average Training Loss: 0.058 | Average Validation Loss: 0.042
Starting the test of epoch 3...
Test of epoch 3 | Accuracy: 62.97%
Checkpoint of epoch 3 saved.
Initializing Epoch 4
Train Iteration 2300 - Loss 0.047
Train Iteration 2400 - Loss 0.045
Train Iteration 2500 - Loss 0.042
...
```

However, the best way to visualize results is using TensorBoard. You can initialize it by calling:

```
python -m mnist_classifier --experiment-name mnist_example tensorboard
```