# MNIST Classifier

In this example, we will implement a model capable of classifying images of
digits. To do so, we will use the MNIST dataset, which contains a total of
60.000 images of numbers ranging from 0 to 9.

The model used in this example is the same used in
the [example provided by PyTorch](https://github.com/pytorch/examples/tree/master/mnist)
. You can check and download the files exposed in this tutorial from
our [official GitHub repository](https://github.com/davidalvarezdlt/skeltorch/tree/master/examples/mnist_classifier)
.

## 1. File structure

The first step is to create the files required to create a Skeltorch project.
You can use the CLI of Skeltorch to do so:

```
skeltorch create --name mnist_classifier
```

## 2. Configuration file

The project is quite simple, and lucky us we will need only a few configuration
parameters. Inside the ``config.default.json`` file, copy:

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

We could decide to also include configuration parameters related to the design
of the model and intermediate layers. To keep things simple, we will assume
that the model is already defined as immutable.

Extra: to validate it, we will also fill an auxiliary file named
``config.schema.json``. You can check it in our GitHub repository if you are
interested in creating them for your projects.

## 3. Data class

The first class that we have to extend is ``skeltorch.Data``. This class serves
as an interface between the file system and the pipelines. We will start by
filling the data class inside ``mnist_classifier/data.py``:

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

The first method to implement is ``load_datasets()``. The function of this
methods is to load ``torch.utils.data.Dataset`` objects in ``self.datasets``.
This attribute is in fact a dictionary with keys identifying each one of the
three natural splits: ``train``, ``validation`` and ``test``.

In our experiment we are going to use MNIST, which already has a
``torch.utils.data.Dataset`` implementation in ``torchvision``. By default,
data items provided by this implementation are ``PIL.Image`` objects, which
must be transformed into ``torch.Tensor`` objects. The parameter
``transform`` receives a set of transformations and applies them to every item.

We create an auxiliary method ``_load_transforms()`` to load this set of
transformations as a class attribute. To help the training, we will also
normalize the data as done in the original implementation:

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
    self.datasets['train'] = torchvision.datasets.MNIST(
        data_path, train=True, transform=self.transforms, download=True
    )
    self.datasets['validation'] = self.datasets['train']
    self.datasets['test'] = torchvision.datasets.MNIST(
        data_path, train=False, transform=self.transforms, download=True
    )
```

Notice that, by default, only training and testing splits are provided. We will
use a portion (given in the configuration parameter ``val_split``) of the
training data to create a validation split. We will perform this task in the
method ``create()``. This method is only called during the creation of a new
experiment and its purpose is to run tasks that should only be done once:

```
def create(self, data_path):
    self.load_datasets(data_path)

    # Create a list containing the indexes of the default MNIST Train split
    train_set_len = len(self.datasets['train'])
    train_set_indexes = list(range(train_set_len))
    random.shuffle(train_set_indexes)

    # Create a validation split using the percentage of data specified in
    # the configuration file
    val_split = self.get_conf('data', 'val_split')
    self.train_indexes = train_set_indexes[
        :round((1 - val_split) * len(train_set_indexes))
    ]
    self.validation_indexes = train_set_indexes[
        round((1 - val_split) * len(train_set_indexes)):
    ]
```

We are finally ready to implement the final method: ``load_loaders()``. This
method uses the previous ones to do exactly the same as ``load_datasets()`` but
with  ``torch.utils.data.DataLoader`` objects. These loaders are then used in
the pipelines to obtain the data of the iterations.

As both the training and validation splits share the same
``torch.data.utils.Dataset`` object, we have to somehow force the loader to use
the indexes created inside the ``create()`` method. The mechanism that PyTorch
provides for this purpose are samplers, which define a procedure to extract
data from a dataset. Specifically, we will use the
``torch.utils.data.SubsetRandomSampler()`` class, which both limits the range
of returned data and shuffles it on every epoch.

For the case of the test split, as there already exists a unique
``torch.utils.data.Dataset`` object, we can simply set ``shuffle=True`` to
shuffle the data on every epoch. We will extract the batch size from the
configuration file:

```
def load_loaders(self, data_path, num_workers):
    self.loaders['train'] = torch.utils.data.DataLoader(
        dataset=self.datasets['train'],
        sampler=torch.utils.data.SubsetRandomSampler(self.train_indexes),
        batch_size=self.get_conf('training', 'batch_size'),
        num_workers=num_workers
    )
    self.loaders['validation'] = torch.utils.data.DataLoader(
        dataset=self.datasets['validation'],
        sampler=torch.utils.data.SubsetRandomSampler(self.validation_indexes),
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

With our data class prepared, it's time to define our pipelines. In this
example, we will use default ``train`` and ``test`` pipelines with some small
extensions. We will use the model used in the PyTorch example and store it
inside ``mnist_example/model.py`` file. For the pipelines, inside our
``mnist_example/runner.py`` file:

```
import skeltorch
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler
from .model import MNISTClassifierModel


class MNISTClassifierRunner(skeltorch.Runner):
    scheduler = None
```

We will start implementing two straightforward but necessary methods:
``init_model()`` and ``init_optimizer()``. These methods initialize both
``self.model`` and ``self.optimizer`` with the ``torch.nn.Module`` and
``torch.optim.Optimizer`` objects to be used in the project. Do not forget to
send the objects to the device with the ``to()`` method:

```
def init_model(self, device):
    self.model = MNISTClassifierModel().to(device[0])

def init_optimizer(self, device):
    self.optimizer = torch.optim.Adadelta(
        params=self.model.parameters(),
        lr=self.get_conf('training', 'lr')
    )
```

We also want to include a scheduler to control the learning rate. We will
extend the ``init_others()`` method to initialize it:

```
def init_others(self, device):
    self.scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=self.optimizer,
        step_size=1,
        gamma=self.get_conf('training', 'lr_gamma')
    )
```

The state of the scheduler is not saved nor restored by default. However,
Skeltorch provides a really easy way of implementing it yourself:

```
def load_states_others(self, checkpoint_data):
    self.scheduler.load_state_dict(checkpoint_data['scheduler'])

def save_states_others(self):
    return {'scheduler': self.scheduler.state_dict()}
```

The ``train`` pipeline is already defined. From our part, we only have to
implement ``step_train()``, which receives the data associated to one iteration
and returns the loss after propagating through the model. The data returned by
the loaders is always stored in the CPU, we should always use the ``to()``
method to move it to other devices that may have been required. As ``device``
is a list, we will select the first item by default:

```
def train_step(self, it_data, device):
    it_data_input = it_data[0].to(device[0])
    it_data_target = it_data[1].to(device[0])
    it_data_prediction = self.model(it_data_input)
    return F.nll_loss(it_data_prediction, it_data_target)
```

We will finally use the ``train_after_epoch_tasks()`` hook to perform a step on
the scheduler at the end of every epoch. We will also call the ``test``
pipeline to compute the accuracy of the current epoch (implemented below). Do
not forget to call ``super().train_after_epoch_tasks()`` to avoid losing
default behavior:

```
def train_after_epoch_tasks(self, device):
    super().train_after_epoch_tasks(device)
    self.scheduler.step()
    self.test(None, device)
```

## 5. Runner class: test pipeline

Negative log-likelihood and other losses are usually good measures of how a
model is performing, but what we want to improve at the end is the accuracy of
our classifier. We will implement the ``test`` pipeline to measure it and log
it using both the default logger and TensorBoard:

```
def test(self, epoch, device):
    if epoch is not None:
        self.restore_states_if_possible(epoch, device)

    # Log the start of the test
    self.logger.info('Starting the test of epoch {}...'.format(
        self.counters['epoch'])
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
```

## 6. Initializing Skeltorch

The last step is to use our custom ``MNISTClassifierData`` and
``MNISTClassifierRunner`` classes to create a Skeltorch project. Inside our
``mnist_example/__main__.py`` file:

```
import skeltorch
from .data import MNISTClassifierData
from .runner import MNISTClassifierRunner

skeltorch.Skeltorch(MNISTClassifierData(), MNISTClassifierRunner()).run()
```

## 7. Running the pipelines

We are ready to run our example. We have implemented the three pipelines, now
it is time to execute them. First, we will start by creating a new experiment
with the ``init`` pipeline:

```
python -m mnist_classifier --experiment-name mnist_classifier_example --verbose init --config-path config.default.json --config-schema-file config.schema.json
```

The next step is to train the model. We can limit the number of epoch to
``--max-epochs 10``. Do not forget to include ``--device cuda`` in case you
want to run it in a GPU:

```
python -m mnist_classifier --experiment-name mnist_classifier_example --verbose train --max-epochs 10
```

We have already trained our model. We have performed the test of every epoch by
calling it inside ``train_after_epoch_tasks()``. In any case, we could run it
again by calling:

```
python -m mnist_classifier --experiment-name mnist_classifier_example --verbose test --epoch 10
```

Where ``--epoch`` may receive any epoch from which the checkpoint exists.

## 8. Visualizing the results

Skeltorch comes with two ways of visualizing results. The simplest one is the
``verbose.log`` file stored inside every experiment, containing a complete log
of everything that has happened since its creation. However, the best way to
visualize results is by using TensorBoard. You can initialize it by calling:

```
python -m mnist_classifier --experiment-name mnist_classifier_example --verbose tensorboard
```

In case you want to check the configuration parameters of your experiment and
the available checkpoints, the ``info`` pipeline is the best way to do so. You
can run it by calling:

```
python -m mnist_classifier --experiment-name mnist_example --verbose info
```

## 9. Creating a release checkpoint

While standard checkpoints keep data related to the training, a release
checkpoint is a checkpoint that only preserves model information. You can
create a release checkpoint if you want to share the model you have trained
without having to share training states.

Creating a release checkpoint is straightforward. Just call the pipeline with
the epoch number from which you want to create the release, let's say epoch 10:

```
python -m mnist_classifier --experiment-name mnist_classifier_example --verbose create_release --epoch 10
```

The new release checkpoint will be stored inside
``experiments/mnist_example/checkpoints/10.checkpoint.release.pkl``.
