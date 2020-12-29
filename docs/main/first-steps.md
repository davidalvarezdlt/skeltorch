# First steps
Skeltorch is designed to work under Python modules. Nowadays, most researchers
create a single script for each different task. For instance, it is normal to
find files named ``train.py`` or ``test.py``, each one with its associated data
pipeline.

**Skeltorch works completely different. Instead of creating different files,
each data pipeline is called using a different command on your module**.

In general, to run a module you can use:

```
python -m <your_module_name> <global_args> command_name <command_args>
```

Where ``your_module_name`` is the name of the folder containing the
``__init__.py`` file and each ``command_name`` is associated with one data
pipeline. By default, Skeltorch provides seven different pipelines:

- ``init``: creates a new experiment.
- ``info``: prints the information associated with an experiment.
- ``train``: trains and validates a model.
- ``test``: tests a previously-trained model.
- ``test_sample``: test a previously-trained model on a single data item.
- ``create_release``: creates a checkpoint containing model state only.
- ``tensorboard``: runs TensorBoard.

In this first steps tutorial, you will learn how to implement the methods
required to make these pipelines work as expected. At the end of it, you will
be ready to create simple projects which will be easily shareable with minimum
effort and focusing on what is important: the data and the model.

## 1. Creating the file structure
To create a Skeltorch project, you need to create a Python module. Skeltorch
comes with a CLI that helps you do precisely that:

```
skeltorch create --name <your_module_name>
```

This will create different files and folders. Specifically, the result will
look like:

```
data/
experiments/
your_module_name/
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

## 2. Creating the data class
The data class, stored in ``/your_module_name/data.py``, handles all functions
related with the data of the project. It also covers the creation of
``torch.utils.data.Dataset`` and ``torch.utils.data.DataLoader`` objects.

To create your own `skeltorch.Data` class, you should extend it and implement:

- ``create()``: called **only** when creating a new experiment. All class
parameters created inside this function are stored inside the experiment and
restored on each prospective load.
- ``load_datasets()``: loads a ``dict`` containing the datasets for the train,
validation and test splits.
- ``load_loaders()``: loads a ``dict`` containing the loaders for the train,
validation and test splits.

```
import skeltorch

class YourModuleNameData(skeltorch.Data):

    def create(self):
        raise NotImplementedError

    def load_datasets(self):
        raise NotImplementedError

    def load_loaders(self):
        raise NotImplementedError
```

Check out our examples to find real implementations of ``skeltorch.Data``
classes.

## 3. Creating the runner class
The runner class, stored in ``/your_module_name/runner.py``, handles all
functions related with the pipelines of the projects. It uses the attributes
and methods of other objects (accessible as class parameters) to train, test
and any other model-related tasks that may be needed for the project. Remember
that the models should be stored inside the ``/your_module_name/model.py``
file.

**Train Pipeline**

To use the default ``train`` pipeline of Skeltorch, you will need to implement
the function ``step_train()``. This function receives the data of one iteration
of the loader and returns the loss after being propagated through the model.

You will also have to initialize your model and optimizer implementing
``init_model()`` and ``init_optimizer()`` respectively. Both of them must be
stored as class parameters inside ``self.model`` and ``self.optimizer``,
respectively.

```
import skeltorch
from .model import YourModuleNameModel


class YourModuleNameRunner(skeltorch.Runner):

    def init_model(self, device):
        self.model = YourModuleNameModel().to(device)

    def init_optimizer(self, device):
        raise NotImplementedError

    def step_train(self, it_data, device):
        raise NotImplementedError

    ...
```

**Test and test Sample Pipelines**

To make the ``test`` pipeline work, you must implement your own ``test()``
method. As every test is different depending on the project you are working on,
you will have to implement the entire functionality of it. Notice that this
function is called when invoking the "test" command on your module.

The ``test_sample`` pipeline is supposed to work the same way as the ``test``,
but for a single data item identified by the parameter "sample".

```
import skeltorch
from .model import YourModuleNameModel

class YourModuleNameRunner(skeltorch.Runner):

    ...

    def test(self, epoch, device):
        raise NotImplementedError

    def test_sample(self, sample, epoch, device):
        raise NotImplementedError
```

Check out our examples to find real implementations of ``skeltorch.Runner``
classes.

## 4. Creating the configuration file

Every time that you create a new experiment (``init`` pipeline), you will be
asked to provide a configuration file associated with it. These configuration
parameters will be accessible through the configuration object of your
experiment. Be careful, because these configuration parameters are immutable.
This means that if you want to change one of them, you need to create a new
experiment.

Configuration files are created using ``.json`` format. You must organize your
configuration parameters in "groups". No more than one level of grouping is
allowed.

```
{
  "data": {
    "dataset": "<dataset_name>"
  },
  "training": {
    "batch_size": 32,
    "lr": 0.0001
  }
}
```

This configuration will be automatically loaded in the ``configuration``
attribute of the ``experiment`` object. In this example, to get the
configuration parameter named "dataset" of the group "data" you should call:

```
dataset = experiment.configuration.get('data', 'dataset')
```

The file ``config.schema.json``, also created automatically, can be used to
validate the configuration parameters when creating a new experiment. This file
is optional and, if used, should be passed in the ``--config-schema-path``
argument of ``init``.

Check the API Documentation for a reference of attributes available in each
object.

## 5. Running Skeltorch
The last step to create a Skeltorch project is to put everything together.
Inside your ``__main__.py`` file:

```
from skeltorch
from .data import YourModuleNameData
from .runner import YourModuleNameRunner

skeltorch.Skeltorch(
    YourModuleNameData(),
    YourModuleNameRunner()
).run()
```

**Congratulations, your project is now ready to be executed!** The next step is
to run it. Check *running default pipelines* for an extensive guide of how
to do it.