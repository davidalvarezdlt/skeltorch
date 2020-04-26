# First steps
Skeltorch is designed in order to work under Python modules. Nowadays, most researchers create a single script for each
different task. For instance, it is normal to find files named ``train.py`` or ``test.py``, each one with its associated
data pipeline.

**Skeltorch works completely different. Instead of creating different files, each data pipeline is called using a 
different command on your own module**.

In general, to run a module you can use:

```
python -m <your_module_name> <global_args> command_name <command_args>
```

Where ``your_module_name`` is the name of the folder containing the ``__init__.py`` file and each ``command_name`` is 
associated to one data pipeline. By default, Skeltorch provides three different pipelines:

- ``init``: creates a new experiment.
- ``train``: trains and validates a model.
- ``test``: tests a previously-trained model.

In this first steps tutorial, you will learn how to implement the methods required in order to make these pipelines work
as expected. At the end of it, you will be ready to create simple projects which will be easily shareable with minimum
effort and focusing on what is really important: the data and the model. 

## 1. Creating the file structure
In order to create a Skeltorch project, you need to create a Python module. To do so, it is enough to create a folder 
with a ``__init__.py`` file inside.

In addition to this file, you will also create different files to handle different parts of your project. Specifically:

- A ``__main__.py`` file to store initialization code.
- A ``data.py`` file to implement the class handling the data used in the project.
- A ``model.py`` file to implement your own models.
- A ``runner.py`` file to implement the class extending default pipeline behavior.

Finally, create a ``config.json`` file to store configuration parameters and, optionally, a ``config.schema.json`` to 
validate it. You may also want to create other auxiliary files such as a ``requirements.txt`` file or a ``README.md`` 
document. These documents should never be placed inside the module's folder.

In the end, you should have a file structure similar to:

```
your_module_name/
    __init__.py
    __main__.py
    data.py
    model.py
    runner.py
config.json
config.schema.json
```

You may also want to have a folder to store experiments and data and, optionally, another for scripts.

## 2. Creating the data class
The data class, stored in ``data.py``, handles all functions related to the data of the project. It also covers the 
creation of ``torch.utils.data.Dataset`` and ``torch.utils.data.DataLoader`` objects.

In order to create your own `skeltorch.Data` class, you should extend it and implement:

- ``create()``: called **only** when creating a new experiment. All class parameters created inside this function are
stored inside the experiment and restored on each prospective load.
- ``load_datasets()``: loads a ``dict`` containing the datasets for the train, validation and test splits.
- ``load_loaders()``: loads a ``dict`` containing the loaders for the train, validation and test splits.

```
import skeltorch

class MyDataClass(skeltorch.Data):
    
    def create(self):
        pass

    def load_datasets(self):
        pass

    def load_loaders(self):
        pass
```

Check out our examples to find real implementations of ``skeltorch.Data`` classes.

## 3. Creating the runner class
The runner class, stored in ``runner.py``, handles all functions related to the pipelines of the projects. It uses the
attributes and methods of other objects (accessible as class parameters) to train, test and any other model-related 
tasks that may be needed for the project. Remember that the models should be stored inside the ``model.py`` file.

**Train Pipeline**

In order to use the default ``train`` pipeline of Skeltorch, you will need to implement the function ``step_train()``.
This function receives the data of one iteration of the loader and returns the loss after being propagated through the
model.

You will also have to initialize your model and optimizer implementing ``init_model()`` and ``init_optimizer()``
respectively. Both of them must be stored as class parameters inside ``self.model`` and ``self.optimizer``, 
respectively.

```
import skeltorch


class MyRunnerClass(skeltorch.Runner):
    
    def init_model(self, device):
        pass

    def init_optimizer(self, device):
        pass

    def step_train(self, it_data, device):
        pass
```

**Test Pipeline**

In order to make the ``test`` pipeline work, you must implement your own ``test()`` method. As every test is different
depending on the project you are working on, you will have to implement the entire functionality of it. Notice that
this function is called when invoking the "test" command on your module.

```
import skeltorch


class MyRunnerClass(skeltorch.Runner):
    
    def test(self, epoch, devices):
        pass
```

Check out our examples to find real implementations of ``skeltorch.Runner`` classes.

## 4. Creating the configuration file

Every time that you create a new experiment (``init`` pipeline), you will be asked to provide a configuration file 
associated with it. These configuration parameters will be accessible through the configuration object of your 
experiment. Be careful, because these configuration parameters are immutable. This means that if you want to change one 
of them, you need to create a new experiment.

Configuration files are created using ``.json`` format. You must group your configuration parameters in "groups". No 
more than one level of grouping is allowed. 

```
{
    "data": {
        "dataset": "mnist",
        "image_size": 32
    },
    "training": {
        "batch_size": 32,
        "lr": 1e-4
    }
}
```

This configuration will be automatically loaded in the ``configuration`` attribute of the ``experiment`` object. In this 
example, to get the configuration parameter named "dataset" of the group "data" you should call:

```
dataset = experiment.configuration.get('data', 'dataset')
```

Check the API Documentation for a reference of attributes available in each object.

## 5. Running Skeltorch
The last step in order to create a Skeltorch project is to put everything together. Inside your ``__main__.py`` file:

```
from skeltorch
from .data import MyDataClass
from .runner import MyRunnerClass

# Create a Skeltorch object with your own Data and Runner classes
skel = skeltorch.Skeltorch(
    MyDataClass(),
    MyRunnerClass()
)

# Run Skeltorch
skel.run()
```

**Congratulations, your project is now ready to be executed!** The next step is to run it. Check *running default 
pipelines* for an extensive guide of how to do it.