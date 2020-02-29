# File structure of an experiment
Every time you create a new experiment, a set of files and folders is created inside your experiments folder. It is 
important to know the purpose of these files and folders, as they will store important data related to your experiment.
A standard experiment is composed of the following files and folders

```
experiment_name/
    checkpoints/
        1.checkpoint.pkl
        2.checkpoint.pkl
        ...
    results/
        ...
    tensorboard/
        ...
    config.pkl
    data.pkl
    verbose.log
```

Where:

- ``experiment_name/``: is the folder containing all data related to the experiment. It can be easily compressed a 
shared.
- ``experiment_name/checkpoints``: contains the checkpoints of the experiment.
- ``experiment_name/results/``: contains results that may be generated in the ``test`` or other custom pipelines.
- ``experiment_name/config.pkl``: contains a binary version of the configuration file used during the creation of an 
experiment. It is saved as a binary to prevent users from modifying it.
- ``experiment_name/data.pkl``: contains the attributes of your custom ``skeltorch.Data`` class, which are initialized
inside the ``create()`` method during the creation of an experiment.
- ``experiment_name/verbose.log``: contains the entire log of the experiment since its creation.

You can access the absolute path of these files and folders by accessing the ``paths`` attribute of the 
``skeltorch.Experiment`` object. For instance, to get the results folder inside your ``test`` method:

```
import skeltorch

class MyCustomRunner(skeltorch.Runner):

    def train_step(self, it_data, device):
        ...

    def test(self, epoch, device):
        results_folder = self.experiment.paths['results']
        ...
```

The dictionary keys of the paths of the files and folders are:  ``experiment``, ``checkpoints``, ``results``, 
``tensorboard``, ``configuration``, ``data`` and ``log``.