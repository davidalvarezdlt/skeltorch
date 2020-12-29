# Running default pipelines

Once you have implemented your custom ``skeltorch.Data`` and
``skeltorch.Runner`` classes, you are ready to run the default pipelines. In
total, Skeltorch comes with seven pre-defined pipelines associated with
different commands.

Before trying to run your Skeltorch project, remember to have it in your
``$PYTHONPATH`` or run your commands inside the folder containing your module.
To execute a pipeline, we will use the syntax:

```
python -m <your_module_name> <global_args> command_name
```

Where ``command_name`` is the command associated with each pipeline and
``<global_args>`` are arguments that are available in all pipelines.

**Global Arguments**

- ``--experiment-name <experiment_name>``: name of the experiment.
  **Required**.
- ``--base-path <base_path>``: base path from which other default paths are
  created. By default, it is the path from where the code is run.
- ``--experiments-path <experiments_path>``: path to the folder where the
  experiments are created/loaded from. Default: ``$base_path/experiments``.
- ``--data-path <data_path>``: path to the folder where the data is stored.
  Default: `$base_path/data`.
- ``--verbose``: whether or not to log using standard output. Default: `True`.

## Init

This pipeline is used to create new experiments and initialize both the
configuration and the data objects of the experiment. To run the ``init``
pipeline successfully, you need:

- A valid implementation of the ``create()`` method of your data object. In
  case you do not need to manage data during bootstrapping, you can always
  include the ``pass`` statement in the body of the method. In any case, you
  are forced to implement the method in your custom class.
- A valid JSON configuration file.
- (Optional) A valid schema file to validate your configuration file.

```
python -m <your_module_name> <global_args> init <init_args>
```

**Init Arguments**

- ``--config-path <config_path>``: path to the configuration file to be used in
  the new experiment. **Required**.
- ``--config-schema-path <config_schema_path>``: path to the schema file used
  to validate the configuration file.
- ``--seed <seed>``: seed to be used in the experiment. Default: ``0``.

## Info

The ``info`` pipeline returns information associated with the experiment.
Specifically, it prints the configuration parameters associated with the
experiment and the available checkpoints, if any. The data is displayed using
the ``logging.Logger`` object of the execution, so make sure to set the global
argument ``--verbose`` if you want to see the information using standard
output.

```
python -m <your_module_name> <global_args> info
```

## Train

The ``train`` pipeline is an implementation of the standard training procedure,
where a `torch.utils.data.DataLoader` is used to obtain the data of each
iteration until no more data is available. At that point, the epoch counter is
increased and a checkpoint is saved inside the experiment. In order to run the
training pipeline successfully, you will need:

- A valid implementation of the methods ``load_datasets()`` and
  ``load_loaders()`` of the data class. These methods must load the data
  attributes ``dataset`` and ``loaders`` with dictionaries with indexes
  ``train``, ``validation`` and ``test`` containing valid
  ``torch.utils.data.Dataset`` and ``torch.utils.data.DataLoader`` objects.
- A valid implementation of both ``init_model()`` and ``init_optimizer()``
  methods of the runner class. Both methods load the model and optimizer as
  class attributes, accessible as ``self.model`` and ``self.optimizer``
  respectively.
- A valid implementation of the ``train_step()`` method of the runner class.
  This class receives as input the data associated with one iteration and
  returns the loss after propagating through the model.

Every epoch is not only trained with the train data split but also validated
using the validation data split.

Notice that the last checkpoint is automatically restored if no ``--epoch`` has
been specified. To know more about the implementation details of the pipeline,
check the implementation of the ``train()`` method inside
``skeltorch/runner.py`` file.

```
python -m <your_module_name> <global_args> train <train_args>
```

**Train Arguments**

- ``--epoch <epoch>``: epoch from which the training should be restored.
  Default: ``None``.
- ``--max-epochs <max_epochs>``: maximum number of epochs to run.
  Default: ``999``.
- ``--log-period <log_period>``: number of iterations to wait between iteration
  logging. Default: ``100``.
- ``--num-workers <num_workers>``: number of workers to use in
  ``torch.utils.data.DataLoader`` objects. Default: ``1``.
- ``--device <device>``: PyTorch-friendly names of the devices where the
  process should be executed. Default: ``cuda`` if available, if not ``cpu``.

## Test

The ``test`` pipeline is an open pipeline devised to test a checkpoint of your
experiment. No default behavior is included in Skeltorch. You are free to test
your model the way you prefer. To run the test pipeline successfully, you will
need:

- A valid implementation of the method ``test()`` of the runner class.
- A valid implementation of auxiliary methods, such as the ones used in the
  training pipeline.

```
python -m <your_module_name> <global_args> test <test_args>
```

**Test Arguments**

- ``--epoch <epoch>``: epoch from which the training should be restored.
  **Required**.
- ``--num-workers <num_workers>``: number of workers to use in the
  ``torch.utils.data.DataLoader`` objects. Default: ``1``.
- ``--device <device>``: PyTorch-friendly names of the devices where the
  process should be executed. Default: ``cuda`` if available, if not ``cpu``.

## Test Sample

The ``test_sample`` pipeline follows the sample principle as the ``test``
pipeline, but for single data instances. The identifier of the data instance is
passed using a CLI argument. To run the test sample pipeline successfully, you
will need:

- A valid implementation of the method ``test_sample()`` of the runner class.
- A valid implementation of auxiliary methods, such as the ones used in the
  training pipeline.

```
python -m <your_module_name> <global_args> test_sample <test_sample_args>
```

**Test Sample Arguments**

- ``--sample <sample>``: identifier of the sample to be tested. **Required**.
- ``--epoch <epoch>``: epoch from which the training should be restored.
  **Required**.
- ``--num-workers <num_workers>``: number of workers to use in the
  ``torch.utils.data.DataLoader`` objects. Default: ``1``.
- ``--devices <devices>``: PyTorch-friendly names of the devices where the
  process should be executed. Default: ``cuda`` if available, if not ``cpu``.

## Create Release

Creates a release checkpoint of a certain epoch. A release checkpoint only
contains the state of the model while removing all other training states.

```
python -m <your_module_name> <global_args> create_release <create_release_args>
```

**Create Release Arguments**

- ``--epoch <epoch>``: epoch from which the checkpoint should be created.
  **Required**.

## TensorBoard

The ``tensorboard`` pipeline is a command wrapper to execute TensorBoard using
the files generated for your experiment. You can decide between running it in
your local machine or using [TensorBoard.dev](https://tensorboard.dev), which
is executed on the cloud and allows for easy sharing of training development.

```
python -m <your_module_name> <global_args> tensorboard <tensorboard_args>
```

**Tensorboard Arguments**

- ``--port <port>``: port number where TensorBoard should be executed, if run
  locally. Default: ``6006``.
- ``--dev``: run using [TensorBoard.dev](https://tensorboard.dev).
- ``--compare``: run Tensorboard over all existing experiments.