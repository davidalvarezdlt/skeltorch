# Introduction

## What is Skeltorch?
Skeltorch is a **light-weight framework that helps researchers to prototype
faster using PyTorch**. To do so, Skeltorch provides developers with a set of
predefined pipelines to organize projects and train/test their models.

Skeltorch is an experiment-based framework. What that means is that every
possible variation of your model will be represented by a different experiment.
Every experiment is uniquely identified by its name and contains:

- A set of immutable configuration parameters, specified during its creation.
- A copy of the data object, also created during the creation of the
experiment.
- The checkpoints of the model associated with the experiment.
- A set of TensorBoard files with a graphical evolution of the losses and other
data that may be logged.
- A textual log of the actions performed on the experiment.

## Features
- Easy creation and loading of experiments.
- Automatic restoration of interrupted training.
- Readable JSON configuration files with the option to validate them using a
schema.
- Visual logging using TensorBoard.
- Automatic logging using the native Python logging package.
- Automatic handling of random seeds, specified during the creation of an
experiment.
- Easy implementation of custom pipelines.
- Automatic handling of multi-GPU training.

## Installing Skeltorch
Use ``pip`` to install Skeltorch in your virtual environment:

```
pip install skeltorch
```

## Where should I start?
Skeltorch has been designed to be easy to use. We provide you with a lot of
material to take your first steps with the framework:

1. Start by reading our [first steps tutorial](https://skeltorch.readthedocs.io/en/latest/main/first-steps.html),
where we give you a high-level overview of how to organize a project.

2. Take a look to one of our examples. If you are totally new to the framework,
you might want to start with our [MNIST Classifier example](https://skeltorch.readthedocs.io/en/latest/examples/mnist.html).

3. Read [our tutorials](https://skeltorch.readthedocs.io/en/latest/tutorials/running-pipelines.html)
to know everything you need to know about Skeltorch and how to customize
default behavior.

4. For a deep understanding of the framework, we recommend you to take a look
to our [API Documentation](https://skeltorch.readthedocs.io/en/latest/api/skeltorch.html).

## Contributing
You are invited to submit your pull requests with new features or bug
corrections. Before creating the pull request, make sure to run all pre-commits
and unit tests.

```
pip install pytest pre-commit
pre-commit run --all
pytest
```
