# Introduction

## What is Skeltorch?
Skeltorch is a **light-weight framework that helps researchers to prototype faster using PyTorch**. To do so, Skeltorch
provides developers with a set of predefined pipelines to organize projects and train/test their models.

Skeltorch is an experiment-based framework. What that means is that every possible variation of your model will be
represented by a different experiment. Every experiment is uniquely identified by its name and contains:

- A set of immutable configuration parameters, specified during its creation.
- A copy of the data object, also created during the creation of the experiment.
- The checkpoints of the model associated with the experiment.
- A set of TensorBoard files with a graphical evolution of the losses and other data that may be logged.
- A textual log of the actions performed on the experiment.

## Features
- Easy creation and loading of experiments.
- Automatic restoration of interrupted training.
- Readable JSON configuration files with the option to validate them using a schema.
- Visual logging using TensorBoard.
- Automatic logging using the native Python logging package.
- Automatic handling of random seeds, specified during the creation of an experiment.
- Easy implementation of custom pipelines.

## Installing Skeltorch
Use ``pip`` to install Skeltorch in your virtual environment:

```
pip install skeltorch
```