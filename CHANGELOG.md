### v2.0.0
+ Major refactoring of the code that breaks compatibility with Skeltorch
  v1.X.X.
+ You can now create Skeltorch projects by calling ``skeltorch create
  --name <project_name>``.
+ Added a new pipeline ``test_sample`` to test individual data samples.
+ Added a new pipeline ``create_release`` to create checkpoints that only
  contain model states.

### v1.1.0
+ You can now train using multiple GPUs. To do so, specify multiple
PyTorch-friendly device names on the `--device` argument.
+ Added a new pipeline ``info`` to obtain information related with an
  experiment.
+ Now you can run local or remote instances of TensorBoard using
  the ``tensorboard`` pipeline of Skeltorch.
+ Other small corrections and efficiency improvements.

### v1.0.0
+ Initial release.
