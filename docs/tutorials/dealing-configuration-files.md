# Dealing with configuration files
When creating a new experiment using the ``init`` pipeline, you are forced to
pass a valid JSON file representing the configuration to be used in that
specific experiment. Configuration parameters are immutable, that is, can not
be changed.

Configuration parameters **must** be placed inside JSON objects. The name of
this object will represent the category of the configuration parameter. You can
create as many configuration categories and parameters as you wish, as long as
the identifiers of the categories are not repeated.

To create a configuration file, create a ``.json`` file containing valid JSON
content:

```
{
  "data": {
    "dataset": "mnist"
  },
  "training": {
    "batch_size": 32,
    "lr": 1e-4
  }
}
```

In this example, we have two configuration categories: ``data`` and
``training``. The first category only contains one configuration parameter,
while the second contains two. Given an instance of a
``skeltorch.Configuration`` object, you could access the parameter ``lr`` of
the ``training`` category as:

```
lr = configuration.get('training', 'lr')
```

You may also want to validate the content of your configuration file. To do so,
you can create a schema file and pass it using the command argument
``--config-schema-path`` of the ``init`` pipeline.

Check  [JSON-Schema.org](https://json-schema.org/) to get details of how to
create a valid schema file.