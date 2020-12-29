# Creating custom pipelines

Every project is different, and so are the procedures that may be needed. The
process of creating a new pipeline in Skeltorch is as easy as executing it. In
this short tutorial you will learn:

- How to create a custom pipeline.
- How to associate this custom pipeline with a certain ``command``.
- How to pass command-line arguments to your custom pipeline to modify its
  behavior.
- How to run the pipeline.

## 1. Creating the pipeline

The first step is to create a function that will be called when the pipeline is
invoked. Ideally, this function will be a method of your ``skeltorch.Runner``
class:

```
import skeltorch

class YourModuleNameRunner(skeltorch.Runner):

    def train_step(self, it_data, device):
        ...

    def my_custom_pipeline(self, custom_arg1, custom_arg2):
        ...
```

In the previous snippet, we have created a method named
``my_custom_pipeline()`` which receives ``arg1`` and ``arg2`` as arguments. The
values of these arguments will be passed as command-line arguments.

## 2. Creating a custom parser and its arguments

Now, we will add a new parser to our Python module. To do so, we will use the
``create_parser()`` method of the ``skeltorch.Skeltorch`` object. Inside our
``__main__.py`` file:

```
import skeltorch
from .data import YourModuleNameData
from .runner import YourModuleNameRunner

# Create a new instance of skeltorch.Skeltorch()
module_name_data = YourModuleNameData()
module_name_runner = YourModuleNameRunner()
skel = skeltorch.Skeltorch(module_name_data, module_name_runner)

# Create a new parser for my custom command named "my_custom_command"
my_custom_parser = skel.create_parser('my_custom_command')
my_custom_parser.add_argument(
    '--custom-arg1', type=int, required=True, help='Argument 1 is an integer.'
)
my_custom_parser.add_argument(
    '--custom-arg2', type=bool, default=False, help='Argument 2 is a boolean.'
)

...
```

## 3. Associating the pipeline with the parser

Created both the function that executes the pipeline and the parser with the
command that invokes it, the last step is to link both of them. To do so, we
will use the ``create_command()`` method of the ``skeltorch.Skeltorch`` object.
Continuing our ``__main__.py`` file:

```
import skeltorch
from .data import YourModuleNameData
from .runner import YourModuleNameRunner

# Create a new instance of skeltorch.Skeltorch()
module_name_data = YourModuleNameData()
module_name_runner = YourModuleNameRunner()
skel = skeltorch.Skeltorch(module_name_data, module_name_runner)

# Create a new parser for my custom command named "my_custom_command"
my_custom_parser = skel.create_parser('my_custom_command')
my_custom_parser.add_argument(
    '--custom-arg1', type=int, required=True, help='Argument 1 is an integer.'
)
my_custom_parser.add_argument(
    '--custom-arg2', type=bool, default=False, help='Argument 2 is a boolean.'
)

# Link the parser with the method of the pipeline
skel.create_command(
    'my_custom_command',
    my_custom_runner.my_custom_pipeline,
    ['custom_arg1', 'custom_arg2']
)

# Run Skeltorch
skel.run()
```

Where the first parameter of the identifier of the command (used in both
the parser and command), the second is the function to be called (if you write
the  ``()`` you are already calling it, which **is not what we want**), and
the third is a list of arguments to pass to the function. Two things are
important when writing the names of the arguments to pass:

- The name of the argument does not include the first two dashes.
- While the arguments of the command are written with dashes
  (``--custom-arg1``), these are automatically converted to underlines
  (``custom_arg1``).

## 4. Running your custom pipeline

Everything is already prepared. Now it is time to execute our new custom
pipeline. To do so, we will follow the same procedure used for default
pipelines:

```
python -m <your_module_name> <global_args> my_custom_command --custom-arg1 <custom_arg1> --custom-arg2 <custom_arg2>
```

Notice that you still have at your disposal all global arguments, which can be
included in the list as done with our custom arguments.

## Extra: Modifying the test pipeline

By default, the ``test`` pipeline only includes two custom arguments: ``epoch``
and ``device``. What if we want to include a new argument named ``arg3``? You
can do that very easily with Skeltorch. As before, we will start modifying the
method which is called when the command is invoked:

```
import skeltorch

class YourModuleNameRunner(skeltorch.Runner):

    def train_step(self, it_data, device):
        ...

    def test(self, epoch, device, arg3):
        ...
```

Now, instead of creating a new parser, we will get it using the
``get_parser()`` method of the ``skeltorch.Skeltorch`` object. We will use that
parser to add our new custom argument:

```
import skeltorch
from .data import YourModuleNameData
from .runner import YourModuleNameRunner

module_name_data = YourModuleNameData()
module_name_runner = YourModuleNameRunner()
skel = skeltorch.Skeltorch(module_name_data, module_name_runner)

# Create a new parser for my custom command named "my_custom_command"
test_parser = skel.get_parser('test')
test_parser.add_argument(
    '--arg3', required=True, help='Argument 3 is super important.'
)

...
```

Finally, we will replace the default parser-function association by calling
``create_command()`` as done in the previous example:

```
import skeltorch
from .data import YourModuleNameData
from .runner import YourModuleNameRunner

# Create a new instance of skeltorch.Skeltorch()
module_name_data = YourModuleNameData()
module_name_runner = YourModuleNameRunner()
skel = skeltorch.Skeltorch(module_name_data, module_name_runner)

# Create a new parser for my custom command named "my_custom_command"
test_parser = skel.get_parser('test')
test_parser.add_argument(
    '--arg3', required=True, help='Argument 3 is super important.'
)

# Replace parser-function association of the test pipeline
skel.create_command(
    'test', my_custom_runner.test, ['epoch', 'device', 'arg3']
)

# Run Skeltorch
skel.run()
```

The ``test`` pipeline is now ready to read the argument ``--arg3`` from the
command-line.