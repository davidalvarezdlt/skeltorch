import argparse
import os
import pkg_resources
import re
import jinja2


class SkeltorchCLI:
    def __init__(self):
        self._fill_args()
        self._fill_empty_args()
        self._fill_tags()
        self.templating = jinja2.Environment(
            loader=jinja2.PackageLoader('skeltorch')
        )

    def _fill_args(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command', required=True)
        subparsers_create = subparsers.add_parser('create', help='a help')
        subparsers_create.add_argument('--name', help='Name of the project')
        self.args = parser.parse_args().__dict__

    def _fill_empty_args(self):
        while not self.args['name'] or \
                not re.match(r'^[a-z\_]+$', self.args['name']):
            if self.args['name']:
                print(
                    '[ERROR] Invalid project name. Make sure to only use '
                    'lower case characters and _.'
                )
            self.args['name'] = input(
                'Write the name of the project using snake case style (i.e. '
                'my_project): '
            )

    def _fill_tags(self):
        self.tags = dict()
        self.tags['project_name'] = self.args['name']
        self.tags['project_name_camelcase'] = ''.join(
            [x.title() for x in self.args['name'].split('_')]
        )
        self.tags['skeltorch_version'] = pkg_resources.get_distribution(
            'skeltorch'
        ).version

    def _create_file(self, template_name, path):
        with open(path, 'w') as f:
            template = self.templating.get_template(template_name).render(
                **self.tags
            ) if template_name else ''
            f.write(template)

    def create(self):
        """Creates the standard file structure of a Skeltorch project."""
        # Create root folders
        os.makedirs('data')
        os.makedirs('experiments')
        os.makedirs(self.args['name'])

        # Create files inside /
        self._create_file('config.default.jinja', 'config.default.json')
        self._create_file('config.schema.jinja', 'config.schema.json')
        self._create_file('requirements.jinja', 'requirements.txt')
        self._create_file('README.jinja', 'README.md')

        # Create files inside /data
        self._create_file(None, os.path.join('data', '.gitkeep'))

        # Create files inside /experiments
        self._create_file(None, os.path.join('experiments', '.gitkeep'))

        # Create files inside /<project_name>
        self._create_file(None, os.path.join(self.args['name'], '__init__.py'))
        self._create_file(
            '__main__.jinja', os.path.join(self.args['name'], '__main__.py')
        )
        self._create_file(
            'data.jinja', os.path.join(self.args['name'], 'data.py')
        )
        self._create_file(
            'model.jinja', os.path.join(self.args['name'], 'model.py')
        )
        self._create_file(
            'runner.jinja', os.path.join(self.args['name'], 'runner.py')
        )


def run():
    """Creates and runs a SkeltorchCLI object."""
    skeltorch_cli = SkeltorchCLI()
    skeltorch_cli.create()
