import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().rstrip().split('\n')

setuptools.setup(
    name='skeltorch',
    version='2.0.0',
    author='David Álvarez de la Torre',
    author_email='davidalvarezdlt@gmail.com',
    description='Light-weight framework that helps researchers to prototype'
                'faster using PyTorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/davidalvarezdlt/skeltorch',
    install_requires=requirements,
    packages=setuptools.find_packages(),
    package_data={'skeltorch': ['templates/*']},
    include_package_data=True,
    entry_points={'console_scripts': ['skeltorch=skeltorch.__main__:run']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
