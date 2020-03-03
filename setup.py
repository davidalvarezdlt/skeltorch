import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skeltorch",
    version="1.0.0b3",
    author="David Álvarez de la Torre",
    author_email="davidalvarezdlt@gmail.com",
    description="Light-weight framework that helps researchers to prototype faster using PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidalvarezdlt/skeltorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
