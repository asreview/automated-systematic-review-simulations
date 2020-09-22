# based on https://github.com/pypa/sampleproject
# MIT License

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Extract version from cbsodata.py
for line in open(path.join("asreviewcontrib", "simulation", "__init__.py")):
    if line.startswith('__version__'):
        exec(line)
        break

setup(
    name='asreview-simulation',
    version=__version__,  # noqa
    description='Simulation project for ASR',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/asreview/automated-systematic-review-simulations',
    author='Utrecht University',
    author_email='asreview@uu.nl',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='asr automated review batch',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    install_requires=[
        "asreview-hyperopt", "mpi4py", "asreview",
    ],

    extras_require={
    },

    entry_points={
        "asreview.entry_points": [
            "batch = asreviewcontrib.simulation:BatchEntryPoint",
        ]
    },

    project_urls={
        'Bug Reports':
            "https://github.com/msdslab/"
            "automated-systematic-review-simulations/issues",
        'Source':
            "https://github.com/msdslab"
            "/automated-systematic-review-simulations",
    },
)
