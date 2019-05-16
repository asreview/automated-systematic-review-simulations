# based on https://github.com/pypa/sampleproject
# MIT License

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Extract version from cbsodata.py
for line in open(path.join("pargrid", "__init__.py")):
    if line.startswith('__version__'):
        exec(line)
        break

setup(
    name='pargrid',
    version=__version__,  # noqa
    description='Parameter grid simulation for ASR',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/msdslab/automated-systematic-review-simulations',
    author='Utrecht University',
    author_email='r.d.schram@uu.nl',
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
    packages=find_packages(exclude=['hpc', 'preparation', 'data', 'simulation']),

    install_requires=[
        "pandas", "numpy", "sklearn", "keras", "matplotlib", "scipy",
        "asr @ git+https://github.com/msdslab/automated-systematic-review",
        "batchgen @ git+https://github.com/UtrechtUniversity/hpc-batch-gen",
    ],

    extras_require={
    },

    # package_data={
    #     'sample': ['data/package_data.dat'],
    # },
    # data_files=[('my_data', ['data/data_file'])],

    entry_points={
        'console_scripts': [
            'pickle_asr=pargrid.pickle:main'],
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
