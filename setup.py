"""A package for building a hierarchy based on multiple partitions on graph nodes.
See:
https://github.com/fanzheng10/HiDeF
"""

from setuptools import setup, find_packages
from os import path
from io import open
import re
import pathlib

here = pathlib.Path(__file__).parent

install_requires = (here / "requirements.txt").read_text().splitlines()

with open(path.join('hidef', '__init__.py')) as ver_file:
    for line in ver_file:
        if line.startswith('__version__'):
            version = re.sub("'", "", line[line.index("'"):])

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hidef',
    version=version,
    description='A package for building a hierarchy based on multiple partitions on graph nodes.', 
    long_description=long_description,  

    long_description_content_type='text/markdown',  
    url='https://github.com/fanzheng10/HiDeF',
    author='Fan Zheng, She Zhang',
    author_email='fanzheng1101@gmail.com',

    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[ 
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    scripts=['hidef/hidef_finder.py'],
    keywords='hierarchy tree DAG',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']), 

    python_requires='>=3.6, <4',
    install_requires=install_requires,

    project_urls={ 
        'Bug Reports': 'https://github.com/fanzheng10/HiDeF/issues',
        'Source': 'https://github.com/fanzheng10/HiDeF',
    },
)