"""A package for building a hierarchy based on multiple partitions on graph nodes.
See:
https://github.com/SHZ66/HierWeaver
"""

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hierweaver',  
    version='1.0.0', 
    description='A package for building a hierarchy based on multiple partitions on graph nodes.', 
    long_description=long_description,  

    long_description_content_type='text/markdown',  
    url='https://github.com/SHZ66/HierWeaver',  
    author='She Zhang',  
    author_email='shz66@pitt.edu',  

    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[ 
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='hierarchy tree DAG',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']), 

    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=['numpy', 'networkx'], 

    project_urls={ 
        'Bug Reports': 'https://github.com/SHZ66/HierWeaver/issues',
        'Source': 'https://github.com/SHZ66/HierWeaver',
    },
)