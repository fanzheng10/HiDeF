"""A package for building a hierarchy based on multiple partitions on graph nodes.
See:
https://github.com/fanzheng10/HiDeF
"""

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hidef',
    version='1.1.3',
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
    install_requires=['numpy',
                      'networkx',
                      'scipy',
                      'pandas',
                      'leidenalg',
                      'scikit-learn'],

    project_urls={ 
        'Bug Reports': 'https://github.com/fanzheng10/HiDeF/issues',
        'Source': 'https://github.com/fanzheng10/HiDeF',
    },
)