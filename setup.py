import setuptools
import sys
from setuptools import setup
from warnings import warn
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='pyVIA',
    version='0.1.61', #Nov-28-2022
    packages=['pyVIA',],
    license='MIT',
    author_email = 'shobana.venkat88@gmail.com',
    url = 'https://github.com/ShobiStassen/VIA',
    setup_requires = ['numpy>=1.17','pybind11'],
    install_requires=['pybind11','numpy>=1.17','scipy','pandas>=0.25','hnswlib','igraph','leidenalg>=0.7.0', 'sklearn', 'termcolor','pygam', 'matplotlib','scanpy','umap-learn','phate','datashader', 'scikit-image', 'pillow','wget','gdown'],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
