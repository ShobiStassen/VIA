import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='VIA',
    version='0.01', #15-Sep-2020
    packages=['VIA',],
    license='MIT',
    author_email = 'shobana.venkat88@gmail.com',
    url = 'https://github.com/ShobiStassen/VIA',
    setup_requires = ['numpy','pybind11'],
    install_requires=['pybind11','numpy','scipy','pandas','hnswlib','python-igraph','leidenalg==0.7.0','umap-learn','math','multiprocessing', 'sklearn', 'termcolor','pygam'],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
