Installation
=============

**install using pip** ::
  
  conda create --name ViaEnv python=3.7 
  pip install pyVIA // tested on linux Ubuntu 16.04 and Windows 10

This usually tries to install hnswlib, produces an error and automatically corrects itself by first installing pybind11 followed by hnswlib. To get a smoother installation, consider installing in the following order after creating a new conda environment::
  
  pip install pybind11
  pip install hnswlib
  pip install pyVIA
           
**install by cloning repository and running setup.py** (ensure dependencies are installed)::

  git clone https://github.com/ShobiStassen/VIA.git 
  python3 setup.py install // cd into the directory of the cloned VIA folder containing setup.py and issue this command


**MAC installation** 
The pie-chart cluster-graph plot does not render correctly for MACs for the time-being. All other outputs are as expected.::

  conda create --name ViaEnv python=3.7 
  pip install pybind11
  conda install -c conda-forge hnswlib
  pip install pyVIA


**Windows installation**::

Note that on Windows if you do not have Visual C++ (required for hnswlib) you can install using `this link <https://www.scivision.dev/python-windows-visual-c-14-required/>`_ . You can also subsequently install dependences separately ::

  pip install pybind11, hnswlib, igraph, leidenalg>=0.7.0, umap-learn, numpy>=1.17, scipy, pandas>=0.25, sklearn, termcolor, pygam, phate, matplotlib,scanpy
  pip install pyVIA





