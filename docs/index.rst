|DOI|

pyVIA - Fast Multi-Omic Single-Cell Trajectory Inference 
==================================================================
**VIA** is a single-cell Trajectory Inference method that offers topology construction, pseudotimes, automated terminal state prediction and automated plotting of temporal gene dynamics along lineages. VIA combines lazy-teleporting random walks and Monte-Carlo Markov Chain simulations to overcome common challenges such as 1) accurate terminal state and lineage inference, 2) ability to capture combination of cyclic, disconnected and tree-like structures, 3) scalability in feature and sample space. 4) Generalizability to multi-omic analysis. In addition to transcriptomic data, VIA works on scATAC-seq, flow and imaging cytometry data. 
Please refer to our `paper <https://www.nature.com/articles/s41467-021-25773-3>`_ for more details. 



|:eight_spoked_asterisk:| **Fine-grained vector field without using RNA-velocity**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/multifurc_animation.gif?raw=true" width="600px" align="center" </a>

Examples and Visualization
--------------------------
There are several `Jupyter Notebooks <https://github.com/ShobiStassen/VIA/tree/master/Jupyter%20Notebooks>`_ here and on the github page with step-by-step code for real and simulated datasets. :eight_spoked_asterisk: **The NB for multifurcating data shows a step-by-step usage tutorial.** 


**scATAC-seq dataset of Human Hematopoiesis represented by VIA graphs** *(click image to open interactive graph)*
-----------------------------------------------------------------------------------------------------------------------------

.. raw:: html

  <img src="https://shobistassen.github.io/toggle_data.html" width="250px" align="center" </a>
  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/scATAC_BuenrostroPCs_MainFig.png?raw=true" width="250px" align="center" </a>


.. toctree::
   :maxdepth: 2
   :caption: General:
   :hidden:

   Installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:
   :hidden:

   Basic tutorial
   Disconnected trajectories
   Hematopoiesis-scRNAseq
   Using RNA-velocity
   Time-series
   Cytometry



.. |DOI| image:: https://zenodo.org/badge/212254929.svg
    :target: https://zenodo.org/badge/latestdoi/212254929
    :alt: DOI
