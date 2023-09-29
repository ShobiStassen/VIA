|DOI|

pyVIA 1.0
======

**VIA** is a single-cell Trajectory Inference method that offers topology construction, pseudotimes, automated terminal state prediction and automated plotting of temporal gene dynamics along lineages. VIA combines lazy-teleporting random walks and Monte-Carlo Markov Chain simulations to overcome common challenges such as 1) accurate terminal state and lineage inference, 2) ability to capture combination of cyclic, disconnected and tree-like structures, 3) scalability in feature and sample space. 4) Generalizability to multi-omic analysis. In addition to transcriptomic data, VIA works on scATAC-seq, flow and imaging cytometry data. 
Please refer to our `paper <https://www.nature.com/articles/s41467-021-25773-3>`_ for more details. 


**VIA visualizes Mouse Gastrulation using time-series and RNA velocity adjusted graphs**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/plasma_pijuansala_annotated.gif?raw=true" width="600px" align="center" </a>
  
**Multifurcating wind maps without RNA velocity**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/multifurc_animation.gif?raw=true" width="600px" align="center" </a>

**VIA plots hi-res edge graph for Mouse Gastrulation (Pijuan Sala) human hematopoiesis**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/milestoneplot_withannots.png?raw=true" width="600px" align="center" </a>

**Via Atlas Animation of connectivity for human hematopoiesis (Setty)**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/human_edgebundle_test.gif?raw=true" width="600px" align="center" </a>


Examples and Visualization
--------------------------
There are several `Jupyter Notebooks <https://github.com/ShobiStassen/VIA/tree/master/Jupyter%20Notebooks>`_ here and on the github page with step-by-step code for real and simulated datasets. |:eight_spoked_asterisk:| **The NB for multifurcating data shows a step-by-step usage tutorial.** 


**scATAC-seq Human Hematopoiesis** `(click to open interactive VIA graph) <https://shobistassen.github.io/toggle_data.html>`_

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/scATAC_BuenrostroPCs_MainFig.png?raw=true" width="600px" align="center" </a>

**Notebooks**

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Notebook
     - details
     - dataset
     - reference

   * - Multifurcation: `Starter Tutorial <https://github.com/ShobiStassen/VIA/blob/master/Jupyter%20Notebooks/ViaJupyter_Toy_Multifurcating.ipynb>`_
     - 4-leaf simulation
     - `4-leaf <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_
     - `DynToy <https://github.com/dynverse/dyntoy>`_

   * - Disconnected `Tutorial <https://github.com/ShobiStassen/VIA/blob/master/Jupyter%20Notebooks/ViaJupyter_Toy_Disconnected.ipynb>`_
     - disconnected simulation
     - `4-leaf <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_
     - `DynToy <https://github.com/dynverse/dyntoy>`_

   * - Human `Embryoid <https://github.com/ShobiStassen/VIA/blob/master/Jupyter%20Notebooks/ViaJupyter_EmbryoidBody.ipynb>`_
     - 16,825 ESCs
     - EB `scRNA-seq <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_ and `embedding <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_
     - Moon et al. (2019)

   * - scATAC-seq `Hematopoiesis <https://github.com/ShobiStassen/VIA/blob/master/Jupyter%20Notebooks/ViaJupyter_scATAC-seq_HumanHematopoiesis.ipynb>`_
     - Human hematopoiesis
     - `scATAC-seq <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_
     - Buenrostro et al. (2018)

   * - scRNA-seq `Hematopoiesis <https://github.com/ShobiStassen/VIA/blob/master/Jupyter%20Notebooks/ViaJupyter_scRNA_Hematopoiesis.ipynb>`_
     - Human hematopoiesis (5780 cells)
     - CD34 `scRNA-seq <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_
     - Setty et al. (2019)




.. |DOI| image:: https://zenodo.org/badge/212254929.svg
    :target: https://zenodo.org/badge/latestdoi/212254929
    :alt: DOI
