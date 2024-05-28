|DOI|

`StaVia <https://www.biorxiv.org/content/10.1101/2024.01.29.577871v1>`_ - Multi-Omic Single-Cell Cartography for Spatial and Temporal Atlases
=============================================================================

**StaVia (Via 2.0)** is our new single-cell trajectory inference method (`preprint <https://www.biorxiv.org/content/10.1101/2024.01.29.577871v1>`_) that explores **single-cell atlas-scale** data and **temporal and spatial studies** enabled by:

#. **Graph augmentation integrating metadata (spatial omics and time-series studies):** Using sequential metadata (temporal labels, hierarchical information, spatial coordinates) to guide the cartography
#. **Higher Order Random Walks:** Leveraging higher order random walks with **memory** of a cell's past states to highlight key end-to-end differentiation pathways along the atlas
#. **Atlas View:** Via 2.0 offers a unique visualization of the predicted trajectory by intuitively merging the cell-cell graph connectivity with the high-resolution of single-cell embeddings.

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/twitter%20gif.gif?raw=true" width="850px" align="center" </a>
|
|

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/rtd_fig1.png?raw=true" width="850px" align="center", class="only-light" </a>

StaVia still offers all the functionality of `Via 1.0 <https://www.nature.com/articles/s41467-021-25773-3>`_ .  in terms of various types of topology construction (disconnected, cyclic), pseudotimes, automated terminal state prediction and automated plotting of temporal gene dynamics along lineages, for details please refer to our `preprint <https://www.biorxiv.org/content/10.1101/2024.01.29.577871v1>`_.

StaVia extends the lazy-teleporting walks to higher order random walks with **memory** to allow better lineage detection, pathway recovery and preservation of global features in terms of computation and visualization. StaVia is generalizable to multi-omic analysis: in addition to transcriptomic data (time-series studies, spatial omics), VIA works on scATAC-seq, flow and imaging cytometry data. 



**Try out the following with StaVia:**

- Combining temporal information with scRNA velocity `temporal study <https://pyvia.readthedocs.io/en/latest/notebooks/Via2.0%20Cartographic%20Mouse%20Gastrualation.html>`_
- Constructing the Atlas View `visualization  <https://pyvia.readthedocs.io/en/latest/notebooks/Zebrahub_tutorial_visualization.html>`_ . H5ad anndata objects are provided on the `github page <https://github.com/ShobiStassen/VIA>`_
- Using spatial information of cells' location on tissues to guide the TI. See tutorials for `MERFISH <https://pyvia.readthedocs.io/en/latest/notebooks/StaVia%20MERFISH%202.html>`_ and Stereoseq data. 
- Datasets used in tutorials can be found on the `github-data <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_ page

**StaVia Cartographic Atlas View for Mouse Gastrulation using time-series and RNA velocity adjusted graphs**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/AtlasGallery/rtd_fig2_v2.png?raw=true" width="850px" align="center" </a>


.. toctree::
   :maxdepth: 1
   :caption: General
   :hidden:
   
   pyVia-home
   Installation
   Release History
   Tutorial Video
   Parameters and Attributes
   api/index
   Basic Example Code

.. toctree::
   :maxdepth: 1
   :caption: StaVia Atlas View Gallery
   :hidden:

   Atlas view examples
   Mouse_to_pup_atlas
   StaVia Atlas Animation
   StaVia Atlas View for Spatial omics

.. toctree::
   :maxdepth: 1
   :caption: StaVia for time-series
   :hidden:

   notebooks/Via2.0 Cartographic Mouse Gastrualation
   notebooks/Zebrahub TI tutorial   
   notebooks/Zebrahub_tutorial_visualization

.. toctree::
   :maxdepth: 1
   :caption: StaVia for spatial-temporal
   :hidden:

   notebooks/StaVia MERFISH 2
   notebooks/Zesta_jp_tutorial

.. toctree::
   :maxdepth: 1
   :caption: scRNA, velocity and scATAC seq Tutorials
   :hidden:

   notebooks/ViaJupyter_scRNA_Hematopoiesis
   notebooks/ViaJupyter_Pancreas_RNAvelocity
   notebooks/ViaJupyter_scRNAVelocity_hematopoiesis
   notebooks/scATAC-seq_HumanHematopoiesis

.. toctree::
   :maxdepth: 1
   :caption: Cytometry
   :hidden:
   
   notebooks/Imaging Cytometry (cell cycle)
   notebooks/mESC_timeseries

.. toctree::
   :maxdepth: 1
   :caption: Basic Tutorials for Toy Data
   :hidden:

   notebooks/ViaJupyter_Toy_Multifurcating
   notebooks/ViaJupyter_Toy_Disconnected

.. |DOI| image:: https://zenodo.org/badge/212254929.svg
    :target: https://zenodo.org/badge/latestdoi/212254929
    :alt: DOI
