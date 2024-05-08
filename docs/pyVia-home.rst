|DOI|

StaVia - Multi-Omic Single-Cell Cartography for Spatial and Temporal Atlases
=============================================================================

**StaVia (Via 2.0)** is our new single-cell trajectory inference method that explores **single-cell atlas-scale** data and **temporal and spatial studies** enabled by:

#. **Graph augmentation using metadata for (temporal) studies:** Using sequential metadata (temporal labels, hierarchical information) and spatial tissue coordinates (from Spatial omics data) to guide the cartography
#. **Higher Order Random Walks:** Leveraging higher order random walks with **memory** to highlight key end-to-end differentiation pathways along the atlas 
#. **Atlas View:** StaVia offers a unique visualization of the predicted trajectory by intuitively merging the cell-cell graph connectivity with the high-resolution of single-cell embeddings.

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/twitter%20gif.gif?raw=true" width="850px" align="center" </a>
|
|

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/rtd_fig1.png?raw=true" width="850px" align="center", class="only-light" </a>

StaVia still offers all the functionality of `Via 1.0 <https://www.nature.com/articles/s41467-021-25773-3>`_ .  in terms of various types of topology construction (disconnected, cyclic), pseudotimes, automated terminal state prediction and automated plotting of temporal gene dynamics along lineages, for details please refer to our `preprint <https://www.biorxiv.org/content/10.1101/2024.01.29.577871v1>`_.

StaVia extends the lazy-teleporting walks to higher order random walks with **memory** to allow better lineage detection, pathway recovery and preservation of global features in terms of computation and visualization. StaVia is generalizable to multi-omic analysis: in addition to transcriptomic data, VIA works on scATAC-seq, flow and imaging cytometry data. 



**Try out the following with StaVia:**

- Combining temporal information with scRNA velocity `temporal study <https://pyvia.readthedocs.io/en/latest/Via2.0%20Cartographic%20Mouse%20Gastrualation.html>`_
- Constructing the Atlas View `visualization  <https://pyvia.readthedocs.io/en/latest/Zebrahub_tutorial_visualization.html>`_ H5ad anndata objects are provided on the `github page <https://github.com/ShobiStassen/VIA>`_
- Using spatial information of cells' location on tissues to guide the TI. See tutorials for `MERFISH <https://pyvia.readthedocs.io/en/latest/notebooks/StaVia%20MERFISH%202.html>`_ and Stereoseq data. 
- Datasets used in tutorials can be found on the `github-data <https://github.com/ShobiStassen/VIA/tree/master/Datasets>`_ page

**StaVia Atlas View plots hi-res edge graph for Mouse Gastrulation (Pijuan Sala)**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/AtlasGallery/rtd_fig2_v2.png?raw=true" width="850px" align="center" </a>

**StaVia visualizes Mouse Gastrulation using time-series and RNA velocity adjusted graphs**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/plasma_pijuansala_annotated.gif?raw=true" width="850px" align="center" </a>





