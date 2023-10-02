|DOI|

pyVIA - Multi-Omic Single-Cell Cartography 
====================================================

**Via 2.0** is our new single-cell trajectory inference method that explores **single-cell atlas-scale** data and **temporal studies** enabled by:

#. **Graph augmentation using metadata for (temporal) studies:** Using sequential metadata (temporal labels, hierarchical information, spatial distances) to guide the cartography
#. **Higher Order Random Walks:** Leveraging higher order random walks with **memory** to highlight key end-to-end differentiation pathways along the atlas 
#. **Atlas View:** Via 2.0 offers a unique visualization of the predicted trajectory by intuitively merging the cell-cell graph connectivity with the high-resolution of single-cell embeddings.


.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/rtd_fig1.png?raw=true" width="850px" align="center", class="only-light" </a>

Via 2.0 still offers all the functionality of Via 1.0 in terms of various types of topology construction (disconnected, cyclic), pseudotimes, automated terminal state prediction and automated plotting of temporal gene dynamics along lineages, for details please refer to our `paper <https://www.nature.com/articles/s41467-021-25773-3>`_ . 

Via 2.0 extends the lazy-teleporting walks to higher order random walks with **memory** to allow better lineage detection, pathway recovery and preservation of global features in terms of computation and visualization. Via 2.0 is generalizable to multi-omic analysis: in addition to transcriptomic data, VIA works on scATAC-seq, flow and imaging cytometry data. 



**Try out the following with Via 2.0:**

- Combining temporal information with scRNA velocity `temporal study <https://pyvia.readthedocs.io/en/latest/Via2.0%20Cartographic%20Mouse%20Gastrualation.html>`_
- Constructing the Atlas View `visualization  <https://pyvia.readthedocs.io/en/latest/Zebrahub_tutorial_visualization.html>`_

**Via 2.0 Atlas View plots hi-res edge graph for Mouse Gastrulation (Pijuan Sala)**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/AtlasGallery/rtd_fig2_v2.png?raw=true" width="850px" align="center" </a>

**Via 2.0 visualizes Mouse Gastrulation using time-series and RNA velocity adjusted graphs**

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/plasma_pijuansala_annotated.gif?raw=true" width="850px" align="center" </a>





