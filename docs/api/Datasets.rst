.. currentmodule:: VIA.datasets_via
    
Datasets
========
To retrieve a dataset (e.g. cell_cycle) use:

.. code-block:: python

    import pyVIA.datasets_via as datasets_via
    adata = datasets_via.cell_cycle()
 
.. autosummary::
    :toctree: _autosummary/VIA.datasets_via

    cell_cycle
    cell_cycle_cyto_data
    embryoid_body
    scATAC_hematopoiesis
    scRNA_hematopoiesis
    toy_disconnected
    toy_multifurcating