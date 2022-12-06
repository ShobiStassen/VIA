Release History
===============

Version 0.1.62
-------------
- New Feature! Heatmap based gene trends (genes x pseudotime) for each lineage
- New Feature! annotate nodes in plot_edge_bundle() by setting text_labels = True uses true_labels as annotations. Optionally provide list of single cell annotations length n_samples to use instead of true_labels. Example figure below

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/milestoneplot_withannots.png?raw=true" width="600px" align="center" </a>

Version 0.1.61
-------------
- Bug fix for import module in examples.py

Version 0.1.60
-------------
- Bug fix for root detection (the initialization for None was over-writing RNA-velocity predicted roots)

Version 0.1.59
-------------
- corrected the auto-scaling in draw_sc_lineage_probability() so that each subplot has the same colorbar scale

Version 0.1.58
-------------
- fix random_seed so pseudotime and branching probabilities are reproducible

Version 0.1.57
-------------
- optionally allow user to fix terminal states based on cell index or group label (corresponding to true_label)
- optionally allow user to plot only selected lineages (by corresponding terminal cluster number) in get_gene_expression and draw_sc_lineage_probability (marker_lineages = [2,8,10])

Version 0.1.56
-------------
- support via-guided embeddings. In particular fast via-mds and via-umap which can be adjusted using known time-series data if available

