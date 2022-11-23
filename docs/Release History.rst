Release History
===============

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

